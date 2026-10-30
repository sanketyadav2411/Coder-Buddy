"""Secure-coding scanner and fixer: detection and correction of violations with a (local) LLM.

Pipeline (see agent/graph.py for the wiring):

    load files -> scanner -> [open violations and rounds left?] -> fixer -> scanner -> ... -> report

Design notes
------------
* Hybrid detection: deterministic rules (agent/rules.py) + LLM analysis. Every LLM finding must be
  anchored to code that really exists in the file (its quoted evidence is located in the file and
  the line numbers are snapped to it); findings that cannot be anchored are dropped.
* No tool-calling: scan and fix are plain "return JSON" calls validated with Pydantic
  (agent/llm.py). The LLM never gets file-system access.
* The fixer returns line-range patches. Our code validates, applies and syntax-checks them; a bad
  patch set is rolled back for that file. A fix only counts once a re-scan no longer reports it.
* The code under review is untrusted data (prompt-injection safe prompts, no tools, output only
  parsed as JSON, never executed).
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from agent.events import emit
from agent.llm import (LLMOutputError, current_model, current_provider, endpoint_warnings, get_llm,
                       structured_call)
from agent.prompts import fixer_prompt, scanner_prompt
from agent.rules import (LLM_SKIP_LANGS, RULES, language_of, mask_secrets, run_rules)
from agent.states import (FixResponse, LLMViolation, Patch, Report, ScanResponse, SkippedFile,
                          Violation, SEVERITY_RANK)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "dist",
             "build", ".idea", ".vscode", "scan_reports", "scan_workspace", ".mypy_cache",
             ".pytest_cache", "__MACOSX"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock"}
MAX_NOTES = 60


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class ScanConfig:
    max_files: int = 200
    max_file_kb: int = 200
    max_total_mb: int = 5
    max_fix_rounds: int = 2
    chunk_lines: int = 200
    chunk_chars: int = 12000
    chunk_overlap: int = 20
    max_patch_span: int = 60
    llm_retries: int = 1
    workers: int = 1          # parallel model calls while scanning (cloud models only; see from_env)
    use_llm: bool = True
    auto_fix: bool = True
    report_dir: str = "scan_reports"

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "ScanConfig":
        try:
            # A local model on one GPU/CPU gains nothing from parallel requests; a cloud API does.
            default_workers = 1
        except ValueError:
            default_workers = 1
        cfg = cls(
            workers=_env_int("SCAN_WORKERS", default_workers),
            max_files=_env_int("SCAN_MAX_FILES", 200),
            max_file_kb=_env_int("SCAN_MAX_FILE_KB", 200),
            max_total_mb=_env_int("SCAN_MAX_TOTAL_MB", 5),
            max_fix_rounds=_env_int("MAX_FIX_ROUNDS", 2),
            chunk_lines=_env_int("SCAN_CHUNK_LINES", 200),
            chunk_chars=_env_int("SCAN_CHUNK_CHARS", 12000),
        )
        known = {f.name for f in fields(cls)}
        for k, v in (overrides or {}).items():
            if k in known and v is not None:
                setattr(cfg, k, v)
        # clamp to sane ranges so a bad value cannot cause unbounded work
        cfg.max_files = max(1, min(int(cfg.max_files), 2000))
        cfg.max_file_kb = max(1, min(int(cfg.max_file_kb), 2048))
        cfg.max_total_mb = max(1, min(int(cfg.max_total_mb), 50))
        cfg.max_fix_rounds = max(0, min(int(cfg.max_fix_rounds), 5))
        cfg.chunk_lines = max(40, min(int(cfg.chunk_lines), 1000))
        cfg.chunk_chars = max(2000, min(int(cfg.chunk_chars), 60000))
        cfg.chunk_overlap = max(0, min(int(cfg.chunk_overlap), cfg.chunk_lines // 2))
        cfg.max_patch_span = max(1, int(cfg.max_patch_span))
        cfg.llm_retries = max(0, min(int(cfg.llm_retries), 5))
        cfg.workers = max(1, min(int(cfg.workers), 8))
        cfg.use_llm = bool(cfg.use_llm)
        cfg.auto_fix = bool(cfg.auto_fix)
        return cfg


def _note(notes: list[str], msg: str) -> None:
    if len(notes) < MAX_NOTES:
        notes.append(msg)
    elif len(notes) == MAX_NOTES:
        notes.append("... further messages omitted")


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def split_text(text: str) -> tuple[list[str], str, bool]:
    """-> (lines, eol, ends_with_newline). Round-trips with join_text (CRLF preserved)."""
    if text == "":
        return [], "\n", False
    eol = "\r\n" if "\r\n" in text else "\n"
    norm = text.replace("\r\n", "\n")
    trailing = norm.endswith("\n")
    if trailing:
        norm = norm[:-1]
    return norm.split("\n"), eol, trailing


def join_text(lines: list[str], eol: str, trailing: bool) -> str:
    out = eol.join(lines)
    return out + eol if (trailing and lines) else out


def numbered(lines: list[str], start: int, end: int, max_line_chars: int = 4000) -> str:
    """Listing 'N | code' for lines start..end (1-based, inclusive) with ABSOLUTE numbers."""
    width = len(str(end))
    rows = []
    for i in range(start, end + 1):
        line = lines[i - 1]
        if len(line) > max_line_chars:
            line = line[:max_line_chars] + " ...[truncated]"
        rows.append(f"{i:>{width}} | {line}")
    return "\n".join(rows)


def make_windows(lines: list[str], max_lines: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Split a file into overlapping (start, end) windows, 1-based inclusive. Line numbers are
    absolute, so no offset arithmetic is needed when mapping findings back to the file."""
    n = len(lines)
    wins: list[tuple[int, int]] = []
    s = 1
    while s <= n:
        e = s
        chars = len(lines[s - 1]) + 8
        while e < n and (e - s + 1) < max_lines and chars + len(lines[e]) + 8 <= max_chars:
            chars += len(lines[e]) + 8
            e += 1
        wins.append((s, e))
        if e >= n:
            break
        s = max(e - overlap + 1, s + 1)
    return wins


_NUM_PREFIX = re.compile(r"^\s*\d+\s*\|\s?")


def _norm(s: str) -> str:
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# file loading
# ---------------------------------------------------------------------------
def _safe_under(root: Path, p: Path) -> bool:
    try:
        rp = p.resolve()
    except OSError:
        return False
    return rp == root or root in rp.parents


def load_files(root: Path | str, cfg: Optional[ScanConfig] = None
               ) -> tuple[dict[str, str], list[SkippedFile]]:
    """Read scannable source files under `root` (posix-style relative paths -> text).

    Skips symlinks, vendored/build directories, binaries, non-UTF-8 files and anything over the
    configured size/count limits, and records why in the returned `skipped` list."""
    cfg = cfg or ScanConfig.from_env()
    root = Path(root).resolve()
    files: dict[str, str] = {}
    skipped: list[SkippedFile] = []
    if not root.is_dir():
        return files, skipped
    total = 0
    max_bytes = cfg.max_file_kb * 1024
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIRS or (dp / d).is_symlink():
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            full = dp / name
            rel = full.relative_to(root).as_posix()
            if name in SKIP_NAMES or name.endswith(".min.js") or language_of(rel) is None:
                continue
            if full.is_symlink():
                skipped.append(SkippedFile(path=rel, reason="symbolic link"))
                continue
            if not _safe_under(root, full):
                skipped.append(SkippedFile(path=rel, reason="outside the scan root"))
                continue
            if len(files) >= cfg.max_files:
                skipped.append(SkippedFile(path=rel, reason=f"file limit ({cfg.max_files}) reached"))
                continue
            try:
                size = full.stat().st_size
                if size > max_bytes:
                    skipped.append(SkippedFile(path=rel, reason=f"larger than {cfg.max_file_kb} KB"))
                    continue
                if total + size > cfg.max_total_mb * 1024 * 1024:
                    skipped.append(SkippedFile(path=rel, reason="total size limit reached"))
                    continue
                with open(full, "rb") as fh:
                    data = fh.read(max_bytes + 1)
            except OSError as e:
                skipped.append(SkippedFile(path=rel, reason=f"unreadable ({e.__class__.__name__})"))
                continue
            if b"\0" in data:
                skipped.append(SkippedFile(path=rel, reason="binary file"))
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append(SkippedFile(path=rel, reason="not valid UTF-8"))
                continue
            total += len(data)
            files[rel] = text
    return files, skipped


def collect_sources(path: Path | str, cfg: Optional[ScanConfig] = None
                    ) -> tuple[dict[str, str], list[SkippedFile]]:
    """Like load_files but also accepts a single file path."""
    p = Path(path)
    if p.is_file():
        cfg = cfg or ScanConfig.from_env()
        if language_of(p.name) is None:
            return {}, [SkippedFile(path=p.name, reason="unsupported file type")]
        with tempfile.TemporaryDirectory() as td:
            shutil.copy2(p, Path(td) / p.name)
            return load_files(td, cfg)
    return load_files(p, cfg)


def write_text_files(root: Path, files: dict[str, str]) -> None:
    """Write {relative path: text} under root (used to copy sources into the scan workspace)."""
    root = root.resolve()
    for rel, text in files.items():
        target = _safe_target(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)


def _safe_target(root: Path, rel: str) -> Path:
    root = root.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the workspace: {rel}")
    if (root / rel).is_symlink():
        raise ValueError(f"refusing to write through a symlink: {rel}")
    return target


# ---------------------------------------------------------------------------
# safe upload handling
# ---------------------------------------------------------------------------
class UnsafeArchiveError(ValueError):
    """The uploaded archive/file was rejected (traversal, symlink, too large, not a zip, ...)."""


def reset_workspace(base: Path | str, name: str = "current") -> Path:
    """Create an empty `base/name` directory (removing previous contents)."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("invalid workspace name")
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    ws = base / name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    return ws


def safe_extract_zip(data: bytes, dest: Path | str, cfg: Optional[ScanConfig] = None
                     ) -> tuple[list[str], list[SkippedFile]]:
    """Extract only scannable source files from a zip into `dest`.

    Aborts the whole archive (UnsafeArchiveError) on path traversal, absolute paths, symlinks,
    declared or actual size above the limits, too many entries, or a corrupt zip. Files of other
    types (images, binaries, ...) are skipped silently."""
    cfg = cfg or ScanConfig.from_env()
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise UnsafeArchiveError("not a valid zip file") from e
    infos = zf.infolist()
    if len(infos) > cfg.max_files * 10:
        raise UnsafeArchiveError(f"archive has too many entries ({len(infos)})")
    max_file = cfg.max_file_kb * 1024
    max_total = cfg.max_total_mb * 1024 * 1024
    declared = sum(i.file_size for i in infos if not i.is_dir())
    if declared > max_total * 20:  # zip-bomb guard on declared size, even before filtering
        raise UnsafeArchiveError("archive expands to an unreasonable size")
    extracted: list[str] = []
    skipped: list[SkippedFile] = []
    total = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if (name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in parts
                or "\x00" in name):
            raise UnsafeArchiveError(f"unsafe path in archive: {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
            raise UnsafeArchiveError(f"symbolic link in archive: {info.filename!r}")
        if info.is_dir() or not parts:
            continue
        rel = "/".join(parts)
        if (any(p in SKIP_DIRS for p in parts[:-1]) or parts[-1] in SKIP_NAMES
                or language_of(rel) is None):
            continue
        if len(extracted) >= cfg.max_files:
            skipped.append(SkippedFile(path=rel, reason=f"file limit ({cfg.max_files}) reached"))
            continue
        if info.file_size > max_file:
            skipped.append(SkippedFile(path=rel, reason=f"larger than {cfg.max_file_kb} KB"))
            continue
        target = (dest / rel).resolve()
        if dest not in target.parents:
            raise UnsafeArchiveError(f"unsafe path in archive: {info.filename!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        too_big = False
        with zf.open(info) as src, open(target, "wb") as out:
            while True:
                chunk = src.read(64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                total += len(chunk)
                if total > max_total:
                    raise UnsafeArchiveError("archive contents exceed the total size limit")
                if written > max_file:       # header lied about the size
                    too_big = True
                    break
                out.write(chunk)
        if too_big:
            target.unlink(missing_ok=True)
            skipped.append(SkippedFile(path=rel, reason=f"larger than {cfg.max_file_kb} KB"))
            continue
        extracted.append(rel)
    return extracted, skipped


def save_uploaded_file(filename: str, data: bytes, dest: Path | str,
                       cfg: Optional[ScanConfig] = None) -> str:
    """Store one uploaded source file (basename only, supported type, size-limited)."""
    cfg = cfg or ScanConfig.from_env()
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    if not name or name in {".", ".."} or "\x00" in name:
        raise UnsafeArchiveError("invalid file name")
    if language_of(name) is None:
        raise UnsafeArchiveError(f"unsupported file type: {name}")
    if len(data) > cfg.max_file_kb * 1024:
        raise UnsafeArchiveError(f"{name} is larger than {cfg.max_file_kb} KB")
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_bytes(data)
    return name


# ---------------------------------------------------------------------------
# evidence validation (guards against hallucinated findings)
# ---------------------------------------------------------------------------
def locate_evidence(norm_lines: list[str], evidence: str, hint: int) -> Optional[tuple[int, int]]:
    """Find where the quoted `evidence` really is in the file (1-based inclusive range).

    Returns None when it cannot be found. When it appears several times, the occurrence nearest
    to the model's claimed line `hint` wins."""
    ev = [_norm(_NUM_PREFIX.sub("", l)) for l in (evidence or "").splitlines()]
    ev = [l for l in ev if l][:5]
    if not ev or len(" ".join(ev)) < 4:
        return None
    n = len(ev)
    exact: list[int] = []
    partial: list[int] = []
    for i in range(0, len(norm_lines) - n + 1):
        if all(ev[k] in norm_lines[i + k] for k in range(n)):
            partial.append(i + 1)
            if all(ev[k] == norm_lines[i + k] for k in range(n)):
                exact.append(i + 1)
    pool = exact or partial
    if not pool:
        return None
    best = min(pool, key=lambda s: (abs(s - hint), s))
    return best, best + n - 1


def validate_llm_violation(lv: LLMViolation, path: str, lines: list[str],
                           norm_lines: list[str]) -> Optional[Violation]:
    """Turn an LLM finding into a Violation anchored to real code, or None if it cannot be anchored."""
    hint = max(1, lv.line_start)
    loc = locate_evidence(norm_lines, lv.evidence, hint)
    if loc is None:
        return None
    start, end = loc
    evidence = "\n".join(l.strip() for l in lines[start - 1:end])[:300]
    if lv.cwe == "CWE-798":
        evidence = mask_secrets(evidence)
    return Violation(
        file=path, line_start=start, line_end=end, cwe=lv.cwe,
        owasp=(lv.owasp or None), cert=(lv.cert or None), severity=lv.severity,  # type: ignore[arg-type]
        title=(lv.title or "Security weakness").strip()[:160], evidence=evidence,
        explanation=(lv.explanation or "").strip()[:600],
        suggested_fix=(lv.suggested_fix or "").strip()[:600], source="llm",
    )


def _overlap(a: Violation, b: Violation) -> bool:
    return a.line_start <= b.line_end and b.line_start <= a.line_end


def merge_violations(rule_vios: list[Violation], llm_vios: list[Violation]) -> list[Violation]:
    """Rule hits win over LLM findings for the same weakness on the same lines; LLM duplicates
    produced by overlapping windows are collapsed (highest severity kept)."""
    merged = list(rule_vios)
    for v in llm_vios:
        dup = next((m for m in merged if m.cwe == v.cwe and _overlap(m, v)), None)
        if dup is None:
            merged.append(v)
        elif dup.source == "llm" and SEVERITY_RANK[v.severity] > SEVERITY_RANK[dup.severity]:
            merged[merged.index(dup)] = v
    merged.sort(key=lambda v: (v.file, v.line_start, v.cwe))
    return merged


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
@dataclass
class ScanContext:
    llm: Any = None
    warnings: list[str] = field(default_factory=list)
    consecutive_failures: int = 0
    bad_output: int = 0
    ok_chunks: int = 0
    dead: bool = False

    @property
    def usable(self) -> bool:
        return self.llm is not None and not self.dead


def scan_files(files: dict[str, str], cfg: ScanConfig, ctx: ScanContext) -> list[Violation]:
    """Rules + LLM over `files`. Never raises because of the LLM: failures become warnings.

    Model calls (one per window of a file) run in a small thread pool when cfg.workers > 1."""
    prepared: dict[str, tuple[list[Violation], list[str], list[str]]] = {}
    jobs: list[tuple[str, int, int, str, list[Violation]]] = []   # (path, start, end, lang, rule_vios)
    for path in sorted(files):
        text = files[path]
        lang = language_of(path)
        if lang is None or not text.strip():
            continue
        rule_vios = run_rules(path, text, lang)
        lines, _, _ = split_text(text)
        prepared[path] = (rule_vios, lines, [_norm(l) for l in lines])
        if cfg.use_llm and ctx.usable and lang not in LLM_SKIP_LANGS:
            for (ws, we) in make_windows(lines, cfg.chunk_lines, cfg.chunk_chars, cfg.chunk_overlap):
                jobs.append((path, ws, we, lang, rule_vios))

    lock = threading.Lock()
    fail_limit = 2
    total = len(jobs)
    counter = {"n": 0}

    def run(job) -> list[Violation]:
        path, ws, we, lang, rule_vios = job
        if not ctx.usable:
            return []
        _, lines, norm_lines = prepared[path]
        with lock:
            counter["n"] += 1
            n = counter["n"]
        emit("progress", f"Scanning {path} (lines {ws}-{we})", done=n - 1, total=total, file=path)
        hints = [f"line {v.line_start}: {v.title} ({v.cwe})"
                 for v in rule_vios if ws <= v.line_start <= we]
        window = "" if (ws == 1 and we == len(lines)) else f" - showing lines {ws}-{we} of {len(lines)}"
        prompt = scanner_prompt(numbered(lines, ws, we), path, lang, hints, secrets.token_hex(4), window)
        try:
            resp = structured_call(ctx.llm, prompt, ScanResponse, retries=cfg.llm_retries)
        except LLMOutputError as e:
            with lock:
                ctx.bad_output += 1
                _note(ctx.warnings, f"{path} lines {ws}-{we}: the model returned unusable output ({e})")
            return []
        except Exception as e:  # noqa: BLE001 - transport/auth/timeouts
            with lock:
                ctx.consecutive_failures += 1
                if ctx.consecutive_failures >= fail_limit:
                    ctx.dead = True
                    _note(ctx.warnings, f"LLM unavailable ({e.__class__.__name__}: {str(e)[:160]}); "
                                        "continuing with rule-based checks only.")
            return []
        with lock:
            ctx.consecutive_failures = 0
            ctx.ok_chunks += 1
        out = []
        for lv in resp.violations[:15]:
            v = validate_llm_violation(lv, path, lines, norm_lines)
            if v is not None:
                out.append(v)
        return out

    if cfg.workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(cfg.workers, total)) as pool:
            outs = list(pool.map(run, jobs))
    else:
        outs = [run(j) for j in jobs]

    llm_by_path: dict[str, list[Violation]] = {}
    for job, vios in zip(jobs, outs):
        llm_by_path.setdefault(job[0], []).extend(vios)
    results: list[Violation] = []
    for path in sorted(prepared):
        results.extend(merge_violations(prepared[path][0], llm_by_path.get(path, [])))
    if total:
        emit("progress", "Scan pass finished", done=total, total=total)
    return results


# ---------------------------------------------------------------------------
# fixing
# ---------------------------------------------------------------------------
_FENCE_LINE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*$")


def clean_replacement(text: str) -> list[str]:
    """Replacement text -> list of lines (strips code fences / copied line-number prefixes)."""
    if text == "":
        return []
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) >= 2 and _FENCE_LINE.match(lines[0]) and _FENCE_LINE.match(lines[-1]):
        lines = lines[1:-1]
    non_empty = [l for l in lines if l.strip()]
    if non_empty and all(_NUM_PREFIX.match(l) for l in non_empty):
        lines = [_NUM_PREFIX.sub("", l) for l in lines]
    return lines


def _conflict(p: Patch, q: Patch) -> bool:
    ps, pe, qs, qe = p.start_line, p.end_line, q.start_line, q.end_line
    p_ins, q_ins = pe < ps, qe < qs
    if p_ins and q_ins:
        return ps == qs
    if p_ins:
        return qs < ps <= qe
    if q_ins:
        return ps < qs <= pe
    return ps <= qe and qs <= pe


def apply_patches(lines: list[str], patches: list[Patch], max_span: int
                  ) -> tuple[list[str], list[Patch], list[tuple[Patch, str]]]:
    """Validate and apply line-range patches (bottom-up so numbers stay valid).

    -> (new_lines, applied, rejected_with_reason). Never mutates `lines`."""
    n = len(lines)
    valid: list[Patch] = []
    rejected: list[tuple[Patch, str]] = []
    for p in patches:
        s, e = p.start_line, p.end_line
        if s < 1 or s > n + 1:
            rejected.append((p, "start line out of range"))
        elif e < s - 1 or e > n:
            rejected.append((p, "end line out of range"))
        elif (e - s + 1) > max_span:
            rejected.append((p, f"patch spans more than {max_span} lines"))
        else:
            valid.append(p)
    valid.sort(key=lambda p: (p.start_line, p.end_line))
    accepted: list[Patch] = []
    for p in valid:
        if any(_conflict(p, q) for q in accepted):
            rejected.append((p, "overlaps another patch"))
        else:
            accepted.append(p)
    result = list(lines)
    for p in sorted(accepted, key=lambda p: (p.start_line, p.end_line), reverse=True):
        result[p.start_line - 1:p.end_line] = clean_replacement(p.replacement)
    return result, accepted, rejected


def _node_syntax_ok(text: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return True, "node not available"
    last = ""
    with tempfile.TemporaryDirectory() as td:
        for ext in (".cjs", ".mjs"):
            f = Path(td) / f"check{ext}"
            f.write_text(text, encoding="utf-8", newline="")
            try:
                r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True,
                                   timeout=15, cwd=td)
            except (OSError, subprocess.TimeoutExpired) as e:
                return True, f"node check skipped ({e.__class__.__name__})"
            if r.returncode == 0:
                return True, ""
            last = (r.stderr or "").strip().splitlines()[-1:] or ["syntax error"]
            last = last[0]
    return False, last


def _syntax_ok(path: str, text: str) -> tuple[bool, str]:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".py":
        try:
            compile(text, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            return False, f"{e.__class__.__name__}: {e}"
        return True, ""
    if suffix in (".js", ".mjs", ".cjs"):
        return _node_syntax_ok(text)
    if suffix == ".json":
        try:
            json.loads(text)
        except ValueError as e:
            return False, f"invalid JSON: {e}"
        return True, ""
    return True, ""   # html / c / others: no reliable standalone check


def check_syntax(path: str, new_text: str, original_text: str) -> tuple[bool, str]:
    """Reject a patched file that no longer parses - unless the original already did not parse
    (then the check proves nothing)."""
    ok, msg = _syntax_ok(path, new_text)
    if ok:
        return True, ""
    orig_ok, _ = _syntax_ok(path, original_text)
    if not orig_ok:
        return True, "original file already had syntax errors; syntax gate skipped"
    return False, msg


CONTEXT_LINES = 20


def cluster_violations(vs: list[Violation], lines: list[str], cfg: ScanConfig
                       ) -> list[tuple[list[Violation], int, int]]:
    """Group a file's violations into context windows small enough for the model."""
    n = len(lines)
    vs = sorted(vs, key=lambda v: (v.line_start, v.line_end))
    clusters: list[tuple[list[Violation], int, int]] = []
    cur: list[Violation] = []
    lo = hi = 0
    for v in vs:
        v_lo, v_hi = max(1, v.line_start - CONTEXT_LINES), min(n, v.line_end + CONTEXT_LINES)
        if cur and (max(hi, v_hi) - lo + 1) <= cfg.chunk_lines and \
                sum(len(lines[i - 1]) + 8 for i in range(lo, max(hi, v_hi) + 1)) <= cfg.chunk_chars:
            cur.append(v)
            hi = max(hi, v_hi)
        else:
            if cur:
                clusters.append((cur, lo, hi))
            cur, lo, hi = [v], v_lo, v_hi
    if cur:
        clusters.append((cur, lo, hi))
    return clusters


def _atomic_write(target: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fix_files(root: Path, violations: list[Violation], cfg: ScanConfig, llm: Any,
              notes: list[str], backup_dir: Optional[Path] = None) -> list[str]:
    """Ask the LLM for patches for `violations` and apply the safe ones. -> list of changed files."""
    root = root.resolve()
    by_file: dict[str, list[Violation]] = {}
    for v in violations:
        by_file.setdefault(v.file, []).append(v)
    changed: list[str] = []
    failures = 0
    for path in sorted(by_file):
        if failures >= 2:
            _note(notes, "fixer: stopped after repeated LLM errors")
            break
        try:
            target = _safe_target(root, path)
            original = target.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as e:
            _note(notes, f"fixer: cannot read {path} ({e.__class__.__name__})")
            continue
        lines, eol, trailing = split_text(original)
        lang = language_of(path) or "text"
        emit("progress", f"Fixing {path}", file=path)
        patches: list[Patch] = []
        for cluster, ws, we in cluster_violations(by_file[path], lines, cfg):
            if sum(len(lines[i - 1]) + 8 for i in range(ws, we + 1)) > cfg.chunk_chars * 2:
                _note(notes, f"fixer: {path} lines {ws}-{we} skipped (lines too long for the model)")
                continue
            window = "" if (ws == 1 and we == len(lines)) else f" - showing lines {ws}-{we} of {len(lines)}"
            items = [{"id": v.id, "line_start": v.line_start, "line_end": v.line_end, "cwe": v.cwe,
                      "severity": v.severity, "title": v.title, "evidence": v.evidence,
                      "suggested_fix": v.suggested_fix} for v in cluster]
            prompt = fixer_prompt(numbered(lines, ws, we), path, lang, items, secrets.token_hex(4), window)
            try:
                resp = structured_call(llm, prompt, FixResponse, retries=cfg.llm_retries)
                failures = 0
            except LLMOutputError as e:
                _note(notes, f"fixer: {path} lines {ws}-{we}: model returned unusable output ({e})")
                continue
            except Exception as e:  # noqa: BLE001
                failures += 1
                _note(notes, f"fixer: LLM error on {path} ({e.__class__.__name__}: {str(e)[:120]})")
                if failures >= 2:
                    break
                continue
            for p in resp.patches:
                # a patch may only touch the window the model was shown (plus an insertion at its end)
                if p.start_line < ws or p.end_line > we or p.start_line > we + 1:
                    _note(notes, f"fixer: {path}: patch for lines {p.start_line}-{p.end_line} is outside "
                                 f"the shown window {ws}-{we}; rejected")
                else:
                    patches.append(p)
        if not patches:
            continue
        new_lines, applied, rejected = apply_patches(lines, patches, cfg.max_patch_span)
        for p, why in rejected:
            _note(notes, f"fixer: {path}: patch {p.start_line}-{p.end_line} rejected ({why})")
        if not applied:
            continue
        new_text = join_text(new_lines, eol, trailing)
        if new_text == original:
            continue
        ok, msg = check_syntax(path, new_text, original)
        if not ok:
            _note(notes, f"fixer: {path}: patches rolled back, result failed the syntax check ({msg})")
            continue
        try:
            if backup_dir is not None:
                bpath = _safe_target(backup_dir, path)
                bpath.parent.mkdir(parents=True, exist_ok=True)
                if not bpath.exists():
                    bpath.write_text(original, encoding="utf-8", newline="")
            _atomic_write(target, new_text)
        except (OSError, ValueError) as e:
            _note(notes, f"fixer: could not write {path} ({e.__class__.__name__})")
            continue
        changed.append(path)
    return changed


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
_SECRET_RULES = [r for r in RULES if r.id in {"GEN001", "GEN002"}]


def _mask_diff_line(line: str) -> str:
    prefix, body = (line[:1], line[1:]) if line[:1] in ("+", "-", " ") else ("", line)
    if any(r.pattern.search(body) for r in _SECRET_RULES):
        body = mask_secrets(body)
    return prefix + body


def make_diffs(originals: dict[str, str], current: dict[str, str]) -> dict[str, str]:
    diffs: dict[str, str] = {}
    for path, before in originals.items():
        after = current.get(path)
        if after is None or after == before:
            continue
        raw = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                   fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
        text = "\n".join(_mask_diff_line(l) for l in raw)
        diffs[path] = text[:20000] + ("\n... diff truncated ..." if len(text) > 20000 else "")
    return diffs


def build_report(*, root: Path, cfg: ScanConfig, files: dict[str, str], skipped: list[SkippedFile],
                 baseline: list[Violation], current: list[Violation], fix_rounds: int,
                 originals: dict[str, str], warnings: list[str], llm_used: bool,
                 first_round: bool) -> Report:
    cur_by_key = {v.key(): v for v in current}
    base_keys = {b.key() for b in baseline}
    out: list[Violation] = []
    if first_round:
        out = [v.model_copy(update={"status": "open"}) for v in baseline]
    else:
        for b in baseline:
            now = cur_by_key.get(b.key())
            if now is None:
                out.append(b.model_copy(update={"status": "fixed"}))
            else:
                out.append(now.model_copy(update={"id": b.id, "status": "unresolved"}))
        out.extend(v.model_copy(update={"status": "new"}) for v in current if v.key() not in base_keys)
    out.sort(key=lambda v: (-SEVERITY_RANK[v.severity], v.file, v.line_start))
    remaining = [v for v in out if v.status in ("open", "unresolved", "new")]
    summary = {
        "files_scanned": len(files),
        "found": len(baseline),
        "fixed": sum(v.status == "fixed" for v in out),
        "unresolved": sum(v.status == "unresolved" for v in out),
        "new": sum(v.status == "new" for v in out),
        "remaining": len(remaining),
    }
    for sev in ("critical", "high", "medium", "low", "info"):
        summary[f"remaining_{sev}"] = sum(v.severity == sev for v in remaining)
    provider = model = ""
    if llm_used or cfg.use_llm:
        try:
            provider, model = current_provider(), current_model()
        except ValueError:
            pass
    return Report(
        root=str(root), provider=provider, model=model, llm_used=llm_used,
        files_scanned=sorted(files), skipped=skipped, fix_rounds=fix_rounds, violations=out,
        summary=summary, diffs=make_diffs(originals, files), warnings=list(dict.fromkeys(warnings)),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def report_to_markdown(r: Report) -> str:
    def fence(s: str) -> str:
        return s.replace("```", "'''")

    s = r.summary
    out = ["# Secure-coding scan report", "",
           f"- Generated: {r.generated_at}",
           f"- Files scanned: {s.get('files_scanned', 0)}",
           f"- LLM analysis: {'yes - ' + r.provider + ' / ' + r.model if r.llm_used else 'no (rule-based only)'}",
           f"- Fix rounds run: {r.fix_rounds}", "",
           "| Found | Fixed | Unresolved | New (after fixes) | Remaining |",
           "|---|---|---|---|---|",
           f"| {s.get('found', 0)} | {s.get('fixed', 0)} | {s.get('unresolved', 0)} | "
           f"{s.get('new', 0)} | {s.get('remaining', 0)} |", ""]
    if r.warnings:
        out += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]
    out += ["## Violations", ""]
    if not r.violations:
        out += ["No violations found.", ""]
    for v in r.violations:
        refs = ", ".join(x for x in (v.cwe, v.owasp, v.cert) if x)
        loc = f"{v.line_start}" if v.line_start == v.line_end else f"{v.line_start}-{v.line_end}"
        out += [f"### {v.id or '-'} [{v.severity.upper()}] {v.title}",
                f"- File: `{v.file}` line {loc}  |  Status: **{v.status}**  |  Source: {v.source}",
                f"- References: {refs}",
                f"- Why it matters: {v.explanation or 'n/a'}",
                f"- Suggested fix: {v.suggested_fix or 'n/a'}",
                "", "```", fence(v.evidence), "```", ""]
    if r.diffs:
        out += ["## Changes applied", ""]
        for path, d in r.diffs.items():
            out += [f"### {path}", "", "```diff", fence(d), "```", ""]
    if r.skipped:
        out += ["## Skipped files", ""] + [f"- `{k.path}`: {k.reason}" for k in r.skipped] + [""]
    out += ["---", "LLM findings are anchored to real code but are not proven exploitable; "
            "review every automatic change before using it."]
    return "\n".join(out) + "\n"


def write_report(report: Report, run_dir: Path) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = run_dir / "report.json", run_dir / "report.md"
    jp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    mp.write_text(report_to_markdown(report), encoding="utf-8")
    return {"json": str(jp), "markdown": str(mp)}


# ---------------------------------------------------------------------------
# graph nodes
# ---------------------------------------------------------------------------
def _default_root() -> Path:
    from agent.tools import PROJECT_ROOT
    return PROJECT_ROOT


def _assign_ids(vios: list[Violation], prefix: str) -> list[Violation]:
    ordered = sorted(vios, key=lambda v: (-SEVERITY_RANK[v.severity], v.file, v.line_start))
    return [v.model_copy(update={"id": f"{prefix}{i:03d}"}) for i, v in enumerate(ordered, start=1)]


def scanner_agent(state: dict) -> dict:
    """Scan (round 0: everything; later rounds: only files the fixer changed)."""
    cfg = ScanConfig.from_env(state.get("scan_options"))
    root = Path(state.get("scan_root") or _default_root()).resolve()
    rnd = int(state.get("scan_round", 0))
    fix_rounds = int(state.get("fix_rounds", 0))
    prev_report: Optional[Report] = state.get("report")
    warnings: list[str] = list(prev_report.warnings) if prev_report else []

    files, skipped = load_files(root, cfg)
    originals: dict[str, str] = dict(state.get("originals") or {})
    for p, t in files.items():
        originals.setdefault(p, t)
    run_dir = state.get("run_dir") or str(
        Path(cfg.report_dir).resolve() / f"scan-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}")

    emit("log", f"Scanner: round {rnd + 1} over {len(files)} file(s)")
    ctx = ScanContext(warnings=warnings)
    if cfg.use_llm:
        try:
            ctx.llm = get_llm(temperature=0, json_mode=True, max_tokens=1500)
            for w in endpoint_warnings():
                _note(warnings, w)
        except Exception as e:  # noqa: BLE001 - missing key/package, bad provider
            _note(warnings, f"LLM unavailable ({e.__class__.__name__}: {str(e)[:160]}); "
                            "continuing with rule-based checks only.")
    if not files:
        _note(warnings, "No scannable source files were found.")

    prev_current: list[Violation] = list(state.get("violations") or [])
    if rnd == 0:
        current = scan_files(files, cfg, ctx)
        baseline = _assign_ids(current, "V")
    else:
        baseline = list(state.get("baseline") or [])
        changed = set(state.get("changed_files") or [])
        carried = [v for v in prev_current if v.file in files and v.file not in changed]
        fresh = scan_files({p: t for p, t in files.items() if p in changed}, cfg, ctx)
        current = carried + fresh
        current.sort(key=lambda v: (v.file, v.line_start, v.cwe))
        # give identical findings the id they had before, and ids to genuinely new ones
        known = {b.key(): b.id for b in baseline}
        fresh_new = [v for v in current if v.key() not in known]
        for v in _assign_ids(fresh_new, "N"):
            known[v.key()] = v.id
        current = [v.model_copy(update={"id": known.get(v.key(), v.id)}) for v in current]

    llm_used = ctx.ok_chunks > 0 or bool(prev_report and prev_report.llm_used)
    if cfg.use_llm and ctx.llm is not None and not llm_used and ctx.bad_output and not ctx.dead:
        _note(warnings, "The model never returned usable output; results come from rules only.")

    # decide what happens next
    prev_keys = {tuple(k) for k in (state.get("prev_keys") or [])}
    cur_keys = {v.key() for v in current}
    stuck = fix_rounds > 0 and bool(prev_keys) and prev_keys <= cur_keys
    can_fix = (cfg.auto_fix and cfg.use_llm and ctx.usable and bool(current)
               and fix_rounds < cfg.max_fix_rounds and not stuck)
    if current and cfg.auto_fix and not can_fix:
        if not cfg.use_llm or ctx.llm is None or ctx.dead:
            _note(warnings, "Automatic fixing needs the LLM; violations were reported but not fixed.")
        elif stuck:
            _note(warnings, "Fixing stopped: the last round resolved nothing.")
        elif fix_rounds >= cfg.max_fix_rounds and cfg.max_fix_rounds > 0:
            _note(warnings, f"Fixing stopped after {fix_rounds} round(s) (limit reached).")
    nxt = "fixer" if can_fix else "END"

    report = build_report(root=root, cfg=cfg, files=files, skipped=skipped, baseline=baseline,
                          current=current, fix_rounds=fix_rounds, originals=originals,
                          warnings=warnings + list(state.get("notes") or []), llm_used=llm_used,
                          first_round=(rnd == 0))
    if nxt == "END":
        try:
            report = report.model_copy(update={"artifacts": write_report(report, Path(run_dir))})
        except OSError as e:
            report.warnings.append(f"Could not write the report files ({e.__class__.__name__}).")

    return {
        "scan_root": str(root), "scan_round": rnd + 1, "violations": current, "baseline": baseline,
        "originals": originals, "report": report, "scan_next": nxt, "run_dir": run_dir,
        "changed_files": [],
    }


def fixer_agent(state: dict) -> dict:
    cfg = ScanConfig.from_env(state.get("scan_options"))
    root = Path(state.get("scan_root") or _default_root()).resolve()
    violations: list[Violation] = list(state.get("violations") or [])
    notes: list[str] = list(state.get("notes") or [])
    llm = get_llm(temperature=0, json_mode=True, max_tokens=2000)
    backup = Path(state["run_dir"]) / "originals" if state.get("run_dir") else None
    changed = fix_files(root, violations, cfg, llm, notes, backup)
    return {
        "fix_rounds": int(state.get("fix_rounds", 0)) + 1,
        "prev_keys": [list(v.key()) for v in violations],
        "changed_files": changed,
        "notes": notes,
    }


def route_after_scan(state: dict) -> str:
    return "fixer" if state.get("scan_next") == "fixer" else "END"
