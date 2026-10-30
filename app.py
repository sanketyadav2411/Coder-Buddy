"""Coder Buddy - Streamlit frontend.

Run with:  streamlit run app.py

Two modes:
  * Generate project      - Planner -> Architect -> Coder (optionally followed by a secure-coding scan + fix)
  * Scan existing code    - upload source files or a .zip; violations are detected and fixed on a COPY

Runs happen in a background thread (agent/jobs.py), so the page shows live progress, an activity log
and a Stop button instead of freezing until the run is over.
"""
import html as _html
import inspect
import io
import json
import os
import pathlib
import re
import shutil
import time
import zipfile
from urllib.parse import urlparse
from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

ROOT = pathlib.Path(__file__).parent.resolve()
os.chdir(ROOT)  # agent/tools.py uses cwd/generated_project
PROJECT_DIR = ROOT / "generated_project"
PROJECT_DIR.mkdir(exist_ok=True)
SCAN_BASE = ROOT / "scan_workspace"   # uploaded code is copied here; originals are never modified

from agent.jobs import MANAGER  # noqa: E402
from agent.llm import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, is_local_url  # noqa: E402
from agent.rules import EXT_LANG  # noqa: E402
from agent.secure import (ScanConfig, UnsafeArchiveError, load_files, report_to_markdown,  # noqa: E402
                          reset_workspace, safe_extract_zip, save_uploaded_file)

EXAMPLES = [
    ("To-do app", "Create a to-do list application using html, css, and javascript."),
    ("Calculator", "Create a simple calculator web application."),
    ("Coffee shop page", "Create a colourful landing page for a coffee shop in html, css and js."),
    ("Pomodoro timer", "Create a pomodoro timer web app with start, pause, reset and a session counter, in html, css and js."),
    ("Quiz game", "Create a multiple-choice quiz game with 5 questions, a score and a restart button, in html, css and js."),
    ("Portfolio site", "Create a clean one-page personal portfolio website with about, projects and contact sections in html and css."),
]

# `use_container_width` is deprecated in newer Streamlit in favour of width="stretch".
STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".html": "html",
    ".css": "css", ".json": "json", ".md": "markdown", ".yml": "yaml",
    ".yaml": "yaml", ".sh": "bash", ".sql": "sql", ".txt": "text",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".java": "java",
}
FILE_ICONS = {".html": "🌐", ".css": "🎨", ".js": "🟨", ".ts": "🟦", ".py": "🐍", ".json": "🧾",
              ".md": "📝", ".c": "⚙️", ".cpp": "⚙️", ".java": "☕", ".sh": "🖥️", ".sql": "🗄️"}

STAGE_LABELS = {"planner": "Planner", "architect": "Architect", "coder": "Coder",
                "scanner": "Scanner", "fixer": "Fixer"}
STAGE_HINT = {"planner": "Plans the project", "architect": "Splits it into tasks", "coder": "Writes the files",
              "scanner": "Finds weaknesses", "fixer": "Patches and re-checks"}
UPLOAD_TYPES = sorted({"zip"} | {e.lstrip(".") for e in EXT_LANG if e != ".env"})

SEV_COLORS = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#38bdf8", "info": "#94a3b8"}
SEV_ORDER = ["critical", "high", "medium", "low", "info"]
SEV_DOT = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
STATUS_COLORS = {"open": "#f97316", "fixed": "#22c55e", "unresolved": "#ef4444", "new": "#a855f7"}

LOGO_SVG = (
    '<svg class="cb-logo" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c83ff"/>'
    '<stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>'
    '<rect width="64" height="64" rx="16" fill="url(#g)"/>'
    '<path d="M25 22 14 32l11 10M39 22l11 10-11 10" fill="none" stroke="#fff" stroke-width="5" '
    'stroke-linecap="round" stroke-linejoin="round"/><path d="M35 19 29 45" stroke="#fff" stroke-width="4" '
    'stroke-linecap="round" opacity=".85"/></svg>'
)


def esc(x) -> str:
    """HTML-escape a value and collapse its whitespace (it is placed inside one-line markup)."""
    return _html.escape(" ".join(str(x).split()), quote=True)


# ---------- helpers ----------
def _list_files(directory: pathlib.Path) -> list[pathlib.Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != ".gitkeep"
    )


def list_project_files() -> list[pathlib.Path]:
    return _list_files(PROJECT_DIR)


def clear_project():
    for item in PROJECT_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def make_zip(directory: pathlib.Path = PROJECT_DIR) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in _list_files(directory):
            zf.write(f, f.relative_to(directory))
    return buf.getvalue()


def md_escape(text: str) -> str:
    """Neutralise markdown in text that comes from scanned code or from the model."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$&:])", r"\\\1", str(text))


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def fmt_secs(s) -> str:
    if s is None:
        return ""
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def html_block(markup: str) -> None:
    """Render trusted-structure HTML. Every dynamic value inside must already be passed through esc()."""
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _css() -> str:
    return (ROOT / "ui" / "style.css").read_text(encoding="utf-8")


@st.cache_data(ttl=15, show_spinner=False)
def ollama_models(url: str) -> list[str] | None:
    """Names of the models installed in a LOCAL Ollama server, or None when it is not reachable."""
    try:
        with urlopen(url.rstrip("/") + "/api/tags", timeout=1.5) as r:  # noqa: S310 - local URL only (checked)
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m["name"] for m in data.get("models", []) if m.get("name"))
    except Exception:  # noqa: BLE001
        return None


# ---------- preview ----------
_EXTERNAL_REF = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*:|//)")


def build_preview_html() -> str | None:
    """Inline local CSS/JS into index.html so it renders inside an iframe."""
    index = PROJECT_DIR / "index.html"
    if not index.exists():
        return None
    html = index.read_text(encoding="utf-8", errors="replace")
    project_root = PROJECT_DIR.resolve()

    def read_local(ref: str) -> str | None:
        ref = ref.split("#", 1)[0].split("?", 1)[0].strip()
        if not ref or _EXTERNAL_REF.match(ref):
            return None  # remote URL (CDN etc.): leave the tag untouched
        while ref.startswith("./"):
            ref = ref[2:]
        ref = ref.lstrip("/")
        p = (project_root / ref).resolve()
        if project_root in p.parents and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        return None

    def css_sub(m):
        href = re.search(r'href=["\']([^"\']+)["\']', m.group(0))
        css = read_local(href.group(1)) if href else None
        return f"<style>{css}</style>" if css is not None else m.group(0)

    deferred_scripts: list[str] = []

    def js_sub(m):
        tag = m.group(0)
        open_tag = re.match(r"<script[^>]*>", tag, re.IGNORECASE).group(0)
        src = re.search(r'src=["\']([^"\']+)["\']', open_tag)
        js = read_local(src.group(1)) if src else None
        if js is None:
            return tag
        js = js.replace("</script", "<\\/script")  # don't let JS close the tag early
        type_m = re.search(r'type=["\']([^"\']+)["\']', open_tag, re.IGNORECASE)
        type_attr = f' type="{type_m.group(1)}"' if type_m else ""
        inline = f"<script{type_attr}>{js}</script>"
        is_module = bool(type_m and type_m.group(1).lower() == "module")
        # `defer` is ignored on inline scripts, so move such scripts to the end of <body>
        # so they still run after the DOM exists. Inline module scripts are deferred natively.
        if re.search(r"\sdefer(\s|=|>|/)", open_tag, re.IGNORECASE) and not is_module:
            deferred_scripts.append(inline)
            return ""
        return inline

    html = re.sub(r'<link[^>]*rel=["\']stylesheet["\'][^>]*>', css_sub, html, flags=re.IGNORECASE)
    html = re.sub(r'<script[^>]*src=["\'][^"\']+["\'][^>]*>\s*</script>', js_sub, html,
                  flags=re.IGNORECASE)
    if deferred_scripts:
        block = "\n".join(deferred_scripts)
        if re.search(r"</body>", html, re.IGNORECASE):
            html = re.sub(r"</body>", lambda _: block + "\n</body>", html, count=1, flags=re.IGNORECASE)
        else:
            html += block
    return html


def device_frame(doc: str, width: int | None, height: int = 620) -> str:
    """Wrap the generated page in a sandboxed iframe of a chosen width (scripts allowed, no same-origin)."""
    w = f"{width}px" if width else "100%"
    return (
        '<div style="display:flex;justify-content:center;padding:14px;background:#0b1020;border-radius:14px">'
        f'<iframe sandbox="allow-scripts allow-forms allow-modals" srcdoc="{_html.escape(doc, quote=True)}" '
        f'style="width:{w};max-width:100%;height:{height}px;border:0;border-radius:10px;background:#fff"></iframe></div>'
    )


# ---------- sidebar / settings ----------
def valid_url(url: str) -> bool:
    try:
        u = urlparse(url.strip())
    except ValueError:
        return False
    return u.scheme in ("http", "https") and bool(u.hostname)


st.set_page_config(page_title="Coder Buddy", page_icon="🛠️", layout="wide")
st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)

with st.sidebar:
    html_block('<div class="cb-brand">' + LOGO_SVG.replace('class="cb-logo"', 'class="cb-logo sm"')
               + '<b>Coder Buddy</b></div>')
    mode = st.radio("Mode", ["Generate project", "Scan existing code"], key="mode")

    st.markdown("##### Model")
    providers = ["groq", "ollama"]
    env_provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    provider = st.selectbox(
        "LLM provider", providers,
        index=providers.index(env_provider) if env_provider in providers else 0,
        format_func=lambda p: {"groq": "Groq (cloud, fast)", "ollama": "Ollama (local, private)"}[p],
        help="Ollama keeps your code on this machine. Groq sends it to a cloud API.",
    )
    api_key = ""
    ollama_model = ollama_url = ""
    if provider == "groq":
        api_key = st.text_input(
            "Groq API key", type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Get one at https://console.groq.com/keys (or put it in .env)",
        )
        st.caption("☁️ Code you generate or scan is sent to Groq.")
    else:
        ollama_url = st.text_input("Ollama server URL", value=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
        default_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        installed = ollama_models(ollama_url.strip()) if valid_url(ollama_url) and is_local_url(ollama_url) else None
        if installed:
            options = installed if default_model in installed else [default_model] + installed
            ollama_model = st.selectbox("Ollama model (installed)", options, index=options.index(default_model))
            st.caption(f"🟢 Ollama reachable · {len(installed)} model(s) installed")
        else:
            ollama_model = st.text_input("Ollama model", value=default_model,
                                         help="Must already be pulled, e.g. `ollama pull qwen2.5-coder:7b`")
            if valid_url(ollama_url) and is_local_url(ollama_url):
                st.caption("🔴 Ollama is not reachable. Start it with `ollama serve`.")
        if valid_url(ollama_url) and not is_local_url(ollama_url):
            st.warning("This URL is not a local address: your source code will leave this machine.")
        if re.search(r":\s*[0-3](\.\d)?b\b", ollama_model.lower()):
            st.caption("⚠️ Very small models (3B or less) are poor at planning and security analysis. "
                       "Use 7B+ for good results.")

    with st.expander("Run settings"):
        recursion_limit = st.number_input("Recursion limit", 20, 1000, 100, step=10,
                                          help="Maximum agent steps for one run.")
        if mode == "Generate project":
            clear_first = st.checkbox("Clear previous generated files first", value=True)
        else:
            clear_first = False

    st.markdown("##### Secure-coding scan")
    if mode == "Generate project":
        run_scan = st.checkbox("Run secure-coding scan after generation", value=False,
                               help="Scans the generated files and, if enabled below, fixes violations in place.")
    else:
        run_scan = True
    auto_fix = st.checkbox("Fix violations automatically", value=True,
                           help="Off = detect and report only; no file is modified.")
    use_llm = st.checkbox("Use the LLM (off = rule-based checks only)", value=True,
                          help="Rules-only is instant and works offline, but finds fewer issues.")
    max_rounds = st.number_input("Max fix rounds", 0, 5, int(os.environ.get("MAX_FIX_ROUNDS", "1") or 1))
    st.divider()
    st.caption("Generated files: `generated_project/`" if mode == "Generate project"
               else "Uploads are copied to `scan_workspace/`; your originals are never modified.")


def apply_model_settings(needs_llm: bool) -> str | None:
    """Push the sidebar choices into the environment (agent/llm.py reads it on every call).
    Returns an error message, or None when everything needed is present."""
    if not needs_llm:
        return None
    os.environ["LLM_PROVIDER"] = provider
    if provider == "groq":
        if not api_key.strip():
            return "Please enter your Groq API key in the sidebar (or set GROQ_API_KEY in .env)."
        os.environ["GROQ_API_KEY"] = api_key.strip()
    else:
        if not ollama_model.strip():
            return "Please enter the Ollama model name in the sidebar."
        if not valid_url(ollama_url):
            return "The Ollama server URL must start with http:// or https://"
        os.environ["OLLAMA_MODEL"] = ollama_model.strip()
        os.environ["OLLAMA_BASE_URL"] = ollama_url.strip()
    return None


def scan_options() -> dict:
    return {"auto_fix": bool(auto_fix), "use_llm": bool(use_llm), "max_fix_rounds": int(max_rounds)}


# ---------- hero ----------
def render_hero():
    if provider == "groq":
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        chips = [("warn", "Cloud · Groq"), ("", model)]
    else:
        local = valid_url(ollama_url) and is_local_url(ollama_url)
        chips = [("ok" if local else "warn", "Local · Ollama" if local else "Remote Ollama"), ("", ollama_model or "no model")]
    chips_html = "".join(f'<span class="cb-chip {c}"><span class="dot"></span>{esc(t)}</span>' for c, t in chips)
    html_block(
        f'<div class="cb-hero">{LOGO_SVG}<div class="cb-grow"><h1>Coder Buddy</h1>'
        '<p>Describe an app and get working files. Scan any code for security weaknesses and get verified fixes.</p></div>'
        f'<div class="cb-chips">{chips_html}</div></div>'
    )


# ---------- live run view ----------
def stepper_html(job) -> str:
    cells = []
    for i, name in enumerate(job.stage_order, start=1):
        state = job.stages.get(name, "waiting")
        sec = job.stage_seconds(name) if state in ("running", "done", "error") else None
        label = {"waiting": "Waiting", "running": "Working…", "done": "Done", "skipped": "Skipped", "error": "Failed"}[state]
        mark = "✓" if state == "done" else ("!" if state == "error" else str(i))
        extra = f" · {fmt_secs(sec)}" if sec is not None else ""
        cells.append(
            f'<div class="cb-step {state}"><span class="n">{mark}</span><span class="t">{STAGE_LABELS[name]}</span>'
            f'<div class="s">{esc(STAGE_HINT[name])}<br>{label}{extra}</div></div>')
    return f'<div class="cb-steps">{"".join(cells)}</div>'


def bar_html(pct: float | None, left: str, right: str = "") -> str:
    inner = f'<i style="width:{max(2, min(100, pct * 100)):.0f}%"></i>' if pct is not None else "<i></i>"
    return (f'<div class="cb-bar-label"><span>{esc(left)}</span><span>{esc(right)}</span></div>'
            f'<div class="cb-bar{"" if pct is not None else " indet"}">{inner}</div>')


def log_html(job, lines: int = 14) -> str:
    with job.lock:
        rows = list(job.log[-lines:])
    body = "".join(
        f'<div><span class="ts">{fmt_secs(t).rjust(5)}</span><span class="{lvl}">{esc(text)}</span></div>'
        for t, lvl, text in rows)
    return f'<div class="cb-term">{body or "Waiting for output…"}</div>'


def tasks_html(job) -> str:
    if not job.steps:
        return ""
    rows = []
    for i, s in enumerate(job.steps):
        if i < job.done_count:
            st_ = "✅"
        elif i == job.done_count and job.stages.get("coder") == "running":
            st_ = "⏳"
        else:
            st_ = "⬜"
        rows.append(f'<div class="cb-task"><span class="st">{st_}</span><code>{esc(s.filepath)}</code></div>')
    return f'<div class="cb-card"><h4>Files ({job.done_count}/{len(job.steps)})</h4>{"".join(rows)}</div>'


def plan_card_html(plan) -> str:
    feats = "".join(f"<li>{esc(f)}</li>" for f in plan.features[:8])
    return (f'<div class="cb-card"><h4>{esc(plan.name)}</h4><div class="muted">{esc(plan.description)}</div>'
            f'<div class="muted" style="margin-top:6px"><b>Stack:</b> {esc(plan.techstack)}</div>'
            f'<ul class="muted" style="margin:8px 0 0 16px;padding:0">{feats}</ul></div>')


@st.fragment(run_every=1.0)
def live_view():
    job = MANAGER.current
    if job is None:
        return
    if job.state != "running":
        st.rerun()          # run finished: redraw the whole page with the results
    title = "Scanning your code" if job.kind == "scan" else "Building your project"
    head, btn = st.columns([5, 1])
    head.markdown(f"### {title} · {fmt_secs(job.elapsed)}")
    btn.button("Stop", key="stop_job", on_click=MANAGER.cancel, disabled=job.cancelling, **STRETCH)
    html_block(stepper_html(job))

    # progress for the active stage
    active = next((n for n in job.stage_order if job.stages[n] == "running"), None)
    if active == "coder" and job.steps:
        html_block(bar_html(job.done_count / len(job.steps), f"Writing files · {job.current_file or ''}",
                            f"{job.done_count}/{len(job.steps)}"))
    elif active in ("scanner", "fixer") and job.progress_total:
        html_block(bar_html(job.progress_done / job.progress_total, job.progress_text or "Scanning",
                            f"{job.progress_done}/{job.progress_total}"))
    else:
        html_block(bar_html(None, job.progress_text or (STAGE_LABELS.get(active, "Working") + "…")))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Activity**")
        html_block(log_html(job))
    with right:
        if job.plan is not None:
            html_block(plan_card_html(job.plan))
        html_block(tasks_html(job))
    if job.cancelling:
        st.caption("Stopping after the current model call finishes…")


# ---------- results ----------
def metric(value, label, cls="") -> str:
    return f'<div class="cb-metric {cls}"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def render_banner(job):
    """Outcome of the latest run."""
    if job is None:
        return
    if job.state == "error":
        if "rate_limit" in job.error or "429" in job.error:
            hint = ("Groq rate limit (free tier tokens per minute). Wait a minute and press Generate again, "
                    "keep prompts short, or set GROQ_MODEL=llama-3.3-70b-versatile in .env. "
                    "A paid Groq tier or a local Ollama model removes the limit.")
        else:
            hint = ("Check the API key (Groq), or that Ollama is running and the model is pulled. "
                    "Small local models (3B or less) often fail at planning: try a 7B+ model.")
        html_block(f'<div class="cb-banner bad"><div>❌</div><div><b>The run failed</b>{esc(job.error)}'
                   f'<div class="cb-muted" style="margin-top:6px">{esc(hint)}</div></div></div>')
        if job.trace:
            with st.expander("Technical details"):
                st.code(job.trace)
    elif job.state == "cancelled":
        html_block('<div class="cb-banner warn"><div>⏹️</div><div><b>Stopped</b>'
                   'Files written before you pressed Stop are kept below.</div></div>')
    elif job.state == "done":
        what = (f"{len(job.files_written)} file(s) generated" if job.kind == "generate"
                else "Scan finished")
        html_block(f'<div class="cb-banner ok"><div>✅</div><div><b>{esc(what)} in {fmt_secs(job.elapsed)}</b>'
                   'Review the results below.</div></div>')


def render_empty_state():
    html_block(
        '<div class="cb-empty">'
        '<div class="cb-card"><div class="ic">💬</div><h4>1 · Describe it</h4><div class="muted">Write what you want in plain English, '
        'or pick an example below.</div></div>'
        '<div class="cb-card"><div class="ic">🧩</div><h4>2 · Three agents build it</h4><div class="muted">Planner, Architect and Coder '
        'create every file. Watch progress live.</div></div>'
        '<div class="cb-card"><div class="ic">🛡️</div><h4>3 · Secure it</h4><div class="muted">Scan for CWE / OWASP issues, '
        'auto-fix and re-verify each fix.</div></div></div>')


def file_label(p: pathlib.Path) -> str:
    return f"{FILE_ICONS.get(p.suffix.lower(), '📄')} {p.relative_to(PROJECT_DIR).as_posix()}"


def render_results():
    """Show file viewer, preview and download for whatever is in generated_project/."""
    files = list_project_files()
    if not files:
        return

    texts = {}
    for f in files:
        try:
            texts[f] = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            texts[f] = None
    total_lines = sum(t.count("\n") + 1 for t in texts.values() if t)
    total_bytes = sum(f.stat().st_size for f in files)

    top, dl = st.columns([4, 1])
    top.subheader("Generated project")
    dl.download_button("Download ZIP", make_zip(), file_name="generated_project.zip",
                       mime="application/zip", type="primary", **STRETCH)
    html_block('<div class="cb-metrics">' + metric(len(files), "Files") + metric(f"{total_lines:,}", "Lines of code")
               + metric(human_size(total_bytes), "Total size") + "</div>")

    rep = st.session_state.get("scan_report") if st.session_state.get("scan_source") == "generate" else None
    names = ["Files", "Live preview"] + ([f"Security ({rep.summary.get('remaining', 0)} open)"] if rep else []) + ["Plan"]
    tabs = st.tabs(names)
    tab_files, tab_preview, tab_plan = tabs[0], tabs[1], tabs[-1]
    if rep:
        with tabs[2]:
            render_security("generate", heading=False)

    with tab_files:
        c1, c2 = st.columns([1, 3])
        with c1:
            choice = st.radio("Files", files, format_func=file_label, label_visibility="collapsed", key="file_choice")
        with c2:
            content = texts.get(choice)
            rel = choice.relative_to(PROJECT_DIR).as_posix()
            html_block(f'<div class="cb-filehead"><code>{esc(rel)}</code>'
                       f'<span class="cb-muted">{human_size(choice.stat().st_size)} · '
                       f'{(content or "").count(chr(10)) + 1 if content else 0} lines</span></div>')
            if content is None:
                st.info("Binary file - cannot display.")
            else:
                st.code(content, language=LANG_BY_EXT.get(choice.suffix.lower(), "text"), line_numbers=True)
                st.download_button("Download this file", content, file_name=choice.name, key=f"dl_{rel}")

    with tab_preview:
        doc = build_preview_html()
        if doc is None:
            st.info("No index.html in the generated project, so there is nothing to preview. "
                    "Download the ZIP to run it.")
        else:
            size = st.radio("Viewport", ["Desktop", "Tablet", "Phone"], horizontal=True, key="viewport")
            width = {"Desktop": None, "Tablet": 768, "Phone": 390}[size]
            components.html(device_frame(doc, width), height=680, scrolling=False)
            st.caption("The preview runs the generated JavaScript in a sandboxed frame. Only preview code you are comfortable running.")

    with tab_plan:
        plan = st.session_state.get("last_plan")
        if plan is None:
            st.info("The plan is shown for the most recent generation in this session.")
        else:
            html_block(plan_card_html(plan))
            st.markdown("**Planned files**")
            for f in plan.files:
                st.markdown(f"- `{md_escape(f.path)}` — {md_escape(f.purpose)}")


# ---------- security report ----------
def render_security(source: str, heading: bool = True):
    """Show the secure-coding report stored by the last run of `source` ('generate' or 'scan')."""
    rep = st.session_state.get("scan_report")
    if rep is None or st.session_state.get("scan_source") != source:
        return

    if heading:
        st.subheader("Security report")
    s = rep.summary
    remaining = s.get("remaining", 0)
    if remaining == 0 and s.get("found", 0) == 0:
        html_block('<div class="cb-banner ok"><div>🛡️</div><div><b>No violations found</b>'
                   'Nothing was reported by the rules' + (" or the model." if rep.llm_used else ".") + '</div></div>')
    elif remaining == 0:
        html_block(f'<div class="cb-banner ok"><div>🛡️</div><div><b>All {s.get("found", 0)} finding(s) fixed and verified</b>'
                   'Each fix was re-scanned. Still review the diffs.</div></div>')
    else:
        html_block(f'<div class="cb-banner warn"><div>⚠️</div><div><b>{remaining} violation(s) still open</b>'
                   'See the list below for details and suggested fixes.</div></div>')

    html_block('<div class="cb-metrics">'
               + metric(s.get("found", 0), "Found")
               + metric(s.get("fixed", 0), "Fixed", "ok")
               + metric(s.get("unresolved", 0), "Unresolved", "bad" if s.get("unresolved", 0) else "")
               + metric(s.get("new", 0), "New after fix", "new" if s.get("new", 0) else "")
               + metric(remaining, "Remaining", "warn" if remaining else "ok") + '</div>')

    # severity distribution of what is still open
    open_v = [v for v in rep.violations if v.status in ("open", "unresolved", "new")]
    if open_v:
        counts = {sev: sum(v.severity == sev for v in open_v) for sev in SEV_ORDER}
        bar = "".join(f'<i style="width:{100 * c / len(open_v):.1f}%;background:{SEV_COLORS[sev]}"></i>'
                      for sev, c in counts.items() if c)
        legend = "".join(f'<span style="--c:{SEV_COLORS[sev]}">{sev.title()} {c}</span>'
                         for sev, c in counts.items() if c)
        html_block(f'<div class="cb-sev">{bar}</div><div class="cb-legend">{legend}</div>')

    if rep.llm_used:
        st.caption(f"LLM analysis: {rep.provider} / {rep.model} · fix rounds run: {rep.fix_rounds}")
    else:
        st.caption("LLM analysis was not used: results come from the rule-based checks only.")
    if rep.warnings:
        with st.expander(f"Notes and warnings ({len(rep.warnings)})", expanded=len(rep.warnings) <= 2):
            for w in rep.warnings:
                st.warning(md_escape(w))

    tab_v, tab_d, tab_s = st.tabs([f"Violations ({len(rep.violations)})", f"Changes ({len(rep.diffs)})",
                                   f"Skipped ({len(rep.skipped)})"])

    with tab_v:
        if not rep.violations:
            st.success("No violations found.")
        else:
            f1, f2, f3 = st.columns(3)
            sev_f = f1.multiselect("Severity", SEV_ORDER, default=SEV_ORDER, key=f"sev_{source}")
            stat_f = f2.multiselect("Status", list(STATUS_COLORS), default=list(STATUS_COLORS), key=f"stat_{source}")
            file_opts = sorted({v.file for v in rep.violations})
            file_f = f3.multiselect("File", file_opts, default=file_opts, key=f"file_{source}")
            shown = [v for v in rep.violations if v.severity in sev_f and v.status in stat_f and v.file in file_f]
            st.caption(f"Showing {len(shown)} of {len(rep.violations)}")
            for v in shown:
                loc = f"{v.line_start}" if v.line_start == v.line_end else f"{v.line_start}-{v.line_end}"
                label = f"{SEV_DOT.get(v.severity, '⚪')} {v.severity.upper()} · {md_escape(v.title)} — {md_escape(v.file)}:{loc} ({v.status})"
                with st.expander(label):
                    refs = " · ".join(esc(x) for x in (v.cwe, v.owasp, v.cert) if x)
                    html_block(
                        '<div class="cb-vio">'
                        f'<div class="row"><span class="cb-badge" style="background:{SEV_COLORS.get(v.severity, "#888")}">{esc(v.severity)}</span>'
                        f'<span class="cb-badge" style="background:{STATUS_COLORS.get(v.status, "#888")}">{esc(v.status)}</span>'
                        f'<span class="cb-muted">{esc(v.id or "-")} · found by {esc(v.source)}</span></div>'
                        f'<div class="row"><span class="k">References</span>{refs}</div>'
                        f'<div class="row"><span class="k">Why it matters</span>{esc(v.explanation or "n/a")}</div>'
                        f'<div class="row"><span class="k">Suggested fix</span>{esc(v.suggested_fix or "n/a")}</div></div>')
                    st.code(v.evidence, language=LANG_BY_EXT.get(pathlib.PurePosixPath(v.file).suffix.lower(), "text"))

    with tab_d:
        if not rep.diffs:
            st.info("No files were changed.")
        for path, d in rep.diffs.items():
            with st.expander(path, expanded=len(rep.diffs) == 1):
                st.code(d, language="diff")

    with tab_s:
        if not rep.skipped:
            st.info("No files were skipped.")
        for k in rep.skipped:
            st.text(f"{k.path}: {k.reason}")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Report (JSON)", rep.model_dump_json(indent=2), file_name="security_report.json",
                       mime="application/json", key=f"dl_json_{source}", **STRETCH)
    d2.download_button("Report (Markdown)", report_to_markdown(rep), file_name="security_report.md",
                       mime="text/markdown", key=f"dl_md_{source}", **STRETCH)
    scan_dir = st.session_state.get("scan_dir")
    if source == "scan" and rep.diffs and scan_dir and pathlib.Path(scan_dir).is_dir():
        d3.download_button("Fixed copy (ZIP)", make_zip(pathlib.Path(st.session_state.scan_dir)),
                           file_name="fixed_code.zip", mime="application/zip", key="dl_fixed_zip",
                           type="primary", **STRETCH)


# ---------- page ----------
render_hero()

job = MANAGER.current
if job is not None and job.state == "running":
    live_view()
    st.stop()

# A run just ended: keep its results in this session.
if job is not None and st.session_state.get("ingested") != job.id:
    st.session_state.ingested = job.id
    if job.report is not None:
        st.session_state.scan_report = job.report
        st.session_state.scan_source = job.kind
    if job.plan is not None:
        st.session_state.last_plan = job.plan
mode_kind = "generate" if mode == "Generate project" else "scan"
last_job = job if (job is not None and job.kind == mode_kind) else None


def start_job(kind, graph_agent, inputs, stages):
    started = MANAGER.start(kind, graph_agent, inputs, stages, int(recursion_limit))
    if started is None:
        st.warning("Another run is already in progress. Please wait for it to finish and try again.")
        st.stop()
    st.rerun()


def _set_prompt(text: str):
    st.session_state.prompt = text


# ---------- mode: generate ----------
if mode == "Generate project":
    # Streamlit forgets the value of a widget that was not drawn in the previous run (the live view hides it).
    st.session_state.prompt = st.session_state.get("prompt", "")

    have_project = bool(list_project_files())
    if not have_project and last_job is None:
        render_empty_state()

    # Once there is a result, fold the input away so the result is the first thing you see.
    box = st.expander("New project", expanded=not have_project, icon=":material/edit:") if have_project \
        else st.container()
    with box:
        st.markdown("#### What do you want to build?")
        prompt = st.text_area("Project prompt", key="prompt", height=110, label_visibility="collapsed",
                              placeholder="e.g. Create a to-do list app with html, css and javascript. "
                                          "Name the files you want for best results.")
        cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(cols, EXAMPLES):
            col.button(name, key=f"ex_{name}", on_click=_set_prompt, args=(text,), help=text, **STRETCH)

        go = st.button("Generate", type="primary", icon=":material/auto_awesome:")
        if run_scan:
            st.caption("🛡️ A secure-coding scan will run on the result.")

    if go:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            st.stop()
        err = apply_model_settings(needs_llm=True)
        if err:
            st.error(err)
            st.stop()
        if MANAGER.busy:
            st.warning("Another run is already in progress. Please wait for it to finish and try again.")
            st.stop()

        if clear_first:
            clear_project()
        for k in ("scan_report", "last_plan"):
            st.session_state.pop(k, None)
        try:
            from agent.graph import agent
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load the agent: {e}")
            st.stop()

        stages = ["planner", "architect", "coder"] + (["scanner", "fixer"] if run_scan else [])
        inputs = {"user_prompt": prompt.strip()}
        if run_scan:
            inputs["secure_scan"] = True
            inputs["scan_options"] = scan_options()
            inputs["scan_root"] = str(PROJECT_DIR)
        start_job("generate", agent, inputs, stages)

    st.markdown("---")
    render_banner(last_job)
    render_results()

# ---------- mode: scan existing code ----------
else:
    if last_job is None and st.session_state.get("scan_report") is None:
        render_empty_state()
    st.markdown("#### Upload code to scan")
    st.caption("Upload source files or a **.zip** of a project. A copy is scanned (and fixed, if enabled); "
               "your original files are never touched.")
    uploads = st.file_uploader("Source files or .zip", type=UPLOAD_TYPES, accept_multiple_files=True,
                               label_visibility="collapsed")
    if uploads:
        st.caption(f"{len(uploads)} file(s) selected · {human_size(sum(u.size for u in uploads))}")
    go = st.button("Scan", type="primary", icon=":material/shield:")

    if go:
        if not uploads:
            st.warning("Please upload at least one file.")
            st.stop()
        err = apply_model_settings(needs_llm=bool(use_llm))
        if err:
            st.error(err)
            st.stop()
        if MANAGER.busy:
            st.warning("Another run is already in progress. Please wait for it to finish and try again.")
            st.stop()

        st.session_state.pop("scan_report", None)
        opts = scan_options()
        cfg = ScanConfig.from_env(opts)
        ws = reset_workspace(SCAN_BASE, "upload")
        try:
            for up in uploads:
                if up.name.lower().endswith(".zip"):
                    safe_extract_zip(up.getvalue(), ws, cfg)
                else:
                    save_uploaded_file(up.name, up.getvalue(), ws, cfg)
        except UnsafeArchiveError as e:
            shutil.rmtree(ws, ignore_errors=True)
            st.error(f"Upload rejected: {e}")
            st.stop()

        found, _ = load_files(ws, cfg)
        if not found:
            st.error("No scannable source files were found in the upload.")
            st.stop()

        try:
            from agent.graph import scan_agent
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load the agent: {e}")
            st.stop()

        st.session_state.scan_dir = str(ws)
        start_job("scan", scan_agent, {"scan_root": str(ws), "scan_options": opts}, ["scanner", "fixer"])

    st.markdown("---")
    render_banner(last_job)
    render_security("scan")
    if st.session_state.get("scan_dir") and pathlib.Path(st.session_state.scan_dir).is_dir():
        if st.button("Delete the uploaded copy from the server"):
            shutil.rmtree(st.session_state.scan_dir, ignore_errors=True)
            st.session_state.pop("scan_dir", None)
            st.info("Uploaded files deleted.")
