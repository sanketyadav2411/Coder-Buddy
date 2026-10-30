"""Hallucination guards, patch application, syntax gate, chunking and the scan/fix engine."""
import json
from pathlib import Path

import pytest

from agent.secure import (ScanConfig, ScanContext, _atomic_write, _safe_target, apply_patches, build_report,
                          check_syntax, clean_replacement, cluster_violations, fix_files, join_text,
                          load_files, locate_evidence, make_diffs, make_windows, merge_violations, numbered,
                          report_to_markdown, scan_files, split_text, validate_llm_violation, _norm)
from agent.states import LLMViolation, Patch, Violation
from conftest import VULNERABLE, FakeLLM, symlink_or_skip


def P(s, e, r="", vid=None):
    return Patch(start_line=s, end_line=e, replacement=r, violation_id=vid)


# ---- text helpers ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["a\nb\nc", "a\nb\nc\n", "a\r\nb\r\n", "", "\n", "one"])
def test_split_join_round_trip(text):
    lines, eol, trailing = split_text(text)
    assert join_text(lines, eol, trailing) == text


def test_numbered_uses_absolute_line_numbers():
    out = numbered(["a", "b", "c", "d"], 3, 4)
    assert out.splitlines() == ["3 | c", "4 | d"]


def test_make_windows_cover_every_line_with_overlap():
    lines = [f"line {i}" for i in range(1, 501)]
    wins = make_windows(lines, max_lines=100, max_chars=100000, overlap=10)
    covered = set()
    for s, e in wins:
        assert 1 <= s <= e <= 500 and (e - s + 1) <= 100
        covered.update(range(s, e + 1))
    assert covered == set(range(1, 501))
    assert all(wins[i + 1][0] <= wins[i][1] for i in range(len(wins) - 1))   # windows overlap
    assert all(wins[i + 1][0] > wins[i][0] for i in range(len(wins) - 1))    # and always progress


def test_make_windows_respects_character_budget_and_terminates():
    lines = ["x" * 500] * 40
    wins = make_windows(lines, max_lines=1000, max_chars=2000, overlap=3)
    assert all(sum(len(lines[i - 1]) + 8 for i in range(s, e + 1)) <= 2000 + 508 for s, e in wins)
    assert wins[-1][1] == 40


def test_make_windows_single_overlong_line_still_progresses():
    assert make_windows(["x" * 100000, "y"], 10, 1000, 2)[-1][1] == 2


# ---- evidence anchoring (the anti-hallucination gate) --------------------------------------------------
LINES = ["import os", "def f(x):", "    return eval(x)", "", "def g(x):", "    return eval(x)"]
NORM = [_norm(l) for l in LINES]


def test_locate_evidence_exact_and_nearest_to_hint():
    assert locate_evidence(NORM, "return eval(x)", 3) == (3, 3)
    assert locate_evidence(NORM, "return eval(x)", 6) == (6, 6)       # two matches: nearest the claimed line wins


def test_locate_evidence_strips_copied_line_numbers_and_whitespace():
    assert locate_evidence(NORM, "3 |     return   eval(x)", 1) == (3, 3)


def test_locate_evidence_rejects_made_up_code():
    assert locate_evidence(NORM, "os.system(cmd)", 3) is None
    assert locate_evidence(NORM, "", 3) is None
    assert locate_evidence(NORM, "x", 3) is None                      # too short to be meaningful


def test_locate_evidence_multiline():
    assert locate_evidence(NORM, "def f(x):\n    return eval(x)", 1) == (2, 3)


def test_validate_llm_violation_drops_hallucinations_and_snaps_lines():
    bad = LLMViolation(line_start=3, cwe="CWE-95", evidence="subprocess.call(cmd, shell=True)")
    assert validate_llm_violation(bad, "a.py", LINES, NORM) is None
    wrong_line = LLMViolation(line_start=99, line_end=99, cwe="CWE-95", severity="high",
                              evidence="return eval(x)", title="eval")
    v = validate_llm_violation(wrong_line, "a.py", LINES, NORM)
    assert v is not None and (v.line_start, v.line_end) == (6, 6)     # the model's line 99 is corrected
    assert v.source == "llm" and v.evidence == "return eval(x)"


def test_validate_llm_violation_masks_secrets_in_evidence():
    lines = ['API_KEY = "sk_live_abcdef123456"']
    lv = LLMViolation(line_start=1, cwe="CWE-798", evidence='API_KEY = "sk_live_abcdef123456"')
    v = validate_llm_violation(lv, "a.py", lines, [_norm(l) for l in lines])
    assert "abcdef123456" not in v.evidence


def test_merge_violations_prefers_rules_and_collapses_llm_duplicates():
    rule = Violation(file="a.py", line_start=3, line_end=3, cwe="CWE-95", evidence="e", source="rule", severity="high")
    llm_same = Violation(file="a.py", line_start=3, line_end=3, cwe="CWE-95", evidence="e", source="llm", severity="critical")
    llm_other = Violation(file="a.py", line_start=5, line_end=5, cwe="CWE-89", evidence="f", source="llm")
    out = merge_violations([rule], [llm_same, llm_other, llm_other])
    assert [(v.line_start, v.source) for v in out] == [(3, "rule"), (5, "llm")]


# ---- clean_replacement ---------------------------------------------------------------------------------
def test_clean_replacement_strips_fences_and_line_number_prefixes():
    assert clean_replacement("```python\nx = 1\ny = 2\n```") == ["x = 1", "y = 2"]
    assert clean_replacement("12 |     x = 1\n13 |     y = 2") == ["    x = 1", "    y = 2"]
    assert clean_replacement("") == []
    assert clean_replacement("a\n") == ["a"]
    assert clean_replacement("x = 1 | 2") == ["x = 1 | 2"]            # not a line-number prefix


# ---- apply_patches -----------------------------------------------------------------------------------
BASE = ["l1", "l2", "l3", "l4", "l5"]


def test_apply_replace_and_does_not_mutate_input():
    snapshot = list(BASE)
    out, applied, rejected = apply_patches(BASE, [P(2, 2, "NEW")], 60)
    assert out == ["l1", "NEW", "l3", "l4", "l5"] and BASE == snapshot and not rejected and len(applied) == 1


def test_apply_insert_before_line_and_at_end():
    assert apply_patches(BASE, [P(1, 0, "import x")], 60)[0][:2] == ["import x", "l1"]
    assert apply_patches(BASE, [P(6, 5, "tail")], 60)[0][-1] == "tail"


def test_apply_multiple_patches_bottom_up_keeps_numbers_valid():
    out, applied, _ = apply_patches(BASE, [P(1, 0, "head"), P(2, 3, "A\nB\nC"), P(5, 5, "")], 60)
    assert out == ["head", "l1", "A", "B", "C", "l4"] and len(applied) == 3


def test_apply_rejects_overlaps_out_of_range_and_oversized():
    out, applied, rejected = apply_patches(BASE, [P(2, 3, "X"), P(3, 4, "Y")], 60)
    assert len(applied) == 1 and "overlaps" in rejected[0][1]
    for bad in (P(0, 1, "x"), P(7, 7, "x"), P(2, 9, "x"), P(3, 1, "x")):
        assert apply_patches(BASE, [bad], 60)[2], bad
    assert "spans more than" in apply_patches(BASE, [P(1, 5, "x")], 3)[2][0][1]


def test_insert_inside_a_replaced_range_conflicts():
    _, applied, rejected = apply_patches(BASE, [P(2, 4, "X"), P(3, 2, "ins")], 60)
    assert len(applied) == 1 and len(rejected) == 1


def test_two_inserts_at_same_position_conflict_but_adjacent_do_not():
    assert len(apply_patches(BASE, [P(2, 1, "a"), P(2, 1, "b")], 60)[1]) == 1
    assert len(apply_patches(BASE, [P(2, 1, "a"), P(3, 2, "b")], 60)[1]) == 2


# ---- syntax gate ---------------------------------------------------------------------------------------
def test_check_syntax_python():
    assert check_syntax("a.py", "x = 1\n", "x = 2\n")[0]
    ok, msg = check_syntax("a.py", "def f(:\n", "x = 1\n")
    assert not ok and "SyntaxError" in msg


def test_check_syntax_is_skipped_when_original_was_already_broken():
    ok, msg = check_syntax("a.py", "def f(:\n", "def g(:\n")
    assert ok and "already had syntax errors" in msg


def test_check_syntax_json_and_unchecked_languages():
    assert not check_syntax("a.json", "{bad", "{}")[0]
    assert check_syntax("a.json", "{}", "{}")[0]
    assert check_syntax("a.c", "int main( {", "x")[0]                 # no reliable checker: pass through


def test_check_syntax_javascript_uses_node_when_available():
    import shutil
    if not shutil.which("node"):
        pytest.skip("node not installed")
    assert check_syntax("a.js", "const a = 1;\n", "const a = 2;\n")[0]
    assert not check_syntax("a.js", "const a = ;\n", "const a = 2;\n")[0]


# ---- scan_files with a fake model ------------------------------------------------------------------------
def _cfg(**kw):
    return ScanConfig.from_env(kw)


def _vuln_py():
    return {"vuln.py": (VULNERABLE / "vuln.py").read_text(encoding="utf-8")}


def test_scan_files_rules_only_when_llm_disabled():
    cfg = _cfg(use_llm=False)
    vios = scan_files(_vuln_py(), cfg, ScanContext())
    assert {v.rule_id for v in vios} >= {"GEN001", "PY003"} and all(v.source == "rule" for v in vios)


def test_scan_files_drops_hallucinated_llm_findings_and_keeps_real_ones():
    reply = json.dumps({"violations": [
        {"line_start": 1, "cwe": "CWE-78", "severity": "high", "evidence": "os.system(user_input)", "title": "made up"},
        {"line_start": 5, "cwe": "CWE-330", "severity": "low", "evidence": "pickle.loads(blob)", "title": "real"},
    ]})
    ctx = ScanContext(llm=FakeLLM([reply]))
    vios = scan_files(_vuln_py(), _cfg(), ctx)
    llm_vios = [v for v in vios if v.source == "llm"]
    titles = {v.title for v in llm_vios}
    assert "made up" not in titles and "real" in titles
    assert all(v.line_start > 1 for v in llm_vios)


def test_scan_files_survives_garbage_model_output():
    ctx = ScanContext(llm=FakeLLM(["total nonsense"]))
    vios = scan_files(_vuln_py(), _cfg(llm_retries=1), ctx)
    assert vios and ctx.bad_output >= 1 and any("unusable output" in w for w in ctx.warnings)


def test_scan_files_degrades_to_rules_when_server_is_down():
    ctx = ScanContext(llm=FakeLLM([ConnectionError("refused")]))
    files = {"a.py": "x = eval(y)\n", "b.py": "z = eval(w)\n", "c.py": "q = eval(r)\n"}
    vios = scan_files(files, _cfg(), ctx)
    assert len(vios) == 3                                             # rule findings still reported
    assert ctx.dead and any("LLM unavailable" in w for w in ctx.warnings)


def test_scan_files_chunks_large_files_with_absolute_line_numbers():
    body = "\n".join(f"x{i} = {i}" for i in range(1, 300)) + "\nresult = eval(data)\n"
    seen_windows = []

    def script(prompt):
        seen_windows.append("showing lines" in prompt)
        return json.dumps({"violations": []})

    scan_files({"big.py": body}, _cfg(chunk_lines=100), ScanContext(llm=FakeLLM(script)))
    assert len(seen_windows) >= 3 and all(seen_windows)


def test_scan_files_ignores_empty_and_unsupported_files():
    assert scan_files({"a.py": "  \n", "n.txt": "eval(x)"}, _cfg(), ScanContext()) == []


# ---- fix_files ----------------------------------------------------------------------------------------------
def _violation(path, text, needle, vid="V001", cwe="CWE-95"):
    lines = text.split("\n")
    n = next(i for i, l in enumerate(lines, 1) if needle in l)
    return Violation(id=vid, file=path, line_start=n, line_end=n, cwe=cwe, severity="high",
                     title="t", evidence=lines[n - 1].strip(), suggested_fix="fix it")


def _setup(tmp_path, text="import os\n\ndef calc(expr):\n    return eval(expr)\n"):
    (tmp_path / "calc.py").write_text(text, encoding="utf-8")
    return tmp_path, _violation("calc.py", text, "eval(expr)")


def test_fix_files_applies_a_valid_patch_and_backs_up_the_original(tmp_path):
    root, v = _setup(tmp_path)
    reply = json.dumps({"patches": [{"violation_id": "V001", "start_line": 4, "end_line": 4,
                                     "replacement": "    return ast.literal_eval(expr)"},
                                    {"violation_id": "V001", "start_line": 1, "end_line": 0,
                                     "replacement": "import ast"}]})
    notes, backup = [], tmp_path / "bk"
    changed = fix_files(root, [v], _cfg(), FakeLLM([reply]), notes, backup)
    assert changed == ["calc.py"]
    assert (root / "calc.py").read_text() == "import ast\nimport os\n\ndef calc(expr):\n    return ast.literal_eval(expr)\n"
    assert (backup / "calc.py").read_text().endswith("return eval(expr)\n")


def test_fix_files_rolls_back_a_patch_that_breaks_syntax(tmp_path):
    root, v = _setup(tmp_path)
    original = (root / "calc.py").read_text()
    reply = json.dumps({"patches": [{"start_line": 4, "end_line": 4, "replacement": "    return (("}]})
    notes = []
    assert fix_files(root, [v], _cfg(), FakeLLM([reply]), notes) == []
    assert (root / "calc.py").read_text() == original
    assert any("syntax check" in n for n in notes)


def test_fix_files_rejects_patches_outside_the_window_the_model_saw(tmp_path):
    text = "\n".join(f"v{i} = {i}" for i in range(1, 200)) + "\nr = eval(x)\n"
    (tmp_path / "big.py").write_text(text)
    v = _violation("big.py", text, "eval(x)")
    reply = json.dumps({"patches": [{"start_line": 1, "end_line": 1, "replacement": "# vandalised"}]})
    notes = []
    assert fix_files(tmp_path, [v], _cfg(), FakeLLM([reply]), notes) == []
    assert (tmp_path / "big.py").read_text() == text and any("outside" in n for n in notes)


def test_fix_files_preserves_crlf_and_trailing_newline(tmp_path):
    text = "def f(x):\r\n    return eval(x)\r\n"
    (tmp_path / "w.py").write_bytes(text.encode())
    v = _violation("w.py", text.replace("\r\n", "\n"), "eval(x)")
    reply = json.dumps({"patches": [{"start_line": 2, "end_line": 2, "replacement": "    return int(x)"}]})
    assert fix_files(tmp_path, [v], _cfg(), FakeLLM([reply]), []) == ["w.py"]
    assert (tmp_path / "w.py").read_bytes() == b"def f(x):\r\n    return int(x)\r\n"


def test_fix_files_stops_after_repeated_llm_errors(tmp_path):
    for n in "abc":
        (tmp_path / f"{n}.py").write_text("x = eval(y)\n")
    vios = [_violation(f"{n}.py", "x = eval(y)\n", "eval", vid=f"V00{i}") for i, n in enumerate("abc")]
    llm = FakeLLM([ConnectionError("down")])
    notes = []
    assert fix_files(tmp_path, vios, _cfg(), llm, notes) == []
    assert len(llm.calls) == 2 and any("stopped" in n or "LLM error" in n for n in notes)


def test_fix_files_never_writes_outside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.py").write_text("x = 1\n")
    v = Violation(id="V1", file="../secret.py", line_start=1, line_end=1, evidence="x = 1")
    notes = []
    assert fix_files(root, [v], _cfg(), FakeLLM(['{"patches": []}']), notes) == []
    assert (tmp_path / "secret.py").read_text() == "x = 1\n"


def test_safe_target_refuses_traversal_and_symlinks(tmp_path):
    with pytest.raises(ValueError):
        _safe_target(tmp_path, "../x.py")
    (tmp_path / "real.py").write_text("x")
    symlink_or_skip(tmp_path / "link.py", tmp_path / "real.py")
    with pytest.raises(ValueError):
        _safe_target(tmp_path, "link.py")


def test_atomic_write_replaces_content(tmp_path):
    t = tmp_path / "f.py"
    t.write_text("old")
    _atomic_write(t, "new")
    assert t.read_text() == "new" and [p.name for p in tmp_path.iterdir()] == ["f.py"]


def test_cluster_violations_groups_nearby_and_splits_distant():
    lines = [f"l{i}" for i in range(1, 1001)]
    vs = [Violation(file="a.py", line_start=n, line_end=n) for n in (10, 15, 800)]
    clusters = cluster_violations(vs, lines, _cfg(chunk_lines=200))
    assert [len(c[0]) for c in clusters] == [2, 1]


# ---- load_files ------------------------------------------------------------------------------------------------
def test_load_files_filters_and_records_reasons(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "notes.txt").write_text("ignored: unsupported type")
    (tmp_path / "bin.py").write_bytes(b"\x00\x01\x02")
    (tmp_path / "latin.py").write_bytes("caf\xe9 = 1".encode("latin-1"))
    (tmp_path / "huge.py").write_text("x = 1\n" * 100000)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("eval(x)")
    (tmp_path / "real.py").write_text("y = 2\n")
    symlink_or_skip(tmp_path / "link.py", tmp_path / "real.py")
    files, skipped = load_files(tmp_path, _cfg(max_file_kb=100))
    assert set(files) == {"ok.py", "real.py"}
    reasons = {s.path: s.reason for s in skipped}
    assert "binary" in reasons["bin.py"] and "UTF-8" in reasons["latin.py"]
    assert "larger" in reasons["huge.py"] and "symbolic" in reasons["link.py"]
    assert "node_modules/dep.js" not in files and "notes.txt" not in reasons


def test_load_files_enforces_file_count(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x = 1\n")
    files, skipped = load_files(tmp_path, _cfg(max_files=3))
    assert len(files) == 3 and len(skipped) == 2 and "file limit" in skipped[0].reason


def test_load_files_on_missing_directory(tmp_path):
    assert load_files(tmp_path / "nope", _cfg()) == ({}, [])


# ---- config clamping --------------------------------------------------------------------------------------------
def test_scan_config_clamps_dangerous_values(monkeypatch):
    monkeypatch.setenv("MAX_FIX_ROUNDS", "9999")
    monkeypatch.setenv("SCAN_MAX_FILES", "not a number")
    cfg = ScanConfig.from_env({"chunk_lines": 1, "llm_retries": 500, "unknown_key": 1})
    assert cfg.max_fix_rounds == 5 and cfg.max_files == 200
    assert cfg.chunk_lines == 40 and cfg.llm_retries == 5
    assert ScanConfig.from_env({"max_fix_rounds": 0}).max_fix_rounds == 0


# ---- reports ---------------------------------------------------------------------------------------------------------
def _mk(path, cwe, ev, vid):
    return Violation(id=vid, file=path, line_start=1, line_end=1, cwe=cwe, severity="high", evidence=ev, title=cwe)


def test_build_report_statuses_and_summary():
    base = [_mk("a.py", "CWE-95", "eval(x)", "V001"), _mk("a.py", "CWE-89", "execute(q)", "V002")]
    current = [_mk("a.py", "CWE-89", "execute(q)", "V002"), _mk("a.py", "CWE-78", "system(c)", "N001")]
    rep = build_report(root=Path("."), cfg=_cfg(use_llm=False), files={"a.py": "x"}, skipped=[], baseline=base,
                       current=current, fix_rounds=1, originals={}, warnings=["w", "w"], llm_used=False,
                       first_round=False)
    status = {v.id: v.status for v in rep.violations}
    assert status == {"V001": "fixed", "V002": "unresolved", "N001": "new"}
    assert rep.summary["found"] == 2 and rep.summary["fixed"] == 1 and rep.summary["remaining"] == 2
    assert rep.warnings == ["w"]


def test_first_round_report_marks_everything_open():
    base = [_mk("a.py", "CWE-95", "eval(x)", "V001")]
    rep = build_report(root=Path("."), cfg=_cfg(use_llm=False), files={"a.py": "x"}, skipped=[], baseline=base,
                       current=base, fix_rounds=0, originals={}, warnings=[], llm_used=False, first_round=True)
    assert rep.violations[0].status == "open" and rep.summary["remaining"] == 1


def test_diffs_and_markdown_never_leak_secrets():
    before = 'API_KEY = "sk_live_abcdef123456"\nx = 1\n'
    after = 'API_KEY = os.environ["API_KEY"]\nx = 1\n'
    diffs = make_diffs({"a.py": before, "same.py": "x"}, {"a.py": after, "same.py": "x"})
    assert set(diffs) == {"a.py"} and "abcdef123456" not in diffs["a.py"]
    rep = build_report(root=Path("."), cfg=_cfg(use_llm=False), files={"a.py": after}, skipped=[],
                       baseline=[_mk("a.py", "CWE-798", "API_KEY = \"sk***\"", "V001")], current=[],
                       fix_rounds=1, originals={"a.py": before}, warnings=[], llm_used=False, first_round=False)
    md = report_to_markdown(rep)
    assert "abcdef123456" not in md and "V001" in md and "Changes applied" in md


def test_markdown_report_cannot_be_broken_out_of_by_evidence():
    v = _mk("a.py", "CWE-1", "```\n# injected heading", "V001")
    rep = build_report(root=Path("."), cfg=_cfg(use_llm=False), files={}, skipped=[], baseline=[v], current=[v],
                       fix_rounds=0, originals={}, warnings=[], llm_used=False, first_round=True)
    assert "```\n# injected" not in report_to_markdown(rep)
