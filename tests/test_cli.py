import json
import os

import pytest

import agent.secure as S
import main as cli
from conftest import CLEAN, VULNERABLE, FakeLLM


@pytest.fixture(autouse=True)
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # main.apply_llm_args writes os.environ; pre-set the keys so monkeypatch restores them afterwards
    for k in ("LLM_PROVIDER", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.setenv(k, os.environ.get(k, "x") if k != "LLM_PROVIDER" else "groq")
    monkeypatch.delenv("OLLAMA_MODEL")
    monkeypatch.delenv("OLLAMA_BASE_URL")
    monkeypatch.delenv("GROQ_MODEL")
    return tmp_path


def run_main(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


# ---- argument parsing ---------------------------------------------------------------------------------
def test_original_interface_is_preserved():
    args = cli.build_parser().parse_args(["-r", "150"])
    assert args.recursion_limit == 150 and args.prompt is None and not args.secure and args.scan is None
    assert cli.build_parser().parse_args([]).recursion_limit == 100


def test_new_flags_parse():
    a = cli.build_parser().parse_args(["--scan", "x", "--no-fix", "--no-llm", "--max-rounds", "3",
                                       "--provider", "ollama", "--model", "m", "--base-url", "http://h:1",
                                       "--out", "o", "--json"])
    assert (a.scan, a.no_fix, a.no_llm, a.max_rounds, a.provider, a.model, a.base_url, a.out, a.json) == \
           ("x", True, True, 3, "ollama", "m", "http://h:1", "o", True)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--provider", "openai"])


def test_conflicting_modes_are_rejected(capsys):
    code, _, err = run_main(["--scan", ".", "--prompt", "x"], capsys)
    assert code == 2 and "cannot be combined" in err
    assert run_main(["--scan", ".", "--secure"], capsys)[0] == 2


def test_max_rounds_is_range_checked(capsys):
    code, _, err = run_main(["--scan", ".", "--max-rounds", "9"], capsys)
    assert code == 2 and "between 0 and 5" in err


def test_llm_args_override_the_environment():
    args = cli.build_parser().parse_args(["--provider", "ollama", "--model", "llama3", "--base-url", "http://h:1"])
    cli.apply_llm_args(args)
    assert (os.environ["LLM_PROVIDER"], os.environ["OLLAMA_MODEL"], os.environ["OLLAMA_BASE_URL"]) == \
           ("ollama", "llama3", "http://h:1")
    args = cli.build_parser().parse_args(["--provider", "groq", "--model", "other"])
    cli.apply_llm_args(args)
    assert os.environ["GROQ_MODEL"] == "other"


# ---- scan mode -----------------------------------------------------------------------------------------------
def test_scan_missing_path_is_an_error(capsys):
    code, _, err = run_main(["--scan", "does-not-exist"], capsys)
    assert code == 1 and "does not exist" in err


def test_scan_with_no_scannable_files_is_an_error(in_tmp, capsys):
    (in_tmp / "empty").mkdir()
    (in_tmp / "empty" / "notes.txt").write_text("hi")
    code, _, err = run_main(["--scan", "empty"], capsys)
    assert code == 1 and "no scannable source files" in err


def test_scan_rules_only_exit_code_and_original_untouched(in_tmp, capsys):
    src = in_tmp / "src"
    src.mkdir()
    (src / "vuln.py").write_text((VULNERABLE / "vuln.py").read_text())
    before = (src / "vuln.py").read_text()
    code, out, _ = run_main(["--scan", "src", "--no-llm", "--no-fix", "--out", "copy"], capsys)
    assert code == 2                                                       # violations remain: CI-friendly
    assert "Remaining" in out and "CWE-798" in out
    assert (src / "vuln.py").read_text() == before and (in_tmp / "copy" / "vuln.py").exists()


def test_scan_clean_code_exits_zero(in_tmp, capsys):
    src = in_tmp / "src"
    src.mkdir()
    (src / "clean.py").write_text((CLEAN / "clean.py").read_text())
    code, out, _ = run_main(["--scan", "src", "--no-llm", "--out", "copy"], capsys)
    assert code == 0 and "Found         : 0" in out


def test_scan_single_file(in_tmp, capsys):
    (in_tmp / "one.py").write_text("x = eval(y)\n")
    code, out, _ = run_main(["--scan", "one.py", "--no-llm", "--out", "copy"], capsys)
    assert code == 2 and (in_tmp / "copy" / "one.py").exists()


def test_scan_json_output_is_valid_json(in_tmp, capsys):
    (in_tmp / "one.py").write_text("x = eval(y)\n")
    code, out, _ = run_main(["--scan", "one.py", "--no-llm", "--out", "copy", "--json"], capsys)
    data = json.loads(out[out.index("{"):])
    assert data["summary"]["found"] == 1 and data["violations"][0]["cwe"] == "CWE-95"


def test_out_directory_is_never_overwritten(in_tmp, capsys):
    (in_tmp / "one.py").write_text("x = eval(y)\n")
    (in_tmp / "copy").mkdir()
    (in_tmp / "copy" / "precious.txt").write_text("keep")
    code, _, err = run_main(["--scan", "one.py", "--no-llm", "--out", "copy"], capsys)
    assert code == 1 and "not empty" in err and (in_tmp / "copy" / "precious.txt").read_text() == "keep"


def test_scan_and_fix_a_copy_with_a_model(in_tmp, capsys, monkeypatch):
    src = in_tmp / "src"
    src.mkdir()
    (src / "calc.py").write_text("def calc(expr):\n    return eval(expr)\n")

    def script(prompt):
        if "SECURE-CODING FIXER" in prompt:
            return json.dumps({"patches": [{"start_line": 2, "end_line": 2, "replacement": "    return int(expr)"}]})
        return json.dumps({"violations": []})

    monkeypatch.setattr(S, "get_llm", lambda *a, **k: FakeLLM(script))
    code, out, _ = run_main(["--scan", "src", "--out", "fixed"], capsys)
    assert code == 0 and "Fixed         : 1" in out
    assert (in_tmp / "fixed" / "calc.py").read_text().endswith("return int(expr)\n")
    assert (src / "calc.py").read_text().endswith("return eval(expr)\n")  # the original is never modified


def test_scan_default_workspace_name_is_sanitised(in_tmp, capsys):
    odd = in_tmp / "we ird$name"
    odd.mkdir()
    (odd / "a.py").write_text("x = 1\n")
    code, _, _ = run_main(["--scan", str(odd), "--no-llm"], capsys)
    assert code == 0 and (in_tmp / "scan_workspace" / "cli-we_ird_name" / "a.py").exists()


def test_format_summary_lists_remaining_items():
    from agent.states import Report, Violation
    rep = Report(violations=[Violation(file="a.py", line_start=3, line_end=3, cwe="CWE-95", severity="high",
                                       title="eval", status="unresolved")],
                 summary={"files_scanned": 1, "found": 1, "remaining": 1}, warnings=["careful"])
    text = cli.format_summary(rep)
    assert "a.py:3" in text and "CWE-95" in text and "warning: careful" in text
