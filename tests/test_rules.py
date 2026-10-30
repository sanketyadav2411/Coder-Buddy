import pytest

from agent.rules import RULES, is_comment, language_of, mask_secrets, run_rules
from conftest import CLEAN, VULNERABLE


def _rule_ids(path):
    return {v.rule_id for v in run_rules(path.name, path.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("name,expected", [
    ("vuln.py", {"GEN001", "PY002", "PY003", "PY004", "PY006"}),
    ("vuln.js", {"GEN001", "JS001", "JS002"}),
    ("vuln.html", {"JS002", "JS005"}),
    ("vuln.c", {"C001", "C002", "C004"}),
])
def test_each_rule_fires_on_its_vulnerable_fixture(name, expected):
    found = _rule_ids(VULNERABLE / name)
    assert expected <= found, f"missing {expected - found}"


@pytest.mark.parametrize("name", ["clean.py", "clean.js", "clean.html", "clean.c"])
def test_clean_fixtures_produce_no_findings(name):
    assert _rule_ids(CLEAN / name) == set()


def test_findings_carry_cwe_and_valid_lines():
    text = (VULNERABLE / "vuln.py").read_text(encoding="utf-8")
    n_lines = len(text.splitlines())
    vios = run_rules("vuln.py", text)
    assert vios
    for v in vios:
        assert v.cwe.startswith("CWE-") and v.cwe != "CWE-unknown"
        assert 1 <= v.line_start == v.line_end <= n_lines
        assert v.source == "rule" and v.rule_id


def test_secret_value_is_never_echoed():
    vios = run_rules("vuln.py", (VULNERABLE / "vuln.py").read_text(encoding="utf-8"))
    gen = [v for v in vios if v.rule_id == "GEN001"]
    assert gen and all("abcdef123456" not in v.evidence for v in gen)
    assert "hunter2hunter2" not in run_rules("a.js", 'const password = "hunter2hunter2";')[0].evidence


def test_placeholders_and_env_lookups_are_not_secrets():
    assert run_rules("a.py", 'API_KEY = "<your-api-key>"') == []
    assert run_rules("a.py", 'password = os.environ["PASSWORD"] or "fallback-secret"') == []
    assert run_rules("a.py", 'token = "${TOKEN_FROM_CI}"') == []


def test_comments_are_ignored():
    assert run_rules("a.py", "# x = eval(y)") == []
    assert run_rules("a.c", "// gets(buf);") == []


def test_ast_literal_eval_is_not_flagged_as_eval():
    assert run_rules("a.py", "x = ast.literal_eval(s)") == []


def test_http_localhost_is_not_flagged_but_remote_is():
    assert run_rules("a.py", 'u = "http://localhost:8000/x"') == []
    assert [v.rule_id for v in run_rules("a.py", 'u = "http://example.org/x"')] == ["GEN003"]


def test_unknown_extension_is_not_scanned():
    assert language_of("notes.txt") is None
    assert language_of("image.png") is None
    assert run_rules("notes.txt", "eval(x)") == []


def test_language_detection_and_env_files():
    assert language_of("a/b/c.PY") == "python"
    assert language_of("x.tsx") == "javascript"
    assert language_of(".env") == "config"
    assert language_of("prod.env") == "config"
    assert language_of(r"dir\file.c") == "c"


def test_mask_secrets():
    assert "abcdef" not in mask_secrets('key = "abcdefgh"')
    assert mask_secrets("AKIAABCDEFGHIJKLMNOP") == "AK***"


def test_is_comment():
    assert is_comment("python", "# hi") and not is_comment("python", "x = 1")
    assert is_comment("c", "/* hi */")


def test_rule_catalogue_is_well_formed():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))
    for r in RULES:
        assert r.cwe.startswith("CWE-") and r.severity in {"critical", "high", "medium", "low", "info"}
        assert r.explanation and r.fix
