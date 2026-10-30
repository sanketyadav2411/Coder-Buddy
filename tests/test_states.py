import pytest
from pydantic import ValidationError

from agent.states import (CoderState, File, FixResponse, ImplementationTask, LLMViolation, Patch, Plan,
                          Report, ScanResponse, TaskPlan, Violation, normalize_cwe, normalize_severity)


# ---- original models are unchanged (regression) ------------------------------------------------
def test_original_models_still_work():
    plan = Plan(name="x", description="d", techstack="python", features=["a"],
                files=[File(path="a.py", purpose="main")])
    tp = TaskPlan(implementation_steps=[ImplementationTask(filepath="a.py", task_description="do it")])
    tp.plan = plan                      # architect_agent relies on extra="allow"
    cs = CoderState(task_plan=tp)
    assert cs.current_step_idx == 0 and cs.current_file_content is None


def test_original_models_still_reject_bad_input():
    with pytest.raises(ValidationError):
        Plan(name="x")


# ---- normalisation of sloppy model output --------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("CWE-89", "CWE-89"), ("cwe 89", "CWE-89"), ("89", "CWE-89"), (89, "CWE-89"),
    ("CWE-089: SQL Injection", "CWE-89"), ("", "CWE-unknown"), (None, "CWE-unknown"),
    ("unknown", "CWE-unknown"),
])
def test_normalize_cwe(raw, expected):
    assert normalize_cwe(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("HIGH", "high"), ("Severe", "high"), ("crit", "critical"), ("moderate", "medium"),
    ("minor", "low"), ("informational", "info"), ("???", "medium"), (None, "medium"),
])
def test_normalize_severity(raw, expected):
    assert normalize_severity(raw) == expected


def test_llm_violation_is_lenient():
    lv = LLMViolation.model_validate({"line_start": "line 12", "line_end": 13.0, "cwe": "89",
                                      "severity": "Severe", "evidence": "x", "bogus": 1})
    assert (lv.line_start, lv.line_end, lv.cwe, lv.severity) == (12, 13, "CWE-89", "high")


def test_llm_violation_requires_a_line_number():
    with pytest.raises(ValidationError):
        LLMViolation.model_validate({"cwe": "CWE-89"})
    with pytest.raises(ValidationError):
        LLMViolation.model_validate({"line_start": True})   # a bool is not a line number


def test_scan_response_defaults_to_empty():
    assert ScanResponse.model_validate({}).violations == []


def test_patch_accepts_strings_and_lists():
    p = Patch.model_validate({"start_line": "5", "end_line": 5, "replacement": ["a", "b"]})
    assert (p.start_line, p.end_line, p.replacement) == (5, 5, "a\nb")
    assert Patch.model_validate({"start_line": 1, "end_line": 0, "replacement": None}).replacement == ""
    assert FixResponse.model_validate({"patches": []}).patches == []


def test_violation_key_survives_line_shifts_and_label_changes():
    a = Violation(file="a.py", line_start=3, line_end=3, cwe="CWE-79", evidence="x  =  eval(y)")
    b = Violation(file="a.py", line_start=9, line_end=9, cwe="CWE-80", evidence="X = eval(Y)")
    assert a.key() == b.key()
    c = Violation(file="b.py", line_start=3, line_end=3, evidence="x = eval(y)")
    assert a.key() != c.key()


def test_report_defaults():
    r = Report()
    assert r.violations == [] and r.summary == {} and r.artifacts == {}
