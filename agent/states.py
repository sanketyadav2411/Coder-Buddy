import re
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Original Coder Buddy models (unchanged)
# ---------------------------------------------------------------------------
class File(BaseModel):
    path: str = Field(description="The path to the file to be created or modified")
    purpose: str = Field(description="The purpose of the file, e.g. 'main application logic', 'data processing module', etc.")


class Plan(BaseModel):
    name: str = Field(description="The name of app to be built")
    description: str = Field(description="A oneline description of the app to be built, e.g. 'A web application for managing personal finances'")
    techstack: str = Field(description="The tech stack to be used for the app, e.g. 'python', 'javascript', 'react', 'flask', etc.")
    features: list[str] = Field(description="A list of features that the app should have, e.g. 'user authentication', 'data visualization', etc.")
    files: list[File] = Field(description="A list of files to be created, each with a 'path' and 'purpose'")


class ImplementationTask(BaseModel):
    filepath: str = Field(description="The path to the file to be modified")
    task_description: str = Field(description="A detailed description of the task to be performed on the file, e.g. 'add user authentication', 'implement data processing logic', etc.")


class TaskPlan(BaseModel):
    implementation_steps: list[ImplementationTask] = Field(description="A list of steps to be taken to implement the task")
    model_config = ConfigDict(extra="allow")


class CoderState(BaseModel):
    task_plan: TaskPlan = Field(description="The plan for the task to be implemented")
    current_step_idx: int = Field(0, description="The index of the current step in the implementation steps")
    current_file_content: Optional[str] = Field(None, description="The content of the file currently being edited or created")


# ---------------------------------------------------------------------------
# Secure-coding scanner models
# ---------------------------------------------------------------------------
SEVERITIES = ("critical", "high", "medium", "low", "info")
SEVERITY_RANK = {s: i for i, s in enumerate(reversed(SEVERITIES))}   # info=0 ... critical=4

_CWE_RE = re.compile(r"(?:CWE)?[\s_:-]*(\d{1,5})", re.IGNORECASE)
_SEVERITY_ALIASES = {
    "critical": "critical", "crit": "critical",
    "high": "high", "severe": "high", "error": "high",
    "medium": "medium", "moderate": "medium", "med": "medium", "warning": "medium",
    "low": "low", "minor": "low",
    "info": "info", "informational": "info",
}


def normalize_cwe(value: Any) -> str:
    """'89', 'cwe 89', 'CWE-089: SQL Injection' -> 'CWE-89'. Anything else -> 'CWE-unknown'."""
    m = _CWE_RE.search(str(value or ""))
    return f"CWE-{int(m.group(1))}" if m else "CWE-unknown"


def normalize_severity(value: Any) -> str:
    return _SEVERITY_ALIASES.get(str(value or "").strip().lower(), "medium")


def _to_int(v: Any) -> Any:
    if isinstance(v, bool):
        raise ValueError("line number expected")
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", str(v))
    if not m:
        raise ValueError("line number expected")
    return int(m.group(0))


class LLMViolation(BaseModel):
    """What the local LLM is asked to return for one finding. Deliberately lenient:
    small local models get field names/values slightly wrong, so we normalise instead of failing."""
    model_config = ConfigDict(extra="ignore")

    line_start: int = Field(description="First affected line (1-based, as numbered in the listing)")
    line_end: Optional[int] = Field(None, description="Last affected line; defaults to line_start")
    cwe: str = Field("CWE-unknown", description="CWE identifier, e.g. CWE-89")
    owasp: Optional[str] = Field(None, description="OWASP Top 10 (2021) category, e.g. A03:2021-Injection")
    cert: Optional[str] = Field(None, description="CERT C/C++ rule id if applicable, e.g. STR31-C")
    severity: str = Field("medium", description="critical | high | medium | low | info")
    title: str = Field("", description="Short name of the weakness")
    evidence: str = Field("", description="The exact offending code copied from the listing")
    explanation: str = Field("", description="Why this is exploitable")
    suggested_fix: str = Field("", description="How to correct it")

    @field_validator("cwe", mode="before")
    @classmethod
    def _cwe(cls, v):
        return normalize_cwe(v)

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v):
        return normalize_severity(v)

    @field_validator("line_start", "line_end", mode="before")
    @classmethod
    def _int(cls, v):
        return None if v is None else _to_int(v)


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    violations: list[LLMViolation] = Field(default_factory=list)


class Violation(BaseModel):
    """A validated secure-coding violation (rule-based or LLM-detected)."""
    id: str = ""
    file: str
    line_start: int
    line_end: int
    cwe: str = "CWE-unknown"
    owasp: Optional[str] = None
    cert: Optional[str] = None
    severity: Literal["critical", "high", "medium", "low", "info"] = "medium"
    title: str = ""
    evidence: str = ""
    explanation: str = ""
    suggested_fix: str = ""
    source: Literal["rule", "llm"] = "llm"
    rule_id: Optional[str] = None
    status: Literal["open", "fixed", "unresolved", "new"] = "open"

    def key(self) -> tuple:
        """Identity that survives line shifts caused by fixes: (file, normalised code text).

        The CWE label is deliberately left out: a small local model may label the same line
        CWE-79 in one run and CWE-80 in the next, which must not look like a different finding."""
        return (self.file, " ".join(self.evidence.split()).lower())


class Patch(BaseModel):
    """Replace lines start_line..end_line (inclusive, 1-based) with `replacement`.
    end_line == start_line - 1 means 'insert before start_line'."""
    model_config = ConfigDict(extra="ignore")

    violation_id: Optional[str] = None
    start_line: int
    end_line: int
    replacement: str = ""

    @field_validator("start_line", "end_line", mode="before")
    @classmethod
    def _int(cls, v):
        return _to_int(v)

    @field_validator("replacement", mode="before")
    @classmethod
    def _repl(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return v if isinstance(v, str) else str(v)


class FixResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    patches: list[Patch] = Field(default_factory=list)


class SkippedFile(BaseModel):
    path: str
    reason: str


class Report(BaseModel):
    """Final output of a scan(+fix) run."""
    root: str = ""
    provider: str = ""
    model: str = ""
    llm_used: bool = False
    files_scanned: list[str] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)
    fix_rounds: int = 0
    violations: list[Violation] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    diffs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


class GraphState(TypedDict, total=False):
    """Typed graph state.

    The original graph used `StateGraph(dict)`, which only forwards the *previous node's output*
    (input keys and earlier outputs are dropped). Scanning needs flags/options/results to survive
    across several nodes, so every key is declared here. Existing nodes read/write the same keys
    as before, so their behaviour is unchanged.
    """
    user_prompt: str
    plan: Plan
    task_plan: TaskPlan
    coder_state: CoderState
    status: str
    secure_scan: bool
    scan_options: dict
    scan_root: str
    scan_round: int
    fix_rounds: int
    scan_next: str
    violations: list[Violation]
    prev_keys: list
    baseline: list[Violation]
    originals: dict
    changed_files: list[str]
    notes: list[str]
    run_dir: str
    report: Report
