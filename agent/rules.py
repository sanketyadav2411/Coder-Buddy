"""Deterministic first pass of the secure-coding scanner.

Each rule is a single-line regex mapped to CWE / OWASP Top 10 (2021) / CERT identifiers. The rules
are intentionally conservative and cheap: they give the scanner findings that do not depend on the
LLM (so a run still produces a useful report if the local model is down or hallucinates), and their
hits are also passed to the LLM as hints to confirm or reject.

OWASP is left as None where there is no honest mapping (e.g. most C memory-safety weaknesses).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

from agent.states import Violation

# ---------------------------------------------------------------------------
# language detection
# ---------------------------------------------------------------------------
EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "javascript", ".tsx": "javascript",
    ".html": "html", ".htm": "html",
    ".c": "c", ".h": "c", ".cpp": "c", ".cc": "c", ".cxx": "c", ".hpp": "c", ".hh": "c",
    ".java": "other", ".go": "other", ".php": "other", ".rb": "other", ".cs": "other",
    ".sh": "other", ".bash": "other", ".sql": "other", ".rs": "other", ".kt": "other",
    ".swift": "other",
    ".json": "config", ".yml": "config", ".yaml": "config", ".toml": "config",
    ".ini": "config", ".cfg": "config", ".env": "config",
}
LLM_SKIP_LANGS = {"config"}  # rules only: configuration files are not analysed by the LLM


def language_of(path: str) -> Optional[str]:
    """Language family for a path, or None if the file type is not scanned."""
    p = PurePosixPath(path.replace("\\", "/"))
    name = p.name.lower()
    if name == ".env" or name.startswith(".env."):
        return "config"
    return EXT_LANG.get(p.suffix.lower())


_COMMENT_PREFIX = {
    "python": ("#",),
    "javascript": ("//", "/*", "*"),
    "html": ("<!--", "//", "/*", "*"),
    "c": ("//", "/*", "*"),
    "config": ("#", ";", "//"),
    "other": ("#", "//", "/*", "*", "--"),
}


def is_comment(lang: str, stripped_line: str) -> bool:
    return stripped_line.startswith(_COMMENT_PREFIX.get(lang, ()))


# ---------------------------------------------------------------------------
# secret masking (reports must never echo a credential)
# ---------------------------------------------------------------------------
_QUOTED = re.compile(r"""(["'])([^"'\n]{6,})\1""")
_AKIA = re.compile(r"\bAKIA[0-9A-Z]{16}\b")


def mask_secrets(text: str) -> str:
    text = _AKIA.sub("AK***", text)
    return _QUOTED.sub(lambda m: f"{m.group(1)}{m.group(2)[:2]}***{m.group(1)}", text)


# ---------------------------------------------------------------------------
# rule catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    cwe: str
    severity: str
    langs: tuple[str, ...]          # language families the rule applies to ("*" = all)
    pattern: re.Pattern
    explanation: str
    fix: str
    owasp: Optional[str] = None
    cert: Optional[str] = None
    mask: bool = False              # mask quoted values in the reported evidence
    skip_if: Optional[re.Pattern] = None   # line-level exclusion
    value_group: Optional[int] = None      # group holding a secret value (checked for placeholders)


def _r(p: str, flags: int = 0) -> re.Pattern:
    return re.compile(p, flags)


A01 = "A01:2021-Broken Access Control"
A02 = "A02:2021-Cryptographic Failures"
A03 = "A03:2021-Injection"
A05 = "A05:2021-Security Misconfiguration"
A07 = "A07:2021-Identification and Authentication Failures"
A08 = "A08:2021-Software and Data Integrity Failures"

RULES: tuple[Rule, ...] = (
    # ---- generic (all languages) -------------------------------------------------------------
    Rule("GEN001", "Hardcoded credential or secret", "CWE-798", "high", ("*",),
         _r(r"""[A-Za-z0-9_.\-]*(?:password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?key|"""
            r"""auth[_-]?token|private[_-]?key|client[_-]?secret|token)["']?\s*[:=]\s*["']([^"'\n]{6,})["']""",
            re.IGNORECASE),
         "A credential is stored in source code, so anyone with access to the code (or its history) has it.",
         "Load the value from an environment variable or a secrets manager and rotate the exposed secret.",
         owasp=A07, mask=True, value_group=1,
         skip_if=_r(r"os\.environ|getenv|process\.env|import\.meta\.env", re.IGNORECASE)),
    Rule("GEN002", "Hardcoded cloud key or private key material", "CWE-798", "critical", ("*",),
         _r(r"\bAKIA[0-9A-Z]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
         "Key material is committed to source code.",
         "Remove it from the code, load it from the environment/secret store, and rotate it.",
         owasp=A02, mask=True),
    Rule("GEN003", "Cleartext HTTP URL", "CWE-319", "low", ("*",),
         _r(r"""["'(]http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|host\.docker\.internal|"""
            r"""www\.w3\.org|[\w.-]*schemas?\.|xmlns)[^\s"')]+"""),
         "Data sent over plain HTTP can be read or modified in transit.",
         "Use an https:// URL.",
         owasp=A02),
    # ---- Python -----------------------------------------------------------------------------
    Rule("PY001", "Dynamic code execution (eval/exec)", "CWE-95", "high", ("python",),
         _r(r"(?<![\w.])(?:eval|exec)\s*\("),
         "eval/exec run arbitrary code; with any attacker-influenced input this is remote code execution.",
         "Parse data with ast.literal_eval/json.loads or use an explicit dispatch table instead.",
         owasp=A03),
    Rule("PY002", "OS command injection risk (shell=True / os.system)", "CWE-78", "high", ("python",),
         _r(r"subprocess\.\w+\s*\(.*shell\s*=\s*True|(?<![\w.])os\.(?:system|popen)\s*\("),
         "Building a shell command string lets crafted input run extra commands.",
         "Pass an argument list to subprocess.run([...]) without shell=True and validate the arguments.",
         owasp=A03),
    Rule("PY003", "SQL built from strings (SQL injection)", "CWE-89", "high", ("python",),
         _r(r"""\.(?:execute|executemany)\s*\(\s*(?:[fF]["']|["'][^"']*["']\s*(?:%|\+)|["'][^"']*["']\s*\.format)"""),
         "Interpolating values into SQL text lets input change the query.",
         "Use parameterised queries: cursor.execute('... WHERE id = ?', (value,)).",
         owasp=A03),
    Rule("PY004", "Insecure deserialisation (pickle/marshal)", "CWE-502", "high", ("python",),
         _r(r"\b(?:pickle|cPickle|marshal)\.loads?\s*\("),
         "Unpickling untrusted data executes code chosen by the sender.",
         "Use a data-only format such as JSON, or authenticate the data before loading.",
         owasp=A08),
    Rule("PY005", "yaml.load without a safe loader", "CWE-502", "high", ("python",),
         _r(r"\byaml\.load\s*\((?!.*(?:SafeLoader|safe_load))"),
         "yaml.load with the default loader can construct arbitrary Python objects.",
         "Use yaml.safe_load(...).",
         owasp=A08),
    Rule("PY006", "Weak hash algorithm (MD5/SHA-1)", "CWE-327", "medium", ("python",),
         _r(r"\bhashlib\.(?:md5|sha1)\s*\("),
         "MD5 and SHA-1 are broken for security purposes (collisions, fast brute force).",
         "Use hashlib.sha256 (or a password KDF such as bcrypt/argon2/scrypt for passwords).",
         owasp=A02),
    Rule("PY007", "TLS certificate verification disabled", "CWE-295", "medium", ("python",),
         _r(r"\bverify\s*=\s*False"),
         "Disabling certificate checks allows man-in-the-middle attacks.",
         "Remove verify=False (or point verify= at the correct CA bundle).",
         owasp=A07),
    Rule("PY008", "Debug mode enabled", "CWE-489", "medium", ("python",),
         _r(r"\.run\s*\(.*\bdebug\s*=\s*True"),
         "Debug servers expose an interactive console and stack traces.",
         "Do not enable debug in deployed code; read it from configuration and default to False.",
         owasp=A05),
    Rule("PY009", "Predictable temporary file (tempfile.mktemp)", "CWE-377", "medium", ("python",),
         _r(r"\btempfile\.mktemp\s*\("),
         "mktemp returns a name without creating the file, which enables race-condition attacks.",
         "Use tempfile.mkstemp() or tempfile.NamedTemporaryFile().",
         owasp=A01),
    Rule("PY010", "Template auto-escaping disabled", "CWE-79", "medium", ("python",),
         _r(r"\bautoescape\s*=\s*False"),
         "Without auto-escaping, user input rendered into HTML enables cross-site scripting.",
         "Enable autoescape (autoescape=True / select_autoescape()).",
         owasp=A03),
    # ---- JavaScript / HTML ------------------------------------------------------------------
    Rule("JS001", "Dynamic code execution (eval / new Function)", "CWE-95", "high", ("javascript", "html"),
         _r(r"(?<![\w.$])eval\s*\(|\bnew\s+Function\s*\(|\bsetTimeout\s*\(\s*[\"'`]|\bsetInterval\s*\(\s*[\"'`]"),
         "Evaluating strings as code allows script injection.",
         "Avoid eval; use JSON.parse for data and pass functions (not strings) to timers.",
         owasp=A03),
    Rule("JS002", "DOM XSS sink (innerHTML / document.write)", "CWE-79", "medium", ("javascript", "html"),
         _r(r"\.(?:innerHTML|outerHTML)\s*\+?=(?!=)|\binsertAdjacentHTML\s*\(|\bdocument\.write(?:ln)?\s*\("),
         "Writing untrusted text into HTML lets an attacker inject script (cross-site scripting).",
         "Use textContent / createElement + appendChild, or sanitise with a vetted library such as DOMPurify.",
         owasp=A03,
         skip_if=_r(r"\.(?:innerHTML|outerHTML)\s*=\s*(?:\"\"|''|``)\s*;?\s*$")),
    Rule("JS003", "SQL built from strings (SQL injection)", "CWE-89", "high", ("javascript",),
         _r(r"""\.(?:query|execute|run|all|get)\s*\(\s*(?:["'][^"']*["']\s*\+|`[^`]*\$\{)"""),
         "Concatenating values into SQL text lets input change the query.",
         "Use parameterised queries / prepared statements.",
         owasp=A03),
    Rule("JS004", "Sensitive data kept in web storage", "CWE-922", "medium", ("javascript", "html"),
         _r(r"""(?:localStorage|sessionStorage)\.setItem\s*\(\s*["'][^"']*(?:token|password|passwd|secret|jwt|api[_-]?key)[^"']*["']""",
            re.IGNORECASE),
         "Web storage is readable by any script on the page, so one XSS bug exposes the secret.",
         "Keep session tokens in HttpOnly, Secure cookies and never store passwords client-side.",
         owasp=A02),
    Rule("JS005", "Link opens a new tab without rel=noopener", "CWE-1022", "low", ("html", "javascript"),
         _r(r"""<a\b(?=[^>]*\btarget\s*=\s*["']_blank["'])(?![^>]*\brel\s*=\s*["'][^"']*noopener)[^>]*>""",
            re.IGNORECASE),
         "The opened page can control the opener through window.opener (reverse tabnabbing).",
         'Add rel="noopener noreferrer" to the link.',
         owasp=A05),
    Rule("JS006", "OS command built from strings", "CWE-78", "high", ("javascript",),
         _r(r"""\b(?:execSync|exec)\s*\(\s*(?:`[^`]*\$\{|["'][^"']*["']\s*\+)"""),
         "Building a shell command from input lets crafted values run extra commands.",
         "Use execFile/spawn with an argument array and validate the arguments.",
         owasp=A03),
    Rule("JS007", "Weak hash algorithm (MD5/SHA-1)", "CWE-327", "medium", ("javascript",),
         _r(r"""createHash\s*\(\s*["'](?:md5|sha1)["']"""),
         "MD5 and SHA-1 are broken for security purposes.",
         "Use sha256 (or a password KDF such as bcrypt/argon2/scrypt for passwords).",
         owasp=A02),
    # ---- C / C++ ----------------------------------------------------------------------------
    Rule("C001", "Use of gets()", "CWE-242", "critical", ("c",),
         _r(r"(?<![\w.>])(?:std::)?gets\s*\("),
         "gets cannot limit input length; it always allows a buffer overflow (removed in C11).",
         "Use fgets(buf, sizeof buf, stdin).",
         cert="MSC24-C"),
    Rule("C002", "Unbounded string copy/format", "CWE-120", "high", ("c",),
         _r(r"(?<![\w.>])(?:std::)?(?:strcpy|strcat|sprintf|vsprintf)\s*\("),
         "These functions do not check the destination size, which can overflow the buffer.",
         "Use snprintf / strncat with explicit sizes, or std::string in C++.",
         cert="STR31-C"),
    Rule("C003", "scanf %s without a width", "CWE-120", "high", ("c",),
         _r(r"""(?<![\w.>])(?:f?scanf|sscanf)\s*\([^;]*%s"""),
         "%s without a maximum width writes past the end of the buffer.",
         'Give a width (e.g. "%63s" for a 64-byte buffer) or use fgets.',
         cert="STR31-C"),
    Rule("C004", "Command execution via system()/popen()", "CWE-78", "high", ("c",),
         _r(r"(?<![\w.>])(?:std::)?(?:system|popen)\s*\("),
         "Passing a command string to the shell lets input inject extra commands.",
         "Avoid the shell: use execve-family calls with a fixed argument vector.",
         cert="ENV33-C"),
    Rule("C005", "Format string from a variable", "CWE-134", "high", ("c",),
         _r(r"(?<![\w.>])printf\s*\(\s*[A-Za-z_]\w*\s*\)|(?<![\w.>])fprintf\s*\(\s*\w+\s*,\s*[A-Za-z_]\w*\s*\)"),
         "If the variable contains % directives an attacker can read or write memory.",
         'Use a constant format string: printf("%s", value).',
         cert="FIO30-C"),
    Rule("C006", "Weak random number generator (rand)", "CWE-338", "low", ("c",),
         _r(r"(?<![\w.>])(?:std::)?rand\s*\(\s*\)"),
         "rand() is predictable and unsuitable for security decisions.",
         "Use a CSPRNG (getrandom, /dev/urandom, std::random_device) for anything security related.",
         cert="MSC30-C"),
    Rule("C007", "Unchecked string-to-number conversion", "CWE-20", "low", ("c",),
         _r(r"(?<![\w.>])(?:std::)?(?:atoi|atol|atoll|atof)\s*\("),
         "atoi-style functions give no error indication, so bad input is silently accepted.",
         "Use strtol/strtod (or std::stoi with exception handling) and check the result.",
         cert="ERR34-C"),
)


_PLACEHOLDER = re.compile(r"^(?:<[^>]*>|\$\{[^}]*\}|\{\{[^}]*\}\}|%\([^)]*\)s?|\[[^\]]*\])$")


def _applies(rule: Rule, langs: set[str]) -> bool:
    return "*" in rule.langs or bool(langs.intersection(rule.langs))


def run_rules(path: str, text: str, lang: Optional[str] = None) -> list[Violation]:
    """Apply every applicable rule to `text`. At most one finding per (rule, line)."""
    lang = lang or language_of(path)
    if lang is None:
        return []
    langs = {lang}
    rules = [r for r in RULES if _applies(r, langs)]
    out: list[Violation] = []
    for lineno, line in enumerate(text.replace("\r\n", "\n").split("\n"), start=1):
        stripped = line.strip()
        if not stripped or is_comment(lang, stripped):
            continue
        for rule in rules:
            m = rule.pattern.search(line)
            if not m:
                continue
            if rule.skip_if is not None and rule.skip_if.search(line):
                continue
            if rule.value_group is not None:
                value = (m.group(rule.value_group) or "").strip()
                if _PLACEHOLDER.match(value):
                    continue
            evidence = stripped[:300]
            if rule.mask:
                evidence = mask_secrets(evidence)
            out.append(Violation(
                file=path, line_start=lineno, line_end=lineno, cwe=rule.cwe, owasp=rule.owasp,
                cert=rule.cert, severity=rule.severity, title=rule.title, evidence=evidence,
                explanation=rule.explanation, suggested_fix=rule.fix, source="rule", rule_id=rule.id,
            ))
    return out
