"""Shared test helpers. Nothing here needs network access, a GPU or a running model server."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"
VULNERABLE = FIXTURES / "vulnerable"
CLEAN = FIXTURES / "clean"

_ENV_VARS = (
    "LLM_PROVIDER", "GROQ_MODEL", "GROQ_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_NUM_CTX",
    "OLLAMA_TIMEOUT", "OLLAMA_TEMPERATURE", "SCAN_MAX_FILES", "SCAN_MAX_FILE_KB", "SCAN_MAX_TOTAL_MB",
    "SCAN_CHUNK_LINES", "SCAN_CHUNK_CHARS", "MAX_FIX_ROUNDS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from a known environment, whatever the developer's .env contains."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeLLM:
    """Scripted chat model.

    `script` is either a callable ``f(prompt_text) -> str | Exception`` or a list of replies
    (strings or exceptions); the last list entry repeats once the list is exhausted.
    """

    def __init__(self, script):
        self.script = script
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if callable(self.script):
            first = messages[0]["content"] if isinstance(messages[0], dict) else str(messages[0])
            out = self.script(first)
        else:
            idx = min(len(self.calls) - 1, len(self.script) - 1)
            out = self.script[idx]
        if isinstance(out, BaseException):
            raise out
        return types.SimpleNamespace(content=out)


@pytest.fixture
def fake_llm():
    return FakeLLM


def symlink_or_skip(link: Path, target: Path) -> None:
    """Create a symlink, or skip the test where the OS refuses (Windows needs Developer Mode or admin)."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")
