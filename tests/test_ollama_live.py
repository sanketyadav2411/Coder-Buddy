"""Optional smoke test against a REAL local model. Skipped by default.

    ollama pull qwen2.5-coder:3b        (any code model works)
    pytest -m ollama -s

The model is chosen automatically from the ones you have pulled. To force one, set CODER_BUDDY_LIVE_MODEL
(the usual OLLAMA_MODEL cannot be used here because the test suite clears it before every test).

It prints a small quality table (detection of the known weaknesses in the fixtures and how many
fixes survived re-verification). The numbers depend on the model; the assertions only check that
the pipeline completes and never damages a file.
"""
import json
import os
import urllib.request

import pytest

import agent.graph as G
from agent.llm import DEFAULT_OLLAMA_URL, current_model
from agent.secure import ScanConfig, load_files
from conftest import VULNERABLE

pytestmark = pytest.mark.ollama


def _installed_models(url: str):
    """Names of the models pulled into the Ollama server, or None when it cannot be reached."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:  # noqa: BLE001
        return None


def pick_model(installed, preferred=None):
    """Choose a chat model from what is installed: the requested one, else a coder model, else None."""
    chat = [m for m in installed if "embed" not in m.lower()]       # embedding models cannot chat
    if preferred and preferred in installed:
        return preferred
    for wanted in ("qwen2.5-coder:7b", "qwen2.5-coder:3b"):
        if wanted in chat:
            return wanted
    coders = [m for m in chat if "coder" in m.lower()]
    return coders[0] if coders else (chat[0] if chat else None)


@pytest.fixture
def live(monkeypatch):
    url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    installed = _installed_models(url)
    if installed is None:
        pytest.skip(f"no Ollama server at {url}")
    model = pick_model(installed, os.environ.get("CODER_BUDDY_LIVE_MODEL"))
    if model is None:
        pytest.skip("Ollama is running but has no chat model; run: ollama pull qwen2.5-coder:3b")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    monkeypatch.setenv("OLLAMA_MODEL", model)
    return url


def test_live_scan_and_fix_on_vulnerable_fixtures(live, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    originals = {}
    for p in VULNERABLE.iterdir():
        originals[p.name] = p.read_text()
        (root / p.name).write_text(originals[p.name])

    result = G.scan_agent.invoke(
        {"scan_root": str(root), "scan_options": {"report_dir": str(tmp_path / "reports"), "max_fix_rounds": 2}},
        {"recursion_limit": 60})
    rep = result["report"]
    s = rep.summary
    print(f"\nmodel={current_model()} llm_used={rep.llm_used} found={s['found']} fixed={s['fixed']} "
          f"unresolved={s['unresolved']} new={s['new']} remaining={s['remaining']}")
    print("fix verification rate:", f"{s['fixed'] / s['found']:.0%}" if s["found"] else "n/a")
    for w in rep.warnings:
        print("warning:", w)

    assert rep.llm_used, "the local model never produced usable output"
    assert s["found"] >= 8                                             # rules alone already find these
    files, _ = load_files(root, ScanConfig.from_env())
    assert set(files) == set(originals)                                # nothing deleted or renamed
    for name, text in files.items():
        if name.endswith(".py"):
            compile(text, name, "exec")                                # a fixed file must still parse