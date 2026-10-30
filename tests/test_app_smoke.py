"""Smoke tests for the Streamlit UI (headless, via streamlit.testing). No model is ever called."""
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Headless app that is isolated from the developer's machine: no .env, no real model calls."""
    import agent.graph as graph_mod
    import agent.llm as llm_mod
    import agent.secure as secure_mod

    def no_model(*args, **kwargs):
        raise AssertionError("a UI smoke test tried to call a real language model")

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)      # ignore the real .env (keys, provider)
    for mod in (graph_mod, llm_mod, secure_mod):
        monkeypatch.setattr(mod, "get_llm", no_model)
    monkeypatch.chdir(tmp_path)               # restores the original cwd afterwards (app.py chdirs itself)
    existed = (APP.parent / "generated_project").exists()
    at = AppTest.from_file(str(APP), default_timeout=60)
    yield at
    if not existed:                           # the app creates this folder on import; keep the repo clean
        shutil.rmtree(APP.parent / "generated_project", ignore_errors=True)


def _select(at, label):
    """Find a selectbox by its label (position changes when the Files tab shows another selectbox)."""
    return next(sb for sb in at.selectbox if sb.label == label)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_starts_in_generate_mode_without_errors(app):
    app.run()
    assert not app.exception
    assert app.radio(key="mode").value == "Generate project"
    labels = [b.label for b in app.button]
    assert "Generate" in labels                                           # original UI is still there
    assert any(c.label.startswith("Run secure-coding scan") for c in app.checkbox)
    assert not [c for c in app.checkbox if c.label.startswith("Run secure-coding scan")][0].value   # off by default


def test_generate_without_prompt_warns(app):
    app.run()
    _button(app, "Generate").click().run()
    assert not app.exception and any("enter a prompt" in w.value for w in app.warning)


def test_generate_without_groq_key_asks_for_it(app):
    app.run()
    _select(app, "LLM provider").set_value("groq").run()
    [t for t in app.text_input if t.label == "Groq API key"][0].set_value("").run()
    app.text_area(key="prompt").set_value("make a todo app")
    _button(app, "Generate").click().run()
    assert any("Groq API key" in e.value for e in app.error)


def test_ollama_provider_hides_the_groq_key_and_validates_the_url(app):
    app.run()
    _select(app, "LLM provider").set_value("ollama").run()
    assert not app.exception
    assert [t for t in app.text_input if t.label == "Ollama model"]
    assert not [t for t in app.text_input if t.label == "Groq API key"]   # no cloud key needed for local mode
    [t for t in app.text_input if t.label == "Ollama server URL"][0].set_value("ftp://nope").run()
    app.text_area(key="prompt").set_value("make a todo app")
    _button(app, "Generate").click().run()
    assert any("http://" in e.value for e in app.error)


def test_remote_ollama_url_triggers_a_privacy_warning(app):
    app.run()
    _select(app, "LLM provider").set_value("ollama").run()
    [t for t in app.text_input if t.label == "Ollama server URL"][0].set_value("https://gpu.example.com").run()
    assert any("not a local address" in w.value for w in app.sidebar.warning)


def test_scan_mode_renders_and_requires_an_upload(app):
    app.run()
    app.radio(key="mode").set_value("Scan existing code").run()
    assert not app.exception
    _button(app, "Scan").click().run()
    assert any("upload at least one file" in w.value for w in app.warning)
