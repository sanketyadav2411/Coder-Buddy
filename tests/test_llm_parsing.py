import pytest

from agent.llm import (LLMOutputError, current_model, current_provider, endpoint_warnings, extract_json,
                       get_llm, is_local_url, message_text, structured_call)
from agent.states import FixResponse, ScanResponse
from conftest import FakeLLM

GOOD = '{"violations": [{"line_start": 3, "cwe": "CWE-89", "severity": "high", "evidence": "x"}]}'


# ---- extract_json -------------------------------------------------------------------------------
@pytest.mark.parametrize("reply", [
    GOOD,
    "```json\n" + GOOD + "\n```",
    "Sure! Here is the result:\n" + GOOD + "\nHope that helps.",
    "<think>let me think {not json}</think>\n" + GOOD,
    "  \n" + GOOD + "  ",
])
def test_extract_json_tolerates_fences_prose_and_think_blocks(reply):
    assert extract_json(reply)["violations"][0]["line_start"] == 3


@pytest.mark.parametrize("reply", ["", "   ", "no json here", "{broken", "<think>only thoughts</think>"])
def test_extract_json_raises_on_garbage(reply):
    with pytest.raises(ValueError):
        extract_json(reply)


def test_message_text_handles_content_blocks():
    class R:
        content = [{"type": "text", "text": "ab"}, "cd"]
    assert message_text(R()) == "abcd"
    assert message_text("plain") == "plain"


# ---- structured_call ----------------------------------------------------------------------------
def test_structured_call_happy_path():
    llm = FakeLLM(["```json\n" + GOOD + "\n```"])
    resp = structured_call(llm, "prompt", ScanResponse)
    assert resp.violations[0].cwe == "CWE-89" and len(llm.calls) == 1


def test_structured_call_retries_with_error_feedback():
    llm = FakeLLM(["I could not find anything, sorry", GOOD])
    resp = structured_call(llm, "prompt", ScanResponse, retries=2)
    assert len(resp.violations) == 1 and len(llm.calls) == 2
    retry_messages = llm.calls[1]
    assert retry_messages[0]["content"] == "prompt"
    assert retry_messages[1]["role"] == "assistant"
    assert "not valid" in retry_messages[2]["content"]          # the model is told what was wrong


def test_structured_call_retries_on_schema_violation():
    llm = FakeLLM(['{"violations": [{"cwe": "CWE-89"}]}', GOOD])   # missing line_start
    assert len(structured_call(llm, "p", ScanResponse).violations) == 1
    assert len(llm.calls) == 2


def test_structured_call_gives_up_cleanly():
    llm = FakeLLM(["nope"])
    with pytest.raises(LLMOutputError):
        structured_call(llm, "p", ScanResponse, retries=2)
    assert len(llm.calls) == 3                                    # retries + 1 attempts, then stop


def test_structured_call_zero_retries_means_one_attempt():
    llm = FakeLLM(["nope"])
    with pytest.raises(LLMOutputError):
        structured_call(llm, "p", ScanResponse, retries=0)
    assert len(llm.calls) == 1


def test_structured_call_accepts_bare_list_for_single_field_models():
    llm = FakeLLM(['[{"start_line": 1, "end_line": 1, "replacement": "x"}]'])
    assert structured_call(llm, "p", FixResponse).patches[0].replacement == "x"


def test_transport_errors_are_not_swallowed():
    llm = FakeLLM([ConnectionError("server down")])
    with pytest.raises(ConnectionError):
        structured_call(llm, "p", ScanResponse)
    assert len(llm.calls) == 1                                    # no pointless retry


# ---- provider / endpoint configuration ------------------------------------------------------------
def test_default_provider_is_groq_to_preserve_existing_behaviour():
    assert current_provider() == "groq"
    assert current_model() == "openai/gpt-oss-120b"


def test_provider_from_env_and_argument(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Ollama")
    assert current_provider() == "ollama"
    assert current_provider("groq") == "groq"
    assert current_model() == "qwen2.5-coder:7b"
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert current_model() == "llama3.1:8b"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        current_provider()


@pytest.mark.parametrize("url,local", [
    ("http://localhost:11434", True), ("http://127.0.0.1:11434", True), ("http://[::1]:11434", True),
    ("http://host.docker.internal:11434", True), ("http://192.168.1.20:11434", True),
    ("http://10.0.0.5", True), ("localhost:11434", True),
    ("https://api.example.com", False), ("http://8.8.8.8:11434", False), ("http://my-gpu-box.com", False),
])
def test_is_local_url(url, local):
    assert is_local_url(url) is local


def test_endpoint_warnings(monkeypatch):
    assert "cloud" in endpoint_warnings()[0]                      # groq default: code leaves the machine
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert endpoint_warnings() == []
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://api.example.com")
    assert "leave this machine" in endpoint_warnings()[0]


def test_get_llm_default_is_groq_with_the_original_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = get_llm()
    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == "openai/gpt-oss-120b"


def test_get_llm_ollama(monkeypatch):
    pytest.importorskip("langchain_ollama")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    llm = get_llm(temperature=0, json_mode=True)
    assert type(llm).__name__ == "ChatOllama"
    assert llm.base_url == "http://localhost:9999"
    assert llm.num_ctx == 4096 and llm.format == "json" and llm.temperature == 0
