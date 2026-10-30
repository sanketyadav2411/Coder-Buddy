"""LLM factory (Groq or local Ollama) and a tolerant "ask for JSON, validate it" helper.

Provider selection (first match wins):
    1. explicit ``get_llm(provider=...)`` argument
    2. ``LLM_PROVIDER`` environment variable ("groq" | "ollama")
    3. default: "groq"  (keeps the original Coder Buddy behaviour)

Everything is read from the environment at call time, so a value set after import (for example from
the Streamlit sidebar) is always honoured - the same pattern the original ``get_llm`` used for the key.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
from typing import Any, Optional, Type, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Kept for backwards compatibility with code that imported MODEL_NAME from agent.graph.
MODEL_NAME = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

PROVIDERS = ("groq", "ollama")


class LLMOutputError(RuntimeError):
    """The model kept returning output that is not valid for the requested schema."""


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------
def current_provider(provider: Optional[str] = None) -> str:
    p = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if p not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{p}'. Use one of: {', '.join(PROVIDERS)}.")
    return p


def current_model(provider: Optional[str] = None) -> str:
    if current_provider(provider) == "ollama":
        return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL


def is_local_url(url: str) -> bool:
    """True when the URL points at this machine or a private network (loopback, RFC1918,
    link-local, localhost, host.docker.internal)."""
    try:
        host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    except ValueError:
        return False
    if host in {"localhost", "host.docker.internal", "gateway.docker.internal"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a public-looking hostname: treat as remote
    return ip.is_loopback or ip.is_private or ip.is_link_local


def endpoint_warnings(provider: Optional[str] = None) -> list[str]:
    """Human-readable warnings about where source code will be sent."""
    p = current_provider(provider)
    if p == "groq":
        return ["Provider is Groq (cloud): source code is sent to a third party. "
                "Choose the Ollama provider for a fully local run."]
    url = ollama_base_url()
    if not is_local_url(url):
        return [f"OLLAMA_BASE_URL ({url}) is not a local address: source code will leave this machine."]
    return []


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def get_llm(provider: Optional[str] = None, *, temperature: Optional[float] = None,
            json_mode: bool = False, max_tokens: Optional[int] = None):
    """Create the chat model on demand so the *current* environment is always used.

    Called with no arguments and LLM_PROVIDER unset it returns exactly what the original code did:
    ``ChatGroq(model=GROQ_MODEL)``.
    """
    p = current_provider(provider)

    if p == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The Ollama provider needs the 'langchain-ollama' package. "
                "Install it with:  uv pip install langchain-ollama"
            ) from e
        kwargs: dict[str, Any] = dict(
            model=current_model(p),
            base_url=ollama_base_url(),
            # Ollama silently truncates prompts beyond num_ctx (default is only a few thousand
            # tokens). A small window would make the scanner "see" half a file and miss findings.
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            temperature=(float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
                         if temperature is None else temperature),
            client_kwargs={"timeout": float(os.getenv("OLLAMA_TIMEOUT", "300"))},
            # Keep the model in memory between calls: reloading a 7B model from disk costs 10-60 s.
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        )
        if max_tokens:
            # Without a cap a small model can ramble until the context window is full.
            kwargs["num_predict"] = int(max_tokens)
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    from langchain_groq.chat_models import ChatGroq
    # Groq's free tier has a small tokens-per-minute budget; the client waits for the "try again in Ns"
    # window and retries, so a burst of requests slows down instead of failing the whole run.
    gkw: dict[str, Any] = dict(model=current_model(p), max_retries=int(os.getenv("GROQ_MAX_RETRIES", "8")))
    if temperature is not None:
        gkw["temperature"] = temperature
    if max_tokens:
        gkw["max_tokens"] = int(max_tokens)
    effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()
    if "gpt-oss" in gkw["model"] and effort in {"low", "medium", "high"}:
        # gpt-oss is a reasoning model; "low" cuts latency a lot for code generation/scanning.
        gkw["reasoning_effort"] = effort
    return ChatGroq(**gkw)


# ---------------------------------------------------------------------------
# JSON handling
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```", re.DOTALL)


def message_text(resp: Any) -> str:
    """Text of a chat-model reply (handles plain strings and content-block lists)."""
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply.

    Tolerates <think> blocks, ```json fences and prose before/after the JSON.
    Raises ValueError when nothing parses.
    """
    text = _THINK_RE.sub("", text or "").strip()
    if not text:
        raise ValueError("empty reply")
    try:
        return json.loads(text)
    except ValueError:
        pass
    for m in _FENCE_RE.finditer(text):
        try:
            return json.loads(m.group(1).strip())
        except ValueError:
            continue
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    raise ValueError("no JSON object found in the reply")


def structured_call(llm: Any, prompt: str, model_cls: Type[T], retries: int = 2) -> T:
    """Ask `llm` for JSON matching `model_cls`; on invalid output, show the model its error and retry.

    No tool-calling is used on purpose: small local models are unreliable at it, and the original
    project already needed a plain-text fallback for the same reason. Transport errors (server down,
    timeout) are NOT caught here - they propagate so the caller can decide what to do.
    Raises LLMOutputError after retries+1 invalid replies.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    last_err = "unknown error"
    fields = list(model_cls.model_fields)
    for _ in range(retries + 1):
        reply = message_text(llm.invoke(messages))
        try:
            data = extract_json(reply)
            if isinstance(data, list) and len(fields) == 1:
                data = {fields[0]: data}
            return model_cls.model_validate(data)
        except (ValueError, ValidationError) as e:
            last_err = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
            messages = messages + [
                {"role": "assistant", "content": reply[:4000]},
                {"role": "user", "content": (
                    f"That reply was not valid: {last_err}\n"
                    "Reply again with ONLY the corrected JSON object. No prose, no code fences.")},
            ]
    raise LLMOutputError(f"model did not return valid JSON after {retries + 1} attempts: {last_err}")
