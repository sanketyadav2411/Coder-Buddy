"""Tiny progress/cancel channel between the agent code and the UI.

The agent nodes call ``emit(...)`` to report what they are doing ("writing index.html",
"scanning app.py lines 1-120"). The UI installs a listener that stores the events and shows them
live. Without a listener every call is a cheap no-op, so the CLI and the tests are unaffected.

``request_cancel()`` asks the running job to stop: the next ``emit`` / ``check_cancel`` raises
``Cancelled`` (a BaseException, so generic ``except Exception`` blocks do not swallow it).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

_listener: Optional[Callable[[dict], None]] = None
_cancel = threading.Event()


class Cancelled(BaseException):
    """The user pressed Stop."""


def set_listener(fn: Optional[Callable[[dict], None]]) -> None:
    global _listener
    _listener = fn


def reset_cancel() -> None:
    _cancel.clear()


def request_cancel() -> None:
    _cancel.set()


def cancel_requested() -> bool:
    return _cancel.is_set()


def check_cancel() -> None:
    if _cancel.is_set():
        raise Cancelled()


def emit(kind: str, message: str = "", **data: Any) -> None:
    """Report progress. kind: 'log' | 'file' | 'progress' | 'warn'."""
    check_cancel()
    fn = _listener
    if fn is not None:
        try:
            fn({"kind": kind, "message": message, **data})
        except Exception:  # noqa: BLE001 - a broken listener must never break the agent
            pass
