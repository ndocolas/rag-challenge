"""Helpers for encoding Server-Sent Events (SSE) frames."""

import json
from typing import Any

EventType = str  # "token" | "tool_call" | "tool_result" | "sources" | "done" | "error"


def encode_event(event_type: EventType, payload: Any) -> str:
    """Encode a payload as a JSON SSE frame: ``event: X\\ndata: {json}\\n\\n``."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


def encode_text_event(event_type: EventType, text: str) -> str:
    """Encode a plain-text SSE frame.

    Per the SSE spec, each of ``\\r``, ``\\n``, ``\\r\\n`` is a line terminator
    that closes a ``data:`` field. We normalize all three to ``\\n`` before
    splitting so a multi-line token never produces a bare ``\\r`` that the
    client would interpret as an extra frame boundary.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    data_lines = "\n".join(f"data: {line}" for line in normalized.split("\n"))
    return f"event: {event_type}\n{data_lines}\n\n"


def encode_stream_error(
    *, code: str, message: str = "internal error", trace_id: str | None = None
) -> str:
    """Encode a sanitized SSE ``error`` frame for use in streaming handlers.

    ``handle_errors`` only covers code that runs before response headers are
    flushed; anything raised inside a ``StreamingResponse`` generator must use
    this helper to surface failures without leaking internal details.
    """
    payload: dict[str, str] = {"code": code, "message": message}
    if trace_id:
        payload["trace_id"] = trace_id
    return encode_event("error", payload)
