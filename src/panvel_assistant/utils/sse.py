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

    Per the SSE spec, each ``\\n`` in *text* terminates a ``data:`` field and
    starts a new one; the client reassembles them by joining with ``\\n``. This
    keeps multi-line LLM tokens intact without forcing the client to parse JSON.
    """
    data_lines = "\n".join(f"data: {line}" for line in text.split("\n"))
    return f"event: {event_type}\n{data_lines}\n\n"
