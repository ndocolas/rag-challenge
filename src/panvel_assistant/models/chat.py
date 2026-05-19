"""Chat-domain Pydantic schemas: messages, requests, citations and SSE events."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant", "system", "tool"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class Citation(BaseModel):
    bula_id: str
    med_name: str
    med_variant: str | None = None
    section_canonical: str
    section_label: str
    source_page: int | None = None
    snippet: str


class ToolCallTrace(BaseModel):
    name: str
    args: dict[str, Any]
    result_preview: str | None = None
    latency_ms: float | None = None
    error: str | None = None


StreamEventType = Literal[
    "token", "tool_call", "tool_result", "sources", "done", "error"
]


class StreamEvent(BaseModel):
    event_type: StreamEventType
    payload: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
