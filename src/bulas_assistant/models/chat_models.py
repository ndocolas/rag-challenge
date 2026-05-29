"""Chat-domain Pydantic schemas: messages, requests, citations and SSE events."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ChatRole = Literal["user", "assistant", "system", "tool"]

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN,
        description=(
            "Opaque session identifier, alphanumeric plus '-' and '_' only. "
            "Used as a Redis key suffix; rejecting other characters prevents "
            "key pollution and ambiguous logs."
        ),
    )
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def _strip_and_require_non_empty(cls, value: str) -> str:
        """Trim outer whitespace and reject inputs that collapse to empty.

        Pydantic's ``min_length=1`` accepts ``"   "``; we additionally enforce
        a non-whitespace character so the LLM never sees a blank prompt.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must contain non-whitespace characters")
        return stripped


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
    "token", "tool_call", "tool_result", "sources", "done", "error", "trace_id"
]


class StreamEvent(BaseModel):
    event_type: StreamEventType
    payload: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
