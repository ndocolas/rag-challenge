"""Demo echo route — exercises Pydantic models end-to-end.

This endpoint is a smoke/fixture target left from the early bootstrap; the
production-facing surface is :mod:`panvel_assistant.routes.chat`. Marked as
deprecated via the ``Deprecation`` header so any consumer pinned to it gets a
clear signal before it is removed.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from panvel_assistant.models.chat import ChatMessage, ChatRequest
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/v1/echo", tags=["echo"], deprecated=True)


@router.post("/chat", response_model=ChatMessage)
@handle_errors
async def echo_chat(req: ChatRequest, response: Response) -> ChatMessage:
    """Echo back the user's message; deprecated, kept for backwards compat."""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Wed, 01 Jul 2026 00:00:00 GMT"
    response.headers["Link"] = '</chat>; rel="successor-version"'
    return ChatMessage(role="assistant", content=f"echo: {req.message}")
