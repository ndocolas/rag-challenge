"""Sessions routes — list known sessions and load message history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from bulas_assistant.services.chat_history_service import (
    RedisHistoryStore,
    get_history_store,
)
from bulas_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
@handle_errors
async def list_sessions(store: RedisHistoryStore = Depends(get_history_store)) -> list[dict]:
    """Return all known sessions ordered newest-first."""
    return await store.list_sessions()


@router.get("/{session_id}/history")
@handle_errors
async def get_session_history(
    session_id: str,
    store: RedisHistoryStore = Depends(get_history_store),
) -> list[dict[str, Any]]:
    """Return the full message history for a session as frontend ChatMessage objects."""
    history = store.get_session_history(session_id)
    messages = await history.aget_messages()
    if not messages:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"session '{session_id}' not found or expired",
            },
        )
    result = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            result.append(
                {
                    "id": f"msg_{i}",
                    "role": "user",
                    "content": msg.content,
                    "citations": None,
                    "status": "done",
                }
            )
        elif isinstance(msg, AIMessage):
            citations = msg.additional_kwargs.get("citations") or None
            result.append(
                {
                    "id": f"msg_{i}",
                    "role": "assistant",
                    "content": msg.content,
                    "citations": citations,
                    "status": "done",
                }
            )
    return result
