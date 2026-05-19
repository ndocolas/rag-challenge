"""Unit tests for ChatService.handle_turn — token/done/error SSE flow + history persistence."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services import (
    chat_history_service,
)
from panvel_assistant.services import (
    chat_service as chat_service_module,
)
from panvel_assistant.services import (
    llm_service as llm_service_module,
)
from panvel_assistant.services.chat_service import chat_service


class _FakeHistory:
    """Stand-in for RedisChatMessageHistory; records aadd_messages calls."""

    def __init__(self, past: list[BaseMessage] | None = None) -> None:
        self.past = list(past or [])
        self.appended: list[list[BaseMessage]] = []

    async def aget_messages(self) -> list[BaseMessage]:
        return self.past

    async def aadd_messages(self, messages):
        self.appended.append(list(messages))


def _parse_sse_event(frame: str) -> tuple[str, dict]:
    """Decode a JSON SSE frame into (event_type, json_payload)."""
    event_line, data_line = frame.strip().split("\n", 1)
    event_type = event_line.removeprefix("event: ").strip()
    data_json = data_line.removeprefix("data: ").strip()
    return event_type, json.loads(data_json)


def _parse_sse_text_event(frame: str) -> tuple[str, str]:
    """Decode a plain-text SSE frame into (event_type, joined_text)."""
    lines = frame.strip().split("\n")
    event_type = lines[0].removeprefix("event: ").strip()
    data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
    return event_type, data


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


@pytest.fixture
def fake_history(monkeypatch):
    """Patch history_store.get_session_history to return a deterministic fake."""
    fake = _FakeHistory()
    monkeypatch.setattr(
        chat_history_service.history_store,
        "get_session_history",
        lambda session_id: fake,
    )
    return fake


async def test_handle_turn_emits_tokens_then_done_and_appends_history(
    monkeypatch, fake_history
):
    captured_messages: list[list[BaseMessage]] = []

    async def fake_stream(messages):
        captured_messages.append(list(messages))
        async for chunk in _async_iter(["Olá", " mundo"]):
            yield chunk

    monkeypatch.setattr(llm_service_module.llm_service, "stream_response", fake_stream)

    frames = [
        frame
        async for frame in chat_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert len(frames) == 3
    assert _parse_sse_text_event(frames[0]) == ("token", "Olá")
    assert _parse_sse_text_event(frames[1]) == ("token", " mundo")
    assert _parse_sse_event(frames[2]) == ("done", {"session_id": "s"})

    assert len(fake_history.appended) == 1
    user_msg, assistant_msg = fake_history.appended[0]
    assert isinstance(user_msg, HumanMessage)
    assert user_msg.content == "oi"
    assert isinstance(assistant_msg, AIMessage)
    assert assistant_msg.content == "Olá mundo"

    sent = captured_messages[0]
    assert sent[0].type == "system"
    assert sent[-1].content == "oi"


async def test_handle_turn_includes_past_history_in_llm_call(monkeypatch, fake_history):
    fake_history.past = [
        HumanMessage(content="anterior?"),
        AIMessage(content="resposta antiga"),
    ]
    captured: list[list[BaseMessage]] = []

    async def fake_stream(messages):
        captured.append(list(messages))
        async for chunk in _async_iter(["ok"]):
            yield chunk

    monkeypatch.setattr(llm_service_module.llm_service, "stream_response", fake_stream)

    frames = [
        frame
        async for frame in chat_service.handle_turn(
            ChatRequest(session_id="s", message="novo")
        )
    ]

    assert len(frames) == 2
    sent = captured[0]
    assert sent[0].type == "system"
    assert sent[1].content == "anterior?"
    assert sent[2].content == "resposta antiga"
    assert sent[3].content == "novo"


async def test_handle_turn_emits_error_event_and_skips_history_on_failure(
    monkeypatch, fake_history
):
    async def boom_stream(messages):
        if False:
            yield ""
        raise RuntimeError("upstream blew up")

    monkeypatch.setattr(llm_service_module.llm_service, "stream_response", boom_stream)

    frames = [
        frame
        async for frame in chat_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert len(frames) == 1
    event_type, payload = _parse_sse_event(frames[0])
    assert event_type == "error"
    assert payload == {"message": "upstream blew up"}
    assert fake_history.appended == []


async def test_handle_turn_preserves_newlines_inside_token_deltas(
    monkeypatch, fake_history
):
    async def fake_stream(messages):
        async for chunk in _async_iter(["linha1\nlinha2", "\n\nfim"]):
            yield chunk

    monkeypatch.setattr(llm_service_module.llm_service, "stream_response", fake_stream)

    frames = [
        frame
        async for frame in chat_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert _parse_sse_text_event(frames[0]) == ("token", "linha1\nlinha2")
    assert _parse_sse_text_event(frames[1]) == ("token", "\n\nfim")
    assert _parse_sse_event(frames[2]) == ("done", {"session_id": "s"})


def test_chat_service_singleton_is_the_module_export():
    assert chat_service is chat_service_module.chat_service
