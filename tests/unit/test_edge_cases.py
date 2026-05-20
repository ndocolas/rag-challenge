"""Edge-case tests covering scenarios the unit suites might miss.

E1 — LLM yields 0 chunks: chat_service must emit ``empty_response`` and skip history.
E2 — empty past history: ``messages`` passed to the LLM still has the system prompt.
E3 — multimodal segments: ``_coerce_content_to_text`` filters them quietly.
E4 — ``CancelledError`` mid-stream: propagates and persists the partial turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from panvel_assistant.assistant.assistant_service import (
    _coerce_content_to_text,
    assistant_service,
)
from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services import chat_history_service


class _FakeHistory:
    def __init__(self, past: list[BaseMessage] | None = None) -> None:
        self.past = list(past or [])
        self.appended: list[list[BaseMessage]] = []

    async def aget_messages(self) -> list[BaseMessage]:
        return self.past

    async def aadd_messages(self, messages):
        self.appended.append(list(messages))


async def _async_iter(items: list[dict]) -> AsyncIterator[dict]:
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# E1 — empty LLM output.
# ---------------------------------------------------------------------------


async def test_e1_empty_llm_output_yields_empty_response_event_and_skips_history(
    monkeypatch,
):
    fake = _FakeHistory()
    monkeypatch.setattr(
        chat_history_service.history_store,
        "get_session_history",
        lambda session_id: fake,
    )

    async def empty_stream(_msgs):
        async for ev in _async_iter([{"type": "done"}]):
            yield ev

    monkeypatch.setattr(assistant_service, "stream_with_tools", empty_stream)

    frames = [
        f
        async for f in assistant_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    # First frame is the trace_id event, second is the empty_response error.
    assert len(frames) == 2
    assert "event: trace_id" in frames[0]
    assert "event: error" in frames[1]
    assert "empty_response" in frames[1]
    assert fake.appended == []


# ---------------------------------------------------------------------------
# E2 — empty past history.
# ---------------------------------------------------------------------------


async def test_e2_empty_history_still_passes_system_and_human(monkeypatch):
    fake = _FakeHistory(past=[])
    monkeypatch.setattr(
        chat_history_service.history_store,
        "get_session_history",
        lambda session_id: fake,
    )

    captured: list[list[BaseMessage]] = []

    async def fake_stream(msgs):
        captured.append(list(msgs))
        async for ev in _async_iter(
            [{"type": "token", "text": "ok"}, {"type": "done"}]
        ):
            yield ev

    monkeypatch.setattr(assistant_service, "stream_with_tools", fake_stream)

    _ = [
        f
        async for f in assistant_service.handle_turn(
            ChatRequest(session_id="s", message="primeiro")
        )
    ]

    sent = captured[0]
    assert len(sent) == 2
    assert sent[0].type == "system"
    assert isinstance(sent[1], HumanMessage)
    assert sent[1].content == "primeiro"


# ---------------------------------------------------------------------------
# E3 — multimodal/tool-call segments.
# ---------------------------------------------------------------------------


def test_e3_multimodal_segments_filtered_to_text():
    content = [
        {"text": "Olá "},
        {"function_call": {"name": "buscar_filiais", "args": {}}},
        {"text": "mundo"},
        {"image_url": "https://example.com/x.png"},
    ]
    assert _coerce_content_to_text(content) == "Olá mundo"


# ---------------------------------------------------------------------------
# E4 — CancelledError mid-stream propagates and persists partial.
# ---------------------------------------------------------------------------


async def test_e4_cancellation_persists_partial_and_propagates(monkeypatch):
    fake = _FakeHistory()
    monkeypatch.setattr(
        chat_history_service.history_store,
        "get_session_history",
        lambda session_id: fake,
    )

    async def cancelling_stream(_msgs):
        yield {"type": "token", "text": "primeiro "}
        yield {"type": "token", "text": "pedaço"}
        raise asyncio.CancelledError

    monkeypatch.setattr(assistant_service, "stream_with_tools", cancelling_stream)

    received: list[str] = []

    async def consume():
        async for f in assistant_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        ):
            received.append(f)

    raised: BaseException | None = None
    try:
        await consume()
    except asyncio.CancelledError as exc:
        raised = exc

    assert isinstance(raised, asyncio.CancelledError)
    # trace_id frame + the two partial token frames before cancellation
    assert len(received) == 3
    await chat_history_service.history_store.drain_pending(timeout=2.0)
    assert len(fake.appended) == 1
    user_msg, ai_msg = fake.appended[0]
    assert isinstance(user_msg, HumanMessage)
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "primeiro pedaço"
