"""Integration test for POST /chat — end-to-end SSE stream + Redis persistence."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import BaseMessage

from panvel_assistant.main import app
from panvel_assistant.services.chat_history_service import (
    _deserialize,
    history_store,
)
from panvel_assistant.services.llm_service import llm_service


def _parse_sse_frames(raw: str) -> list[tuple[str, str | dict]]:
    """Split a raw ``text/event-stream`` body into ``[(event, payload), ...]``.

    ``token`` frames carry plain text (possibly multi-line, recomposed by joining
    the consecutive ``data:`` fields with ``\\n``); ``done``/``error`` frames
    carry a JSON payload.
    """
    events: list[tuple[str, str | dict]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        event_type = lines[0].removeprefix("event: ")
        data_lines = [line.removeprefix("data: ") for line in lines[1:]]
        data = "\n".join(data_lines)
        if event_type == "token":
            events.append((event_type, data))
        else:
            events.append((event_type, json.loads(data)))
    return events


@pytest.fixture
def fake_redis_client():
    """Inject a fakeredis client into history_store and bypass connect/disconnect."""
    client = fake_aioredis.FakeRedis(decode_responses=True)
    original_connect = history_store.connect
    original_disconnect = history_store.disconnect
    original_client = history_store._client

    async def _noop_connect() -> None:
        history_store._client = client

    async def _noop_disconnect() -> None:
        history_store._client = None

    history_store.connect = _noop_connect  # type: ignore[method-assign]
    history_store.disconnect = _noop_disconnect  # type: ignore[method-assign]

    try:
        yield client
    finally:
        history_store.connect = original_connect  # type: ignore[method-assign]
        history_store.disconnect = original_disconnect  # type: ignore[method-assign]
        history_store._client = original_client


@pytest.fixture
def stub_gemini(monkeypatch):
    """Replace LLMService.stream_response with a deterministic token generator.

    Patching the public service method rather than the underlying
    ``ChatGoogleGenerativeAI`` client because Pydantic models reject ad-hoc
    attribute assignment on bound methods.
    """

    async def _stream(messages: Sequence[BaseMessage]) -> AsyncIterator[str]:
        for piece in ["Eu ", "sou ", "Panvel."]:
            yield piece

    monkeypatch.setattr(llm_service, "stream_response", _stream)


async def test_post_chat_streams_tokens_and_persists_history_across_turns(
    fake_redis_client, stub_gemini
):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/chat",
                json={"session_id": "itest", "message": "oi"},
                headers={"Accept": "text/event-stream"},
            )

            assert r1.status_code == 200
            assert r1.headers["content-type"].startswith("text/event-stream")
            events_1 = _parse_sse_frames(r1.text)

            event_types_1 = [evt for evt, _ in events_1]
            assert event_types_1 == ["token", "token", "token", "done"]
            assert [payload for _, payload in events_1[:3]] == [
                "Eu ",
                "sou ",
                "Panvel.",
            ]
            assert events_1[-1][1] == {"session_id": "itest"}

            assert await fake_redis_client.llen("chat:history:itest") == 2

            r2 = await client.post(
                "/chat",
                json={"session_id": "itest", "message": "e ai?"},
                headers={"Accept": "text/event-stream"},
            )
            assert r2.status_code == 200
            assert [evt for evt, _ in _parse_sse_frames(r2.text)] == [
                "token",
                "token",
                "token",
                "done",
            ]
            assert await fake_redis_client.llen("chat:history:itest") == 4

            raw_entries = await fake_redis_client.lrange("chat:history:itest", 0, -1)
            messages = [_deserialize(raw) for raw in raw_entries]
            assert [m.type for m in messages] == ["human", "ai", "human", "ai"]
            assert messages[0].content == "oi"
            assert messages[1].content == "Eu sou Panvel."
            assert messages[2].content == "e ai?"
            assert messages[3].content == "Eu sou Panvel."


async def test_post_chat_returns_422_for_empty_message(fake_redis_client, stub_gemini):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/chat",
                json={"session_id": "x", "message": ""},
            )

    assert response.status_code == 422
