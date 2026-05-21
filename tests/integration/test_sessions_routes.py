"""Integration tests for GET /sessions and GET /sessions/{session_id}/history."""

from __future__ import annotations

import pytest
import httpx
from asgi_lifespan import LifespanManager
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import AIMessage, HumanMessage

from panvel_assistant.main import app
from panvel_assistant.services.chat_history_service import (
    RedisChatMessageHistory,
    history_store,
)
from panvel_assistant.services.trace_service import trace_service


@pytest.fixture
def fake_redis():
    """Inject fakeredis into history_store and trace_service; bypass connect/disconnect."""
    client = fake_aioredis.FakeRedis(decode_responses=True)
    trace_client = fake_aioredis.FakeRedis(decode_responses=True)

    original_hs_connect = history_store.connect
    original_hs_disconnect = history_store.disconnect
    original_hs_client = history_store._client

    original_ts_connect = trace_service.connect
    original_ts_disconnect = trace_service.disconnect

    async def _hs_noop_connect() -> None:
        history_store._client = client

    async def _hs_noop_disconnect() -> None:
        history_store._client = None

    async def _ts_noop_connect() -> None:
        trace_service._client = trace_client  # type: ignore[attr-defined]

    async def _ts_noop_disconnect() -> None:
        trace_service._client = None  # type: ignore[attr-defined]

    history_store.connect = _hs_noop_connect  # type: ignore[method-assign]
    history_store.disconnect = _hs_noop_disconnect  # type: ignore[method-assign]
    trace_service.connect = _ts_noop_connect  # type: ignore[method-assign]
    trace_service.disconnect = _ts_noop_disconnect  # type: ignore[method-assign]
    history_store._client = client

    try:
        yield client
    finally:
        history_store._client = original_hs_client
        history_store.connect = original_hs_connect  # type: ignore[method-assign]
        history_store.disconnect = original_hs_disconnect  # type: ignore[method-assign]
        trace_service.connect = original_ts_connect  # type: ignore[method-assign]
        trace_service.disconnect = original_ts_disconnect  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_list_sessions_empty(fake_redis):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_sessions_returns_metadata(fake_redis):
    await history_store.save_session_meta("sess-x", "Posologia do Advil")
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["session_id"] == "sess-x"
    assert data[0]["title"] == "Posologia do Advil"


@pytest.mark.asyncio
async def test_get_history_not_found(fake_redis):
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sessions/nonexistent/history")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_get_history_returns_messages(fake_redis):
    sess = RedisChatMessageHistory(
        "hist-1", history_store.client, ttl_seconds=300
    )
    await sess.aadd_messages(
        [
            HumanMessage(content="Qual a posologia?"),
            AIMessage(
                content="A posologia é X.",
                additional_kwargs={
                    "citations": [
                        {"bula_id": "b1", "med_name": "Advil", "snippet": "..."}
                    ]
                },
            ),
        ]
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sessions/hist-1/history")
    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Qual a posologia?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["citations"][0]["med_name"] == "Advil"
