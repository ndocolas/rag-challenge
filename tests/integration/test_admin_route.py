"""Integration tests for GET /admin/traces/{trace_id}.

Exercises the full round-trip: POST /chat → capture trace_id from first SSE
event → GET /admin/traces/{trace_id} → verify the persisted trace structure.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import BaseMessage

from panvel_assistant.assistant.assistant_service import assistant_service
from panvel_assistant.main import app
from panvel_assistant.services.chat_history_service import history_store
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.services.trace_service import trace_service


def _parse_sse_frames(raw: str) -> list[tuple[str, str | dict]]:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    """Inject fakeredis into both history_store and trace_service."""
    history_client = fake_aioredis.FakeRedis(decode_responses=True)
    trace_client = fake_aioredis.FakeRedis(decode_responses=True)

    # history_store
    orig_hs_connect = history_store.connect
    orig_hs_disconnect = history_store.disconnect
    orig_hs_client = history_store._client

    async def _hs_connect() -> None:
        history_store._client = history_client

    async def _hs_disconnect() -> None:
        history_store._client = None

    history_store.connect = _hs_connect  # type: ignore[method-assign]
    history_store.disconnect = _hs_disconnect  # type: ignore[method-assign]

    # trace_service
    orig_ts_connect = trace_service.connect
    orig_ts_disconnect = trace_service.disconnect
    orig_ts_redis = trace_service._redis

    async def _ts_connect() -> None:
        trace_service._redis = trace_client

    async def _ts_disconnect() -> None:
        trace_service._redis = None

    trace_service.connect = _ts_connect  # type: ignore[method-assign]
    trace_service.disconnect = _ts_disconnect  # type: ignore[method-assign]

    try:
        yield trace_client
    finally:
        history_store.connect = orig_hs_connect  # type: ignore[method-assign]
        history_store.disconnect = orig_hs_disconnect  # type: ignore[method-assign]
        history_store._client = orig_hs_client
        trace_service.connect = orig_ts_connect  # type: ignore[method-assign]
        trace_service.disconnect = orig_ts_disconnect  # type: ignore[method-assign]
        trace_service._redis = orig_ts_redis


@pytest.fixture(autouse=True)
def stub_filiais_load(monkeypatch):
    monkeypatch.setattr(filiais_service, "load", lambda *_, **__: None)


@pytest.fixture(autouse=True)
def stub_session_lock(monkeypatch):
    async def _acquire(self, session_id: str) -> str:
        return "test-token"

    async def _release(self, session_id: str, token: str) -> None:
        return None

    from panvel_assistant.services.chat_history_service import RedisHistoryStore

    monkeypatch.setattr(RedisHistoryStore, "acquire_lock", _acquire)
    monkeypatch.setattr(RedisHistoryStore, "release_lock", _release)


@pytest.fixture
def stub_gemini_with_sources(monkeypatch):
    """Stub that emits a sources event so citations appear in the trace."""
    _citations = [
        {
            "bula_id": "bula-1",
            "med_name": "Ritalina",
            "med_variant": None,
            "section_canonical": "IAP_3_CONTRAINDICACOES",
            "section_label": "Quando não devo usar",
            "source_page": None,
            "snippet": "Contraindicado em hipertensão.",
        }
    ]

    async def _stream(messages: Sequence[BaseMessage]) -> AsyncIterator[dict]:
        yield {"type": "sources", "citations": _citations}
        yield {"type": "token", "text": "Não use com hipertensão."}
        yield {"type": "done", "tokens_in": 42, "tokens_out": 18}

    monkeypatch.setattr(assistant_service, "stream_with_tools", _stream)


@pytest.fixture
def stub_gemini_simple(monkeypatch):
    """Minimal stub — no tools, no sources."""

    async def _stream(messages: Sequence[BaseMessage]) -> AsyncIterator[dict]:
        yield {"type": "token", "text": "Olá!"}
        yield {"type": "done", "tokens_in": 10, "tokens_out": 5}

    monkeypatch.setattr(assistant_service, "stream_with_tools", _stream)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_chat_emits_trace_id_as_first_sse_event(fake_redis, stub_gemini_simple):
    """The first SSE frame from POST /chat must be a trace_id event."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/chat",
                json={"session_id": "admin-t1", "message": "oi"},
                headers={"Accept": "text/event-stream"},
            )

    assert r.status_code == 200
    events = _parse_sse_frames(r.text)
    assert events[0][0] == "trace_id"
    assert isinstance(events[0][1]["trace_id"], str)
    assert len(events[0][1]["trace_id"]) > 0


async def test_admin_get_trace_returns_full_structure(fake_redis, stub_gemini_with_sources):
    """GET /admin/traces/{trace_id} must return the persisted audit JSON."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_chat = await client.post(
                "/chat",
                json={"session_id": "admin-t2", "message": "contraindicações ritalina"},
                headers={"Accept": "text/event-stream"},
            )
            assert r_chat.status_code == 200

            events = _parse_sse_frames(r_chat.text)
            assert events[0][0] == "trace_id"
            tid = events[0][1]["trace_id"]

            r_trace = await client.get(f"/admin/traces/{tid}")

    assert r_trace.status_code == 200
    trace = r_trace.json()

    assert trace["trace_id"] == tid
    assert trace["session_id"] == "admin-t2"
    assert trace["user_message"] == "contraindicações ritalina"
    assert trace["final_response"] == "Não use com hipertensão."
    assert trace["tokens_in"] == 42
    assert trace["tokens_out"] == 18
    assert trace["duration_ms"] > 0

    # citations were emitted by the sources event
    assert len(trace["citations"]) == 1
    assert trace["citations"][0]["med_name"] == "Ritalina"

    # tool_calls is empty (stub emits no tools)
    assert trace["tool_calls"] == []


async def test_admin_get_trace_returns_404_for_unknown_id(fake_redis):
    """A non-existent trace_id returns 404 with the standard error envelope."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/admin/traces/does-not-exist")

    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "http_404"
    assert body["error"]["status_code"] == 404
    assert "trace_id" in body["error"]


async def test_admin_trace_records_llm_steps(fake_redis, stub_gemini_simple):
    """The trace must contain at least one llm_stream step from stream_with_tools.

    With the simple stub, stream_with_tools is replaced entirely so no
    trace_service.add_step calls come from it. This test verifies the
    end-to-end round-trip produces a trace retrievable via the admin API.
    """
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_chat = await client.post(
                "/chat",
                json={"session_id": "admin-t3", "message": "posologia"},
                headers={"Accept": "text/event-stream"},
            )
            assert r_chat.status_code == 200
            tid = _parse_sse_frames(r_chat.text)[0][1]["trace_id"]

            r_trace = await client.get(f"/admin/traces/{tid}")

    assert r_trace.status_code == 200
    trace = r_trace.json()
    assert trace["trace_id"] == tid
    assert trace["session_id"] == "admin-t3"
    assert trace["final_response"] == "Olá!"
