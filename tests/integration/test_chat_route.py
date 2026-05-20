"""Integration test for POST /chat — end-to-end SSE stream + Redis persistence."""

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
from panvel_assistant.services.chat_history_service import (
    _deserialize,
    history_store,
)
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.services.trace_service import trace_service


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
    """Inject fakeredis into history_store and trace_service; bypass connect/disconnect."""
    client = fake_aioredis.FakeRedis(decode_responses=True)
    trace_client = fake_aioredis.FakeRedis(decode_responses=True)

    # --- history_store ---
    original_hs_connect = history_store.connect
    original_hs_disconnect = history_store.disconnect
    original_hs_client = history_store._client

    async def _hs_noop_connect() -> None:
        history_store._client = client

    async def _hs_noop_disconnect() -> None:
        history_store._client = None

    history_store.connect = _hs_noop_connect  # type: ignore[method-assign]
    history_store.disconnect = _hs_noop_disconnect  # type: ignore[method-assign]

    # --- trace_service ---
    original_ts_connect = trace_service.connect
    original_ts_disconnect = trace_service.disconnect
    original_ts_redis = trace_service._redis

    async def _ts_noop_connect() -> None:
        trace_service._redis = trace_client

    async def _ts_noop_disconnect() -> None:
        trace_service._redis = None

    trace_service.connect = _ts_noop_connect  # type: ignore[method-assign]
    trace_service.disconnect = _ts_noop_disconnect  # type: ignore[method-assign]

    try:
        yield client
    finally:
        history_store.connect = original_hs_connect  # type: ignore[method-assign]
        history_store.disconnect = original_hs_disconnect  # type: ignore[method-assign]
        history_store._client = original_hs_client
        trace_service.connect = original_ts_connect  # type: ignore[method-assign]
        trace_service.disconnect = original_ts_disconnect  # type: ignore[method-assign]
        trace_service._redis = original_ts_redis


@pytest.fixture(autouse=True)
def stub_filiais_load(monkeypatch):
    """Skip the real parquet read inside the lifespan hook; tests don't need it."""
    monkeypatch.setattr(filiais_service, "load", lambda *_, **__: None)


@pytest.fixture(autouse=True)
def stub_session_lock(monkeypatch):
    """Bypass the route-level session lock — fakeredis does not round-trip the
    CAS release reliably, which is unrelated to what these tests cover.
    """

    async def _acquire(self, session_id: str) -> str:
        return "test-token"

    async def _release(self, session_id: str, token: str) -> None:
        return None

    from panvel_assistant.services.chat_history_service import RedisHistoryStore

    monkeypatch.setattr(RedisHistoryStore, "acquire_lock", _acquire)
    monkeypatch.setattr(RedisHistoryStore, "release_lock", _release)


@pytest.fixture
def stub_gemini(monkeypatch):
    """Replace AssistantService.stream_with_tools with a deterministic event generator.

    Patching the public service method rather than the underlying
    ``ChatGoogleGenerativeAI`` client because Pydantic models reject ad-hoc
    attribute assignment on bound methods.
    """

    async def _stream(messages: Sequence[BaseMessage]) -> AsyncIterator[dict]:
        for piece in ["Eu ", "sou ", "Panvel."]:
            yield {"type": "token", "text": piece}
        yield {"type": "done"}

    monkeypatch.setattr(assistant_service, "stream_with_tools", _stream)


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
            assert event_types_1 == ["trace_id", "token", "token", "token", "done"]
            assert "trace_id" in events_1[0][1]
            assert [payload for _, payload in events_1[1:4]] == [
                "Eu ",
                "sou ",
                "Panvel.",
            ]
            assert events_1[-1][1] == {"session_id": "itest"}

            # The route drains this session's pending persistence before
            # releasing the lock, so by the time ``r1`` returned the LIST is
            # already populated — no explicit drain needed here.
            assert await fake_redis_client.llen("chat:history:itest") == 2

            r2 = await client.post(
                "/chat",
                json={"session_id": "itest", "message": "e ai?"},
                headers={"Accept": "text/event-stream"},
            )
            assert r2.status_code == 200
            assert [evt for evt, _ in _parse_sse_frames(r2.text)] == [
                "trace_id",
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
    body = response.json()
    # 422 must travel through the same envelope as every other error response.
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["status_code"] == 422
    assert "trace_id" in body["error"]
    assert isinstance(body["error"]["fields"], list)


@pytest.fixture
def small_body_limit(monkeypatch):
    """Shrink ``MAX_REQUEST_BODY_BYTES`` for the duration of a single test.

    The body-size middleware reads the limit via ``get_settings()`` on each
    request, so monkeypatching the cached singleton applies immediately.
    """
    from panvel_assistant.utils.settings import get_settings

    real = get_settings()

    def _set(limit_bytes: int):
        patched = real.model_copy(update={"MAX_REQUEST_BODY_BYTES": limit_bytes})
        monkeypatch.setattr(
            "panvel_assistant.main.get_settings", lambda: patched
        )

    return _set


async def test_body_size_middleware_rejects_chunked_oversized_body(
    fake_redis_client, stub_gemini, small_body_limit
):
    """A ``Transfer-Encoding: chunked`` POST that exceeds the cap must 413."""
    small_body_limit(4096)

    async def chunked_body():
        # 32 KB pushed through chunked encoding without a Content-Length header.
        chunk = b"x" * 8192
        for _ in range(4):
            yield chunk

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/chat",
                content=chunked_body(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 413
            body = resp.json()
            assert body["error"]["code"] == "payload_too_large"
            assert body["error"]["status_code"] == 413
            assert body["error"]["limit_bytes"] == 4096
            assert "trace_id" in body["error"]


async def test_turn_n_plus_1_sees_history_persisted_by_turn_n(
    fake_redis_client, stub_gemini
):
    """The lock must be held until persistence settles so turn N+1 isn't racy."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r1 = await client.post(
                "/chat",
                json={"session_id": "consist", "message": "primeira"},
                headers={"Accept": "text/event-stream"},
            )
            assert r1.status_code == 200
            # No explicit drain here — the route must have done it.
            assert await fake_redis_client.llen("chat:history:consist") == 2

            r2 = await client.post(
                "/chat",
                json={"session_id": "consist", "message": "segunda"},
                headers={"Accept": "text/event-stream"},
            )
            assert r2.status_code == 200
            assert await fake_redis_client.llen("chat:history:consist") == 4


async def test_body_size_middleware_rejects_content_length_oversized(
    fake_redis_client, stub_gemini, small_body_limit
):
    """A declared ``Content-Length`` above the cap is rejected pre-read."""
    small_body_limit(256)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/chat",
                json={"session_id": "x", "message": "x" * 1000},
            )
            assert resp.status_code == 413
            assert resp.json()["error"]["code"] == "payload_too_large"


async def test_post_chat_enforces_rate_limit(
    fake_redis_client, stub_gemini, monkeypatch
):
    """After exceeding the per-minute budget the route must return 429."""
    monkeypatch.setattr(
        "panvel_assistant.utils.settings.Settings.model_config",
        # No-op; rate limit is overridden via the dependency below.
        (__import__(
                "panvel_assistant.utils.settings", fromlist=["Settings"]
            ).Settings).model_config,
    )

    from panvel_assistant.main import app as fastapi_app
    from panvel_assistant.utils.settings import get_settings

    real_settings = get_settings()
    # 3 hits/min is enough to assert a 429 on the 4th call.
    overridden = real_settings.model_copy(update={"CHAT_RATE_LIMIT_PER_MINUTE": 3})
    fastapi_app.dependency_overrides[get_settings] = lambda: overridden

    try:
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                for _ in range(3):
                    r = await client.post(
                        "/chat",
                        json={"session_id": "ratelimit", "message": "oi"},
                        headers={"Accept": "text/event-stream"},
                    )
                    assert r.status_code == 200
                blocked = await client.post(
                    "/chat",
                    json={"session_id": "ratelimit", "message": "oi"},
                )
                assert blocked.status_code == 429
                body = blocked.json()
                assert body["error"]["code"] == "rate_limited"
                assert body["error"]["status_code"] == 429
                assert body["error"]["limit"] == 3
                assert "trace_id" in body["error"]
                assert blocked.headers["Retry-After"] == "60"
                assert blocked.headers["X-RateLimit-Limit"] == "3"
                assert blocked.headers["X-RateLimit-Remaining"] == "0"
    finally:
        fastapi_app.dependency_overrides.clear()
