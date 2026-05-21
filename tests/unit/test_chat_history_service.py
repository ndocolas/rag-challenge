"""Unit tests for RedisChatMessageHistory + RedisHistoryStore + helpers."""

from __future__ import annotations

import pytest
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import AIMessage, HumanMessage

from panvel_assistant.services.chat_history_service import (
    MAX_MESSAGES,
    LockBusyError,
    RedisChatMessageHistory,
    RedisHistoryStore,
    _redact_url,
)
from panvel_assistant.utils.exceptions import SessionBusyError
from panvel_assistant.utils.settings import settings


@pytest.fixture
async def fake_client():
    """A standalone fakeredis client with text decoding enabled."""
    client = fake_aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def history(fake_client):
    """RedisChatMessageHistory bound to a fresh session id."""
    return RedisChatMessageHistory(session_id="t-session", client=fake_client)


async def test_round_trip_preserves_message_types_and_content(history):
    await history.aadd_messages([HumanMessage(content="oi"), AIMessage(content="olá")])
    messages = await history.aget_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "oi"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "olá"


async def test_aget_messages_returns_empty_for_unknown_session(history):
    assert await history.aget_messages() == []


async def test_aadd_messages_caps_history_at_max_messages(history):
    for i in range(MAX_MESSAGES + 5):
        await history.aadd_messages([HumanMessage(content=f"msg-{i}")])

    messages = await history.aget_messages()
    assert len(messages) == MAX_MESSAGES
    assert messages[0].content == f"msg-{5}"
    assert messages[-1].content == f"msg-{MAX_MESSAGES + 4}"


async def test_aadd_messages_renews_ttl(history, fake_client):
    await history.aadd_messages([HumanMessage(content="x")])
    ttl = await fake_client.ttl("chat:history:t-session")

    assert 0 < ttl <= settings.CHAT_HISTORY_TTL_SECONDS


async def test_aadd_messages_empty_is_noop(history, fake_client):
    await history.aadd_messages([])
    assert await fake_client.exists("chat:history:t-session") == 0


async def test_aclear_deletes_the_key(history, fake_client):
    await history.aadd_messages([HumanMessage(content="x")])
    await history.aclear()

    assert await fake_client.exists("chat:history:t-session") == 0
    assert await history.aget_messages() == []


def test_sync_messages_property_raises(history):
    with pytest.raises(NotImplementedError):
        _ = history.messages


def test_sync_add_message_raises(history):
    with pytest.raises(NotImplementedError):
        history.add_message(HumanMessage(content="x"))


def test_sync_clear_raises(history):
    with pytest.raises(NotImplementedError):
        history.clear()



async def test_store_get_session_history_requires_connect():
    store = RedisHistoryStore()
    with pytest.raises(RuntimeError, match="not connected"):
        store.get_session_history("s1")


async def test_store_get_session_history_returns_history_after_inject(fake_client):
    store = RedisHistoryStore()
    store._client = fake_client

    history = store.get_session_history("abc")
    await history.aadd_messages([HumanMessage(content="hi")])

    refetched = store.get_session_history("abc")
    messages = await refetched.aget_messages()
    assert len(messages) == 1
    assert messages[0].content == "hi"


# A6 — corrupted entries on read must not abort the whole turn.
async def test_aget_messages_skips_corrupted_entries(history, fake_client):
    """Garbage entries in a session list are silently dropped, valid ones kept."""
    key = "chat:history:t-session"
    await fake_client.rpush(
        key,
        "not-json",
        "v1:not-json-either",
        # A correctly-shaped v1 payload (HumanMessage):
        'v1:{"type": "human", "data": {"content": "real", "type": "human"}}',
    )

    messages = await history.aget_messages()
    assert len(messages) == 1
    assert messages[0].content == "real"


# A7 — credentials in REDIS_URL must be redacted from logs.
def test_redact_url_strips_password():
    assert _redact_url("redis://user:secret@host:6379/0") == "redis://user:***@host:6379/0"


def test_redact_url_passthrough_when_no_password():
    assert _redact_url("redis://host:6379/0") == "redis://host:6379/0"


# A2 — turn locking serializes per-session concurrent requests.
async def test_acquire_lock_then_busy_raises_session_busy(fake_client):
    store = RedisHistoryStore()
    store._client = fake_client

    token = await store.acquire_lock("s")
    assert isinstance(token, str) and len(token) >= 16

    with pytest.raises(SessionBusyError):
        await store.acquire_lock("s")

    await store.release_lock("s", token)
    # After release another caller can acquire.
    second = await store.acquire_lock("s")
    assert second != token


async def test_lock_busy_error_is_session_busy_subclass():
    assert issubclass(LockBusyError, SessionBusyError)


async def test_release_lock_no_op_when_token_mismatch(fake_client):
    """Releasing with the wrong token leaves the lock intact."""
    store = RedisHistoryStore()
    store._client = fake_client

    real = await store.acquire_lock("s")
    await store.release_lock("s", token="not-the-real-token")
    with pytest.raises(SessionBusyError):
        await store.acquire_lock("s")
    await store.release_lock("s", real)


async def test_ping_returns_false_when_disconnected():
    store = RedisHistoryStore()
    assert await store.ping(timeout=0.05) is False


async def test_ping_returns_true_when_connected(fake_client):
    store = RedisHistoryStore()
    store._client = fake_client
    assert await store.ping(timeout=0.5) is True


# C-3 — release_lock must complete even when the caller is being cancelled.
async def test_release_lock_completes_under_caller_cancellation(fake_client):
    """Cancellation upstream must not leak the per-session lock for the full TTL."""
    import asyncio as _asyncio

    store = RedisHistoryStore()
    store._client = fake_client

    token = await store.acquire_lock("s")
    # Confirm the lock is held.
    with pytest.raises(SessionBusyError):
        await store.acquire_lock("s")

    async def release_and_get_cancelled():
        import contextlib

        task = _asyncio.create_task(store.release_lock("s", token))
        # Yield once so the shielded body has a chance to start.
        await _asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(_asyncio.CancelledError):
            await task

    await release_and_get_cancelled()

    # Despite the cancellation, the key must have been deleted by the shielded
    # body, so the next acquire succeeds.
    second = await store.acquire_lock("s")
    assert second != token


async def test_drain_pending_filters_by_session(fake_client):
    """``drain_pending(session_id=...)`` only awaits that session's tasks."""
    import asyncio as _asyncio

    store = RedisHistoryStore()
    store._client = fake_client

    done_a = _asyncio.Event()
    done_b = _asyncio.Event()

    async def slow_a():
        await _asyncio.sleep(0.05)
        done_a.set()

    async def slow_b():
        await _asyncio.sleep(2.0)
        done_b.set()

    loop = _asyncio.get_running_loop()
    task_a = loop.create_task(slow_a())
    task_b = loop.create_task(slow_b())
    store.register_pending(task_a, session_id="A")
    store.register_pending(task_b, session_id="B")

    await store.drain_pending(timeout=1.0, session_id="A")
    assert done_a.is_set()
    assert not done_b.is_set()
    task_b.cancel()
