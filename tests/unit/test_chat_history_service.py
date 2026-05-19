"""Unit tests for RedisChatMessageHistory + RedisHistoryStore + helpers."""

from __future__ import annotations

import pytest
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from panvel_assistant.services.chat_history_service import (
    MAX_MESSAGES,
    RedisChatMessageHistory,
    RedisHistoryStore,
    serialize_messages_to_text,
)
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


def test_serialize_messages_to_text_empty_returns_empty_string():
    assert serialize_messages_to_text([]) == ""


def test_serialize_messages_to_text_formats_known_roles():
    out = serialize_messages_to_text(
        [
            HumanMessage(content="q"),
            AIMessage(content="a"),
            SystemMessage(content="s"),
        ]
    )
    assert out == "Human: `q`\nAI: `a`\nSystem: `s`"


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
