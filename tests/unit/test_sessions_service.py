"""Unit tests for RedisHistoryStore session-metadata methods."""
from __future__ import annotations

import pytest
from fakeredis import aioredis as fake_aioredis

from panvel_assistant.services.chat_history_service import RedisHistoryStore


@pytest.fixture
async def store():
    client = fake_aioredis.FakeRedis(decode_responses=True)
    s = RedisHistoryStore()
    s._client = client
    return s


async def test_save_session_meta_stores_entry(store):
    await store.save_session_meta("sess-1", "Quais são as contraindicações?")
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-1"
    assert sessions[0]["title"] == "Quais são as contraindicações?"
    assert "created_at" in sessions[0]


async def test_save_session_meta_idempotent(store):
    await store.save_session_meta("sess-1", "Primeira mensagem")
    await store.save_session_meta("sess-1", "Segunda — não deve sobrescrever")
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["title"] == "Primeira mensagem"


async def test_list_sessions_returns_newest_first(store):
    await store.save_session_meta("sess-a", "Mensagem A")
    await store.save_session_meta("sess-b", "Mensagem B")
    sessions = await store.list_sessions()
    assert sessions[0]["session_id"] == "sess-b"
    assert sessions[1]["session_id"] == "sess-a"


async def test_list_sessions_empty(store):
    sessions = await store.list_sessions()
    assert sessions == []


async def test_save_session_meta_truncates_title(store):
    long_title = "x" * 100
    await store.save_session_meta("sess-1", long_title)
    sessions = await store.list_sessions()
    assert len(sessions[0]["title"]) == 60
