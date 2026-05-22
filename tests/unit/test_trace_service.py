"""Unit tests for TraceService — in-memory buffer + Redis persistence."""

from __future__ import annotations

import json

import pytest
from fakeredis import aioredis as fake_aioredis

from bulas_assistant.services.trace_service import TraceService
from bulas_assistant.utils.logger import trace_id_var


@pytest.fixture
def svc():
    """A fresh TraceService instance (not connected to real Redis)."""
    return TraceService()


@pytest.fixture
async def connected_svc():
    """TraceService backed by a fakeredis client."""
    service = TraceService()
    service._redis = fake_aioredis.FakeRedis(decode_responses=True)
    yield service
    await service._redis.aclose()


@pytest.fixture
def trace_id(svc):
    """Set a stable trace_id contextvar and return it."""
    tid = "test-trace-abc123"
    token = trace_id_var.set(tid)
    yield tid
    trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

async def test_start_creates_buffer(svc, trace_id):
    tid = svc.start("sess-1", "hello world")

    assert tid == trace_id
    assert tid in svc._buffers
    buf = svc._buffers[tid]
    assert buf["session_id"] == "sess-1"
    assert buf["user_message"] == "hello world"
    assert buf["steps"] == []
    assert buf["tool_calls"] == []
    assert buf["citations"] == []
    assert buf["final_response"] is None
    assert buf["tokens_in"] is None
    assert buf["tokens_out"] is None


# ---------------------------------------------------------------------------
# add_step
# ---------------------------------------------------------------------------

async def test_add_step_appends(svc, trace_id):
    svc.start("s", "msg")
    svc.add_step("retrieval", 42.5, k=4, returned=3)

    steps = svc._buffers[trace_id]["steps"]
    assert len(steps) == 1
    assert steps[0] == {"name": "retrieval", "latency_ms": 42.5, "k": 4, "returned": 3}


async def test_add_step_noop_without_buffer(svc):
    token = trace_id_var.set("no-buffer-id")
    try:
        svc.add_step("retrieval", 10.0)  # should not raise
    finally:
        trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# add_tool_call
# ---------------------------------------------------------------------------

async def test_add_tool_call_appends(svc, trace_id):
    svc.start("s", "msg")
    svc.add_tool_call("buscar_bulas", {"query": "ritalina"}, "preview text", 88.0)

    calls = svc._buffers[trace_id]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "buscar_bulas"
    assert calls[0]["args"] == {"query": "ritalina"}
    assert calls[0]["result_preview"] == "preview text"
    assert calls[0]["latency_ms"] == 88.0
    assert calls[0]["error"] is None


async def test_add_tool_call_with_error(svc, trace_id):
    svc.start("s", "msg")
    svc.add_tool_call("bad_tool", {}, None, 5.0, error="tool_execution_failed")

    assert svc._buffers[trace_id]["tool_calls"][0]["error"] == "tool_execution_failed"


# ---------------------------------------------------------------------------
# set_citations
# ---------------------------------------------------------------------------

async def test_set_citations(svc, trace_id):
    svc.start("s", "msg")
    citations = [{"bula_id": "bula-1", "med_name": "Ritalina"}]
    svc.set_citations(citations)

    assert svc._buffers[trace_id]["citations"] == citations


# ---------------------------------------------------------------------------
# set_response
# ---------------------------------------------------------------------------

async def test_set_response(svc, trace_id):
    svc.start("s", "msg")
    svc.set_response("Final answer text", tokens_in=100, tokens_out=50)

    buf = svc._buffers[trace_id]
    assert buf["final_response"] == "Final answer text"
    assert buf["tokens_in"] == 100
    assert buf["tokens_out"] == 50


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

async def test_finalize_persists_to_redis_and_removes_buffer(connected_svc):
    tid = "finalize-trace-xyz"
    token = trace_id_var.set(tid)
    try:
        connected_svc.start("sess-fin", "finalize me")
        connected_svc.add_step("retrieval", 30.0, k=4, returned=2)
        connected_svc.set_response("answer", tokens_in=10, tokens_out=5)

        await connected_svc.finalize()

        # Buffer removed from memory
        assert tid not in connected_svc._buffers

        # Data persisted to Redis
        raw = await connected_svc._redis.get(f"trace:{tid}")
        assert raw is not None
        data = json.loads(raw)
        assert data["trace_id"] == tid
        assert data["session_id"] == "sess-fin"
        assert data["final_response"] == "answer"
        assert len(data["steps"]) == 1
        assert data["duration_ms"] > 0
    finally:
        trace_id_var.reset(token)


async def test_finalize_noop_when_no_buffer(connected_svc):
    token = trace_id_var.set("ghost-trace")
    try:
        await connected_svc.finalize()  # must not raise
    finally:
        trace_id_var.reset(token)


async def test_finalize_noop_when_redis_not_connected(svc):
    tid = "no-redis-trace"
    token = trace_id_var.set(tid)
    try:
        svc.start("s", "m")
        await svc.finalize()  # _redis is None — must not raise
        assert tid not in svc._buffers
    finally:
        trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

async def test_get_returns_stored_trace(connected_svc):
    tid = "get-trace-001"
    token = trace_id_var.set(tid)
    try:
        connected_svc.start("sess-g", "query")
        connected_svc.set_response("text", 20, 10)
        await connected_svc.finalize()

        result = await connected_svc.get(tid)
        assert result is not None
        assert result["trace_id"] == tid
        assert result["final_response"] == "text"
    finally:
        trace_id_var.reset(token)


async def test_get_returns_none_for_unknown_trace(connected_svc):
    result = await connected_svc.get("unknown-trace-id")
    assert result is None


async def test_get_returns_none_when_redis_not_connected(svc):
    result = await svc.get("any-id")
    assert result is None
