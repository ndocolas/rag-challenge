"""Unit tests for ``AssistantService`` — content coercion, tool loop, and handle_turn."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import tool as tool_decorator

from panvel_assistant.assistant import assistant_service as assistant_module
from panvel_assistant.assistant.assistant_service import (
    MAX_TOOL_ITERATIONS,
    AssistantService,
    _coerce_content_to_text,
    get_assistant_service,
)
from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services import chat_history_service

# ---------------------------------------------------------------------------
# _coerce_content_to_text — pure function over LangChain chunk content shapes.
# ---------------------------------------------------------------------------


def test_coerce_text_passthrough_when_str():
    assert _coerce_content_to_text("hello") == "hello"


def test_coerce_text_joins_str_segments():
    assert _coerce_content_to_text(["a", "b", "c"]) == "abc"


def test_coerce_text_extracts_dict_text_fields():
    content = [{"text": "hi "}, {"text": "there"}]
    assert _coerce_content_to_text(content) == "hi there"


def test_coerce_text_drops_non_text_dicts_and_unknown_types():
    content = [
        "prefix-",
        {"text": "middle"},
        {"function_call": {"name": "foo"}},
        42,
    ]
    assert _coerce_content_to_text(content) == "prefix-middle"


def test_coerce_text_returns_empty_for_none():
    assert _coerce_content_to_text(None) == ""


def test_coerce_text_returns_empty_for_random_object():
    class Whatever:
        pass

    assert _coerce_content_to_text(Whatever()) == ""


# ---------------------------------------------------------------------------
# stream_with_tools — tool loop event sequence.
# ---------------------------------------------------------------------------


def _text_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


def _tool_call_chunk(
    name: str, args_json: str, call_id: str = "call_1"
) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": name, "args": args_json, "id": call_id, "index": 0}
        ],
    )


class _StubClient:
    """Minimal stand-in for the bound ``ChatGoogleGenerativeAI`` client."""

    def __init__(self, rounds: list[list[AIMessageChunk]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[list[BaseMessage]] = []

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        self.calls.append(list(messages))
        chunks = self.rounds.pop(0) if self.rounds else []
        for chunk in chunks:
            yield chunk


def _make_service_with(client, *, tools_by_name: dict | None = None) -> AssistantService:
    """Build an AssistantService whose underlying client is the supplied stub."""
    svc = AssistantService.__new__(AssistantService)
    svc._llm = client
    svc._tools_by_name = tools_by_name or {}
    svc._history = None  # not used by stream_with_tools
    return svc


async def _drain(svc: AssistantService) -> list[dict]:
    return [evt async for evt in svc.stream_with_tools([])]


async def test_single_tool_call_then_final_text_emits_full_event_sequence():
    @tool_decorator("echo_args")
    def echo_args(value: str) -> str:
        """Echo a string back; used for assertions."""
        return f"echoed:{value}"

    client = _StubClient(
        rounds=[
            [_tool_call_chunk("echo_args", '{"value": "hi"}', call_id="c1")],
            [_text_chunk("Resposta "), _text_chunk("final.")],
        ]
    )
    svc = _make_service_with(client, tools_by_name={"echo_args": echo_args})

    events = await _drain(svc)
    types = [e["type"] for e in events]
    assert types == ["tool_call", "tool_result", "token", "token", "done"]

    tc = events[0]
    assert tc["name"] == "echo_args"
    assert tc["args"] == {"value": "hi"}

    tr = events[1]
    assert tr["name"] == "echo_args"
    assert tr["error"] is None
    assert tr["preview"] == "echoed:hi"
    assert isinstance(tr["latency_ms"], float)

    assert events[2]["text"] == "Resposta "
    assert events[3]["text"] == "final."

    assert len(client.calls) == 2
    assert len(client.calls[1]) == len(client.calls[0]) + 2


async def test_tool_that_raises_yields_error_in_tool_result_event():
    @tool_decorator("explode")
    def explode() -> str:
        """Raise unconditionally; used to test failure handling."""
        raise RuntimeError("boom")

    client = _StubClient(
        rounds=[
            [_tool_call_chunk("explode", "{}", call_id="c1")],
            [_text_chunk("desculpe")],
        ]
    )
    svc = _make_service_with(client, tools_by_name={"explode": explode})

    events = await _drain(svc)
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["error"] == "boom"
    assert tool_result["preview"] is None
    assert events[-1]["type"] == "done"


async def test_unknown_tool_emits_error_trace_without_raising():
    client = _StubClient(
        rounds=[
            [_tool_call_chunk("missing_tool", "{}", call_id="c1")],
            [_text_chunk("ok")],
        ]
    )
    svc = _make_service_with(client, tools_by_name={})

    events = await _drain(svc)
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["name"] == "missing_tool"
    assert tool_result["error"] == "unknown_tool"


async def test_max_iterations_exceeded_emits_terminal_error():
    @tool_decorator("noop")
    def noop() -> str:
        """Return an empty JSON object."""
        return "{}"

    call_count = {"n": 0}

    class _LoopingClient:
        def __init__(self) -> None:
            self.calls: list[list[BaseMessage]] = []

        async def astream(self, messages):
            self.calls.append(list(messages))
            call_count["n"] += 1
            yield _tool_call_chunk("noop", "{}", call_id=f"c{call_count['n']}")

    svc = _make_service_with(_LoopingClient(), tools_by_name={"noop": noop})

    events = await _drain(svc)

    assert call_count["n"] == MAX_TOOL_ITERATIONS
    assert events[-1] == {
        "type": "error",
        "message": "max_tool_iterations_exceeded",
    }
    assert sum(1 for e in events if e["type"] == "tool_call") == MAX_TOOL_ITERATIONS
    assert sum(1 for e in events if e["type"] == "tool_result") == MAX_TOOL_ITERATIONS


async def test_no_tool_calls_yields_only_tokens_then_done():
    client = _StubClient(rounds=[[_text_chunk("Olá"), _text_chunk(" mundo")]])
    svc = _make_service_with(client)

    events = await _drain(svc)
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[0]["text"] == "Olá"
    assert events[1]["text"] == " mundo"


async def test_empty_stream_yields_done_without_error():
    client = _StubClient(rounds=[[]])
    svc = _make_service_with(client)

    events = await _drain(svc)
    assert events == [{"type": "done"}]


# ---------------------------------------------------------------------------
# CancelledError propagation in stream_with_tools.
# ---------------------------------------------------------------------------


class _CancellingClient:
    async def astream(self, _msgs) -> AsyncIterator[AIMessageChunk]:
        yield AIMessageChunk(content="partial")
        raise asyncio.CancelledError


async def test_stream_with_tools_propagates_cancelled_error():
    svc = _make_service_with(_CancellingClient())

    collected: list[dict] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in svc.stream_with_tools([]):
            collected.append(event)

    assert collected == [{"type": "token", "text": "partial"}]


# ---------------------------------------------------------------------------
# get_assistant_service factory caching + module-level proxy.
# ---------------------------------------------------------------------------


def test_get_assistant_service_is_cached():
    a = get_assistant_service()
    b = get_assistant_service()
    assert a is b


def test_module_proxy_returns_singleton():
    assert assistant_module.assistant_service is get_assistant_service()


def test_constructor_accepts_injected_llm_without_settings():
    """``AssistantService`` must build cleanly when a stub LLM is supplied.

    This exercises the helper-style DI: no Settings/API key consulted because
    the LLM is provided explicitly. ``bind_tools`` is the only contract we
    need from the injected client.
    """

    class _BindOnly:
        def bind_tools(self, _tools):
            return _StubClient(rounds=[])

    class _NoopRag:
        async def retrieve(self, **_kwargs):
            return []

        async def list_medicamentos(self):
            return []

    svc = AssistantService(
        llm=_BindOnly(),  # type: ignore[arg-type]
        rag_service=_NoopRag(),  # type: ignore[arg-type]
    )
    assert set(svc._tools_by_name) == {
        "buscar_filiais",
        "detalhes_filial",
        "listar_cidades_atendidas",
        "buscar_bulas",
        "listar_medicamentos_disponiveis",
    }


# ---------------------------------------------------------------------------
# handle_turn — SSE event mapping + history persistence.
# ---------------------------------------------------------------------------


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
    event_line, data_line = frame.strip().split("\n", 1)
    event_type = event_line.removeprefix("event: ").strip()
    data_json = data_line.removeprefix("data: ").strip()
    return event_type, json.loads(data_json)


def _parse_sse_text_event(frame: str) -> tuple[str, str]:
    lines = frame.strip().split("\n")
    event_type = lines[0].removeprefix("event: ").strip()
    data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
    return event_type, data


async def _async_iter(items: list[dict]) -> AsyncIterator[dict]:
    for item in items:
        yield item


@pytest.fixture
def fake_history(monkeypatch):
    """Patch the history store to return a deterministic fake."""
    fake = _FakeHistory()
    monkeypatch.setattr(
        chat_history_service.history_store,
        "get_session_history",
        lambda session_id: fake,
    )
    return fake


async def _drain_pending() -> None:
    await chat_history_service.history_store.drain_pending(timeout=2.0)


async def test_handle_turn_emits_tokens_then_done_and_appends_history(
    monkeypatch, fake_history
):
    captured_messages: list[list[BaseMessage]] = []

    async def fake_stream(messages):
        captured_messages.append(list(messages))
        async for event in _async_iter(
            [
                {"type": "token", "text": "Olá"},
                {"type": "token", "text": " mundo"},
                {"type": "done"},
            ]
        ):
            yield event

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", fake_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert len(frames) == 3
    assert _parse_sse_text_event(frames[0]) == ("token", "Olá")
    assert _parse_sse_text_event(frames[1]) == ("token", " mundo")
    assert _parse_sse_event(frames[2]) == ("done", {"session_id": "s"})

    await _drain_pending()
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
        async for event in _async_iter(
            [{"type": "token", "text": "ok"}, {"type": "done"}]
        ):
            yield event

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", fake_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
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
            yield {}
        raise RuntimeError("upstream blew up")

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", boom_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert len(frames) == 1
    event_type, payload = _parse_sse_event(frames[0])
    assert event_type == "error"
    assert payload["code"] == "stream_failed"
    assert payload["message"] == "internal error"
    assert "trace_id" in payload
    assert fake_history.appended == []


async def test_handle_turn_preserves_newlines_inside_token_deltas(
    monkeypatch, fake_history
):
    async def fake_stream(messages):
        async for event in _async_iter(
            [
                {"type": "token", "text": "linha1\nlinha2"},
                {"type": "token", "text": "\n\nfim"},
                {"type": "done"},
            ]
        ):
            yield event

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", fake_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
            ChatRequest(session_id="s", message="oi")
        )
    ]

    assert _parse_sse_text_event(frames[0]) == ("token", "linha1\nlinha2")
    assert _parse_sse_text_event(frames[1]) == ("token", "\n\nfim")
    assert _parse_sse_event(frames[2]) == ("done", {"session_id": "s"})


async def test_handle_turn_forwards_tool_call_and_tool_result_events(
    monkeypatch, fake_history
):
    """Tool events must be mapped to SSE frames in order, around any tokens."""

    async def fake_stream(messages):
        async for event in _async_iter(
            [
                {
                    "type": "tool_call",
                    "name": "buscar_filiais",
                    "args": {"cidade": "CURITIBA"},
                },
                {
                    "type": "tool_result",
                    "name": "buscar_filiais",
                    "preview": "{\"total\": 3}",
                    "error": None,
                    "latency_ms": 12.5,
                },
                {"type": "token", "text": "Achei 3 lojas."},
                {"type": "done"},
            ]
        ):
            yield event

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", fake_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
            ChatRequest(session_id="s", message="lojas em Curitiba?")
        )
    ]

    assert len(frames) == 4
    assert _parse_sse_event(frames[0]) == (
        "tool_call",
        {"name": "buscar_filiais", "args": {"cidade": "CURITIBA"}},
    )
    assert _parse_sse_event(frames[1]) == (
        "tool_result",
        {
            "name": "buscar_filiais",
            "preview": "{\"total\": 3}",
            "error": None,
            "latency_ms": 12.5,
        },
    )
    assert _parse_sse_text_event(frames[2]) == ("token", "Achei 3 lojas.")
    assert _parse_sse_event(frames[3]) == ("done", {"session_id": "s"})

    await _drain_pending()
    assert fake_history.appended[0][1].content == "Achei 3 lojas."


async def test_handle_turn_terminates_on_internal_error_event(
    monkeypatch, fake_history
):
    async def fake_stream(messages):
        async for event in _async_iter(
            [
                {"type": "token", "text": "tentando..."},
                {"type": "error", "message": "max_tool_iterations_exceeded"},
            ]
        ):
            yield event

    monkeypatch.setattr(
        assistant_module.assistant_service, "stream_with_tools", fake_stream
    )

    frames = [
        frame
        async for frame in assistant_module.assistant_service.handle_turn(
            ChatRequest(session_id="s", message="x")
        )
    ]

    assert len(frames) == 2
    assert _parse_sse_text_event(frames[0]) == ("token", "tentando...")
    err_type, err_payload = _parse_sse_event(frames[1])
    assert err_type == "error"
    assert err_payload["message"] == "max_tool_iterations_exceeded"
    assert "trace_id" in err_payload
    assert fake_history.appended == []
