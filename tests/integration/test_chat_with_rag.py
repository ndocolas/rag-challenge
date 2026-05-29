"""End-to-end-ish test for the agentic RAG path in ``AssistantService``.

Stubs the LLM client and the ``RAGService`` so the test runs offline, but
exercises the real ``buscar_bulas`` tool, the real ``_execute_tool`` async
dispatch, and the real ``sources`` SSE emission inside ``stream_with_tools``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage

from bulas_assistant.assistant.agent_tools import build_tools
from bulas_assistant.assistant.assistant_service import AssistantService
from bulas_assistant.models.bula_models import BulaChunk, BulaMetadata
from bulas_assistant.services.filiais_service import filiais_service


def _text_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


def _tool_call_chunk(name: str, args_json: str, call_id: str = "c1") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": name, "args": args_json, "id": call_id, "index": 0}
        ],
    )


class _StubLLM:
    """Replays canned rounds of AIMessageChunks instead of calling Gemini."""

    def __init__(self, rounds: list[list[AIMessageChunk]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[list[BaseMessage]] = []

    async def astream(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[AIMessageChunk]:
        self.calls.append(list(messages))
        chunks = self.rounds.pop(0) if self.rounds else []
        for chunk in chunks:
            yield chunk


class _FakeRAGService:
    """Returns a fixed list of chunks regardless of query — deterministic."""

    def __init__(
        self,
        chunks: list[BulaChunk],
        meds: list[str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.meds = meds or ["Ritalina Metilfenidato", "Pantoprazol"]
        self.last_call: dict[str, Any] | None = None

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        med_name: str | None = None,
        med_variant: str | None = None,
        section_hint: str | None = None,
        patient_facing_only: bool = True,
    ) -> list[BulaChunk]:
        self.last_call = {
            "query": query,
            "k": k,
            "med_name": med_name,
            "section_hint": section_hint,
            "patient_facing_only": patient_facing_only,
        }
        return self.chunks[:k]

    async def list_medicamentos(self) -> list[str]:
        return list(self.meds)


def _make_chunk(chunk_id: str, *, section: str = "IAP_3_CONTRAINDICACOES") -> BulaChunk:
    return BulaChunk(
        chunk_id=chunk_id,
        text=f"Texto do chunk {chunk_id} sobre contraindicações da Ritalina.",
        metadata=BulaMetadata(
            bula_id="123",
            med_name="Ritalina",
            anvisa_code="123",
            med_variant=None,
            section_canonical=section,
            section_raw_header=None,
            chunk_idx=0,
            source_page=None,
            patient_facing=True,
            is_full_section=True,
            section_char_len=80,
        ),
        score=0.9,
    )


def _make_service(llm: _StubLLM, rag: _FakeRAGService) -> AssistantService:
    """Build an AssistantService bypassing __init__ (no API key required)."""
    svc = AssistantService.__new__(AssistantService)
    tools = build_tools(filiais_service, rag)  # type: ignore[arg-type]
    svc._llm = llm
    svc._tools_by_name = {t.name: t for t in tools}
    svc._history = None
    return svc


@pytest.mark.asyncio
async def test_buscar_bulas_emits_sources_event():
    """Tool round 1 = buscar_bulas call; round 2 = final answer text."""
    chunks = [_make_chunk("c1"), _make_chunk("c2", section="IAP_6_POSOLOGIA")]
    rag = _FakeRAGService(chunks)
    llm = _StubLLM(
        rounds=[
            [
                _tool_call_chunk(
                    "buscar_bulas",
                    '{"query": "contraindicacoes", "med_name": "Ritalina"}',
                )
            ],
            [_text_chunk("Resposta com [Ritalina — Quando não devo usar].")],
        ]
    )
    svc = _make_service(llm, rag)

    events = [evt async for evt in svc.stream_with_tools([])]
    types = [e["type"] for e in events]

    assert types == ["tool_call", "tool_result", "sources", "token", "done"]

    sources = next(e for e in events if e["type"] == "sources")
    cits = sources["citations"]
    assert len(cits) == 2
    assert all(c["med_name"] == "Ritalina" for c in cits)
    assert cits[0]["section_label"] == "Quando não devo usar"
    assert cits[1]["section_label"] == "Como devo usar"

    assert rag.last_call is not None
    assert rag.last_call["med_name"] == "Ritalina"


@pytest.mark.asyncio
async def test_buscar_bulas_unknown_med_returns_error_and_no_sources():
    """med_name set + 0 chunks -> medicamento_nao_encontrado, no sources."""
    rag = _FakeRAGService(chunks=[])
    llm = _StubLLM(
        rounds=[
            [
                _tool_call_chunk(
                    "buscar_bulas",
                    '{"query": "x", "med_name": "NaoExiste"}',
                )
            ],
            [_text_chunk("Não temos a bula desse medicamento.")],
        ]
    )
    svc = _make_service(llm, rag)

    events = [evt async for evt in svc.stream_with_tools([])]
    types = [e["type"] for e in events]
    assert "sources" not in types
    assert types == ["tool_call", "tool_result", "token", "done"]

    # The ToolMessage carries the structured error payload.
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert "medicamento_nao_encontrado" in (tool_result["preview"] or "")


@pytest.mark.asyncio
async def test_buscar_bulas_no_filter_zero_chunks_returns_nenhum_resultado():
    """No med_name + 0 chunks -> nenhum_resultado, no sources."""
    rag = _FakeRAGService(chunks=[])
    llm = _StubLLM(
        rounds=[
            [_tool_call_chunk("buscar_bulas", '{"query": "dipirona posologia"}')],
            [_text_chunk("Esse medicamento não está nas bulas indexadas.")],
        ]
    )
    svc = _make_service(llm, rag)

    events = [evt async for evt in svc.stream_with_tools([])]
    types = [e["type"] for e in events]
    assert "sources" not in types

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert "nenhum_resultado" in (tool_result["preview"] or "")


@pytest.mark.asyncio
async def test_small_talk_does_not_invoke_tool_nor_emit_sources():
    rag = _FakeRAGService(chunks=[_make_chunk("c1")])
    llm = _StubLLM(rounds=[[_text_chunk("Oi! Tudo ótimo, como posso ajudar?")]])
    svc = _make_service(llm, rag)

    events = [evt async for evt in svc.stream_with_tools([])]
    types = [e["type"] for e in events]
    assert types == ["token", "done"]
    assert rag.last_call is None
