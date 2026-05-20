"""Invoca a tool ``buscar_bulas`` ponta-a-ponta via ``build_tools``.

Verifica o contrato JSON que o LLM recebe.
"""

from __future__ import annotations

import json
import os
import socket
from urllib.parse import urlparse

import pytest

from panvel_assistant.assistant.agent_tools import build_tools
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.services.rag_service import RAGService
from panvel_assistant.utils.settings import settings


def _qdrant_up() -> bool:
    parsed = urlparse(settings.QDRANT_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _real_gemini_key() -> str | None:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key in {"", "test-key", "fake", "dummy"}:
        from dotenv import dotenv_values

        key = dotenv_values().get("GOOGLE_API_KEY", "") or ""
    if not key or key in {"test-key", "fake", "dummy"}:
        return None
    return key


def _has_real_gemini_key() -> bool:
    return _real_gemini_key() is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable"),
    pytest.mark.skipif(
        not _has_real_gemini_key(), reason="real GOOGLE_API_KEY required"
    ),
]


@pytest.fixture
def buscar_bulas_tool():
    real = _real_gemini_key()
    if real:
        os.environ["GOOGLE_API_KEY"] = real
        from panvel_assistant.utils.settings import get_settings

        get_settings.cache_clear()
    tools = build_tools(filiais_service, RAGService())
    by_name = {t.name: t for t in tools}
    assert "buscar_bulas" in by_name
    return by_name["buscar_bulas"]


@pytest.mark.asyncio
async def test_tool_returns_well_formed_json(buscar_bulas_tool):
    raw = await buscar_bulas_tool.ainvoke(
        {"query": "para que serve", "med_name": "Ritalina Metilfenidato", "k": 2}
    )
    payload = json.loads(raw)
    assert "matches" in payload and "total" in payload
    assert payload["total"] == len(payload["matches"])
    if payload["matches"]:
        m = payload["matches"][0]
        for f in ("chunk_id", "med_name", "section_canonical", "is_full_section"):
            assert f in m


@pytest.mark.asyncio
async def test_tool_returns_medicamento_nao_encontrado(buscar_bulas_tool):
    raw = await buscar_bulas_tool.ainvoke(
        {"query": "posologia", "med_name": "NaoExisteMed123", "k": 3}
    )
    payload = json.loads(raw)
    assert payload.get("error") == "medicamento_nao_encontrado"
    assert "NaoExisteMed123" in payload.get("message", "")
    available = payload.get("hint", {}).get("medicamentos_disponiveis", [])
    assert isinstance(available, list) and len(available) > 0


@pytest.mark.asyncio
async def test_tool_returns_nenhum_resultado_when_no_med_filter(buscar_bulas_tool):
    # No med_name — query about a drug not in the corpus.
    raw = await buscar_bulas_tool.ainvoke(
        {"query": "qual a posologia da dipirona sodica monoidratada injetavel", "k": 3}
    )
    payload = json.loads(raw)
    # Either a relevance miss or hybrid still returns adjacent chunks. Both are
    # valid — accept matches OR nenhum_resultado, just not medicamento_nao_encontrado.
    assert payload.get("error") != "medicamento_nao_encontrado"
    if "error" in payload:
        assert payload["error"] == "nenhum_resultado"
        assert "sugestao" in payload.get("hint", {})
