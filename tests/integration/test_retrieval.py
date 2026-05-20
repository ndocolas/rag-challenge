"""End-to-end retrieval tests against the live (already-indexed) Qdrant.

Pré-condição: corpus já indexado (Task 05). Skipa se Qdrant indisponível,
coleção vazia ou ``GOOGLE_API_KEY`` ausente/placeholder.
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest

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
    """Return the real Gemini key from env/.env, ignoring conftest placeholder."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key in {"", "test-key", "fake", "dummy"}:
        from dotenv import dotenv_values

        key = dotenv_values().get("GOOGLE_API_KEY", "") or ""
    if not key or key in {"test-key", "fake", "dummy"}:
        return None
    return key


def _has_real_gemini_key() -> bool:
    return _real_gemini_key() is not None


def _collection_populated() -> bool:
    if not _qdrant_up():
        return False
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(url=settings.QDRANT_URL)
        info = c.get_collection(settings.QDRANT_COLLECTION)
        return (info.points_count or 0) > 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable"),
    pytest.mark.skipif(
        not _has_real_gemini_key(), reason="real GOOGLE_API_KEY required"
    ),
    pytest.mark.skipif(
        not _collection_populated(),
        reason="collection bulas_panvel empty — run scripts/ingest_bulas.py first",
    ),
]


@pytest.fixture
def rag() -> RAGService:
    # The conftest.py sets GOOGLE_API_KEY=test-key via ``setdefault`` for unit
    # tests; for integration we must override with the real key picked up from
    # ``.env`` so Gemini calls actually succeed.
    real = _real_gemini_key()
    if real:
        os.environ["GOOGLE_API_KEY"] = real
        from panvel_assistant.utils.settings import get_settings

        get_settings.cache_clear()
    return RAGService()


@pytest.mark.asyncio
async def test_retrieve_with_med_name_filter(rag: RAGService):
    # parse_filename joins all post-id segments as the canonical med_name.
    chunks = await rag.retrieve(
        query="contraindicações", k=4, med_name="Ritalina Metilfenidato"
    )
    assert chunks, "Ritalina deve existir no corpus indexado"
    assert all(c.metadata.med_name == "Ritalina Metilfenidato" for c in chunks)


@pytest.mark.asyncio
async def test_retrieve_patient_facing_only_filters_IT(rag: RAGService):
    chunks = await rag.retrieve(
        query="interações", k=8, patient_facing_only=True
    )
    assert chunks
    for c in chunks:
        assert not c.metadata.section_canonical.startswith("IT_"), (
            f"chunk {c.chunk_id} é IT_* mas patient_facing_only=True"
        )


@pytest.mark.asyncio
async def test_retrieve_section_hint_filters_section(rag: RAGService):
    chunks = await rag.retrieve(
        query="qualquer texto",
        k=3,
        section_hint="IAP_6_POSOLOGIA",
        patient_facing_only=False,
    )
    assert chunks
    assert all(
        c.metadata.section_canonical == "IAP_6_POSOLOGIA" for c in chunks
    )


@pytest.mark.asyncio
async def test_retrieve_unknown_med_returns_empty(rag: RAGService):
    chunks = await rag.retrieve(
        query="posologia", k=4, med_name="MedicamentoQueNaoExiste123"
    )
    assert chunks == []


@pytest.mark.asyncio
async def test_list_medicamentos_returns_canonical_set(rag: RAGService):
    meds = await rag.list_medicamentos()
    assert isinstance(meds, list)
    assert len(meds) >= 10
    assert "Ritalina Metilfenidato" in meds
    # cached — second call returns identical object (no extra Qdrant round-trip)
    assert (await rag.list_medicamentos()) is meds


@pytest.mark.asyncio
async def test_format_tool_payload_shape(rag: RAGService):
    chunks = await rag.retrieve(query="indicação", k=2)
    payload = RAGService.format_tool_payload(chunks)
    assert payload["total"] == len(payload["matches"])
    if payload["matches"]:
        m = payload["matches"][0]
        for field in (
            "chunk_id",
            "med_name",
            "section_canonical",
            "section_label",
            "is_full_section",
            "text",
        ):
            assert field in m
