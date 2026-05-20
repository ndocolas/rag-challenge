"""End-to-end ingestion test against a live Qdrant.

Skipped automatically when Qdrant is not reachable. Also skipped when
``GOOGLE_API_KEY`` is missing or a placeholder, because dense embeddings
require a real Gemini key.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from panvel_assistant.services.ingestion_service import (
    IngestionService,
    chunks_for_pdf,
)
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


def _has_real_gemini_key() -> bool:
    key = os.environ.get("GOOGLE_API_KEY", "")
    return bool(key) and key not in {"test-key", "fake", "dummy"}


@pytest.fixture
def sample_pdf():
    pdfs = sorted(settings.BULAS_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no PDFs in corpus")
    return pdfs[0]


def test_chunks_for_pdf_runs_offline(sample_pdf):
    """Pure helper must work without Qdrant or Gemini."""
    chunks = chunks_for_pdf(sample_pdf)
    assert chunks, "expected at least one chunk"
    assert all(c.text and c.chunk_id for c in chunks)
    assert all(c.metadata.section_char_len > 0 for c in chunks)


@pytest.mark.integration
@pytest.mark.skipif(not _qdrant_up(), reason="Qdrant not reachable")
@pytest.mark.skipif(
    not _has_real_gemini_key(), reason="real GOOGLE_API_KEY required"
)
@pytest.mark.asyncio
async def test_ingest_single_pdf_is_idempotent(sample_pdf, tmp_path):
    service = IngestionService()
    # Use an isolated collection so the test never disturbs the real one.
    original = settings.QDRANT_COLLECTION
    settings.QDRANT_COLLECTION = f"test_ingest_{os.getpid()}"
    try:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / sample_pdf.name).symlink_to(sample_pdf.resolve())

        m1 = await service.ingest_corpus(corpus_dir=corpus, rebuild=True)
        assert m1["total_chunks"] > 0
        info = m1["bulas"][sample_pdf.name]
        assert info["full_sections"] + info["sub_chunks"] == info["chunks"]

        # second run without rebuild: same chunk_ids -> same uuid5 -> upserts
        # over existing points, count stays identical.
        m2 = await service.ingest_corpus(corpus_dir=corpus, rebuild=False)
        assert m2["total_chunks"] == m1["total_chunks"]

        info = await service._qdrant.get_collection(settings.QDRANT_COLLECTION)
        assert info.points_count == m1["total_chunks"]

        scrolled, _ = await service._qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1,
            with_payload=True,
        )
        payload = scrolled[0].payload
        for required in (
            "chunk_id",
            "text",
            "bula_id",
            "section_canonical",
            "is_full_section",
            "section_char_len",
            "patient_facing",
        ):
            assert required in payload
    finally:
        try:
            await service._qdrant.delete_collection(settings.QDRANT_COLLECTION)
        except Exception:  # noqa: BLE001
            pass
        settings.QDRANT_COLLECTION = original
