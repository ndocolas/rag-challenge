"""Offline RAG ingestion service.

Reads bula PDFs, sections them via Anvisa regex, applies the hybrid chunking
policy (whole section when it fits ``SECTION_WHOLE_THRESHOLD``, otherwise
sub-split with header prepended), embeds dense vectors with Gemini and sparse
BM25 vectors with fastembed, and upserts everything into Qdrant.

The service is meant to be invoked from ``scripts/ingest_bulas.py``; the chat
runtime never touches it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from bulas_assistant.assistant.sectionizer import (
    Section,
    extract_variant_names,
    sectionize,
)
from bulas_assistant.models.bula_models import BulaChunk, BulaMetadata
from bulas_assistant.utils.logger import get_logger
from bulas_assistant.utils.pdf import extract_text, file_hash
from bulas_assistant.utils.settings import settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "IngestionService", "component.version": "v1"}

# Tuned from scripts/profile_sections.py: p75 of section sizes is ~1800 chars,
# p90 ~6800. 3500 keeps almost all IAP_1/2/3/5/6/7 sections intact while
# splitting only the truly long IAP_4 (precautions) and IAP_8 (reactions).
SECTION_WHOLE_THRESHOLD = 3500
SUB_CHUNK_SIZE = 1600
SUB_CHUNK_OVERLAP = 200
MIN_SECTION_CHARS = 150  # drop noisy snippets like "VIA ORAL"

EMBED_DIM = 3072  # gemini-embedding-001 default output dimensionality
EMBED_CONCURRENCY = 5  # Gemini free tier: ~1500 RPM
UPSERT_BATCH = 32

# Deterministic UUID namespace so chunk_id -> point id is stable across runs.
_NAMESPACE = uuid.UUID("6e6c4f12-3a8f-4e0f-9a2c-1a9bc1a3a000")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SUB_CHUNK_SIZE,
    chunk_overlap=SUB_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def parse_filename(stem: str) -> tuple[str, str, str | None]:
    """Decode a filename stem into ``(bula_id, med_name, anvisa_code)``.

    Convention: ``<id>_<med>_<variant>.pdf``. The first segment is treated as
    the bula id; the remainder becomes the human-readable name. ``anvisa_code``
    is currently the same as ``bula_id`` (numeric prefix) and exposed
    separately so future records with explicit ANVISA registration codes can
    diverge without touching callers.
    """
    parts = stem.split("_", 1)
    bula_id = parts[0]
    med_name = parts[1].replace("_", " ").title() if len(parts) > 1 else stem
    anvisa_code = bula_id if bula_id.isdigit() else None
    return bula_id, med_name, anvisa_code


def section_to_chunks(
    section: Section,
    bula_id: str,
    med_name: str,
    med_variant: str | None,
    anvisa_code: str | None,
) -> list[BulaChunk]:
    """Apply the hybrid chunking policy to a single :class:`Section`.

    Short sections (``<= SECTION_WHOLE_THRESHOLD``) become a single chunk that
    preserves the full section verbatim. Long sections are sub-split with
    overlap and each sub-chunk gets the raw section header prepended so the
    embedding/BM25 vectors still carry the section identity. Sections shorter
    than ``MIN_SECTION_CHARS`` are dropped as extraction noise.
    """
    content = section.content.strip()
    if len(content) < MIN_SECTION_CHARS:
        return []

    base_meta: dict[str, Any] = {
        "bula_id": bula_id,
        "med_name": med_name,
        "anvisa_code": anvisa_code,
        "med_variant": med_variant,
        "section_canonical": section.canonical,
        "section_raw_header": section.raw_header or None,
        "source_page": None,
        "patient_facing": section.patient_facing,
        "section_char_len": len(content),
    }

    if len(content) <= SECTION_WHOLE_THRESHOLD:
        chunk_id = (
            f"{bula_id}__{section.canonical}__{section.occurrence}__full"
        )
        return [
            BulaChunk(
                chunk_id=chunk_id,
                text=content,
                metadata=BulaMetadata(
                    **base_meta,
                    chunk_idx=0,
                    is_full_section=True,
                ),
            )
        ]

    pieces = _splitter.split_text(content)
    raw_header_stripped = section.raw_header.strip() if section.raw_header else ""
    header_prefix = f"{raw_header_stripped}\n\n" if raw_header_stripped else ""
    chunks: list[BulaChunk] = []
    for idx, piece in enumerate(pieces):
        # Always prepend the header so each sub-chunk carries section identity
        # in both the dense embedding and the BM25 sparse representation.
        text = piece if piece.startswith(raw_header_stripped) else f"{header_prefix}{piece}"
        chunk_id = (
            f"{bula_id}__{section.canonical}__{section.occurrence}__part{idx}"
        )
        chunks.append(
            BulaChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=BulaMetadata(
                    **base_meta,
                    chunk_idx=idx,
                    is_full_section=False,
                ),
            )
        )
    return chunks


def chunks_for_pdf(pdf_path: Path) -> list[BulaChunk]:
    """Pure helper: extract + sectionize + hybrid chunk a single PDF.

    Kept free of embedding/Qdrant side-effects so unit tests and the offline
    profiler can call it directly.
    """
    bula_id, med_name, anvisa_code = parse_filename(pdf_path.stem)
    text = extract_text(pdf_path)
    sections = sectionize(text)
    variants = extract_variant_names(text, sections)
    all_chunks: list[BulaChunk] = []
    for s in sections:
        variant = variants.get(s.occurrence)
        all_chunks.extend(
            section_to_chunks(s, bula_id, med_name, variant, anvisa_code)
        )
    return all_chunks


class IngestionService:
    """Wires extraction → chunking → embedding → Qdrant upsert."""

    def __init__(self) -> None:
        self._embedder = GoogleGenerativeAIEmbeddings(  # type: ignore[call-arg]
            model=settings.GEMINI_EMBED_MODEL,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            task_type="RETRIEVAL_DOCUMENT",
        )
        self._sparse = SparseTextEmbedding("Qdrant/bm25")
        self._qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
        self._sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Stable UUIDv5 used as the Qdrant point id."""
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    async def ensure_collection(self, recreate: bool = False) -> None:
        collections = await self._qdrant.get_collections()
        names = {c.name for c in collections.collections}
        if settings.QDRANT_COLLECTION in names:
            if not recreate:
                return
            await self._qdrant.delete_collection(settings.QDRANT_COLLECTION)
        await self._qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )
        logger.info(
            "collection criada",
            extra={**_logger_extra, "step": "ingest", "collection": settings.QDRANT_COLLECTION},
        )

    async def _embed_dense(self, text: str) -> list[float]:
        async with self._sem:
            return await asyncio.to_thread(self._embedder.embed_query, text)

    def _embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        emb = next(iter(self._sparse.embed([text])))
        return emb.indices.tolist(), emb.values.tolist()

    async def _build_point(self, chunk: BulaChunk) -> PointStruct:
        dense = await self._embed_dense(chunk.text)
        sparse_idx, sparse_val = self._embed_sparse(chunk.text)
        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            **chunk.metadata.model_dump(),
        }
        return PointStruct(
            id=self.point_id(chunk.chunk_id),
            vector={
                "dense": dense,
                "bm25": SparseVector(indices=sparse_idx, values=sparse_val),
            },
            payload=payload,
        )

    async def _upsert_chunks(self, chunks: list[BulaChunk]) -> None:
        if not chunks:
            return
        points = await asyncio.gather(
            *(self._build_point(c) for c in chunks)
        )
        for start in range(0, len(points), UPSERT_BATCH):
            batch = points[start : start + UPSERT_BATCH]
            await self._qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=batch,
            )

    async def ingest_corpus(
        self, corpus_dir: Path | None = None, rebuild: bool = False
    ) -> dict[str, Any]:
        corpus_dir = corpus_dir or settings.BULAS_DIR
        pdfs = sorted(corpus_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"no PDFs found in {corpus_dir}")

        await self.ensure_collection(recreate=rebuild)

        manifest: dict[str, Any] = {
            "collection": settings.QDRANT_COLLECTION,
            "embed_model": settings.GEMINI_EMBED_MODEL,
            "threshold": SECTION_WHOLE_THRESHOLD,
            "bulas": {},
            "total_chunks": 0,
            "total_full_sections": 0,
            "total_sub_chunks": 0,
        }

        for pdf in pdfs:
            started = time.perf_counter()
            chunks = chunks_for_pdf(pdf)
            await self._upsert_chunks(chunks)
            n_full = sum(1 for c in chunks if c.metadata.is_full_section)
            n_split = len(chunks) - n_full
            manifest["bulas"][pdf.name] = {
                "hash": file_hash(pdf),
                "chunks": len(chunks),
                "full_sections": n_full,
                "sub_chunks": n_split,
            }
            manifest["total_chunks"] += len(chunks)
            manifest["total_full_sections"] += n_full
            manifest["total_sub_chunks"] += n_split
            logger.info(
                "bula indexada",
                extra={
                    **_logger_extra,
                    "step": "ingest",
                    "file": pdf.name,
                    "chunks": len(chunks),
                    "full_sections": n_full,
                    "sub_chunks": n_split,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )

        manifest_path = settings.CACHE_DIR / "ingest_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        return manifest


ingestion_service = IngestionService()
