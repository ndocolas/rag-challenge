"""Retrieval service for the agentic ``buscar_bulas`` tool.

Hybrid search over Qdrant: dense (Gemini ``RETRIEVAL_QUERY`` embeddings) +
sparse (fastembed BM25), fused with Reciprocal Rank Fusion server-side. Filters
(``med_name``, ``section_canonical``, ``patient_facing``) are pushed into each
``Prefetch`` so the RRF pool doesn't get polluted by off-target candidates.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from bulas_assistant.models.bula_models import BulaChunk, BulaMetadata
from bulas_assistant.models.chat_models import Citation
from bulas_assistant.services.trace_service import trace_service
from bulas_assistant.utils.logger import get_logger
from bulas_assistant.utils.settings import settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "RagService", "component.version": "v1"}
_BULA_METADATA_FIELDS = frozenset(BulaMetadata.model_fields)


SECTION_LABEL: dict[str, str] = {
    "IAP_1_INDICACOES": "Para que é indicado",
    "IAP_2_MECANISMO": "Como funciona",
    "IAP_3_CONTRAINDICACOES": "Quando não devo usar",
    "IAP_4_PRECAUCOES_ADVERTENCIAS": "O que devo saber antes de usar",
    "IAP_5_ARMAZENAMENTO": "Como guardar",
    "IAP_6_POSOLOGIA": "Como devo usar",
    "IAP_7_ESQUECIMENTO_DOSE": "Esquecimento de dose",
    "IAP_8_REACOES_ADVERSAS": "Reações adversas",
    "IAP_9_SUPERDOSE": "Superdose",
    "IT_CARACTERISTICAS_FARMACOLOGICAS": "Características farmacológicas",
    "IT_INTERACOES_MEDICAMENTOSAS": "Interações medicamentosas",
    "IT_REACOES_ADVERSAS_TECNICAS": "Reações adversas (técnico)",
    "IDENT_APRESENTACOES": "Apresentações",
    "IDENT_COMPOSICAO": "Composição",
    "IDENT_VIA_USO": "Via/Uso",
    "DIZERES_LEGAIS": "Dizeres legais",
    "UNCLASSIFIED": "—",
}


def _section_label(canonical: str) -> str:
    return SECTION_LABEL.get(canonical, canonical)


class RAGService:
    """Hybrid retrieval over the ``bulas`` Qdrant collection."""

    def __init__(self) -> None:
        self._embedder = GoogleGenerativeAIEmbeddings(  # type: ignore[call-arg]
            model=settings.GEMINI_EMBED_MODEL,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            task_type="RETRIEVAL_QUERY",
        )
        self._sparse = SparseTextEmbedding("Qdrant/bm25")
        self._qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
        # Canonical med_name list, materialized on first access via a Qdrant
        # scroll. The corpus only changes through the offline ingest pipeline,
        # so it's safe to memoize for the process lifetime.
        self._meds_cache: list[str] | None = None

    @staticmethod
    def _build_filter(
        med_name: str | None,
        section_hint: str | None,
        patient_facing_only: bool,
        med_variant: str | None = None,
    ) -> Filter | None:
        """Compose a Qdrant ``Filter`` from the optional search dimensions.

        Returns ``None`` when no filter is needed so the prefetch stays
        unfiltered (avoids constructing empty ``must`` lists).
        """
        must: list[FieldCondition] = []
        if med_name:
            must.append(
                FieldCondition(key="med_name", match=MatchValue(value=med_name))
            )
        if med_variant:
            must.append(
                FieldCondition(key="med_variant", match=MatchValue(value=med_variant))
            )
        if section_hint:
            must.append(
                FieldCondition(
                    key="section_canonical", match=MatchValue(value=section_hint)
                )
            )
        if patient_facing_only:
            must.append(
                FieldCondition(
                    key="patient_facing", match=MatchValue(value=True)
                )
            )
        return Filter(must=must) if must else None  # type: ignore[arg-type]

    async def _embed_dense(self, query: str) -> list[float]:
        return await self._embedder.aembed_query(query)

    async def _embed_sparse(self, query: str) -> tuple[list[int], list[float]]:
        # fastembed is sync; offload so we don't block the event loop.
        def _run() -> tuple[list[int], list[float]]:
            results = list(self._sparse.query_embed([query]))
            if not results:
                raise RuntimeError("sparse embedding returned empty result")
            emb = results[0]
            return emb.indices.tolist(), emb.values.tolist()

        return await asyncio.to_thread(_run)

    async def _query_and_dedup(
        self,
        dense: list[float],
        sparse_idx: list[int],
        sparse_val: list[float],
        qfilter: Filter | None,
        k: int,
    ) -> list[BulaChunk]:
        results = await self._qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=k * 4, filter=qfilter),
                Prefetch(
                    query=SparseVector(indices=sparse_idx, values=sparse_val),
                    using="bm25",
                    limit=k * 4,
                    filter=qfilter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k * 3,
            with_payload=True,
        )
        return self._dedup_points(results.points, k)

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        med_name: str | None = None,
        med_variant: str | None = None,
        section_hint: str | None = None,
        patient_facing_only: bool = True,
    ) -> list[BulaChunk]:
        """Hybrid search with RRF fusion + filtered prefetch + smart dedup.

        Filters live INSIDE each ``Prefetch`` so RRF only fuses candidates that
        already passed the structural constraints. Without this the fusion
        score is contaminated by candidates that would later be discarded.
        """
        started = time.perf_counter()
        qfilter = RAGService._build_filter(med_name, section_hint, patient_facing_only, med_variant)
        dense, (sparse_idx, sparse_val) = await asyncio.gather(
            self._embed_dense(query),
            self._embed_sparse(query),
        )

        chunks = await self._query_and_dedup(dense, sparse_idx, sparse_val, qfilter, k)

        if not chunks and section_hint:
            qfilter_broad = RAGService._build_filter(
                med_name, None, patient_facing_only, med_variant
            )
            chunks = await self._query_and_dedup(dense, sparse_idx, sparse_val, qfilter_broad, k)
            logger.info(
                "section_hint_fallback",
                extra={
                    **_logger_extra,
                    "step": "retrieval",
                    "section_hint": section_hint,
                    "med_name": med_name,
                    "returned": len(chunks),
                },
            )

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "retrieval concluido",
            extra={
                **_logger_extra,
                "step": "retrieval",
                "query_len": len(query),
                "k": k,
                "med_name": med_name,
                "med_variant": med_variant,
                "section_hint": section_hint,
                "patient_facing_only": patient_facing_only,
                "returned": len(chunks),
                "latency_ms": latency_ms,
            },
        )
        trace_service.add_step(
            "retrieval",
            latency_ms,
            k=k,
            returned=len(chunks),
            hint=section_hint,
        )
        return chunks

    async def list_medicamentos(self) -> list[str]:
        """Distinct medication entries currently indexed in Qdrant.

        Returns one entry per (med_name, med_variant) pair so multi-product
        PDFs (e.g. Ritalina + Ritalina LA) appear as separate items.
        Format: "Ritalina Metilfenidato" for the base product and
        "Ritalina Metilfenidato — RITALINA LA" for named variants.

        Memoized on the instance because the corpus only changes via the
        offline ingestion pipeline.
        """
        if self._meds_cache is not None:
            return self._meds_cache
        seen: set[tuple[str, str | None]] = set()
        offset: Any = None
        while True:
            points, offset = await self._qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=256,
                with_payload=["med_name", "med_variant"],
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                name = payload.get("med_name")
                if name:
                    seen.add((name, payload.get("med_variant")))
            if offset is None:
                break
        entries: list[str] = []
        for med_name, med_variant in seen:
            if med_variant:
                entries.append(f"{med_name} — {med_variant}")
            else:
                entries.append(med_name)
        self._meds_cache = sorted(entries)
        logger.info(
            "med_name cache populado",
            extra={**_logger_extra, "step": "retrieval", "total_meds": len(self._meds_cache)},
        )
        return self._meds_cache

    @staticmethod
    def _dedup_points(points: list[Any], k: int) -> list[BulaChunk]:
        """Apply chunk_id + (bula_id, section) dedup, capped at ``k``.

        Rules:
        - duplicate ``chunk_id`` → always drop;
        - ``is_full_section=True`` duplicates within the same
          ``(bula_id, section_canonical)`` → keep only the best-scoring one;
        - ``is_full_section=False`` (sub-chunks) → NOT deduped by section so
          two distinct slices of the same long section can co-occur.
        """
        seen_chunk_ids: set[str] = set()
        seen_full_sections: set[tuple[str, str, str | None]] = set()
        chunks: list[BulaChunk] = []
        for p in points:
            payload = p.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            if payload.get("is_full_section"):
                key = (
                    payload.get("bula_id", ""),
                    payload.get("section_canonical", ""),
                    payload.get("med_variant"),
                )
                if key in seen_full_sections:
                    continue
                seen_full_sections.add(key)
            md_kwargs = {f: payload[f] for f in _BULA_METADATA_FIELDS if f in payload}
            chunks.append(
                BulaChunk(
                    chunk_id=chunk_id,
                    text=payload.get("text", ""),
                    metadata=BulaMetadata(**md_kwargs),
                    score=getattr(p, "score", None),
                )
            )
            if len(chunks) >= k:
                break
        return chunks

    @staticmethod
    def format_tool_payload(chunks: list[BulaChunk]) -> dict[str, Any]:
        """JSON-serializable shape returned by the tool to the LLM."""
        items = []
        for c in chunks:
            md = c.metadata
            items.append(
                {
                    "chunk_id": c.chunk_id,
                    "bula_id": md.bula_id,
                    "med_name": md.med_name,
                    "med_variant": md.med_variant,
                    "section_canonical": md.section_canonical,
                    "section_label": _section_label(md.section_canonical),
                    "is_full_section": md.is_full_section,
                    "source_page": md.source_page,
                    "text": c.text,
                    "score": c.score,
                }
            )
        return {"matches": items, "total": len(items)}

    @staticmethod
    def citations_from_matches(matches: list[dict[str, Any]]) -> list[Citation]:
        """Build ``Citation`` objects from the tool's JSON payload.

        Used by the chat stream to emit the ``sources`` SSE event without
        re-querying Qdrant: the tool payload already carries everything
        needed for citation rendering.
        """
        out: list[Citation] = []
        for m in matches:
            text = m.get("text", "") or ""
            canonical = m.get("section_canonical", "UNCLASSIFIED")
            out.append(
                Citation(
                    bula_id=m.get("bula_id", ""),
                    med_name=m.get("med_name", ""),
                    med_variant=m.get("med_variant"),
                    section_canonical=canonical,
                    section_label=m.get("section_label") or _section_label(canonical),
                    source_page=m.get("source_page"),
                    snippet=text[:200].strip(),
                )
            )
        return out


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the process-wide ``RAGService`` (lazy, cached)."""
    return RAGService()


def __getattr__(name: str) -> object:
    """Lazy ``rag_service`` proxy for backward-compatible imports."""
    if name == "rag_service":
        return get_rag_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
