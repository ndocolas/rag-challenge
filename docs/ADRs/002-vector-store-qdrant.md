# ADR 002: Vector store — Qdrant

**Status:** Accepted
**Date:** 2026-05-20

## Context

The system needs a vector store that supports:

- Hybrid search (dense + sparse vectors) with server-side fusion
- Payload metadata filters (medication name, canonical section, reader profile)
- Simple Docker deploy with volume persistence
- Production quality (healthcheck, configurable indexes, no in-memory state)

Evaluated alternatives:

| Alternative | Reason for rejection |
|---|---|
| pgvector | No native hybrid search; depends on Postgres extension; higher latency |
| Chroma | No hybrid search; focused on prototyping |
| FAISS | In-memory; no payload filters; no native persistence |
| Weaviate | Configuration overhead; more rigid schema; heavier Docker image |

## Decision

Use **Qdrant v1.13.0**:

- Collection `bulas` with two named vectors: `dense` (cosine, 3072-dim) and `bm25` (sparse, via `fastembed` + `Qdrant/bm25` model)
- Hybrid search with two parallel `Prefetch` + server-side `Fusion.RRF`
- Payload filters: `med_name`, `section_canonical`, `patient_facing`
- Named Docker volume `qdrant_data` for persistence across restarts
- Exposed on port `6333`; `rag` profile in compose

## Consequences

**Positive:**
- Runs in 1 container with no external dependencies
- Server-side RRF (Reciprocal Rank Fusion) eliminates manual fusion in Python
- Rich payload filters allow section-hint queries without multiple collections
- UUIDv5 point IDs support idempotent deduplication during ingestion

**Negative / trade-offs:**
- Adds a container to the stack (vs using pgvector in an existing Postgres)
- Requires offline ingestion before any RAG query works
- fastembed downloads the BM25 model on first run (~50 MB)
