# ADR 003: Anvisa Section-Aware Chunking

**Status:** Accepted
**Date:** 2026-05-20

## Context

The corpus consists of 20 Anvisa drug leaflets in RDC 47/2009 format, which defines two structures:

- **IAP (Patient Information):** 9 questions in accessible language (e.g., "What is this medication indicated for?", "How should I use this medication?")
- **IT (Technical Information):** clinical sections (dosage, contraindications, interactions, pharmacokinetics)

Generic size-based chunking loses section semantics: a 400-token chunk may cross the boundary between dosage and contraindications, degrading recall for specific queries.

Evaluated alternatives:

| Strategy | Reason for rejection |
|---|---|
| Pure recursive char split | Ignores structure; chunks cross sections; imprecise citations |
| Semantic chunking (embedding-based) | 2× embedding cost during ingestion; non-deterministic; slow |
| Page-level | Chunks too large (>2000 tokens); diffuse context; no section citation |

## Decision

**Section-aware chunking** with 16 canonical keys mapping regex headers from the leaflets:

- `IAP_*` prefix for patient-facing sections (e.g., `IAP_6_POSOLOGIA`, `IAP_8_REACOES_ADVERSAS`)
- `IT_*` prefix for technical sections (e.g., `IT_INTERACOES_MEDICAMENTOSAS`, `IT_FARMACOCINETICA`)
- Sections with content ≤ 3500 chars → single chunk (`is_full_section=True`)
- Long sections → recursive split with 1600 tokens, 120 overlap, section header prefixed on each sub-chunk
- Sections without detected header → `UNCLASSIFIED` key (100% coverage fallback)
- Multi-product leaflets (e.g., Ritalina IR/LA) → `med_variant` field in payload metadata

## Consequences

**Positive:**
- Rich citations `(bula_id, section_canonical, page_range)` displayed in frontend
- High recall for specific queries: `section_hint` filters directly by canonical key
- Full coverage: UNCLASSIFIED covers leaflets with irregular text extraction
- Deterministic and idempotent ingestion (UUIDv5 IDs per chunk_id)

**Negative / trade-offs:**
- Header detection regex may fail on leaflets with non-standard formatting (mitigated by fallback)
- 16 canonical keys require maintenance if Anvisa updates RDC 47/2009
- Section hints in agent code must stay in sync with the canonical keys
