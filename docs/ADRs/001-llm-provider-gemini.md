# ADR 001: LLM provider — Google Gemini

**Status:** Accepted
**Date:** 2026-05-20

## Context

The assistant needs a language model that simultaneously supports:

- Native tool calling (to invoke branch and RAG tools)
- Token-by-token streaming (SSE)
- Embedding generation (for ingestion and retrieval)
- Good quality in Brazilian Portuguese
- Affordable cost for an MVP with 20 leaflets and ~100 req/day

Evaluated alternatives:

| Provider | Reason for rejection |
|---|---|
| OpenAI GPT-4o | High cost; separate embeddings (`text-embedding-3`) increase complexity |
| Anthropic Claude | No proprietary embeddings; would require a second provider |
| AWS Bedrock | Heavy setup (IAM, VPC, temporary credentials); extra latency |
| Local models (Ollama) | Insufficient quality in PT-BR for tool calling + RAG |

## Decision

Use **Google Gemini** as the sole provider:

- Chat: `gemini-3-flash-preview` (`temperature=0.2`, streaming enabled)
- Embeddings: `gemini-embedding-001` (3072 dimensions, task types `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`)
- Integration via **LangChain** (`ChatGoogleGenerativeAI` + `GoogleGenerativeAIEmbeddings`)

## Consequences

**Positive:**
- Generous free tier covers MVP volume at no cost
- Unified client via LangChain: changing provider requires modifying only `builders.py`
- LangSmith traces Gemini calls automatically (via LangChain callbacks)
- 3072-dim embeddings sufficient for 20-leaflet corpus; BM25 complements recall

**Negative / trade-offs:**
- Dependency on a single external provider (Google availability and rate limits)
- 3072-dim embeddings vs 1536-dim from OpenAI: larger vector, but small corpus is not a bottleneck
- `gemini-3-flash-preview` is not the most capable model in the family; acceptable for the scope
