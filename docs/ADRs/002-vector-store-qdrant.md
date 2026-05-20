# ADR 002: Vector store — Qdrant

**Status:** Aceito
**Data:** 2026-05-20

## Contexto

O sistema precisa de um vector store que suporte:

- Busca híbrida (vetores densos + esparsos) com fusão no servidor
- Filtros por metadados de payload (nome do medicamento, seção canônica, perfil do leitor)
- Deploy simples via Docker, com persistência em volume
- Qualidade de produção (healthcheck, índices configuráveis, sem estado em memória)

Alternativas avaliadas:

| Alternativa | Motivo de descarte |
|---|---|
| pgvector | Sem hybrid search nativo; depende de extensão Postgres; latência maior |
| Chroma | Sem hybrid search; foco em prototipagem |
| FAISS | In-memory; sem filtros de payload; sem persistência nativa |
| Weaviate | Overhead de configuração; esquema mais rígido; imagem Docker mais pesada |

## Decisão

Usar **Qdrant v1.13.0**:

- Coleção `bulas_panvel` com dois vetores nomeados: `dense` (cosine, 3072-dim) e `bm25` (sparse, via `fastembed` + modelo `Qdrant/bm25`)
- Busca híbrida com dois `Prefetch` paralelos + `Fusion.RRF` server-side
- Filtros por payload: `med_name`, `section_canonical`, `patient_facing`
- Volume Docker nomeado `qdrant_data` para persistência entre reinicializações
- Exposto na porta `6333`; profile `rag` no compose

## Consequências

**Positivas:**
- Roda em 1 container sem dependências externas
- RRF (Reciprocal Rank Fusion) server-side elimina fusão manual no Python
- Filtros ricos por payload permitem queries com section hint sem múltiplas coleções
- Suporte a ponto IDs UUIDv5 facilita deduplicação idempotente na ingestão

**Negativas / trade-offs:**
- Adiciona um container à stack (vs usar pgvector em Postgres já existente)
- Requer ingestão offline antes de qualquer query RAG funcionar
- fastembed baixa o modelo BM25 na primeira execução (~50 MB)
