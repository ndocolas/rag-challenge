# Overview — Panvel Assistente Conversacional Farmacêutico

## Sobre o desafio

Construir assistente conversacional baseado em LLM que responde:
1. **Informação farmacológica** via RAG sobre corpus de 20 bulas reais da Anvisa
2. **Consulta a filiais** Panvel-PR (124 lojas) via tool calling

Backend FastAPI com streaming SSE, frontend React, observabilidade ponta-a-ponta,
docker-compose para subir tudo, e ADRs justificando decisões.

## Decisões técnicas fechadas

| Item | Escolha |
|---|---|
| Stack backend | Python 3.12 + FastAPI |
| Package manager | uv |
| LLM | Gemini (chat + embeddings) via LangChain |
| Orquestração | LangChain puro (`bind_tools` + loop manual) — sem LangGraph |
| Vector store | Qdrant (docker) |
| Memória conversa | Redis (chave session_id, TTL 30min) |
| Observabilidade | LangSmith + logs JSON estruturados |
| Chunking | Section-aware (16 chaves canônicas Anvisa RDC 47/2009) + fallback recursive 800 tokens / overlap 120 |
| Retrieval | Hybrid search dense + BM25 (RRF) no Qdrant |
| Tools filiais | 3: `buscar_filiais`, `detalhes_filial`, `listar_cidades_atendidas` |
| Frontend | React + Vite + TypeScript + Tailwind + shadcn/ui |
| Streaming | SSE (cliente: fetch + ReadableStream) |
| Escopo | Só RAG bulas + tools cadastrais filiais (sem analytics de vendas) |

## Princípios

- **Vertical slices**: LLM básico funcional antes de adicionar tools e RAG.
- **Clean separation**: `routes → services → utils`. Domain models em Pydantic.
- **Provider-agnostic**: trocar Gemini → outro LLM deve mexer só no `llm_service`.
- **Observable**: trace_id por request, logs JSON estruturados, LangSmith.
- **Testável**: cada service e tool com unit tests; integration tests com Qdrant docker.
- **Helper-backend como referência**: espelha padrões de `~/Desktop/helper-backend`
  (app factory, settings, handle_errors decorator, logger).

## Ordem de execução (linear)

```
01 Bootstrap + Utils
02 Models Pydantic
03 LLM básico (chat SSE + Redis multi-turno)
04 Tools filiais
05 RAG ingestão
06 RAG retrieval + integração chat
07 Observabilidade
08 Docker compose backend
09 Docs + ADRs + README
10 Frontend setup
11 Frontend integração
12 Frontend polish final
```

Cada task tem critério de verificação independente. Ao fim da Task 09 o backend está
completo e demonstrável via `curl`. Frontend cobre Tasks 10–12.

## Como usar estes arquivos

Cada `NN-nome.md` em `tasks/` é auto-contido:
- **Objetivo** — o que essa task entrega
- **Pré-requisitos** — tasks anteriores necessárias
- **Subtarefas** — passo a passo
- **Arquivos** — paths a criar/modificar
- **Snippets de referência** — quando útil
- **Verificação** — como provar que está pronto
- **Gotchas** — armadilhas conhecidas

Implementar uma task por vez em sessão separada para isolar contexto e qualidade.

## Dados disponíveis

- `~/Downloads/Case IA Generativa - Panvel/filiais.parquet` — 124 filiais PR
- `~/Downloads/Case IA Generativa - Panvel/corpus_bulas/` — 20 PDFs Anvisa
- `~/Downloads/Case IA Generativa - Panvel/dicionario_dados.xlsx` — dicionário

Mover/copiar para `data/` no repo na Task 01.
