# Documentação técnica

## Visão geral

Backend conversacional com:

- RAG sobre 20 bulas Anvisa (corpus em PDF, indexado no Qdrant)
- Tool calling sobre 124 filiais Panvel-PR (in-memory a partir de parquet)
- Streaming SSE token-a-token
- Memória conversacional por sessão via Redis
- Observabilidade via LangSmith + logs JSON estruturados

## Arquitetura

```mermaid
sequenceDiagram
    participant U as Usuário (browser)
    participant API as FastAPI /chat
    participant CS as AssistantService
    participant RS as RAGService
    participant Q as Qdrant
    participant G as Gemini
    participant R as Redis
    participant T as tools

    U->>API: POST /chat (session_id, message)
    API->>CS: handle_turn
    CS->>R: load history
    CS->>RS: retrieve(message, k=4)
    RS->>Q: hybrid query (dense + BM25 RRF)
    Q-->>RS: chunks
    RS-->>CS: chunks + citations
    CS-->>U: SSE event: sources
    CS->>G: astream (messages + contexto + tools)
    loop até MAX_TOOL_ITERATIONS ou sem tool_calls
        G-->>CS: tool_calls?
        alt tem tool_calls
            CS-->>U: SSE event: tool_call
            CS->>T: invoke parallel
            T-->>CS: results
            CS-->>U: SSE event: tool_result
        else só texto
            G-->>CS: tokens
            CS-->>U: SSE event: token
        end
    end
    CS-->>U: SSE event: done
    CS->>R: persist updated history
```

## Componentes

| Camada | Responsabilidade |
|---|---|
| `routes/` | Validação de request, rate limit, session lock, encoding SSE |
| `assistant/` | Loop de tool calling (`AssistantService`), prompts, tools, sectionizer Anvisa |
| `services/` | Orquestração — RAG (`rag_service`), ingestão (`ingestion_service`), filiais, histórico Redis |
| `models/` | Schemas Pydantic (bula, chat, tool, filial) |
| `utils/` | Settings, logger JSON, handle_errors, sse, pdf (pdfplumber + cache) |

### Fluxo de ingestão (offline)

```
PDF
  → pdfplumber (texto + cache .txt)
  → sectionizer (regex Anvisa RDC 47/2009, 16 chaves canônicas)
  → chunks (seção inteira ≤3500 chars; recursive split 1600/200 para seções longas)
  → embeddings dense (Gemini 3072-dim) + sparse BM25 (fastembed)
  → upsert Qdrant (batch 32, concorrência 5)
```

### Eventos SSE

| Evento | Descrição |
|---|---|
| `sources` | Citações retornadas pelo RAG antes de iniciar geração |
| `token` | Fragmento de texto gerado pelo LLM |
| `tool_call` | Nome e argumentos da tool invocada |
| `tool_result` | Resultado retornado pela tool |
| `done` | Fim da resposta; inclui `trace_id` |
| `error` | Erro estruturado |

## ADRs

- [001 — LLM provider Gemini](ADRs/001-llm-provider-gemini.md)
- [002 — Vector store Qdrant](ADRs/002-vector-store-qdrant.md)
- [003 — Chunking section-aware Anvisa](ADRs/003-chunking-section-aware-anvisa.md)
- [004 — LangChain sem LangGraph](ADRs/004-langchain-sem-langgraph.md)
- [005 — Streaming SSE](ADRs/005-streaming-sse.md)

## Queries piloto

Bateria de 10 perguntas para validação manual: [queries-piloto.md](queries-piloto.md).

## Setup

Veja o [README raiz](../README.md).
