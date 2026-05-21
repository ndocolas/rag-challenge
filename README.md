# RAG Bulas — Assistente Farmacêutico

Assistente LLM para informação farmacológica (RAG sobre 20 bulas Anvisa) e consulta a filiais do PR (tool calling). Respostas em streaming SSE.

## Stack

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.12 + FastAPI + uv |
| LLM + Embeddings | Gemini (`gemini-3-flash-preview` + `gemini-embedding-001`) via LangChain |
| Vector store | Qdrant v1.13.0 (dense + BM25 hybrid, RRF) |
| Memória | Redis 7 |
| Observabilidade | LangSmith + logs JSON |
| Streaming | SSE (`text/event-stream`) |
| Frontend | React + Vite + TypeScript + Tailwind + shadcn/ui *(em desenvolvimento)* |

## Quick start (Docker)

**Pré-requisitos:** Docker, `data/corpus_bulas/*.pdf` (20 PDFs) e `data/filiais.parquet`.

```bash
cp .env.example .env                               # preencha GOOGLE_API_KEY

docker compose up -d qdrant redis                  # sobe infra
docker compose --profile ingest up ingest          # ingestão das bulas (1x)
docker compose up -d api                           # sobe API

curl http://localhost:8000/health                  # {"status":"ok","env":"dev"}
```

Teste de chat:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"contraindicações da ritalina"}'
```

Para derrubar tudo (preservando volumes):

```bash
docker compose down
```

Para resetar dados (apaga volumes Qdrant):

```bash
docker compose down -v
```

## Dev local (sem Docker)

```bash
uv sync
cp .env.example .env        # preencha GOOGLE_API_KEY + ajuste URLs se necessário
uv run uvicorn bulas_assistant.main:app --reload
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

## Testes

```bash
uv run pytest                        # unit tests (≥90% coverage)
uv run pytest -m integration         # integration tests (requer Qdrant + Redis)
uv run ruff check src/ tests/
```

## Layout

```
rag-challenge/
├── src/bulas_assistant/   # FastAPI backend
├── tests/                  # Unit + integration
├── scripts/                # ingest_bulas.py
├── data/                   # corpus_bulas/, filiais.parquet
├── requests/               # .http para testes manuais
├── tasks/                  # Plano de execução (00–12)
├── docs/                   # Documentação técnica
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Documentação

- [Arquitetura e componentes](docs/README.md)
- [ADRs — decisões técnicas](docs/ADRs/)
- [Queries piloto](docs/queries-piloto.md)
