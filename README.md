# RAG Bulas — Pharmaceutical Assistant

LLM assistant for pharmacological information (RAG over 20 Anvisa drug leaflets) and branch lookup (tool calling). Token-by-token SSE streaming.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 + FastAPI + uv |
| LLM + Embeddings | Gemini (`gemini-3-flash-preview` + `gemini-embedding-001`) via LangChain |
| Vector store | Qdrant v1.13.0 (dense + BM25 hybrid, RRF) |
| Memory | Redis 7 |
| Observability | LangSmith + structured JSON logs |
| Streaming | SSE (`text/event-stream`) |
| Frontend | React + Vite + TypeScript + Tailwind + shadcn/ui *(in development)* |

## Quick start (Docker)

**Prerequisites:** Docker, `data/corpus_bulas/*.pdf` (20 PDFs) and `data/filiais.parquet`.

```bash
cp .env.example .env                               # fill in GOOGLE_API_KEY

docker compose up -d qdrant redis                  # start infrastructure
docker compose --profile ingest up ingest          # ingest drug leaflets (once)
docker compose up -d api                           # start API

curl http://localhost:8000/health                  # {"status":"ok","env":"dev"}
```

Chat test:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"contraindicações da ritalina"}'
```

Stop everything (preserve volumes):

```bash
docker compose down
```

Reset data (delete Qdrant volumes):

```bash
docker compose down -v
```

## Local dev (without Docker)

```bash
uv sync
cp .env.example .env        # fill in GOOGLE_API_KEY + adjust URLs if needed
uv run uvicorn bulas_assistant.main:app --reload
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

## Tests

```bash
uv run pytest                        # unit tests (≥90% coverage)
uv run pytest -m integration         # integration tests (requires Qdrant + Redis)
uv run ruff check src/ tests/
```

## Layout

```
rag-challenge/
├── src/bulas_assistant/   # FastAPI backend
├── tests/                  # Unit + integration
├── scripts/                # ingest_bulas.py
├── data/                   # corpus_bulas/, filiais.parquet
├── requests/               # .http files for manual testing
├── tasks/                  # Execution plan (00–12)
├── docs/                   # Technical documentation
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Documentation

- [Architecture and components](docs/README.md)
- [ADRs — technical decisions](docs/ADRs/)
- [Pilot queries](docs/queries-piloto.md)
