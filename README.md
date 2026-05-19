# Panvel Pharmaceutical Conversational Assistant

LLM-powered conversational assistant that answers two kinds of questions:

1. **Pharmacological information** — RAG over a corpus of 20 real Anvisa drug leaflets.
2. **Branch (filial) lookups** — Panvel-PR (124 stores) via tool calling.

Stack: Python 3.12 + FastAPI + Gemini (via LangChain) + Qdrant + Redis + React/Vite.

Full execution plan: see [tasks/00-overview.md](tasks/00-overview.md).

## Layout

```
rag-challenge/
├── src/panvel_assistant/   # FastAPI service (Task 01+)
├── tests/                  # Unit + integration tests
├── data/                   # Leaflets, branches parquet, dictionary (Task 01)
├── requests/               # .rest files for manual route testing
├── tasks/                  # Linear execution plan (00–12)
├── Dockerfile              # Backend image
├── docker-compose.yml      # Local orchestration (backend-only for now)
└── pyproject.toml          # uv + ruff + pytest config
```

## Quickstart

```bash
uv sync
cp .env.example .env        # fill GOOGLE_API_KEY
uv run uvicorn panvel_assistant.main:app --reload
```

Healthcheck:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","env":"dev"}
```

## Tests

```bash
uv run pytest               # unit tests + coverage (>=70%)
uv run ruff check src/ tests/
```

## Docker

```bash
docker compose build backend
docker compose up -d backend
curl http://localhost:8000/health
docker compose down
```

If you don't have a `.env` checked out, the compose file tolerates it
(`env_file` is `required: false`). Provide your variables via shell env
or `--env-file path/to/file` as needed.
