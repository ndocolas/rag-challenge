# Task 08 — Docker Compose backend

## Objetivo

`docker compose up` traz backend completo: API + Qdrant + Redis. Comando dedicado
para rodar ingestão. Healthchecks em todos services. Volumes nomeados para
persistência. Frontend ainda não — vem na Task 11.

## Pré-requisitos

- Tasks 01–07 (backend funcional via `uv run uvicorn`).

## Subtarefas

### 1. `docker-compose.yml` (raiz do repo)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    container_name: panvel-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: panvel-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: panvel-api
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./data:/app/data:ro
    healthcheck:
      test: ["CMD-SHELL", "python -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\")'"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s

  ingest:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: panvel-ingest
    profiles: ["ingest"]
    env_file: .env
    environment:
      QDRANT_URL: http://qdrant:6333
      REDIS_URL: redis://redis:6379/0
    depends_on:
      qdrant:
        condition: service_healthy
    volumes:
      - ./data:/app/data
    command: ["uv", "run", "python", "/app/scripts/ingest_bulas.py", "--rebuild"]

volumes:
  qdrant_data:
  redis_data:
```

### 2. Ajustar `Dockerfile` backend

Garantir que `scripts/` é copiado também (para o service `ingest`):

```dockerfile
FROM python:3.12-slim

RUN groupadd --system --gid 1001 app && useradd --system --uid 1001 --gid 1001 app

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY --chown=app:app backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=app:app backend/src/ ./src/
COPY --chown=app:app scripts/ ./scripts/

ENV PYTHONPATH=/app/src

USER app
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "panvel_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Nota: o Dockerfile vive em `backend/` mas o compose usa `context: ./backend`. Se quiser
manter o context em `./backend` e ainda copiar `scripts/` (que está na raiz), há 2
opções:
- (a) Mover `scripts/` para `backend/scripts/`
- (b) Trocar `context: ./` no compose, e ajustar paths do Dockerfile

Recomendado (a) — `scripts/` é parte do backend.

### 3. `.dockerignore` backend

```
__pycache__
*.pyc
.venv
.pytest_cache
.ruff_cache
tests/
.env
.env.*
```

### 4. `.env.example` na raiz do repo

(complementar ao da Task 01)

```env
# Required
GOOGLE_API_KEY=

# LangSmith (optional)
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=panvel-assistant
LANGCHAIN_TRACING_V2=true

# Defaults
GEMINI_CHAT_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=text-embedding-004
QDRANT_COLLECTION=bulas_panvel
ENV=dev
LOG_LEVEL=INFO
ALLOWED_ORIGINS=["http://localhost:5173"]
```

### 5. `Makefile` (opcional, conveniência)

```makefile
.PHONY: up down ingest logs reset

up:
\tdocker compose up -d

down:
\tdocker compose down

ingest:
\tdocker compose --profile ingest up ingest

logs:
\tdocker compose logs -f api

reset:
\tdocker compose down -v
```

### 6. README setup

Adicionar na raiz README.md:

```markdown
## Setup local com Docker

1. `cp .env.example .env` e preencha `GOOGLE_API_KEY`
2. Garanta que `data/filiais.parquet` e `data/corpus_bulas/*.pdf` existem
3. Suba serviços: `docker compose up -d qdrant redis`
4. Rode ingestão (1x): `docker compose --profile ingest up ingest`
5. Suba API: `docker compose up -d api`
6. Verifique: `curl http://localhost:8000/health`
7. Teste chat: `curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"t","message":"oi"}'`
```

## Verificação

```bash
# clone limpo (simular)
cd /tmp && git clone <repo> panvel-test && cd panvel-test
cp .env.example .env
# editar .env com GOOGLE_API_KEY válida + copiar data/

# subir infra
docker compose up -d qdrant redis
docker compose ps  # ambos healthy em ~15s

# ingerir bulas
docker compose --profile ingest up ingest
# espera log "OK: ~200 chunks indexados"

# subir API
docker compose up -d api
docker compose logs api | grep "app starting"
curl http://localhost:8000/health  # {"status":"ok"}

# turno chat
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"d1","message":"contraindicações ritalina"}'
# stream com sources + tokens + done

# trace
curl http://localhost:8000/admin/traces/<id> | jq

# parar tudo
docker compose down

# parar e apagar volumes (resetar dados)
docker compose down -v
```

## Gotchas

- Healthcheck do qdrant via TCP raw (`/dev/tcp/...`) porque a imagem oficial não tem
  curl/wget. Outra opção: `qdrant/qdrant:latest-curl` se existir, ou usar
  `start_period` longo e confiar no depends_on do API.
- `data/` montado como `:ro` no api: protege contra escrita acidental. Mas
  `data/cache/` precisa ser escrito pelo PDF extractor — ou monta cache em volume
  separado, ou roda extração só no service `ingest` que tem mount RW.
- Recomendação prática: usar 2 volumes:
  ```yaml
  api:
    volumes:
      - ./data/filiais.parquet:/app/data/filiais.parquet:ro
      - ./data/corpus_bulas:/app/data/corpus_bulas:ro
      - cache_data:/app/data/cache
  ingest:
    volumes:
      - ./data:/app/data
      - cache_data:/app/data/cache
  ```
  Cache é compartilhado entre api e ingest.
- `depends_on.condition: service_healthy` requer Compose v2.
- `.env` na raiz é lido pelo compose para substituição de variáveis em `${VAR}`; o
  `env_file:` injeta no container.
- Em produção, qdrant precisa de tuning de memória (`QDRANT__SERVICE__MAX_REQUEST_SIZE_MB`).
  MVP: defaults OK.
- Logs do api: stdout em JSON; redirecionar para shipper (Loki/Datadog) em prod.
