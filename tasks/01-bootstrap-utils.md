# Task 01 — Bootstrap + Utils

## Objetivo

Repositório executável: FastAPI rodando com `GET /health`, toda a base de utilitários
no lugar (settings, exceptions, handle_errors, logger, sse), Dockerfile pronto.
Nenhuma lógica de domínio ainda.

## Pré-requisitos

- Nenhum (primeira task).

## Referência

Esta task espelha convenções de `~/Desktop/helper-backend`. Antes de começar, ler:
- `helper-backend/src/helper_backend/main.py` — app factory, lifespan, middlewares
- `helper-backend/src/helper_backend/utils/settings.py` — pattern pydantic-settings
- `helper-backend/src/helper_backend/utils/handle_errors.py` — decorator
- `helper-backend/src/helper_backend/utils/logger.py` — get_logger pattern
- `helper-backend/pyproject.toml` — uv + ruff + pytest config
- `helper-backend/Dockerfile` — uv não-root

## Subtarefas

### 1. Inicializar projeto Python com uv

```bash
cd backend
uv init --package panvel-assistant --python 3.12
```

Editar `pyproject.toml`:
```toml
[project]
name = "panvel-assistant"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = [
    "ruff>=0.7",
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "httpx>=0.27",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "RUF"]
ignore = ["D100", "D104"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=panvel_assistant --cov-report=term-missing --cov-fail-under=70"
```

### 2. Criar estrutura de pastas

```
backend/
├── src/panvel_assistant/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/__init__.py
│   ├── services/__init__.py
│   ├── models/__init__.py
│   ├── assistant/
│   │   ├── __init__.py
│   │   └── tools/__init__.py
│   └── utils/
│       ├── __init__.py
│       ├── settings.py
│       ├── exceptions.py
│       ├── handle_errors.py
│       ├── logger.py
│       └── sse.py
├── tests/
│   ├── __init__.py
│   ├── unit/__init__.py
│   ├── integration/__init__.py
│   └── conftest.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
└── .dockerignore
```

Criar também na raiz do repo:
- `data/` (copiar `filiais.parquet`, `corpus_bulas/`, `dicionario_dados.xlsx` de
  `~/Downloads/Case IA Generativa - Panvel/`)
- `data/cache/.gitkeep`
- `.env.example`
- `.gitignore` (Python, IDE, .env, .venv, data/cache/, dist/)
- `README.md` raiz mínimo

### 3. `utils/settings.py`

Espelhar `helper-backend/utils/settings.py`:

```python
from pathlib import Path
from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    GOOGLE_API_KEY: str
    GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    # Vector store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "bulas_panvel"

    # Cache / memória
    REDIS_URL: str = "redis://localhost:6379/0"
    CHAT_HISTORY_TTL_SECONDS: int = 1800
    TRACE_TTL_SECONDS: int = 3600

    # Observabilidade
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "panvel-assistant"
    LANGCHAIN_TRACING_V2: bool = True

    # API
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # Caminhos
    DATA_DIR: Path = Path(__file__).resolve().parents[3] / "data"
    BULAS_DIR: Path = DATA_DIR / "corpus_bulas"
    FILIAIS_PARQUET: Path = DATA_DIR / "filiais.parquet"
    CACHE_DIR: Path = DATA_DIR / "cache"


settings = Settings()
```

### 4. `utils/exceptions.py`

```python
class AppError(Exception):
    """Base para erros de domínio."""


class ResourceNotFoundError(AppError):
    pass


class InvalidRequestError(AppError):
    pass


class LLMProviderError(AppError):
    pass


class RetrievalError(AppError):
    pass


class ToolExecutionError(AppError):
    pass
```

### 5. `utils/handle_errors.py`

Espelhar `helper-backend/utils/handle_errors.py`, adaptado:

```python
import inspect
from functools import wraps

from fastapi import HTTPException
from pydantic import ValidationError

from panvel_assistant.utils.exceptions import (
    InvalidRequestError,
    LLMProviderError,
    ResourceNotFoundError,
    RetrievalError,
    ToolExecutionError,
)
from panvel_assistant.utils.logger import get_logger

logger = get_logger(__name__)


def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        except ResourceNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidRequestError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
        except (LLMProviderError, RetrievalError, ToolExecutionError) as e:
            logger.exception("Erro de provider/recurso: %s", e)
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            logger.exception("Erro inesperado")
            raise HTTPException(status_code=500, detail="Erro interno") from e

    return wrapper
```

### 6. `utils/logger.py` ⚠️ IMPORTANTE

**Portar exatamente o padrão de `helper-backend/src/helper_backend/utils/logger.py`**
(stdlib logging, `get_logger(__name__)` preconfigurado, formato JSON estruturado,
suporte a `extra={...}`). Adicionar:

- Injeção de `trace_id` via `contextvars.ContextVar` (lido pelo JSON formatter)
- Campos padrão: `timestamp`, `level`, `logger`, `message`, `trace_id`, `extra fields`

**Regra obrigatória para todo o projeto:**
```python
from panvel_assistant.utils.logger import get_logger
logger = get_logger(__name__)
```
Proibido `print()` ou `logging.getLogger()` direto em qualquer arquivo do projeto.

Logs sempre estruturados:
```python
logger.info(
    "retrieval concluído",
    extra={"step": "retrieval", "latency_ms": 120, "k": 4},
)
```

### 7. `utils/sse.py`

```python
import json
from typing import Any

EventType = str  # "token" | "tool_call" | "tool_result" | "sources" | "done" | "error"


def encode_event(event_type: EventType, payload: Any) -> str:
    """Encoda payload em formato SSE: `event: X\\ndata: {json}\\n\\n`."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"
```

### 8. `main.py` — app factory

```python
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from panvel_assistant.utils.logger import get_logger, trace_id_var
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app starting", extra={"env": settings.ENV})
    # placeholders para tasks futuras: carregar parquet, abrir clients Redis/Qdrant
    yield
    logger.info("app stopping")


def create_app() -> FastAPI:
    app = FastAPI(title="Panvel Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            trace_id_var.reset(token)

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": settings.ENV}

    return app


app = create_app()
```

### 9. `Dockerfile` backend

Espelhar `helper-backend/Dockerfile`:
```dockerfile
FROM python:3.12-slim

RUN groupadd --system --gid 1001 app && useradd --system --uid 1001 --gid 1001 app

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=app:app src/ ./src/
ENV PYTHONPATH=/app/src

USER app
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "panvel_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10. `.env.example`

```env
GOOGLE_API_KEY=sua-chave-aqui
GEMINI_CHAT_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=text-embedding-004

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=bulas_panvel

REDIS_URL=redis://localhost:6379/0

LANGSMITH_API_KEY=
LANGSMITH_PROJECT=panvel-assistant
LANGCHAIN_TRACING_V2=true

ALLOWED_ORIGINS=["http://localhost:5173"]
ENV=dev
LOG_LEVEL=INFO
```

### 11. Testes unitários

`tests/unit/test_utils.py`:
- `test_settings_load_from_env` — define env var, instancia Settings, valida
- `test_handle_errors_resource_not_found` — decora função que levanta, valida 404
- `test_handle_errors_validation_error` — valida 422
- `test_logger_emits_json_with_trace_id` — captura log, parse JSON, checa campo
- `test_sse_encode_event` — checa formato `event: X\ndata: {...}\n\n`
- `test_exceptions_hierarchy` — todos derivam de AppError

`tests/conftest.py`:
- fixture que carrega `.env.test` ou define env vars mínimas

## Verificação

```bash
cd backend
uv sync
uv run pytest                              # todos passam, cobertura >70%
uv run ruff check src/                     # zero erros
uv run uvicorn panvel_assistant.main:app   # sobe em :8000
curl http://localhost:8000/health          # {"status":"ok","env":"dev"}
curl -i http://localhost:8000/health | grep -i x-trace-id  # header presente

# Docker
docker build -t panvel-api backend/
docker run --rm -p 8000:8000 --env-file .env panvel-api
curl http://localhost:8000/health
```

## Gotchas

- `pydantic-settings` requer `extra="ignore"` ou explode com env vars extras (LangChain
  injeta vários).
- `find_dotenv()` precisa do arquivo `.env` na hierarquia; em Docker, use `--env-file`
  ou injete vars direto.
- `trace_id_var` deve ser `ContextVar[str]("trace_id", default="-")` para funcionar em
  async sem token reset issues.
- Logger JSON: usar `json.dumps(..., default=str)` para suportar datetime/UUID em extra.
- Path em `settings.DATA_DIR`: ajustar `parents[3]` conforme profundidade real do file.
