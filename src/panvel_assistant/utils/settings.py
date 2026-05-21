"""Centralized application settings backed by pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path:
    """Walk upwards from this file until a directory contains ``pyproject.toml``.

    More robust than ``parents[N]`` because it survives package re-layouts and
    deeper editable installs.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


_REPO_ROOT = _find_repo_root()


class Settings(BaseSettings):
    """Application configuration.

    Reads environment variables (and the nearest ``.env`` up the tree) and
    ignores unknown keys so we can coexist with vars injected by LangChain,
    LangSmith and similar tooling.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    GOOGLE_API_KEY: SecretStr
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_MAX_OUTPUT_TOKENS: int = Field(1024, ge=64, le=8192)
    GEMINI_TIMEOUT_SECONDS: float = Field(60.0, ge=1.0, le=600.0)
    GEMINI_MAX_RETRIES: int = Field(2, ge=0, le=10)

    # Vector store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "bulas_panvel"

    # Cache / memory
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = Field(20, ge=1, le=200)
    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(2.0, ge=0.1, le=30.0)
    REDIS_CONNECT_TIMEOUT_SECONDS: float = Field(2.0, ge=0.1, le=30.0)
    REDIS_HEALTHCHECK_TIMEOUT_SECONDS: float = Field(0.2, ge=0.05, le=5.0)
    CHAT_HISTORY_TTL_SECONDS: int = Field(1800, ge=60, le=86_400)
    CHAT_LOCK_TTL_SECONDS: int = Field(60, ge=5, le=600)
    TRACE_TTL_SECONDS: int = 3600

    # Observability
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "panvel-assistant"
    LANGCHAIN_TRACING_V2: bool = True

    # API
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    ALLOW_AUTHORIZATION_HEADER: bool = False
    MAX_REQUEST_BODY_BYTES: int = Field(16 * 1024, ge=1024, le=1024 * 1024)
    CHAT_RATE_LIMIT_PER_MINUTE: int = Field(20, ge=1, le=600)
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # Paths (relative to the repo root)
    DATA_DIR: Path = _REPO_ROOT / "data"
    BULAS_DIR: Path = _REPO_ROOT / "data" / "corpus_bulas"
    FILIAIS_PARQUET: Path = _REPO_ROOT / "data" / "filiais.parquet"
    CACHE_DIR: Path = _REPO_ROOT / "data" / "cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide ``Settings`` instance.

    Cached so consumers can call it in tight loops (FastAPI ``Depends``) without
    re-parsing the environment. Tests can monkeypatch this function or call
    ``get_settings.cache_clear()`` between cases.
    """
    return Settings()  # type: ignore[call-arg]


def __getattr__(name: str) -> Settings:
    """Lazy ``settings`` proxy for legacy ``from .settings import settings`` imports.

    Defers ``Settings()`` instantiation until the first attribute access, which
    keeps imports side-effect-free (tests that don't need real env still work)
    while preserving the previous public API.
    """
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Mypy can't introspect module-level ``__getattr__``; declare the expected
# attribute so type checkers see ``settings: Settings``.
settings: Settings
