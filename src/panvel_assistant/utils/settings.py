"""Centralized application settings backed by pydantic-settings."""

from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


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
    GOOGLE_API_KEY: str
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    # Vector store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "bulas_panvel"

    # Cache / memory
    REDIS_URL: str = "redis://localhost:6379/0"
    CHAT_HISTORY_TTL_SECONDS: int = 1800
    TRACE_TTL_SECONDS: int = 3600

    # Observability
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "panvel-assistant"
    LANGCHAIN_TRACING_V2: bool = True

    # API
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    ENV: str = "dev"
    LOG_LEVEL: str = "INFO"

    # Paths (relative to the repo root)
    DATA_DIR: Path = _REPO_ROOT / "data"
    BULAS_DIR: Path = _REPO_ROOT / "data" / "corpus_bulas"
    FILIAIS_PARQUET: Path = _REPO_ROOT / "data" / "filiais.parquet"
    CACHE_DIR: Path = _REPO_ROOT / "data" / "cache"


settings = Settings()  # type: ignore[call-arg]
