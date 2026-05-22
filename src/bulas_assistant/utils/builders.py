"""Lazy, cached factories for heavyweight LLM client objects.

Mirrors the helper-backend ``utils/builders.py`` pattern: keep the wiring
between :class:`Settings` and the LLM SDK isolated from the orchestration
layer (:class:`AssistantService`), so the orchestrator can be constructed in
tests with a fake LLM and without ``GOOGLE_API_KEY`` in the environment.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from bulas_assistant.utils.settings import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return the process-wide ``ChatGoogleGenerativeAI`` (lazy, cached).

    Reads from :func:`get_settings` on first call only. Subsequent calls
    return the same instance, which is what we want for the single-tenant
    assistant: one warm HTTP keep-alive pool against the Gemini endpoint.
    """
    cfg = get_settings()
    return ChatGoogleGenerativeAI(
        model=cfg.GEMINI_CHAT_MODEL,
        google_api_key=cfg.GOOGLE_API_KEY.get_secret_value(),
        temperature=0.2,
        streaming=True,
        max_output_tokens=cfg.GEMINI_MAX_OUTPUT_TOKENS,
        timeout=cfg.GEMINI_TIMEOUT_SECONDS,
        max_retries=cfg.GEMINI_MAX_RETRIES,
    )
