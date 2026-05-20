"""Integration-suite bootstrap.

The root ``tests/conftest.py`` force-injects ``GOOGLE_API_KEY=test-key`` via
``setdefault`` so unit tests can import ``panvel_assistant`` without a real
Gemini key. Integration tests, however, need the real key from ``.env``.

This module runs at collection time (before any panvel import inside a test
collects), reads ``.env`` and OVERWRITES the placeholder so ``Settings`` (and
all downstream singletons) see the genuine credential.
"""

from __future__ import annotations

import os

from dotenv import dotenv_values

_PLACEHOLDERS = {"", "test-key", "fake", "dummy"}


def _load_real_key() -> None:
    current = os.environ.get("GOOGLE_API_KEY", "")
    if current and current not in _PLACEHOLDERS:
        return
    real = dotenv_values().get("GOOGLE_API_KEY", "") or ""
    if real and real not in _PLACEHOLDERS:
        os.environ["GOOGLE_API_KEY"] = real
        # Bust the cached Settings() so subsequent ``settings.GOOGLE_API_KEY``
        # reads pick up the real value. Already-bound module references are
        # rebuilt lazily via the ``__getattr__`` proxy in settings.py.
        try:
            from panvel_assistant.utils.settings import get_settings

            get_settings.cache_clear()
        except Exception:
            pass


_load_real_key()
