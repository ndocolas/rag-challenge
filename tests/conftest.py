"""Test bootstrap.

Sets the minimal environment variables BEFORE any ``bulas_assistant.*``
import, because ``Settings()`` is instantiated at import time inside
``settings.py`` and ``GOOGLE_API_KEY`` has no default.
"""

import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("LOG_LEVEL", "INFO")
