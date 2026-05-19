"""Structured JSON logger with trace_id injection via ContextVar.

Project-wide rule:

    from panvel_assistant.utils.logger import get_logger
    logger = get_logger(__name__)

Direct ``print()`` or ``logging.getLogger()`` calls are forbidden. Logs must
always be structured through ``extra={...}``.
"""

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

# Standard LogRecord attributes. Any key in record.__dict__ outside this set
# came from the logger call's `extra={...}` argument and is promoted to a
# top-level field in the emitted JSON.
_STANDARD_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Serializes each LogRecord as a single structured JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the LogRecord as JSON with an ISO-8601 UTC timestamp."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_ATTRS or key.startswith("_"):
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a single JSON handler, isolated from the root logger.

    Idempotent: subsequent calls for the same ``name`` reuse the already-configured
    handler, preventing duplicate log lines.
    """
    logger = logging.getLogger(name)
    if not getattr(logger, "_panvel_configured", False):
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        logger._panvel_configured = True  # type: ignore[attr-defined]
    return logger
