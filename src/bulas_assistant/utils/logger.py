"""Text and JSON loggers with trace_id injection via ContextVar.

Project-wide rules:

    from bulas_assistant.utils.logger import get_logger
    logger = get_logger(__name__)
    _logger_extra = {"component.name": "MyModule", "component.version": "v1"}

    logger.info("something happened", extra=_logger_extra)

Direct ``print()`` or ``logging.getLogger()`` calls are forbidden.
"""

import json
import logging
from contextvars import ContextVar
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class TextFormatter(logging.Formatter):
    """Formats each log record as a human-readable text line with trace_id.

    Injects the current ``trace_id_var`` value into the record so it can be
    referenced as ``%(trace_id)s`` in the format string without callers having
    to pass it explicitly.
    """

    _FMT = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace=%(trace_id)s] - %(message)s"  # noqa: E501

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT)

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = trace_id_var.get()
        return super().format(record)


# Standard LogRecord attributes that should not be surfaced as extra fields.
_LOGRECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
    | {"message", "asctime"}
)


class JsonFormatter(logging.Formatter):
    """Formats each log record as a single JSON line.

    Standard fields emitted: ``timestamp``, ``level``, ``logger``,
    ``message``, ``trace_id``.  Any ``extra`` keys passed to the logger
    call are merged into the top-level JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "trace_id": trace_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _LOGRECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _resolve_log_level() -> str:
    """Read the configured log level from ``Settings``, with a safe fallback.

    Imports are deferred so this module can be used during early bootstrap
    (e.g. logging from inside ``Settings`` itself) without a circular import.
    """
    try:
        from bulas_assistant.utils.settings import get_settings

        return get_settings().LOG_LEVEL
    except Exception:
        return "INFO"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a single text handler, isolated from the root logger.

    Idempotent: subsequent calls for the same ``name`` reuse the already-configured
    handler, preventing duplicate log lines. Reads ``LOG_LEVEL`` from the
    centralized ``Settings`` so the env value is parsed in exactly one place.
    """
    logger = logging.getLogger(name)
    if not getattr(logger, "_bulas_configured", False):
        logger.setLevel(_resolve_log_level())
        handler = logging.StreamHandler()
        handler.setFormatter(TextFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        logger._bulas_configured = True  # type: ignore[attr-defined]
    return logger
