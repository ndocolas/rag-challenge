"""Text logger with trace_id injection via ContextVar.

Project-wide rules:

    from panvel_assistant.utils.logger import get_logger
    logger = get_logger(__name__)
    _logger_extra = {"component.name": "MyModule", "component.version": "v1"}

    logger.info("something happened", extra=_logger_extra)

Direct ``print()`` or ``logging.getLogger()`` calls are forbidden.
"""

import logging
from contextvars import ContextVar

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


def _resolve_log_level() -> str:
    """Read the configured log level from ``Settings``, with a safe fallback.

    Imports are deferred so this module can be used during early bootstrap
    (e.g. logging from inside ``Settings`` itself) without a circular import.
    """
    try:
        from panvel_assistant.utils.settings import get_settings

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
    if not getattr(logger, "_panvel_configured", False):
        logger.setLevel(_resolve_log_level())
        handler = logging.StreamHandler()
        handler.setFormatter(TextFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        logger._panvel_configured = True  # type: ignore[attr-defined]
    return logger
