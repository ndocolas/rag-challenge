"""Domain-exception → HTTP-response mapping and unified error envelopes.

Every non-2xx response leaves this module with the same JSON shape::

    {"error": {"code": str, "message": str, "status_code": int,
               "trace_id": str, ...optional}}

That includes Pydantic 422 validation failures, ``HTTPException`` raised
anywhere in the stack, our own ``AppError`` hierarchy, the body-size
middleware's 413, and the unhandled-exception fallback. Clients only ever need
to parse one shape.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from bulas_assistant.utils.exceptions import (
    AppError,
    InvalidRequestError,
    LLMProviderError,
    RateLimitedError,
    ResourceNotFoundError,
    RetrievalError,
    SessionBusyError,
    ToolExecutionError,
)
from bulas_assistant.utils.logger import get_logger, trace_id_var

logger = get_logger(__name__)
_logger_extra = {"component.name": "ErrorHandler", "component.version": "v1"}


def error_response_payload(
    *,
    code: str,
    message: str,
    status_code: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical ``{"error": {...}}`` envelope."""
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "status_code": status_code,
        "trace_id": trace_id_var.get(),
    }
    if extra:
        error.update(extra)
    return {"error": error}


def _map_app_error(exc: Exception) -> tuple[int, str, str]:
    """Return ``(status_code, code, public_message)`` for a domain exception."""
    if isinstance(exc, ResourceNotFoundError):
        return 404, "not_found", str(exc)
    if isinstance(exc, SessionBusyError):
        return 409, "session_busy", str(exc)
    if isinstance(exc, RateLimitedError):
        return 429, "rate_limited", str(exc) or "too many requests"
    if isinstance(exc, InvalidRequestError):
        return 400, "invalid_request", str(exc)
    if isinstance(exc, (LLMProviderError, RetrievalError, ToolExecutionError)):
        # External-resource failures: do not leak provider error strings to
        # clients (they may include URLs, request IDs, etc).
        return 503, "upstream_unavailable", "upstream service unavailable"
    return 500, "internal_error", "internal server error"


def handle_errors(func):
    """Translate domain errors into ``HTTPException`` for route handlers.

    Wraps both sync and async handlers. Errors thrown by ``StreamingResponse``
    generators are surfaced via SSE ``error`` frames in the service layer; this
    decorator only fires before the response body starts streaming.

    The raised ``HTTPException`` is then re-rendered by ``http_exception_handler``
    below so the wire format stays consistent with the rest of the API.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        except HTTPException:
            raise
        except RateLimitedError:
            # Re-raise so the global app_error_handler handles it with the
            # full rate-limit context (limit, count, Retry-After headers).
            raise
        except AppError as exc:
            status_code, code, message = _map_app_error(exc)
            if status_code >= 500:
                logger.exception("domain error: %s", code, extra=_logger_extra)
            else:
                logger.info(
                    "domain error",
                    extra={**_logger_extra, "code": code, "status_code": status_code},
                )
            raise HTTPException(
                status_code=status_code,
                detail=error_response_payload(
                    code=code, message=message, status_code=status_code
                ),
            ) from exc
        except Exception as exc:
            logger.exception("unexpected error", extra=_logger_extra)
            raise HTTPException(
                status_code=500,
                detail=error_response_payload(
                    code="internal_error",
                    message="internal server error",
                    status_code=500,
                ),
            ) from exc

    return wrapper


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Global handler for any ``AppError`` not caught by ``handle_errors``."""
    status_code, code, message = _map_app_error(exc)
    if status_code >= 500:
        logger.exception("domain error (global): %s", code, extra=_logger_extra)

    extra: dict[str, Any] = {}
    headers: dict[str, str] | None = None
    if isinstance(exc, RateLimitedError):
        if exc.limit is not None:
            extra["limit"] = exc.limit
        if exc.count is not None:
            extra["count"] = exc.count
        headers = {
            "Retry-After": str(exc.retry_after),
        }
        if exc.limit is not None:
            headers["X-RateLimit-Limit"] = str(exc.limit)
            headers["X-RateLimit-Remaining"] = "0"

    return JSONResponse(
        status_code=status_code,
        content=error_response_payload(
            code=code, message=message, status_code=status_code, extra=extra or None
        ),
        headers=headers,
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Re-render every ``HTTPException`` through the canonical envelope.

    If ``detail`` is already a dict carrying our ``error`` envelope (e.g. from
    ``handle_errors`` above), unwrap and replay it without nesting. Otherwise
    treat the detail as a free-form message and synthesize the envelope.
    """
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        content = detail
    elif isinstance(detail, dict):
        content = error_response_payload(
            code=f"http_{exc.status_code}",
            message=detail.get("message") or "request failed",
            status_code=exc.status_code,
            extra={k: v for k, v in detail.items() if k != "message"},
        )
    else:
        # FastAPI emits ``HTTPException(detail="...")`` for raw 404/405/etc;
        # surface the string as ``message`` rather than echoing as-is.
        content = error_response_payload(
            code=f"http_{exc.status_code}",
            message=str(detail) if detail else "request failed",
            status_code=exc.status_code,
        )
    return JSONResponse(
        status_code=exc.status_code, content=content, headers=exc.headers or None
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Re-render 422s in the unified envelope, keeping field details under ``fields``."""
    return JSONResponse(
        status_code=422,
        content=error_response_payload(
            code="validation_error",
            message="request payload failed validation",
            status_code=422,
            extra={"fields": exc.errors()},
        ),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all so no 500 leaves the process without a structured body."""
    logger.exception("unhandled exception", extra=_logger_extra)
    return JSONResponse(
        status_code=500,
        content=error_response_payload(
            code="internal_error",
            message="internal server error",
            status_code=500,
        ),
    )
