"""FastAPI application factory for the Panvel Assistant."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from panvel_assistant.routes.admin import router as admin_router
from panvel_assistant.routes.chat import router as chat_router
from panvel_assistant.routes.echo import router as echo_router
from panvel_assistant.services.chat_history_service import get_history_store
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.services.trace_service import trace_service
from panvel_assistant.utils.exceptions import AppError
from panvel_assistant.utils.handle_errors import (
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from panvel_assistant.utils.logger import get_logger, trace_id_var
from panvel_assistant.utils.settings import Settings, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t%(name)s - %(message)s",
)
logger = get_logger(__name__)
_logger_extra = {"component.name": "App", "component.version": "v1"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle hook."""
    settings = get_settings()
    history_store = get_history_store()
    logger.info("app starting", extra={**_logger_extra, "env": settings.ENV})

    if settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        logger.info(
            "langsmith tracing enabled",
            extra={**_logger_extra, "project": settings.LANGSMITH_PROJECT},
        )

    await history_store.connect()
    await trace_service.connect()
    filiais_service.load()
    try:
        yield
    finally:
        await trace_service.disconnect()
        await history_store.disconnect()
        logger.info("app stopping", extra=_logger_extra)


class BodySizeLimitMiddleware:
    """ASGI middleware that caps the request body at the configured size.

    Implemented at the raw ASGI layer (not ``BaseHTTPMiddleware``) so we can
    intercept ``http.request`` chunks as they arrive. A pure ``Content-Length``
    check is insufficient because ``Transfer-Encoding: chunked`` clients can
    omit that header and stream an arbitrarily large body — a real DoS vector
    on ``/chat``. We buffer the body locally (the cap is tiny, ~16 KB) and
    reject as soon as the byte tally crosses the threshold; the wrapped app
    then sees a single ``http.request`` message that replays the bytes.
    """

    def __init__(self, app, *, max_bytes_provider) -> None:
        self._app = app
        self._max_bytes_provider = max_bytes_provider

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method", "GET").upper() in {
            "GET",
            "HEAD",
            "OPTIONS",
            "DELETE",
        }:
            await self._app(scope, receive, send)
            return

        max_bytes = self._max_bytes_provider()
        cl = next(
            (
                v.decode("latin-1")
                for k, v in scope["headers"]
                if k.lower() == b"content-length"
            ),
            None,
        )
        if cl is not None:
            try:
                if int(cl) > max_bytes:
                    await _send_413(send, max_bytes=max_bytes, declared=int(cl))
                    return
            except ValueError:
                pass

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                # Disconnect or unknown event: forward unchanged.
                async def passthrough_receive(_msg=message):
                    return _msg

                await self._app(scope, passthrough_receive, send)
                return
            body.extend(message.get("body", b""))
            more_body = message.get("more_body", False)
            if len(body) > max_bytes:
                await _send_413(send, max_bytes=max_bytes, declared=len(body))
                return

        replayed = bytes(body)
        consumed = False

        async def replay_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {
                    "type": "http.request",
                    "body": replayed,
                    "more_body": False,
                }
            return await receive()

        await self._app(scope, replay_receive, send)


async def _send_413(send, *, max_bytes: int, declared: int) -> None:
    from panvel_assistant.utils.handle_errors import error_response_payload

    payload = error_response_payload(
        code="payload_too_large",
        message="request body exceeds the configured limit",
        status_code=413,
        extra={"limit_bytes": max_bytes, "declared_bytes": declared},
    )
    body = JSONResponse(status_code=413, content=payload).body
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add baseline browser hardening headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI instance."""
    cfg = settings or get_settings()
    app = FastAPI(title="Panvel Assistant", lifespan=lifespan)

    allow_headers = ["Content-Type"]
    if cfg.ALLOW_AUTHORIZATION_HEADER:
        allow_headers.append("Authorization")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=allow_headers,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    # Resolve the cap per-request via ``get_settings()`` so dependency overrides
    # (and hot-reloaded env values in dev) take effect without restarting the
    # app. We still pay the lru_cache hit, which is O(1).
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes_provider=lambda: get_settings().MAX_REQUEST_BODY_BYTES,
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
    async def health() -> dict[str, str]:
        """Liveness probe — always succeeds while the process is alive."""
        return {"status": "ok", "env": cfg.ENV}

    @app.get("/ready")
    async def ready() -> Response:
        """Readiness probe — verifies Redis answers a PING within the timeout.

        Returns 200 once the dependent services are reachable, 503 otherwise.
        Designed for orchestrator routing decisions (load balancer / k8s
        readinessProbe) so traffic stops flowing the moment Redis blips.
        """
        store = get_history_store()
        if await store.ping():
            return JSONResponse(
                status_code=200, content={"status": "ready", "redis": "ok"}
            )
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "redis": "down"}
        )

    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(echo_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    return app


app = create_app()
