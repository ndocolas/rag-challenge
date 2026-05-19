"""FastAPI application factory for the Panvel Assistant."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from panvel_assistant.routes.chat import router as chat_router
from panvel_assistant.routes.echo import router as echo_router
from panvel_assistant.services.chat_history_service import history_store
from panvel_assistant.utils.logger import get_logger, trace_id_var
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle hook."""
    logger.info("app starting", extra={"env": settings.ENV})
    await history_store.connect()
    try:
        yield
    finally:
        await history_store.disconnect()
        logger.info("app stopping")


def create_app() -> FastAPI:
    """Build and configure the FastAPI instance."""
    app = FastAPI(title="Panvel Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
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
    async def health():
        return {"status": "ok", "env": settings.ENV}

    app.include_router(echo_router)
    app.include_router(chat_router)

    return app


app = create_app()
