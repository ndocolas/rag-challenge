"""Unit tests for the utils layer and the /health endpoint."""

import io
import json
import logging

import httpx
import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from panvel_assistant.utils.exceptions import (
    AppError,
    InvalidRequestError,
    LLMProviderError,
    ResourceNotFoundError,
    RetrievalError,
    ToolExecutionError,
)
from panvel_assistant.utils.handle_errors import handle_errors
from panvel_assistant.utils.logger import JsonFormatter, get_logger, trace_id_var
from panvel_assistant.utils.settings import Settings
from panvel_assistant.utils.sse import encode_event


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc-123")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

    cfg = Settings(_env_file=None)  # type: ignore[call-arg]

    assert cfg.GOOGLE_API_KEY == "abc-123"
    assert cfg.QDRANT_URL == "http://qdrant.example:6333"
    assert cfg.ENV == "staging"
    assert cfg.GEMINI_CHAT_MODEL == "gemini-2.0-flash"


async def test_handle_errors_resource_not_found():
    @handle_errors
    async def route():
        raise ResourceNotFoundError("leaflet not found")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 404
    assert "leaflet not found" in exc_info.value.detail


async def test_handle_errors_invalid_request():
    @handle_errors
    async def route():
        raise InvalidRequestError("payload missing session_id")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 400


async def test_handle_errors_validation_error():
    class Payload(BaseModel):
        n: int

    @handle_errors
    async def route():
        Payload.model_validate({"n": "not-a-number"})

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 422
    assert isinstance(exc_info.value.detail, list)


async def test_handle_errors_provider_error_is_503():
    @handle_errors
    async def route():
        raise LLMProviderError("gemini timeout")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 503


async def test_handle_errors_unexpected_error_is_500():
    @handle_errors
    async def route():
        raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 500


def test_logger_emits_json_with_trace_id():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("panvel_assistant.test.logger")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    token = trace_id_var.set("trace-xyz")
    try:
        logger.info("retrieval finished", extra={"step": "retrieval", "k": 4})
    finally:
        trace_id_var.reset(token)

    payload = json.loads(buffer.getvalue().strip())
    assert payload["message"] == "retrieval finished"
    assert payload["trace_id"] == "trace-xyz"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "panvel_assistant.test.logger"
    assert payload["step"] == "retrieval"
    assert payload["k"] == 4
    assert "timestamp" in payload


def test_logger_get_logger_is_idempotent():
    a = get_logger("panvel_assistant.test.idempotent")
    b = get_logger("panvel_assistant.test.idempotent")
    assert a is b
    assert len(a.handlers) == 1


def test_sse_encode_event():
    out = encode_event("token", {"x": 1})
    assert out == 'event: token\ndata: {"x": 1}\n\n'


def test_sse_encode_event_unicode():
    out = encode_event("done", {"msg": "olá"})
    assert "olá" in out
    assert out.startswith("event: done\n")
    assert out.endswith("\n\n")


def test_exceptions_hierarchy():
    for cls in (
        ResourceNotFoundError,
        InvalidRequestError,
        LLMProviderError,
        RetrievalError,
        ToolExecutionError,
    ):
        assert issubclass(cls, AppError)
        assert issubclass(cls, Exception)


async def test_health_endpoint_returns_ok_and_trace_id():
    from panvel_assistant.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health", headers={"X-Trace-Id": "abc-trace"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert response.headers.get("X-Trace-Id") == "abc-trace"


async def test_health_endpoint_generates_trace_id_when_absent():
    from panvel_assistant.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    trace_id = response.headers.get("X-Trace-Id")
    assert trace_id and trace_id != "-"


def test_validation_error_raised_outside_route_still_caught():
    class Payload(BaseModel):
        n: int

    with pytest.raises(ValidationError):
        Payload.model_validate({"n": "x"})
