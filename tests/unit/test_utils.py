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
from panvel_assistant.utils.sse import encode_event, encode_text_event


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc-123")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")

    cfg = Settings(_env_file=None)  # type: ignore[call-arg]

    assert cfg.GOOGLE_API_KEY.get_secret_value() == "abc-123"
    assert "abc-123" not in repr(cfg)  # SecretStr keeps the key out of dumps
    assert cfg.QDRANT_URL == "http://qdrant.example:6333"
    assert cfg.ENV == "staging"
    assert cfg.GEMINI_CHAT_MODEL == "gemini-3-flash-preview"


async def test_handle_errors_resource_not_found():
    @handle_errors
    async def route():
        raise ResourceNotFoundError("leaflet not found")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 404
    detail = exc_info.value.detail
    assert detail["error"]["code"] == "not_found"
    assert detail["error"]["message"] == "leaflet not found"
    assert detail["error"]["status_code"] == 404
    assert "trace_id" in detail["error"]


async def test_handle_errors_invalid_request():
    @handle_errors
    async def route():
        raise InvalidRequestError("payload missing session_id")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"]["code"] == "invalid_request"


async def test_handle_errors_wraps_unexpected_exceptions_as_internal_error():
    """Non-AppError, non-HTTPException raises become an opaque 500 envelope.

    Pydantic ``ValidationError`` raised inside a handler is a programmer
    bug (FastAPI validates request models *before* the handler runs and emits
    422 via ``RequestValidationError``). The decorator should not leak field
    details for these.
    """

    class Payload(BaseModel):
        n: int

    @handle_errors
    async def route():
        Payload.model_validate({"n": "not-a-number"})

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["error"]["code"] == "internal_error"


async def test_handle_errors_provider_error_is_503_and_redacted():
    @handle_errors
    async def route():
        raise LLMProviderError("gemini internal URL https://api.example/x?key=abc")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 503
    # Internal details must NOT leak to the client payload.
    detail = exc_info.value.detail
    assert detail["error"]["code"] == "upstream_unavailable"
    assert "api.example" not in detail["error"]["message"]


async def test_handle_errors_session_busy_is_409():
    from panvel_assistant.utils.exceptions import SessionBusyError

    @handle_errors
    async def route():
        raise SessionBusyError("session 'abc' already has a turn in flight")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"]["code"] == "session_busy"


async def test_handle_errors_unexpected_error_is_500():
    @handle_errors
    async def route():
        raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc_info:
        await route()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["error"]["code"] == "internal_error"


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


def test_sse_encode_text_event_normalizes_crlf():
    """LLMs may emit \\r\\n line endings; the SSE frame must collapse them.

    Per the SSE spec each of \\r, \\n, \\r\\n is a line terminator, so leaving a
    bare \\r in the data field splits the frame on the client side.
    """
    out = encode_text_event("token", "linha1\r\nlinha2\rfim")
    assert "\r" not in out
    assert out == "event: token\ndata: linha1\ndata: linha2\ndata: fim\n\n"


def test_sse_encode_text_event_empty_string():
    out = encode_text_event("token", "")
    assert out == "event: token\ndata: \n\n"


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


# ---------------------------------------------------------------------------
# Additional coverage for handle_errors, logger, main, and chat route.
# ---------------------------------------------------------------------------


def test_map_app_error_base_app_error_maps_to_500():
    from panvel_assistant.utils.handle_errors import _map_app_error

    status, code, _ = _map_app_error(AppError("raw error"))
    assert status == 500
    assert code == "internal_error"


async def test_handle_errors_wraps_sync_function():
    @handle_errors
    def route():
        return "ok"

    result = await route()
    assert result == "ok"


async def test_app_error_handler_500_level_error():
    from unittest.mock import MagicMock

    from fastapi import Request

    from panvel_assistant.utils.handle_errors import app_error_handler

    req = MagicMock(spec=Request)
    response = await app_error_handler(req, AppError("raw 500 error"))
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == "internal_error"


async def test_http_exception_handler_dict_with_error_key():
    from unittest.mock import MagicMock

    from fastapi import HTTPException, Request

    from panvel_assistant.utils.handle_errors import http_exception_handler

    req = MagicMock(spec=Request)
    exc = HTTPException(status_code=400, detail={"error": {"code": "x", "status_code": 400}})
    response = await http_exception_handler(req, exc)
    assert response.status_code == 400


async def test_http_exception_handler_plain_dict_detail():
    from unittest.mock import MagicMock

    from fastapi import HTTPException, Request

    from panvel_assistant.utils.handle_errors import http_exception_handler

    req = MagicMock(spec=Request)
    exc = HTTPException(status_code=400, detail={"message": "bad input"})
    response = await http_exception_handler(req, exc)
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["code"] == "http_400"


async def test_unhandled_exception_handler():
    from unittest.mock import MagicMock

    from fastapi import Request

    from panvel_assistant.utils.handle_errors import unhandled_exception_handler

    req = MagicMock(spec=Request)
    response = await unhandled_exception_handler(req, RuntimeError("chaos"))
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == "internal_error"


def test_json_formatter_includes_exc_info():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())

    exc_logger = logging.getLogger("panvel_assistant.test.exc_info")
    exc_logger.handlers.clear()
    exc_logger.addHandler(handler)
    exc_logger.setLevel(logging.ERROR)
    exc_logger.propagate = False

    try:
        raise ValueError("oops")
    except ValueError:
        exc_logger.exception("something failed")

    payload = json.loads(buffer.getvalue().strip())
    assert "exc_info" in payload
    assert "ValueError" in payload["exc_info"]


async def test_ready_endpoint_redis_up(monkeypatch):
    from unittest.mock import AsyncMock

    from panvel_assistant.main import app
    from panvel_assistant.services.chat_history_service import get_history_store

    monkeypatch.setattr(get_history_store(), "ping", AsyncMock(return_value=True))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json()["redis"] == "ok"


async def test_ready_endpoint_redis_down(monkeypatch):
    from unittest.mock import AsyncMock

    from panvel_assistant.main import app
    from panvel_assistant.services.chat_history_service import get_history_store

    monkeypatch.setattr(get_history_store(), "ping", AsyncMock(return_value=False))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["redis"] == "down"


def test_create_app_allows_authorization_header():
    from panvel_assistant.main import create_app

    app_instance = create_app(Settings(GOOGLE_API_KEY="test-key", ALLOW_AUTHORIZATION_HEADER=True))
    assert app_instance is not None


def test_client_ip_uses_x_forwarded_for():
    from unittest.mock import MagicMock

    from panvel_assistant.routes.chat_route import _client_ip

    req = MagicMock()
    req.headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
    assert _client_ip(req) == "203.0.113.5"


def test_resolve_log_level_falls_back_to_info_on_error(monkeypatch):
    from panvel_assistant.utils.logger import _resolve_log_level

    def _raise():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("panvel_assistant.utils.settings.get_settings", _raise)
    assert _resolve_log_level() == "INFO"


async def test_body_size_limit_middleware_rejects_declared_oversized_request():
    from panvel_assistant.main import BodySizeLimitMiddleware

    sent: list[dict] = []

    async def _mock_app(scope, receive, send):
        pass

    async def _capture_send(message):
        sent.append(message)

    middleware = BodySizeLimitMiddleware(_mock_app, max_bytes_provider=lambda: 100)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-length", b"9999999")],
    }

    async def _mock_receive():
        return {"type": "http.request", "body": b"x", "more_body": False}

    await middleware(scope, _mock_receive, _capture_send)
    assert sent[0]["status"] == 413


async def test_body_size_limit_middleware_passes_through_non_http_request_message():
    from panvel_assistant.main import BodySizeLimitMiddleware

    inner_received: list[dict] = []

    async def _mock_app(scope, receive, send):
        inner_received.append(await receive())

    middleware = BodySizeLimitMiddleware(_mock_app, max_bytes_provider=lambda: 100)
    scope = {"type": "http", "method": "POST", "headers": []}
    disconnect = {"type": "http.disconnect"}

    async def _mock_receive():
        return disconnect

    await middleware(scope, _mock_receive, lambda _: None)
    assert inner_received == [disconnect]
