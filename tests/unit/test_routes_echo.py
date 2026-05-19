"""Smoke tests for the demo route POST /v1/echo/chat."""

import httpx

from panvel_assistant.main import app


async def test_echo_chat_happy_path():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/echo/chat",
            json={"session_id": "s1", "message": "hi"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert body["content"] == "echo: hi"
    assert body["tool_calls"] is None


async def test_echo_chat_validation_error_empty_message():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/echo/chat",
            json={"session_id": "s1", "message": ""},
        )

    assert response.status_code == 422
