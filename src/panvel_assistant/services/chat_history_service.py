"""Async Redis-backed chat message history for LangChain.

Mirrors the per-session/factory pattern from ``helper-backend``'s
``PostgresChatMessageHistory``, but persists ``BaseMessage`` entries in a Redis
LIST. Appends are atomic via a single ``RPUSH + LTRIM + EXPIRE`` pipeline and
serialization uses LangChain's canonical ``messages_to_dict`` format so any
future ``ToolMessage`` / multimodal content round-trips without schema changes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import redis.asyncio as redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)

MAX_MESSAGES = 20


def _serialize(message: BaseMessage) -> str:
    """Encode a ``BaseMessage`` as a JSON string via LangChain's canonical schema."""
    return json.dumps(messages_to_dict([message])[0], ensure_ascii=False)


def _deserialize(raw: str) -> BaseMessage:
    """Decode a JSON string back into a ``BaseMessage`` (any subtype)."""
    return messages_from_dict([json.loads(raw)])[0]


def serialize_messages_to_text(messages: list[BaseMessage]) -> str:
    """Flatten a history list into ``Human: \\`...\\`\\nAI: \\`...\\``` lines.

    Kept available for future prompt-template injection (Task 05/06); the MVP
    ``chat_service`` passes ``list[BaseMessage]`` directly to the LLM and does
    not consume this helper.
    """
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prefix = "Human"
        elif isinstance(msg, AIMessage):
            prefix = "AI"
        elif isinstance(msg, SystemMessage):
            prefix = "System"
        else:
            prefix = msg.type.capitalize() if msg.type else "Message"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{prefix}: `{content}`")
    return "\n".join(lines)


class RedisChatMessageHistory(BaseChatMessageHistory):
    """Per-session chat history persisted in a Redis LIST.

    Storage layout: ``chat:history:{session_id}`` → LIST of JSON-encoded
    ``BaseMessage`` dicts. The list is trimmed to the last ``MAX_MESSAGES``
    entries and the key's TTL is refreshed on every write.

    Sync methods (``messages``, ``add_message``, ``clear``) raise
    ``NotImplementedError`` — the application is fully async.
    """

    def __init__(self, session_id: str, client: redis.Redis) -> None:
        self._session_id = session_id
        self._client = client

    def _key(self) -> str:
        return f"chat:history:{self._session_id}"

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        raise NotImplementedError("Use aget_messages() in async context.")

    def add_message(self, message: BaseMessage) -> None:
        raise NotImplementedError("Use aadd_messages() in async context.")

    def clear(self) -> None:
        raise NotImplementedError("Use aclear() in async context.")

    async def aget_messages(self) -> list[BaseMessage]:
        raw_items = await self._client.lrange(self._key(), 0, -1)
        return [_deserialize(raw) for raw in raw_items]

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        payload = [_serialize(m) for m in messages]
        ttl = settings.CHAT_HISTORY_TTL_SECONDS
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.rpush(self._key(), *payload)
            pipe.ltrim(self._key(), -MAX_MESSAGES, -1)
            pipe.expire(self._key(), ttl)
            await pipe.execute()

    async def aclear(self) -> None:
        await self._client.delete(self._key())


class RedisHistoryStore:
    """Owns the async Redis client and produces per-session history instances."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await self._client.ping()
        logger.info("redis connected", extra={"url": settings.REDIS_URL})

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("redis disconnected")

    def get_session_history(self, session_id: str) -> RedisChatMessageHistory:
        if self._client is None:
            raise RuntimeError("RedisHistoryStore not connected")
        return RedisChatMessageHistory(session_id=session_id, client=self._client)


history_store = RedisHistoryStore()
