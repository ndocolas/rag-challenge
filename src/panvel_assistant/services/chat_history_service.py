"""Async Redis-backed chat message history for LangChain.

Persists ``BaseMessage`` entries in a Redis LIST. Appends are atomic via a
single ``RPUSH + LTRIM + EXPIRE`` pipeline and serialization uses LangChain's
canonical ``messages_to_dict`` format so any future ``ToolMessage`` / multimodal
content round-trips without schema changes.

Each entry is JSON-encoded and stored as ``v1:{json}``. The ``v1:`` prefix lets
us evolve the on-wire schema in the future without losing the ability to read
older lists — entries with an unknown prefix are skipped (with a warning) on
read so a single bad message can't take the whole session down.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

import redis.asyncio as redis
import redis.exceptions as redis_exc
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from panvel_assistant.utils.exceptions import SessionBusyError
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import Settings, get_settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "ChatHistoryService", "component.version": "v1"}

MAX_MESSAGES = 20
SCHEMA_PREFIX = "v1:"


def _redact_url(url: str) -> str:
    """Remove password from a Redis URL before logging.

    ``redis://user:pass@host:6379/0`` -> ``redis://user:***@host:6379/0``.
    Falls back to a generic placeholder when the URL can't be parsed.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable>"
    if not parts.password:
        return url
    safe_userinfo = f"{parts.username or ''}:***"
    netloc = (
        f"{safe_userinfo}@{parts.hostname or ''}"
        f"{':' + str(parts.port) if parts.port else ''}"
    )
    return urlunsplit(parts._replace(netloc=netloc))


def _serialize(message: BaseMessage) -> str:
    """Encode a ``BaseMessage`` as ``v1:{json}`` using LangChain's canonical schema."""
    return SCHEMA_PREFIX + json.dumps(
        messages_to_dict([message])[0], ensure_ascii=False
    )


def _deserialize(raw: str) -> BaseMessage | None:
    """Decode a payload back into a ``BaseMessage``; return ``None`` on failure.

    Tolerant on read: corrupted entries (truncated JSON, unknown schema prefix,
    or unknown LangChain message types) yield ``None`` so the caller can skip
    them with a log warning instead of aborting the whole turn.
    """
    payload = raw
    if raw.startswith(SCHEMA_PREFIX):
        payload = raw[len(SCHEMA_PREFIX) :]
    try:
        return messages_from_dict([json.loads(payload)])[0]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "skipping corrupted history entry",
            extra={**_logger_extra, "error": type(exc).__name__, "raw_preview": raw[:80]},
        )
        return None


def serialize_messages_to_text(messages: list[BaseMessage]) -> str:
    """Flatten a history list into ``Human: \\`...\\`\\nAI: \\`...\\``` lines.

    Kept available for future prompt-template injection; the MVP ``AssistantService``
    passes ``list[BaseMessage]`` directly to the LLM and does not consume this
    helper.
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

    Storage layout: ``chat:history:{session_id}`` → LIST of ``v1:{json}``
    entries. The list is trimmed to the last ``MAX_MESSAGES`` items and the
    key's TTL is refreshed on every write.

    Sync methods (``messages``, ``add_message``, ``clear``) raise
    ``NotImplementedError`` — the application is fully async.
    """

    def __init__(
        self,
        session_id: str,
        client: redis.Redis,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._session_id = session_id
        self._client = client
        # Tests can leave ``ttl_seconds`` unset; resolve lazily so importing this
        # module doesn't require ``GOOGLE_API_KEY`` to be present.
        if ttl_seconds is None:
            ttl_seconds = get_settings().CHAT_HISTORY_TTL_SECONDS
        self._ttl_seconds = ttl_seconds

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
        # redis-py's type stubs union sync+async returns; the async client
        # only ever returns an awaitable here, hence the ignore.
        raw_items = await self._client.lrange(self._key(), 0, -1)  # type: ignore[misc]
        out: list[BaseMessage] = []
        for raw in raw_items:
            msg = _deserialize(raw)
            if msg is not None:
                out.append(msg)
        return out

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        payload = [_serialize(m) for m in messages]
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.rpush(self._key(), *payload)
            pipe.ltrim(self._key(), -MAX_MESSAGES, -1)
            pipe.expire(self._key(), self._ttl_seconds)
            await pipe.execute()

    async def aclear(self) -> None:
        await self._client.delete(self._key())


class RedisHistoryStore:
    """Owns the async Redis client and produces per-session history instances."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: redis.Redis | None = None
        # Background persistence tasks registered by ``ChatService``. Tracked
        # globally (for the lifespan ``disconnect`` drain) and indexed by
        # ``session_id`` so the route layer can drain only its own session's
        # tasks before releasing the per-session lock.
        self._pending: set[asyncio.Task] = set()
        self._pending_by_session: dict[str, set[asyncio.Task]] = {}

    async def connect(self) -> None:
        cfg = self._settings
        self._client = redis.from_url(
            cfg.REDIS_URL,
            decode_responses=True,
            max_connections=cfg.REDIS_MAX_CONNECTIONS,
            socket_timeout=cfg.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=cfg.REDIS_CONNECT_TIMEOUT_SECONDS,
        )
        await self._client.ping()  # type: ignore[misc]
        logger.info("redis connected", extra={**_logger_extra, "url": _redact_url(cfg.REDIS_URL)})

    async def disconnect(self) -> None:
        await self.drain_pending()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("redis disconnected", extra=_logger_extra)

    def register_pending(
        self, task: asyncio.Task, *, session_id: str | None = None
    ) -> None:
        """Track a background persistence task globally and (optionally) per session."""
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        if session_id is not None:
            bucket = self._pending_by_session.setdefault(session_id, set())
            bucket.add(task)

            def _cleanup(_t, sid=session_id) -> None:
                b = self._pending_by_session.get(sid)
                if b is None:
                    return
                b.discard(_t)
                if not b:
                    self._pending_by_session.pop(sid, None)

            task.add_done_callback(_cleanup)

    async def drain_pending(
        self,
        timeout: float = 5.0,  # noqa: ASYNC109 — bounded wait, not a request cap
        *,
        session_id: str | None = None,
    ) -> None:
        """Await registered background tasks with an upper bound.

        Without ``session_id`` every pending task is awaited (lifespan
        shutdown). With ``session_id`` only the tasks belonging to that
        session are drained (route layer, before releasing the per-session
        lock so the next turn sees the updated history).
        """
        if session_id is not None:
            pending = list(self._pending_by_session.get(session_id, ()))
        else:
            pending = list(self._pending)
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "drained pending tasks timed out",
                extra={
                    **_logger_extra,
                    "count": len(pending),
                    "timeout": timeout,
                    "session_id": session_id,
                },
            )

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("RedisHistoryStore not connected")
        return self._client

    async def rate_limit_check(
        self,
        bucket: str,
        *,
        max_per_minute: int,
    ) -> tuple[bool, int]:
        """Fixed-window per-minute counter; returns ``(allowed, current_count)``.

        Implementation: ``INCR rl:{bucket}:{epoch_minute}`` + ``EXPIRE`` to 65s
        so the key cleans itself up. Fixed window has well-known burst issues
        at the boundary, but for a chat-style endpoint with N≈20/min the
        tradeoff vs adding a sliding-window dependency is acceptable.
        """
        epoch_minute = int(time.time() // 60)
        key = f"rl:{bucket}:{epoch_minute}"
        try:
            async with self.client.pipeline(transaction=False) as pipe:
                pipe.incr(key)
                pipe.expire(key, 65)
                results = await pipe.execute()
            current = int(results[0])
        except redis_exc.RedisError:
            logger.warning(
                "rate limit check failed; fail-open",
                extra={**_logger_extra, "bucket": bucket},
            )
            return True, 0
        return current <= max_per_minute, current

    async def ping(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Liveness probe for the readiness endpoint.

        Returns ``True`` when the server responds within ``timeout`` seconds,
        ``False`` otherwise. Uses the configured healthcheck timeout when
        ``timeout`` is ``None``.
        """
        if self._client is None:
            return False
        deadline = timeout if timeout is not None else (
            self._settings.REDIS_HEALTHCHECK_TIMEOUT_SECONDS
        )
        try:
            await asyncio.wait_for(self._client.ping(), timeout=deadline)  # type: ignore[arg-type]
            return True
        except (TimeoutError, redis_exc.RedisError):
            return False

    def get_session_history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=session_id,
            client=self.client,
            ttl_seconds=self._settings.CHAT_HISTORY_TTL_SECONDS,
        )

    async def acquire_lock(self, session_id: str) -> str:
        """Acquire the per-session turn lock; return an opaque release token.

        Implementation: ``SET chat:lock:{sid} <token> NX EX {ttl}``. The token
        is a random hex string so ``release_lock`` can safely no-op when the
        lock was already taken over by a successor (TTL expired mid-turn).

        Raises ``LockBusyError`` when the key is already held; the route layer
        translates that into HTTP 409.
        """
        key = self._lock_key(session_id)
        token = secrets.token_hex(16)
        ttl = self._settings.CHAT_LOCK_TTL_SECONDS
        acquired = await self.client.set(key, token, nx=True, ex=ttl)
        if not acquired:
            raise LockBusyError(session_id)
        return token

    async def release_lock(self, session_id: str, token: str) -> None:
        """Best-effort CAS-release shielded against the caller's cancellation.

        Releasing the lock matters even (especially) when the route is being
        cancelled by a client disconnect — otherwise the key sits for the full
        ``CHAT_LOCK_TTL_SECONDS`` and the next turn for the same session 409s.

        We avoid a Lua ``EVAL`` because the dependency injected by tests
        (``fakeredis``) handles scripts differently than a real server. The
        tiny race between ``GET`` and ``DEL`` is harmless: the TTL is much
        longer than any chat turn, so a successor cannot have started yet.
        ``asyncio.shield`` guarantees the round-trip completes regardless of
        cancellation upstream.
        """
        key = self._lock_key(session_id)
        try:
            await asyncio.shield(self._release_lock_inner(key, token))
        except asyncio.CancelledError:
            # Shielded body keeps running; the cancellation still propagates
            # after the inner task settles, so re-raise to honor the contract.
            raise
        except redis_exc.RedisError:
            logger.warning("failed to release session lock", extra={**_logger_extra, "key": key})

    async def _release_lock_inner(self, key: str, token: str) -> None:
        current = await self.client.get(key)
        if current == token:
            await self.client.delete(key)

    @asynccontextmanager
    async def session_lock(self, session_id: str):
        """Acquire/release the turn lock as a context manager (test helper)."""
        token = await self.acquire_lock(session_id)
        try:
            yield
        finally:
            await self.release_lock(session_id, token)

    @staticmethod
    def _lock_key(session_id: str) -> str:
        return f"chat:lock:{session_id}"


class LockBusyError(SessionBusyError):
    """Raised when another turn is already in flight for the same session.

    Inherits from :class:`SessionBusyError` so the global ``handle_errors``
    decorator maps it to HTTP 409 without any route-level glue.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id!r} already has a turn in flight")
        self.session_id = session_id


@lru_cache(maxsize=1)
def get_history_store() -> RedisHistoryStore:
    """Return the process-wide ``RedisHistoryStore`` (lazy, cached)."""
    return RedisHistoryStore()


def __getattr__(name: str) -> object:
    """Lazy ``history_store`` proxy for backward-compatible imports."""
    if name == "history_store":
        return get_history_store()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
