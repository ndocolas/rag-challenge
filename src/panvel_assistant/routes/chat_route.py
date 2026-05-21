"""``POST /chat`` route — token-by-token SSE stream backed by Gemini + Redis memory.

Lock handling lives in the route (not the streaming generator) so a busy
session can be rejected with HTTP 409 *before* response headers are flushed —
the SSE body would otherwise be the only place to report the conflict.

Two correctness guarantees that the route layer enforces:

- ``release_lock`` is shielded inside ``RedisHistoryStore``, so a client
  disconnect mid-stream still frees the per-session lock instead of pinning
  the next turn behind the 60-second TTL.
- The lock is held until the background persistence task for this turn
  completes (or the configured drain timeout elapses), so an immediate turn
  N+1 from the same session never reads stale history.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from panvel_assistant.assistant.assistant_service import (
    AssistantService,
    get_assistant_service,
)
from panvel_assistant.models.chat_models import ChatRequest
from panvel_assistant.services.chat_history_service import (
    RedisHistoryStore,
    get_history_store,
)
from panvel_assistant.utils.exceptions import RateLimitedError
from panvel_assistant.utils.handle_errors import handle_errors
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import Settings, get_settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "ChatRoute", "component.version": "v1"}

router = APIRouter(tags=["chat"])


def _client_ip(request: Request) -> str:
    """Best-effort client IP, falling back to direct peer when no XFF is set."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/chat")
@handle_errors
async def chat(
    request: Request,
    req: ChatRequest,
    assistant_service: AssistantService = Depends(get_assistant_service),
    history_store: RedisHistoryStore = Depends(get_history_store),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream the assistant's response for one chat turn as SSE frames."""
    bucket = f"chat:{_client_ip(request)}:{req.session_id}"
    allowed, count = await history_store.rate_limit_check(
        bucket, max_per_minute=settings.CHAT_RATE_LIMIT_PER_MINUTE
    )
    if not allowed:
        raise RateLimitedError(
            "too many requests; slow down",
            retry_after=60,
            limit=settings.CHAT_RATE_LIMIT_PER_MINUTE,
            count=count,
        )

    token = await history_store.acquire_lock(req.session_id)

    async def stream():
        # Ensures the lock is freed even if the generator is closed (GC,
        # client disconnect, server shutdown) before its body completes —
        # ``finally`` inside ``async for`` does not run when ``aclose`` is
        # invoked on a generator that never iterated. We use a flag instead
        # of a ``try/finally`` around the loop so the release path runs
        # exactly once regardless of how the generator exits.
        try:
            async for frame in assistant_service.handle_turn(req):
                yield frame
        finally:
            try:
                await history_store.drain_pending(
                    timeout=2.0, session_id=req.session_id
                )
            except Exception:
                logger.warning("drain_pending failed", extra=_logger_extra, exc_info=True)
            # Cancellation upstream must not skip the lock release; the
            # store shields the Redis round-trip internally.
            await history_store.release_lock(req.session_id, token)

    try:
        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-RateLimit-Limit": str(settings.CHAT_RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": str(
                    max(settings.CHAT_RATE_LIMIT_PER_MINUTE - count, 0)
                ),
            },
        )
    except BaseException:
        # If wiring the response itself fails (or the request is cancelled
        # between ``acquire_lock`` above and Starlette taking ownership of
        # the generator), the ``stream()`` ``finally`` would never run and
        # the lock would sit pinned for the full TTL. Release it here so
        # the next turn for this session isn't stuck behind a 409.
        await history_store.release_lock(req.session_id, token)
        raise
