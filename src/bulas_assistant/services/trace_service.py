"""Internal trace buffer: per-turn audit log persisted to Redis.

Each chat turn is captured as a structured blob keyed by trace_id and stored
with a 1-hour TTL so ``GET /admin/traces/{trace_id}`` can replay it after the
fact. The in-memory buffer is keyed by ``trace_id_var`` (contextvar), so
concurrent async requests never cross-contaminate each other.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as redis

from bulas_assistant.utils.logger import get_logger, trace_id_var
from bulas_assistant.utils.settings import settings

logger = get_logger(__name__)


class TraceService:
    """Collects per-turn telemetry and persists it to Redis on finalize."""

    def __init__(self) -> None:
        self._redis: redis.Redis | None = None
        self._buffers: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def start(self, session_id: str, user_message: str) -> str:
        """Open a new trace buffer for the current contextvar trace_id."""
        tid = trace_id_var.get()
        self._buffers[tid] = {
            "trace_id": tid,
            "session_id": session_id,
            "user_message": user_message,
            "started_at": time.time(),
            "steps": [],
            "tool_calls": [],
            "citations": [],
            "tokens_in": None,
            "tokens_out": None,
            "final_response": None,
        }
        return tid

    def add_step(self, name: str, latency_ms: float, **extra: Any) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["steps"].append(
                {"name": name, "latency_ms": latency_ms, **extra}
            )

    def add_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        result_preview: str | None,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["tool_calls"].append(
                {
                    "name": name,
                    "args": args,
                    "result_preview": result_preview,
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    def set_citations(self, citations: list[dict[str, Any]]) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["citations"] = citations

    def set_response(
        self,
        text: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["final_response"] = text
            self._buffers[tid]["tokens_in"] = tokens_in
            self._buffers[tid]["tokens_out"] = tokens_out

    async def finalize(self) -> None:
        """Flush the current buffer to Redis and remove it from memory."""
        tid = trace_id_var.get()
        if tid not in self._buffers:
            return
        buf = self._buffers.pop(tid)
        buf["duration_ms"] = (time.time() - buf["started_at"]) * 1000
        if self._redis is not None:
            await self._redis.set(
                f"trace:{tid}",
                json.dumps(buf, ensure_ascii=False, default=str),
                ex=settings.TRACE_TTL_SECONDS,
            )
        logger.info(
            "trace finalizado",
            extra={
                "trace_id": tid,
                "duration_ms": buf["duration_ms"],
                "tool_count": len(buf["tool_calls"]),
                "citation_count": len(buf["citations"]),
            },
        )

    async def get(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieve a previously finalized trace from Redis."""
        if self._redis is None:
            return None
        raw = await self._redis.get(f"trace:{trace_id}")
        return json.loads(raw) if raw else None


trace_service = TraceService()
