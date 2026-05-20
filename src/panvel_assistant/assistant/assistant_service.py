"""Core assistant service: tool-aware Gemini streaming + Redis-backed chat memory.

Mirrors the layout of ``helper-backend/.../assistant/assistant_service.py``:
the LLM client, the bounded tool-calling loop, and the per-turn orchestration
(history load + SSE encoding + background persistence) all live in a single
``AssistantService``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import add_ai_message_chunks
from langchain_google_genai import ChatGoogleGenerativeAI

from panvel_assistant.assistant.agent_tools import build_tools
from panvel_assistant.assistant.prompts import SYSTEM_PROMPT_MVP
from panvel_assistant.models.chat import ChatRequest, ToolCallTrace
from panvel_assistant.services.chat_history_service import (
    RedisHistoryStore,
    get_history_store,
)
from panvel_assistant.services.filiais_service import FiliaisService
from panvel_assistant.services.filiais_service import (
    filiais_service as default_filiais_service,
)
from panvel_assistant.services.rag_service import RAGService, get_rag_service
from panvel_assistant.services.trace_service import trace_service
from panvel_assistant.utils.builders import get_llm
from panvel_assistant.utils.logger import get_logger, trace_id_var
from panvel_assistant.utils.sse import encode_event, encode_stream_error, encode_text_event

logger = get_logger(__name__)
_logger_extra = {"component.name": "AssistantService", "component.version": "v1"}

_SYSTEM_MSG = SystemMessage(content=SYSTEM_PROMPT_MVP)


def _citations_from_tool_message(content: object) -> list[dict[str, Any]]:
    """Parse a ``buscar_bulas`` ToolMessage payload into Citation dicts.

    Returns ``[]`` on any parsing failure so a malformed payload never breaks
    the stream — the tokens still flow, only the ``sources`` SSE event is
    skipped.
    """
    if not isinstance(content, str) or not content:
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("buscar_bulas payload nao parsavel; sources omitido", extra=_logger_extra)
        return []
    if not isinstance(payload, dict):
        return []
    # Structured error payload (medicamento_nao_encontrado, nenhum_resultado):
    # carries no citation content, skip the sources event entirely.
    if "error" in payload:
        return []
    matches = payload.get("matches")
    if not matches:
        return []
    try:
        from panvel_assistant.services.rag_service import RAGService

        citations = RAGService.citations_from_matches(matches)
        return [c.model_dump() for c in citations]
    except Exception:
        logger.warning("falha ao construir citations a partir de matches", extra=_logger_extra)
        return []


def _extract_token_counts(
    ai_msg: object,
) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, output_tokens)`` from a merged AIMessage.

    Gemini surfaces usage via ``usage_metadata`` which may be a dict (older
    LangChain builds) or a dataclass-like object (newer builds). We handle
    both to stay compatible across patch versions.
    """
    usage = getattr(ai_msg, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)


def _coerce_content_to_text(content: object) -> str:
    """Reduce a chunk's ``content`` to a plain string.

    LangChain v1 chunks may carry either a ``str`` (Gemini text-only path) or a
    list of segment dicts (multimodal / tool-call chunks). For streaming we
    only forward textual deltas; non-text segments are dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        dropped = 0
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                dropped += 1
        if dropped:
            logger.debug("dropped %d non-text chunk segments", dropped, extra=_logger_extra)
        return "".join(parts)
    return ""


class AssistantService:
    """Tool-bound Gemini client plus single-turn orchestration."""

    def __init__(
        self,
        filiais_service: FiliaisService | None = None,
        rag_service: RAGService | None = None,
        history: RedisHistoryStore | None = None,
        llm: ChatGoogleGenerativeAI | None = None,
    ) -> None:
        base = llm or get_llm()
        tools = build_tools(
            filiais_service or default_filiais_service,
            rag_service or get_rag_service(),
        )
        self._llm = base.bind_tools(tools)
        self._tools_by_name = {t.name: t for t in tools}
        self._history = history or get_history_store()

    async def _execute_tool(
        self, call: dict[str, Any]
    ) -> tuple[ToolMessage, ToolCallTrace]:
        """Invoke a single tool call; always returns (ToolMessage, trace).

        Unknown tools and tool exceptions both surface as JSON error blobs
        embedded in the ``ToolMessage``, never as raised exceptions, so the
        outer loop can keep streaming.
        """
        name = call["name"]
        args = call.get("args", {}) or {}
        # Gemini normally emits a non-empty ``id`` per tool call, but the
        # contract is "optional". A blank ``tool_call_id`` would unpair the
        # request from its ``ToolMessage`` reply on the next ``astream``
        # turn — synthesize a stable fallback and log so we notice if this
        # path actually starts firing in production.
        call_id = call.get("id")
        if not call_id:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            logger.warning(
                "tool call missing id; synthesized fallback",
                extra={**_logger_extra, "tool": name, "fallback_id": call_id},
            )
        tool = self._tools_by_name.get(name)
        started = time.perf_counter()
        if tool is None:
            latency = (time.perf_counter() - started) * 1000
            content = json.dumps({"error": "unknown_tool", "name": name})
            trace_service.add_tool_call(
                name=name, args=args, result_preview=None,
                latency_ms=latency, error="unknown_tool",
            )
            return (
                ToolMessage(content=content, tool_call_id=call_id, name=name),
                ToolCallTrace(name=name, args=args, error="unknown_tool", latency_ms=latency),
            )
        try:
            # Async tools (``@tool`` over a coroutine function) carry a
            # non-None ``coroutine`` attribute; dispatch via ``ainvoke`` so we
            # don't spawn a worker thread just to drive an event loop.
            if getattr(tool, "coroutine", None) is not None:
                result = await tool.ainvoke(args)
            else:
                result = await asyncio.to_thread(tool.invoke, args)
            latency = (time.perf_counter() - started) * 1000
            text = str(result)
            trace_service.add_tool_call(
                name=name, args=args, result_preview=text[:500],
                latency_ms=latency, error=None,
            )
            return (
                ToolMessage(content=text, tool_call_id=call_id, name=name),
                ToolCallTrace(
                    name=name,
                    args=args,
                    result_preview=text[:500],
                    latency_ms=latency,
                ),
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            err = json.dumps({"error": "tool_execution_failed", "message": str(exc)})
            trace_service.add_tool_call(
                name=name, args=args, result_preview=None,
                latency_ms=latency, error=str(exc),
            )
            return (
                ToolMessage(content=err, tool_call_id=call_id, name=name),
                ToolCallTrace(name=name, args=args, error=str(exc), latency_ms=latency),
            )

    def _detect_tool_loop(
        self, tool_calls: list[dict], seen_calls: set[str]
    ) -> str | None:
        """Return the name of the first repeated tool call, or None.

        Mutates ``seen_calls`` with fingerprints of non-duplicate calls so the
        caller can keep accumulating across iterations.
        """
        for tc in tool_calls:
            fingerprint = json.dumps(
                {"name": tc["name"], "args": tc.get("args") or {}},
                sort_keys=True,
            )
            if fingerprint in seen_calls:
                return tc["name"]
            seen_calls.add(fingerprint)
        return None

    async def stream_with_tools(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield ``{"type": ...}`` events for one chat turn (may include tools).

        Event shapes:
        - ``{"type": "token", "text": str}`` — content delta from the LLM.
        - ``{"type": "tool_call", "name": str, "args": dict}`` — emitted
          **before** the corresponding tool executes (UI can show "calling…").
        - ``{"type": "tool_result", "name": str, "preview": str | None,
          "error": str | None, "latency_ms": float | None}`` — emitted after
          each tool finishes.
        - ``{"type": "sources", "citations": list[dict]}`` — emitted right
          after a successful ``buscar_bulas`` tool result so the UI can render
          source attributions before the LLM's textual answer streams in.
        - ``{"type": "done"}`` — terminal event when the LLM stops requesting
          tools and produces its final answer.
        - ``{"type": "error", "message": "repeated_tool_call"}`` — terminal
          event when the model calls the same tool with identical arguments
          twice in the same turn (loop detected).
        """
        msgs: list[BaseMessage] = list(messages)
        # Fingerprints of every (name, args) pair dispatched this turn.
        # A duplicate fingerprint means the model is stuck in a loop.
        seen_calls: set[str] = set()
        iteration = 0

        while True:
            ai_chunks: list[AIMessageChunk] = []
            llm_started = time.perf_counter()

            try:
                async for chunk in self._llm.astream(msgs):
                    text = _coerce_content_to_text(chunk.content)
                    if text:
                        yield {"type": "token", "text": text}
                    # ``astream`` is typed as yielding BaseMessage; in practice
                    # Gemini emits AIMessageChunk so chunk arithmetic works.
                    ai_chunks.append(chunk)  # type: ignore[arg-type]
            except asyncio.CancelledError:
                logger.info("llm stream cancelled by caller", extra=_logger_extra)
                raise

            llm_latency = (time.perf_counter() - llm_started) * 1000
            trace_service.add_step("llm_stream", llm_latency, iteration=iteration)
            iteration += 1

            if not ai_chunks:
                yield {"type": "done", "tokens_in": None, "tokens_out": None}
                return

            # Single-pass merge (O(N)). The naive ``reduce(+, chunks)``
            # reallocates the accumulated ``content`` on every step, which
            # is O(N²) for long streams (a 1k-token response with ~1 token
            # per chunk does ~500k extra bytes copied).
            final_ai = add_ai_message_chunks(ai_chunks[0], *ai_chunks[1:])
            msgs.append(final_ai)

            tool_calls = list(getattr(final_ai, "tool_calls", []) or [])
            if not tool_calls:
                tokens_in, tokens_out = _extract_token_counts(final_ai)
                yield {"type": "done", "tokens_in": tokens_in, "tokens_out": tokens_out}
                return

            # Detect loops: abort if any call in this batch was already
            # dispatched with identical arguments this turn.
            repeated_tool = self._detect_tool_loop(tool_calls, seen_calls)
            if repeated_tool:
                logger.warning(
                    "repeated tool call detected; aborting loop",
                    extra={**_logger_extra, "tool": repeated_tool},
                )
                yield {"type": "error", "message": "repeated_tool_call"}
                return

            for tc in tool_calls:
                yield {
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc.get("args", {}) or {},
                }

            results = await asyncio.gather(
                *(self._execute_tool(tc) for tc in tool_calls)
            )
            for tool_msg, trace in results:
                msgs.append(tool_msg)
                yield {
                    "type": "tool_result",
                    "name": trace.name,
                    "preview": trace.result_preview,
                    "error": trace.error,
                    "latency_ms": trace.latency_ms,
                }
                if trace.name == "buscar_bulas" and not trace.error:
                    citations = _citations_from_tool_message(tool_msg.content)
                    if citations:
                        yield {"type": "sources", "citations": citations}

    async def handle_turn(self, req: ChatRequest) -> AsyncIterator[str]:
        """Run a full chat turn and yield SSE-encoded frames.

        Errors raised after response headers were flushed surface as a final
        ``error`` SSE frame whose payload omits the raw exception message — the
        client only sees a ``trace_id`` while operators correlate via logs.
        """
        history = self._history.get_session_history(req.session_id)
        past = await history.aget_messages()

        user_msg = HumanMessage(content=req.message)
        messages = [_SYSTEM_MSG, *past, user_msg]

        tid = trace_service.start(req.session_id, req.message)
        yield encode_event("trace_id", {"trace_id": tid})

        text_chunks: list[str] = []
        cancelled = False
        persisted = False
        try:
            async for event in self.stream_with_tools(messages):
                event_type = event["type"]
                if event_type == "token":
                    text = event["text"]
                    text_chunks.append(text)
                    yield encode_text_event("token", text)
                elif event_type == "tool_call":
                    yield encode_event(
                        "tool_call",
                        {"name": event["name"], "args": event["args"]},
                    )
                elif event_type == "tool_result":
                    yield encode_event(
                        "tool_result",
                        {
                            "name": event["name"],
                            "preview": event.get("preview"),
                            "error": event.get("error"),
                            "latency_ms": event.get("latency_ms"),
                        },
                    )
                elif event_type == "sources":
                    trace_service.set_citations(event["citations"])
                    yield encode_event(
                        "sources", {"citations": event["citations"]}
                    )
                elif event_type == "error":
                    # LLM-loop-level error (e.g. max iterations exceeded);
                    # the message is already a stable opaque code. Persist the
                    # user turn (with whatever partial text we already
                    # streamed, plus a placeholder if none) so the next turn
                    # for this session still has continuity.
                    self._persist_failed_turn(
                        history,
                        user_msg=user_msg,
                        partial_text=text_chunks,
                        reason=event["message"],
                        session_id=req.session_id,
                    )
                    persisted = True
                    yield encode_stream_error(
                        code=event["message"],
                        message=event["message"],
                        trace_id=trace_id_var.get(),
                    )
                    return
                elif event_type == "done":
                    trace_service.set_response(
                        "".join(text_chunks),
                        tokens_in=event.get("tokens_in"),
                        tokens_out=event.get("tokens_out"),
                    )
                    break
        except asyncio.CancelledError:
            cancelled = True
            logger.info(
                "chat stream cancelled by client",
                extra={**_logger_extra, "session_id": req.session_id, "step": "cancelled"},
            )
            raise
        except Exception as exc:
            logger.exception(
                "chat stream failed",
                extra={
                    **_logger_extra,
                    "session_id": req.session_id,
                    "error_type": type(exc).__name__,
                },
            )
            self._persist_failed_turn(
                history,
                user_msg=user_msg,
                partial_text=text_chunks,
                reason="stream_failed",
                session_id=req.session_id,
            )
            persisted = True
            yield encode_stream_error(
                code="stream_failed", trace_id=trace_id_var.get()
            )
            return
        finally:
            await trace_service.finalize()
            if cancelled and text_chunks and not persisted:
                # Best-effort partial-turn persistence even on disconnect, so a
                # cancelled stream isn't lost from history.
                self._spawn_persist(
                    history,
                    [user_msg, AIMessage(content="".join(text_chunks))],
                    step="persist_partial",
                    session_id=req.session_id,
                )

        if not text_chunks:
            # The LLM produced no textual content — don't pollute history with
            # an empty AIMessage; surface a structured error instead. We still
            # persist the user message so the next turn knows what was asked.
            self._persist_failed_turn(
                history,
                user_msg=user_msg,
                partial_text=text_chunks,
                reason="empty_response",
                session_id=req.session_id,
            )
            yield encode_stream_error(
                code="empty_response",
                message="empty response",
                trace_id=trace_id_var.get(),
            )
            return

        # Persist in the background so ``done`` ships without waiting on Redis.
        # The history store tracks the task and ``drain_pending`` is awaited
        # during shutdown so no turn is dropped on a clean stop.
        assistant_msg = AIMessage(content="".join(text_chunks))
        self._spawn_persist(
            history,
            [user_msg, assistant_msg],
            step="persist",
            session_id=req.session_id,
        )
        yield encode_event("done", {"session_id": req.session_id})

    def _persist_failed_turn(
        self,
        history,
        *,
        user_msg: HumanMessage,
        partial_text: list[str],
        reason: str,
        session_id: str,
    ) -> None:
        """Persist the user turn paired with a synthetic AI marker on failure.

        The synthetic ``AIMessage`` carries any partial text the model
        produced before the failure plus a bracketed reason tag, so the LLM
        can see the prior turn ended badly and adjust on the next turn.
        """
        partial = "".join(partial_text)
        marker = f"[turn_ended_with: {reason}]"
        content = f"{partial}\n{marker}" if partial else marker
        self._spawn_persist(
            history,
            [user_msg, AIMessage(content=content)],
            step=f"persist_failed:{reason}",
            session_id=session_id,
        )

    def _spawn_persist(
        self,
        history,
        messages: list[BaseMessage],
        *,
        step: str,
        session_id: str | None = None,
    ) -> None:
        """Schedule a background write and register it with the history store.

        ``session_id`` is forwarded so the route layer can drain only this
        turn's pending writes before releasing the per-session lock, keeping
        the lock-protected critical section small while still guaranteeing
        ordering between consecutive turns.
        """

        async def _save() -> None:
            try:
                await history.aadd_messages(messages)
            except Exception:
                logger.warning("failed to persist turn", extra={**_logger_extra, "step": step})

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(_save())
        self._history.register_pending(task, session_id=session_id)


@lru_cache(maxsize=1)
def get_assistant_service() -> AssistantService:
    """Return the process-wide ``AssistantService`` (lazy, cached)."""
    return AssistantService()


def __getattr__(name: str) -> object:
    """Lazy ``assistant_service`` proxy for backward-compatible imports."""
    if name == "assistant_service":
        return get_assistant_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
