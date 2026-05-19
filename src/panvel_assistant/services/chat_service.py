"""Chat orchestration: bridges the route layer to LLM streaming + Redis memory."""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from panvel_assistant.assistant.prompts import SYSTEM_PROMPT_MVP
from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services.chat_history_service import history_store
from panvel_assistant.services.llm_service import llm_service
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.sse import encode_event, encode_text_event

logger = get_logger(__name__)


class ChatService:
    """Runs a single chat turn end-to-end and yields SSE-encoded events."""

    async def handle_turn(self, req: ChatRequest) -> AsyncIterator[str]:
        """Stream the assistant response for one user turn.

        Emits ``token`` frames per delta, a terminal ``done`` frame on success,
        or a single ``error`` frame if the stream fails mid-flight (the
        ``@handle_errors`` decorator on the route cannot capture exceptions
        raised after response headers were already flushed).
        """
        history = history_store.get_session_history(req.session_id)
        past = await history.aget_messages()

        user_msg = HumanMessage(content=req.message)
        messages = [SystemMessage(content=SYSTEM_PROMPT_MVP), *past, user_msg]

        chunks: list[str] = []
        try:
            async for delta in llm_service.stream_response(messages):
                chunks.append(delta)
                yield encode_text_event("token", delta)
        except Exception as exc:
            logger.exception("chat stream failed", extra={"session_id": req.session_id})
            yield encode_event("error", {"message": str(exc)})
            return

        assistant_msg = AIMessage(content="".join(chunks))
        await history.aadd_messages([user_msg, assistant_msg])
        yield encode_event("done", {"session_id": req.session_id})


chat_service = ChatService()
