"""``POST /chat`` route — token-by-token SSE stream backed by Gemini + Redis memory."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services.chat_service import chat_service
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter(tags=["chat"])


@router.post("/chat")
@handle_errors
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the assistant's response for one chat turn as SSE frames."""
    return StreamingResponse(
        chat_service.handle_turn(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
