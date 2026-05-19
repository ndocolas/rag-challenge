"""Demo routes that exercise the Pydantic schemas through FastAPI.

These endpoints are smoke/fixture routes: they prove that the models validate
end-to-end and serve as targets for the ``.rest`` files. They will be replaced
by real routes in upcoming tasks (e.g. Task 03 — ``/v1/chat`` with SSE).
"""

from fastapi import APIRouter

from panvel_assistant.models.chat import ChatMessage, ChatRequest
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/v1/echo", tags=["echo"])


@router.post("/chat", response_model=ChatMessage)
@handle_errors
async def echo_chat(req: ChatRequest) -> ChatMessage:
    return ChatMessage(role="assistant", content=f"echo: {req.message}")
