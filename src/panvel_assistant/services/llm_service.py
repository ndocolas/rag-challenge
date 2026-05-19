"""Thin async wrapper around Gemini's streaming chat completions via LangChain."""

from collections.abc import AsyncIterator, Sequence

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


def _coerce_content_to_text(content: object) -> str:
    """Reduce a chunk's ``content`` to a plain string.

    LangChain v1 chunks may carry either a ``str`` (Gemini text-only path) or a
    list of segment dicts (multimodal / tool-call chunks). For the MVP we only
    stream textual deltas, so we join string segments and drop the rest.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


class LLMService:
    """Singleton wrapper around ``ChatGoogleGenerativeAI`` for streaming chat."""

    def __init__(self) -> None:
        self._client = ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=SecretStr(settings.GOOGLE_API_KEY),
            temperature=0.2,
        )

    async def stream_response(
        self, messages: Sequence[BaseMessage]
    ) -> AsyncIterator[str]:
        """Yield textual deltas from the model, dropping empty/non-text chunks."""
        async for chunk in self._client.astream(list(messages)):
            text = _coerce_content_to_text(chunk.content)
            if text:
                yield text


llm_service = LLMService()
