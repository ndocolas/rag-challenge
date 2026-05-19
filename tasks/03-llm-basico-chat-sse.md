# Task 03 — LLM básico (chat SSE + Redis multi-turno)

## Objetivo

`POST /chat` funcional end-to-end: recebe mensagem, chama Gemini, devolve stream SSE
token-a-token. Histórico persistido em Redis por `session_id`. **Sem tools, sem RAG.**
Validar pipeline de streaming antes de enriquecer.

## Pré-requisitos

- Task 01 (utils, settings, sse encoder)
- Task 02 (models ChatRequest, ChatMessage, StreamEvent)

## Dependências novas

Adicionar ao `pyproject.toml`:
```toml
"langchain-core>=0.3",
"langchain-google-genai>=2.0",
"redis>=5.2",
```

## Subtarefas

### 1. `services/llm_service.py`

```python
from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


class LLMService:
    def __init__(self) -> None:
        self._client = ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            streaming=True,
        )

    async def stream_response(
        self, messages: list[BaseMessage]
    ) -> AsyncIterator[str]:
        """Stream de tokens (string deltas) da resposta do LLM."""
        async for chunk in self._client.astream(messages):
            content = chunk.content
            if content:
                yield content


llm_service = LLMService()
```

### 2. `services/chat_history_service.py`

Wrapper sobre Redis para histórico conversacional.

```python
import json

import redis.asyncio as redis

from panvel_assistant.models.chat import ChatMessage
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


class ChatHistoryService:
    def __init__(self) -> None:
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await self._redis.ping()
        logger.info("redis conectado", extra={"url": settings.REDIS_URL})

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def _key(self, session_id: str) -> str:
        return f"chat:history:{session_id}"

    async def load(self, session_id: str) -> list[ChatMessage]:
        raw = await self._redis.get(self._key(session_id))
        if not raw:
            return []
        return [ChatMessage(**m) for m in json.loads(raw)]

    async def append(self, session_id: str, messages: list[ChatMessage]) -> None:
        existing = await self.load(session_id)
        existing.extend(messages)
        # mantém só últimas 20 mensagens p/ não estourar contexto
        existing = existing[-20:]
        await self._redis.set(
            self._key(session_id),
            json.dumps([m.model_dump() for m in existing], ensure_ascii=False),
            ex=settings.CHAT_HISTORY_TTL_SECONDS,
        )


chat_history_service = ChatHistoryService()
```

### 3. `services/chat_service.py` — orquestração

```python
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from panvel_assistant.assistant.prompts import SYSTEM_PROMPT_MVP
from panvel_assistant.models.chat import ChatMessage, ChatRequest
from panvel_assistant.services.chat_history_service import chat_history_service
from panvel_assistant.services.llm_service import llm_service
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.sse import encode_event

logger = get_logger(__name__)


def _to_lc(messages: list[ChatMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT_MVP)]
    for m in messages:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


class ChatService:
    async def handle_turn(self, req: ChatRequest) -> AsyncIterator[str]:
        history = await chat_history_service.load(req.session_id)
        user_msg = ChatMessage(role="user", content=req.message)
        history_for_llm = history + [user_msg]

        full_response = []
        try:
            async for delta in llm_service.stream_response(_to_lc(history_for_llm)):
                full_response.append(delta)
                yield encode_event("token", {"text": delta})
        except Exception as e:
            logger.exception("erro no stream")
            yield encode_event("error", {"message": str(e)})
            return

        assistant_msg = ChatMessage(role="assistant", content="".join(full_response))
        await chat_history_service.append(req.session_id, [user_msg, assistant_msg])
        yield encode_event("done", {"session_id": req.session_id})


chat_service = ChatService()
```

### 4. `assistant/prompts.py`

```python
SYSTEM_PROMPT_MVP = """\
Você é o assistente conversacional da Panvel para a operação no Paraná (PR).
Responde em português brasileiro, de forma clara e concisa.

Regras:
- Só responde sobre: medicamentos (informações farmacológicas) e filiais Panvel-PR.
- Se a pergunta sai do escopo, recuse educadamente e ofereça redirecionar.
- Nunca substitui orientação médica — sempre lembre disso ao falar de medicamentos.
- Não invente informações; se não souber, diga.
"""
```

### 5. `routes/chat.py`

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from panvel_assistant.models.chat import ChatRequest
from panvel_assistant.services.chat_service import chat_service
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter()


@router.post("/chat")
@handle_errors
async def chat(req: ChatRequest):
    return StreamingResponse(
        chat_service.handle_turn(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 6. Atualizar `main.py`

- Adicionar `chat_history_service.connect()` no lifespan startup
- Adicionar `chat_history_service.disconnect()` no shutdown
- Incluir router: `app.include_router(chat_router)`

### 7. Testes

`tests/unit/test_chat_service.py`:
- mock do `llm_service.stream_response` retornando ["Olá", " mundo"]
- mock do `chat_history_service` (load=[], append=async noop)
- chama `chat_service.handle_turn(ChatRequest(...))`, coleta eventos
- valida: eventos token na ordem, evento done ao final

`tests/integration/test_chat_route.py`:
- usa `httpx.AsyncClient` contra `app`
- mocka `LLMService._client` retornando chunks fake
- usa redis local (ou fakeredis)
- POST /chat → parse stream → valida eventos

## Verificação

```bash
# subir redis local
docker run --rm -d -p 6379:6379 --name redis-dev redis:7-alpine

# rodar API
cd backend
uv run uvicorn panvel_assistant.main:app

# turno 1
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"Quem é você?"}'

# esperado: stream de eventos `event: token` com text incremental, terminando em
# `event: done`

# turno 2 — testa memória
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"O que perguntei antes?"}'

# esperado: LLM deve referenciar o "Quem é você?"

# verifica Redis
docker exec redis-dev redis-cli GET "chat:history:s1"
```

## Gotchas

- `redis.asyncio` vs `redis` (sync): usar SEMPRE asyncio em código FastAPI.
- `decode_responses=True` evita lidar com `bytes` em todo lugar.
- LangChain v0.3+: `astream` retorna `AIMessageChunk`; `chunk.content` pode ser
  string OU lista — checar tipo. Para Gemini text-only fica string.
- Cache-Control e X-Accel-Buffering nos headers SSE evitam buffering em proxies/nginx.
- `StreamingResponse` precisa de async generator — `handle_turn` é `async def` com
  `yield`.
- `@handle_errors` em endpoint que retorna `StreamingResponse`: exceptions DURANTE o
  stream NÃO são capturadas pelo decorator (já enviou headers). Por isso o try/except
  dentro de `handle_turn` que emite `event: error`.
- Histórico em Redis: limitar tamanho (últimas N mensagens) ou usar `RPUSH+LTRIM`
  para performance. MVP: 20 msgs JSON serializado está OK.
- Gemini key vazia → erro silencioso em algumas versões; validar no startup.
