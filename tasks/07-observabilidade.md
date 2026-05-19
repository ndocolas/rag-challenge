# Task 07 — Observabilidade

## Objetivo

Auditoria completa por turno: trace_id, latência por etapa, tokens consumidos, tools
chamadas, documentos recuperados, fontes citadas. Dois canais: LangSmith (visual UI
plug-and-play) + logs JSON estruturados (stdout, para shipping a qualquer backend).
Endpoint `GET /admin/traces/{trace_id}` retorna o trace estruturado.

## Pré-requisitos

- Task 06 (chat completo com RAG + tools).

## Dependências novas

Nenhuma — LangSmith ativa via env vars. Cliente Python opcional para queries
programáticas (não usado neste MVP).

## Subtarefas

### 1. LangSmith automático

Em `main.py` ou no startup do lifespan, garantir que essas vars estão setadas no
processo:

```python
import os
from panvel_assistant.utils.settings import settings

if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
```

LangChain detecta as vars e envia traces automaticamente para todas chamadas
`astream`, `embed_query`, tool invocations. **Não precisa instrumentar manualmente.**

### 2. `services/trace_service.py` — buffer interno de traces

Para o endpoint `/admin/traces/{trace_id}`, persistir um resumo estruturado por turno
em Redis (TTL maior, 1h).

```python
import json
import time
from typing import Any

import redis.asyncio as redis

from panvel_assistant.utils.logger import get_logger, trace_id_var
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


class TraceService:
    def __init__(self) -> None:
        self._redis: redis.Redis | None = None
        self._buffers: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def start(self, session_id: str, user_message: str) -> str:
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

    def add_step(self, name: str, latency_ms: float, **extra) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["steps"].append({
                "name": name, "latency_ms": latency_ms, **extra,
            })

    def add_tool_call(self, name: str, args: dict, result_preview: str | None,
                      latency_ms: float, error: str | None = None) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["tool_calls"].append({
                "name": name, "args": args,
                "result_preview": result_preview,
                "latency_ms": latency_ms, "error": error,
            })

    def set_citations(self, citations: list[dict]) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["citations"] = citations

    def set_response(self, text: str, tokens_in: int | None = None,
                     tokens_out: int | None = None) -> None:
        tid = trace_id_var.get()
        if tid in self._buffers:
            self._buffers[tid]["final_response"] = text
            self._buffers[tid]["tokens_in"] = tokens_in
            self._buffers[tid]["tokens_out"] = tokens_out

    async def finalize(self) -> None:
        tid = trace_id_var.get()
        if tid not in self._buffers:
            return
        buf = self._buffers.pop(tid)
        buf["duration_ms"] = (time.time() - buf["started_at"]) * 1000
        await self._redis.set(
            f"trace:{tid}",
            json.dumps(buf, ensure_ascii=False, default=str),
            ex=settings.TRACE_TTL_SECONDS,
        )
        logger.info(
            "trace finalizado",
            extra={"trace_id": tid, "duration_ms": buf["duration_ms"],
                   "tool_count": len(buf["tool_calls"]),
                   "citation_count": len(buf["citations"])},
        )

    async def get(self, trace_id: str) -> dict | None:
        raw = await self._redis.get(f"trace:{trace_id}")
        return json.loads(raw) if raw else None


trace_service = TraceService()
```

### 3. Instrumentar services existentes

Em pontos chave, chamar `trace_service.add_step / add_tool_call / set_citations`.

**`rag_service.retrieve`:**
```python
import time
started = time.perf_counter()
chunks = ... # lógica atual
latency = (time.perf_counter() - started) * 1000
trace_service.add_step("retrieval", latency, k=k, returned=len(chunks))
```

**`llm_service._execute_tool`:** já está calculando latency; passar trace:
```python
trace_service.add_tool_call(
    name=name, args=args,
    result_preview=str(result)[:500],
    latency_ms=latency,
)
```

**`chat_service.handle_turn`:**
- Início: `trace_service.start(session_id, message)` → retorna trace_id
- Emite primeiro evento SSE com trace_id pro cliente
- Após retrieval: `trace_service.set_citations([c.model_dump() for c in citations])`
- Fim do stream: `trace_service.set_response(full_text)`
- Sempre no finally: `await trace_service.finalize()`

### 4. Evento SSE `trace_id`

Emitir como primeiro evento do stream para o frontend exibir e permitir auditoria:

```python
yield encode_event("trace_id", {"trace_id": tid})
```

(Adicionar `"trace_id"` aos tipos válidos em `models/chat.py::StreamEventType`.)

### 5. `routes/admin.py`

```python
from fastapi import APIRouter, HTTPException

from panvel_assistant.services.trace_service import trace_service
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/traces/{trace_id}")
@handle_errors
async def get_trace(trace_id: str):
    trace = await trace_service.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"trace {trace_id} não encontrado")
    return trace
```

Incluir router em `main.py`.

### 6. Logs JSON estruturados — checklist de campos

Padronizar `extra={...}` em todos pontos chave (já era prática estabelecida na
Task 01, agora reforça):

| Local | Campos esperados |
|---|---|
| chat_service.handle_turn entry | `step=turn_start, session_id, message_len` |
| rag_service.retrieve | `step=retrieval, latency_ms, k, returned, hint?` |
| llm_service._execute_tool | `step=tool, tool=<name>, latency_ms, error?` |
| llm_service.stream_with_tools | `step=llm_stream, iteration, latency_ms` |
| chat_service finalize | `step=turn_end, duration_ms, tokens_in, tokens_out` |

### 7. Testes

`tests/unit/test_trace_service.py`:
- `start` → cria buffer com trace_id atual
- `add_step` registra
- `finalize` persiste em Redis (mock) e remove do buffer
- `get` recupera

`tests/integration/test_admin_route.py`:
- POST /chat → recebe trace_id no primeiro evento
- GET /admin/traces/{trace_id} → retorna JSON com steps, tool_calls, citations,
  final_response preenchidos

## Verificação

```bash
# faz um turno
TRACE_ID=$(curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"o1","message":"contraindicações ritalina"}' \
  | grep -m1 trace_id | sed 's/.*"trace_id":"\([^"]*\)".*/\1/')

echo "trace: $TRACE_ID"

# busca o trace
curl http://localhost:8000/admin/traces/$TRACE_ID | jq

# esperado: JSON com:
# {
#   "trace_id": "...",
#   "session_id": "o1",
#   "user_message": "contraindicações ritalina",
#   "duration_ms": 1234,
#   "steps": [{"name":"retrieval","latency_ms":120,"k":4,"returned":4}],
#   "tool_calls": [],
#   "citations": [{"med_name":"Ritalina","section_canonical":"IAP_3_..."}, ...],
#   "final_response": "...",
#   "tokens_in": ..., "tokens_out": ...
# }

# LangSmith: abrir https://smith.langchain.com → projeto panvel-assistant
# Ver árvore de spans: retrieval → tool calls → llm stream
```

## Gotchas

- Tokens in/out: Gemini retorna `usage_metadata` no AIMessage final
  (`response.usage_metadata.input_tokens`). Acessar do último chunk consolidado.
- LangSmith pode falhar silenciosamente se a API key for inválida; verificar logs
  do LangChain ao subir (`LANGCHAIN_VERBOSE=true` para debug).
- `trace_id_var` é contextvar — funciona em async, mas se você criar uma `asyncio.Task`
  separada o contexto precisa ser propagado (`contextvars.copy_context().run(...)`).
  Em tool execution via `asyncio.gather(...)`, normalmente herda contexto.
- Buffer in-memory (`self._buffers`) NÃO escala horizontalmente — se houver múltiplas
  réplicas da API, trace_id pode estar em outra. Solução produção: Redis com hash
  partial-updates. MVP fica OK pois usa só persistência final.
- Em logs JSON, evite incluir conteúdo grande (chunks completos) — só preview.
- LangSmith free tier: 5k traces/mês. Suficiente pro desafio.
