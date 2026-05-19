# Task 04 — Tools filiais (tool calling sobre cadastro)

## Objetivo

Enriquecer o chat com 3 tools que o LLM pode chamar para consultar `filiais.parquet`.
Implementar loop de tool calling com guard rails (max iterations, erros acionáveis,
execução paralela). Streaming reflete chamadas de tools como eventos SSE.

## Pré-requisitos

- Task 03 (chat funcional sem tools).

## Dependências novas

Adicionar:
```toml
"pandas>=2.2",
"pyarrow>=18.0",
"langchain>=0.3",  # para o decorator @tool
```

## Subtarefas

### 1. `services/filiais_service.py`

Carrega parquet 1x no startup, expõe operações puras.

```python
import unicodedata
from pathlib import Path

import pandas as pd

from panvel_assistant.models.filial import FilialCompleta, FilialResumo, ServicoFilial
from panvel_assistant.utils.exceptions import ResourceNotFoundError
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


def _normalize(s: str) -> str:
    """Upper, strip, remove acentos."""
    s = s.strip().upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _bool(v: str) -> bool:
    return v == "SIM"


def _row_to_completa(row: dict) -> FilialCompleta:
    return FilialCompleta(
        codigo_filial=str(row["codigo_filial"]),
        faixa_vida=row["faixa_vida"],
        localidade=row["localidade"],
        uf=row["uf"],
        tipo_estabelecimento=row["tipo_estabelecimento"],
        delivery=_bool(row["delivery"]),
        metragem_area_venda=float(row["metragem_area_venda"]),
        panvel_clinic=_bool(row["panvel_clinic"]),
        estacionamento=_bool(row["estacionamento"]),
        atendimento_24_horas=_bool(row["atendimento_24_horas"]),
    )


def _to_resumo(f: FilialCompleta) -> FilialResumo:
    servicos: list[ServicoFilial] = []
    if f.panvel_clinic: servicos.append("panvel_clinic")
    if f.delivery: servicos.append("delivery")
    if f.estacionamento: servicos.append("estacionamento")
    if f.atendimento_24_horas: servicos.append("atendimento_24_horas")
    return FilialResumo(
        codigo_filial=f.codigo_filial,
        localidade=f.localidade,
        tipo_estabelecimento=f.tipo_estabelecimento,
        servicos_ativos=servicos,
    )


class FiliaisService:
    def __init__(self) -> None:
        self._loaded = False
        self._by_codigo: dict[str, FilialCompleta] = {}
        self._all: list[FilialCompleta] = []
        self._cidades: list[str] = []

    def load(self, parquet_path: Path | None = None) -> None:
        path = parquet_path or settings.FILIAIS_PARQUET
        df = pd.read_parquet(path)
        self._all = [_row_to_completa(r) for r in df.to_dict(orient="records")]
        self._by_codigo = {f.codigo_filial: f for f in self._all}
        self._cidades = sorted({f.localidade for f in self._all})
        self._loaded = True
        logger.info(
            "filiais carregadas",
            extra={"total": len(self._all), "cidades": len(self._cidades)},
        )

    def listar_cidades(self) -> list[str]:
        return list(self._cidades)

    def detalhar(self, codigo_filial: str) -> FilialCompleta:
        f = self._by_codigo.get(str(codigo_filial).strip())
        if not f:
            raise ResourceNotFoundError(f"código {codigo_filial} não encontrado")
        return f

    def buscar(
        self,
        cidade: str | None = None,
        servicos: list[ServicoFilial] | None = None,
        tipo_estabelecimento: str | None = None,
        faixa_vida: str | None = None,
        min_metragem: float | None = None,
        limit: int = 10,
    ) -> tuple[int, list[FilialResumo]]:
        result = self._all
        if cidade:
            target = _normalize(cidade)
            if target not in (_normalize(c) for c in self._cidades):
                from panvel_assistant.utils.exceptions import InvalidRequestError
                raise InvalidRequestError(
                    f"cidade_nao_encontrada:{cidade}"
                )  # tool wrapper trata e devolve payload acionável
            result = [f for f in result if _normalize(f.localidade) == target]
        if servicos:
            for s in servicos:
                result = [f for f in result if getattr(f, s)]
        if tipo_estabelecimento:
            result = [f for f in result if f.tipo_estabelecimento == tipo_estabelecimento]
        if faixa_vida:
            result = [f for f in result if f.faixa_vida == faixa_vida]
        if min_metragem is not None:
            result = [f for f in result if f.metragem_area_venda >= min_metragem]

        total = len(result)
        return total, [_to_resumo(f) for f in result[:limit]]


filiais_service = FiliaisService()
```

Atualizar `main.py` lifespan: `filiais_service.load()` no startup.

### 2. Tools LangChain

`assistant/tools/buscar_filiais.py`:

```python
import json
import time

from langchain_core.tools import tool

from panvel_assistant.models.tool import (
    BuscarFiliaisInput,
    BuscarFiliaisOutput,
    ToolErrorPayload,
)
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.utils.exceptions import InvalidRequestError
from panvel_assistant.utils.logger import get_logger

logger = get_logger(__name__)


@tool("buscar_filiais", args_schema=BuscarFiliaisInput)
def buscar_filiais(
    cidade: str | None = None,
    servicos: list[str] | None = None,
    tipo_estabelecimento: str | None = None,
    faixa_vida: str | None = None,
    min_metragem: float | None = None,
    limit: int = 10,
) -> str:
    """Busca filiais Panvel no Paraná aplicando filtros opcionais.

    Use esta tool quando o usuário quiser encontrar lojas que atendam critérios
    como: cidade específica, serviços (panvel_clinic / delivery / estacionamento /
    atendimento_24_horas), tipo (BAIRRO / CENTRO / SHOPPING / MALL / SUPERMERCADO),
    faixa de operação, ou metragem mínima.

    Retorna lista resumida. Para detalhes completos de uma filial específica, use
    detalhes_filial(codigo_filial).
    """
    started = time.perf_counter()
    try:
        total, filiais = filiais_service.buscar(
            cidade=cidade,
            servicos=servicos,
            tipo_estabelecimento=tipo_estabelecimento,
            faixa_vida=faixa_vida,
            min_metragem=min_metragem,
            limit=limit,
        )
        out = BuscarFiliaisOutput(
            total_match=total, returned=len(filiais), filiais=filiais
        )
        return out.model_dump_json()
    except InvalidRequestError as e:
        if str(e).startswith("cidade_nao_encontrada:"):
            err = ToolErrorPayload(
                error="cidade_nao_encontrada",
                message=f"'{cidade}' não está no Paraná atendido pela Panvel.",
                hint={"cidades_disponiveis": filiais_service.listar_cidades()},
            )
            return err.model_dump_json()
        raise
    finally:
        logger.info(
            "tool buscar_filiais",
            extra={
                "step": "tool",
                "tool": "buscar_filiais",
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
        )
```

`assistant/tools/detalhes_filial.py`:

```python
import time

from langchain_core.tools import tool

from panvel_assistant.models.tool import (
    DetalhesFilialInput,
    DetalhesFilialOutput,
    ToolErrorPayload,
)
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.utils.exceptions import ResourceNotFoundError
from panvel_assistant.utils.logger import get_logger

logger = get_logger(__name__)


@tool("detalhes_filial", args_schema=DetalhesFilialInput)
def detalhes_filial(codigo_filial: str) -> str:
    """Retorna o cadastro completo de uma filial específica pelo código.

    Use depois que o usuário identificar uma filial (via buscar_filiais) e quiser
    detalhes adicionais (metragem, todos os serviços, faixa de vida, etc.).
    """
    started = time.perf_counter()
    try:
        f = filiais_service.detalhar(codigo_filial)
        return DetalhesFilialOutput(filial=f).model_dump_json()
    except ResourceNotFoundError:
        return ToolErrorPayload(
            error="codigo_invalido",
            message=f"Filial '{codigo_filial}' não existe no cadastro.",
            hint={"sugestao": "use buscar_filiais para listar códigos válidos"},
        ).model_dump_json()
    finally:
        logger.info(
            "tool detalhes_filial",
            extra={"step": "tool", "latency_ms": (time.perf_counter() - started) * 1000},
        )
```

`assistant/tools/listar_cidades.py`:

```python
import time

from langchain_core.tools import tool

from panvel_assistant.models.tool import ListarCidadesOutput
from panvel_assistant.services.filiais_service import filiais_service
from panvel_assistant.utils.logger import get_logger

logger = get_logger(__name__)


@tool("listar_cidades_atendidas")
def listar_cidades_atendidas() -> str:
    """Lista todas as cidades do Paraná onde existem filiais Panvel.

    Use ANTES de buscar_filiais quando o usuário mencionar uma cidade que pode
    não estar coberta — esta tool confirma o escopo.
    """
    started = time.perf_counter()
    cidades = filiais_service.listar_cidades()
    logger.info(
        "tool listar_cidades",
        extra={"step": "tool", "latency_ms": (time.perf_counter() - started) * 1000},
    )
    return ListarCidadesOutput(cidades=cidades, total=len(cidades)).model_dump_json()
```

Registrar todas em `assistant/tools/__init__.py`:

```python
from panvel_assistant.assistant.tools.buscar_filiais import buscar_filiais
from panvel_assistant.assistant.tools.detalhes_filial import detalhes_filial
from panvel_assistant.assistant.tools.listar_cidades import listar_cidades_atendidas

ALL_TOOLS = [buscar_filiais, detalhes_filial, listar_cidades_atendidas]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
```

### 3. Atualizar `llm_service.py` com tool loop

```python
import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from panvel_assistant.assistant.tools import ALL_TOOLS, TOOLS_BY_NAME
from panvel_assistant.models.chat import ToolCallTrace
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = 4


class LLMService:
    def __init__(self) -> None:
        self._client = ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            streaming=True,
        ).bind_tools(ALL_TOOLS)

    async def _execute_tool(self, call: dict[str, Any]) -> tuple[ToolMessage, ToolCallTrace]:
        name = call["name"]
        args = call.get("args", {}) or {}
        tool = TOOLS_BY_NAME.get(name)
        started = time.perf_counter()
        if not tool:
            content = json.dumps({"error": "unknown_tool", "name": name})
            latency = (time.perf_counter() - started) * 1000
            return (
                ToolMessage(content=content, tool_call_id=call["id"], name=name),
                ToolCallTrace(name=name, args=args, error="unknown_tool", latency_ms=latency),
            )
        try:
            result = await asyncio.to_thread(tool.invoke, args)
            latency = (time.perf_counter() - started) * 1000
            return (
                ToolMessage(content=str(result), tool_call_id=call["id"], name=name),
                ToolCallTrace(
                    name=name, args=args,
                    result_preview=str(result)[:500],
                    latency_ms=latency,
                ),
            )
        except Exception as e:
            latency = (time.perf_counter() - started) * 1000
            err = json.dumps({"error": "tool_execution_failed", "message": str(e)})
            return (
                ToolMessage(content=err, tool_call_id=call["id"], name=name),
                ToolCallTrace(name=name, args=args, error=str(e), latency_ms=latency),
            )

    async def stream_with_tools(
        self, messages: list[BaseMessage]
    ) -> AsyncIterator[dict]:
        """Yield eventos: {"type":"tool_call"|"tool_result"|"token"|"done", ...}"""
        msgs = list(messages)

        for iteration in range(MAX_TOOL_ITERATIONS):
            ai_chunks: list[AIMessage] = []
            collected_text = []
            collected_tool_calls = []

            async for chunk in self._client.astream(msgs):
                if chunk.tool_call_chunks:
                    # Gemini emite tool_call_chunks ao longo do stream; acumular
                    pass
                if chunk.content:
                    collected_text.append(chunk.content)
                    yield {"type": "token", "text": chunk.content}
                ai_chunks.append(chunk)

            # consolida AIMessage final
            final_ai = ai_chunks[0]
            for c in ai_chunks[1:]:
                final_ai = final_ai + c
            msgs.append(final_ai)

            if not final_ai.tool_calls:
                yield {"type": "done"}
                return

            # executa tools em paralelo
            for tc in final_ai.tool_calls:
                yield {"type": "tool_call", "name": tc["name"], "args": tc.get("args", {})}

            results = await asyncio.gather(
                *(self._execute_tool(tc) for tc in final_ai.tool_calls)
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

        # excedeu iterações
        logger.warning("max tool iterations atingido")
        yield {"type": "error", "message": "max_tool_iterations_exceeded"}


llm_service = LLMService()
```

### 4. Atualizar `chat_service.py`

Trocar `llm_service.stream_response` por `llm_service.stream_with_tools`. Mapear cada
evento interno para um `encode_event` SSE apropriado (`token`, `tool_call`,
`tool_result`, `done`, `error`).

### 5. Atualizar `prompts.py`

Adicionar ao system prompt instruções de uso de tools:

```python
SYSTEM_PROMPT_MVP = """\
... (texto anterior) ...

Você tem 3 ferramentas para consultar filiais Panvel-PR:
- listar_cidades_atendidas(): use para confirmar quais cidades atendemos
- buscar_filiais(cidade?, servicos?, tipo?, faixa_vida?, min_metragem?, limit?):
  busca filiais por critérios. Combina filtros com AND.
- detalhes_filial(codigo_filial): cadastro completo de uma filial.

Quando o usuário perguntar sobre filiais, USE as tools — não invente.
Se uma tool retornar erro (campo "error" no JSON), explique ao usuário e ofereça
alternativas com base no campo "hint".
"""
```

### 6. Testes

`tests/unit/test_tools.py`:
- Carrega FiliaisService com parquet de teste (fixture)
- `test_buscar_filiais_cidade_valida` → retorna lista
- `test_buscar_filiais_cidade_invalida` → retorna payload error com cidades_disponiveis
- `test_buscar_filiais_multiplos_servicos` → AND lógico
- `test_detalhes_filial_codigo_invalido` → ToolErrorPayload
- `test_listar_cidades_zero_arg` → 28 cidades

`tests/unit/test_llm_tool_loop.py`:
- Mocka `_client.astream` retornando sequência: tool_call → tool_call → response
- Valida: tools executadas, ToolMessage injetadas, evento done no fim
- Mocka loop infinito (sempre tool_call) → para em 4 iterações
- Mocka tool que levanta → evento tool_result com error preenchido

## Verificação

```bash
# turno com filial
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"f1","message":"quais lojas 24h em Curitiba com Panvel Clinic?"}'

# esperado:
# event: tool_call    {name: "buscar_filiais", args: {cidade:"CURITIBA", servicos:[...]}}
# event: tool_result  {name: "buscar_filiais", preview: "..."}
# event: token        ... (resposta natural citando filial 1557)
# event: done

# fora de escopo
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"f2","message":"tem em Florianópolis?"}'

# esperado: tool retorna error cidade_nao_encontrada → LLM responde "só PR..."
```

## Gotchas

- LangChain `astream` com tool_calls: chunks vêm fragmentados; precisa concatenar
  via `chunk + chunk` (AIMessageChunk suporta `__add__`).
- `tool.invoke(args)` é sync; usar `asyncio.to_thread` para não bloquear loop.
- Gemini tool_calls: schema deve ser Pydantic v2; `args_schema=` no `@tool`.
- `bind_tools` retorna NOVO objeto; reusar o mesmo cliente bindado.
- Erros de tool NUNCA propagam exception — sempre retornam JSON com `error` field
  pra LLM se recuperar.
- `MAX_TOOL_ITERATIONS=4` é defensivo; ajustar se LLM precisar de mais saltos legítimos.
- Eventos `tool_call` devem ser emitidos ANTES da execução (UI mostra "calling...").
- Filiais carregadas in-memory: 124 rows × 10 cols = nada, sem problema de memória.
