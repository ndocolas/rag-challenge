# Task 02 — Modelos Pydantic

## Objetivo

Centralizar todos os schemas de domínio em `models/`. Sem lógica, só dados. Servirão
para validação de request/response, contratos das tools e tipagem interna.

## Pré-requisitos

- Task 01 concluída (estrutura, utils, settings).

## Subtarefas

### 1. `models/chat.py`

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant", "system", "tool"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None  # quando role=assistant + tool_use
    tool_call_id: str | None = None                  # quando role=tool
    name: str | None = None                          # nome da tool


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class Citation(BaseModel):
    bula_id: str            # ex: "927100"
    med_name: str           # ex: "Ritalina"
    med_variant: str | None = None  # ex: "Ritalina LA"
    section_canonical: str  # ex: "IAP_3_CONTRAINDICACOES"
    section_label: str      # ex: "Quando não devo usar este medicamento?"
    source_page: int | None = None
    snippet: str            # primeiras 200 chars do chunk


class ToolCallTrace(BaseModel):
    name: str
    args: dict[str, Any]
    result_preview: str | None = None    # primeiros 500 chars do JSON resultado
    latency_ms: float | None = None
    error: str | None = None


StreamEventType = Literal[
    "token", "tool_call", "tool_result", "sources", "done", "error"
]


class StreamEvent(BaseModel):
    event_type: StreamEventType
    payload: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 2. `models/filial.py`

```python
from typing import Literal

from pydantic import BaseModel, Field

ServicoFilial = Literal[
    "panvel_clinic", "delivery", "estacionamento", "atendimento_24_horas"
]
TipoEstabelecimento = Literal["BAIRRO", "CENTRO", "SHOPPING", "MALL", "SUPERMERCADO"]
FaixaVida = Literal[
    "MENOS DE 1 ANO", "ENTRE 1-2 ANOS", "ENTRE 2-3 ANOS", "MAIS DE 3 ANOS"
]


class FilialResumo(BaseModel):
    codigo_filial: str
    localidade: str
    tipo_estabelecimento: TipoEstabelecimento
    servicos_ativos: list[ServicoFilial] = Field(default_factory=list)


class FilialCompleta(BaseModel):
    codigo_filial: str
    faixa_vida: FaixaVida
    localidade: str
    uf: str = "PR"
    tipo_estabelecimento: TipoEstabelecimento
    delivery: bool
    metragem_area_venda: float
    panvel_clinic: bool
    estacionamento: bool
    atendimento_24_horas: bool
```

### 3. `models/tool.py`

Schemas para input/output das 3 tools — usados pelos decoradores `@tool` LangChain
na Task 04.

```python
from pydantic import BaseModel, Field

from panvel_assistant.models.filial import (
    FaixaVida,
    FilialCompleta,
    FilialResumo,
    ServicoFilial,
    TipoEstabelecimento,
)


class BuscarFiliaisInput(BaseModel):
    cidade: str | None = Field(
        None,
        description="Nome da cidade no Paraná (ex: CURITIBA). Normalização "
                    "automática para upper sem acento.",
    )
    servicos: list[ServicoFilial] | None = Field(
        None,
        description="Lista de serviços que a filial DEVE ter (AND lógico). "
                    "Valores válidos: panvel_clinic, delivery, estacionamento, "
                    "atendimento_24_horas.",
    )
    tipo_estabelecimento: TipoEstabelecimento | None = None
    faixa_vida: FaixaVida | None = None
    min_metragem: float | None = Field(None, description="Metragem mínima em m².")
    limit: int = Field(10, ge=1, le=50)


class BuscarFiliaisOutput(BaseModel):
    total_match: int
    returned: int
    filiais: list[FilialResumo]


class DetalhesFilialInput(BaseModel):
    codigo_filial: str = Field(..., description="Código identificador da filial.")


class DetalhesFilialOutput(BaseModel):
    filial: FilialCompleta


class ListarCidadesOutput(BaseModel):
    cidades: list[str]
    total: int


class ToolErrorPayload(BaseModel):
    """Erro estruturado retornado quando tool falha — LLM lê e se recupera."""

    error: str         # código curto: "cidade_nao_encontrada", "codigo_invalido"
    message: str       # explicação humana
    hint: dict | None = None  # info adicional (ex: cidades_disponiveis)
```

### 4. `models/bula.py`

```python
from typing import Literal

from pydantic import BaseModel, Field

SectionCanonical = Literal[
    # Identificação (pré-bloco)
    "IDENT_APRESENTACOES",
    "IDENT_COMPOSICAO",
    "IDENT_VIA_USO",
    # Informações ao Paciente (RDC 47/2009 — 9 perguntas)
    "IAP_1_INDICACOES",
    "IAP_2_MECANISMO",
    "IAP_3_CONTRAINDICACOES",
    "IAP_4_PRECAUCOES_ADVERTENCIAS",
    "IAP_5_ARMAZENAMENTO",
    "IAP_6_POSOLOGIA",
    "IAP_7_ESQUECIMENTO_DOSE",
    "IAP_8_REACOES_ADVERSAS",
    "IAP_9_SUPERDOSE",
    # Informações Técnicas (profissionais)
    "IT_CARACTERISTICAS_FARMACOLOGICAS",
    "IT_INTERACOES_MEDICAMENTOSAS",
    "IT_REACOES_ADVERSAS_TECNICAS",
    # Outros
    "DIZERES_LEGAIS",
    "UNCLASSIFIED",
]


class BulaMetadata(BaseModel):
    bula_id: str              # ex: "927100"
    med_name: str             # ex: "Ritalina"
    anvisa_code: str | None = None
    med_variant: str | None = None  # ex: "Ritalina LA" (multi-produto)
    section_canonical: SectionCanonical
    section_raw_header: str | None = None  # header original detectado
    chunk_idx: int            # ordem dentro da seção
    source_page: int | None = None
    patient_facing: bool      # True para IAP_*, False para IT_*


class BulaChunk(BaseModel):
    chunk_id: str             # ex: "927100__IAP_6_POSOLOGIA__0"
    text: str
    metadata: BulaMetadata
    score: float | None = None  # populado pelo retriever
```

### 5. Testes unitários

`tests/unit/test_models.py`:

- `test_chat_message_serialization` — round-trip JSON
- `test_chat_request_validation` — vazio, muito longo, session_id missing → erro
- `test_citation_required_fields`
- `test_stream_event_discriminator` — cria de cada tipo
- `test_filial_servico_enum` — valor inválido falha
- `test_tool_input_buscar_filiais_optional_fields` — todos opcionais
- `test_tool_error_payload_structure`
- `test_bula_metadata_section_canonical_literal` — valor inválido falha
- `test_bula_chunk_id_format`

## Verificação

```bash
cd backend
uv run pytest tests/unit/test_models.py -v   # todos passam
uv run python -c "from panvel_assistant.models.chat import ChatMessage; \
    print(ChatMessage(role='user', content='oi').model_dump_json())"
```

## Gotchas

- `Literal` em pydantic v2 valida pelo membro; usar tuple/list não funciona — apenas
  `Literal["a", "b"]`.
- `Field(default_factory=list)` para listas mutáveis (não `default=[]`).
- `datetime.utcnow` está deprecado em Python 3.12; trocar para
  `datetime.now(timezone.utc)` se ruff reclamar.
- Discriminated unions: para `StreamEvent`, se quiser tipagem forte por `event_type`,
  usar `Annotated[Union[...], Field(discriminator="event_type")]`. MVP: `Any` no payload
  está OK.
- `BulaChunk.chunk_id` formato sugerido: `{bula_id}__{section_canonical}__{chunk_idx}` —
  facilita debug e dedup.
