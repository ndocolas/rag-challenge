"""Pydantic input/output schemas for the LangChain branch (filial) tools."""

from typing import Literal

from pydantic import BaseModel, Field

from panvel_assistant.models.filial_models import (
    FaixaVida,
    FilialCompleta,
    FilialResumo,
    ServicoFilial,
    TipoEstabelecimento,
)

SectionHint = Literal[
    "IAP_1_INDICACOES",
    "IAP_2_MECANISMO",
    "IAP_3_CONTRAINDICACOES",
    "IAP_4_PRECAUCOES_ADVERTENCIAS",
    "IAP_5_ARMAZENAMENTO",
    "IAP_6_POSOLOGIA",
    "IAP_7_ESQUECIMENTO_DOSE",
    "IAP_8_REACOES_ADVERSAS",
    "IAP_9_SUPERDOSE",
    "IT_INTERACOES_MEDICAMENTOSAS",
    "IT_REACOES_ADVERSAS_TECNICAS",
    "IT_CARACTERISTICAS_FARMACOLOGICAS",
]


class BuscarFiliaisInput(BaseModel):
    cidade: str | None = Field(
        None,
        description="City name in Paraná state (e.g. CURITIBA). Normalized "
                    "automatically to uppercase, no accents.",
    )
    servicos: list[ServicoFilial] | None = Field(
        None,
        description="Services the branch MUST have (logical AND). Valid "
                    "values: panvel_clinic, delivery, estacionamento, "
                    "atendimento_24_horas.",
    )
    tipo_estabelecimento: TipoEstabelecimento | None = None
    faixa_vida: FaixaVida | None = None
    min_metragem: float | None = Field(None, description="Minimum sales-area in m².")
    limit: int = Field(10, ge=1, le=50)


class BuscarFiliaisOutput(BaseModel):
    total_match: int
    returned: int
    filiais: list[FilialResumo]


class DetalhesFilialInput(BaseModel):
    codigo_filial: str = Field(..., description="Branch identifier code.")


class DetalhesFilialOutput(BaseModel):
    filial: FilialCompleta


class ListarCidadesOutput(BaseModel):
    cidades: list[str]
    total: int


class BuscarBulasInput(BaseModel):
    """Input schema for the agentic ``buscar_bulas`` retrieval tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pergunta/termos sobre o medicamento (em PT-BR).",
    )
    med_name: str | None = Field(
        None,
        description=(
            "Nome canônico do medicamento (ex.: 'Ritalina', 'Losartana'). "
            "Filtra match exato no payload — passe sempre que conseguir "
            "identificar o nome a partir da conversa."
        ),
    )
    med_variant: str | None = Field(
        None,
        description=(
            "Variante do medicamento quando há múltiplos produtos no mesmo PDF "
            "(ex.: 'RITALINA LA'). Obtido via listar_medicamentos_disponiveis — "
            "aparece após ' — ' no nome. Ex.: 'Ritalina Metilfenidato — RITALINA LA' "
            "→ med_name='Ritalina Metilfenidato', med_variant='RITALINA LA'."
        ),
    )
    section_hint: SectionHint | None = Field(
        None,
        description=(
            "Hint da seção Anvisa quando o intent for claro: "
            "'posologia' → IAP_6_POSOLOGIA, 'reações' → IAP_8_REACOES_ADVERSAS, "
            "'contraindicações' → IAP_3_CONTRAINDICACOES, etc."
        ),
    )
    patient_facing_only: bool = Field(
        True,
        description=(
            "True (default) restringe a seções IAP_* (linguagem ao paciente). "
            "Use False só para perguntas técnicas explícitas."
        ),
    )
    k: int = Field(
        4,
        ge=1,
        le=10,
        description="Número de chunks retornados após dedup.",
    )


class ToolErrorPayload(BaseModel):
    """Structured error returned when a tool fails — the LLM reads this and recovers."""

    error: str
    message: str
    hint: dict | None = None
