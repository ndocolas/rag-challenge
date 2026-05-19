"""Pydantic input/output schemas for the LangChain branch (filial) tools."""

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


class ToolErrorPayload(BaseModel):
    """Structured error returned when a tool fails — the LLM reads this and recovers."""

    error: str
    message: str
    hint: dict | None = None
