"""Pydantic schemas for the PR branch (filial) domain."""

from typing import Literal

from pydantic import BaseModel, Field

ServicoFilial = Literal[
    "clinic", "delivery", "estacionamento", "atendimento_24_horas"
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
    clinic: bool
    estacionamento: bool
    atendimento_24_horas: bool
