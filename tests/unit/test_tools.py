"""Unit tests for the LangChain branch tools.

Tools are constructed via :func:`build_tools` with an explicit ``FiliaisService``
backed by a tmp parquet so the tests are hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from panvel_assistant.assistant.agent_tools import build_tools
from panvel_assistant.models.tool import (
    BuscarFiliaisOutput,
    DetalhesFilialOutput,
    ListarCidadesOutput,
    ToolErrorPayload,
)
from panvel_assistant.services.filiais_service import FiliaisService

_ROWS = [
    {
        "codigo_filial": "100",
        "faixa_vida": "MAIS DE 3 ANOS",
        "localidade": "CURITIBA",
        "uf": "PR",
        "tipo_estabelecimento": "BAIRRO",
        "delivery": "SIM",
        "metragem_area_venda": 200.0,
        "panvel_clinic": "SIM",
        "estacionamento": "SIM",
        "atendimento_24_horas": "SIM",
    },
    {
        "codigo_filial": "101",
        "faixa_vida": "ENTRE 1-2 ANOS",
        "localidade": "CURITIBA",
        "uf": "PR",
        "tipo_estabelecimento": "SHOPPING",
        "delivery": "NÃO",
        "metragem_area_venda": 80.0,
        "panvel_clinic": "NÃO",
        "estacionamento": "SIM",
        "atendimento_24_horas": "NÃO",
    },
    {
        "codigo_filial": "200",
        "faixa_vida": "MAIS DE 3 ANOS",
        "localidade": "LONDRINA",
        "uf": "PR",
        "tipo_estabelecimento": "CENTRO",
        "delivery": "SIM",
        "metragem_area_venda": 500.0,
        "panvel_clinic": "SIM",
        "estacionamento": "NÃO",
        "atendimento_24_horas": "NÃO",
    },
]


@pytest.fixture
def tools(tmp_path: Path) -> dict:
    """Build the agent tools against an in-memory FiliaisService backed by tmp parquet."""
    parquet = tmp_path / "filiais.parquet"
    pd.DataFrame(_ROWS).to_parquet(parquet)

    svc = FiliaisService()
    svc.load(parquet)
    # Stub RAGService: these tests cover only the branch (filial) tools.
    class _NoopRag:
        async def retrieve(self, **_kwargs):
            return []

        async def list_medicamentos(self):
            return []

    return {t.name: t for t in build_tools(svc, _NoopRag())}  # type: ignore[arg-type]


def _invoke(tool, args: dict) -> str:
    """Helper: invoke a LangChain @tool returning its raw string output."""
    return tool.invoke(args)


def test_buscar_filiais_happy_path_returns_parseable_json(tools):
    raw = _invoke(tools["buscar_filiais"], {"cidade": "CURITIBA"})
    parsed = BuscarFiliaisOutput.model_validate_json(raw)
    assert parsed.total_match == 2
    assert parsed.returned == 2
    assert {f.codigo_filial for f in parsed.filiais} == {"100", "101"}


def test_buscar_filiais_cidade_invalida_returns_structured_error(tools):
    raw = _invoke(tools["buscar_filiais"], {"cidade": "Florianópolis"})
    payload = json.loads(raw)
    assert payload["error"] == "cidade_nao_encontrada"
    assert "Florianópolis" in payload["message"]
    assert "cidades_disponiveis" in payload["hint"]
    assert set(payload["hint"]["cidades_disponiveis"]) == {"CURITIBA", "LONDRINA"}


def test_buscar_filiais_multi_service_and_logic(tools):
    raw = _invoke(
        tools["buscar_filiais"],
        {
            "cidade": "CURITIBA",
            "servicos": ["panvel_clinic", "atendimento_24_horas"],
        },
    )
    parsed = BuscarFiliaisOutput.model_validate_json(raw)
    assert parsed.total_match == 1
    assert parsed.filiais[0].codigo_filial == "100"


def test_buscar_filiais_returns_zero_match_payload(tools):
    raw = _invoke(tools["buscar_filiais"], {"cidade": "CURITIBA", "min_metragem": 999.0})
    parsed = BuscarFiliaisOutput.model_validate_json(raw)
    assert parsed.total_match == 0
    assert parsed.filiais == []


def test_detalhes_filial_happy_path(tools):
    raw = _invoke(tools["detalhes_filial"], {"codigo_filial": "100"})
    parsed = DetalhesFilialOutput.model_validate_json(raw)
    assert parsed.filial.codigo_filial == "100"
    assert parsed.filial.atendimento_24_horas is True


def test_detalhes_filial_codigo_invalido_returns_error_payload(tools):
    raw = _invoke(tools["detalhes_filial"], {"codigo_filial": "999"})
    payload = ToolErrorPayload.model_validate_json(raw)
    assert payload.error == "codigo_invalido"
    assert "999" in payload.message
    assert payload.hint is not None and "sugestao" in payload.hint


def test_listar_cidades_atendidas_returns_full_set(tools):
    raw = _invoke(tools["listar_cidades_atendidas"], {})
    parsed = ListarCidadesOutput.model_validate_json(raw)
    assert parsed.total == 2
    assert parsed.cidades == ["CURITIBA", "LONDRINA"]
