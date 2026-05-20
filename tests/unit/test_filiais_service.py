"""Unit tests for FiliaisService — load, listar_cidades, detalhar, buscar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from panvel_assistant.services.filiais_service import FiliaisService
from panvel_assistant.utils.exceptions import InvalidRequestError, ResourceNotFoundError

# ---------------------------------------------------------------------------
# Fixture: a tiny parquet covering 3 cities and a mix of services / tipos.
# ---------------------------------------------------------------------------

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
    {
        "codigo_filial": "300",
        "faixa_vida": "MENOS DE 1 ANO",
        "localidade": "MARINGÁ",
        "uf": "PR",
        "tipo_estabelecimento": "BAIRRO",
        "delivery": "SIM",
        "metragem_area_venda": 150.0,
        "panvel_clinic": "NÃO",
        "estacionamento": "NÃO",
        "atendimento_24_horas": "NÃO",
    },
]


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    """Write the fixture rows to a temporary parquet and return its path."""
    path = tmp_path / "filiais.parquet"
    pd.DataFrame(_ROWS).to_parquet(path)
    return path


@pytest.fixture
def service(parquet_path: Path) -> FiliaisService:
    """Fresh FiliaisService loaded from the fixture parquet."""
    s = FiliaisService()
    s.load(parquet_path)
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_populates_totals_and_sorted_unique_cidades(service: FiliaisService):
    cidades = service.listar_cidades()
    assert len(cidades) == 3
    assert cidades == sorted(set(cidades))
    assert set(cidades) == {"CURITIBA", "LONDRINA", "MARINGÁ"}


def test_detalhar_returns_full_record(service: FiliaisService):
    f = service.detalhar("100")
    assert f.codigo_filial == "100"
    assert f.localidade == "CURITIBA"
    assert f.delivery is True
    assert f.atendimento_24_horas is True
    assert f.metragem_area_venda == 200.0


def test_detalhar_strips_input_and_coerces_to_str(service: FiliaisService):
    assert service.detalhar(" 100 ").codigo_filial == "100"


def test_detalhar_unknown_raises_resource_not_found(service: FiliaisService):
    with pytest.raises(ResourceNotFoundError):
        service.detalhar("999")


def test_buscar_no_filters_returns_all(service: FiliaisService):
    total, items = service.buscar()
    assert total == 4
    assert len(items) == 4


def test_buscar_cidade_normalizes_accents_and_case(service: FiliaisService):
    total, items = service.buscar(cidade="maringa")
    assert total == 1
    assert items[0].codigo_filial == "300"
    assert items[0].localidade == "MARINGÁ"


def test_buscar_cidade_invalida_raises_with_marker_prefix(service: FiliaisService):
    with pytest.raises(InvalidRequestError) as exc:
        service.buscar(cidade="Florianópolis")
    assert str(exc.value).startswith("cidade_nao_encontrada:")


def test_buscar_servicos_and_combination(service: FiliaisService):
    total, items = service.buscar(
        cidade="CURITIBA",
        servicos=["panvel_clinic", "atendimento_24_horas"],
    )
    assert total == 1
    assert items[0].codigo_filial == "100"


def test_buscar_filters_by_tipo_estabelecimento(service: FiliaisService):
    total, items = service.buscar(tipo_estabelecimento="SHOPPING")
    assert total == 1
    assert items[0].codigo_filial == "101"


def test_buscar_filters_by_faixa_vida(service: FiliaisService):
    total, items = service.buscar(faixa_vida="MENOS DE 1 ANO")
    assert total == 1
    assert items[0].codigo_filial == "300"


def test_buscar_min_metragem_inclusive(service: FiliaisService):
    total, items = service.buscar(min_metragem=200.0)
    assert total == 2
    codigos = {it.codigo_filial for it in items}
    assert codigos == {"100", "200"}


def test_buscar_limit_truncates_but_preserves_total(service: FiliaisService):
    total, items = service.buscar(limit=2)
    assert total == 4
    assert len(items) == 2


def test_buscar_resumo_lists_only_active_services(service: FiliaisService):
    _, items = service.buscar(cidade="LONDRINA")
    assert items[0].servicos_ativos == ["panvel_clinic", "delivery"]


# D14 — confirm that every enum literal in models/filial.py still matches the
# values actually present in the real parquet shipped with the project.
def test_real_parquet_values_satisfy_model_literals():
    """Load the production parquet and assert the Pydantic models accept every row."""
    from typing import get_args

    from panvel_assistant.models.filial import FaixaVida, TipoEstabelecimento
    from panvel_assistant.utils.settings import get_settings

    parquet = get_settings().FILIAIS_PARQUET
    if not parquet.is_file():
        pytest.skip(f"production parquet missing at {parquet}")

    svc = FiliaisService()
    svc.load(parquet)
    allowed_faixa = set(get_args(FaixaVida))
    allowed_tipo = set(get_args(TipoEstabelecimento))
    for f in svc._all:  # type: ignore[attr-defined]
        assert f.faixa_vida in allowed_faixa, f.faixa_vida
        assert f.tipo_estabelecimento in allowed_tipo, f.tipo_estabelecimento
        assert f.uf == "PR"
