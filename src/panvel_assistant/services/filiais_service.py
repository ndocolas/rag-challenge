"""In-memory cache + query layer over the Panvel-PR branch catalog (``filiais.parquet``).

The parquet is read once at application startup (``load()`` is invoked from the
FastAPI lifespan hook) and decoded into immutable ``FilialCompleta`` Pydantic
models. All subsequent reads are pure list comprehensions: the dataset is
~124 rows x 10 columns, so an in-memory store keeps the tool latency in the
sub-millisecond range without any I/O on the hot path.

Query behavior is intentionally narrow — the LLM-facing tools wrap this service
and translate domain exceptions into structured JSON payloads the model can
inspect and recover from.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd

from panvel_assistant.models.filial_models import (
    FaixaVida,
    FilialCompleta,
    FilialResumo,
    ServicoFilial,
    TipoEstabelecimento,
)
from panvel_assistant.utils.exceptions import InvalidRequestError, ResourceNotFoundError
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "FiliaisService", "component.version": "v1"}


@lru_cache(maxsize=256)
def _normalize(value: str) -> str:
    """Upper-case, strip whitespace, drop diacritics (NFD).

    Used so ``"Curitiba"``, ``" CURITIBA "`` and ``"CURITIBÁ"`` all collapse
    to the same lookup key when matching cities.
    """
    value = value.strip().upper()
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def _safe_str(value: object, *, field: str, codigo: str | None = None) -> str:
    """Coerce a parquet cell to ``str``, rejecting ``NaN``/``None``.

    Pandas surfaces missing values as ``float("nan")``; passing that to
    ``_normalize`` would blow up during startup with a confusing
    ``AttributeError``. Raise a clear ``ValueError`` instead so the operator
    sees which column/row needs fixing.
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN
        raise ValueError(
            f"campo {field!r} ausente"
            + (f" para filial {codigo!r}" if codigo else "")
        )
    return str(value)


def _bool(value: object, *, field: str, codigo: str | None = None) -> bool:
    """Decode the parquet's ``"SIM"``/``"NÃO"`` flags.

    Tolerant of case and surrounding whitespace; raises ``ValueError`` on
    unrecognized payloads so a typo in the source data fails loudly at load
    instead of being silently coerced to ``False``.
    """
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"campo {field!r} esperava 'SIM'/'NÃO', recebeu {type(value).__name__}"
            + (f" (filial {codigo!r})" if codigo else "")
        )
    normalized = value.strip().upper()
    if normalized == "SIM":
        return True
    if normalized in {"NÃO", "NAO", "NO", ""}:
        return False
    raise ValueError(
        f"campo {field!r} com valor inesperado {value!r}"
        + (f" (filial {codigo!r})" if codigo else "")
    )


def _row_to_completa(row: dict) -> FilialCompleta:
    """Project a raw parquet row dict into a ``FilialCompleta``."""
    codigo = _safe_str(row.get("codigo_filial"), field="codigo_filial").strip()
    return FilialCompleta(
        codigo_filial=codigo,
        faixa_vida=cast(
            FaixaVida,
            _safe_str(row.get("faixa_vida"), field="faixa_vida", codigo=codigo),
        ),
        localidade=_safe_str(row.get("localidade"), field="localidade", codigo=codigo),
        uf=_safe_str(row.get("uf"), field="uf", codigo=codigo),
        tipo_estabelecimento=cast(
            TipoEstabelecimento,
            _safe_str(
                row.get("tipo_estabelecimento"),
                field="tipo_estabelecimento",
                codigo=codigo,
            ),
        ),
        delivery=_bool(row.get("delivery"), field="delivery", codigo=codigo),
        metragem_area_venda=float(row["metragem_area_venda"]),
        panvel_clinic=_bool(
            row.get("panvel_clinic"), field="panvel_clinic", codigo=codigo
        ),
        estacionamento=_bool(
            row.get("estacionamento"), field="estacionamento", codigo=codigo
        ),
        atendimento_24_horas=_bool(
            row.get("atendimento_24_horas"),
            field="atendimento_24_horas",
            codigo=codigo,
        ),
    )


def _to_resumo(f: FilialCompleta) -> FilialResumo:
    """Compact representation returned by ``buscar`` (omits non-essential fields)."""
    servicos: list[ServicoFilial] = []
    if f.panvel_clinic:
        servicos.append("panvel_clinic")
    if f.delivery:
        servicos.append("delivery")
    if f.estacionamento:
        servicos.append("estacionamento")
    if f.atendimento_24_horas:
        servicos.append("atendimento_24_horas")
    return FilialResumo(
        codigo_filial=f.codigo_filial,
        localidade=f.localidade,
        tipo_estabelecimento=f.tipo_estabelecimento,
        servicos_ativos=servicos,
    )


class FiliaisService:
    """In-memory query layer over the branch catalog.

    Not thread-safe to *load*, but read paths are read-only and concurrency-safe
    after ``load()`` returns.
    """

    def __init__(self) -> None:
        self._loaded: bool = False
        self._by_codigo: dict[str, FilialCompleta] = {}
        self._all: list[FilialCompleta] = []
        self._cidades: list[str] = []
        # Pre-built lookup indexes so the hot path (`buscar`) avoids repeated
        # ``_normalize`` calls and full-list scans.
        self._cidades_norm: set[str] = set()
        self._por_cidade_norm: dict[str, list[FilialCompleta]] = {}

    def load(self, parquet_path: Path | None = None) -> None:
        """Read ``parquet_path`` (defaults to ``settings.FILIAIS_PARQUET``) into memory."""
        path = parquet_path or settings.FILIAIS_PARQUET
        df = pd.read_parquet(path)
        self._all = [_row_to_completa(r) for r in df.to_dict(orient="records")]
        # ``_row_to_completa`` already strips ``codigo_filial``; keep the
        # indexing key consistent so ``detalhar`` lookups can't miss because
        # of source-side whitespace.
        by_codigo: dict[str, FilialCompleta] = {}
        by_cidade: dict[str, list[FilialCompleta]] = {}
        cidades_set: set[str] = set()
        for f in self._all:
            by_codigo[f.codigo_filial] = f
            cidades_set.add(f.localidade)
            by_cidade.setdefault(_normalize(f.localidade), []).append(f)
        self._by_codigo = by_codigo
        self._cidades = sorted(cidades_set)
        self._cidades_norm = {_normalize(c) for c in self._cidades}
        self._por_cidade_norm = by_cidade
        self._loaded = True
        logger.info(
            "filiais carregadas",
            extra={**_logger_extra, "total": len(self._all), "cidades": len(self._cidades)},
        )

    def listar_cidades(self) -> list[str]:
        """All Paraná cities served by Panvel, sorted alphabetically."""
        return list(self._cidades)

    def detalhar(self, codigo_filial: str) -> FilialCompleta:
        """Full record for one branch; raises ``ResourceNotFoundError`` if unknown."""
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
        """Filter branches with AND-combined criteria.

        Returns ``(total_match, sliced_list)`` so callers can surface how many
        records matched before truncation to ``limit``. Unknown ``cidade``
        raises ``InvalidRequestError("cidade_nao_encontrada:<input>")`` — the
        tool wrapper translates that into a structured payload with the list
        of cities the LLM can fall back to.

        Performance: when ``cidade`` is provided we look up the pre-computed
        per-city bucket in O(1) instead of scanning the full list. The
        remaining predicates run in a single pass over the candidate slice
        (one allocation, instead of one list per filter).
        """
        if cidade:
            target = _normalize(cidade)
            if target not in self._cidades_norm:
                raise InvalidRequestError(f"cidade_nao_encontrada:{cidade}")
            candidates: list[FilialCompleta] = self._por_cidade_norm.get(target, [])
        else:
            candidates = self._all

        servicos_tuple = tuple(servicos) if servicos else ()

        def _matches(f: FilialCompleta) -> bool:
            if tipo_estabelecimento and f.tipo_estabelecimento != tipo_estabelecimento:
                return False
            if faixa_vida and f.faixa_vida != faixa_vida:
                return False
            if min_metragem is not None and f.metragem_area_venda < min_metragem:
                return False
            return all(getattr(f, s) for s in servicos_tuple)

        # Fast path: no extra filters → reuse the bucket directly.
        no_extra_filters = (
            not servicos_tuple
            and not tipo_estabelecimento
            and not faixa_vida
            and min_metragem is None
        )
        filtered = candidates if no_extra_filters else [f for f in candidates if _matches(f)]

        total = len(filtered)
        return total, [_to_resumo(f) for f in filtered[:limit]]


filiais_service = FiliaisService()
