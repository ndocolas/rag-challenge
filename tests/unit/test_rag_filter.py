"""Unit tests for ``RAGService._build_filter``.

The method is static and uses no IO, so we can call it without instantiating
the service (avoids needing GOOGLE_API_KEY in unit-test env).
"""

from __future__ import annotations

from qdrant_client.models import Filter

from bulas_assistant.services.rag_service import RAGService


def _conditions(f: Filter) -> list[tuple[str, object]]:
    """Flatten a Filter's ``must`` list to ``[(key, value), ...]`` for asserts."""
    out: list[tuple[str, object]] = []
    for cond in f.must or []:
        # FieldCondition with MatchValue
        out.append((cond.key, cond.match.value))  # type: ignore[union-attr]
    return out


def test_no_filters_returns_none():
    assert RAGService._build_filter(None, None, False) is None


def test_only_med_name_single_must():
    f = RAGService._build_filter("Ritalina", None, False)
    assert f is not None
    assert _conditions(f) == [("med_name", "Ritalina")]


def test_only_patient_facing_single_must():
    f = RAGService._build_filter(None, None, True)
    assert f is not None
    assert _conditions(f) == [("patient_facing", True)]


def test_only_section_hint_single_must():
    f = RAGService._build_filter(None, "IAP_6_POSOLOGIA", False)
    assert f is not None
    assert _conditions(f) == [("section_canonical", "IAP_6_POSOLOGIA")]


def test_all_three_combined_three_musts():
    f = RAGService._build_filter("Ritalina", "IAP_3_CONTRAINDICACOES", True)
    assert f is not None
    conds = _conditions(f)
    assert ("med_name", "Ritalina") in conds
    assert ("section_canonical", "IAP_3_CONTRAINDICACOES") in conds
    assert ("patient_facing", True) in conds
    assert len(conds) == 3


def test_empty_med_name_is_treated_as_absent():
    # Empty string should be skipped (falsy) — avoids matching empty payloads.
    f = RAGService._build_filter("", None, False)
    assert f is None


def test_med_variant_adds_condition():
    f = RAGService._build_filter("Ritalina", None, False, med_variant="RITALINA LA")
    assert f is not None
    conds = _conditions(f)
    assert ("med_name", "Ritalina") in conds
    assert ("med_variant", "RITALINA LA") in conds
