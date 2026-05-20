"""Unit tests for ``RAGService._dedup_points``.

Verifies:
- duplicate ``chunk_id`` is always dropped;
- ``is_full_section=True`` duplicates within the same (bula, section) collapse
  to a single chunk;
- ``is_full_section=False`` sub-chunks of the same section are NOT collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from panvel_assistant.services.rag_service import RAGService


@dataclass
class _FakePoint:
    payload: dict[str, Any]
    score: float = 0.0


def _payload(
    chunk_id: str,
    *,
    bula_id: str = "1",
    section: str = "IAP_6_POSOLOGIA",
    is_full: bool = True,
    med_name: str = "Ritalina",
    text: str = "lorem ipsum",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "bula_id": bula_id,
        "med_name": med_name,
        "section_canonical": section,
        "is_full_section": is_full,
        "patient_facing": True,
        "chunk_idx": 0,
        "section_char_len": len(text),
        "text": text,
    }


def test_duplicate_chunk_id_dropped():
    points = [
        _FakePoint(_payload("a", bula_id="1"), score=0.9),
        _FakePoint(_payload("a", bula_id="1"), score=0.5),
        _FakePoint(_payload("b", bula_id="2"), score=0.4),
    ]
    out = RAGService._dedup_points(points, k=10)
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_full_section_duplicates_collapse():
    points = [
        _FakePoint(_payload("a", bula_id="1", section="IAP_6_POSOLOGIA", is_full=True)),
        _FakePoint(_payload("b", bula_id="1", section="IAP_6_POSOLOGIA", is_full=True)),
        _FakePoint(_payload("c", bula_id="2", section="IAP_6_POSOLOGIA", is_full=True)),
    ]
    out = RAGService._dedup_points(points, k=10)
    # First full-section (a) keeps; b is dropped (same bula+section); c is kept (other bula).
    assert [c.chunk_id for c in out] == ["a", "c"]


def test_sub_chunks_same_section_kept():
    points = [
        _FakePoint(_payload("a__part0", bula_id="1", section="IAP_4_PRECAUCOES_ADVERTENCIAS", is_full=False)),
        _FakePoint(_payload("a__part1", bula_id="1", section="IAP_4_PRECAUCOES_ADVERTENCIAS", is_full=False)),
        _FakePoint(_payload("a__part2", bula_id="1", section="IAP_4_PRECAUCOES_ADVERTENCIAS", is_full=False)),
    ]
    out = RAGService._dedup_points(points, k=10)
    assert [c.chunk_id for c in out] == ["a__part0", "a__part1", "a__part2"]


def test_respects_k_limit():
    points = [_FakePoint(_payload(f"c{i}", bula_id=str(i))) for i in range(10)]
    out = RAGService._dedup_points(points, k=3)
    assert len(out) == 3
    assert [c.chunk_id for c in out] == ["c0", "c1", "c2"]


def test_skips_payload_without_chunk_id():
    points = [
        _FakePoint({"text": "lixo"}),
        _FakePoint(_payload("a")),
    ]
    out = RAGService._dedup_points(points, k=10)
    assert [c.chunk_id for c in out] == ["a"]


def test_full_and_sub_in_same_section_both_kept():
    # is_full_section=True dedupes; the False ones never enter the seen set.
    points = [
        _FakePoint(_payload("full", section="IAP_8_REACOES_ADVERSAS", is_full=True)),
        _FakePoint(_payload("part0", section="IAP_8_REACOES_ADVERSAS", is_full=False)),
        _FakePoint(_payload("part1", section="IAP_8_REACOES_ADVERSAS", is_full=False)),
    ]
    out = RAGService._dedup_points(points, k=10)
    assert {c.chunk_id for c in out} == {"full", "part0", "part1"}


def test_citations_from_matches_truncates_snippet():
    matches = [
        {
            "bula_id": "1",
            "med_name": "Ritalina",
            "med_variant": None,
            "section_canonical": "IAP_3_CONTRAINDICACOES",
            "section_label": "Quando não devo usar",
            "source_page": None,
            "text": "x" * 500,
        }
    ]
    cits = RAGService.citations_from_matches(matches)
    assert len(cits) == 1
    assert len(cits[0].snippet) == 200
    assert cits[0].section_label == "Quando não devo usar"
