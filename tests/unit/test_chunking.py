"""Unit tests for the hybrid section-to-chunks policy."""

from __future__ import annotations

from bulas_assistant.assistant.sectionizer import Section
from bulas_assistant.services.ingestion_service import (
    MIN_SECTION_CHARS,
    SECTION_WHOLE_THRESHOLD,
    parse_filename,
    section_to_chunks,
)


def _make_section(content: str, canonical: str = "IAP_6_POSOLOGIA",
                  raw_header: str = "6. COMO DEVO USAR ESTE MEDICAMENTO?",
                  occurrence: int = 0) -> Section:
    return Section(
        canonical=canonical,
        raw_header=raw_header,
        start=0,
        end=len(content),
        content=content,
        patient_facing=True,
        occurrence=occurrence,
    )


def test_short_section_becomes_single_full_chunk() -> None:
    content = (
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n\n"
        + ("A dose habitual é de um comprimido por dia, "
           "preferencialmente após as refeições. " * 4)
    )
    assert MIN_SECTION_CHARS <= len(content) <= SECTION_WHOLE_THRESHOLD
    section = _make_section(content)

    chunks = section_to_chunks(section, "123", "Foo", None, "123")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata.is_full_section is True
    assert chunk.text == content.strip()
    assert chunk.metadata.section_char_len == len(content.strip())
    assert chunk.chunk_id == "123__IAP_6_POSOLOGIA__0__full"
    assert chunk.metadata.bula_id == "123"
    assert chunk.metadata.patient_facing is True


def test_long_section_is_subsplit_and_headers_are_prepended() -> None:
    long_content = (
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n\n"
        + ("Parágrafo cheio de detalhes sobre posologia. " * 200)
    )
    assert len(long_content) > SECTION_WHOLE_THRESHOLD
    section = _make_section(long_content)

    chunks = section_to_chunks(section, "927100", "Ritalina", None, "927100")

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.is_full_section is False
        assert chunk.text.startswith("6. COMO DEVO USAR ESTE MEDICAMENTO?")
        assert chunk.chunk_id == f"927100__IAP_6_POSOLOGIA__0__part{idx}"

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids)


def test_noisy_short_section_is_discarded() -> None:
    section = _make_section("VIA ORAL", canonical="IDENT_VIA_USO", raw_header="VIA ORAL")
    assert len(section.content) < MIN_SECTION_CHARS

    chunks = section_to_chunks(section, "1", "Foo", None, "1")
    assert chunks == []


def test_parse_filename_extracts_id_med_anvisa() -> None:
    bula_id, med_name, anvisa = parse_filename("927100_ritalina_metilfenidato")
    assert bula_id == "927100"
    assert med_name == "Ritalina Metilfenidato"
    assert anvisa == "927100"


def test_parse_filename_handles_stem_without_underscore() -> None:
    bula_id, med_name, anvisa = parse_filename("nonstandard")
    assert bula_id == "nonstandard"
    assert med_name == "nonstandard"
    assert anvisa is None
