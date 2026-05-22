"""Unit tests for the Anvisa sectionizer regex parser."""

from __future__ import annotations

from bulas_assistant.assistant.sectionizer import (
    detect_med_variants,
    extract_variant_names,
    sectionize,
)


def test_sections_returned_in_text_order() -> None:
    text = (
        "APRESENTAÇÕES\n"
        "Comprimidos 10mg.\n"
        "COMPOSIÇÃO\n"
        "Princípio ativo X.\n"
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\n"
        "Indicado para Y.\n"
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n"
        "Uma vez ao dia.\n"
    )

    sections = sectionize(text)
    canonicals = [s.canonical for s in sections]

    assert canonicals == [
        "IDENT_APRESENTACOES",
        "IDENT_COMPOSICAO",
        "IAP_1_INDICACOES",
        "IAP_6_POSOLOGIA",
    ]
    # Sections are contiguous and ordered by start offset.
    starts = [s.start for s in sections]
    assert starts == sorted(starts)


def test_text_without_known_headers_falls_back_to_unclassified() -> None:
    text = "Some prose without any anvisa headers. Lorem ipsum dolor sit amet."

    sections = sectionize(text)

    assert len(sections) == 1
    assert sections[0].canonical == "UNCLASSIFIED"
    assert sections[0].content == text
    assert sections[0].patient_facing is False


def test_multi_product_increments_occurrence() -> None:
    """Multi-product bulas (Ritalina IR/LA) repeat IAP_1 inside the same file."""
    text = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\n"
        "Variante A: indicação 1.\n"
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n"
        "Variante A dosage.\n"
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\n"
        "Variante B: indicação 2.\n"
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n"
        "Variante B dosage.\n"
    )

    sections = sectionize(text)
    iap1_sections = [s for s in sections if s.canonical == "IAP_1_INDICACOES"]
    iap6_sections = [s for s in sections if s.canonical == "IAP_6_POSOLOGIA"]

    assert len(iap1_sections) == 2
    assert [s.occurrence for s in iap1_sections] == [0, 1]
    assert [s.occurrence for s in iap6_sections] == [0, 1]

    variants = detect_med_variants(sections)
    assert variants[0] is None
    assert variants[1] == "variante_2"


def test_iap_and_it_collide_on_numbering_but_regex_disambiguates() -> None:
    """Both IAP and IT use ``3.`` / ``6.`` / ``9.`` — IAP keeps the trailing ``?``."""
    text = (
        "3. QUANDO NÃO DEVO USAR ESTE MEDICAMENTO?\n"
        "Contraindicação ao paciente.\n"
        "3. CARACTERÍSTICAS FARMACOLÓGICAS\n"
        "Bloco técnico farmacológico.\n"
        "6. COMO DEVO USAR ESTE MEDICAMENTO?\n"
        "Dose ao paciente.\n"
        "6. INTERAÇÕES MEDICAMENTOSAS\n"
        "Bloco técnico de interações.\n"
        "9. O QUE FAZER SE ALGUÉM USAR UMA QUANTIDADE MAIOR?\n"
        "Superdose paciente.\n"
        "9. REAÇÕES ADVERSAS\n"
        "Bloco técnico de reações.\n"
    )

    sections = sectionize(text)
    canonicals = [s.canonical for s in sections]

    assert canonicals == [
        "IAP_3_CONTRAINDICACOES",
        "IT_CARACTERISTICAS_FARMACOLOGICAS",
        "IAP_6_POSOLOGIA",
        "IT_INTERACOES_MEDICAMENTOSAS",
        "IAP_9_SUPERDOSE",
        "IT_REACOES_ADVERSAS_TECNICAS",
    ]

    iap = [s for s in sections if s.canonical.startswith("IAP_")]
    it = [s for s in sections if s.canonical.startswith("IT_")]
    assert all(s.patient_facing for s in iap)
    assert all(not s.patient_facing for s in it)


def test_dizeres_legais_full_content_preserved() -> None:
    """DIZERES LEGAIS section is no longer truncated at parse time.

    Truncation was removed so multi-product PDFs (e.g. Ritalina + Ritalina LA)
    are fully indexed. Long sections are sub-chunked by ingestion_service.
    """
    tail = "histórico regulatório " * 1000  # ~22k chars
    text = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\n"
        "Indicação real.\n"
        "DIZERES LEGAIS\n"
        f"{tail}"
    )

    sections = sectionize(text)
    dizeres = [s for s in sections if s.canonical == "DIZERES_LEGAIS"][0]

    # Full content must be present — chunking is handled downstream.
    assert len(dizeres.content) > 20_000


def test_extract_variant_names_single_product() -> None:
    """Single-product bula returns {0: None}."""
    text = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\nIndicação.\n"
        "DIZERES LEGAIS\nFabricante X.\n"
    )
    sections = sectionize(text)
    assert extract_variant_names(text, sections) == {0: None}


def test_extract_variant_names_extracts_title() -> None:
    """Product name found in preamble before second IAP_1 is used as variant label."""
    product1 = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\nIndicação produto 1.\n"
        "DIZERES LEGAIS\nFabricante.\n"
    )
    separator = "\nRITALINA LA\n"
    product2 = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\nIndicação produto 2.\n"
        "DIZERES LEGAIS\nFabricante.\n"
    )
    text = product1 + separator + product2
    sections = sectionize(text)
    variants = extract_variant_names(text, sections)
    assert variants[0] is None
    assert variants[1] == "RITALINA LA"


def test_extract_variant_names_fallback_when_no_title() -> None:
    """Falls back to 'variante_N' when no uppercase title is found in preamble."""
    product1 = (
        "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\nIndicação produto 1.\n"
        "DIZERES LEGAIS\nFabricante.\n"
    )
    # No uppercase title line before second IAP_1
    separator = "\ncontinuação da bula sem título destacado\n"
    product2 = "1. PARA QUE ESTE MEDICAMENTO É INDICADO?\nIndicação produto 2.\n"
    text = product1 + separator + product2
    sections = sectionize(text)
    variants = extract_variant_names(text, sections)
    assert variants[0] is None
    assert variants[1] == "variante_2"
