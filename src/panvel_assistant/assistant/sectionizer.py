"""Parser of Anvisa drug-leaflet (bula) sections.

Recognises the canonical section headers defined by RDC 47/2009 and splits a
raw bula text into ordered :class:`Section` slices. The parser is purely
regex-based: anchors require a header to be alone on its line (``re.M``) so
table-of-contents lines and inline mentions do not produce false positives.

Outputs are consumed by :mod:`panvel_assistant.services.ingestion_service`
which then applies the hybrid chunking policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from panvel_assistant.models.bula_models import SectionCanonical


@dataclass(frozen=True)
class HeaderPattern:
    canonical: SectionCanonical
    patterns: tuple[re.Pattern[str], ...]
    patient_facing: bool


# Each entry: (canonical, regex aliases, patient_facing flag).
# IAP regex requires a trailing "?" because Anvisa templates IAP as questions;
# IT uses declarative headers with the same numbering ("3.", "6.", "9.") and
# would otherwise collide.
HEADERS: tuple[HeaderPattern, ...] = (
    HeaderPattern(
        "IDENT_APRESENTACOES",
        (re.compile(r"^\s*APRESENTA[ÇC][ÕO]ES\s*$", re.I | re.M),),
        False,
    ),
    HeaderPattern(
        "IDENT_COMPOSICAO",
        (re.compile(r"^\s*COMPOSI[ÇC][ÃA]O\s*$", re.I | re.M),),
        False,
    ),
    HeaderPattern(
        "IDENT_VIA_USO",
        (
            re.compile(
                r"^\s*(USO|VIA)\s+(ORAL|ADULTO|PEDI[ÁA]TRICO|INTRAVENOSO|"
                r"INTRAMUSCULAR|T[ÓO]PICO|SUBLINGUAL)\b.*$",
                re.I | re.M,
            ),
        ),
        False,
    ),
    HeaderPattern(
        "IAP_1_INDICACOES",
        (
            re.compile(
                r"^\s*1\.\s*PARA QUE ESTE MEDICAMENTO [ÉE] INDICADO\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_2_MECANISMO",
        (
            re.compile(
                r"^\s*2\.\s*COMO ESTE MEDICAMENTO FUNCIONA\?\s*$", re.I | re.M
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_3_CONTRAINDICACOES",
        (
            re.compile(
                r"^\s*3\.\s*QUANDO N[ÃA]O DEVO USAR ESTE MEDICAMENTO\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_4_PRECAUCOES_ADVERTENCIAS",
        (
            re.compile(
                r"^\s*4\.\s*O QUE DEVO SABER ANTES DE USAR ESTE MEDICAMENTO\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_5_ARMAZENAMENTO",
        (
            re.compile(
                r"^\s*5\.\s*ONDE.{0,5}COMO E POR QUANTO TEMPO POSSO GUARDAR.*?\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_6_POSOLOGIA",
        (
            re.compile(
                r"^\s*6\.\s*COMO DEVO USAR ESTE MEDICAMENTO\?\s*$", re.I | re.M
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_7_ESQUECIMENTO_DOSE",
        (
            re.compile(
                r"^\s*7\.\s*O QUE DEVO FAZER QUANDO EU ME ESQUECER.*?\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_8_REACOES_ADVERSAS",
        (
            re.compile(
                r"^\s*8\.\s*QUAIS OS MALES QUE ESTE MEDICAMENTO PODE.*?\?\s*$",
                re.I | re.M,
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IAP_9_SUPERDOSE",
        (
            re.compile(
                r"^\s*9\.\s*O QUE FAZER SE ALGU[ÉE]M USAR.*?\?\s*$", re.I | re.M
            ),
        ),
        True,
    ),
    HeaderPattern(
        "IT_CARACTERISTICAS_FARMACOLOGICAS",
        (
            re.compile(
                r"^\s*3\.\s*CARACTER[ÍI]STICAS FARMACOL[ÓO]GICAS\s*$",
                re.I | re.M,
            ),
        ),
        False,
    ),
    HeaderPattern(
        "IT_INTERACOES_MEDICAMENTOSAS",
        (
            re.compile(
                r"^\s*6\.\s*INTERA[ÇC][ÕO]ES MEDICAMENTOSAS\s*$", re.I | re.M
            ),
        ),
        False,
    ),
    HeaderPattern(
        "IT_REACOES_ADVERSAS_TECNICAS",
        (re.compile(r"^\s*9\.\s*REA[ÇC][ÕO]ES ADVERSAS\s*$", re.I | re.M),),
        False,
    ),
    HeaderPattern(
        "DIZERES_LEGAIS",
        (re.compile(r"^\s*DIZERES LEGAIS\s*$", re.I | re.M),),
        False,
    ),
)

_DIZERES_PATTERN = re.compile(r"^\s*DIZERES LEGAIS\s*$", re.I | re.M)


@dataclass
class Section:
    """A contiguous slice of bula text bound to a canonical Anvisa header."""

    canonical: SectionCanonical
    raw_header: str
    start: int
    end: int
    content: str
    patient_facing: bool
    occurrence: int  # 0 = first occurrence; >0 = subsequent (multi-product bulas)


def _truncate_after_dizeres(text: str) -> str:
    """Cut text after the first ``DIZERES LEGAIS`` block.

    Regulatory boilerplate that follows ``DIZERES LEGAIS`` (revision history,
    listings of approvals) consumes a large fraction of the longest bulas
    (Ritalina, Spidufen) without adding retrieval value. We keep the
    ``DIZERES LEGAIS`` header itself so the sectionizer still emits the section
    (small, contains manufacturer info), but cap it at ~2k chars to avoid the
    long historical tails.
    """
    match = _DIZERES_PATTERN.search(text)
    if not match:
        return text
    end = min(len(text), match.end() + 2000)
    return text[:end]


def sectionize(text: str) -> list[Section]:
    """Split ``text`` into canonical Anvisa sections.

    Each section runs from its header match up to the start of the next
    header (or end of text). When no headers are found, returns a single
    ``UNCLASSIFIED`` section spanning the entire text.

    The full text is processed without early truncation so multi-product PDFs
    (e.g. Ritalina + Ritalina LA in the same file) are indexed completely.
    Per-section caps are applied downstream in the chunking stage.
    """

    matches: list[tuple[int, HeaderPattern, str]] = []
    for hp in HEADERS:
        for pat in hp.patterns:
            for m in pat.finditer(text):
                matches.append((m.start(), hp, m.group(0).strip()))

    matches.sort(key=lambda x: x[0])
    if not matches:
        return [
            Section(
                canonical="UNCLASSIFIED",
                raw_header="",
                start=0,
                end=len(text),
                content=text,
                patient_facing=False,
                occurrence=0,
            )
        ]

    sections: list[Section] = []
    occurrence_counter: dict[str, int] = {}
    for i, (pos, hp, raw) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        occ = occurrence_counter.get(hp.canonical, 0)
        occurrence_counter[hp.canonical] = occ + 1
        sections.append(
            Section(
                canonical=hp.canonical,
                raw_header=raw,
                start=pos,
                end=end,
                content=text[pos:end].strip(),
                patient_facing=hp.patient_facing,
                occurrence=occ,
            )
        )
    return sections


def detect_med_variants(sections: list[Section]) -> dict[int, str | None]:
    """Map ``occurrence`` index to a synthetic variant label.

    Heuristic: when ``IAP_1_INDICACOES`` appears more than once the bula
    covers multiple presentations (e.g. Ritalina IR vs Ritalina LA). The first
    occurrence stays ``None`` (default variant); subsequent ones get a
    ``variante_N`` label so downstream payloads can disambiguate.

    Prefer :func:`extract_variant_names` when the raw PDF text is available —
    it returns human-readable names (e.g. "RITALINA LA") instead of synthetic
    labels.
    """
    last_occ = next(
        (s.occurrence for s in reversed(sections) if s.canonical == "IAP_1_INDICACOES"),
        -1,
    )
    iap1_count = last_occ + 1
    if iap1_count <= 1:
        return {0: None}
    return {i: (f"variante_{i + 1}" if i > 0 else None) for i in range(iap1_count)}


# Matches a short all-caps line — typical product name in Brazilian bulas.
_PRODUCT_NAME_RE = re.compile(
    r"^\s*([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s\d®\-\.]{1,49})\s*$",
    re.M,
)

# Known Anvisa section header words to exclude from variant name detection.
_KNOWN_HEADER_WORDS = frozenset({
    "DIZERES LEGAIS", "APRESENTAÇÕES", "APRESENTACOES", "COMPOSIÇÃO",
    "COMPOSICAO", "INFORMAÇÕES", "INFORMACOES",
})


def extract_variant_names(text: str, sections: list[Section]) -> dict[int, str | None]:
    """Return meaningful variant names extracted from the raw PDF text.

    For each occurrence > 0, scans the 600 chars before the second (or later)
    ``IAP_1_INDICACOES`` section for a capitalised product-name line. Takes the
    LAST such match (closest to the new section) and skips known Anvisa headers.
    Falls back to ``variante_N`` when the pattern is not found.
    """
    iap1_sections = [s for s in sections if s.canonical == "IAP_1_INDICACOES"]
    if len(iap1_sections) <= 1:
        return {0: None}

    result: dict[int, str | None] = {0: None}
    for sec in iap1_sections[1:]:
        preamble_start = max(0, sec.start - 600)
        preamble = text[preamble_start : sec.start]
        # Collect all matches and take the last one, skipping known headers.
        candidates = [
            m.group(1).strip()
            for m in _PRODUCT_NAME_RE.finditer(preamble)
            if m.group(1).strip().upper() not in _KNOWN_HEADER_WORDS
        ]
        if candidates:
            result[sec.occurrence] = candidates[-1]
        else:
            result[sec.occurrence] = f"variante_{sec.occurrence + 1}"
    return result
