"""Pydantic schemas for the drug-leaflet (bula) domain.

Covers the Anvisa canonical section keys plus chunk/metadata wrappers used by
the retriever.
"""

from typing import Literal

from pydantic import BaseModel

SectionCanonical = Literal[
    # Identification (pre-block)
    "IDENT_APRESENTACOES",
    "IDENT_COMPOSICAO",
    "IDENT_VIA_USO",
    # Patient-facing information (RDC 47/2009 — 9 questions)
    "IAP_1_INDICACOES",
    "IAP_2_MECANISMO",
    "IAP_3_CONTRAINDICACOES",
    "IAP_4_PRECAUCOES_ADVERTENCIAS",
    "IAP_5_ARMAZENAMENTO",
    "IAP_6_POSOLOGIA",
    "IAP_7_ESQUECIMENTO_DOSE",
    "IAP_8_REACOES_ADVERSAS",
    "IAP_9_SUPERDOSE",
    # Technical information (healthcare professionals)
    "IT_CARACTERISTICAS_FARMACOLOGICAS",
    "IT_INTERACOES_MEDICAMENTOSAS",
    "IT_REACOES_ADVERSAS_TECNICAS",
    # Others
    "DIZERES_LEGAIS",
    "UNCLASSIFIED",
]


class BulaMetadata(BaseModel):
    bula_id: str
    med_name: str
    anvisa_code: str | None = None
    med_variant: str | None = None
    section_canonical: SectionCanonical
    section_raw_header: str | None = None
    chunk_idx: int
    source_page: int | None = None
    patient_facing: bool


class BulaChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: BulaMetadata
    score: float | None = None
