"""PDF text extraction with on-disk caching for the bula ingestion pipeline.

``pdfplumber`` is the heaviest call in the offline pipeline; caching the
plain-text output keeps ``profile_sections`` and re-ingestion iterations
near-instant after the first run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pdfplumber

from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "PdfUtils", "component.version": "v1"}

PAGE_SEPARATOR = "\n\n[PAGE]\n\n"


def extract_text(pdf_path: Path, force: bool = False) -> str:
    """Extract text from ``pdf_path`` and cache the result.

    The cache lives at ``settings.CACHE_DIR / "<stem>.txt"``. Pass ``force=True``
    to bypass it (useful when the PDF was replaced in place). Pages are joined
    with a ``[PAGE]`` marker so downstream code can recover page boundaries.
    """
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = settings.CACHE_DIR / f"{pdf_path.stem}.txt"
    if cache_file.exists() and not force:
        return cache_file.read_text(encoding="utf-8")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = PAGE_SEPARATOR.join(pages)
    cache_file.write_text(text, encoding="utf-8")
    logger.info(
        "pdf extraido",
        extra={**_logger_extra, "step": "pdf_extract", "path": str(pdf_path), "chars": len(text)},
    )
    return text


def file_hash(pdf_path: Path) -> str:
    """Return MD5 of the PDF bytes. Used in the ingestion manifest."""
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()
