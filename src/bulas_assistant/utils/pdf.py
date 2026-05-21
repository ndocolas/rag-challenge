"""PDF text extraction with on-disk caching for the bula ingestion pipeline.

``pdfplumber`` is the heaviest call in the offline pipeline; caching the
plain-text output keeps ``profile_sections`` and re-ingestion iterations
near-instant after the first run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pdfplumber

from bulas_assistant.utils.logger import get_logger
from bulas_assistant.utils.settings import settings

logger = get_logger(__name__)
_logger_extra = {"component.name": "PdfUtils", "component.version": "v1"}

PAGE_SEPARATOR = "\n\n[PAGE]\n\n"


def _page_text(page) -> str:
    """Extract text from a single page, appending structured table content.

    pdfplumber's extract_text() often garbles multi-column tables (dates, doses).
    We append a structured version of each detected table so the raw cell values
    are always present in the indexed text, even when the layout pass fails.
    """
    raw = page.extract_text() or ""
    tables = page.extract_tables() or []
    if not tables:
        return raw
    table_parts: list[str] = []
    for table in tables:
        rows: list[str] = []
        for row in (table or []):
            cells = ["" if c is None else str(c).strip() for c in (row or [])]
            row_text = "  |  ".join(cells)
            if row_text.strip():
                rows.append(row_text)
        if rows:
            table_parts.append("\n".join(rows))
    if not table_parts:
        return raw
    return raw + "\n\n[TABELAS]\n" + "\n\n".join(table_parts)


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
        pages = [_page_text(p) for p in pdf.pages]
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
