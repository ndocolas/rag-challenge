"""CLI for ingesting bulas into Qdrant.

Usage::

    uv run python scripts/ingest_bulas.py
    uv run python scripts/ingest_bulas.py --rebuild
    uv run python scripts/ingest_bulas.py --corpus-dir /path/to/pdfs
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from panvel_assistant.services.ingestion_service import ingestion_service


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the Qdrant collection before re-indexing",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="override the default corpus directory (settings.BULAS_DIR)",
    )
    args = parser.parse_args()

    manifest = await ingestion_service.ingest_corpus(
        corpus_dir=args.corpus_dir, rebuild=args.rebuild
    )
    print(
        f"OK: {manifest['total_chunks']} chunks indexed "
        f"({manifest['total_full_sections']} full sections, "
        f"{manifest['total_sub_chunks']} sub-chunks) "
        f"into '{manifest['collection']}'"
    )
    for fname, info in sorted(manifest["bulas"].items()):
        print(
            f"  {fname:<55s} -> {info['chunks']:3d} chunks "
            f"(full={info['full_sections']:2d}, sub={info['sub_chunks']:2d})"
        )


if __name__ == "__main__":
    asyncio.run(_main())
