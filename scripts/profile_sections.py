"""Profile section sizes across the bula corpus to pick a chunking threshold.

Run this once after ``utils/pdf.py`` and ``assistant/sectionizer.py`` are in
place, before settling on ``SECTION_WHOLE_THRESHOLD`` for ingestion. The
report prints aggregate and per-canonical-section percentiles plus a
simulation of how many chunks each candidate threshold would produce.

Usage::

    uv run python scripts/profile_sections.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from panvel_assistant.assistant.sectionizer import sectionize
from panvel_assistant.utils.pdf import extract_text
from panvel_assistant.utils.settings import settings

THRESHOLD_CANDIDATES = (2000, 3500, 6000)
SUB_CHUNK_SIZE = 1600  # matches ingestion_service splitter
SUB_CHUNK_OVERLAP = 200
MIN_SECTION_CHARS = 150  # below this we drop the section as noise


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    return int(values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f))


def _simulate_chunk_count(
    sizes: list[int], threshold: int, sub_size: int, overlap: int
) -> int:
    """Approximate chunk count if we applied the hybrid policy with ``threshold``."""
    total = 0
    for size in sizes:
        if size < MIN_SECTION_CHARS:
            continue
        if size <= threshold:
            total += 1
            continue
        stride = max(sub_size - overlap, 1)
        total += max(1, (size + stride - 1) // stride)
    return total


def main() -> None:
    pdfs = sorted(settings.BULAS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {settings.BULAS_DIR}")

    all_sizes: list[int] = []
    by_canonical: dict[str, list[int]] = defaultdict(list)
    largest: list[tuple[int, str, str]] = []  # (size, bula_stem, canonical)
    per_bula: dict[str, dict] = {}

    for pdf in pdfs:
        text = extract_text(pdf)
        sections = sectionize(text)
        bula_sizes: list[int] = []
        for s in sections:
            size = len(s.content)
            all_sizes.append(size)
            by_canonical[s.canonical].append(size)
            largest.append((size, pdf.stem, s.canonical))
            bula_sizes.append(size)
        per_bula[pdf.stem] = {
            "n_sections": len(sections),
            "max_section_chars": max(bula_sizes) if bula_sizes else 0,
            "total_chars": sum(bula_sizes),
        }

    largest.sort(reverse=True)

    print("=" * 80)
    print(f"Corpus: {len(pdfs)} bulas, {len(all_sizes)} sections total")
    print("=" * 80)

    print("\nAggregate section_char_len distribution:")
    print(f"  min={min(all_sizes)}  max={max(all_sizes)}  mean={int(statistics.mean(all_sizes))}")
    for p in (0.50, 0.75, 0.90, 0.99):
        print(f"  p{int(p * 100):>2}: {_percentile(all_sizes, p)}")

    print("\nBy section_canonical (p50 / p90 / max, n):")
    for canonical in sorted(by_canonical):
        sizes = by_canonical[canonical]
        print(
            f"  {canonical:<38s} "
            f"p50={_percentile(sizes, 0.5):>6d}  "
            f"p90={_percentile(sizes, 0.9):>6d}  "
            f"max={max(sizes):>6d}  "
            f"n={len(sizes)}"
        )

    print("\nTop 10 largest sections:")
    for size, stem, canonical in largest[:10]:
        print(f"  {size:>6d} chars  {stem:<40s}  {canonical}")

    print("\nChunk count simulation per threshold (drops sections < 150 chars):")
    print(
        f"  {'threshold':>12s}  {'n_full_sections':>18s}  "
        f"{'n_split_sections':>18s}  {'total_chunks':>14s}"
    )
    for th in THRESHOLD_CANDIDATES:
        kept = [s for s in all_sizes if s >= MIN_SECTION_CHARS]
        n_full = sum(1 for s in kept if s <= th)
        n_split = sum(1 for s in kept if s > th)
        total = _simulate_chunk_count(all_sizes, th, SUB_CHUNK_SIZE, SUB_CHUNK_OVERLAP)
        print(f"  {th:>12d}  {n_full:>18d}  {n_split:>18d}  {total:>14d}")

    report = {
        "n_bulas": len(pdfs),
        "n_sections": len(all_sizes),
        "aggregate": {
            "min": min(all_sizes),
            "max": max(all_sizes),
            "mean": int(statistics.mean(all_sizes)),
            "p50": _percentile(all_sizes, 0.5),
            "p75": _percentile(all_sizes, 0.75),
            "p90": _percentile(all_sizes, 0.9),
            "p99": _percentile(all_sizes, 0.99),
        },
        "by_canonical": {
            c: {
                "n": len(sizes),
                "p50": _percentile(sizes, 0.5),
                "p90": _percentile(sizes, 0.9),
                "max": max(sizes),
            }
            for c, sizes in by_canonical.items()
        },
        "top_largest": [
            {"chars": size, "bula": stem, "canonical": canonical}
            for size, stem, canonical in largest[:10]
        ],
        "threshold_simulation": [
            {
                "threshold": th,
                "n_full_sections": sum(
                    1 for s in all_sizes if MIN_SECTION_CHARS <= s <= th
                ),
                "n_split_sections": sum(1 for s in all_sizes if s > th),
                "total_chunks": _simulate_chunk_count(
                    all_sizes, th, SUB_CHUNK_SIZE, SUB_CHUNK_OVERLAP
                ),
            }
            for th in THRESHOLD_CANDIDATES
        ],
        "per_bula": per_bula,
    }
    out_path = settings.CACHE_DIR / "section_profile.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
