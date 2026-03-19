#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Costums de Lleida (1228).

Основная структурная единица:
- статья [1]...[171]
- отдельный вариант [60 bis]

Крупные разделы [I], [II], [III] используются как структурные маркеры,
но не отдаются как сегменты.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

_SECTION_RE = re.compile(r"^[IVX]+$", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^\d+(?:\s*bis)?$", re.IGNORECASE)

_MARKER_RE = re.compile(
    r"(?m)^\s*\[(?P<id>\d+(?:\s*bis)?|[IVX]+)\]\s*(?P<rest>.*)$"
)

_END_MARKERS = (
    "Expliciunt consuetudines civitatis Ilerde.",
)

_FOOTNOTE_RE = re.compile(r"^\s*\d+\s+[A-ZÁÉÍÓÚ]")
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_MULTI_SPACE_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
_LEADING_MARKER_RE = re.compile(r"^\s*\[(?:\d+(?:\s*bis)?|[IVX]+)\]\s*", re.IGNORECASE)


def normalize_article_no(raw_id: str) -> str:
    return raw_id.strip().lower().replace(" ", "")


def make_segment_id(source_name: str, article_no: str) -> str:
    return f"{source_name}_Art{article_no}"


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _looks_like_editorial_or_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s in _END_MARKERS:
        return True
    if _PAGE_RE.match(s):
        return True
    if _FOOTNOTE_RE.match(s):
        return True
    return False


def extract_article_text(lines: List[str]) -> str:
    """
    Clean article body and robustly remove the leading structural marker.
    This version removes [1], [2], [60 bis] even if blank/noisy lines appear first.
    """
    result: List[str] = []

    for line in lines:
        s = line.strip().replace("\xa0", " ")
        if _looks_like_editorial_or_noise(s):
            if s in _END_MARKERS:
                break
            continue
        result.append(s)

    text = clean_text(" ".join(result))
    text = _LEADING_MARKER_RE.sub("", text)
    text = re.sub(r"^[\]\)\}.,:;\-\s]+", "", text)
    text = clean_text(text)
    return text


def _scan_markers(text: str) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    for m in _MARKER_RE.finditer(text):
        raw_id = m.group("id").strip()
        is_section = bool(_SECTION_RE.fullmatch(raw_id))
        is_article = bool(_ARTICLE_RE.fullmatch(raw_id))
        if not (is_section or is_article):
            continue
        markers.append({
            "raw_id": raw_id,
            "start": m.start(),
            "end": m.end(),
            "is_section": is_section,
            "is_article": is_article,
        })
    return markers


def segment_lleida(text: str, source_name: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Costums de Lleida into numbered articles.

    Output IDs use a unified style:
      ObychaiLleidy12271228_Art1
      ObychaiLleidy12271228_Art60bis
      ObychaiLleidy12271228_Art171
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    markers = _scan_markers(text)

    if debug:
        print(f"Total structural markers found: {len(markers)}")
        print(f"  Sections: {sum(1 for m in markers if m['is_section'])}")
        print(f"  Articles: {sum(1 for m in markers if m['is_article'])}")

    segments: List[Tuple[str, str]] = []
    seen_article_ids = set()

    for idx, marker in enumerate(markers):
        if not marker["is_article"]:
            continue

        article_no = normalize_article_no(marker["raw_id"])
        if article_no in seen_article_ids:
            continue

        start = marker["start"]
        end = markers[idx + 1]["start"] if idx + 1 < len(markers) else len(text)
        block = text[start:end]

        lines = block.split("\n")
        article_text = extract_article_text(lines)
        if not article_text:
            continue

        seg_id = make_segment_id(source_name, article_no)
        segments.append((seg_id, article_text))
        seen_article_ids.add(article_no)

        if debug and len(segments) <= 12:
            print(f"  {seg_id}: {article_text[:100]}")

    return segments


def analyze_and_save(
    text: str,
    output_file: str,
    source_name: str = "ObychaiLleidy12271228",
    expected_main_count: int = 171,
) -> List[Tuple[str, str]]:
    print("=" * 80)
    print("COSTUMS DE LLEIDA (1228) - ARTICLE SEGMENTATION")
    print("=" * 80)

    articles = segment_lleida(text, source_name=source_name, debug=False)

    article_ids = [seg_id.rsplit("_Art", 1)[1] for seg_id, _ in articles]
    bis_articles = [aid for aid in article_ids if aid.endswith("bis")]
    main_articles = [aid for aid in article_ids if aid.isdigit()]

    coverage = len(main_articles) / expected_main_count * 100 if expected_main_count else 0.0

    print(f"Expected main articles: {expected_main_count}")
    print(f"Found total segments: {len(articles)}")
    print(f"  - Main numbered articles: {len(main_articles)}")
    print(f"  - Bis articles: {len(bis_articles)}")
    print(f"Coverage of main sequence: {coverage:.1f}%")

    from collections import Counter
    duplicates = [(n, c) for n, c in Counter(article_ids).items() if c > 1]
    if duplicates:
        print(f"\nDuplicates: {len(duplicates)}")
        for num, count in sorted(duplicates)[:5]:
            print(f"  Article {num}: {count} times")
    else:
        print("\nNo duplicates")

    expected_nums = set(str(i) for i in range(1, expected_main_count + 1))
    found_nums = set(main_articles)
    missing = sorted(int(n) for n in (expected_nums - found_nums))
    if missing:
        print(f"\nMissing main articles: {len(missing)}")
        if len(missing) <= 15:
            print(f"  {missing}")
        else:
            print(f"  First 10: {missing[:10]}")
            print(f"  Last 5: {missing[-5:]}")
    else:
        print("\nNo missing main articles")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total segments: {len(articles)}\n")
        f.write(f"Main articles: {len(main_articles)}\n")
        f.write(f"Bis articles: {len(bis_articles)}\n")
        f.write("=" * 80 + "\n\n")
        for seg_id, article_text in articles:
            f.write("=" * 80 + "\n")
            f.write(f"{seg_id}\n")
            f.write("=" * 80 + "\n")
            f.write(article_text)
            f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    return articles


def segment_lleida_unified(source_file, source_name):
    """
    Unified Lleida segmenter.
    Reads the source file and returns list[(id, text)] with unified IDs.
    """
    from .seg_common import read_source_file, validate_segments

    text = read_source_file(source_file)
    raw_segments = segment_lleida(text, source_name=source_name, debug=False)
    return validate_segments(raw_segments, source_name)


def main():
    candidates = [
        Path("data/ObychaiLleidy12271228_v2.txt"),
        Path("ObychaiLleidy12271228_v2.txt"),
        Path("/mnt/data/ObychaiLleidy12271228_v2.txt"),
    ]

    file_lleida = next((p for p in candidates if p.exists()), None)
    if file_lleida is None:
        print("Error: source file not found. Tried:")
        for p in candidates:
            print(f"  - {p}")
        raise SystemExit(1)

    print(f"Processing Costums de Lleida from {file_lleida}...")
    text = file_lleida.read_text(encoding="utf-8", errors="replace")
    articles = analyze_and_save(
        text,
        output_file="costums_lleida_segmented.txt",
        source_name="ObychaiLleidy12271228",
        expected_main_count=171,
    )

    print("\nFirst 5 segments:")
    for seg_id, seg_text in articles[:5]:
        print(f"  {seg_id}: {seg_text[:120]}")

    print("\nLast 5 segments:")
    for seg_id, seg_text in articles[-5:]:
        print(f"  {seg_id}: {seg_text[:120]}")


if __name__ == "__main__":
    main()
