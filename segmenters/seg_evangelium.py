#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmenter for Evangelium (Matthew, Mark, Luke, John)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


_GOSPEL_RE = re.compile(
    r"(?m)^\s*EVANGELIUM\s+SECUNDUM\s+(MATTHAEUM|MARCUM|LUCAM|IOANNEM)\s*$",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(r"\[(\d+)\]")
_INLINE_VERSE_RE = re.compile(r"(?<!\[)\b(\d{1,3})\b(?!\])")


_GOSPEL_CODE = {
    "MATTHAEUM": "Mt",
    "MARCUM": "Mc",
    "LUCAM": "Lc",
    "IOANNEM": "Io",
}


def _normalize_gospel_name(name: str) -> str:
    return name.strip().upper()


def _cleanup_chapter_text(text: str) -> str:
    """
    Убираем номера стихов внутри главы, оставляя только собственно текст.
    Главные структурные номера (Евангелие + глава) идут в ID.
    """
    text = _INLINE_VERSE_RE.sub(" ", text)
    text = clean_text(text)
    return text


def _make_segment_id(source_name: str, gospel_name: str, chapter_num: str) -> str:
    code = _GOSPEL_CODE[_normalize_gospel_name(gospel_name)]
    return f"{source_name}_{code}.{int(chapter_num)}"


def segment_evangelium(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Чистая структурная сегментация Evangelium:
    один сегмент = одна глава конкретного Евангелия.

    IDs содержат реальные номера из документа:
      Evangelium_Mt_1
      Evangelium_Mc_5
      Evangelium_Lc_12
      Evangelium_Io_3
    """
    gospel_matches = list(_GOSPEL_RE.finditer(text))
    if not gospel_matches:
        return []

    segments: List[Tuple[str, str]] = []

    for i, gm in enumerate(gospel_matches):
        gospel_name = _normalize_gospel_name(gm.group(1))
        start = gm.end()
        end = gospel_matches[i + 1].start() if i + 1 < len(gospel_matches) else len(text)
        gospel_block = text[start:end].strip()
        if not gospel_block:
            continue

        chapter_matches = list(_CHAPTER_RE.finditer(gospel_block))
        if not chapter_matches:
            continue

        for j, cm in enumerate(chapter_matches):
            chapter_num = cm.group(1).strip()
            c_start = cm.end()
            c_end = chapter_matches[j + 1].start() if j + 1 < len(chapter_matches) else len(gospel_block)

            chapter_text = gospel_block[c_start:c_end].strip()
            chapter_text = _cleanup_chapter_text(chapter_text)
            if not chapter_text:
                continue

            seg_id = _make_segment_id(source_name, gospel_name, chapter_num)
            segments.append((seg_id, chapter_text))

    return validate_segments(segments, source_name)


def segment_evangelium_unified(source_file, source_name):
    """
    Unified segmenter for Evangelium.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "Evangelium".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_evangelium(text, source_name)


def main() -> None:
    candidates = [
        Path("data/Evangelium_v2.txt"),
        Path("Evangelium_v2.txt"),
        Path("/mnt/data/Evangelium_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_evangelium_unified(src, "Evangelium")
    print(f"Evangelium: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()