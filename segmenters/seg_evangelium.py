"""Segmenter for Evangelium (Matthew, Mark, Luke, John)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, validate_segments, read_source_file


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
    return f"{source_name}_{code}_{int(chapter_num)}"


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
    Унифицированная сегментация Evangelium.
    Сегментер сам читает файл и возвращает list[(id, text)].
    """
    text = read_source_file(source_file)
    raw_segments = segment_evangelium(text, source_name)
    return validate_segments(raw_segments, source_name)


if __name__ == "__main__":
    candidates = [
        Path("data/Evangelium_v2.txt"),
    ]

    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        print("Not found. Expected one of:")
        for c in candidates:
            print(f"  - {c}")
        raise SystemExit(1)

    text = read_source_file(p)
    segs = segment_evangelium(text, "Evangelium")

    print(f"Evangelium: {len(segs)} segments")
    print("Expected structural unit: one segment per chapter of each Gospel")
    print()

    if segs:
        print("First 5 segments:")
        for sid, txt in segs[:5]:
            print(f"  {sid}: {txt[:140]}")
        print()

        print("Last 5 segments:")
        for sid, txt in segs[-5:]:
            print(f"  {sid}: {txt[:140]}")