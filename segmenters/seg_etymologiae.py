#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmenter for Isidori Hispalensis Episcopi Etymologiarum."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


# В этом OCR-файле структура очень чистая:
#   Caput I.
#   DE AUCTORIBUS LEGUM.
#   [1] ...
#   [2] ...
#
# Главы идут последовательно от I до XXXIX.
# Судя по содержанию, это юридическая книга Etymologiarum, поэтому для
# сопоставимых ID полезно фиксировать номер caput.
#
# Примеры ID:
#   Etymologiae_C1
#   Etymologiae_C2
#   Etymologiae_C39

_CAPUT_RE = re.compile(
    r"(?m)^\s*Caput\s+([IVXLCDM]+)\.\s*$"
)

_TITLE_RE = re.compile(
    r"(?m)^\s*([A-Z][A-Z\s,\-\';:\?\!\[\]À-Ý]+)\s*$"
)

_INLINE_SECTION_NUM_RE = re.compile(r"\[\d+\]\s*")
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")


_ROMAN_MAP = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def _roman_to_int(s: str) -> int:
    s = s.strip().upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _clean_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _PAGE_RE.match(s):
            continue
        lines.append(s)

    joined = " ".join(lines)
    joined = _INLINE_SECTION_NUM_RE.sub("", joined)
    return clean_text(joined)


def _make_segment_id(source_name: str, caput_roman: str) -> str:
    caput_no = _roman_to_int(caput_roman)
    return f"{source_name}_C{caput_no}"


def segment_etymologiae(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Чистая структурная сегментация Etymologiae:
    один сегмент = один Caput.

    IDs содержат фактический номер главы из документа:
      Etymologiae_C1
      Etymologiae_C2
      ...
      Etymologiae_C39
    """
    matches = list(_CAPUT_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        caput_roman = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        if not block:
            continue

        lines = [ln.rstrip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln.strip()]

        title = ""
        body_lines = lines
        if lines and _TITLE_RE.match(lines[0].strip()):
            title = lines[0].strip()
            body_lines = lines[1:]

        body = _clean_block("\n".join(body_lines))
        if title:
            seg_text = clean_text(f"{title} {body}")
        else:
            seg_text = body

        if not seg_text:
            continue

        seg_id = _make_segment_id(source_name, caput_roman)
        segments.append((seg_id, seg_text))

    return validate_segments(segments, source_name)


def segment_etymologiae_unified(source_file, source_name):
    """
    Unified segmenter for Isidori Etymologiae.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "Etymologiae".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_etymologiae(text, source_name)


def main() -> None:
    candidates = [
        Path("data/Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt"),
        Path("Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt"),
        Path("/mnt/data/Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_etymologiae_unified(src, "Etymologiae")
    print(f"Etymologiae: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()