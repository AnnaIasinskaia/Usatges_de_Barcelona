#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Privilegi de la Querimònia (Val d'Aran, 1313).

Ориентация на ObychaiValdArana1313_v2.txt.

Принципы:
- unified-выход только в формате (id, text)
- единый стиль id: ObychaiValdArana1313_ArtN
- Art0 = латинская преамбула
- Art1..Art23 = статьи [I]..[XXIII]
- режем только документ 38 и не заходим в следующий текст
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .seg_common import read_source_file, validate_segments


_DOC_HEADER_RE = re.compile(
    r'^\s*38\.\s*PRIVILEGI DIT GENERALMENT DE LA QUERIMONIA',
    re.IGNORECASE,
)
_NEXT_DOC_RE = re.compile(
    r'^\s*(?:c\)\s*Compilaciones de derecho feudal|39\.\s*LAS COSTUMAS DE CATHALUNYA)',
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(r'(?m)^[\s\u3000]*\[([IVXLCDM]+)\]\s*')

_FOOTNOTE_RE = re.compile(r'^\s*\d+\s+[A-ZÁÉÍÓÚ]')
_PAGE_RE = re.compile(r'^\s*\d{1,4}\s*$')
_MULTI_SPACE_RE = re.compile(r'\s+')
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\s+(\w)')

_SKIP_CONTAINS = [
    "Jaime II de Aragón concede",
    "Original, perdido",
    "Copia de la misma fecha que el original",
    "Copia en documento de confirmación",
    "***",
    "Publicado por",
    "reedición a cargo de",
    "Una traducción",
    "Agradecemos la ayuda",
]

_END_MARKERS = [
    "In cuius rei testimonium",
    "Data Ilerde",
]

_ROMAN_VALUES: Dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}


def roman_to_int(roman: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(roman.upper()):
        value = _ROMAN_VALUES.get(ch, 0)
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _slice_target_document(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start_idx = None
    for i, line in enumerate(lines):
        if _DOC_HEADER_RE.search(line):
            start_idx = i
            break

    if start_idx is None:
        return "\n".join(lines)

    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        if _NEXT_DOC_RE.search(lines[i]):
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


def _trim_tail(text: str) -> str:
    for marker in _END_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            return text[:pos]
    return text


def _drop_editorial_noise(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        s = raw.strip().replace("\xa0", " ").replace("\u3000", " ")
        if not s:
            continue
        if _PAGE_RE.match(s):
            continue
        if _FOOTNOTE_RE.match(s):
            continue
        if any(mark in s for mark in _SKIP_CONTAINS):
            continue
        if _DOC_HEADER_RE.search(s):
            continue
        out.append(s)
    return out


def _extract_preamble(text: str) -> str:
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return ""

    preamble_raw = text[:matches[0].start()]
    lines = _drop_editorial_noise(preamble_raw.splitlines())
    joined = "\n".join(lines)

    start_markers = [
        "In Christi nomine. Pateat universis",
        "Pateat universis, quod coram Nobis",
    ]
    start_pos = -1
    for marker in start_markers:
        pos = joined.find(marker)
        if pos != -1:
            start_pos = pos
            break
    if start_pos != -1:
        joined = joined[start_pos:]

    return clean_text(joined)


def _extract_articles(text: str, source_name: str) -> List[Tuple[str, str]]:
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    for idx, match in enumerate(matches):
        roman = match.group(1)
        art_no = roman_to_int(roman)

        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]

        block = re.sub(r'^\s*\[[IVXLCDM]+\]\s*', '', block, flags=re.IGNORECASE)
        lines = _drop_editorial_noise(block.splitlines())
        article_text = clean_text(" ".join(lines))

        if article_text:
            segments.append((f"{source_name}_{art_no}", article_text))

    return segments


def segment_vald_aran(
    text: str,
    source_name: str,
    min_words: int = 10,
) -> List[Tuple[str, str]]:
    """
    Unified-style segmentation for Querimònia.

    IDs:
      ObychaiValdArana1313_Preamble
      ObychaiValdArana1313_Art1
      ...
      ObychaiValdArana1313_Art23
    """
    text = _slice_target_document(text)
    text = _trim_tail(text)

    segments: List[Tuple[str, str]] = []

    preamble = _extract_preamble(text)
    if preamble and len(preamble.split()) >= min_words:
        segments.append((f"{source_name}_Preamble", preamble))

    for seg_id, seg_text in _extract_articles(text, source_name):
        if len(seg_text.split()) >= min_words:
            segments.append((seg_id, seg_text))

    return segments


def segment_vald_aran_unified(source_file, source_name):
    """
    Unified Val d'Aran segmenter.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "ObychaiValdArana1313".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_segments = segment_vald_aran(text, source_name=source_name, min_words=10)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/ObychaiValdArana1313_v2.txt"),
        Path("ObychaiValdArana1313_v2.txt"),
        Path("/mnt/data/ObychaiValdArana1313_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_vald_aran_unified(src, "ObychaiValdArana1313")
    print(f"ObychaiValdArana1313: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            preview = txt[:120] + "..." if len(txt) > 120 else txt
            print(f"  {sid}: {preview}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            preview = txt[:120] + "..." if len(txt) > 120 else txt
            print(f"  {sid}: {preview}")


if __name__ == "__main__":
    main()