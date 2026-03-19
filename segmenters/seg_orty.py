#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Costums d'Orta (1296).

Новая логика ориентирована на v2.txt, где статьи даны явными маркерами:
  [I] ... [LXXXI]

Выходной формат:
  ObychaiOrty1296_Art0   -> преамбула
  ObychaiOrty1296_Art1   -> статья [I]
  ...
  ObychaiOrty1296_Art81  -> статья [LXXXI]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict


_ARTICLE_MARK_RE = re.compile(r'(?m)^[\s\u3000]*\[([IVXLCDM]+)\]\s*')
_DOC_HEADER_RE = re.compile(r'^\s*37\.\s*COSTUMBRES\s+DE\s+ORTA', re.IGNORECASE)

_CLOSING_MARKERS = [
    "Mandantes universis",
    "Ad hec autem nos",
    "Quod fuit actum",
    "Sig+num fratris",
    "Testes sunt",
    "38.PRIVILEGI DIT GENERALMENT",
]

_SKIP_PREFIXES = [
    "***",
    "Original en pergamino",
]

_FOOTNOTE_RE = re.compile(r'^\s*\d+\s+[A-ZÁÉÍÓÚ]')
_PAGE_RE = re.compile(r'^\s*\d{1,4}\s*$')
_MULTI_SPACE_RE = re.compile(r'\s+')
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\s+(\w)')


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
    """
    Берём только документ 37 (Orta), от его заголовка до начала следующего документа
    или до конца файла.
    """
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
        line = lines[i].strip()
        if line.startswith("38.PRIVILEGI DIT GENERALMENT"):
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


def _trim_tail(text: str) -> str:
    for marker in _CLOSING_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            return text[:pos]
    return text


def _drop_editorial_noise(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        s = line.strip().replace("\xa0", " ").replace("\u3000", " ")
        if not s:
            continue
        if _PAGE_RE.match(s):
            continue
        if _FOOTNOTE_RE.match(s):
            continue
        if any(s.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        if "COSTUMBRES DE ORTA" in s:
            continue
        out.append(s)
    return out


def _extract_preamble(text: str) -> str:
    matches = list(_ARTICLE_MARK_RE.finditer(text))
    if not matches:
        return ""

    preamble_raw = text[:matches[0].start()]
    lines = _drop_editorial_noise(preamble_raw.split("\n"))

    # стараемся начинать именно с юридической преамбулы
    joined = "\n".join(lines)
    start_markers = [
        "Pateat universis",
        "In Christi nomine. Pateat universis",
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
    matches = list(_ARTICLE_MARK_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    for idx, match in enumerate(matches):
        roman = match.group(1)
        art_no = roman_to_int(roman)

        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]

        # remove the leading [ROMAN] marker
        block = re.sub(r'^\s*\[[IVXLCDM]+\]\s*', '', block, flags=re.IGNORECASE)
        lines = _drop_editorial_noise(block.split("\n"))
        article_text = clean_text(" ".join(lines))

        if article_text:
            segments.append((f"{source_name}_Art{art_no}", article_text))

    return segments


def segment_orty(text: str, source_name: str, min_words: int = 10) -> List[Tuple[str, str]]:
    """
    Основная функция сегментации.

    Unified-style IDs:
      ObychaiOrty1296_Art0
      ObychaiOrty1296_Art1
      ...
      ObychaiOrty1296_Art81
    """
    text = _slice_target_document(text)
    text = _trim_tail(text)

    segments: List[Tuple[str, str]] = []

    preamble = _extract_preamble(text)
    if preamble and len(preamble.split()) >= min_words:
        segments.append((f"{source_name}_Art0", preamble))

    article_segments = _extract_articles(text, source_name)
    for seg_id, seg_text in article_segments:
        if len(seg_text.split()) >= min_words:
            segments.append((seg_id, seg_text))

    return segments


def segment_orty_unified(source_file, source_name):
    """
    Unified Orty segmenter.
    """
    from .seg_common import read_source_file, validate_segments

    text = read_source_file(source_file)
    raw_segments = segment_orty(text, source_name=source_name, min_words=10)
    return validate_segments(raw_segments, source_name)


def main():
    candidates = [
        Path("data/ObychaiOrty1296_v2.txt"),
        Path("ObychaiOrty1296_v2.txt"),
        Path("/mnt/data/ObychaiOrty1296_v2.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found. Tried:")
        for p in candidates:
            print(f"  - {p}")
        raise SystemExit(1)

    text = src.read_text(encoding="utf-8", errors="replace")
    segs = segment_orty(text, "ObychaiOrty1296", min_words=10)

    print("=" * 80)
    print("COSTUMS D'ORTA (1296) — SEGMENTATION RESULT")
    print("=" * 80)
    print(f"Source: {src}")
    print(f"Total segments: {len(segs)}")

    if segs:
        print("\nFirst 5 segments:")
        for sid, stxt in segs[:5]:
            preview = stxt[:120] + "..." if len(stxt) > 120 else stxt
            print(f"  {sid}: {preview}")

        print("\nLast 5 segments:")
        for sid, stxt in segs[-5:]:
            preview = stxt[:120] + "..." if len(stxt) > 120 else stxt
            print(f"  {sid}: {preview}")


if __name__ == "__main__":
    main()