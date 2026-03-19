#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Costums de Tarrega (1290 / privilege of 1242).

Новая версия ориентирована на ObychaiTarregi1290E_v2.txt.

Что меняется:
- опора не на суррогатные "Unmarked" куски, а на реальную нумерацию [1]...[25]
- единый стиль id: ObychaiTarregi1290E_ArtN
- латинская преамбула выделяется как Art0
- сегментер режет именно юридическое ядро, начиная с
  "Noverint universi ..." и заканчивая датировкой "Datum Ilerde..."
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional


# Маркеры статей в этом OCR:
# [1] [2] ... [12] [13) [14] (15] [17) [22} (25]
_ARTICLE_MARKER_RE = re.compile(r'(?m)^[\s\u3000\.,;:·•\-–—]*[\[\(](\d{1,2})[\]\)\}]\s*')

_START_MARKERS = [
    "Noverint universi quod nos",
    "[CONSUETUDINES VILLE TAREGE.",
]

_END_MARKERS = [
    "Datum Ilerde",
    "Sig t. num Iacobi",
    "Signum Guillemoni scriba",
]

_SKIP_LINE_PATTERNS = [
    re.compile(r'^\s*\d{1,4}\s*$'),                    # page number
    re.compile(r'^\s*[a-z]\s+B\b'),                    # editorial note like "a B"
    re.compile(r'^\s*[a-z]\s+C\b'),
    re.compile(r'^\s*[a-z]\s+D\b'),
    re.compile(r'^\s*\d+\.'),                          # scholarly footnotes/comments
]

_MULTI_SPACE_RE = re.compile(r'\s+')
_HYPHEN_BREAK_RE = re.compile(r'(\w)[\-\u00ad]\s+(\w)')
_LEADING_MARKER_RE = re.compile(r'^[\s\.,;:·•\-–—]*[\[\(]\d{1,2}[\]\)\}]\s*')
_INLINE_FOOTNOTE_RE = re.compile(r'(?<=\w)\s*\d+\s*[•º°*]?(?=[\s\.,;:])')


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _slice_legal_core(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start_pos: Optional[int] = None
    for marker in _START_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            start_pos = pos
            break
    if start_pos is None:
        start_pos = 0

    text = text[start_pos:]

    end_pos: Optional[int] = None
    for marker in _END_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break
    if end_pos is not None:
        text = text[:end_pos]

    return text


def _keep_line(line: str) -> bool:
    s = line.strip().replace("\xa0", " ").replace("\u3000", " ")
    if not s:
        return False
    for pat in _SKIP_LINE_PATTERNS:
        if pat.match(s):
            return False
    return True


def _normalize_block(block: str) -> str:
    lines = []
    for raw in block.splitlines():
        if not _keep_line(raw):
            continue
        s = raw.strip().replace("\xa0", " ").replace("\u3000", " ")
        lines.append(s)

    text = " ".join(lines)
    text = _LEADING_MARKER_RE.sub("", text)
    text = _INLINE_FOOTNOTE_RE.sub("", text)
    text = clean_text(text)
    text = re.sub(r'^[\]\)\}.,:;\-\s]+', '', text)
    return clean_text(text)


def _extract_preamble(core_text: str) -> str:
    matches = list(_ARTICLE_MARKER_RE.finditer(core_text))
    if not matches:
        return ""
    preamble = core_text[:matches[0].start()]
    return _normalize_block(preamble)


def segment_tarregi(
    text: str,
    source_name: str = "ObychaiTarregi1290E",
    min_words: int = 8,
) -> List[Tuple[str, str]]:
    """
    Unified-style segmentation of Tarrega.

    Output IDs:
      ObychaiTarregi1290E_Art0
      ObychaiTarregi1290E_Art1
      ...
      ObychaiTarregi1290E_Art25
    """
    core_text = _slice_legal_core(text)
    matches = list(_ARTICLE_MARKER_RE.finditer(core_text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    preamble = _extract_preamble(core_text)
    if preamble and len(preamble.split()) >= min_words:
        segments.append((f"{source_name}_Art0", preamble))

    for idx, match in enumerate(matches):
        art_no = int(match.group(1))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(core_text)
        block = core_text[start:end]
        article_text = _normalize_block(block)

        if article_text and len(article_text.split()) >= min_words:
            seg_id = f"{source_name}_Art{art_no}"
            segments.append((seg_id, article_text))

    return segments


def segment_tarregi_unified(source_file, source_name):
    """
    Unified Tarregi segmenter.
    """
    from .seg_common import read_source_file, validate_segments

    text = read_source_file(source_file)
    raw_segments = segment_tarregi(text, source_name=source_name, min_words=8)
    return validate_segments(raw_segments, source_name)


def analyze_and_save(
    text: str,
    output_file: str,
    source_name: str = "ObychaiTarregi1290E",
) -> List[Tuple[str, str]]:
    print("=" * 80)
    print("COSTUMS DE TARREGA (1290 / PRIVILEGE OF 1242) - SEGMENTATION")
    print("=" * 80)

    segments = segment_tarregi(text, source_name=source_name, min_words=8)

    print(f"Found segments: {len(segments)}")
    if segments:
        print(f"First id: {segments[0][0]}")
        print(f"Last id:  {segments[-1][0]}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total segments: {len(segments)}\n")
        f.write("=" * 80 + "\n\n")
        for seg_id, seg_text in segments:
            f.write("=" * 80 + "\n")
            f.write(f"{seg_id}\n")
            f.write("=" * 80 + "\n")
            f.write(seg_text)
            f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    return segments


def main():
    candidates = [
        Path("data/ObychaiTarregi1290E_v2.txt"),
        Path("ObychaiTarregi1290E_v2.txt"),
        Path("/mnt/data/ObychaiTarregi1290E_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found. Tried:")
        for p in candidates:
            print(f"  - {p}")
        raise SystemExit(1)

    print(f"Processing {src}...")
    text = src.read_text(encoding="utf-8", errors="replace")
    segs = analyze_and_save(
        text,
        output_file="tarregi_segmented.txt",
        source_name="ObychaiTarregi1290E",
    )

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
