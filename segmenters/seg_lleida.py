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
from typing import Any, Dict, List, Tuple

from .seg_common import read_source_file, validate_segments

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
_LEADING_MARKER_RE = re.compile(
    r"^\s*\[(?:\d+(?:\s*bis)?|[IVX]+)\]\s*",
    re.IGNORECASE,
)


def normalize_article_no(raw_id: str) -> str:
    return raw_id.strip().lower().replace(" ", "")


def make_segment_id(source_name: str, article_no: str) -> str:
    return f"{source_name}_{article_no}"


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
        markers.append(
            {
                "raw_id": raw_id,
                "start": m.start(),
                "end": m.end(),
                "is_section": is_section,
                "is_article": is_article,
            }
        )
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


def segment_lleida_unified(source_file, source_name):
    """
    Unified Lleida segmenter.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "ObychaiLleidy12271228".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_segments = segment_lleida(text, source_name=source_name, debug=False)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/ObychaiLleidy12271228_v2.txt"),
        Path("ObychaiLleidy12271228_v2.txt"),
        Path("/mnt/data/ObychaiLleidy12271228_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_lleida_unified(src, "ObychaiLleidy12271228")
    print(f"ObychaiLleidy12271228: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()