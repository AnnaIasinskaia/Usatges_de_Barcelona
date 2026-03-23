#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Constituciones de Miravet (1319).

Рабочий контракт:
- выход только в формате (id, text)
- preamble -> Art0
- "Primo confirmant..." -> Art1
- далее статьи -> ArtN по внутренним маркерам, если они видимы в тексте
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import read_source_file, validate_segments


PREAMBLE_START_RE = re.compile(
    r"In\s+Christi\s+nomine\s+et\s+individue\s+Trinitatis",
    re.IGNORECASE,
)
PRIMO_RE = re.compile(r"Primo\s+confirmant\s+nobis", re.IGNORECASE)
ARTICLE_MARKER_RE = re.compile(r"(?m)^\s*[•*'\"]?\s*(\d(?: ?\d){0,2})\s*\.\s*(?=\S)")
CF_REFERENCE_RE = re.compile(r"^\s*\d(?: ?\d){0,2}\s*\.\s*Cf\.", re.IGNORECASE)
FOOTNOTE_LINE_RE = re.compile(r"^\s*\d+\)\s")
FOLIO_LINE_RE = re.compile(r"\b(?:Folio|Polio|Foho)\b", re.IGNORECASE)

CLOSING_BODY_MARKERS = [
    "perpetuo ei mandamus",
    "Sigffinum ffratris Martini Petri de Oros",
    "Et nos frater Elionus",
]

HEADER_PATTERNS = [
    re.compile(r"^\s*CONSTITUC", re.IGNORECASE),
    re.compile(r"^\s*CO\s*NS\s*TI", re.IGNORECASE),
    re.compile(r"^\s*BA\s*I?U?L?I?E?\s+MIRAB", re.IGNORECASE),
    re.compile(r"^\s*B\s*JI\s*I\s*UL", re.IGNORECASE),
]


def _normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = line.replace("\u3000", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    if not line:
        return True
    if FOLIO_LINE_RE.search(line):
        return True
    if FOOTNOTE_LINE_RE.match(line):
        return True
    if CF_REFERENCE_RE.match(line):
        return True
    if any(p.search(line) for p in HEADER_PATTERNS):
        return True
    if "Villanueva" in line and "Viage" in line:
        return True
    if len(line) >= 8 and len(re.sub(r"[^A-ZÁÉÍÓÚÜÑ ]", "", line)) >= max(6, int(len(line) * 0.6)):
        return True
    return False


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"(\w)[-‐-‒–—]\s+(\w)", r"\1\2", text)
    text = re.sub(r"[•*'\"]?\s*(?:Folio|Polio|Foho)\s+[^.]*\.", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bft\d{3,4}\b", " ", text)
    text = re.sub(r"\bCf\.\s*Consuetudines[^.]*\.", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Villanueva[^.]*\.", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Viage[^.]*\.", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slice_legal_core(text: str) -> str:
    m = PREAMBLE_START_RE.search(text)
    if not m:
        return ""
    start = m.start()
    end = len(text)
    for marker in CLOSING_BODY_MARKERS:
        idx = text.find(marker, start)
        if idx != -1:
            end = min(end, idx)
    return text[start:end]


def _strip_noise_blocks(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if _is_noise_line(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(
        r"(?mis)^\s*\d(?: ?\d){0,2}\s*\.\s*Cf\..*?(?=^\s*(?:\d(?: ?\d){0,2}\s*\.|\Z))",
        "",
        text,
    )
    text = re.sub(r"(?m)^\s*13-?t\.?\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _parse_articles(cleaned_core: str, source_name: str, min_words: int) -> List[Tuple[str, str]]:
    primo_match = PRIMO_RE.search(cleaned_core)
    if not primo_match:
        return []

    segments: List[Tuple[str, str]] = []

    preamble_text = clean_text(cleaned_core[:primo_match.start()])
    if len(preamble_text.split()) >= min_words:
        segments.append((f"{source_name}_Preamble", preamble_text))

    body = cleaned_core[primo_match.start():]
    markers = list(ARTICLE_MARKER_RE.finditer(body))

    if not markers:
        primo_text = clean_text(body)
        if len(primo_text.split()) >= min_words:
            segments.append((f"{source_name}_1", primo_text))
        return segments

    primo_text = clean_text(body[:markers[0].start()])
    if len(primo_text.split()) >= min_words:
        segments.append((f"{source_name}_1", primo_text))

    seen_numbers = {0, 1}
    for i, m in enumerate(markers):
        num = int(m.group(1).replace(" ", ""))
        if num <= 1 or num in seen_numbers:
            continue
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        article_text = clean_text(body[start:end])
        if not article_text:
            continue
        if len(article_text.split()) >= min_words:
            segments.append((f"{source_name}_{num}", article_text))
            seen_numbers.add(num)

    return segments


def segment_miravet(
    text: str,
    source_name: str = "ObychaiMiraveta1319Fix",
    min_words: int = 10,
) -> List[Tuple[str, str]]:
    legal_core = _slice_legal_core(text)
    if not legal_core:
        return []
    cleaned_core = _strip_noise_blocks(legal_core)
    return _parse_articles(cleaned_core, source_name=source_name, min_words=min_words)


def segment_miravet_unified(source_file, source_name):
    """
    Unified segmenter for Miravet.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "ObychaiMiraveta1319Fix".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_segments = segment_miravet(text, source_name=source_name, min_words=10)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/ObychaiMiraveta1319Fix_v2.txt"),
        Path("data/ObychaiMiraveta1319Fix.txt"),
        Path("ObychaiMiraveta1319Fix_v2.txt"),
        Path("/mnt/data/ObychaiMiraveta1319Fix_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_miravet_unified(src, "ObychaiMiraveta1319Fix")
    print(f"ObychaiMiraveta1319Fix: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()