#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default example segmenter.

Назначение:
- служить простым эталонным сегментером для новых источников;
- показывать рекомендуемый стиль unified-сегментеров;
- давать предсказуемую базовую сегментацию без source-specific эвристик.

Логика:
1. режем текст на абзацы по пустым строкам;
2. выбрасываем очевидный шум и apparatus-like строки;
3. нормализуем текст;
4. склеиваем соседние короткие абзацы в сегменты разумного размера;
5. возвращаем strict list[(id, text)].
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, is_apparatus_line, read_source_file, validate_segments


_BLANK_BLOCK_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"^[A-ZА-ЯIVXLCM0-9][A-ZА-ЯIVXLCM0-9\s\-\.,:;]{3,}$")
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")


def _split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = _BLANK_BLOCK_RE.split(text)

    paragraphs: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue

        joined = " ".join(lines)
        joined = _MULTI_SPACE_RE.sub(" ", joined).strip()
        if joined:
            paragraphs.append(joined)

    return paragraphs


def _is_noise_paragraph(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if _PAGE_RE.match(s):
        return True
    if len(s) < 20:
        return True
    if is_apparatus_line(s):
        return True
    return False


def _looks_like_heading(text: str) -> bool:
    s = text.strip()
    if len(s) > 120:
        return False
    return bool(_HEADING_RE.match(s))


def _merge_paragraphs(
    paragraphs: List[str],
    source_name: str,
    *,
    target_words: int = 140,
    min_words: int = 20,
    max_words: int = 220,
) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []

    current_parts: List[str] = []
    current_words = 0
    seg_idx = 1

    def flush() -> None:
        nonlocal current_parts, current_words, seg_idx
        if not current_parts:
            return
        text = clean_text(" ".join(current_parts))
        if len(text.split()) >= min_words:
            segments.append((f"{source_name}_S{seg_idx}", text))
            seg_idx += 1
        current_parts = []
        current_words = 0

    for para in paragraphs:
        para = clean_text(para)
        if not para:
            continue

        para_words = len(para.split())

        if _looks_like_heading(para):
            flush()
            # заголовок не выбрасываем: он будет приклеен к следующему абзацу
            current_parts.append(para)
            current_words = para_words
            continue

        if current_words == 0:
            current_parts.append(para)
            current_words = para_words
            continue

        if current_words >= target_words or current_words + para_words > max_words:
            flush()
            current_parts.append(para)
            current_words = para_words
            continue

        current_parts.append(para)
        current_words += para_words

    flush()
    return segments


def segment_default(
    text: str,
    source_name: str,
    *,
    target_words: int = 140,
    min_words: int = 20,
    max_words: int = 220,
) -> List[Tuple[str, str]]:
    """
    Простая базовая сегментация по абзацам.

    IDs:
      <source_name>_S1
      <source_name>_S2
      ...

    Это fallback / reference-сегментер без source-specific правил.
    """
    paragraphs = _split_into_paragraphs(text)
    paragraphs = [p for p in paragraphs if not _is_noise_paragraph(p)]

    raw_segments = _merge_paragraphs(
        paragraphs,
        source_name,
        target_words=target_words,
        min_words=min_words,
        max_words=max_words,
    )
    return validate_segments(raw_segments, source_name)


def segment_default_unified(source_file, source_name):
    """
    Unified default segmenter.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name.

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_default(
        text,
        source_name,
        target_words=140,
        min_words=20,
        max_words=220,
    )


def main() -> None:
    candidates = [
        Path("data/Bastardas_Usatges_de_Barcelona_djvu.txt"),
        Path("data/Bastardas Usatges de Barcelona_djvu.txt"),
        Path("Bastardas_Usatges_de_Barcelona_djvu.txt"),
        Path("/mnt/data/Bastardas_Usatges_de_Barcelona_djvu.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_default_unified(src, "SampleSource")
    print(f"SampleSource: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()