#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable segmenter for Corpus Juris Civilis (Digesta / Pandectae).

Подход:
- максимально простой line-based parser;
- regex только на структурном номере Dig., без шумового слоя;
- дубликаты id схлопываются: сохраняется самый длинный текст;
- title-headers вида Dig. x.y.0. не становятся сегментами;
- unified-вход без повторной лишней обвязки.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


_DIG_HEADER_RE = re.compile(
    r"^\s*Dig\.\s*([0-9]+\.[0-9]+\.[0-9]+(?:pr|\.[0-9]+)?)\.?\s*$",
    re.IGNORECASE,
)

_AUTHOR_LINE_RE = re.compile(
    r"^[A-Z][A-Za-z\-]+\s+.+$"
)

_SKIP_PREFIXES = (
    "DOMINI NOSTRI",
    "IURIS ENUCLEATI",
    "DIGESTORUM SEU",
    "LIBER ",
)

_STOP_MARKERS = (
    "INDEX",
    "APPENDIX",
    "APPENDICES",
    "GLOSSARIUM",
    "ERRATA",
)


def _safe_str(obj) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return str(obj)
    except Exception:
        return ""


def _normalize_dig_num(num: str) -> str:
    return "".join(ch for ch in _safe_str(num).strip() if not ch.isspace())


def _is_title_header_num(dig_num: str) -> bool:
    # Dig. 1.1.0. = title header, not a comparable fragment
    return dig_num.endswith(".0")


def _make_segment_id(source_name: str, dig_num: str) -> str:
    return f"{source_name}_Dig_{_normalize_dig_num(dig_num)}"


def _is_page_line(s: str) -> bool:
    s = s.strip()
    return s.isdigit() and 1 <= len(s) <= 4


def _is_noise_line(line: str) -> bool:
    s = _safe_str(line).strip()
    if not s:
        return True

    if _is_page_line(s):
        return True

    upper = s.upper()
    if upper.startswith(_SKIP_PREFIXES):
        return True

    if any(marker in upper for marker in _STOP_MARKERS):
        return True

    return False


def _extract_author_and_body(lines: List[str]) -> Tuple[Optional[str], List[str]]:
    if not lines:
        return None, []

    first = lines[0].strip()
    if _AUTHOR_LINE_RE.match(first) and not first.upper().startswith("DIG."):
        return first, lines[1:]
    return None, lines


def _finalize_segment(
    out: Dict[str, str],
    source_name: str,
    dig_num: Optional[str],
    body_lines: List[str],
) -> None:
    if not dig_num or _is_title_header_num(dig_num):
        return

    cleaned_lines: List[str] = []
    for line in body_lines:
        s = _safe_str(line).strip()
        if not s:
            continue
        if _is_noise_line(s):
            continue
        if _DIG_HEADER_RE.match(s):
            continue
        cleaned_lines.append(s)

    if not cleaned_lines:
        return

    author_line, remaining = _extract_author_and_body(cleaned_lines)
    body_text = clean_text(" ".join(remaining))
    if author_line:
        seg_text = clean_text(f"{author_line} {body_text}")
    else:
        seg_text = body_text

    if len(seg_text.split()) < 5:
        return

    seg_id = _make_segment_id(source_name, dig_num)

    prev = out.get(seg_id)
    if prev is None or len(seg_text) > len(prev):
        out[seg_id] = seg_text


def segment_corpus_juris(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Устойчивая структурная сегментация Corpus Juris Civilis:
    один Dig.-фрагмент = один сегмент.

    Возвращает list[(segment_id, segment_text)].
    """
    text = _safe_str(text)
    if not text.strip():
        return []

    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if _DIG_HEADER_RE.match(_safe_str(line).strip()):
            start_idx = i
            break

    seg_map: Dict[str, str] = {}
    current_num: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in lines[start_idx:]:
        line = _safe_str(raw_line).rstrip("\n")
        stripped = line.strip()

        if not stripped:
            if current_num is not None:
                current_lines.append("")
            continue

        m = _DIG_HEADER_RE.match(stripped)
        if m:
            _finalize_segment(seg_map, source_name, current_num, current_lines)
            current_num = _normalize_dig_num(m.group(1))
            current_lines = []
            continue

        upper = stripped.upper()
        if current_num is not None and any(marker in upper for marker in _STOP_MARKERS):
            break

        if current_num is not None:
            current_lines.append(stripped)

    _finalize_segment(seg_map, source_name, current_num, current_lines)

    def _sort_key(item: Tuple[str, str]):
        seg_id = item[0]
        dig = seg_id.split("_Dig_", 1)[-1]
        head = dig[:-2] if dig.endswith("pr") else dig
        parts = []
        for p in head.split("."):
            try:
                parts.append(int(p))
            except Exception:
                parts.append(10**9)
        if dig.endswith("pr"):
            parts.append(-1)
        return tuple(parts) + (seg_id,)

    segments = sorted(seg_map.items(), key=_sort_key)
    return validate_segments(segments, source_name)


def segment_corpus_juris_unified(source_file, source_name):
    """
    Unified segmenter for Corpus Juris Civilis.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "CorpusJuris".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_corpus_juris(text, source_name)


def main() -> None:
    candidates = [
        Path("data/Corpus_Juris_Civilis_v2.txt"),
        Path("Corpus_Juris_Civilis_v2.txt"),
        Path("/mnt/data/Corpus_Juris_Civilis_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_corpus_juris_unified(src, "CorpusJuris")
    print(f"CorpusJuris: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()