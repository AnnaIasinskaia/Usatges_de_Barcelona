#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Recognoverunt Proceres (Barcelona privilege, 1283).

Рабочий контракт:
- выход только в формате (id, text)
- id в едином стиле: RecognovrentProceres12831284_ArtN
- опора на латинскую нумерацию [I]...[CXVI]
- каталонский параллельный текст и редакционный шум отбрасываются
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .seg_common import read_source_file, validate_segments


_ARTICLE_RE = re.compile(r"(?m)^[\s\u3000]*\[([IVXLCDM]+)\]\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_FOOTNOTE_RE = re.compile(r"^\s*\d+\s+[A-ZÁÉÍÓÚ]")

_ROMAN_VALUES: Dict[str, int] = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

_LATIN_WORDS = {
    "quod", "item", "si", "nisi", "cum", "vel", "et", "non", "que", "est", "sunt",
    "potest", "quilibet", "aliquis", "dominus", "uxor", "mariti", "bona", "civis",
    "concedimus", "consuetudo", "emphiteosim", "debet", "debitor", "creditor",
    "hereditas", "testamentum", "dotem", "vicarius", "baiulus", "capitur", "heres",
    "mulier", "mortem", "causa", "instrumentum", "fructus", "appellatur",
    "venditione", "habere", "petere", "fideiussor", "fideiussorem", "prediorum",
    "predicte", "predicatas", "similiter", "recognoverunt", "proceres",
    "barchinone", "consuetudinem", "legitima", "hereditatem", "sponsalicio",
}

_CATALAN_WORDS = {
    "encara", "dels", "del", "muller", "veguer", "bens", "ciutada", "senyor",
    "puscha", "nuyl", "nengu", "hom", "aquells", "promens", "costuma", "fruyts",
    "creador", "deuma", "luernes", "foriscacio", "ciutadans", "ciutat", "barcelona",
    "sagrament", "apell", "deutor", "cascun", "vehi", "paret", "fembra", "vassayl",
    "atorgam", "costumes", "promeya", "lexar", "deu", "aço", "axo", "aquella",
    "pagat", "paga", "fferma", "meylor", "bovatge", "leuda", "usatges",
}

_SKIP_CONTAINS = [
    "Pedro II el Grande concede",
    "Original, en el Archivo",
    "Copia en versión catalana",
    "reseñado por",
    "Publicado por",
    "Versión publicada por",
    "En nom Jesu Crist.",
]


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


def _line_scores(line: str) -> Tuple[int, int]:
    words = re.findall(r"[A-Za-zÀ-ÿ']+", line.lower())
    latin = sum(w in _LATIN_WORDS for w in words)
    catalan = sum(w in _CATALAN_WORDS for w in words)

    if re.match(r"^\d+\.", line.strip()):
        catalan += 3

    if any(ch in line.lower() for ch in ["ç", "·"]):
        catalan += 1

    return latin, catalan


def _is_catalan_line(line: str) -> bool:
    latin, catalan = _line_scores(line)
    return catalan >= latin + 2 or re.match(r"^\d+\.", line.strip()) is not None


def _drop_global_noise(lines: List[str]) -> List[str]:
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
        out.append(s)
    return out


def _extract_latin_preamble(text: str) -> str:
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return ""

    preamble_raw = text[:matches[0].start()]
    lines = _drop_global_noise(preamble_raw.splitlines())

    latin_lines: List[str] = []
    for s in lines:
        if _is_catalan_line(s):
            continue
        latin_lines.append(s)

    return clean_text(" ".join(latin_lines))


def _extract_latin_from_block(block: str) -> str:
    """
    Берёт один блок между [ROMAN]-маркерами и оставляет только латинскую часть.
    Как только началась каталонская колонка/перевод, обрываем блок.
    """
    block = re.sub(r"^\s*\[[IVXLCDM]+\]\s*", "", block, flags=re.IGNORECASE)
    lines = _drop_global_noise(block.splitlines())

    kept: List[str] = []
    for s in lines:
        if _is_catalan_line(s):
            break
        kept.append(s)

    text = clean_text(" ".join(kept))
    text = re.sub(r"\bCap\.\s*[IVXLCDM]+\.*\s*$", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def segment_privileges(text: str, source_name: str = "", debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Recognoverunt Proceres into preamble + numbered Latin articles.

    Возвращает
    ----------
    list[tuple[str, str]]
        Список сегментов в формате (segment_id, segment_text).
    """
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    preamble = _extract_latin_preamble(text)
    if preamble:
        prefix = source_name or "RecognovrentProceres12831284"
        segments.append((f"{prefix}_Preamble", preamble))

    prefix = source_name or "RecognovrentProceres12831284"

    for i, match in enumerate(matches):
        roman = match.group(1).upper()
        art_num = roman_to_int(roman)

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        latin_text = _extract_latin_from_block(block)
        if not latin_text:
            continue

        seg_id = f"{prefix}_Art{art_num}"
        segments.append((seg_id, latin_text))

        if debug and len(segments) <= 10:
            print(f"{seg_id}: {latin_text[:100]}")

    return segments


def segment_privileges_unified(source_file, source_name):
    """
    Unified segmenter for Recognoverunt Proceres.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "RecognovrentProceres12831284".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_segments = segment_privileges(text, source_name=source_name, debug=False)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/RecognovrentProceres12831284_v2.txt"),
        Path("RecognovrentProceres12831284_v2.txt"),
        Path("/mnt/data/RecognovrentProceres12831284_v2.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_privileges_unified(src, "RecognovrentProceres12831284")
    print(f"RecognovrentProceres12831284: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()