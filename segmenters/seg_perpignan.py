#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Customs of Perpignan (1243-1246).

Design:
- extract only the Perpignan source from a larger compilation file;
- keep the Latin prologue only (drop editorial intro and Catalan translation);
- segment the 69 Latin chapters [I]...[LXIX];
- segment the 10 Latin tithes/first-fruits usages that follow the main corpus;
- ignore the next source in the compilation (e.g. Recognoverunt Proceres).

Unified output:
    list[(segment_id, segment_text)]

IDs:
    CostumsDePerpinya_Preamble
    CostumsDePerpinya_1 ... CostumsDePerpinya_69
    CostumsDePerpinya_Decima1 ... CostumsDePerpinya_Decima10
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .seg_common import read_source_file, validate_segments

_ROMAN_VALUES: Dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}

_ARTICLE_RE = re.compile(r"(?m)^\s*\[([IVXLCDM]+)\]\s*")
_MAIN_HEADER_RE = re.compile(r"35\.\s+COSTUMBRES\s+DE\s+PERPIÑ[ÁA]N", re.IGNORECASE)
_NEXT_SOURCE_RE = re.compile(r"(?m)^\s*36\.\s+RECOGNOVERUNT\s+PROCERES\b", re.IGNORECASE)
_END_MAIN_RE = re.compile(r"Quas\s+predictas\s+consuetudines", re.IGNORECASE)
_SECTION_II_RE = re.compile(r"(?m)^\s*II\s*(?:\d+)?\s*$")
_FOOTNOTE_LINE_RE = re.compile(r"^\s*\d+\s+[A-ZÁÉÍÓÚÀÈÌÒÙÜÏÇÑ]")
_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_MULTI_SPACE_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")

_LATIN_WORDS = {
    "item", "nullus", "nulla", "nullum", "quod", "si", "nisi", "cum", "vel", "et",
    "non", "debet", "debent", "potest", "possunt", "consuetudo", "consuetum",
    "habitator", "habitatores", "perpinyani", "perpiniani", "baiulus", "vicarius",
    "dominus", "creditor", "debitor", "decima", "primicia", "villa", "placitare",
    "iudicari", "firmancia", "pignus", "querimonia", "causa", "captus", "vendi",
    "testamentum", "domum", "vinum", "oleum", "bladi", "carnes", "foriscapium",
    "consules", "universitatis", "privilegia", "libertates", "immunitates",
}

_CATALAN_WORDS = {
    "los", "les", "homens", "homes", "perpinya", "costumes", "costuma", "veguer",
    "balle", "senyor", "bens", "aquel", "aquell", "aquesta", "aço", "aixo",
    "deu", "deuen", "pot", "poden", "playdeiar", "jutjats", "fermances", "penyora",
    "clam", "questio", "cort", "vila", "vi", "oli", "blat", "forn", "flaquer",
    "juheus", "carn", "strayn", "ven", "venut", "pagua", "forescapi", "dret",
}

_SKIP_CONTAINS = [
    "Compilaciones de derecho municipal",
    "Jaime I confirma las costumbres",
    "Original, en el Archivo Municipal",
    "El texto se conserva en tres cartularios",
    "Aunque el texto carece de la cláusula",
    "Las primeras ediciones del texto son",
    "Massot-Reynier",
    "Valls-Taberner",
    "García Edo",
    "Pedro II el Grande concede",
    "Original, en el Archivo de la Corona",
    "35. COSTUMBRES DE PERPIÑÁN",
    "***",
    "Els Costums de",
    "Les Coutumes de Perpignan",
    "de Perpinyà, pp.",
    "de Perpignan, pp.",
]

_INLINE_FOOTNOTE_TAIL_RE = re.compile(r"\bde\s+Perpiny[aà],\s*pp\.?\s*\d+(?:-\d+)?\.?$", re.IGNORECASE)


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
    text = re.sub(r"\[(\w+)\]", r"\1", text)
    text = _INLINE_FOOTNOTE_TAIL_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip(" ;,\n\t")


def _line_scores(line: str) -> Tuple[int, int]:
    words = re.findall(r"[A-Za-zÀ-ÿ·']+", line.lower())
    latin = sum(w in _LATIN_WORDS for w in words)
    catalan = sum(w in _CATALAN_WORDS for w in words)
    if any(ch in line for ch in ("ç", "·", "l’", "qu’", "d’")):
        catalan += 1
    return latin, catalan


def _is_catalan_line(line: str) -> bool:
    latin, catalan = _line_scores(line)
    s = line.strip().lower()
    if not s:
        return False
    if re.match(r"^los\s+homens\b", s):
        return True
    return catalan >= latin + 2


def _drop_global_noise(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        s = raw.strip().replace("\xa0", " ").replace("\u3000", " ")
        if not s:
            continue
        if _PAGE_RE.match(s):
            continue
        if _FOOTNOTE_LINE_RE.match(s):
            continue
        if any(mark in s for mark in _SKIP_CONTAINS):
            continue
        out.append(s)
    return out


def _extract_perpignan_only(text: str) -> str:
    start_match = _MAIN_HEADER_RE.search(text)
    if not start_match:
        return text
    start = start_match.start()

    next_match = _NEXT_SOURCE_RE.search(text, start)
    end = next_match.start() if next_match else len(text)
    return text[start:end]


def _split_main_and_decima(text: str) -> Tuple[str, str]:
    end_main = _END_MAIN_RE.search(text)
    if not end_main:
        return text, ""

    section_ii = _SECTION_II_RE.search(text, end_main.end())
    if not section_ii:
        return text, ""

    return text[:end_main.start()], text[section_ii.start():]


def _extract_latin_preamble(main_text: str) -> str:
    matches = list(_ARTICLE_RE.finditer(main_text))
    if not matches:
        return ""

    preamble_raw = main_text[:matches[0].start()]
    lines = _drop_global_noise(preamble_raw.splitlines())

    latin_lines: List[str] = []
    for s in lines:
        if s == "I":
            continue
        if _is_catalan_line(s):
            continue
        latin_lines.append(s)

    return clean_text(" ".join(latin_lines))


def _extract_latin_from_block(block: str) -> str:
    block = re.sub(r"^\s*\[([IVXLCDM]+)\]\s*", "", block, flags=re.IGNORECASE)
    lines = _drop_global_noise(block.splitlines())

    kept: List[str] = []
    for s in lines:
        if _is_catalan_line(s):
            break
        kept.append(s)

    return clean_text(" ".join(kept))


def _segment_numbered_blocks(text: str, prefix: str, id_prefix: str = "") -> List[Tuple[str, str]]:
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        roman = match.group(1).upper()
        num = roman_to_int(roman)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        latin_text = _extract_latin_from_block(block)
        if not latin_text:
            continue

        seg_id = f"{prefix}_{id_prefix}{num}"
        segments.append((seg_id, latin_text))

    return segments


def segment_perpignan(text: str, source_name: str = "CostumsDePerpinya", debug: bool = False) -> List[Tuple[str, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _extract_perpignan_only(text)
    main_text, decima_text = _split_main_and_decima(text)

    segments: List[Tuple[str, str]] = []

    preamble = _extract_latin_preamble(main_text)
    if preamble:
        segments.append((f"{source_name}_Preamble", preamble))

    segments.extend(_segment_numbered_blocks(main_text, source_name))
    segments.extend(_segment_numbered_blocks(decima_text, source_name, id_prefix="Decima"))

    best: Dict[str, str] = {}
    order: List[str] = []
    for seg_id, seg_text in segments:
        prev = best.get(seg_id)
        if prev is None:
            best[seg_id] = seg_text
            order.append(seg_id)
        elif len(seg_text) > len(prev):
            best[seg_id] = seg_text

    final_segments = [(seg_id, best[seg_id]) for seg_id in order if seg_id in best]

    if debug:
        print(f"Segments: {len(final_segments)}")
        for seg_id, seg_text in final_segments[:6]:
            print(seg_id, '->', seg_text[:120])
        print('...')
        for seg_id, seg_text in final_segments[-6:]:
            print(seg_id, '->', seg_text[:120])

    return final_segments


def segment_perpignan_unified(source_file, source_name):
    text = read_source_file(source_file)
    raw_segments = segment_perpignan(text, source_name=source_name, debug=False)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/Customs_of_Perpignan_v2.txt"),
        Path("Customs_of_Perpignan_v2.txt"),
        Path("/mnt/data/Customs_of_Perpignan_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_perpignan_unified(src, "CostumsDePerpinya")
    print(f"CostumsDePerpinya: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()
