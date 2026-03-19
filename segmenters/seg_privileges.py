#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Recognoverunt Proceres (Barcelona privilege, 1283).

Новая версия ориентирована на RecognovrentProceres12831284_v2.txt.
Главный принцип:
- выход только в формате (id, text)
- id в едином стиле: RecognovrentProceres12831284_ArtN
- опора на латинскую нумерацию [I]...[CXVI]
- каталонский параллельный текст и редакционный шум отбрасываются
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict


_ARTICLE_RE = re.compile(r'(?m)^[\s\u3000]*\[([IVXLCDM]+)\]\s*')
_MULTI_SPACE_RE = re.compile(r'\s+')
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\s+(\w)')
_PAGE_RE = re.compile(r'^\s*\d{1,4}\s*$')
_FOOTNOTE_RE = re.compile(r'^\s*\d+\s+[A-ZÁÉÍÓÚ]')

_ROMAN_VALUES: Dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}

# Упрощённые словари для отделения латинского блока от каталонского
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

    # Арабские пункты вроде "41." почти всегда каталонская колонка
    if re.match(r'^\d+\.', line.strip()):
        catalan += 3

    if any(ch in line.lower() for ch in ['ç', '·']):
        catalan += 1

    return latin, catalan


def _is_catalan_line(line: str) -> bool:
    latin, catalan = _line_scores(line)
    return catalan >= latin + 2 or re.match(r'^\d+\.', line.strip()) is not None


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

    text = clean_text(" ".join(latin_lines))
    return text


def _extract_latin_from_block(block: str) -> str:
    """
    Берёт один блок между [ROMAN]-маркерами и оставляет только латинскую часть.
    Как только началась каталонская колонка/перевод, обрываем блок.
    """
    block = re.sub(r'^\s*\[[IVXLCDM]+\]\s*', '', block, flags=re.IGNORECASE)
    lines = _drop_global_noise(block.splitlines())

    kept: List[str] = []
    for s in lines:
        if _is_catalan_line(s):
            break
        kept.append(s)

    text = clean_text(" ".join(kept))
    text = re.sub(r'\bCap\.\s*[IVXLCDM]+\.*\s*$', '', text, flags=re.IGNORECASE)
    text = clean_text(text)
    return text


def segment_privileges(text: str, source_name: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Новая сегментация Recognoverunt Proceres.

    IDs:
      RecognovrentProceres12831284_Art0   -> латинская преамбула
      RecognovrentProceres12831284_Art1   -> [I]
      ...
      RecognovrentProceres12831284_Art116 -> [CXVI]
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_ARTICLE_RE.finditer(text))

    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    preamble = _extract_latin_preamble(text)
    if preamble:
        segments.append((f"{source_name}_Art0", preamble))

    for idx, m in enumerate(matches):
        roman = m.group(1).upper()
        art_no = roman_to_int(roman)

        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]

        chapter_text = _extract_latin_from_block(block)
        if not chapter_text:
            continue

        segments.append((f"{source_name}_Art{art_no}", chapter_text))

        if debug and art_no <= 5:
            print(f"Art{art_no}: {chapter_text[:120]}")

    return segments


def analyze_and_save(
    text: str,
    output_file: str,
    source_name: str = "RecognovrentProceres12831284",
) -> List[Tuple[str, str]]:
    print("=" * 80)
    print("RECOGNOVERUNT PROCERES (1283) - CHAPTER SEGMENTATION")
    print("=" * 80)

    chapters = segment_privileges(text, source_name=source_name, debug=False)

    print(f"\nFound: {len(chapters)} segments")
    if chapters:
        print(f"First id: {chapters[0][0]}")
        print(f"Last id:  {chapters[-1][0]}")

    from collections import Counter
    ids = [cid for cid, _ in chapters]
    duplicates = [(n, c) for n, c in Counter(ids).items() if c > 1]

    if duplicates:
        print(f"\nDuplicates: {len(duplicates)}")
        for num, count in sorted(duplicates)[:5]:
            print(f"  {num}: {count} times")
    else:
        print("\nNo duplicates")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total chapters: {len(chapters)}\n")
        f.write("=" * 80 + "\n\n")
        for chapter_id, chapter_text in chapters:
            f.write("=" * 80 + "\n")
            f.write(f"{chapter_id}\n")
            f.write("=" * 80 + "\n")
            f.write(chapter_text)
            f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    return chapters


def segment_privileges_unified(source_file, source_name):
    """
    Unified-сегментация Privileges.
    """
    from .seg_common import read_source_file, validate_segments

    text = read_source_file(source_file)
    raw_segments = segment_privileges(text, source_name=source_name, debug=False)
    return validate_segments(raw_segments, source_name)


def main():
    candidates = [
        Path('data/RecognovrentProceres12831284_v2.txt'),
        Path('RecognovrentProceres12831284_v2.txt'),
        Path('/mnt/data/RecognovrentProceres12831284_v2.txt'),
    ]

    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        print("Error: source file not found. Tried:")
        for p in candidates:
            print(f"  - {p}")
        raise SystemExit(1)

    print(f"Processing {file_path}...")
    text = file_path.read_text(encoding='utf-8', errors='replace')
    docs = analyze_and_save(
        text,
        output_file='privileges_segmented.txt',
        source_name='RecognovrentProceres12831284',
    )

    if docs:
        print("\nFirst 5 segments:")
        for sid, txt in docs[:5]:
            print(f"  {sid}: {txt[:120]}")

        print("\nLast 5 segments:")
        for sid, txt in docs[-5:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == '__main__':
    main()