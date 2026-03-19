"""Segmenter for Exceptiones Legum Romanorum Petri."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, validate_segments, read_source_file


# Структура документа:
#   Prologus.
#   Liber primus.
#   Cap. 1. De ...
#   Cap. 2. ...
#
# Сегментируем по главам внутри книги.
# В id выносим фактические номера книги и главы из документа:
#   ExceptPetri_L1_C1
#   ExceptPetri_L1_C2
#   ExceptPetri_L2_C14
#
# При этом "Prologus" сохраняем отдельно:
#   ExceptPetri_Prologus

_LIBER_RE = re.compile(
    r"(?im)^\s*Liber\s+(primus|secundus|tertius|quartus|quintus|sextus|septimus|octavus|nonus|decimus|\w+)\.?\s*$"
)

_CAP_RE = re.compile(
    r"(?im)^\s*Cap\.\s*(\d+)\.?\s*(.*)$"
)

_PROLOGUS_RE = re.compile(
    r"(?im)^\s*P\s*r\s*o\s*l\s*o\s*g\s*u\s*s\.?\s*$|^\s*Prologus\.?\s*$"
)

_PAGE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_RUNNING_RE = re.compile(r"^\s*Liber\s+[ivxlcdm0-9]+\s*[\.,]?\s*\d*\s*$", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"^\s*\d+\)\s+")
_PAREN_REF_RE = re.compile(r"^\s*\(.*\)\s*$")
_INLINE_FOOTNOTE_NUM_RE = re.compile(r"\b\d+\)")
_HYPHEN_SPLIT_RE = re.compile(r"(\w)-\s+(\w)")

_ORDINALS = {
    "primus": 1,
    "secundus": 2,
    "tertius": 3,
    "quartus": 4,
    "quintus": 5,
    "sextus": 6,
    "septimus": 7,
    "octavus": 8,
    "nonus": 9,
    "decimus": 10,
    # fallback OCR variants
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
}


def _roman_or_ordinal_to_int(token: str) -> int:
    t = token.strip().lower().rstrip(".")
    if t in _ORDINALS:
        return _ORDINALS[t]
    raise ValueError(f"Unsupported Liber token: {token!r}")


def _clean_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _PAGE_RE.match(s):
            continue
        if _RUNNING_RE.match(s):
            continue
        if _FOOTNOTE_RE.match(s):
            continue
        if _PAREN_REF_RE.match(s):
            continue
        lines.append(s)

    out = " ".join(lines)
    out = _INLINE_FOOTNOTE_NUM_RE.sub("", out)
    out = _HYPHEN_SPLIT_RE.sub(r"\1\2", out)
    out = clean_text(out)
    return out


def _segment_prologus(text: str, source_name: str) -> List[Tuple[str, str]]:
    m = _PROLOGUS_RE.search(text)
    if not m:
        return []

    # Пролог заканчивается перед первым Liber
    lm = _LIBER_RE.search(text, m.end())
    end = lm.start() if lm else len(text)
    block = text[m.end():end].strip()
    cleaned = _clean_block(block)
    if not cleaned:
        return []
    return [(f"{source_name}_Prologus", cleaned)]


def segment_exceptiones_petri(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Чистая структурная сегментация Exceptiones Petri:
    один сегмент = одна глава внутри конкретной книги.

    IDs:
      ExceptPetri_Prologus
      ExceptPetri_L1_C1
      ExceptPetri_L1_C2
      ExceptPetri_L2_C14
    """
    segments: List[Tuple[str, str]] = []

    # 1) Пролог отдельно
    segments.extend(_segment_prologus(text, source_name))

    # 2) Книги
    liber_matches = list(_LIBER_RE.finditer(text))
    if not liber_matches:
        return validate_segments(segments, source_name)

    for i, lm in enumerate(liber_matches):
        try:
            liber_no = _roman_or_ordinal_to_int(lm.group(1))
        except ValueError:
            continue

        start = lm.end()
        end = liber_matches[i + 1].start() if i + 1 < len(liber_matches) else len(text)
        liber_block = text[start:end].strip()
        if not liber_block:
            continue

        cap_matches = list(_CAP_RE.finditer(liber_block))
        if not cap_matches:
            continue

        for j, cm in enumerate(cap_matches):
            cap_no = int(cm.group(1))
            cap_head = (cm.group(2) or "").strip()

            c_start = cm.end()
            c_end = cap_matches[j + 1].start() if j + 1 < len(cap_matches) else len(liber_block)
            cap_body = liber_block[c_start:c_end].strip()

            cleaned_body = _clean_block(cap_body)
            if cap_head:
                cleaned = clean_text(f"{cap_head} {cleaned_body}")
            else:
                cleaned = cleaned_body

            if not cleaned:
                continue

            seg_id = f"{source_name}_L{liber_no}_C{cap_no}"
            segments.append((seg_id, cleaned))

    return validate_segments(segments, source_name)


def segment_exceptiones_petri_unified(source_file, source_name):
    """
    Унифицированная сегментация Exceptiones Legum Romanorum Petri.
    Сегментер сам читает файл и возвращает list[(id, text)].
    """
    text = read_source_file(source_file)
    raw_segments = segment_exceptiones_petri(text, source_name)
    return validate_segments(raw_segments, source_name)


if __name__ == "__main__":
    candidates = [
        Path("data/Exeptionis_Legum_Romanorum_Petri_v2.txt"),
    ]

    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        print("Not found. Expected one of:")
        for c in candidates:
            print(f"  - {c}")
        raise SystemExit(1)

    text = read_source_file(p)
    segs = segment_exceptiones_petri(text, "ExceptPetri")

    print(f"ExceptPetri: {len(segs)} segments")
    print("Expected structural unit: prologus + one segment per chapter within each Liber")
    print()

    if segs:
        print("First 8 segments:")
        for sid, txt in segs[:8]:
            print(f"  {sid}: {txt[:160]}")
        print()

        print("Last 8 segments:")
        for sid, txt in segs[-8:]:
            print(f"  {sid}: {txt[:160]}")