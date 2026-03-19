"""Segmenter for Lex Visigothorum (Liber Iudiciorum)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, validate_segments, read_source_file


# Основная структура в тексте:
#   LIBER PRIMUS.
#   I. TITULUS: DE LEGISLATORE.
#   ...
#   1,1,1. Quod sit artificium condendarum legum.
#   ...
#   I, 1, 2. II. Quo modo uti debeat artifex legum.
#
# То есть фактический сопоставимый уровень — отдельная lex с номером вида:
#   1,1,1
#   I, 1, 2
#   II, 1, 1
#
# Для ID нормализуем к арабской форме:
#   1,1,1 -> 1.1.1
#   I, 1, 2 -> 1.1.2
#
# В документе также много критического аппарата и OCR-мусора; его надо отсеивать.


_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IIII": 4, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "VIIII": 9, "IX": 9,
    "X": 10, "XI": 11, "XII": 12,
}

_LIBER_RE = re.compile(r"(?m)^\s*LIBER\s+([A-Z]+)\.?\s*$", re.IGNORECASE)
_TITULUS_RE = re.compile(r"(?m)^\s*([IVX]+)\.\s+TITULUS:\s*(.+?)\s*$", re.IGNORECASE)

# Леммы статей внутри корпуса:
# 1,1,1.
# I, 1, 2.
# II, 1, 1.
# IIII, 2, 5.
_LEX_RE = re.compile(
    r"(?m)^\s*(?P<book>[IVX]+|\d+)\s*,\s*(?P<title>\d+)\s*,\s*(?P<law>\d+)\.\s*(?P<head>.*)$",
    re.IGNORECASE,
)

# Строки критического аппарата и пагинации
_PAGE_RE = re.compile(r"^\s*\d{1,3}\s*$")
_RUNNING_HEADER_RE = re.compile(r"^\s*(?:LEX\s+VISIGOTH|LIBER\s+IUDIC|Recc\.|Erv\.|J?Recc\.|Mecc\.)", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"^\s*\d+\)\s+")
_LINEAPP_RE = re.compile(r"^\s*l\.\s*\d+", re.IGNORECASE)
_LIB_SECTION_RE = re.compile(r"^\s*Ll\.\s+Sect\.", re.IGNORECASE)
_EDITORIAL_RE = re.compile(r"^\s*(?:Inscriptio|Ad hunc librum|LIB\.\s+[IVX]+\.|TIT\.\s+[IVX]+\.|Codd\.)", re.IGNORECASE)


def _roman_to_int(token: str) -> int:
    token = token.strip().upper().rstrip(".")
    if token.isdigit():
        return int(token)
    if token not in _ROMAN_MAP:
        raise ValueError(f"Unsupported Roman numeral: {token}")
    return _ROMAN_MAP[token]


def _normalize_ref(book_token: str, title_token: str, law_token: str) -> Tuple[int, int, int]:
    return _roman_to_int(book_token), int(title_token), int(law_token)


def _make_segment_id(source_name: str, book_no: int, title_no: int, law_no: int) -> str:
    return f"{source_name}_{book_no}.{title_no}.{law_no}"


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _PAGE_RE.match(s):
        return True
    if _RUNNING_HEADER_RE.match(s):
        return True
    if _FOOTNOTE_RE.match(s):
        return True
    if _LINEAPP_RE.match(s):
        return True
    if _LIB_SECTION_RE.match(s):
        return True
    if _EDITORIAL_RE.match(s):
        return True
    # OCR-аппарат: очень мало букв при большой длине
    alpha = sum(ch.isalpha() for ch in s)
    if len(s) > 20 and alpha / max(1, len(s)) < 0.45:
        return True
    return False


def _clean_block_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _is_noise_line(s):
            continue
        # Убираем служебные вставки в начале строки, если осталась нумерация аппарата
        s = re.sub(r"^\d+\s+", "", s)
        out.append(s)
    return out


def segment_lex_visigothorum(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Чистая структурная сегментация Lex Visigothorum:
    один сегмент = одна lex с номером book.title.law

    Примеры ID:
      LexVisigoth_1.1.1
      LexVisigoth_1.1.2
      LexVisigoth_2.1.1
    """
    matches = list(_LEX_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        try:
            book_no, title_no, law_no = _normalize_ref(
                m.group("book"),
                m.group("title"),
                m.group("law"),
            )
        except ValueError:
            continue

        headline = (m.group("head") or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        raw_lines = [ln.rstrip() for ln in block.splitlines()]
        raw_lines = _clean_block_lines(raw_lines)

        body = clean_text(" ".join(raw_lines))
        if headline:
            seg_text = clean_text(f"{headline} {body}")
        else:
            seg_text = body

        if not seg_text:
            continue

        seg_id = _make_segment_id(source_name, book_no, title_no, law_no)
        segments.append((seg_id, seg_text))

    return validate_segments(segments, source_name)


def segment_lex_visigothorum_unified(source_file, source_name):
    """
    Унифицированная сегментация Lex Visigothorum.
    Сегментер сам читает файл и возвращает list[(id, text)].
    """
    text = read_source_file(source_file)
    raw_segments = segment_lex_visigothorum(text, source_name)
    return validate_segments(raw_segments, source_name)


if __name__ == "__main__":
    candidates = [
        Path("data/Lex_visigothorum_v2.txt"),
    ]

    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        print("Not found. Expected one of:")
        for c in candidates:
            print(f"  - {c}")
        raise SystemExit(1)

    text = read_source_file(p)
    segs = segment_lex_visigothorum(text, "LexVisigoth")

    print(f"LexVisigoth: {len(segs)} segments")
    print("Expected structural unit: one segment per numbered lex (book.title.law)")
    print()

    if segs:
        print("First 5 segments:")
        for sid, txt in segs[:5]:
            print(f"  {sid}: {txt[:140]}")
        print()

        print("Last 5 segments:")
        for sid, txt in segs[-5:]:
            print(f"  {sid}: {txt[:140]}")