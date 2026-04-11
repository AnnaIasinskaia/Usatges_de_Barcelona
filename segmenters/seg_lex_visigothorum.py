#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Lex Visigothorum segmenter, migration version v3.

Design goals for this version:
- keep v2 behavior where it works;
- fix OCR-corrupted marker lines (esp. books 3 and 11);
- stay close to the old high-coverage segmenter;
- keep Novella laws instead of dropping them;
- drop books 13+ as OCR noise;
- treat ANTIQUA / FLAVIUS / NOVELLA as heading metadata, not body text;
- strip the editor-added pseudo-title conservatively;
- preserve physical ids like 2.5.3(4), but keep base-slot validation anchored
  to the printed law number before parentheses.
"""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .seg_common import clean_text, read_source_file, validate_segments

EXPECTED_IDS_BY_TITLE: Dict[Tuple[int, int], List[int]] = {
    (1, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (1, 2): [1, 2, 3, 4, 5, 6],
    (2, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    (2, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (2, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (2, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (2, 5): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    (3, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (3, 2): [1, 2, 3, 4, 5, 6, 7, 8],
    (3, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (3, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    (3, 5): [1, 2, 3, 4, 5],
    (3, 6): [1, 2, 3],
    (4, 1): [1, 2, 3, 4, 5, 6, 7],
    (4, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (4, 3): [1, 2, 3, 4],
    (4, 4): [1, 2, 3],
    (4, 5): [1, 2, 3, 4, 5, 6, 7],
    (5, 1): [1, 2, 3, 4],
    (5, 2): [1, 2, 3, 4, 5, 6, 7],
    (5, 3): [1, 2, 3, 4],
    (5, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    (5, 5): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (5, 6): [1, 2, 3, 4, 5, 6],
    (5, 7): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    (6, 1): [1, 2, 3, 4, 5, 6, 7],
    (6, 2): [1, 2, 3, 4, 5],
    (6, 3): [1, 2, 3, 4, 5, 6, 7],
    (6, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    (7, 1): [1, 2, 3, 4, 5],
    (7, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    (7, 3): [1, 2, 3, 4, 5, 6],
    (7, 4): [1, 2, 3, 4, 5, 6, 7],
    (7, 5): [1, 2, 3, 4, 5, 6, 7, 8],
    (7, 6): [1, 2, 3, 4, 5],
    (8, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    (8, 2): [1, 2, 3],
    (8, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    (8, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    (8, 5): [1, 2, 3, 4, 5, 6, 7, 8],
    (8, 6): [1, 2, 3],
    (9, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    (9, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (9, 3): [1, 2, 3, 4],
    (10, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    (10, 2): [1, 2, 3, 4, 5, 6],
    (10, 3): [1, 2, 3, 4, 5],
    (11, 1): [1, 2, 3, 4, 5, 6, 7, 8],
    (11, 2): [1, 2],
    (11, 3): [1, 2, 3, 4],
    (12, 1): [1, 2],
    (12, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    (12, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
}
EXPECTED_ID_SET: Set[Tuple[int, int, int]] = {
    (b, t, l) for (b, t), laws in EXPECTED_IDS_BY_TITLE.items() for l in laws
}

_ORDINAL_BOOK_MAP: Dict[str, int] = {
    "PRIMUS": 1,
    "SECUNDUS": 2,
    "TERTIUS": 3,
    "QUARTUS": 4,
    "QUINTUS": 5,
    "SEXTUS": 6,
    "SEPTIMUS": 7,
    "OCTAVUS": 8,
    "OCTABUS": 8,
    "OCTAUUS": 8,
    "NONUS": 9,
    "DECIMUS": 10,
    "UNDECIMUS": 11,
    "DUODECIMUS": 12,
}

_LEAD_TOKEN = r"(?P<booktok>[IVXLCDM1l|\\/№]+|\d+)"
_BOOK_RE = re.compile(r"^\s*Liber\s+(?P<book>[A-Za-z]+)\.?(?:\s+.+)?$", re.IGNORECASE)
_TITLE_RE = re.compile(
    rf"^\s*[\\|/№]*\s*{_LEAD_TOKEN}\.(?P<title>\d{{1,2}})\.\s+(?P<rest>.+?)\s*$",
    re.IGNORECASE,
)
_LAW_RE = re.compile(
    rf"^\s*\(?\s*[\\|/№]*\s*{_LEAD_TOKEN}\.(?P<title>\d{{1,2}})\.(?P<law>\d{{1,3}})(?:\((?P<altlaw>\d{{1,3}})\))?\.\s*\)?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_RUSSIAN_HEADING_RE = re.compile(r"^\s*Книга\s+|^\s*ТЕКСТ\.", re.IGNORECASE)
_FRAGMENTA_RE = re.compile(r"^\s*Fragmenta\s+Parisina\b", re.IGNORECASE)
_TIT_SHORT_RE = re.compile(r"^\s*Tit\.\s+", re.IGNORECASE)
_HEAD_LABEL_RE = re.compile(r"^(?:ANTIQUA\b|NOVELLA\b|LEX\b|ITEM\b)", re.IGNORECASE)
_FLAVIUS_RE = re.compile(r"^FLAVIUS\b[^.]{0,120}(?:REX|PRINCEPS|GLORIOSUS)\b\.?", re.IGNORECASE)
_PSEUDOTITLE_START_RE = re.compile(r"^(?:De|Quod|Qualis|Quo\s+modo|Ut|Ne|Si|Nullus|Qui|Quoties|Maritus|Mater|Pater|Femina)\b", re.IGNORECASE)
_BODY_START_RE = re.compile(r"^(?:Si|Siquis|Si\s+quis|Quod|Quodsi|Quicumque|Quoties|Qui|Nullus|Omnis|Maritus|Formandarum|Tunc|Erit|Venditio|Conmutatio|Romanus|Mater|Pater|Femina|Salutare|Pacta|Placita)\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


_MARKER_CYRILLIC_MAP = str.maketrans({
    "Х": "X", "х": "x",
    "І": "I", "і": "i",
    "Ѵ": "V", "ѵ": "v",
    "Ү": "Y", "ү": "y",
})


def _normalize_marker_line(line: str) -> str:
    s = line.translate(_MARKER_CYRILLIC_MAP)
    # OCR sometimes turns roman I into ! inside law/title markers: Х!.2.1., X!.1.1., etc.
    s = re.sub(r'^([IVXLCDMivxlcdmXx])!', r'\1I', s)
    s = re.sub(r'^([IVXLCDMivxlcdmXx]{2})!', r'\1I', s)
    return s



def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = re.sub(r"([A-Za-z])-\n\s*([A-Za-z])", r"\1\2", text)
    return text


def _clean_line(line: str) -> str:
    s = line.replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _contains_cyrillic(s: str) -> bool:
    return any("\u0400" <= ch <= "\u04FF" for ch in s)


def _roman_like_to_int(token: str) -> Optional[int]:
    tok = token.strip().upper().replace("J", "I").replace("1", "I")
    tok = tok.replace("X1I", "XII").replace("X1", "XI")
    tok = tok.replace("|", "I").replace("\\", "").replace("/", "").replace("№", "")
    tok = tok.replace("ILL", "III").replace("LLL", "III")
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not tok or any(ch not in values for ch in tok):
        return None
    total = 0
    prev = 0
    for ch in reversed(tok):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if total > 0 else None


def _book_token_to_int(token: str) -> Optional[int]:
    tok = token.strip().upper().replace("|", "I")
    if tok.isdigit():
        val = int(tok)
        return val if 1 <= val <= 12 else None
    if tok in _ORDINAL_BOOK_MAP:
        return _ORDINAL_BOOK_MAP[tok]
    val = _roman_like_to_int(tok)
    return val if val is not None and 1 <= val <= 12 else None


def _resolve_book_from_context(booktok: str, current_book: Optional[int]) -> Optional[int]:
    parsed = _book_token_to_int(booktok)
    if current_book is None:
        return parsed
    if parsed == current_book:
        return current_book
    tok = (booktok or "").strip()
    if any(ch in tok for ch in "|\\/№"):
        return current_book
    if current_book in {2, 3} and parsed in {1, 2, 3, 11}:
        return current_book
    if parsed is None:
        return current_book
    return parsed if 1 <= parsed <= 12 else None


def _looks_like_noise(line: str) -> bool:
    if not line:
        return True
    if _PAGE_NUMBER_RE.match(line):
        return True
    if _contains_cyrillic(line):
        return True
    if _RUSSIAN_HEADING_RE.match(line):
        return True
    if line.startswith("og ") or "RON" in line:
        return True
    return False


def _normalize_body_text(lines: List[str]) -> str:
    return re.sub(r"\s+", " ", clean_text(" ".join(lines))).strip()


def _strip_head_labels(text: str) -> Tuple[str, bool]:
    s = text.strip()
    changed = False
    while s:
        m = _HEAD_LABEL_RE.match(s)
        if m:
            changed = True
            s = s[m.end():].lstrip(" .:;,-")
            continue
        m = _FLAVIUS_RE.match(s)
        if m:
            changed = True
            s = s[m.end():].lstrip(" .:;,-")
            continue
        break
    return s.strip(), changed


def _looks_like_editorial_pseudotitle(line: str, next_line: Optional[str]) -> bool:
    s = line.strip()
    if not s:
        return False
    words = s.split()
    if len(words) > 18:
        return False
    parts = _SENTENCE_SPLIT_RE.split(s, maxsplit=1)
    if len(parts) >= 2:
        first = parts[0].strip()
        second = parts[1].strip()
        if first and second and len(first.split()) <= 16 and _BODY_START_RE.match(second):
            return True
    if next_line and s.endswith('.') and len(words) <= 16 and _PSEUDOTITLE_START_RE.match(s):
        if _BODY_START_RE.match(next_line.strip()):
            return True
    return False


def _strip_editorial_prefix(lines: List[str]) -> List[str]:
    out = list(lines)
    if not out:
        return out

    # strip heading labels from the first few lines only; do not overreach
    for i in range(min(3, len(out))):
        cleaned, _ = _strip_head_labels(out[i])
        out[i] = cleaned

    while out and not out[0].strip():
        out.pop(0)
    if not out:
        return out

    next_line = out[1].strip() if len(out) > 1 else None
    if _looks_like_editorial_pseudotitle(out[0], next_line):
        parts = _SENTENCE_SPLIT_RE.split(out[0].strip(), maxsplit=1)
        if len(parts) >= 2 and len(parts[0].split()) <= 16 and _BODY_START_RE.match(parts[1].strip()):
            out[0] = parts[1].strip()
        else:
            out.pop(0)

    while out and not out[0].strip():
        out.pop(0)
    return out


def _is_reasonable_body(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 6:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text)) >= 0.55


def _title_match(line: str, current_book: Optional[int]) -> Optional[Tuple[int, int]]:
    m = _TITLE_RE.match(line)
    if not m:
        return None
    book_no = _resolve_book_from_context(m.group("booktok"), current_book)
    if book_no is None:
        return None
    title_no = int(m.group("title"))
    if not (1 <= title_no <= 60):
        return None
    return (book_no, title_no)


def _law_match(line: str, current_book: Optional[int]) -> Optional[Tuple[int, int, int, Optional[int], str]]:
    m = _LAW_RE.match(line)
    if not m:
        return None
    book_no = _resolve_book_from_context(m.group("booktok"), current_book)
    if book_no is None or not (1 <= book_no <= 12):
        return None
    title_no = int(m.group("title"))
    law_no = int(m.group("law"))
    altlaw = int(m.group("altlaw")) if m.group("altlaw") else None
    if not (1 <= title_no <= 60 and 1 <= law_no <= 400):
        return None
    rest = (m.group("rest") or "").strip()
    rest, _ = _strip_head_labels(rest)
    return (book_no, title_no, law_no, altlaw, rest)


def _format_seg_id(source_name: str, book: int, title: int, law: int, altlaw: Optional[int]) -> str:
    if altlaw is not None:
        return f"{source_name}_{book}.{title}.{law}({altlaw})"
    return f"{source_name}_{book}.{title}.{law}"


def _sort_key(seg_id: str) -> Tuple[int, int, int, int]:
    tail = seg_id.split("_", 1)[1]
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\((\d+)\))?$", tail)
    if not m:
        return (999, 999, 999, 999)
    return tuple(int(m.group(i) or 0) for i in range(1, 5))


def segment_lex_visigothorum(text: str, source_name: str) -> List[Tuple[str, str]]:
    text = _normalize_text(text)
    raw_lines = text.split("\n")

    segments: List[Tuple[str, str]] = []
    current_book: Optional[int] = None
    current_title: Optional[int] = None
    current_law: Optional[int] = None
    current_altlaw: Optional[int] = None
    current_lines: List[str] = []
    started = False

    def flush_current() -> None:
        nonlocal current_law, current_altlaw, current_lines
        if current_book is None or current_title is None or current_law is None:
            current_law = None
            current_altlaw = None
            current_lines = []
            return
        key = (current_book, current_title, current_law)
        body_lines = _strip_editorial_prefix(current_lines)
        body = _normalize_body_text(body_lines)
        if key in EXPECTED_ID_SET and _is_reasonable_body(body):
            seg_id = _format_seg_id(source_name, current_book, current_title, current_law, current_altlaw)
            segments.append((seg_id, body))
        current_law = None
        current_altlaw = None
        current_lines = []

    for raw in raw_lines:
        line = _clean_line(raw)
        if not line:
            continue

        marker_line = _normalize_marker_line(line)

        m_b = _BOOK_RE.match(marker_line)
        if m_b:
            book_no = _book_token_to_int(m_b.group("book"))
            if book_no == 1:
                started = True
            if started:
                flush_current()
                if book_no is not None and 1 <= book_no <= 12:
                    current_book = book_no
                else:
                    current_book = None
                current_title = None
            continue

        if not started:
            continue

        # Try structural matches on a normalized marker line before discarding OCR-noisy lines.
        law = _law_match(marker_line, current_book)
        if law is not None:
            book_no, title_no, law_no, altlaw, rest = law
            flush_current()
            current_book = book_no
            current_title = title_no
            current_law = law_no
            current_altlaw = altlaw
            current_lines = [rest] if rest else []
            continue

        title = _title_match(marker_line, current_book)
        if title is not None:
            book_no, title_no = title
            flush_current()
            current_book = book_no
            current_title = title_no
            continue

        if _looks_like_noise(line):
            continue

        if _FRAGMENTA_RE.match(marker_line):
            flush_current()
            current_book = None
            current_title = None
            continue

        if _TIT_SHORT_RE.match(marker_line):
            if current_law is not None and len(current_lines) <= 1:
                current_lines.append(line)
            continue

        if current_law is not None:
            current_lines.append(line)

    flush_current()

    dedup: "OrderedDict[str, str]" = OrderedDict()
    for seg_id, seg_text in segments:
        prev = dedup.get(seg_id)
        if prev is None or len(seg_text) > len(prev):
            dedup[seg_id] = seg_text

    items = list(dedup.items())
    items.sort(key=lambda x: _sort_key(x[0]))
    return validate_segments(items, source_name)


def segment_lex_visigothorum_unified(source_file, source_name):
    return segment_lex_visigothorum(read_source_file(source_file), source_name)


def main() -> None:
    candidates = [
        Path("data/Lex_visigothorum_lat-1.txt"),
        Path("Lex_visigothorum_lat-1.txt"),
        Path("/mnt/data/Lex_visigothorum_lat-1.txt"),
        Path("/mnt/data/Lex_visigothorum_lat-1(1).txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)
    segs = segment_lex_visigothorum_unified(src, "LexVisigothorum")
    print(f"LexVisigothorum: {len(segs)} segments")
    for sid, txt in segs[:5]:
        print(f"  {sid}: {txt[:160]}")


if __name__ == "__main__":
    main()
