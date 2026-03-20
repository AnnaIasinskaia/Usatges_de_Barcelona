#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Lex Visigothorum based on the new Google Books text edition.

Новый вариант ориентирован на OCR-текст из издания Zeumer и на unified-интерфейс:
сегментер сам читает файл и возвращает list[(id, text)].

Основная структурная единица:
    одна lex = один сегмент

ID строится осмысленно и стабильно:
    LexVisigoth_<book>.<title>.<law>
например:
    LexVisigoth_1.1.1
    LexVisigoth_2.5.19
    LexVisigoth_12.3.28
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


# ---------------------------------------------------------------------------
# OCR / heading helpers
# ---------------------------------------------------------------------------

_ORDINAL_BOOK_MAP: Dict[str, int] = {
    "PRIMUS": 1,
    "SECUNDUS": 2,
    "TERTIUS": 3,
    "QUARTUS": 4,
    "QUINTUS": 5,
    "SEXTUS": 6,
    "SEPTIMUS": 7,
    "OCTAVUS": 8,
    "OCTAUUS": 8,
    "NONUS": 9,
    "DECIMUS": 10,
    "UNDECIMUS": 11,
    "DUODECIMUS": 12,
}

_LIBER_RE = re.compile(r"^\s*LIBER\s+([A-Z0-9IVXLCM]+)\.?\s*$", re.IGNORECASE)
_TITULUS_RE = re.compile(
    r"^\s*([IVXLCM]+)\.\s*TITULUS\s*[:.]?\s*(.+?)\s*$",
    re.IGNORECASE,
)

_NUMERIC_LEX_RE = re.compile(
    r"^\s*(?P<book>[IVX1l]+|\d+)\s*,\s*(?P<title>\d+)\s*,\s*(?P<law>\d+)\.\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

_INLINE_ROMAN_HEADING_RE = re.compile(
    r"^\s*(?P<roman>[A-Za-z0-9]+)\.?\s+(?P<rest>.+?)\s*$"
)

_PURE_ROMAN_HEADING_RE = re.compile(r"^\s*(?P<roman>[A-Za-z0-9]+)\.\s*$")

_PAGE_HEADER_RE = re.compile(r"LEX\s+VISIGOTHORUM|LIBER\s+IUDICIORUM", re.IGNORECASE)


def _normalize_body_window(text: str) -> str:
    text = text.replace("\f", "\n")
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)
    return text


def _roman_like_to_int(token: str) -> int:
    tok = token.strip().upper().replace("J", "I")
    tok = tok.replace("VUII", "VIII")
    tok = tok.replace("UIII", "VIII")
    tok = tok.replace("NIL", "IIII")
    tok = tok.replace("INI", "III")
    tok = tok.replace("1111", "IIII")
    tok = tok.replace("111", "III")
    tok = tok.replace("11", "II")

    if tok == "N":
        tok = "II"
    if tok == "L":
        tok = "I"

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not tok or any(ch not in values for ch in tok):
        raise ValueError(f"Unsupported Roman token: {token!r} -> {tok!r}")

    total = 0
    prev = 0
    for ch in reversed(tok):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def _book_token_to_int(token: str) -> Optional[int]:
    tok = token.strip().upper().replace("0", "O").replace("J", "I")
    if tok in _ORDINAL_BOOK_MAP:
        return _ORDINAL_BOOK_MAP[tok]
    try:
        return _roman_like_to_int(tok)
    except Exception:
        return None


def _is_roman_heading_token(token: str) -> bool:
    tok = token.strip().upper()
    if tok in {"N", "L", "11", "111", "1111", "VUII", "NIL", "INI"}:
        return True
    return bool(re.fullmatch(r"[IVXLCDM]+", tok))


def _clean_line(raw) -> str:
    s = str(raw).replace("\t", " ").rstrip()
    s = re.sub(r"^\s*\d+\s+(?=[A-Za-zIVXivx])", "", s)
    return s.rstrip()


def _safe_to_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return str(obj)
    except Exception:
        pass
    try:
        return "".join(str(x) for x in obj)
    except Exception:
        return ""


def _ascii_alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    alpha = 0
    total = 0
    for ch in s:
        total += 1
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            alpha += 1
    return alpha / max(1, total)


def _looks_like_sigla_line(lower: str) -> bool:
    sigla_hits = 0

    for marker in (
        " recc.",
        " erv.",
        " recc ",
        " erv ",
        " cod.",
        " codd.",
        " ms.",
        " mss.",
        " emend",
        " corr.",
        " desunt",
        " deest",
        " gl.",
        " schol.",
    ):
        if marker in lower:
            sigla_hits += 1

    compact = lower.replace(" ", "")
    for marker in ("r1", "r2", "e1", "e2", "v1", "v2", "b2"):
        if marker in compact:
            sigla_hits += 1

    return sigla_hits > 0


def _is_apparatus_or_noise(line) -> bool:
    s = _safe_to_text(line).strip()
    if not s:
        return True

    lower = s.lower()

    try:
        if _PAGE_HEADER_RE.search(s):
            return True
    except RuntimeError:
        if "lex visigothorum" in lower or "liber iudiciorum" in lower:
            return True

    if lower.startswith(("lib. ", "tit. ")):
        return True
    if "adhibui codices" in lower:
        return True

    if len(s) > 10 and "l." in s and any(ch.isdigit() for ch in s):
        return True
    if s.startswith(("1)", "2)", "3)", "4)", "5)", "*", "†")):
        return True

    if len(s) > 15 and _looks_like_sigla_line(lower):
        return True

    if len(s) > 30 and _ascii_alpha_ratio(s) < 0.45:
        return True

    return False


def _strip_heading_tail_noise(text: str) -> str:
    text = re.sub(r"\s+[A-Za-z><\^*«»0-9]+$", "", text).strip()
    return text


def _candidate_score(text: str) -> Tuple[int, int]:
    """Лучший кандидат для duplicate-id: меньше apparatus, длиннее тело."""
    penalty = 0
    lower = text.lower()
    for bad in ("adhibui codices", "recc.", "erv.", " l. ", " r 1", " e 2", " v 1"):
        if bad in lower:
            penalty += 1
    return (-penalty, len(text))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_main_legal_block(text: str) -> str:
    starts = list(re.finditer(r"LIBER\s+PRIMUS\.", text, re.IGNORECASE))
    body_start = None
    for m in starts:
        window = text[m.start():m.start() + 3500]
        if re.search(r"(?m)^\s*1\s*,\s*1\s*,\s*1\.", window):
            body_start = m.start()
            break

    if body_start is None:
        raise ValueError("Could not locate the main legal text of Lex Visigothorum")

    end_markers = [
        "Chronica regum Visigothorum",
        "CHRONICA REGUM VISIGOTHORUM",
        "Additamentum.",
        "Supplementa",
    ]
    body_end = None
    for marker in end_markers:
        pos = text.find(marker, body_start)
        if pos != -1:
            body_end = pos
            break

    if body_end is None:
        body_end = len(text)

    return _normalize_body_window(text[body_start:body_end])


def segment_lex_visigothorum(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Структурная сегментация нового OCR-издания Lex Visigothorum.

    Опорная логика:
    - находим основной блок текста законов;
    - отслеживаем LIBER -> TITULUS -> отдельную lex;
    - строим id как <source_name>_<book>.<title>.<law>.
    """
    legal_text = _extract_main_legal_block(text)
    lines = legal_text.splitlines()

    current_book: Optional[int] = None
    current_title: Optional[int] = None
    current_segment_id: Optional[str] = None
    current_parts: List[str] = []
    waiting_for_title_after_pure_roman = False

    raw_segments: List[Tuple[str, str]] = []

    def flush_current() -> None:
        nonlocal current_segment_id, current_parts, waiting_for_title_after_pure_roman
        if current_segment_id is None:
            return
        joined = clean_text(" ".join(part for part in current_parts if part))
        if len(joined.split()) >= 5:
            raw_segments.append((current_segment_id, joined))
        current_segment_id = None
        current_parts = []
        waiting_for_title_after_pure_roman = False

    for raw in lines:
        line = _clean_line(raw)
        if not line.strip():
            continue

        m = _LIBER_RE.match(line)
        if m:
            flush_current()
            current_book = _book_token_to_int(m.group(1))
            current_title = None
            continue

        m = _TITULUS_RE.match(line)
        if m:
            flush_current()
            try:
                current_title = _roman_like_to_int(m.group(1))
            except Exception:
                current_title = None
            continue

        if current_book is None or current_title is None:
            continue

        m = _NUMERIC_LEX_RE.match(line)
        if m:
            try:
                book_no = int(m.group("book")) if m.group("book").isdigit() else _roman_like_to_int(
                    m.group("book").replace("1", "I").replace("l", "I")
                )
                title_no = int(m.group("title"))
                law_no = int(m.group("law"))
            except Exception:
                book_no = title_no = law_no = None

            if (
                book_no == current_book
                and title_no == current_title
                and law_no is not None
                and 1 <= law_no <= 200
            ):
                flush_current()
                rest = m.group("rest").strip()
                m_head = _INLINE_ROMAN_HEADING_RE.match(rest)
                if m_head and _is_roman_heading_token(m_head.group("roman")):
                    rest = m_head.group("rest").strip()
                current_segment_id = f"{source_name}_{current_book}.{current_title}.{law_no}"
                if rest:
                    current_parts.append(_strip_heading_tail_noise(rest))
                continue

        m = _PURE_ROMAN_HEADING_RE.match(line)
        if m and _is_roman_heading_token(m.group("roman")):
            try:
                law_no = _roman_like_to_int(m.group("roman"))
            except Exception:
                law_no = None
            if law_no is not None and 1 <= law_no <= 200:
                flush_current()
                current_segment_id = f"{source_name}_{current_book}.{current_title}.{law_no}"
                waiting_for_title_after_pure_roman = True
                continue

        m = _INLINE_ROMAN_HEADING_RE.match(line)
        if m and _is_roman_heading_token(m.group("roman")):
            try:
                law_no = _roman_like_to_int(m.group("roman"))
            except Exception:
                law_no = None

            rest = m.group("rest").strip()
            if law_no is not None and 1 <= law_no <= 200 and len(rest.split()) >= 2:
                flush_current()
                current_segment_id = f"{source_name}_{current_book}.{current_title}.{law_no}"
                current_parts.append(_strip_heading_tail_noise(rest))
                waiting_for_title_after_pure_roman = False
                continue

        if _is_apparatus_or_noise(line):
            continue

        if waiting_for_title_after_pure_roman and current_segment_id is not None:
            current_parts.append(_strip_heading_tail_noise(line))
            waiting_for_title_after_pure_roman = False
            continue

        if current_segment_id is not None:
            current_parts.append(line.strip())

    flush_current()

    dedup: "OrderedDict[str, str]" = OrderedDict()
    for seg_id, seg_text in raw_segments:
        if seg_id not in dedup:
            dedup[seg_id] = seg_text
            continue
        if _candidate_score(seg_text) > _candidate_score(dedup[seg_id]):
            dedup[seg_id] = seg_text

    segments = list(dedup.items())
    return validate_segments(segments, source_name)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def segment_lex_visigothorum_unified(source_file, source_name):
    """
    Unified segmenter for Lex Visigothorum.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "LexVisigoth".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_lex_visigothorum(text, source_name)


def main() -> None:
    candidates = [
        Path("data/legesvisigothor00zeumgoog_text.txt"),
        Path("legesvisigothor00zeumgoog_text.txt"),
        Path("/mnt/data/legesvisigothor00zeumgoog_text.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_lex_visigothorum_unified(src, "LexVisigoth")
    print(f"LexVisigoth: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()