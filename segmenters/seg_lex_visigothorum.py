#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified segmenter v8 for Lex Visigothorum (Zeumer OCR).

Public contract preserved:
    segment_lex_visigothorum_unified(source_file, source_name) -> list[(id, text)]

Main ideas vs v7
----------------
1. Stronger rhetorical mode for books 1/10/11.
2. Mandatory post-rubric extension attempts.
3. Earlier / stronger apparatus stripping.
4. Stricter boundary handling for books 4/5 and selected titles.
5. Expected-id-driven extraction only.
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


# ---------------------------------------------------------------------------
# Expected ids
# ---------------------------------------------------------------------------

EXPECTED_IDS_BY_TITLE: Dict[Tuple[int, int], List[int]] = {
    (1, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 13],
    (1, 2): [1, 2, 3, 4, 5, 6, 8],
    (1, 5): [2],
    (2, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    (2, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (2, 3): [1, 3, 4, 5, 6, 7, 8, 9, 10],
    (2, 4): [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    (2, 5): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    (2, 6): [11, 18, 19],
    (2, 8): [1, 2, 9],
    (2, 12): [25],
    (3, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (3, 2): [1, 2, 3, 4, 6, 7, 8],
    (3, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (3, 4): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    (3, 5): [1, 2, 3, 4, 5, 6],
    (3, 6): [1, 2, 3],
    (3, 8): [2, 6, 9, 12],
    (3, 16): [1, 2],
    (3, 17): [4],
    (3, 51): [1],
    (4, 1): [1, 2, 3, 4, 5, 6, 7, 14],
    (4, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (4, 3): [1, 2, 3, 4],
    (4, 4): [1, 2, 3],
    (4, 5): [1, 2, 3, 4, 5, 6, 7],
    (4, 6): [1],
    (4, 10): [2, 6],
    (4, 11): [100],
    (4, 21): [16],
    (5, 1): [1, 2, 3, 4, 8, 14],
    (5, 2): [1, 2, 3, 4, 5, 6, 7],
    (5, 3): [1, 2, 3, 4],
    (5, 4): [1, 2, 3, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 21, 22],
    (5, 5): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    (5, 6): [1, 2, 3, 4, 5, 6, 7],
    (5, 7): [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (5, 8): [1],
    (6, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (6, 2): [1, 2, 3, 4, 5, 6, 8],
    (6, 3): [1, 2, 3, 4, 5, 6, 7],
    (6, 4): [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    (6, 5): [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    (6, 6): [7, 9, 10, 11, 15, 17, 19],
    (6, 8): [5, 6, 7],
    (6, 22): [9],
    (6, 56): [4],
    (7, 1): [5],
    (7, 2): [21],
    (7, 3): [1],
    (7, 6): [1],
    (9, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    (9, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9],
    (9, 3): [1, 2, 3, 4],
    (9, 4): [2],
    (9, 8): [1, 2, 4],
    (9, 9): [1],
    (9, 15): [1],
    (9, 24): [1],
    (9, 45): [1, 2],
    (10, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100],
    (10, 2): [1, 2, 3, 4, 5, 6, 7, 8],
    (10, 3): [1, 2, 3, 4, 5],
    (10, 8): [2, 4, 8],
    (11, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 24, 88],
    (11, 2): [1, 2, 4, 5, 8],
    (11, 3): [1, 2, 3, 4],
    (11, 5): [3, 5],
    (11, 6): [8],
    (11, 8): [1, 2, 8],
}

BOOK_PROFILE = {
    1: {"body_mode": "rhetorical", "body_start_window": 72, "anchor_candidates_limit": 18, "prefer_anchor_roles": ["mixed", "rubric", "body_leading"], "rubric_weight_bonus": 26, "mixed_weight_bonus": 32, "allow_rubric_extension": True, "allow_long_preamble_as_body": True, "min_words_if_no_continuation": 22, "min_words_relaxed": 11, "editorial_tolerance_at_start": 1, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "very_high", "dedupe_guard": "normal"},
    2: {"body_mode": "casuistic", "body_start_window": 42, "anchor_candidates_limit": 14, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 12, "mixed_weight_bonus": 18, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 15, "min_words_relaxed": 9, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "high", "dedupe_guard": "normal"},
    3: {"body_mode": "casuistic", "body_start_window": 38, "anchor_candidates_limit": 12, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 10, "mixed_weight_bonus": 14, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 15, "min_words_relaxed": 9, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "high", "dedupe_guard": "normal"},
    4: {"body_mode": "casuistic", "body_start_window": 36, "anchor_candidates_limit": 12, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 8, "mixed_weight_bonus": 12, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 15, "min_words_relaxed": 9, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "very_high", "tail_trim_aggressiveness": "very_high", "dedupe_guard": "very_strict"},
    5: {"body_mode": "casuistic", "body_start_window": 46, "anchor_candidates_limit": 14, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 12, "mixed_weight_bonus": 18, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 15, "min_words_relaxed": 9, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "very_high", "tail_trim_aggressiveness": "very_high", "dedupe_guard": "very_strict"},
    6: {"body_mode": "casuistic", "body_start_window": 40, "anchor_candidates_limit": 12, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 10, "mixed_weight_bonus": 14, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 15, "min_words_relaxed": 9, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "high", "dedupe_guard": "normal"},
    7: {"body_mode": "casuistic", "body_start_window": 30, "anchor_candidates_limit": 8, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 6, "mixed_weight_bonus": 9, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 12, "min_words_relaxed": 8, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "high", "dedupe_guard": "strict"},
    9: {"body_mode": "casuistic", "body_start_window": 48, "anchor_candidates_limit": 14, "prefer_anchor_roles": ["body_leading", "mixed", "rubric"], "rubric_weight_bonus": 14, "mixed_weight_bonus": 20, "allow_rubric_extension": True, "allow_long_preamble_as_body": False, "min_words_if_no_continuation": 14, "min_words_relaxed": 8, "editorial_tolerance_at_start": 0, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "high", "dedupe_guard": "normal"},
    10: {"body_mode": "rhetorical", "body_start_window": 84, "anchor_candidates_limit": 22, "prefer_anchor_roles": ["mixed", "rubric", "body_leading"], "rubric_weight_bonus": 28, "mixed_weight_bonus": 34, "allow_rubric_extension": True, "allow_long_preamble_as_body": True, "min_words_if_no_continuation": 24, "min_words_relaxed": 11, "editorial_tolerance_at_start": 1, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "very_high", "dedupe_guard": "normal"},
    11: {"body_mode": "rhetorical", "body_start_window": 86, "anchor_candidates_limit": 22, "prefer_anchor_roles": ["mixed", "rubric", "body_leading"], "rubric_weight_bonus": 28, "mixed_weight_bonus": 32, "allow_rubric_extension": True, "allow_long_preamble_as_body": True, "min_words_if_no_continuation": 24, "min_words_relaxed": 11, "editorial_tolerance_at_start": 1, "editorial_tolerance_at_end": 0, "apparatus_hard_reject_threshold": 2, "boundary_break_strictness": "medium", "tail_trim_aggressiveness": "very_high", "dedupe_guard": "normal"},
}

TITLE_OVERRIDE = {
    (1, 1): {"body_start_window": 90, "anchor_candidates_limit": 24},
    (1, 2): {"body_start_window": 90, "anchor_candidates_limit": 24},
    (4, 1): {"body_start_window": 46},
    (4, 2): {"boundary_break_strictness": "very_high", "dedupe_guard": "very_strict"},
    (5, 6): {"body_start_window": 58, "allow_rubric_extension": True},
    (5, 7): {"body_start_window": 58, "allow_rubric_extension": True, "tail_trim_aggressiveness": "very_high"},
    (6, 2): {"body_start_window": 50},
    (6, 3): {"body_start_window": 50},
    (9, 1): {"body_start_window": 52, "anchor_candidates_limit": 16},
    (10, 1): {"body_start_window": 96, "anchor_candidates_limit": 26},
    (10, 2): {"body_start_window": 96, "anchor_candidates_limit": 26},
    (11, 1): {"body_start_window": 96, "anchor_candidates_limit": 26},
    (11, 2): {"body_start_window": 96, "anchor_candidates_limit": 26},
    (11, 3): {"body_start_window": 92, "anchor_candidates_limit": 24},
    (11, 5): {"body_start_window": 88, "anchor_candidates_limit": 20},
}


def _cfg_for(book: int, title: int) -> Dict[str, object]:
    cfg = dict(BOOK_PROFILE[book])
    cfg.update(TITLE_OVERRIDE.get((book, title), {}))
    return cfg


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_ORDINAL_BOOK_MAP: Dict[str, int] = {
    "PRIMUS": 1, "SECUNDUS": 2, "TERTIUS": 3, "QUARTUS": 4, "QUINTUS": 5,
    "SEXTUS": 6, "SEPTIMUS": 7, "OCTAVUS": 8, "OCTAUUS": 8, "NONUS": 9,
    "DECIMUS": 10, "UNDECIMUS": 11, "DUODECIMUS": 12,
}
_PAGE_HEADER_RE = re.compile(r"LEX\s+VISIGOTHORUM|LIBER\s+IUDICIORUM", re.IGNORECASE)
_LIBER_RE = re.compile(r"^\s*LIBER\s+([A-Z0-9IVXLCM]+)\.?\s*$", re.IGNORECASE)
_TITULUS_RE = re.compile(r"^\s*([IVXLCM]+)\.\s*TITULUS\b", re.IGNORECASE)
_ROMAN_ONLY_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_LAW_ID_RE = re.compile(
    r"(?P<book>\b(?:[IVXLCDM]+|\d+|[IVX1l]{1,8})\b)\s*[,|]\s*(?P<title>\d{1,2}|[IVXLCDM]{1,6})\s*[,|]\s*(?P<law>\d{1,3}|[IVXLCDM]{1,8})\.",
    re.IGNORECASE,
)

_APPARATUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bcodd?\b", r"\bcod\.\b", r"\brecc?\b", r"\berv\b", r"\bdeest\b", r"\bdes\.\b",
    r"\bcorr\.\b", r"\bemend\b", r"\badd\.\b", r"\bita\b", r"\bmad\.\b", r"\bpith\.\b",
    r"\bcf\.\b", r"\binfra\b", r"\bsupra\b", r"\bgaud\.\b", r"\binterp\b", r"\binthrpr\b",
    r"\bintrpr\b", r"\bappend\.\b", r"\bn\.\s*arch\.\b", r"\bmarca\s+hispanica\b",
    r"\bmonasterii\b", r"\bcartae\b", r"\bconc\.\b", r"\btolet\.\b", r"\bdig\.\b",
    r"\bl\.\s*burg\.\b", r"\blex\s+rom\.\b", r"\bl\.\s*\d", r"/\.", r"\|\|",
    r"\bgls\b", r"\brcds\b", r"\bmss?\b", r"\blegis\s+baiuv\.\b", r"\bdahn\b", r"\bstudien\b",
]]
_RUBRIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bantiqua\b", r"\bflavius\b", r"\brex\b", r"\brecc\b", r"\berv\b",
    r"\begica\b", r"\bchindasvind\w*\b", r"\breccessvind\w*\b", r"\bnov\.\b",
]]
_EDITORIAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bcf\.\b", r"\bn\.\s*arch\.\b", r"\bzeitschr\.\b", r"\bmarca\s+hispanica\b",
    r"\bmonasterii\b", r"\bcartae\b", r"\ba\.\s*\d{3,4}\b", r"\bappend\.\b",
    r"\bhandbuch\b", r"\bgoldschm", r"\bde\s+hac\s+regula\b",
    r"\bquae\s+de\s+hac\s+re\s+exposui\b", r"\bvidetur\b", r"\bcrediderim\b",
    r"\bl\.\s*burg\.\b", r"\blex\s+rom\.\b", r"\bconc\.\s*tolet\.\b", r"\bdig\.\b",
    r"\bmad,\s*addit\b", r"\baddit\s+formulam\b", r"\blegis\s+baiuv\.\b",
    r"\bdahn\b", r"\bstudien\b",
]]
_MAIN_TEXT_HINTS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bsi\b", r"\bquis\b", r"\bquod\b", r"\bquicumque\b", r"\but\b", r"\bdum\b",
    r"\bomnis\b", r"\bpresentis\s+legis\b", r"\bde\b", r"\bservus\b", r"\bingenuus\b",
    r"\biudex\b", r"\bdominus\b", r"\bdomino\b", r"\bfugitiv\w*\b", r"\bmancipi\w*\b",
    r"\blatron\w*\b", r"\bhomicid\w*\b", r"\baccusat\w*\b", r"\btest\w*\b",
    r"\bpena\b", r"\bsolidos?\b", r"\bconpon\w*\b", r"\brestitu\w*\b",
    r"\bvoluerit\b", r"\bpresumpserit\b", r"\bsanction\w*\b", r"\blegibus\b",
    r"\bmanifeste\b", r"\bdepromitur\b", r"\bsententia\b", r"\bordinante\b", r"\buniversis\b",
]]
_STRONG_BODY_OPENERS = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*si\b", r"^\s*quod\b", r"^\s*ut\b", r"^\s*de\b", r"^\s*dum\b",
    r"^\s*omnis\b", r"^\s*presentis\s+legis\b", r"^\s*priscarum\b",
    r"^\s*infra\b", r"^\s*nam\s+si\b", r"^\s*si\s+vero\b",
    r"^\s*quod\s+si\b", r"^\s*ceterum\b",
    r"^\s*flavius\b", r"^\s*universis\b", r"^\s*domino\s+ordinante\b",
]]
_CONTINUATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bsi\s+vero\b", r"\bquod\s+si\b", r"\bnam\s+si\b",
    r"\bsi\s+autem\b", r"\bceterum\b", r"\bet\s+si\b", r"\bcerte\s+si\b",
]]
_STOP_AT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bindex capitulorum\b", r"\blemmata\b", r"\bchronica regum visigothorum\b",
    r"\badditamentum\b", r"\bsupplementa\b",
]]
_FOOTNOTE_START_RE = re.compile(r"^\s*\d+\)")
_SHORT_CITATION_RE = re.compile(
    r"^\s*(?:cf\.|mad\.|pith\.|dig\.|l\.\s*burg\.|conc\.|interp\.|inthrpr\.|intrpr\.|legis\s+baiuv\.)",
    re.IGNORECASE,
)
_INLINE_BODY_START_RE = re.compile(
    r"(?P<body>(?:Si|Quod|Ut|Dum|Omnis|Presentis\s+legis|De|Priscarum|Nam\s+si|Si\s+vero|Quod\s+si|Flavius|Universis|Domino\s+ordinante)\b.*$)",
    re.IGNORECASE,
)
_VARIANT_CHAIN_RE = re.compile(
    r"(?:\b[A-Z][a-z]?\s*\d?\.?\s*;|\bdes\.;|\bita\b|\badd\.)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalize_body_window(text: str) -> str:
    text = text.replace("\f", "\n")
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)
    return text


def _roman_like_to_int(token: str) -> int:
    tok = token.strip().upper().replace("J", "I")
    tok = tok.replace("VUII", "VIII").replace("UIII", "VIII")
    tok = tok.replace("NIL", "IIII").replace("INI", "III")
    tok = tok.replace("1111", "IIII").replace("111", "III").replace("11", "II")
    if tok == "N":
        tok = "II"
    if tok == "L":
        tok = "I"
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(tok):
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def _to_int_maybe(token: str) -> Optional[int]:
    tok = token.strip()
    if tok.isdigit():
        return int(tok)
    try:
        return _roman_like_to_int(tok if _ROMAN_ONLY_RE.fullmatch(tok) else tok.replace("1", "I").replace("l", "I"))
    except Exception:
        return None


def _book_token_to_int(token: str) -> Optional[int]:
    tok = token.strip().upper().replace("0", "O").replace("J", "I")
    if tok in _ORDINAL_BOOK_MAP:
        return _ORDINAL_BOOK_MAP[tok]
    try:
        return _roman_like_to_int(tok)
    except Exception:
        return None


def _clean_line(raw: str) -> str:
    s = str(raw).replace("\t", " ").rstrip()
    s = re.sub(r"^\s*\d+\s+(?=[A-Za-zIVXivx])", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", s or "")


def _ascii_alpha_ratio(s: str) -> float:
    alpha = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in (s or ""))
    return alpha / max(1, len(s or ""))


def _apparatus_hits(s: str) -> int:
    return sum(1 for p in _APPARATUS_PATTERNS if p.search(s))


def _rubric_hits(s: str) -> int:
    return sum(1 for p in _RUBRIC_PATTERNS if p.search(s))


def _editorial_hits(s: str) -> int:
    return sum(1 for p in _EDITORIAL_PATTERNS if p.search(s))


def _main_text_hits(s: str) -> int:
    return sum(1 for p in _MAIN_TEXT_HINTS if p.search(s))


def _has_strong_body_opener(s: str) -> bool:
    return any(p.search(s) for p in _STRONG_BODY_OPENERS)


def _has_body_continuation(s: str) -> bool:
    return any(p.search(s) for p in _CONTINUATION_PATTERNS)


def _is_page_header_or_furniture(s: str) -> bool:
    low = s.lower()
    return (
        (not s)
        or bool(_PAGE_HEADER_RE.search(s))
        or bool(re.fullmatch(r"\d+", s))
        or bool(re.search(r"LEX\s+VISIGOTHORUM.*\b\d+\b", s, re.IGNORECASE))
        or (len(s) < 8 and sum(ch.isdigit() for ch in s) >= 1)
        or low.startswith("lib. ")
        or low.startswith("tit. ")
    )


def _is_structural_stop_line(s: str) -> bool:
    return any(p.search(s) for p in _STOP_AT_PATTERNS)


# ---------------------------------------------------------------------------
# Parsing structure
# ---------------------------------------------------------------------------

def _extract_core_liber_iudiciorum_block(text: str) -> str:
    starts = list(re.finditer(r"LIBER\s+PRIMUS\.", text, re.IGNORECASE))
    body_start = None
    for m in starts:
        window = text[m.start():m.start() + 7000]
        if re.search(r"(?m)^\s*1\s*[,|]\s*1\s*[,|]\s*1\.", window):
            body_start = m.start()
            break
    if body_start is None:
        raise ValueError("Could not locate main legal text")
    for marker in ["Chronica regum Visigothorum", "CHRONICA REGUM VISIGOTHORUM", "Additamentum.", "Supplementa"]:
        pos = text.find(marker, body_start)
        if pos != -1:
            return _normalize_body_window(text[body_start:pos])
    return _normalize_body_window(text[body_start:])


def _parse_structure(lines):
    current_book = None
    current_title = None
    annotated = []
    for idx, raw in enumerate(lines):
        line = _clean_line(raw)
        if not line:
            annotated.append((idx, current_book, current_title, line))
            continue
        m = _LIBER_RE.match(line)
        if m:
            current_book = _book_token_to_int(m.group(1))
            current_title = None
            annotated.append((idx, current_book, current_title, line))
            continue
        m = _TITULUS_RE.match(line)
        if m:
            try:
                current_title = _roman_like_to_int(m.group(1))
            except Exception:
                current_title = None
            annotated.append((idx, current_book, current_title, line))
            continue
        annotated.append((idx, current_book, current_title, line))
    return annotated


def _strip_title_frontmatter(annotated_lines):
    result = []
    buckets = defaultdict(list)
    for row in annotated_lines:
        buckets[(row[1], row[2])].append(row)
    ordered_keys, seen = [], set()
    for _, book, title, _ in annotated_lines:
        key = (book, title)
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)
    for key in ordered_keys:
        rows = buckets[key]
        if key[0] is None or key[1] is None:
            result.extend(rows)
            continue
        first_law_idx = None
        for pos, row in enumerate(rows):
            line = row[3]
            if not line:
                continue
            if re.search(r"\blemmata\b|\bindex capitulorum\b", line, re.IGNORECASE):
                continue
            if _LAW_ID_RE.search(line):
                first_law_idx = pos
                break
        result.extend(rows if first_law_idx is None else rows[first_law_idx:])
    return result


def _find_law_occurrences(annotated_lines):
    occ = defaultdict(list)
    for idx, current_book, current_title, line in annotated_lines:
        if not line or current_book is None or current_title is None:
            continue
        for m in _LAW_ID_RE.finditer(line):
            b = _to_int_maybe(m.group("book"))
            t = _to_int_maybe(m.group("title"))
            l = _to_int_maybe(m.group("law"))
            if b is None or t is None or l is None:
                continue
            if 1 <= b <= 12 and 1 <= t <= 60 and 1 <= l <= 400:
                occ[(b, t, l)].append(idx)
    return occ


# ---------------------------------------------------------------------------
# Cleaning / classification
# ---------------------------------------------------------------------------

def _trim_editorial_prefix_inline(text: str) -> str:
    if not text:
        return text
    m = _INLINE_BODY_START_RE.search(text)
    if not m:
        return text
    cand = m.group("body").strip()
    return cand if cand and len(cand) < len(text) else text


def _trim_editorial_suffix_inline(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in [
        re.compile(r"\s+(?:Cf\.|Mad\.|Pith\.|Dig\.|L\.\s*Burg\.|Conc\.|N\.\s*Arch\.|Interp\.|Inthrpr\.|Legis\s+Baiuv\.).*$", re.IGNORECASE),
        re.compile(r"\s+\d+\)\s+.*$", re.IGNORECASE),
        re.compile(r"\s+\|\|.*$", re.IGNORECASE),
    ]:
        out = pat.sub("", out).strip()
    return out


def _strip_variant_chains(text: str) -> str:
    if not text:
        return text
    out = text
    # Bracketed / sigla-heavy tails
    out = re.sub(r"\b(?:R|E|V|B|F|Pith|Mad|L|Ji)\s*\\?\.?\s*\d?\b(?:\s*;\s*)?", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:des\.|ita|add\.)\b(?:\s*;\s*)?", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _preclean_candidate_text(text: str, cfg: Dict[str, object]) -> str:
    out = _trim_editorial_prefix_inline(text)
    out = _trim_editorial_suffix_inline(out)
    out = _strip_variant_chains(out)
    return re.sub(r"\s+", " ", out).strip()


def _classify_editorial_fragment(line: str, cfg: Dict[str, object]) -> Optional[str]:
    low = line.lower()
    if _is_page_header_or_furniture(line):
        return "page_header"
    if _FOOTNOTE_START_RE.match(line):
        return "footnote"
    if _SHORT_CITATION_RE.match(low):
        return "cross_reference"
    if _is_structural_stop_line(line):
        return "index_entry"
    app = _apparatus_hits(low)
    edt = _editorial_hits(low)
    if "codd" in low or "/." in low or "||" in low:
        return "apparatus"
    if _VARIANT_CHAIN_RE.search(line) and app >= 1:
        return "apparatus"
    if edt >= 1 and _main_text_hits(low) <= 1:
        return "cross_reference"
    if app >= int(cfg["apparatus_hard_reject_threshold"]) and _main_text_hits(low) <= 1:
        return "apparatus"
    return None


def _classify_occurrence_role(line: str, cfg: Dict[str, object]) -> str:
    ed = _classify_editorial_fragment(line, cfg)
    if ed == "page_header":
        return "page_header"
    if ed in {"footnote", "cross_reference", "index_entry"}:
        return "tail_reference"
    if ed == "apparatus":
        return "apparatus"
    low = line.lower()
    app = _apparatus_hits(low)
    rubric = _rubric_hits(low)
    editor = _editorial_hits(low)
    main = _main_text_hits(low)
    if rubric >= 1 and main == 0:
        return "rubric"
    if main >= 2 and app == 0 and editor == 0:
        return "body_leading"
    if main >= 1 and rubric >= 1 and editor <= int(cfg["editorial_tolerance_at_start"]):
        return "mixed"
    return "apparatus"


def _role_base_score(role: str, cfg: Dict[str, object]) -> int:
    order = list(cfg["prefer_anchor_roles"])
    if role == order[0]:
        return 180
    if len(order) > 1 and role == order[1]:
        return 130
    if len(order) > 2 and role == order[2]:
        return 90
    return {"apparatus": -40, "tail_reference": -160, "page_header": -220}.get(role, -60)


def _score_anchor(line: str, cfg: Dict[str, object], recovery=False) -> int:
    low = line.lower()
    role = _classify_occurrence_role(line, cfg)
    score = _role_base_score(role, cfg)
    score += _main_text_hits(low) * 10
    score -= _apparatus_hits(low) * 16
    score -= _editorial_hits(low) * 30
    if role == "rubric":
        score += int(cfg["rubric_weight_bonus"])
    if role == "mixed":
        score += int(cfg["mixed_weight_bonus"])
    if _has_strong_body_opener(low):
        score += 20
    if _SHORT_CITATION_RE.match(low):
        score -= 120
    if "codd" in low:
        score -= 100
    if recovery:
        score += 12
    return score


def _candidate_anchor_indices(key, line_indices, annotated_lines, cfg, recovery=False):
    scored = []
    for idx in line_indices:
        line = annotated_lines[idx][3]
        scored.append((_score_anchor(line, cfg, recovery=recovery) - min(idx, 5000) // 350, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored]


# ---------------------------------------------------------------------------
# Body detection
# ---------------------------------------------------------------------------

def _rhetorical_opening_like(low: str) -> bool:
    return bool(
        re.search(r"^(flavius|universis|domino\s+ordinante|salutare|quod\s+sit|presentis\s+legis|de\s+nuptiarum)", low)
    )


def _looks_like_main_law_body(line: str, cfg: Dict[str, object]) -> bool:
    if not line:
        return False
    line = _preclean_candidate_text(line, cfg)
    low = line.lower()
    if _classify_editorial_fragment(line, cfg) is not None:
        return False

    app = _apparatus_hits(low)
    editor = _editorial_hits(low)
    rubric = _rubric_hits(low)
    main = _main_text_hits(low)
    toks = _tokenize_words(line)
    digits = sum(tok.isdigit() for tok in toks)
    alpha_ratio = _ascii_alpha_ratio(line)

    if editor > int(cfg["editorial_tolerance_at_start"]) and main <= 1:
        return False
    if app >= int(cfg["apparatus_hard_reject_threshold"]) and main <= 1:
        return False
    if digits >= 4 and alpha_ratio < 0.70:
        return False
    if len(toks) <= 3:
        return False

    if str(cfg["body_mode"]) == "rhetorical":
        if _rhetorical_opening_like(low):
            return True
        if main >= 1 and editor <= int(cfg["editorial_tolerance_at_start"]):
            return True
        if _has_strong_body_opener(low):
            return True
        if rubric >= 1 and bool(cfg["allow_long_preamble_as_body"]):
            return True
        return False

    return (
        (main >= 2 and editor == 0)
        or (_has_strong_body_opener(low) and editor <= int(cfg["editorial_tolerance_at_start"]) and app <= 1)
        or (rubric >= 1 and main >= 1 and app <= 1 and editor <= int(cfg["editorial_tolerance_at_start"]))
    )


def _looks_like_body_end_break(line: str, cfg: Dict[str, object]) -> bool:
    if not line:
        return False
    if _classify_editorial_fragment(line, cfg) is not None:
        return True
    low = line.lower()
    app = _apparatus_hits(low)
    main = _main_text_hits(low)
    digits = sum(tok.isdigit() for tok in _tokenize_words(line))
    return (app >= int(cfg["apparatus_hard_reject_threshold"]) and main <= 1) or (digits >= 4 and _ascii_alpha_ratio(line) < 0.72)


def _find_body_start(anchor_idx: int, annotated_lines, cfg: Dict[str, object]) -> Optional[int]:
    window = int(cfg["body_start_window"])
    for j in range(anchor_idx, min(anchor_idx + window, len(annotated_lines))):
        line = _preclean_candidate_text(annotated_lines[j][3], cfg)
        if _looks_like_main_law_body(line, cfg):
            low = line.lower()
            if _editorial_hits(low) > int(cfg["editorial_tolerance_at_start"]):
                continue
            if _classify_editorial_fragment(line, cfg) is not None:
                continue
            return j
    return None


def _trim_head(lines: List[str], cfg: Dict[str, object]) -> List[str]:
    for i, line in enumerate(lines):
        line2 = _preclean_candidate_text(line, cfg)
        low = line2.lower()
        if _looks_like_main_law_body(line2, cfg) and _classify_editorial_fragment(line2, cfg) is None:
            if _editorial_hits(low) <= int(cfg["editorial_tolerance_at_start"]) or _main_text_hits(low) >= _editorial_hits(low) + 1:
                out = lines[i:].copy()
                out[0] = line2
                return out
    return lines


def _trim_tail(lines: List[str], cfg: Dict[str, object]) -> List[str]:
    cleaned = [_preclean_candidate_text(x, cfg) for x in lines]
    end = len(cleaned)
    for i in range(len(cleaned) - 1, -1, -1):
        line = cleaned[i]
        if not line:
            end = i
            continue
        low = line.lower()
        if _looks_like_body_end_break(line, cfg) or _editorial_hits(low) > int(cfg["editorial_tolerance_at_end"]):
            end = i
            continue
        break
    return [x for x in cleaned[:end] if x]


def _is_rubric_only(text: str, cfg: Dict[str, object]) -> bool:
    low = text.lower()
    min_words = int(cfg["min_words_if_no_continuation"])
    return (
        len(_tokenize_words(text)) <= min_words
        and (not _has_body_continuation(low))
        and (_has_strong_body_opener(low) or low.startswith("de ") or _rhetorical_opening_like(low))
    )


# ---------------------------------------------------------------------------
# Extraction and validation
# ---------------------------------------------------------------------------

def _segment_quality(text: str, cfg: Dict[str, object]) -> Dict[str, float]:
    low = text.lower()
    word_count = len(text.split())
    apparatus = _apparatus_hits(low)
    editorial = _editorial_hits(low)
    main = _main_text_hits(low)
    first = text[:220] if text else ""
    last = text[-220:] if text else ""

    start_bad = (
        _classify_editorial_fragment(first, cfg) is not None
        or _apparatus_hits(first.lower()) + _editorial_hits(first.lower()) >= 2
        or bool(_SHORT_CITATION_RE.match(first.lower()))
    )
    end_bad = (
        _classify_editorial_fragment(last, cfg) is not None
        or _apparatus_hits(last.lower()) + _editorial_hits(last.lower()) >= 2
        or bool(_SHORT_CITATION_RE.match(last.lower()))
    )

    score = min(main, 8) * 1.3 - apparatus * 2.0 - editorial * 2.8
    if word_count < int(cfg["min_words_relaxed"]):
        score -= 10.0
    if start_bad:
        score -= 10.0
    if end_bad:
        score -= 8.0
    if _has_body_continuation(low):
        score += 4.0

    keep = (
        word_count >= int(cfg["min_words_relaxed"])
        and not start_bad
        and not end_bad
        and not _is_rubric_only(text, cfg)
        and (editorial <= 1 or main >= editorial + 1)
        and not (apparatus >= int(cfg["apparatus_hard_reject_threshold"]) and main <= 1)
    )
    return {"score": score, "keep": float(keep)}


def _next_anchor_line(current_key, occ, annotated_lines, cfg):
    book, title, law = current_key
    candidates = []
    for (b, t, l), idxs in occ.items():
        if b == book and t == title and l > law:
            anchors = _candidate_anchor_indices((b, t, l), idxs, annotated_lines, cfg)
            if anchors:
                candidates.append(anchors[0])
    return min(candidates) if candidates else None


def _extend_after_rubric(collected: List[str], stop_at: int, next_anchor: Optional[int], annotated_lines, cfg: Dict[str, object]) -> List[str]:
    extra_lines: List[str] = []
    ext_limit = 18 if str(cfg["body_mode"]) == "rhetorical" else 12
    extra_stop = min(len(annotated_lines), (next_anchor if next_anchor is not None else len(annotated_lines)) + ext_limit)
    for j in range(stop_at, extra_stop):
        raw = annotated_lines[j][3]
        if not raw:
            continue
        line = _preclean_candidate_text(raw, cfg)
        if not line:
            continue
        if _looks_like_body_end_break(line, cfg):
            break
        if _looks_like_main_law_body(line, cfg):
            extra_lines.append(line)
    return collected + extra_lines if extra_lines else collected


def _extract_from_anchor(key, anchor_idx, annotated_lines, occ, cfg) -> str:
    body_start = _find_body_start(anchor_idx, annotated_lines, cfg)
    if body_start is None:
        return ""
    next_anchor = _next_anchor_line(key, occ, annotated_lines, cfg)
    stop_at = next_anchor if next_anchor is not None else len(annotated_lines)

    collected: List[str] = []
    bad_run = 0
    good_run = 0
    break_limit = 3 if str(cfg["boundary_break_strictness"]) in {"low", "medium"} else 2

    for j in range(body_start, stop_at):
        raw = annotated_lines[j][3]
        if not raw:
            if good_run > 0:
                bad_run += 1
                if bad_run >= break_limit + 1:
                    break
            continue

        stop_for_new_law = False
        for m in _LAW_ID_RE.finditer(raw):
            b = _to_int_maybe(m.group("book"))
            t = _to_int_maybe(m.group("title"))
            l = _to_int_maybe(m.group("law"))
            if b == key[0] and t == key[1] and l is not None and l != key[2]:
                stop_for_new_law = True
                break
        if stop_for_new_law:
            break

        line = _preclean_candidate_text(raw, cfg)
        if not line:
            continue

        if _looks_like_body_end_break(line, cfg):
            if good_run == 0:
                continue
            bad_run += 1
            if bad_run >= break_limit:
                break
            continue

        if _looks_like_main_law_body(line, cfg):
            collected.append(line)
            good_run += 1
            bad_run = 0
        elif good_run > 0:
            bad_run += 1
            if bad_run >= break_limit:
                break

    collected = _trim_head(collected, cfg)
    collected = _trim_tail(collected, cfg)
    text = clean_text(" ".join(collected))

    # Mandatory post-rubric extension
    if cfg.get("allow_rubric_extension", False) and (_is_rubric_only(text, cfg) or len(_tokenize_words(text)) < int(cfg["min_words_if_no_continuation"])):
        extended = _extend_after_rubric(collected, stop_at, next_anchor, annotated_lines, cfg)
        extended = _trim_tail(_trim_head(extended, cfg), cfg)
        text2 = clean_text(" ".join(extended))
        if len(text2.split()) > len(text.split()):
            text = text2

    return text


def _normalize_for_similarity(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _too_similar(prev_text: str, cur_text: str, cfg: Dict[str, object]) -> bool:
    if not prev_text or not cur_text:
        return False
    a = _normalize_for_similarity(prev_text)
    b = _normalize_for_similarity(cur_text)
    if a == b:
        return True
    head_n = 160 if str(cfg["dedupe_guard"]) in {"strict", "very_strict"} else 140
    tail_n = head_n
    if str(cfg["dedupe_guard"]) == "very_strict":
        return a[:head_n] == b[:head_n] or a[-tail_n:] == b[-tail_n:]
    return a[:head_n] == b[:head_n] and a[-tail_n:] == b[-tail_n:]


# ---------------------------------------------------------------------------
# Public segmentation
# ---------------------------------------------------------------------------

def segment_lex_visigothorum(text: str, source_name: str) -> List[Tuple[str, str]]:
    expected_ids: Set[Tuple[int, int, int]] = {
        (b, t, l) for (b, t), laws in EXPECTED_IDS_BY_TITLE.items() for l in laws
    }

    legal_text = _extract_core_liber_iudiciorum_block(text)
    annotated_lines = _strip_title_frontmatter(_parse_structure(legal_text.splitlines()))
    occ = _find_law_occurrences(annotated_lines)

    produced: "OrderedDict[str, str]" = OrderedDict()
    last_text_by_book_title: Dict[Tuple[int, int], str] = {}

    # Pass 1: expected-id-driven base extraction
    for key in sorted(expected_ids):
        if key not in occ:
            continue
        book, title, law = key
        cfg = _cfg_for(book, title)
        best_text = ""
        best_score = float("-inf")

        for anchor_idx in _candidate_anchor_indices(key, occ[key], annotated_lines, cfg, recovery=False)[: int(cfg["anchor_candidates_limit"])]:
            candidate = _extract_from_anchor(key, anchor_idx, annotated_lines, occ, cfg)
            if not candidate:
                continue
            q = _segment_quality(candidate, cfg)
            if q["score"] > best_score:
                best_score = q["score"]
                best_text = candidate
            if q["keep"] >= 1.0 and q["score"] >= 1.0:
                best_text = candidate
                break

        if not best_text:
            continue
        if _segment_quality(best_text, cfg)["keep"] < 1.0:
            continue
        prev = last_text_by_book_title.get((book, title))
        if prev and _too_similar(prev, best_text, cfg):
            continue

        seg_id = f"{source_name}_{book}.{title}.{law}"
        produced[seg_id] = best_text
        last_text_by_book_title[(book, title)] = best_text

    # Pass 2: missing-id recovery
    produced_ids = set()
    for seg_id in produced:
        b, t, l = map(int, seg_id.split("_", 1)[1].split("."))
        produced_ids.add((b, t, l))

    for key in sorted(expected_ids - produced_ids):
        if key not in occ:
            continue
        book, title, law = key
        cfg = _cfg_for(book, title)
        best_text = ""
        best_score = float("-inf")
        limit = max(int(cfg["anchor_candidates_limit"]), 12)

        for anchor_idx in _candidate_anchor_indices(key, occ[key], annotated_lines, cfg, recovery=True)[:limit]:
            candidate = _extract_from_anchor(key, anchor_idx, annotated_lines, occ, cfg)
            if not candidate:
                continue
            q = _segment_quality(candidate, cfg)
            if q["score"] > best_score:
                best_score = q["score"]
                best_text = candidate
            if q["keep"] >= 1.0 and q["score"] >= 0.5:
                best_text = candidate
                break

        if not best_text:
            continue
        if _segment_quality(best_text, cfg)["keep"] < 1.0:
            continue
        prev = last_text_by_book_title.get((book, title))
        if prev and _too_similar(prev, best_text, cfg):
            continue

        seg_id = f"{source_name}_{book}.{title}.{law}"
        produced[seg_id] = best_text
        last_text_by_book_title[(book, title)] = best_text

    filtered = []
    for seg_id, seg_text in produced.items():
        b, t, l = map(int, seg_id.split("_", 1)[1].split("."))
        if (b, t, l) in expected_ids:
            filtered.append((seg_id, seg_text))

    filtered.sort(key=lambda x: tuple(map(int, x[0].split("_", 1)[1].split("."))))
    return validate_segments(filtered, source_name)


def segment_lex_visigothorum_unified(source_file, source_name):
    return segment_lex_visigothorum(read_source_file(source_file), source_name)


def main():
    candidates = [
        Path("data/legesvisigothor00zeumgoog_text.txt"),
        Path("legesvisigothor00zeumgoog_text.txt"),
        Path("/mnt/data/legesvisigothor00zeumgoog_text.txt"),
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
