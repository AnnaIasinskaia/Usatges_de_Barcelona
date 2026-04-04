#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Consuetudines Dertusae (Customs of Tortosa), 1272–1279.

Unified contract: returns list[(id, text)] with meaningful IDs like:
    Tort_1.1.1
    Tort_4.12.7
    Tort_9.30.1

This version is tuned for the OCR in ObychaiTortosy1272to1279_v2.txt.
The goal is not perfect philological cleanup, but stable structural segmentation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .seg_common import read_source_file, validate_segments


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_STRONG_LATIN = {
    "enim", "autem", "quidem", "igitur", "ergo", "namque", "similiter",
    "constituerunt", "statuerunt", "laudaverunt", "approbant", "repprobant",
    "intelligitur", "intelliguntur", "continetur", "continentur", "dicitur",
    "videtur", "tenetur", "debet", "possit", "fuerit", "fuerint",
    "obligacio", "accio", "excepcio", "iuris", "cuiusque", "quamque",
    "aliquis", "quisque", "nemo", "sicut", "inde", "hinc", "ideo",
    "idem", "quoque", "ubi", "unde", "dum", "tunc", "item",
}
_STRONG_CATALAN = {
    "ciutadans", "senyoria", "veguer", "costuma", "costum", "habitadors",
    "franquea", "enaxi", "seynor", "tennen", "termen", "alberc",
    "sayg", "pleyts", "deliure", "usan", "clams", "emenda",
    "axí", "axi", "tots", "totz", "ésser", "esser",
}
_LATIN_FUNC = {"qui", "vel", "aut", "nisi", "sive", "atque", "cum", "ab", "ut", "ne", "etiam", "non", "sed", "ac"}
_CATALAN_FUNC = {"los", "les", "del", "dels", "als", "son", "pot", "deu", "per", "tots", "tot", "han", "la", "lo", "que"}


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    words = re.findall(r"\b[a-záàãâéèêíïóòôúùûçñ]+\b", text.lower())
    if len(words) < 5:
        return "unknown"

    lat_hits = sum(1 for w in words if w in _STRONG_LATIN)
    cat_hits = sum(1 for w in words if w in _STRONG_CATALAN)
    lat_func = sum(1 for w in words if w in _LATIN_FUNC) / len(words)
    cat_func = sum(1 for w in words if w in _CATALAN_FUNC) / len(words)

    if lat_hits > 0 and cat_hits == 0:
        return "latin"
    if cat_hits > 0 and lat_hits == 0:
        return "catalan"
    if lat_hits > 0 and cat_hits > 0:
        return "mixed"
    if lat_func > cat_func * 1.5:
        return "latin"
    return "catalan"


# ---------------------------------------------------------------------------
# Book boundary detection
# ---------------------------------------------------------------------------

_DEFAULT_BOOK_START = {
    1: 0,
    2: 293807,
    3: 451562,
    4: 627154,
    5: 885872,
    6: 1007028,
    7: 1153596,
    8: 1240831,
    9: 1358971,
}
_LLIBRE_WORDS = {
    "PRIMER": 1, "SEGON": 2, "TERCER": 3, "QUART": 4,
    "CINQUÈ": 5, "CINQUÉ": 5, "CINQUE": 5,
    "SISÈ": 6, "SISÉ": 6, "SISE": 6,
    "SETÉ": 7, "SETÈ": 7, "SETE": 7,
    "VUIT'E": 8, "VUITÉ": 8, "VUITÈ": 8, "VUITE": 8,
}


def _detect_book_boundaries(text: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    book_start = dict(_DEFAULT_BOOK_START)
    for m in re.finditer(r"(?:^|\n)\s*LLIB\w+\s+(\S+)", text, re.IGNORECASE):
        word = m.group(1).upper().rstrip("'.,")
        if word in _LLIBRE_WORDS:
            bnum = _LLIBRE_WORDS[word]
            prev_pos = book_start.get(bnum - 1, 0)
            if m.start() > prev_pos:
                book_start[bnum] = m.start()

    sorted_books = sorted(book_start.items())
    book_end: Dict[int, int] = {}
    for i, (b, pos) in enumerate(sorted_books):
        book_end[b] = sorted_books[i + 1][1] if i + 1 < len(sorted_books) else len(text)
    return book_start, book_end


# ---------------------------------------------------------------------------
# Article number detection
# ---------------------------------------------------------------------------

_ARTICLE_LINE_RE = re.compile(
    r"(?m)^[ \t\u00a0\u2000-\u200b\u202f\u3000]*([1-9])\.([1-9]|[12]\d|30)\.([1-9]|[1-4]\d)\b"
)

_ARTICLE_ANY_RE = re.compile(
    r"(?<!\d)([1-9])\.([1-9]|[12]\d|30)\.([1-9]|[1-4]\d)(?!\d)"
)


def _collect_article_positions(text: str, book_start: Dict[int, int], book_end: Dict[int, int]) -> Dict[str, int]:
    """
    Prefer line-start article numbers. If an article was not seen there,
    allow a fallback to any-position matches inside the correct book range.
    """
    articles: Dict[str, int] = {}

    for rx in (_ARTICLE_LINE_RE, _ARTICLE_ANY_RE):
        for m in rx.finditer(text):
            book = int(m.group(1))
            rub = int(m.group(2))
            art = int(m.group(3))
            pos = m.start()
            if pos < book_start[book] or pos >= book_end[book]:
                continue
            num = f"{book}.{rub}.{art}"
            if num not in articles:
                articles[num] = pos
    return articles


# ---------------------------------------------------------------------------
# OCR / note cleanup
# ---------------------------------------------------------------------------

_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_LINE_NUM_RE = re.compile(r"^\d+\s{2,}")
_LEADING_ARTICLE_RE = re.compile(r"^[1-9]\.([1-9]|[12]\d|30)\.([1-9]|[1-4]\d)\b")

_START_NOTE_RE = re.compile(
    r"^\s*\d+\.\s*(?:En|El|La|Les|Els|Prim|Secund|Terci|Terc|Quart|Quint|Sext|Sept|Oct|Non|Coronatges|Host|Comu|Privilegis|Sent[eè]ncia)",
    re.IGNORECASE,
)
_START_LATAPP_RE = re.compile(
    r"^\s*(?:Primam?|Secundam?|Terciam?|Quartam?|Quintam?|Sextam?|Septimam?|Octavam?|Nonam?)\b.{0,120}?(?:incipit|approbant|repprobant|dicunt arbi)",
    re.IGNORECASE,
)

_INLINE_CUT_PATTERNS = [
    re.compile(r"\s+\d+\.\([^)]+\):", re.IGNORECASE),
    re.compile(
        r"\s+\d+\.\s*(?:En|El|La|Les|Els|Prim|Secund|Terci|Terc|Quart|Quint|Sext|Sept|Oct|Non|Coronatges|Host|Comu|Privilegis|Sent[eè]ncia)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:Primam?|Secundam?|Terciam?|Quartam?|Quintam?|Sextam?|Septimam?|Octavam?|Nonam?)\b.{0,120}?(?:incipit|approbant|repprobant|dicunt arbi)",
        re.IGNORECASE,
    ),
    re.compile(r"\s+(?:Coronatges num\.|Host [IVX]+ num\.|Comu de Tortosa|Privilegis|La sent[èe]ncia|Els [àa]rbitres)\b", re.IGNORECASE),
]

_INLINE_CUT_SUBSTRINGS = [
    "ad primam consuetud",
    "ad secundam consuetud",
    "ad terciam consuetud",
    "approbant huius",
    "dicunt arbitri",
    "repprobant",
    "coronatges num.",
    "host ii num.",
    "host i num.",
    "comu de tortosa",
    "privilegis",
    "la sentència",
    "els àrbitres",
]


def _clean_article_text(raw: str) -> str:
    """
    Keep the article body, but stop before large editorial / apparatus blocks.
    This is intentionally heuristic and conservative.
    """
    lines = raw.splitlines()
    kept: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _PAGE_NUM_RE.match(s):
            continue

        s = _LINE_NUM_RE.sub("", s).strip()
        if not s:
            continue

        if kept and _LEADING_ARTICLE_RE.match(s):
            break

        if _START_NOTE_RE.match(s) or _START_LATAPP_RE.match(s):
            break

        kept.append(s)

    out = " ".join(kept)
    out = out.replace("\xad", "")
    out = re.sub(r"(\w)-\s+(\w)", r"\1\2", out)
    out = re.sub(r"\s+", " ", out).strip()

    for rx in _INLINE_CUT_PATTERNS:
        m = rx.search(out)
        if m:
            out = out[:m.start()].rstrip()

    low = out.lower()
    cut_pos = None
    for marker in _INLINE_CUT_SUBSTRINGS:
        idx = low.find(marker)
        if idx != -1 and (cut_pos is None or idx < cut_pos):
            cut_pos = idx
    if cut_pos is not None:
        out = out[:cut_pos].rstrip()

    out = re.sub(r"\s+", " ", out).strip(" ;,.\u00b7")
    return out


# ---------------------------------------------------------------------------
# Main segmentation
# ---------------------------------------------------------------------------

def segment_tortosa(text: str, source_name: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (article_id, article_text, language).
    Only first occurrence of each article number is kept.
    """
    book_start, book_end = _detect_book_boundaries(text)
    articles = _collect_article_positions(text, book_start, book_end)
    sorted_articles = sorted(articles.items(), key=lambda x: x[1])

    segments: List[Tuple[str, str, str]] = []
    for i, (num, pos) in enumerate(sorted_articles):
        m_num = re.search(re.escape(num) + r"[\s\t]*", text[pos: pos + 40])
        text_start = pos + (m_num.end() if m_num else len(num) + 1)
        next_pos = sorted_articles[i + 1][1] if i + 1 < len(sorted_articles) else len(text)
        raw = text[text_start:next_pos]

        art_text = _clean_article_text(raw)
        if not art_text:
            continue

        lang = detect_language(art_text)
        segments.append((f"{source_name}_{num}", art_text, lang))

    return segments


def segment_costums_tortosa(text: str, source_name: str) -> List[Tuple[str, str]]:
    raw_triples = segment_tortosa(text, source_name)
    
    # оставить только книгу 9
    raw_triples = [(sid, txt, lang) for sid, txt, lang in raw_triples if sid.split("_")[1].startswith("9.")]

    
    return [(seg_id, art_text) for seg_id, art_text, _ in raw_triples]


def segment_costums_tortosa_unified(source_file, source_name):
    text = read_source_file(source_file)
    raw_segments = segment_costums_tortosa(text, source_name)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/ObychaiTortosy1272to1279_v2.txt"),
        Path("ObychaiTortosy1272to1279_v2.txt"),
        Path("/mnt/data/ObychaiTortosy1272to1279_v2.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_costums_tortosa_unified(src, "ObychaiTortosy1272to1279")
    print(f"ObychaiTortosy1272to1279: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()