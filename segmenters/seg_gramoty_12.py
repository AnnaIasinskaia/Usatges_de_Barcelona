#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Gramoty XII.

Правила:
- segment_id сохраняются
- сверху выкидываются датировка, regest и аппарат
- в segment_text остаётся только латинский текст грамоты
- снизу аппарат / поздние комментарии / повторная датировка обрезаются
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import read_source_file, validate_segments


DOC_NUM_PATTERN1 = re.compile(r"^\s*(\d+)\s*$")
DOC_NUM_PATTERN2 = re.compile(r"^\s*###\s*DOC\s+(\d+)\s*###\s*$", re.IGNORECASE)

DATE_PATTERNS = [
    re.compile(r"\b1[012]\d{2}\b"),
    re.compile(r"\[.*?1[012]\d{2}.*?\]"),
    re.compile(r"Segle\s+xi{1,3}", re.IGNORECASE),
]
MONTH_RE = re.compile(
    r"\b(gener|febrer|mar[cç]|abril|maig|juny|juliol|agost|setembre|octubre|novembre|desembre)\b",
    re.IGNORECASE,
)

ARCHIVE_CODES = r"(ACA|BC|ADG|ACU|AHAT|ACC|ADS|AMM|ADC|BRAH|BNF|ADM|ACT|BPT)"

APPARATUS_LINE_PATTERNS = [
    re.compile(r"^\[A\]\s+", re.IGNORECASE),
    re.compile(r"^\[B\]\s+", re.IGNORECASE),
    re.compile(r"^\[C\]\s+", re.IGNORECASE),
    re.compile(r"^[A-Z]\s+" + ARCHIVE_CODES + r"\b", re.IGNORECASE),
    re.compile(r"^\[?[A-Z]\]?\s+" + ARCHIVE_CODES + r"\b", re.IGNORECASE),
    re.compile(r"^Ed\.\s+", re.IGNORECASE),
    re.compile(r"^\*", re.IGNORECASE),
    re.compile(r"^Original no localitzat", re.IGNORECASE),
    re.compile(r"^Seguim\b", re.IGNORECASE),
    re.compile(r"^Vegeu\b", re.IGNORECASE),
    re.compile(r"^Aquesta declaració\b", re.IGNORECASE),
    re.compile(r"^Aquest document\b", re.IGNORECASE),
    re.compile(r"^Bibliografia\b", re.IGNORECASE),
    re.compile(r"^FONTS I BIBLIOGRAFIA\b", re.IGNORECASE),
]

PAGE_HEADER_PATTERNS = [
    re.compile(r"^(Documents|TEXTOS JUR[IÍ]DICS CATALANS|JUST[IÍ]CIA I RESOLUCI[ÓO])\b", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
]

LATIN_INCIPIT_RE = re.compile(
    r"^(Notum|Noverint|Hec|Haec|Hoc|Ista|Iste|Ego|Nos|Placitum|Conuenientia|Convenientia|Carta|Noticia|Quoniam|In nomine|Facta|Vox)\b"
)

LATIN_WORDS = {
    "et", "in", "cum", "per", "ad", "de", "ab", "seu", "non", "sic", "ut",
    "unde", "qui", "quod", "iam", "est", "erat", "fuit", "eius", "ipsa", "ipse",
    "ipsi", "ipsum", "sibi", "suo", "sua", "suus", "dictus", "presentibus",
    "futuris", "hanc", "cartam", "carta", "homines", "testes", "comes", "comite",
    "abbas", "abba", "episcopus", "episcopi", "clericus", "monasterii", "villa",
    "anno", "die", "firmamus", "successoribus", "amen", "prior", "alodem",
    "iurato", "dixit", "respondit", "interrogatus", "coluit", "mandavit",
    "donamus", "definimus", "ecclesie", "domini", "dominus", "regnante",
    "sig", "signum", "presbiter", "levita", "scripsit", "actum", "huius",
}

ROMANCE_MARKERS = {
    "després", "declaració", "resolució", "conflicte", "jutge", "jutges", "comte",
    "comtessa", "esposa", "esposes", "presència", "església", "monestir", "castell",
    "davant", "entre", "sobre", "molts", "altres", "causa", "donen", "venen",
    "defineixen", "testimonis", "disputa", "sentència", "arriba", "acord",
    "perquè", "seva", "seu", "seus", "volia", "homes", "dona", "reconeixen",
    "reconeix", "remei", "ànima", "pecats", "proposta", "datació",
}


def segment_gramoty_12(text: str, debug: bool = False) -> List[Tuple[int, str]]:
    lines = text.split("\n")
    doc_boundaries = []

    for i, line in enumerate(lines):
        doc_num = None
        format_type = None

        m1 = DOC_NUM_PATTERN1.match(line)
        if m1:
            doc_num = int(m1.group(1))
            format_type = "standard"

        m2 = DOC_NUM_PATTERN2.match(line)
        if m2:
            doc_num = int(m2.group(1))
            format_type = "marked"

        if doc_num is None:
            continue

        has_date = False
        date_line = None

        if format_type == "marked":
            has_date = True
            if i > 0 and any(p.search(lines[i - 1]) for p in DATE_PATTERNS):
                date_line = lines[i - 1].strip()
        else:
            for offset in range(1, min(7, len(lines) - i)):
                nxt = lines[i + offset]
                if not nxt.strip():
                    continue
                if any(p.search(nxt) for p in DATE_PATTERNS):
                    has_date = True
                    date_line = nxt.strip()
                    break
                if offset > 4:
                    break

        if format_type == "marked" or has_date:
            doc_boundaries.append({
                "num": doc_num,
                "line": i,
                "format": format_type,
                "date_line": date_line,
            })

            if debug and len(doc_boundaries) <= 20:
                print(f"Found doc {doc_num} at line {i} [{format_type}]")

    documents: List[Tuple[int, str]] = []
    for idx, boundary in enumerate(doc_boundaries):
        start_line = boundary["line"]
        end_line = doc_boundaries[idx + 1]["line"] if idx + 1 < len(doc_boundaries) else len(lines)
        doc_lines = lines[start_line:end_line]

        extracted = _extract_latin_12(doc_lines[1:])
        if extracted.strip():
            documents.append((boundary["num"], extracted))
        elif debug:
            print(f"WARNING: Doc {boundary['num']}: empty after extraction")

    return documents


def _is_page_header(stripped: str) -> bool:
    return any(p.match(stripped) for p in PAGE_HEADER_PATTERNS)


def _is_apparatus_line(stripped: str) -> bool:
    return any(p.match(stripped) for p in APPARATUS_LINE_PATTERNS)


def _normalize_words(text: str) -> list[str]:
    low = text.lower().replace("v", "u").replace("j", "i")
    return re.findall(r"[a-zà-ÿ]+", low, re.IGNORECASE)


def _romance_score(stripped: str) -> float:
    low = stripped.lower()
    words = _normalize_words(stripped)
    score = 0.0

    if re.match(r"^\d{4}\s*,", stripped):
        score += 6.0
    if stripped.startswith("[") and any(p.search(stripped) for p in DATE_PATTERNS):
        score += 5.0
    if MONTH_RE.search(stripped):
        score += 3.0
    if "l’" in low or "d’" in low or "s’ha" in low:
        score += 2.0

    for marker in ROMANCE_MARKERS:
        if marker in low:
            score += 1.2

    # typical prose regest lines
    if len(words) >= 8 and any(w in {"que", "els", "les", "dels", "del", "una", "molts"} for w in words):
        score += 2.0

    return score


def _latin_score(stripped: str) -> float:
    words = _normalize_words(stripped)
    if not words:
        return 0.0

    score = 0.0
    if LATIN_INCIPIT_RE.match(stripped):
        score += 8.0

    for w in words:
        if w in LATIN_WORDS:
            score += 1.2
        if w.endswith(("us", "um", "am", "em", "ibus", "orum", "arum", "ae", "tur", "unt", "imus", "nt", "is")):
            score += 0.35

    # signature / dating zone often still Latin, so allow it
    if "Sig+" in stripped or "Signum" in stripped or "scripsit" in stripped:
        score += 2.0

    return score


def _is_probably_latin_line(stripped: str) -> bool:
    if not stripped:
        return False
    if _is_page_header(stripped) or _is_apparatus_line(stripped):
        return False
    latin = _latin_score(stripped)
    romance = _romance_score(stripped)

    if LATIN_INCIPIT_RE.match(stripped):
        return True
    if latin >= 6.0 and latin >= romance + 1.5:
        return True
    if latin >= 4.5 and romance <= 1.5:
        return True
    return False


def _find_latin_start(lines: List[str]) -> int:
    """
    Надёжнее всего здесь не классифицировать верхние блоки по отдельности,
    а найти первую строку, которая действительно выглядит латинской.
    """
    if not lines:
        return 0

    for i, line in enumerate(lines[:180]):
        stripped = line.strip()
        if _is_probably_latin_line(stripped):
            return i

    # fallback: best line by latin-romance margin
    best_i = 0
    best_margin = float("-inf")
    for i, line in enumerate(lines[:180]):
        stripped = line.strip()
        if not stripped or _is_page_header(stripped) or _is_apparatus_line(stripped):
            continue
        margin = _latin_score(stripped) - _romance_score(stripped)
        if margin > best_margin:
            best_margin = margin
            best_i = i
    return best_i


def _should_stop_latin(stripped: str) -> bool:
    if _is_apparatus_line(stripped):
        return True
    if re.match(r"^\[?\d{4}(?:[/-]\d{2})?\]?$", stripped):
        return True
    if re.match(r"^\[?\d{4}(?:[/-]\d{2})?,\s*(?:gener|febrer|mar[cç]|abril|maig|juny|juliol|agost|setembre|octubre|novembre|desembre)", stripped, re.IGNORECASE):
        return True
    if re.search(r"\b(pàg\.|f\.\s*\d|núm\.\s*\d|Cartoral|perg\.)\b", stripped, re.IGNORECASE):
        return True
    # once charter text ended, new regest/comment line should stop extraction
    if _romance_score(stripped) >= 4.0 and _latin_score(stripped) <= 2.0:
        return True
    return False


def _extract_latin_12(lines: List[str]) -> str:
    if not lines:
        return ""

    start_idx = _find_latin_start(lines)
    latin_lines: List[str] = []
    empty_count = 0

    for line in lines[start_idx:]:
        stripped = line.strip()

        if _is_page_header(stripped):
            continue

        if not stripped:
            empty_count += 1
            if empty_count >= 3:
                break
            if latin_lines:
                latin_lines.append(line)
            continue

        empty_count = 0

        if _should_stop_latin(stripped):
            break

        latin_lines.append(line)

    text = "\n".join(latin_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def segment_gramoty_12_unified(source_file, source_name):
    text = read_source_file(source_file)
    raw_pairs = segment_gramoty_12(text, debug=False)
    segments = [(f"{source_name}_{doc_num}", doc_text) for doc_num, doc_text in raw_pairs]
    return validate_segments(segments, source_name)


def main() -> None:
    candidates = [
        Path("data/Gramoty12.txt"),
        Path("Gramoty12.txt"),
        Path("/mnt/data/Gramoty12.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_gramoty_12_unified(src, "Gramoty12")
    print(f"Gramoty12: {len(segs)} segments")
    for sid, txt in segs[:3]:
        print(f"{sid}: {txt[:120]}")


if __name__ == "__main__":
    main()
