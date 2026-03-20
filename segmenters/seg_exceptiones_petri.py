#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmenter for Exceptiones Legum Romanorum Petri (new edition / OCR v3)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

_ORDINAL_WORDS = {
    "primum": 1,
    "primi": 1,
    "primo": 1,
    "primit": 1,
    "primiti": 1,
    "secundum": 2,
    "secundi": 2,
    "tertium": 3,
    "tertii": 3,
    "quartum": 4,
    "quarti": 4,
    "quintum": 5,
    "quinti": 5,
    "sextum": 6,
    "sexti": 6,
    "septimum": 7,
    "septimi": 7,
    "octavum": 8,
    "octavi": 8,
    "nonum": 9,
    "noni": 9,
    "decimum": 10,
    "decimi": 10,
}

_BOOK_PATTERNS: List[Tuple[int, re.Pattern[str]]] = [
    (1, re.compile(r"LIBER\s*P\s*RIMV?S")),
    (2, re.compile(r"LIBER\s*S\s*E\s*C\s*V?N?D?V?S")),
    (3, re.compile(r"LIBER\s*T\s*E\s*R\s*T\s*I\s*V?S")),
    (4, re.compile(r"LIBER\s*Q\s*V?\s*A\s*R\s*T\s*V?S")),
    (5, re.compile(r"LIBER\s*Q\s*V?\s*I\s*N\s*T\s*V?S")),
]

_CHAPTER_HEAD_RE = re.compile(
    r"(?P<title>"
    r"(?:De|Decodem|Q\s*ui|Tudex|Ne\s+quis|Mulieres|Dos|Inter|Nuptie|Cum)"
    r"[^\.]{1,80}?"
    r")"
    r"\s*[,.:;—\-| ]+\s*"
    r"(?:"
    r"Capitulum\s+(?P<cap_word>[A-Za-z]+)"
    r"|"
    r"(?:ca|c[ao@]|@)\.?\s*(?P<cap_tok>[A-Za-z0-9£íivxljfg]{1,10})"
    r")"
)


def _fold_ascii(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _safe_strip(text: str) -> str:
    return " ".join(text.split()).strip()


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    low = _fold_ascii(s).lower()

    if low.startswith(
        (
            "erstellungsdatum",
            "titel:",
            "ort:",
            "verlag:",
            "jahr:",
            "doi /",
            "nutzungsbedingungen",
            "http",
        )
    ):
        return True

    if "digi.ub.uni" in low:
        return True

    if "universitats-bibliothek" in low or "universitatsbibliothek" in low:
        return True

    if "heidelberg" in low and ("universit" in low or "digi." in low or "http" in low):
        return True

    if "baden-wurtt" in low or "badenwurtt" in low or "wilrtiemberg" in low:
        return True

    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 2 and len(s) < 8:
        return True

    return False


def _prepare_text(text: str) -> str:
    kept: List[str] = []
    for line in text.splitlines():
        if _is_noise_line(line):
            continue
        kept.append(line.strip())

    joined = " ".join(kept)
    joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def _roman_to_int(token: str) -> Optional[int]:
    t = _fold_ascii(token).lower()
    t = re.sub(r"[^a-z0-9£]", "", t)
    if not t:
        return None

    if t in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[t]

    t = (
        t.replace("j", "i")
        .replace("£", "i")
        .replace("f", "i")
        .replace("t", "i")
        .replace("r", "i")
        .replace("g", "ii")
    )
    t = "".join(ch for ch in t if ch in "ivxlcdm")
    if not t:
        return None

    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(t):
        val = vals[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val

    if 0 < total < 200:
        return total
    return None


def _normalize_title(title: str) -> str:
    t = _safe_strip(title)
    t = re.sub(r"^[\"'`\-–—:;,.| ]+", "", t)
    t = re.sub(r"[\"'`\-–—:;,.| ]+$", "", t)
    t = t.replace("Q ui", "Qui")
    return t


def _find_books(text: str) -> List[Tuple[int, int, int]]:
    found: List[Tuple[int, int, int]] = []
    for book_no, pat in _BOOK_PATTERNS:
        for m in pat.finditer(text):
            found.append((m.start(), m.end(), book_no))

    found.sort()
    dedup: List[Tuple[int, int, int]] = []
    last_pos = -10**9
    for start, end, book_no in found:
        if start - last_pos < 40:
            continue
        dedup.append((start, end, book_no))
        last_pos = start
    return dedup


def _find_chapter_starts(block: str) -> List[Tuple[int, int, str]]:
    starts: List[Tuple[int, int, str]] = []

    expected_next = 1
    for m in _CHAPTER_HEAD_RE.finditer(block):
        title = _normalize_title(m.group("title") or "")
        if not title:
            continue

        token = (m.group("cap_word") or m.group("cap_tok") or "").strip()
        parsed = _roman_to_int(token)

        if parsed is None or parsed < expected_next or parsed > expected_next + 5:
            cap_no = expected_next
        else:
            cap_no = parsed

        starts.append((m.start(), cap_no, title))
        expected_next = cap_no + 1

    collapsed: List[Tuple[int, int, str]] = []
    last_pos = -10**9
    last_no = None
    for pos, cap_no, title in starts:
        if pos - last_pos < 25 and last_no == cap_no:
            continue
        collapsed.append((pos, cap_no, title))
        last_pos = pos
        last_no = cap_no

    return collapsed


def _clean_segment_text(text: str) -> str:
    t = text
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    t = re.sub(r"\bFol\.?\s*[A-Za-z0-9'«»\.\-]+\b", " ", t)
    t = re.sub(r"\bFo[blf]?\.?\s*[A-Za-z0-9'«»\.\-]+\b", " ", t)
    t = re.sub(r"\bLI+\s*FO\.?\s*[A-Za-z0-9]+\b", " ", t)
    t = re.sub(r"\bCapitula\s+[A-Z][^.]{0,120}", " ", t)
    t = re.sub(r"\bEXCEPCIONVM\s+LEGVM[^.]{0,80}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return clean_text(t)


# ---------------------------------------------------------
# Main segmentation
# ---------------------------------------------------------

def segment_exceptiones_petri(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    OCR-aware segmentation for Exceptiones Petri.

    Structural unit:
      one segment = one chapter inside a detected Liber.

    IDs:
      ExceptPetri_L1_C1
      ExceptPetri_L1_C2
      ExceptPetri_L4_C1
    """
    prepared = _prepare_text(text)
    books = _find_books(prepared)
    if not books:
        return []

    raw_segments: List[Tuple[str, str]] = []

    for i, (book_start, book_head_end, book_no) in enumerate(books):
        book_end = books[i + 1][0] if i + 1 < len(books) else len(prepared)
        block = prepared[book_head_end:book_end].strip()
        if not block:
            continue

        chapter_starts = _find_chapter_starts(block)
        if not chapter_starts:
            continue

        for j, (start_pos, cap_no, title) in enumerate(chapter_starts):
            end_pos = chapter_starts[j + 1][0] if j + 1 < len(chapter_starts) else len(block)
            chunk = block[start_pos:end_pos].strip()
            cleaned = _clean_segment_text(chunk)
            if not cleaned:
                continue

            seg_id = f"{source_name}_L{book_no}_C{cap_no}"
            raw_segments.append((seg_id, cleaned))

    best: Dict[str, str] = {}
    for seg_id, seg_text in raw_segments:
        cur = best.get(seg_id)
        if cur is None or len(seg_text) > len(cur):
            best[seg_id] = seg_text

    segments = sorted(best.items(), key=lambda x: x[0])
    return validate_segments(segments, source_name)


def segment_exceptiones_petri_unified(source_file, source_name):
    """
    Unified segmenter for Exceptiones Petri.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "ExceptPetri".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    return segment_exceptiones_petri(text, source_name)


def main() -> None:
    candidates = [
        Path("data/Exeptionis_Legum_Romanorum_Petri_v3.txt"),
        Path("Exeptionis_Legum_Romanorum_Petri_v3.txt"),
        Path("/mnt/data/Exeptionis_Legum_Romanorum_Petri_v3.txt"),
        Path("data/Exeptionis_Legum_Romanorum_Petri_v2.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_exceptiones_petri_unified(src, "ExceptPetri")
    print(f"ExceptPetri: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()