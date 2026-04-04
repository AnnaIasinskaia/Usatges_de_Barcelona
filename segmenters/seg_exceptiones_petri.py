#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmenter for Exceptiones Legum Romanorum Petri (Transkribus OCR v4)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .seg_common import clean_text, read_source_file, validate_segments


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

_BOOK_LABELS = {
    1: "PRIMVS",
    2: "SECVNDVS",
    3: "TERCIVS",
    4: "QVARTVS",
    5: "QVINTVS",
}

_ORDINAL_WORDS = {
    "primum": 1,
    "primo": 1,
    "primi": 1,
    "primum.": 1,
    "secundum": 2,
    "secundo": 2,
    "secundi": 2,
    "tertium": 3,
    "tertio": 3,
    "tertii": 3,
    "quartum": 4,
    "quarto": 4,
    "quarti": 4,
    "quintum": 5,
    "quinto": 5,
    "quinti": 5,
    "sextum": 6,
    "sexto": 6,
    "sexti": 6,
    "septimum": 7,
    "septimo": 7,
    "septimi": 7,
    "octauum": 8,
    "octavum": 8,
    "octauo": 8,
    "octavo": 8,
    "octaui": 8,
    "octavi": 8,
    "nonum": 9,
    "nono": 9,
    "noni": 9,
    "decimum": 10,
    "decimo": 10,
    "decimi": 10,
    "primu": 1,
    "secundu": 2,
    "tertiu": 3,
    "quartu": 4,
    "quintu": 5,
    "sextu": 6,
    "septimu": 7,
    "octavu": 8,
    "nonu": 9,
    "decimu": 10,
}

_BOOK_RE = re.compile(
    r"\bLIBER\s+(PRIMVS|SECVNDVS|SECVNDUS|TERCIVS|TERTIUS|QVARTVS|QUARTVS|QVINTVS|QUINTUS)\b",
    re.IGNORECASE,
)

_MARKER_ONLY_RE = re.compile(
    r"^(?:ca\.?|capitulum)\s+([^\s]+)[\.:,;\)]*$",
    re.IGNORECASE,
)

_INLINE_HEADING_RE = re.compile(
    r"^(?P<title>.+?)\s+(?:ca\.?|Capitulum)\s+(?P<num>[^\s]+)[\.:,;\)]*$",
    re.IGNORECASE,
)

_STOP_MARKERS = (
    "TRACTATVS DE ACTIONVM",
    "TRACTATUS DE ACTIONVM",
)


def _fold_ascii(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip()


def _roman_to_int(token: str) -> Optional[int]:
    raw = _fold_ascii(token).lower().replace("j", "i")
    raw = re.sub(r"[^a-z0-9]", "", raw)
    if not raw:
        return None

    if raw in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[raw]

    if raw.endswith("u") and (raw + "m") in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[raw + "m"]

    t = raw.replace("u", "v")
    roman = "".join(ch for ch in t if ch in "ivxlcdm")
    if not roman:
        return None

    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(roman):
        val = vals[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    if 0 < total < 500:
        return total
    return None


def _book_no_from_line(line: str) -> Optional[int]:
    m = _BOOK_RE.search(_fold_ascii(line))
    if not m:
        return None
    lab = m.group(1).upper().replace("U", "V")
    for num, name in _BOOK_LABELS.items():
        if lab == name:
            return num
    return None


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    low = _fold_ascii(s).lower()

    starts = (
        "erstellungsdatum",
        "titel:",
        "ort:",
        "verlag:",
        "jahr:",
        "doi",
        "nutzungsbedingungen",
        "http",
    )
    if low.startswith(starts):
        return True

    noisy_substrings = (
        "heidelberg",
        "universitatsbibliothek",
        "universitats",
        "bibliothek",
        "baden",
        "digi.ub.uni",
        "control grey",
        "chart",
    )
    if any(x in low for x in noisy_substrings):
        return True

    if re.fullmatch(r"[LI0-9 .,'/:;\-]+", s) and len(s) <= 20:
        return True
    if re.fullmatch(r"FO\.?\s*[A-Za-z0-9ivxlcdm\.]+", _fold_ascii(s), re.IGNORECASE):
        return True
    if re.fullmatch(r"L[Il1\.]*(?:\s*FO\.?\s*[A-Za-z0-9ivxlcdm\.]+)?", _fold_ascii(s), re.IGNORECASE):
        return True

    letters = sum(ch.isalpha() for ch in s)
    if letters <= 2 and len(s) <= 8:
        return True

    return False


def _drop_front_matter(lines: List[str]) -> List[str]:
    start_idx = None
    for i, line in enumerate(lines):
        if _book_no_from_line(line) == 1:
            start_idx = i
            break
    if start_idx is None:
        return []
    return lines[start_idx:]


def _iter_clean_lines(text: str) -> List[str]:
    raw_lines = [ln.rstrip() for ln in text.splitlines()]
    raw_lines = _drop_front_matter(raw_lines)
    out: List[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if any(marker in s for marker in _STOP_MARKERS):
            break
        if _is_noise_line(s):
            continue
        out.append(s)
    return out


def _is_heading_like(line: str) -> bool:
    s = _normalize_ws(line)
    if not s:
        return False
    if _book_no_from_line(s) is not None:
        return True
    if _MARKER_ONLY_RE.match(s):
        return True
    if _INLINE_HEADING_RE.match(s):
        return True
    if s.startswith(("De ", "Qui ", "Cum ", "Ne ", "Si ")) and len(s) <= 120:
        return True
    return False


def _parse_heading(lines: List[str], i: int) -> Optional[Tuple[int, int, str, int]]:
    """
    Returns (chapter_no, consumed_lines, heading_text, inline_title_len)
    consumed_lines = number of lines used by the heading.
    """
    cur = _normalize_ws(lines[i])

    m = _INLINE_HEADING_RE.match(cur)
    if m:
        no = _roman_to_int(m.group("num"))
        if no is not None:
            title = _normalize_ws(m.group("title"))
            return no, 1, f"{title}. ca. {m.group('num')}", 1

    if i + 1 < len(lines):
        nxt = _normalize_ws(lines[i + 1])
        m2 = _MARKER_ONLY_RE.match(nxt)
        if m2 and not _book_no_from_line(cur):
            no = _roman_to_int(m2.group(1))
            if no is not None and len(cur) <= 160:
                return no, 2, f"{cur}. ca. {m2.group(1)}", 1

    return None


def _clean_joined_text(lines: Iterable[str]) -> str:
    joined = "\n".join(lines)
    joined = joined.replace("¬\n", "")
    joined = re.sub(r"(\w)[\-¬]\n(\w)", r"\1\2", joined)
    joined = re.sub(r"\n+", " ", joined)
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r"\bFo\.?\s*[A-Za-z0-9ivxlcdm\.]+\b", " ", joined, flags=re.IGNORECASE)
    joined = re.sub(r"\bFol\.?\s*[A-Za-z0-9ivxlcdm\.]+\b", " ", joined, flags=re.IGNORECASE)
    joined = re.sub(r"\bCapitula\b[^\.]{0,160}", " ", joined, flags=re.IGNORECASE)
    joined = re.sub(r"\bIncipiunt\b[^\.]{0,160}", " ", joined, flags=re.IGNORECASE)
    joined = re.sub(r"\s+", " ", joined).strip()
    return clean_text(joined)


# ---------------------------------------------------------
# Main segmentation
# ---------------------------------------------------------


def segment_exceptiones_petri(text: str, source_name: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Exceptiones Petri v4 into large structural units.

    Structural unit:
      one segment = one numbered chapter inside a detected Liber.

    IDs:
      ExceptPetri_L1_C1
      ExceptPetri_L1_C2
      ExceptPetri_L4_C1

    Important:
      chapter numbers are taken from the document itself;
      no inferred/guessed numbering is used.
    """
    lines = _iter_clean_lines(text)
    if not lines:
        return []

    segments: List[Tuple[str, str]] = []
    current_book: Optional[int] = None
    current_chapter: Optional[int] = None
    current_heading: Optional[str] = None
    current_body: List[str] = []

    def flush() -> None:
        nonlocal current_book, current_chapter, current_heading, current_body, segments
        if current_book is None or current_chapter is None or current_heading is None:
            current_body = []
            return
        text_lines = [current_heading] + current_body
        seg_text = _clean_joined_text(text_lines)
        if seg_text:
            seg_id = f"{source_name}_{current_book}.{current_chapter}"
            segments.append((seg_id, seg_text))
        current_body = []

    i = 0
    while i < len(lines):
        line = lines[i]

        book_no = _book_no_from_line(line)
        if book_no is not None:
            flush()
            current_book = book_no
            current_chapter = None
            current_heading = None
            current_body = []
            i += 1
            continue

        if current_book is None:
            i += 1
            continue

        parsed = _parse_heading(lines, i)
        if parsed is not None:
            chap_no, consumed, heading_text, _ = parsed
            flush()
            current_chapter = chap_no
            current_heading = heading_text
            current_body = []
            i += consumed
            continue

        if current_chapter is not None:
            # guard against stray TOC / page furniture leaking into body
            if not _is_noise_line(line):
                current_body.append(line)
        i += 1

    flush()

    # keep longest duplicate if the OCR produced repeated headers/pages
    best: Dict[str, str] = {}
    for seg_id, seg_text in segments:
        prev = best.get(seg_id)
        if prev is None or len(seg_text) > len(prev):
            best[seg_id] = seg_text

    ordered = sorted(
        best.items(),
        key=lambda x: tuple(int(n) for n in re.findall(r"\d+", x[0])) or (9999,),
    )

    if debug:
        print(f"Detected segments: {len(ordered)}")
        for sid, txt in ordered[:5]:
            print(sid, txt[:100])

    return validate_segments(ordered, source_name)



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
        Path("data/Exeptionis_Legum_Romanorum_Petri_v4.txt"),
        Path("Exeptionis_Legum_Romanorum_Petri_v4.txt"),
        Path("/mnt/data/Exeptionis_Legum_Romanorum_Petri_v4.txt"),
        Path("data/Exeptionis_Legum_Romanorum_Petri_v3.txt"),
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
