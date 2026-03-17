#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for Pragmática de Jaime II 1301
(Doc 83: ordinances on advocates, procurators, notaries)

Structure:
  - Single document (83.)
  - Spanish editorial header (skip)
  - Preamble: Latin "Capitula facta..." + "Iacobus..." + Catalan intro
  - Articles [I]...[XXIV] with Roman numeral markers
  - Footnotes starting with digit + space (skip, may split article I)
  - Closing Latin formula: "Igitur..." + "Datum..." (skip)

Format differences from 1295:
  - No nested doc numbering — single document only
  - Footnote on line between [I] header and its continuation body
"""

import re
from pathlib import Path
from typing import List, Tuple
from collections import Counter


# ── Patterns ──────────────────────────────────────────────────────────────────

ARTICLE_PATTERN    = re.compile(r'^[\s\u3000]*\[([IVX]+)\](.*)', re.DOTALL)
FOOTNOTE_PATTERN   = re.compile(r'^\s*\d+\s+[A-ZÁÉÍÓÚ]')          # "1 Reseñado…"
DOC_HEADER_PATTERN = re.compile(r'^\d+\.PRAGMÁTICA')               # "83.PRAGMÁTICA…"

CLOSING_MARKERS = [
    "Igitur cum sit nobis",
    "Datum Valencie",
]

SKIP_MARKERS = [
    "Original, en el Archivo",
    "***",
    "Reseñado por",
    "Publicado por",
    "Copia en el Llibre",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def roman_to_int(roman: str) -> int:
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total, prev = 0, 0
    for ch in reversed(roman.upper()):
        v = vals.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = v
    return total


def clean_text(text: str) -> str:
    text = re.sub(r'-\s+', '', text)        # remove hyphenation breaks
    text = re.sub(r'\s+', ' ', text)        # normalise whitespace
    return text.strip()


# ── Core segmenter ────────────────────────────────────────────────────────────

def segment_zhaime1301_text(
    text: str,
    debug: bool = False
) -> List[Tuple[str, str, str]]:
    """
    Segment Pragmática de Jaime II 1301 into articles.

    Returns:
        List of (doc_id, article_id, article_text) where doc_id == "83".
        article_id "0" is the preamble (before [I]).
    """
    lines = text.split('\n')
    DOC_ID = "83"

    articles: List[Tuple[str, str, str]] = []
    current_roman: str | None = None
    current_lines: List[str] = []
    in_articles: bool = False
    preamble_lines: List[str] = []

    def flush():
        nonlocal current_roman, current_lines
        if current_roman is not None and current_lines:
            t = clean_text(" ".join(current_lines))
            if t:
                articles.append((DOC_ID, current_roman, t))
        current_roman = None
        current_lines = []

    for i, raw_line in enumerate(lines):
        line = raw_line.strip().replace('\u3000', ' ')

        if not line:
            continue

        # Closing formula — stop article extraction
        if any(line.startswith(m) for m in CLOSING_MARKERS):
            flush()
            if debug:
                print(f"  Closing formula at line {i}: {line[:60]}")
            break

        # Skip doc header
        if DOC_HEADER_PATTERN.match(line):
            continue

        # Skip footnotes (e.g. "1 Reseñado por…")
        if FOOTNOTE_PATTERN.match(line):
            if debug:
                print(f"  Footnote at {i}: {line[:60]}")
            continue

        # Skip misc editorial markers
        if any(m in line for m in SKIP_MARKERS):
            continue

        # Article marker [ROMAN]
        art_match = ARTICLE_PATTERN.match(line)
        if art_match:
            flush()
            in_articles = True
            current_roman = art_match.group(1)
            rest = art_match.group(2).strip()
            current_lines = [rest] if rest else []
            if debug:
                print(f"  Article [{current_roman}] at {i}: {rest[:50]}")
            continue

        # Accumulate lines
        if in_articles:
            current_lines.append(line)
        else:
            preamble_lines.append(line)

    flush()  # last article

    # Prepend preamble as Art 0
    if preamble_lines:
        preamble_text = clean_text(" ".join(preamble_lines))
        if preamble_text:
            articles.insert(0, (DOC_ID, "0", preamble_text))

    return articles


# ── Analysis + save ───────────────────────────────────────────────────────────

def analyze_and_save_1301(text: str, output_file: str) -> List[Tuple[str, str, str]]:
    """Analyse segmentation, print stats, save to file."""

    print("=" * 80)
    print("PRAGMÁTICA DE JAIME II 1301 – ARTICLE SEGMENTATION")
    print("=" * 80)

    articles = segment_zhaime1301_text(text, debug=True)

    docs: dict = {}
    for doc_id, art_id, art_text in articles:
        docs.setdefault(doc_id, []).append((art_id, art_text))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(docs)}")
    for doc_id in sorted(docs):
        print(f"  Document {doc_id}: {len(docs[doc_id])} articles")
    print(f"\nTotal articles: {len(articles)}")

    # Duplicate check
    for doc_id, art_list in docs.items():
        ids = [aid for aid, _ in art_list]
        dupes = [(n, c) for n, c in Counter(ids).items() if c > 1]
        if dupes:
            print(f"\nWARNING: Duplicates in document {doc_id}:")
            for art_id, count in dupes:
                print(f"  Article {art_id} × {count}")
        else:
            print(f"\nDocument {doc_id}: No duplicates ✓")

    print("=" * 80)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PRAGMÁTICA DE JAIME II 1301 – SEGMENTED ARTICLES\n")
        f.write(f"Total documents: {len(docs)}\n")
        f.write(f"Total articles: {len(articles)}\n")
        f.write("=" * 80 + "\n\n")

        for doc_id in sorted(docs):
            sorted_arts = sorted(
                docs[doc_id],
                key=lambda x: roman_to_int(x[0]) if x[0] != "0" else 0
            )
            f.write(f"\n{'=' * 80}\n")
            f.write(f"DOCUMENT {doc_id}\n")
            f.write(f"Articles: {len(sorted_arts)}\n")
            f.write(f"{'=' * 80}\n\n")
            for art_id, art_text in sorted_arts:
                f.write(f"{'=' * 80}\n")
                f.write(f"ARTICLE {doc_id}.{art_id}\n")
                f.write(f"{'=' * 80}\n")
                f.write(art_text)
                f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    print("=" * 80)
    return articles


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    file_path = Path("data/PragmatikaZhaumeII1301_v2.txt")
    if file_path.exists():
        text = file_path.read_text(encoding='utf-8')
        analyze_and_save_1301(text, output_file="pragmatica_zhaime1301_segmented.txt")
    else:
        print(f"Error: {file_path} not found")


if __name__ == '__main__':
    main()
