"""
Specialized segmenter for Constituciones de Miravet (1319).

Structure:
- Historical introduction (editorial matter, skip)
- Preamble: "In Christi nomine et individue Trinitatis..."
- Article 0: "Primo confirmant nobis dominus castellanus..."
- Articles 1-N: Each starting with "Item" or "ltem" (lowercase L)
- Closing formulas: Signatures, notarial confirmations (skip)

Strategy:
1. Skip editorial introduction and metadata
2. Extract preamble (In Christi nomine paragraph)
3. Extract Primo article as Article 0
4. Extract all Item/ltem articles sequentially
5. Skip references to Consuetudines Ilerdenses
6. Skip closing formulas
"""

import re
from typing import List, Tuple

# Skip markers for editorial matter and closing formulas
SKIP_MARKERS = [
    'CONSTITUClONES B AIULI',
    'DON EDUARDO DE HINOJOSA',
    'HISTORIA Y FUENTES',
    'LA PRESENTE EDICIN',
    'LOSMANUSCRITOS',
    'LA REDACCIN CATALANA',
    'Et nos frater Elionus',
    'Sigffinum',
    'Testes huius',
    'Ego Raymundus',
    'Nos Petrus Sancii',
    'Folio',
    'Polio',
    'Villanueva, Viage',
    'El manuscrito',
    'TITLE DOC',
]

# Patterns for source references (to skip)
REFERENCE_PATTERN = re.compile(r'^\s*\d+\.\s*Cf\.\s+Consuetudines', re.IGNORECASE)
CF_PATTERN = re.compile(r'^\s*Cf\.\s+Cons', re.IGNORECASE)

# Main patterns
PREAMBLE_PATTERN = re.compile(r'In Christi nomine et individue Trinitatis', re.IGNORECASE)
PRIMO_PATTERN = re.compile(r'Primo\s+confirmant\s+nobis', re.IGNORECASE)
ITEM_PATTERN = re.compile(r'^l?tem\s+', re.IGNORECASE)  # Matches "Item" or "ltem" (lowercase L)


def segment_miravet(text: str, source_name: str = "Miravet", min_words: int = 10) -> List[Tuple[str, str]]:
    """
    Extract articles from Constituciones de Miravet.

    Args:
        text: Raw text from document file
        source_name: Source identifier (default "Miravet")
        min_words: Minimum words per segment (default 10)

    Returns:
        List of (segment_id, segment_text) tuples
    """

    lines = text.split('\n')

    segments = []
    current_article_num = None
    current_article_lines = []
    in_articles = False
    found_preamble = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty or very short lines
        if not line or len(line) < 10:
            i += 1
            continue

        # Skip editorial matter and metadata
        if any(marker in line for marker in SKIP_MARKERS):
            i += 1
            continue

        # Skip source references
        if REFERENCE_PATTERN.match(line) or CF_PATTERN.match(line):
            i += 1
            continue

        # Extract preamble
        if PREAMBLE_PATTERN.search(line) and not found_preamble:
            found_preamble = True
            preamble_text = line
            segments.append((f"{source_name}:Preamble", clean_text(preamble_text)))
            i += 1
            continue

        # Start of articles: "Primo confirmant nobis..."
        if PRIMO_PATTERN.search(line) and not in_articles:
            in_articles = True
            current_article_num = 0
            current_article_lines = [line]
            i += 1
            continue

        # Process articles
        if in_articles:
            # New article starts with "Item" or "ltem"
            if ITEM_PATTERN.match(line):
                # Save previous article
                if current_article_num is not None and current_article_lines:
                    article_text = ' '.join(current_article_lines)
                    if len(article_text.split()) >= min_words:
                        segments.append((f"{source_name}:Art{current_article_num}", clean_text(article_text)))

                # Start new article
                current_article_num += 1
                current_article_lines = [line]
                i += 1
                continue
            else:
                # Continuation of current article
                if current_article_lines:
                    # Don't add source references
                    if not (CF_PATTERN.match(line) or REFERENCE_PATTERN.match(line)):
                        current_article_lines.append(line)
                i += 1
                continue

        i += 1

    # Don't forget last article
    if current_article_num is not None and current_article_lines:
        article_text = ' '.join(current_article_lines)
        if len(article_text.split()) >= min_words:
            segments.append((f"{source_name}:Art{current_article_num}", clean_text(article_text)))

    return segments


def clean_text(text: str) -> str:
    """Clean text: remove hyphenation artifacts, normalize whitespace."""
    # Remove hyphenation artifacts
    text = re.sub(r'-(\s+)', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


if __name__ == "__main__":
    from pathlib import Path

    # Test with text file
    txt_path = Path("data/ObychaiMiraveta1319Fix.txt")

    if txt_path.exists():
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        segs = segment_miravet(text, "Miravet")

        print("=" * 80)
        print("CONSTITUCIONES DE MIRAVET (1319) SEGMENTATION RESULT")
        print("=" * 80)
        print(f"Total segments: {len(segs)}")
        print("=" * 80)

        # Statistics
        total_words = sum(len(s[1].split()) for s in segs)
        avg_words = total_words / len(segs) if segs else 0
        print(f"\nTotal words: {total_words}")
        print(f"Average words per segment: {avg_words:.1f}")
        print("=" * 80)

        print("\nFIRST 10 SEGMENTS:")
        print("=" * 80)
        for sid, stxt in segs[:10]:
            words = len(stxt.split())
            preview = stxt[:80] + "..." if len(stxt) > 80 else stxt
            print(f"{sid:25s} ({words:3d}w) {preview}")

        print("\n" + "=" * 80)
        print("LAST 10 SEGMENTS:")
        print("=" * 80)
        for sid, stxt in segs[-10:]:
            words = len(stxt.split())
            preview = stxt[:80] + "..." if len(stxt) > 80 else stxt
            print(f"{sid:25s} ({words:3d}w) {preview}")
    else:
        print(f"File not found: {txt_path}")
