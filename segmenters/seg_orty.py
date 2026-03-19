"""Specialized segmenter for Costums d'Orta (1296).

Structure:
  - Preamble: "Pateat universis..."
  - Article 0: "Primo, quod..."
  - Articles 1-38: "Item, quod..." (unmarked)
  - Articles 39-78: "[XL]...[LXXXI] Item, quod..." (roman numerals)
  - Closing formulas (Mandantes, signatures, dates)

Strategy:
  1. Merge continuation paragraphs (DOCX formatting artifacts)
  2. Extract preamble (Pateat paragraph)
  3. Extract Primo article
  4. Extract all "Item, quod" paragraphs (both unmarked and roman-numbered)
  5. Ignore closing formulas after [LXXXI]
"""
import re
from typing import List, Tuple

# Roman numeral marker at start: [XL], [XLI], etc.
_ROMAN_ARTICLE_RE = re.compile(r'^\[([IVXLCDM]+)\]\s*Item,')

# Closing formulas to skip
_CLOSING_MARKERS = [
    "Mandantes universis",
    "Ad hec autem nos",
    "Quod fuit actum",
    "Sig+num fratris",
    "Testes sunt",
    "PRIVILEGI DIT GENERALMENT",  # Start of next document
]

# Patterns indicating incomplete paragraph (ends mid-sentence)
_INCOMPLETE_END_RE = re.compile(r'(\s+de|\s+in|\s+vel|\s+et)$')


def segment_orty(text: str, source_name: str, 
                  min_words: int = 10) -> List[Tuple[str, str]]:
    """Extract 80 articles from Costums d'Orta.

    Args:
        text: Raw text from DOCX
        source_name: Source identifier (e.g., "CostOrty")
        min_words: Minimum words per segment (default 10)

    Returns:
        List of (segment_id, segment_text) tuples
    """
    lines = text.split('\n')
    raw_paragraphs = [line.strip() for line in lines if line.strip()]

    # Step 1: Merge continuation paragraphs
    paragraphs = _merge_continuations(raw_paragraphs)

    segments = []
    unmarked_item_count = 0

    for para in paragraphs:
        # Skip very short paragraphs
        if len(para) < 30:
            continue

        # Skip closing formulas
        if any(marker in para for marker in _CLOSING_MARKERS):
            continue

        # Skip title and metadata
        if "COSTUMBRES DE ORTA" in para or "Original en pergamino" in para:
            continue

        words = len(para.split())
        if words < min_words:
            continue

        # Clean text
        cleaned = _clean_text(para)

        # 1. Extract preamble
        if cleaned.startswith("Pateat universis"):
            segments.append((f"{source_name}_Preamble", cleaned))
            continue

        # 2. Extract Primo article
        if cleaned.startswith("Primo, quod") or cleaned.startswith("Primo,"):
            segments.append((f"{source_name}_Art0", cleaned))
            continue

        # 3. Extract roman-numbered articles [XL], [XLI], etc.
        roman_match = _ROMAN_ARTICLE_RE.match(cleaned)
        if roman_match:
            roman_num = roman_match.group(1)
            # Remove the [NUM] prefix for cleaner text
            article_text = re.sub(r'^\[[IVXLCDM]+\]\s*', '', cleaned)
            segments.append((f"{source_name}_Art{roman_num}", article_text))
            continue

        # 4. Extract unmarked "Item, quod" articles
        if cleaned.startswith("Item, quod") or cleaned.startswith("Item,"):
            unmarked_item_count += 1
            segments.append((f"{source_name}_Art{unmarked_item_count}", cleaned))
            continue

    return segments


def _merge_continuations(paragraphs: List[str]) -> List[str]:
    """Merge paragraph continuations split by DOCX formatting.

    Strategy: If paragraph ends with preposition/conjunction + next paragraph 
    doesn't start with article marker → merge.
    """
    merged = []
    i = 0

    while i < len(paragraphs):
        para = paragraphs[i]

        # Check if incomplete (ends with de, in, vel, et, etc.)
        while i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1]

            # Don't merge if next starts with article marker
            if (next_para.startswith("Item,") or 
                next_para.startswith("Primo,") or
                next_para.startswith("[") or
                next_para.startswith("Pateat")):
                break

            # Don't merge if current seems complete (ends with period)
            if para.rstrip().endswith('.'):
                break

            # Merge continuation
            para = para + " " + next_para
            i += 1

        merged.append(para)
        i += 1

    return merged


def _clean_text(text: str) -> str:
    """Clean text: normalize whitespace, remove artifacts."""
    # Remove hyphenation artifacts
    text = re.sub(r'-(\s+)', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def segment_orty_unified(
    source_file, source_name
):
    """
    Унифицированная сегментация Orty.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с min_words (передаём min_words)
    min_words = 10
    raw_segments = segment_orty(text, source_name, min_words=min_words)

    # Валидация
    return validate_segments(raw_segments, source_name)
if __name__ == "__main__":
    from pathlib import Path
    import docx

    doc_path = Path("data/ObychaiOrty1296.docx")
    if doc_path.exists():
        doc = docx.Document(str(doc_path))
        text = "\n".join(par.text for par in doc.paragraphs)

        segs = segment_orty(text, "CostOrty")

        print(f"{'='*80}")
        print(f"COSTUMS D'ORTY (1296) — SEGMENTATION RESULT")
        print(f"{'='*80}")
        print(f"Total segments: {len(segs)}")
        print(f"\n{'='*80}")
        print("ALL SEGMENTS")
        print(f"{'='*80}\n")

        for sid, stxt in segs:
            words = len(stxt.split())
            preview = stxt[:80] + "..." if len(stxt) > 80 else stxt
            print(f"{sid:25s} | {words:3d}w | {preview}")
    else:
        print(f"File not found: {doc_path}")
