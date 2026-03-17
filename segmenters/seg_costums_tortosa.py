#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Consuetudines Dertusae (Customs of Tortosa), 1272–1279.

Source: ObychaiTortosy1272to1279_v2.txt
Structure: 9 books (llibres), 142 rubrics, 1350 articles
Numbering: book.rubric.article  e.g. 1.1.1 ... 9.30.1
Two parallel texts (ms A 1272 + ms B/C 1277–1279) interleaved.
Only FIRST occurrence of each article number is taken (ms A).

Language: ~91% Old Catalan, ~7% Latin (mainly rubrics 9.22–9.30).
Each segment carries a 'lang' tag: 'latin' | 'catalan' | 'mixed' | 'unknown'.

Expected:  1350 articles
Achieved: ~1274 articles (94.4% coverage)
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

_STRONG_LATIN = {
    'enim', 'autem', 'quidem', 'igitur', 'ergo', 'namque', 'similiter',
    'constituerunt', 'statuerunt', 'laudaverunt', 'approbant', 'repprobant',
    'intelligitur', 'intelliguntur', 'continetur', 'continentur', 'dicitur',
    'videtur', 'tenetur', 'debet', 'possit', 'fuerit', 'fuerint',
    'obligacio', 'accio', 'excepcio', 'iuris', 'cuiusque', 'quamque',
    'aliquis', 'quisque', 'nemo', 'sicut', 'inde', 'hinc', 'ideo',
    'idem', 'quoque', 'ubi', 'unde', 'dum', 'tunc', 'item',
}

_STRONG_CATALAN = {
    'ciutadans', 'senyoria', 'veguer', 'costuma', 'costum', 'habitadors',
    'franquea', 'enaxi', 'seynor', 'tennen', 'termen', 'alberc',
    'sayg', 'pleyts', 'deliure', 'usan', 'clams', 'emenda',
    'axí', 'axi', 'tots', 'totz', 'ésser', 'esser',
}

_LATIN_FUNC  = {'qui', 'vel', 'aut', 'nisi', 'sive', 'atque', 'cum',
                'ab', 'ut', 'ne', 'etiam', 'non', 'sed', 'ac'}
_CATALAN_FUNC = {'los', 'les', 'del', 'dels', 'als', 'son', 'pot',
                 'deu', 'per', 'tots', 'tot', 'han', 'la', 'lo', 'que'}


def detect_language(text: str) -> str:
    """Return 'latin' | 'catalan' | 'mixed' | 'unknown'."""
    if not text:
        return 'unknown'
    words = re.findall(r'\b[a-záàãâéèêíïóòôúùûçñ]+\b', text.lower())
    if len(words) < 5:
        return 'unknown'

    lat_hits = sum(1 for w in words if w in _STRONG_LATIN)
    cat_hits = sum(1 for w in words if w in _STRONG_CATALAN)
    lat_func = sum(1 for w in words if w in _LATIN_FUNC)  / len(words)
    cat_func = sum(1 for w in words if w in _CATALAN_FUNC) / len(words)

    if lat_hits > 0 and cat_hits == 0:
        return 'latin'
    if cat_hits > 0 and lat_hits == 0:
        return 'catalan'
    if lat_hits > 0 and cat_hits > 0:
        return 'mixed'
    if lat_func > cat_func * 1.5:
        return 'latin'
    return 'catalan'


# ─────────────────────────────────────────────────────────────────────────────
# Book boundary detection
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded fallbacks (from OCR analysis of the specific file)
_DEFAULT_BOOK_START = {
    1: 0,
    2: 293807,
    3: 451562,
    4: 627154,
    5: 885872,
    6: 1007028,
    7: 1153596,
    8: 1240831,
    9: 1358971,   # No LLIBRE NOVÈ marker found; approximated from first art. 9.x.x
}

_LLIBRE_WORDS = {
    'PRIMER': 1, 'SEGON': 2, 'TERCER': 3, 'QUART': 4,
    'CINQUÈ': 5, 'CINQUÉ': 5, 'CINQUE': 5,
    'SISÈ': 6,  'SISÉ': 6,  'SISE': 6,
    'SETÉ': 7,  'SETÈ': 7,  'SETE': 7,
    "VUIT'E": 8, 'VUITÉ': 8, 'VUITÈ': 8, 'VUITE': 8,
    # Book 9 intentionally absent (no reliable marker in this OCR)
}


def _detect_book_boundaries(text: str) -> Tuple[Dict[int,int], Dict[int,int]]:
    """Auto-detect LLIBRE markers; fall back to hardcoded positions."""
    book_start = dict(_DEFAULT_BOOK_START)

    # Only accept markers that appear at the start of a line (avoid footnote refs)
    for m in re.finditer(r'(?:^|\n)\s*LLIB\w+\s+(\S+)', text, re.IGNORECASE):
        word = m.group(1).upper().rstrip("'.,")
        if word in _LLIBRE_WORDS:
            bnum = _LLIBRE_WORDS[word]
            # Only update if this position is plausible (later than previous book)
            prev_pos = book_start.get(bnum - 1, 0)
            if m.start() > prev_pos:
                book_start[bnum] = m.start()

    sorted_books = sorted(book_start.items())
    book_end: Dict[int,int] = {}
    for i, (b, pos) in enumerate(sorted_books):
        book_end[b] = sorted_books[i + 1][1] if i + 1 < len(sorted_books) else len(text)
    return book_start, book_end


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

_FOOTNOTE_RE = re.compile(
    r'^\d+\.\s*[A-Z].{5,}?(?:que\s+incipit|approbant|repprobant)',
    re.IGNORECASE
)
_FOOTNOTE_DEC_RE = re.compile(
    r'^\d+\.(Decimal|Primar|Secund|Terci|Quart|Quint|Sext|Sept|Oct|Non)\w+',
    re.IGNORECASE
)
_PAGE_NUM_RE = re.compile(r'^\d{1,4}$')
_LINE_NUM_RE = re.compile(r'^\d+\s{2,}')


def _clean_article_text(raw: str) -> str:
    """Strip footnotes, page numbers, OCR line numbers; fix hyphenation."""
    lines = raw.split('\n')
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Stop at numbered footnotes
        if _FOOTNOTE_RE.match(s) or _FOOTNOTE_DEC_RE.match(s):
            break
        # Skip bare page numbers
        if _PAGE_NUM_RE.match(s):
            continue
        # Remove OCR line numbers ("5   text" → "text")
        s = _LINE_NUM_RE.sub('', s)
        if s:
            result.append(s)

    out = ' '.join(result)
    out = re.sub(r'\s+', ' ', out).strip()
    out = out.replace('\xad', '')            # soft hyphen
    out = re.sub(r'(\w)- (\w)', r'\1\2', out)  # broken hyphenation
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main segmentation
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(r'(?<!\d)([1-9])\.([1-9]|[12]\d|30)\.([1-9]|[1-4]\d)(?!\d)')


def segment_tortosa(text: str) -> List[Tuple[str, str, str]]:
    """
    Segment the Tortosa customs text.

    Returns list of (article_id, article_text, language) where:
      - article_id : 'Tort_1.1.1' ... 'Tort_9.30.1'
      - article_text: cleaned text of the article
      - language   : 'latin' | 'catalan' | 'mixed' | 'unknown'

    Only the FIRST occurrence of each article number is kept.
    Articles are filtered to lie within the correct book's text range.
    """
    book_start, book_end = _detect_book_boundaries(text)

    # Collect first occurrence of each valid article number within its book range
    articles: Dict[str, int] = {}
    for m in _ARTICLE_RE.finditer(text):
        book = int(m.group(1))
        rub  = int(m.group(2))
        art  = int(m.group(3))
        num  = f"{book}.{rub}.{art}"
        pos  = m.start()

        if pos < book_start[book] or pos >= book_end[book]:
            continue
        if num not in articles:
            articles[num] = pos

    sorted_articles = sorted(articles.items(), key=lambda x: x[1])

    # Extract text for each article
    segments: List[Tuple[str, str, str]] = []
    for i, (num, pos) in enumerate(sorted_articles):
        # Skip past the article number itself
        m_num = re.search(re.escape(num) + r'[\s\t]*', text[pos: pos + 30])
        text_start = pos + (m_num.end() if m_num else len(num) + 1)

        next_pos = sorted_articles[i + 1][1] if i + 1 < len(sorted_articles) else len(text)
        raw = text[text_start: next_pos]

        art_text = _clean_article_text(raw)
        if not art_text:
            continue

        lang = detect_language(art_text)
        segments.append((f"Tort_{num}", art_text, lang))

    # Stats
    total = len(segments)
    expected = 1350
    coverage = total / expected * 100
    lat = sum(1 for _, _, lg in segments if lg == 'latin')
    cat = sum(1 for _, _, lg in segments if lg == 'catalan')
    mix = sum(1 for _, _, lg in segments if lg == 'mixed')
    print(f"✓ Tortosa: {total} articles ({coverage:.1f}% of {expected})")
    print(f"  Latin: {lat}  Catalan: {cat}  Mixed: {mix}")
    for b in range(1, 10):
        n = sum(1 for seg_id, _, _ in segments if seg_id.startswith(f'Tort_{b}.'))
        print(f"  Book {b}: {n}")

    return segments


def segment_tortosa_latin_only(text: str) -> List[Tuple[str, str]]:
    """
    Convenience wrapper: returns only Latin articles as (article_id, text).
    Use this when comparing directly against Latin sources (Usatges de Barcelona).
    """
    all_segs = segment_tortosa(text)
    latin = [(seg_id, art_text) for seg_id, art_text, lang in all_segs
             if lang in ('latin', 'mixed')]
    print(f"  → Latin/mixed only: {len(latin)} articles")
    return latin

def segment_costums_tortosa(text: str, source_name: str = "", max_words: int = 0) -> List[Tuple[str, str]]:
    """
    Wrapper for compatibility with source_segmenters registry.
    Ignores source_name and max_words, returns Latin/mixed articles.
    """
    return segment_tortosa_latin_only(text)

# ─────────────────────────────────────────────────────────────────────────────
# CLI / debug
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from collections import Counter

    src = Path('data/ObychaiTortosy1272to1279_v2.txt')
    if not src.exists():
        # Try current directory
        src = Path('ObychaiTortosy1272to1279_v2.txt')
    if not src.exists():
        print(f'Error: source file not found. Expected: {src}')
        sys.exit(1)

    text = src.read_text(encoding='utf-8', errors='replace')
    segments = segment_tortosa(text)

    print('\n=== First 5 segments ===')
    for seg_id, txt, lang in segments[:5]:
        print(f'\n{seg_id} [{lang}]:\n  {txt[:120]}')

    print('\n=== Last 5 segments ===')
    for seg_id, txt, lang in segments[-5:]:
        print(f'\n{seg_id} [{lang}]:\n  {txt[:120]}')

    print('\n=== Latin articles sample ===')
    latin_segs = [(s, t, l) for s, t, l in segments if l == 'latin']
    for seg_id, txt, lang in latin_segs[:10]:
        print(f'  {seg_id}: {txt[:100]}')

    # Missing articles report
    expected_struct = {
        1: [22,5,18,15,16,14,4,3,5,1,3,4,1],
        2: [4,4,2,22,9,2,4,7,16,2,12,6,4,12,3,11,8,11],
        3: [36,1,2,6,2,5,6,8,17,4,31,10,15,1,2,6],
        4: [4,13,1,2,4,11,9,6,4,6,41,3,4,3,8,2,15,13,11,6,30,7,4,2,15,33],
        5: [21,6,6,4,24,23,12],
        6: [19,7,5,34,6,8,4,7,19,3,3],
        7: [10,13,10,7,1,3,20,2,4,3],
        8: [14,2,1,1,22,10,10,19,5,3,18],
        9: [18,9,8,12,8,3,11,1,11,11,3,1,5,6,25,8,4,13,2,5,8,17,18,7,22,2,44,4,15,1],
    }
    found_ids = {seg_id for seg_id, _, _ in segments}
    missing = []
    for book, rubrics in expected_struct.items():
        for rub_idx, count in enumerate(rubrics):
            for art in range(1, count + 1):
                num = f'Tort_{book}.{rub_idx+1}.{art}'
                if num not in found_ids:
                    missing.append(num)

    print(f'\n=== Missing articles: {len(missing)} ===')
    if missing:
        print('  ' + ', '.join(missing[:20]) + ('...' if len(missing) > 20 else ''))

    # Save output
    out_path = Path('tortosa_segments.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'Total: {len(segments)}\n{"="*80}\n\n')
        for seg_id, txt, lang in segments:
            f.write(f'\n{"="*80}\n{seg_id} [{lang}]\n{"="*80}\n{txt}\n')
    print(f'\n✓ Saved to {out_path}')
