#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for Costums de Lleida (1228)
Обычаи Лериды - сегментация латинского текста

Format: Articles marked with [N] where N is article number (1-171)
Also contains Roman numeral sections [I], [II], [III] for major divisions
"""

import re
from pathlib import Path
from typing import List, Tuple


def segment_lleida(text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Costums de Lleida into individual articles.

    Args:
        text: Full text of Costums de Lleida
        debug: If True, print debugging information

    Returns:
        List of (article_id, article_text) tuples
    """
    lines = text.split('\n')

    # Pattern for articles: [N] De ... or [I], [II], [III] for sections
    article_pattern = re.compile(r'^\s*\[(\d+|[IVX]+)\]\s+(.+?)$')

    boundaries = []

    for i, line in enumerate(lines):
        match = article_pattern.match(line)
        if match:
            article_id = match.group(1)
            title = match.group(2).strip()
            boundaries.append({
                'id': article_id,
                'line': i,
                'title': title
            })

            if debug and len(boundaries) <= 20:
                print(f"Article [{article_id:>3s}] at line {i:4d}: {title[:60]}...")

    if debug:
        print(f"\nTotal articles found: {len(boundaries)}")

    # Extract text for each article
    articles = []

    for idx, boundary in enumerate(boundaries):
        article_id = boundary['id']
        start_line = boundary['line']

        # End is the start of next article or end of document
        if idx + 1 < len(boundaries):
            end_line = boundaries[idx + 1]['line']
        else:
            end_line = len(lines)

        # Extract lines for this article
        article_lines = lines[start_line:end_line]

        # Clean up the text
        article_text = extract_article_text(article_lines)

        if article_text.strip():
            articles.append((article_id, article_text))

    # Sort by article number (convert to int where possible)
    def sort_key(item):
        article_id = item[0]
        # Roman numerals come first (converted to negative numbers)
        if re.match(r'^[IVX]+$', article_id):
            roman_values = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
            return (-1000, roman_values.get(article_id, 0))
        # Arabic numbers
        try:
            return (int(article_id), 0)
        except ValueError:
            return (9999, 0)

    articles.sort(key=sort_key)
    
    # Новый шаг: фильтруем только арабские номера
    articles = [
        (aid, txt)
        for aid, txt in articles
        if aid.isdigit()
    ]

    return articles


def extract_article_text(lines: List[str]) -> str:
    """
    Extract clean article text from lines.
    Includes the article header and body.
    """
    result = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        result.append(stripped)

    # Join with spaces
    text = ' '.join(result)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def analyze_and_save(text: str, output_file: str, expected_count: int = 171):
    """
    Analyze segmentation and save results.
    """
    print("=" * 80)
    print("COSTUMS DE LLEIDA (1228) - ARTICLE SEGMENTATION")
    print("=" * 80)

    articles = segment_lleida(text, debug=False)

    # Statistics
    article_ids = [aid for aid, _ in articles]
    roman_articles = [aid for aid in article_ids if re.match(r'^[IVX]+$', aid)]
    arabic_articles = [aid for aid in article_ids if re.match(r'^\d+$', aid)]

    coverage = len(arabic_articles) / expected_count * 100

    print(f"Expected: {expected_count} articles")
    print(f"Found: {len(articles)} total articles")
    print(f"  - Roman numeral sections: {len(roman_articles)}")
    print(f"  - Numbered articles: {len(arabic_articles)}")
    print(f"Coverage: {coverage:.1f}%")

    # Check for duplicates
    from collections import Counter
    duplicates = [(n, c) for n, c in Counter(article_ids).items() if c > 1]

    if duplicates:
        print(f"\nDuplicates: {len(duplicates)}")
        for num, count in sorted(duplicates)[:5]:
            print(f"  Article {num}: {count} times")
    else:
        print("\nNo duplicates")

    # Check for missing articles
    expected_nums = set(str(i) for i in range(1, expected_count + 1))
    found_nums = set(arabic_articles)
    missing = sorted([int(n) for n in (expected_nums - found_nums)])

    if missing:
        print(f"\nMissing: {len(missing)} articles")
        if len(missing) <= 15:
            print(f"  {missing}")
        else:
            print(f"  First 10: {missing[:10]}")
            print(f"  Last 5: {missing[-5:]}")
    else:
        print("\nNo missing articles")

    print("=" * 80)

    # Verdict
    if coverage >= 90 and not duplicates:
        print(f"✓ SUCCESS: {coverage:.1f}% coverage, no duplicates")
    else:
        print(f"⚠ {coverage:.1f}% coverage, {len(duplicates)} duplicates")

    print("=" * 80)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total articles: {len(articles)}\n")
        f.write(f"Roman sections: {len(roman_articles)}, Numbered: {len(arabic_articles)}\n")
        f.write("=" * 80 + "\n\n")

        for article_id, text in articles:
            f.write("=" * 80 + "\n")
            f.write(f"ARTICLE [{article_id}]\n")
            f.write("=" * 80 + "\n")
            f.write(text)
            f.write("\n\n")

    print(f"Results saved to {output_file}")

    return articles


def main():
    """Main entry point"""
    file_lleida = Path('data/ObychaiLleidy12271228_v2.txt')

    if file_lleida.exists():
        print(f"Processing Costums de Lleida from {file_lleida}...")
        text = file_lleida.read_text(encoding='utf-8')
        docs = analyze_and_save(text, 
                               output_file='costums_lleida_segmented.txt',
                               expected_count=171)
    else:
        print(f"Error: {file_lleida} not found")


def segment_lleida_unified(
    source_file, source_name
):
    """
    Унифицированная сегментация Lleida.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с debug=False
    raw_segments = segment_lleida(text, debug=False)

    # Валидация
    return validate_segments(raw_segments, source_name)
if __name__ == '__main__':
    main()
