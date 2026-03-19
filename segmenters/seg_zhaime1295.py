#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for Pragmática de Jaime II 1295
Format: Articles marked with Roman numerals in brackets [I], [II], etc.
Contains multiple documents (82, 83, etc.)
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict

def segment_zhaime1295_text(text: str, debug: bool = False) -> List[Tuple[str, str, str]]:
    """
    Segment Pragmática de Jaime II 1295 into individual articles.

    Args:
        text: Full text of Pragmática
        debug: If True, print debugging information

    Returns:
        List of (doc_id, article_id, article_text) tuples
    """
    lines = text.split('\n')

    # Find document boundaries (82., 83., etc.)
    doc_pattern = re.compile(r'^(\d+)\.')
    doc_boundaries = []

    for i, line in enumerate(lines):
        match = doc_pattern.match(line.strip())
        if match:
            doc_num = match.group(1)
            doc_boundaries.append({'num': doc_num, 'line': i})
            if debug:
                print(f"Document {doc_num} found at line {i}")

    if debug:
        print(f"\nTotal documents found: {len(doc_boundaries)}")

    # Process each document separately
    all_articles = []

    for doc_idx, doc_info in enumerate(doc_boundaries):
        doc_num = doc_info['num']
        start_line = doc_info['line']

        # Determine end line for this document
        if doc_idx + 1 < len(doc_boundaries):
            end_line = doc_boundaries[doc_idx + 1]['line']
        else:
            end_line = len(lines)

        doc_lines = lines[start_line:end_line]
        doc_text = '\n'.join(doc_lines)

        # Find articles within this document
        articles = segment_articles_in_doc_1295(doc_text, doc_num, start_line, debug)
        all_articles.extend(articles)

    return all_articles


def segment_articles_in_doc_1295(text: str, doc_id: str, offset: int, debug: bool = False) -> List[Tuple[str, str, str]]:
    """
    Segment articles within a single document from Pragmática 1295.

    Args:
        text: Text of the document
        doc_id: Document ID (e.g., "82", "83")
        offset: Line offset in the original file
        debug: Debug flag

    Returns:
        List of (doc_id, article_id, article_text) tuples
    """
    lines = text.split('\n')

    # Pattern for article markers: [I], [II], [III], etc.
    # Also handles variations like [I]Quod or [II]Item
    article_pattern = re.compile(r'^[\s\u3000]*\[([IVX]+)\](.*)$')

    boundaries = []

    for i, line in enumerate(lines):
        match = article_pattern.match(line)
        if match:
            article_id = match.group(1)
            title = match.group(2).strip()

            boundaries.append({
                'id': article_id,
                'line': i,
                'title': title[:60] if title else '(no title)'
            })

            if debug and len(boundaries) <= 25:
                print(f"  Doc {doc_id}, Article {article_id:>4s} at line {offset+i:4d}: {title[:60]}")

    if debug:
        print(f"  Document {doc_id}: {len(boundaries)} articles found")

    # Extract text for each article
    articles = []

    for idx, boundary in enumerate(boundaries):
        article_id = boundary['id']
        start_line = boundary['line']

        # Find end: next article or end of document
        if idx + 1 < len(boundaries):
            end_line = boundaries[idx + 1]['line']
        else:
            end_line = len(lines)

        article_lines = lines[start_line:end_line]
        article_text = extract_article_text_1295(article_lines)

        if article_text.strip():
            articles.append((doc_id, article_id, article_text))

    return articles


def extract_article_text_1295(lines: List[str]) -> str:
    """
    Extract clean article text from lines for Pragmática 1295.
    Removes empty lines and excessive whitespace.
    """
    result = []

    for line in lines:
        # Remove leading ideographic spaces and regular spaces
        stripped = line.strip().replace('\u3000', ' ').strip()
        if not stripped:
            continue
        result.append(stripped)

    text = ' '.join(result)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer for sorting."""
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }

    total = 0
    prev_value = 0

    for char in reversed(roman.upper()):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


def segment_zhaime1295_unified(
    source_file,
    source_name
):
    """
    Унифицированная функция сегментации для Pragmática de Jaime II 1295.
    Соответствует контракту из INTERFACE.md.

    Параметры
    ---------
    source_file : str или Path
        Путь к файлу с текстом (формат .txt или .docx).
    source_name : str
        Имя источника (например, "PragmatikaZhaumeII1295").
    min_words : int, optional
        Минимальное количество слов в сегменте (по умолчанию 10).
    max_words : int, optional
        Максимальное количество слов в сегменте (по умолчанию 150).

    Возвращает
    ----------
    List[Tuple[str, str]]
        Список сегментов в формате (segment_id, segment_text).
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments
    text = read_source_file(source_file)
    raw_triples = segment_zhaime1295_text(text, debug=False)
    # Преобразуем тройки (doc_id, article_id, text) в пары (id, text)
    segments = []
    for doc_id, art_id, art_text in raw_triples:
        seg_id = f"{source_name}_Doc{doc_id}_Art{art_id}"
        segments.append((seg_id, art_text))
    # Применяем ограничения по словам
    # Валидация
    return validate_segments(segments, source_name)
def analyze_and_save_1295(text: str, output_file: str):
    """Analyze segmentation and save results for Pragmática 1295."""

    print("=" * 80)
    print("PRAGMÁTICA DE JAIME II 1295 - ARTICLE SEGMENTATION")
    print("=" * 80)

    articles = segment_zhaime1295_text(text, debug=True)

    # Group by document
    docs = {}
    for doc_id, art_id, art_text in articles:
        if doc_id not in docs:
            docs[doc_id] = []
        docs[doc_id].append((art_id, art_text))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(docs)}")
    for doc_id in sorted(docs.keys(), key=lambda x: int(x)):
        print(f"  Document {doc_id}: {len(docs[doc_id])} articles")

    print(f"\nTotal articles: {len(articles)}")

    # Check for duplicates within each document
    from collections import Counter
    for doc_id, articles_in_doc in docs.items():
        article_ids = [aid for aid, _ in articles_in_doc]
        duplicates = [(n, c) for n, c in Counter(article_ids).items() if c > 1]
        if duplicates:
            print(f"\nWARNING: Duplicates in document {doc_id}:")
            for art_id, count in duplicates:
                print(f"  Article {art_id} appears {count} times")
        else:
            print(f"\nDocument {doc_id}: No duplicates ✓")

    print("=" * 80)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"PRAGMÁTICA DE JAIME II 1295 - SEGMENTED ARTICLES\n")
        f.write(f"Total documents: {len(docs)}\n")
        f.write(f"Total articles: {len(articles)}\n")
        f.write("=" * 80 + "\n\n")

        for doc_id in sorted(docs.keys(), key=lambda x: int(x)):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"DOCUMENT {doc_id}\n")
            f.write(f"Articles: {len(docs[doc_id])}\n")
            f.write(f"{'=' * 80}\n\n")

            # Sort articles by Roman numeral value
            sorted_articles = sorted(docs[doc_id], key=lambda x: roman_to_int(x[0]))

            for article_id, text in sorted_articles:
                f.write(f"{'=' * 80}\n")
                f.write(f"ARTICLE {doc_id}.{article_id}\n")
                f.write(f"{'=' * 80}\n")
                f.write(text)
                f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    print("=" * 80)
    return articles


def main():
    """Main entry point"""
    file_zhaime1295 = Path("data/PragmatikaZhaumeII1295_v2.txt")

    if file_zhaime1295.exists():
        print(f"Processing Pragmática de Jaime II 1295 from {file_zhaime1295}...\n")
        text = file_zhaime1295.read_text(encoding='utf-8')
        docs = analyze_and_save_1295(text, output_file="pragmatica_zhaime1295_segmented.txt")
    else:
        print(f"Error: {file_zhaime1295} not found")


if __name__ == '__main__':
    main()
