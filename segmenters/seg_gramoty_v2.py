#!/usr/bin/env python3
"""
Segmentation script for Gramoty documents (12th century)
Improved version with support for multiple document number formats
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import Counter


def segment_gramoty_12(text: str, debug: bool = False) -> List[Tuple[int, str]]:
    """
    Segment Gramoty12 document into individual documents.

    Supports two formats:
    1. Standard: number on separate line followed by date
    2. Alternative: ### DOC number ### format

    Returns:
        List of (doc_number, latin_text) tuples
    """
    lines = text.split('\n')
    doc_boundaries = []

    # Pattern 1: Document number (just digits on a line)
    doc_num_pattern1 = re.compile(r'^\s*(\d+)\s*$')

    # Pattern 2: ### DOC number ###
    doc_num_pattern2 = re.compile(r'^\s*###\s*DOC\s+(\d+)\s*###\s*$', re.IGNORECASE)

    # Patterns for dates - more flexible
    date_patterns = [
        re.compile(r'\b1[012]\d{2}\b'),  # Year 1000-1299
        re.compile(r'\[.*?1[012]\d{2}.*?\]'),  # [1094-1103] or [c. 1105]
        re.compile(r'Segle\s+xi{1,3}', re.IGNORECASE),  # Segle xii, Segle xi
    ]

    for i, line in enumerate(lines):
        doc_num = None
        format_type = None

        # Try pattern 1: number on separate line
        match1 = doc_num_pattern1.match(line)
        if match1:
            doc_num = int(match1.group(1))
            format_type = 'standard'

        # Try pattern 2: ### DOC number ###
        match2 = doc_num_pattern2.match(line)
        if match2:
            doc_num = int(match2.group(1))
            format_type = 'marked'

        if doc_num is not None:
            # Check if this looks like a document boundary
            # For marked format, date is usually in previous or next line
            # For standard format, date is in next 1-5 lines

            has_date = False
            date_line = None

            # Search range depends on format
            if format_type == 'marked':
                # Check previous line (often date before ### DOC ###)
                if i > 0:
                    prev_line = lines[i - 1]
                    for date_pattern in date_patterns:
                        if date_pattern.search(prev_line):
                            has_date = True
                            date_line = prev_line.strip()
                            break

                # If not found, check next few lines
                if not has_date:
                    for offset in range(1, 4):
                        if i + offset < len(lines):
                            next_line = lines[i + offset]
                            if not next_line.strip():
                                continue
                            for date_pattern in date_patterns:
                                if date_pattern.search(next_line):
                                    has_date = True
                                    date_line = next_line.strip()
                                    break
                            if has_date:
                                break
            else:
                # Standard format: check next 1-6 lines
                for offset in range(1, min(7, len(lines) - i)):
                    next_line = lines[i + offset]

                    # Skip empty lines
                    if not next_line.strip():
                        continue

                    # Check for date patterns
                    for date_pattern in date_patterns:
                        if date_pattern.search(next_line):
                            has_date = True
                            date_line = next_line.strip()
                            break

                    if has_date:
                        break

                    # Stop after too many lines
                    if offset > 4:
                        break

            # For ### DOC ### format, accept even without explicit date
            # (it's a more reliable marker)
            if format_type == 'marked' or has_date:
                doc_boundaries.append({
                    'num': doc_num,
                    'line': i,
                    'format': format_type,
                    'date_line': date_line
                })

                if debug and len(doc_boundaries) <= 20:
                    date_str = date_line[:60] if date_line else "NO DATE"
                    print(f"Found doc {doc_num:4d} at line {i:5d} [{format_type:8s}] date: {date_str}")

    if debug:
        print(f"\nTotal document boundaries found: {len(doc_boundaries)}")
        standard_count = sum(1 for b in doc_boundaries if b['format'] == 'standard')
        marked_count = sum(1 for b in doc_boundaries if b['format'] == 'marked')
        print(f"  Standard format: {standard_count}")
        print(f"  Marked format (### DOC ###): {marked_count}")

    # Extract document texts
    documents = []

    for idx, boundary in enumerate(doc_boundaries):
        doc_num = boundary['num']
        start_line = boundary['line']

        # Find end: next document or end of file
        if idx + 1 < len(doc_boundaries):
            end_line = doc_boundaries[idx + 1]['line']
        else:
            end_line = len(lines)

        # Extract text from start to end
        doc_lines = lines[start_line:end_line]
        doc_text = '\n'.join(doc_lines)

        # Extract Latin portion
        latin_text = _extract_latin_12(doc_lines[1:])  # Skip the number line

        if latin_text.strip():
            documents.append((doc_num, latin_text))
        elif debug:
            print(f"⚠️  Doc {doc_num}: empty after extraction")

    return documents


def _extract_latin_12(lines: List[str]) -> str:
    """
    Extract Latin text from document lines.
    Takes everything until bibliography/archive references.
    """
    latin_lines = []

    # Patterns that indicate end of Latin text
    end_patterns = [
        re.compile(r'^\[?[A-Z]\]?\s+(ACA|BC|ADG|ACU|AHAT|ACC|ADS|AMM|ADC|BRAH)', re.IGNORECASE),
        re.compile(r'^Ed\.\s+'),  # Editorial references
        re.compile(r'^\*'),  # Bibliography markers
    ]

    empty_count = 0
    for line in lines:
        stripped = line.strip()

        # Count consecutive empty lines
        if not stripped:
            empty_count += 1
            if empty_count >= 4:  # 4+ empty lines = likely end
                break
            latin_lines.append(line)
            continue
        else:
            empty_count = 0

        # Check for end patterns
        is_end = False
        for pattern in end_patterns:
            if pattern.match(stripped):
                is_end = True
                break

        if is_end:
            break

        latin_lines.append(line)

    return '\n'.join(latin_lines).strip()


def parse_expected_docs_from_file(text: str) -> Set[int]:
    """
    Find all document numbers that appear in the file (including index).
    This gives us the "expected" set of documents.
    """
    # Find all numbers that look like document references
    # Format: "123," or "123 " or "###DOC 123###" or standalone "123"
    pattern = re.compile(r'(?:###\s*DOC\s+(\d+)\s*###|\b(\d+)[,\s])')

    all_nums = set()
    for match in pattern.finditer(text):
        if match.group(1):
            all_nums.add(int(match.group(1)))
        elif match.group(2):
            num = int(match.group(2))
            # Only include reasonable document numbers (1-2000)
            if 1 <= num <= 2000:
                all_nums.add(num)

    return all_nums


def segment_gramoty_12_unified(
    source_file,
    source_name,
    min_words=10,
    max_words=150
):
    """
    Унифицированная функция сегментации для Gramoty XII.
    Соответствует контракту из INTERFACE.md.

    Параметры
    ---------
    source_file : str или Path
        Путь к файлу с текстом (формат .txt или .docx).
    source_name : str
        Имя источника (например, "Gramoty12").
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
    raw_pairs = segment_gramoty_12(text, debug=False)
    # Преобразуем пары (doc_num, text) в пары (id, text)
    segments = []
    for doc_num, doc_text in raw_pairs:
        seg_id = f"{source_name}_Doc{doc_num}"
        segments.append((seg_id, doc_text))
    # Применяем ограничения по словам
    filtered = apply_word_limits(segments, min_words, max_words)
    # Валидация
    return validate_segments(filtered, source_name)
def analyze_documents_12(text: str, expected_count: int = 873):
    """
    Analyze Gramoty12 segmentation and provide debugging info.
    """
    print("="*80)
    print("GRAMOTY XII CENTURY - DOCUMENT ANALYSIS")
    print("="*80)

    # Segment documents
    documents = segment_gramoty_12(text, debug=True)

    print(f"\nExpected documents: {expected_count}")
    print(f"Found documents: {len(documents)}")
    print(f"Coverage: {len(documents)/expected_count*100:.1f}%")

    # Check for duplicates
    doc_nums = [num for num, _ in documents]
    num_counts = Counter(doc_nums)
    duplicates = {num: count for num, count in num_counts.items() if count > 1}

    if duplicates:
        print(f"\n⚠️  DUPLICATES FOUND: {len(duplicates)}")
        for num, count in sorted(duplicates.items())[:10]:
            print(f"   Doc {num}: appears {count} times")
    else:
        print(f"\n✓ No duplicates found")

    # Check number range
    if doc_nums:
        sorted_unique = sorted(set(doc_nums))
        print(f"\nNumber range: {min(doc_nums)} to {max(doc_nums)}")
        print(f"Unique documents: {len(sorted_unique)}")

        # Check for gaps in first 200
        if len(sorted_unique) > 1:
            print(f"\nChecking for gaps in first 200 numbers:")
            gaps = []
            for i in range(len(sorted_unique) - 1):
                if sorted_unique[i] > 200:
                    break
                if sorted_unique[i+1] - sorted_unique[i] > 1:
                    gaps.append((sorted_unique[i], sorted_unique[i+1]))

            if gaps:
                print(f"  Found {len(gaps)} gaps:")
                for start, end in gaps[:10]:
                    missing = list(range(start + 1, end))
                    print(f"    After {start}: missing {missing}")
            else:
                print(f"  ✓ No gaps found in first 200")

    # Show sample documents
    print(f"\nFirst 5 documents:")
    for num, text in documents[:5]:
        preview = text[:80].replace('\n', ' ')
        print(f"   Doc {num:3d}: {preview}...")

    if len(documents) > 5:
        print(f"\nLast 5 documents:")
        for num, text in documents[-5:]:
            preview = text[:80].replace('\n', ' ')
            print(f"   Doc {num:3d}: {preview}...")

    return documents


def main():
    """Main function for testing"""

    # Test Gramoty12
    file_12 = Path('data/Gramoty12.txt')
    if file_12.exists():
        print("Reading Gramoty12.txt...")
        text_12 = file_12.read_text(encoding='utf-8')
        docs_12 = analyze_documents_12(text_12, expected_count=873)

        # Save results
        output_file = Path('gramoty12_segmented.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Total documents: {len(docs_12)}\n")
            f.write("="*80 + "\n\n")
            for num, text in docs_12:
                f.write(f"\n{'='*80}\n")
                f.write(f"DOCUMENT {num}\n")
                f.write(f"{'='*80}\n")
                f.write(text)
                f.write("\n\n")
        print(f"\n✓ Results saved to {output_file}")
    else:
        print(f"Error: {file_12} not found")


if __name__ == '__main__':
    main()
