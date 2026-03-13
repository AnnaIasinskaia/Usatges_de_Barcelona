#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for Gramoty IX-XI (9th-11th centuries)
==========================================================

Documents from 9th-11th centuries (years 800-1100)
Expected: 557 documents (552 regular + 5 "bis")
"""

import re
from pathlib import Path
from typing import List, Tuple
from collections import Counter


def segment_gramoty_911(text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Gramoty911 document into individual documents.

    Format: Document number on separate line followed by date (8XX-11XX)
    Example:
        1
        812, abril, 2. Aquisgrà
        [text...]

    Also handles "bis" documents: "15 bis"

    Args:
        text: Full document text
        debug: If True, print debugging information

    Returns:
        List of (doc_number, extracted_text) tuples, sorted by document number
    """
    lines = text.split('\n')
    doc_boundaries = []

    # Pattern for document number (digits or "digits bis")
    doc_num_pattern = re.compile(r'^\s*(\d+(?:\s+bis)?)\s*$', re.IGNORECASE)

    # Date pattern: years 800-1100
    date_pattern = re.compile(r'\b(8\d{2}|9\d{2}|10\d{2}|110[0])\b')

    for i, line in enumerate(lines):
        match = doc_num_pattern.match(line)
        if match:
            doc_num = match.group(1)

            # Check next 1-6 lines for date
            has_date = False
            date_line = None

            for offset in range(1, min(7, len(lines) - i)):
                next_line = lines[i + offset]

                # Skip empty lines
                if not next_line.strip():
                    continue

                # Check for year 8XX-11XX
                if date_pattern.search(next_line):
                    has_date = True
                    date_line = next_line.strip()
                    break

                # Stop if we hit bibliography or next document
                if offset > 4:
                    break

            if has_date:
                doc_boundaries.append({
                    'num': doc_num,
                    'line': i,
                    'date_line': date_line
                })

                if debug and len(doc_boundaries) <= 20:
                    print(f"Doc {doc_num:>7s} at line {i:5d}, date: {date_line[:60]}")

    # Sort by line number
    doc_boundaries.sort(key=lambda x: x['line'])

    if debug:
        print(f"\nTotal document boundaries found: {len(doc_boundaries)}")

    # Extract document texts
    documents = []

    for idx, boundary in enumerate(doc_boundaries):
        doc_num = boundary['num']
        start_line = boundary['line']
        end_line = doc_boundaries[idx + 1]['line'] if idx + 1 < len(doc_boundaries) else len(lines)

        doc_lines = lines[start_line:end_line]
        meaningful_text = _extract_text_911(doc_lines)

        if meaningful_text.strip():
            documents.append((doc_num, meaningful_text))

    # Sort by document number (handle "bis" properly)
    def sort_key(doc_tuple):
        num_str = doc_tuple[0]
        if 'bis' in num_str.lower():
            base_num = int(num_str.split()[0])
            return (base_num, 1)  # bis comes after regular
        else:
            return (int(num_str), 0)

    documents.sort(key=sort_key)
    return documents


def _extract_text_911(doc_lines: List[str]) -> str:
    """
    Extract meaningful text from document lines.
    Includes: date, regests, Latin text
    Excludes: bibliography, archive references
    """
    # Skip document number line
    start_idx = 0
    for i, line in enumerate(doc_lines):
        stripped = line.strip()
        if re.match(r'^\s*\d+(?:\s+bis)?\s*$', stripped, re.IGNORECASE):
            start_idx = i + 1
            continue
        if stripped:
            start_idx = i
            break

    # Extract until bibliography
    end_patterns = [
        re.compile(r'^\[?[A-Z]\]?\s+(ACA|BC|ADG|ACU|BNF|ACG|AHAT)', re.IGNORECASE),
        re.compile(r'^Original no localitzat', re.IGNORECASE),
        re.compile(r'^Ed\.\s+'),
        re.compile(r'^\*'),
    ]

    result_lines = []
    empty_count = 0

    for line in doc_lines[start_idx:]:
        stripped = line.strip()

        if not stripped:
            empty_count += 1
            if empty_count >= 4:
                break
            result_lines.append(line)
            continue
        else:
            empty_count = 0

        # Check for bibliography
        is_end = any(pattern.match(stripped) for pattern in end_patterns)
        if is_end:
            break

        result_lines.append(line)

    return '\n'.join(result_lines).strip()


def analyze_and_save(text: str, expected_count: int, output_file: str):
    """
    Analyze segmentation and save results.
    """
    print("="*80)
    print("GRAMOTY IX-XI CENTURY - DOCUMENT SEGMENTATION")
    print("="*80)

    documents = segment_gramoty_911(text, debug=False)

    # Statistics
    doc_nums = [num for num, _ in documents]
    coverage = len(documents) / expected_count * 100
    duplicates = {n: c for n, c in Counter(doc_nums).items() if c > 1}

    # Count bis documents
    bis_docs = [n for n in doc_nums if 'bis' in n.lower()]
    regular_docs = [n for n in doc_nums if 'bis' not in n.lower()]

    print(f"\nExpected: {expected_count} documents (552 regular + 5 bis)")
    print(f"Found: {len(documents)} documents")
    print(f"  Regular: {len(regular_docs)}")
    print(f"  Bis: {len(bis_docs)}")
    print(f"Coverage: {coverage:.1f}%")

    if duplicates:
        print(f"\n⚠️  Duplicates: {len(duplicates)}")
        for num, count in sorted(duplicates.items())[:5]:
            print(f"   Doc {num}: {count} times")
    else:
        print("\n✓ No duplicates")

    # Check for missing
    expected_nums = set(str(i) for i in range(1, 553))
    expected_bis = {'15 bis', '119 bis', '215 bis', '314 bis', '401 bis'}  # Known bis docs
    expected_all = expected_nums | expected_bis

    found_set = set(doc_nums)
    missing = sorted(expected_all - found_set, key=lambda x: (int(x.split()[0]), 'bis' in x))

    if missing:
        print(f"\n⚠️  Missing: {len(missing)} documents")
        if len(missing) <= 15:
            print(f"   {missing}")
        else:
            print(f"   First 10: {missing[:10]}")
            print(f"   Last 5: {missing[-5:]}")
    else:
        print("\n✓ No missing documents")

    # Verdict
    print("\n" + "="*80)
    if coverage >= 90 and not duplicates:
        print(f"✓ SUCCESS: {coverage:.1f}% coverage, no duplicates")
    else:
        print(f"⚠️  {coverage:.1f}% coverage, {len(duplicates)} duplicates")
    print("="*80)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total documents: {len(documents)}\n")
        f.write(f"Regular: {len(regular_docs)}, Bis: {len(bis_docs)}\n")
        f.write("="*80 + "\n\n")
        for num, text in documents:
            f.write(f"\n{'='*80}\n")
            f.write(f"DOCUMENT {num}\n")
            f.write(f"{'='*80}\n")
            f.write(text)
            f.write("\n\n")

    print(f"\n✓ Results saved to {output_file}")

    return documents


def main():
    """Main entry point"""

    file_911 = Path('data/Gramoty911.txt')
    if file_911.exists():
        print("Processing Gramoty IX-XI centuries...")
        text = file_911.read_text(encoding='utf-8')
        docs = analyze_and_save(text, expected_count=557, output_file='gramoty911_final.txt')
    else:
        print(f"Error: {file_911} not found")


if __name__ == '__main__':
    main()
