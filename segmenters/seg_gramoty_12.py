#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Gramoty XII.

Рабочий контракт:
- основной парсер возвращает list[(doc_number, text)]
- unified entrypoint возвращает list[(segment_id, segment_text)]
- тестирование выполняется через test_unified_segmenters.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import read_source_file, validate_segments


def segment_gramoty_12(text: str, debug: bool = False) -> List[Tuple[int, str]]:
    """
    Segment Gramoty12 document into individual documents.

    Supports two formats:
    1. Standard: number on separate line followed by date
    2. Alternative: ### DOC number ### format

    Returns
    -------
    list[tuple[int, str]]
        List of (doc_number, extracted_text) tuples.
    """
    lines = text.split("\n")
    doc_boundaries = []

    doc_num_pattern1 = re.compile(r"^\s*(\d+)\s*$")
    doc_num_pattern2 = re.compile(r"^\s*###\s*DOC\s+(\d+)\s*###\s*$", re.IGNORECASE)

    date_patterns = [
        re.compile(r"\b1[012]\d{2}\b"),                   # 1000-1299
        re.compile(r"\[.*?1[012]\d{2}.*?\]"),            # [1094-1103], [c. 1105]
        re.compile(r"Segle\s+xi{1,3}", re.IGNORECASE),   # Segle xi / xii / xiii
    ]

    for i, line in enumerate(lines):
        doc_num = None
        format_type = None

        match1 = doc_num_pattern1.match(line)
        if match1:
            doc_num = int(match1.group(1))
            format_type = "standard"

        match2 = doc_num_pattern2.match(line)
        if match2:
            doc_num = int(match2.group(1))
            format_type = "marked"

        if doc_num is None:
            continue

        has_date = False
        date_line = None

        if format_type == "marked":
            if i > 0:
                prev_line = lines[i - 1]
                for date_pattern in date_patterns:
                    if date_pattern.search(prev_line):
                        has_date = True
                        date_line = prev_line.strip()
                        break

            if not has_date:
                for offset in range(1, 4):
                    if i + offset >= len(lines):
                        break
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
            for offset in range(1, min(7, len(lines) - i)):
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

                if offset > 4:
                    break

        # Для ### DOC ### формат сам по себе считаем надёжным маркером.
        if format_type == "marked" or has_date:
            doc_boundaries.append({
                "num": doc_num,
                "line": i,
                "format": format_type,
                "date_line": date_line,
            })

            if debug and len(doc_boundaries) <= 20:
                date_str = date_line[:60] if date_line else "NO DATE"
                print(
                    f"Found doc {doc_num:4d} at line {i:5d} "
                    f"[{format_type:8s}] date: {date_str}"
                )

    if debug:
        print(f"\nTotal document boundaries found: {len(doc_boundaries)}")
        standard_count = sum(1 for b in doc_boundaries if b["format"] == "standard")
        marked_count = sum(1 for b in doc_boundaries if b["format"] == "marked")
        print(f"  Standard format: {standard_count}")
        print(f"  Marked format (### DOC ###): {marked_count}")

    documents: List[Tuple[int, str]] = []

    for idx, boundary in enumerate(doc_boundaries):
        doc_num = boundary["num"]
        start_line = boundary["line"]
        end_line = doc_boundaries[idx + 1]["line"] if idx + 1 < len(doc_boundaries) else len(lines)

        doc_lines = lines[start_line:end_line]

        if boundary["format"] == "marked":
            latin_text = _extract_latin_12(doc_lines[1:])  # пропускаем ### DOC ###
        else:
            latin_text = _extract_latin_12(doc_lines[1:])  # пропускаем строку с номером

        if latin_text.strip():
            documents.append((doc_num, latin_text))
        elif debug:
            print(f"WARNING: Doc {doc_num}: empty after extraction")

    return documents


def _extract_latin_12(lines: List[str]) -> str:
    """
    Extract Latin text from document lines.
    Takes everything until bibliography/archive references.
    """
    latin_lines: List[str] = []

    end_patterns = [
        re.compile(r"^\[?[A-Z]\]?\s+(ACA|BC|ADG|ACU|AHAT|ACC|ADS|AMM|ADC|BRAH)", re.IGNORECASE),
        re.compile(r"^Ed\.\s+"),
        re.compile(r"^\*"),
    ]

    empty_count = 0
    for line in lines:
        stripped = line.strip()

        if not stripped:
            empty_count += 1
            if empty_count >= 4:
                break
            latin_lines.append(line)
            continue

        empty_count = 0

        if any(pattern.match(stripped) for pattern in end_patterns):
            break

        latin_lines.append(line)

    return "\n".join(latin_lines).strip()


def segment_gramoty_12_unified(source_file, source_name):
    """
    Unified segmenter for Gramoty XII.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "Gramoty12".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_pairs = segment_gramoty_12(text, debug=False)

    segments = []
    for doc_num, doc_text in raw_pairs:
        seg_id = f"{source_name}_{doc_num}"
        segments.append((seg_id, doc_text))

    return validate_segments(segments, source_name)


def main() -> None:
    candidates = [
        Path("data/Gramoty12.txt"),
        Path("Gramoty12.txt"),
        Path("/mnt/data/Gramoty12.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_gramoty_12_unified(src, "Gramoty12")
    print(f"Gramoty12: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()