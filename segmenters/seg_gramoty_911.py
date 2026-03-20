#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Gramoty IX–XI.

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


def segment_gramoty_911(text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Gramoty911 document into individual documents.

    Format: document number on a separate line followed by a date (8XX–11XX).
    Also handles "bis" documents such as "15 bis".

    Returns
    -------
    list[tuple[str, str]]
        List of (doc_number, extracted_text) tuples, sorted by document number.
    """
    lines = text.split("\n")
    doc_boundaries = []

    doc_num_pattern = re.compile(r"^\s*(\d+(?:\s+bis)?)\s*$", re.IGNORECASE)
    date_pattern = re.compile(r"\b(8\d{2}|9\d{2}|10\d{2}|110[0])\b")

    for i, line in enumerate(lines):
        match = doc_num_pattern.match(line)
        if not match:
            continue

        doc_num = match.group(1)
        has_date = False
        date_line = None

        for offset in range(1, min(7, len(lines) - i)):
            next_line = lines[i + offset]
            if not next_line.strip():
                continue

            if date_pattern.search(next_line):
                has_date = True
                date_line = next_line.strip()
                break

            if offset > 4:
                break

        if has_date:
            doc_boundaries.append({
                "num": doc_num,
                "line": i,
                "date_line": date_line,
            })

            if debug and len(doc_boundaries) <= 20:
                print(f"Doc {doc_num:>7s} at line {i:5d}, date: {date_line[:60]}")

    doc_boundaries.sort(key=lambda x: x["line"])

    if debug:
        print(f"\nTotal document boundaries found: {len(doc_boundaries)}")

    documents: List[Tuple[str, str]] = []
    for idx, boundary in enumerate(doc_boundaries):
        doc_num = boundary["num"]
        start_line = boundary["line"]
        end_line = doc_boundaries[idx + 1]["line"] if idx + 1 < len(doc_boundaries) else len(lines)

        doc_lines = lines[start_line:end_line]
        meaningful_text = _extract_text_911(doc_lines)
        if meaningful_text.strip():
            documents.append((doc_num, meaningful_text))

    def sort_key(doc_tuple: Tuple[str, str]) -> Tuple[int, int]:
        num_str = doc_tuple[0]
        if "bis" in num_str.lower():
            base_num = int(num_str.split()[0])
            return (base_num, 1)
        return (int(num_str), 0)

    documents.sort(key=sort_key)
    return documents



def _extract_text_911(doc_lines: List[str]) -> str:
    """Extract the meaningful text of a single document block."""
    start_idx = 0
    for i, line in enumerate(doc_lines):
        stripped = line.strip()
        if re.match(r"^\s*\d+(?:\s+bis)?\s*$", stripped, re.IGNORECASE):
            start_idx = i + 1
            continue
        if stripped:
            start_idx = i
            break

    end_patterns = [
        re.compile(r"^\[?[A-Z]\]?\s+(ACA|BC|ADG|ACU|BNF|ACG|AHAT)", re.IGNORECASE),
        re.compile(r"^Original no localitzat", re.IGNORECASE),
        re.compile(r"^Ed\.\s+"),
        re.compile(r"^\*"),
    ]

    result_lines: List[str] = []
    empty_count = 0

    for line in doc_lines[start_idx:]:
        stripped = line.strip()

        if not stripped:
            empty_count += 1
            if empty_count >= 4:
                break
            result_lines.append(line)
            continue

        empty_count = 0
        if any(pattern.match(stripped) for pattern in end_patterns):
            break
        result_lines.append(line)

    return "\n".join(result_lines).strip()



def segment_gramoty_911_unified(source_file, source_name):
    """
    Unified segmenter for Gramoty IX–XI.

    Parameters
    ----------
    source_file : str | Path
        Path to the source file.
    source_name : str
        Canonical source name, e.g. "Gramoty911".

    Returns
    -------
    list[tuple[str, str]]
        List of (segment_id, segment_text) pairs.
    """
    text = read_source_file(source_file)
    raw_pairs = segment_gramoty_911(text, debug=False)

    segments = []
    for doc_num, doc_text in raw_pairs:
        seg_id = f"{source_name}_Doc{doc_num}"
        segments.append((seg_id, doc_text))

    return validate_segments(segments, source_name)



def main() -> None:
    candidates = [
        Path("data/Gramoty911.txt"),
        Path("Gramoty911.txt"),
        Path("/mnt/data/Gramoty911.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_gramoty_911_unified(src, "Gramoty911")
    print(f"Gramoty911: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()
