#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Recognoverunt Proceres (Barcelona Privilege, 1283)
Extracts individual chapters from bilingual (Latin/Catalan) medieval legal document
"""

import re
from pathlib import Path
from typing import List, Tuple


def segment_privileges(text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Recognoverunt Proceres into individual chapters.
    
    Chapters marked with [I], [II], [III]... or Cap. I., Cap. II., etc.
    Also handles Alia vero capitula section with numbered chapters.
    
    Args:
        text: Full document text
        debug: If True, print debugging information
    
    Returns:
        List of (chapter_id, chapter_text) tuples
    """
    lines = text.split('\n')
    
    # Pattern for chapters: [ROMAN] or Cap. ROMAN or Cap. NUMBER
    # Matches: [I], [II], [LXIX], Cap. I, Cap. II, etc.
    chapter_pattern = re.compile(r'^\s*(?:\[(([IVXLCDM]{1,6}))\]|Cap\.\s*([IVXLCDM]{1,6}|\d{1,3}))', re.MULTILINE)
    
    boundaries = []
    
    for i, line in enumerate(lines):
        match = chapter_pattern.search(line)
        if match:
            # Extract chapter ID from either bracket format [I] or Cap. I
            roman = match.group(2) if match.group(2) else match.group(3)
            if roman:
                boundaries.append({
                    'id': roman,
                    'line': i,
                    'full_line': line.strip()
                })
    
    if debug:
        print(f"Found {len(boundaries)} chapter boundaries")
        for b in boundaries[:10]:
            print(f"  Cap_{b['id']} at line {b['line']}: {b['full_line'][:70]}...")
    
    # Extract chapter texts
    chapters = []
    
    for idx, boundary in enumerate(boundaries):
        chapter_id = f"Cap_{boundary['id']}"
        start_line = boundary['line']
        
        # End is start of next chapter or end of document
        if idx + 1 < len(boundaries):
            end_line = boundaries[idx + 1]['line']
        else:
            end_line = len(lines)
        
        # Extract lines for this chapter
        chapter_lines = lines[start_line:end_line]
        
        # Remove the header line (the [I] or Cap. I line itself)
        if chapter_lines and chapter_pattern.match(chapter_lines[0].strip()):
            chapter_lines = chapter_lines[1:]
        
        # Clean and join
        chapter_text = extract_chapter_text(chapter_lines)
        
        if chapter_text.strip():
            chapters.append((chapter_id, chapter_text))
    
    if debug:
        print(f"\nExtracted {len(chapters)} chapters with text")
    
    return chapters


def extract_chapter_text(lines: List[str]) -> str:
    """
    Extract clean chapter text from lines.
    Removes leading chapter markers and artifacts.
    """
    # Pattern to detect pure chapter markers
    chapter_marker = re.compile(r'^\s*(?:\[(([IVXLCDM]{1,6}))\]|Cap\.\s*([IVXLCDM]{1,6}|\d{1,3}))')
    
    result = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Skip pure chapter markers (e.g., "[I]" or "Cap. I.")
        if chapter_marker.match(stripped):
            # Check if line is ONLY the marker (no additional text)
            # by seeing if removing the marker leaves nothing
            test = re.sub(r'^\s*\[([IVXLCDM]{1,6})\]\s*', '', stripped)
            test = re.sub(r'^\s*Cap\.\s*([IVXLCDM]{1,6}|\d{1,3})\s*', '', test)
            if not test:  # Line was only a marker
                continue
        
        # Skip apparatus markers
        if re.match(r'^\[?[A-Z]?\]?\s+(PHN|HNV|PNV)', stripped):
            continue
        
        result.append(stripped)
    
    # Join with spaces and clean
    text = ' '.join(result)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def analyze_and_save(text: str, output_file: str):
    """
    Analyze segmentation and save results.
    """
    print("=" * 80)
    print("RECOGNOVERUNT PROCERES (1283) - CHAPTER SEGMENTATION")
    print("=" * 80)
    
    chapters = segment_privileges(text, debug=False)
    
    # Statistics
    chapter_ids = [cid for cid, _ in chapters]
    
    print(f"\nExpected: ~106 chapters")
    print(f"Found: {len(chapters)} chapters")
    
    # Check for duplicates
    from collections import Counter
    duplicates = [(n, c) for n, c in Counter(chapter_ids).items() if c > 1]
    
    if duplicates:
        print(f"\nDuplicates: {len(duplicates)}")
        for num, count in sorted(duplicates)[:5]:
            print(f"  Cap_{num}: {count} times")
    else:
        print(f"\n✓ No duplicates")
    
    # Show sample chapters
    print(f"\nSample chapters:")
    for cid, text in chapters[:5]:
        print(f"  {cid}: {len(text)} chars - {text[:60]}...")
    
    print("=" * 80)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total chapters: {len(chapters)}\n")
        f.write("=" * 80 + "\n\n")
        
        for chapter_id, text in chapters:
            f.write("=" * 80 + "\n")
            f.write(f"CHAPTER {chapter_id}\n")
            f.write("=" * 80 + "\n")
            f.write(text)
            f.write("\n\n")
    
    print(f"✓ Results saved to {output_file}")
    
    return chapters


def main():
    """Main entry point"""
    file_path = Path('data/RecognovrentProceres12831284_v2.txt')
    
    if file_path.exists():
        print(f"Processing {file_path}...")
        text = file_path.read_text(encoding='utf-8', errors='replace')
        docs = analyze_and_save(text, output_file='privileges_segmented.txt')
    else:
        print(f"Error: {file_path} not found")


def segment_privileges_unified(
    source_file, source_name, min_words=10, max_words=150
):
    """
    Унифицированная сегментация Privileges.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с debug=False
    raw_segments = segment_privileges(text, debug=False)

    # Применяем ограничения по словам
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == '__main__':
    main()
