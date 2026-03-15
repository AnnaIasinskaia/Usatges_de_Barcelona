"""Segmenter for Costums de Tortosa (1272–1279) - Text format version.

Structure: RUBRICA N / Rubrica de ... → paragraphs
Falls back to segment_default if structural markers are sparse.
"""
import re
from seg_common import clean_text, is_apparatus_line, group_segments, validate_segments
from seg_default import segment_default

# Rubrica patterns - various formats:
# 1. "RUBRICA N" where N is a number
# 2. "Rubrica de ..." - descriptive title
# 3. "RUBRICA DE ..." - descriptive title in uppercase
_RUBRICA_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(?:'
    r'(?:RUBRICA\s+\d+)'  # RUBRICA 2
    r'|(?:Rubrica\s+de\s+[^\n]{10,150})'  # Rubrica de les pastures...
    r'|(?:RUBRICA\s+DE\s+[^\n]{10,150})'  # RUBRICA DE L'OFFICI...
    r')',
    re.MULTILINE | re.IGNORECASE
)

# Apparatus markers specific to Costums de Tortosa
_CT_APPARATUS = re.compile(
    r'(?:'
    r'\bMs\b|\bmss\b|\bCod\b|\bmanuscrit\b'
    r'|v\.\s*l\.'
    r'|\bop\.\s*cit\b'
    r'|\bLoc\.\s*cit\b'
    r'|\bp\.\s*\d+'
    r'|\bque\s+incipit\b'  # sentència markers
    r'|\bapprobant\b|\brepprobant\b'  # arbitral sentence
    r')',
    re.IGNORECASE
)

def segment_costums_tortosa(text, source_name="CostumsTortosa", max_segment_words=200):
    """
    Segment Costums de Tortosa text by Rubrica divisions.

    Args:
        text: Full text content
        source_name: Source identifier for segment labels
        max_segment_words: Maximum words per segment for grouping

    Returns:
        List of (segment_id, segment_text) tuples
    """
    # Try Rubrica-level segmentation first
    matches = list(_RUBRICA_RE.finditer(text))

    if len(matches) >= 5:
        segments = _split_by_rubrica(matches, text, source_name)
        if segments:
            return validate_segments(
                group_segments(segments, source_name, max_segment_words),
                source_name
            )

    # Fallback to default segmentation if structure is sparse
    return segment_default(text, source_name, max_segment_words)

def _split_by_rubrica(matches, text, source_name):
    """
    Split text by Rubrica matches.

    Args:
        matches: List of regex match objects
        text: Full text
        source_name: Source identifier

    Returns:
        List of (label, cleaned_text) tuples
    """
    segments = []

    for i, m in enumerate(matches):
        # Extract rubrica title/number
        rubrica_header = m.group(0).strip()

        # Content starts after the header
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_content = text[start:end]

        # Clean the content
        lines = raw_content.split('\n')
        clean_lines = []

        for line in lines:
            s = line.strip()

            # Skip empty or very short lines
            if not s or len(s) < 5:
                continue

            # Skip apparatus/editorial notes
            if is_apparatus_line(s):
                continue

            # Skip lines with apparatus markers and low alpha ratio
            if _CT_APPARATUS.search(s) and sum(c.isalpha() for c in s) < len(s) * 0.4:
                continue

            # Skip standalone numbers (page numbers, etc.)
            if re.match(r'^\s*\d{1,3}\s*$', s):
                continue

            clean_lines.append(s)

        # Join cleaned lines
        cleaned = clean_text(' '.join(clean_lines))

        # Generate segment label from rubrica header
        label = _generate_label(source_name, rubrica_header, i + 1)

        # Only add if sufficient content
        if len(cleaned) >= 30:
            segments.append((label, cleaned))

    return segments

def _generate_label(source_name, rubrica_header, index):
    """
    Generate a clean segment label from rubrica header.

    Args:
        source_name: Source identifier
        rubrica_header: Header text from match
        index: Rubrica index number

    Returns:
        Formatted label string
    """
    # Extract number if present
    num_match = re.search(r'\d+', rubrica_header)
    if num_match:
        num = num_match.group(0)
        return f"{source_name}_Rub{num}"

    # Otherwise use index
    return f"{source_name}_Rub{index}"

# Test function for standalone execution
if __name__ == "__main__":
    from pathlib import Path

    # Try to load text file
    txt_path = Path("data/ObychaiTortosy1272to1279.txt")

    if txt_path.exists():
        print(f"Loading {txt_path}...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Text length: {len(text)} characters")

        # Segment the text
        segments = segment_costums_tortosa(text, "CostumsTortosa")

        print(f"\nCostumsTortosa: {len(segments)} segments\n")

        # Display first 5 segments
        for seg_id, seg_text in segments[:5]:
            print(f"{seg_id}:")
            print(f"  {seg_text[:120]}...")
            print(f"  Length: {len(seg_text)} chars\n")
    else:
        print(f"File not found: {txt_path}")
        print("Expected file: data/ObychaiTortosy1272to1279.txt in current directory")
