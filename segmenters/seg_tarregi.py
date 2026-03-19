"""Specialized segmenter for Costums de Tarrega (1290).

Edition: Font Rius critical edition with Spanish commentary.

Structure issues:
  - Only 13 of 25 articles have explicit markers [N], [N), (N]
  - Articles 5-12 are unmarked between [4] and [13)
  - Spanish commentary interspersed without clear boundaries
  - No consistent pattern for unmarked articles

Strategy:
  1. Extract explicitly marked articles [N], [N), (N], {N}
  2. Between marked articles: extract Latin paragraphs as separate segments
  3. Filter Spanish commentary using language ratio
  4. Label unmarked segments as Tarregi_UnmarkedN

This is a CONSERVATIVE approach: prefers precision over recall.
Only extracts content we're confident is Latin legal text.
"""
import re

def is_apparatus_line(text):
    """Check if line is critical apparatus."""
    if len(text) < 3:
        return False
    patterns = [r'\bMs\b', r'\bmss\b', r'\bCod\b', r'\bfol\.', 
                r'\bp\.\s*\d+', r'v\.\s*l\.', r'\bop\.\s*cit\b',
                r'\bcf\.']
    count = sum(1 for p in patterns if re.search(p, text, re.I))
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return count >= 2 or (count >= 1 and alpha_ratio < 0.5)

def clean_text(text):
    """Normalize whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[­\u00ad]', '', text)  # soft hyphens
    return text.strip()

def detect_latin_ratio(text):
    """Estimate Latin vs Spanish content ratio (1.0 = pure Latin)."""
    # Latin endings and words
    latin_patterns = [
        r'\b\w+tur\b', r'\b\w+um\b', r'\b\w+is\b', r'\b\w+it\b',
        r'\b\w+nt\b', r'\b\w+tis\b', r'\bvel\b', r'\bet\b',
        r'\bnon\b', r'\bsi\b', r'\bper\b', r'\bin\b',
        r'\baliquo\b', r'\baliqua\b', r'\bqui\b', r'\bquod\b',
        r'\bfuerit\b', r'\bdebere\b', r'\bteneantur\b',
    ]
    # Spanish words
    spanish_patterns = [
        r'\bel\b', r'\bla\b', r'\blos\b', r'\blas\b', r'\bde\b',
        r'\bdel\b', r'\bcon\b', r'\bpor\b', r'\ben\b', r'\bque\b',
        r'\bpara\b', r'\bción\b', r'\bmiento\b', r'\bpero\b',
        r'\bmás\b', r'\bsobre\b', r'\besta\b', r'\beste\b',
        r'\baños\b', r'\bdonde\b', r'\bsolo\b', r'\bcomo\b',
    ]

    latin_count = sum(len(re.findall(p, text, re.I)) for p in latin_patterns)
    spanish_count = sum(len(re.findall(p, text, re.I)) for p in spanish_patterns)

    total = latin_count + spanish_count
    return latin_count / total if total > 0 else 0.5

# Article markers: [N] or [N) or (N] or {N} or [N}
# Must be at line start or after whitespace, max 2 digits
_ARTICLE_MARKER = re.compile(
    r'(?:^|\s)([\[\(]\d{1,2}[\]\)\}])',
    re.MULTILINE
)

def segment_tarregi(text, source_name="Tarregi", min_latin_ratio=0.70, min_words=15):
    """Segment Costums de Tarrega with conservative Latin-only extraction.

    Args:
        text: Full document text
        source_name: Source identifier
        min_latin_ratio: Minimum Latin ratio to accept segment (default 0.70 = strict)
        min_words: Minimum words per segment

    Returns:
        List of (segment_id, segment_text) tuples
    """
    lines = text.split('\n')

    # Find all article markers and their line numbers
    markers = []
    for i, line in enumerate(lines):
        match = _ARTICLE_MARKER.search(line)
        if match:
            # Extract article number
            marker_str = match.group(1).strip()
            num_match = re.search(r'(\d+)', marker_str)
            if num_match:
                art_num = int(num_match.group(1))
                markers.append((i, art_num, marker_str))

    if not markers:
        return []

    segments = []

    # Process each marked article
    for idx, (line_num, art_num, marker_str) in enumerate(markers):
        # Determine content region for this article
        start_line = line_num
        end_line = markers[idx + 1][0] if idx + 1 < len(markers) else len(lines)

        # === MARKED ARTICLE: Extract article text ===
        article_lines = []

        # Line with marker
        marker_line = lines[start_line].strip()
        # Remove marker itself from line
        article_text = re.sub(r'[\[\(]\d{1,2}[\]\)\}]', '', marker_line).strip()
        if article_text:
            article_lines.append(article_text)

        # Following lines until next marker (but only first paragraph/sentence)
        for i in range(start_line + 1, min(start_line + 5, end_line)):
            line = lines[i].strip()
            if not line or len(line) < 5:
                continue
            if is_apparatus_line(line):
                continue
            # Stop at Spanish commentary
            if detect_latin_ratio(line) < 0.5:
                break
            # Stop at numbered footnotes
            if re.match(r'^\d{1,2}\.\s', line):
                break
            article_lines.append(line)

        if article_lines:
            article_text = clean_text(' '.join(article_lines))
            # Language filter
            if detect_latin_ratio(article_text) >= min_latin_ratio:
                word_count = len(article_text.split())
                if word_count >= min_words:
                    seg_id = f"{source_name}_Art{art_num}"
                    segments.append((seg_id, article_text))

        # === UNMARKED CONTENT between this article and next ===
        # Look for Latin paragraphs in the gap
        gap_start = start_line + 6  # Skip first few lines (part of marked article)
        gap_end = end_line - 1

        if gap_end - gap_start > 3:  # Only if gap is significant
            unmarked_segments = _extract_unmarked_latin(
                lines, gap_start, gap_end, 
                f"{source_name}_Unmarked{idx+1}",
                min_latin_ratio, min_words
            )
            segments.extend(unmarked_segments)

    return segments

def _extract_unmarked_latin(lines, start, end, base_id, min_ratio, min_words):
    """Extract unmarked Latin paragraphs from a region.

    Strategy: Group consecutive Latin lines into paragraphs.
    """
    segments = []
    current_para = []
    current_words = 0
    para_num = 1

    for i in range(start, end):
        line = lines[i].strip()

        # Skip short, empty, apparatus
        if not line or len(line) < 20:
            # End current paragraph if any
            if current_para:
                para_text = clean_text(' '.join(current_para))
                if current_words >= min_words and detect_latin_ratio(para_text) >= min_ratio:
                    seg_id = f"{base_id}_{chr(96+para_num)}"  # _a, _b, _c
                    segments.append((seg_id, para_text))
                    para_num += 1
                current_para = []
                current_words = 0
            continue

        if is_apparatus_line(line):
            continue

        # Skip numbered footnotes
        if re.match(r'^\d{1,2}\.\s', line):
            continue

        # Check if line is Latin
        line_ratio = detect_latin_ratio(line)

        if line_ratio >= 0.6:  # Accept line
            current_para.append(line)
            current_words += len(line.split())
        else:  # Spanish commentary - end paragraph
            if current_para:
                para_text = clean_text(' '.join(current_para))
                if current_words >= min_words and detect_latin_ratio(para_text) >= min_ratio:
                    seg_id = f"{base_id}_{chr(96+para_num)}"
                    segments.append((seg_id, para_text))
                    para_num += 1
                current_para = []
                current_words = 0

    # Final paragraph
    if current_para:
        para_text = clean_text(' '.join(current_para))
        if current_words >= min_words and detect_latin_ratio(para_text) >= min_ratio:
            seg_id = f"{base_id}_{chr(96+para_num)}"
            segments.append((seg_id, para_text))

    return segments


def segment_tarregi_unified(
    source_file, source_name
):
    """
    Унифицированная сегментация Tarregi.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с параметрами по умолчанию (min_latin_ratio=0.70, min_words=15)
    raw_segments = segment_tarregi(text, source_name, min_latin_ratio=0.70, min_words=15)
    # Валидация
    return validate_segments(raw_segments, source_name)
if __name__ == "__main__":
    import docx
    from pathlib import Path

    p = Path("data/ObychaiTarregi1290E.docx")
    if p.exists():
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)

        segs = segment_tarregi(text, "Tarregi", min_latin_ratio=0.70, min_words=15)

        print(f"Tarregi: {len(segs)} segments\n")

        for sid, stxt in segs:
            words = len(stxt.split())
            ratio = detect_latin_ratio(stxt)
            print(f"{sid:25s} | {words:3d}w | LAT{ratio:.2f} | {stxt[:70]}...")
    else:
        print("data/ObychaiTarregi1290E.docx not found")
