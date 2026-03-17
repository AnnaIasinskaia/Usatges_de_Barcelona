"""Common utilities for source segmenters."""
import re
from typing import List, Tuple

_APPARATUS_SIGLA = re.compile(
    r'(?:'
    r'\b[A-Z]\s*\d\s*\.'
    r'|\b(?:Ms|Ed|Cod)[sd]?\.'
    r'|\b(?:deest|evan|add|corr|des)\b'
    r'|\*[A-Z]'
    r'|\|\|'
    r'|\bsic\b.*\b(?:Ms|Ed)\b'
    r')',
    re.IGNORECASE,
)
_LINE_NUM_RE = re.compile(r'^\s*\d{1,3}\s*$')
_FOOTNOTE_RE = re.compile(r'\[\d+\]')
_MULTI_SPACE = re.compile(r'\s+')
_HYPHEN_SPLIT = re.compile(r'(\w)[-\u00ad]\s+(\w)')


def clean_text(text: str) -> str:
    text = _FOOTNOTE_RE.sub('', text)
    text = _HYPHEN_SPLIT.sub(r'\1\2', text)
    text = _MULTI_SPACE.sub(' ', text)
    return text.strip()


def is_apparatus_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3:
        return False
    if _APPARATUS_SIGLA.search(s):
        return True
    alpha_count = sum(1 for c in s if c.isalpha())
    if len(s) > 10 and alpha_count / len(s) < 0.3:
        return True
    return False


def group_segments(raw, source_name, max_words):
    if not raw:
        return []
    grouped = []
    buf_id = raw[0][0]
    buf_text = raw[0][1]
    for seg_id, text in raw[1:]:
        buf_words = len(buf_text.split())
        new_words = len(text.split())
        if buf_words + new_words <= max_words:
            buf_text += " " + text
        else:
            if len(buf_text.strip()) >= 30:
                grouped.append((buf_id, buf_text.strip()))
            buf_id = seg_id
            buf_text = text
    if buf_text and len(buf_text.strip()) >= 30:
        grouped.append((buf_id, buf_text.strip()))
    result = []
    for i, (_sid, text) in enumerate(grouped, 1):
        result.append((f"{source_name}_S{i}", text))
    return result


def validate_segments(segments, source_name):
    valid = []
    for i, item in enumerate(segments):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"[{source_name}] Segment #{i}: expected tuple(str,str), "
                f"got {type(item).__name__}: {repr(item)[:100]}"
            )
        seg_id, seg_text = item
        if not isinstance(seg_id, str):
            raise TypeError(
                f"[{source_name}] Segment #{i}: seg_id must be str, "
                f"got {type(seg_id).__name__}: {repr(seg_id)[:100]}"
            )
        if not isinstance(seg_text, str):
            raise TypeError(
                f"[{source_name}] Segment #{i} ({seg_id}): seg_text must be str, "
                f"got {type(seg_text).__name__}: {repr(seg_text)[:100]}"
            )
        if len(seg_text.strip()) >= 20:
            valid.append((seg_id, seg_text.strip()))
    return valid
