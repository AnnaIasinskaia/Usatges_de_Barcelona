"""Common utilities for unified source segmenters."""

from __future__ import annotations

import re
from pathlib import Path


_APPARATUS_SIGLA = re.compile(
    r"(?:"
    r"\b[A-Z]\s*\d\s*\."
    r"|\b(?:Ms|Ed|Cod)[sd]?\."
    r"|\b(?:deest|evan|add|corr|des)\b"
    r"|\*[A-Z]"
    r"|\|\|"
    r"|\bsic\b.*\b(?:Ms|Ed)\b"
    r")",
    re.IGNORECASE,
)

_FOOTNOTE_RE = re.compile(r"\[\d+\]")
_MULTI_SPACE = re.compile(r"\s+")
_HYPHEN_SPLIT = re.compile(r"(\w)[-\u00ad]\s+(\w)")


def clean_text(text: str) -> str:
    text = _FOOTNOTE_RE.sub("", text)
    text = _HYPHEN_SPLIT.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
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


def validate_segments(
    segments: list[tuple[str, str]],
    source_name: str,
    *,
    min_chars: int = 20,
) -> list[tuple[str, str]]:
    """
    Validate strict unified segment format.

    Expected contract:
        list[tuple[str, str]]

    Segments shorter than `min_chars` after stripping are dropped.
    """
    valid: list[tuple[str, str]] = []

    for i, item in enumerate(segments):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"[{source_name}] Segment #{i}: expected tuple(str, str), "
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

        seg_text = seg_text.strip()
        if len(seg_text) >= min_chars:
            valid.append((seg_id, seg_text))

    return valid


def read_source_file(source_file: str | Path) -> str:
    """
    Read text from .txt or .docx file.
    """
    from pathlib import Path
    path = Path(source_file)
    if path.suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        text = "\n".join(par.text for par in doc.paragraphs)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text

