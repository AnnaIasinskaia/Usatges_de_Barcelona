"""Segmenter for Corpus Juris Civilis (Digesta / Pandectae)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import clean_text, validate_segments, read_source_file


# Ищем заголовки фрагментов Digest:
# Dig. 1.1.1pr.
# Dig. 1.1.1.1
# Dig. 1.2.2.14
#
# В тексте встречаются:
# - с завершающей точкой
# - без неё
# - с пробелами
_DIG_RE = re.compile(
    r"(?m)^\s*Dig\.\s*"
    r"(?P<num>\d+\.\d+\.\d+(?:pr|\.\d+)?)"
    r"\.?\s*$",
    re.IGNORECASE,
)

# Строки-атрибуции обычно выглядят так:
# Ulpianus 1 inst.
# Pomponius l.S. enchir.
# Gaius 1 ad l. xii tab.
# Paulus 54 ad ed.
#
# Берём первую содержательную строку после Dig., если она похожа на attribution line.
_AUTHOR_LINE_RE = re.compile(
    r"^[A-Z][A-Za-z\-]+\s+.+$"
)

# Явные заголовки книги/титула, которые не должны попадать в сегмент
_BOOK_TITLE_RE = re.compile(
    r"^(?:DIGESTORUM|SEU|LIBER\s+[A-Z]+|Dig\.\s*\d+\.\d+\.\d+(?:pr|\.\d+)?)",
    re.IGNORECASE,
)


def _normalize_dig_num(num: str) -> str:
    """
    Нормализует номер Dig.-фрагмента для ID:
    1.1.1pr  -> 1.1.1pr
    1.1.1.1  -> 1.1.1.1
    """
    return re.sub(r"\s+", "", num.strip())


def _make_segment_id(source_name: str, dig_num: str) -> str:
    """
    Вынесение фактического номера из документа в ID.
    """
    dig_num = _normalize_dig_num(dig_num)
    return f"{source_name}_Dig_{dig_num}"


def _cleanup_body_lines(lines: List[str]) -> List[str]:
    clean_lines: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _BOOK_TITLE_RE.match(s):
            continue
        clean_lines.append(s)
    return clean_lines


def segment_corpus_juris(text: str, source_name: str) -> List[Tuple[str, str]]:
    """
    Чистая структурная сегментация Corpus Juris Civilis:
    один Dig.-фрагмент = один сегмент.

    Возвращает list[(segment_id, segment_text)].
    """
    matches = list(_DIG_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, str]] = []

    for i, m in enumerate(matches):
        dig_num = _normalize_dig_num(m.group("num"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        if not block:
            continue

        raw_lines = [ln.rstrip() for ln in block.splitlines()]
        raw_lines = [ln for ln in raw_lines if ln.strip()]
        if not raw_lines:
            continue

        author_line = None
        body_lines = raw_lines

        # Первая строка часто — attribution line, сохраняем её.
        first = raw_lines[0].strip()
        if _AUTHOR_LINE_RE.match(first) and not _BOOK_TITLE_RE.match(first):
            author_line = first
            body_lines = raw_lines[1:]

        body_lines = _cleanup_body_lines(body_lines)
        body_text = clean_text(" ".join(body_lines))

        if not body_text:
            continue

        if author_line:
            seg_text = clean_text(f"{author_line} {body_text}")
        else:
            seg_text = body_text

        seg_id = _make_segment_id(source_name, dig_num)
        segments.append((seg_id, seg_text))

    return validate_segments(segments, source_name)


def segment_corpus_juris_unified(source_file, source_name):
    """
    Унифицированная сегментация Corpus Juris Civilis.
    Сегментер сам читает файл и возвращает list[(id, text)].
    """
    text = read_source_file(source_file)
    raw_segments = segment_corpus_juris(text, source_name)
    return validate_segments(raw_segments, source_name)


if __name__ == "__main__":
    import time

    candidates = [
        Path("data/Corpus_Juris_Civilis_v2.txt"),

    ]

    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        print("Not found. Expected one of:")
        for c in candidates:
            print(f"  - {c}")
        raise SystemExit(1)

    t0 = time.time()
    text = read_source_file(p)
    t1 = time.time()

    segs = segment_corpus_juris(text, "CorpusJuris")
    t2 = time.time()

    print(f"Loaded in {t1 - t0:.2f}s: {len(text)} chars")
    print(f"CorpusJuris: {len(segs)} segments in {t2 - t1:.2f}s")

    if segs:
        print("\nFirst 5 segments:")
        for sid, txt in segs[:5]:
            print(f"  {sid}: {txt[:140]}")

        print("\nLast 5 segments:")
        for sid, txt in segs[-5:]:
            print(f"  {sid}: {txt[:140]}")