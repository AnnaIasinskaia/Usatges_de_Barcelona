#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified segmenter for Gramoty files.

Compatibility goals:
- keep the old public interface from seg_gramoty_stable.py
- delegate real segmentation to the newer specialized parsers
- do not modify the behavior of seg_gramoty_911.py / seg_gramoty_v2.py

Public API kept intact:
- segment_gramoty(text, source_name='gramoty', min_length=100)
- segment_gramoty_from_file(filepath, source_name='gramoty', min_length=100)
- validate_segments(segments, expected_range=None)
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


_DOC911_RE = re.compile(r'^\s*(\d+(?:\s+bis)?)\s*$', re.IGNORECASE | re.MULTILINE)
_DOC12_MARKED_RE = re.compile(r'^\s*###\s*DOC\s+\d+\s*###\s*$', re.IGNORECASE | re.MULTILINE)
_DOC12_STABLE_RE = re.compile(r'\b1[012]\d{2}\b')
_DOC911_YEAR_RE = re.compile(r'\b(8\d{2}|9\d{2}|10\d{2}|1100)\b')
_DATE_LINE_RE = re.compile(
    r'^\s*(?:abans del \d+.*del\s+)?'
    r'(\d{3,4})'
    r'(?:\s*,\s*([A-Za-zÀ-ÿçÇ\.]+)'
    r'(?:\s*,?\s*(\d{1,2}))?)?'
    r'(?:\s*\.\s*(.+))?'
    r'\s*$'
)

_CATALAN_MONTHS = {
    'gener': 1, 'gen': 1, 'febrer': 2, 'feb': 2, 'març': 3, 'mar': 3,
    'abril': 4, 'abr': 4, 'maig': 5, 'juny': 6, 'jun': 6,
    'juliol': 7, 'jul': 7, 'agost': 8, 'ag': 8, 'setembre': 9, 'set': 9,
    'octubre': 10, 'oct': 10, 'novembre': 11, 'nov': 11, 'desembre': 12, 'des': 12,
}


def _load_module(module_filename: str, module_name: str):
    """Load a sibling module without requiring package installation."""
    module_path = Path(__file__).with_name(module_filename)
    if not module_path.exists():
        raise FileNotFoundError(f"Required module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOD_911 = _load_module('seg_gramoty_911.py', 'seg_gramoty_911_module')
_MOD_12 = _load_module('seg_gramoty_v2.py', 'seg_gramoty_v2_module')

segment_gramoty_911 = _MOD_911.segment_gramoty_911
segment_gramoty_12 = _MOD_12.segment_gramoty_12


def _detect_format(text: str, source_name: str = '') -> str:
    """Choose specialized parser using filename/source_name first, then content."""
    source_l = (source_name or '').lower()

    if '911' in source_l or 'ix' in source_l or 'xi' in source_l:
        return '911'
    if re.search(r'(^|[^0-9])12([^0-9]|$)', source_l) or 'xii' in source_l:
        return '12'

    if _DOC12_MARKED_RE.search(text):
        return '12'

    # 911: many standalone numeric document lines with optional "bis"
    doc911_hits = len(_DOC911_RE.findall(text))
    y911_hits = len(_DOC911_YEAR_RE.findall(text))

    # 12: dates in the 1000-1299 range and/or ### DOC markers
    y12_hits = len(_DOC12_STABLE_RE.findall(text))

    if doc911_hits >= 3 and y911_hits >= 3:
        return '911'
    if y12_hits >= 3:
        return '12'

    # Safe default: the newer 12th-century parser handles more input styles.
    return '12'


def _parse_date_prefix(text: str) -> Optional[Dict[str, Any]]:
    """Extract date metadata from the first non-empty lines, if present."""
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped:
            continue
        m = _DATE_LINE_RE.match(stripped)
        if not m:
            continue

        year = int(m.group(1))
        month = (m.group(2) or '').rstrip('.').strip()
        day = (m.group(3) or '').strip()
        location = (m.group(4) or '').strip()
        month_num = _CATALAN_MONTHS.get(month.lower(), 0) if month else 0
        return {
            'year': year,
            'month': month,
            'month_num': month_num,
            'day': day,
            'location': location,
            'century': (year // 100) * 100 if year else 0,
        }
    return None


def _doc_sort_key(doc_num: Union[int, str]) -> Tuple[int, int, str]:
    """Sort regular docs before bis docs with the same base number."""
    if isinstance(doc_num, int):
        return (doc_num, 0, str(doc_num))

    s = str(doc_num).strip()
    m = re.match(r'^(\d+)(?:\s+(bis))?$', s, re.IGNORECASE)
    if m:
        base = int(m.group(1))
        is_bis = 1 if m.group(2) else 0
        return (base, is_bis, s.lower())

    nums = re.findall(r'\d+', s)
    base = int(nums[0]) if nums else 10**9
    return (base, 2, s.lower())


def _build_segment(source_name: str, doc_num: Union[int, str], text: str) -> Dict[str, Any]:
    """Normalize parser output to the legacy stable interface."""
    date_info = _parse_date_prefix(text) or {
        'year': 0,
        'month': '',
        'month_num': 0,
        'day': '',
        'location': '',
        'century': 0,
    }

    doc_str = str(doc_num).strip()
    doc_id_part = re.sub(r'\s+', '_', doc_str)

    if date_info['year']:
        date_id = f"Y{date_info['year']}"
        if date_info['month']:
            date_id += f"M{date_info['month'][:3]}"
        if date_info['day']:
            date_id += f"D{date_info['day']}"
    else:
        date_id = 'undated'

    metadata = {
        'doc_num': doc_num,
        'year': date_info['year'],
        'month': date_info['month'],
        'month_num': date_info['month_num'],
        'day': date_info['day'],
        'location': date_info['location'],
        'century': date_info['century'],
        'regest_preview': text[:200].replace('\n', ' ').strip(),
    }

    return {
        'id': f'{source_name}_doc{doc_id_part}_{date_id}',
        'doc_num': doc_num,
        'text': text,
        'regest': '',
        'metadata': metadata,
    }


def segment_gramoty(text: str, source_name: str = 'gramoty',
                    min_length: int = 100) -> List[Dict[str, Any]]:
    """
    Stable interface wrapper.

    Dispatches to one of the specialized segmenters and normalizes their output
    back to the legacy dict structure expected by downstream scripts.
    """
    fmt = _detect_format(text, source_name=source_name)

    if fmt == '911':
        raw_docs = segment_gramoty_911(text, debug=False)
    else:
        raw_docs = segment_gramoty_12(text, debug=False)

    segments: List[Dict[str, Any]] = []
    for doc_num, extracted_text in raw_docs:
        if len(extracted_text.strip()) < min_length:
            continue
        segments.append(_build_segment(source_name, doc_num, extracted_text))

    segments.sort(key=lambda s: _doc_sort_key(s['doc_num']))
    return segments


def segment_gramoty_from_file(filepath: Path, source_name: str = 'gramoty',
                              min_length: int = 100) -> List[Dict[str, Any]]:
    """Load a file and segment it using the stable interface."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8')

    effective_source = source_name
    if source_name == 'gramoty':
        effective_source = filepath.stem

    return segment_gramoty(text, source_name=effective_source, min_length=min_length)


def validate_segments(segments: List[Dict[str, Any]],
                      expected_range: Tuple[Optional[int], Optional[int]] = None):
    """Compatibility helper kept from the stable interface."""
    issues: List[str] = []

    doc_nums = [s['doc_num'] for s in segments]
    doc_keys = [str(n) for n in doc_nums]

    dupes = sorted({n for n in doc_keys if doc_keys.count(n) > 1}, key=_doc_sort_key)
    if dupes:
        issues.append(f"Duplicate doc numbers: {dupes}")

    numeric_regular = sorted(
        int(str(n)) for n in doc_nums
        if re.fullmatch(r'\d+', str(n).strip())
    )
    if numeric_regular:
        full_range = set(range(min(numeric_regular), max(numeric_regular) + 1))
        missing = sorted(full_range - set(numeric_regular))
        if missing and len(missing) < 50:
            issues.append(f"Missing doc numbers ({len(missing)}): {missing}")
        elif missing:
            issues.append(f"Missing doc numbers: {len(missing)} gaps")

    lengths = [len(s.get('text', '')) for s in segments]
    if lengths:
        very_short = [str(s['doc_num']) for s in segments if len(s.get('text', '')) < 200]
        very_long = [str(s['doc_num']) for s in segments if len(s.get('text', '')) > 50000]
        if very_short:
            issues.append(f"Very short (<200 chars): docs {very_short[:10]}")
        if very_long:
            issues.append(f"Very long (>50k chars): docs {very_long[:10]}")

    if expected_range and numeric_regular:
        exp_min, exp_max = expected_range
        if exp_min is not None and min(numeric_regular) != exp_min:
            issues.append(f"Expected first doc {exp_min}, got {min(numeric_regular)}")
        if exp_max is not None and max(numeric_regular) != exp_max:
            issues.append(f"Expected last doc {exp_max}, got {max(numeric_regular)}")

    return issues


def segment_gramoty_unified(
    source_file,
    source_name,
    min_words=10,
    max_words=150
):
    """
    Унифицированная функция сегментации для Gramoty (объединённая).
    Соответствует контракту из INTERFACE.md.

    Параметры
    ---------
    source_file : str или Path
        Путь к файлу с текстом (формат .txt или .docx).
    source_name : str
        Имя источника (например, "Gramoty911" или "Gramoty12").
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
    # Используем существующую функцию segment_gramoty, которая возвращает словари
    raw_dicts = segment_gramoty(text, source_name=source_name, min_length=1)
    # Преобразуем словари в пары (id, text)
    segments = []
    for d in raw_dicts:
        seg_id = d['id']  # уже содержит source_name_doc...
        seg_text = d['text']
        segments.append((seg_id, seg_text))
    # Применяем ограничения по словам
    filtered = apply_word_limits(segments, min_words, max_words)
    # Валидация
    return validate_segments(filtered, source_name)
def main():
    test_files = [
        (Path('data/Gramoty911.txt'), 'gramoty911', (1, 552)),
        (Path('data/Gramoty12.txt'), 'gramoty12', (1, 873)),
    ]

    found_files = [(p, n, r) for p, n, r in test_files if p.exists()]
    if not found_files:
        print('No data files found. Expected data/Gramoty911.txt and/or data/Gramoty12.txt')
        return

    all_segments: List[Dict[str, Any]] = []
    for filepath, source_name, expected_range in found_files:
        print('=' * 70)
        print(f' {source_name}: {filepath}')
        print('=' * 70)

        segments = segment_gramoty_from_file(filepath, source_name=source_name)
        issues = validate_segments(segments, expected_range)
        all_segments.extend(segments)

        print(f'  Extracted: {len(segments)}')
        if issues:
            print('  Validation issues:')
            for issue in issues:
                print(f'    - {issue}')
        else:
            print('  Validation passed')

        for seg in segments[:5]:
            print(f"    doc {seg['doc_num']}: {seg['text'][:100].replace(chr(10), ' ')}...")

    print('=' * 70)
    print(f' TOTAL: {len(all_segments)} charters from {len(found_files)} files')
    print('=' * 70)


if __name__ == '__main__':
    main()
