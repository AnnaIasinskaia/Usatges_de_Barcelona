"""
Usatges de Barcelona segmenter (Bastardas edition).

Этот модуль выделен из preprocessing.py для модульной архитектуры.
Логика сегментации Usatges ИДЕНТИЧНА оригиналу из preprocessing.py.

Парсер Bastardas: 125 основных глав + до 20 адвентивных из аппендиксов A-D.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .seg_common import read_source_file, validate_segments


# =====================================================================
#  Segmentation helpers
# =====================================================================

_SIGLA_RE = re.compile(
    r'\bPHN\b|\bHNV\b|\bPNV\b|\bPHV\b|\bPH\b|\bPN\b|\bHV\b|\bNV\b'
    r'|\bom\.\b|\|\|'
)


def _is_apparatus_line(line: str) -> bool:
    """True if line looks like a critical apparatus entry."""
    s = line.strip()
    if not s:
        return False
    if _SIGLA_RE.search(s):
        return True
    if re.match(r'^\d+\s*\(us\..*\)\.\s*\d', s):
        return True
    return False


def _is_catalan_or_note(line: str) -> bool:
    """True if line is clearly Catalan text or editorial note."""
    s = line.strip()
    if not s:
        return False
    catalan_markers = [
        r'\bque\b.*\blos\b', r'\bde\b.*\bla\b', r'\bdels\b', r'\bels\b',
        r'\bsie\b', r'\bhom\b', r'\btots\b', r'\baquesta\b', r'\bpleyts\b',
        r'\bemenat\b', r'\bcavaler\b', r'\bsenyors?\b', r'\bhómens\b',
        r'\bf\.\s*\d+[rv]',
        r'^\[', r'\[A\]',
    ]
    sl = s.lower()
    for pat in catalan_markers:
        if re.search(pat, sl):
            return True
    if re.match(r'^\d+\.\s+[A-Z]', s) and any(w in sl for w in ['captol', 'manuscrit', 'traducci']):
        return True
    return False


def _classify_block_as_latin(lines) -> bool:
    """Check if the first few lines of a block are Latin (not Catalan/apparatus)."""
    if isinstance(lines, str):
        lines = [l.strip() for l in lines.split('\n') if l.strip()]
    sample = lines[:5]
    if not sample:
        return False
    if _is_apparatus_line(sample[0]):
        return False
    catalan_count = sum(1 for l in sample if _is_catalan_or_note(l))
    if catalan_count > len(sample) * 0.5:
        return False
    first = sample[0].lower()
    latin_indicators = [
        'quis', 'uel', 'aut', 'iudic', 'emend', 'usatic', 'princip',
        'ecclesi', 'homini', 'milite', 'solido', 'compos', 'senior',
    ]
    has_latin = any(ind in first for ind in latin_indicators)
    has_caps = bool(re.match(r'^[A-Z]{3,}', sample[0].strip()))
    return has_latin or has_caps or not _is_catalan_or_note(sample[0])


def _extract_latin_portion(lines) -> str:
    """Extract Latin text, stopping at apparatus or Catalan."""
    if isinstance(lines, str):
        lines = [l.strip() for l in lines.split('\n') if l.strip()]
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _is_apparatus_line(s):
            break
        if _is_catalan_or_note(s):
            break
        if re.match(r'^\d+\s+VSATICI\b', s) or re.match(r'^\d+\s+USATGES\b', s):
            continue
        if s in ('VSATICI BARCHINONAE', 'USATGES DE BARCELONA'):
            continue
        if re.match(r'^\d+$', s):
            continue
        result.append(s)
    text = ' '.join(result)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[\d+\]', '', text)
    return text


def _fix_split_headers(text: str) -> str:
    """Fix OCR line-splits: '106\n(us. 127)' -> '106 (us. 127)'."""
    return re.sub(
        r'^(\d{1,3})\s*\n+\s*(\(us\.)',
        r'\1 \2',
        text,
        flags=re.MULTILINE,
    )


# =====================================================================
#  Main Bastardas parser (chapters 1-125)
# =====================================================================

def segment_usatges_bastardas(text: str) -> List[Tuple[str, str]]:
    """
    Parse the Bastardas critical edition (djvu.txt).
    Returns list of (article_id, latin_text) for chapters 1-125.
    """
    text = _fix_split_headers(text)

    chapter_re = re.compile(
        r'^\s*(\d+)\s*\(us\.\s*([\d\-,\.\s]+)\)',
        re.MULTILINE,
    )
    bare_re = re.compile(
        r'^\s*\(us\.\s*([\d\-,\.\s]+)\)',
        re.MULTILINE,
    )

    first_chapter = re.search(r'^\s*1\s*\(us\.\s*1', text, re.MULTILINE)
    anteqvam_pos = first_chapter.start() if first_chapter else 0
    apendix_pos = text.find('APENDIX A', len(text) // 2)
    if anteqvam_pos < 0:
        anteqvam_pos = 0
    if apendix_pos < 0:
        apendix_pos = len(text)

    all_headers = []
    primary_positions = set()

    for m in chapter_re.finditer(text):
        if m.start() < anteqvam_pos or m.start() > apendix_pos:
            continue
        all_headers.append((m.start(), m.end(), m.group(2).strip(), m.group(1)))
        primary_positions.add(m.start())

    for m in bare_re.finditer(text):
        if any(abs(m.start() - pp) < 10 for pp in primary_positions):
            continue
        if m.start() < anteqvam_pos or m.start() > apendix_pos:
            continue
        after = text[m.end():m.end() + 200]
        after_lines = [l.strip() for l in after.split('\n') if l.strip()]
        if after_lines and re.match(r'^[A-Z]{3,}', after_lines[0]):
            all_headers.append((m.start(), m.end(), m.group(1).strip(), None))

    all_headers.sort(key=lambda x: x[0])
    if not all_headers:
        return []

    latin_blocks = {}

    for i, (pos, hdr_end, us_str, cap_str) in enumerate(all_headers):
        us_key = us_str.replace(' ', '')
        if us_key in latin_blocks:
            continue

        block_end = all_headers[i + 1][0] if i + 1 < len(all_headers) else apendix_pos
        raw = text[hdr_end:block_end]

        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        if not lines:
            continue
        if not _classify_block_as_latin(lines):
            continue

        latin_text = _extract_latin_portion(lines)
        if len(latin_text) > 15:
            if cap_str:
                sort_key = int(cap_str)
            else:
                nums = re.findall(r'\d+', us_str)
                sort_key = int(nums[0]) + 1000 if nums else 9999
            latin_blocks[us_key] = {'us': us_str, 'text': latin_text, 'sort_key': sort_key}

    segments = []
    for us_key in sorted(latin_blocks, key=lambda k: latin_blocks[k]['sort_key']):
        b = latin_blocks[us_key]
        seg_id = f"Us_{b['us'].replace(' ', '')}"
        segments.append((seg_id, b['text']))

    return segments


# =====================================================================
#  Appendix parser (adventitious usatges from Appendices A-D)
# =====================================================================

def _parse_appendix_usatges(text: str) -> List[Tuple[str, str]]:
    """
    Parse the 20 adventitious usatges from Appendices A-D.
      A: us. 16, 63, 96
      B: us. 82, 85-90
      C: us. 145-152
      D: us. 139-140
    """
    appendix_re = re.compile(
        r'^([A-D])([l1-8])\s*\(us\.\s*(\d+)\)',
        re.MULTILINE,
    )

    text_headers = []
    for m in appendix_re.finditer(text):
        after = text[m.end():m.end() + 30]
        if re.match(r'\.\s*\d', after):
            continue
        appendix = m.group(1)
        item = m.group(2).replace('l', '1')
        us_num = m.group(3)
        text_headers.append((appendix, item, us_num, m.start()))

    if not text_headers:
        return []

    segments = []
    for i, (app, item, us_num, pos) in enumerate(text_headers):
        header_end = text.index('\n', pos) + 1

        if i + 1 < len(text_headers):
            end = text_headers[i + 1][3]
        else:
            for marker in ['ÍNDEXS', 'NDEXS', 'INDEX']:
                idx = text.find(marker, pos)
                if idx != -1:
                    end = idx
                    break
            else:
                end = min(pos + 2000, len(text))

        raw = text[header_end:end].strip()
        lines = raw.split('\n')
        clean = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.match(r'^[A-D][l1-8]\s*\(us\.', s):
                break
            if re.match(r'^AP[EÈ]NDIX', s):
                break
            if re.match(r'^APENDIXS\s+\d', s):
                break
            if s.startswith('Observacions'):
                break
            if re.match(r'^\d+\s+APENDIXS', s) or re.match(r'^\d+\s+VSATICI', s):
                continue
            if re.match(r'^\d+$', s):
                continue
            clean.append(line.rstrip())

        latin = ' '.join(clean)
        latin = re.sub(r'(\w)- (\w)', r'\1\2', latin)
        latin = re.sub(r'\s+', ' ', latin).strip()

        if len(latin) > 20:
            segments.append((f"Us_{us_num}", latin))

    return segments


# =====================================================================
#  Entry points
# =====================================================================

def segment_usatges(text: str) -> List[Tuple[str, str]]:
    """
    Сегментация Usatges из текстового файла Bastardas (djvu.txt).
    125 основных обычаев + до 20 адвентивных из аппендиксов A-D.
    """
    segments = segment_usatges_bastardas(text)
    appendix_segments = _parse_appendix_usatges(text)

    existing_ids = {seg_id for seg_id, _ in segments}
    for seg_id, seg_text in appendix_segments:
        if seg_id not in existing_ids:
            segments.append((seg_id, seg_text))
            existing_ids.add(seg_id)

    return segments


def segment_usatges_unified(source_file, source_name):
    """
    Унифицированная функция сегментации для Usatges de Barcelona.

    Параметры
    ---------
    source_file : str | Path
        Путь к файлу с текстом источника.
    source_name : str
        Имя источника.

    Возвращает
    ----------
    List[Tuple[str, str]]
        Список сегментов в формате (segment_id, segment_text).
    """
    text = read_source_file(source_file)
    raw_segments = segment_usatges(text)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/Bastardas_Usatges_de_Barcelona_djvu.txt"),
        Path("data/Bastardas Usatges de Barcelona_djvu.txt"),
        Path("Bastardas_Usatges_de_Barcelona_djvu.txt"),
        Path("/mnt/data/Bastardas_Usatges_de_Barcelona_djvu.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segments = segment_usatges_unified(src, "UsatgesBarcelona")
    print(f"UsatgesBarcelona: {len(segments)} segments")

    if segments:
        print("First 3 segments:")
        for seg_id, seg_text in segments[:3]:
            preview = seg_text[:120] + "..." if len(seg_text) > 120 else seg_text
            print(f"  {seg_id}: {preview}")

        print("Last 3 segments:")
        for seg_id, seg_text in segments[-3:]:
            preview = seg_text[:120] + "..." if len(seg_text) > 120 else seg_text
            print(f"  {seg_id}: {preview}")


if __name__ == "__main__":
    main()
