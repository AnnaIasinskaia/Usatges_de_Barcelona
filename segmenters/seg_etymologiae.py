"""Segmenter for Isidori Etymologiae."""
import re
from .seg_common import clean_text, group_segments, validate_segments
from .seg_default import segment_default


def segment_etymologiae(text, source_name, max_segment_words=200):
    caput_re = re.compile(
        r'(?:^|\n)\s*Caput\s+([IVXLCDMivxlcdm]+)\s*[\.\:]',
        re.MULTILINE,
    )
    matches = list(caput_re.finditer(text))
    segments = []
    for i, m in enumerate(matches):
        caput_num = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_text = re.sub(r'\[\d+\]\s*', '', section_text)
        section_text = clean_text(section_text)
        if len(section_text) >= 30:
            segments.append((f"{source_name}_Cap{caput_num}", section_text))
    if not segments:
        return segment_default(text, source_name, max_segment_words)
    grouped = group_segments(segments, source_name, max_segment_words)
    return validate_segments(grouped, source_name)


def segment_etymologiae_unified(
    source_file, source_name
):
    """
    Унифицированная сегментация Isidori Etymologiae.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с max_segment_words = max_words
    min_words=10
    max_words=150
    raw_segments = segment_etymologiae(text, source_name, max_segment_words=max_words)
    # Применяем ограничения по словам
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == "__main__":
    from pathlib import Path
    import docx
    p = Path("data/Isidori Hispalensis Episcopi Etymologiarum.docx")
    if p.exists():
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        segs = segment_etymologiae(text, "Etymologiae")
        print(f"Etymologiae: {len(segs)} segments")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:80]}")
    else:
        print(f"Not found: {p}")
