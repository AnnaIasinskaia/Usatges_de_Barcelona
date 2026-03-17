"""Segmenter for Evangelium."""
import re
from .seg_common import clean_text, validate_segments
from .seg_default import segment_default


def segment_evangelium(text, source_name, max_segment_words=200):
    chapter_re = re.compile(r'\[(\d+)\]')
    parts = chapter_re.split(text)
    segments = []
    i = 1
    while i < len(parts) - 1:
        ch_num = parts[i].strip()
        ch_text = parts[i + 1].strip()
        i += 2
        if not ch_text:
            continue
        ch_text = re.sub(r'(?<!\[)\b(\d{1,3})\b(?!\])', ' ', ch_text)
        ch_text = clean_text(ch_text)
        if len(ch_text) < 30:
            continue
        words = ch_text.split()
        if len(words) <= max_segment_words:
            segments.append((f"{source_name}_Ch{ch_num}", ch_text))
        else:
            chunk_idx = 1
            for start in range(0, len(words), max_segment_words):
                chunk = " ".join(words[start:start + max_segment_words])
                if len(chunk) >= 30:
                    segments.append((f"{source_name}_Ch{ch_num}_{chunk_idx}", chunk))
                    chunk_idx += 1
    if not segments:
        return segment_default(text, source_name, max_segment_words)
    return validate_segments(segments, source_name)


def segment_evangelium_unified(
    source_file, source_name, min_words=10, max_words=150
):
    """
    Унифицированная сегментация Evangelium.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с max_segment_words = max_words
    raw_segments = segment_evangelium(text, source_name, max_segment_words=max_words)

    # Применяем ограничения по словам (min_words уже частично учтено, но проверим)
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == "__main__":
    from pathlib import Path
    import docx
    p = Path("data/Evangelium.docx")
    if p.exists():
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        segs = segment_evangelium(text, "Evangelium")
        print(f"Evangelium: {len(segs)} segments")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:100]}")
    else:
        print(f"Not found: {p}")
