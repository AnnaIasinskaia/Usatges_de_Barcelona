"""Segmenter for Corpus Juris Civilis."""
import re
from .seg_common import clean_text, is_apparatus_line, group_segments, validate_segments
from .seg_default import segment_default


def segment_corpus_juris(text, source_name, max_segment_words=200):
    section_re = re.compile(
        r'(Dig\.\s*\d+\.\d+\.\d+(?:pr)?(?:\.\d+)?)',
        re.IGNORECASE,
    )
    parts = section_re.split(text)
    raw_segments = []
    i = 1
    while i < len(parts) - 1:
        content = parts[i + 1].strip()
        i += 2
        lines = content.split('\n')
        clean_lines = []
        skip_next = True
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if skip_next:
                if re.match(r'^[A-Z][a-z]+\s+\d+\s+', s) or re.match(r'^[A-Z][a-z]+\s+l\.', s):
                    skip_next = False
                    continue
                skip_next = False
            if is_apparatus_line(s):
                continue
            if re.match(r'^\*?\*?Dig\.\s*\d+', s):
                continue
            clean_lines.append(s)
        cleaned = clean_text(' '.join(clean_lines))
        if len(cleaned) >= 20:
            raw_segments.append((f"{source_name}_Dig", cleaned))
    if not raw_segments:
        return segment_default(text, source_name, max_segment_words)
    grouped = group_segments(raw_segments, source_name, max_segment_words)
    return validate_segments(grouped, source_name)


def segment_corpus_juris_unified(
    source_file, source_name, min_words=10, max_words=150
):
    """
    Унифицированная сегментация Corpus Juris Civilis.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с max_segment_words = max_words
    raw_segments = segment_corpus_juris(text, source_name, max_segment_words=max_words)

    # Применяем ограничения по словам
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == "__main__":
    from pathlib import Path
    import docx, time
    p = Path("data/Corpus Juris Civilis.docx")
    if p.exists():
        t0 = time.time()
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        t1 = time.time()
        print(f"Loaded in {t1-t0:.1f}s: {len(text)} chars")
        segs = segment_corpus_juris(text, "CorpusJuris")
        t2 = time.time()
        print(f"CorpusJuris: {len(segs)} segments in {t2-t1:.1f}s")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:80]}")
        print("  ...")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:80]}")
    else:
        print(f"Not found: {p}")
