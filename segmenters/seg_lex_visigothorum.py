"""Segmenter for Lex Visigothorum."""
import re
from .seg_common import clean_text, is_apparatus_line, group_segments, validate_segments
from .seg_default import segment_default

_LV_APPARATUS = re.compile(
    r'(?:'
    r'\b[RBEF]\s*\d'
    r'|\bRecc\b|\bErv\b|\bNov\b'
    r'|\bCodd?\.\s*[RBEF]'
    r'|\bcf\.\s'
    r'|\bL\.\s+Rom\.'
    r'|\bGrimm\b|\bDahn\b|\bBrunner\b'
    r'|\bp\.\s*\d+'
    r'|\bsqq?\.'
    r')',
    re.IGNORECASE,
)


def segment_lex_visigothorum(text, source_name, max_segment_words=200):
    paragraphs = text.split('\n')
    clean_paras = []
    for para in paragraphs:
        s = para.strip()
        if not s or len(s) < 10:
            continue
        if is_apparatus_line(s):
            continue
        if _LV_APPARATUS.search(s):
            alpha = sum(1 for c in s if c.isalpha())
            if len(s) > 0 and alpha < len(s) * 0.5:
                continue
        if re.match(r'^\d+\s+LEX\s+VISIGOTH', s, re.IGNORECASE):
            continue
        if re.match(r'^LL\.\s+Sect\.', s):
            continue
        if s.startswith('*') and s.endswith('*'):
            continue
        if re.match(r'^\*[A-Z]', s) and len(s) < 40:
            continue
        if re.match(r'^\s*\d{1,3}\s*$', s):
            continue
        clean_paras.append(s)
    joined = '\n'.join(clean_paras)
    law_re = re.compile(
        r'(?:^|\n)\s*(?:'
        r'([IVX]+\s*,\s*\d+\s*,\s*\d+)'
        r'|([IVX]+\.\s+Titulus)'
        r'|(LIBER\s+\w+)'
        r'|(FLAVIUS\s+\w+)'
        r')',
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(law_re.finditer(joined))
    if len(matches) < 5:
        return segment_default(joined, source_name, max_segment_words)
    segments = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        section_text = joined[start:end].strip()
        label = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "")
        label = re.sub(r'\s+', '_', label.strip())[:20]
        section_text = clean_text(section_text)
        if len(section_text) >= 30:
            segments.append((f"{source_name}_{label}", section_text))
    if not segments:
        return segment_default(joined, source_name, max_segment_words)
    grouped = group_segments(segments, source_name, max_segment_words)
    return validate_segments(grouped, source_name)


def segment_lex_visigothorum_unified(
    source_file, source_name
):
    """
    Унифицированная сегментация Lex Visigothorum.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с max_segment_words = max_words
    min_words=10
    max_words=150
    raw_segments = segment_lex_visigothorum(text, source_name, max_segment_words=max_words)
    # Применяем ограничения по словам
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == "__main__":
    from pathlib import Path
    import docx
    p = Path("data/Lex visigothorum.docx")
    if p.exists():
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        segs = segment_lex_visigothorum(text, "LexVisigoth")
        print(f"LexVisigoth: {len(segs)} segments")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:80]}")
    else:
        print(f"Not found: {p}")
