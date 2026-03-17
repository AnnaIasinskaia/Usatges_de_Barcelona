"""Segmenter for Exceptiones Legum Romanorum Petri."""
import re
from .seg_common import clean_text, is_apparatus_line, group_segments, validate_segments
from .seg_default import segment_default


def segment_exceptiones_petri(text, source_name, max_segment_words=200):
    cap_re = re.compile(
        r'(?:^|\n)\s*(?:C\s*a\s*p|Cap)\s*[\.\)]\s*(\d+)\s*[\.\)]?\s*',
        re.MULTILINE | re.IGNORECASE,
    )
    liber_re = re.compile(r'Liber\s+(\w+)\s*\.?', re.IGNORECASE)
    matches = list(cap_re.finditer(text))
    current_liber = "1"
    segments = []
    for i, m in enumerate(matches):
        cap_num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        for lm in liber_re.finditer(text[:m.start()]):
            current_liber = lm.group(1)
        lines = section_text.split('\n')
        clean_lines = []
        for line in lines:
            s = line.strip()
            if not s or len(s) < 5:
                continue
            if is_apparatus_line(s):
                continue
            if re.match(r'^\d+\)\s', s):
                continue
            if re.match(r'^\(.*\)\s*$', s):
                continue
            clean_lines.append(s)
        cleaned = clean_text(' '.join(clean_lines))
        if len(cleaned) >= 30:
            segments.append((f"{source_name}_L{current_liber}_C{cap_num}", cleaned))
    if not segments:
        return segment_default(text, source_name, max_segment_words)
    grouped = group_segments(segments, source_name, max_segment_words)
    return validate_segments(grouped, source_name)


if __name__ == "__main__":
    from pathlib import Path
    import docx
    p = Path("data/Exeptionis Legum Romanorum Petri.docx")
    if p.exists():
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        segs = segment_exceptiones_petri(text, "ExceptPetri")
        print(f"ExceptPetri: {len(segs)} segments")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:80]}")
    else:
        print(f"Not found: {p}")
