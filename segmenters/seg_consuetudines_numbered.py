"""Segmenter for numbered-article customs.

Handles: Consuetudines Ilerdenses (1228), Costums de Tarrega (1290),
         Recognoverunt Proceres (1283-84), Consuetudines Miraveti (1319).

Supports mixed Latin / Catalan layouts with various article markers.
"""

import re
from .seg_common import clean_text, is_apparatus_line, group_segments
from .seg_default import segment_default

# ---------------------------------------------------------------------------
# Marker regexes
# ---------------------------------------------------------------------------

# [N]  (Evangelium-style)
_BRACKET_NUM_RE = re.compile(r'\[(\d+)\]')

# [N)  /  [N )  — Tarrega Latin; допускаем '>' и пробелы перед маркером
_BRACKET_PAREN_NUM_RE = re.compile(
    r'[>\s]\[(\d+)\)',           # не привязан к началу строки
)

# Decimal number with dot at line start or после '>'
# Примеры: "107. *Que de plets ...*", " 108. *Dels plets ...*"
_DECIMAL_DOT_RE = re.compile(
    r'(?:^|\n|[>\s])(\d{1,4})\.\s+\*?',
)

# Roman numeral in square brackets: [CXI] (Recognoverunt Latin)
_BRACKET_ROMAN_RE = re.compile(
    r'\[([IVXLCDMivxlcdm]{1,8})\]',
)

# Cap./Art./Capitulum header
_CAP_RE = re.compile(
    r'(?:^|\n|[>\s])(?:Cap(?:itulum|\.)?|Art(?:iculus|\.)?)\s*\.?\s*'
    r'(\d+|[IVXivx]+)\b[^\n]*',
    re.IGNORECASE,
)

# Roman numeral at start of line: "I." (необязательно в начале строки)
_ROMAN_RE = re.compile(
    r'(?:^|\n|[>\s])([IVXLCDMivxlcdm]{1,8})\.\s+\*?',
)

# § N paragraph
_PARA_NUM_RE = re.compile(
    r'(?:^|\n|[>\s])§\s*(\d+)',
)

# Аппарат
_NOISE_RE = re.compile(
    r'(?:\bp\.\s*\d+|\bfol\.\s*\d+|\bMs\b|\bcf\.\s)',
    re.IGNORECASE,
)

# Источники без group_segments
_NO_GROUP_SOURCES = {"CostLleida", "CostTarregi", "RecogProc"}


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------

def _local_validate(segments, source_name):
    """Щадящая проверка длины для обычаев (>= 5 слов)."""
    valid = []
    for i, (seg_id, seg_text) in enumerate(segments):
        if not isinstance(seg_id, str) or not isinstance(seg_text, str):
            raise TypeError(
                f"[{source_name}] bad segment #{i}: "
                f"{type(seg_id).__name__}, {type(seg_text).__name__}"
            )
        words = seg_text.strip().split()
        if len(words) >= 5:
            valid.append((seg_id, " ".join(words)))
    return valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_consuetudines_numbered(text, source_name, max_segment_words=150):
    """Main dispatcher for numbered customs."""

    # 1. [N]-style (Lleida — работает уже хорошо)
    parts = _BRACKET_NUM_RE.split(text)
    if len(parts) > 5:
        segments = _from_bracket_split(parts, source_name)
        if segments:
            return _postprocess_segments(segments, source_name, max_segment_words)

    # 2. Tarrega: [N) в латинской части
    if source_name == "CostTarregi":
        matches = list(_BRACKET_PAREN_NUM_RE.finditer(text))
        if len(matches) >= 3:
            segments = _from_re_matches(matches, text, source_name, "Art")
            if segments:
                return _postprocess_segments(segments, source_name, max_segment_words)

    # 3. Cap./Art. style
    matches = list(_CAP_RE.finditer(text))
    if len(matches) >= 3:
        segments = _from_re_matches(matches, text, source_name, "Cap")
        if segments:
            return _postprocess_segments(segments, source_name, max_segment_words)

    # 4. § N style
    matches = list(_PARA_NUM_RE.finditer(text))
    if len(matches) >= 3:
        segments = _from_re_matches(matches, text, source_name, "Par")
        if segments:
            return _postprocess_segments(segments, source_name, max_segment_words)

    # 5. Roman numeral style ("I.", "II.", ...) — на всякий случай
    matches = list(_ROMAN_RE.finditer(text))
    if len(matches) >= 3:
        segments = _from_re_matches(matches, text, source_name, "Art")
        if segments:
            return _postprocess_segments(segments, source_name, max_segment_words)

    # 6. Recognoverunt: [ROMAN] в латинской части
    if source_name == "RecogProc":
        matches = list(_BRACKET_ROMAN_RE.finditer(text))
        if len(matches) >= 3:
            segments = _from_re_matches(matches, text, source_name, "Cap")
            if segments:
                return _postprocess_segments(segments, source_name, max_segment_words)

    # 7. Оба корпуса: десятичные заголовки "107.", "108." и т. п.
    if source_name in {"CostTarregi", "RecogProc"}:
        matches = list(_DECIMAL_DOT_RE.finditer(text))
        if len(matches) >= 3:
            segments = _from_re_matches(matches, text, source_name, "Art")
            if segments:
                return _postprocess_segments(segments, source_name, max_segment_words)

    # 8. Fallback — общий сегментер (для CostMiraveta и проч.)
    return segment_default(text, source_name, max_segment_words)


def _postprocess_segments(segments, source_name, max_segment_words):
    """Применяет (или нет) group_segments и локальный валидатор."""
    if source_name in _NO_GROUP_SOURCES:
        # Для этих корпусов сохраняем каждую статью отдельным сегментом
        return _local_validate(segments, source_name)
    grouped = group_segments(segments, source_name, max_segment_words)
    return _local_validate(grouped, source_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_article(raw_text):
    """Очистка одной статьи от аппарата и мусора."""
    lines = raw_text.split('\n')
    clean_lines = []
    for line in lines:
        s = line.strip()
        if not s or len(s) < 3:
            continue
        if is_apparatus_line(s):
            continue
        if _NOISE_RE.search(s) and sum(c.isalpha() for c in s) < len(s) * 0.45:
            continue
        if re.match(r'^\s*\d{1,4}\s*$', s):
            continue
        clean_lines.append(s)
    return clean_text(' '.join(clean_lines))


def _from_bracket_split(parts, source_name):
    """Split by [N] bracket markers — аналогично seg_evangelium."""
    segments = []
    i = 1
    while i < len(parts) - 1:
        art_num = parts[i].strip()
        art_text = parts[i + 1].strip()
        i += 2
        if not art_text:
            continue
        art_text = re.sub(r'(?<!\[)\b(\d{1,3})\b(?!\])', ' ', art_text)
        cleaned = _clean_article(art_text)
        if len(cleaned.split()) >= 5:
            segments.append((f"{source_name}_Art{art_num}", cleaned))
    return segments


def _from_re_matches(matches, text, source_name, prefix):
    """Создаёт сегменты по списку совпадений (Cap., §, Roman, decimal etc.)."""
    segments = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cleaned = _clean_article(text[start:end])
        num = (m.group(1) or "").strip().replace(' ', '_')[:12]
        if len(cleaned.split()) >= 5:
            segments.append((f"{source_name}_{prefix}{num}", cleaned))
    return segments


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import docx

    TESTS = [
        ("ObychaiLleidy12271228.docx",        "CostLleida"),
        ("ObychaiTarregi1290E.docx",          "CostTarregi"),
        ("RecognovrentProceres12831284.docx", "RecogProc"),
        ("ObychaiMiraveta1319Fix.docx",       "CostMiraveta"),
    ]

    for fname, sname in TESTS:
        p = Path("data") / fname
        if not p.exists():
            print(f"[SKIP] {fname} not found")
            continue
        doc = docx.Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        segs = segment_consuetudines_numbered(text, sname, 80)
        print(f"\n{sname}: {len(segs)} segments")
        for sid, txt in segs[:5]:
            print(f"  {sid}: {txt[:90]}...")
