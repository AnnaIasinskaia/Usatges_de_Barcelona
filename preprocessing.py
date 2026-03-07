"""
Step 1-2: Text loading, segmentation, normalization, lemmatization.

Collatinus (если доступен) → simplemma → suffix-stemmer.
Парсер Bastardas: 125 основных глав + до 20 адвентивных из аппендиксов A-D.
"""
import re
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import docx
except ImportError:
    docx = None

_COLLATINUS_AVAILABLE = False
try:
    from pycollatinus import Lemmatiseur
    _COLLATINUS_AVAILABLE = True
except ImportError:
    pass

try:
    import simplemma
    _SIMPLEMMA_AVAILABLE = True
except ImportError:
    _SIMPLEMMA_AVAILABLE = False


# =====================================================================
#  Medieval Latin normalisation
# =====================================================================

_ENCLITICS = re.compile(r"(\w+)(que|ue|ne)$", re.IGNORECASE)

LATIN_STOPWORDS = {
    "et", "in", "est", "non", "ad", "ut", "si", "cum", "per", "sed",
    "qui", "que", "quod", "uel", "aut", "de", "ex", "ab", "hic", "ille",
    "ipse", "is", "ea", "id", "hoc", "nec", "neque", "atque", "tamen",
    "enim", "autem", "iam", "sic", "tam", "quam", "dum", "sub", "super",
    "inter", "pro", "sine", "nisi", "ante", "post", "ergo", "igitur",
    "quoque", "etiam", "omnis", "esse", "sum", "fui", "suo", "sua",
    "suum", "eius", "eorum", "nos", "uos", "ego", "tu", "se", "sibi",
    "uel", "siue", "ac", "nam", "quia", "quidem", "nil", "nichil",
}


def normalize_latin(text: str) -> str:
    """Apply medieval Latin orthographic normalisation."""
    t = text.lower()
    t = t.replace('j', 'i').replace('v', 'u')
    t = t.replace('ae', 'e').replace('oe', 'e')
    t = t.replace('ph', 'f').replace('y', 'i')
    if not t:
        return t
    cleaned = []
    prev = None
    for ch in t:
        if ch != prev:
            cleaned.append(ch)
        prev = ch
    return ''.join(cleaned)


def tokenize_latin(text: str) -> List[str]:
    """Tokenize Latin text, splitting enclitics."""
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    result = []
    for tok in tokens:
        m = _ENCLITICS.match(tok)
        if m and len(m.group(1)) > 2:
            result.append(m.group(1))
        else:
            result.append(tok)
    return [t for t in result if len(t) > 1]


# =====================================================================
#  Lemmatization
# =====================================================================

_LATIN_ENDINGS = [
    "arum", "orum", "ibus", "atur", "antur",
    "unt", "int", "ent", "ient", "ant",
    "are", "ere", "ire",
    "eat", "it", "et", "at",
    "um", "us", "em",
    "is", "as",
    "ias", "ia", "ae", "ii",
    "os", "a", "e", "i", "o",
]


def stem_latin(word: str) -> str:
    if len(word) < 4:
        return word
    w = word.lower().replace("j", "i").replace("v", "u")
    for ending in _LATIN_ENDINGS:
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            return w[:-len(ending)]
    return w


class LatinLemmatizer:
    """Collatinus → suffix-stemmer fallback."""

    def __init__(self, use_collatinus: bool = True):
        self._collatinus = None
        if use_collatinus and _COLLATINUS_AVAILABLE:
            try:
                self._collatinus = Lemmatiseur()
                print("✓ LatinLemmatizer: Collatinus")
            except Exception as e:
                print(f"⚠️ Collatinus: {e}")
        if self._collatinus is None:
            print("✓ LatinLemmatizer: suffix-stemmer")

    def lemmatize(self, word: str) -> str:
        if self._collatinus:
            try:
                results = list(self._collatinus.lemmatise(word))
                if results and "lemma" in results[0]:
                    lemma = results[0]["lemma"].lower()
                    if lemma and lemma != word.lower():
                        return lemma
            except Exception:
                pass
        return stem_latin(word)

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.lemmatize(t) for t in tokens]


# =====================================================================
#  Text loading
# =====================================================================

def load_docx(path: Path) -> str:
    if docx is None:
        raise ImportError("python-docx is required: pip install python-docx")
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# =====================================================================
#  Segmentation helpers
# =====================================================================

# Sigla-based apparatus detector
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
    # "106 (us. 127). 1 patris..." — apparatus header
    if re.match(r'^\d+\s*\(us\..*\)\.\s*\d', s):
        return True
    return False


def _is_catalan_or_note(line: str) -> bool:
    """True if line is clearly Catalan text or editorial note."""
    s = line.strip()
    if not s:
        return False
    # Catalan markers
    catalan_markers = [
        r'\bque\b.*\blos\b', r'\bde\b.*\bla\b', r'\bdels\b', r'\bels\b',
        r'\bsie\b', r'\bhom\b', r'\btots\b', r'\baquesta\b', r'\bpleyts\b',
        r'\bemenat\b', r'\bcavaler\b', r'\bsenyors?\b', r'\bhómens\b',
        r'\bf\.\s*\d+[rv]',  # folio references
        r'^\[', r'\[A\]',  # editorial brackets
    ]
    sl = s.lower()
    for pat in catalan_markers:
        if re.search(pat, sl):
            return True
    # Numbered editorial notes: "4. Aquest captol..."
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
    # If the first non-empty line is apparatus, not Latin
    if _is_apparatus_line(sample[0]):
        return False
    # If clearly Catalan
    catalan_count = sum(1 for l in sample if _is_catalan_or_note(l))
    if catalan_count > len(sample) * 0.5:
        return False
    # Check for Latin keywords in first line
    first = sample[0].lower()
    latin_indicators = [
        'quis', 'uel', 'aut', 'iudic', 'emend', 'usatic', 'princip',
        'ecclesi', 'homini', 'milite', 'solido', 'compos', 'senior',
    ]
    has_latin = any(ind in first for ind in latin_indicators)
    # Latin text often starts with ALL CAPS
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
        # Page headers
        if re.match(r'^\d+\s+VSATICI\b', s) or re.match(r'^\d+\s+USATGES\b', s):
            continue
        if s in ('VSATICI BARCHINONAE', 'USATGES DE BARCELONA'):
            continue
        # Pure numbers (line numbers)
        if re.match(r'^\d+$', s):
            continue
        result.append(s)
    text = ' '.join(result)
    # Clean hyphens and whitespace
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove footnote references
    text = re.sub(r'\[\d+\]', '', text)
    return text


def _fix_split_headers(text: str) -> str:
    """Fix OCR line-splits: '106\\n(us. 127)' → '106 (us. 127)'."""
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

    # Primary: "106 (us. 127)"
    chapter_re = re.compile(
        r'^\s*(\d+)\s*\(us\.\s*([\d\-,\.\s]+)\)',
        re.MULTILINE,
    )
    # Bare: "(us. 127)" — OCR artifact (no cap number)
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

    # Collect all headers into unified list
    all_headers = []  # (position, end, us_str, cap_str_or_None)
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
        # Verify: next non-blank line should start with Latin uppercase
        after = text[m.end():m.end() + 200]
        after_lines = [l.strip() for l in after.split('\n') if l.strip()]
        if after_lines and re.match(r'^[A-Z]{3,}', after_lines[0]):
            all_headers.append((m.start(), m.end(), m.group(1).strip(), None))

    all_headers.sort(key=lambda x: x[0])
    if not all_headers:
        return []

    # For each usatge number, take first Latin block
    latin_blocks = {}  # us_key → {us, text, sort_key}

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
    # Headers: "Al (us. 16)", "B2 (us. 85)", "C4 (us. 147)", "D1 (us. 139)"
    # OCR renders "A1" as "Al"
    appendix_re = re.compile(
        r'^([A-D])([l1-8])\s*\(us\.\s*(\d+)\)',
        re.MULTILINE,
    )

    text_headers = []
    for m in appendix_re.finditer(text):
        after = text[m.end():m.end() + 30]
        if re.match(r'\.\s*\d', after):  # apparatus line
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
#  Segmentation — entry point
# =====================================================================

def segment_usatges(text: str) -> List[Tuple[str, str]]:
    """
    Сегментация Usatges из текстового файла Bastardas (djvu.txt).
    125 основных обычаев + до 20 адвентивных из аппендиксов A-D.
    """
    # 1. Основные (chapters 1-125)
    segments = segment_usatges_bastardas(text)
    for s in segments:
        print(s[0])
    main_count = len(segments)

    # 2. Адвентивные (appendices A-D)
    appendix_segments = _parse_appendix_usatges(text)

    existing_ids = {seg[0] for seg in segments}
    added = 0
    for seg_id, seg_text in appendix_segments:
        if seg_id not in existing_ids:
            segments.append((seg_id, seg_text))
            existing_ids.add(seg_id)
            added += 1

    print(f"✓ Bastardas: {main_count} основных + {added} из аппендиксов = {len(segments)} всего")
    return segments


def segment_source(
    text: str, source_name: str, max_segment_words: int = 150
) -> List[Tuple[str, str]]:
    """Segment a source text into chunks of ~max_segment_words."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip() and len(p.strip()) > 20]
    segments = []
    current_text = ""
    current_words = 0
    seg_idx = 1
    for para in paragraphs:
        words = len(para.split())
        if current_words + words > max_segment_words and current_text:
            segments.append((f"{source_name}_S{seg_idx}", current_text.strip()))
            seg_idx += 1
            current_text = para + " "
            current_words = words
        else:
            current_text += para + " "
            current_words += words
    if current_text.strip():
        segments.append((f"{source_name}_S{seg_idx}", current_text.strip()))
    return segments


def preprocess_segment(
    text: str,
    lemmatizer: LatinLemmatizer,
    remove_stopwords: bool = True,
    min_length: int = 3,
) -> List[str]:
    """Full preprocessing pipeline for a single segment."""
    normalized = normalize_latin(text)
    tokens = tokenize_latin(normalized)
    lemmas = lemmatizer.lemmatize_tokens(tokens)
    result = []
    for lem in lemmas:
        if len(lem) < min_length:
            continue
        if remove_stopwords and lem in LATIN_STOPWORDS:
            continue
        if re.match(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iu|u?i{0,3})$", lem):
            continue
        result.append(lem)
    return result


if __name__ == "__main__":
    lem = LatinLemmatizer(use_collatinus=False)
    test_text = "ANTEQVAM VSATICI fuissent missi solebant iudices iudicare"
    print("TEST:", test_text)
    normalized = normalize_latin(test_text)
    print("NORM:", normalized)
    tokens = tokenize_latin(normalized)
    print("TOKENS:", tokens)
    lemmas = lem.lemmatize_tokens(tokens)
    print("LEMMAS:", lemmas)
