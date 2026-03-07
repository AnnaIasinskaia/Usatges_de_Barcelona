"""
Step 1-2: Text loading, segmentation, normalization, lemmatization.

Использует Collatinus (если доступен) или suffix-stemmer как fallback.
"""
import re
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import docx
except ImportError:
    docx = None

# Collatinus (опционально)
_COLLATINUS_AVAILABLE = False
try:
    from pycollatinus import Lemmatiseur
    _COLLATINUS_AVAILABLE = True
except ImportError:
    pass

# Simplemma (fallback)
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
    "siue", "ac", "nam", "quia", "quidem", "nil", "nichil",
}


def normalize_latin(text: str) -> str:
    """Apply medieval Latin orthographic normalisation."""
    t = text.lower()
    t = t.replace("j", "i").replace("v", "u")
    t = t.replace("ae", "e").replace("oe", "e")
    t = t.replace("ph", "f").replace("y", "i")
    if not t:
        return t
    # Remove duplicate consecutive characters
    cleaned = [t[0]]
    for ch in t[1:]:
        if ch != cleaned[-1]:
            cleaned.append(ch)
    return "".join(cleaned)


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
    """Simple suffix-stripping stemmer for Latin."""
    if len(word) < 4:
        return word
    w = word.lower().replace("j", "i").replace("v", "u")
    for ending in _LATIN_ENDINGS:
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            return w[: -len(ending)]
    return w


class LatinLemmatizer:
    """Collatinus с fallback на suffix-stemmer."""

    def __init__(self, use_collatinus: bool = True):
        self._collatinus = None
        if use_collatinus and _COLLATINUS_AVAILABLE:
            try:
                self._collatinus = Lemmatiseur()
                print("✓ LatinLemmatizer: Collatinus доступен")
            except Exception as e:
                print(f"⚠️ Collatinus недоступен: {e}")
        if self._collatinus is None:
            print("✓ LatinLemmatizer: использует suffix-stemmer")

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
#  Segmentation — Bastardas critical edition
# =====================================================================
#
#  Структура файла Bastardas djvu.txt:
#  ─────────────────────────────────────
#  Введение (каталанский) → затем для каждой главы ТРОЙКА блоков:
#
#    N (us. X)     ← Заголовок главы (cap number + usatge number)
#    ЛАТИНСКИЙ ТЕКСТ (начинается с заглавных или типичных лат. слов)
#
#    N (us. X)     ← тот же номер
#    КРИТИЧЕСКИЙ АППАРАТ (сиглы: PHN, HNV... || om. : )
#
#    N (us. X)     ← тот же номер
#    КАТАЛАНСКИЙ ПЕРЕВОД (начинается с [A], [E], etc.)
#
#  Задача: извлечь ТОЛЬКО латинские блоки, один на главу.
# =====================================================================

# Markers for Catalan words (used to distinguish Catalan from Latin)
_CATALAN_MARKERS = [
    "l'", "d'", "és ", "axí ", " sie ", "senyor", "cavaler",
    "pagès", "pleyts", "volrà", "pusque", "fossen",
]

# Markers for Latin words
_LATIN_MARKERS = [
    " uel ", "siue", " sint ", "fuerit", "sicut", "emendet",
    "emendetur", "constituer", "princip", "iudic", " eius",
    "debet", "autem", "faciat", " eos ",
]


def _is_apparatus_line(line: str) -> bool:
    """True if line belongs to the critical apparatus."""
    if re.search(r"\|\||om\.|PH[NV]?\s*:|HN[V]?\s*:|PN[V]?\s*:", line):
        return True
    if re.match(r'^[\.\s]*\d+[\s/\-]+\w+', line) and re.search(r'PH|HN|NV|PN', line):
        return True
    return False


def _is_catalan_start(line: str) -> bool:
    """True if line starts a Catalan translation block."""
    clean = re.sub(r'^[\.\s]+', '', line)
    if re.match(r'^\[', clean):
        return True
    if re.match(r'^f\.\s*\d', clean):
        return True
    return False


def _classify_block_as_latin(lines: List[str]) -> bool:
    """
    Given the first few meaningful lines after a chapter header,
    determine if this block is Latin text (vs apparatus or Catalan).
    """
    # Find first meaningful line
    first = None
    for line in lines:
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'^\d+\s*\(us\.', line):
            continue
        if 'VSATICI BARCHINONAE' in line or 'USATGES DE BARCELONA' in line:
            continue
        first = line
        break

    if not first:
        return False

    # Apparatus?
    if _is_apparatus_line(first):
        return False

    # Catalan?
    if _is_catalan_start(first):
        return False

    # Content analysis
    first_clean = re.sub(r'^[\.\s]+', '', first)
    sample = ' '.join(lines[:5]).lower()
    cat_count = sum(1 for m in _CATALAN_MARKERS if m in sample)
    lat_count = sum(1 for m in _LATIN_MARKERS if m in sample)

    # ALL CAPS start — strong Latin signal
    if re.match(r'^[A-Z]{2,}', first_clean):
        return True

    # Typical Latin opening words
    if re.match(
        r'^(Vt|Item|Set|Si |In |De |Similiter|Quicumque|Quod|Quia|Hoc|'
        r'Omnes|Ipse|Post|Moneta|Eandem|Quoniam|Auctoritate)',
        first_clean,
    ):
        return cat_count < 3

    # Latin content predominates
    if lat_count >= 2 and lat_count > cat_count:
        return True

    # No Catalan markers + Latin vocabulary present
    if cat_count == 0 and re.search(r'\b(uel|siue|sicut|fuerit|sint|eius)\b', sample):
        return True

    return False


def _extract_latin_portion(lines: List[str]) -> str:
    """
    From a raw block, extract only the Latin portion.
    Stops at the first apparatus or Catalan line.
    Returns cleaned, joined text.
    """
    latin_lines = []
    for line in lines:
        s = line.strip()
        if not s or re.match(r'^\d+$', s):
            continue
        if 'VSATICI BARCHINONAE' in s or 'USATGES DE BARCELONA' in s:
            continue
        if re.match(r'^\d+\s*\(us\.', s):
            continue

        # STOP at apparatus
        if _is_apparatus_line(s):
            break
        # STOP at Catalan
        if _is_catalan_start(s):
            break
        # STOP at line with heavy Catalan content
        s_lower = s.lower()
        if sum(1 for m in _CATALAN_MARKERS if m in s_lower) >= 2:
            break

        latin_lines.append(s)

    result = ' '.join(latin_lines)
    result = re.sub(r'(\w)- (\w)', r'\1\2', result)   # fix line-break hyphens
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'^[\.\s]+', '', result)           # leading dots
    return result



def _fix_split_headers(text: str) -> str:
    """
    Fix OCR artifacts where a chapter header is split across lines:
      '106\n(us. 127)' → '106 (us. 127)'
    """
    return re.sub(r'(\d+)\s*\n\s*(\(us\.)', r'\1 \2', text)

def segment_usatges_bastardas(text: str) -> List[Tuple[str, str]]:
    """
    Parse the Bastardas critical edition (djvu.txt).

    Returns list of (article_id, latin_text) for each chapter.
    article_id format: "Us_{usatge_number}" (e.g. "Us_3", "Us_4,1", "Us_66,2").
    """
    # Fix OCR line-splits in chapter headers (e.g. "106\n(us. 127)")
    text = _fix_split_headers(text)

    # Find all chapter headers: "N (us. X)"
    chapter_re = re.compile(
        r'^\s*(\d+)\s*\(us\.\s*([\d\-,\.\s]+)\)',
        re.MULTILINE,
    )
    headers = list(chapter_re.finditer(text))
    if not headers:
        return []

    latin_blocks = {}   # cap_num -> {cap, us, text}

    for i, h in enumerate(headers):
        cap = h.group(1)
        us  = h.group(2).strip()

        # Keep only the FIRST Latin block per chapter number
        if cap in latin_blocks:
            continue

        # Text between this header and the next
        block_start = h.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        raw_block = text[block_start:block_end]

        lines = [l.strip() for l in raw_block.split('\n') if l.strip()]
        if not lines:
            continue

        # Classify: is this block Latin?
        if not _classify_block_as_latin(lines):
            continue

        # Extract only the Latin portion (stop at apparatus/Catalan)
        latin_text = _extract_latin_portion(lines)
        if len(latin_text) > 15:
            latin_blocks[cap] = {'cap': cap, 'us': us, 'text': latin_text}

    # Build output sorted by chapter number
    segments = []
    for cap in sorted(latin_blocks, key=lambda x: int(x)):
        b = latin_blocks[cap]
        seg_id = f"Us_{b['us'].replace(' ', '')}"
        segments.append((seg_id, b['text']))

    return segments


# =====================================================================
#  Segmentation — entry point
# =====================================================================

def segment_usatges(text: str) -> List[Tuple[str, str]]:
    """
    Сегментация Usatges из текстового файла Bastardas (djvu.txt).
    Возвращает список (article_id, latin_text).
    """
    segments = segment_usatges_bastardas(text)
    print(f"✓ Bastardas: {len(segments)} латинских сегментов")
    return segments


# =====================================================================
#  Segmentation — generic source texts
# =====================================================================

def segment_source(
    text: str, source_name: str, max_segment_words: int = 150
) -> List[Tuple[str, str]]:
    """Segment a source text into chunks of ~max_segment_words."""
    paragraphs = [
        p.strip() for p in text.split("\n") if p.strip() and len(p.strip()) > 20
    ]

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


# =====================================================================
#  Full preprocessing pipeline for a segment
# =====================================================================

def preprocess_segment(
    text: str,
    lemmatizer: LatinLemmatizer,
    remove_stopwords: bool = True,
    min_length: int = 3,
) -> List[str]:
    """Normalize → tokenize → lemmatize → filter."""
    normalized = normalize_latin(text)
    tokens = tokenize_latin(normalized)
    lemmas = lemmatizer.lemmatize_tokens(tokens)

    result = []
    for lem in lemmas:
        if len(lem) < min_length:
            continue
        if remove_stopwords and lem in LATIN_STOPWORDS:
            continue
        if re.match(
            r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iu|u?i{0,3})$", lem
        ):
            continue
        result.append(lem)
    return result


# =====================================================================
#  Quick self-test
# =====================================================================

if __name__ == "__main__":
    lem = LatinLemmatizer(use_collatinus=False)

    test = "ANTEQVAM VSATICI fuissent missi solebant iudices iudicare"
    print("INPUT:     ", test)
    print("NORMALIZED:", normalize_latin(test))
    tokens = tokenize_latin(normalize_latin(test))
    print("TOKENS:    ", tokens)
    print("LEMMAS:    ", lem.lemmatize_tokens(tokens))
