"""
Step 1-2: Text loading, normalization, lemmatization.

Segmentation:
  - segmenters.seg_usatges  (Bastardas edition)
  - source_segmenters.py  (dispatcher -> seg_*.py per source)
"""
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

try:
    import docx as _docx_module
except ImportError:
    _docx_module = None

_COLLATINUS_AVAILABLE = False
try:
    from pycollatinus import Lemmatiseur
    _COLLATINUS_AVAILABLE = True
except ImportError:
    pass

# Re-export segmenters so pipeline.py can do:
#   from preprocessing import segment_usatges, segment_source
from segmenters.seg_usatges import segment_usatges  # noqa: F401
from source_segmenters import segment_source    # noqa: F401

__all__ = [
    "segment_usatges", "segment_source",
    "normalize_latin", "tokenize_latin",
    "LatinLemmatizer", "preprocess_segment",
    "load_docx", "load_txt",
    "LATIN_STOPWORDS",
]

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
    def __init__(self, use_collatinus: bool = True):
        self._collatinus = None
        if use_collatinus and _COLLATINUS_AVAILABLE:
            try:
                self._collatinus = Lemmatiseur()
                print("\u2713 LatinLemmatizer: Collatinus")
            except Exception as e:
                print(f"\u26a0\ufe0f Collatinus: {e}")
        if self._collatinus is None:
            print("\u2713 LatinLemmatizer: suffix-stemmer")

    def lemmatize(self, word: str) -> str:
        if not isinstance(word, str) or not word:
            return ""
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
        return [self.lemmatize(t) for t in tokens if t]


# =====================================================================
#  Text loading
# =====================================================================

def _load_docx_fast(path: Path) -> str:
    """Fast docx loader using zipfile (avoids python-docx overhead on large files)."""
    with zipfile.ZipFile(str(path)) as z:
        xml = z.read('word/document.xml').decode('utf-8')
    text = re.sub(r'<w:p[^>]*>', '\n', xml)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def load_docx(path: Path) -> str:
    """
    Load docx file.
    Uses fast zipfile method for large files (>500KB),
    python-docx for small files (preserves paragraph structure better).
    """
    file_size = path.stat().st_size
    if file_size > 500_000:
        return _load_docx_fast(path)
    if _docx_module is None:
        return _load_docx_fast(path)
    doc = _docx_module.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# =====================================================================
#  Segment preprocessing
# =====================================================================

_ROMAN_RE = re.compile(
    r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iu|u?i{0,3})$"
)


def preprocess_segment(
    text: str,
    lemmatizer: LatinLemmatizer,
    remove_stopwords: bool = True,
    min_length: int = 3,
) -> List[str]:
    """Full preprocessing pipeline for a single segment."""
    if not isinstance(text, str) or not text.strip():
        return []

    normalized = normalize_latin(text)
    tokens = tokenize_latin(normalized)
    if not tokens:
        return []

    lemmas = lemmatizer.lemmatize_tokens(tokens)
    result = []
    for lem in lemmas:
        if not isinstance(lem, str) or len(lem) < min_length:
            continue
        if remove_stopwords and lem in LATIN_STOPWORDS:
            continue
        if _ROMAN_RE.match(lem):
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
