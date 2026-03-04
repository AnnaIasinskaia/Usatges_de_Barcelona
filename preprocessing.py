"""
Step 1-2: Text loading, segmentation, normalization, lemmatization.
"""
import re
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import docx
except ImportError:
    docx = None

try:
    from pycollatinus import Lemmatiseur
    _COLLATINUS_AVAILABLE = True
except ImportError:
    _COLLATINUS_AVAILABLE = False


# ----- Medieval Latin normalisation -----

_NORM_RULES = [
    (r"j", "i"), (r"J", "I"),
    (r"v", "u"), (r"V", "U"),
    # common medieval variants
    (r"ae", "e"), (r"oe", "e"),
    (r"ph", "f"),
    (r"y", "i"),
    # double-letter simplification
    (r"(.)\1+", r"\1"),
]

_ENCLITICS = re.compile(r"(\w+)(que|ue|ne)$", re.IGNORECASE)

# Broad Latin stop-word list (function words, pronouns, conjunctions)
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
    for pattern, repl in _NORM_RULES:
        t = re.sub(pattern, repl, t)
    return t


def tokenize_latin(text: str) -> List[str]:
    """Tokenize Latin text, splitting enclitics."""
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    result = []
    for tok in tokens:
        m = _ENCLITICS.match(tok)
        if m and len(m.group(1)) > 2:
            result.append(m.group(1))
            # skip enclitic itself (it's a function word)
        else:
            result.append(tok)
    return [t for t in result if len(t) > 1]


# ----- Stemmer (fallback) -----

_ENDINGS = [
    "arum", "orum", "ibus", "atur", "antur",
    "unt", "int", "ent", "ient", "ant",
    "are", "ere", "ire",
    "eat", "it", "et", "at",
    "um", "us", "es", "am", "em", "is",
    "ias", "ia", "ae", "ii",
    "os", "a", "e", "i", "o", "u",
]

def stem_latin(word: str) -> str:
    """Simple suffix-stripping stemmer for Latin (fallback)."""
    w = word.replace("u", "u").replace("j", "i")
    for end in _ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 3:
            w = w[:-len(end)]
            break
    return w


# ----- Lemmatizer -----

class LatinLemmatizer:
    """Wrapper around Collatinus with fallback to simple stemmer."""

    def __init__(self, use_collatinus: bool = True):
        self._lem = None
        if use_collatinus and _COLLATINUS_AVAILABLE:
            try:
                self._lem = Lemmatiseur()
            except Exception:
                pass

    def lemmatize(self, word: str) -> str:
        if self._lem:
            try:
                results = list(self._lem.lemmatise(word))
                if results and "lemma" in results[0]:
                    lemma = results[0]["lemma"].lower()
                    if lemma:
                        return lemma
            except Exception:
                pass
        return stem_latin(word)

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.lemmatize(t) for t in tokens]


# ----- Text loading -----

def load_docx(path: Path) -> str:
    """Load text from a .docx file."""
    if docx is None:
        raise ImportError("python-docx is required: pip install python-docx")
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    """Load text from a .txt file."""
    return path.read_text(encoding="utf-8", errors="replace")


# ----- Segmentation -----

_USATGE_PATTERN = re.compile(
    r"^\s*(?:\d+)\s*\((?:UB|us|Us)\.?\s*[\d.,\s\-]+\)",
    re.MULTILINE,
)

# Pattern for critical apparatus lines (footnotes with sigla like PHN, V, O)
_APPARATUS_LINE = re.compile(
    r"^\s*(?:\d+\s*)?\(us\.\s*[\d.,]+\)|"
    r"[A-Z]{2,4}\s*[:;]|"
    r"\|\||"
    r"om\.|"
    r"\bPHN\b|\bHNV\b|\bPNV\b",
)


def segment_usatges(text: str) -> List[Tuple[str, str]]:
    """
    Segment Usatges text into individual articles.
    Returns list of (article_id, raw_text).
    """
    # Find all article starts
    matches = list(_USATGE_PATTERN.finditer(text))
    if not matches:
        # Fallback: split by paragraph
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [(f"UsatgesP{i+1}", p) for i, p in enumerate(paras)]

    segments = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        header = m.group(0).strip()
        body = text[start:end]

        # Clean: remove apparatus lines
        clean_lines = []
        for line in body.split("\n"):
            if _APPARATUS_LINE.search(line) and len(line) > 10:
                # Heuristic: if line has mostly uppercase sigla, skip
                upper_ratio = sum(1 for c in line if c.isupper()) / max(len(line), 1)
                if upper_ratio > 0.3:
                    continue
            clean_lines.append(line)

        body_clean = " ".join(clean_lines)
        # Extract article number for ID
        num_match = re.search(r"(\d+)", header)
        art_num = num_match.group(1) if num_match else str(idx + 1)
        segments.append((f"Us_{art_num}", body_clean))

    return segments


def segment_source(text: str, source_name: str, max_segment_words: int = 150) -> List[Tuple[str, str]]:
    """
    Segment a source text into chunks of roughly max_segment_words.
    Uses paragraph boundaries where possible.
    """
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

    # Filter
    result = []
    for lem in lemmas:
        if len(lem) < min_length:
            continue
        if remove_stopwords and lem in LATIN_STOPWORDS:
            continue
        # Skip Roman numerals
        if re.match(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iu|u?i{0,3})$", lem):
            continue
        result.append(lem)

    return result
