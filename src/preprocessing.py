"""
Step 1-2: Text loading, normalization, and stemming-oriented preprocessing.

Unified public interface used by the current pipeline:
- normalize_latin(text)
- tokenize_latin(text)
- LatinLemmatizer(use_collatinus=...)
- preprocess_segment(text, lemmatizer, remove_stopwords=True, min_length=3)
- load_docx(path)
- load_txt(path)

Key properties:
- no pycollatinus dependency
- local mode detection per segment
- unified normalization contract
- mode-aware stemming
- unified graphic normalization (j->i, v->u)
- stronger rule-based stemming with irregular Latin forms
- stronger OCR/apparatus filtering
- safer handling of numbers and Roman numerals
"""
from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import docx as _docx_module
except ImportError:
    _docx_module = None

__all__ = [
    "normalize_latin",
    "tokenize_latin",
    "LatinLemmatizer",
    "preprocess_segment",
    "load_docx",
    "load_txt",
    "LATIN_STOPWORDS",
    "ROMANCE_STOPWORDS",
    "detect_mode",
]

LATIN_FUNCTION_WORDS = {
    "et", "in", "est", "non", "ad", "ut", "si", "cum", "per", "sed", "qui",
    "que", "quod", "uel", "vel", "aut", "de", "ex", "ab", "hic", "ille",
    "ipse", "is", "ea", "id", "hoc", "nec", "neque", "atque", "tamen", "enim",
    "autem", "iam", "sic", "tam", "quam", "dum", "sub", "super", "inter", "pro",
    "sine", "nisi", "ante", "post", "ergo", "igitur", "quoque", "etiam", "omnis",
    "esse", "sum", "fui", "suo", "sua", "suum", "eius", "eorum", "nos", "uos",
    "vos", "ego", "tu", "se", "sibi", "siue", "sive", "ac", "nam", "quia",
    "quidem", "nullus", "quilibet", "inde", "unde", "ita", "idem", "apud",
    "quis", "quid", "quae", "quas", "quos", "quibus", "illud", "illi", "illas",
    "fuerit", "fuerint", "debet", "debent", "iudex", "iudicio", "leges", "lex",
    "ius", "iure", "contra", "apud", "inter", "super", "infra",
}

LATIN_STOPWORDS = {
    "et", "in", "est", "non", "ad", "ut", "si", "cum", "per", "sed", "qui",
    "que", "quod", "uel", "vel", "aut", "de", "ex", "ab", "hic", "ille",
    "ipse", "is", "ea", "id", "hoc", "nec", "neque", "atque", "tamen", "enim",
    "autem", "iam", "sic", "tam", "quam", "dum", "sub", "super", "inter", "pro",
    "sine", "nisi", "ante", "post", "ergo", "igitur", "quoque", "etiam", "omnis",
    "esse", "sum", "fui", "suo", "sua", "suum", "eius", "eorum", "nos", "uos",
    "vos", "ego", "tu", "se", "sibi", "siue", "sive", "ac", "nam", "quia",
    "quidem", "nil", "nichil", "inde", "unde", "ita", "idem", "apud",
}

ROMANCE_FUNCTION_WORDS = {
    "d", "l", "que", "dels", "deles", "dellas", "dones",
    "els", "les", "los", "amb", "senyor", "senyoria", "ciutat",
    "usatge", "usatges", "monestir", "batlle", "barcelona", "casa", "vila",
}

ROMANCE_STOPWORDS = {
    "e", "o", "de", "del", "dels", "la", "las", "les", "lo", "los",
    "el", "els", "al", "als", "un", "una", "uns", "unes", "en", "ab", "amb",
    "per", "que", "qui", "com", "si", "car", "on", "tot", "tots", "totes",
    "aquel", "aquell", "aquells", "aquella", "aquelles", "d", "l",
}

MIXED_STOPWORDS = LATIN_STOPWORDS | ROMANCE_STOPWORDS

APPARATUS_TOKENS = {
    "brv", "stv", "cod", "codd", "ms", "mss", "cet", "cett", "cap", "tit",
    "gl", "glos", "var", "lect", "ed", "fol", "pag", "app", "cf", "ibid",
    "uar", "uariant", "variant", "vari", "sigl",
}

PROTECTED_LATIN_WORDS = {
    "petrus", "didymus", "iesus", "maria", "monumentum", "hereditatem",
    "hereditas", "iustitia", "iustitiam", "discipulum", "operam",
    "appellatum", "nosse", "aequi", "aequum", "iudicium", "iudicio",
    "abraham", "generationis", "narrationem", "fratres",
    "tradiderunt", "dominicae", "institutis", "sacerdotes", "notitiam",
}

LATIN_IRREGULAR_STEMS = {
    "possum": "poss",
    "potes": "pot",
    "potest": "pot",
    "possumus": "poss",
    "potestis": "potest",
    "possunt": "poss",
    "posse": "poss",
}

LATIN_VERB_SUFFIXES = [
    ("erunt", "er"),
    ("uerunt", "u"),
    ("untur", "unt"),
    ("buntur", "b"),
    ("antur", "ant"),
    ("entur", "ent"),
    ("ientes", "ient"),
    ("entes", "ent"),
    ("antes", "ant"),
    ("ntes", "nt"),
    ("bamus", "b"),
    ("batis", "b"),
    ("bant", "b"),
    ("mus", ""),
    ("tis", ""),
    ("nt", ""),
    ("tur", ""),
    ("mur", ""),
    ("mini", ""),
    ("ri", ""),
    ("re", ""),
    ("it", ""),
    ("at", ""),
    ("et", ""),
]

LATIN_SUFFIXES_LONG = [
    "ationibus", "itionibus",
    "ationem", "itionem", "ationis", "itionis",
    "mentorum", "mentarum", "mentibus",
    "tionibus", "sionibus", "tionem", "sionem", "tionis", "sionis",
    "itudinis", "itudine",
    "tudinis", "tatibus", "tatis", "tatem",
    "issima", "issimi", "issimum", "issimis", "issimam",
    "biliter", "abiliter", "ibiliter",
    "orum", "arum", "ibus", "ium", "ius",
    "atus", "atum", "atae", "atis", "ator", "atio",
    "mentum", "torum", "turis",
]

LATIN_SUFFIXES_SHORT = [
    "are", "ere", "ire",
    "ae", "am", "em", "um", "us", "is", "es", "os", "as",
]

ROMANCE_SUFFIXES = [
    "aments", "ament", "acions", "acion", "atges", "atge", "itats", "itat",
    "ments", "ment", "tats", "tat", "ures", "ura", "dor", "dora", "dors",
]

COMMON_EDITORIAL_PATTERNS = [
    r"\[\s*\d+\s*\]",
    r"\(\s*\d+\s*\)",
    r"fol\.\s*\d+[rv]?",
    r"p\.\s*\d+",
    r"cap\.\s*[ivxlcdm]+",
]

_ROMAN_RE = re.compile(
    r"^(?=[mdclxviuij]+$)m{0,4}(cm|cd|d?c{0,3})"
    r"(xc|xl|l?x{0,3})(ix|iv|iu|u?i{0,3})$",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-zà-ÿ]+(?:['’][a-zà-ÿ]+)?|\d+", re.IGNORECASE)
# Split only productive enclitics that are safe for retrieval purposes.
# We intentionally do NOT split final -ne, because it creates false breaks
# inside ordinary lexical forms like "condicione" -> "condicio" + "ne".
_ENCLITIC_RE = re.compile(r"^([a-zà-ÿ]{2,}?)(que|ue|ve)$", re.IGNORECASE)
_LETTER_RE = re.compile(r"[a-zà-ÿ]", re.IGNORECASE)

# Lexicalized forms that should stay whole and not be mechanically split as base + enclitic.
_NO_ENCLITIC_SPLIT = {
    "atque",
    "neque",
    "quoque",
    "itaque",
    "namque",
    "ubique",
    "undique",
    "usque",
}

# =============================================================================
# Text loading
# =============================================================================

def _load_docx_fast(path: Path) -> str:
    with zipfile.ZipFile(str(path)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    text = re.sub(r"<w:p[^>]*>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_docx(path: Path) -> str:
    file_size = path.stat().st_size
    if file_size > 500_000:
        return _load_docx_fast(path)
    if _docx_module is None:
        return _load_docx_fast(path)
    doc = _docx_module.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _squash_repeated_nonletters(text: str) -> str:
    text = re.sub(r"[_=~•·]+", " ", text)
    text = re.sub(r"[-]{2,}", " ", text)
    text = re.sub(r"[|¦]+", " ", text)
    return text


def basic_cleanup(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("Æ", "Ae").replace("æ", "ae").replace("Œ", "Oe").replace("œ", "oe")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = _squash_repeated_nonletters(text)

    for pattern in COMMON_EDITORIAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"(?m)^\s*[\d\W_]{3,}\s*$", " ", text)
    text = re.sub(r"(?<=\w)[/\\](?=\w)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =============================================================================
# Tokenization / unified compatibility API
# =============================================================================

def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _normalize_uv_latin(token: str) -> str:
    """
    Historical Latin normalization:
    collapse graphic u/v variation to a unified u-based representation.

    Examples:
        venerit  -> uenerit
        privato  -> priuato
        novum    -> nouum
        uidentur -> uidentur
    """
    t = token.lower()
    return t.replace("v", "u")


def normalize_latin(text: str) -> str:
    cleaned = basic_cleanup(text).lower()
    cleaned = _strip_accents(cleaned)
    cleaned = cleaned.replace("j", "i")
    cleaned = cleaned.replace("æ", "ae").replace("œ", "oe")
    cleaned = _normalize_uv_latin(cleaned)
def _normalize_graphics(text: str) -> str:
    text = text.lower()
    text = _strip_accents(text)
    text = text.replace("j", "i")
    text = text.replace("v", "u")
    text = text.replace("æ", "ae").replace("œ", "oe")
    return text


def normalize_latin(text: str) -> str:
    cleaned = basic_cleanup(text)
    cleaned = _normalize_graphics(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_latin(text: str) -> List[str]:
    tokens = tokenize_text(text)
    result = []
    for tok in tokens:
        lower = tok.lower()
        if lower in _NO_ENCLITIC_SPLIT:
            result.append(lower)
            continue
        lower = _normalize_graphics(tok)
        m = _ENCLITIC_RE.match(lower)
        if m and m.group(2) in {"que", "ue", "ve"}:
            base = m.group(1)
            enclitic = m.group(2)
            if base:
                result.append(base)
            result.append(enclitic)
        else:
            result.append(lower)
    return [t for t in result if t]


def is_roman_numeral(token: str) -> bool:
    if not token:
        return False
    return bool(_ROMAN_RE.match(token.lower().replace("j", "i").replace("v", "u")))


def _latin_shape_score(token: str) -> float:
    t = _normalize_graphics(token)
    score = 0.0
    if t in LATIN_FUNCTION_WORDS:
        score += 3.0
    if re.search(r"(orum|arum|ibus|que|ius|ae|am|em|um|us|is)$", t):
        score += 0.8
    if re.search(r"^(quod|quia|quae|quibus|nullus|omnis|ips|illi|apud|contra)", t):
        score += 1.2
    if t.endswith("tur") or t.endswith("ntur"):
        score += 1.0
    return score


def _is_apparatus_like(token: str) -> bool:
    t = _normalize_graphics(token)
    if t in APPARATUS_TOKENS:
        return True
    if len(t) <= 4 and not t.isdigit():
        vowels = sum(ch in "aeiouy" for ch in t)
        if vowels == 0:
            return True
    if re.search(r"^(cod|codd|ms|mss|fol|pag|tit|cap)\d*$", t):
        return True
    return False


def _is_noise_token(token: str) -> bool:
    if not token:
        return True
    if token.isdigit():
        return False
    if is_roman_numeral(token):
        return False

    t = _normalize_graphics(token)
    if len(t) == 1 and t not in {"a", "e", "i", "o"}:
        return True
    if re.search(r"\d", t) and re.search(r"[a-z]", t):
        return True
    if _is_apparatus_like(t):
        return True

    vowels = sum(ch in "aeiouy" for ch in t)
    if len(t) >= 8 and vowels <= 1:
        return True
    if len(t) >= 6 and vowels == 0:
        return True
    if re.search(r"(.)\1\1\1", t):
        return True
    if re.search(r"(q[^u]|[bcdfghiklmnopqrstuwxyz]{5,})", t):
        return True

    return False


def score_latin(tokens: List[str]) -> float:
    return sum(_latin_shape_score(tok) for tok in tokens)


def _is_strong_romance_token(token: str) -> bool:
    raw = token.lower()
    t = _strip_accents(raw)

    if "'" in raw:
        return True
    if t in ROMANCE_FUNCTION_WORDS and _latin_shape_score(t) < 1.0:
        return True
    if re.search(r"(ny|ll)$", t):
        return True
    if re.search(r"(atge|atges|itats|itat|cio|cions|ment|ments|dret|senyor|ciutat)$", t):
        return True
    if re.search(r"^(dels|les|els|los|senyor|ciutat|barcelona|monestir|batlle)", t):
        return True
    return False


def score_romance(tokens: List[str]) -> float:
    score = 0.0
    for tok in tokens:
        raw = tok.lower()
        t = _strip_accents(raw)

        if re.search(r"(orum|arum|ibus|que|ius|ae|am|em|um|us|is)$", t):
            continue
        if "'" in raw:
            score += 2.4
            continue
        if t in {"dels", "les", "els", "los"}:
            score += 2.0
            continue
        if _is_strong_romance_token(tok):
            score += 1.5
    return score


def score_ocr_noise(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    bad = 0
    strange = 0
    apparatus = 0
    mixed_alnum = 0

    for tok in tokens:
        low = tok.lower()
        if _is_noise_token(low):
            bad += 1
        if _is_apparatus_like(low):
            apparatus += 1
        if re.search(r"\d", low) and re.search(r"[a-zà-ÿ]", low, re.IGNORECASE):
            mixed_alnum += 1
        if len(low) >= 8:
            stripped = _normalize_graphics(low)
            vowels = sum(ch in "aeiouy" for ch in stripped)
            if vowels <= 1:
                strange += 1

    total = max(1, len(tokens))
    return (
        (bad / total) * 10.0
        + (strange / total) * 5.0
        + (apparatus / total) * 5.0
        + (mixed_alnum / total) * 8.0
    )


def detect_mode(tokens: List[str]) -> Tuple[str, Dict[str, float]]:
    if not tokens:
        return "unknown", {"latin": 0.0, "romance": 0.0, "ocr_noise": 0.0}

    latin = score_latin(tokens)
    romance = score_romance(tokens)
    noise = score_ocr_noise(tokens)

    scores = {
        "latin": round(latin, 3),
        "romance": round(romance, 3),
        "ocr_noise": round(noise, 3),
    }

    if noise >= 3.2 and (noise >= latin * 0.50 or latin < 4.0):
        return "ocr_noise", scores
    if latin >= 4.0 and latin >= romance * 1.8 and noise < latin * 0.60:
        return "latin", scores
    if romance >= 4.0 and romance >= latin * 1.8 and noise < romance * 0.60:
        return "romance", scores
    if latin >= 2.5 and romance >= 2.5:
        ratio = max(latin, romance) / max(1.0, min(latin, romance))
        if ratio <= 1.6:
            return "mixed", scores
    if latin > romance and latin >= 2.0:
        return "latin", scores
    if romance > latin and romance >= 2.0:
        return "romance", scores
    if noise >= 2.2:
        return "ocr_noise", scores

    return "unknown", scores

# =============================================================================
# Mode-aware normalization
# =============================================================================

def normalize_token_latin(token: str) -> str:
    t = _normalize_graphics(token)
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_romance(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("j", "i").replace("v", "u")
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_mixed(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("j", "i").replace("v", "u")
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_ocr(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("j", "i").replace("v", "u")
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = re.sub(r"(^'+|'+$)", "", t)
    t = re.sub(r"[^a-z0-9']", "", t)
    return t


def normalize_tokens(tokens: List[str], mode: str) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if mode == "latin":
            out.append(normalize_token_latin(tok))
        elif mode == "romance":
            out.append(normalize_token_romance(tok))
        elif mode == "mixed":
            out.append(normalize_token_mixed(tok))
        elif mode == "ocr_noise":
            out.append(normalize_token_ocr(tok))
        else:
            out.append(normalize_token_mixed(tok))
    return out


def _is_healthy_stem(stem: str, original: str, min_len: int = 4, min_ratio: float = 0.50) -> bool:
    if len(stem) < min_len:
        return False
    if len(stem) / max(1, len(original)) < min_ratio:
        return False
    if sum(ch in "aeiouy" for ch in stem) == 0:
        return False
    if re.search(r"(q[^u]|[bcdfghiklmnopqrstuwxyz]{5,})", stem):
        return False
    return True


def stem_latin(word: str) -> str:
    w = normalize_token_latin(word)
    if len(w) < 5:
        return w
    if w in PROTECTED_LATIN_WORDS:
        return w
    if w in LATIN_IRREGULAR_STEMS:
        return LATIN_IRREGULAR_STEMS[w]

    for ending, replacement in LATIN_VERB_SUFFIXES:
        if not w.endswith(ending):
            continue
        stem = w[:-len(ending)] + replacement
        if _is_healthy_stem(stem, w, min_len=3, min_ratio=0.40):
            return stem

    for ending in LATIN_SUFFIXES_LONG:
        if not w.endswith(ending):
            continue
        stem = w[:-len(ending)]
        if _is_healthy_stem(stem, w, min_len=4, min_ratio=0.50):
            return stem

    for ending in LATIN_SUFFIXES_SHORT:
        if not w.endswith(ending):
            continue
        if len(w) < 7:
            continue
        stem = w[:-len(ending)]
        if _is_healthy_stem(stem, w, min_len=4, min_ratio=0.55):
            return stem

    return w


def stem_romance(word: str) -> str:
    w = normalize_token_romance(word)
    if len(w) < 6:
        return w
    for ending in ROMANCE_SUFFIXES:
        if not w.endswith(ending):
            continue
        stem = w[:-len(ending)]
        if _is_healthy_stem(stem, w, min_len=4, min_ratio=0.55):
            return stem
    for ending in ("es", "os", "as", "s"):
        if w.endswith(ending) and len(w) >= 8:
            stem = w[:-len(ending)]
            if _is_healthy_stem(stem, w, min_len=5, min_ratio=0.75):
                return stem
    return w


def stem_mixed(word: str) -> str:
    w = normalize_token_mixed(word)
    if len(w) < 7:
        return w
    for ending in ("ments", "ment", "atge", "atges", "orum", "arum", "ibus"):
        if not w.endswith(ending):
            continue
        stem = w[:-len(ending)]
        if _is_healthy_stem(stem, w, min_len=5, min_ratio=0.70):
            return stem
    return w


def stem_ocr(word: str) -> str:
    return normalize_token_ocr(word)


class LatinLemmatizer:
    def __init__(self, use_collatinus: bool = True):
        self.use_collatinus = False
        print("LatinLemmatizer: mode-aware rule-based stemmer")

    def lemmatize(self, word: str) -> str:
        if not isinstance(word, str) or not word:
            return ""
        return stem_latin(word)

    def lemmatize_tokens(self, tokens: List[str], mode: str = "latin") -> List[str]:
        out: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            if mode == "latin":
                out.append(stem_latin(tok))
            elif mode == "romance":
                out.append(stem_romance(tok))
            elif mode == "mixed":
                out.append(stem_mixed(tok))
            elif mode == "ocr_noise":
                out.append(stem_ocr(tok))
            else:
                out.append(stem_mixed(tok))
        return out


def _stopword_set(mode: str) -> set:
    if mode == "latin":
        return LATIN_STOPWORDS
    if mode == "romance":
        return ROMANCE_STOPWORDS
    return MIXED_STOPWORDS


def _token_quality(token: str, mode: str, segment_scores: Dict[str, float] | None = None) -> float:
    if not token:
        return -10.0
    if token.isdigit():
        return 2.0
    if is_roman_numeral(token):
        return 2.5

    stripped = token.replace("'", "")
    quality = 0.0

    if _LETTER_RE.search(stripped):
        quality += 1.0
    else:
        return -10.0

    vowels = sum(ch in "aeiouy" for ch in stripped)
    if vowels >= 1:
        quality += 0.8
    if len(stripped) >= 5 and vowels >= 2:
        quality += 0.7
    if len(stripped) >= 7 and vowels <= 1:
        quality -= 2.0

    if _is_apparatus_like(stripped):
        quality -= 3.0
    if re.search(r"(q[^u]|[bcdfghiklmnopqrstuwxyz]{5,})", stripped):
        quality -= 2.0
    if re.search(r"\d", stripped) and re.search(r"[a-z]", stripped):
        quality -= 3.0

    if mode == "latin":
        quality += min(1.6, _latin_shape_score(stripped) * 0.45)
    elif mode == "romance":
        if _is_strong_romance_token(stripped):
            quality += 1.2
    elif mode == "mixed":
        quality += min(0.8, _latin_shape_score(stripped) * 0.20)
        if _is_strong_romance_token(stripped):
            quality += 0.6

    if segment_scores:
        noise = segment_scores.get("ocr_noise", 0.0)
        latin = segment_scores.get("latin", 0.0)
        romance = segment_scores.get("romance", 0.0)

        if noise >= 3.0 and len(stripped) <= 4:
            quality -= 0.8
        if noise >= 3.0 and _latin_shape_score(stripped) < 0.5 and not _is_strong_romance_token(stripped):
            quality -= 0.6
        if latin >= 5.0 and mode == "latin":
            quality += 0.4
        if romance >= 4.0 and mode == "romance":
            quality += 0.4

    return quality



LATIN_CONTENT_SUFFIXES = (
    "tio", "tios", "tion", "tionem", "tionis",
    "sio", "sion", "sionem", "sionis",
    "tor", "toris", "toria", "toriam", "torio", "torium", "torius",
    "orium", "oria", "orius",
    "mentum", "menta", "menti", "mentis",
    "tas", "tat", "tatis", "tatem", "itas", "itatem",
    "itia", "itiam", "itio", "ition",
    "o", "os", "orum", "arum",
    "tur", "ntur", "mur", "mus", "tis",
)

def _looks_contentful_latin(token: str) -> bool:
    t = token.replace("'", "")
    if len(t) < 4:
        return False
    if t in LATIN_STOPWORDS:
        return False
    if t.isdigit() or is_roman_numeral(t):
        return True
    if _is_apparatus_like(t):
        return False
    if re.search(r"\d", t) and re.search(r"[a-z]", t):
        return False

    vowels = sum(ch in "aeiouy" for ch in t)
    latin_score = _latin_shape_score(t)

    if latin_score >= 0.8:
        return True
    if len(t) >= 6 and vowels >= 2 and t.endswith(LATIN_CONTENT_SUFFIXES):
        return True
    if len(t) >= 7 and vowels >= 3:
        return True
    return False

def _keep_numeric_token(token: str, segment_scores: Dict[str, float] | None = None) -> bool:
    if not token.isdigit():
        return False
    noise = 0.0
    if segment_scores:
        noise = segment_scores.get("ocr_noise", 0.0)
    if len(token) in {3, 4}:
        return True
    if len(token) >= 5:
        return True
    if len(token) <= 2:
        return noise < 1.5
    return False


def filter_tokens(
    tokens: List[str],
    mode: str,
    remove_stopwords: bool = True,
    min_length: int = 3,
    segment_scores: Dict[str, float] | None = None,
) -> List[str]:
    result: List[str] = []
    stops = _stopword_set(mode)

    for idx, tok in enumerate(tokens):
        if not tok:
            continue
        if tok.isdigit():
            if _keep_numeric_token(tok, segment_scores=segment_scores):
                result.append(tok)
            continue
        if is_roman_numeral(tok):
            result.append(tok)
            continue
        if len(tok) < min_length:
            continue
        if remove_stopwords and tok in stops:
            continue

        quality = _token_quality(tok, mode, segment_scores=segment_scores)
        left = tokens[max(0, idx - 2):idx]
        right = tokens[idx + 1:idx + 3]
        neighbors = left + right
        bad_neighbors = sum(
            1 for n in neighbors
            if _token_quality(n, mode, segment_scores=segment_scores) < 0.0
        )

        if bad_neighbors >= 2:
            quality -= 0.6
        elif bad_neighbors == 1:
            quality -= 0.25

        threshold = 0.35
        if mode == "ocr_noise":
            threshold = 0.75
        elif segment_scores and segment_scores.get("ocr_noise", 0.0) >= 3.0:
            threshold = 0.65
        elif mode == "mixed":
            threshold = 0.45

        if mode == "latin" and _looks_contentful_latin(tok):
            noise = 0.0
            if segment_scores:
                noise = segment_scores.get("ocr_noise", 0.0)

            if noise >= 3.0:
                threshold = min(threshold, 0.25)
            else:
                threshold = min(threshold, 0.30)

            if _latin_shape_score(tok) >= 0.8:
                quality += 0.20
            if len(tok) >= 7:
                quality += 0.10

        if quality < threshold:
            continue
        result.append(tok)

    return result


def preprocess_segment(
    text: str,
    lemmatizer: LatinLemmatizer,
    remove_stopwords: bool = True,
    min_length: int = 3,
    return_debug: bool = False,
):
    if not isinstance(text, str) or not text.strip():
        if return_debug:
            return {
                "mode": "unknown",
                "scores": {"latin": 0.0, "romance": 0.0, "ocr_noise": 0.0},
                "cleaned": "",
                "raw_tokens": [],
                "normalized_tokens": [],
                "stemmed_tokens": [],
                "final_tokens": [],
            }
        return []

    cleaned = basic_cleanup(text)
    raw_tokens = tokenize_latin(cleaned)
    mode, scores = detect_mode(raw_tokens)

    normalized_tokens = normalize_tokens(raw_tokens, mode)
    stemmed_tokens = lemmatizer.lemmatize_tokens(normalized_tokens, mode=mode)
    final_tokens = filter_tokens(
        stemmed_tokens,
        mode=mode,
        remove_stopwords=remove_stopwords,
        min_length=min_length,
        segment_scores=scores,
    )

    if return_debug:
        return {
            "mode": mode,
            "scores": scores,
            "cleaned": cleaned,
            "raw_tokens": raw_tokens,
            "normalized_tokens": normalized_tokens,
            "stemmed_tokens": stemmed_tokens,
            "final_tokens": final_tokens,
        }

    return final_tokens
