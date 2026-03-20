
"""
Step 1-2: Text loading, normalization, and stemming-oriented preprocessing.

Public interface kept compatible with the current pipeline:
- normalize_latin(text)
- tokenize_latin(text)
- LatinLemmatizer(use_collatinus=...)
- preprocess_segment(text, lemmatizer, remove_stopwords=True, min_length=3)
- load_docx(path)
- load_txt(path)

Internal changes:
- no pycollatinus dependency
- local mode detection per segment
- more conservative normalization
- softer Latin stemming
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

# =============================================================================
# Lexical resources
# =============================================================================

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

# Intentionally narrow: only strongly Romance/Catalan signals.
ROMANCE_FUNCTION_WORDS = {
    "d", "l", "que", "dels", "deles", "dellas", "dels", "dones",
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

# Frequent OCR/apparatus markers observed in legal editions.
APPARATUS_TOKENS = {
    "brv", "stv", "cod", "codd", "ms", "mss", "cet", "cett", "cap", "tit",
    "gl", "glos", "var", "lect", "ed", "fol", "pag", "app", "cf", "ibid",
    "uar", "uariant", "variant", "vari", "sigl",
}

# Protected words that were over-stemmed in inspection.
PROTECTED_LATIN_WORDS = {
    "petrus", "didymus", "iesus", "maria", "monumentum", "hereditatem",
    "hereditas", "iustitia", "iustitiam", "discipulum", "operam",
    "appellatum", "nosse", "aequi", "aequum", "iudicium", "iudicio",
}

# Long and reliable suffixes first. Short endings are handled separately and conservatively.
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
    "ntur", "tur", "mus", "tis", "nt",
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
_ENCLITIC_RE = re.compile(r"^([a-zà-ÿ]{3,})(que|ue|ve|ne)$", re.IGNORECASE)
_LETTER_RE = re.compile(r"[a-zà-ÿ]", re.IGNORECASE)

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

# =============================================================================
# Cleanup helpers
# =============================================================================

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
# Tokenization / compatibility API
# =============================================================================

def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def normalize_latin(text: str) -> str:
    """
    Compatibility helper for inspect scripts.
    Keeps normalization intentionally conservative.
    """
    cleaned = basic_cleanup(text).lower()
    cleaned = cleaned.replace("j", "i")
    cleaned = cleaned.replace("æ", "ae").replace("œ", "oe")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_latin(text: str) -> List[str]:
    tokens = tokenize_text(text)
    result = []
    for tok in tokens:
        lower = tok.lower()
        m = _ENCLITIC_RE.match(lower)
        if m and m.group(2) in {"que", "ue", "ve", "ne"}:
            result.append(m.group(1))
        else:
            result.append(lower)
    return [t for t in result if t]

# =============================================================================
# Detection helpers
# =============================================================================

def is_roman_numeral(token: str) -> bool:
    if not token:
        return False
    return bool(_ROMAN_RE.match(token.lower().replace("j", "i")))


def _latin_shape_score(token: str) -> float:
    t = _strip_accents(token.lower().replace("j", "i"))
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
    t = _strip_accents(token.lower())
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

    t = _strip_accents(token.lower())
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

    # Suspicious OCR clusters
    if re.search(r"(q[^u]|[bcdfghjklmnpqrstvwxz]{5,})", t):
        return True

    return False


def score_latin(tokens: List[str]) -> float:
    score = 0.0
    for tok in tokens:
        score += _latin_shape_score(tok)
    return score


def _is_strong_romance_token(token: str) -> bool:
    raw = token.lower()
    t = _strip_accents(raw)

    if "'" in raw:
        return True

    if t in ROMANCE_FUNCTION_WORDS:
        # Do not count it as Romance if it also looks strongly Latin.
        if _latin_shape_score(t) < 1.0:
            return True

    # Strongly Romance/Catalan patterns only.
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

        # Reject obvious Latin-looking tokens from Romance scoring.
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
            stripped = _strip_accents(low)
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

    # Let heavy noise win more often than before.
    if noise >= 3.5 and noise >= latin * 0.55:
        return "ocr_noise", scores

    if latin >= 4.0 and latin >= romance * 1.6:
        return "latin", scores

    if romance >= 4.0 and romance >= latin * 1.6:
        return "romance", scores

    if latin >= 3.0 and romance >= 3.0 and max(latin, romance) / max(1.0, min(latin, romance)) <= 1.8:
        return "mixed", scores

    if latin > romance and latin >= 2.0:
        return "latin", scores

    if romance > latin and romance >= 2.0:
        return "romance", scores

    if noise >= 2.5:
        return "ocr_noise", scores

    return "unknown", scores

# =============================================================================
# Mode-aware normalization
# =============================================================================

def _normalize_uv_latin(token: str) -> str:
    """
    Conservative Latin u/v normalization.
    Avoid global v->u replacement because it damages Romance material.
    """
    t = token.lower()
    if len(t) >= 2 and t[0] == "v" and t[1] in "aeiouy":
        t = "u" + t[1:]
    t = t.replace("vv", "uv")
    return t


def normalize_token_latin(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("j", "i")
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = _normalize_uv_latin(t)
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_romance(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_mixed(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
    t = t.replace("æ", "ae").replace("œ", "oe")
    t = re.sub(r"(^'+|'+$)", "", t)
    return t


def normalize_token_ocr(token: str) -> str:
    t = token.lower()
    t = _strip_accents(t)
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

# =============================================================================
# Stemming
# =============================================================================

def stem_latin(word: str) -> str:
    w = normalize_token_latin(word)
    if len(w) < 5:
        return w

    if w in PROTECTED_LATIN_WORDS:
        return w

    # Strong, reliable reductions first.
    for ending in LATIN_SUFFIXES_LONG:
        if w.endswith(ending):
            stem = w[:-len(ending)]
            if len(stem) >= 5:
                return stem

    # Moderate reductions, only for sufficiently long words.
    for ending in LATIN_SUFFIXES_SHORT:
        if not w.endswith(ending):
            continue
        if len(w) < 7:
            continue
        stem = w[:-len(ending)]
        if len(stem) >= 5:
            return stem

    return w


def stem_romance(word: str) -> str:
    w = normalize_token_romance(word)
    if len(w) < 6:
        return w

    for ending in ROMANCE_SUFFIXES:
        if w.endswith(ending):
            stem = w[:-len(ending)]
            if len(stem) >= 4:
                return stem

    # Very conservative plural stripping.
    for ending in ("es", "os", "as", "s"):
        if w.endswith(ending) and len(w) >= 8:
            stem = w[:-len(ending)]
            if len(stem) >= 5:
                return stem

    return w


def stem_mixed(word: str) -> str:
    w = normalize_token_mixed(word)
    if len(w) < 7:
        return w

    for ending in ("ments", "ment", "atge", "atges", "orum", "arum", "ibus"):
        if w.endswith(ending):
            stem = w[:-len(ending)]
            if len(stem) >= 5:
                return stem

    return w


def stem_ocr(word: str) -> str:
    return normalize_token_ocr(word)


class LatinLemmatizer:
    """
    Backward-compatible wrapper.
    The name is kept for the existing pipeline, but internally this is a
    mode-aware rule-based stemmer.
    """
    def __init__(self, use_collatinus: bool = True):
        self.use_collatinus = False
        print("✓ LatinLemmatizer: mode-aware rule-based stemmer")

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

# =============================================================================
# Final filtering
# =============================================================================

def _stopword_set(mode: str) -> set:
    if mode == "latin":
        return LATIN_STOPWORDS
    if mode == "romance":
        return ROMANCE_STOPWORDS
    return MIXED_STOPWORDS


def _looks_like_garbage(token: str) -> bool:
    if not token:
        return True
    if token.isdigit():
        return False
    if is_roman_numeral(token):
        return False
    if _is_apparatus_like(token):
        return True

    stripped = token.replace("'", "")
    if len(stripped) == 1 and stripped not in {"a", "e", "i", "o"}:
        return True

    if not _LETTER_RE.search(stripped):
        return True

    if len(stripped) <= 4 and stripped not in LATIN_FUNCTION_WORDS and stripped not in ROMANCE_FUNCTION_WORDS:
        vowels = sum(ch in "aeiouy" for ch in stripped)
        if vowels == 0:
            return True

    if len(stripped) >= 7:
        vowels = sum(ch in "aeiouy" for ch in stripped)
        if vowels <= 1:
            return True

    if re.search(r"(q[^u]|[bcdfghjklmnpqrstvwxz]{5,})", stripped):
        return True

    return False


def _keep_numeric_token(token: str) -> bool:
    # Keep likely dates, drop short apparatus-like numbers.
    if not token.isdigit():
        return False
    if len(token) in {3, 4}:
        return True
    if len(token) >= 5:
        return True
    return False


def filter_tokens(
    tokens: List[str],
    mode: str,
    remove_stopwords: bool = True,
    min_length: int = 3,
) -> List[str]:
    result: List[str] = []
    stops = _stopword_set(mode)

    for tok in tokens:
        if not tok:
            continue

        if tok.isdigit():
            if _keep_numeric_token(tok):
                result.append(tok)
            continue

        if is_roman_numeral(tok):
            result.append(tok)
            continue

        if _looks_like_garbage(tok):
            continue

        if len(tok) < min_length:
            continue

        if remove_stopwords and tok in stops:
            continue

        result.append(tok)

    return result

# =============================================================================
# Public segment preprocessing
# =============================================================================

def preprocess_segment(
    text: str,
    lemmatizer: LatinLemmatizer,
    remove_stopwords: bool = True,
    min_length: int = 3,
    return_debug: bool = False,
):
    """
    Full preprocessing pipeline for a single segment.

    Default return value:
        List[str]

    Debug mode:
        dict with mode/scores/intermediate stages
    """
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


if __name__ == "__main__":
    lem = LatinLemmatizer(use_collatinus=False)
    samples = [
        "ANTEQVAM VSATICI fuissent missi solebant iudices iudicare",
        "Et los prohòmens de la ciutat de Barcelona tenguen llur dret.",
        "Si quis hoc fecerit, in iudicio respondeat.",
        "Aqvesta carta fo feta en l'any MCCXIII.",
        "Deammonicioneluper conltringiic olmu'iivfraa plcript fectenrur",
    ]
    for sample in samples:
        dbg = preprocess_segment(sample, lem, return_debug=True)
        print("=" * 80)
        print("RAW:", sample)
        print("MODE:", dbg["mode"], dbg["scores"])
        print("CLEANED:", dbg["cleaned"])
        print("RAW TOKENS:", dbg["raw_tokens"])
        print("NORMALIZED:", dbg["normalized_tokens"])
        print("STEMMED:", dbg["stemmed_tokens"])
        print("FINAL:", dbg["final_tokens"])
