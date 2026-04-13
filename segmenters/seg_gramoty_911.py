#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmenter for Gramoty IX–XI (v3).

Цель v3:
- сохранить хорошее покрытие по границам документов;
- извлекать в качестве segment_text исключительно латинский текст документа;
- не возвращать каталанские regests, архивно-издательский аппарат и редакторские вставки.

Рабочий контракт:
- основной парсер возвращает list[(doc_number, text)]
- unified entrypoint возвращает list[(segment_id, segment_text)]
- тестирование выполняется через test_unified_segmenters.py
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

from .seg_common import read_source_file, validate_segments


_DOC_NUM_RE = re.compile(r"^\s*(\d+(?:\s+bis)?)\s*$", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(8\d{2}|9\d{2}|10\d{2}|1100)\b")
_CENTURY_RE = re.compile(r"^\s*Segle\s+[xivlcdm]+\s*$", re.IGNORECASE)

# Архивно-издательский аппарат
_EDITORIAL_PATTERNS = [
    re.compile(
        r"^\s*\[?[A-Z]\]?\s+"
        r"(?:AAC|ACA|ACB|ACC|ACG|ACM|ACS|ACT|ACU|ACV|ADB|ADC|ADG|ADM|ADPO|ADS|AES|AEV|"
        r"AHC|AHCM|AHCT|AHN|AMA|AMB|AMM|AMR|AMSBB|AMSJA|AMU|APSMV|ASD|BC|BNF|BRAH|BSB|BUB|"
        r"LA|LDEU|MEV|RAH)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\[?[A-Z]\]?\s*Original no localitzat\.?", re.IGNORECASE),
    re.compile(r"^\s*Original no localitzat\.?", re.IGNORECASE),
    re.compile(r"^\s*Ed\.\s*", re.IGNORECASE),
    re.compile(r"^\s*\*"),
]

# Отдельные редакторские/научные вставки, которые не должны попадать в латинский сегмент
_SKIP_AFTER_START_PATTERNS = [
    re.compile(r"^\s*(?:\d+\.\s*)?Lex Visigothorum\b", re.IGNORECASE),
]

# Явные каталанские комментарии редактора
_CATALAN_COMMENT_RE = re.compile(
    r"\b(?:Aquesta|Aquest|Seguim|Segons|La recerca|En el decurs|"
    r"Davant|Butlla|Declaració|Acord|Notícia|Nou testimonis|Per manament)\b",
    re.IGNORECASE,
)

# Типичные латинские инципиты / сильные стартовые формулы
_LATIN_START_RE = re.compile(
    r"""^\s*
    (?!\()  # не брать редакторские заголовки в круглых скобках
    (?:In|Notum|Hec|Haec|Hoc|He|Hes|Istae|Iste|Conditiones|Condiciones|Condictiones|Conditionies|
       Noticia|Notitia|Vox|Si|Universis|Johannes|Benedictus|Gregorius|Urbanus|Renerius|
       Quanta|Annus|Incarnationis|Placuit|Mironi|Reverentissimo|Satis|Sume|
       Ego|Nos|Quapropter|Idcirco|Igitur|Quoniam|Manifestum|Convenit|Dum|Cum|
       Antiquitus|Auctore|Omnibus)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LATIN_STOPWORDS = {
    "in", "nomine", "domini", "notum", "sit", "omnibus", "hominibus", "tam",
    "presentibus", "futuris", "qualiter", "quia", "quod", "ego", "nos", "hec",
    "haec", "hoc", "est", "sunt", "unde", "autem", "igitur", "quapropter",
    "idcirco", "cum", "dum", "et", "qui", "quae", "carta", "noticia", "notitia",
    "conditiones", "condiciones", "actum", "acta", "data", "sig", "signum",
    "iudicio", "iudices", "presbiter", "comes", "episcopus",
}

_CATALAN_STOPWORDS = {
    "els", "les", "del", "dels", "de", "la", "el", "que", "amb", "davant",
    "contra", "seva", "seus", "fills", "bisbe", "comte", "jutge", "greuges",
    "per", "i", "al", "als", "una", "un",
}


def segment_gramoty_911(text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Segment Gramoty911 document into individual documents.

    Format: document number on a separate line followed by a date (8XX–1100).
    Also handles "bis" documents such as "79 bis".

    Returns
    -------
    list[tuple[str, str]]
        List of (doc_number, extracted_latin_text) tuples, sorted by document number.
    """
    lines = text.split("\n")
    doc_boundaries = []

    for i, line in enumerate(lines):
        match = _DOC_NUM_RE.match(line)
        if not match:
            continue

        doc_num = match.group(1)
        has_date = False
        date_line = None

        for offset in range(1, min(7, len(lines) - i)):
            next_line = lines[i + offset]
            if not next_line.strip():
                continue

            if _DATE_RE.search(next_line) or _CENTURY_RE.match(next_line.strip()):
                has_date = True
                date_line = next_line.strip()
                break

            if offset > 4:
                break

        if has_date:
            doc_boundaries.append({
                "num": doc_num,
                "line": i,
                "date_line": date_line,
            })

            if debug and len(doc_boundaries) <= 20:
                print(f"Doc {doc_num:>7s} at line {i:5d}, date: {date_line[:60]}")

    doc_boundaries.sort(key=lambda x: x["line"])

    if debug:
        print(f"\nTotal document boundaries found: {len(doc_boundaries)}")

    documents: List[Tuple[str, str]] = []
    for idx, boundary in enumerate(doc_boundaries):
        doc_num = boundary["num"]
        start_line = boundary["line"]
        end_line = doc_boundaries[idx + 1]["line"] if idx + 1 < len(doc_boundaries) else len(lines)

        doc_lines = lines[start_line:end_line]
        latin_text = _extract_latin_text_911(doc_lines)
        if latin_text.strip():
            documents.append((doc_num, latin_text))

    def sort_key(doc_tuple: Tuple[str, str]) -> Tuple[int, int]:
        num_str = doc_tuple[0]
        if "bis" in num_str.lower():
            base_num = int(num_str.split()[0])
            return (base_num, 1)
        return (int(num_str), 0)

    documents.sort(key=sort_key)
    return documents


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ]+", text)


def _normalize_for_start_detection(line: str) -> str:
    # [I]n -> In, [G]regorius -> Gregorius
    return line.replace("[", "").replace("]", "").strip()


def _is_editorial_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if any(p.match(stripped) for p in _EDITORIAL_PATTERNS):
        return True

    # Частые продолжения длинных bibliographic/editorial строк
    if re.search(
        r"\b(Còpia|doc\.|pàg\.|f\.\s*\d|ff\.\s*\d|segle|núm\.|Cartoral|"
        r"Diplomatari|Col·lecció|Pergamins|Marca hispanica|Histoire générale|"
        r"Patrologiae|Concilia|Recueil)\b",
        stripped,
        re.IGNORECASE,
    ):
        toks = [_strip_accents(w).lower() for w in _tokenize_words(stripped)]
        latin_hits = sum(t in _LATIN_STOPWORDS for t in toks[:12])
        if latin_hits < 2:
            return True

    return False


def _looks_like_latin_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_editorial_line(stripped):
        return False
    if any(p.match(stripped) for p in _SKIP_AFTER_START_PATTERNS):
        return False
    if _CATALAN_COMMENT_RE.search(stripped):
        return False

    normalized = _normalize_for_start_detection(stripped)

    if _LATIN_START_RE.match(normalized):
        return True

    toks = [_strip_accents(w).lower() for w in _tokenize_words(normalized)[:12]]
    if len(toks) < 5:
        return False

    latin_hits = sum(t in _LATIN_STOPWORDS for t in toks)
    catalan_hits = sum(t in _CATALAN_STOPWORDS for t in toks)

    if re.search(r"[àèéíïòóúüç]", stripped.lower()):
        catalan_hits += 1

    return latin_hits >= 3 and latin_hits > catalan_hits + 1


def _should_skip_after_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_editorial_line(stripped):
        return True
    if any(p.match(stripped) for p in _SKIP_AFTER_START_PATTERNS):
        return True
    if _CATALAN_COMMENT_RE.search(stripped):
        return True
    return False


def _extract_latin_text_911(doc_lines: List[str]) -> str:
    """
    Extract only the Latin body of a single Gramoty911 document.

    Strategy
    --------
    1. Ignore document number / date / regest / apparatus.
    2. Find the first line that strongly looks like the Latin incipit/body.
    3. From that point on, keep the body, but skip editorial insertions
       such as 'Lex Visigothorum...' notes.
    """
    start_idx = None

    for i, line in enumerate(doc_lines):
        stripped = line.strip()

        if not stripped:
            continue
        if _DOC_NUM_RE.match(stripped):
            continue
        if _DATE_RE.search(stripped) and len(stripped) < 80:
            # типичная короткая строка даты
            continue

        if _looks_like_latin_start(stripped):
            start_idx = i
            break

    if start_idx is None:
        return ""

    result_lines: List[str] = []
    prev_blank = False

    for line in doc_lines[start_idx:]:
        stripped = line.strip()

        if _should_skip_after_start(stripped):
            continue

        if not stripped:
            if result_lines and not prev_blank:
                result_lines.append("")
            prev_blank = True
            continue

        result_lines.append(stripped)
        prev_blank = False

    return "\n".join(result_lines).strip()


def segment_gramoty_911_unified(source_file, source_name):
    """
    Unified segmenter for Gramoty IX–XI.
    """
    text = read_source_file(source_file)
    raw_pairs = segment_gramoty_911(text, debug=False)

    segments = []
    for doc_num, doc_text in raw_pairs:
        seg_id = f"{source_name}_{doc_num}"
        segments.append((seg_id, doc_text))

    return validate_segments(segments, source_name)


def main() -> None:
    candidates = [
        Path("data/Gramoty911.txt"),
        Path("Gramoty911.txt"),
        Path("/mnt/data/Gramoty911.txt"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_gramoty_911_unified(src, "Gramoty911")
    print(f"Gramoty911: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:180]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:180]}")


if __name__ == "__main__":
    main()
