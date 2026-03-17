#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation script for:
38. PRIVILEGI DIT GENERALMENT DE LA QUERIMONIA (COSTUMBRES DEL VALLE DE ARÁN). 1313

Основан на seg_zhaime1301.py:
- один документ (38.)
- испанский редакционный заголовок + аннотация
- латинский текст с артикулами [I]...[XXIII]
- хвостовая формула ("In cuius rei testimonium...", "Data Ilerde...") — отбрасывается
"""

import re
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# ── Patterns ──────────────────────────────────────────────────────────────────

# [I] ...  (с возможными пробелами и \u3000 перед скобкой)
ARTICLE_PATTERN = re.compile(r'^[\s\u3000]*\[([IVX]+)\](.*)', re.DOTALL)

# "1 Publicado..." и подобные редакторские сноски
FOOTNOTE_PATTERN = re.compile(r'^\s*\d+\s+[A-ZÁÉÍÓÚ]')

# Заголовок документа: "38.PRIVILEGI DIT GENERALMENT DE LA QUERIMONIA ..."
DOC_HEADER_PATTERN = re.compile(r'^(\d+)\.\s*PRIVILEGI DIT GENERALMENT DE LA QUERIMONIA', re.IGNORECASE)

# Маркеры конца хартии (чтобы отрезать нотариальный хвост)
CLOSING_MARKERS = [
    "In cuius rei testimonium",
    "Data Ilerde",
]

# Всё, что идёт после привилегии, нас не интересует
AFTER_DOC_MARKERS = [
    "c) Compilaciones de derecho feudal",
    "39.LAS COSTUMAS DE CATHALUNYA",
]

# Редакционные маркеры, которые можно пропускать
SKIP_MARKERS = [
    "Original, perdido",
    "Original, en el Archivo",
    "***",
    "Reseñado por",
    "Publicado por",
    "Copia en el Llibre",
    "Copia de la misma fecha que el original",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def roman_to_int(roman: str) -> int:
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total, prev = 0, 0
    for ch in reversed(roman.upper()):
        v = vals.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = v
    return total


def clean_text(text: str) -> str:
    # убрать переносы с дефисом и нормализовать пробелы
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Core segmenter ────────────────────────────────────────────────────────────

def segment_vald_aran_text(
    text: str,
    debug: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Сегментирует Privilegi de la Querimònia (Val d'Aran, 1313) на статьи.

    Возвращает:
      List[(doc_id, article_id, article_text)],
      где doc_id == "38",
      article_id "0" — преамбула (до [I]).
    """
    lines = text.split('\n')

    DOC_ID = "38"  # по умолчанию; при наличии заголовка обновим из него

    articles: List[Tuple[str, str, str]] = []
    current_roman: str | None = None
    current_lines: List[str] = []
    in_articles: bool = False
    preamble_lines: List[str] = []
    in_target_doc: bool = False  # начали ли саму Querimònia

    def flush():
        nonlocal current_roman, current_lines
        if current_roman is not None and current_lines:
            t = clean_text(" ".join(current_lines))
            if t:
                articles.append((DOC_ID, current_roman, t))
        current_roman = None
        current_lines = []

    for i, raw_line in enumerate(lines):
        # сохраняем оригинальную строку для match'ей по началу
        line_raw = raw_line.rstrip('\r\n')
        # нормализованный вариант для остальной логики
        line = line_raw.strip().replace('\u3000', ' ')

        if not line:
            continue

        # Если наткнулись на блок, который идёт уже после нашей привилегии — выходим.
        if any(line_raw.startswith(m) for m in AFTER_DOC_MARKERS):
            if debug:
                print(f" After-doc marker at line {i}: {line_raw[:60]}")
            flush()
            break

        # Если ещё не вошли в целевой документ, ждём его заголовка
        if not in_target_doc:
            m = DOC_HEADER_PATTERN.match(line_raw)
            if m:
                DOC_ID = m.group(1)
                in_target_doc = True
                if debug:
                    print(f" Doc header for {DOC_ID} at line {i}: {line_raw[:60]}")
                # заголовок не включаем в преамбулу
                continue
            # до целевого заголовка ничего не собираем
            continue

        # После начала целевого документа:

        # Конечная формула — прекращаем извлечение статей
        if any(line_raw.startswith(m) for m in CLOSING_MARKERS):
            flush()
            if debug:
                print(f" Closing formula at line {i}: {line_raw[:60]}")
            break

        # Пропускаем редакционные маркеры
        if any(m in line_raw for m in SKIP_MARKERS):
            continue

        # Сноски "1 Publicado..." и т.п.
        if FOOTNOTE_PATTERN.match(line):
            if debug:
                print(f" Footnote at {i}: {line_raw[:60]}")
            continue

        # Маркер статьи [ROMAN]
        art_match = ARTICLE_PATTERN.match(line_raw.replace('\u3000', ' '))
        if art_match:
            flush()
            in_articles = True
            current_roman = art_match.group(1)
            rest = art_match.group(2).strip()
            current_lines = [rest] if rest else []
            if debug:
                print(f" Article [{current_roman}] at {i}: {rest[:50]}")
            continue

        # Накопление строк
        if in_articles:
            current_lines.append(line_raw.strip())
        else:
            # Всё между заголовком и [I] идёт в преамбулу
            preamble_lines.append(line_raw.strip())

    # последний артикул
    flush()

    # добавляем преамбулу как статью 0
    if preamble_lines:
        preamble_text = clean_text(" ".join(preamble_lines))
        if preamble_text:
            articles.insert(0, (DOC_ID, "0", preamble_text))

    return articles


# ── Analysis + save ───────────────────────────────────────────────────────────

def analyze_and_save_vald_aran(text: str, output_file: str) -> List[Tuple[str, str, str]]:
    """
    Анализ сегментации, печать статистики и сохранение в файл.
    """
    print("=" * 80)
    print("PRIVILEGI DE LA QUERIMÒNIA (VAL D'ARAN, 1313) – ARTICLE SEGMENTATION")
    print("=" * 80)

    articles = segment_vald_aran_text(text, debug=True)

    docs: dict = {}
    for doc_id, art_id, art_text in articles:
        docs.setdefault(doc_id, []).append((art_id, art_text))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(docs)}")
    for doc_id in sorted(docs):
        print(f" Document {doc_id}: {len(docs[doc_id])} articles")
    print(f"\nTotal articles: {len(articles)}")

    # Проверка дубликатов
    for doc_id, art_list in docs.items():
        ids = [aid for aid, _ in art_list]
        dupes = [(n, c) for n, c in Counter(ids).items() if c > 1]
        if dupes:
            print(f"\nWARNING: Duplicates in document {doc_id}:")
            for art_id, count in dupes:
                print(f" Article {art_id} × {count}")
        else:
            print(f"\nDocument {doc_id}: No duplicates ✓")

    print("=" * 80)

    # Сохранение результатов
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PRIVILEGI DE LA QUERIMÒNIA (VAL D'ARAN, 1313) – SEGMENTED ARTICLES\n")
        f.write(f"Total documents: {len(docs)}\n")
        f.write(f"Total articles: {len(articles)}\n")
        f.write("=" * 80 + "\n\n")

        for doc_id in sorted(docs):
            sorted_arts = sorted(
                docs[doc_id],
                key=lambda x: roman_to_int(x[0]) if x[0] != "0" else 0
            )

            f.write(f"\n{'=' * 80}\n")
            f.write(f"DOCUMENT {doc_id}\n")
            f.write(f"Articles: {len(sorted_arts)}\n")
            f.write(f"{'=' * 80}\n\n")

            for art_id, art_text in sorted_arts:
                f.write(f"{'=' * 80}\n")
                f.write(f"ARTICLE {doc_id}.{art_id}\n")
                f.write(f"{'=' * 80}\n")
                f.write(art_text)
                f.write("\n\n")

    print(f"\nResults saved to {output_file}")
    print("=" * 80)

    return articles


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    file_path = Path("data/ObychaiValdArana1313_v2.txt")
    if file_path.exists():
        print(f"Processing Val d'Aran customs from {file_path}...")
        text = file_path.read_text(encoding='utf-8')
        analyze_and_save_vald_aran(text, output_file="vald_aran_querimonia_segmented.txt")
    else:
        print(f"Error: {file_path} not found")


def segment_vald_aran_unified(
    source_file, source_name, min_words=10, max_words=150
):
    """
    Унифицированная сегментация Val d'Aran.
    Читает файл, применяет ограничения по словам.
    """
    from .seg_common import read_source_file, apply_word_limits, validate_segments

    text = read_source_file(source_file)
    # Вызов старого сегментера с debug=False
    raw_triples = segment_vald_aran_text(text, debug=False)

    # Преобразуем тройки в пары (segment_id, text)
    raw_segments = []
    for doc_id, art_id, art_text in raw_triples:
        seg_id = f"{source_name}_{doc_id}_{art_id}"
        raw_segments.append((seg_id, art_text))

    # Применяем ограничения по словам
    filtered = apply_word_limits(raw_segments, min_words, max_words)

    # Валидация
    return validate_segments(filtered, source_name)
if __name__ == '__main__':
    main()
