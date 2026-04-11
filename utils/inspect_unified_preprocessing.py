#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_unified_preprocessing.py

Диагностический просмотр unified-сегментеров + текущего preprocessing.

Что делает:
- берет актуальный список источников из config.CORPORA
- запускает unified-сегментер через src.source_segmenters
- проверяет строгий контракт list[tuple[str, str]]
- выбирает умеренное число сегментов из начала / середины / конца
- для каждого сегмента показывает:
    * исходный текст
    * cleaned
    * mode
    * scores
    * raw_tokens
    * normalized_tokens
    * stemmed_tokens
    * final_tokens
    * token-level changes между стадиями
    * removed after filtering
- дополнительно печатает backward-compatible normalize_latin(...) / tokenize_latin(...)
- печатает краткую статистику по источнику

Цель:
- вручную сравнивать поведение preprocessing на разных типах корпусов
- находить места, где локальная детекция режима ошибается
- быстро дебажить нормализацию, стемминг и фильтрацию
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

# Позволяет запускать скрипт как:
#   python utils/inspect_unified_preprocessing.py
# из корня репозитория, не требуя PYTHONPATH или -m.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CORPORA
from src.preprocessing import (
    LatinLemmatizer,
    preprocess_segment,
)
from src.source_segmenters import get_available_segmenters, segment_source


def validate_segments(segments: object, source_name: str) -> list[tuple[str, str]]:
    if not isinstance(segments, list):
        raise TypeError(f"[{source_name}] Возвращен не list: {type(segments).__name__}")

    validated: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(segments):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"[{source_name}] Элемент #{i}: ожидался tuple(str, str), "
                f"получено {type(item).__name__}: {item!r}"
            )

        seg_id, seg_text = item
        if not isinstance(seg_id, str):
            raise TypeError(
                f"[{source_name}] Элемент #{i}: seg_id должен быть str, "
                f"получено {type(seg_id).__name__}: {seg_id!r}"
            )
        if not isinstance(seg_text, str):
            raise TypeError(
                f"[{source_name}] Элемент #{i}: seg_text должен быть str, "
                f"получено {type(seg_text).__name__}: {seg_text!r}"
            )

        if seg_id in seen_ids:
            raise ValueError(f"[{source_name}] Найден duplicate segment_id: {seg_id!r}")
        seen_ids.add(seg_id)

        validated.append((seg_id, seg_text))

    return validated


def compute_basic_stats(segments: list[tuple[str, str]]) -> dict[str, float | int]:
    word_lengths = [len(seg_text.split()) for _, seg_text in segments]
    char_lengths = [len(seg_text) for _, seg_text in segments]

    if word_lengths:
        min_words = min(word_lengths)
        median_words = median(word_lengths)
        max_words = max(word_lengths)
    else:
        min_words = median_words = max_words = 0

    if char_lengths:
        min_chars = min(char_lengths)
        median_chars = median(char_lengths)
        max_chars = max(char_lengths)
    else:
        min_chars = median_chars = max_chars = 0

    return {
        "count": len(segments),
        "min_words": min_words,
        "median_words": median_words,
        "max_words": max_words,
        "min_chars": min_chars,
        "median_chars": median_chars,
        "max_chars": max_chars,
    }


def sample_indices(n: int, per_zone: int) -> list[int]:
    """
    Выбирает индексы сегментов из начала / середины / конца без дубликатов.
    """
    if n <= 0:
        return []

    if n <= per_zone * 3:
        return list(range(n))

    result: list[int] = []

    result.extend(range(0, min(per_zone, n)))

    mid = n // 2
    half = per_zone // 2
    start_mid = max(0, mid - half)
    end_mid = min(n, start_mid + per_zone)
    result.extend(range(start_mid, end_mid))

    result.extend(range(max(0, n - per_zone), n))

    seen = set()
    ordered = []
    for idx in result:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def shorten(text: str, limit: int = 220) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def format_tokens(tokens: Iterable[str], max_items: int = 30) -> str:
    items = list(tokens)
    if not items:
        return "(empty)"
    if len(items) <= max_items:
        return " ".join(items)
    head = " ".join(items[:max_items])
    return f"{head} ... (+{len(items) - max_items} more)"


def format_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "(none)"
    parts = []
    for key in ("latin", "romance", "ocr_noise"):
        if key in scores:
            parts.append(f"{key}={scores[key]}")
    for key, value in scores.items():
        if key not in {"latin", "romance", "ocr_noise"}:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def token_changes(before: Sequence[str], after: Sequence[str]) -> list[str]:
    """
    Показывает покомпонентные изменения по позициям до min(len(before), len(after)).
    Удобно именно для стадий raw -> normalized и normalized -> stemmed.
    """
    out: list[str] = []
    for b, a in zip(before, after):
        if b != a:
            out.append(f"{b} -> {a}")
    return out


def removed_tokens(before: Sequence[str], after: Sequence[str]) -> list[str]:
    """
    Показывает токены, исчезнувшие между стадиями. Сохраняет кратности в порядке обхода.
    """
    remaining = list(after)
    removed: list[str] = []
    for token in before:
        if token in remaining:
            remaining.remove(token)
        else:
            removed.append(token)
    return removed


def format_change_list(items: Sequence[str], max_items: int = 20) -> str:
    if not items:
        return "(no visible token-level changes)"
    if len(items) <= max_items:
        return "; ".join(items)
    head = "; ".join(items[:max_items])
    return f"{head}; ... (+{len(items) - max_items} more)"



def inspect_segment(
    seg_id: str,
    seg_text: str,
    lemmatizer: LatinLemmatizer,
    min_length: int,
    raw_preview: int,
) -> str:
    debug = preprocess_segment(
        seg_text,
        lemmatizer,
        remove_stopwords=True,
        min_length=min_length,
        return_debug=True,
    )

    raw_to_norm = token_changes(debug["raw_tokens"], debug["normalized_tokens"])
    norm_to_stem = token_changes(debug["normalized_tokens"], debug["stemmed_tokens"])
    removed_after_filter = removed_tokens(debug["stemmed_tokens"], debug["final_tokens"])

    lines = []
    lines.append(f"  id: {seg_id}")
    lines.append(f"  raw:               {shorten(seg_text, raw_preview)}")
    lines.append(f"  cleaned:           {shorten(debug['cleaned'], raw_preview)}")
    lines.append(f"  mode:              {debug['mode']}")
    lines.append(f"  scores:            {format_scores(debug['scores'])}")
    lines.append("")
    lines.append(f"  raw_tokens[{len(debug['raw_tokens'])}]:        {format_tokens(debug['raw_tokens'])}")
    lines.append(f"  normalized[{len(debug['normalized_tokens'])}]:     {format_tokens(debug['normalized_tokens'])}")
    lines.append(f"  stemmed[{len(debug['stemmed_tokens'])}]:        {format_tokens(debug['stemmed_tokens'])}")
    lines.append(f"  final[{len(debug['final_tokens'])}]:          {format_tokens(debug['final_tokens'])}")
    lines.append("")
    lines.append(f"  raw->normalized:    {format_change_list(raw_to_norm)}")
    lines.append(f"  normalized->stemmed:{' ' if norm_to_stem else ' '}{format_change_list(norm_to_stem)}")
    lines.append(f"  removed_after_filter: {format_change_list(removed_after_filter)}")
    return "\n".join(lines)


def run_one_source(
    source_name: str,
    source_path: Path,
    per_zone: int,
    min_length: int,
    raw_preview: int,
) -> tuple[bool, str]:
    try:
        if source_name not in get_available_segmenters():
            available = ", ".join(sorted(get_available_segmenters()))
            raise KeyError(
                f"Для {source_name!r} нет unified-сегментера в src.source_segmenters. "
                f"Available values: {available}"
            )

        if not source_path.exists():
            raise FileNotFoundError(f"Файл не найден: {source_path}")

        segments = validate_segments(segment_source(source_path, source_name), source_name)
        stats = compute_basic_stats(segments)

        lemmatizer = LatinLemmatizer(use_collatinus=False)

        lines = []
        lines.append("=" * 100)
        lines.append(source_name)
        lines.append(f"file: {source_path}")
        lines.append(
            f"segments={stats['count']}, "
            f"min_words={stats['min_words']}, median_words={stats['median_words']}, max_words={stats['max_words']}, "
            f"min_chars={stats['min_chars']}, median_chars={stats['median_chars']}, max_chars={stats['max_chars']}"
        )

        idxs = sample_indices(len(segments), per_zone=per_zone)
        lines.append(f"shown_segments={len(idxs)}")
        lines.append("-" * 100)

        for local_no, idx in enumerate(idxs, 1):
            seg_id, seg_text = segments[idx]
            lines.append(f"[{local_no}/{len(idxs)}] segment_index={idx}")
            lines.append(
                inspect_segment(
                    seg_id=seg_id,
                    seg_text=seg_text,
                    lemmatizer=lemmatizer,
                    min_length=min_length,
                    raw_preview=raw_preview,
                )
            )
            lines.append("-" * 100)

        return True, "\n".join(lines)

    except Exception as exc:
        return False, (
            "=" * 100 + "\n"
            f"{source_name}\n"
            f"file: {source_path}\n"
            f"ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )


def iter_selected_sources(only: set[str] | None = None) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for source_name, meta in CORPORA.items():
        if only is not None and source_name not in only:
            continue
        path = Path(meta["path"])
        items.append((source_name, path))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-zone",
        type=int,
        default=2,
        help="Сколько сегментов показывать из начала, середины и конца каждого источника",
    )
    ap.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="min_length для preprocess_segment",
    )
    ap.add_argument(
        "--raw-preview",
        type=int,
        default=220,
        help="Максимальная длина превью raw/cleaned/normalized текста",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Список source_name для выборочного запуска",
    )
    ap.add_argument(
        "--list-sources",
        action="store_true",
        help="Показать доступные source_name и выйти",
    )
    args = ap.parse_args()


    if args.list_sources:
        print("Available sources:")
        for source_name in sorted(CORPORA):
            path = CORPORA[source_name]["path"]
            has_segmenter = source_name in get_available_segmenters()
            mark = "OK" if has_segmenter else "NO_SEGMENTER"
            print(f"  - {source_name:30} [{mark}]  path={path}")
        return

    only = set(args.only) if args.only else None

    total = 0
    ok_count = 0

    for source_name, source_path in iter_selected_sources(only=only):
        total += 1
        ok, report = run_one_source(
            source_name=source_name,
            source_path=source_path,
            per_zone=args.per_zone,
            min_length=args.min_length,
            raw_preview=args.raw_preview,
        )
        print(report, flush=True)
        if ok:
            ok_count += 1

    print("=" * 100)
    print(f"SUMMARY: total={total}, ok={ok_count}, failed={total - ok_count}")

    if ok_count != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
