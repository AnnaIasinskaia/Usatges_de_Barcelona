
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_unified_preprocessing.py

Диагностический просмотр unified-сегментеров + текущего preprocessing.

Что делает:
- запускает каждый unified-сегментер
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
- дополнительно печатает backward-compatible normalize_latin(...) / tokenize_latin(...)
- печатает краткую статистику по источнику

Цель:
- вручную сравнивать поведение preprocessing на разных типах корпусов
- находить места, где локальная детекция режима ошибается
- быстро дебажить нормализацию, стемминг и фильтрацию
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from statistics import median
from typing import Iterable

from preprocessing import (
    LatinLemmatizer,
    normalize_latin,
    tokenize_latin,
    preprocess_segment,
)


TEST_CONFIG = {
    "seg_corpus_juris": (["data/Corpus_Juris_Civilis_v2.txt"], "CorpusJuris"),
    "seg_evangelium": (["data/Evangelium_v2.txt"], "Evangelium"),
    "seg_lex_visigothorum": (["data/legesvisigothor00zeumgoog_text.txt"], "LexVisigoth"),
    "seg_exceptiones_petri": (["data/Exeptionis_Legum_Romanorum_Petri_v3.txt"], "ExceptPetri"),
    "seg_etymologiae": (["data/Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt"], "Etymologiae"),
    "seg_costums_tortosa": (["data/ObychaiTortosy1272to1279_v2.txt"], "ObychaiTortosy1272to1279"),
    "seg_lleida": (["data/ObychaiLleidy12271228_v2.txt"], "ObychaiLleidy12271228"),
    "seg_miravet": (["data/ObychaiMiraveta1319Fix_v2.txt"], "ObychaiMiraveta1319Fix"),
    "seg_orty": (["data/ObychaiOrty1296_v2.txt"], "ObychaiOrty1296"),
    "seg_privileges": (["data/RecognovrentProceres12831284_v2.txt"], "RecognovrentProceres12831284"),
    "seg_tarregi": (["data/ObychaiTarregi1290E_v2.txt"], "ObychaiTarregi1290E"),
    "seg_vald_aran": (["data/ObychaiValdArana1313_v2.txt"], "ObychaiValdArana1313"),
    "seg_zhaime1295": (["data/PragmatikaZhaumeII1295_v2.txt"], "PragmatikaZhaumeII1295"),
    "seg_zhaime1301": (["data/PragmatikaZhaumeII1301_v2.txt"], "PragmatikaZhaumeII1301"),
    "seg_gramoty_911": (["data/Gramoty911.txt"], "Gramoty911"),
    "seg_gramoty_12": (["data/Gramoty12.txt"], "Gramoty12"),
    "seg_usatges": (["data/Bastardas_Usatges_de_Barcelona_djvu.txt"], "UsatgesBarcelona"),
}


def find_file(paths: list[str]) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def expected_unified_name(module_name: str) -> str:
    if not module_name.startswith("seg_"):
        raise ValueError(f"Ожидался модуль seg_*, получено: {module_name}")
    return f"segment_{module_name[4:]}_unified"


def validate_segments(segments: object, source_name: str) -> list[tuple[str, str]]:
    if not isinstance(segments, list):
        raise TypeError(f"[{source_name}] Возвращён не list: {type(segments).__name__}")

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

    if word_lengths:
        min_words = min(word_lengths)
        median_words = median(word_lengths)
        max_words = max(word_lengths)
    else:
        min_words = median_words = max_words = 0

    return {
        "count": len(segments),
        "min_words": min_words,
        "median_words": median_words,
        "max_words": max_words,
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
    if len(items) <= max_items:
        return " ".join(items)
    head = " ".join(items[:max_items])
    return f"{head} ... (+{len(items) - max_items} more)"


def format_scores(scores: dict[str, float]) -> str:
    parts = []
    for key in ("latin", "romance", "ocr_noise"):
        if key in scores:
            parts.append(f"{key}={scores[key]}")
    for key, value in scores.items():
        if key not in {"latin", "romance", "ocr_noise"}:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def inspect_segment(
    seg_id: str,
    seg_text: str,
    lemmatizer: LatinLemmatizer,
    min_length: int,
    raw_preview: int,
) -> str:
    normalized_compat = normalize_latin(seg_text)
    tokens_compat = tokenize_latin(normalized_compat)
    lemmas_compat = lemmatizer.lemmatize_tokens(tokens_compat)

    debug = preprocess_segment(
        seg_text,
        lemmatizer,
        remove_stopwords=True,
        min_length=min_length,
        return_debug=True,
    )

    lines = []
    lines.append(f"  id: {seg_id}")
    lines.append(f"  raw:               {shorten(seg_text, raw_preview)}")
    lines.append(f"  cleaned:           {shorten(debug['cleaned'], raw_preview)}")
    lines.append(f"  mode:              {debug['mode']}")
    lines.append(f"  scores:            {format_scores(debug['scores'])}")
    lines.append("")
    lines.append(f"  compat_normalized: {shorten(normalized_compat, raw_preview)}")
    lines.append(f"  compat_tokens[{len(tokens_compat)}]:  {format_tokens(tokens_compat)}")
    lines.append(f"  compat_lemmas[{len(lemmas_compat)}]:  {format_tokens(lemmas_compat)}")
    lines.append("")
    lines.append(f"  raw_tokens[{len(debug['raw_tokens'])}]:         {format_tokens(debug['raw_tokens'])}")
    lines.append(f"  normalized[{len(debug['normalized_tokens'])}]:      {format_tokens(debug['normalized_tokens'])}")
    lines.append(f"  stemmed[{len(debug['stemmed_tokens'])}]:         {format_tokens(debug['stemmed_tokens'])}")
    lines.append(f"  final[{len(debug['final_tokens'])}]:           {format_tokens(debug['final_tokens'])}")
    return "\n".join(lines)


def run_one_source(
    module_name: str,
    possible_paths: list[str],
    source_name: str,
    per_zone: int,
    min_length: int,
    raw_preview: int,
) -> tuple[bool, str]:
    try:
        module = importlib.import_module(f"segmenters.{module_name}")
        func_name = expected_unified_name(module_name)

        if not hasattr(module, func_name):
            raise AttributeError(
                f"В модуле segmenters.{module_name} не найдена функция {func_name}"
            )

        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(f"{func_name} существует, но не является callable")

        path = find_file(possible_paths)
        if path is None:
            raise FileNotFoundError(f"Файл не найден: {possible_paths}")

        segments = validate_segments(func(str(path), source_name), source_name)
        stats = compute_basic_stats(segments)

        lemmatizer = LatinLemmatizer(use_collatinus=False)

        lines = []
        lines.append("=" * 100)
        lines.append(f"{source_name}  ({module_name})")
        lines.append(f"file: {path}")
        lines.append(
            f"segments={stats['count']}, "
            f"min_words={stats['min_words']}, "
            f"median_words={stats['median_words']}, "
            f"max_words={stats['max_words']}"
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
            f"{source_name}  ({module_name})\n"
            f"ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )


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
    args = ap.parse_args()

    sys.path.insert(0, ".")

    only = set(args.only) if args.only else None

    total = 0
    ok_count = 0

    for module_name, (possible_paths, source_name) in TEST_CONFIG.items():
        if only is not None and source_name not in only:
            continue

        total += 1
        ok, report = run_one_source(
            module_name=module_name,
            possible_paths=possible_paths,
            source_name=source_name,
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
