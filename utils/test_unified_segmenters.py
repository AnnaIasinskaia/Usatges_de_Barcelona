#!/usr/bin/env python3
"""
Строгое тестирование unified-сегментеров.

Инварианты:
- модуль обязан экспортировать ожидаемую unified-функцию
- unified-функция обязана принимать (source_file, source_name)
- возвращаемое значение обязано быть list[tuple[str, str]]
- segment_id обязаны быть уникальны в пределах источника
- TEST_CONFIG, config.CORPORA и source_segmenters должны быть согласованы
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from statistics import median


TEST_CONFIG = {
    "seg_corpus_juris": (["data/Corpus_Juris_Civilis_v2.txt"], "CorpusJuris"),
    "seg_evangelium": (["data/Evangelium_v2.txt"], "Evangelium"),
    "seg_lex_visigothorum": (["data/Lex_visigothorum_lat-1.txt"], "LexVisigothorum"),
    "seg_exceptiones_petri": (["data/Exeptionis_Legum_Romanorum_Petri_v4.txt"], "ExceptPetri"),
    "seg_etymologiae": (["data/Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt"], "Etymologiae"),
    "seg_costums_tortosa": (["data/ObychaiTortosy1272to1279_v2.txt"], "CostumsDeTortosa"),
    "seg_lleida": (["data/ObychaiLleidy12271228_v2.txt"], "CostumsDeLleida"),
    "seg_miravet": (["data/ObychaiMiraveta1319Fix_v2.txt"], "ConstitucionesBaiulieMirabeti"),
    "seg_orty": (["data/ObychaiOrty1296_v2.txt"], "CostumsDeOrta"),
    "seg_privileges": (["data/RecognovrentProceres12831284_v2.txt"], "RecognovrentProceres"),
    "seg_tarregi": (["data/ObychaiTarregi1290E_v2.txt"], "CostumresDeTarrega"),
    "seg_vald_aran": (["data/ObychaiValdArana1313_v2.txt"], "CostumsDeValdAran"),
    "seg_zhaime1295": (["data/PragmatikaZhaumeII1295_v2.txt"], "PragmaticaJaimeII1295"),
    "seg_zhaime1301": (["data/PragmatikaZhaumeII1301_v2.txt"], "PragmaticaJaimeII1301"),
    "seg_gramoty_911": (["data/Gramoty911.txt"], "Acta911"),
    "seg_gramoty_12": (["data/Gramoty12.txt"], "Acta12"),
    "seg_usatges": (["data/Bastardas_Usatges_de_Barcelona_djvu.txt"], "UsatgesBarcelona"),
    "seg_perpignan":(["data/Customs_of_Perpignan_v2.txt"],"CostumsDePerpinya"),
}


def find_file(paths: list[str]) -> Path | None:
    """Возвращает первый существующий путь из списка."""
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def expected_unified_name(module_name: str) -> str:
    """Строит строгое имя unified-функции для модуля segmenters.seg_*."""
    if not module_name.startswith("seg_"):
        raise ValueError(f"Ожидался модуль вида seg_*, получено: {module_name}")
    return f"segment_{module_name[4:]}_unified"


def validate_segments(segments: object, source_name: str) -> list[tuple[str, str]]:
    """
    Проверяет строгий контракт list[tuple[str, str]] и уникальность id.
    Возвращает тот же список в типизированном виде.
    """
    if not isinstance(segments, list):
        raise TypeError(f"[{source_name}] Возвращён не list: {type(segments).__name__}")

    validated: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

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
                f"[{source_name}] Элемент #{i} ({seg_id!r}): seg_text должен быть str, "
                f"получено {type(seg_text).__name__}: {seg_text!r}"
            )

        if seg_id in seen_ids:
            duplicate_ids.append(seg_id)
        else:
            seen_ids.add(seg_id)

        validated.append((seg_id, seg_text))

    if duplicate_ids:
        preview = ", ".join(repr(x) for x in duplicate_ids[:10])
        more = " ..." if len(duplicate_ids) > 10 else ""
        raise ValueError(
            f"[{source_name}] Найдены дубликаты segment_id ({len(duplicate_ids)}): {preview}{more}"
        )

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


def validate_registry_consistency() -> tuple[bool, str]:
    """
    Проверяет согласованность трёх реестров:
    - TEST_CONFIG
    - config.CORPORA
    - source_segmenters.get_available_segmenters()
    """
    try:
        cfg_mod = importlib.import_module("config")
        corpora = dict(getattr(cfg_mod, "CORPORA"))
    except Exception as exc:
        return False, f"Не удалось импортировать config.CORPORA: {exc}\n{traceback.format_exc()}"

    try:
        seg_mod = importlib.import_module("src.source_segmenters")
        segmenters = dict(seg_mod.get_available_segmenters())
    except Exception as exc:
        return False, f"Не удалось импортировать source_segmenters: {exc}\n{traceback.format_exc()}"

    test_sources = {source_name for _, source_name in TEST_CONFIG.values()}
    corpora_sources = set(corpora.keys())
    segmenter_sources = set(segmenters.keys())

    errors: list[str] = []

    only_in_test = sorted(test_sources - corpora_sources)
    if only_in_test:
        errors.append(
            "Источник есть в TEST_CONFIG, но отсутствует в config.CORPORA: "
            + ", ".join(only_in_test)
        )

    only_in_corpora = sorted(corpora_sources - test_sources)
    if only_in_corpora:
        errors.append(
            "Источник есть в config.CORPORA, но отсутствует в TEST_CONFIG: "
            + ", ".join(only_in_corpora)
        )

    only_in_segmenters_vs_corpora = sorted(segmenter_sources - corpora_sources)
    if only_in_segmenters_vs_corpora:
        errors.append(
            "Источник есть в source_segmenters, но отсутствует в config.CORPORA: "
            + ", ".join(only_in_segmenters_vs_corpora)
        )

    only_in_corpora_vs_segmenters = sorted(corpora_sources - segmenter_sources)
    if only_in_corpora_vs_segmenters:
        errors.append(
            "Источник есть в config.CORPORA, но отсутствует в source_segmenters: "
            + ", ".join(only_in_corpora_vs_segmenters)
        )

    only_in_test_vs_segmenters = sorted(test_sources - segmenter_sources)
    if only_in_test_vs_segmenters:
        errors.append(
            "Источник есть в TEST_CONFIG, но отсутствует в source_segmenters: "
            + ", ".join(only_in_test_vs_segmenters)
        )

    if errors:
        return False, "\n".join(errors)

    return True, (
        "Реестры согласованы: "
        f"TEST_CONFIG={len(test_sources)}, "
        f"CORPORA={len(corpora_sources)}, "
        f"segmenters={len(segmenter_sources)}"
    )


def test_unified_function(
    module_name: str,
    possible_paths: list[str],
    source_name: str,
) -> tuple[bool, str, list[tuple[str, str]] | None, dict[str, float | int] | None]:
    """Импортирует строго ожидаемую unified-функцию и запускает её."""
    try:
        module = importlib.import_module(f"segmenters.{module_name}")
        func_name = expected_unified_name(module_name)

        if not hasattr(module, func_name):
            raise AttributeError(
                f"[{source_name}] В модуле segmenters.{module_name} "
                f"не найдена обязательная функция {func_name}"
            )

        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(
                f"[{source_name}] Атрибут {func_name} существует, но не является вызываемым"
            )

        path = find_file(possible_paths)
        if path is None:
            raise FileNotFoundError(f"[{source_name}] Файл не найден: {possible_paths}")

        segments = func(str(path), source_name)
        validated = validate_segments(segments, source_name)
        stats = compute_basic_stats(validated)
        return True, "Успех", validated, stats

    except Exception as exc:
        return False, f"Ошибка: {exc}\n{traceback.format_exc()}", None, None


def main() -> None:
    print("=== Строгое тестирование unified-сегментеров ===\n")
    sys.path.insert(0, ".")

    print("Проверяем согласованность TEST_CONFIG ↔ config ↔ source_segmenters...")
    registry_ok, registry_msg = validate_registry_consistency()
    registry_status = "✓" if registry_ok else "✗"
    print(f"  {registry_status} {registry_msg}")

    if not registry_ok:
        print("\nОстановка: сначала исправьте рассогласование реестров.")
        sys.exit(1)

    print()

    results: list[tuple[str, bool, str]] = []

    for module_name, (possible_paths, source_name) in TEST_CONFIG.items():
        print(f"Тестируем {module_name}...")
        success, msg, segments, stats = test_unified_function(module_name, possible_paths, source_name)
        status = "✓" if success else "✗"
        print(f"  {status} {msg}")

        if success and segments is not None and stats is not None:
            print(f"    Количество сегментов: {stats['count']}")
            print(
                f"    Длины в словах: min={stats['min_words']}, "
                f"median={stats['median_words']}, max={stats['max_words']}"
            )

            if segments:
                print("    Примеры первых трёх сегментов:")
                for i, (seg_id, seg_text) in enumerate(segments[:3], 1):
                    preview = seg_text[:100] + "..." if len(seg_text) > 100 else seg_text
                    preview = preview.replace("\n", "\\n")
                    print(f"      {i}. id={seg_id!r} text={preview!r}")
            else:
                print("    Нет сегментов")

        results.append((module_name, success, msg))

    print("\n=== Итог ===")
    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    failed = total - passed
    print(f"Всего: {total}, Успешно: {passed}, Ошибок: {failed}")

    if failed > 0:
        print("\nДетали ошибок:")
        for module_name, success, msg in results:
            if not success:
                print(f"- {module_name}: {msg}")
        sys.exit(1)

    print("\nВсе тесты пройдены.")
    sys.exit(0)


if __name__ == "__main__":
    main()
