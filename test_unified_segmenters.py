#!/usr/bin/env python3
"""
Тестирование унифицированных функций сегментеров.
Запускает каждую функцию на тестовых файлах, указанных в их main.

Дополнительно печатает базовую статистику:
- число сегментов
- WARNING о дубликатах id
- min / median / max длины в словах
"""
import sys
import traceback
from pathlib import Path
from statistics import median

# ВАЖНО:
# Берём те же пути, что и в рабочем тесте, чтобы статистика считалась
# на тех же самых файлах и не меняла число сегментов.
TEST_CONFIG = {
    'seg_corpus_juris': (['data/Corpus_Juris_Civilis_v2.txt'], 'CorpusJuris'),
    'seg_evangelium': (['data/Evangelium_v2.txt'], 'Evangelium'),
    'seg_lex_visigothorum': (['data/legesvisigothor00zeumgoog_text.txt'], 'LexVisigoth'),
    'seg_exceptiones_petri': (['data/Exeptionis_Legum_Romanorum_Petri_v3.txt'], 'ExceptPetri'),
    'seg_etymologiae': (['data/Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt'], 'Etymologiae'),
    'seg_costums_tortosa': (['data/ObychaiTortosy1272to1279_v2.txt'], 'ObychaiTortosy1272to1279'),
    'seg_lleida': (['data/ObychaiLleidy12271228_v2.txt'], 'ObychaiLleidy12271228'),
    'seg_miravet': (['data/ObychaiMiraveta1319Fix_v2.txt'], 'ObychaiMiraveta1319Fix'),
    'seg_orty': (['data/ObychaiOrty1296_v2.txt'], 'ObychaiOrty1296'),
    'seg_privileges': (['data/RecognovrentProceres12831284_v2.txt'], 'RecognovrentProceres12831284'),
    'seg_tarregi': (['data/ObychaiTarregi1290E_v2.txt'], 'ObychaiTarregi1290E'),
    'seg_vald_aran': (['data/ObychaiValdArana1313_v2.txt'], 'ObychaiValdArana1313'),
    'seg_zhaime1295': (['data/PragmatikaZhaumeII1295_v2.txt'], 'PragmatikaZhaumeII1295'),
    'seg_zhaime1301': (['data/PragmatikaZhaumeII1301_v2.txt'], 'PragmatikaZhaumeII1301'),
    'seg_gramoty_911': (['data/Gramoty911.txt'], 'Gramoty911'),
    'seg_gramoty_12': (['data/Gramoty12.txt'], 'Gramoty12'),
    'seg_usatges': (['data/Bastardas_Usatges_de_Barcelona_djvu.txt'], 'UsatgesBarcelona'),
}


def find_file(paths):
    """Возвращает первый существующий путь из списка."""
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def compute_basic_stats(segments):
    ids = [seg_id for seg_id, _ in segments]
    word_lengths = [len(seg_text.split()) for _, seg_text in segments]

    seen = set()
    duplicates = []
    for seg_id in ids:
        if seg_id in seen and seg_id not in duplicates:
            duplicates.append(seg_id)
        seen.add(seg_id)

    if word_lengths:
        min_words = min(word_lengths)
        median_words = median(word_lengths)
        max_words = max(word_lengths)
    else:
        min_words = median_words = max_words = 0

    return {
        "count": len(segments),
        "duplicates": duplicates,
        "min_words": min_words,
        "median_words": median_words,
        "max_words": max_words,
    }


def test_unified_function(module_name, possible_paths, source_name):
    """Импортирует unified функцию и запускает её."""
    try:
        module = __import__(f'segmenters.{module_name}', fromlist=[module_name])
        func_name = f'segment_{module_name[4:]}_unified' if module_name.startswith('seg_') else f'{module_name}_unified'
        if not hasattr(module, func_name):
            if module_name == 'seg_gramoty_stable_merged':
                func_name = 'segment_gramoty_unified'
            else:
                candidates = [attr for attr in dir(module) if 'unified' in attr and callable(getattr(module, attr))]
                if not candidates:
                    return False, "Функция unified не найдена", None, None
                func_name = candidates[0]
        func = getattr(module, func_name)

        path = find_file(possible_paths)
        if path is None:
            return False, f"Файл не найден: {possible_paths}", None, None

        segments = func(str(path), source_name)

        if not isinstance(segments, list):
            return False, f"Возвращён не список: {type(segments)}", None, None
        for seg in segments:
            if not (isinstance(seg, tuple) and len(seg) == 2 and isinstance(seg[0], str) and isinstance(seg[1], str)):
                return False, f"Некорректный элемент: {seg}", None, None

        stats = compute_basic_stats(segments)
        return True, "Успех", segments, stats

    except Exception as e:
        return False, f"Ошибка: {e}\n{traceback.format_exc()}", None, None


def main():
    print("=== Тестирование унифицированных функций сегментеров ===\n")
    sys.path.insert(0, '.')

    results = []
    for module_name, (possible_paths, source_name) in TEST_CONFIG.items():
        print(f"Тестируем {module_name}...")
        success, msg, segments, stats = test_unified_function(module_name, possible_paths, source_name)
        status = "✓" if success else "✗"
        print(f"  {status} {msg}")

        if success and segments is not None:
            print(f"    Количество сегментов: {stats['count']}")
            print(
                f"    Длины в словах: min={stats['min_words']}, "
                f"median={stats['median_words']}, max={stats['max_words']}"
            )
            if stats["duplicates"]:
                preview = ", ".join(repr(x) for x in stats["duplicates"][:10])
                more = " ..." if len(stats["duplicates"]) > 10 else ""
                print(f"    WARNING: дубликаты id ({len(stats['duplicates'])}): {preview}{more}")

            if segments:
                print("    Примеры первых трёх сегментов:")
                for i, (seg_id, seg_text) in enumerate(segments[:3]):
                    preview = seg_text[:100] + "..." if len(seg_text) > 100 else seg_text
                    preview = preview.replace('\n', '\\n')
                    print(f"      {i+1}. id={seg_id!r} text={preview!r}")
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
    else:
        print("\nВсе тесты пройдены.")
        sys.exit(0)


if __name__ == '__main__':
    main()
