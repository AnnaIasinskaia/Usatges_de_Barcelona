#!/usr/bin/env python3
"""
Тестирование унифицированных функций сегментеров.
Запускает каждую функцию на тестовых файлах, указанных в их main.
"""
import sys
import traceback
from pathlib import Path

# Словарь: имя модуля -> (список возможных путей, source_name)
# закомменитировал успешно пройденные тесты
TEST_CONFIG = {
    'seg_corpus_juris': (['data/Corpus Juris Civilis.docx'], 'CorpusJuris'),
    'seg_evangelium': (['data/Evangelium.docx'], 'Evangelium'),
    'seg_lex_visigothorum': (['data/Lex visigothorum.docx'], 'LexVisigoth'),
    'seg_exceptiones_petri': (['data/Exeptionis Legum Romanorum Petri.docx'], 'ExceptPetri'),
    'seg_etymologiae': (['data/Isidori Hispalensis Episcopi Etymologiarum.docx'], 'Etymologiae'),
    'seg_costums_tortosa': (['data/ObychaiTortosy1272to1279_v2.txt'], 'ObychaiTortosy1272to1279'),
    'seg_lleida': (['data/ObychaiLleidy12271228_v2.txt'], 'ObychaiLleidy12271228'),
    'seg_miravet': (['data/ObychaiMiraveta1319Fix.txt'], 'ObychaiMiraveta1319Fix'),
    'seg_orty': (['data/ObychaiOrty1296.txt'], 'ObychaiOrty1296'),
    'seg_privileges': (['data/RecognovrentProceres12831284_v2.txt'], 'RecognovrentProceres12831284'),
    'seg_tarregi': (['data/ObychaiTarregi1290E.txt'], 'ObychaiTarregi1290E'),
    'seg_vald_aran': (['data/ObychaiValdArana1313_v2.txt'], 'ObychaiValdArana1313'),
    'seg_zhaime1295': (['data/PragmatikaZhaumeII1295_v2.txt'], 'PragmatikaZhaumeII1295'),
    'seg_zhaime1301': (['data/PragmatikaZhaumeII1301_v2.txt'], 'PragmatikaZhaumeII1301'),
    'seg_gramoty_911': (['data/Gramoty911.txt'], 'Gramoty911'),
    'seg_gramoty_v2': (['data/Gramoty12.txt'], 'Gramoty12'),
    'seg_usatges': (['data/Bastardas Usatges de Barcelona_djvu.txt'], 'UsatgesBarcelona'),
}

def find_file(paths):
    """Возвращает первый существующий путь из списка."""
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None

def test_unified_function(module_name, possible_paths, source_name):
    """Импортирует unified функцию и запускает её."""
    try:
        # Динамический импорт
        module = __import__(f'segmenters.{module_name}', fromlist=[module_name])
        func_name = f'segment_{module_name[4:]}_unified' if module_name.startswith('seg_') else f'{module_name}_unified'
        if not hasattr(module, func_name):
            # Попробуем найти функцию с другим именем
            # Для seg_gramoty_stable_merged функция называется segment_gramoty_unified
            if module_name == 'seg_gramoty_stable_merged':
                func_name = 'segment_gramoty_unified'
            else:
                # Поиск любой функции, содержащей 'unified'
                candidates = [attr for attr in dir(module) if 'unified' in attr and callable(getattr(module, attr))]
                if not candidates:
                    return False, f"Функция unified не найдена в {module_name}", None
                func_name = candidates[0]
        func = getattr(module, func_name)
        
        # Находим файл
        path = find_file(possible_paths)
        if path is None:
            return False, f"Файл не найден: {possible_paths}", None
        
        # Вызов функции с параметрами по умолчанию
        segments = func(str(path), source_name)
        
        # Проверка формата
        if not isinstance(segments, list):
            return False, f"Возвращён не список: {type(segments)}", None
        for seg in segments:
            if not (isinstance(seg, tuple) and len(seg) == 2 and isinstance(seg[0], str) and isinstance(seg[1], str)):
                return False, f"Некорректный элемент: {seg}", None
        
        return True, f"Успех: {len(segments)} сегментов", segments
    
    except Exception as e:
        return False, f"Ошибка: {e}\n{traceback.format_exc()}", None

def main():
    print("=== Тестирование унифицированных функций сегментеров ===\n")
    sys.path.insert(0, '.')
    
    results = []
    for module_name, (possible_paths, source_name) in TEST_CONFIG.items():
        print(f"Тестируем {module_name}...")
        success, msg, segments = test_unified_function(module_name, possible_paths, source_name)
        status = "✓" if success else "✗"
        print(f"  {status} {msg}")
        if success and segments is not None:
            if segments:
                print("    Примеры первых трёх сегментов:")
                for i, (seg_id, seg_text) in enumerate(segments[:3]):
                    # Обрежем текст для удобства чтения
                    preview = seg_text[:100] + "..." if len(seg_text) > 100 else seg_text
                    # Заменим переносы строк на \n для компактности
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