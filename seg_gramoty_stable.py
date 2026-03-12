#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сегментация каталонских грамот (Diplomatari) по документам.
Работает с .txt файлами.
Версия 2.0 - распознавание по структурным паттернам.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


def segment_gramoty(text: str, source_name: str = "gramoty", 
                    min_length: int = 200) -> List[Dict[str, str]]:
    """
    Сегментирует грамоты по документам.

    Паттерн распознавания:
      - Строка с датой: "1102, gener, 10" или "812, abril, 2. Місце"
      - Следующая строка может быть локацией
      - Далее идёт текст документа до следующей даты

    Args:
        text: Исходный текст грамот
        source_name: Имя источника
        min_length: Минимальная длина сегмента (символов)

    Returns:
        Список словарей с полями 'id', 'text', 'metadata'
    """
    segments = []

    # Паттерн: год, месяц, день(опц). Место(опц)
    # Примеры: "1102, gener, 10" или "812, abril, 2. Aquisgrà"
    date_line_pattern = re.compile(
        r'^(\d{3,4}),\s*(\w+),?\s*(\d{1,2})?(?:\.\s*(.*))?$',
        re.MULTILINE
    )

    matches = list(date_line_pattern.finditer(text))
    print(f"Найдено {len(matches)} дат-заголовков")

    if not matches:
        print("⚠️  Грамоты не найдены")
        return segments

    doc_counter = 1

    for i, match in enumerate(matches):
        year = int(match.group(1))
        month = match.group(2)
        day = match.group(3) if match.group(3) else ""
        location = match.group(4) if match.group(4) else ""

        # Фильтр по годам (800-1299)
        if year < 800 or year > 1299:
            continue

        # Определяем границы текста документа
        start_pos = match.start()

        # Конец = начало следующей даты или конец файла
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        # Извлекаем полный текст документа
        doc_text = text[start_pos:end_pos].strip()

        # Очищаем текст
        cleaned = clean_text(doc_text)

        # Фильтр по длине
        if len(cleaned) < min_length:
            continue

        # Формируем ID
        if day:
            date_id = f"Y{year}M{month[:3]}D{day}"
        else:
            date_id = f"Y{year}M{month[:3]}"

        segment_id = f"{source_name}_D{doc_counter}_{date_id}"

        metadata = {
            "doc_number": str(doc_counter),
            "year": year,
            "month": month,
            "day": day if day else "",
            "location": location[:50] if location else "",
            "century": (year // 100) * 100
        }

        segments.append({
            "id": segment_id,
            "text": cleaned,
            "metadata": metadata
        })

        doc_counter += 1

    print(f"✓ Сегментировано {len(segments)} документов")

    # Статистика
    if segments:
        centuries = {}
        for seg in segments:
            century = seg['metadata']['century']
            centuries[century] = centuries.get(century, 0) + 1

        print(f"\nРаспределение по векам:")
        for c in sorted(centuries.keys()):
            print(f"  {c}s: {centuries[c]} документов")

    return segments


def clean_text(text: str) -> str:
    """Очистка текста документа."""
    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def segment_gramoty_from_file(filepath: Path, source_name: str = "gramoty",
                               min_length: int = 200) -> List[Dict[str, str]]:
    """
    Сегментирует грамоты из файла.

    Args:
        filepath: Путь к .txt файлу
        source_name: Имя источника
        min_length: Минимальная длина документа

    Returns:
        Список сегментов
    """
    print(f"Загрузка: {filepath}")

    text = filepath.read_text(encoding='utf-8')
    print(f"Загружено {len(text):,} символов")

    return segment_gramoty(text, source_name=source_name, min_length=min_length)


def main():
    """Отладка сегментера."""

    # Проверяем оба файла грамот
    test_files = [
        (Path("data/Gramoty911.txt"), "gramoty911"),
        (Path("data/Gramoty12.txt"), "gramoty12"),
    ]

    found_files = []
    for fpath, fname in test_files:
        if fpath.exists():
            found_files.append((fpath, fname))

    # Убираем дубликаты
    unique_sources = {}
    for fpath, fname in found_files:
        if fname not in unique_sources:
            unique_sources[fname] = fpath

    if not unique_sources:
        print("❌ Файлы грамот не найдены")
        print("Ожидаемые файлы:")
        print("  - Gramoty911.txt (IX-XI вв.)")
        print("  - Gramoty12.txt (XII в.)")
        return

    print("="*70)
    print("СЕГМЕНТАЦИЯ ГРАМОТ")
    print("="*70)

    all_segments = []

    for source_name, filepath in sorted(unique_sources.items()):
        print(f"\n{'─'*70}")
        print(f"Файл: {filepath}")
        print(f"{'─'*70}")

        try:
            segments = segment_gramoty_from_file(
                filepath,
                source_name=source_name,
                min_length=200
            )

            all_segments.extend(segments)

            # Показываем примеры
            print(f"\nПримеры (первые 5):")
            for i, seg in enumerate(segments[:5], 1):
                meta = seg['metadata']
                date_str = f"{meta['year']}, {meta['month']}"
                if meta['day']:
                    date_str += f", {meta['day']}"

                print(f"\n[{i}] {seg['id']}")
                print(f"    Дата: {date_str}")
                if meta['location']:
                    print(f"    Место: {meta['location']}")
                print(f"    Длина: {len(seg['text'])} символов")
                print(f"    Текст: {seg['text'][:80]}...")

            if len(segments) > 5:
                print(f"\n... и ещё {len(segments) - 5} документов")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    # Итоговая статистика
    print(f"\n{'='*70}")
    print(f"ИТОГО")
    print(f"{'='*70}")
    print(f"Всего сегментов: {len(all_segments)}")

    if all_segments:
        # По векам
        centuries = {}
        for seg in all_segments:
            c = seg['metadata']['century']
            centuries[c] = centuries.get(c, 0) + 1

        print(f"\nРаспределение по векам:")
        for c in sorted(centuries.keys()):
            print(f"  {c}s: {centuries[c]}")

        # Диапазон лет
        years = [seg['metadata']['year'] for seg in all_segments]
        print(f"\nДиапазон лет: {min(years)} - {max(years)}")

    print(f"\n✓ Сегментация завершена")


if __name__ == "__main__":
    main()
