#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер .docx → .txt для грамот и обычаев.
Сохраняет структуру с жирными номерами документов.
Пропускает уже сконвертированные файлы.
"""

from pathlib import Path
from docx import Document
import re


def convert_docx_to_txt(docx_path: Path, txt_path: Path = None, 
                        preserve_bold_markers: bool = True,
                        skip_existing: bool = True):
    """
    Конвертирует .docx в .txt с опциональным сохранением маркеров.

    Args:
        docx_path: Путь к .docx файлу
        txt_path: Путь для сохранения .txt (если None - рядом с исходным)
        preserve_bold_markers: Если True, жирные номера выделяются маркером
        skip_existing: Если True, пропускает уже существующие .txt файлы

    Returns:
        Path к созданному .txt файлу или None при ошибке
    """
    if txt_path is None:
        txt_path = docx_path.with_suffix('.txt')

    # Проверка существования .txt
    if skip_existing and txt_path.exists():
        size = txt_path.stat().st_size
        print(f"⏭️  Пропуск: {docx_path.name} (уже есть {txt_path.name}, {size:,} байт)")
        return txt_path

    print(f"🔄 Конвертация: {docx_path.name}", end=" ... ")

    try:
        doc = Document(str(docx_path))

        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()

            if not text:
                lines.append('')  # Пустая строка
                continue

            # Проверяем, является ли параграф жирным номером документа
            is_bold = any(run.bold for run in para.runs) if para.runs else False
            is_doc_number = re.match(r'^\s*\d{1,4}\s*$', text)

            if preserve_bold_markers and is_bold and is_doc_number:
                # Выделяем номер документа специальным маркером
                lines.append(f"### DOC {text} ###")
            else:
                lines.append(text)

        # Сохраняем в текстовый файл
        txt_path.write_text('\n'.join(lines), encoding='utf-8')

        size = txt_path.stat().st_size
        print(f"✅ {txt_path.name} ({size:,} байт)")

        return txt_path

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return None


def batch_convert_directory(directory: Path, pattern: str = "*.docx",
                            output_dir: Path = None,
                            skip_existing: bool = True):
    """
    Конвертирует все .docx файлы в директории.

    Args:
        directory: Директория с .docx файлами
        pattern: Маска файлов (по умолчанию *.docx)
        output_dir: Директория для .txt файлов (если None - та же)
        skip_existing: Пропускать уже сконвертированные файлы
    """
    if output_dir is None:
        output_dir = directory

    output_dir.mkdir(parents=True, exist_ok=True)

    docx_files = list(directory.glob(pattern))

    if not docx_files:
        print(f"Не найдено файлов {pattern} в {directory}")
        return

    print("="*70)
    print(f"КОНВЕРТАЦИЯ .DOCX → .TXT")
    print("="*70)
    print(f"Директория: {directory}")
    print(f"Найдено файлов: {len(docx_files)}")
    print(f"Режим: {'пропуск существующих' if skip_existing else 'перезапись'}\n")

    converted = []
    skipped = []
    errors = []

    for docx_file in sorted(docx_files):
        txt_file = output_dir / docx_file.with_suffix('.txt').name

        # Проверка перед конвертацией
        if skip_existing and txt_file.exists():
            size = txt_file.stat().st_size
            print(f"⏭️  Пропуск: {docx_file.name} (уже есть {txt_file.name}, {size:,} байт)")
            skipped.append(txt_file)
            continue

        try:
            result = convert_docx_to_txt(
                docx_file, 
                txt_file, 
                skip_existing=False  # Уже проверили выше
            )
            if result:
                converted.append(result)
            else:
                errors.append((docx_file, "Конвертация вернула None"))
        except Exception as e:
            print(f"❌ {docx_file.name}: {e}")
            errors.append((docx_file, e))

    print("\n" + "="*70)
    print("ИТОГО")
    print("="*70)
    print(f"✅ Успешно конвертировано: {len(converted)}")
    print(f"⏭️  Пропущено (уже есть):    {len(skipped)}")
    print(f"❌ Ошибок:                  {len(errors)}\n")

    if converted:
        print("Новые файлы:")
        total_size = 0
        for txt_file in converted:
            size = txt_file.stat().st_size
            total_size += size
            print(f"  {txt_file.name:40} {size:>12,} байт")

        print(f"\nОбщий размер новых: {total_size:,} байт ({total_size/1024/1024:.1f} МБ)")

    if skipped:
        total_skipped_size = sum(f.stat().st_size for f in skipped)
        print(f"\nПропущено файлов: {len(skipped)} ({total_skipped_size/1024/1024:.1f} МБ)")

    if errors:
        print("\n⚠️  Ошибки при конвертации:")
        for docx_file, error in errors:
            print(f"  • {docx_file.name}: {error}")
        print("\nЭти файлы могут быть повреждены или иметь несовместимый формат.")
        print("Попробуйте открыть их в Word и пересохранить.")


def main():
    """Основная функция для запуска конвертации."""

    # Варианты директорий
    possible_dirs = [
        Path("data"),
        Path("data/raw"),
        Path("."),
    ]

    data_dir = None
    for d in possible_dirs:
        if d.exists() and list(d.glob("*.docx")):
            data_dir = d
            break

    if data_dir:
        batch_convert_directory(data_dir)
    else:
        print("Не найдено .docx файлов в стандартных директориях.")
        print("Укажите путь вручную:")
        print()
        print("  from pathlib import Path")
        print("  from convert_docx_to_txt import batch_convert_directory")
        print("  batch_convert_directory(Path('путь/к/директории'))")


if __name__ == "__main__":
    main()
