# Контракт сегментеров

## Назначение

Документ описывает унифицированный интерфейс для всех модулей сегментации,
расположенных в папке `segmenters/`.

Цель контракта:
- обеспечить совместимость сегментеров с текущим unified-пайплайном;
- упростить добавление новых источников;
- зафиксировать единый минимальный формат сегментов;
- исключить legacy-варианты интерфейса.

Сегментеры предназначены для работы с:

- `pipeline_unified.py`
- `source_segmenters.py`
- `test_unified_segmenters.py`

---

## Главный принцип

Каждый сегментер должен быть **простым библиотечным модулем**, который:

1. самостоятельно читает текстовый файл источника;
2. режет его на структурные сегменты;
3. возвращает только:

```python
list[tuple[str, str]]
```

то есть список пар:

```python
(segment_id, segment_text)
```

Никакие дополнительные metadata-словари, dataclass-объекты, dict-сегменты и прочие расширенные форматы не используются.

---

## Унифицированный интерфейс

Каждый сегментер обязан экспортировать функцию с именем:

```python
segment_<source>_unified
```

где `<source>` — каноническое имя источника внутри модуля.

Сигнатура:

```python
from pathlib import Path

def segment_source_unified(
    source_file: str | Path,
    source_name: str,
) -> list[tuple[str, str]]:
    ...
```

---

## Параметры

### `source_file`
Путь к текстовому файлу источника.

Допустимый рабочий формат:
- `.txt`

Сегментер обязан прочитать файл самостоятельно, обычно через:

```python
from .seg_common import read_source_file
```

### `source_name`
Каноническое имя источника из `config_unified.py` / `source_segmenters.py`.

Используется:
- для формирования `segment_id`
- для унификации формата выходных идентификаторов

Примеры:
- `CorpusJuris`
- `Evangelium`
- `UsatgesBarcelona`
- `Gramoty911`

---

## Формат возвращаемых данных

Сегментер обязан вернуть:

```python
list[tuple[str, str]]
```

где каждый элемент — это:

### `segment_id: str`
Уникальный идентификатор сегмента в пределах источника.

`segment_id` должен:
- быть стабильным;
- отражать реальную структурную единицу источника;
- быть пригодным для дальнейшей трассировки результата.

Рекомендуемый стиль:
```python
{source_name}_{структурный_идентификатор}
```

Примеры:
- `UsatgesBarcelona_Us_1`
- `CorpusJuris_Dig_1.1.1`
- `Etymologiae_C3`
- `LexVisigoth_2.4.7`
- `Gramoty911_Doc15`
- `PragmatikaZhaumeII1295_Art4`

### `segment_text: str`
Очищенный текст сегмента.

Текст должен:
- быть строкой;
- не быть пустым;
- по возможности не содержать номера страниц, аппаратных помет и явного редакционного шума;
- соответствовать именно той структурной единице, которую сегментер выделяет.

---

## Что сегментер не должен делать

Сегментер **не должен**:

- возвращать dict вместо tuple;
- возвращать `(id, text, meta)` или любые другие расширенные форматы;
- опираться на старые пайплайны;
- сохранять сегментированные версии текста на диск;
- писать `*_segmented.txt`;
- включать в рабочую логику `analyze_and_save(...)`;
- управлять chunking;
- делать постобработку уровня графа, кандидатов, сходства и т.д.

Сегментер отвечает только за:
- чтение файла;
- структурную сегментацию;
- возврат строгого unified-формата.

---

## Роль `seg_common.py`

Общие утилиты выносятся в `segmenters/seg_common.py`.

Разрешённый типичный набор:
- `read_source_file(...)`
- `clean_text(...)`
- `is_apparatus_line(...)`
- `validate_segments(...)`

Сегментеры могут использовать эти функции, но не обязаны переносить в `seg_common.py`
узкоспециальную source-specific логику.

---

## Рекомендуемая структура модуля

Рекомендуемый шаблон:

```python
from __future__ import annotations

from pathlib import Path

from .seg_common import read_source_file, validate_segments


def segment_source(text: str, source_name: str) -> list[tuple[str, str]]:
    ...
    return segments


def segment_source_unified(source_file, source_name):
    text = read_source_file(source_file)
    raw_segments = segment_source(text, source_name=source_name)
    return validate_segments(raw_segments, source_name)


def main() -> None:
    candidates = [
        Path("data/source.txt"),
        Path("source.txt"),
        Path("/mnt/data/source.txt"),
    ]

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("Source file not found.")
        raise SystemExit(1)

    segs = segment_source_unified(src, "SourceName")
    print(f"SourceName: {len(segs)} segments")

    if segs:
        print("First 3 segments:")
        for sid, txt in segs[:3]:
            print(f"  {sid}: {txt[:120]}")

        print("Last 3 segments:")
        for sid, txt in segs[-3:]:
            print(f"  {sid}: {txt[:120]}")


if __name__ == "__main__":
    main()
```

---

## Правила именования

### Имя файла
Файл сегментера:
```python
seg_<source>.py
```

Примеры:
- `seg_usatges.py`
- `seg_gramoty_911.py`
- `seg_corpus_juris.py`

### Имя unified-функции
```python
segment_<source>_unified
```

Примеры:
- `segment_usatges_unified`
- `segment_gramoty_911_unified`
- `segment_corpus_juris_unified`

Именно это имя ожидает `test_unified_segmenters.py`.

---

## Проверка сегментеров

Основная автоматическая проверка выполняется через:

```bash
python test_unified_segmenters.py
```

Ручной smoke-run конкретного сегментера:

```bash
python -m segmenters.seg_usatges
python -m segmenters.seg_corpus_juris
python -m segmenters.seg_default
```

Тестирование должно подтверждать:
- наличие ожидаемой unified-функции;
- корректную сигнатуру;
- возврат строгого `list[tuple[str, str]]`;
- уникальность `segment_id` внутри источника.

---

## Инварианты контракта

На текущем этапе архитектуры считаются обязательными следующие инварианты:

1. **Один сегмент = одна реальная структурная единица источника**
2. **Возврат только `(id, text)`**
3. **`id` уникален внутри источника**
4. **Сегментер ничего не пишет на диск**
5. **Сегментер не знает о scoring, matching, graph-building**
6. **Сегментер совместим с `pipeline_unified.py`**
7. **Сегментер проходит `test_unified_segmenters.py`**

---

## Итог

Минимальный и обязательный контракт сегментера:

```python
segment_<source>_unified(source_file, source_name) -> list[tuple[str, str]]
```

Это и есть единственный рабочий интерфейс сегментеров в текущем репозитории.
