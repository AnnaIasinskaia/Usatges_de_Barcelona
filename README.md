# Historical Borrowing Detection Pipeline

Пайплайн для поиска текстуальных заимствований между историческими корпусами
(латынь, кутюмы, грамоты и др.).


## Быстрый запуск

1. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Проверить сегментеры:
   ```bash
   python utils/test_unified_segmenters.py
   ```
3. Запустить нужный эксперимент:
   ```bash
   python pipeline.py --config config --experiment test
   ```
4. Смотреть результаты в директории, указанной в `config.py`


## Метод

Пайплайн реализует 7-шаговый NLP-алгоритм:

1. **Загрузка источников**  
   Источники режутся по их собственной структурной единице:
   статья, глава, дигест, капитул, грамота и т.д.

2. **Сегментирование**  
   Разбиение крупных структурных единиц

3. **Предобработка**  
   Нормализация текста, mode-aware stemming/token reduction, очистка шумов.

4. **Предварительный отбор**  
   Отбор ограниченного числа кандидатов по TF-IDF

5. **Расчёт метрик**  
   TF-IDF cosine + Tesserae-style overlap + soft cosine + Smith–Waterman

6. **Pareto + ranking**  
   Парето-фильтрацирование + ранжирование по метрикам

7. **Граф**  
   Экспорт агрегированных связей в CSV / GEXF / PNG.

---

## Эксперименты

В `config.py` сейчас заданы следующие основные эксперименты:

| Эксперимент | Смысл |
|---|---|
| `test` | тестовый запуск |
| `latin_to_usatges` | латинские источники → Usatges |
| `left_to_gramoty` | латинские источники + Usatges → грамоты |
| `usatges_to_other_codes` | Usatges → обычаи других городов |

Примеры запуска:

```bash
python pipeline.py --config config --experiment test
python pipeline.py --config config --experiment latin_to_usatges
python pipeline.py --config config --experiment left_to_gramoty
python pipeline.py --config config --experiment usatges_to_other_codes
```

