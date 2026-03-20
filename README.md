# Usatges de Barcelona — пайплайн обнаружения заимствований

NLP-пайплайн для выявления текстуальных заимствований между
**Usatges de Barcelona** и корпусом латинских, каталонских и дипломатических источников.

Проект использует **единый unified-контур**:
- одна точка входа: `pipeline_unified.py`
- один конфиг экспериментов: `config_unified.py`
- один диспетчер сегментеров: `source_segmenters.py`
- один контракт сегментеров: `list[tuple[str, str]]`

---

## Текущая архитектура

### Точка входа
Основной запуск выполняется через:

```bash
python pipeline_unified.py --config config_unified --experiment <experiment_name>
```

### Конфиг
Эксперименты и корпуса задаются в:

```bash
config_unified.py
```

### Сегментеры
Все сегментеры находятся в:

```bash
segmenters/
```

Они работают по единому правилу:

```python
segment_<source>_unified(source_file, source_name) -> list[tuple[str, str]]
```

Каждый сегмент — это строго пара:

```python
(segment_id, segment_text)
```

`segment_id` содержит реальную структурную идентичность фрагмента внутри источника
(номер статьи, главы, дигеста, грамоты и т.д.).

### Диспетчер сегментации
Маршрутизация по источникам централизована в:

```bash
source_segmenters.py
```

---

## Метод

Пайплайн реализует 7-шаговый NLP-воркфлоу **без машинного обучения**:

1. **Сегментация**  
   Источники режутся по их собственной структурной единице:
   статья, глава, дигест, капитул, грамота и т.д.

2. **Предобработка**  
   Нормализация латинского текста, лемматизация, очистка шумов.

3. **Извлечение признаков**  
   TF-IDF векторы с n-граммами.

4. **Поиск кандидатов**  
   Косинусное сходство по парам сегментов.

5. **Скоринг**  
   Комбинация нескольких сигналов:
   TF-IDF cosine + Tesserae-style overlap + soft cosine.

6. **Выравнивание**  
   Smith–Waterman для локального выравнивания пар-кандидатов.

7. **Граф**  
   Экспорт агрегированных связей в CSV / GEXF / PNG.

---

## Источники

Все рабочие входные данные сейчас представлены как текстовые файлы в `data/`.

| Ключ в конфиге | Файл | Русское название |
|---|---|---|
| `UsatgesBarcelona` | `Bastardas_Usatges_de_Barcelona_djvu.txt` | Обычаи Барселоны |
| `Evangelium` | `Evangelium_v2.txt` | Евангелие (Вульгата) |
| `CorpusJuris` | `Corpus_Juris_Civilis_v2.txt` | Свод гражданского права Юстиниана |
| `Etymologiae` | `Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt` | Этимологии Исидора Севильского |
| `LexVisigoth` | `legesvisigothor00zeumgoog_text.txt` | Вестготская правда |
| `ExceptPetri` | `Exeptionis_Legum_Romanorum_Petri_v3.txt` | Извлечения из римских законов Петра |
| `ObychaiTortosy1272to1279` | `ObychaiTortosy1272to1279_v2.txt` | Обычаи Тортосы |
| `ObychaiLleidy12271228` | `ObychaiLleidy12271228_v2.txt` | Обычаи Ллейды |
| `ObychaiMiraveta1319Fix` | `ObychaiMiraveta1319Fix_v2.txt` | Обычаи Миравета |
| `ObychaiOrty1296` | `ObychaiOrty1296_v2.txt` | Обычаи Орты |
| `RecognovrentProceres12831284` | `RecognovrentProceres12831284_v2.txt` | Recognovrent Proceres |
| `ObychaiTarregi1290E` | `ObychaiTarregi1290E_v2.txt` | Обычаи Тарреги |
| `ObychaiValdArana1313` | `ObychaiValdArana1313_v2.txt` | Обычаи Валь-д’Арана |
| `PragmatikaZhaumeII1295` | `PragmatikaZhaumeII1295_v2.txt` | Прагматика Жауме II (1295) |
| `PragmatikaZhaumeII1301` | `PragmatikaZhaumeII1301_v2.txt` | Прагматика Жауме II (1301) |
| `Gramoty911` | `Gramoty911.txt` | Грамоты IX–XI вв. |
| `Gramoty12` | `Gramoty12.txt` | Грамоты XII в. |

Структура каталога:

```text
data/
├── Bastardas_Usatges_de_Barcelona_djvu.txt
├── Evangelium_v2.txt
├── Corpus_Juris_Civilis_v2.txt
├── Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt
├── legesvisigothor00zeumgoog_text.txt
├── Exeptionis_Legum_Romanorum_Petri_v3.txt
├── ObychaiTortosy1272to1279_v2.txt
├── ObychaiLleidy12271228_v2.txt
├── ObychaiMiraveta1319Fix_v2.txt
├── ObychaiOrty1296_v2.txt
├── RecognovrentProceres12831284_v2.txt
├── ObychaiTarregi1290E_v2.txt
├── ObychaiValdArana1313_v2.txt
├── PragmatikaZhaumeII1295_v2.txt
├── PragmatikaZhaumeII1301_v2.txt
├── Gramoty911.txt
└── Gramoty12.txt
```

---

## Эксперименты

В `config_unified.py` сейчас заданы следующие основные эксперименты:

| Эксперимент | Смысл |
|---|---|
| `test` | smoke test: `Evangelium → UsatgesBarcelona` |
| `latin_to_usatges` | латинские источники → Usatges |
| `left_to_gramoty` | латинские источники + Usatges → грамоты |

Примеры запуска:

```bash
python pipeline_unified.py --config config_unified --experiment test
python pipeline_unified.py --config config_unified --experiment latin_to_usatges
python pipeline_unified.py --config config_unified --experiment left_to_gramoty
```

---

## Сегментеры

Все рабочие сегментеры приведены к unified-стилю.

### Проверка сегментеров
Основной тестовый прогон:

```bash
python test_unified_segmenters.py
```

Для ручной проверки отдельного сегментера:

```bash
python -m segmenters.seg_usatges
python -m segmenters.seg_gramoty_911
python -m segmenters.seg_gramoty_12
python -m segmenters.seg_costums_tortosa
python -m segmenters.seg_lleida
python -m segmenters.seg_miravet
python -m segmenters.seg_orty
python -m segmenters.seg_tarregi
python -m segmenters.seg_vald_aran
python -m segmenters.seg_privileges
python -m segmenters.seg_zhaime1295
python -m segmenters.seg_zhaime1301
python -m segmenters.seg_lex_visigothorum
python -m segmenters.seg_corpus_juris
python -m segmenters.seg_evangelium
python -m segmenters.seg_etymologiae
python -m segmenters.seg_exceptiones_petri
python -m segmenters.seg_default
```

### Текущая статистика сегментации

Статистика ниже отражает текущий unified-контур и служит ориентиром, а не формальной гарантией.

#### Основной текст

| Ключ | Файл | Единица сегментации | Найдено |
|---|---|---|---:|
| `UsatgesBarcelona` | `Bastardas_Usatges_de_Barcelona_djvu.txt` | статья | 145 |

#### Латинские источники

| Ключ | Файл | Единица сегментации | Найдено |
|---|---|---|---:|
| `Evangelium` | `Evangelium_v2.txt` | глава | 89 |
| `CorpusJuris` | `Corpus_Juris_Civilis_v2.txt` | дигест | 9190 |
| `Etymologiae` | `Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt` | глава | 39 |
| `LexVisigoth` | `legesvisigothor00zeumgoog_text.txt` | закон | 188 |
| `ExceptPetri` | `Exeptionis_Legum_Romanorum_Petri_v3.txt` | глава | 115 |

#### Каталанские кутюмы и акты

| Ключ | Файл | Единица сегментации | Найдено |
|---|---|---|---:|
| `ObychaiTortosy1272to1279` | `ObychaiTortosy1272to1279_v2.txt` | статья | 1267 |
| `ObychaiTarregi1290E` | `ObychaiTarregi1290E_v2.txt` | статья | 26 |
| `ObychaiMiraveta1319Fix` | `ObychaiMiraveta1319Fix_v2.txt` | статья | 100 |
| `ObychaiLleidy12271228` | `ObychaiLleidy12271228_v2.txt` | статья | 172 |
| `PragmatikaZhaumeII1295` | `PragmatikaZhaumeII1295_v2.txt` | статья | 17 |
| `PragmatikaZhaumeII1301` | `PragmatikaZhaumeII1301_v2.txt` | статья | 25 |
| `ObychaiOrty1296` | `ObychaiOrty1296_v2.txt` | статья | 79 |
| `ObychaiValdArana1313` | `ObychaiValdArana1313_v2.txt` | статья | 24 |
| `RecognovrentProceres12831284` | `RecognovrentProceres12831284_v2.txt` | статья | 117 |
| `Gramoty911` | `Gramoty911.txt` | грамота | 556 |
| `Gramoty12` | `Gramoty12.txt` | грамота | 871 |

---

## Установка

```bash
pip install -r requirements.txt
```

---

## Минимальный рабочий цикл

1. Положить текстовые источники в `data/`
2. Проверить сегментеры:
   ```bash
   python test_unified_segmenters.py
   ```
3. Запустить нужный эксперимент:
   ```bash
   python pipeline_unified.py --config config_unified --experiment latin_to_usatges
   ```
4. Смотреть результаты в директории, указанной в `config_unified.py`


