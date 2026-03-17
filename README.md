# Usatges de Barcelona — Пайплайн обнаружения заимствований

NLP-пайплайн для выявления текстуальных заимствований между
**Usatges de Barcelona** (XI–XII вв.) и пятью латинскими правовыми источниками.

---

## Метод

Пайплайн реализует 7-шаговый NLP-воркфлоу **без машинного обучения**:

1. **Сегментация** — Usatges делятся по статьям (по маркерам издания Бастарда);
   источники — по структурным элементам документа (главы, дигесты, капитулы)
2. **Предобработка** — нормализация средневековой латыни (J→I, V→U, AE→E, PH→F),
   разбиение энклитик, лемматизация через Collatinus (или правиловой стеммер)
3. **Извлечение признаков** — TF-IDF векторы с н-граммами (1–3),
   фильтрация по порогам document frequency
4. **Поиск кандидатов** — косинусное сходство по всем парам (usatge, сегмент источника)
5. **Скоринг** — комбинированный BorrowScore:
   TF-IDF косинус (α) + Tesserae IDF-взвешенное пересечение (β) + мягкий косинус с Левенштейном (γ)
6. **Выравнивание** — семантическое локальное выравнивание Smith–Waterman для пар-кандидатов
7. **Граф** — взвешенный ориентированный граф (источник → обычай),
   экспорт в GEXF для Gephi и PNG для быстрого просмотра

---

## Источники

| Ключ в конфиге | Файл | Русское название |
|---|---|---|
| `Evangelium` | `Evangelium.docx` | Евангелие (Вульгата, пер. блаж. Иеронима, 382–405) |
| `CorpusJuris` | `Corpus Juris Civilis.docx` | Свод гражданского права Юстиниана |
| `Etymologiae` | `Isidori Hispalensis Episcopi Etymologiarum.docx` | Этимологии Исидора Севильского |
| `LexVisigoth` | `Lex visigothorum.docx` | Вестготская правда (Liber Iudiciorum) |
| `ExceptPetri` | `Exeptionis Legum Romanorum Petri.docx` | Извлечения из римских законов Петра |
| `ObychaiMiraveta1319Fix`  | `Obychai_Miraveta_1319_fix.docx`  | Обычаи Миравета 1319 fix |
| `PragmatikaZhaumeII1295` | `Pragmatika_Zhaume_II_1295.docx`  | Прагматика Жауме II 1295 |
| `PragmatikaZhaumeII1301` | `Pragmatika_Zhaume_II_1301.docx`  | Прагматика Жауме II 1301 |
| `ObychaiOrty1296` | `Obychai_Orty_1296.docx` | Обычаи Орты 1296 |
| `ObychaiTarregi1290E` | `Obychai_Tarregi_1290_e.docx` | Обычаи Тарреги 1290 e |
| `ObychaiTortosy1272to1279` | `Obychai_Tortosy_1272–1279.docx` | Обычаи Тортосы 1272–1279 |
| `RecognovrentProceres12831284` | `Recognovrent_proceres_1283_1284.docx` | Recognovrent proceres 1283 1284 |
| `Gramoty911` | `Gramoty_9_11.docx` | Грамоты 9 11 |
| `Gramoty12` | `Gramoty_12.docx` | Грамоты 12 |
| `ObychaiValdArana1313` | `Obychai_Val-d'Arana_1313.docx` | Обычаи Валь-д'Арана 1313 |
| `ObychaiLleidy12271228` | `Obychai_Lleidy_1227_1228.docx` | Обычаи Ллейды 1227 1228 |

Файлы источников размещаются в директории `data/`:
```
data/
├── Bastardas Usatges de Barcelona_djvu.txt # Usatges (издание Бастарда, основной)
├── Evangelium.docx
├── Corpus Juris Civilis.docx
├── Isidori Hispalensis Episcopi Etymologiarum.docx
├── Lex visigothorum.docx
├── Exeptionis Legum Romanorum Petri.docx
├── Obychai_Miraveta_1319_fix.docx
├── Pragmatika_Zhaume_II_1295.docx
├── Pragmatika_Zhaume_II_1301.docx
├── Obychai_Orty_1296.docx
├── Obychai_Tarregi_1290_e.docx
├── Obychai_Tortosy_1272–1279.docx
├── Recognovrent_proceres_1283_1284.docx
├── Gramoty_9_11.docx
├── Gramoty_12.docx
├── Obychai_Val-d'Arana_1313.docx
└── Obychai_Lleidy_1227_1228.docx
```
---
## Сегментеры

# Трекинг сегментации источников — Usatges de Barcelona

## Основной текст

| Ключ в конфиге        | Файл                                         | Название                         | Ед. сегментации | Ожидаемое кол-во | Найдено | Покрытие (%) | Сегментер |
|-----------------------|----------------------------------------------|----------------------------------|-----------------|------------------|--------|-------------|-----------|
| `UsatgesBarcelona`    | `Bastardas Usatges de Barcelona_djvu.txt`    | Барселонские обычаи              | статья          | 145              | **145**    | 100.0       | seg_usatges.py |

## Латинские источники

| Ключ в конфиге   | Файл                                         | Название                                   | Ед. сегментации | Ожидаемое кол-во | Найдено | Покрытие (%) | Сегментер |
|------------------|----------------------------------------------|--------------------------------------------|-----------------|------------------|--------|-------------|-----------|
| `Evangelium`     | `Evangelium.docx`                            | Евангелие (Вульгата)                       | стих                |              | 439    |    | seg_evangelium.py |
| `CorpusJuris`    | `Corpus Juris Civilis.docx`                  | Свод Юстиниана                             |                     |              | 7361   |    | seg_corpus_juris.py |
| `Etymologiae`    | `Isidori Hispalensis Episcopi Etymologiarum.docx` | «Этимологии» Исидора Севильского      |                     |              | 24     |    | seg_etymologiae.py |
| `LexVisigoth`    | `Lex visigothorum.docx`                      | Вестготская правда                         | закон               |              | 292    |    | seg_lex_visigothorum.py |
| `ExceptPetri`    | `Exeptionis Legum Romanorum Petri.docx`      | Составленные Петром извлечения из римских законов | параграф     |              | 123    |    | seg_exceptiones_petri.py |

## Каталанские кутюмы и акты

| Ключ в конфиге                 | Файл                                   | Название                             | Ед. сегментации | Ожидаемое кол-во | Найдено | Покрытие (%) | Сегментер |
|--------------------------------|----------------------------------------|--------------------------------------|-----------------|------------------|--------|-------------|-----------|
| `ObychaiTortosy1272to1279`     | `Obychai_Tortosy_1272–1279.docx`       | Обычаи Тортосы                       | статья          | 1350             | 1274   | 94.4 | seg_costums_tortosa.py |
| `ObychaiTarregi1290E`          | `Obychai_Tarregi_1290_e.docx`          | Обычаи Тарреги 1290 e                | статья          | 25               | 19     | 76.0 | seg_tarregi.py |
| `ObychaiMiraveta1319Fix`       | `Obychai_Miraveta_1319_fix.docx`       | Обычаи Миравета 1319                 | статья          | 130              | 40     | 30.8 | seg_miravet.py |
| `ObychaiLleidy12271228`        | `Obychai_Lleidy_1227_1228.docx`        | Обычаи Ллейды 1227–1228              | статья          | 171              | 171    | 100  | seg_lleida.py |
| `PragmatikaZhaumeII1295`       | `Pragmatika_Zhaume_II_1295.docx`       | Прагматика Жауме II 1295             | статья          |                  | 17     |      | seg_zhaime1295.py |
| `PragmatikaZhaumeII1301`       | `Pragmatika_Zhaume_II_1301.docx`       | Прагматика Жауме II 1301             | статья          |                  | 25     |      | seg_zhaime1301.py |
| `ObychaiOrty1296`              | `Obychai_Orty_1296.docx`               | Обычаи Орты 1296                     | статья          |                  | 80     |      | seg_orty.py |
| `ObychaiValdArana1313`         | `Obychai_Val-d'Arana_1313.docx`        | Обычаи Валь-д’Арана 1313             | статья          |                  | 28     |      | seg_vald_aran.py |
| `RecognovrentProceres12831284` | `Recognovrent_proceres_1283_1284.docx` | Recognovrent proceres 1283–1284      | статья          |                  | 112    |      | seg_privileges.py |
| `Gramoty911`                   | `Gramoty_9_11.docx`                    | Грамоты IX–XI вв.                    | грамота         | 557              | 555    |      | seg_gramoty_911.py |
| `Gramoty12`                    | `Gramoty_12.docx`                      | Грамоты XII в.                       | грамота         | 873              | 862    |      | seg_gramoty_v2.py |


## Установка и запуск

```bash
pip install -r requirements.txt
python pipeline.py
```
