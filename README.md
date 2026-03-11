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

## Установка и запуск

```bash
pip install -r requirements.txt
python pipeline.py
```
