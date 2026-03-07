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

Файлы источников размещаются в директории `data/`:
```
data/
├── Bastardas Usatges de Barcelona_djvu.txt # Usatges (издание Бастарда, основной)
├── Evangelium.docx
├── Corpus Juris Civilis.docx
├── Isidori Hispalensis Episcopi Etymologiarum.docx
├── Lex visigothorum.docx
└── Exeptionis Legum Romanorum Petri.docx
```
---

## Установка и запуск

```bash
pip install -r requirements.txt
python pipeline.py
```
