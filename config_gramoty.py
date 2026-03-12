"""
Конфигурация пайплайна обнаружения заимствований из латинских источников
(включая «Обычаи Барселоны») В ГРАМОТЫ (оба тома).

Пайплайн зеркален исходному: слева — латинские источники, справа — грамоты.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Базовые пути
# ----------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_charters")
OUTPUT_DIR.mkdir(exist_ok=True)

# Тексты Usatges уже сегментируются в исходном пайплайне,
# здесь мы используем те же источники + грамоты как "цели".
USATGES_TXT = DATA_DIR / "Bastardas Usatges de Barcelona_djvu.txt"

# ----------------------------------------------------------------------
# Источники (left) — Латинские тексты
# ----------------------------------------------------------------------

SOURCES = {
    "Evangelium":   DATA_DIR / "Evangelium.docx",
    "CorpusJuris":  DATA_DIR / "Corpus Juris Civilis.docx",
    "Etymologiae":  DATA_DIR / "Isidori Hispalensis Episcopi Etymologiarum.docx",
    "LexVisigoth":  DATA_DIR / "Lex visigothorum.docx",
    "ExceptPetri":  DATA_DIR / "Exeptionis Legum Romanorum Petri.docx",
    "Usatges":      USATGES_TXT,
}

# Русские отображаемые названия (для графиков и таблиц)
SOURCE_NAMES_RU = {
    "Evangelium":  "Евангелие\n(Вульгата)",
    "CorpusJuris": "Свод\nЮстиниана",
    "Etymologiae": "Этимологии\nИсидора",
    "LexVisigoth": "Вестготская\nправда",
    "ExceptPetri": "Извлечения\nПетра",
    "Usatges":     "Обычаи\nБарселоны",
}

# Краткие (однострочные) названия
SOURCE_NAMES_RU_SHORT = {
    "Evangelium":  "Евангелие",
    "CorpusJuris": "Свод Юстиниана",
    "Etymologiae": "Этимологии Исидора",
    "LexVisigoth": "Вестготская правда",
    "ExceptPetri": "Извлечения Петра",
    "Usatges":     "Обычаи Барселоны",
}

# ----------------------------------------------------------------------
# Конфиги сегментации источников (как в старом пайплайне)
# ----------------------------------------------------------------------

SOURCE_CONFIGS = {
    "Evangelium": {
        "type": "evangelium",
        "max_segment_words": 200,
    },
    "CorpusJuris": {
        "type": "corpus_juris",
        "max_segment_words": 200,
    },
    "LexVisigoth": {
        "type": "lex_visigothorum",
        "max_segment_words": 200,
    },
    "ExceptPetri": {
        "type": "exceptiones_petri",
        "max_segment_words": 200,
    },
    "Etymologiae": {
        "type": "etymologiae",
        "max_segment_words": 200,
    },
    "Usatges": {
        "type": "usatges",
        "max_segment_words": 200,
    },
}

DEFAULT_SOURCE_CONFIG = {
    "type": "default",
    "max_segment_words": 150,
}

# ----------------------------------------------------------------------
# Цели (right) — грамоты двух томов
# ----------------------------------------------------------------------

# Имена target-корпусов (обрабатываются seg_gramoty.py)
CHARTER_CORPORA = {
    "Gramoty911": DATA_DIR / "Gramoty911.txt",   # IX–XI вв.
    "Gramoty12":  DATA_DIR / "Gramoty12.txt",    # XII в.
}

# Русские подписи корпусов (для легенд и графиков)
CHARTER_NAMES_RU = {
    "Gramoty911": "Грамоты IX–XI вв.",
    "Gramoty12":  "Грамоты XII в.",
}

# ----------------------------------------------------------------------
# Препроцессинг / признаки / скоринг — те же параметры,
# чтобы обеспечить сопоставимость результатов
# ----------------------------------------------------------------------

MIN_LEMMA_LENGTH = 3
USE_COLLATINUS = False

NGRAM_RANGE = (1, 3)
MAX_DF = 0.50
MIN_DF = 2

TFIDF_COSINE_THRESHOLD = 0.08
ALPHA = 0.30
BETA = 0.40
GAMMA = 0.30
FINAL_THRESHOLD = 0.10

SW_MATCH = 2
SW_MISMATCH = -1
SW_GAP = -1
SW_LEVENSHTEIN_BONUS_THRESHOLD = 2

SOFT_COSINE_MAX_TERMS = 500
SW_MAX_SEQ_LEN = 300

MIN_SEGMENT_LENGTH = 30

# ----------------------------------------------------------------------
# Новые имена выходных файлов (чтобы не пересекаться со старым пайплайном)
# ----------------------------------------------------------------------

GRAPH_GEXF = OUTPUT_DIR / "borrowings_charters_graph.gexf"
GRAPH_PNG  = OUTPUT_DIR / "borrowings_charters_graph.png"
RESULTS_CSV = OUTPUT_DIR / "borrowings_charters_pairs.csv"

# Файлы для дополнительных статистик и датировки Usatges
USATGES_TIMELINE_PNG = OUTPUT_DIR / "usatges_borrowings_timeline.png"
USATGES_YEAR_HIST_PNG = OUTPUT_DIR / "usatges_borrowings_year_hist.png"
STATS_SUMMARY_PNG = OUTPUT_DIR / "charters_borrowings_stats.png"
