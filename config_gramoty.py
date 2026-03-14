"""
Конфигурация отдельного пайплайна исследования грамот.

Задача:
    латинские источники + Usatges  ->  грамоты (оба тома)

Этот файл совместим с pipeline_gramoty.py и теперь отражает основной
7-шаговый подход: TF-IDF кандидаты -> BorrowScore -> Smith-Waterman -> граф.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Базовые пути
# ----------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_charters")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESUME_STEP = 1

# ----------------------------------------------------------------------
# Основной текст Usatges
# ----------------------------------------------------------------------

USATGES_PATH = DATA_DIR / "Bastardas Usatges de Barcelona_djvu.txt"
USATGES_TXT = USATGES_PATH  # alias for compatibility

# ----------------------------------------------------------------------
# Источники (левая колонка графа)
# ----------------------------------------------------------------------

SOURCES = {
    "Evangelium": DATA_DIR / "Evangelium.docx",
    "CorpusJuris": DATA_DIR / "Corpus Juris Civilis.docx",
    "Etymologiae": DATA_DIR / "Isidori Hispalensis Episcopi Etymologiarum.docx",
    "LexVisigoth": DATA_DIR / "Lex visigothorum.docx",
    "ExceptPetri": DATA_DIR / "Exeptionis Legum Romanorum Petri.docx",
    "Usatges": USATGES_PATH,
}

SOURCE_NAMES_RU = {
    "Evangelium": "Евангелие\n(Вульгата)",
    "CorpusJuris": "Свод\nЮстиниана",
    "Etymologiae": "Этимологии\nИсидора",
    "LexVisigoth": "Вестготская\nправда",
    "ExceptPetri": "Извлечения\nПетра",
    "Usatges": "Обычаи\nБарселоны",
}

SOURCE_NAMES_RU_SHORT = {
    "Evangelium": "Евангелие",
    "CorpusJuris": "Свод Юстиниана",
    "Etymologiae": "Этимологии Исидора",
    "LexVisigoth": "Вестготская правда",
    "ExceptPetri": "Извлечения Петра",
    "Usatges": "Обычаи Барселоны",
}

# ----------------------------------------------------------------------
# Сегментация источников
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

MAX_SEGMENT_WORDS = 200
MIN_SEGMENT_WORDS = 12
GRAMOTY_MIN_WORDS = MIN_SEGMENT_WORDS

# ----------------------------------------------------------------------
# Грамоты (правая колонка графа)
# ----------------------------------------------------------------------

GRAMOTY = {
    "Gramoty911": DATA_DIR / "Gramoty911.txt",   # IX–XI вв.
    "Gramoty12": DATA_DIR / "Gramoty12.txt",     # XII в.
}

CHARTERS = GRAMOTY
CHARTER_CORPORA = GRAMOTY

CHARTER_NAMES_RU = {
    "Gramoty911": "Грамоты IX–XI вв.",
    "Gramoty12": "Грамоты XII в.",
}

GRAMOTY_CONFIGS = {
    "Gramoty911": {
        "type": "gramoty",
        "max_segment_words": 200,
    },
    "Gramoty12": {
        "type": "gramoty",
        "max_segment_words": 200,
    },
}

CHARTER_CONFIGS = GRAMOTY_CONFIGS
SEGMENT_CONFIGS = GRAMOTY_CONFIGS

# ----------------------------------------------------------------------
# Пороговые параметры и скоринг
# ----------------------------------------------------------------------

MIN_LEMMA_LENGTH = 3
USE_COLLATINUS = False

NGRAM_RANGE = (1, 3)
MAX_DF = 0.50
MIN_DF = 2

# Step 4: TF-IDF candidate threshold
GRAMOTY_TFIDF_COSINE_THRESHOLD = 0.08
TFIDF_COSINE_THRESHOLD = GRAMOTY_TFIDF_COSINE_THRESHOLD
GRAMOTY_COSINE_THRESHOLD = GRAMOTY_TFIDF_COSINE_THRESHOLD
CHARTER_COSINE_THRESHOLD = GRAMOTY_TFIDF_COSINE_THRESHOLD
COSINE_THRESHOLD = GRAMOTY_TFIDF_COSINE_THRESHOLD

GRAMOTY_TOP_K = 5
TOP_K = GRAMOTY_TOP_K

ALPHA = 0.30
BETA = 0.40
GAMMA = 0.30
GRAMOTY_FINAL_THRESHOLD = 0.12
FINAL_THRESHOLD = GRAMOTY_FINAL_THRESHOLD
GRAMOTY_MIN_HITS = 2

SW_MATCH = 2
SW_MISMATCH = -1
SW_GAP = -1
SW_LEVENSHTEIN_BONUS_THRESHOLD = 2
GRAMOTY_SW_MIN_SCORE = 0.0

SOFT_COSINE_MAX_TERMS = 500
SW_MAX_SEQ_LEN = 300
MIN_SEGMENT_LENGTH = 30

# ----------------------------------------------------------------------
# Исторические ограничения по датам
# ----------------------------------------------------------------------

SOURCE_NOT_BEFORE = {
    "ExceptPetri": 1100,
    "Usatges": 1068,
}

# ----------------------------------------------------------------------
# Входной граф предыдущего исследования: sources -> Usatges
# ----------------------------------------------------------------------

SOURCE_GRAPH_GEXF = Path("output") / "borrowing_graph.gexf"
USATGES_GRAPH_GEXF = SOURCE_GRAPH_GEXF
BORROWING_GRAPH_GEXF = SOURCE_GRAPH_GEXF

# ----------------------------------------------------------------------
# Выходные файлы исследования грамот
# ----------------------------------------------------------------------

GRAPH_GEXF = OUTPUT_DIR / "borrowings_charters_graph.gexf"
GRAPH_PNG = OUTPUT_DIR / "borrowings_charters_graph.png"
RESULTS_CSV = OUTPUT_DIR / "borrowings_charters_pairs.csv"

USATGES_TIMELINE_PNG = OUTPUT_DIR / "usatges_borrowings_timeline.png"
USATGES_YEAR_HIST_PNG = OUTPUT_DIR / "usatges_borrowings_year_hist.png"
STATS_SUMMARY_PNG = OUTPUT_DIR / "charters_borrowings_stats.png"

OUT_DIR = OUTPUT_DIR
