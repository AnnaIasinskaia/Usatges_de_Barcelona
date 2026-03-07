"""
Конфигурация пайплайна обнаружения заимствований в «Обычаях Барселоны».

Usatges de Barcelona (кат. Usatges de Barcelona, лат. Usatici Barchinonae) —
свод обычного права Каталонии, кодифицированный при Рамоне Беренгере I (сер. XI в.).
"""
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

USATGES_TXT = DATA_DIR / "Bastardas Usatges de Barcelona_djvu.txt"

# =====================================================================
#  Источники — латинские тексты, из которых предполагаются заимствования
# =====================================================================
#
# Русские названия приводятся по следующим изданиям:
#
# 1. «Свод Юстиниана» (Corpus Iuris Civilis) — каноническое русское
#    название юстиниановой кодификации римского права.
#    См.: Покровский И. А. История римского права. — СПб., 1913.
#    Энциклопедический словарь Брокгауза и Ефрона: «Corpus juris civilis».
#
# 2. «Вестготская правда» (Lex Visigothorum / Liber Iudiciorum) —
#    первый полный русский перевод:
#    Вестготская правда (Книга приговоров) / под ред. О. В. Аурова,
#    Д. Ю. Полдникова. — М.: Русский Фонд Содействия Образованию
#    и Науке, 2012. — 944 с. (ISBN 978-5-91244-069-4)
#
# 3. «Извлечения Петра из римских законов» (Petri Exceptiones Legum
#    Romanorum) — русское название по: Томсинов В. А. О сущности
#    явления, называемого «рецепцией римского права» // Вестник
#    МГУ. Серия 11, Право. 1998.
#    Перевод: Полдников Д. Ю. Составленные Петром извлечения из
#    римских законов / Под ред. Л. Л. Кофанова. — М., 2010.
#
# 4. «Евангелие» (Вульгата) — латинский перевод Священного Писания
#    блаж. Иеронима Стридонского (382–405 гг.).
#
# 5. «Этимологии» Исидора Севильского (Etymologiae sive Origines) —
#    энциклопедический труд в 20 книгах (ок. 615–636 гг.).
#    Русский перевод кн. V: Павлов А. А. Исидор Севильский. Этимологии,
#    или Начала. Книга V: О законах и временах // Диалог со временем.
#    2014. Вып. 46. С. 354–373.

SOURCES = {
    "Evangelium":    DATA_DIR / "Evangelium.docx",
    "CorpusJuris":  DATA_DIR / "Corpus Juris Civilis.docx",
    "Etymologiae":  DATA_DIR / "Isidori Hispalensis Episcopi Etymologiarum.docx",
    "LexVisigoth":  DATA_DIR / "Lex visigothorum.docx",
    "ExceptPetri":  DATA_DIR / "Exeptionis Legum Romanorum Petri.docx",
}

# Русские отображаемые названия (для графиков и таблиц)
SOURCE_NAMES_RU = {
    "Evangelium":   "Евангелие\n(Вульгата)",
    "CorpusJuris":  "Свод\nЮстиниана",
    "Etymologiae":  "Этимологии\nИсидора",
    "LexVisigoth":  "Вестготская\nправда",
    "ExceptPetri":  "Извлечения\nПетра",
    "Usatges":      "Обычаи\nБарселоны",
}

# Краткие (однострочные) русские названия
SOURCE_NAMES_RU_SHORT = {
    "Evangelium":   "Евангелие",
    "CorpusJuris":  "Свод Юстиниана",
    "Etymologiae":  "Этимологии Исидора",
    "LexVisigoth":  "Вестготская правда",
    "ExceptPetri":  "Извлечения Петра",
    "Usatges":      "Обычаи Барселоны",
}

# --- Modular source segmentation configs ---
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
}

DEFAULT_SOURCE_CONFIG = {
    "type": "default",
    "max_segment_words": 150,
}

# --- Preprocessing ---
MIN_LEMMA_LENGTH = 3
USE_COLLATINUS = False

# --- Feature extraction ---
NGRAM_RANGE = (1, 3)
MAX_DF = 0.50
MIN_DF = 2

# --- Scoring ---
TFIDF_COSINE_THRESHOLD = 0.08
ALPHA = 0.30
BETA  = 0.40
GAMMA = 0.30
FINAL_THRESHOLD = 0.10

# --- Smith-Waterman ---
SW_MATCH = 2
SW_MISMATCH = -1
SW_GAP = -1
SW_LEVENSHTEIN_BONUS_THRESHOLD = 2

# --- Safety caps ---
SOFT_COSINE_MAX_TERMS = 500
SW_MAX_SEQ_LEN = 300

# --- Output ---
GRAPH_GEXF = OUTPUT_DIR / "borrowing_graph.gexf"
GRAPH_PNG  = OUTPUT_DIR / "borrowing_graph.png"
RESULTS_CSV = OUTPUT_DIR / "borrowing_pairs.csv"

MIN_SEGMENT_LENGTH = 30
