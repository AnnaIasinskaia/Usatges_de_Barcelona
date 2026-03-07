"""
Configuration for the Usatges de Barcelona borrowing detection pipeline.
"""
from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Source texts: key = short name, value = filename in DATA_DIR
#USATGES_FILE = DATA_DIR / "Latin.docx"  # Usatges de Barcelona
USATGES_TXT = DATA_DIR / "Bastardas Usatges de Barcelona_djvu.txt"

# Параметры для фильтрации
MIN_SEGMENT_LENGTH = 100  # Минимум символов в обычае
SOURCES = {
    "Evangelium":    DATA_DIR / "Evangelium.docx",
    "CorpusJuris":  DATA_DIR / "Corpus Juris Civilis.docx",
    "Etymologiae":  DATA_DIR / "Isidori Hispalensis Episcopi Etymologiarum.docx",
    "LexVisigoth":  DATA_DIR / "Lex visigothorum.docx",
    "ExceptPetri":  DATA_DIR / "Exeptionis Legum Romanorum Petri.docx",
}

# If you have .txt version of Usatges with article markers like "1 (UB. 1-2)"
#USATGES_TXT = DATA_DIR / "Latin.txt"

# --- Preprocessing ---
MIN_LEMMA_LENGTH = 3
USE_COLLATINUS = False  # set False to use fallback stemmer

# --- Feature extraction ---
NGRAM_RANGE = (1, 3)          # unigrams + bigrams + trigrams
MAX_DF = 0.50                 # discard terms in >50% of segments
MIN_DF = 2                    # term must appear in >=2 segments

# --- Scoring ---
TFIDF_COSINE_THRESHOLD = 0.08   # min TF-IDF cosine to be a candidate pair
ALPHA = 0.30                    # weight for TF-IDF cosine
BETA  = 0.40                    # weight for Tesserae-style score
GAMMA = 0.30                    # weight for soft cosine
FINAL_THRESHOLD = 0.10          # min BorrowScore to include in graph

# --- Smith-Waterman ---
SW_MATCH = 2
SW_MISMATCH = -1
SW_GAP = -1
SW_LEVENSHTEIN_BONUS_THRESHOLD = 2  # if lev_dist <= this, partial match

# --- Output ---
GRAPH_GEXF = OUTPUT_DIR / "borrowing_graph.gexf"
GRAPH_PNG  = OUTPUT_DIR / "borrowing_graph.png"
RESULTS_CSV = OUTPUT_DIR / "borrowing_pairs.csv"
