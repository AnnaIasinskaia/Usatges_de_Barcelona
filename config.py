"""
Configuration for the Usatges de Barcelona borrowing detection pipeline.
"""
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

USATGES_TXT = DATA_DIR / "Bastardas Usatges de Barcelona_djvu.txt"

SOURCES = {
    "Evangelium":    DATA_DIR / "Evangelium.docx",
    "CorpusJuris":  DATA_DIR / "Corpus Juris Civilis.docx",
    "Etymologiae":  DATA_DIR / "Isidori Hispalensis Episcopi Etymologiarum.docx",
    "LexVisigoth":  DATA_DIR / "Lex visigothorum.docx",
    "ExceptPetri":  DATA_DIR / "Exeptionis Legum Romanorum Petri.docx",
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
SOFT_COSINE_MAX_TERMS = 500   # max unique terms for soft-cosine matrix
SW_MAX_SEQ_LEN = 300          # max lemma sequence length for Smith-Waterman

# --- Output ---
GRAPH_GEXF = OUTPUT_DIR / "borrowing_graph.gexf"
GRAPH_PNG  = OUTPUT_DIR / "borrowing_graph.png"
RESULTS_CSV = OUTPUT_DIR / "borrowing_pairs.csv"

MIN_SEGMENT_LENGTH = 30
