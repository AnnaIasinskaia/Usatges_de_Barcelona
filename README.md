# Usatges de Barcelona — Borrowing Detection Pipeline

NLP pipeline for detecting textual borrowings between the **Usatges de Barcelona** (XI–XII c.) and five Latin legal sources.

## Method

The pipeline implements a 7-step NLP workflow **without machine learning**:

1. **Segmentation** — Usatges split by articles; sources by paragraphs (~150 words)
2. **Preprocessing** — Medieval Latin normalization (J→I, V→U, AE→E, PH→F), enclitic splitting, lemmatization via Collatinus
3. **Feature Extraction** — TF-IDF vectors with n-grams (1–3), filtering by DF thresholds
4. **Candidate Linking** — Cosine similarity between all (usatge, source) pairs
5. **Scoring** — Combined BorrowScore: TF-IDF cosine (α) + Tesserae IDF-weighted overlap (β) + Soft cosine with Levenshtein (γ)
6. **Alignment** — Semantic Smith-Waterman local alignment for matched pairs
7. **Graph Construction** — Directed weighted graph (source → usatge), exported as GEXF for Gephi

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `pycollatinus` requires compilation. If installation fails, the pipeline falls back to a rule-based Latin stemmer. Use `--no-collatinus` flag.

## Data

Place source texts as `.docx` files in the `data/` directory:

```
data/
├── Latin.txt                                          # Usatges (with article markers)
├── Latin.docx                                         # Usatges (alternative)
├── Evangelium.docx
├── Corpus Juris Civilis.docx
├── Isidori Hispalensis Episcopi Etymologiarum.docx
├── Lex visigothorum.docx
└── Exeptionis Legum Romanorum Petri.docx
```

## Usage

```bash
# Full pipeline with Collatinus
python pipeline.py

# With fallback stemmer
python pipeline.py --no-collatinus

# Custom threshold
python pipeline.py --threshold 0.15
```

## Output

Results are saved to `output/`:

| File | Description |
|------|-------------|
| `borrowing_graph.gexf` | Graph for Gephi visualization |
| `borrowing_graph.png`  | Quick visualization (matplotlib) |
| `borrowing_pairs.csv`  | All detected borrowings with scores |

## Configuration

Edit `config.py` to adjust:

- **Scoring weights**: `ALPHA`, `BETA`, `GAMMA` (must sum to 1.0)
- **Thresholds**: `TFIDF_COSINE_THRESHOLD`, `FINAL_THRESHOLD`
- **N-gram range**: `NGRAM_RANGE`
- **Document frequency filters**: `MAX_DF`, `MIN_DF`

## Architecture

```
pipeline.py          ← main entry point
├── config.py        ← all parameters
├── preprocessing.py ← normalization, tokenization, lemmatization, segmentation
├── features.py      ← TF-IDF, Tesserae scoring, soft cosine, IDF
├── alignment.py     ← Smith-Waterman local alignment
└── graph_builder.py ← NetworkX graph construction & export
```

## References

- Sidorov G. et al. *Soft Cosine Measure*. Computación y Sistemas, 2014.
- Manning C.D. et al. *Introduction to Information Retrieval*. Cambridge, 2008.
- Büchler M. et al. *TRACER* — text reuse detection framework.
- Coffee N. et al. *Tesserae* — intertextuality detection for classical texts.
- Smith D.A. *Passim* — alignment-based text reuse detection.
