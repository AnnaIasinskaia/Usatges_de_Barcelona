#!/usr/bin/env python3
"""
Main pipeline: Detect borrowings between Usatges de Barcelona and Latin sources.

Usage:
    python pipeline.py [--no-collatinus] [--threshold 0.10] [--top-n 30]

Requires:
    pip install python-docx numpy networkx matplotlib
    pip install pycollatinus  (optional, for better lemmatization)
"""
import argparse
import sys
import time
import logging
import textwrap
from pathlib import Path

import numpy as np

from config import *
from preprocessing import (
    LatinLemmatizer, load_docx, load_txt,
    segment_usatges, segment_source, preprocess_segment,
)
from features import (
    build_tfidf_matrix, cosine_similarity_matrix,
    find_candidate_pairs, tesserae_score, soft_cosine_similarity,
    compute_idf,
)
from alignment import smith_waterman
from graph_builder import BorrowingGraph


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def run_pipeline(
    use_collatinus: bool = True,
    final_threshold: float = None,
    top_n: int = 30,
):
    """Execute the full borrowing detection pipeline."""

    if final_threshold is None:
        final_threshold = FINAL_THRESHOLD

    t0 = time.time()

    # ---- Step 1: Load & Segment ----
    log.info("Step 1: Loading and segmenting texts...")

    if USATGES_TXT.exists():
        usatges_raw = load_txt(USATGES_TXT)
    elif USATGES_FILE.exists():
        usatges_raw = load_docx(USATGES_FILE)
    else:
        log.error(f"Usatges file not found at {USATGES_TXT} or {USATGES_FILE}")
        sys.exit(1)

    usatges_segments = segment_usatges(usatges_raw)
    log.info(f"  Usatges: {len(usatges_segments)} segments")

    # Store raw text for each segment (for readable output)
    usatge_raw_texts = {seg_id: text for seg_id, text in usatges_segments}
    source_raw_texts = {}

    source_segments = {}
    all_source_segs = []

    for src_name, src_path in SOURCES.items():
        if not src_path.exists():
            log.warning(f"  Source '{src_name}' not found at {src_path}, skipping.")
            continue
        raw = load_docx(src_path)
        segs = segment_source(raw, src_name)
        source_segments[src_name] = segs
        all_source_segs.extend(segs)
        for seg_id, text in segs:
            source_raw_texts[seg_id] = text
        log.info(f"  {src_name}: {len(segs)} segments")

    if not all_source_segs:
        log.error("No source texts found. Place .docx files in data/ directory.")
        sys.exit(1)

    # ---- Step 2: Preprocess ----
    log.info("Step 2: Preprocessing (normalize, tokenize, lemmatize)...")
    lemmatizer = LatinLemmatizer(use_collatinus=use_collatinus)

    usatges_lemmas = {}
    for seg_id, text in usatges_segments:
        usatges_lemmas[seg_id] = preprocess_segment(text, lemmatizer, min_length=MIN_LEMMA_LENGTH)

    source_lemmas = {}
    for seg_id, text in all_source_segs:
        source_lemmas[seg_id] = preprocess_segment(text, lemmatizer, min_length=MIN_LEMMA_LENGTH)

    log.info(f"  Preprocessed {len(usatges_lemmas)} usatge + {len(source_lemmas)} source segments")

    # ---- Step 3: Feature extraction (TF-IDF) ----
    log.info("Step 3: Building TF-IDF matrix...")

    u_ids = list(usatges_lemmas.keys())
    s_ids = list(source_lemmas.keys())

    corpus = [usatges_lemmas[uid] for uid in u_ids] + [source_lemmas[sid] for sid in s_ids]

    tfidf_matrix, vocab, term2idx = build_tfidf_matrix(
        corpus, ngram_range=NGRAM_RANGE, max_df=MAX_DF, min_df=MIN_DF,
    )
    log.info(f"  Vocabulary size: {len(vocab)} terms")

    n_u = len(u_ids)
    tfidf_usatges = tfidf_matrix[:n_u]
    tfidf_sources = tfidf_matrix[n_u:]

    # ---- Step 4: Candidate linking ----
    log.info("Step 4: Finding candidate pairs (TF-IDF cosine)...")

    sim_matrix = cosine_similarity_matrix(tfidf_usatges, tfidf_sources)
    candidates = find_candidate_pairs(sim_matrix, TFIDF_COSINE_THRESHOLD, u_ids, s_ids)
    log.info(f"  Found {len(candidates)} candidate pairs above threshold {TFIDF_COSINE_THRESHOLD}")

    idf = compute_idf(corpus)

    # ---- Step 5: Scoring ----
    log.info("Step 5: Computing combined BorrowScore...")

    scored_pairs = []
    for u_id, s_id, cos_sim in candidates:
        u_lem = usatges_lemmas[u_id]
        s_lem = source_lemmas[s_id]

        tess = tesserae_score(u_lem, s_lem, idf)

        if cos_sim + tess * BETA > final_threshold * 0.5:
            soft_cos = soft_cosine_similarity(u_lem, s_lem)
        else:
            soft_cos = 0.0

        borrow_score = ALPHA * cos_sim + BETA * tess + GAMMA * soft_cos

        if borrow_score >= final_threshold:
            scored_pairs.append((u_id, s_id, borrow_score, cos_sim, tess, soft_cos))

    log.info(f"  {len(scored_pairs)} pairs above final threshold {final_threshold}")

    # ---- Step 6: Alignment ----
    log.info("Step 6: Running Smith-Waterman alignment on top pairs...")

    aligned_pairs = []
    for u_id, s_id, bscore, cos_sim, tess, soft_cos in scored_pairs:
        u_lem = usatges_lemmas[u_id]
        s_lem = source_lemmas[s_id]

        al_a, al_b, al_score = smith_waterman(
            u_lem, s_lem,
            match_score=SW_MATCH,
            mismatch_score=SW_MISMATCH,
            gap_penalty=SW_GAP,
            lev_bonus_threshold=SW_LEVENSHTEIN_BONUS_THRESHOLD,
        )

        aligned_pairs.append({
            "usatge": u_id,
            "source": s_id,
            "borrow_score": bscore,
            "cos_sim": cos_sim,
            "tesserae": tess,
            "soft_cos": soft_cos,
            "sw_score": al_score,
            "alignment_a": al_a,
            "alignment_b": al_b,
        })

    # ---- Step 7: Build graph ----
    log.info("Step 7: Building borrowing graph...")

    graph = BorrowingGraph()

    seg_to_group = {}
    for src_name, segs in source_segments.items():
        for seg_id, _ in segs:
            seg_to_group[seg_id] = src_name

    for pair in aligned_pairs:
        graph.add_borrowing(
            source_id=pair["source"],
            target_id=pair["usatge"],
            weight=pair["borrow_score"],
            source_name=seg_to_group.get(pair["source"], "unknown"),
            alignment_a=pair["alignment_a"],
            alignment_b=pair["alignment_b"],
            usatge_text=usatge_raw_texts.get(pair["usatge"], ""),
            source_text=source_raw_texts.get(pair["source"], ""),
        )

    # ---- Output ----
    log.info("Exporting results...")

    # GEXF for Gephi
    graph.export_gexf(GRAPH_GEXF)
    log.info(f"  Graph GEXF: {GRAPH_GEXF}")

    # CSV with text snippets
    graph.export_csv(RESULTS_CSV, usatge_texts=usatge_raw_texts,
                     source_texts=source_raw_texts)
    log.info(f"  Results CSV: {RESULTS_CSV}")

    # Visualization 1: Heatmap (most readable overview)
    heatmap_path = OUTPUT_DIR / "heatmap_borrowings.png"
    graph.visualize_heatmap(heatmap_path, usatge_texts=usatge_raw_texts)
    log.info(f"  Heatmap: {heatmap_path}")

    # Visualization 2: Per-source graphs (clean bipartite)
    graph.visualize_per_source(OUTPUT_DIR, usatge_texts=usatge_raw_texts)
    log.info(f"  Per-source graphs: {OUTPUT_DIR}/graph_*.png")

    # Visualization 3: Top-N strongest borrowings
    top_path = OUTPUT_DIR / "top_borrowings_graph.png"
    graph.visualize_top_borrowings(top_path, top_n=top_n,
                                    usatge_texts=usatge_raw_texts)
    log.info(f"  Top-{top_n} graph: {top_path}")

    # Visualization 4: Stacked bar chart
    bar_path = OUTPUT_DIR / "bar_chart_borrowings.png"
    graph.visualize_bar_chart(bar_path, usatge_texts=usatge_raw_texts)
    log.info(f"  Bar chart: {bar_path}")

    # Formatted stats with text snippets
    stats_text = graph.format_stats(usatge_texts=usatge_raw_texts)
    log.info("\n" + stats_text)

    elapsed = time.time() - t0
    log.info(f"\n  Pipeline completed in {elapsed:.1f}s")

    return graph, aligned_pairs, graph.get_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Detect textual borrowings between Usatges de Barcelona and Latin sources."
    )
    parser.add_argument(
        "--no-collatinus", action="store_true",
        help="Use fallback stemmer instead of Collatinus lemmatizer",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=f"Final BorrowScore threshold (default: {FINAL_THRESHOLD})",
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Number of top borrowings for the focused graph (default: 30)",
    )
    args = parser.parse_args()

    run_pipeline(
        use_collatinus=not args.no_collatinus,
        final_threshold=args.threshold,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
