#!/usr/bin/env python3
"""
Main pipeline: Detect borrowings between Usatges de Barcelona and Latin sources.
"""
import argparse
import sys
import time
import logging
from pathlib import Path

import numpy as np

from config import (
    USATGES_TXT, SOURCES, SOURCE_CONFIGS, DEFAULT_SOURCE_CONFIG,
    NGRAM_RANGE, MAX_DF, MIN_DF,
    TFIDF_COSINE_THRESHOLD, ALPHA, BETA, GAMMA, FINAL_THRESHOLD,
    SW_MATCH, SW_MISMATCH, SW_GAP, SW_LEVENSHTEIN_BONUS_THRESHOLD,
    SOFT_COSINE_MAX_TERMS, SW_MAX_SEQ_LEN, MIN_LEMMA_LENGTH,
    GRAPH_GEXF, RESULTS_CSV, OUTPUT_DIR,
)
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


def _safe_preprocess(seg_id, text, lemmatizer, min_length):
    """Preprocess a segment with error handling."""
    try:
        return preprocess_segment(text, lemmatizer, min_length=min_length)
    except Exception as e:
        log.warning(f"Preprocessing failed for {seg_id}: {e}")
        return []


def run_pipeline(
    use_collatinus: bool = True,
    final_threshold: float = None,
    top_n: int = 30,
):
    if final_threshold is None:
        final_threshold = FINAL_THRESHOLD

    t0 = time.time()

    # ---- Step 1: Load & Segment ----
    log.info("Step 1: Loading and segmenting texts...")

    if USATGES_TXT.exists():
        usatges_raw = load_txt(USATGES_TXT)
    else:
        log.error(f"Usatges file not found at {USATGES_TXT}")
        sys.exit(1)

    usatges_segments = segment_usatges(usatges_raw)
    log.info(f"  Usatges: {len(usatges_segments)} segments")

    usatge_raw_texts = {seg_id: text for seg_id, text in usatges_segments}
    source_raw_texts = {}

    source_segments = {}
    all_source_segs = []

    for src_name, src_path in SOURCES.items():
        if not src_path.exists():
            log.warning(f"  Source '{src_name}' not found at {src_path}, skipping.")
            continue
        try:
            raw = load_docx(src_path)
        except Exception as e:
            log.warning(f"  Failed to load '{src_name}': {e}, skipping.")
            continue

        # Use modular config
        cfg = SOURCE_CONFIGS.get(src_name, DEFAULT_SOURCE_CONFIG)
        segs = segment_source(raw, src_name, cfg)

        # Validate segments
        valid_segs = []
        for seg_id, text in segs:
            if text and len(text.strip()) >= 20:
                valid_segs.append((seg_id, text))

        source_segments[src_name] = valid_segs
        all_source_segs.extend(valid_segs)
        for seg_id, text in valid_segs:
            source_raw_texts[seg_id] = text
        log.info(f"  {src_name}: {len(valid_segs)} segments")

    if not all_source_segs:
        log.error("No source texts found. Place .docx files in data/ directory.")
        sys.exit(1)

    log.info(f"  Total source segments: {len(all_source_segs)}")

    # ---- Step 2: Preprocess ----
    log.info("Step 2: Preprocessing (normalize, tokenize, lemmatize)...")
    lemmatizer = LatinLemmatizer(use_collatinus=use_collatinus)

    usatges_lemmas = {}
    for seg_id, text in usatges_segments:
        lemmas = _safe_preprocess(seg_id, text, lemmatizer, MIN_LEMMA_LENGTH)
        if lemmas:
            usatges_lemmas[seg_id] = lemmas

    source_lemmas = {}
    skipped = 0
    for seg_id, text in all_source_segs:
        lemmas = _safe_preprocess(seg_id, text, lemmatizer, MIN_LEMMA_LENGTH)
        if lemmas:
            source_lemmas[seg_id] = lemmas
        else:
            skipped += 1

    log.info(f"  Preprocessed {len(usatges_lemmas)} usatge + {len(source_lemmas)} source segments"
             f" (skipped {skipped} empty)")

    if not usatges_lemmas or not source_lemmas:
        log.error("No valid segments after preprocessing.")
        sys.exit(1)

    # ---- Step 3: Feature extraction (TF-IDF) ----
    log.info("Step 3: Building TF-IDF matrix...")

    u_ids = list(usatges_lemmas.keys())
    s_ids = list(source_lemmas.keys())

    corpus = [usatges_lemmas[uid] for uid in u_ids] + [source_lemmas[sid] for sid in s_ids]

    tfidf_matrix, vocab, term2idx = build_tfidf_matrix(
        corpus, ngram_range=NGRAM_RANGE, max_df=MAX_DF, min_df=MIN_DF,
    )
    log.info(f"  Vocabulary size: {len(vocab)} terms, matrix shape: {tfidf_matrix.shape}")

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
        u_lem = usatges_lemmas.get(u_id, [])
        s_lem = source_lemmas.get(s_id, [])

        if not u_lem or not s_lem:
            continue

        tess = tesserae_score(u_lem, s_lem, idf)

        if cos_sim + tess * BETA > final_threshold * 0.5:
            soft_cos = soft_cosine_similarity(u_lem, s_lem, max_terms=SOFT_COSINE_MAX_TERMS)
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
        u_lem = usatges_lemmas.get(u_id, [])
        s_lem = source_lemmas.get(s_id, [])

        try:
            al_a, al_b, al_score = smith_waterman(
                u_lem, s_lem,
                match_score=SW_MATCH,
                mismatch_score=SW_MISMATCH,
                gap_penalty=SW_GAP,
                lev_bonus_threshold=SW_LEVENSHTEIN_BONUS_THRESHOLD,
                max_seq_len=SW_MAX_SEQ_LEN,
            )
        except Exception as e:
            log.warning(f"Smith-Waterman failed for {u_id} <-> {s_id}: {e}")
            al_a, al_b, al_score = [], [], 0.0

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

    graph.export_gexf(GRAPH_GEXF)
    log.info(f"  Graph GEXF: {GRAPH_GEXF}")

    graph.export_csv(RESULTS_CSV, usatge_texts=usatge_raw_texts,
                     source_texts=source_raw_texts)
    log.info(f"  Results CSV: {RESULTS_CSV}")

    table_path = OUTPUT_DIR / "borrowing_table.md"
    graph.export_borrowing_table(table_path, usatge_texts=usatge_raw_texts)
    log.info(f"  Borrowing table: {table_path}")

    heatmap_path = OUTPUT_DIR / "heatmap_borrowings.png"
    graph.visualize_heatmap(heatmap_path, usatge_texts=usatge_raw_texts)
    log.info(f"  Heatmap: {heatmap_path}")

    graph.visualize_per_source(OUTPUT_DIR, usatge_texts=usatge_raw_texts)
    log.info(f"  Per-source graphs: {OUTPUT_DIR}/graph_*.png")

    top_path = OUTPUT_DIR / "top_borrowings_graph.png"
    graph.visualize_top_borrowings(top_path, top_n=top_n,
                                    usatge_texts=usatge_raw_texts)
    log.info(f"  Top-{top_n} graph: {top_path}")

    bar_path = OUTPUT_DIR / "bar_chart_borrowings.png"
    graph.visualize_bar_chart(bar_path, usatge_texts=usatge_raw_texts)
    log.info(f"  Bar chart: {bar_path}")

    stats_text = graph.format_stats(usatge_texts=usatge_raw_texts)
    log.info("\n" + stats_text)

    elapsed = time.time() - t0
    log.info(f"\nPipeline completed in {elapsed:.1f}s")

    return graph, aligned_pairs, graph.get_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Detect textual borrowings between Usatges de Barcelona and Latin sources."
    )
    parser.add_argument("--no-collatinus", action="store_true",
                        help="Use fallback stemmer instead of Collatinus")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Final BorrowScore threshold (default: {FINAL_THRESHOLD})")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of top borrowings for the focused graph")
    args = parser.parse_args()
    run_pipeline(
        use_collatinus=not args.no_collatinus,
        final_threshold=args.threshold,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
