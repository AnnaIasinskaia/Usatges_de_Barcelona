#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_unified_scoring.py

Диагностический скрипт для разбора Step 5/7:
Computing BorrowScore and filtering candidates.

Ничего не меняет в основном pipeline.
Просто воспроизводит шаги 1-4 и детально профилирует scoring по кандидатам.

Запуск:
    python inspect_unified_scoring.py --config config_unified --experiment test
    python inspect_unified_scoring.py --config config_unified --experiment latin_to_usatges
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from preprocessing import LatinLemmatizer
from pipeline_unified import (
    Segment,
    ProgressLogger,
    resolve_group_tokens,
    segment_corpus,
    chunk_segments,
    lemmatize_segments,
    compute_borrow_score,
)
from features import (
    build_tfidf_matrix,
    compute_idf,
    select_tfidf_candidates,
    tesserae_score,
    soft_cosine_similarity,
)

# ----------------------------
# Helpers
# ----------------------------

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def safe_mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs_sorted[int(k)])
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return float(d0 + d1)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def token_stats(tokens: Sequence[str]) -> Dict[str, Any]:
    uniq = set(tokens)
    lens = [len(t) for t in tokens]
    return {
        "n_tokens": len(tokens),
        "n_unique": len(uniq),
        "avg_token_len": round(safe_mean(lens), 3) if lens else 0.0,
        "max_token_len": max(lens) if lens else 0,
    }


def ocrish_token_count(tokens: Sequence[str]) -> int:
    bad = 0
    for t in tokens:
        if not t:
            continue
        vowels = sum(ch in "aeiouy" for ch in t.lower())
        has_digit = any(ch.isdigit() for ch in t)
        has_alpha = any(ch.isalpha() for ch in t)
        if len(t) >= 8 and vowels <= 1:
            bad += 1
        elif has_digit and has_alpha:
            bad += 1
        elif any(ch * 4 in t for ch in set(t)):
            bad += 1
    return bad


def maybe_bool_str(v: bool) -> str:
    return "yes" if v else "no"


# ----------------------------
# Wrapped soft cosine diagnostics
# ----------------------------

def inspect_soft_cosine(
    left_lemmas: List[str],
    right_lemmas: List[str],
    max_terms: int,
    lev_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Отдельный диагностический вызов soft cosine.
    Не меняет оригинальную функцию; просто заранее считает характеристики входа.
    """
    uniq_left = set(left_lemmas)
    uniq_right = set(right_lemmas)
    all_terms = sorted(uniq_left | uniq_right)
    n_terms = len(all_terms)
    will_fallback = n_terms > max_terms

    t0 = now_ms()
    value = float(
        soft_cosine_similarity(
            left_lemmas,
            right_lemmas,
            lev_threshold=lev_threshold,
            max_terms=max_terms,
        )
    )
    t1 = now_ms()

    return {
        "soft_cos": value,
        "soft_time_ms": round(t1 - t0, 3),
        "soft_unique_terms_union": n_terms,
        "soft_unique_terms_left": len(uniq_left),
        "soft_unique_terms_right": len(uniq_right),
        "soft_will_fallback": will_fallback,
        "soft_max_terms": max_terms,
        "soft_lev_threshold": lev_threshold,
    }


# ----------------------------
# Experiment preparation
# ----------------------------

def load_experiment(config_module_name: str, experiment_id: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, List[str]]]:
    cfg = importlib.import_module(config_module_name)
    corpora = dict(getattr(cfg, "CORPORA"))
    groups = dict(getattr(cfg, "GROUPS", {}))
    experiments = dict(getattr(cfg, "EXPERIMENTS"))
    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")
    return cfg, experiments[experiment_id], corpora, groups


def build_segments_for_experiment(
    corpora: Dict[str, Dict[str, Any]],
    groups: Dict[str, List[str]],
    exp: Dict[str, Any],
    logger: ProgressLogger,
) -> Tuple[List[Segment], List[Segment], Dict[str, int]]:
    left_corpora = resolve_group_tokens(exp["graph_sides"]["left"], groups)
    right_corpora = resolve_group_tokens(exp["graph_sides"]["right"], groups)
    used_corpora = sorted(set(left_corpora + right_corpora))

    base_segments: List[Segment] = []
    segmentation_counts: Dict[str, int] = {}

    for cid in used_corpora:
        segs = segment_corpus(cid, corpora[cid], logger)
        segmentation_counts[cid] = len(segs)
        for seg_id, seg_text in segs:
            base_segments.append(
                Segment(
                    id=seg_id,
                    text=seg_text,
                    corpus=cid,
                    side="__unassigned__",
                    parent_id=seg_id,
                )
            )

    base_by_corpus: Dict[str, List[Segment]] = {cid: [] for cid in used_corpora}
    for seg in base_segments:
        base_by_corpus[seg.corpus].append(seg)

    left_base: List[Segment] = []
    for cid in left_corpora:
        for seg in base_by_corpus[cid]:
            left_base.append(Segment(seg.id, seg.text, seg.corpus, "left", seg.parent_id))

    right_base: List[Segment] = []
    for cid in right_corpora:
        for seg in base_by_corpus[cid]:
            right_base.append(Segment(seg.id, seg.text, seg.corpus, "right", seg.parent_id))

    chunk_cfg = dict(exp.get("chunking") or {})
    left_leaf = chunk_segments(left_base, chunk_cfg, logger, "left")
    right_leaf = chunk_segments(right_base, chunk_cfg, logger, "right")

    return left_leaf, right_leaf, segmentation_counts


def build_candidates_for_experiment(
    exp: Dict[str, Any],
    left_leaf: List[Segment],
    right_leaf: List[Segment],
    logger: ProgressLogger,
) -> Tuple[
    List[Tuple[str, str, float]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, float],
    Dict[str, Segment],
    Dict[str, Segment],
]:
    model = dict(exp.get("model") or {})
    lemmatizer = LatinLemmatizer()
    min_lemma_length = int(model.get("min_lemma_length", 3))

    left_lemmas = lemmatize_segments(
        left_leaf,
        lemmatizer,
        min_lemma_length,
        logger,
        "left",
        progress_every=None,
    )
    right_lemmas = lemmatize_segments(
        right_leaf,
        lemmatizer,
        min_lemma_length,
        logger,
        "right",
        progress_every=None,
    )

    left_ids = [s.id for s in left_leaf]
    right_ids = [s.id for s in right_leaf]
    corpus_lemmas = [left_lemmas[i] for i in left_ids] + [right_lemmas[i] for i in right_ids]

    tfidf, vocab, term2idx = build_tfidf_matrix(
        corpus_lemmas,
        ngram_range=tuple(model.get("ngram_range", (1, 3))),
        max_df=float(model.get("max_df", 0.5)),
        min_df=int(model.get("min_df", 2)),
    )

    n_left = len(left_ids)
    tfidf_left = tfidf[:n_left]
    tfidf_right = tfidf[n_left:]

    cand_cfg = dict(exp.get("candidate_selection") or {})
    threshold = float(cand_cfg.get("threshold", model.get("tfidf_cosine_threshold", 0.08)))
    top_k = cand_cfg.get("top_k_per_left")

    candidates = select_tfidf_candidates(
        tfidf_left,
        tfidf_right,
        left_ids=left_ids,
        right_ids=right_ids,
        threshold=threshold,
        top_k_per_left=top_k,
        progress_every=None,
        progress_callback=None,
    )

    idf = compute_idf(corpus_lemmas)
    left_seg_by_id = {s.id: s for s in left_leaf}
    right_seg_by_id = {s.id: s for s in right_leaf}

    return candidates, left_lemmas, right_lemmas, idf, left_seg_by_id, right_seg_by_id


# ----------------------------
# Step 5 detailed inspection
# ----------------------------

def inspect_scoring(
    exp: Dict[str, Any],
    groups: Dict[str, List[str]],
    candidates: List[Tuple[str, str, float]],
    left_lemmas: Dict[str, List[str]],
    right_lemmas: Dict[str, List[str]],
    idf: Dict[str, float],
    left_seg_by_id: Dict[str, Segment],
    right_seg_by_id: Dict[str, Segment],
    logger: ProgressLogger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = dict(exp.get("model") or {})
    mappings = exp.get("mappings") or []
    resolved_mappings = []
    for m in mappings:
        frm = resolve_group_tokens(m.get("from", []), groups)
        to = resolve_group_tokens(m.get("to", []), groups)
        resolved_mappings.append((frm, to))

    final_threshold = float(model.get("final_threshold", 0.10))
    beta = float(model.get("beta", 0.40))
    soft_gate_factor = float(model.get("soft_cosine_gate_factor", 0.50))
    soft_max_terms = int(model.get("soft_cosine_max_terms", 500))

    rows: List[Dict[str, Any]] = []

    total_step5_t0 = now_ms()
    total_tess_ms = 0.0
    total_soft_ms = 0.0
    soft_calls = 0
    soft_fallbacks = 0
    mapping_skips = 0
    empty_skips = 0
    final_rejects = 0

    corpus_pair_counter = Counter()

    for idx, (left_id, right_id, cos_sim) in enumerate(candidates, 1):
        row_t0 = now_ms()

        ls = left_seg_by_id.get(left_id)
        rs = right_seg_by_id.get(right_id)
        if ls is None or rs is None:
            continue

        allowed = False
        for frm, to in resolved_mappings:
            if ls.corpus in frm and rs.corpus in to:
                allowed = True
                break

        llem = left_lemmas.get(left_id, [])
        rlem = right_lemmas.get(right_id, [])

        uniq_l = set(llem)
        uniq_r = set(rlem)
        shared = uniq_l & uniq_r
        union = uniq_l | uniq_r

        left_stats = token_stats(llem)
        right_stats = token_stats(rlem)

        row: Dict[str, Any] = {
            "candidate_idx": idx,
            "left_id": left_id,
            "right_id": right_id,
            "left_corpus": ls.corpus,
            "right_corpus": rs.corpus,
            "left_parent_id": ls.parent_id,
            "right_parent_id": rs.parent_id,
            "left_text_chars": len(ls.text),
            "right_text_chars": len(rs.text),
            "cos_sim": round(float(cos_sim), 6),

            "left_n_lemmas": left_stats["n_tokens"],
            "right_n_lemmas": right_stats["n_tokens"],
            "left_unique_lemmas": left_stats["n_unique"],
            "right_unique_lemmas": right_stats["n_unique"],
            "union_unique_lemmas": len(union),
            "shared_unique_lemmas": len(shared),
            "left_avg_token_len": left_stats["avg_token_len"],
            "right_avg_token_len": right_stats["avg_token_len"],
            "left_max_token_len": left_stats["max_token_len"],
            "right_max_token_len": right_stats["max_token_len"],
            "left_ocrish_tokens": ocrish_token_count(llem),
            "right_ocrish_tokens": ocrish_token_count(rlem),

            "mapping_allowed": allowed,
            "left_empty_lemmas": not bool(llem),
            "right_empty_lemmas": not bool(rlem),
        }

        if not allowed:
            mapping_skips += 1
            row["status"] = "skipped_mapping"
            row["step5_total_ms"] = round(now_ms() - row_t0, 3)
            rows.append(row)
            continue

        if not llem or not rlem:
            empty_skips += 1
            row["status"] = "skipped_empty_lemmas"
            row["step5_total_ms"] = round(now_ms() - row_t0, 3)
            rows.append(row)
            continue

        t_tess0 = now_ms()
        tess = float(tesserae_score(llem, rlem, idf))
        t_tess1 = now_ms()
        tess_ms = t_tess1 - t_tess0
        total_tess_ms += tess_ms

        gate_value = float(cos_sim) + tess * beta
        gate_threshold = final_threshold * soft_gate_factor
        soft_called = gate_value > gate_threshold

        row["tesserae"] = round(tess, 6)
        row["t_tesserae_ms"] = round(tess_ms, 3)
        row["soft_gate_value"] = round(gate_value, 6)
        row["soft_gate_threshold"] = round(gate_threshold, 6)
        row["soft_called"] = soft_called

        soft_value = 0.0
        soft_ms = 0.0
        soft_union = len(union)
        soft_fallback = False

        if soft_called:
            soft_calls += 1
            soft_info = inspect_soft_cosine(
                llem,
                rlem,
                max_terms=soft_max_terms,
                lev_threshold=2,
            )
            soft_value = float(soft_info["soft_cos"])
            soft_ms = float(soft_info["soft_time_ms"])
            soft_union = int(soft_info["soft_unique_terms_union"])
            soft_fallback = bool(soft_info["soft_will_fallback"])
            total_soft_ms += soft_ms
            if soft_fallback:
                soft_fallbacks += 1

        row["soft_cos"] = round(soft_value, 6)
        row["t_soft_cosine_ms"] = round(soft_ms, 3)
        row["soft_union_terms"] = soft_union
        row["soft_fallback"] = soft_fallback

        # Для полноты сравниваем с оригинальной логикой score
        # Чтобы диагностический скрипт не расходился с основным pipeline.
        t_borrow0 = now_ms()
        score, tess2, soft2 = compute_borrow_score(
            float(cos_sim),
            llem,
            rlem,
            idf,
            model,
        )
        t_borrow1 = now_ms()

        row["borrow_score"] = round(float(score), 6)
        row["borrow_tesserae_from_pipeline"] = round(float(tess2), 6)
        row["borrow_soft_from_pipeline"] = round(float(soft2), 6)
        row["t_compute_borrow_score_ms"] = round(t_borrow1 - t_borrow0, 3)

        passed_final = float(score) >= final_threshold
        row["final_threshold"] = final_threshold
        row["passed_final_threshold"] = passed_final

        if not passed_final:
            final_rejects += 1
            row["status"] = "rejected_final_threshold"
        else:
            row["status"] = "kept_after_final_threshold"

        row["step5_total_ms"] = round(now_ms() - row_t0, 3)

        corpus_pair_counter[(ls.corpus, rs.corpus)] += 1
        rows.append(row)

        if idx % 100 == 0 or idx == len(candidates):
            logger.log(f"Scoring inspect progress: {idx}/{len(candidates)}")

    total_step5_ms = now_ms() - total_step5_t0

    step5_times = [float(r.get("step5_total_ms", 0.0)) for r in rows if "step5_total_ms" in r]
    borrow_times = [float(r.get("t_compute_borrow_score_ms", 0.0)) for r in rows if "t_compute_borrow_score_ms" in r]
    soft_times = [float(r.get("t_soft_cosine_ms", 0.0)) for r in rows if float(r.get("t_soft_cosine_ms", 0.0)) > 0]
    union_terms_soft = [int(r.get("soft_union_terms", 0)) for r in rows if r.get("soft_called")]

    slowest = sorted(rows, key=lambda r: float(r.get("step5_total_ms", 0.0)), reverse=True)[:20]

    summary = {
        "total_candidates": len(candidates),
        "rows_written": len(rows),
        "mapping_skips": mapping_skips,
        "empty_skips": empty_skips,
        "final_rejects": final_rejects,
        "soft_calls": soft_calls,
        "soft_fallbacks": soft_fallbacks,
        "soft_call_rate": round(soft_calls / len(candidates), 6) if candidates else 0.0,

        "timing_ms": {
            "step5_total_ms": round(total_step5_ms, 3),
            "tesserae_total_ms": round(total_tess_ms, 3),
            "soft_cosine_total_ms": round(total_soft_ms, 3),
            "step5_avg_ms": round(safe_mean(step5_times), 3),
            "step5_median_ms": round(statistics.median(step5_times), 3) if step5_times else 0.0,
            "step5_p90_ms": round(percentile(step5_times, 0.90), 3),
            "step5_p95_ms": round(percentile(step5_times, 0.95), 3),
            "step5_max_ms": round(max(step5_times), 3) if step5_times else 0.0,

            "borrow_avg_ms": round(safe_mean(borrow_times), 3),
            "borrow_median_ms": round(statistics.median(borrow_times), 3) if borrow_times else 0.0,
            "borrow_max_ms": round(max(borrow_times), 3) if borrow_times else 0.0,

            "soft_avg_ms": round(safe_mean(soft_times), 3),
            "soft_median_ms": round(statistics.median(soft_times), 3) if soft_times else 0.0,
            "soft_max_ms": round(max(soft_times), 3) if soft_times else 0.0,
        },

        "soft_union_terms": {
            "avg": round(safe_mean(union_terms_soft), 3),
            "median": round(statistics.median(union_terms_soft), 3) if union_terms_soft else 0.0,
            "p90": round(percentile(union_terms_soft, 0.90), 3),
            "max": max(union_terms_soft) if union_terms_soft else 0,
        },

        "top_corpus_pairs_by_count": [
            {"left_corpus": k[0], "right_corpus": k[1], "n": v}
            for k, v in corpus_pair_counter.most_common(20)
        ],

        "top_slowest_pairs": [
            {
                "candidate_idx": r.get("candidate_idx"),
                "left_id": r.get("left_id"),
                "right_id": r.get("right_id"),
                "left_corpus": r.get("left_corpus"),
                "right_corpus": r.get("right_corpus"),
                "step5_total_ms": r.get("step5_total_ms"),
                "t_compute_borrow_score_ms": r.get("t_compute_borrow_score_ms"),
                "t_soft_cosine_ms": r.get("t_soft_cosine_ms"),
                "soft_called": r.get("soft_called"),
                "soft_fallback": r.get("soft_fallback"),
                "left_n_lemmas": r.get("left_n_lemmas"),
                "right_n_lemmas": r.get("right_n_lemmas"),
                "union_unique_lemmas": r.get("union_unique_lemmas"),
                "shared_unique_lemmas": r.get("shared_unique_lemmas"),
                "cos_sim": r.get("cos_sim"),
                "tesserae": r.get("tesserae"),
                "borrow_score": r.get("borrow_score"),
                "status": r.get("status"),
            }
            for r in slowest
        ],
    }

    return rows, summary


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_unified", help="Python config module")
    ap.add_argument("--experiment", required=True, help="Experiment id from config")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for scoring inspection. Default: <experiment_output_dir>/inspect_scoring",
    )
    args = ap.parse_args()

    logger = ProgressLogger(enabled=True)

    cfg, exp, corpora, groups = load_experiment(args.config, args.experiment)

    out_dir = Path(args.out_dir) if args.out_dir else Path(exp["output"]["dir"]) / "inspect_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.log("Step A: build segments")
    left_leaf, right_leaf, segmentation_counts = build_segments_for_experiment(corpora, groups, exp, logger)

    logger.log("Step B: build candidates")
    candidates, left_lemmas, right_lemmas, idf, left_seg_by_id, right_seg_by_id = build_candidates_for_experiment(
        exp,
        left_leaf,
        right_leaf,
        logger,
    )

    logger.log("Step C: inspect scoring")
    rows, summary = inspect_scoring(
        exp,
        groups,
        candidates,
        left_lemmas,
        right_lemmas,
        idf,
        left_seg_by_id,
        right_seg_by_id,
        logger,
    )

    logger.log("Step D: write outputs")
    write_csv(rows, out_dir / "scoring_candidates.csv")
    write_json(summary, out_dir / "scoring_summary.json")

    print("\n" + "=" * 100)
    print("SCORING INSPECT SUMMARY")
    print("=" * 100)
    print(f"experiment: {args.experiment}")
    print(f"out_dir: {out_dir}")
    print(f"left_leaf: {len(left_leaf)}")
    print(f"right_leaf: {len(right_leaf)}")
    print(f"candidates: {len(candidates)}")
    print(f"soft_calls: {summary['soft_calls']}")
    print(f"soft_fallbacks: {summary['soft_fallbacks']}")
    print(f"step5_total_ms: {summary['timing_ms']['step5_total_ms']}")
    print(f"step5_avg_ms: {summary['timing_ms']['step5_avg_ms']}")
    print(f"step5_median_ms: {summary['timing_ms']['step5_median_ms']}")
    print(f"step5_p95_ms: {summary['timing_ms']['step5_p95_ms']}")
    print(f"step5_max_ms: {summary['timing_ms']['step5_max_ms']}")
    print(f"soft_avg_ms: {summary['timing_ms']['soft_avg_ms']}")
    print(f"soft_max_ms: {summary['timing_ms']['soft_max_ms']}")

    print("\nTop slowest pairs:")
    for i, item in enumerate(summary["top_slowest_pairs"][:10], 1):
        print(
            f"{i:02d}. "
            f"{item['left_corpus']}:{item['left_id']} -> "
            f"{item['right_corpus']}:{item['right_id']} | "
            f"step5={item['step5_total_ms']} ms | "
            f"borrow={item['t_compute_borrow_score_ms']} ms | "
            f"soft={item['t_soft_cosine_ms']} ms | "
            f"soft_called={item['soft_called']} | "
            f"fallback={item['soft_fallback']} | "
            f"union={item['union_unique_lemmas']} | "
            f"shared={item['shared_unique_lemmas']} | "
            f"status={item['status']}"
        )

    print("\nFiles written:")
    print(f"  {out_dir / 'scoring_candidates.csv'}")
    print(f"  {out_dir / 'scoring_summary.json'}")


if __name__ == "__main__":
    main()