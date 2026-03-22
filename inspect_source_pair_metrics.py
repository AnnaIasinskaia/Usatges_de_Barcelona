#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_source_pair_metrics_checkpointed.py

Большой единый стенд для анализа metric-space по всем парам источников.

Новая версия:
- НЕ использует EXPERIMENTS
- режет сравнения так: top1-per-left -> global-top5-per-source-pair
- добавляет локальные checkpoint'ы:
    * по каждому корпусу
    * по каждой паре источников
- умеет продолжать работу после падения
- пишет один большой TXT-отчет

Checkpoint layout:
    checkpoint_metric_lab/
        corpora/
            <corpus_id>.pkl
        pairs/
            <left>__VS__<right>.pkl
"""

from __future__ import annotations

import argparse
import importlib
import math
import pickle
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from preprocessing import LatinLemmatizer
from pipeline_unified import (
    Segment,
    ProgressLogger,
    segment_corpus,
    chunk_segments,
    lemmatize_segments,
    maybe_align,
)
from features import (
    build_tfidf_matrix,
    compute_idf,
    select_tfidf_candidates,
    tesserae_score,
    soft_cosine_similarity,
)

# --------------------------------------------------------------------------------------
# Global knobs: intentionally simple
# --------------------------------------------------------------------------------------

RETRIEVAL_TOPK_PER_LEFT = 1          # first-stage local pruning
GLOBAL_TOP_PER_SOURCE_PAIR = 5       # second-stage global pruning
REPORT_TOP_N_PER_SOURCE_PAIR = 5     # how many examples print per section

CHUNKING_ENABLED = False
CHUNK_WINDOW_WORDS = 180
CHUNK_OVERLAP_WORDS = 60
CHUNK_MIN_WORDS = 20

WRITE_ALIGNMENT_PREVIEW_TOKENS = 25

DEFAULT_SW_MODEL = {
    "sw_match": 2,
    "sw_mismatch": -1,
    "sw_gap": -1,
    "sw_lev_bonus_threshold": 2,
    "sw_max_seq_len": 300,
}

DEFAULT_SOFT_MAX_TERMS = 500
DEFAULT_MIN_LEMMA_LENGTH = 3

CHECKPOINT_DIRNAME = "checkpoint_metric_lab"


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------

@dataclass
class CorpusPrepared:
    corpus_id: str
    segments: List[Segment]
    lemmas_by_id: Dict[str, List[str]]


@dataclass
class PairMetricRow:
    left_id: str
    right_id: str
    left_parent_id: str
    right_parent_id: str
    left_corpus: str
    right_corpus: str

    cos_sim: float
    tesserae: float
    soft_cos: float
    soft_fallback: bool
    sw_score: float

    left_n_lemmas: int
    right_n_lemmas: int
    left_unique_lemmas: int
    right_unique_lemmas: int
    shared_unique_lemmas: int
    union_unique_lemmas: int
    shared_ratio_union: float
    shared_ratio_min_side: float

    sw_alignment_len: int
    sw_exact_matches: int
    sw_gap_count: int
    sw_left_coverage: float
    sw_right_coverage: float
    sw_norm_minlen: float
    sw_norm_maxlen: float
    sw_norm_alignlen: float

    t_tess_ms: float
    t_soft_ms: float
    t_sw_ms: float
    t_total_ms: float

    left_snippet: str
    right_snippet: str

    sw_alignment_a_preview: List[str]
    sw_alignment_b_preview: List[str]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def short_text(text: str, limit: int = 220) -> str:
    return text[:limit].replace("\n", " ").strip()


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def token_stats(tokens: Sequence[str]) -> Dict[str, int]:
    return {
        "n_tokens": len(tokens),
        "n_unique": len(set(tokens)),
    }


def shared_stats(left_lem: List[str], right_lem: List[str]) -> Dict[str, float]:
    ul = set(left_lem)
    ur = set(right_lem)
    shared = ul & ur
    union = ul | ur
    return {
        "shared_unique": len(shared),
        "union_unique": len(union),
        "shared_ratio_union": (len(shared) / len(union)) if union else 0.0,
        "shared_ratio_min_side": (len(shared) / max(1, min(len(ul), len(ur)))) if ul and ur else 0.0,
    }


def flatten_alignment(al_a: List[str], al_b: List[str]) -> Dict[str, Any]:
    exact = sum(1 for x, y in zip(al_a, al_b) if x == y and x != "-" and y != "-")
    gaps = sum(1 for x, y in zip(al_a, al_b) if x == "-" or y == "-")
    return {
        "alignment_len": len(al_a),
        "exact_matches": exact,
        "gap_count": gaps,
        "a_preview": al_a[:WRITE_ALIGNMENT_PREVIEW_TOKENS],
        "b_preview": al_b[:WRITE_ALIGNMENT_PREVIEW_TOKENS],
    }


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    return float(num / (denx * deny))


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


def checkpoint_root(out_path: Path) -> Path:
    return out_path.parent / CHECKPOINT_DIRNAME


def corpus_ckpt_path(out_path: Path, corpus_id: str) -> Path:
    root = checkpoint_root(out_path) / "corpora"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_name(corpus_id)}.pkl"


def pair_ckpt_path(out_path: Path, left_id: str, right_id: str) -> Path:
    root = checkpoint_root(out_path) / "pairs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_name(left_id)}__VS__{safe_name(right_id)}.pkl"


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------
# Corpus selection
# --------------------------------------------------------------------------------------

def load_registry() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    cfg = importlib.import_module("config_unified")
    corpora = dict(getattr(cfg, "CORPORA"))
    groups = dict(getattr(cfg, "GROUPS", {}))
    return corpora, groups


def parse_sources_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def select_corpora(
    corpora: Dict[str, Dict[str, Any]],
    groups: Dict[str, List[str]],
    sources: List[str],
    left_group: Optional[str],
    right_group: Optional[str],
    left_sources: List[str],
    right_sources: List[str],
) -> Tuple[List[str], List[str]]:
    all_ids = sorted(corpora.keys())

    if not sources and not left_group and not right_group and not left_sources and not right_sources:
        return all_ids, all_ids

    if sources:
        chosen = [x for x in sources if x in corpora]
        return chosen, chosen

    left: List[str] = []
    right: List[str] = []

    if left_group:
        left.extend(groups.get(left_group, []))
    if right_group:
        right.extend(groups.get(right_group, []))

    left.extend([x for x in left_sources if x in corpora])
    right.extend([x for x in right_sources if x in corpora])

    if not left:
        left = all_ids
    if not right:
        right = all_ids

    def dedup(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return dedup(left), dedup(right)


def generate_source_pairs(left_ids: List[str], right_ids: List[str]) -> List[Tuple[str, str]]:
    if left_ids == right_ids:
        out = []
        for i, a in enumerate(left_ids):
            for b in left_ids[i+1:]:
                out.append((a, b))
        return out

    out = []
    for a in left_ids:
        for b in right_ids:
            if a != b:
                out.append((a, b))
    return out


# --------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------

def maybe_chunk(segments: List[Segment], logger: ProgressLogger, side: str) -> List[Segment]:
    if not CHUNKING_ENABLED:
        return segments
    chunk_cfg = {
        "enabled": True,
        "mode": "sliding_window_words",
        "window_words": CHUNK_WINDOW_WORDS,
        "overlap_words": CHUNK_OVERLAP_WORDS,
        "min_words": CHUNK_MIN_WORDS,
        "per_corpus": {},
    }
    return chunk_segments(segments, chunk_cfg, logger, side)


def prepare_one_corpus(
    corpus_id: str,
    corpora: Dict[str, Dict[str, Any]],
    logger: ProgressLogger,
) -> CorpusPrepared:
    segs_raw = segment_corpus(corpus_id, corpora[corpus_id], logger)
    segs = [
        Segment(
            id=seg_id,
            text=seg_text,
            corpus=corpus_id,
            side="generic",
            parent_id=seg_id,
        )
        for seg_id, seg_text in segs_raw
    ]
    segs = maybe_chunk(segs, logger, side="generic")
    lemmatizer = LatinLemmatizer()
    lemmas = lemmatize_segments(
        segs,
        lemmatizer,
        DEFAULT_MIN_LEMMA_LENGTH,
        logger,
        corpus_id,
        progress_every=None,
    )
    return CorpusPrepared(
        corpus_id=corpus_id,
        segments=segs,
        lemmas_by_id=lemmas,
    )


def prepare_all_corpora(
    selected_corpora: Sequence[str],
    corpora: Dict[str, Dict[str, Any]],
    logger: ProgressLogger,
    out_path: Path,
    force_rebuild: bool = False,
) -> Dict[str, CorpusPrepared]:
    prepared: Dict[str, CorpusPrepared] = {}

    for corpus_id in selected_corpora:
        ckpt = corpus_ckpt_path(out_path, corpus_id)

        if ckpt.exists() and not force_rebuild:
            logger.log(f"Loading corpus checkpoint: {corpus_id}")
            prepared[corpus_id] = load_pickle(ckpt)
            logger.log(
                f"Loaded {corpus_id}: segments={len(prepared[corpus_id].segments)}, "
                f"nonempty_lemmas={sum(1 for x in prepared[corpus_id].lemmas_by_id.values() if x)}"
            )
            continue

        logger.log(f"Preparing corpus from scratch: {corpus_id}")
        obj = prepare_one_corpus(corpus_id, corpora, logger)
        save_pickle(obj, ckpt)
        logger.log(
            f"Saved corpus checkpoint: {corpus_id} -> {ckpt.name} | "
            f"segments={len(obj.segments)}, nonempty_lemmas={sum(1 for x in obj.lemmas_by_id.values() if x)}"
        )
        prepared[corpus_id] = obj

    return prepared


# --------------------------------------------------------------------------------------
# Retrieval stage per source pair
# --------------------------------------------------------------------------------------

def build_candidates_for_source_pair(
    left_corpus: CorpusPrepared,
    right_corpus: CorpusPrepared,
    logger: ProgressLogger,
) -> Tuple[
    List[Tuple[str, str, float]],
    Dict[str, float],
    Dict[str, Segment],
    Dict[str, Segment],
]:
    left_ids = [s.id for s in left_corpus.segments]
    right_ids = [s.id for s in right_corpus.segments]

    corpus_lemmas = (
        [left_corpus.lemmas_by_id[i] for i in left_ids] +
        [right_corpus.lemmas_by_id[i] for i in right_ids]
    )

    tfidf, vocab, term2idx = build_tfidf_matrix(
        corpus_lemmas,
        ngram_range=(1, 3),
        max_df=0.5,
        min_df=2,
    )

    n_left = len(left_ids)
    tfidf_left = tfidf[:n_left]
    tfidf_right = tfidf[n_left:]

    full_pairs = len(left_ids) * len(right_ids)

    local_candidates = select_tfidf_candidates(
        tfidf_left,
        tfidf_right,
        left_ids=left_ids,
        right_ids=right_ids,
        threshold=0.0,
        top_k_per_left=RETRIEVAL_TOPK_PER_LEFT,
        progress_every=None,
        progress_callback=None,
    )

    local_candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = local_candidates[:GLOBAL_TOP_PER_SOURCE_PAIR]

    idf = compute_idf(corpus_lemmas)
    left_seg_by_id = {s.id: s for s in left_corpus.segments}
    right_seg_by_id = {s.id: s for s in right_corpus.segments}

    logger.log(f"Retrieval {left_corpus.corpus_id} x {right_corpus.corpus_id}:")
    logger.log(f"  left={len(left_ids)}, right={len(right_ids)}, full_pairs={full_pairs}")
    logger.log(
        f"  local_topk_per_left={RETRIEVAL_TOPK_PER_LEFT} "
        f"-> {len(local_candidates)} candidates"
    )
    logger.log(
        f"  global_top_per_source_pair={GLOBAL_TOP_PER_SOURCE_PAIR} "
        f"-> {len(candidates)} candidates kept"
    )

    return candidates, idf, left_seg_by_id, right_seg_by_id


# --------------------------------------------------------------------------------------
# Metric extraction per candidate
# --------------------------------------------------------------------------------------

def extract_metrics_for_candidate(
    left_id: str,
    right_id: str,
    cos_sim: float,
    idf: Dict[str, float],
    left_corpus: CorpusPrepared,
    right_corpus: CorpusPrepared,
    left_seg_by_id: Dict[str, Segment],
    right_seg_by_id: Dict[str, Segment],
) -> PairMetricRow:
    t0 = now_ms()

    ls = left_seg_by_id[left_id]
    rs = right_seg_by_id[right_id]
    llem = left_corpus.lemmas_by_id.get(left_id, [])
    rlem = right_corpus.lemmas_by_id.get(right_id, [])

    lstat = token_stats(llem)
    rstat = token_stats(rlem)
    sstat = shared_stats(llem, rlem)

    t_tess0 = now_ms()
    tess = float(tesserae_score(llem, rlem, idf))
    t_tess1 = now_ms()

    union_unique = len(set(llem) | set(rlem))
    soft_fallback = union_unique > DEFAULT_SOFT_MAX_TERMS

    t_soft0 = now_ms()
    soft = float(soft_cosine_similarity(llem, rlem, max_terms=DEFAULT_SOFT_MAX_TERMS))
    t_soft1 = now_ms()

    t_sw0 = now_ms()
    al_a, al_b, sw = maybe_align(llem, rlem, DEFAULT_SW_MODEL, enabled=True)
    t_sw1 = now_ms()

    ali = flatten_alignment(al_a, al_b)
    min_side = max(1, min(len(llem), len(rlem)))
    max_side = max(1, max(len(llem), len(rlem)))

    row = PairMetricRow(
        left_id=left_id,
        right_id=right_id,
        left_parent_id=ls.parent_id,
        right_parent_id=rs.parent_id,
        left_corpus=ls.corpus,
        right_corpus=rs.corpus,

        cos_sim=float(cos_sim),
        tesserae=tess,
        soft_cos=soft,
        soft_fallback=soft_fallback,
        sw_score=float(sw),

        left_n_lemmas=lstat["n_tokens"],
        right_n_lemmas=rstat["n_tokens"],
        left_unique_lemmas=lstat["n_unique"],
        right_unique_lemmas=rstat["n_unique"],
        shared_unique_lemmas=int(sstat["shared_unique"]),
        union_unique_lemmas=int(sstat["union_unique"]),
        shared_ratio_union=float(sstat["shared_ratio_union"]),
        shared_ratio_min_side=float(sstat["shared_ratio_min_side"]),

        sw_alignment_len=int(ali["alignment_len"]),
        sw_exact_matches=int(ali["exact_matches"]),
        sw_gap_count=int(ali["gap_count"]),
        sw_left_coverage=float(ali["alignment_len"] / max(1, len(llem))),
        sw_right_coverage=float(ali["alignment_len"] / max(1, len(rlem))),
        sw_norm_minlen=float(sw) / min_side,
        sw_norm_maxlen=float(sw) / max_side,
        sw_norm_alignlen=float(sw) / max(1, ali["alignment_len"]),

        t_tess_ms=round(t_tess1 - t_tess0, 3),
        t_soft_ms=round(t_soft1 - t_soft0, 3),
        t_sw_ms=round(t_sw1 - t_sw0, 3),
        t_total_ms=round(now_ms() - t0, 3),

        left_snippet=short_text(ls.text),
        right_snippet=short_text(rs.text),

        sw_alignment_a_preview=list(ali["a_preview"]),
        sw_alignment_b_preview=list(ali["b_preview"]),
    )
    return row


def analyze_source_pair(
    left_corpus: CorpusPrepared,
    right_corpus: CorpusPrepared,
    logger: ProgressLogger,
) -> Dict[str, Any]:
    candidates, idf, left_seg_by_id, right_seg_by_id = build_candidates_for_source_pair(left_corpus, right_corpus, logger)

    metric_rows: List[PairMetricRow] = []
    for left_id, right_id, cos_sim in candidates:
        row = extract_metrics_for_candidate(
            left_id=left_id,
            right_id=right_id,
            cos_sim=cos_sim,
            idf=idf,
            left_corpus=left_corpus,
            right_corpus=right_corpus,
            left_seg_by_id=left_seg_by_id,
            right_seg_by_id=right_seg_by_id,
        )
        metric_rows.append(row)

    return {
        "left_corpus": left_corpus.corpus_id,
        "right_corpus": right_corpus.corpus_id,
        "n_left_segments": len(left_corpus.segments),
        "n_right_segments": len(right_corpus.segments),
        "n_candidates": len(candidates),
        "rows": metric_rows,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def rows_by_metric(rows: List[PairMetricRow], metric_name: str, top_n: int) -> List[PairMetricRow]:
    return sorted(rows, key=lambda r: getattr(r, metric_name), reverse=True)[:top_n]


def disagreement_soft_vs_tess(rows: List[PairMetricRow], top_n: int) -> List[PairMetricRow]:
    return sorted(rows, key=lambda r: (r.soft_cos - r.tesserae), reverse=True)[:top_n]


def disagreement_sw_vs_cos(rows: List[PairMetricRow], top_n: int) -> List[PairMetricRow]:
    return sorted(rows, key=lambda r: (r.sw_norm_minlen - r.cos_sim), reverse=True)[:top_n]


def summarize_rows(rows: List[PairMetricRow]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "cos_mean": 0.0,
            "tess_mean": 0.0,
            "soft_mean": 0.0,
            "sw_mean": 0.0,
            "soft_fallbacks": 0,
            "time_mean_ms": 0.0,
        }

    cos_vals = [r.cos_sim for r in rows]
    tess_vals = [r.tesserae for r in rows]
    soft_vals = [r.soft_cos for r in rows]
    sw_vals = [r.sw_score for r in rows]
    time_vals = [r.t_total_ms for r in rows]

    return {
        "n": len(rows),
        "cos_mean": mean(cos_vals),
        "cos_median": median(cos_vals),
        "cos_max": max(cos_vals),

        "tess_mean": mean(tess_vals),
        "tess_median": median(tess_vals),
        "tess_max": max(tess_vals),

        "soft_mean": mean(soft_vals),
        "soft_median": median(soft_vals),
        "soft_max": max(soft_vals),

        "sw_mean": mean(sw_vals),
        "sw_median": median(sw_vals),
        "sw_max": max(sw_vals),

        "soft_fallbacks": sum(1 for r in rows if r.soft_fallback),
        "time_mean_ms": mean(time_vals),
        "time_p95_ms": pct(time_vals, 0.95),
        "time_max_ms": max(time_vals),
    }


def global_correlation_report(all_rows: List[PairMetricRow]) -> List[Tuple[str, str, float]]:
    if len(all_rows) < 2:
        return []

    metrics = {
        "cos_sim": [r.cos_sim for r in all_rows],
        "tesserae": [r.tesserae for r in all_rows],
        "soft_cos": [r.soft_cos for r in all_rows],
        "sw_score": [r.sw_score for r in all_rows],
        "sw_norm_minlen": [r.sw_norm_minlen for r in all_rows],
        "shared_ratio_union": [r.shared_ratio_union for r in all_rows],
        "shared_unique_lemmas": [float(r.shared_unique_lemmas) for r in all_rows],
        "union_unique_lemmas": [float(r.union_unique_lemmas) for r in all_rows],
        "t_total_ms": [r.t_total_ms for r in all_rows],
    }

    out: List[Tuple[str, str, float]] = []
    keys = list(metrics.keys())
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            out.append((a, b, pearson(metrics[a], metrics[b])))

    out.sort(key=lambda x: abs(x[2]), reverse=True)
    return out


def write_row_block(f, title: str, rows: List[PairMetricRow]) -> None:
    f.write(f"\n{title}\n")
    f.write("-" * len(title) + "\n")

    if not rows:
        f.write("  [no rows]\n")
        return

    for i, r in enumerate(rows, 1):
        f.write(
            f"{i:02d}. {r.left_id} -> {r.right_id} | "
            f"cos={fmt(r.cos_sim)} tess={fmt(r.tesserae)} soft={fmt(r.soft_cos)} "
            f"sw={fmt(r.sw_score)} sw_norm={fmt(r.sw_norm_minlen)} | "
            f"shared={r.shared_unique_lemmas}/{r.union_unique_lemmas} | "
            f"time={fmt(r.t_total_ms, 1)}ms\n"
        )
        f.write(f"    left : {r.left_snippet}\n")
        f.write(f"    right: {r.right_snippet}\n")
        f.write(
            f"    sizes: left_lem={r.left_n_lemmas}, right_lem={r.right_n_lemmas}, "
            f"left_u={r.left_unique_lemmas}, right_u={r.right_unique_lemmas}, "
            f"shared_ratio_union={fmt(r.shared_ratio_union)}\n"
        )
        f.write(
            f"    SW: align_len={r.sw_alignment_len}, exact={r.sw_exact_matches}, gaps={r.sw_gap_count}, "
            f"covL={fmt(r.sw_left_coverage)}, covR={fmt(r.sw_right_coverage)}\n"
        )
        f.write(f"    SW A: {' '.join(r.sw_alignment_a_preview)}\n")
        f.write(f"    SW B: {' '.join(r.sw_alignment_b_preview)}\n")


def write_report(
    output_path: Path,
    pair_results: List[Dict[str, Any]],
    selected_left: List[str],
    selected_right: List[str],
    source_pairs: List[Tuple[str, str]],
) -> None:
    all_rows: List[PairMetricRow] = []
    for item in pair_results:
        all_rows.extend(item["rows"])

    global_corr = global_correlation_report(all_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("UNIFIED SOURCE-PAIR METRIC LAB REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("SETUP\n")
        f.write("-" * 5 + "\n")
        f.write(f"RETRIEVAL_TOPK_PER_LEFT = {RETRIEVAL_TOPK_PER_LEFT}\n")
        f.write(f"GLOBAL_TOP_PER_SOURCE_PAIR = {GLOBAL_TOP_PER_SOURCE_PAIR}\n")
        f.write(f"REPORT_TOP_N_PER_SOURCE_PAIR = {REPORT_TOP_N_PER_SOURCE_PAIR}\n")
        f.write(f"CHUNKING_ENABLED = {CHUNKING_ENABLED}\n")
        f.write(f"DEFAULT_SOFT_MAX_TERMS = {DEFAULT_SOFT_MAX_TERMS}\n")
        f.write(f"DEFAULT_MIN_LEMMA_LENGTH = {DEFAULT_MIN_LEMMA_LENGTH}\n\n")

        f.write(f"LEFT SOURCES ({len(selected_left)}): {', '.join(selected_left)}\n")
        f.write(f"RIGHT SOURCES ({len(selected_right)}): {', '.join(selected_right)}\n")
        f.write(f"SOURCE PAIRS ({len(source_pairs)}):\n")
        for a, b in source_pairs:
            f.write(f"  - {a} x {b}\n")

        f.write("\nGLOBAL SUMMARY\n")
        f.write("-" * 14 + "\n")
        f.write(f"total_pair_sections = {len(pair_results)}\n")
        f.write(f"total_metric_rows   = {len(all_rows)}\n\n")

        f.write("TOP GLOBAL CORRELATIONS\n")
        f.write("-" * 23 + "\n")
        for a, b, corr in global_corr[:20]:
            f.write(f"  {a:20s} vs {b:20s} -> {corr:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PER SOURCE-PAIR REPORT\n")
        f.write("=" * 80 + "\n")

        for item in pair_results:
            left = item["left_corpus"]
            right = item["right_corpus"]
            rows = item["rows"]
            summary = summarize_rows(rows)

            f.write(f"\n\nSOURCE PAIR: {left} x {right}\n")
            f.write("=" * (14 + len(left) + len(right)) + "\n")
            f.write(f"left_segments  = {item['n_left_segments']}\n")
            f.write(f"right_segments = {item['n_right_segments']}\n")
            f.write(f"candidates     = {item['n_candidates']}\n")
            f.write(f"rows           = {summary['n']}\n")
            f.write(
                f"means          = cos:{fmt(summary['cos_mean'])} "
                f"tess:{fmt(summary['tess_mean'])} "
                f"soft:{fmt(summary['soft_mean'])} "
                f"sw:{fmt(summary['sw_mean'])}\n"
            )
            f.write(
                f"maxima         = cos:{fmt(summary['cos_max'])} "
                f"tess:{fmt(summary['tess_max'])} "
                f"soft:{fmt(summary['soft_max'])} "
                f"sw:{fmt(summary['sw_max'])}\n"
            )
            f.write(
                f"timing         = mean:{fmt(summary['time_mean_ms'],1)}ms "
                f"p95:{fmt(summary['time_p95_ms'],1)}ms "
                f"max:{fmt(summary['time_max_ms'],1)}ms\n"
            )
            f.write(f"soft_fallbacks = {summary['soft_fallbacks']}\n")

            write_row_block(f, "TOP BY COSINE", rows_by_metric(rows, "cos_sim", REPORT_TOP_N_PER_SOURCE_PAIR))
            write_row_block(f, "TOP BY TESSERAE", rows_by_metric(rows, "tesserae", REPORT_TOP_N_PER_SOURCE_PAIR))
            write_row_block(f, "TOP BY SOFT COSINE", rows_by_metric(rows, "soft_cos", REPORT_TOP_N_PER_SOURCE_PAIR))
            write_row_block(f, "TOP BY SMITH-WATERMAN", rows_by_metric(rows, "sw_score", REPORT_TOP_N_PER_SOURCE_PAIR))
            write_row_block(f, "TOP SOFT >> TESS DISAGREEMENTS", disagreement_soft_vs_tess(rows, REPORT_TOP_N_PER_SOURCE_PAIR))
            write_row_block(f, "TOP SW >> COS DISAGREEMENTS", disagreement_sw_vs_cos(rows, REPORT_TOP_N_PER_SOURCE_PAIR))

        f.write("\n\nEND OF REPORT\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="", help="Comma-separated source ids")
    ap.add_argument("--left-group", default="", help="Group name from GROUPS")
    ap.add_argument("--right-group", default="", help="Group name from GROUPS")
    ap.add_argument("--left-sources", default="", help="Comma-separated left source ids")
    ap.add_argument("--right-sources", default="", help="Comma-separated right source ids")
    ap.add_argument("--out", default="metric_lab_report.txt", help="Single TXT output")
    ap.add_argument("--force-rebuild-corpora", action="store_true", help="Ignore corpus checkpoints")
    ap.add_argument("--force-rebuild-pairs", action="store_true", help="Ignore pair checkpoints")
    args = ap.parse_args()

    logger = ProgressLogger(enabled=True)
    corpora, groups = load_registry()

    sources = parse_sources_arg(args.sources)
    left_sources = parse_sources_arg(args.left_sources)
    right_sources = parse_sources_arg(args.right_sources)

    selected_left, selected_right = select_corpora(
        corpora=corpora,
        groups=groups,
        sources=sources,
        left_group=args.left_group or None,
        right_group=args.right_group or None,
        left_sources=left_sources,
        right_sources=right_sources,
    )

    selected_all = sorted(set(selected_left + selected_right))
    source_pairs = generate_source_pairs(selected_left, selected_right)

    out_path = Path(args.out)
    checkpoint_root(out_path).mkdir(parents=True, exist_ok=True)

    logger.log(f"Selected corpora: {selected_all}")
    logger.log(f"Source pairs to analyze: {len(source_pairs)}")
    logger.log(f"Checkpoint root: {checkpoint_root(out_path)}")

    prepared = prepare_all_corpora(
        selected_all,
        corpora,
        logger,
        out_path=out_path,
        force_rebuild=args.force_rebuild_corpora,
    )

    pair_results: List[Dict[str, Any]] = []
    for left_id, right_id in source_pairs:
        ckpt = pair_ckpt_path(out_path, left_id, right_id)

        if ckpt.exists() and not args.force_rebuild_pairs:
            logger.log(f"Loading pair checkpoint: {left_id} x {right_id}")
            result = load_pickle(ckpt)
            pair_results.append(result)
            continue

        logger.log(f"Analyzing source pair from scratch: {left_id} x {right_id}")
        result = analyze_source_pair(
            left_corpus=prepared[left_id],
            right_corpus=prepared[right_id],
            logger=logger,
        )
        save_pickle(result, ckpt)
        logger.log(f"Saved pair checkpoint: {left_id} x {right_id} -> {ckpt.name}")
        pair_results.append(result)

    write_report(
        output_path=out_path,
        pair_results=pair_results,
        selected_left=selected_left,
        selected_right=selected_right,
        source_pairs=source_pairs,
    )

    print("\nDone.")
    print(f"Report written to: {out_path.resolve()}")
    print(f"Checkpoints: {checkpoint_root(out_path).resolve()}")


if __name__ == "__main__":
    main()
