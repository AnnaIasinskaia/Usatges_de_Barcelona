#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_source_pair_strategies.py

Отдельный стенд для проверки двух radically different способов работы
с уже посчитанными метриками:

1) Rank aggregation
2) Pareto frontier

ВАЖНО:
- ничего не пересчитывает;
- читает уже сохранённые pair-checkpoint'ы;
- работает только с тем, что уже лежит в checkpoint_metric_lab/pairs/*.pkl;
- на каждую пару источников выводит РОВНО те кандидаты, которые уже были сохранены
  в checkpoint'е (обычно их 5, т.к. global-top5-per-source-pair);
- не печатает N разных топов по каждой метрике.

Идея отчёта:
для каждой пары источников мы берём уже сохранённые 5 кандидатов и:
- считаем ranks по 4 метрикам;
- строим rank-aggregation score/rank;
- строим Pareto layers;
- печатаем одну компактную таблицу на 5 кандидатов:
    * ranks по метрикам
    * aggregate rank
    * pareto layer / on_front
    * ключевые raw metrics

Это не replacement старого стенда, а отдельный аналитический взгляд на те же самые checkpoint'ы.
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

CHECKPOINT_DIRNAME = "checkpoint_metric_lab"
PAIR_DIRNAME = "pairs"


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------

@dataclass
class StrategyRow:
    left_id: str
    right_id: str
    left_corpus: str
    right_corpus: str

    cos_sim: float
    tesserae: float
    soft_cos: float
    sw_score: float
    sw_norm_minlen: float

    shared_unique_lemmas: int
    union_unique_lemmas: int
    shared_ratio_union: float
    t_total_ms: float

    left_snippet: str
    right_snippet: str

    rank_cos: int = 0
    rank_tess: int = 0
    rank_soft: int = 0
    rank_sw: int = 0

    rank_sum: int = 0
    rank_mean: float = 0.0
    rank_borda: float = 0.0
    rank_final_position: int = 0

    pareto_layer: int = 0
    pareto_on_front: bool = False


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_root(out_path: Path) -> Path:
    return out_path.parent / CHECKPOINT_DIRNAME


def pair_checkpoint_dir(out_path: Path) -> Path:
    return checkpoint_root(out_path) / PAIR_DIRNAME


# --------------------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------------------

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def rank_desc(values: List[float]) -> List[int]:
    """
    Dense-ish positional ranking with stable order by original index.
    Highest value -> rank 1.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: (-float(x[1]), x[0]))
    ranks = [0] * len(values)
    for pos, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = pos
    return ranks


# --------------------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------------------

def dominates(a: StrategyRow, b: StrategyRow) -> bool:
    """
    a dominates b iff a is >= in all metrics and > in at least one metric.
    We use:
        cos_sim
        tesserae
        soft_cos
        sw_norm_minlen
    """
    av = [a.cos_sim, a.tesserae, a.soft_cos, a.sw_norm_minlen]
    bv = [b.cos_sim, b.tesserae, b.soft_cos, b.sw_norm_minlen]

    ge_all = all(x >= y for x, y in zip(av, bv))
    gt_any = any(x > y for x, y in zip(av, bv))
    return ge_all and gt_any


def pareto_layers(rows: List[StrategyRow]) -> None:
    """
    Assign Pareto layer numbers in-place:
    layer 1 = frontier, layer 2 = frontier after removing layer 1, etc.
    """
    remaining = list(range(len(rows)))
    layer = 1

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(rows[j], rows[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            rows[i].pareto_layer = layer
            rows[i].pareto_on_front = (layer == 1)

        remaining = [i for i in remaining if i not in set(front)]
        layer += 1


# --------------------------------------------------------------------------------------
# Strategy computation
# --------------------------------------------------------------------------------------

def convert_checkpoint_rows(raw_rows: List[Any]) -> List[StrategyRow]:
    out: List[StrategyRow] = []
    for r in raw_rows:
        # r is expected to be PairMetricRow from the saved checkpoint
        out.append(
            StrategyRow(
                left_id=r.left_id,
                right_id=r.right_id,
                left_corpus=r.left_corpus,
                right_corpus=r.right_corpus,

                cos_sim=float(r.cos_sim),
                tesserae=float(r.tesserae),
                soft_cos=float(r.soft_cos),
                sw_score=float(r.sw_score),
                sw_norm_minlen=float(r.sw_norm_minlen),

                shared_unique_lemmas=int(r.shared_unique_lemmas),
                union_unique_lemmas=int(r.union_unique_lemmas),
                shared_ratio_union=float(r.shared_ratio_union),
                t_total_ms=float(r.t_total_ms),

                left_snippet=str(r.left_snippet),
                right_snippet=str(r.right_snippet),
            )
        )
    return out


def apply_rank_aggregation(rows: List[StrategyRow]) -> None:
    cos_ranks = rank_desc([r.cos_sim for r in rows])
    tess_ranks = rank_desc([r.tesserae for r in rows])
    soft_ranks = rank_desc([r.soft_cos for r in rows])
    sw_ranks = rank_desc([r.sw_norm_minlen for r in rows])

    n = len(rows)
    for i, r in enumerate(rows):
        r.rank_cos = cos_ranks[i]
        r.rank_tess = tess_ranks[i]
        r.rank_soft = soft_ranks[i]
        r.rank_sw = sw_ranks[i]

        r.rank_sum = r.rank_cos + r.rank_tess + r.rank_soft + r.rank_sw
        r.rank_mean = r.rank_sum / 4.0

        # Simple Borda-like score: bigger is better
        # If n=5 and rank=1 -> +5, rank=5 -> +1
        r.rank_borda = float((n + 1 - r.rank_cos) +
                             (n + 1 - r.rank_tess) +
                             (n + 1 - r.rank_soft) +
                             (n + 1 - r.rank_sw))

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i].rank_sum,
            -rows[i].rank_borda,
            rows[i].rank_mean,
            -rows[i].tesserae,
            -rows[i].sw_norm_minlen,
            -rows[i].soft_cos,
            -rows[i].cos_sim,
        )
    )
    for pos, i in enumerate(order, 1):
        rows[i].rank_final_position = pos


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def summarize_global(rows: List[StrategyRow]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0,
            "front1_share": 0.0,
            "rank1_share": 0.0,
            "avg_pareto_layer": 0.0,
        }

    return {
        "n_rows": len(rows),
        "front1_share": sum(1 for r in rows if r.pareto_on_front) / len(rows),
        "rank1_share": sum(1 for r in rows if r.rank_final_position == 1) / len(rows),
        "avg_pareto_layer": mean([float(r.pareto_layer) for r in rows]),
        "cos_mean": mean([r.cos_sim for r in rows]),
        "tess_mean": mean([r.tesserae for r in rows]),
        "soft_mean": mean([r.soft_cos for r in rows]),
        "sw_norm_mean": mean([r.sw_norm_minlen for r in rows]),
    }


def write_pair_section(f, pair_result: Dict[str, Any]) -> None:
    left = pair_result["left_corpus"]
    right = pair_result["right_corpus"]
    rows: List[StrategyRow] = pair_result["strategy_rows"]

    f.write(f"\nSOURCE PAIR: {left} x {right}\n")
    f.write("=" * (14 + len(left) + len(right)) + "\n")
    f.write(f"left_segments   = {pair_result['n_left_segments']}\n")
    f.write(f"right_segments  = {pair_result['n_right_segments']}\n")
    f.write(f"saved_candidates= {pair_result['n_candidates']}\n")
    f.write(f"rows_loaded     = {len(rows)}\n")

    f.write("\nInterpretation:\n")
    f.write("  rank_final_position = итоговый порядок по rank aggregation\n")
    f.write("  pareto_layer=1      = Pareto frontier\n")
    f.write("  rank_*              = место кандидата по отдельной метрике внутри этой пары источников\n")

    # One compact table for ALL 5 rows — no N x 5 blocks
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.rank_final_position,
            r.pareto_layer,
            -r.tesserae,
            -r.sw_norm_minlen,
        )
    )

    f.write("\nCANDIDATES (all saved rows for this source pair)\n")
    f.write("-" * 48 + "\n")

    for i, r in enumerate(rows_sorted, 1):
        f.write(
            f"{i:02d}. rankAgg={r.rank_final_position} | "
            f"pareto_layer={r.pareto_layer} | front={str(r.pareto_on_front):5s} | "
            f"ranks[c,t,s,sw]=[{r.rank_cos},{r.rank_tess},{r.rank_soft},{r.rank_sw}] | "
            f"rank_sum={r.rank_sum:2d} | borda={fmt(r.rank_borda,1)}\n"
        )
        f.write(
            f"    {r.left_id} -> {r.right_id}\n"
            f"    cos={fmt(r.cos_sim)} tess={fmt(r.tesserae)} soft={fmt(r.soft_cos)} "
            f"sw={fmt(r.sw_score)} sw_norm={fmt(r.sw_norm_minlen)} | "
            f"shared={r.shared_unique_lemmas}/{r.union_unique_lemmas} "
            f"ratio={fmt(r.shared_ratio_union)} | time={fmt(r.t_total_ms,1)}ms\n"
        )
        f.write(f"    left : {r.left_snippet}\n")
        f.write(f"    right: {r.right_snippet}\n")

    # Tiny local summary
    front_count = sum(1 for r in rows if r.pareto_on_front)
    rank1 = next((r for r in rows if r.rank_final_position == 1), None)

    f.write("\nLOCAL SUMMARY\n")
    f.write("-" * 13 + "\n")
    f.write(f"pareto_front_size = {front_count}\n")
    if rank1:
        f.write(
            f"rank_aggregation_winner = {rank1.left_id} -> {rank1.right_id} "
            f"(tess={fmt(rank1.tesserae)}, soft={fmt(rank1.soft_cos)}, sw_norm={fmt(rank1.sw_norm_minlen)})\n"
        )
    if front_count > 0:
        front_rows = [r for r in rows if r.pareto_on_front]
        f.write("pareto_front_members:\n")
        for r in front_rows:
            f.write(f"  - {r.left_id} -> {r.right_id}\n")


def write_report(out_path: Path, pair_results: List[Dict[str, Any]]) -> None:
    all_rows: List[StrategyRow] = []
    for pr in pair_results:
        all_rows.extend(pr["strategy_rows"])

    global_summary = summarize_global(all_rows)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("RANK AGGREGATION + PARETO FRONTIER REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("WHAT THIS REPORT DOES\n")
        f.write("-" * 22 + "\n")
        f.write("1) loads already saved pair checkpoints\n")
        f.write("2) does NOT recompute metrics\n")
        f.write("3) compares two approaches over the same saved candidate rows:\n")
        f.write("   - Rank aggregation\n")
        f.write("   - Pareto frontier\n")
        f.write("4) prints only the saved rows per source-pair (typically 5), not N x 5 blocks\n\n")

        f.write("GLOBAL SUMMARY\n")
        f.write("-" * 14 + "\n")
        for k, v in global_summary.items():
            if isinstance(v, float):
                f.write(f"{k:20s} = {fmt(v)}\n")
            else:
                f.write(f"{k:20s} = {v}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PER SOURCE-PAIR ANALYSIS\n")
        f.write("=" * 80 + "\n")

        for pr in pair_results:
            write_pair_section(f, pr)

        f.write("\n\nEND OF REPORT\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def load_all_pair_checkpoints(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(checkpoint_dir.glob("*.pkl"))
    out = []
    for fp in files:
        obj = load_pickle(fp)
        out.append(obj)
    return out


def build_strategy_view(pair_obj: Dict[str, Any]) -> Dict[str, Any]:
    strategy_rows = convert_checkpoint_rows(pair_obj["rows"])
    apply_rank_aggregation(strategy_rows)
    pareto_layers(strategy_rows)

    return {
        "left_corpus": pair_obj["left_corpus"],
        "right_corpus": pair_obj["right_corpus"],
        "n_left_segments": pair_obj["n_left_segments"],
        "n_right_segments": pair_obj["n_right_segments"],
        "n_candidates": pair_obj["n_candidates"],
        "strategy_rows": strategy_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-report",
        default="metric_lab_report.txt",
        help="Path to the original metric-lab TXT report. Used only to locate checkpoint_metric_lab/",
    )
    ap.add_argument(
        "--out",
        default="metric_strategy_report.txt",
        help="Output TXT report for rank aggregation + Pareto frontier",
    )
    args = ap.parse_args()

    base_report = Path(args.base_report)
    pair_dir = pair_checkpoint_dir(base_report)

    if not pair_dir.exists():
        raise FileNotFoundError(
            f"Pair checkpoint dir not found: {pair_dir}\n"
            f"Expected existing checkpoints produced by inspect_source_pair_metrics_checkpointed.py"
        )

    raw_pair_objs = load_all_pair_checkpoints(pair_dir)
    pair_results = [build_strategy_view(obj) for obj in raw_pair_objs]

    out_path = Path(args.out)
    write_report(out_path, pair_results)

    print(f"Loaded pair checkpoints from: {pair_dir}")
    print(f"Wrote strategy report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
