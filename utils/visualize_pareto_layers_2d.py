#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_pareto_layers_2d_v4.py

Чистая 2D-визуализация Pareto-слоёв для статьи.
Оси:
- x = tesserae
- y = sw_norm

Особенности:
- НЕ запускает pipeline
- читает только готовый step_05_metrics.pkl
- строит слои как ломаные, а не как набор маркеров
- без лог-шкалы, заливок и декоративных эффектов
"""

from __future__ import annotations

import argparse
import csv
import importlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline as pl


class PipelineCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "Segment":
            return pl.Segment
        if module == "__main__" and name == "CandidateMetrics":
            return pl.CandidateMetrics
        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as f:
        return PipelineCheckpointUnpickler(f).load()


def load_config_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def resolve_experiment_context(config_module: Any, experiment_id: str) -> Dict[str, Any]:
    corpora: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "CORPORA"))
    groups: Dict[str, List[str]] = dict(getattr(config_module, "GROUPS", {}))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "EXPERIMENTS"))

    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")

    exp = dict(experiments[experiment_id])

    left_corpora = pl.resolve_group_tokens(exp["graph_sides"]["left"], groups)
    right_corpora = pl.resolve_group_tokens(exp["graph_sides"]["right"], groups)

    mappings = exp.get("mappings") or [{"from": left_corpora, "to": right_corpora}]
    resolved_mappings: List[Tuple[List[str], List[str]]] = []
    for m in mappings:
        frm = pl.resolve_group_tokens(m.get("from", []), groups)
        to = pl.resolve_group_tokens(m.get("to", []), groups)
        resolved_mappings.append((frm, to))

    model = dict(exp.get("model") or {})
    retrieval_cfg = dict(exp.get("retrieval") or {})
    agg_cfg = dict(exp.get("aggregation") or {})
    pareto_cfg = dict(exp.get("pareto") or {})
    selection_cfg = dict(exp.get("selection") or {})
    chunk_cfg = dict(exp.get("chunking") or {})
    output_cfg = dict(exp.get("output") or {})

    used_corpora = sorted(set(left_corpora + right_corpora))
    step_fingerprints = pl.build_checkpoint_fingerprints(
        experiment_id=experiment_id,
        left_corpora=left_corpora,
        right_corpora=right_corpora,
        resolved_mappings=resolved_mappings,
        chunk_cfg=chunk_cfg,
        model=model,
        retrieval_cfg=retrieval_cfg,
        agg_cfg=agg_cfg,
        pareto_cfg=pareto_cfg,
        selection_cfg=selection_cfg,
        corpora=corpora,
        used_corpora=used_corpora,
    )

    checkpoint_dir = Path(output_cfg.get("dir", ".")) / "checkpoints"

    return {
        "output_cfg": output_cfg,
        "checkpoint_dir": checkpoint_dir,
        "step_fingerprints": step_fingerprints,
    }


def load_checkpoint_strict(
    checkpoint_dir: Path,
    expected_fingerprint: str,
    step_no: int,
    step_key: str,
) -> Any:
    path = checkpoint_dir / f"step_{step_no:02d}_{step_key}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Этот скрипт работает только по готовому эксперименту."
        )

    payload = load_pickle_compat(path)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload type in {path}: {type(payload)}")

    got = payload.get("fingerprint")
    if got != expected_fingerprint:
        raise RuntimeError(
            f"Fingerprint mismatch for {path.name}: expected={expected_fingerprint}, got={got}"
        )

    return payload.get("data")


def to_float_scalar(value: Any, field_name: str, pair_hint: str) -> float:
    v = value
    depth = 0
    while isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise TypeError(
                f"Field {field_name} for pair {pair_hint} is not scalar: {type(value)}={value!r}"
            )
        v = v[0]
        depth += 1
        if depth > 8:
            raise TypeError(
                f"Field {field_name} for pair {pair_hint} is nested too deeply: {value!r}"
            )
    try:
        return float(v)
    except Exception as e:
        raise TypeError(
            f"Field {field_name} for pair {pair_hint} cannot be converted to float: {value!r}"
        ) from e


def metric_row_to_plot_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        raw = dict(row)
        pair_hint = f"{raw.get('left_leaf_id', '?')} -> {raw.get('right_leaf_id', '?')}"
        return {
            "left_leaf_id": str(raw.get("left_leaf_id", "")),
            "right_leaf_id": str(raw.get("right_leaf_id", "")),
            "left_parent_id": str(raw.get("left_parent_id", "")),
            "right_parent_id": str(raw.get("right_parent_id", "")),
            "tesserae": to_float_scalar(raw.get("tesserae", 0.0), "tesserae", pair_hint),
            "sw_norm": to_float_scalar(raw.get("sw_norm", 0.0), "sw_norm", pair_hint),
        }

    pair_hint = f"{getattr(row, 'left_leaf_id', '?')} -> {getattr(row, 'right_leaf_id', '?')}"
    return {
        "left_leaf_id": str(getattr(row, "left_leaf_id")),
        "right_leaf_id": str(getattr(row, "right_leaf_id")),
        "left_parent_id": str(getattr(row, "left_parent_id")),
        "right_parent_id": str(getattr(row, "right_parent_id")),
        "tesserae": to_float_scalar(getattr(row, "tesserae"), "tesserae", pair_hint),
        "sw_norm": to_float_scalar(getattr(row, "sw_norm"), "sw_norm", pair_hint),
    }


class FenwickMax:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = [0] * (self.n + 1)

    def update(self, idx: int, value: int) -> None:
        i = int(idx)
        v = int(value)
        while i <= self.n:
            if v > self.bit[i]:
                self.bit[i] = v
            i += i & -i

    def query_prefix_max(self, idx: int) -> int:
        i = int(idx)
        out = 0
        while i > 0:
            if self.bit[i] > out:
                out = self.bit[i]
            i -= i & -i
        return out


def assign_pareto_layers_2d_fast(coords: Sequence[Tuple[float, float]]) -> List[int]:
    n = len(coords)
    if n == 0:
        return []

    ys_unique = sorted({float(y) for _, y in coords}, reverse=True)
    y_to_rank = {y: i + 1 for i, y in enumerate(ys_unique)}
    fenwick = FenwickMax(len(ys_unique))

    order = sorted(range(n), key=lambda i: (-float(coords[i][0]), -float(coords[i][1]), i))
    layers = [0] * n
    pos = 0

    while pos < n:
        x_val = float(coords[order[pos]][0])
        same_x: List[int] = []
        while pos < n and float(coords[order[pos]][0]) == x_val:
            same_x.append(order[pos])
            pos += 1

        same_x.sort(key=lambda i: (-float(coords[i][1]), i))

        k = 0
        while k < len(same_x):
            y_val = float(coords[same_x[k]][1])
            same_xy: List[int] = []
            while k < len(same_x) and float(coords[same_x[k]][1]) == y_val:
                same_xy.append(same_x[k])
                k += 1

            rank = y_to_rank[y_val]
            layer = fenwick.query_prefix_max(rank) + 1

            for idx in same_xy:
                layers[idx] = layer

            fenwick.update(rank, layer)

    return layers


def build_front_curve(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Для слоя берём уникальные x и оставляем максимальный y для каждого x,
    затем сортируем по x слева направо. Этого достаточно для чистой
    журнальной иллюстрации фронта.
    """
    best_by_x: Dict[float, float] = {}
    for x, y in points:
        x = float(x)
        y = float(y)
        prev = best_by_x.get(x)
        if prev is None or y > prev:
            best_by_x[x] = y

    curve = sorted(best_by_x.items(), key=lambda p: p[0])
    return [(float(x), float(y)) for x, y in curve]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config", help="Python module name, e.g. config")
    ap.add_argument("--experiment", required=True, help="Experiment id from config.EXPERIMENTS")
    ap.add_argument("--max-layers", type=int, default=6, help="How many layers to draw")
    ap.add_argument("--line-width", type=float, default=2.0, help="Line width")
    ap.add_argument("--output-prefix", default=None, help="Custom output path prefix without extension")
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)
    ctx = resolve_experiment_context(cfg_mod, args.experiment)
    checkpoint_dir: Path = ctx["checkpoint_dir"]
    step_fingerprints: Dict[str, str] = ctx["step_fingerprints"]
    output_cfg: Dict[str, Any] = ctx["output_cfg"]

    step5 = load_checkpoint_strict(
        checkpoint_dir,
        step_fingerprints["step_05_metrics"],
        5,
        "metrics",
    )

    rows = [metric_row_to_plot_dict(r) for r in list(step5["metric_rows"])]
    coords = [(float(r["tesserae"]), float(r["sw_norm"])) for r in rows]
    layers = assign_pareto_layers_2d_fast(coords)

    for row, layer in zip(rows, layers):
        row["pareto_layer_2d"] = int(layer)

    rows.sort(
        key=lambda r: (
            int(r["pareto_layer_2d"]),
            -float(r["tesserae"]),
            -float(r["sw_norm"]),
            str(r["left_leaf_id"]),
            str(r["right_leaf_id"]),
        )
    )

    out_dir = Path(output_cfg.get("dir", "."))
    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = out_dir / "pareto_2d_tesserae_vs_sw_norm_clean"

    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    write_csv(rows, csv_path)

    max_layer_seen = max(layers) if layers else 0
    visible_layers = max(1, min(int(args.max_layers), max_layer_seen if max_layer_seen else 1))

    fig, ax = plt.subplots(figsize=(10, 7))

    for layer in range(1, visible_layers + 1):
        pts = [
            (float(r["tesserae"]), float(r["sw_norm"]))
            for r in rows
            if int(r["pareto_layer_2d"]) == layer
        ]
        if not pts:
            continue

        curve = build_front_curve(pts)
        xs = [x for x, _ in curve]
        ys = [y for _, y in curve]

        ax.plot(
            xs,
            ys,
            linewidth=float(args.line_width),
            label=f"layer {layer}",
        )

    ax.set_title(f"2D Pareto layers: {args.experiment} (tesserae vs sw_norm)")
    ax.set_xlabel("tesserae")
    ax.set_ylabel("sw_norm")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[pareto-2d-v4] experiment={args.experiment}")
    print(f"[pareto-2d-v4] points={len(rows)}")
    print(f"[pareto-2d-v4] max_layer={max_layer_seen}")
    print(f"[pareto-2d-v4] csv={csv_path}")
    print(f"[pareto-2d-v4] png={png_path}")


if __name__ == "__main__":
    main()
