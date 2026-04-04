#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_graph_from_csv.py

Рендерит PNG-граф только на основе уже готового graph_rows.csv.
Ничего не пересчитывает: ни сегментацию, ни TF-IDF, ни scoring.

Источник graph_rows.csv по умолчанию берётся из:
  EXPERIMENTS[experiment]["output"]["dir"] / "graph_rows.csv"

Скрипт предназначен для быстрого повторного рендера после изменения
визуальных параметров, без повторного запуска полного pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import time
from pathlib import Path
from typing import Any, Dict, List

from src.graph_rendering import build_node_metadata_from_graph_rows, render_bipartite_graph


class ProgressLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        dt = time.time() - self.t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)


def load_config_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def run_render_from_csv(
    config_module: Any,
    experiment_id: str,
    graph_csv: str | Path | None = None,
    out_png: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    logger = ProgressLogger(enabled=verbose)

    logger.log("Step 1/3: Loading config and graph rows...")

    corpora: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "CORPORA"))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "EXPERIMENTS"))

    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")

    exp = experiments[experiment_id]
    out_dir = Path(exp["output"]["dir"])
    viz_cfg = dict(exp.get("viz") or {})
    agg_cfg = dict(exp.get("aggregation") or {})

    graph_csv_path = Path(graph_csv) if graph_csv else out_dir / "graph_rows.csv"
    out_png_path = Path(out_png) if out_png else out_dir / "graph.png"

    rows = read_csv_rows(graph_csv_path)
    logger.log(f"  Graph rows loaded: {len(rows)} from {graph_csv_path}")

    logger.log("Step 2/3: Building node metadata...")
    # build_node_metadata_from_graph_rows теперь добавляет legend_label,
    # чтобы легенда отображала только реально присутствующие на картинке корпуса.
    left_level = str(agg_cfg.get("left_node_level", "parent"))
    right_level = str(agg_cfg.get("right_node_level", "parent"))
    left_nodes, right_nodes = build_node_metadata_from_graph_rows(
        rows,
        corpora=corpora,
        left_level=left_level,
        right_level=right_level,
    )
    logger.log(f"  Node metadata done: left_nodes={len(left_nodes)}, right_nodes={len(right_nodes)}")

    logger.log("Step 3/3: Rendering PNG from graph_rows.csv...")
    top_n = viz_cfg.get("top_n_edges")
    logger.log(f"  Rendering PNG: {out_png_path.name} (top_n_edges={top_n}, total_graph_rows={len(rows)})")
    rendered_edges = render_bipartite_graph(
        graph_rows=rows,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        out_png=out_png_path,
        straight_edges=bool(viz_cfg.get("straight_edges", True)),
        label_left=bool(viz_cfg.get("label_left", True)),
        label_right=bool(viz_cfg.get("label_right", True)),
        top_n_edges=top_n,
    )
    logger.log(f"  Rendered edges: {rendered_edges}")
    logger.log("Render finished")
    logger.log(
        f"Summary: rows={len(rows)}, left_nodes={len(left_nodes)}, "
        f"right_nodes={len(right_nodes)}, rendered_edges={rendered_edges}"
    )
    logger.log(f"Output PNG: {out_png_path}")

    return {
        "experiment_id": experiment_id,
        "graph_csv": str(graph_csv_path),
        "out_png": str(out_png_path),
        "rendered_edges": int(rendered_edges),
        "rows": len(rows),
        "left_nodes": len(left_nodes),
        "right_nodes": len(right_nodes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_unified", help="Python module name, e.g. config_unified")
    ap.add_argument("--experiment", required=True, help="Experiment id, see EXPERIMENTS in config")
    ap.add_argument("--quiet", action="store_true", help="Disable progress logs")
    ap.add_argument("--graph-csv", default=None, help="Override path to graph_rows.csv")
    ap.add_argument("--out-png", default=None, help="Override output PNG path")
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)
    result = run_render_from_csv(
        config_module=cfg_mod,
        experiment_id=args.experiment,
        graph_csv=args.graph_csv,
        out_png=args.out_png,
        verbose=not args.quiet,
    )

    print(f"[render] experiment={result['experiment_id']}", flush=True)
    print(f"[render] graph_csv={result['graph_csv']}", flush=True)
    print(f"[render] out_png={result['out_png']}", flush=True)
    print(f"[render] rendered_edges={result['rendered_edges']}", flush=True)


if __name__ == "__main__":
    main()
