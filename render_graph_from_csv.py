#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_graph_from_csv.py

Рендерит PNG-граф только на основе уже готового graph_rows.csv.
Ничего не пересчитывает: ни сегментацию, ни TF-IDF, ни scoring.

CLI совместим по основным параметрам с pipeline_unified.py:
  --config
  --experiment
  --quiet
  --progress-every   (сохраняется только для CLI-совместимости, не используется)

Источник graph_rows.csv по умолчанию берётся из:
  EXPERIMENTS[experiment]["output"]["dir"] / "graph_rows.csv"
"""

from __future__ import annotations

import argparse
import csv
import importlib
import time
from pathlib import Path
from typing import Any, Dict, List

from graph_rendering import build_node_metadata_from_graph_rows, render_bipartite_graph



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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_unified", help="Python module name, e.g. config_unified")
    ap.add_argument("--experiment", required=True, help="Experiment id, see EXPERIMENTS in config")
    ap.add_argument("--quiet", action="store_true", help="Disable progress logs")
    ap.add_argument("--progress-every", type=int, default=1000, help="Compatibility arg; ignored")
    ap.add_argument("--graph-csv", default=None, help="Override path to graph_rows.csv")
    ap.add_argument("--out-png", default=None, help="Override output PNG path")
    args = ap.parse_args()

    logger = ProgressLogger(enabled=not args.quiet)

    logger.log("Step 1/3: Loading config and graph rows...")
    cfg_mod = load_config_module(args.config)

    corpora: Dict[str, Dict[str, Any]] = dict(getattr(cfg_mod, "CORPORA"))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(cfg_mod, "EXPERIMENTS"))

    if args.experiment not in experiments:
        raise KeyError(f"Unknown experiment_id={args.experiment}. Available: {sorted(experiments.keys())}")

    exp = experiments[args.experiment]
    out_dir = Path(exp["output"]["dir"])
    viz_cfg = dict(exp.get("viz") or {})
    agg_cfg = dict(exp.get("aggregation") or {})

    graph_csv = Path(args.graph_csv) if args.graph_csv else out_dir / "graph_rows.csv"
    out_png = Path(args.out_png) if args.out_png else out_dir / "graph.png"

    rows = read_csv_rows(graph_csv)
    logger.log(f"  Graph rows loaded: {len(rows)} from {graph_csv}")

    logger.log("Step 2/3: Building node metadata...")
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
    logger.log(f"  Rendering PNG: {out_png.name} (top_n_edges={top_n}, total_graph_rows={len(rows)})")
    rendered_edges = render_bipartite_graph(
        graph_rows=rows,
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        out_png=out_png,
        straight_edges=bool(viz_cfg.get("straight_edges", True)),
        label_left=bool(viz_cfg.get("label_left", True)),
        label_right=bool(viz_cfg.get("label_right", True)),
        top_n_edges=top_n,
    )
    logger.log(f"  Rendered edges: {rendered_edges}")
    logger.log("Render finished")
    logger.log(f"Summary: rows={len(rows)}, left_nodes={len(left_nodes)}, right_nodes={len(right_nodes)}, rendered_edges={rendered_edges}")
    logger.log(f"Output PNG: {out_png}")
    print(f"[render] experiment={args.experiment}", flush=True)
    print(f"[render] graph_csv={graph_csv}", flush=True)
    print(f"[render] out_png={out_png}", flush=True)
    print(f"[render] rendered_edges={rendered_edges}", flush=True)


if __name__ == "__main__":
    main()
