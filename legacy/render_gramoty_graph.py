#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render gramoty graph PNG/GEXF from cached graph rows CSV.

Usage:
    python render_gramoty_graph.py
    python render_gramoty_graph.py output/gramoty/graph_rows_gramoty.csv
    python render_gramoty_graph.py output/gramoty/graph_rows_gramoty.csv output/gramoty
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

from graph_builder_gramoty import build_gramoty_graph


def _load_source_names(config_module_name: str = "config_gramoty"):
    try:
        cfg = importlib.import_module(config_module_name)
    except Exception:
        return {}
    return getattr(cfg, "SOURCE_NAMES_RU", None) or getattr(cfg, "SOURCE_NAMES_RU_SHORT", None) or {}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_charters/graph_rows_gramoty.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    if not rows:
        raise SystemExit(f"CSV is empty: {csv_path}")

    graph, paths = build_gramoty_graph(
        rows,
        out_dir=out_dir,
        graph_name="gramoty_graph",
        source_names_ru=_load_source_names(),
    )
    print(f"Rendered graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(paths["gexf"])
    print(paths["png"])


if __name__ == "__main__":
    main()
