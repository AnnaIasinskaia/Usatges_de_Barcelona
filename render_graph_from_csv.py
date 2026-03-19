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
  --progress-every   (принимается для совместимости, но не используется)

Источник graph_rows.csv по умолчанию берётся из:
  EXPERIMENTS[experiment]["output"]["dir"] / "graph_rows.csv"
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:  # pragma: no cover
    plt = None


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


_NUM_RE = re.compile(r"\d+")


def generic_numeric_sort_key(seg_id: str) -> Tuple[int, str]:
    nums = _NUM_RE.findall(str(seg_id))
    if nums:
        return (int(nums[0]), str(seg_id))
    return (10**9, str(seg_id))


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def build_node_metadata_from_graph_rows(
    graph_rows: Sequence[Dict[str, Any]],
    corpora: Dict[str, Dict[str, Any]],
    left_level: str,
    right_level: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Собирает минимально достаточные метаданные узлов прямо из graph_rows.csv.
    Для corpus-level слева использует display_ru/color из конфига.
    Для parent/leaf справа использует node_id как label.
    """
    left_nodes: Dict[str, Dict[str, Any]] = {}
    right_nodes: Dict[str, Dict[str, Any]] = {}

    for row in graph_rows:
        left_node = str(row.get("left_node", ""))
        right_node = str(row.get("right_node", ""))
        left_corpus = str(row.get("left_corpus", ""))
        right_corpus = str(row.get("right_corpus", ""))

        if left_node and left_node not in left_nodes:
            corpus_spec = corpora.get(left_corpus, {})
            if left_level == "corpus":
                label = corpus_spec.get("display_ru", left_node)
            else:
                label = left_node

            left_nodes[left_node] = {
                "side": "left",
                "group": left_corpus,
                "label": label,
                "color": corpus_spec.get("color", "#999999"),
                "sort_key": (left_corpus, left_node),
            }

        if right_node and right_node not in right_nodes:
            corpus_spec = corpora.get(right_corpus, {})
            label = right_node if right_level in ("parent", "leaf") else corpus_spec.get("display_ru", right_node)

            right_year = row.get("right_year", "")
            right_doc_no = row.get("right_doc_no", "")

            if right_year != "" or right_doc_no != "":
                sort_key = (_to_int(right_year, 10**9), _to_int(right_doc_no, 10**9), right_node)
            else:
                sort_key = generic_numeric_sort_key(right_node)

            right_nodes[right_node] = {
                "side": "right",
                "group": right_corpus,
                "label": label,
                "color": corpus_spec.get("color", "#17becf"),
                "sort_key": sort_key,
            }

    return left_nodes, right_nodes


def render_bipartite_graph(
    graph_rows: Sequence[Dict[str, Any]],
    left_nodes: Dict[str, Dict[str, Any]],
    right_nodes: Dict[str, Dict[str, Any]],
    out_png: Path,
    straight_edges: bool = True,
    label_left: bool = True,
    label_right: bool = True,
    top_n_edges: Optional[int] = None,
) -> int:
    if plt is None or nx is None:
        return 0

    rows = list(graph_rows)
    if top_n_edges is not None:
        try:
            k = int(top_n_edges)
        except Exception:
            k = None
        if k is not None and k > 0 and len(rows) > k:
            rows = sorted(rows, key=lambda r: -_to_float(r.get("weight", 0.0)))[:k]

    G = nx.DiGraph()
    for r in rows:
        u = str(r["left_node"])
        v = str(r["right_node"])
        w = _to_float(r.get("weight", 0.0))
        if u not in left_nodes or v not in right_nodes:
            continue
        G.add_node(u, **left_nodes[u])
        G.add_node(v, **right_nodes[v])
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() == 0:
        return 0

    left_list = [n for n in G.nodes() if G.nodes[n].get("side") == "left"]
    right_list = [n for n in G.nodes() if G.nodes[n].get("side") == "right"]

    left_list = sorted(left_list, key=lambda n: (str(G.nodes[n].get("group", "")), str(n)))
    right_list = sorted(
        right_list,
        key=lambda n: G.nodes[n].get("sort_key", (10**9, 10**9, str(n))),
    )

    pos: Dict[str, Tuple[float, float]] = {}
    y = 0.0
    last_group = None
    for n in left_list:
        g = G.nodes[n].get("group", "")
        if last_group is not None and g != last_group:
            y += 0.8
        pos[n] = (0.0, -y)
        y += 1.0
        last_group = g

    total_left_h = max(1.0, y)
    for i, n in enumerate(right_list):
        pos[n] = (4.0, -(i * (total_left_h / max(1, len(right_list)))))

    fig_h = max(10, 0.35 * max(len(left_list), len(right_list)) + 6)
    plt.figure(figsize=(20, fig_h))
    ax = plt.gca()
    ax.set_facecolor("#f7f7f7")

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=left_list,
        node_size=300,
        node_color=[G.nodes[n].get("color", "#999999") for n in left_list],
        linewidths=0.8,
        edgecolors="white",
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=right_list,
        node_size=520,
        node_color=[G.nodes[n].get("color", "#999999") for n in right_list],
        linewidths=0.8,
        edgecolors="white",
    )

    edgelist = list(G.edges())
    weights = [
        max(0.6, min(4.0, 0.6 + 3.0 * float(G.edges[e].get("weight", 0.0))))
        for e in edgelist
    ]
    edge_colors = [G.nodes[u].get("color", "#666666") for (u, v) in edgelist]

    edge_kwargs = {}
    if not straight_edges:
        edge_kwargs["connectionstyle"] = "arc3,rad=0.03"

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edgelist,
        width=weights,
        edge_color=edge_colors,
        alpha=0.45,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        **edge_kwargs,
    )

    if label_left:
        for n in left_list:
            x, y = pos[n]
            ax.text(
                x - 0.06, y,
                G.nodes[n].get("label", str(n)),
                fontsize=13,
                ha="right",
                va="center",
            )

    if label_right:
        for n in right_list:
            x, y = pos[n]
            ax.text(
                x + 0.06, y,
                G.nodes[n].get("label", str(n)),
                fontsize=12,
                ha="left",
                va="center",
            )

    legend = []
    seen = set()
    for n in left_list:
        lbl = G.nodes[n].get("label", str(n))
        col = G.nodes[n].get("color", "#999999")
        key = (lbl, col)
        if key in seen:
            continue
        legend.append(Patch(facecolor=col, label=lbl))
        seen.add(key)

    if legend:
        plt.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=12,
            frameon=True,
            framealpha=0.95,
        )

    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return G.number_of_edges()


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

    logger.log(f"load config: start experiment={args.experiment}")
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

    logger.log(f"graph csv: {graph_csv}")
    rows = read_csv_rows(graph_csv)
    logger.log(f"rows loaded: {len(rows)}")

    left_level = str(agg_cfg.get("left_node_level", "parent"))
    right_level = str(agg_cfg.get("right_node_level", "parent"))
    left_nodes, right_nodes = build_node_metadata_from_graph_rows(
        rows,
        corpora=corpora,
        left_level=left_level,
        right_level=right_level,
    )
    logger.log(f"left nodes: {len(left_nodes)}, right nodes: {len(right_nodes)}")

    top_n = viz_cfg.get("top_n_edges")
    logger.log(f"render png: {out_png} (top_n_edges={top_n})")
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
    logger.log(f"rendered edges: {rendered_edges}")
    print(f"[render] experiment={args.experiment}", flush=True)
    print(f"[render] graph_csv={graph_csv}", flush=True)
    print(f"[render] out_png={out_png}", flush=True)
    print(f"[render] rendered_edges={rendered_edges}", flush=True)


if __name__ == "__main__":
    main()
