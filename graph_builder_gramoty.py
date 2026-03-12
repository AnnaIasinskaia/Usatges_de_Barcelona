
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_builder_gramoty.py

Graph builder for the charters study:
- left column: sources
- right column: charters
- charter volume I and II are colored differently
- charters are vertically sorted by date/year extracted from their ids
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


SOURCE_COLOR = "#bdbdbd"
VOLUME_COLORS = {
    "I": "#4c78a8",
    "II": "#e45756",
    "?": "#72b7b2",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _charter_volume(seg_id: str, row: Dict[str, Any]) -> str:
    if row.get("volume") in {"I", "II"}:
        return str(row["volume"])
    lower = str(seg_id).lower()
    if "911" in lower or "9_11" in lower:
        return "I"
    if "gramoty12" in lower or re.search(r"(^|[_\-])12([_\-]|$)", lower):
        return "II"
    return "?"


def _charter_year(seg_id: str, row: Dict[str, Any]) -> int:
    if row.get("year") is not None:
        try:
            return int(row["year"])
        except Exception:
            pass
    for token in re.split(r"[_\-]", str(seg_id)):
        if re.fullmatch(r"\d{3,4}", token):
            return int(token)
    return 999999


def _positions_for_bipartite(left_nodes: List[str], right_nodes: List[str], right_rows: Dict[str, Dict[str, Any]]):
    pos: Dict[str, Tuple[float, float]] = {}

    # sources on the left
    for i, node in enumerate(sorted(left_nodes)):
        pos[node] = (0.0, float(-i))

    # charters on the right, sorted by date and then id
    right_sorted = sorted(
        right_nodes,
        key=lambda node: (_charter_year(node, right_rows.get(node, {})), str(node)),
    )
    for i, node in enumerate(right_sorted):
        pos[node] = (1.0, float(-i))

    return pos


def build_gramoty_graph(rows: List[Dict[str, Any]], out_dir: Path, graph_name: str = "gramoty_graph"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    right_rows: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        source = str(row["source"])
        charter = str(row["charter_id"])
        weight = _safe_float(row.get("weight", 1.0), 1.0)
        volume = _charter_volume(charter, row)
        year = _charter_year(charter, row)

        g.add_node(source, kind="source", bipartite="left", color=SOURCE_COLOR)
        g.add_node(
            charter,
            kind="charter",
            bipartite="right",
            volume=volume,
            year=year,
            color=VOLUME_COLORS.get(volume, VOLUME_COLORS["?"]),
        )
        g.add_edge(source, charter, weight=weight)
        right_rows[charter] = {"volume": volume, "year": year}

    left_nodes = [n for n, a in g.nodes(data=True) if a.get("bipartite") == "left"]
    right_nodes = [n for n, a in g.nodes(data=True) if a.get("bipartite") == "right"]
    pos = _positions_for_bipartite(left_nodes, right_nodes, right_rows)

    # write GEXF first
    gexf_path = out_dir / f"{graph_name}.gexf"
    nx.write_gexf(g, gexf_path)

    # render PNG
    png_path = out_dir / f"{graph_name}.png"
    plt.figure(figsize=(18, max(8, len(right_nodes) * 0.18)))
    node_colors = [a.get("color", "#cccccc") for _, a in g.nodes(data=True)]
    edge_widths = [max(0.5, math.sqrt(_safe_float(a.get("weight", 1.0)))) for _, _, a in g.edges(data=True)]

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=700, alpha=0.95)
    nx.draw_networkx_edges(
        g,
        pos,
        width=edge_widths,
        alpha=0.35,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(g, pos, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    return g, {"gexf": gexf_path, "png": png_path}
