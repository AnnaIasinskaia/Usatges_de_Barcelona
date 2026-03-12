#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_builder_gramoty.py

Graph builder for the charters study.

Expected rows format:
    {
        "left_id": "CorpusJuris" | "Usatges",
        "left_group": "CorpusJuris" | "Usatges",
        "charter_id": "Gramoty911_D336_Y1066MjunD10",
        "weight": 0.123,
        "edge_type": "source_direct" | "usatge_direct",
        "volume": "I" | "II" | "?",     # optional, can be re-derived
        "year": 812,                         # optional, can be re-derived
    }

Layout:
    - left column: source groups + Usatges
    - right column: charter documents sorted by date/year
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


LEFT_GROUP_COLORS = {
    "CorpusJuris": "#d62728",
    "Etymologiae": "#2ca02c",
    "LexVisigoth": "#9467bd",
    "ExceptPetri": "#ff7f0e",
    "Evangelium": "#8c564b",
    "Usatges": "#17becf",
}

VOLUME_COLORS = {
    "I": "#4c78a8",
    "II": "#e45756",
    "?": "#72b7b2",
}

LEFT_GROUP_ORDER = ["CorpusJuris", "Etymologiae", "LexVisigoth", "ExceptPetri", "Evangelium", "Usatges"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _charter_volume(seg_id: str, row: Dict[str, Any]) -> str:
    volume = row.get("volume")
    if volume in {"I", "II"}:
        return str(volume)
    lower = str(seg_id).lower()
    if "gramoty911" in lower:
        return "I"
    if "gramoty12" in lower:
        return "II"
    return "?"


def _charter_year(seg_id: str, row: Dict[str, Any]) -> int:
    if row.get("year") is not None:
        try:
            return int(row["year"])
        except Exception:
            pass
    m = re.search(r"_Y(\d{3,4})", str(seg_id))
    if m:
        return int(m.group(1))
    return 999999


def _charter_doc_no(seg_id: str) -> int:
    m = re.search(r"_D(\d+)", str(seg_id))
    return int(m.group(1)) if m else 999999


def _short_charter_label(seg_id: str) -> str:
    m = re.search(r"_D(\d+)", str(seg_id))
    if m:
        return f"D{m.group(1)}"
    return str(seg_id)


def _left_group(left_id: str, row: Dict[str, Any]) -> str:
    group = row.get("left_group")
    if group:
        return str(group)
    return str(left_id)


def _left_color(group: str) -> str:
    return LEFT_GROUP_COLORS.get(group, "#7f7f7f")


def _group_sort_key(group: str):
    if group in LEFT_GROUP_ORDER:
        return (LEFT_GROUP_ORDER.index(group), group)
    return (len(LEFT_GROUP_ORDER) + 1, group)


def _positions_for_graph(left_nodes: List[str], right_nodes: List[str], right_meta: Dict[str, Dict[str, Any]]):
    pos: Dict[str, Tuple[float, float]] = {}

    for i, node in enumerate(sorted(left_nodes, key=_group_sort_key)):
        pos[node] = (0.0, float(-i * 1.2))

    right_sorted = sorted(
        right_nodes,
        key=lambda node: (
            right_meta[node].get("year", 999999),
            _charter_doc_no(node),
            str(node),
        ),
    )
    for i, node in enumerate(right_sorted):
        pos[node] = (1.0, float(-i))

    return pos


def build_gramoty_graph(rows: List[Dict[str, Any]], out_dir: Path, graph_name: str = "gramoty_graph", source_names_ru: Dict[str, str] | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_names_ru = source_names_ru or {}

    g = nx.DiGraph()
    right_meta: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        left_group = _left_group(str(row.get("left_id", row["left_group"])), row)
        charter = str(row["charter_id"])
        weight = _safe_float(row.get("weight", 1.0), 1.0)
        edge_type = str(row.get("edge_type", "source_direct"))
        volume = _charter_volume(charter, row)
        year = _charter_year(charter, row)

        g.add_node(
            left_group,
            kind="left",
            left_group=left_group,
            bipartite="left",
            color=_left_color(left_group),
            label=source_names_ru.get(left_group, left_group),
        )
        g.add_node(
            charter,
            kind="charter",
            bipartite="right",
            volume=volume,
            year=year,
            color=VOLUME_COLORS.get(volume, VOLUME_COLORS["?"]),
            label=_short_charter_label(charter),
        )
        g.add_edge(
            left_group,
            charter,
            weight=weight,
            edge_type=edge_type,
            best_left_id=str(row.get("best_left_id", "")),
            hit_count=int(row.get("hit_count", 1)),
        )

        right_meta[charter] = {"volume": volume, "year": year}

    left_nodes = [n for n, a in g.nodes(data=True) if a.get("bipartite") == "left"]
    right_nodes = [n for n, a in g.nodes(data=True) if a.get("bipartite") == "right"]
    pos = _positions_for_graph(left_nodes, right_nodes, right_meta)

    gexf_path = out_dir / f"{graph_name}.gexf"
    nx.write_gexf(g, gexf_path)

    png_path = out_dir / f"{graph_name}.png"
    plt.figure(figsize=(18, max(10, len(right_nodes) * 0.18, len(left_nodes) * 0.6)))

    left_node_list = [n for n in g.nodes() if g.nodes[n].get("bipartite") == "left"]
    right_node_list = [n for n in g.nodes() if g.nodes[n].get("bipartite") == "right"]

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=left_node_list,
        node_color=[g.nodes[n].get("color", "#cccccc") for n in left_node_list],
        node_size=180,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=right_node_list,
        node_color=[g.nodes[n].get("color", "#cccccc") for n in right_node_list],
        node_size=420,
        alpha=0.95,
    )

    for edge_type, alpha in (("source_direct", 0.35), ("usatge_direct", 0.60)):
        edgelist = [(u, v) for u, v, a in g.edges(data=True) if a.get("edge_type") == edge_type]
        if not edgelist:
            continue
        edge_colors = [g.nodes[u].get("color", "#999999") for u, v in edgelist]
        edge_widths = [max(0.8, math.sqrt(_safe_float(g.edges[u, v].get("weight", 1.0))) * 2.6) for u, v in edgelist]
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edgelist,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=alpha,
            arrows=True,
            arrowsize=11,
            connectionstyle="arc3,rad=0.04",
        )

    nx.draw_networkx_labels(
        g,
        pos,
        labels={n: g.nodes[n].get("label", n) for n in left_node_list},
        font_size=10,
        horizontalalignment="right",
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels={n: g.nodes[n].get("label", n) for n in right_node_list},
        font_size=8,
    )

    legend_handles = []
    for group in LEFT_GROUP_ORDER:
        if group in left_nodes:
            legend_handles.append(mpatches.Patch(color=_left_color(group), label=source_names_ru.get(group, group)))
    if any(right_meta[n].get("volume") == "I" for n in right_nodes):
        legend_handles.append(mpatches.Patch(color=VOLUME_COLORS["I"], label="Грамоты IX–XI вв."))
    if any(right_meta[n].get("volume") == "II" for n in right_nodes):
        legend_handles.append(mpatches.Patch(color=VOLUME_COLORS["II"], label="Грамоты XII в."))

    if legend_handles:
        plt.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    plt.title("Заимствования в грамоты: источники и Usatges → грамоты", fontsize=18, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    return g, {"gexf": gexf_path, "png": png_path}
