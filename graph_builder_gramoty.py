#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_builder_gramoty.py

Graph builder for the charters study.

Rows format:
    {
        "left_id": "LexVisigoth" | "Usatges",
        "left_group": "LexVisigoth" | "Usatges",
        "charter_id": "Gramoty911_doc116_Y990...",
        "weight": 0.123,
        "edge_type": "source_direct" | "usatge_direct" | "source_projection",
        "volume": "I" | "II" | "?",
        "year": 812,
        "doc_no": 1,
    }
"""
from __future__ import annotations

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


def _charter_doc_no(seg_id: str, row: Dict[str, Any]) -> int:
    if row.get("doc_no") is not None:
        try:
            return int(row["doc_no"])
        except Exception:
            pass
    m = re.search(r"_doc(\d+)", str(seg_id), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_D(\d+)", str(seg_id))
    return int(m.group(1)) if m else 999999


def _charter_display_label(seg_id: str, row: Dict[str, Any]) -> str:
    doc_no = _charter_doc_no(seg_id, row)
    if doc_no == 999999:
        return str(seg_id)
    return f"{seg_id}"


def _left_group(left_id: str, row: Dict[str, Any]) -> str:
    group = row.get("left_group")
    if group:
        return str(group)
    text = str(left_id)
    if text.startswith("Us_") or text.lower().startswith("usatge"):
        return "Usatges"
    if "_S" in text:
        return text.split("_S", 1)[0]
    return text.split("_", 1)[0]


def _left_color(group: str) -> str:
    return LEFT_GROUP_COLORS.get(group, "#7f7f7f")


def _group_sort_key(group: str):
    if group in LEFT_GROUP_ORDER:
        return (LEFT_GROUP_ORDER.index(group), group)
    return (len(LEFT_GROUP_ORDER) + 1, group)


def _positions_for_graph(
    left_nodes: List[str],
    left_meta: Dict[str, Dict[str, Any]],
    right_nodes: List[str],
    right_meta: Dict[str, Dict[str, Any]],
):
    pos: Dict[str, Tuple[float, float]] = {}

    grouped: Dict[str, List[str]] = {}
    for node in left_nodes:
        group = left_meta[node]["group"]
        grouped.setdefault(group, []).append(node)

    y = 0.0
    group_gap = 1.85
    node_gap = 1.2
    for group in sorted(grouped.keys(), key=_group_sort_key):
        nodes = sorted(grouped[group])
        for node in nodes:
            pos[node] = (0.0, -y)
            y += node_gap
        y += group_gap

    vol_i = [n for n in right_nodes if right_meta[n].get("volume") == "I"]
    vol_ii = [n for n in right_nodes if right_meta[n].get("volume") == "II"]
    vol_other = [n for n in right_nodes if right_meta[n].get("volume") not in {"I", "II"}]

    vol_i_sorted = sorted(vol_i, key=lambda n: (_charter_doc_no(n, right_meta[n]), str(n)))
    vol_ii_sorted = sorted(vol_ii, key=lambda n: (_charter_doc_no(n, right_meta[n]), str(n)))
    vol_other_sorted = sorted(vol_other, key=lambda n: (_charter_doc_no(n, right_meta[n]), str(n)))

    right_y = 0.0
    right_gap = 1.8

    for node in vol_i_sorted:
        pos[node] = (1.0, -right_y)
        right_y += 1.0

    if vol_i_sorted and vol_ii_sorted:
        right_y += right_gap

    for node in vol_ii_sorted:
        pos[node] = (1.0, -right_y)
        right_y += 1.0

    if vol_other_sorted:
        right_y += right_gap
        for node in vol_other_sorted:
            pos[node] = (1.0, -right_y)
            right_y += 1.0

    return pos


def build_gramoty_graph(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    graph_name: str = "gramoty_graph",
    source_names_ru: Dict[str, str] | None = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gexf_path = out_dir / f"{graph_name}.gexf"
    png_path = out_dir / f"{graph_name}.png"

    G = nx.DiGraph()

    source_names_ru = source_names_ru or {}

    left_nodes: Dict[str, Dict[str, Any]] = {}
    right_nodes: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        left_id = str(row["left_id"])
        left_group = _left_group(left_id, row)
        charter_id = str(row["charter_id"])
        weight = _safe_float(row.get("weight", 0.0), 0.0)
        edge_type = str(row.get("edge_type", "source_direct"))

        left_nodes[left_id] = {
            "kind": "left",
            "group": left_group,
            "display_label": source_names_ru.get(left_group, left_group),
            "color": _left_color(left_group),
        }
        right_nodes[charter_id] = {
            "kind": "charter",
            "volume": _charter_volume(charter_id, row),
            "year": _charter_year(charter_id, row),
            "doc_no": _charter_doc_no(charter_id, row),
            "display_label": _charter_display_label(charter_id, row),
            "color": VOLUME_COLORS.get(_charter_volume(charter_id, row), VOLUME_COLORS["?"]),
        }

        G.add_node(left_id, **left_nodes[left_id])
        G.add_node(charter_id, **right_nodes[charter_id])
        G.add_edge(left_id, charter_id, weight=weight, edge_type=edge_type)

    left_node_list = sorted(left_nodes.keys(), key=lambda n: _group_sort_key(left_nodes[n]["group"]))
    right_node_list = list(right_nodes.keys())
    pos = _positions_for_graph(left_node_list, left_nodes, right_node_list, right_nodes)

    fig_h = max(12, 0.55 * max(len(left_node_list), len(right_node_list)) + 4)
    plt.figure(figsize=(18, fig_h))
    ax = plt.gca()
    ax.set_facecolor("#f7f7f7")

    nx.draw_networkx_nodes(
        G, pos, nodelist=left_node_list,
        node_size=280, node_color=[left_nodes[n]["color"] for n in left_node_list],
        linewidths=0.8, edgecolors="white"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=right_node_list,
        node_size=520, node_color=[right_nodes[n]["color"] for n in right_node_list],
        linewidths=0.8, edgecolors="white"
    )

    edge_groups = {}
    for u, v, d in G.edges(data=True):
        edge_groups.setdefault(d.get("edge_type", "source_direct"), []).append((u, v, d))

    for edge_type, triples in edge_groups.items():
        edgelist = [(u, v) for u, v, _ in triples]
        raw_weights = [_safe_float(d.get("weight", 0.0)) for _, _, d in triples]
        scaled_weights = [max(0.0, min(1.0, x)) for x in raw_weights]
        widths = [max(0.6, min(3.2, 0.8 + 2.2 * x)) for x in scaled_weights]
        edge_color = None
        alpha = 0.35
        if edge_type == "usatge_direct":
            edge_color = LEFT_GROUP_COLORS["Usatges"]
            alpha = 0.55
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            width=widths,
            edge_color=edge_color,
            alpha=alpha,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            connectionstyle="arc3,rad=0.03",
        )

    for node in left_node_list:
        x, y = pos[node]
        ax.text(
            x - 0.03, y, left_nodes[node]["display_label"],
            fontsize=15, ha="right", va="center",
        )

    for node in right_node_list:
        x, y = pos[node]
        ax.text(
            x + 0.018, y + 0.10, right_nodes[node]["display_label"],
            fontsize=14, ha="left", va="center",
        )

    plt.title(
        "Заимствования в грамоты (7 шагов: BorrowScore+SW): источники и Usatges → грамоты",
        fontsize=22,
        fontweight="bold",
        pad=28,
    )

    legend_patches = []
    seen = set()
    for group in LEFT_GROUP_ORDER:
        if group in {meta["group"] for meta in left_nodes.values()} and group not in seen:
            legend_patches.append(
                mpatches.Patch(color=LEFT_GROUP_COLORS[group], label=source_names_ru.get(group, group))
            )
            seen.add(group)
    legend_patches.extend([
        mpatches.Patch(color=VOLUME_COLORS["I"], label="Грамоты IX–XI вв."),
        mpatches.Patch(color=VOLUME_COLORS["II"], label="Грамоты XII в."),
    ])

    plt.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        fontsize=13,
        frameon=True,
        framealpha=0.95,
    )

    plt.axis("off")
    plt.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    nx.write_gexf(G, gexf_path)
    return G, {"gexf": gexf_path, "png": png_path}
