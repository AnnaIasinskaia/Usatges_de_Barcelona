#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_rendering.py

Общий модуль для рендера bipartite-графа и связанных helper-функций.

Нужен, чтобы:
- pipeline_unified.py и render_graph_from_csv.py использовали один и тот же renderer
- не дублировались сортировка узлов, отбор top_n_edges и стиль PNG-рендера
- render_graph_from_csv.py был синхронизирован с текущей схемой graph_rows.csv
"""

from __future__ import annotations

import re
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


_NUM_RE = re.compile(r"\d+")


def generic_numeric_sort_key(seg_id: str) -> Tuple[int, str]:
    nums = _NUM_RE.findall(str(seg_id))
    if nums:
        return (int(nums[0]), str(seg_id))
    return (10**9, str(seg_id))


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
    Собирает метаданные узлов из graph_rows.csv в логике, синхронизированной
    с pipeline_unified.py.

    В текущем unified pipeline graph_rows.csv гарантированно содержит:
    - left_node / right_node
    - left_corpus / right_corpus
    - right_doc_no (опционально для сортировки справа)

    Поле right_year больше не ожидается и не используется.
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
            label = corpus_spec.get("display_ru", left_node) if left_level == "corpus" else left_node
            left_nodes[left_node] = {
                "side": "left",
                "group": left_corpus,
                "label": label,
                "legend_label": corpus_spec.get("display_ru", left_corpus),
                "color": corpus_spec.get("color", "#999999"),
                "sort_key": (left_corpus, left_node),
            }

        if right_node and right_node not in right_nodes:
            corpus_spec = corpora.get(right_corpus, {})
            label = corpus_spec.get("display_ru", right_node) if right_level == "corpus" else right_node

            right_doc_no = row.get("right_doc_no", "")
            if right_doc_no != "":
                sort_key = (_to_int(right_doc_no, 10**9), right_node)
            else:
                sort_key = generic_numeric_sort_key(right_node)

            right_nodes[right_node] = {
                "side": "right",
                "group": right_corpus,
                "label": label,
                "legend_label": corpus_spec.get("display_ru", right_corpus),
                "color": corpus_spec.get("color", "#999999"),
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
    # --- выделяем top-3 рёбер по весу ---
    sorted_edges = sorted(
        edgelist,
        key=lambda e: float(G.edges[e].get("weight", 0.0)),
        reverse=True
    )

    top_edges = set(sorted_edges[:3])

    # --- задаём толщины ---
    weights = []
    for e in edgelist:
        if e in top_edges:
            weights.append(4.0)   # толстые (фиксированная ширина)
        else:
            weights.append(1.5)   # обычные
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
    seen_groups = set()
    for n in G.nodes():
        group = G.nodes[n].get("group", "")
        if not group or group in seen_groups:
            continue

        legend_label = G.nodes[n].get("legend_label", group)
        color = G.nodes[n].get("color", "#999999")
        legend.append(Patch(facecolor=color, label=legend_label))
        seen_groups.add(group)

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
