#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
borrowing_map.py

Строит "карту заимствований":
- без стрелок
- без жирных линий
- связанные источники располагаются ближе друг к другу
- направление A->B / B->A схлопывается в одну неориентированную связь

Поддерживает вход:
1) .gexf
2) .csv с рёбрами

Примеры:
    python borrowing_map.py \
        --input output/catalan_plus_usatges_upper_triangle/graph.gexf \
        --output borrowing_map.png

    python borrowing_map.py \
        --input output/catalan_plus_usatges_upper_triangle/graph.csv \
        --output borrowing_map.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ---------------------- Known colors from config.py ----------------------

KNOWN_NODE_STYLES = {
    "CorpusJuris": {"label": "Свод\nЮстиниана", "color": "#d62728"},
    "Evangelium": {"label": "Евангелие\n(Вульгата)", "color": "#1f77b4"},
    "LexVisigothorum": {"label": "Вестготская\nправда", "color": "#9467bd"},
    "ExceptPetri": {"label": "Составленные Петром\nизвлечения", "color": "#ff7f0e"},
    "Etymologiae": {"label": "Этимологии\nИсидора", "color": "#2ca02c"},
    "CostumsDeTortosa": {"label": "Обычаи\nТортосы", "color": "#8c564b"},
    "CostumsDeLleida": {"label": "Обычаи\nЛьейды", "color": "#e377c2"},
    "ConstitucionesBaiulieMirabeti": {"label": "Обычаи\nМиравета", "color": "#7f7f7f"},
    "CostumsDeOrta": {"label": "Обычаи\nОрты", "color": "#bcbd22"},
    "RecognovrentProceres": {"label": "Recognovrent\nProceres", "color": "#17becf"},
    "CostumresDeTarrega": {"label": "Обычаи\nТарреги", "color": "#aec7e8"},
    "CostumsDeValdAran": {"label": "Обычаи\nВаль-д’Арана", "color": "#98df8a"},
    "PragmaticaJaimeII1295": {"label": "Прагматика\nЖауме II (1295)", "color": "#ff9896"},
    "PragmaticaJaimeII1301": {"label": "Прагматика\nЖауме II (1301)", "color": "#c5b0d5"},
    "Acta911": {"label": "Грамоты\nIX–XI вв.", "color": "#2c7fb8"},
    "Acta12": {"label": "Грамоты\nXII в.", "color": "#f03b20"},
    "UsatgesBarcelona": {"label": "Барселонские\nОбычаи", "color": "#17becf"},
}

FALLBACK_COLOR = "#4c78a8"


# ---------------------- Helpers ----------------------

def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_node_name(name: str) -> str:
    return str(name).strip()


def get_label(node: str, attrs: Dict) -> str:
    for key in ("display_ru", "label", "name", "title"):
        value = attrs.get(key)
        if value:
            return str(value)

    if node in KNOWN_NODE_STYLES:
        return KNOWN_NODE_STYLES[node]["label"]

    return node


def get_color(node: str, attrs: Dict) -> str:
    for key in ("color", "node_color", "fillcolor"):
        value = attrs.get(key)
        if value:
            return str(value)

    if node in KNOWN_NODE_STYLES:
        return KNOWN_NODE_STYLES[node]["color"]

    return FALLBACK_COLOR


def collapse_to_undirected_sum(G_in: nx.Graph) -> nx.Graph:
    """
    Схлопывает ориентированный/дублированный граф в неориентированный.
    Если есть A->B и B->A, веса суммируются.
    """
    G = nx.Graph()

    for node, attrs in G_in.nodes(data=True):
        G.add_node(node, **attrs)

    for u, v, attrs in G_in.edges(data=True):
        if u == v:
            continue

        w = attrs.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


def build_graph_from_csv(csv_path: Path) -> nx.Graph:
    df = pd.read_csv(csv_path)

    left_col = pick_first_existing(df, [
        "left", "source", "src", "from",
        "left_node", "left_corpus", "node_u", "u"
    ])
    right_col = pick_first_existing(df, [
        "right", "target", "dst", "to",
        "right_node", "right_corpus", "node_v", "v"
    ])
    weight_col = pick_first_existing(df, [
        "weight", "edge_weight", "score", "value", "hits", "count"
    ])

    if left_col is None or right_col is None:
        raise ValueError(
            f"Не удалось распознать колонки рёбер в {csv_path}. "
            f"Нашёл колонки: {list(df.columns)}"
        )

    G = nx.DiGraph()

    for _, row in df.iterrows():
        u = normalize_node_name(row[left_col])
        v = normalize_node_name(row[right_col])

        if not u or not v or u == "nan" or v == "nan":
            continue

        w = row[weight_col] if weight_col is not None else 1.0
        try:
            w = float(w)
        except Exception:
            w = 1.0

        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)

        G.add_edge(u, v, weight=w)

    return collapse_to_undirected_sum(G)


def build_graph_from_gexf(gexf_path: Path) -> nx.Graph:
    G_in = nx.read_gexf(gexf_path)

    # На всякий случай нормализуем имена
    H = nx.DiGraph() if G_in.is_directed() else nx.Graph()

    for node, attrs in G_in.nodes(data=True):
        H.add_node(normalize_node_name(node), **attrs)

    for u, v, attrs in G_in.edges(data=True):
        H.add_edge(normalize_node_name(u), normalize_node_name(v), **attrs)

    return collapse_to_undirected_sum(H)


def load_graph(input_path: Path) -> nx.Graph:
    suffix = input_path.suffix.lower()
    if suffix == ".gexf":
        return build_graph_from_gexf(input_path)
    if suffix == ".csv":
        return build_graph_from_csv(input_path)
    raise ValueError("Поддерживаются только .gexf и .csv")


def compute_layout(
    G: nx.Graph,
    seed: int = 42,
    layout: str = "spring",
    iterations: int = 300,
    k: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}

    if k is None:
        k = 2.2 / math.sqrt(max(G.number_of_nodes(), 1))

    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, weight="weight")

    return nx.spring_layout(
        G,
        weight="weight",
        seed=seed,
        iterations=iterations,
        k=k,
        scale=1.25,
    )


def compute_label_positions(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    node_sizes: Dict[str, float],
    base_offset: float = 0.04,
) -> Dict[str, Tuple[float, float]]:
    """
    Смещает подписи наружу от центра графа, чтобы они не перекрывали узлы.
    Смещение зависит от размера узла.
    """
    if not pos:
        return {}

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    label_pos = {}
    for node, (x, y) in pos.items():
        dx = x - cx
        dy = y - cy
        norm = math.hypot(dx, dy)

        if norm < 1e-9:
            dx, dy = 1.0, 0.0
            norm = 1.0

        # matplotlib node_size задаётся в points^2; переводим грубо в радиус
        radius_hint = math.sqrt(max(node_sizes.get(node, 900.0), 1.0)) / 900.0
        offset = base_offset + radius_hint

        ox = (dx / norm) * offset
        oy = (dy / norm) * offset
        label_pos[node] = (x + ox, y + oy)

    return label_pos


def export_positions(pos: Dict[str, Tuple[float, float]], G: nx.Graph, output_csv: Path) -> None:
    rows = []
    for node, (x, y) in pos.items():
        rows.append({
            "node": node,
            "label": get_label(node, G.nodes[node]),
            "x": float(x),
            "y": float(y),
            "degree": int(G.degree(node)),
            "weighted_degree": float(G.degree(node, weight="weight")),
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def draw_map(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    output_path: Path,
    title: str = "Карта заимствований",
    node_size_base: int = 900,
    font_size: int = 11,
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    node_colors = [get_color(node, G.nodes[node]) for node in G.nodes()]
    weighted_degree = dict(G.degree(weight="weight"))

    node_sizes_list = []
    node_sizes_map: Dict[str, float] = {}
    for node in G.nodes():
        wd = weighted_degree.get(node, 0.0)
        size = node_size_base + 90 * math.sqrt(max(wd, 0.0))
        node_sizes_list.append(size)
        node_sizes_map[node] = size

    # Все линии одинаковой толщины, но чуть светлее для читаемости
    nx.draw_networkx_edges(
        G,
        pos,
        width=1.0,
        alpha=0.30,
        edge_color="#7a7a7a",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes_list,
        node_color=node_colors,
        linewidths=1.0,
        edgecolors="white",
        ax=ax,
    )

    labels = {node: get_label(node, G.nodes[node]) for node in G.nodes()}
    label_pos = compute_label_positions(
        G,
        pos,
        node_sizes=node_sizes_map,
        base_offset=0.05,
    )

    for node, label in labels.items():
        x, y = label_pos[node]
        ax.text(
            x,
            y,
            label,
            fontsize=font_size,
            fontfamily="DejaVu Sans",
            ha="center",
            va="center",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
            ),
        )

    ax.set_title(title, fontsize=16)
    ax.axis("off")
    ax.margins(0.14)
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


# ---------------------- Main ----------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь к graph.gexf или graph.csv")
    parser.add_argument("--output", required=True, help="Путь к итоговой PNG")
    parser.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--title", default="Карта заимствований")
    parser.add_argument(
        "--positions-csv",
        default=None,
        help="Опционально: сохранить координаты узлов в CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    G = load_graph(input_path)

    if G.number_of_nodes() == 0:
        raise SystemExit("Граф пустой")

    # Убираем изолированные узлы, если вдруг они есть
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    pos = compute_layout(
        G,
        seed=args.seed,
        layout=args.layout,
        iterations=args.iterations,
        k=args.k,
    )

    draw_map(
        G,
        pos,
        output_path=output_path,
        title=args.title,
    )

    if args.positions_csv:
        export_positions(pos, G, Path(args.positions_csv))

    print(f"[OK] Saved map to: {output_path}")
    print(f"[OK] Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()