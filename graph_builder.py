"""
Step 7: Построение и анализ графа заимствований.

BorrowingGraph:
- хранит ориентированный взвешенный граф
- экспортирует GEXF/CSV
- строит несколько разных визуализаций, рассчитанных на печать на A4.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
import textwrap

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

# Matplotlib для статичных картинок под печать
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Глобальные настройки под А4 (портрет), крупные шрифты и печать
    plt.rcParams.update({
        "figure.figsize": (8.27, 11.69),  # A4, портрет
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
    })
    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False


class BorrowingGraph:
    """Ориентированный взвешенный граф текстовых заимствований."""

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError("Требуется пакет networkx: pip install networkx")
        self.G = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # Базовые операции
    # ------------------------------------------------------------------ #

    def add_borrowing(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        source_name: str = "",
        alignment_a: Optional[List[str]] = None,
        alignment_b: Optional[List[str]] = None,
        usatge_text: str = "",
        source_text: str = "",
    ):
        """Добавить ориентированное ребро «источник → обычай»."""
        if not self.G.has_node(source_id):
            self.G.add_node(
                source_id,
                text_group=source_name,
                node_type="source",
                text_snippet=source_text[:200] if source_text else "",
            )
        if not self.G.has_node(target_id):
            self.G.add_node(
                target_id,
                text_group="Usatges",
                node_type="usatge",
                text_snippet=usatge_text[:200] if usatge_text else "",
            )

        align_str = ""
        if alignment_a and alignment_b:
            align_str = " | ".join(
                f"{a} ~ {b}" for a, b in zip(alignment_a[:20], alignment_b[:20])
            )

        self.G.add_edge(
            source_id,
            target_id,
            weight=round(weight, 4),
            alignment=align_str,
        )

    def export_gexf(self, path: Path):
        """Экспорт графа в формат GEXF (для Gephi)."""
        nx.write_gexf(self.G, str(path))

    def export_csv(
        self,
        path: Path,
        usatge_texts: Optional[Dict[str, str]] = None,
        source_texts: Optional[Dict[str, str]] = None,
    ):
        """
        Экспорт рёбер в CSV с добавлением текстовых фрагментов,
        чтобы результат можно было читать без обращения к исходным файлам.
        """
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source_segment",
                    "source_group",
                    "target_usatge",
                    "borrow_score",
                    "alignment_sample",
                    "usatge_text_snippet",
                    "source_text_snippet",
                ]
            )
            for u, v, data in sorted(
                self.G.edges(data=True),
                key=lambda x: x[2].get("weight", 0),
                reverse=True,
            ):
                u_text = ""
                v_text = ""
                if usatge_texts and v in usatge_texts:
                    v_text = usatge_texts[v][:150].replace("\n", " ").strip()
                if source_texts and u in source_texts:
                    u_text = source_texts[u][:150].replace("\n", " ").strip()

                writer.writerow(
                    [
                        u,
                        self.G.nodes[u].get("text_group", ""),
                        v,
                        data.get("weight", 0),
                        data.get("alignment", ""),
                        v_text,
                        u_text,
                    ]
                )

    # ------------------------------------------------------------------ #
    # Статистика и форматированный вывод
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        """Базовая статистика по графу."""
        usatge_nodes = [
            n for n, d in self.G.nodes(data=True) if d.get("node_type") == "usatge"
        ]
        source_nodes = [
            n for n, d in self.G.nodes(data=True) if d.get("node_type") == "source"
        ]

        stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "usatge_nodes": len(usatge_nodes),
            "source_nodes": len(source_nodes),
        }

        if usatge_nodes:
            in_deg = sorted(
                [(n, self.G.in_degree(n)) for n in usatge_nodes],
                key=lambda x: x[1],
                reverse=True,
            )
            stats["most_dependent_usatges"] = in_deg[:10]

        if source_nodes:
            out_deg = sorted(
                [(n, self.G.out_degree(n)) for n in source_nodes],
                key=lambda x: x[1],
                reverse=True,
            )
            stats["most_influential_sources"] = out_deg[:10]

        group_counts: Dict[str, int] = {}
        group_weights: Dict[str, float] = {}
        for u, v, data in self.G.edges(data=True):
            grp = self.G.nodes[u].get("text_group", "unknown")
            group_counts[grp] = group_counts.get(grp, 0) + 1
            group_weights[grp] = group_weights.get(grp, 0.0) + data.get("weight", 0.0)

        stats["edges_by_source"] = group_counts
        stats["total_weight_by_source"] = {k: round(v, 3) for k, v in group_weights.items()}

        return stats

    def format_stats(self, usatge_texts: Optional[Dict[str, str]] = None) -> str:
        """Человеко-читаемый текст статистики с фрагментами обычаев."""
        stats = self.get_stats()
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("РЕЗУЛЬТАТЫ ПАЙПЛАЙНА")
        lines.append("=" * 70)
        lines.append(f"  Узлов всего: {stats['total_nodes']}")
        lines.append(f"  Рёбер (заимствований): {stats['total_edges']}")
        lines.append(f"  Узлов-Usatges: {stats['usatge_nodes']}")
        lines.append(f"  Узлов-источников: {stats['source_nodes']}")
        lines.append(f"  Рёбер по источникам: {stats.get('edges_by_source', {})}")
        lines.append(
            f"  Суммарный вес по источникам: {stats.get('total_weight_by_source', {})}"
        )

        if "most_dependent_usatges" in stats:
            lines.append("")
            lines.append(
                "  Наиболее зависимые Usatges (по числу входящих заимствований):"
            )
            lines.append("  " + "-" * 66)
            for name, deg in stats["most_dependent_usatges"]:
                snippet = ""
                if usatge_texts and name in usatge_texts:
                    raw = usatge_texts[name].replace("\n", " ").strip()
                    snippet = textwrap.shorten(raw, width=80, placeholder="...")
                lines.append(f"    {name} ({deg} заимствований)")
                if snippet:
                    lines.append(f"      Текст: {snippet}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # ВИЗУАЛИЗАЦИИ
    # ------------------------------------------------------------------ #

    # 1) Тепловая карта «обычай × источник»

    def visualize_heatmap(
        self,
        output_path: Path,
        usatge_texts: Optional[Dict[str, str]] = None,
    ):
        """
        Тепловая карта: строки — Usatges, столбцы — источники,
        цвет — суммарный BorrowScore. Предназначена для печати на A4.
        """
        if not _PLT_AVAILABLE:
            return

        # Собираем множество узлов-Usatges и групп источников
        usatge_nodes = sorted(
            {
                v
                for _, v, _ in self.G.edges(data=True)
                if self.G.nodes[v].get("node_type") == "usatge"
            },
            key=lambda x: (int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0),
        )

        source_groups = sorted(
            {
                self.G.nodes[u].get("text_group", "?")
                for u, _, _ in self.G.edges(data=True)
            }
        )

        if not usatge_nodes or not source_groups:
            return

        # Матрица суммарных весов
        matrix = np.zeros((len(usatge_nodes), len(source_groups)))
        u_idx = {n: i for i, n in enumerate(usatge_nodes)}
        g_idx = {g: i for i, g in enumerate(source_groups)}

        for src, tgt, data in self.G.edges(data=True):
            grp = self.G.nodes[src].get("text_group", "?")
            if tgt in u_idx and grp in g_idx:
                matrix[u_idx[tgt], g_idx[grp]] += data.get("weight", 0.0)

        # Подписи строк: ID + начало латинского текста
        row_labels = list(usatge_nodes)

        # Фигура под A4
        fig, ax = plt.subplots()

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )

        ax.set_xticks(range(len(source_groups)))
        ax.set_xticklabels(
            source_groups,
            rotation=45,
            ha="right",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_yticks(range(len(usatge_nodes)))
        ax.set_yticklabels(row_labels, fontsize=9)

        # Подписываем ячейки только там, где есть значение
        max_val = matrix.max() if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if max_val and val > max_val * 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )

        cbar = plt.colorbar(
            im,
            ax=ax,
            label="Суммарный вес заимствований (BorrowScore)",
            shrink=0.85,
        )
        cbar.ax.tick_params(labelsize=9)

        ax.set_title(
            'Интенсивность заимствований: "Usatges de Barcelona" из латинских источников',
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel("Текст-источник")
        ax.set_ylabel("Статья обычая (Usatge)")

        plt.tight_layout(pad=0.8)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2) Отдельный двудольный граф для каждого источника

    def visualize_per_source(
        self,
        output_dir: Path,
        usatge_texts: Optional[Dict[str, str]] = None,
    ):
        """
        Один аккуратный двудольный граф на источник.
        Источники слева, Usatges справа, с текстовыми фрагментами.
        """
        if not _PLT_AVAILABLE:
            return

        source_groups = {
            self.G.nodes[u].get("text_group", "?")
            for u, _, _ in self.G.edges(data=True)
        }

        source_colors = {
            "CorpusJuris": "#d62728",
            "Evangelium": "#1f77b4",
            "Etymologiae": "#2ca02c",
            "LexVisigoth": "#9467bd",
            "ExceptPetri": "#ff7f0e",
        }

        for grp in sorted(source_groups):
            edges = [
                (u, v, d)
                for u, v, d in self.G.edges(data=True)
                if self.G.nodes[u].get("text_group") == grp
            ]
            if not edges:
                continue

            sub = nx.DiGraph()
            for u, v, d in edges:
                sub.add_node(u, node_type="source")
                sub.add_node(v, node_type="usatge")
                sub.add_edge(u, v, **d)

            us_nodes = [n for n, d in sub.nodes(data=True) if d.get("node_type") == "usatge"]
            src_nodes = [n for n, d in sub.nodes(data=True) if d.get("node_type") == "source"]

            pos: Dict[str, Tuple[float, float]] = {}
            # источники слева
            for i, n in enumerate(sorted(src_nodes)):
                pos[n] = (0.0, -i)

            # Usatges справа, отсортированы по номеру
            for i, n in enumerate(
                sorted(
                    us_nodes,
                    key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
                )
            ):
                pos[n] = (3.0, -i * (len(src_nodes) / max(len(us_nodes), 1)))

            fig, ax = plt.subplots(figsize=(8.27, 11.69))

            color = source_colors.get(grp, "#7f8c8d")

            nx.draw_networkx_nodes(
                sub,
                pos,
                nodelist=src_nodes,
                node_color=color,
                node_size=40,
                alpha=0.6,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                sub,
                pos,
                nodelist=us_nodes,
                node_color="#17becf",
                node_size=160,
                alpha=0.85,
                ax=ax,
            )

            weights = [d.get("weight", 0.1) * 5 for _, _, d in sub.edges(data=True)]
            nx.draw_networkx_edges(
                sub,
                pos,
                width=weights,
                alpha=0.35,
                edge_color=color,
                arrows=True,
                arrowsize=9,
                ax=ax,
            )

            labels: Dict[str, str] = {}
            for n in us_nodes:
                labels[n] = n

            nx.draw_networkx_labels(
                sub,
                pos,
                labels,
                font_size=8,
                font_color="#2c3e50",
                ax=ax,
                horizontalalignment="left",
            )

            n_edges = sub.number_of_edges()
            n_usatges = len(us_nodes)
            ax.set_title(
                f"{grp} → Usatges ({n_edges} заимствований, {n_usatges} статей)",
                fontsize=15,
                fontweight="bold",
            )
            ax.axis("off")

            plt.tight_layout(pad=0.8)
            fname = output_dir / f"graph_{grp}.png"
            plt.savefig(str(fname), dpi=300, bbox_inches="tight")
            plt.close(fig)

    # 3) Граф только из N сильнейших заимствований

    def visualize_top_borrowings(
        self,
        output_path: Path,
        top_n: int = 30,
        usatge_texts: Optional[Dict[str, str]] = None,
    ):
        """
        Компактный граф из N сильнейших рёбер.
        Kamada–Kawai layout, уплотнён под страницу A4.
        """
        if not _PLT_AVAILABLE:
            return

        all_edges = sorted(
            self.G.edges(data=True),
            key=lambda x: x[2].get("weight", 0),
            reverse=True,
        )
        top_edges = all_edges[:top_n]

        sub = nx.DiGraph()
        for u, v, d in top_edges:
            grp = self.G.nodes[u].get("text_group", "?")
            sub.add_node(u, text_group=grp, node_type="source")
            sub.add_node(v, text_group="Usatges", node_type="usatge")
            sub.add_edge(u, v, **d)

        source_colors = {
            "CorpusJuris": "#d62728",
            "Evangelium": "#1f77b4",
            "Etymologiae": "#2ca02c",
            "LexVisigoth": "#9467bd",
            "ExceptPetri": "#ff7f0e",
            "Usatges": "#17becf",
        }

        node_colors = [
            source_colors.get(sub.nodes[n].get("text_group", ""), "#95a5a6")
            for n in sub.nodes()
        ]
        node_sizes = [
            260 if sub.nodes[n].get("node_type") == "usatge" else 70
            for n in sub.nodes()
        ]

        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        pos = nx.kamada_kawai_layout(sub, scale=1.8, center=(0.0, 0.0))

        weights = [d.get("weight", 0.1) * 5 for _, _, d in sub.edges(data=True)]
        edge_colors = [
            source_colors.get(sub.nodes[u].get("text_group", ""), "#7f8c8d")
            for u, v in sub.edges()
        ]

        nx.draw_networkx_nodes(
            sub,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_edges(
            sub,
            pos,
            width=weights,
            alpha=0.4,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=10,
            ax=ax,
            connectionstyle="arc3,rad=0.08",
        )

        # Подписываем только Usatges с фрагментами текста
        labels: Dict[str, str] = {}
        for n in sub.nodes():
            if sub.nodes[n].get("node_type") == "usatge":
                labels[n] = n

        nx.draw_networkx_labels(
            sub,
            pos,
            labels,
            font_size=9,
            font_weight="bold",
            font_color="#2c3e50",
            ax=ax,
        )

        from matplotlib.patches import Patch

        legend_elems = [
            Patch(facecolor=c, label=g)
            for g, c in source_colors.items()
            if g in {sub.nodes[n].get("text_group", "") for n in sub.nodes()}
        ]
        ax.legend(
            handles=legend_elems,
            loc="upper left",
            fontsize=10,
            framealpha=0.9,
        )

        ax.set_title(
            f'Топ-{top_n} сильнейших заимствований: "Usatges de Barcelona" ← латинские источники',
            fontsize=15,
            fontweight="bold",
        )
        ax.axis("off")

        plt.tight_layout(pad=0.8)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 4) Горизонтальные стэкованные бары по обычаям

    def visualize_bar_chart(
        self,
        output_path: Path,
        usatge_texts: Optional[Dict[str, str]] = None,
    ):
        """
        Горизонтальная стэкованная диаграмма:
        одна строка — один обычай, цветовые сегменты — вклад источников.
        """
        if not _PLT_AVAILABLE:
            return

        source_colors = {
            "CorpusJuris": "#d62728",
            "Evangelium": "#1f77b4",
            "Etymologiae": "#2ca02c",
            "LexVisigoth": "#9467bd",
            "ExceptPetri": "#ff7f0e",
        }

        usatge_source_weight: Dict[str, Dict[str, float]] = {}
        for u, v, d in self.G.edges(data=True):
            grp = self.G.nodes[u].get("text_group", "?")
            if v not in usatge_source_weight:
                usatge_source_weight[v] = {}
            usatge_source_weight[v][grp] = (
                usatge_source_weight[v].get(grp, 0.0) + d.get("weight", 0.0)
            )

        usatge_ids = sorted(
            usatge_source_weight.keys(),
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0,
        )
        groups = ["CorpusJuris", "LexVisigoth", "ExceptPetri", "Evangelium", "Etymologiae"]

        labels: List[str] = list(usatge_ids)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        y_pos = np.arange(len(usatge_ids))
        left = np.zeros(len(usatge_ids))

        for grp in groups:
            vals = [usatge_source_weight[uid].get(grp, 0.0) for uid in usatge_ids]
            ax.barh(
                y_pos,
                vals,
                left=left,
                height=0.7,
                color=source_colors.get(grp, "#95a5a6"),
                label=grp,
                alpha=0.9,
            )
            left += np.array(vals)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Суммарный вес заимствований")
        ax.set_title(
            "Интенсивность заимствований по обычаям (по источникам)",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout(pad=0.8)
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
