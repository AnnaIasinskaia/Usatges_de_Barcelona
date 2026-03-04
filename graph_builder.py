"""
Step 7: Build and analyze the borrowing graph.
Includes multiple visualization strategies for readability.
"""
import csv
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


class BorrowingGraph:
    """Directed weighted graph of textual borrowings."""

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError("networkx is required: pip install networkx")
        self.G = nx.DiGraph()

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
        """Add a directed edge from source segment to usatge."""
        if not self.G.has_node(source_id):
            self.G.add_node(source_id, text_group=source_name, node_type="source",
                            text_snippet=source_text[:200] if source_text else "")
        if not self.G.has_node(target_id):
            self.G.add_node(target_id, text_group="Usatges", node_type="usatge",
                            text_snippet=usatge_text[:200] if usatge_text else "")

        align_str = ""
        if alignment_a and alignment_b:
            align_str = " | ".join(
                f"{a} ~ {b}" for a, b in zip(alignment_a[:20], alignment_b[:20])
            )

        self.G.add_edge(
            source_id, target_id,
            weight=round(weight, 4),
            alignment=align_str,
        )

    def export_gexf(self, path: Path):
        """Export graph in GEXF format (for Gephi)."""
        nx.write_gexf(self.G, str(path))

    def export_csv(self, path: Path, usatge_texts: Optional[Dict[str, str]] = None,
                   source_texts: Optional[Dict[str, str]] = None):
        """Export edges as CSV with text snippets for readability."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source_segment", "source_group", "target_usatge",
                "borrow_score", "alignment_sample",
                "usatge_text_snippet", "source_text_snippet",
            ])
            for u, v, data in sorted(self.G.edges(data=True),
                                      key=lambda x: x[2].get("weight", 0),
                                      reverse=True):
                u_text = ""
                v_text = ""
                if usatge_texts and v in usatge_texts:
                    v_text = usatge_texts[v][:150].replace("\n", " ").strip()
                if source_texts and u in source_texts:
                    u_text = source_texts[u][:150].replace("\n", " ").strip()

                writer.writerow([
                    u, self.G.nodes[u].get("text_group", ""),
                    v, data.get("weight", 0),
                    data.get("alignment", ""),
                    v_text, u_text,
                ])

    def get_stats(self) -> Dict:
        """Basic graph statistics."""
        usatge_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "usatge"]
        source_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "source"]

        stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "usatge_nodes": len(usatge_nodes),
            "source_nodes": len(source_nodes),
        }

        if usatge_nodes:
            in_deg = sorted(
                [(n, self.G.in_degree(n)) for n in usatge_nodes],
                key=lambda x: x[1], reverse=True,
            )
            stats["most_dependent_usatges"] = in_deg[:10]

        if source_nodes:
            out_deg = sorted(
                [(n, self.G.out_degree(n)) for n in source_nodes],
                key=lambda x: x[1], reverse=True,
            )
            stats["most_influential_sources"] = out_deg[:10]

        group_counts = {}
        group_weights = {}
        for u, v, data in self.G.edges(data=True):
            grp = self.G.nodes[u].get("text_group", "unknown")
            group_counts[grp] = group_counts.get(grp, 0) + 1
            group_weights[grp] = group_weights.get(grp, 0) + data.get("weight", 0)
        stats["edges_by_source"] = group_counts
        stats["total_weight_by_source"] = {
            k: round(v, 3) for k, v in group_weights.items()
        }

        return stats

    def format_stats(self, usatge_texts: Optional[Dict[str, str]] = None) -> str:
        """Format stats as human-readable text with usatge snippets."""
        stats = self.get_stats()
        lines = []
        lines.append("=" * 70)
        lines.append("PIPELINE RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Total nodes: {stats['total_nodes']}")
        lines.append(f"  Total edges (borrowings): {stats['total_edges']}")
        lines.append(f"  Usatge nodes: {stats['usatge_nodes']}")
        lines.append(f"  Source nodes: {stats['source_nodes']}")
        lines.append(f"  Edges by source: {stats.get('edges_by_source', {})}")
        lines.append(f"  Total weight by source: {stats.get('total_weight_by_source', {})}")

        if "most_dependent_usatges" in stats:
            lines.append("")
            lines.append("  Top most dependent Usatges (highest incoming borrowings):")
            lines.append("  " + "-" * 66)
            for name, deg in stats["most_dependent_usatges"][:10]:
                snippet = ""
                if usatge_texts and name in usatge_texts:
                    raw = usatge_texts[name].replace("\n", " ").strip()
                    snippet = textwrap.shorten(raw, width=80, placeholder="...")
                lines.append(f"    {name} ({deg} borrowings)")
                if snippet:
                    lines.append(f"      Text: {snippet}")

        return "\n".join(lines)

    # ---------- VISUALIZATION 1: Aggregated bipartite heatmap ----------

    def visualize_heatmap(self, output_path: Path,
                          usatge_texts: Optional[Dict[str, str]] = None):
        """
        Heatmap: rows = Usatges, columns = source groups.
        Cell color = sum of BorrowScores for that pair.
        Much more readable than a raw graph for 180+ edges.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        # Aggregate edge weights by (usatge, source_group)
        usatge_nodes = sorted(set(
            v for _, v, _ in self.G.edges(data=True)
            if self.G.nodes[v].get("node_type") == "usatge"
        ), key=lambda x: (int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0))

        source_groups = sorted(set(
            self.G.nodes[u].get("text_group", "?")
            for u, _, _ in self.G.edges(data=True)
        ))

        if not usatge_nodes or not source_groups:
            return

        # Build matrix
        matrix = np.zeros((len(usatge_nodes), len(source_groups)))
        u_idx = {n: i for i, n in enumerate(usatge_nodes)}
        g_idx = {g: i for i, g in enumerate(source_groups)}

        for src, tgt, data in self.G.edges(data=True):
            grp = self.G.nodes[src].get("text_group", "?")
            if tgt in u_idx and grp in g_idx:
                matrix[u_idx[tgt], g_idx[grp]] += data.get("weight", 0)

        # Build row labels with text snippets
        row_labels = []
        for u in usatge_nodes:
            label = u
            if usatge_texts and u in usatge_texts:
                raw = usatge_texts[u].replace("\n", " ").strip()
                snippet = textwrap.shorten(raw, width=50, placeholder="...")
                label = f"{u}: {snippet}"
            row_labels.append(label)

        fig, ax = plt.subplots(figsize=(10, max(8, len(usatge_nodes) * 0.38)))

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

        ax.set_xticks(range(len(source_groups)))
        ax.set_xticklabels(source_groups, rotation=45, ha="right", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(usatge_nodes)))
        ax.set_yticklabels(row_labels, fontsize=7.5)

        # Annotate cells with values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() * 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, label="Cumulative BorrowScore", shrink=0.8)
        ax.set_title("Borrowing Intensity: Usatges de Barcelona ← Latin Sources",
                      fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Source Text", fontsize=11)
        ax.set_ylabel("Usatge Article", fontsize=11)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- VISUALIZATION 2: Per-source focused graphs ----------

    def visualize_per_source(self, output_dir: Path,
                             usatge_texts: Optional[Dict[str, str]] = None):
        """
        One clean graph per source: only edges from that source.
        Much less cluttered than one giant graph.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        source_groups = set(
            self.G.nodes[u].get("text_group", "?")
            for u, _, _ in self.G.edges(data=True)
        )

        source_colors = {
            "CorpusJuris": "#e74c3c",
            "Evangelium": "#3498db",
            "Etymologiae": "#f39c12",
            "LexVisigoth": "#9b59b6",
            "ExceptPetri": "#1abc9c",
        }

        for grp in sorted(source_groups):
            # Subgraph for this source
            edges = [
                (u, v, d) for u, v, d in self.G.edges(data=True)
                if self.G.nodes[u].get("text_group") == grp
            ]
            if not edges:
                continue

            sub = nx.DiGraph()
            for u, v, d in edges:
                sub.add_node(u, node_type="source")
                sub.add_node(v, node_type="usatge")
                sub.add_edge(u, v, **d)

            usatge_nodes = [n for n, d in sub.nodes(data=True) if d.get("node_type") == "usatge"]
            source_nodes = [n for n, d in sub.nodes(data=True) if d.get("node_type") == "source"]

            # Bipartite layout: sources on left, usatges on right
            pos = {}
            for i, n in enumerate(sorted(source_nodes)):
                pos[n] = (0, -i)
            for i, n in enumerate(sorted(usatge_nodes,
                    key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)):
                pos[n] = (3, -i * (len(source_nodes) / max(len(usatge_nodes), 1)))

            fig, ax = plt.subplots(figsize=(14, max(6, len(source_nodes) * 0.25)))

            color = source_colors.get(grp, "#7f8c8d")

            # Draw source nodes
            nx.draw_networkx_nodes(sub, pos, nodelist=source_nodes,
                                   node_color=color, node_size=40, alpha=0.6, ax=ax)
            # Draw usatge nodes
            nx.draw_networkx_nodes(sub, pos, nodelist=usatge_nodes,
                                   node_color="#2ecc71", node_size=120, alpha=0.8, ax=ax)

            # Edges with width proportional to weight
            weights = [d.get("weight", 0.1) * 5 for _, _, d in sub.edges(data=True)]
            nx.draw_networkx_edges(sub, pos, width=weights, alpha=0.35,
                                   edge_color=color, arrows=True, arrowsize=8, ax=ax)

            # Labels for usatge nodes with text snippets
            labels = {}
            for n in usatge_nodes:
                label = n
                if usatge_texts and n in usatge_texts:
                    raw = usatge_texts[n].replace("\n", " ").strip()
                    snippet = textwrap.shorten(raw, width=40, placeholder="...")
                    label = f"{n}\n{snippet}"
                labels[n] = label

            nx.draw_networkx_labels(sub, pos, labels, font_size=6,
                                    font_color="#2c3e50", ax=ax,
                                    horizontalalignment="left")

            n_edges = sub.number_of_edges()
            n_usatges = len(usatge_nodes)
            ax.set_title(f"{grp} → Usatges ({n_edges} borrowings, {n_usatges} articles)",
                          fontsize=13, fontweight="bold")
            ax.axis("off")
            plt.tight_layout()

            fname = output_dir / f"graph_{grp}.png"
            plt.savefig(str(fname), dpi=150, bbox_inches="tight")
            plt.close()

    # ---------- VISUALIZATION 3: Top-N strongest borrowings ----------

    def visualize_top_borrowings(self, output_path: Path, top_n: int = 30,
                                  usatge_texts: Optional[Dict[str, str]] = None):
        """
        Clean graph of only the top-N strongest borrowing edges.
        Removes clutter while preserving the most important connections.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            return

        # Get top edges
        all_edges = sorted(self.G.edges(data=True),
                           key=lambda x: x[2].get("weight", 0), reverse=True)
        top_edges = all_edges[:top_n]

        sub = nx.DiGraph()
        for u, v, d in top_edges:
            grp = self.G.nodes[u].get("text_group", "?")
            sub.add_node(u, text_group=grp, node_type="source")
            sub.add_node(v, text_group="Usatges", node_type="usatge")
            sub.add_edge(u, v, **d)

        source_colors = {
            "CorpusJuris": "#e74c3c",
            "Evangelium": "#3498db",
            "Etymologiae": "#f39c12",
            "LexVisigoth": "#9b59b6",
            "ExceptPetri": "#1abc9c",
            "Usatges": "#2ecc71",
        }

        node_colors = [
            source_colors.get(sub.nodes[n].get("text_group", ""), "#95a5a6")
            for n in sub.nodes()
        ]

        node_sizes = [
            200 if sub.nodes[n].get("node_type") == "usatge" else 60
            for n in sub.nodes()
        ]

        fig, ax = plt.subplots(figsize=(16, 12))

        pos = nx.kamada_kawai_layout(sub)

        weights = [d.get("weight", 0.1) * 6 for _, _, d in sub.edges(data=True)]
        edge_colors = [
            source_colors.get(sub.nodes[u].get("text_group", ""), "#7f8c8d")
            for u, v in sub.edges()
        ]

        nx.draw_networkx_nodes(sub, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.85, ax=ax)
        nx.draw_networkx_edges(sub, pos, width=weights, alpha=0.4,
                               edge_color=edge_colors, arrows=True,
                               arrowsize=10, ax=ax)

        # Labels: usatge nodes get text snippets
        labels = {}
        for n in sub.nodes():
            if sub.nodes[n].get("node_type") == "usatge":
                label = n
                if usatge_texts and n in usatge_texts:
                    raw = usatge_texts[n].replace("\n", " ").strip()
                    snippet = textwrap.shorten(raw, width=30, placeholder="...")
                    label = f"{n}\n{snippet}"
                labels[n] = label

        nx.draw_networkx_labels(sub, pos, labels, font_size=6.5,
                                font_color="#2c3e50", ax=ax)

        legend_elements = [
            Patch(facecolor=c, label=g)
            for g, c in source_colors.items()
            if g in set(sub.nodes[n].get("text_group", "") for n in sub.nodes())
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
                  framealpha=0.9)

        ax.set_title(f"Top {top_n} Strongest Borrowings: Usatges ← Latin Sources",
                      fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- VISUALIZATION 4: Horizontal bar chart by usatge ----------

    def visualize_bar_chart(self, output_path: Path,
                            usatge_texts: Optional[Dict[str, str]] = None):
        """
        Stacked horizontal bar chart: one bar per usatge, 
        stacked by source contribution. Very readable overview.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        source_colors = {
            "CorpusJuris": "#e74c3c",
            "Evangelium": "#3498db",
            "Etymologiae": "#f39c12",
            "LexVisigoth": "#9b59b6",
            "ExceptPetri": "#1abc9c",
        }

        # Aggregate
        usatge_source_weight = {}
        for u, v, d in self.G.edges(data=True):
            grp = self.G.nodes[u].get("text_group", "?")
            if v not in usatge_source_weight:
                usatge_source_weight[v] = {}
            usatge_source_weight[v][grp] = usatge_source_weight[v].get(grp, 0) + d.get("weight", 0)

        # Sort usatges by number
        usatge_ids = sorted(usatge_source_weight.keys(),
                            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

        source_groups = sorted(source_colors.keys())

        # Build labels
        labels = []
        for uid in usatge_ids:
            label = uid
            if usatge_texts and uid in usatge_texts:
                raw = usatge_texts[uid].replace("\n", " ").strip()
                snippet = textwrap.shorten(raw, width=45, placeholder="...")
                label = f"{uid}: {snippet}"
            labels.append(label)

        fig, ax = plt.subplots(figsize=(12, max(6, len(usatge_ids) * 0.35)))

        y_pos = np.arange(len(usatge_ids))
        left = np.zeros(len(usatge_ids))

        for grp in source_groups:
            vals = [usatge_source_weight[uid].get(grp, 0) for uid in usatge_ids]
            bars = ax.barh(y_pos, vals, left=left, height=0.7,
                           color=source_colors.get(grp, "#95a5a6"), label=grp, alpha=0.85)
            left += np.array(vals)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Cumulative BorrowScore", fontsize=11)
        ax.set_title("Borrowing Intensity per Usatge (by Source)",
                      fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
