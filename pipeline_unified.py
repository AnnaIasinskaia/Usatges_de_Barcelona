#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_unified.py

Версия с подробным пошаговым выводом.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:  # pragma: no cover
    plt = None

from preprocessing import LatinLemmatizer, preprocess_segment
from alignment import smith_waterman
from features import (
    build_tfidf_matrix,
    compute_idf,
    soft_cosine_similarity,
    tesserae_score,
)


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    corpus: str
    side: str
    parent_id: str
    meta: Dict[str, Any] = field(default_factory=dict)


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


def resolve_group_tokens(items: Sequence[str], groups: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x.startswith("@"):
            out.extend(groups.get(x[1:], []))
        else:
            out.append(x)
    return out


def _normalize_segment_list(raw: Any) -> List[Tuple[str, str, Dict[str, Any]]]:
    if raw is None:
        return []

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for item in raw:
        if isinstance(item, dict):
            seg_id = str(item.get("id", ""))
            seg_text = str(item.get("text", ""))
            meta = dict(item.get("metadata") or {})
            for k in ("year", "doc_no", "doc_num", "volume", "date_key"):
                if k in item and k not in meta:
                    meta[k] = item[k]
            if seg_id and seg_text:
                out.append((seg_id, seg_text, meta))
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1]), {}))
            continue

        raise TypeError(f"Unsupported segment shape: {type(item)} => {repr(item)[:200]}")
    return out


def segment_corpus(corpus_id: str, corpus_spec: Dict[str, Any], logger: ProgressLogger) -> List[Tuple[str, str, Dict[str, Any]]]:
    source_file = Path(corpus_spec["path"])
    logger.log(f"segment_corpus: {corpus_id} -> {source_file}")
    if not source_file.exists():
        raise FileNotFoundError(f"Corpus file not found for {corpus_id}: {source_file}")

    seg_mod = importlib.import_module("source_segmenters")
    raw = seg_mod.segment_source(source_file, corpus_id)
    norm = _normalize_segment_list(raw)
    logger.log(f"segment_corpus: {corpus_id} done, segments={len(norm)}")
    return norm


def _word_tokens(text: str) -> List[str]:
    return [w for w in text.split() if w]


def chunk_segments(base: Sequence[Segment], chunking_cfg: Dict[str, Any], logger: ProgressLogger, side: str) -> List[Segment]:
    if not chunking_cfg or not bool(chunking_cfg.get("enabled")):
        logger.log(f"chunking[{side}]: disabled, keep {len(base)} segments")
        return list(base)

    mode = chunking_cfg.get("mode", "sliding_window_words")
    if mode != "sliding_window_words":
        raise ValueError(f"Unsupported chunking mode: {mode}")

    w_default = int(chunking_cfg.get("window_words", 180))
    o_default = int(chunking_cfg.get("overlap_words", 60))
    min_default = int(chunking_cfg.get("min_words", 20))
    per = dict(chunking_cfg.get("per_corpus") or {})

    out: List[Segment] = []
    produced = 0

    for seg in base:
        ov = dict(per.get(seg.corpus) or {})
        if ov.get("enabled", True) is False:
            out.append(seg)
            produced += 1
            continue

        window_words = int(ov.get("window_words", w_default))
        overlap_words = int(ov.get("overlap_words", o_default))
        min_words = int(ov.get("min_words", min_default))

        words = _word_tokens(seg.text)
        if len(words) < min_words or len(words) <= window_words:
            out.append(seg)
            produced += 1
            continue

        step = max(1, window_words - max(0, overlap_words))
        idx = 0
        for start in range(0, len(words), step):
            end = min(len(words), start + window_words)
            if end - start < min_words:
                continue
            chunk_text = " ".join(words[start:end])
            leaf_id = f"{seg.parent_id}__w{idx}_W{start}-{end}"

            meta = dict(seg.meta)
            meta.update({
                "window_start": start,
                "window_end": end,
                "window_words": end - start,
            })

            out.append(Segment(
                id=leaf_id,
                text=chunk_text,
                corpus=seg.corpus,
                side=seg.side,
                parent_id=seg.parent_id,
                meta=meta,
            ))
            produced += 1
            idx += 1
            if end == len(words):
                break

    logger.log(f"chunking[{side}]: input={len(base)} output={produced}")
    return out


_YEAR_RE_1 = re.compile(r"_Y(\d{3,4})")
_DOC_RE_1 = re.compile(r"_doc(\d+)", re.IGNORECASE)
_DOC_RE_2 = re.compile(r"_D(\d+)")
_NUM_RE = re.compile(r"\d+")


def extract_charter_year(seg: Segment) -> Optional[int]:
    for key in ("year",):
        if seg.meta.get(key) is not None:
            try:
                return int(seg.meta[key])
            except Exception:
                pass
    m = _YEAR_RE_1.search(seg.parent_id)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def extract_charter_doc_no(seg: Segment) -> Optional[int]:
    for key in ("doc_no", "doc_num"):
        if seg.meta.get(key) is not None:
            try:
                return int(seg.meta[key])
            except Exception:
                pass
    m = _DOC_RE_1.search(seg.parent_id)
    if m:
        return int(m.group(1))
    m = _DOC_RE_2.search(seg.parent_id)
    if m:
        return int(m.group(1))
    return None


def generic_numeric_sort_key(seg_id: str) -> Tuple[int, str]:
    nums = _NUM_RE.findall(str(seg_id))
    if nums:
        return (int(nums[0]), str(seg_id))
    return (10**9, str(seg_id))


def is_charter_corpus(corpus_id: str, corpus_spec: Dict[str, Any]) -> bool:
    kind = str(corpus_spec.get("kind", "")).lower()
    if kind == "charters":
        return True
    return corpus_id in {"Gramoty911", "Gramoty12"}


def lemmatize_segments(
    segments: Sequence[Segment],
    lemmatizer: LatinLemmatizer,
    min_lemma_length: int,
    logger: ProgressLogger,
    label: str,
) -> Dict[str, List[str]]:
    total = len(segments)
    logger.log(f"lemmatize[{label}]: start, segments={total}")
    lemmas_by_id: Dict[str, List[str]] = {}
    for i, seg in enumerate(segments, 1):
        try:
            lem = preprocess_segment(seg.text, lemmatizer, min_length=min_lemma_length)
        except Exception:
            lem = []
        lemmas_by_id[seg.id] = lem
        if i % 100 == 0 or i == total:
            logger.log(f"lemmatize[{label}]: {i}/{total}")
    return lemmas_by_id


def cosine_candidates_dense(
    tfidf_left: np.ndarray,
    tfidf_right: np.ndarray,
    left_ids: List[str],
    right_ids: List[str],
    threshold: float,
    top_k_per_left: Optional[int],
    logger: ProgressLogger,
) -> List[Tuple[str, str, float]]:
    logger.log(
        f"candidates: cosine start, left={tfidf_left.shape}, right={tfidf_right.shape}, "
        f"threshold={threshold}, top_k={top_k_per_left}"
    )
    sim = tfidf_left @ tfidf_right.T

    pairs: List[Tuple[str, str, float]] = []
    if top_k_per_left is None:
        rows, cols = np.where(sim >= threshold)
        for i, j in zip(rows, cols):
            pairs.append((left_ids[int(i)], right_ids[int(j)], float(sim[int(i), int(j)])))
        logger.log(f"candidates: found={len(pairs)}")
        return pairs

    k = int(top_k_per_left)
    for i in range(sim.shape[0]):
        row = sim[i]
        if row.size == 0:
            continue
        if k >= row.size:
            idx = np.argsort(row)[::-1]
        else:
            idx = np.argpartition(row, -k)[-k:]
            idx = idx[np.argsort(row[idx])[::-1]]

        for j in idx:
            v = float(row[int(j)])
            if v >= threshold:
                pairs.append((left_ids[i], right_ids[int(j)], v))

        if (i + 1) % 100 == 0 or i + 1 == sim.shape[0]:
            logger.log(f"candidates: processed left rows {i + 1}/{sim.shape[0]}, pairs={len(pairs)}")

    logger.log(f"candidates: found={len(pairs)}")
    return pairs


def compute_borrow_score(
    cos_sim: float,
    left_lem: List[str],
    right_lem: List[str],
    idf: Dict[str, float],
    model: Dict[str, Any],
) -> Tuple[float, float, float]:
    tess = float(tesserae_score(left_lem, right_lem, idf))
    if cos_sim + tess * float(model["beta"]) > float(model["final_threshold"]) * 0.5:
        soft = float(
            soft_cosine_similarity(
                left_lem,
                right_lem,
                max_terms=int(model["soft_cosine_max_terms"]),
            )
        )
    else:
        soft = 0.0

    score = (
        float(model["alpha"]) * float(cos_sim)
        + float(model["beta"]) * tess
        + float(model["gamma"]) * soft
    )
    return score, tess, soft


def maybe_align(
    left_lem: List[str],
    right_lem: List[str],
    model: Dict[str, Any],
    enabled: bool,
) -> Tuple[List[str], List[str], float]:
    if not enabled:
        return [], [], 0.0

    al_a, al_b, sw = smith_waterman(
        left_lem,
        right_lem,
        match_score=int(model["sw_match"]),
        mismatch_score=int(model["sw_mismatch"]),
        gap_penalty=int(model["sw_gap"]),
        lev_bonus_threshold=int(model["sw_lev_bonus_threshold"]),
        max_seq_len=int(model["sw_max_seq_len"]),
    )
    return al_a, al_b, float(sw)


def node_id_for_level(seg: Segment, level: str) -> str:
    if level == "leaf":
        return seg.id
    if level == "parent":
        return seg.parent_id
    if level == "corpus":
        return seg.corpus
    raise ValueError(f"Unknown node level: {level}")


def aggregate_rows(
    detail_rows: Sequence[Dict[str, Any]],
    left_level: str,
    right_level: str,
    weight_mode: str,
    min_hits: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in detail_rows:
        left_node = str(row["left_node"])
        right_node = str(row["right_node"])
        key = (left_node, right_node)
        score = float(row["score"])

        cur = grouped.get(key)
        if cur is None:
            grouped[key] = {
                "left_node": left_node,
                "right_node": right_node,
                "left_level": left_level,
                "right_level": right_level,
                "weight": score,
                "hit_count": 1,
                "best_score": score,
                "best_left_leaf_id": row.get("left_leaf_id", ""),
                "best_right_leaf_id": row.get("right_leaf_id", ""),
                "best_left_parent_id": row.get("left_parent_id", ""),
                "best_right_parent_id": row.get("right_parent_id", ""),
                "left_corpus": row.get("left_corpus", ""),
                "right_corpus": row.get("right_corpus", ""),
                "right_year": row.get("right_year"),
                "right_doc_no": row.get("right_doc_no"),
            }
        else:
            cur["hit_count"] += 1
            if weight_mode == "sum":
                cur["weight"] = float(cur["weight"]) + score
            elif weight_mode == "max":
                cur["weight"] = max(float(cur["weight"]), score)
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            if score > float(cur["best_score"]):
                cur["best_score"] = score
                cur["best_left_leaf_id"] = row.get("left_leaf_id", "")
                cur["best_right_leaf_id"] = row.get("right_leaf_id", "")
                cur["best_left_parent_id"] = row.get("left_parent_id", "")
                cur["best_right_parent_id"] = row.get("right_parent_id", "")

    rows = [r for r in grouped.values() if int(r.get("hit_count", 1)) >= int(min_hits)]
    rows.sort(key=lambda r: (-float(r["weight"]), str(r["left_node"]), str(r["right_node"])))
    return rows

def render_bipartite_graph(
    graph_rows: Sequence[Dict[str, Any]],
    left_nodes: Dict[str, Dict[str, Any]],
    right_nodes: Dict[str, Dict[str, Any]],
    out_png: Path,
    straight_edges: bool = True,
    label_left: bool = True,
    label_right: bool = True,
    top_n_edges: Optional[int] = None,
) -> None:
    if plt is None or nx is None:
        return

    rows = list(graph_rows)
    if top_n_edges is not None:
        try:
            k = int(top_n_edges)
        except Exception:
            k = None
        if k is not None and k > 0 and len(rows) > k:
            rows = sorted(rows, key=lambda r: -float(r.get("weight", 0.0)))[:k]

    G = nx.DiGraph()
    for r in rows:
        u = str(r["left_node"])
        v = str(r["right_node"])
        w = float(r.get("weight", 0.0))
        if u not in left_nodes or v not in right_nodes:
            continue
        G.add_node(u, **left_nodes[u])
        G.add_node(v, **right_nodes[v])
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() == 0:
        return

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


def build_node_metadata(
    segments: Sequence[Segment],
    corpora: Dict[str, Dict[str, Any]],
    node_level: str,
    side: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for seg in segments:
        if seg.side != side:
            continue

        node_id = node_id_for_level(seg, node_level)
        if node_id in out:
            continue

        corpus_spec = corpora.get(seg.corpus, {})
        color = corpus_spec.get("color", "#999999")
        label = corpus_spec.get("display_ru", seg.corpus) if node_level == "corpus" else node_id

        if is_charter_corpus(seg.corpus, corpus_spec):
            year = extract_charter_year(seg) or 999999
            doc_no = extract_charter_doc_no(seg) or 999999
            sort_key = (year, doc_no, node_id)
        else:
            sort_key = generic_numeric_sort_key(node_id)

        out[node_id] = {
            "side": side,
            "group": seg.corpus,
            "label": label,
            "color": color,
            "sort_key": sort_key,
        }
    return out


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)



def _make_gexf_safe_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит атрибуты узла/ребра к типам, которые умеет писать networkx.write_gexf.
    В частности:
    - tuple/list/dict -> строка
    - Path -> строка
    - None -> пустая строка
    Простые скаляры (str/int/float/bool) оставляются как есть.
    """
    safe: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif value is None:
            safe[key] = ""
        else:
            safe[key] = repr(value)
    return safe

def run_experiment(
    config_module: Any,
    experiment_id: str,
    verbose: bool = True,
    progress_every: int = 1000,
) -> Dict[str, Any]:
    logger = ProgressLogger(enabled=verbose)

    logger.log(f"load config: start experiment={experiment_id}")
    corpora: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "CORPORA"))
    groups: Dict[str, List[str]] = dict(getattr(config_module, "GROUPS", {}))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "EXPERIMENTS"))

    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")

    exp = experiments[experiment_id]
    out_dir = Path(exp["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"output dir: {out_dir}")

    left_corpora = resolve_group_tokens(exp["graph_sides"]["left"], groups)
    right_corpora = resolve_group_tokens(exp["graph_sides"]["right"], groups)
    logger.log(f"left corpora: {left_corpora}")
    logger.log(f"right corpora: {right_corpora}")

    mappings = exp.get("mappings") or [{"from": left_corpora, "to": right_corpora}]
    resolved_mappings = []
    for m in mappings:
        frm = resolve_group_tokens(m.get("from", []), groups)
        to = resolve_group_tokens(m.get("to", []), groups)
        resolved_mappings.append((frm, to))
    logger.log(f"mappings: {resolved_mappings}")

    base_segments: List[Segment] = []
    used_corpora = sorted(set(left_corpora + right_corpora))
    logger.log(f"segment all corpora: count={len(used_corpora)}")

    for cid in used_corpora:
        if cid not in corpora:
            raise KeyError(f"Corpus {cid} is not defined in CORPORA")
        segs = segment_corpus(cid, corpora[cid], logger)
        for seg_id, seg_text, meta in segs:
            base_segments.append(Segment(
                id=seg_id,
                text=seg_text,
                corpus=cid,
                side="__unassigned__",
                parent_id=seg_id,
                meta=meta,
            ))
    logger.log(f"base segments total: {len(base_segments)}")

    base_by_corpus: Dict[str, List[Segment]] = {cid: [] for cid in used_corpora}
    for seg in base_segments:
        base_by_corpus[seg.corpus].append(seg)

    left_base: List[Segment] = []
    for cid in left_corpora:
        for seg in base_by_corpus[cid]:
            left_base.append(Segment(seg.id, seg.text, seg.corpus, "left", seg.parent_id, dict(seg.meta)))

    right_base: List[Segment] = []
    for cid in right_corpora:
        for seg in base_by_corpus[cid]:
            right_base.append(Segment(seg.id, seg.text, seg.corpus, "right", seg.parent_id, dict(seg.meta)))

    logger.log(f"left base={len(left_base)}, right base={len(right_base)}")

    chunk_cfg = dict(exp.get("chunking") or {})
    left_leaf = chunk_segments(left_base, chunk_cfg, logger, "left")
    right_leaf = chunk_segments(right_base, chunk_cfg, logger, "right")
    logger.log(f"left leaf={len(left_leaf)}, right leaf={len(right_leaf)}")

    model = dict(exp.get("model") or {})
    logger.log("lemmatizer: init")
    lemmatizer = LatinLemmatizer(use_collatinus=bool(model.get("use_collatinus", False)))
    min_lemma_length = int(model.get("min_lemma_length", 3))

    left_lemmas = lemmatize_segments(left_leaf, lemmatizer, min_lemma_length, logger, "left")
    right_lemmas = lemmatize_segments(right_leaf, lemmatizer, min_lemma_length, logger, "right")

    left_ids = [s.id for s in left_leaf]
    right_ids = [s.id for s in right_leaf]
    corpus_lemmas = [left_lemmas[i] for i in left_ids] + [right_lemmas[i] for i in right_ids]

    logger.log(f"tfidf: start, docs={len(corpus_lemmas)}")
    tfidf, vocab, term2idx = build_tfidf_matrix(
        corpus_lemmas,
        ngram_range=tuple(model.get("ngram_range", (1, 3))),
        max_df=float(model.get("max_df", 0.5)),
        min_df=int(model.get("min_df", 2)),
    )
    logger.log(f"tfidf: done, matrix_shape={getattr(tfidf, 'shape', None)}, vocab={len(vocab)}")

    n_left = len(left_ids)
    tfidf_left = tfidf[:n_left]
    tfidf_right = tfidf[n_left:]

    cand_cfg = dict(exp.get("candidate_selection") or {})
    threshold = float(cand_cfg.get("threshold", model.get("tfidf_cosine_threshold", 0.08)))
    top_k = cand_cfg.get("top_k_per_left")

    candidates = cosine_candidates_dense(
        tfidf_left,
        tfidf_right,
        left_ids=left_ids,
        right_ids=right_ids,
        threshold=threshold,
        top_k_per_left=top_k,
        logger=logger,
    )

    logger.log("idf: compute")
    idf = compute_idf(corpus_lemmas)
    align_enabled = bool((exp.get("alignment") or {}).get("enabled", True))
    logger.log(f"alignment enabled={align_enabled}")

    left_seg_by_id = {s.id: s for s in left_leaf}
    right_seg_by_id = {s.id: s for s in right_leaf}

    filters = dict(exp.get("filters") or {})
    not_before_map = dict(filters.get("right_not_before_by_left") or {})

    agg = dict(exp.get("aggregation") or {})
    left_level = str(agg.get("left_node_level", "parent"))
    right_level = str(agg.get("right_node_level", "parent"))

    detail_rows: List[Dict[str, Any]] = []
    logger.log(f"score candidates: total={len(candidates)}")

    for idx, (left_id, right_id, cos_sim) in enumerate(candidates, 1):
        ls = left_seg_by_id.get(left_id)
        rs = right_seg_by_id.get(right_id)
        if ls is None or rs is None:
            continue

        allowed = False
        for frm, to in resolved_mappings:
            if ls.corpus in frm and rs.corpus in to:
                allowed = True
                break
        if not allowed:
            continue

        nb = not_before_map.get(ls.corpus)
        if nb is not None:
            y = extract_charter_year(rs)
            if y is not None and int(y) < int(nb):
                continue

        llem = left_lemmas.get(left_id, [])
        rlem = right_lemmas.get(right_id, [])
        if not llem or not rlem:
            continue

        score, tess, soft = compute_borrow_score(float(cos_sim), llem, rlem, idf, model)
        if score < float(model.get("final_threshold", 0.10)):
            continue

        al_a, al_b, sw = maybe_align(llem, rlem, model, enabled=align_enabled)
        if sw < float(model.get("sw_min_score", 0.0)):
            continue

        detail_rows.append({
            "left_leaf_id": left_id,
            "right_leaf_id": right_id,
            "left_parent_id": ls.parent_id,
            "right_parent_id": rs.parent_id,
            "left_corpus": ls.corpus,
            "right_corpus": rs.corpus,
            "cos_sim": float(cos_sim),
            "tesserae": float(tess),
            "soft_cos": float(soft),
            "score": float(score),
            "sw_score": float(sw),
            "left_node": node_id_for_level(ls, left_level),
            "right_node": node_id_for_level(rs, right_level),
            "alignment_a": al_a[:25],
            "alignment_b": al_b[:25],
            "right_year": extract_charter_year(rs),
            "right_doc_no": extract_charter_doc_no(rs),
            "left_text_snippet": ls.text[:220].replace("\n", " ").strip(),
            "right_text_snippet": rs.text[:220].replace("\n", " ").strip(),
        })

        if idx % max(1, progress_every) == 0 or idx == len(candidates):
            logger.log(f"score candidates: {idx}/{len(candidates)}, kept={len(detail_rows)}")

    detail_rows.sort(
        key=lambda r: (
            -float(r["score"]),
            str(r["left_corpus"]),
            str(r["right_corpus"]),
            str(r["left_leaf_id"]),
            str(r["right_leaf_id"]),
        )
    )
    logger.log(f"detail rows: {len(detail_rows)}")

    graph_rows = aggregate_rows(
        detail_rows,
        left_level=left_level,
        right_level=right_level,
        weight_mode=str(agg.get("weight_mode", "max")),
        min_hits=int(agg.get("min_hits", 1)),
    )
    logger.log(f"graph rows: {len(graph_rows)}")

    output_cfg = dict(exp.get("output") or {})
    if bool(output_cfg.get("write_detail_csv")):
        path = out_dir / "detail_pairs.csv"
        logger.log(f"write csv: {path}")
        write_csv(detail_rows, path)
    if bool(output_cfg.get("write_graph_csv")):
        path = out_dir / "graph_rows.csv"
        logger.log(f"write csv: {path}")
        write_csv(graph_rows, path)

    left_meta = build_node_metadata(left_leaf, corpora, node_level=left_level, side="left")
    right_meta = build_node_metadata(right_leaf, corpora, node_level=right_level, side="right")

    G = None
    if nx is not None:
        logger.log("build networkx graph")
        G = nx.DiGraph()
        for nid, meta in left_meta.items():
            G.add_node(nid, **meta)
        for nid, meta in right_meta.items():
            G.add_node(nid, **meta)
        for r in graph_rows:
            u = str(r["left_node"])
            v = str(r["right_node"])
            if u in left_meta and v in right_meta:
                G.add_edge(u, v, weight=float(r.get("weight", 0.0)), hit_count=int(r.get("hit_count", 1)))

        if bool(output_cfg.get("write_gexf")):
            path = out_dir / "graph.gexf"
            logger.log(f"write gexf: {path}")

            # Для GEXF нужен отдельный "безопасный" граф:
            # networkx.write_gexf не принимает tuple/list/dict как значения атрибутов.
            G_gexf = nx.DiGraph()
            for nid, meta in left_meta.items():
                G_gexf.add_node(nid, **_make_gexf_safe_attrs(meta))
            for nid, meta in right_meta.items():
                G_gexf.add_node(nid, **_make_gexf_safe_attrs(meta))
            for r in graph_rows:
                u = str(r["left_node"])
                v = str(r["right_node"])
                if u in left_meta and v in right_meta:
                    edge_attrs = _make_gexf_safe_attrs({
                        "weight": float(r.get("weight", 0.0)),
                        "hit_count": int(r.get("hit_count", 1)),
                        "left_level": r.get("left_level", ""),
                        "right_level": r.get("right_level", ""),
                        "left_corpus": r.get("left_corpus", ""),
                        "right_corpus": r.get("right_corpus", ""),
                        "best_score": r.get("best_score", ""),
                        "best_left_leaf_id": r.get("best_left_leaf_id", ""),
                        "best_right_leaf_id": r.get("best_right_leaf_id", ""),
                        "best_left_parent_id": r.get("best_left_parent_id", ""),
                        "best_right_parent_id": r.get("best_right_parent_id", ""),
                        "right_year": r.get("right_year", ""),
                        "right_doc_no": r.get("right_doc_no", ""),
                    })
                    G_gexf.add_edge(u, v, **edge_attrs)

            nx.write_gexf(G_gexf, path)

    viz_cfg = dict(exp.get("viz") or {})
    if bool(viz_cfg.get("enabled")) and bool(output_cfg.get("write_png")):
        path = out_dir / "graph.png"
        logger.log(
            f"render png: {path} (top_n_edges={viz_cfg.get('top_n_edges')}, total_graph_rows={len(graph_rows)})"
        )
        render_bipartite_graph(
            graph_rows=graph_rows,
            left_nodes=left_meta,
            right_nodes=right_meta,
            out_png=path,
            straight_edges=bool(viz_cfg.get("straight_edges", True)),
            label_left=bool(viz_cfg.get("label_left", True)),
            label_right=bool(viz_cfg.get("label_right", True)),
            top_n_edges=viz_cfg.get("top_n_edges"),
        )

    logger.log("experiment finished")
    return {
        "experiment_id": experiment_id,
        "detail_rows": detail_rows,
        "graph_rows": graph_rows,
        "out_dir": str(out_dir),
        "graph": G,
        "stats": {
            "left_leaf": len(left_leaf),
            "right_leaf": len(right_leaf),
            "candidates": len(candidates),
            "detail_rows": len(detail_rows),
            "graph_rows": len(graph_rows),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_unified", help="Python module name, e.g. config_unified")
    ap.add_argument("--experiment", required=True, help="Experiment id, see EXPERIMENTS in config")
    ap.add_argument("--quiet", action="store_true", help="Disable progress logs")
    ap.add_argument("--progress-every", type=int, default=1000, help="Progress print period for candidate scoring")
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)
    result = run_experiment(
        cfg_mod,
        args.experiment,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )

    stats = result.get("stats") or {}
    print(f"[unified] experiment={args.experiment}", flush=True)
    for k in sorted(stats.keys()):
        print(f"[unified] {k}: {stats[k]}", flush=True)
    print(f"[unified] out_dir: {result.get('out_dir')}", flush=True)


if __name__ == "__main__":
    main()
