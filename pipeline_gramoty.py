#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_gramoty.py

Separate pipeline for the charters study.

Target graph shape
------------------
Left column:
    - source segments from the old source->Usatges graph, grouped by source
    - Usatges segments matched directly to charters, grouped as "Usatges"
Right column:
    - charter segments from both charter volumes, sorted by date

Edges:
    1) direct:   Usatge segment -> Charter segment
    2) projected: Source segment -> Charter segment (through Usatges)
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import docx  # python-docx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("python-docx is required for pipeline_gramoty.py") from exc

from graph_builder_gramoty import build_gramoty_graph

Segment = Tuple[str, str]


# ----------------------------- configuration helpers -----------------------------


def _load_module(name: str):
    return importlib.import_module(name)


def _first_attr(module: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


def _resolve_path(value: Any, data_dir: Optional[Path] = None) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if data_dir is not None:
        candidate = data_dir / path
        if candidate.exists():
            return candidate
    return path


def _discover_config(cfg_module: Any) -> Dict[str, Any]:
    data_dir = Path(_first_attr(cfg_module, "DATA_DIR", default="data"))
    output_dir = Path(_first_attr(cfg_module, "OUTPUT_DIR", default="output"))

    sources = _first_attr(cfg_module, "SOURCES", "SOURCE_FILES", default=None)
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("config_gramoty.py must expose SOURCES (dict[name -> path])")

    charters = _first_attr(
        cfg_module,
        "GRAMOTY",
        "CHARTERS",
        "GRAMOTY_FILES",
        "CHARTER_FILES",
        default=None,
    )
    if not isinstance(charters, dict) or not charters:
        raise RuntimeError("config_gramoty.py must expose GRAMOTY/CHARTERS (dict[name -> path])")

    usatges_path = _first_attr(
        cfg_module,
        "USATGES_PATH",
        "USATGES_FILE",
        "USATGES",
        "USATGES_TXT",
        default=None,
    )
    if usatges_path is None:
        fallback = data_dir / "Bastardas Usatges de Barcelona_djvu.txt"
        if fallback.exists():
            usatges_path = fallback
        else:
            raise RuntimeError("Could not discover USATGES_PATH from config_gramoty.py")

    source_graph_path = _first_attr(
        cfg_module,
        "SOURCE_GRAPH_GEXF",
        "USATGES_GRAPH_GEXF",
        "BORROWING_GRAPH_GEXF",
        default=Path("output") / "borrowing_graph.gexf",
    )

    cosine_threshold = float(
        _first_attr(
            cfg_module,
            "GRAMOTY_COSINE_THRESHOLD",
            "CHARTER_COSINE_THRESHOLD",
            "TFIDF_COSINE_THRESHOLD",
            "COSINE_THRESHOLD",
            default=0.08,
        )
    )
    top_k = int(_first_attr(cfg_module, "GRAMOTY_TOP_K", "TOP_K", default=5))
    min_words = int(_first_attr(cfg_module, "GRAMOTY_MIN_WORDS", "MIN_SEGMENT_WORDS", default=12))
    max_segment_words = int(_first_attr(cfg_module, "MAX_SEGMENT_WORDS", default=200))
    graph_top_n = int(_first_attr(cfg_module, "GRAPH_TOP_N", default=120))

    source_names_ru = _first_attr(cfg_module, "SOURCE_NAMES_RU", default={})
    source_names_ru_short = _first_attr(cfg_module, "SOURCE_NAMES_RU_SHORT", default={})

    return {
        "DATA_DIR": data_dir,
        "OUTPUT_DIR": output_dir,
        "SOURCES": {k: _resolve_path(v, data_dir) for k, v in sources.items()},
        "GRAMOTY": {k: _resolve_path(v, data_dir) for k, v in charters.items()},
        "USATGES_PATH": _resolve_path(usatges_path, data_dir),
        "SOURCE_GRAPH_GEXF": _resolve_path(source_graph_path, output_dir.parent),
        "COSINE_THRESHOLD": cosine_threshold,
        "TOP_K": top_k,
        "MIN_WORDS": min_words,
        "MAX_SEGMENT_WORDS": max_segment_words,
        "GRAPH_TOP_N": graph_top_n,
        "SOURCE_NAMES_RU": source_names_ru,
        "SOURCE_NAMES_RU_SHORT": source_names_ru_short,
    }


# ----------------------------- loading -----------------------------


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        d = docx.Document(str(path))
        return "\n".join(par.text for par in d.paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


# ----------------------------- segmentation adapters -----------------------------


def _call_with_supported_kwargs(func: Callable, *args, **kwargs):
    sig = inspect.signature(func)
    accepted = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return func(*args, **accepted)


def segment_usatges(text: str, max_segment_words: int = 200) -> List[Segment]:
    mod = importlib.import_module("usatges_segmenter")
    if hasattr(mod, "segment_usatges"):
        return _call_with_supported_kwargs(mod.segment_usatges, text, max_segment_words=max_segment_words)
    for name in dir(mod):
        if name.startswith("segment_") and callable(getattr(mod, name)):
            return _call_with_supported_kwargs(getattr(mod, name), text, max_segment_words=max_segment_words)
    raise RuntimeError("Could not find segment_usatges in usatges_segmenter.py")


def segment_charter_text(text: str, source_name: str, max_segment_words: int = 200):
    mod = importlib.import_module("seg_gramoty_stable")
    for candidate in ("segment_gramoty_stable", "segment_gramoty", f"segment_{source_name.lower()}"):
        if hasattr(mod, candidate):
            return _call_with_supported_kwargs(
                getattr(mod, candidate),
                text,
                source_name,
                max_segment_words=max_segment_words,
            )
    for name in dir(mod):
        if name.startswith("segment_") and callable(getattr(mod, name)):
            return _call_with_supported_kwargs(
                getattr(mod, name),
                text,
                source_name,
                max_segment_words=max_segment_words,
            )
    raise RuntimeError("Could not find a charter segmenter in seg_gramoty_stable.py")


# ----------------------------- normalization -----------------------------


_LATIN_MAP = str.maketrans({"j": "i", "J": "I", "v": "u", "V": "U"})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LATIN_MAP)
    text = re.sub(r"æ", "e", text, flags=re.IGNORECASE)
    text = re.sub(r"œ", "e", text, flags=re.IGNORECASE)
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def maybe_preprocess_segments(segments: Sequence[Segment]) -> List[Segment]:
    try:
        preprocessing = importlib.import_module("preprocessing")
    except Exception:
        return [(sid, normalize_text(text)) for sid, text in segments]

    for candidate in ("preprocess_text", "normalize_latin", "preprocess_segment"):
        if hasattr(preprocessing, candidate):
            func = getattr(preprocessing, candidate)
            processed = []
            for sid, text in segments:
                try:
                    processed_text = func(text)
                except TypeError:
                    processed_text = func(text, lemmatize=False)
                processed.append((sid, processed_text))
            return processed

    return [(sid, normalize_text(text)) for sid, text in segments]


# ----------------------------- similarity -----------------------------


def _segment_word_count(text: str) -> int:
    return len([w for w in text.split() if w])


def _segment_to_pair(seg):
    if isinstance(seg, dict):
        seg_id = seg.get("id")
        seg_text = seg.get("text")
        if seg_id is None or seg_text is None:
            raise ValueError(f"Segment dict must contain 'id' and 'text': {seg}")
        return str(seg_id), str(seg_text)

    if isinstance(seg, (tuple, list)):
        if len(seg) < 2:
            raise ValueError(f"Segment tuple/list too short: {seg}")
        return str(seg[0]), str(seg[1])

    raise TypeError(f"Unsupported segment type: {type(seg)}")


def _normalize_segment_list(segments):
    return [_segment_to_pair(seg) for seg in segments]


def build_tfidf(texts: Sequence[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(texts)
    return vectorizer


def compute_usatges_to_charters(
    usatges_segments,
    charter_segments,
    cosine_threshold=0.08,
    top_k=5,
    min_words=12,
):
    usatges_segments = _normalize_segment_list(usatges_segments)
    charter_segments = _normalize_segment_list(charter_segments)

    usatges_segments = [(sid, txt) for sid, txt in usatges_segments if _segment_word_count(txt) >= min_words]
    charter_segments = [(sid, txt) for sid, txt in charter_segments if _segment_word_count(txt) >= min_words]

    prep_usatges = maybe_preprocess_segments(usatges_segments)
    prep_charters = maybe_preprocess_segments(charter_segments)

    all_texts = [txt for _, txt in prep_usatges] + [txt for _, txt in prep_charters]
    vectorizer = build_tfidf(all_texts)

    u_matrix = vectorizer.transform([txt for _, txt in prep_usatges])
    c_matrix = vectorizer.transform([txt for _, txt in prep_charters])
    sim = cosine_similarity(u_matrix, c_matrix)

    rows: List[Dict[str, Any]] = []
    for i, (u_id, u_raw) in enumerate(usatges_segments):
        order = sim[i].argsort()[::-1]
        kept = 0
        for j in order:
            score = float(sim[i, j])
            if score < cosine_threshold:
                break
            c_id, c_raw = charter_segments[j]
            rows.append(
                {
                    "usatge_id": u_id,
                    "charter_id": c_id,
                    "score": score,
                    "usatge_text": u_raw,
                    "charter_text": c_raw,
                }
            )
            kept += 1
            if kept >= top_k:
                break

    rows.sort(key=lambda row: (-float(row["score"]), str(row["usatge_id"]), str(row["charter_id"])))
    return rows


# ----------------------------- source graph projection -----------------------------


def _derive_source_group(source_segment_id: str, attrs: Optional[Dict[str, Any]] = None) -> str:
    attrs = attrs or {}
    text_group = attrs.get("text_group")
    if text_group:
        return str(text_group)
    src = str(source_segment_id)
    if "_S" in src:
        return src.split("_S", 1)[0]
    return src.split("_", 1)[0]


def load_source_to_usatges_edges(gexf_path: Path) -> List[Dict[str, Any]]:
    """
    Real format in borrowing_graph.gexf:
      - source nodes: node_type='source'
      - usatge nodes: node_type='usatge'
      - edges: source_segment -> usatge_segment
      - source nodes also store text_group (source corpus)
    """
    G = nx.read_gexf(gexf_path)
    edges: List[Dict[str, Any]] = []

    for u, v, attrs in G.edges(data=True):
        u_attrs = dict(G.nodes[u])
        v_attrs = dict(G.nodes[v])
        u_type = str(u_attrs.get("node_type", "")).strip().lower()
        v_type = str(v_attrs.get("node_type", "")).strip().lower()

        if u_type == "source" and v_type == "usatge":
            edges.append(
                {
                    "source_segment_id": str(u),
                    "source_group": _derive_source_group(u, u_attrs),
                    "usatge_id": str(v),
                    "weight": float(attrs.get("weight", 1.0)),
                    "alignment": attrs.get("alignment", ""),
                }
            )

    if not edges:
        raise RuntimeError(f"No source->Usatges edges found in {gexf_path}")

    print(f"[gramoty] loaded {len(edges)} source->Usatges edges from {gexf_path}")
    return edges


def project_source_segments_to_charters(
    source_usatges: Sequence[Dict[str, Any]],
    usatges_charters: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Project old source_segment -> usatge edges through newly computed
    usatge -> charter edges.
    """
    by_usatge: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in usatges_charters:
        by_usatge[str(row["usatge_id"])].append(row)

    projected: List[Dict[str, Any]] = []
    for row in source_usatges:
        usatge_id = str(row["usatge_id"])
        for hit in by_usatge.get(usatge_id, []):
            projected.append(
                {
                    "left_id": str(row["source_segment_id"]),
                    "left_group": str(row["source_group"]),
                    "charter_id": str(hit["charter_id"]),
                    "weight": float(row["weight"]) * float(hit["score"]),
                    "edge_type": "source_projection",
                    "via_usatge_id": usatge_id,
                    "source_to_usatge_weight": float(row["weight"]),
                    "usatge_to_charter_score": float(hit["score"]),
                }
            )

    projected.sort(key=lambda row: (-float(row["weight"]), row["left_group"], row["left_id"], row["charter_id"]))
    return projected


def prepare_usatges_direct_rows(usatges_charters: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in usatges_charters:
        rows.append(
            {
                "left_id": str(row["usatge_id"]),
                "left_group": "Usatges",
                "charter_id": str(row["charter_id"]),
                "weight": float(row["score"]),
                "edge_type": "usatge_direct",
                "via_usatge_id": str(row["usatge_id"]),
            }
        )
    rows.sort(key=lambda row: (-float(row["weight"]), row["left_id"], row["charter_id"]))
    return rows


# ----------------------------- metadata -----------------------------


def parse_charter_metadata(seg_id: str) -> Dict[str, Any]:
    seg_id = str(seg_id)
    lower = seg_id.lower()

    volume = None
    if "gramoty911" in lower or "911" in lower:
        volume = "I"
    elif "gramoty12" in lower:
        volume = "II"

    year = None
    m = re.search(r"_Y(\d{3,4})", seg_id)
    if m:
        year = int(m.group(1))
    else:
        m = re.search(r"(?:_|^)(\d{3,4})(?:_|$)", seg_id)
        if m:
            year = int(m.group(1))

    doc_no = None
    m = re.search(r"_D(\d+)", seg_id)
    if m:
        doc_no = int(m.group(1))

    date_key = year if year is not None else 999999
    return {"volume": volume or "?", "year": year, "doc_no": doc_no, "date_key": date_key}


def enrich_graph_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row.update(parse_charter_metadata(str(row["charter_id"])))
        out.append(new_row)
    return out


# ----------------------------- I/O -----------------------------


def save_csv(rows: Sequence[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------- main pipeline -----------------------------


def run(config_module_name: str = "config_gramoty"):
    cfg_mod = _load_module(config_module_name)
    cfg = _discover_config(cfg_mod)
    out_dir = cfg["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gramoty] loading source->Usatges graph: {cfg['SOURCE_GRAPH_GEXF']}")
    source_to_usatges = load_source_to_usatges_edges(cfg["SOURCE_GRAPH_GEXF"])
    print(f"[gramoty] loaded {len(source_to_usatges)} source->Usatges edges")

    print(f"[gramoty] loading Usatges: {cfg['USATGES_PATH']}")
    usatges_text = load_text(cfg["USATGES_PATH"])
    usatges_segments = segment_usatges(usatges_text, max_segment_words=cfg["MAX_SEGMENT_WORDS"])
    print(f"[gramoty] Usatges segments: {len(usatges_segments)}")

    charter_segments: List[Any] = []
    for source_name, path in cfg["GRAMOTY"].items():
        print(f"[gramoty] loading {source_name}: {path}")
        raw = load_text(path)
        segs = segment_charter_text(raw, source_name, max_segment_words=cfg["MAX_SEGMENT_WORDS"])
        print(f"[gramoty] {source_name}: {len(segs)} segments")
        charter_segments.extend(segs)

    if not charter_segments:
        raise RuntimeError("No charter segments were produced")

    print("[gramoty] computing Usatges -> charters similarities")
    uc_rows = compute_usatges_to_charters(
        usatges_segments=usatges_segments,
        charter_segments=charter_segments,
        cosine_threshold=cfg["COSINE_THRESHOLD"],
        top_k=cfg["TOP_K"],
        min_words=cfg["MIN_WORDS"],
    )
    print(f"[gramoty] retained {len(uc_rows)} Usatges->charter links")

    print("[gramoty] projecting source segments -> charters")
    sc_rows = project_source_segments_to_charters(source_to_usatges, uc_rows)
    print(f"[gramoty] projected {len(sc_rows)} source-segment->charter links")

    direct_rows = prepare_usatges_direct_rows(uc_rows)
    print(f"[gramoty] prepared {len(direct_rows)} direct Usatges->charter links")

    graph_rows = enrich_graph_rows(sc_rows + direct_rows)

    # Optional trimming for legibility, preserving direct links.
    graph_top_n = cfg.get("GRAPH_TOP_N", 0)
    if graph_top_n and len(graph_rows) > graph_top_n:
        graph_rows = sorted(graph_rows, key=lambda row: (-float(row["weight"]), row["left_group"], row["charter_id"]))[:graph_top_n]
        print(f"[gramoty] trimmed graph rows to top {graph_top_n}")

    save_csv(uc_rows, out_dir / "usatges_to_gramoty.csv")
    save_csv(sc_rows, out_dir / "sources_to_gramoty.csv")
    save_csv(graph_rows, out_dir / "graph_rows_gramoty.csv")

    graph, graph_paths = build_gramoty_graph(
        graph_rows,
        out_dir=out_dir,
        graph_name="gramoty_graph",
        source_names_ru=cfg.get("SOURCE_NAMES_RU") or cfg.get("SOURCE_NAMES_RU_SHORT") or {},
    )

    print(f"[gramoty] graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    print(f"[gramoty] wrote: {graph_paths['gexf']}")
    print(f"[gramoty] wrote: {graph_paths['png']}")

    return {
        "config": cfg,
        "usatges_to_charters": uc_rows,
        "source_segments_to_charters": sc_rows,
        "graph_rows": graph_rows,
        "graph": graph,
        "paths": graph_paths,
    }


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_gramoty"
    run(config_name)
