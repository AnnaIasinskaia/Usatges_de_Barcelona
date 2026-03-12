#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_gramoty.py

Direct pipeline for the charters study.

Target graph shape
------------------
Left column:
    - source groups from config SOURCES
    - Usatges as one more source group
Right column:
    - charter documents from both charter volumes, sorted by date

Edges:
    - direct source-segment -> charter matches are computed first
    - graph rows are aggregated to source-group -> charter
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    output_dir = Path(_first_attr(cfg_module, "OUTPUT_DIR", default="output_charters"))

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

    source_configs = _first_attr(cfg_module, "SOURCE_CONFIGS", default={}) or {}
    source_names_ru = _first_attr(cfg_module, "SOURCE_NAMES_RU", default={}) or {}
    source_names_ru_short = _first_attr(cfg_module, "SOURCE_NAMES_RU_SHORT", default={}) or {}

    # Defaults tuned for charters: lower threshold, more candidates, allow shorter legal formulas.
    cosine_threshold = float(
        _first_attr(
            cfg_module,
            "GRAMOTY_COSINE_THRESHOLD",
            "CHARTER_COSINE_THRESHOLD",
            "TFIDF_COSINE_THRESHOLD",
            "COSINE_THRESHOLD",
            default=0.045,
        )
    )
    top_k = int(_first_attr(cfg_module, "GRAMOTY_TOP_K", "TOP_K", default=15))
    min_words = int(_first_attr(cfg_module, "GRAMOTY_MIN_WORDS", "MIN_SEGMENT_WORDS", default=6))
    max_segment_words = int(_first_attr(cfg_module, "MAX_SEGMENT_WORDS", default=120))
    graph_top_n = int(_first_attr(cfg_module, "GRAPH_TOP_N", default=160))

    return {
        "DATA_DIR": data_dir,
        "OUTPUT_DIR": output_dir,
        "SOURCES": {k: _resolve_path(v, data_dir) for k, v in sources.items()},
        "GRAMOTY": {k: _resolve_path(v, data_dir) for k, v in charters.items()},
        "SOURCE_CONFIGS": source_configs,
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


def segment_source_text(text: str, source_name: str, cfg: Optional[Dict[str, Any]] = None, max_segment_words: int = 120):
    if source_name == "Usatges":
        return segment_usatges(text, max_segment_words=max_segment_words)

    mod = importlib.import_module("source_segmenters")
    if not hasattr(mod, "segment_source"):
        raise RuntimeError("Could not find segment_source in source_segmenters.py")
    return _call_with_supported_kwargs(mod.segment_source, text, source_name, cfg=cfg or {"max_segment_words": max_segment_words})


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


def compute_source_groups_to_charters(
    source_segments_by_group: Dict[str, Sequence[Any]],
    charter_segments: Sequence[Any],
    cosine_threshold: float = 0.045,
    top_k: int = 15,
    min_words: int = 6,
):
    charter_segments = _normalize_segment_list(charter_segments)
    charter_segments = [(sid, txt) for sid, txt in charter_segments if _segment_word_count(txt) >= min_words]
    prep_charters = maybe_preprocess_segments(charter_segments)

    detail_rows: List[Dict[str, Any]] = []

    for group_name, raw_source_segments in source_segments_by_group.items():
        source_segments = _normalize_segment_list(raw_source_segments)
        source_segments = [(sid, txt) for sid, txt in source_segments if _segment_word_count(txt) >= min_words]
        if not source_segments or not charter_segments:
            continue

        prep_sources = maybe_preprocess_segments(source_segments)
        all_texts = [txt for _, txt in prep_sources] + [txt for _, txt in prep_charters]
        vectorizer = build_tfidf(all_texts)

        s_matrix = vectorizer.transform([txt for _, txt in prep_sources])
        c_matrix = vectorizer.transform([txt for _, txt in prep_charters])
        sim = cosine_similarity(s_matrix, c_matrix)

        for i, (left_id, left_raw) in enumerate(source_segments):
            order = sim[i].argsort()[::-1]
            kept = 0
            for j in order:
                score = float(sim[i, j])
                if score < cosine_threshold:
                    break
                charter_id, charter_raw = charter_segments[j]
                detail_rows.append(
                    {
                        "left_id": left_id,
                        "left_group": group_name,
                        "charter_id": charter_id,
                        "score": score,
                        "left_text": left_raw,
                        "charter_text": charter_raw,
                        "edge_type": "usatge_direct" if group_name == "Usatges" else "source_direct",
                    }
                )
                kept += 1
                if kept >= top_k:
                    break

    detail_rows.sort(key=lambda row: (-float(row["score"]), str(row["left_group"]), str(row["left_id"]), str(row["charter_id"])))
    return detail_rows


def aggregate_detail_rows(detail_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, Dict[str, Any]] = {}

    for row in detail_rows:
        key = (str(row["left_group"]), str(row["charter_id"]))
        score = float(row["score"])
        current = grouped.get(key)

        if current is None:
            grouped[key] = {
                "left_id": str(row["left_group"]),
                "left_group": str(row["left_group"]),
                "charter_id": str(row["charter_id"]),
                "weight": score,
                "edge_type": "usatge_direct" if str(row["left_group"]) == "Usatges" else "source_direct",
                "best_left_id": str(row["left_id"]),
                "best_score": score,
                "hit_count": 1,
            }
        else:
            current["hit_count"] += 1
            if score > float(current["best_score"]):
                current["weight"] = score
                current["best_score"] = score
                current["best_left_id"] = str(row["left_id"])

    rows = list(grouped.values())
    rows.sort(key=lambda row: (-float(row["weight"]), str(row["left_group"]), str(row["charter_id"])))
    return rows


# ----------------------------- metadata -----------------------------


def parse_charter_metadata(seg_id: str) -> Dict[str, Any]:
    seg_id = str(seg_id)
    lower = seg_id.lower()

    volume = None
    if "gramoty911" in lower or lower.startswith("gramoty911"):
        volume = "I"
    elif "gramoty12" in lower or lower.startswith("gramoty12"):
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

    date_key = (year if year is not None else 999999, doc_no if doc_no is not None else 999999)
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

    source_segments_by_group: Dict[str, List[Any]] = {}
    for source_name, path in cfg["SOURCES"].items():
        print(f"[gramoty] loading source {source_name}: {path}")
        raw = load_text(path)
        source_cfg = (cfg.get("SOURCE_CONFIGS") or {}).get(source_name, {"max_segment_words": cfg["MAX_SEGMENT_WORDS"]})
        if source_name == "Usatges":
            segs = segment_usatges(raw, max_segment_words=source_cfg.get("max_segment_words", cfg["MAX_SEGMENT_WORDS"]))
        else:
            segs = segment_source_text(raw, source_name, cfg=source_cfg, max_segment_words=source_cfg.get("max_segment_words", cfg["MAX_SEGMENT_WORDS"]))
        print(f"[gramoty] {source_name}: {len(segs)} segments")
        source_segments_by_group[source_name] = list(segs)

    charter_segments: List[Any] = []
    for source_name, path in cfg["GRAMOTY"].items():
        print(f"[gramoty] loading {source_name}: {path}")
        raw = load_text(path)
        segs = segment_charter_text(raw, source_name, max_segment_words=cfg["MAX_SEGMENT_WORDS"])
        print(f"[gramoty] {source_name}: {len(segs)} segments")
        charter_segments.extend(segs)

    if not charter_segments:
        raise RuntimeError("No charter segments were produced")

    print("[gramoty] computing direct source/Usatges -> charters similarities")
    detail_rows = compute_source_groups_to_charters(
        source_segments_by_group=source_segments_by_group,
        charter_segments=charter_segments,
        cosine_threshold=cfg["COSINE_THRESHOLD"],
        top_k=cfg["TOP_K"],
        min_words=cfg["MIN_WORDS"],
    )
    print(f"[gramoty] retained {len(detail_rows)} direct segment->charter links")

    graph_rows = aggregate_detail_rows(detail_rows)
    graph_rows = enrich_graph_rows(graph_rows)
    print(f"[gramoty] aggregated to {len(graph_rows)} left-group->charter links")

    graph_top_n = cfg.get("GRAPH_TOP_N", 0)
    if graph_top_n and len(graph_rows) > graph_top_n:
        graph_rows = sorted(graph_rows, key=lambda row: (-float(row["weight"]), row["left_group"], row["charter_id"]))[:graph_top_n]
        print(f"[gramoty] trimmed graph rows to top {graph_top_n}")

    sources_rows = [row for row in detail_rows if row["left_group"] != "Usatges"]
    usatges_rows = [row for row in detail_rows if row["left_group"] == "Usatges"]

    save_csv(detail_rows, out_dir / "all_left_to_gramoty.csv")
    save_csv(sources_rows, out_dir / "sources_to_gramoty.csv")
    save_csv(usatges_rows, out_dir / "usatges_to_gramoty.csv")
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
        "detail_rows": detail_rows,
        "graph_rows": graph_rows,
        "graph": graph,
        "paths": graph_paths,
    }


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_gramoty"
    run(config_name)
