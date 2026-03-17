#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_gramoty.py

7-step pipeline for the charters study, aligned with pipeline.py while
preserving charter-specific inputs/outputs and graph layout.

Target graph shape
------------------
Left column:
    - source groups from config SOURCES
    - Usatges as one more source group
Right column:
    - charter documents from both charter volumes, sorted by date

Steps
-----
1. Load and segment texts
2. Preprocess and lemmatize
3. Build TF-IDF matrix
4. Find TF-IDF candidate pairs
5. Compute BorrowScore (cos + tesserae + soft cosine)
6. Run Smith-Waterman alignment
7. Aggregate to group -> charter and build graph

Checkpoints
-----------
A checkpoint is written after every step, so the pipeline can be resumed from
an arbitrary step via --resume-step.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import docx  # python-docx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("python-docx is required for pipeline_gramoty.py") from exc

from alignment import smith_waterman
from features import (
    build_tfidf_matrix,
    compute_idf,
    cosine_similarity_matrix,
    find_candidate_pairs,
    soft_cosine_similarity,
    tesserae_score,
)
from graph_builder_gramoty import build_gramoty_graph
from preprocessing import LatinLemmatizer, preprocess_segment

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

    tfidf_cosine_threshold = float(
        _first_attr(
            cfg_module,
            "GRAMOTY_TFIDF_COSINE_THRESHOLD",
            "TFIDF_COSINE_THRESHOLD",
            "GRAMOTY_COSINE_THRESHOLD",
            "CHARTER_COSINE_THRESHOLD",
            "COSINE_THRESHOLD",
            default=0.08,
        )
    )

    top_k = int(_first_attr(cfg_module, "GRAMOTY_TOP_K", "TOP_K", default=5))
    min_words = int(_first_attr(cfg_module, "GRAMOTY_MIN_WORDS", "MIN_SEGMENT_WORDS", default=12))
    max_segment_words = int(_first_attr(cfg_module, "MAX_SEGMENT_WORDS", default=200))
    graph_top_n = int(_first_attr(cfg_module, "GRAPH_TOP_N", default=160))

    ngram_range = tuple(_first_attr(cfg_module, "NGRAM_RANGE", default=(1, 3)))
    max_df = float(_first_attr(cfg_module, "MAX_DF", default=0.50))
    min_df = int(_first_attr(cfg_module, "MIN_DF", default=2))

    alpha = float(_first_attr(cfg_module, "ALPHA", default=0.30))
    beta = float(_first_attr(cfg_module, "BETA", default=0.40))
    gamma = float(_first_attr(cfg_module, "GAMMA", default=0.30))
    final_threshold = float(_first_attr(cfg_module, "GRAMOTY_FINAL_THRESHOLD", "FINAL_THRESHOLD", default=0.12))
    min_hits = int(_first_attr(cfg_module, "GRAMOTY_MIN_HITS", default=2))

    use_collatinus = bool(_first_attr(cfg_module, "USE_COLLATINUS", default=False))
    min_lemma_length = int(_first_attr(cfg_module, "MIN_LEMMA_LENGTH", default=3))

    soft_cosine_max_terms = int(_first_attr(cfg_module, "SOFT_COSINE_MAX_TERMS", default=500))
    sw_match = int(_first_attr(cfg_module, "SW_MATCH", default=2))
    sw_mismatch = int(_first_attr(cfg_module, "SW_MISMATCH", default=-1))
    sw_gap = int(_first_attr(cfg_module, "SW_GAP", default=-1))
    sw_lev_threshold = int(_first_attr(cfg_module, "SW_LEVENSHTEIN_BONUS_THRESHOLD", default=2))
    sw_max_seq_len = int(_first_attr(cfg_module, "SW_MAX_SEQ_LEN", default=300))
    sw_min_score = float(_first_attr(cfg_module, "GRAMOTY_SW_MIN_SCORE", default=0.0))

    source_not_before = dict(_first_attr(cfg_module, "SOURCE_NOT_BEFORE", default={}) or {})

    checkpoint_dir = Path(_first_attr(cfg_module, "CHECKPOINT_DIR", default=output_dir / "checkpoints"))
    resume_step = int(_first_attr(cfg_module, "RESUME_STEP", default=1))

    return {
        "DATA_DIR": data_dir,
        "OUTPUT_DIR": output_dir,
        "SOURCES": {k: _resolve_path(v, data_dir) for k, v in sources.items()},
        "GRAMOTY": {k: _resolve_path(v, data_dir) for k, v in charters.items()},
        "SOURCE_CONFIGS": source_configs,
        "TFIDF_COSINE_THRESHOLD": tfidf_cosine_threshold,
        "TOP_K": top_k,
        "MIN_WORDS": min_words,
        "MAX_SEGMENT_WORDS": max_segment_words,
        "GRAPH_TOP_N": graph_top_n,
        "NGRAM_RANGE": ngram_range,
        "MAX_DF": max_df,
        "MIN_DF": min_df,
        "ALPHA": alpha,
        "BETA": beta,
        "GAMMA": gamma,
        "FINAL_THRESHOLD": final_threshold,
        "MIN_HITS": min_hits,
        "USE_COLLATINUS": use_collatinus,
        "MIN_LEMMA_LENGTH": min_lemma_length,
        "SOFT_COSINE_MAX_TERMS": soft_cosine_max_terms,
        "SW_MATCH": sw_match,
        "SW_MISMATCH": sw_mismatch,
        "SW_GAP": sw_gap,
        "SW_LEVENSHTEIN_BONUS_THRESHOLD": sw_lev_threshold,
        "SW_MAX_SEQ_LEN": sw_max_seq_len,
        "GRAMOTY_SW_MIN_SCORE": sw_min_score,
        "SOURCE_NOT_BEFORE": source_not_before,
        "CHECKPOINT_DIR": checkpoint_dir,
        "RESUME_STEP": resume_step,
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
    mod = importlib.import_module("segmenters.seg_usatges")
    if hasattr(mod, "segment_usatges"):
        return _call_with_supported_kwargs(mod.segment_usatges, text, max_segment_words=max_segment_words)
    for name in dir(mod):
        if name.startswith("segment_") and callable(getattr(mod, name)):
            return _call_with_supported_kwargs(getattr(mod, name), text, max_segment_words=max_segment_words)
    raise RuntimeError("Could not find segment_usatges in segmenters.seg_usatges")


def segment_source_text(text: str, source_name: str, cfg: Optional[Dict[str, Any]] = None, max_segment_words: int = 120):
    if source_name == "Usatges":
        return segment_usatges(text, max_segment_words=max_segment_words)

    mod = importlib.import_module("source_segmenters")
    if not hasattr(mod, "segment_source"):
        raise RuntimeError("Could not find segment_source in source_segmenters.py")
    return _call_with_supported_kwargs(mod.segment_source, text, source_name, cfg=cfg or {"max_segment_words": max_segment_words})


def segment_charter_text(text: str, source_name: str, max_segment_words: int = 200):
    mod = importlib.import_module("segmenters.seg_gramoty_stable_merged")
    for candidate in ("seg_gramoty_stable_merged", "segment_gramoty", f"segment_{source_name.lower()}"):
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
    raise RuntimeError("Could not find a charter segmenter in seg_gramoty_stable_merged.py")


# ----------------------------- helpers -----------------------------


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


def _safe_preprocess(seg_id: str, text: str, lemmatizer: Any, min_length: int) -> List[str]:
    try:
        return preprocess_segment(text, lemmatizer, min_length=min_length)
    except Exception as exc:
        print(f"[gramoty] warning: preprocessing failed for {seg_id}: {exc}")
        return []


def _ckpt_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"step{step}.pkl"


def save_checkpoint(step: int, obj: Any, ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(ckpt_dir, step)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_checkpoint(step: int, ckpt_dir: Path) -> Any:
    path = _ckpt_path(ckpt_dir, step)
    with open(path, "rb") as f:
        return pickle.load(f)


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
    m = re.search(r"_doc(\d+)", seg_id, flags=re.IGNORECASE)
    if m:
        doc_no = int(m.group(1))
    else:
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


# ----------------------------- aggregation -----------------------------


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


# ----------------------------- I/O -----------------------------


def save_csv(rows: Sequence[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------- main pipeline -----------------------------


def _load_and_segment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    source_segments_by_group: Dict[str, List[Any]] = {}
    for source_name, path in cfg["SOURCES"].items():
        print(f"[gramoty] Step 1: loading source {source_name}: {path}")
        raw = load_text(path)
        source_cfg = (cfg.get("SOURCE_CONFIGS") or {}).get(source_name, {"max_segment_words": cfg["MAX_SEGMENT_WORDS"]})
        if source_name == "Usatges":
            segs = segment_usatges(raw, max_segment_words=source_cfg.get("max_segment_words", cfg["MAX_SEGMENT_WORDS"]))
        else:
            segs = segment_source_text(
                raw,
                source_name,
                cfg=source_cfg,
                max_segment_words=source_cfg.get("max_segment_words", cfg["MAX_SEGMENT_WORDS"]),
            )
        print(f"[gramoty]   {source_name}: {len(segs)} segments")
        source_segments_by_group[source_name] = list(segs)

    charter_segments: List[Any] = []
    for corpus_name, path in cfg["GRAMOTY"].items():
        print(f"[gramoty] Step 1: loading {corpus_name}: {path}")
        raw = load_text(path)
        segs = segment_charter_text(raw, corpus_name, max_segment_words=cfg["MAX_SEGMENT_WORDS"])
        print(f"[gramoty]   {corpus_name}: {len(segs)} segments")
        charter_segments.extend(segs)

    if not charter_segments:
        raise RuntimeError("No charter segments were produced")

    return {
        "source_segments_by_group": source_segments_by_group,
        "charter_segments": charter_segments,
    }


def _preprocess_segments(step1: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    lemmatizer = LatinLemmatizer(use_collatinus=cfg["USE_COLLATINUS"])

    source_raw_by_id: Dict[str, str] = {}
    source_group_by_id: Dict[str, str] = {}
    source_lemmas: Dict[str, List[str]] = {}

    for group_name, raw_segments in step1["source_segments_by_group"].items():
        for sid, text in _normalize_segment_list(raw_segments):
            if _segment_word_count(text) < cfg["MIN_WORDS"]:
                continue
            source_raw_by_id[sid] = text
            source_group_by_id[sid] = group_name
            lem = _safe_preprocess(sid, text, lemmatizer, cfg["MIN_LEMMA_LENGTH"])
            if lem:
                source_lemmas[sid] = lem

    charter_raw_by_id: Dict[str, str] = {}
    charter_lemmas: Dict[str, List[str]] = {}
    for cid, text in _normalize_segment_list(step1["charter_segments"]):
        if _segment_word_count(text) < cfg["MIN_WORDS"]:
            continue
        charter_raw_by_id[cid] = text
        lem = _safe_preprocess(cid, text, lemmatizer, cfg["MIN_LEMMA_LENGTH"])
        if lem:
            charter_lemmas[cid] = lem

    if not source_lemmas or not charter_lemmas:
        raise RuntimeError("No valid segments after preprocessing")

    return {
        "source_raw_by_id": source_raw_by_id,
        "source_group_by_id": source_group_by_id,
        "source_lemmas": source_lemmas,
        "charter_raw_by_id": charter_raw_by_id,
        "charter_lemmas": charter_lemmas,
    }


def _build_tfidf(step2: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    left_ids = list(step2["source_lemmas"].keys())
    charter_ids = list(step2["charter_lemmas"].keys())
    corpus = [step2["source_lemmas"][sid] for sid in left_ids] + [step2["charter_lemmas"][cid] for cid in charter_ids]

    tfidf_matrix, vocab, term2idx = build_tfidf_matrix(
        corpus,
        ngram_range=cfg["NGRAM_RANGE"],
        max_df=cfg["MAX_DF"],
        min_df=cfg["MIN_DF"],
    )

    n_left = len(left_ids)
    tfidf_left = tfidf_matrix[:n_left]
    tfidf_charters = tfidf_matrix[n_left:]

    return {
        "left_ids": left_ids,
        "charter_ids": charter_ids,
        "corpus": corpus,
        "tfidf_left": tfidf_left,
        "tfidf_charters": tfidf_charters,
        "vocab": vocab,
        "term2idx": term2idx,
    }


def _find_candidates(step3: Dict[str, Any], step2: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    sim_matrix = cosine_similarity_matrix(step3["tfidf_left"], step3["tfidf_charters"])
    raw_candidates = find_candidate_pairs(
        sim_matrix,
        cfg["TFIDF_COSINE_THRESHOLD"],
        step3["left_ids"],
        step3["charter_ids"],
    )

    grouped: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    for left_id, charter_id, cos_sim in raw_candidates:
        grouped[left_id].append((left_id, charter_id, float(cos_sim)))

    filtered: List[Tuple[str, str, float]] = []
    for left_id, triples in grouped.items():
        triples = sorted(triples, key=lambda t: t[2], reverse=True)[: cfg["TOP_K"]]
        group_name = step2["source_group_by_id"].get(left_id)
        not_before = cfg["SOURCE_NOT_BEFORE"].get(group_name)
        for _, charter_id, cos_sim in triples:
            if not_before is not None:
                year = parse_charter_metadata(charter_id).get("year")
                if year is not None and int(year) < int(not_before):
                    continue
            filtered.append((left_id, charter_id, cos_sim))

    return {"candidates": filtered}


def _score_candidates(step4: Dict[str, Any], step2: Dict[str, Any], step3: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    idf = compute_idf(step3["corpus"])
    scored_pairs: List[Dict[str, Any]] = []

    for left_id, charter_id, cos_sim in step4["candidates"]:
        left_lem = step2["source_lemmas"].get(left_id, [])
        charter_lem = step2["charter_lemmas"].get(charter_id, [])
        if not left_lem or not charter_lem:
            continue

        tess = float(tesserae_score(left_lem, charter_lem, idf))
        if cos_sim + tess * cfg["BETA"] > cfg["FINAL_THRESHOLD"] * 0.5:
            soft_cos = float(soft_cosine_similarity(left_lem, charter_lem, max_terms=cfg["SOFT_COSINE_MAX_TERMS"]))
        else:
            soft_cos = 0.0

        borrow_score = float(cfg["ALPHA"] * cos_sim + cfg["BETA"] * tess + cfg["GAMMA"] * soft_cos)
        if borrow_score < cfg["FINAL_THRESHOLD"]:
            continue

        scored_pairs.append(
            {
                "left_id": left_id,
                "left_group": step2["source_group_by_id"].get(left_id, "unknown"),
                "charter_id": charter_id,
                "score": borrow_score,
                "cos_sim": float(cos_sim),
                "tesserae": tess,
                "soft_cos": soft_cos,
                "left_text": step2["source_raw_by_id"].get(left_id, ""),
                "charter_text": step2["charter_raw_by_id"].get(charter_id, ""),
                "edge_type": "usatge_direct" if step2["source_group_by_id"].get(left_id) == "Usatges" else "source_direct",
            }
        )

    scored_pairs.sort(key=lambda row: (-float(row["score"]), row["left_group"], row["left_id"], row["charter_id"]))
    return {"idf": idf, "scored_pairs": scored_pairs}


def _align_pairs(step5: Dict[str, Any], step2: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    detail_rows: List[Dict[str, Any]] = []

    for row in step5["scored_pairs"]:
        left_id = row["left_id"]
        charter_id = row["charter_id"]
        left_lem = step2["source_lemmas"].get(left_id, [])
        charter_lem = step2["charter_lemmas"].get(charter_id, [])
        if not left_lem or not charter_lem:
            continue

        try:
            al_a, al_b, sw_score = smith_waterman(
                left_lem,
                charter_lem,
                match_score=cfg["SW_MATCH"],
                mismatch_score=cfg["SW_MISMATCH"],
                gap_penalty=cfg["SW_GAP"],
                lev_bonus_threshold=cfg["SW_LEVENSHTEIN_BONUS_THRESHOLD"],
                max_seq_len=cfg["SW_MAX_SEQ_LEN"],
            )
        except Exception as exc:
            print(f"[gramoty] warning: Smith-Waterman failed for {left_id} <-> {charter_id}: {exc}")
            al_a, al_b, sw_score = [], [], 0.0

        if float(sw_score) < cfg["GRAMOTY_SW_MIN_SCORE"]:
            continue

        new_row = dict(row)
        new_row.update(
            {
                "sw_score": float(sw_score),
                "alignment_a": al_a,
                "alignment_b": al_b,
            }
        )
        detail_rows.append(new_row)

    detail_rows.sort(key=lambda row: (-float(row["score"]), row["left_group"], row["left_id"], row["charter_id"]))
    return {"detail_rows": detail_rows}


def _build_graph_rows(step6: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    graph_rows = aggregate_detail_rows(step6["detail_rows"])
    graph_rows = enrich_graph_rows(graph_rows)
    if cfg["MIN_HITS"] > 1:
        graph_rows = [row for row in graph_rows if int(row.get("hit_count", 1)) >= cfg["MIN_HITS"]]
    if cfg["GRAPH_TOP_N"] and len(graph_rows) > cfg["GRAPH_TOP_N"]:
        graph_rows = sorted(graph_rows, key=lambda row: (-float(row["weight"]), row["left_group"], row["charter_id"]))[: cfg["GRAPH_TOP_N"]]
    return {"graph_rows": graph_rows}


def run(config_module_name: str = "config_gramoty"):
    cfg_mod = _load_module(config_module_name)
    cfg = _discover_config(cfg_mod)
    out_dir = cfg["OUTPUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["CHECKPOINT_DIR"])
    resume_step = int(cfg["RESUME_STEP"])

    # Step 1
    if resume_step > 1 and _ckpt_path(ckpt_dir, 1).exists():
        print(f"[gramoty] loading checkpoint step 1 from {_ckpt_path(ckpt_dir, 1)}")
        step1 = load_checkpoint(1, ckpt_dir)
    else:
        step1 = _load_and_segment(cfg)
        save_checkpoint(1, step1, ckpt_dir)

    # Step 2
    if resume_step > 2 and _ckpt_path(ckpt_dir, 2).exists():
        print(f"[gramoty] loading checkpoint step 2 from {_ckpt_path(ckpt_dir, 2)}")
        step2 = load_checkpoint(2, ckpt_dir)
    else:
        print("[gramoty] Step 2: preprocessing and lemmatization")
        step2 = _preprocess_segments(step1, cfg)
        save_checkpoint(2, step2, ckpt_dir)

    # Step 3
    if resume_step > 3 and _ckpt_path(ckpt_dir, 3).exists():
        print(f"[gramoty] loading checkpoint step 3 from {_ckpt_path(ckpt_dir, 3)}")
        step3 = load_checkpoint(3, ckpt_dir)
    else:
        print("[gramoty] Step 3: building TF-IDF matrix")
        step3 = _build_tfidf(step2, cfg)
        save_checkpoint(3, step3, ckpt_dir)

    # Step 4
    if resume_step > 4 and _ckpt_path(ckpt_dir, 4).exists():
        print(f"[gramoty] loading checkpoint step 4 from {_ckpt_path(ckpt_dir, 4)}")
        step4 = load_checkpoint(4, ckpt_dir)
    else:
        print("[gramoty] Step 4: candidate linking by TF-IDF cosine")
        step4 = _find_candidates(step3, step2, cfg)
        print(f"[gramoty]   retained {len(step4['candidates'])} candidates")
        save_checkpoint(4, step4, ckpt_dir)

    # Step 5
    if resume_step > 5 and _ckpt_path(ckpt_dir, 5).exists():
        print(f"[gramoty] loading checkpoint step 5 from {_ckpt_path(ckpt_dir, 5)}")
        step5 = load_checkpoint(5, ckpt_dir)
    else:
        print("[gramoty] Step 5: computing BorrowScore")
        step5 = _score_candidates(step4, step2, step3, cfg)
        print(f"[gramoty]   retained {len(step5['scored_pairs'])} scored pairs")
        save_checkpoint(5, step5, ckpt_dir)

    # Step 6
    if resume_step > 6 and _ckpt_path(ckpt_dir, 6).exists():
        print(f"[gramoty] loading checkpoint step 6 from {_ckpt_path(ckpt_dir, 6)}")
        step6 = load_checkpoint(6, ckpt_dir)
    else:
        print("[gramoty] Step 6: Smith-Waterman alignment")
        step6 = _align_pairs(step5, step2, cfg)
        print(f"[gramoty]   retained {len(step6['detail_rows'])} aligned pairs")
        save_checkpoint(6, step6, ckpt_dir)

    # Step 7
    if resume_step > 7 and _ckpt_path(ckpt_dir, 7).exists():
        print(f"[gramoty] loading checkpoint step 7 from {_ckpt_path(ckpt_dir, 7)}")
        step7 = load_checkpoint(7, ckpt_dir)
    else:
        print("[gramoty] Step 7: aggregating and building graph rows")
        step7 = _build_graph_rows(step6, cfg)
        print(f"[gramoty]   aggregated to {len(step7['graph_rows'])} group->charter links")
        save_checkpoint(7, step7, ckpt_dir)

    detail_rows = step6["detail_rows"]
    graph_rows = step7["graph_rows"]

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
    ap = argparse.ArgumentParser(description="7-step borrowings pipeline for charters (gramoty)")
    ap.add_argument("config", nargs="?", default="config_gramoty", help="Config module name")
    ap.add_argument("--resume-step", type=int, default=None, help="Resume from step N (1..7)")
    ap.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    args = ap.parse_args()

    cfg_mod = _load_module(args.config)
    if args.resume_step is not None:
        setattr(cfg_mod, "RESUME_STEP", int(args.resume_step))
    if args.checkpoint_dir is not None:
        setattr(cfg_mod, "CHECKPOINT_DIR", Path(args.checkpoint_dir))

    run(args.config)
