#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_unified_refactored.py

Новая архитектура unified pipeline:

1) TF-IDF используется только как retrieval-ranking engine
2) Никаких threshold / top_k_per_left
3) Один понятный параметр retrieval_budget
4) Для retrieval-кандидатов всегда считаются 4 метрики:
   - cos_sim
   - tesserae
   - soft_cos
   - sw_norm
5) Дальше:
   - Pareto filtering
   - rank aggregation
   - top_N graph edges

Старая BorrowScore-логика, soft-gate и SW post-filter убраны.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib
import json
import pickle
import re
import time
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

from preprocessing import LatinLemmatizer, preprocess_segment
from alignment import smith_waterman
from features import (
    build_tfidf_matrix,
    compute_idf,
    soft_cosine_similarity,
    tesserae_score,
)
from graph_rendering import generic_numeric_sort_key, render_bipartite_graph


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    corpus: str
    side: str
    parent_id: str


@dataclass
class CandidateMetrics:
    left_leaf_id: str
    right_leaf_id: str
    left_parent_id: str
    right_parent_id: str
    left_corpus: str
    right_corpus: str

    left_node: str
    right_node: str
    right_doc_no: Optional[int]

    cos_sim: float
    tesserae: float
    soft_cos: float
    sw_score_raw: float
    sw_norm: float

    alignment_a: List[str]
    alignment_b: List[str]

    left_text_snippet: str
    right_text_snippet: str

    pareto_layer: int = 0
    pareto_on_front: bool = False

    rank_cos: int = 0
    rank_tess: int = 0
    rank_soft: int = 0
    rank_sw: int = 0
    rank_sum: int = 0
    rank_final_position: int = 0

    rank_score: float = 0.0


class ProgressLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        dt = time.time() - self.t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)



def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def compute_checkpoint_fingerprint(
    *,
    experiment_id: str,
    left_corpora: Sequence[str],
    right_corpora: Sequence[str],
    resolved_mappings: Sequence[Tuple[List[str], List[str]]],
    chunk_cfg: Dict[str, Any],
    model: Dict[str, Any],
    agg_cfg: Dict[str, Any],
    pareto_cfg: Dict[str, Any],
    selection_cfg: Dict[str, Any],
    viz_cfg: Dict[str, Any],
    corpora: Dict[str, Dict[str, Any]],
    used_corpora: Sequence[str],
) -> str:
    source_meta: Dict[str, Dict[str, Any]] = {}
    for cid in used_corpora:
        spec = corpora.get(cid, {})
        path = Path(spec["path"])
        stat = path.stat()
        source_meta[cid] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    payload = {
        "experiment_id": experiment_id,
        "left_corpora": list(left_corpora),
        "right_corpora": list(right_corpora),
        "resolved_mappings": list(resolved_mappings),
        "chunk_cfg": _json_safe(chunk_cfg),
        "model": _json_safe(model),
        "aggregation": _json_safe(agg_cfg),
        "pareto": _json_safe(pareto_cfg),
        "selection": _json_safe(selection_cfg),
        "viz": _json_safe(viz_cfg),
        "source_meta": source_meta,
        "pipeline_version": "smart-checkpoints-v2-step5-partial",
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(blob).hexdigest()


class CheckpointManager:
    def __init__(
        self,
        root_dir: Path,
        fingerprint: str,
        logger: ProgressLogger,
        enabled: bool = True,
        force_from_step: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.logger = logger
        self.enabled = enabled
        self.force_from_step = force_from_step
        self.meta_path = self.root_dir / "checkpoint_meta.json"

        if self.enabled:
            self._write_meta()

    def _write_meta(self) -> None:
        payload = {
            "fingerprint": self.fingerprint,
            "updated_at_epoch": time.time(),
        }
        self.meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _step_path(self, step_no: int, step_key: str) -> Path:
        return self.root_dir / f"step_{step_no:02d}_{step_key}.pkl"

    def load(self, step_no: int, step_key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if self.force_from_step is not None and step_no >= int(self.force_from_step):
            return None

        path = self._step_path(step_no, step_key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            self.logger.log(
                f"  Checkpoint ignored for step {step_no} ({step_key}): "
                f"cannot read checkpoint ({type(e).__name__}: {e})"
            )
            return None

        if not isinstance(payload, dict):
            self.logger.log(f"  Checkpoint ignored for step {step_no} ({step_key}): invalid payload type")
            return None

        if payload.get("fingerprint") != self.fingerprint:
            self.logger.log(
                f"  Checkpoint ignored for step {step_no} ({step_key}): fingerprint mismatch"
            )
            return None

        self.logger.log(f"  Checkpoint hit: step {step_no} ({step_key})")
        return payload.get("data")

    def save(self, step_no: int, step_key: str, data: Any) -> None:
        if not self.enabled:
            return
        path = self._step_path(step_no, step_key)
        payload = {
            "fingerprint": self.fingerprint,
            "step_no": step_no,
            "step_key": step_key,
            "saved_at_epoch": time.time(),
            "data": data,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.log(f"  Checkpoint saved: step {step_no} ({step_key})")


class Step3CorpusCheckpointManager:
    def __init__(
        self,
        root_dir: Path,
        fingerprint: Dict[str, Any],
        logger: ProgressLogger,
        enabled: bool = True,
        force_from_step: Optional[int] = None,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.logger = logger
        self.enabled = enabled
        self.force_from_step = force_from_step

    def _path(self, side: str, corpus_id: str) -> Path:
        safe_corpus = re.sub(r"[^A-Za-z0-9_.-]+", "_", corpus_id)
        safe_side = re.sub(r"[^A-Za-z0-9_.-]+", "_", side)
        return self.root_dir / f"{safe_side}__{safe_corpus}.pkl"

    def load(self, side: str, corpus_id: str) -> Optional[Dict[str, List[str]]]:
        if not self.enabled:
            return None
        if self.force_from_step is not None and self.force_from_step <= 3:
            return None
        path = self._path(side, corpus_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            self.logger.log(
                f"  Step 3 corpus checkpoint ignored ({side}/{corpus_id}): cannot read ({type(e).__name__}: {e})"
            )
            return None
        if payload.get("fingerprint") != self.fingerprint:
            self.logger.log(f"  Step 3 corpus checkpoint ignored ({side}/{corpus_id}): fingerprint mismatch")
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            self.logger.log(f"  Step 3 corpus checkpoint ignored ({side}/{corpus_id}): bad payload")
            return None
        self.logger.log(f"  Step 3 corpus checkpoint hit: {side}/{corpus_id}")
        return data

    def save(self, side: str, corpus_id: str, data: Dict[str, List[str]]) -> None:
        if not self.enabled:
            return
        path = self._path(side, corpus_id)
        payload = {
            "fingerprint": self.fingerprint,
            "side": side,
            "corpus_id": corpus_id,
            "data": data,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        self.logger.log(f"  Step 3 corpus checkpoint saved: {side}/{corpus_id}, segments={len(data)}")

    def clear(self) -> None:
        if not self.root_dir.exists():
            return
        for path in self.root_dir.glob("*.pkl"):
            try:
                path.unlink()
            except Exception:
                pass


class Step5PartialCheckpointManager:
    def __init__(self, root_dir: Path, fingerprint: str, logger: ProgressLogger, enabled: bool = True, force_from_step: Optional[int] = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.logger = logger
        self.enabled = enabled
        self.force_from_step = force_from_step
        self.meta_path = self.root_dir / "progress.json"

    def _part_path(self, upto_index: int) -> Path:
        return self.root_dir / f"part_{upto_index:08d}.pkl"

    def load_progress(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if self.force_from_step is not None and 5 >= int(self.force_from_step):
            return None
        if not self.meta_path.exists():
            return None
        try:
            payload = json.loads(self.meta_path.read_text(encoding='utf-8'))
        except Exception as e:
            self.logger.log(f"  Step 5 partial checkpoint ignored: cannot read progress ({type(e).__name__}: {e})")
            return None
        if payload.get("fingerprint") != self.fingerprint:
            self.logger.log("  Step 5 partial checkpoint ignored: fingerprint mismatch")
            return None
        return payload

    def load_parts(self, upto_index: int) -> List[CandidateMetrics]:
        rows: List[CandidateMetrics] = []
        if not self.enabled or upto_index <= 0:
            return rows
        for path in sorted(self.root_dir.glob('part_*.pkl')):
            try:
                payload = pickle.loads(path.read_bytes())
            except Exception as e:
                self.logger.log(f"  Step 5 part ignored: {path.name} unreadable ({type(e).__name__}: {e})")
                continue
            if not isinstance(payload, dict) or payload.get('fingerprint') != self.fingerprint:
                continue
            part_upto = int(payload.get('upto_index', 0))
            if part_upto > upto_index:
                continue
            data = payload.get('metric_rows') or []
            rows.extend(data)
        return rows

    def save_progress(self, *, upto_index: int, part_metric_rows: List[CandidateMetrics], total_computed: int, skipped_empty_lemmas: int, total_candidates: int) -> None:
        if not self.enabled:
            return
        part_payload = {
            'fingerprint': self.fingerprint,
            'upto_index': int(upto_index),
            'metric_rows': list(part_metric_rows),
            'saved_at_epoch': time.time(),
        }
        with open(self._part_path(int(upto_index)), 'wb') as f:
            pickle.dump(part_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            'fingerprint': self.fingerprint,
            'next_index': int(upto_index) + 1,
            'last_completed_index': int(upto_index),
            'skipped_empty_lemmas': int(skipped_empty_lemmas),
            'total_candidates': int(total_candidates),
            'saved_at_epoch': time.time(),
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        self.logger.log(
            f"  Step 5 partial checkpoint saved: upto={upto_index}/{total_candidates}, "
            f"computed={total_computed}, batch_saved={len(part_metric_rows)}, skipped_empty={skipped_empty_lemmas}"
        )

    def clear(self) -> None:
        if not self.root_dir.exists():
            return
        for path in self.root_dir.glob('*'):
            try:
                path.unlink()
            except Exception:
                pass


def load_config_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def resolve_logging_config(
    config_module: Any,
    experiment_cfg: Dict[str, Any],
    progress_every_override: Optional[int] = None,
) -> Dict[str, Any]:
    defaults = dict(getattr(config_module, "LOGGING_DEFAULTS", {}) or {})
    local = dict(experiment_cfg.get("logging") or {})
    merged = dict(defaults)
    merged.update(local)
    if progress_every_override is not None:
        merged["scoring_progress_every"] = progress_every_override
    return merged


def resolve_group_tokens(items: Sequence[str], groups: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x.startswith("@"):
            out.extend(groups.get(x[1:], []))
        else:
            out.append(x)
    return out


def segment_corpus(corpus_id: str, corpus_spec: Dict[str, Any], logger: ProgressLogger) -> List[Tuple[str, str]]:
    source_file = Path(corpus_spec["path"])
    if not source_file.exists():
        raise FileNotFoundError(f"Corpus file not found for {corpus_id}: {source_file}")

    seg_mod = importlib.import_module("source_segmenters")
    raw = seg_mod.segment_source(source_file, corpus_id)

    if raw is None:
        return []

    segments: List[Tuple[str, str]] = []
    for item in raw:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise TypeError(
                f"Unsupported segment shape for {corpus_id}: expected tuple(str, str), "
                f"got {type(item)} => {repr(item)[:200]}"
            )
        seg_id, seg_text = item
        if not isinstance(seg_id, str) or not isinstance(seg_text, str):
            raise TypeError(
                f"Unsupported segment value types for {corpus_id}: "
                f"id={type(seg_id)}, text={type(seg_text)}"
            )
        segments.append((seg_id, seg_text))
    return segments


def _word_tokens(text: str) -> List[str]:
    return [w for w in text.split() if w]


def chunk_segments(base: Sequence[Segment], chunking_cfg: Dict[str, Any], logger: ProgressLogger, side: str) -> List[Segment]:
    if not chunking_cfg or not bool(chunking_cfg.get("enabled")):
        return list(base)

    mode = chunking_cfg.get("mode", "sliding_window_words")
    if mode != "sliding_window_words":
        raise ValueError(f"Unsupported chunking mode: {mode}")

    w_default = int(chunking_cfg.get("window_words", 180))
    o_default = int(chunking_cfg.get("overlap_words", 60))
    min_default = int(chunking_cfg.get("min_words", 20))
    per = dict(chunking_cfg.get("per_corpus") or {})

    out: List[Segment] = []

    for seg in base:
        ov = dict(per.get(seg.corpus) or {})
        if ov.get("enabled", True) is False:
            out.append(seg)
            continue

        window_words = int(ov.get("window_words", w_default))
        overlap_words = int(ov.get("overlap_words", o_default))
        min_words = int(ov.get("min_words", min_default))

        words = _word_tokens(seg.text)
        if len(words) < min_words or len(words) <= window_words:
            out.append(seg)
            continue

        step = max(1, window_words - max(0, overlap_words))
        idx = 0
        for start in range(0, len(words), step):
            end = min(len(words), start + window_words)
            if end - start < min_words:
                continue
            chunk_text = " ".join(words[start:end])
            leaf_id = f"{seg.parent_id}__w{idx}_W{start}-{end}"
            out.append(Segment(
                id=leaf_id,
                text=chunk_text,
                corpus=seg.corpus,
                side=seg.side,
                parent_id=seg.parent_id,
            ))
            idx += 1
            if end == len(words):
                break

    return out


_DOC_RE = re.compile(r"_doc(\d+)", re.IGNORECASE)


def extract_doc_no(seg: Segment) -> Optional[int]:
    m = _DOC_RE.search(seg.parent_id)
    if m:
        return int(m.group(1))
    return None


def lemmatize_segments_strict(
    segments: Sequence[Segment],
    lemmatizer: LatinLemmatizer,
    min_lemma_length: int,
    logger: ProgressLogger,
    label: str,
    corpus_id: str,
    progress_every: Optional[int] = None,
) -> Dict[str, List[str]]:
    lemmas_by_id: Dict[str, List[str]] = {}
    total = len(segments)
    for idx, seg in enumerate(segments, 1):
        try:
            lem = preprocess_segment(seg.text, lemmatizer, min_length=min_lemma_length)
        except Exception as e:
            logger.log(
                f"  Preprocessing fatal error ({label}) seg_id={seg.id} corpus={seg.corpus}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(
                f"Preprocessing failed for corpus={corpus_id} side={label} seg_id={seg.id}: {type(e).__name__}: {e}"
            ) from e
        lemmas_by_id[seg.id] = lem

        if progress_every and idx % max(1, int(progress_every)) == 0:
            logger.log(f"  Preprocessing progress ({label}/{corpus_id}): {idx}/{total}")

    return lemmas_by_id


def compute_sw_metrics(
    left_lem: List[str],
    right_lem: List[str],
    model: Dict[str, Any],
) -> Tuple[List[str], List[str], float, float]:
    al_a, al_b, sw = smith_waterman(
        left_lem,
        right_lem,
        match_score=int(model.get("sw_match", 2)),
        mismatch_score=int(model.get("sw_mismatch", -1)),
        gap_penalty=int(model.get("sw_gap", -1)),
        lev_bonus_threshold=int(model.get("sw_lev_bonus_threshold", 2)),
        max_seq_len=int(model.get("sw_max_seq_len", 300)),
    )
    sw_raw = float(sw)
    base = max(1, min(len(left_lem), len(right_lem)))
    sw_norm = sw_raw / float(base)
    return al_a, al_b, sw_raw, sw_norm


def compute_candidate_metrics(
    cos_sim: float,
    left_lem: List[str],
    right_lem: List[str],
    idf: Dict[str, float],
    model: Dict[str, Any],
) -> Tuple[float, float, float, float, List[str], List[str]]:
    tess = float(tesserae_score(left_lem, right_lem, idf))
    soft = float(
        soft_cosine_similarity(
            left_lem,
            right_lem,
            max_terms=int(model.get("soft_cosine_max_terms", 500)),
        )
    )
    al_a, al_b, sw_raw, sw_norm = compute_sw_metrics(left_lem, right_lem, model)
    return tess, soft, sw_raw, sw_norm, al_a, al_b


def build_allowed_right_corpora_by_left(
    left_corpora: Sequence[str],
    right_corpora: Sequence[str],
    mappings: Sequence[Tuple[List[str], List[str]]],
) -> Dict[str, set]:
    allowed: Dict[str, set] = {lc: set() for lc in left_corpora}
    for frm, to in mappings:
        to_set = set(to)
        for lc in frm:
            if lc in allowed:
                allowed[lc].update(to_set)
    for lc in left_corpora:
        if not allowed[lc]:
            allowed[lc].update(right_corpora)
    return allowed


def iter_sparse_row_scores(row_vec, right_matrix):
    sim_row = row_vec.dot(right_matrix.T)
    if hasattr(sim_row, "tocoo"):
        coo = sim_row.tocoo()
        for col, val in zip(coo.col, coo.data):
            yield int(col), float(val)
    else:
        dense = sim_row.ravel().tolist()
        for col, val in enumerate(dense):
            if val:
                yield col, float(val)


def select_retrieval_candidates_budgeted(
    tfidf_left,
    tfidf_right,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    left_seg_by_id: Dict[str, Segment],
    right_seg_by_id: Dict[str, Segment],
    allowed_right_corpora_by_left: Dict[str, set],
    retrieval_budget: int,
    logger: ProgressLogger,
    progress_every: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    if retrieval_budget <= 0:
        raise ValueError(f"retrieval_budget must be positive, got {retrieval_budget}")

    heap: List[Tuple[float, str, str]] = []
    total_left = len(left_ids)

    for idx, left_id in enumerate(left_ids, 1):
        left_seg = left_seg_by_id[left_id]
        allowed_right_corpora = allowed_right_corpora_by_left.get(left_seg.corpus, set())

        row_vec = tfidf_left[idx - 1]
        for right_idx, sim in iter_sparse_row_scores(row_vec, tfidf_right):
            if sim <= 0.0:
                continue
            right_id = right_ids[right_idx]
            right_seg = right_seg_by_id[right_id]
            if right_seg.corpus not in allowed_right_corpora:
                continue

            item = (sim, left_id, right_id)
            if len(heap) < retrieval_budget:
                heapq.heappush(heap, item)
            else:
                if sim > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if progress_every and (idx % max(1, int(progress_every)) == 0 or idx == total_left):
            logger.log(f"  Retrieval progress: {idx}/{total_left}, heap={len(heap)}")

    out = [(left_id, right_id, float(sim)) for sim, left_id, right_id in heap]
    out.sort(key=lambda x: (-float(x[2]), x[0], x[1]))
    return out


def dominates(a: CandidateMetrics, b: CandidateMetrics) -> bool:
    ge_all = (
        a.cos_sim >= b.cos_sim
        and a.tesserae >= b.tesserae
        and a.soft_cos >= b.soft_cos
        and a.sw_norm >= b.sw_norm
    )
    gt_any = (
        a.cos_sim > b.cos_sim
        or a.tesserae > b.tesserae
        or a.soft_cos > b.soft_cos
        or a.sw_norm > b.sw_norm
    )
    return ge_all and gt_any


def assign_pareto_layers(rows: List[CandidateMetrics]) -> None:
    remaining = list(range(len(rows)))
    layer = 1
    while remaining:
        front: List[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if dominates(rows[j], rows[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        front_set = set(front)
        for i in front:
            rows[i].pareto_layer = layer
            rows[i].pareto_on_front = (layer == 1)

        remaining = [i for i in remaining if i not in front_set]
        layer += 1


def rank_desc(values: Sequence[float]) -> List[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: (-float(x[1]), x[0]))
    ranks = [0] * len(values)
    for pos, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = pos
    return ranks


def rank_aggregate(rows: List[CandidateMetrics]) -> None:
    if not rows:
        return

    rank_cos = rank_desc([r.cos_sim for r in rows])
    rank_tess = rank_desc([r.tesserae for r in rows])
    rank_soft = rank_desc([r.soft_cos for r in rows])
    rank_sw = rank_desc([r.sw_norm for r in rows])

    for i, r in enumerate(rows):
        r.rank_cos = rank_cos[i]
        r.rank_tess = rank_tess[i]
        r.rank_soft = rank_soft[i]
        r.rank_sw = rank_sw[i]
        r.rank_sum = r.rank_cos + r.rank_tess + r.rank_soft + r.rank_sw

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i].rank_sum,
            -rows[i].tesserae,
            -rows[i].sw_norm,
            -rows[i].soft_cos,
            -rows[i].cos_sim,
            rows[i].left_leaf_id,
            rows[i].right_leaf_id,
        )
    )

    for pos, i in enumerate(order, 1):
        rows[i].rank_final_position = pos
        rows[i].rank_score = 1.0 / float(pos)


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
        rank_score = float(row["rank_score"])
        rank_final = int(row["rank_final_position"])

        cur = grouped.get(key)
        if cur is None:
            grouped[key] = {
                "left_node": left_node,
                "right_node": right_node,
                "left_level": left_level,
                "right_level": right_level,
                "weight": rank_score,
                "hit_count": 1,
                "best_rank": rank_final,
                "best_rank_score": rank_score,
                "best_left_leaf_id": row.get("left_leaf_id", ""),
                "best_right_leaf_id": row.get("right_leaf_id", ""),
                "best_left_parent_id": row.get("left_parent_id", ""),
                "best_right_parent_id": row.get("right_parent_id", ""),
                "left_corpus": row.get("left_corpus", ""),
                "right_corpus": row.get("right_corpus", ""),
                "right_doc_no": row.get("right_doc_no"),
                "best_tesserae": row.get("tesserae", 0.0),
                "best_soft_cos": row.get("soft_cos", 0.0),
                "best_sw_norm": row.get("sw_norm", 0.0),
                "best_cos_sim": row.get("cos_sim", 0.0),
                "best_pareto_layer": row.get("pareto_layer", 0),
            }
        else:
            cur["hit_count"] += 1
            if weight_mode == "sum":
                cur["weight"] = float(cur["weight"]) + rank_score
            elif weight_mode == "max":
                cur["weight"] = max(float(cur["weight"]), rank_score)
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            if rank_final < int(cur["best_rank"]):
                cur["best_rank"] = rank_final
                cur["best_rank_score"] = rank_score
                cur["best_left_leaf_id"] = row.get("left_leaf_id", "")
                cur["best_right_leaf_id"] = row.get("right_leaf_id", "")
                cur["best_left_parent_id"] = row.get("left_parent_id", "")
                cur["best_right_parent_id"] = row.get("right_parent_id", "")
                cur["best_tesserae"] = row.get("tesserae", 0.0)
                cur["best_soft_cos"] = row.get("soft_cos", 0.0)
                cur["best_sw_norm"] = row.get("sw_norm", 0.0)
                cur["best_cos_sim"] = row.get("cos_sim", 0.0)
                cur["best_pareto_layer"] = row.get("pareto_layer", 0)

    rows = [r for r in grouped.values() if int(r.get("hit_count", 1)) >= int(min_hits)]
    rows.sort(
        key=lambda r: (
            -float(r["weight"]),
            int(r.get("best_rank", 10**9)),
            str(r["left_node"]),
            str(r["right_node"]),
        )
    )
    return rows


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

        doc_no = extract_doc_no(seg)
        if doc_no is not None:
            sort_key = (doc_no, node_id)
        else:
            sort_key = generic_numeric_sort_key(node_id)

        legend_label = corpus_spec.get("display_ru", seg.corpus)

        out[node_id] = {
            "side": side,
            "group": seg.corpus,
            "label": label,
            "legend_label": legend_label,
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
    safe: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif value is None:
            safe[key] = ""
        else:
            safe[key] = repr(value)
    return safe


def metric_row_to_dict(r: CandidateMetrics) -> Dict[str, Any]:
    return {
        "left_leaf_id": r.left_leaf_id,
        "right_leaf_id": r.right_leaf_id,
        "left_parent_id": r.left_parent_id,
        "right_parent_id": r.right_parent_id,
        "left_corpus": r.left_corpus,
        "right_corpus": r.right_corpus,
        "left_node": r.left_node,
        "right_node": r.right_node,
        "right_doc_no": r.right_doc_no,
        "cos_sim": r.cos_sim,
        "tesserae": r.tesserae,
        "soft_cos": r.soft_cos,
        "sw_score_raw": r.sw_score_raw,
        "sw_norm": r.sw_norm,
        "alignment_a": r.alignment_a[:25],
        "alignment_b": r.alignment_b[:25],
        "left_text_snippet": r.left_text_snippet,
        "right_text_snippet": r.right_text_snippet,
        "pareto_layer": r.pareto_layer,
        "pareto_on_front": r.pareto_on_front,
        "rank_cos": r.rank_cos,
        "rank_tess": r.rank_tess,
        "rank_soft": r.rank_soft,
        "rank_sw": r.rank_sw,
        "rank_sum": r.rank_sum,
        "rank_final_position": r.rank_final_position,
        "rank_score": r.rank_score,
    }


def run_experiment(
    config_module: Any,
    experiment_id: str,
    verbose: bool = True,
    progress_every: Optional[int] = None,
    use_checkpoints: bool = True,
    force_from_step: Optional[int] = None,
) -> Dict[str, Any]:
    logger = ProgressLogger(enabled=verbose)

    corpora: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "CORPORA"))
    groups: Dict[str, List[str]] = dict(getattr(config_module, "GROUPS", {}))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "EXPERIMENTS"))

    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")

    exp = experiments[experiment_id]
    logging_cfg = resolve_logging_config(config_module, exp, progress_every_override=progress_every)
    lemmatize_progress_every = logging_cfg.get("lemmatize_progress_every")
    candidate_progress_every = logging_cfg.get("candidate_progress_every")
    scoring_progress_every = logging_cfg.get("scoring_progress_every")

    out_dir = Path(exp["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_cfg = dict(exp.get("retrieval") or {})
    selection_cfg = dict(exp.get("selection") or {})
    pareto_cfg = dict(exp.get("pareto") or {})
    agg_cfg = dict(exp.get("aggregation") or {})
    viz_cfg = dict(exp.get("viz") or {})
    output_cfg = dict(exp.get("output") or {})
    model = dict(exp.get("model") or {})

    checkpoint_dir = Path(output_cfg.get("dir", ".")) / "checkpoints"

    retrieval_budget = retrieval_cfg.get("budget")
    if retrieval_budget is None:
        raise ValueError(
            f"Experiment '{experiment_id}' must define retrieval.budget in config_unified.py "
            f"for the new threshold-free architecture."
        )
    retrieval_budget = int(retrieval_budget)
    if retrieval_budget <= 0:
        raise ValueError(f"retrieval.budget must be positive, got {retrieval_budget}")

    graph_top_n = selection_cfg.get("graph_top_n", viz_cfg.get("top_n_edges"))
    if graph_top_n is None:
        raise ValueError(
            f"Experiment '{experiment_id}' must define selection.graph_top_n "
            f"(or viz.top_n_edges) for graph selection."
        )
    graph_top_n = int(graph_top_n)
    if graph_top_n <= 0:
        raise ValueError(f"graph_top_n must be positive, got {graph_top_n}")

    pareto_keep_layers = int(pareto_cfg.get("keep_layers", 1))
    if pareto_keep_layers <= 0:
        raise ValueError(f"pareto.keep_layers must be positive, got {pareto_keep_layers}")

    left_level = str(agg_cfg.get("left_node_level", "parent"))
    right_level = str(agg_cfg.get("right_node_level", "parent"))
    weight_mode = str(agg_cfg.get("weight_mode", "sum"))
    min_hits = int(agg_cfg.get("min_hits", 1))

    logger.log("Step 1/7: Loading config and segmenting corpora...")

    left_corpora = resolve_group_tokens(exp["graph_sides"]["left"], groups)
    right_corpora = resolve_group_tokens(exp["graph_sides"]["right"], groups)

    mappings = exp.get("mappings") or [{"from": left_corpora, "to": right_corpora}]
    resolved_mappings: List[Tuple[List[str], List[str]]] = []
    for m in mappings:
        frm = resolve_group_tokens(m.get("from", []), groups)
        to = resolve_group_tokens(m.get("to", []), groups)
        resolved_mappings.append((frm, to))

    used_corpora = sorted(set(left_corpora + right_corpora))
    chunk_cfg = dict(exp.get("chunking") or {})
    checkpoint_fingerprint = compute_checkpoint_fingerprint(
        experiment_id=experiment_id,
        left_corpora=left_corpora,
        right_corpora=right_corpora,
        resolved_mappings=resolved_mappings,
        chunk_cfg=chunk_cfg,
        model=model,
        agg_cfg=agg_cfg,
        pareto_cfg=pareto_cfg,
        selection_cfg=selection_cfg,
        viz_cfg=viz_cfg,
        corpora=corpora,
        used_corpora=used_corpora,
    )
    checkpoints = CheckpointManager(
        checkpoint_dir,
        checkpoint_fingerprint,
        logger,
        enabled=use_checkpoints,
        force_from_step=force_from_step,
    )
    if use_checkpoints:
        logger.log(f"  Checkpoints: enabled ({checkpoint_dir})")
    else:
        logger.log("  Checkpoints: disabled")

    step1_data = checkpoints.load(1, "segmentation")
    if step1_data is None:
        base_segments: List[Segment] = []
        segmentation_counts: Dict[str, int] = {}

        for cid in used_corpora:
            if cid not in corpora:
                raise KeyError(f"Corpus {cid} is not defined in CORPORA")
            segs = segment_corpus(cid, corpora[cid], logger)
            segmentation_counts[cid] = len(segs)
            for seg_id, seg_text in segs:
                base_segments.append(Segment(
                    id=seg_id,
                    text=seg_text,
                    corpus=cid,
                    side="__unassigned__",
                    parent_id=seg_id,
                ))

        base_by_corpus: Dict[str, List[Segment]] = {cid: [] for cid in used_corpora}
        for seg in base_segments:
            base_by_corpus[seg.corpus].append(seg)

        left_base: List[Segment] = []
        for cid in left_corpora:
            for seg in base_by_corpus[cid]:
                left_base.append(Segment(seg.id, seg.text, seg.corpus, "left", seg.parent_id))

        right_base: List[Segment] = []
        for cid in right_corpora:
            for seg in base_by_corpus[cid]:
                right_base.append(Segment(seg.id, seg.text, seg.corpus, "right", seg.parent_id))

        step1_data = {
            "segmentation_counts": segmentation_counts,
            "left_base": left_base,
            "right_base": right_base,
            "used_corpora": used_corpora,
        }
        checkpoints.save(1, "segmentation", step1_data)

    segmentation_counts = dict(step1_data["segmentation_counts"])
    left_base = list(step1_data["left_base"])
    right_base = list(step1_data["right_base"])

    seg_summary = ", ".join(f"{cid}={segmentation_counts[cid]}" for cid in used_corpora)
    logger.log(f"  Segmentation done: {seg_summary}")
    logger.log(f"  Left base={len(left_base)}, right base={len(right_base)}, total={len(left_base) + len(right_base)}")

    logger.log("Step 2/7: Chunking segments...")
    step2_data = checkpoints.load(2, "chunking")
    if step2_data is None:
        left_leaf = chunk_segments(left_base, chunk_cfg, logger, "left")
        right_leaf = chunk_segments(right_base, chunk_cfg, logger, "right")
        step2_data = {
            "left_leaf": left_leaf,
            "right_leaf": right_leaf,
        }
        checkpoints.save(2, "chunking", step2_data)
    left_leaf = list(step2_data["left_leaf"])
    right_leaf = list(step2_data["right_leaf"])
    logger.log(
        f"  Chunking done: left {len(left_base)} -> {len(left_leaf)}, "
        f"right {len(right_base)} -> {len(right_leaf)}"
    )

    logger.log("Step 3/7: Preprocessing and token reduction...")
    min_lemma_length = int(model.get("min_lemma_length", 3))
    step3_data = checkpoints.load(3, "preprocessing")
    if step3_data is None:
        lemmatizer = LatinLemmatizer()
        step3_partial = Step3CorpusCheckpointManager(
            checkpoints.root_dir / "step_03_preprocessing_parts",
            fingerprint=checkpoints.fingerprint,
            logger=logger,
            enabled=checkpoints.enabled,
            force_from_step=checkpoints.force_from_step,
        )
        left_lemmas: Dict[str, List[str]] = {}
        right_lemmas: Dict[str, List[str]] = {}

        left_by_corpus: Dict[str, List[Segment]] = {}
        for seg in left_leaf:
            left_by_corpus.setdefault(seg.corpus, []).append(seg)
        right_by_corpus: Dict[str, List[Segment]] = {}
        for seg in right_leaf:
            right_by_corpus.setdefault(seg.corpus, []).append(seg)

        for corpus_id in left_corpora:
            corpus_segments = left_by_corpus.get(corpus_id, [])
            cached = step3_partial.load("left", corpus_id)
            if cached is None:
                logger.log(f"  Preprocessing corpus start: left/{corpus_id}, segments={len(corpus_segments)}")
                cached = lemmatize_segments_strict(
                    corpus_segments,
                    lemmatizer,
                    min_lemma_length,
                    logger,
                    "left",
                    corpus_id,
                    progress_every=lemmatize_progress_every,
                )
                step3_partial.save("left", corpus_id, cached)
            left_lemmas.update(cached)

        for corpus_id in right_corpora:
            corpus_segments = right_by_corpus.get(corpus_id, [])
            cached = step3_partial.load("right", corpus_id)
            if cached is None:
                logger.log(f"  Preprocessing corpus start: right/{corpus_id}, segments={len(corpus_segments)}")
                cached = lemmatize_segments_strict(
                    corpus_segments,
                    lemmatizer,
                    min_lemma_length,
                    logger,
                    "right",
                    corpus_id,
                    progress_every=lemmatize_progress_every,
                )
                step3_partial.save("right", corpus_id, cached)
            right_lemmas.update(cached)

        step3_data = {
            "left_lemmas": left_lemmas,
            "right_lemmas": right_lemmas,
        }
        checkpoints.save(3, "preprocessing", step3_data)
        step3_partial.clear()
    left_lemmas = dict(step3_data["left_lemmas"])
    right_lemmas = dict(step3_data["right_lemmas"])
    logger.log(f"  Preprocessing done: left={len(left_lemmas)}, right={len(right_lemmas)}")

    left_ids = [s.id for s in left_leaf]
    right_ids = [s.id for s in right_leaf]
    corpus_lemmas = [left_lemmas[i] for i in left_ids] + [right_lemmas[i] for i in right_ids]

    logger.log("Step 4/7: Building TF-IDF features and budgeted retrieval...")
    left_seg_by_id = {s.id: s for s in left_leaf}
    right_seg_by_id = {s.id: s for s in right_leaf}
    step4_data = checkpoints.load(4, "retrieval")
    if step4_data is None:
        tfidf, vocab, term2idx = build_tfidf_matrix(
            corpus_lemmas,
            ngram_range=tuple(model.get("ngram_range", (1, 3))),
            max_df=float(model.get("max_df", 0.5)),
            min_df=int(model.get("min_df", 2)),
        )
        logger.log(
            f"  TF-IDF done: docs={len(corpus_lemmas)}, "
            f"vocab={len(vocab)}, matrix={getattr(tfidf, 'shape', None)}"
        )

        n_left = len(left_ids)
        tfidf_left = tfidf[:n_left]
        tfidf_right = tfidf[n_left:]

        allowed_right_corpora_by_left = build_allowed_right_corpora_by_left(
            left_corpora, right_corpora, resolved_mappings
        )

        candidates = select_retrieval_candidates_budgeted(
            tfidf_left=tfidf_left,
            tfidf_right=tfidf_right,
            left_ids=left_ids,
            right_ids=right_ids,
            left_seg_by_id=left_seg_by_id,
            right_seg_by_id=right_seg_by_id,
            allowed_right_corpora_by_left=allowed_right_corpora_by_left,
            retrieval_budget=retrieval_budget,
            logger=logger,
            progress_every=candidate_progress_every,
        )
        idf = compute_idf(corpus_lemmas)
        step4_data = {
            "candidates": candidates,
            "idf": idf,
        }
        checkpoints.save(4, "retrieval", step4_data)
    candidates = list(step4_data["candidates"])
    idf = dict(step4_data["idf"])
    logger.log(f"  Retrieval done: budget={retrieval_budget}, kept={len(candidates)}")

    logger.log("Step 5/7: Computing all 4 metrics for retrieval candidates...")
    step5_data = checkpoints.load(5, "metrics")
    if step5_data is None:
        partial_dir = checkpoints.root_dir / "step_05_metrics_parts"
        step5_partial = Step5PartialCheckpointManager(
            root_dir=partial_dir,
            fingerprint=checkpoints.fingerprint,
            logger=logger,
            enabled=checkpoints.enabled,
            force_from_step=checkpoints.force_from_step,
        )

        total_candidates = len(candidates)
        save_every = max(1, int(scoring_progress_every or 250))
        progress = step5_partial.load_progress()
        start_idx = 1
        metric_rows: List[CandidateMetrics] = []
        skipped_empty_lemmas = 0

        if progress is not None:
            start_idx = max(1, int(progress.get("next_index", 1)))
            skipped_empty_lemmas = int(progress.get("skipped_empty_lemmas", 0))
            metric_rows = step5_partial.load_parts(start_idx - 1)
            if start_idx > 1:
                logger.log(
                    f"  Step 5 resume: starting from candidate {start_idx}/{total_candidates}, "
                    f"restored={len(metric_rows)}, skipped_empty={skipped_empty_lemmas}"
                )

        batch_rows: List[CandidateMetrics] = []

        for idx, (left_id, right_id, cos_sim) in enumerate(candidates, 1):
            if idx < start_idx:
                continue

            ls = left_seg_by_id[left_id]
            rs = right_seg_by_id[right_id]
            llem = left_lemmas.get(left_id, [])
            rlem = right_lemmas.get(right_id, [])

            if not llem or not rlem:
                skipped_empty_lemmas += 1
            else:
                tess, soft, sw_raw, sw_norm, al_a, al_b = compute_candidate_metrics(
                    cos_sim=float(cos_sim),
                    left_lem=llem,
                    right_lem=rlem,
                    idf=idf,
                    model=model,
                )

                batch_rows.append(CandidateMetrics(
                    left_leaf_id=left_id,
                    right_leaf_id=right_id,
                    left_parent_id=ls.parent_id,
                    right_parent_id=rs.parent_id,
                    left_corpus=ls.corpus,
                    right_corpus=rs.corpus,
                    left_node=node_id_for_level(ls, left_level),
                    right_node=node_id_for_level(rs, right_level),
                    right_doc_no=extract_doc_no(rs),
                    cos_sim=float(cos_sim),
                    tesserae=float(tess),
                    soft_cos=float(soft),
                    sw_score_raw=float(sw_raw),
                    sw_norm=float(sw_norm),
                    alignment_a=al_a[:25],
                    alignment_b=al_b[:25],
                    left_text_snippet=ls.text[:220].replace("\n", " ").strip(),
                    right_text_snippet=rs.text[:220].replace("\n", " ").strip(),
                ))

            should_flush = (idx % save_every == 0) or (idx == total_candidates)
            if should_flush:
                step5_partial.save_progress(
                    upto_index=idx,
                    part_metric_rows=batch_rows,
                    total_computed=len(metric_rows) + len(batch_rows),
                    skipped_empty_lemmas=skipped_empty_lemmas,
                    total_candidates=total_candidates,
                )
                metric_rows.extend(batch_rows)
                batch_rows = []

            if scoring_progress_every and (
                idx % max(1, int(scoring_progress_every)) == 0 or idx == total_candidates
            ):
                logger.log(
                    f"  Metric progress: {idx}/{len(candidates)}, "
                    f"computed={len(metric_rows) + len(batch_rows)}, skipped_empty={skipped_empty_lemmas}"
                )

        if batch_rows:
            metric_rows.extend(batch_rows)

        step5_data = {
            "metric_rows": metric_rows,
            "skipped_empty_lemmas": skipped_empty_lemmas,
        }
        checkpoints.save(5, "metrics", step5_data)
        step5_partial.clear()

    metric_rows = list(step5_data["metric_rows"])
    skipped_empty_lemmas = int(step5_data.get("skipped_empty_lemmas", 0))
    if len(metric_rows) > len(candidates):
        raise RuntimeError(
            f"Corrupted step 5 checkpoint: computed metric_rows={len(metric_rows)} exceeds "
            f"retrieval candidates={len(candidates)}. Delete checkpoints/step_05_metrics.pkl "
            f"and checkpoints/step_05_metrics_parts and rerun from step 5."
        )
    logger.log(
        f"  Metric computation done: computed={len(metric_rows)}, skipped_empty={skipped_empty_lemmas}"
    )

    logger.log("Step 6/7: Pareto filtering and rank aggregation...")
    step6_data = checkpoints.load(6, "pareto_rank")
    if step6_data is None:
        if len(metric_rows) > len(candidates):
            raise RuntimeError(
                f"Refusing Step 6: metric_rows={len(metric_rows)} exceeds candidates={len(candidates)}. "
                f"Most likely Step 5 partial checkpoints were saved cumulatively and reloaded with duplicates."
            )
        assign_pareto_layers(metric_rows)
        pareto_rows = [r for r in metric_rows if r.pareto_layer <= pareto_keep_layers]
        rank_aggregate(pareto_rows)

        pareto_rows.sort(
            key=lambda r: (
                r.rank_final_position,
                r.left_corpus,
                r.right_corpus,
                r.left_leaf_id,
                r.right_leaf_id,
            )
        )

        detail_rows = [metric_row_to_dict(r) for r in pareto_rows]
        step6_data = {
            "detail_rows": detail_rows,
            "metric_rows_total": len(metric_rows),
        }
        checkpoints.save(6, "pareto_rank", step6_data)

    detail_rows = list(step6_data["detail_rows"])
    logger.log(
        f"  Pareto done: total={int(step6_data.get('metric_rows_total', len(metric_rows)))}, "
        f"kept_layers<={pareto_keep_layers} -> {len(detail_rows)}"
    )

    logger.log("Step 7/7: Aggregating graph rows and exporting outputs...")
    step7_data = checkpoints.load(7, "graph_rows")
    if step7_data is None:
        graph_rows = aggregate_rows(
            detail_rows,
            left_level=left_level,
            right_level=right_level,
            weight_mode=weight_mode,
            min_hits=min_hits,
        )
        graph_rows = graph_rows[:graph_top_n]
        step7_data = {
            "graph_rows": graph_rows,
        }
        checkpoints.save(7, "graph_rows", step7_data)
    graph_rows = list(step7_data["graph_rows"])
    logger.log(f"  Graph aggregation done: detail_rows={len(detail_rows)}, graph_rows={len(graph_rows)}")

    if bool(output_cfg.get("write_detail_csv")):
        path = out_dir / "detail_pairs.csv"
        write_csv(detail_rows, path)
        logger.log(f"  Wrote CSV: {path.name}")

    if bool(output_cfg.get("write_graph_csv")):
        path = out_dir / "graph_rows.csv"
        write_csv(graph_rows, path)
        logger.log(f"  Wrote CSV: {path.name}")

    left_meta = build_node_metadata(left_leaf, corpora, node_level=left_level, side="left")
    right_meta = build_node_metadata(right_leaf, corpora, node_level=right_level, side="right")

    G = None
    if nx is not None:
        G = nx.DiGraph()
        for nid, meta in left_meta.items():
            G.add_node(nid, **meta)
        for nid, meta in right_meta.items():
            G.add_node(nid, **meta)
        for r in graph_rows:
            u = str(r["left_node"])
            v = str(r["right_node"])
            if u in left_meta and v in right_meta:
                G.add_edge(
                    u,
                    v,
                    weight=float(r.get("weight", 0.0)),
                    hit_count=int(r.get("hit_count", 1)),
                    best_rank=int(r.get("best_rank", 0)),
                )

        if bool(output_cfg.get("write_gexf")):
            path = out_dir / "graph.gexf"
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
                        "best_rank": int(r.get("best_rank", 0)),
                        "best_rank_score": float(r.get("best_rank_score", 0.0)),
                        "left_level": r.get("left_level", ""),
                        "right_level": r.get("right_level", ""),
                        "left_corpus": r.get("left_corpus", ""),
                        "right_corpus": r.get("right_corpus", ""),
                        "best_left_leaf_id": r.get("best_left_leaf_id", ""),
                        "best_right_leaf_id": r.get("best_right_leaf_id", ""),
                        "best_left_parent_id": r.get("best_left_parent_id", ""),
                        "best_right_parent_id": r.get("best_right_parent_id", ""),
                        "right_doc_no": r.get("right_doc_no", ""),
                        "best_tesserae": float(r.get("best_tesserae", 0.0)),
                        "best_soft_cos": float(r.get("best_soft_cos", 0.0)),
                        "best_sw_norm": float(r.get("best_sw_norm", 0.0)),
                        "best_cos_sim": float(r.get("best_cos_sim", 0.0)),
                        "best_pareto_layer": int(r.get("best_pareto_layer", 0)),
                    })
                    G_gexf.add_edge(u, v, **edge_attrs)

            nx.write_gexf(G_gexf, path)
            logger.log(f"  Wrote GEXF: {path.name}")

    if bool(viz_cfg.get("enabled")) and bool(output_cfg.get("write_png")):
        path = out_dir / "graph.png"
        logger.log(
            f"  Rendering PNG: {path.name} "
            f"(top_n_edges={graph_top_n}, total_graph_rows={len(graph_rows)})"
        )
        render_bipartite_graph(
            graph_rows=graph_rows,
            left_nodes=left_meta,
            right_nodes=right_meta,
            out_png=path,
            straight_edges=bool(viz_cfg.get("straight_edges", True)),
            label_left=bool(viz_cfg.get("label_left", True)),
            label_right=bool(viz_cfg.get("label_right", True)),
            top_n_edges=graph_top_n,
        )
        logger.log(f"  Wrote PNG: {path.name}")

    logger.log("Experiment finished")
    logger.log(
        f"Summary: left_leaf={len(left_leaf)}, right_leaf={len(right_leaf)}, "
        f"retrieval_candidates={len(candidates)}, pareto_rows={len(detail_rows)}, graph_rows={len(graph_rows)}"
    )
    logger.log(f"Output dir: {out_dir}")

    return {
        "experiment_id": experiment_id,
        "detail_rows": detail_rows,
        "graph_rows": graph_rows,
        "out_dir": str(out_dir),
        "graph": G,
        "checkpoints_enabled": use_checkpoints,
        "checkpoint_dir": str(checkpoint_dir),
        "stats": {
            "left_leaf": len(left_leaf),
            "right_leaf": len(right_leaf),
            "retrieval_candidates": len(candidates),
            "pareto_rows": len(detail_rows),
            "graph_rows": len(graph_rows),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_unified", help="Python module name, e.g. config_unified")
    ap.add_argument("--experiment", required=True, help="Experiment id, see EXPERIMENTS in config")
    ap.add_argument("--quiet", action="store_true", help="Disable progress logs")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Override scoring progress period from logging config",
    )
    ap.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable smart checkpoints and always recompute all steps",
    )
    ap.add_argument(
        "--force-from-step",
        type=int,
        default=None,
        choices=list(range(1, 8)),
        help="Ignore checkpoints начиная с указанного шага и пересчитать его и все следующие",
    )
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)
    result = run_experiment(
        cfg_mod,
        args.experiment,
        verbose=not args.quiet,
        progress_every=args.progress_every,
        use_checkpoints=not args.no_checkpoints,
        force_from_step=args.force_from_step,
    )

    stats = result.get("stats") or {}
    print(f"[unified] experiment={args.experiment}", flush=True)
    for k in sorted(stats.keys()):
        print(f"[unified] {k}: {stats[k]}", flush=True)
    print(f"[unified] out_dir: {result.get('out_dir')}", flush=True)
    print(f"[unified] checkpoint_dir: {result.get('checkpoint_dir')}", flush=True)


if __name__ == "__main__":
    main()
