#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
explain_top_borrowings_from_ready_experiment_v3.py

Сильно улучшенный explain-скрипт для УЖЕ ГОТОВОГО эксперимента.

Что делает:
- НЕ запускает pipeline
- читает готовые checkpoints шагов 2/3/4/6
- строит человекочитаемый markdown-отчёт по top-K парам
- показывает:
  * raw сегменты
  * preprocessing как тексты, а не как списки
  * какие токены исчезли / изменились
  * TF-IDF evidence
  * Tesserae: какие именно леммы совпали и сколько каждая даёт в score
  * soft cosine: exact / fuzzy overlaps
  * сохранённые метрики и ранги

Поддерживает pickle, созданные когда pipeline.py запускался как __main__.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import pipeline as pl
from src.features import build_tfidf_matrix, compute_idf, levenshtein_distance
from src.preprocessing import LatinLemmatizer, preprocess_segment


class PipelineCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "Segment":
            return pl.Segment
        if module == "__main__" and name == "CandidateMetrics":
            return pl.CandidateMetrics
        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> Any:
    with open(path, "rb") as f:
        return PipelineCheckpointUnpickler(f).load()


def load_config_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def resolve_experiment_context(config_module: Any, experiment_id: str) -> Dict[str, Any]:
    corpora: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "CORPORA"))
    groups: Dict[str, List[str]] = dict(getattr(config_module, "GROUPS", {}))
    experiments: Dict[str, Dict[str, Any]] = dict(getattr(config_module, "EXPERIMENTS"))

    if experiment_id not in experiments:
        raise KeyError(f"Unknown experiment_id={experiment_id}. Available: {sorted(experiments.keys())}")

    exp = dict(experiments[experiment_id])

    left_corpora = pl.resolve_group_tokens(exp["graph_sides"]["left"], groups)
    right_corpora = pl.resolve_group_tokens(exp["graph_sides"]["right"], groups)

    mappings = exp.get("mappings") or [{"from": left_corpora, "to": right_corpora}]
    resolved_mappings: List[Tuple[List[str], List[str]]] = []
    for m in mappings:
        frm = pl.resolve_group_tokens(m.get("from", []), groups)
        to = pl.resolve_group_tokens(m.get("to", []), groups)
        resolved_mappings.append((frm, to))

    model = dict(exp.get("model") or {})
    retrieval_cfg = dict(exp.get("retrieval") or {})
    agg_cfg = dict(exp.get("aggregation") or {})
    pareto_cfg = dict(exp.get("pareto") or {})
    selection_cfg = dict(exp.get("selection") or {})
    chunk_cfg = dict(exp.get("chunking") or {})
    output_cfg = dict(exp.get("output") or {})

    used_corpora = sorted(set(left_corpora + right_corpora))
    step_fingerprints = pl.build_checkpoint_fingerprints(
        experiment_id=experiment_id,
        left_corpora=left_corpora,
        right_corpora=right_corpora,
        resolved_mappings=resolved_mappings,
        chunk_cfg=chunk_cfg,
        model=model,
        retrieval_cfg=retrieval_cfg,
        agg_cfg=agg_cfg,
        pareto_cfg=pareto_cfg,
        selection_cfg=selection_cfg,
        corpora=corpora,
        used_corpora=used_corpora,
    )

    checkpoint_dir = Path(output_cfg.get("dir", ".")) / "checkpoints"

    return {
        "corpora": corpora,
        "exp": exp,
        "model": model,
        "output_cfg": output_cfg,
        "checkpoint_dir": checkpoint_dir,
        "step_fingerprints": step_fingerprints,
    }


def load_checkpoint_strict(
    checkpoint_dir: Path,
    expected_fingerprint: str,
    step_no: int,
    step_key: str,
) -> Any:
    path = checkpoint_dir / f"step_{step_no:02d}_{step_key}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Этот скрипт работает только по готовому эксперименту."
        )

    payload = load_pickle_compat(path)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload type in {path}: {type(payload)}")

    got = payload.get("fingerprint")
    if got != expected_fingerprint:
        raise RuntimeError(
            f"Fingerprint mismatch for {path.name}: expected={expected_fingerprint}, got={got}\n"
            f"Похоже, конфиг/исходники поменялись после запуска эксперимента."
        )

    return payload.get("data")


def text_block(text: str) -> str:
    return "```text\n" + (text or "") + "\n```"


def json_block(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def join_tokens(tokens: Sequence[str]) -> str:
    return " ".join(str(t) for t in tokens)


def unique_preserve_order(tokens: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def multiset_removed(before: Sequence[str], after: Sequence[str]) -> List[str]:
    cb = Counter(before)
    ca = Counter(after)
    out: List[str] = []
    for tok in before:
        if cb[tok] > ca.get(tok, 0):
            out.append(tok)
            cb[tok] -= 1
    return out


def pairwise_token_changes(a: Sequence[str], b: Sequence[str], limit: int = 30) -> List[str]:
    n = min(len(a), len(b))
    rows: List[str] = []
    for i in range(n):
        if a[i] != b[i]:
            rows.append(f"{a[i]} → {b[i]}")
    if len(a) > n:
        rows.extend(f"{x} → ∅" for x in a[n:])
    elif len(b) > n:
        rows.extend(f"∅ → {x}" for x in b[n:])
    return rows[:limit]


def format_preproc_trace(dbg: Dict[str, Any]) -> str:
    raw = dbg["raw_tokens"]
    norm = dbg["normalized_tokens"]
    stem = dbg["stemmed_tokens"]
    final = dbg["final_tokens"]

    changed_raw_norm = pairwise_token_changes(raw, norm)
    changed_norm_stem = pairwise_token_changes(norm, stem)
    removed_after_filter = multiset_removed(stem, final)

    lines = []
    lines.append("MODE:")
    lines.append(f"{dbg['mode']}  scores={dbg['scores']}")
    lines.append("")
    lines.append("RAW:")
    lines.append(join_tokens(raw))
    lines.append("")
    lines.append("NORMALIZED:")
    lines.append(join_tokens(norm))
    lines.append("")
    lines.append("STEMMED:")
    lines.append(join_tokens(stem))
    lines.append("")
    lines.append("FINAL TOKENS:")
    lines.append(join_tokens(final))
    lines.append("")

    lines.append("RAW → NORMALIZED changes:")
    lines.append("; ".join(changed_raw_norm) if changed_raw_norm else "(no visible token-level changes)")
    lines.append("")
    lines.append("NORMALIZED → STEMMED changes:")
    lines.append("; ".join(changed_norm_stem) if changed_norm_stem else "(no visible token-level changes)")
    lines.append("")
    lines.append("Removed after filtering:")
    lines.append(", ".join(removed_after_filter) if removed_after_filter else "(nothing removed)")

    return text_block("\n".join(lines))


def top_terms_from_vector(vec: np.ndarray, vocab: Sequence[str], topn: int = 15) -> List[Tuple[str, float]]:
    arr = np.asarray(vec).ravel()
    nz = np.flatnonzero(arr)
    if len(nz) == 0:
        return []
    rows = [(str(vocab[i]), float(arr[i])) for i in nz]
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows[:topn]


def shared_terms(
    left_vec: np.ndarray,
    right_vec: np.ndarray,
    vocab: Sequence[str],
    topn: int = 20,
) -> List[Tuple[str, float, float, float]]:
    lv = np.asarray(left_vec).ravel()
    rv = np.asarray(right_vec).ravel()
    common = np.flatnonzero((lv > 0) & (rv > 0))
    rows: List[Tuple[str, float, float, float]] = []
    for idx in common:
        lw = float(lv[idx])
        rw = float(rv[idx])
        rows.append((str(vocab[idx]), lw, rw, lw * rw))
    rows.sort(key=lambda x: (-x[3], x[0]))
    return rows[:topn]


def compute_tesserae_explanation(
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
    idf: Dict[str, float],
) -> Dict[str, Any]:
    left_unique = set(left_tokens)
    right_unique = set(right_tokens)
    shared = sorted(left_unique & right_unique, key=lambda t: (-idf.get(t, 0.0), t))
    contributions = [
        {
            "lemma": tok,
            "idf": round(float(idf.get(tok, 0.0)), 6),
        }
        for tok in shared
    ]
    numerator = sum(idf.get(tok, 0.0) for tok in shared)
    norm = math.sqrt(max(1, len(left_unique)) * max(1, len(right_unique)))
    score = numerator / norm if norm > 0 and len(shared) >= 2 else 0.0
    return {
        "shared_lemmas": shared,
        "shared_lemma_count": len(shared),
        "idf_contributions": contributions,
        "numerator_sum_idf": round(float(numerator), 6),
        "normalizer_sqrt_unique_product": round(float(norm), 6),
        "recomputed_tesserae": round(float(score), 6),
    }


def compute_soft_overlap_explanation(
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
    lev_threshold: int = 2,
    limit: int = 25,
) -> Dict[str, Any]:
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)

    exact = sorted(set(left_counts) & set(right_counts))
    fuzzy_rows = []

    for lt in left_counts:
        for rt in right_counts:
            if lt == rt:
                continue
            d = levenshtein_distance(lt, rt)
            if d <= lev_threshold:
                max_len = max(len(lt), len(rt))
                sim = 0.0 if max_len == 0 else 1.0 - d / max_len
                if sim > 0.0:
                    fuzzy_rows.append({
                        "left": lt,
                        "right": rt,
                        "distance": int(d),
                        "similarity": round(float(sim), 6),
                        "left_count": int(left_counts[lt]),
                        "right_count": int(right_counts[rt]),
                        "weighted_contribution_hint": round(float(sim * left_counts[lt] * right_counts[rt]), 6),
                    })

    fuzzy_rows.sort(
        key=lambda r: (-r["weighted_contribution_hint"], r["left"], r["right"])
    )

    return {
        "exact_shared_lemmas": exact[:limit],
        "fuzzy_near_matches": fuzzy_rows[:limit],
    }


def build_report(
    *,
    experiment_id: str,
    top_rows: List[Dict[str, Any]],
    left_seg_by_id: Dict[str, Any],
    right_seg_by_id: Dict[str, Any],
    tfidf: np.ndarray,
    vocab: Sequence[str],
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    candidate_rank: Dict[Tuple[str, str], int],
    model: Dict[str, Any],
    idf: Dict[str, float],
    top_terms_n: int,
) -> str:
    n_left = len(left_ids)
    left_index = {seg_id: i for i, seg_id in enumerate(left_ids)}
    right_index = {seg_id: i for i, seg_id in enumerate(right_ids)}

    lemmatizer = LatinLemmatizer()
    min_lemma_length = int(model.get("min_lemma_length", 3))
    soft_lev_threshold = 2

    parts: List[str] = []
    parts.append(f"# Top borrowings explanation from ready experiment: {experiment_id}")
    parts.append("")
    parts.append("Источник данных: готовые checkpoint-артефакты шага 2/3/4/6. Pipeline не запускался.")
    parts.append("Формат отчёта: raw text → preprocessing trace → retrieval evidence → tesserae / soft overlaps → saved metrics.")
    parts.append("")

    for ordinal, row in enumerate(top_rows, 1):
        left_id = str(row["left_leaf_id"])
        right_id = str(row["right_leaf_id"])
        left_seg = left_seg_by_id[left_id]
        right_seg = right_seg_by_id[right_id]

        li = left_index[left_id]
        ri = right_index[right_id]

        left_vec = tfidf[li]
        right_vec = tfidf[n_left + ri]

        left_dbg = preprocess_segment(
            left_seg.text,
            lemmatizer,
            min_length=min_lemma_length,
            return_debug=True,
        )
        right_dbg = preprocess_segment(
            right_seg.text,
            lemmatizer,
            min_length=min_lemma_length,
            return_debug=True,
        )

        left_final = left_dbg["final_tokens"]
        right_final = right_dbg["final_tokens"]

        pair_key = (left_id, right_id)
        retrieval_rank = candidate_rank.get(pair_key)
        cosine_from_tfidf = float(np.dot(np.asarray(left_vec).ravel(), np.asarray(right_vec).ravel()))

        left_top_terms = top_terms_from_vector(left_vec, vocab, topn=top_terms_n)
        right_top_terms = top_terms_from_vector(right_vec, vocab, topn=top_terms_n)
        shared_top = shared_terms(left_vec, right_vec, vocab, topn=top_terms_n)
        tess_info = compute_tesserae_explanation(left_final, right_final, idf)
        soft_info = compute_soft_overlap_explanation(left_final, right_final, lev_threshold=soft_lev_threshold, limit=top_terms_n)

        left_only = [t for t in unique_preserve_order(left_final) if t not in set(right_final)]
        right_only = [t for t in unique_preserve_order(right_final) if t not in set(left_final)]

        parts.append(f"## #{ordinal}: rank_final_position = {row['rank_final_position']}")
        parts.append("")
        parts.append(
            f"**Pair:** `{left_id}` → `{right_id}`  \n"
            f"**Parents:** `{row['left_parent_id']}` → `{row['right_parent_id']}`  \n"
            f"**Corpora:** `{row['left_corpus']}` → `{row['right_corpus']}`"
        )
        parts.append("")

        parts.append("### 1. Raw leaf segments")
        parts.append("")
        parts.append("**Left raw**")
        parts.append(text_block(left_seg.text))
        parts.append("")
        parts.append("**Right raw**")
        parts.append(text_block(right_seg.text))
        parts.append("")

        parts.append("### 2. Preprocessing trace")
        parts.append("")
        parts.append("**Left preprocessing**")
        parts.append(format_preproc_trace(left_dbg))
        parts.append("")
        parts.append("**Right preprocessing**")
        parts.append(format_preproc_trace(right_dbg))
        parts.append("")
        parts.append("**Final token overlap after preprocessing**")
        parts.append(text_block(
            "COMMON FINAL TOKENS:\n"
            + (", ".join(tess_info["shared_lemmas"]) if tess_info["shared_lemmas"] else "(none)")
            + "\n\nLEFT ONLY:\n"
            + (", ".join(left_only) if left_only else "(none)")
            + "\n\nRIGHT ONLY:\n"
            + (", ".join(right_only) if right_only else "(none)")
        ))
        parts.append("")

        parts.append("### 3. TF-IDF retrieval evidence")
        parts.append("")
        parts.append(json_block({
            "candidate_rank_in_step4": retrieval_rank,
            "cosine_from_rebuilt_tfidf": round(float(cosine_from_tfidf), 6),
            "cos_sim_saved_in_step6": round(float(row["cos_sim"]), 6),
        }))
        parts.append("")
        parts.append("**Left top TF-IDF terms**")
        parts.append(text_block("\n".join(
            f"{i+1:>2}. {term}    weight={weight:.6f}"
            for i, (term, weight) in enumerate(left_top_terms)
        ) or "(no non-zero TF-IDF terms)"))
        parts.append("")
        parts.append("**Right top TF-IDF terms**")
        parts.append(text_block("\n".join(
            f"{i+1:>2}. {term}    weight={weight:.6f}"
            for i, (term, weight) in enumerate(right_top_terms)
        ) or "(no non-zero TF-IDF terms)"))
        parts.append("")
        parts.append("**Shared TF-IDF terms contributing to cosine**")
        parts.append(text_block("\n".join(
            f"{i+1:>2}. {term}    left={lw:.6f}    right={rw:.6f}    product={prod:.6f}"
            for i, (term, lw, rw, prod) in enumerate(shared_top)
        ) or "(no shared non-zero TF-IDF terms)"))
        parts.append("")

        parts.append("### 4. Tesserae: what exactly matched")
        parts.append("")
        parts.append(text_block(
            "SHARED LEMMAS USED BY TESSERAE:\n"
            + (", ".join(tess_info["shared_lemmas"]) if tess_info["shared_lemmas"] else "(none)")
        ))
        parts.append("")
        parts.append("**Per-lemma IDF contributions**")
        parts.append(text_block("\n".join(
            f"{i+1:>2}. {row2['lemma']}    idf={row2['idf']:.6f}"
            for i, row2 in enumerate(tess_info["idf_contributions"])
        ) or "(shared lemmas < 2, so tesserae score collapses to 0)"))
        parts.append("")
        parts.append(json_block({
            "shared_lemma_count": tess_info["shared_lemma_count"],
            "numerator_sum_idf": tess_info["numerator_sum_idf"],
            "normalizer_sqrt_unique_product": tess_info["normalizer_sqrt_unique_product"],
            "recomputed_tesserae": tess_info["recomputed_tesserae"],
            "saved_tesserae_in_step6": round(float(row["tesserae"]), 6),
        }))
        parts.append("")

        parts.append("### 5. Soft cosine: exact and fuzzy overlaps")
        parts.append("")
        parts.append("**Exact overlaps**")
        parts.append(text_block(", ".join(soft_info["exact_shared_lemmas"]) or "(none)"))
        parts.append("")
        parts.append("**Near matches within Levenshtein threshold**")
        parts.append(text_block("\n".join(
            f"{i+1:>2}. {r['left']} ~ {r['right']}    dist={r['distance']}    sim={r['similarity']:.6f}    hint={r['weighted_contribution_hint']:.6f}"
            for i, r in enumerate(soft_info["fuzzy_near_matches"])
        ) or "(no fuzzy near-matches found)"))
        parts.append("")

        parts.append("### 6. Saved metrics from ready experiment")
        parts.append("")
        parts.append(json_block({
            "cos_sim": row["cos_sim"],
            "tesserae": row["tesserae"],
            "soft_cos": row["soft_cos"],
            "sw_score_raw": row["sw_score_raw"],
            "sw_norm": row["sw_norm"],
            "alignment_a": row["alignment_a"],
            "alignment_b": row["alignment_b"],
        }))
        parts.append("")

        parts.append("### 7. Why this pair ended up in top")
        parts.append("")
        parts.append(text_block(
            f"Pareto layer: {row['pareto_layer']}  (front={row['pareto_on_front']})\n"
            f"rank_cos={row['rank_cos']}, rank_tess={row['rank_tess']}, rank_soft={row['rank_soft']}, rank_sw={row['rank_sw']}\n"
            f"rank_sum={row['rank_sum']}\n"
            f"rank_final_position={row['rank_final_position']}\n"
            f"rank_score={row['rank_score']}"
        ))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config", help="Python module name, e.g. config")
    ap.add_argument("--experiment", required=True, help="Experiment id from config.EXPERIMENTS")
    ap.add_argument("--top-k", type=int, default=3, help="How many top pairs to explain")
    ap.add_argument("--top-terms", type=int, default=15, help="How many TF-IDF / tesserae / soft overlap rows to show")
    ap.add_argument("--output", default=None, help="Markdown output path")
    args = ap.parse_args()

    cfg_mod = load_config_module(args.config)
    ctx = resolve_experiment_context(cfg_mod, args.experiment)
    checkpoint_dir: Path = ctx["checkpoint_dir"]
    step_fingerprints: Dict[str, str] = ctx["step_fingerprints"]
    model: Dict[str, Any] = ctx["model"]
    output_cfg: Dict[str, Any] = ctx["output_cfg"]

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}\n"
            f"Сначала должен существовать готовый эксперимент."
        )

    step2 = load_checkpoint_strict(checkpoint_dir, step_fingerprints["step_02_chunking"], 2, "chunking")
    step3 = load_checkpoint_strict(checkpoint_dir, step_fingerprints["step_03_preprocessing"], 3, "preprocessing")
    step4 = load_checkpoint_strict(checkpoint_dir, step_fingerprints["step_04_retrieval"], 4, "retrieval")
    step6 = load_checkpoint_strict(checkpoint_dir, step_fingerprints["step_06_pareto_rank"], 6, "pareto_rank")

    left_leaf = list(step2["left_leaf"])
    right_leaf = list(step2["right_leaf"])
    left_lemmas = dict(step3["left_lemmas"])
    right_lemmas = dict(step3["right_lemmas"])
    candidates = list(step4["candidates"])
    detail_rows = list(step6["detail_rows"])

    left_ids = [s.id for s in left_leaf]
    right_ids = [s.id for s in right_leaf]
    corpus_lemmas = [left_lemmas[i] for i in left_ids] + [right_lemmas[i] for i in right_ids]

    tfidf, vocab, _term2idx = build_tfidf_matrix(
        corpus_lemmas,
        ngram_range=tuple(model.get("ngram_range", (1, 3))),
        max_df=float(model.get("max_df", 0.5)),
        min_df=int(model.get("min_df", 2)),
    )
    idf = compute_idf(corpus_lemmas)

    left_seg_by_id = {s.id: s for s in left_leaf}
    right_seg_by_id = {s.id: s for s in right_leaf}
    candidate_rank = {
        (str(left_id), str(right_id)): idx + 1
        for idx, (left_id, right_id, _cos) in enumerate(candidates)
    }

    detail_rows.sort(
        key=lambda r: (
            int(r["rank_final_position"]),
            str(r["left_leaf_id"]),
            str(r["right_leaf_id"]),
        )
    )
    top_rows = detail_rows[:max(1, int(args.top_k))]

    report = build_report(
        experiment_id=args.experiment,
        top_rows=top_rows,
        left_seg_by_id=left_seg_by_id,
        right_seg_by_id=right_seg_by_id,
        tfidf=tfidf,
        vocab=vocab,
        left_ids=left_ids,
        right_ids=right_ids,
        candidate_rank=candidate_rank,
        model=model,
        idf=idf,
        top_terms_n=max(1, int(args.top_terms)),
    )

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(output_cfg.get("dir", "."))
        out_path = out_dir / f"top_{max(1, int(args.top_k))}_borrowings_explained_from_ready_experiment_v3.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"[explain-ready-v3] experiment={args.experiment}")
    print(f"[explain-ready-v3] checkpoint_dir={checkpoint_dir}")
    print(f"[explain-ready-v3] output={out_path}")


if __name__ == "__main__":
    main()
