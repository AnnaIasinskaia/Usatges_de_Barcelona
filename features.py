"""
Step 3-4: Feature extraction (TF-IDF), candidate selection, Tesserae scoring,
Soft Cosine.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def build_tfidf_matrix(
    corpus: List[List[str]],
    ngram_range: Tuple[int, int] = (1, 3),
    max_df: float = 0.5,
    min_df: int = 2,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    ngram_docs = []
    for doc in corpus:
        ngrams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(doc) - n + 1):
                ngrams.append(" ".join(doc[i:i+n]))
        ngram_docs.append(ngrams)

    n_docs = len(ngram_docs)
    if n_docs == 0:
        return np.zeros((0, 1)), ["_empty_"], {"_empty_": 0}

    df = Counter()
    for doc in ngram_docs:
        for term in set(doc):
            df[term] += 1

    max_df_abs = int(max_df * n_docs) if isinstance(max_df, float) else max_df
    vocab = sorted([
        t for t, f in df.items()
        if min_df <= f <= max_df_abs
    ])
    term2idx = {t: i for i, t in enumerate(vocab)}

    if not vocab:
        return np.zeros((n_docs, 1)), ["_empty_"], {"_empty_": 0}

    n_terms = len(vocab)
    tfidf = np.zeros((n_docs, n_terms), dtype=np.float32)

    for doc_i, doc in enumerate(ngram_docs):
        tf = Counter(doc)
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            if term in term2idx:
                idx = term2idx[term]
                tf_val = count / max_tf
                idf_val = math.log(n_docs / max(df[term], 1))
                tfidf[doc_i, idx] = tf_val * idf_val

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf = tfidf / norms

    return tfidf, vocab, term2idx


def select_tfidf_candidates(
    tfidf_left: np.ndarray,
    tfidf_right: np.ndarray,
    left_ids: List[str],
    right_ids: List[str],
    threshold: float,
    top_k_per_left: Optional[int],
    progress_every: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Build candidate pairs from two normalized TF-IDF matrices.

    If ``top_k_per_left`` is None, returns all pairs above ``threshold``.
    Otherwise, keeps at most top-k candidates for each left row, still subject
    to the same threshold.
    """
    sim = tfidf_left @ tfidf_right.T

    pairs: List[Tuple[str, str, float]] = []
    if top_k_per_left is None:
        rows, cols = np.where(sim >= threshold)
        for i, j in zip(rows, cols):
            pairs.append((left_ids[int(i)], right_ids[int(j)], float(sim[int(i), int(j)])))
        return pairs

    k = int(top_k_per_left)
    total_rows = sim.shape[0]
    for i in range(total_rows):
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

        if (
            progress_every is not None
            and progress_callback is not None
            and (i + 1) % max(1, int(progress_every)) == 0
        ):
            progress_callback(f"  Candidate selection progress: {i + 1}/{total_rows}")

    return pairs


def tesserae_score(
    lemmas_a: List[str],
    lemmas_b: List[str],
    idf: Dict[str, float],
) -> float:
    if not lemmas_a or not lemmas_b:
        return 0.0
    set_a = set(lemmas_a)
    set_b = set(lemmas_b)
    shared = set_a & set_b

    if len(shared) < 2:
        return 0.0

    score = sum(idf.get(w, 0.0) for w in shared)
    norm = math.sqrt(len(set_a) * len(set_b))
    return score / norm if norm > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    if not isinstance(s1, str):
        s1 = str(s1)
    if not isinstance(s2, str):
        s2 = str(s2)
    try:
        from Levenshtein import distance as lev
        return lev(s1, s2)
    except ImportError:
        pass
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            current_row.append(min(
                current_row[j] + 1,
                previous_row[j + 1] + 1,
                previous_row[j] + cost,
            ))
        previous_row = current_row
    return previous_row[-1]


def soft_cosine_similarity(
    lemmas_a: List[str],
    lemmas_b: List[str],
    lev_threshold: int = 2,
    max_terms: int = 500,
) -> float:
    """
    Soft cosine similarity with a safety cap on the number of unique terms
    to prevent O(n^2) memory explosions.
    """
    if not lemmas_a or not lemmas_b:
        return 0.0

    all_terms = sorted(set(lemmas_a) | set(lemmas_b))

    # Safety: if too many unique terms, fall back to hard cosine
    if len(all_terms) > max_terms:
        # Simple hard cosine as fallback
        set_a = Counter(lemmas_a)
        set_b = Counter(lemmas_b)
        common = set(set_a.keys()) & set(set_b.keys())
        if not common:
            return 0.0
        dot = sum(set_a[t] * set_b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in set_a.values()))
        norm_b = math.sqrt(sum(v * v for v in set_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    term2idx = {t: i for i, t in enumerate(all_terms)}
    n = len(all_terms)

    vec_a = np.zeros(n, dtype=np.float32)
    vec_b = np.zeros(n, dtype=np.float32)
    for w in lemmas_a:
        vec_a[term2idx[w]] += 1
    for w in lemmas_b:
        vec_b[term2idx[w]] += 1

    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            lev = levenshtein_distance(all_terms[i], all_terms[j])
            max_len = max(len(all_terms[i]), len(all_terms[j]))
            if lev <= lev_threshold and max_len > 0:
                sim = 1.0 - lev / max_len
                S[i, j] = sim
                S[j, i] = sim

    numerator = float(vec_a @ S @ vec_b)
    denom_a = math.sqrt(float(vec_a @ S @ vec_a))
    denom_b = math.sqrt(float(vec_b @ S @ vec_b))

    if denom_a == 0 or denom_b == 0:
        return 0.0

    return numerator / (denom_a * denom_b)


def compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    n_docs = len(corpus)
    if n_docs == 0:
        return {}
    df = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    return {t: math.log(n_docs / max(f, 1)) for t, f in df.items()}
