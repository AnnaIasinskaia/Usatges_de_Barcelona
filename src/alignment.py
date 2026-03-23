"""
Step 6: Smith-Waterman local text alignment.
"""
import numpy as np
import logging
from typing import List, Tuple
from src.features import levenshtein_distance

log = logging.getLogger(__name__)


def smith_waterman(
    seq_a: List[str],
    seq_b: List[str],
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -1,
    lev_bonus_threshold: int = 2,
    max_seq_len: int = 300,
) -> Tuple[List[str], List[str], float]:
    """
    Semantic Smith-Waterman alignment for lemma sequences.
    Sequences are truncated to max_seq_len to prevent memory issues.
    """
    if not seq_a or not seq_b:
        return [], [], 0.0

    # Ensure all elements are strings
    seq_a = [str(x) if not isinstance(x, str) else x for x in seq_a if x]
    seq_b = [str(x) if not isinstance(x, str) else x for x in seq_b if x]

    # Safety truncation
    seq_a = seq_a[:max_seq_len]
    seq_b = seq_b[:max_seq_len]

    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return [], [], 0.0

    try:
        H = np.zeros((m + 1, n + 1), dtype=np.float32)
        traceback = np.zeros((m + 1, n + 1), dtype=np.int8)
    except MemoryError:
        log.warning(f"Smith-Waterman: MemoryError for sequences of length {m} x {n}")
        return [], [], 0.0

    max_score = 0.0
    max_pos = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a_tok = seq_a[i-1]
            b_tok = seq_b[j-1]

            if a_tok == b_tok:
                diag = H[i-1, j-1] + match_score
            else:
                try:
                    lev = levenshtein_distance(a_tok, b_tok)
                except Exception:
                    lev = lev_bonus_threshold + 1

                if lev <= lev_bonus_threshold:
                    max_len = max(len(a_tok), len(b_tok))
                    bonus = match_score * (1.0 - lev / max_len) if max_len > 0 else 0
                    diag = H[i-1, j-1] + bonus
                else:
                    diag = H[i-1, j-1] + mismatch_score

            up = H[i-1, j] + gap_penalty
            left = H[i, j-1] + gap_penalty

            best = max(0.0, diag, up, left)
            H[i, j] = best

            if best == 0:
                traceback[i, j] = 0
            elif best == diag:
                traceback[i, j] = 1
            elif best == up:
                traceback[i, j] = 2
            else:
                traceback[i, j] = 3

            if best > max_score:
                max_score = best
                max_pos = (i, j)

    aligned_a, aligned_b = [], []
    i, j = max_pos
    while i > 0 and j > 0 and H[i, j] > 0:
        if traceback[i, j] == 1:
            aligned_a.append(seq_a[i-1])
            aligned_b.append(seq_b[j-1])
            i -= 1
            j -= 1
        elif traceback[i, j] == 2:
            aligned_a.append(seq_a[i-1])
            aligned_b.append("-")
            i -= 1
        elif traceback[i, j] == 3:
            aligned_a.append("-")
            aligned_b.append(seq_b[j-1])
            j -= 1
        else:
            break

    aligned_a.reverse()
    aligned_b.reverse()

    return aligned_a, aligned_b, float(max_score)
