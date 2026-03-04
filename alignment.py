"""
Step 6: Smith-Waterman local text alignment.
"""
import numpy as np
from typing import List, Tuple, Optional
from features import levenshtein_distance


def smith_waterman(
    seq_a: List[str],
    seq_b: List[str],
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -1,
    lev_bonus_threshold: int = 2,
) -> Tuple[List[str], List[str], float]:
    """
    Semantic Smith-Waterman alignment for lemma sequences.

    Mismatches are softened if Levenshtein distance between lemmas
    is below lev_bonus_threshold (partial morphological match).

    Returns:
        aligned_a: aligned tokens from seq_a (with '-' for gaps)
        aligned_b: aligned tokens from seq_b (with '-' for gaps)
        score: alignment score
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return [], [], 0.0

    # Scoring matrix
    H = np.zeros((m + 1, n + 1))
    traceback = np.zeros((m + 1, n + 1), dtype=int)
    # 0=stop, 1=diag, 2=up, 3=left

    max_score = 0
    max_pos = (0, 0)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch score
            if seq_a[i-1] == seq_b[j-1]:
                diag = H[i-1, j-1] + match_score
            else:
                lev = levenshtein_distance(seq_a[i-1], seq_b[j-1])
                if lev <= lev_bonus_threshold:
                    # Partial match: reduced penalty
                    max_len = max(len(seq_a[i-1]), len(seq_b[j-1]))
                    bonus = match_score * (1.0 - lev / max_len) if max_len > 0 else 0
                    diag = H[i-1, j-1] + bonus
                else:
                    diag = H[i-1, j-1] + mismatch_score

            up = H[i-1, j] + gap_penalty
            left = H[i, j-1] + gap_penalty

            best = max(0, diag, up, left)
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

    # Traceback
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
