"""Herding-based exemplar selection (iCaRL-style)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings (N, D), got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Expected at least one embedding")
    return x


def select_exemplar_indices(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Select exemplar indices that best approximate the class mean.

    The objective is greedy: each step picks one not-yet-selected embedding that
    minimises the distance between current selected-mean and the full class mean.
    """
    x = _validate_embeddings(embeddings)
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    n = x.shape[0]
    target_k = min(int(k), n)
    class_mean = x.mean(axis=0)

    selected: List[int] = []
    running_sum = np.zeros_like(class_mean, dtype=np.float32)
    all_indices = np.arange(n)

    for step in range(target_k):
        available = np.setdiff1d(all_indices, np.asarray(selected, dtype=np.int64), assume_unique=False)
        # Score each candidate by ||mean(selected + candidate) - class_mean||_2.
        candidate_sums = running_sum[None, :] + x[available]
        candidate_means = candidate_sums / float(step + 1)
        distances = np.linalg.norm(candidate_means - class_mean[None, :], axis=1)
        best_local = int(np.argmin(distances))
        best_idx = int(available[best_local])
        selected.append(best_idx)
        running_sum += x[best_idx]

    return np.asarray(selected, dtype=np.int64)


def herding_select(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (selected_embeddings, selected_indices)."""
    idx = select_exemplar_indices(embeddings=embeddings, k=k)
    x = _validate_embeddings(embeddings)
    return x[idx], idx

