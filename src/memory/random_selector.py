"""Random exemplar selection baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RandomSelector:
    """Uniform random selector for exemplar baseline experiments."""

    seed: int | None = None

    def select(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D embeddings (N, D), got shape {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("Expected at least one embedding")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        n = x.shape[0]
        target_k = min(int(k), n)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=target_k, replace=False)
        idx = np.asarray(idx, dtype=np.int64)
        return x[idx], idx
