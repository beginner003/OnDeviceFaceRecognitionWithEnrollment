"""Gaussian-based synthetic memory for continual learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass(frozen=True)
class GaussianParams:
    """Per-class Gaussian parameters for synthetic replay."""

    mean: np.ndarray
    cov: np.ndarray
    n_samples: int

    def total_bytes(self) -> int:
        """Approximate in-memory bytes for this Gaussian."""
        return int(self.mean.nbytes + self.cov.nbytes)


class GaussianStore:
    """Store per-class Gaussian parameters instead of exemplars."""

    def __init__(self) -> None:
        self.gaussians: Dict[str, GaussianParams] = {}

    def fit_gaussian(self, identity: str, embeddings: np.ndarray) -> None:
        """Fit Gaussian to embeddings for the given identity."""
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Embeddings must be (N, D), got {emb.shape}")
        if emb.shape[0] < 1:
            raise ValueError("Need at least 1 embedding to fit Gaussian")

        mean = np.mean(emb, axis=0)
        if emb.shape[0] == 1:
            cov = np.eye(emb.shape[1], dtype=np.float32) * 1e-6
        else:
            cov = np.cov(emb.T)
            cov += np.eye(emb.shape[1], dtype=np.float32) * 1e-6

        self.gaussians[identity] = GaussianParams(
            mean=mean,
            cov=cov.astype(np.float32, copy=False),
            n_samples=emb.shape[0],
        )

    def sample_synthetic(self, identity: str, n_samples: int) -> np.ndarray:
        """Sample synthetic embeddings from the Gaussian."""
        if identity not in self.gaussians:
            raise ValueError(f"No Gaussian fitted for identity {identity}")
        params = self.gaussians[identity]
        samples = np.random.multivariate_normal(
            params.mean, params.cov, size=n_samples
        ).astype(np.float32)
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return samples / norms

    def identities(self) -> List[str]:
        """List of stored identities."""
        return sorted(self.gaussians.keys())

    def total_bytes(self) -> int:
        """Approximate total in-memory bytes for all stored Gaussians."""
        return int(sum(params.total_bytes() for params in self.gaussians.values()))

    def remove(self, identity: str) -> None:
        """Remove Gaussian for identity."""
        self.gaussians.pop(identity, None)