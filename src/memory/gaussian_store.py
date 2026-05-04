"""Gaussian-based synthetic memory for continual learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        if self.root_dir is not None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
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
            mean=mean.astype(np.float32, copy=False),
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
        if self.root_dir is not None:
            class_dir = self.root_dir / identity
            gaussian_path = class_dir / "gaussian.npz"
            if gaussian_path.is_file():
                gaussian_path.unlink()
            if class_dir.is_dir():
                try:
                    class_dir.rmdir()
                except OSError:
                    pass

    def save_class(self, identity: str) -> Path:
        """Persist the Gaussian parameters for one identity."""
        if self.root_dir is None:
            raise ValueError("GaussianStore has no root_dir for persistence")
        if identity not in self.gaussians:
            raise KeyError(f"Identity not found: {identity}")

        payload = self.gaussians[identity]
        class_dir = self.root_dir / identity
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / "gaussian.npz"
        np.savez_compressed(
            out_path,
            mean=payload.mean.astype(np.float32, copy=False),
            cov=payload.cov.astype(np.float32, copy=False),
            n_samples=np.array(payload.n_samples, dtype=np.int64),
        )
        return out_path

    def load_class(self, identity: str) -> GaussianParams:
        """Load Gaussian parameters for one identity from disk."""
        if self.root_dir is None:
            raise ValueError("GaussianStore has no root_dir for persistence")

        in_path = self.root_dir / identity / "gaussian.npz"
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing Gaussian file: {in_path}")

        with np.load(in_path, allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            cov = np.asarray(data["cov"], dtype=np.float32)
            n_samples = int(data["n_samples"].tolist())

        if mean.ndim != 1:
            raise ValueError(f"Invalid stored mean shape: {mean.shape}")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mean.shape[0]:
            raise ValueError(f"Invalid stored cov shape: {cov.shape}")

        params = GaussianParams(mean=mean, cov=cov, n_samples=n_samples)
        self.gaussians[identity] = params
        return params

    def save_all(self) -> List[Path]:
        """Persist all stored Gaussian parameters."""
        return [self.save_class(identity) for identity in self.identities()]

    def load_all(self) -> List[str]:
        """Load all Gaussian parameter files from disk."""
        if self.root_dir is None:
            raise ValueError("GaussianStore has no root_dir for persistence")

        loaded: List[str] = []
        for p in sorted(self.root_dir.iterdir()):
            if not p.is_dir():
                continue
            gaussian_path = p / "gaussian.npz"
            if gaussian_path.is_file():
                self.load_class(p.name)
                loaded.append(p.name)
        return loaded