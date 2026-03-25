"""Nearest Class Mean recognizer using exemplar prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


def _l2_normalize(v: np.ndarray, *, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(v, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return v / denom


@dataclass
class NCMRecognizer:
    """Cosine similarity against class prototypes stored in ExemplarStore."""

    confidence_threshold: float = 0.5

    def predict(
        self,
        embedding: np.ndarray,
        store: ExemplarStore,
        classifier: CosineLinear,
    ) -> Tuple[str, float]:
        _ = classifier
        ids = store.identities()
        if not ids:
            return "unknown", 0.0

        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] == 0:
            return "unknown", 0.0

        query = _l2_normalize(query)
        protos = np.stack([store.get(identity).prototype.astype(np.float32) for identity in ids], axis=0)
        protos = _l2_normalize(protos, axis=1)
        sims = protos @ query
        best_idx = int(np.argmax(sims))
        best_conf = float(sims[best_idx])
        if best_conf < self.confidence_threshold:
            return "unknown", best_conf
        return ids[best_idx], best_conf
