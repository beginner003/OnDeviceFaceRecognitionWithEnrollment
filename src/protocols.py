"""Protocol interfaces for pluggable system strategies."""

from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


class RegistrationStrategy(Protocol):
    """Incremental learning update when a new identity is registered."""

    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear:
        """Return an updated classifier after registration."""
        ...


class ExemplarSelector(Protocol):
    """Select K representative exemplars from N candidate embeddings."""

    def select(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (selected_embeddings, selected_indices)."""
        ...


class RecognitionStrategy(Protocol):
    """Predict identity from a single query embedding."""

    def predict(
        self,
        embedding: np.ndarray,
        store: ExemplarStore,
        classifier: CosineLinear,
    ) -> Tuple[str, float]:
        """
        Return (identity_name, confidence).
        Return ("unknown", conf) if below threshold.
        """
        ...
