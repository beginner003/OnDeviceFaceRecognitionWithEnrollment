"""Exemplar replay strategy (high-level stub for orchestrator wiring)."""

from __future__ import annotations

import numpy as np

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


class ExemplarReplayStrategy:
    """
    Strategy hook for iCaRL-style replay.

    TODO: Implement balanced replay training loop from IMPLEMENTATION_PLAN §6.2.
    """

    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear:
        _ = (classifier, store, new_embeddings, identity)
        raise NotImplementedError(
            "TODO(§6.2): ExemplarReplayStrategy.update is not implemented yet."
        )
