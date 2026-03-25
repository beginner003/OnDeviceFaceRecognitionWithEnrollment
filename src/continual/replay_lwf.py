"""Replay + LwF strategy (high-level stub for orchestrator wiring)."""

from __future__ import annotations

import numpy as np

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


class ReplayLwFStrategy:
    """
    Strategy hook for replay plus distillation.

    TODO: Implement replay + LwF training from IMPLEMENTATION_PLAN §6.3.
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
            "TODO(§6.3): ReplayLwFStrategy.update is not implemented yet."
        )
