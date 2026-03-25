"""Classifier-based recognizer using CosineLinear logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


@dataclass
class ClassifierRecognizer:
    """Use classifier softmax confidence for prediction."""

    confidence_threshold: float = 0.5
    device: str = "cpu"

    def predict(
        self,
        embedding: np.ndarray,
        store: ExemplarStore,
        classifier: CosineLinear,
    ) -> Tuple[str, float]:
        _ = store
        if classifier.out_features == 0:
            return "unknown", 0.0

        x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != classifier.in_features:
            raise ValueError(
                f"Expected embedding dim {classifier.in_features}, got {x.shape[1]}"
            )

        class_names = getattr(classifier, "_class_names", None)
        if not isinstance(class_names, list):
            class_names = [f"class_{i}" for i in range(classifier.out_features)]

        classifier = classifier.to(self.device)
        classifier.eval()
        with torch.no_grad():
            logits = classifier(torch.from_numpy(x).to(self.device))
            probs = torch.softmax(logits, dim=1)
            conf_t, idx_t = torch.max(probs, dim=1)

        conf = float(conf_t.item())
        idx = int(idx_t.item())
        if conf < self.confidence_threshold:
            return "unknown", conf
        if idx >= len(class_names):
            return "unknown", conf
        return str(class_names[idx]), conf
