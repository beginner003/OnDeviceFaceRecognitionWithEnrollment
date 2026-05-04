"""Learning without Forgetting (LwF) strategy without exemplar replay."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore
from src.memory.gaussian_store import GaussianStore


@dataclass
class LwFConfig:
    """Training hyperparameters for no-replay LwF distillation."""

    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 10
    batch_size: int = 32
    temperature: float = 2.0
    distill_weight: float = 1.0


def incremental_train_lwf(
    classifier: CosineLinear,
    new_embeddings: np.ndarray,
    *,
    init_new_class_from_mean: bool = True,
    config: LwFConfig | None = None,
    device: torch.device | str | None = None,
) -> CosineLinear:
    """
    Add one class and train on new embeddings with LwF distillation from a frozen teacher.

    This variant intentionally does not replay old exemplars. The old classifier is
    copied before expansion and used to preserve its old-class posterior on new data.
    """
    cfg = config or LwFConfig()
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")

    dev = torch.device(device or "cpu")
    classifier = classifier.to(dev)
    n_old = classifier.out_features

    teacher: CosineLinear | None = None
    if n_old > 0:
        teacher = copy.deepcopy(classifier).to(dev)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    emb = np.asarray(new_embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != classifier.in_features:
        raise ValueError(
            f"new_embeddings must be (N, {classifier.in_features}), got {emb.shape}"
        )
    if emb.shape[0] == 0:
        raise ValueError("new_embeddings must contain at least one vector")

    init_arg = emb if init_new_class_from_mean else None
    classifier.expand(1, init_from_embeddings=init_arg)
    new_class_idx = classifier.out_features - 1

    x_t = torch.from_numpy(emb).to(device=dev, dtype=torch.float32)
    y_t = torch.full((emb.shape[0],), new_class_idx, dtype=torch.long, device=dev)

    bs = max(1, min(cfg.batch_size, emb.shape[0]))
    loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=bs,
        shuffle=True,
        drop_last=False,
    )

    optimizer = SGD(classifier.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()
    temperature = float(cfg.temperature)

    classifier.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            student_logits = classifier(xb)
            loss = criterion(student_logits, yb)

            if teacher is not None and n_old > 0 and cfg.distill_weight != 0:
                with torch.no_grad():
                    teacher_logits = teacher(xb)
                student_old = student_logits[:, :n_old]
                distill_loss = F.kl_div(
                    F.log_softmax(student_old / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = loss + float(cfg.distill_weight) * distill_loss

            loss.backward()
            optimizer.step()

    return classifier


@dataclass
class LwFStrategy:
    """Protocol-compatible wrapper for no-replay LwF distillation."""

    config: LwFConfig = field(default_factory=LwFConfig)
    device: torch.device | str | None = None
    init_new_class_from_mean: bool = True

    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,
        gaussian_store: GaussianStore | None,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear:
        # `store`, `gaussian_store`, and `identity` are kept in signature for protocol consistency.
        _ = (store, gaussian_store, identity)
        return incremental_train_lwf(
            classifier=classifier,
            new_embeddings=new_embeddings,
            init_new_class_from_mean=self.init_new_class_from_mean,
            config=self.config,
            device=self.device,
        )
