"""Naive fine-tuning baseline: expand head and train only on the new identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.continual.classifier import CosineLinear


@dataclass
class NaiveFTConfig:
    """Training hyperparameters aligned with IMPLEMENTATION_PLAN §5.5 / §6.2."""

    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 5
    batch_size: int = 32


def incremental_train_naive(
    classifier: CosineLinear,
    new_embeddings: np.ndarray,
    *,
    init_new_class_from_mean: bool = True,
    config: NaiveFTConfig | None = None,
    device: torch.device | str | None = None,
) -> CosineLinear:
    """
    Add one output class (index = previous class count) and fine-tune **only** on
    `new_embeddings` with cross-entropy. No replay; old-class accuracy typically drops
    (catastrophic forgetting baseline per §6.1).

    Parameters
    ----------
    classifier
        Cosine head; moved to `device` if given.
    new_embeddings
        Array of shape (N, 128), float32 recommended (L2-normalised embeddings expected).
    init_new_class_from_mean
        If True, initialise the new weight row from the mean direction of `new_embeddings`.
    config
        SGD / epoch / batch settings.
    device
        Training device; defaults to CPU.
    """
    cfg = config or NaiveFTConfig()
    dev = torch.device(device or "cpu")
    classifier = classifier.to(dev)

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

    classifier.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return classifier
