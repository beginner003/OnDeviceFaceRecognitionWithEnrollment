"""Synthetic replay strategy: generate synthetic embeddings via per-class Gaussians."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore
from src.memory.gaussian_store import GaussianStore


@dataclass
class SyntheticReplayConfig:
    """Training hyperparameters for synthetic replay."""

    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 40
    batch_size: int = 32
    synthetic_samples_per_class: int = 50  # How many synthetic samples to generate per replay class


def incremental_train_synthetic_replay(
    classifier: CosineLinear,
    store: GaussianStore,
    new_embeddings: np.ndarray,
    identity: str,
    *,
    init_new_class_from_mean: bool = True,
    config: SyntheticReplayConfig | None = None,
    device: torch.device | str | None = None,
) -> CosineLinear:
    """
    Add one class and fine-tune on new embeddings + synthetic samples from stored Gaussians.

    This is the synthetic replay variant: instead of storing exemplars, we fit Gaussians
    and sample synthetic embeddings for replay (§7 Strategy 3).
    """
    cfg = config or SyntheticReplayConfig()
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

    all_x, all_y = [], []

    # Synthetic replay: generate samples from stored Gaussians for old classes in
    # the classifier's current class index order.
    for cls_idx, ident in enumerate(store.identities()):
        if ident == identity:
            continue
        synthetic = store.sample_synthetic(ident, cfg.synthetic_samples_per_class)
        all_x.append(synthetic)
        all_y.append(np.full(synthetic.shape[0], cls_idx, dtype=np.int64))

    # New class data
    all_x.append(emb)
    all_y.append(np.full(emb.shape[0], new_class_idx, dtype=np.int64))

    x_t = torch.from_numpy(np.concatenate(all_x)).to(device=dev, dtype=torch.float32)
    y_t = torch.from_numpy(np.concatenate(all_y)).to(device=dev)

    bs = max(1, min(cfg.batch_size, x_t.shape[0]))
    loader = DataLoader(
        TensorDataset(x_t, y_t), batch_size=bs, shuffle=True, drop_last=False,
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


@dataclass
class SyntheticReplayStrategy:
    """Protocol-compatible wrapper for Gaussian synthetic replay."""

    config: SyntheticReplayConfig = field(default_factory=SyntheticReplayConfig)
    device: torch.device | str | None = None
    init_new_class_from_mean: bool = True

    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,   # not used, just comply with the protocol
        gaussian_store: GaussianStore | None,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear:
        if gaussian_store is None:
            raise ValueError("SyntheticReplayStrategy requires gaussian_store")
        _ = store

        return incremental_train_synthetic_replay(
            classifier=classifier,
            store=gaussian_store,
            new_embeddings=new_embeddings,
            identity=identity,
            init_new_class_from_mean=self.init_new_class_from_mean,
            config=self.config,
            device=self.device,
        )