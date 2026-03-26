"""Exemplar replay strategy: balanced training on new + stored exemplars (iCaRL-style)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore


@dataclass
class ExemplarReplayConfig:
    """Training hyperparameters aligned with IMPLEMENTATION_PLAN §6.2."""

    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 10
    batch_size: int = 32
    exemplar_k: int = 30


def incremental_train_replay(
    classifier: CosineLinear,
    store: ExemplarStore,
    new_embeddings: np.ndarray,
    identity: str,
    *,
    init_new_class_from_mean: bool = True,
    config: ExemplarReplayConfig | None = None,
    device: torch.device | str | None = None,
) -> CosineLinear:
    """
    Add one class and fine-tune on a balanced mix of new embeddings + all stored exemplars.

    Unlike naive FT, this replays old-class exemplars each epoch so the classifier
    retains accuracy on previously registered identities (§6.2).
    """
    cfg = config or ExemplarReplayConfig()
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

    # Replay: gather stored exemplars for all previously registered classes
    for cls_idx, ident in enumerate(store.identities()):
        stored = store.get(ident).embeddings.astype(np.float32)
        # sample from stored exemplars: n per class
        max_exemplars = cfg.exemplar_k
        if stored.shape[0] > max_exemplars:
            idxs = np.random.choice(stored.shape[0], max_exemplars, replace=False)
            stored = stored[idxs]
        print(f"Replaying {stored.shape[0]} exemplars for identity '{ident}' (class {cls_idx})")
        all_x.append(stored)
        all_y.append(np.full(stored.shape[0], cls_idx, dtype=np.int64))

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
class ExemplarReplayStrategy:
    """Protocol-compatible wrapper for iCaRL-style exemplar replay."""

    config: ExemplarReplayConfig = field(default_factory=ExemplarReplayConfig)
    device: torch.device | str | None = None
    init_new_class_from_mean: bool = True

    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear:
        return incremental_train_replay(
            classifier=classifier,
            store=store,
            new_embeddings=new_embeddings,
            identity=identity,
            init_new_class_from_mean=self.init_new_class_from_mean,
            config=self.config,
            device=self.device,
        )
