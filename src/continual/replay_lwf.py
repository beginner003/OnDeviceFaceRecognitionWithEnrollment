"""Replay + LwF strategy: exemplar replay with old-logit distillation."""

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
class ReplayLwFConfig:
    """Training hyperparameters for replay-assisted LwF distillation."""

    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 10
    batch_size: int = 32
    temperature: float = 2.0
    distill_weight: float = 1.0
    replay_per_identity: int = 5


def _resolve_old_identity_order(class_names: list[str] | None, store: ExemplarStore) -> list[str]:
    if isinstance(class_names, list) and class_names:
        return [str(name) for name in class_names]
    return [str(name) for name in store.identities()]


def incremental_train_replay_lwf(
    classifier: CosineLinear,
    store: ExemplarStore,
    new_embeddings: np.ndarray,
    identity: str,
    *,
    init_new_class_from_mean: bool = True,
    config: ReplayLwFConfig | None = None,
    device: torch.device | str | None = None,
) -> CosineLinear:
    """
    Add one class and train with old-exemplar replay plus LwF distillation.

    Training dataset:
    - all new-class embeddings;
    - up to `replay_per_identity` stored exemplars for each previously registered identity.
    """
    cfg = config or ReplayLwFConfig()
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if cfg.replay_per_identity <= 0:
        raise ValueError("replay_per_identity must be > 0")

    dev = torch.device(device or "cpu")
    classifier = classifier.to(dev)
    n_old = classifier.out_features
    old_class_names = getattr(classifier, "_class_names", None)
    if isinstance(old_class_names, list):
        old_class_names = [str(name) for name in old_class_names]
    else:
        old_class_names = None

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

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    replay_k = int(cfg.replay_per_identity)
    available_identities = set(store.identities())
    old_order = _resolve_old_identity_order(class_names=old_class_names, store=store)
    for cls_idx, old_identity in enumerate(old_order):
        if old_identity == identity:
            continue
        if old_identity not in available_identities:
            continue
        old_emb = store.get(old_identity).embeddings.astype(np.float32)
        take_k = min(replay_k, old_emb.shape[0])
        all_x.append(old_emb[:take_k])
        all_y.append(np.full(take_k, cls_idx, dtype=np.int64))

    all_x.append(emb)
    all_y.append(np.full(emb.shape[0], new_class_idx, dtype=np.int64))

    x_t = torch.from_numpy(np.concatenate(all_x, axis=0)).to(device=dev, dtype=torch.float32)
    y_t = torch.from_numpy(np.concatenate(all_y, axis=0)).to(device=dev)
    bs = max(1, min(int(cfg.batch_size), int(x_t.shape[0])))
    loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=bs,
        shuffle=True,
        drop_last=False,
    )

    optimizer = SGD(classifier.parameters(), lr=float(cfg.lr), momentum=float(cfg.momentum))
    criterion = nn.CrossEntropyLoss()
    temperature = float(cfg.temperature)

    classifier.train()
    for _ in range(int(cfg.epochs)):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            student_logits = classifier(xb)
            loss = criterion(student_logits, yb)

            if teacher is not None and n_old > 0 and float(cfg.distill_weight) != 0.0:
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
class ReplayLwFStrategy:
    """Protocol-compatible wrapper for replay-assisted LwF distillation."""

    config: ReplayLwFConfig = field(default_factory=ReplayLwFConfig)
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
        _ = gaussian_store
        return incremental_train_replay_lwf(
            classifier=classifier,
            store=store,
            new_embeddings=new_embeddings,
            identity=identity,
            init_new_class_from_mean=self.init_new_class_from_mean,
            config=self.config,
            device=self.device,
        )


__all__ = [
    "ReplayLwFConfig",
    "ReplayLwFStrategy",
    "incremental_train_replay_lwf",
]
