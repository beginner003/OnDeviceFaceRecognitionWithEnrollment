"""Unit tests for cosine classifier head and naive fine-tuning baseline."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.continual.classifier import CosineLinear
from src.continual.naive_ft import NaiveFTConfig, incremental_train_naive


def test_cosine_linear_expand_and_forward_shape() -> None:
    m = CosineLinear(in_features=128, out_features=0)
    assert m.out_features == 0
    with pytest.raises(RuntimeError, match="no classes"):
        m(torch.zeros(2, 128))

    m.expand(2)
    assert m.out_features == 2
    logits = m(torch.randn(4, 128))
    assert logits.shape == (4, 2)


def test_cosine_linear_expand_init_aligns_with_mean_direction() -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5, 128)).astype(np.float32)
    mean_dir = emb.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)

    m = CosineLinear(in_features=128, out_features=0)
    m.expand(1, init_from_embeddings=emb)
    w = m.weight.detach().numpy()[0]
    w = w / (np.linalg.norm(w) + 1e-12)
    cos = float(np.dot(mean_dir, w))
    assert cos > 0.999


def test_incremental_train_naive_first_class_reduces_loss() -> None:
    rng = np.random.default_rng(1)
    # Nearly identical vectors around one direction -> easy CE
    base = rng.normal(size=(128,)).astype(np.float32)
    base = base / (np.linalg.norm(base) + 1e-12)
    noise = rng.normal(size=(8, 128)).astype(np.float32) * 0.05
    emb = base + noise

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = NaiveFTConfig(epochs=15, batch_size=8, lr=0.05, momentum=0.9)
    incremental_train_naive(clf, emb, config=cfg, device="cpu")

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.from_numpy(emb))
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(8, dtype=torch.long))
    assert loss.item() < 0.5


def test_incremental_train_naive_second_step_has_two_classes() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 128)).astype(np.float32)
    b = rng.normal(size=(4, 128)).astype(np.float32)

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = NaiveFTConfig(epochs=3, batch_size=4, lr=0.02, momentum=0.9)
    incremental_train_naive(clf, a, config=cfg, device="cpu")
    assert clf.out_features == 1
    incremental_train_naive(clf, b, config=cfg, device="cpu")
    assert clf.out_features == 2
    logits = clf(torch.from_numpy(b))
    assert logits.shape == (4, 2)
