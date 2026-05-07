"""Unit tests for cosine classifier head, naive fine-tuning, exemplar replay, and LwF."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.continual.classifier import CosineLinear
from src.continual.exemplar_replay import ExemplarReplayConfig, incremental_train_replay
from src.continual.lwf import LwFConfig, incremental_train_lwf
from src.continual.naive_ft import NaiveFTConfig, incremental_train_naive
from src.continual.replay_lwf import ReplayLwFConfig, incremental_train_replay_lwf
from src.continual.synthetic_replay import SyntheticReplayConfig, incremental_train_synthetic_replay
from src.memory.exemplar_store import ExemplarStore


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


def test_incremental_train_naive_first_identity_weights_match_expand_init() -> None:
    """CE with C=1 is degenerate (loss 0, no grad); weights stay at mean-direction init."""
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(8, 128)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = NaiveFTConfig(epochs=15, batch_size=8, lr=0.05, momentum=0.9)
    incremental_train_naive(clf, emb, config=cfg, device="cpu")

    ref = CosineLinear(in_features=128, out_features=0)
    ref.expand(1, init_from_embeddings=emb)
    assert torch.allclose(clf.weight, ref.weight, atol=1e-5)


def test_incremental_train_naive_second_identity_updates_weight_rows() -> None:
    """With C>=2, CE produces non-zero loss and SGD moves classifier rows."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(8, 128)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(8, 128)).astype(np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    cfg = NaiveFTConfig(epochs=10, batch_size=8, lr=0.05, momentum=0.9)
    clf = CosineLinear(in_features=128, out_features=0)
    incremental_train_naive(clf, a, config=cfg, device="cpu")
    w_row0_after_first = clf.weight[0].detach().clone()

    incremental_train_naive(clf, b, config=cfg, device="cpu")
    assert clf.out_features == 2
    assert not torch.allclose(w_row0_after_first, clf.weight[0], atol=1e-5)

    ref_b = CosineLinear(in_features=128, out_features=1)
    ref_b.expand(1, init_from_embeddings=b)
    assert not torch.allclose(ref_b.weight[0], clf.weight[1], atol=1e-5)


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


# ---------------------------------------------------------------------------
# Exemplar replay tests
# ---------------------------------------------------------------------------

def _make_class_embeddings(rng: np.random.Generator, direction_seed: int, n: int = 8) -> np.ndarray:
    """Generate tight cluster of embeddings around a random unit direction."""
    base = np.random.default_rng(direction_seed).normal(size=(128,)).astype(np.float32)
    base /= np.linalg.norm(base) + 1e-12
    noise = rng.normal(size=(n, 128)).astype(np.float32) * 0.05
    return base + noise


def test_replay_first_class_reduces_loss(tmp_path) -> None:
    rng = np.random.default_rng(10)
    emb_a = _make_class_embeddings(rng, direction_seed=0)
    store = ExemplarStore(tmp_path / "exemplars")
    store.upsert_class("alice", emb_a)

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = ExemplarReplayConfig(epochs=15, batch_size=8, lr=0.05)
    incremental_train_replay(clf, store, emb_a, "alice", config=cfg, device="cpu")

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.from_numpy(emb_a.astype(np.float32)))
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(8, dtype=torch.long))
    assert loss.item() < 0.5


def test_replay_second_class_output_shape(tmp_path) -> None:
    rng = np.random.default_rng(20)
    emb_a = _make_class_embeddings(rng, direction_seed=0)
    emb_b = _make_class_embeddings(rng, direction_seed=100)
    store = ExemplarStore(tmp_path / "exemplars")
    cfg = ExemplarReplayConfig(epochs=5, batch_size=8, lr=0.03)

    clf = CosineLinear(in_features=128, out_features=0)
    store.upsert_class("alice", emb_a)
    incremental_train_replay(clf, store, emb_a, "alice", config=cfg, device="cpu")
    assert clf.out_features == 1

    store.upsert_class("bob", emb_b)
    incremental_train_replay(clf, store, emb_b, "bob", config=cfg, device="cpu")
    assert clf.out_features == 2

    logits = clf(torch.from_numpy(emb_b.astype(np.float32)))
    assert logits.shape == (8, 2)


def test_replay_retains_old_class_better_than_naive(tmp_path) -> None:
    """After adding class B, replay should classify class-A samples more accurately than naive."""
    rng = np.random.default_rng(30)
    emb_a = _make_class_embeddings(rng, direction_seed=0, n=10)
    emb_b = _make_class_embeddings(rng, direction_seed=999, n=10)

    # --- Naive path ---
    clf_naive = CosineLinear(in_features=128, out_features=0)
    naive_cfg = NaiveFTConfig(epochs=20, batch_size=10, lr=0.05)
    incremental_train_naive(clf_naive, emb_a, config=naive_cfg, device="cpu")
    incremental_train_naive(clf_naive, emb_b, config=naive_cfg, device="cpu")

    clf_naive.eval()
    with torch.no_grad():
        preds_naive = clf_naive(torch.from_numpy(emb_a.astype(np.float32))).argmax(dim=1)
    naive_acc = (preds_naive == 0).float().mean().item()

    # --- Replay path ---
    clf_replay = CosineLinear(in_features=128, out_features=0)
    replay_cfg = ExemplarReplayConfig(epochs=20, batch_size=10, lr=0.05)
    store = ExemplarStore(tmp_path / "exemplars")
    store.upsert_class("alice", emb_a)
    incremental_train_replay(clf_replay, store, emb_a, "alice", config=replay_cfg, device="cpu")
    store.upsert_class("bob", emb_b)
    incremental_train_replay(clf_replay, store, emb_b, "bob", config=replay_cfg, device="cpu")

    clf_replay.eval()
    with torch.no_grad():
        preds_replay = clf_replay(torch.from_numpy(emb_a.astype(np.float32))).argmax(dim=1)
    replay_acc = (preds_replay == 0).float().mean().item()

    assert replay_acc >= naive_acc, (
        f"Replay old-class acc ({replay_acc:.2f}) should be >= naive ({naive_acc:.2f})"
    )


# ---------------------------------------------------------------------------
# LwF tests
# ---------------------------------------------------------------------------

def test_lwf_first_class_output_shape() -> None:
    rng = np.random.default_rng(40)
    emb_a = _make_class_embeddings(rng, direction_seed=0)

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = LwFConfig(epochs=5, batch_size=8, lr=0.03)
    incremental_train_lwf(clf, emb_a, config=cfg, device="cpu")

    assert clf.out_features == 1
    logits = clf(torch.from_numpy(emb_a.astype(np.float32)))
    assert logits.shape == (8, 1)


def test_lwf_second_class_output_shape() -> None:
    rng = np.random.default_rng(50)
    emb_a = _make_class_embeddings(rng, direction_seed=0)
    emb_b = _make_class_embeddings(rng, direction_seed=100)

    clf = CosineLinear(in_features=128, out_features=0)
    cfg = LwFConfig(epochs=5, batch_size=8, lr=0.03)
    incremental_train_lwf(clf, emb_a, config=cfg, device="cpu")
    incremental_train_lwf(clf, emb_b, config=cfg, device="cpu")

    assert clf.out_features == 2
    logits = clf(torch.from_numpy(emb_b.astype(np.float32)))
    assert logits.shape == (8, 2)


def test_lwf_distillation_changes_old_class_optimization() -> None:
    rng = np.random.default_rng(60)
    emb_a = _make_class_embeddings(rng, direction_seed=0, n=10)
    emb_b = _make_class_embeddings(rng, direction_seed=100, n=10)
    emb_c = _make_class_embeddings(rng, direction_seed=999, n=10)

    base = CosineLinear(in_features=128, out_features=0)
    warmup_cfg = LwFConfig(epochs=10, batch_size=10, lr=0.05, distill_weight=0.0)
    incremental_train_lwf(base, emb_a, config=warmup_cfg, device="cpu")
    incremental_train_lwf(base, emb_b, config=warmup_cfg, device="cpu")
    assert base.out_features == 2

    no_distill = copy.deepcopy(base)
    with_distill = copy.deepcopy(base)

    torch.manual_seed(0)
    incremental_train_lwf(
        no_distill,
        emb_c,
        config=LwFConfig(epochs=20, batch_size=10, lr=0.05, distill_weight=0.0),
        device="cpu",
    )
    torch.manual_seed(0)
    incremental_train_lwf(
        with_distill,
        emb_c,
        config=LwFConfig(
            epochs=20,
            batch_size=10,
            lr=0.05,
            temperature=2.0,
            distill_weight=10.0,
        ),
        device="cpu",
    )

    assert with_distill.out_features == 3
    assert no_distill.out_features == 3
    assert not torch.allclose(no_distill.weight[:2], with_distill.weight[:2], atol=1e-5)


def test_replay_lwf_second_class_output_shape(tmp_path) -> None:
    rng = np.random.default_rng(70)
    emb_a = _make_class_embeddings(rng, direction_seed=0)
    emb_b = _make_class_embeddings(rng, direction_seed=100)

    clf = CosineLinear(in_features=128, out_features=0)
    store = ExemplarStore(tmp_path / "exemplars")
    cfg = ReplayLwFConfig(epochs=5, batch_size=8, lr=0.03, replay_per_identity=5)

    store.upsert_class("alice", emb_a)
    incremental_train_replay_lwf(clf, store, emb_a, "alice", config=cfg, device="cpu")
    assert clf.out_features == 1
    setattr(clf, "_class_names", ["alice"])

    store.upsert_class("bob", emb_b)
    incremental_train_replay_lwf(clf, store, emb_b, "bob", config=cfg, device="cpu")

    assert clf.out_features == 2
    logits = clf(torch.from_numpy(emb_b.astype(np.float32)))
    assert logits.shape == (8, 2)


def test_replay_lwf_retains_old_class_better_than_no_replay_lwf(tmp_path) -> None:
    rng = np.random.default_rng(80)
    emb_a = _make_class_embeddings(rng, direction_seed=1, n=10)
    emb_b = _make_class_embeddings(rng, direction_seed=999, n=10)

    no_replay = CosineLinear(in_features=128, out_features=0)
    lwf_cfg = LwFConfig(epochs=20, batch_size=10, lr=0.05, temperature=2.0, distill_weight=1.0)
    incremental_train_lwf(no_replay, emb_a, config=lwf_cfg, device="cpu")
    setattr(no_replay, "_class_names", ["alice"])
    incremental_train_lwf(no_replay, emb_b, config=lwf_cfg, device="cpu")
    no_replay.eval()
    with torch.no_grad():
        acc_no_replay = (
            no_replay(torch.from_numpy(emb_a.astype(np.float32))).argmax(dim=1) == 0
        ).float().mean().item()

    replay = CosineLinear(in_features=128, out_features=0)
    store = ExemplarStore(tmp_path / "replay_lwf_exemplars")
    replay_cfg = ReplayLwFConfig(
        epochs=20,
        batch_size=10,
        lr=0.05,
        temperature=2.0,
        distill_weight=1.0,
        replay_per_identity=5,
    )
    store.upsert_class("alice", emb_a)
    incremental_train_replay_lwf(replay, store, emb_a, "alice", config=replay_cfg, device="cpu")
    setattr(replay, "_class_names", ["alice"])
    store.upsert_class("bob", emb_b)
    incremental_train_replay_lwf(replay, store, emb_b, "bob", config=replay_cfg, device="cpu")
    replay.eval()
    with torch.no_grad():
        acc_replay_lwf = (
            replay(torch.from_numpy(emb_a.astype(np.float32))).argmax(dim=1) == 0
        ).float().mean().item()

    assert acc_replay_lwf >= acc_no_replay


def test_replay_lwf_validates_replay_per_identity(tmp_path) -> None:
    rng = np.random.default_rng(90)
    emb_a = _make_class_embeddings(rng, direction_seed=0, n=8)
    clf = CosineLinear(in_features=128, out_features=0)
    store = ExemplarStore(tmp_path / "exemplars")
    store.upsert_class("alice", emb_a)
    with pytest.raises(ValueError, match="replay_per_identity must be > 0"):
        incremental_train_replay_lwf(
            clf,
            store,
            emb_a,
            "alice",
            config=ReplayLwFConfig(replay_per_identity=0),
            device="cpu",
        )


def test_synthetic_replay_uses_classifier_order_for_old_classes(tmp_path) -> None:
    from src.memory.gaussian_store import GaussianStore

    rng = np.random.default_rng(60)
    emb_a = _make_class_embeddings(rng, direction_seed=0, n=50)
    emb_b = _make_class_embeddings(rng, direction_seed=100, n=50)
    emb_c = _make_class_embeddings(rng, direction_seed=999, n=50)

    base = CosineLinear(in_features=128, out_features=0)
    warmup_cfg = SyntheticReplayConfig()
    store = GaussianStore()
    incremental_train_synthetic_replay(base, store, emb_a, "alice", config=warmup_cfg, device="cpu")
    incremental_train_synthetic_replay(base, store, emb_b, "bob", config=warmup_cfg, device="cpu")
    assert base.out_features == 2
    assert getattr(base, "_class_names", ["alice", "bob"]) == ["alice", "bob"]

    torch.manual_seed(0)
    incremental_train_synthetic_replay(
        base, store, emb_c, "carol", config=warmup_cfg, device="cpu"
    )
    assert base.out_features == 3
    assert getattr(base, "_class_names", ["alice", "bob", "carol"]) == ["alice", "bob", "carol"]
