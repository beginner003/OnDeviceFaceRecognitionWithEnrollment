"""Unit tests for FaceRecognitionSystem orchestrator (IMPLEMENTATION_PLAN §5.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.system import FaceRecognitionSystem, SystemConfig


def _cluster(center: np.ndarray, n: int = 12, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = center[None, :] + rng.normal(scale=noise, size=(n, center.shape[0])).astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x.astype(np.float32)


def test_system_register_and_recognize_ncm(tmp_path: Path) -> None:
    cfg = SystemConfig(
        registration="naive",
        exemplar_selection="herding",
        recognition="ncm",
        exemplar_k=8,
        confidence_threshold=0.3,
    )
    system = FaceRecognitionSystem.from_config(cfg, workspace=tmp_path)
    center = np.ones((128,), dtype=np.float32)
    emb = _cluster(center, n=14, seed=11)

    result = system.register("alice", emb)
    assert result.identity == "alice"
    assert result.total_identities == 1
    assert system.identities() == ["alice"]

    q = _cluster(center, n=1, seed=99)[0]
    pred, conf = system.recognize(q)
    assert pred == "alice"
    assert conf >= cfg.confidence_threshold


def test_system_save_load_roundtrip_classifier_recognition(tmp_path: Path) -> None:
    cfg = SystemConfig(
        registration="naive",
        exemplar_selection="random",
        recognition="classifier",
        exemplar_k=6,
        confidence_threshold=0.2,
    )
    system = FaceRecognitionSystem.from_config(cfg, workspace=tmp_path)
    c1 = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
    c2 = np.eye(1, 128, 1, dtype=np.float32).reshape(-1)
    system.register("alice", _cluster(c1, n=10, seed=1))
    system.register("bob", _cluster(c2, n=10, seed=2))

    loaded = FaceRecognitionSystem.from_config(cfg, workspace=tmp_path)
    assert loaded.identities() == ["alice", "bob"]

    pred_a, _ = loaded.recognize(_cluster(c1, n=1, seed=3)[0])
    pred_b, _ = loaded.recognize(_cluster(c2, n=1, seed=4)[0])
    assert pred_a == "alice"
    assert pred_b == "bob"


def test_system_replay_stub_raises_clear_todo(tmp_path: Path) -> None:
    cfg = SystemConfig(
        registration="replay",
        exemplar_selection="herding",
        recognition="ncm",
        exemplar_k=5,
    )
    system = FaceRecognitionSystem.from_config(cfg, workspace=tmp_path)
    emb = _cluster(np.ones((128,), dtype=np.float32), n=6, seed=20)
    with pytest.raises(NotImplementedError, match="TODO\\(§6.2\\)"):
        system.register("alice", emb)
