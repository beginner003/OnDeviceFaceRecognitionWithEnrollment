"""Unit tests for exemplar memory manager and herding selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.memory.exemplar_store import ExemplarStore
from src.memory.herding import herding_select, select_exemplar_indices


def test_herding_select_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2D embeddings"):
        select_exemplar_indices(np.zeros((3,), dtype=np.float32), k=2)
    with pytest.raises(ValueError, match="at least one embedding"):
        select_exemplar_indices(np.zeros((0, 128), dtype=np.float32), k=2)
    with pytest.raises(ValueError, match="k must be > 0"):
        select_exemplar_indices(np.zeros((3, 128), dtype=np.float32), k=0)


def test_herding_select_returns_unique_indices_and_expected_shapes() -> None:
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(10, 128)).astype(np.float32)
    selected, indices = herding_select(embeddings=emb, k=4)
    assert selected.shape == (4, 128)
    assert indices.shape == (4,)
    assert np.unique(indices).size == 4
    assert np.all(indices >= 0)
    assert np.all(indices < emb.shape[0])


def test_herding_select_k_larger_than_n_clips_to_n() -> None:
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(3, 128)).astype(np.float32)
    selected, indices = herding_select(embeddings=emb, k=8)
    assert selected.shape[0] == 3
    assert indices.shape[0] == 3


def test_exemplar_store_upsert_validates_image_paths_count(tmp_path: Path) -> None:
    store = ExemplarStore(tmp_path)
    emb = np.ones((2, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="same length as embeddings"):
        store.upsert_class("alice", emb, image_paths=["a.png"])


def test_exemplar_store_roundtrip_save_load_and_total_bytes(tmp_path: Path) -> None:
    store = ExemplarStore(tmp_path)
    rng = np.random.default_rng(123)
    emb = rng.normal(size=(5, 128)).astype(np.float32)
    image_paths = [f"img_{i}.png" for i in range(5)]

    saved = store.upsert_class("alice", emb, image_paths=image_paths)
    assert saved.embeddings.dtype == np.float16
    assert saved.prototype.shape == (128,)
    assert saved.total_bytes() > 0

    npz_path = store.save_class("alice")
    assert npz_path.is_file()

    loaded_store = ExemplarStore(tmp_path)
    loaded = loaded_store.load_class("alice")
    assert loaded.embeddings.shape == (5, 128)
    assert loaded.embeddings.dtype == np.float16
    assert loaded.prototype.shape == (128,)
    assert loaded.image_paths == image_paths
    assert loaded_store.total_bytes() > 0


def test_exemplar_store_load_all_only_loads_valid_identity_dirs(tmp_path: Path) -> None:
    store = ExemplarStore(tmp_path)
    emb = np.eye(4, dtype=np.float32)
    store.upsert_class("alice", emb)
    store.save_class("alice")

    orphan_dir = tmp_path / "notes"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "readme.txt").write_text("ignore me", encoding="utf-8")

    reloaded = ExemplarStore(tmp_path)
    ids = reloaded.load_all()
    assert ids == ["alice"]
    assert reloaded.identities() == ["alice"]

