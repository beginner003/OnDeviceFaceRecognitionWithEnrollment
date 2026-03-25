"""Exemplar storage with per-class persistence and memory accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class ExemplarSet:
    """Stored exemplars for one identity."""

    embeddings: np.ndarray
    prototype: np.ndarray
    image_paths: List[str]

    def total_bytes(self) -> int:
        """Approximate in-memory bytes for this set."""
        paths_bytes = sum(len(p.encode("utf-8")) for p in self.image_paths)
        return int(self.embeddings.nbytes + self.prototype.nbytes + paths_bytes)


class ExemplarStore:
    """
    In-memory + on-disk exemplar manager.

    Persistence format:
    - root/<identity>/exemplars.npz with arrays:
      - embeddings: (K, D) float16
      - prototype: (D,) float32
      - image_paths: (K,) object/str
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, ExemplarSet] = {}

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings)
        if x.ndim != 2:
            raise ValueError(f"Expected embeddings shape (K, D), got {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty")
        return x

    @staticmethod
    def _validate_image_paths(image_paths: Iterable[str] | None, k: int) -> List[str]:
        if image_paths is None:
            return []
        paths = [str(p) for p in image_paths]
        if paths and len(paths) != k:
            raise ValueError(
                f"image_paths must have same length as embeddings ({k}), got {len(paths)}"
            )
        return paths

    def upsert_class(
        self,
        identity: str,
        embeddings: np.ndarray,
        image_paths: Iterable[str] | None = None,
    ) -> ExemplarSet:
        x = self._validate_embeddings(embeddings)
        k, d = x.shape
        paths = self._validate_image_paths(image_paths=image_paths, k=k)

        x_f16 = x.astype(np.float16, copy=False)
        prototype = x.astype(np.float32, copy=False).mean(axis=0)
        if prototype.shape != (d,):
            raise RuntimeError("Prototype shape mismatch")

        exemplar_set = ExemplarSet(embeddings=x_f16, prototype=prototype, image_paths=paths)
        self._data[identity] = exemplar_set
        return exemplar_set

    def remove_class(self, identity: str) -> None:
        self._data.pop(identity, None)
        class_dir = self.root_dir / identity
        npz_path = class_dir / "exemplars.npz"
        if npz_path.is_file():
            npz_path.unlink()
        if class_dir.is_dir():
            try:
                class_dir.rmdir()
            except OSError:
                # Keep directory if it still has user-managed files.
                pass

    def identities(self) -> List[str]:
        return sorted(self._data.keys())

    def get(self, identity: str) -> ExemplarSet:
        return self._data[identity]

    def total_bytes(self) -> int:
        return int(sum(v.total_bytes() for v in self._data.values()))

    def save_class(self, identity: str) -> Path:
        if identity not in self._data:
            raise KeyError(f"Identity not found: {identity}")

        payload = self._data[identity]
        class_dir = self.root_dir / identity
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / "exemplars.npz"
        np.savez_compressed(
            out_path,
            embeddings=payload.embeddings.astype(np.float16, copy=False),
            prototype=payload.prototype.astype(np.float32, copy=False),
            image_paths=np.asarray(payload.image_paths, dtype=object),
        )
        return out_path

    def load_class(self, identity: str) -> ExemplarSet:
        in_path = self.root_dir / identity / "exemplars.npz"
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing exemplar file: {in_path}")

        with np.load(in_path, allow_pickle=True) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float16)
            prototype = np.asarray(data["prototype"], dtype=np.float32)
            image_paths = [str(p) for p in data["image_paths"].tolist()]

        if embeddings.ndim != 2:
            raise ValueError(f"Invalid stored embeddings shape: {embeddings.shape}")
        if prototype.ndim != 1 or prototype.shape[0] != embeddings.shape[1]:
            raise ValueError(
                "Invalid prototype shape: expected "
                f"({embeddings.shape[1]},), got {prototype.shape}"
            )

        exemplar_set = ExemplarSet(
            embeddings=embeddings,
            prototype=prototype,
            image_paths=image_paths,
        )
        self._data[identity] = exemplar_set
        return exemplar_set

    def save_all(self) -> List[Path]:
        return [self.save_class(identity) for identity in self.identities()]

    def load_all(self) -> List[str]:
        loaded: List[str] = []
        for p in sorted(self.root_dir.iterdir()):
            if not p.is_dir():
                continue
            if (p / "exemplars.npz").is_file():
                self.load_class(p.name)
                loaded.append(p.name)
        return loaded

