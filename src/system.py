"""High-level orchestrator for registration and recognition workflows."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.continual.classifier import CosineLinear
from src.continual.exemplar_replay import ExemplarReplayStrategy
from src.continual.naive_ft import NaiveFTStrategy
from src.continual.replay_lwf import ReplayLwFStrategy
from src.memory.exemplar_store import ExemplarStore
from src.memory.herding import HerdingSelector
from src.memory.random_selector import RandomSelector
from src.protocols import ExemplarSelector, RecognitionStrategy, RegistrationStrategy
from src.recognition.classifier_based import ClassifierRecognizer
from src.recognition.ncm import NCMRecognizer


@dataclass(frozen=True)
class SystemConfig:
    """Factory config for strategy and threshold selection."""

    registration: str = "replay_lwf"  # "naive" | "replay" | "replay_lwf"
    exemplar_selection: str = "herding"  # "herding" | "random"
    recognition: str = "classifier"  # "ncm" | "classifier"
    exemplar_k: int = 50
    confidence_threshold: float = 0.5


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome metrics for a single registration transaction."""

    identity: str
    selected_count: int
    total_identities: int
    elapsed_s: float
    exemplar_bytes: int


class FaceRecognitionSystem:
    """
    Top-level orchestrator for registration and recognition.

    The application layer is responsible for capture/detect/align/embed. This system
    works on 128-d embeddings and pluggable strategy components only.
    """

    _CLASSIFIER_CKPT = "classifier.pt"
    _STATE_JSON = "system_state.json"

    def __init__(
        self,
        registration_strategy: RegistrationStrategy,
        exemplar_selector: ExemplarSelector,
        recognition_strategy: RecognitionStrategy,
        workspace: Path,
        *,
        exemplar_k: int = 50,
        confidence_threshold: float = 0.5,
    ) -> None:
        if exemplar_k <= 0:
            raise ValueError("exemplar_k must be > 0")
        self.registration_strategy = registration_strategy
        self.exemplar_selector = exemplar_selector
        self.recognition_strategy = recognition_strategy
        self.exemplar_k = int(exemplar_k)
        self.confidence_threshold = float(confidence_threshold)

        self.workspace = Path(workspace)
        self.exemplars_dir = self.workspace / "exemplars"
        self.checkpoints_dir = self.workspace / "checkpoints"
        self.logs_dir = self.workspace / "logs"
        self.exemplars_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.store = ExemplarStore(self.exemplars_dir)
        self.classifier = CosineLinear(in_features=128, out_features=0)
        self._identity_to_class: Dict[str, int] = {}

    def register(self, name: str, embeddings: np.ndarray) -> RegistrationResult:
        """
        Register a new identity from pre-extracted embeddings.

        NOTE: Re-registering an existing identity is intentionally blocked for now
        because class-index semantics for "replace" need method-specific handling.
        """
        start_t = time.perf_counter()
        identity = str(name).strip()
        if not identity:
            raise ValueError("Identity name cannot be empty")
        if identity in self._identity_to_class:
            raise ValueError(
                f"Identity '{identity}' already exists. TODO: implement re-registration flow."
            )

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.classifier.in_features:
            raise ValueError(
                f"embeddings must be (N, {self.classifier.in_features}), got {emb.shape}"
            )
        if emb.shape[0] == 0:
            raise ValueError("embeddings must contain at least one vector")

        selected, _indices = self.exemplar_selector.select(emb, self.exemplar_k)
        self.store.upsert_class(identity, selected)

        # Strategy owns update policy (naive, replay, replay+lwf...).
        self.classifier = self.registration_strategy.update(
            classifier=self.classifier,
            store=self.store,
            new_embeddings=emb,
            identity=identity,
        )

        self._identity_to_class[identity] = self.classifier.out_features - 1
        self._sync_classifier_class_names()
        self.save()

        elapsed = time.perf_counter() - start_t
        result = RegistrationResult(
            identity=identity,
            selected_count=int(selected.shape[0]),
            total_identities=len(self._identity_to_class),
            elapsed_s=float(elapsed),
            exemplar_bytes=self.store.total_bytes(),
        )
        self._append_registration_log(result)
        return result

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Identify one query embedding; unknown if below strategy threshold."""
        self._sync_classifier_class_names()
        return self.recognition_strategy.predict(
            embedding=embedding,
            store=self.store,
            classifier=self.classifier,
        )

    def identities(self) -> List[str]:
        """List registered identities ordered by class index."""
        return [name for name, _idx in sorted(self._identity_to_class.items(), key=lambda kv: kv[1])]

    def remove_identity(self, name: str) -> None:
        """
        Remove one identity and rebuild class indices.

        For cosine heads this can be done by dropping one row and remapping.
        """
        identity = str(name)
        if identity not in self._identity_to_class:
            return
        remove_idx = self._identity_to_class[identity]
        self.store.remove_class(identity)

        old_weight = None if self.classifier.weight is None else self.classifier.weight.detach().clone()
        old_sigma = self.classifier.sigma.detach().clone()
        keep = [n for n in self.identities() if n != identity]
        self._identity_to_class = {n: i for i, n in enumerate(keep)}

        rebuilt = CosineLinear(in_features=self.classifier.in_features, out_features=len(keep))
        with torch.no_grad():
            rebuilt.sigma.copy_(old_sigma)
            if old_weight is not None and len(keep) > 0:
                mask = [i for i in range(old_weight.shape[0]) if i != remove_idx]
                rebuilt.weight.copy_(old_weight[mask])
        self.classifier = rebuilt
        self._sync_classifier_class_names()
        self.save()

    def save(self) -> None:
        """Persist classifier, exemplar store, and class-index mapping."""
        self.store.save_all()
        ckpt_path = self.checkpoints_dir / self._CLASSIFIER_CKPT
        state_path = self.workspace / self._STATE_JSON

        torch.save(
            {
                "in_features": self.classifier.in_features,
                "out_features": self.classifier.out_features,
                "state_dict": self.classifier.state_dict(),
            },
            ckpt_path,
        )
        state_payload = {"identity_to_class": self._identity_to_class}
        state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Restore classifier, exemplar store, and class-index mapping."""
        self.store = ExemplarStore(self.exemplars_dir)
        self.store.load_all()

        ckpt_path = self.checkpoints_dir / self._CLASSIFIER_CKPT
        state_path = self.workspace / self._STATE_JSON
        if not ckpt_path.is_file() or not state_path.is_file():
            # Fresh workspace: nothing to load.
            self.classifier = CosineLinear(in_features=128, out_features=0)
            self._identity_to_class = {}
            return

        raw = json.loads(state_path.read_text(encoding="utf-8"))
        mapping = raw.get("identity_to_class", {})
        self._identity_to_class = {str(k): int(v) for k, v in mapping.items()}

        payload = torch.load(ckpt_path, map_location="cpu")
        in_features = int(payload["in_features"])
        out_features = int(payload["out_features"])
        clf = CosineLinear(in_features=in_features, out_features=out_features)
        clf.load_state_dict(payload["state_dict"])
        self.classifier = clf
        self._sync_classifier_class_names()

    @classmethod
    def from_config(cls, config: SystemConfig, workspace: Path) -> "FaceRecognitionSystem":
        """Build a system with strategy objects chosen from string config."""
        registration_map: Dict[str, RegistrationStrategy] = {
            "naive": NaiveFTStrategy(),
            "replay": ExemplarReplayStrategy(),
            "replay_lwf": ReplayLwFStrategy(),
        }
        selector_map: Dict[str, ExemplarSelector] = {
            "herding": HerdingSelector(),
            "random": RandomSelector(),
        }
        recognition_map: Dict[str, RecognitionStrategy] = {
            "ncm": NCMRecognizer(confidence_threshold=config.confidence_threshold),
            "classifier": ClassifierRecognizer(confidence_threshold=config.confidence_threshold),
        }

        if config.registration not in registration_map:
            raise ValueError(f"Unknown registration strategy: {config.registration}")
        if config.exemplar_selection not in selector_map:
            raise ValueError(f"Unknown exemplar selection strategy: {config.exemplar_selection}")
        if config.recognition not in recognition_map:
            raise ValueError(f"Unknown recognition strategy: {config.recognition}")

        system = cls(
            registration_strategy=registration_map[config.registration],
            exemplar_selector=selector_map[config.exemplar_selection],
            recognition_strategy=recognition_map[config.recognition],
            workspace=workspace,
            exemplar_k=config.exemplar_k,
            confidence_threshold=config.confidence_threshold,
        )
        system.load()
        return system

    def _sync_classifier_class_names(self) -> None:
        class_names = self.identities()
        setattr(self.classifier, "_class_names", class_names)

    def _append_registration_log(self, result: RegistrationResult) -> None:
        log_path = self.logs_dir / "registration.log"
        event = {
            "identity": result.identity,
            "selected_count": result.selected_count,
            "total_identities": result.total_identities,
            "elapsed_s": round(result.elapsed_s, 4),
            "exemplar_bytes": result.exemplar_bytes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
