"""
Integration test for ExemplarReplayStrategy using FaceRecognitionSystem.

Extracts real embeddings from data/val/ images via embedding_helper,
uses pre-split train/test sets from supertask_8_2.json,
registers identities incrementally per task order,
and evaluates recognition accuracy on held-out test embeddings after each task step.

Usage:
    python -m experiments.replay_classifier.run --reset-workspace
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.embedding_helper import embed_supertask_identities_to_root
from experiments.eval_utils import (
    compute_accuracy_matrix,
    compute_forgetting,
    evaluate_system,
    print_confusion_matrix,
    print_per_task_table,
    print_summary_metrics,
)
from experiments.experiment_logging import setup_experiment_logging
from src.continual.exemplar_replay import ExemplarReplayConfig, ExemplarReplayStrategy
from src.memory.herding import HerdingSelector
from src.recognition.classifier_based import ClassifierRecognizer
from src.system import FaceRecognitionSystem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUPERTASK_JSON = REPO_ROOT / "data" / "supertask_8_2.json"
EXPERIMENT_ROOT = Path(__file__).resolve().parent

EPOCHS = 10
BATCH_SIZE = 10
CONFIDENCE_THRESHOLD = 0.1
MAX_NEW_EXEMPLARS = 50
EXEMPLAR_K = 5
SEED = 42


@dataclass(frozen=True)
class TaskSpec:
    name: str
    identities: List[str]


def _load_supertask(path: Path) -> Tuple[List[TaskSpec], Dict[str, int], List[str]]:
    """Return (task_order, identity->task_index map, flat identity list)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks_raw: Dict[str, List[str]] = data["tasks"]

    def task_sort_key(t: str) -> Tuple[int, str]:
        suffix = t.removeprefix("task")
        return (int(suffix) if suffix.isdigit() else 10**9, t)

    task_names = sorted(tasks_raw.keys(), key=task_sort_key)
    task_order = [TaskSpec(name=t, identities=list(tasks_raw[t])) for t in task_names]

    identity_task_map: Dict[str, int] = {}
    for t_idx, t in enumerate(task_order):
        for ident in t.identities:
            identity_task_map[str(ident)] = t_idx

    identity_order: List[str] = []
    for t in task_order:
        identity_order.extend(t.identities)
    return task_order, identity_task_map, identity_order


def _load_embeddings(embeddings_root: Path, identity: str, split: str) -> np.ndarray:
    emb_path = embeddings_root / identity / split / "embeddings.npy"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    emb = np.load(str(emb_path))
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != 128:
        raise RuntimeError(f"Unexpected embeddings shape for {identity}/{split}: {emb.shape}")
    return emb


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    embeddings_root = EXPERIMENT_ROOT / "embeddings"
    workspace_dir = Path(tempfile.mkdtemp(prefix="replay_integration_"))
    logs_dir = EXPERIMENT_ROOT / "logs"

    loggers = setup_experiment_logging(log_dir=logs_dir, experiment_name="replay_integration")
    progress = loggers.progress
    metrics = loggers.metrics
    metrics.info("========================================")
    metrics.info("Run configuration")
    metrics.info("  strategy: exemplar_replay")
    metrics.info("  epochs: %d", EPOCHS)
    metrics.info("  batch_size: %d", BATCH_SIZE)
    metrics.info("  confidence_threshold: %.4f", CONFIDENCE_THRESHOLD)
    metrics.info("  exemplar_k (stored per identity): %d", EXEMPLAR_K)
    metrics.info("  max new-identity training samples: %d", MAX_NEW_EXEMPLARS)
    metrics.info("========================================")

    task_order, id_to_task_idx, identity_order = _load_supertask(SUPERTASK_JSON)
    all_identities = list(identity_order)
    task_column_names = [t.name for t in task_order]

    for ident in all_identities:
        progress.info("Embedding %s (train)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=SUPERTASK_JSON,
            output_root=embeddings_root,
            split="train",
            identities_filter=[ident],
        )
    for ident in all_identities:
        progress.info("Embedding %s (test)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=SUPERTASK_JSON,
            output_root=embeddings_root,
            split="test",
            identities_filter=[ident],
        )

    train_embeddings: Dict[str, np.ndarray] = {
        ident: _load_embeddings(embeddings_root, ident, "train") for ident in all_identities
    }
    test_embeddings: Dict[str, np.ndarray] = {
        ident: _load_embeddings(embeddings_root, ident, "test") for ident in all_identities
    }

    for ident in all_identities:
        progress.info(
            "  %s: %d train, %d test",
            ident, train_embeddings[ident].shape[0], test_embeddings[ident].shape[0],
        )

    try:
        system = FaceRecognitionSystem(
            registration_strategy=ExemplarReplayStrategy(
                config=ExemplarReplayConfig(
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                )
            ),
            exemplar_selector=HerdingSelector(),
            recognition_strategy=ClassifierRecognizer(confidence_threshold=CONFIDENCE_THRESHOLD),
            workspace=workspace_dir,
            exemplar_k=EXEMPLAR_K,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        system.load()

        per_task_results: List[Dict[str, float]] = []
        final_predictions = None
        registered: List[str] = []
        rng = np.random.default_rng(SEED)

        for task in task_order:
            for ident in task.identities:
                emb = train_embeddings[ident]
                if emb.shape[0] > MAX_NEW_EXEMPLARS:
                    idx = rng.choice(emb.shape[0], MAX_NEW_EXEMPLARS, replace=False)
                    emb = emb[idx]
                progress.info("Registering %s (%d embeddings)", ident, emb.shape[0])
                system.register(ident, emb)
                registered.append(ident)

            test_by_identity = {ident: test_embeddings[ident] for ident in registered}
            per_class_acc, preds = evaluate_system(
                system,
                test_by_identity,
                on_identity_start=lambda i: progress.info("Recognising %s on test images", i),
            )
            per_task_results.append(per_class_acc)
            final_predictions = preds

        A, identity_names = compute_accuracy_matrix(
            per_task_results, task_column_names, id_to_task_idx,
        )
        print_per_task_table(A, task_column_names, identity_names, logger=metrics)
        summary = print_summary_metrics(A, logger=metrics)

        forgetting = compute_forgetting(A)
        for ident, f in zip(identity_names, forgetting.tolist()):
            if math.isfinite(float(f)):
                metrics.info("Forgetting %s: %.4f", ident, float(f))

        if final_predictions is not None:
            print_confusion_matrix(
                final_predictions,
                registered_identities=identity_names,
                logger=metrics,
            )

        progress.info("=" * 50)
        progress.info("Results  (full detail in %s)", logs_dir / "evaluation.log")
        progress.info("  average_accuracy:   %.4f", summary["average_accuracy"])
        progress.info("  average_forgetting: %.4f", summary["average_forgetting"])
        progress.info("  backward_transfer:  %.4f", summary["backward_transfer"])
        progress.info("=" * 50)

    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
