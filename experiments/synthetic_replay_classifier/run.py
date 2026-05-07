"""
Integration test for SyntheticReplayStrategy using FaceRecognitionSystem.

Extracts real embeddings from data/val/ images via embedding_helper,
uses pre-split train/test sets from supertask_8_2.json,
registers identities incrementally per task order,
and evaluates recognition accuracy on held-out test embeddings after each task step.

Usage:
    python -m experiments.synthetic_replay_classifier.run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
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
from experiments.memory_metrics import peak_delta_mb, snapshot_peak_memory
from experiments.registration_metrics import (
    RegistrationEvent,
    log_registration_event,
    log_registration_summary,
    take_storage_snapshot,
)
from src.continual.synthetic_replay import SyntheticReplayConfig, SyntheticReplayStrategy
from src.memory.herding import HerdingSelector
from src.recognition.classifier_based import ClassifierRecognizer
from src.system import FaceRecognitionSystem


@dataclass(frozen=True)
class TaskSpec:
    name: str
    identities: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_supertask(path: Path) -> Tuple[List[TaskSpec], Dict[str, str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks_raw: Dict[str, List[str]] = data["tasks"]

    def task_sort_key(t: str) -> Tuple[int, str]:
        suffix = t.removeprefix("task")
        return (int(suffix) if suffix.isdigit() else 10**9, t)

    task_names = sorted(tasks_raw.keys(), key=task_sort_key)
    task_order = [TaskSpec(name=t, identities=list(tasks_raw[t])) for t in task_names]

    identity_task_map: Dict[str, str] = {}
    for t in task_order:
        for ident in t.identities:
            identity_task_map[str(ident)] = t.name

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


def _identity_task_index_map(task_order: Sequence[TaskSpec]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t_idx, task in enumerate(task_order):
        for ident in task.identities:
            out[str(ident)] = t_idx
    return out


def _ensure_workspace(workspace: Path, *, reset: bool) -> None:
    if reset and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic replay continual FR experiment.")
    parser.add_argument(
        "--supertask-json",
        type=str,
        default=str(_repo_root() / "data/supertask_8_2.json"),
        help="Path to the supertask JSON (default: data/supertask_8_2.json).",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default=str(_repo_root() / "experiments" / "synthetic_replay_classifier"),
        help="Experiment directory root (default: experiments/synthetic_replay_classifier).",
    )
    parser.add_argument(
        "--reset-workspace",
        action="store_true",
        help="Delete and recreate the experiment workspace before running.",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Recompute embeddings even if cached embeddings exist.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Recognition threshold for classifier-based recognition.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of SGD epochs for each incremental update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="SGD mini-batch size for each incremental update.",
    )
    parser.add_argument(
        "--synthetic-samples-per-class",
        type=int,
        default=50,
        help="Number of synthetic replay samples generated per old class.",
    )
    # not used
    parser.add_argument(
        "--exemplar-k",
        type=int,
        default=5,
        help="Number of exemplars stored per identity for Gaussian estimation.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING)

    supertask_path = Path(args.supertask_json).expanduser().resolve()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    embeddings_root = experiment_root / "embeddings"
    workspace_dir = experiment_root / "workspace"
    _ensure_workspace(workspace_dir, reset=bool(args.reset_workspace))
    logs_dir = experiment_root / "logs"

    loggers = setup_experiment_logging(log_dir=logs_dir, experiment_name="synthetic_replay_classifier")
    progress = loggers.progress
    metrics = loggers.metrics
    mem_run_start = snapshot_peak_memory()
    run_started_at = time.perf_counter()
    metrics.info("========================================")
    metrics.info("Run configuration")
    metrics.info("  epochs: %d", int(args.epochs))
    metrics.info("  batch_size: %d", int(args.batch_size))
    metrics.info("  synthetic_samples_per_class: %d", int(args.synthetic_samples_per_class))
    metrics.info("  exemplar_k: %d, not used for synthetic replay", int(args.exemplar_k))
    metrics.info("  confidence_threshold: %.4f", float(args.confidence_threshold))
    metrics.info("========================================")

    task_order, _, identity_order = _load_supertask(supertask_path)
    all_identities = list(identity_order)
    id_to_task_idx = _identity_task_index_map(task_order)
    task_column_names = [t.name for t in task_order]

    embedding_started_at = time.perf_counter()
    for ident in all_identities:
        progress.info("Embedding started for person %s (train)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=supertask_path,
            output_root=embeddings_root,
            split="train",
            identities_filter=[ident],
            overwrite=bool(args.overwrite_embeddings),
        )
    for ident in all_identities:
        progress.info("Embedding started for person %s (test)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=supertask_path,
            output_root=embeddings_root,
            split="test",
            identities_filter=[ident],
            overwrite=bool(args.overwrite_embeddings),
        )
    mem_after_embedding = snapshot_peak_memory()
    metrics.info(
        "Peak memory after embedding: %.2f MB (delta +%.2f MB)",
        mem_after_embedding.peak_rss_mb,
        peak_delta_mb(mem_run_start, mem_after_embedding),
    )
    metrics.info("Embedding time seconds: %.3f", time.perf_counter() - embedding_started_at)

    confidence_threshold = float(args.confidence_threshold)
    system = FaceRecognitionSystem(
        registration_strategy=SyntheticReplayStrategy(
            config=SyntheticReplayConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                synthetic_samples_per_class=int(args.synthetic_samples_per_class),
            )
        ),
        exemplar_selector=HerdingSelector(),
        recognition_strategy=ClassifierRecognizer(confidence_threshold=confidence_threshold),
        workspace=workspace_dir,
        exemplar_k=int(args.exemplar_k),
        confidence_threshold=confidence_threshold,
    )
    system.load()
    if not args.reset_workspace and system.identities():
        raise RuntimeError(
            "Workspace already contains a saved system state. "
            "Use --reset-workspace to start fresh or remove the workspace manually."
        )

    train_embeddings: Dict[str, np.ndarray] = {
        ident: _load_embeddings(embeddings_root, ident, "train") for ident in all_identities
    }
    test_embeddings: Dict[str, np.ndarray] = {
        ident: _load_embeddings(embeddings_root, ident, "test") for ident in all_identities
    }

    per_task_results: List[Dict[str, float]] = []
    final_predictions = None

    registered: List[str] = []
    registration_events: List[RegistrationEvent] = []
    try:
        for task in task_order:
            task_started_at = time.perf_counter()
            mem_task_before = snapshot_peak_memory()
            registration_started_at = time.perf_counter()
            for ident in task.identities:
                progress.info("Registering person %s", ident)
                storage_before = take_storage_snapshot(workspace_dir)
                reg_result = system.register(ident, train_embeddings[ident])
                storage_after = take_storage_snapshot(workspace_dir)
                registration_events.append(
                    log_registration_event(
                        metrics=metrics,
                        task_name=task.name,
                        result=reg_result,
                        before=storage_before,
                        after=storage_after,
                    )
                )
                registered.append(ident)
            registration_seconds = time.perf_counter() - registration_started_at

            test_by_identity = {ident: test_embeddings[ident] for ident in registered}
            evaluation_started_at = time.perf_counter()
            per_class_acc, preds = evaluate_system(
                system,
                test_by_identity,
                on_identity_start=lambda i: progress.info("Recognising %s on test images", i),
            )
            evaluation_seconds = time.perf_counter() - evaluation_started_at
            per_task_results.append(per_class_acc)
            final_predictions = preds
            mem_task_after = snapshot_peak_memory()
            metrics.info(
                "Timing after %s: registration/update %.3f s, evaluation %.3f s, task total %.3f s",
                task.name,
                registration_seconds,
                evaluation_seconds,
                time.perf_counter() - task_started_at,
            )
            metrics.info(
                "Peak memory after %s: %.2f MB (delta +%.2f MB)",
                task.name,
                mem_task_after.peak_rss_mb,
                peak_delta_mb(mem_task_before, mem_task_after),
            )

        A, identity_names = compute_accuracy_matrix(per_task_results, task_column_names, id_to_task_idx)
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
        mem_run_end = snapshot_peak_memory()
        log_registration_summary(metrics=metrics, events=registration_events)
        metrics.info(
            "Overall peak memory: %.2f MB (run delta +%.2f MB)",
            mem_run_end.peak_rss_mb,
            peak_delta_mb(mem_run_start, mem_run_end),
        )
        metrics.info("Overall run time seconds: %.3f", time.perf_counter() - run_started_at)

    finally:
        progress.info("Workspace retained at %s", workspace_dir)
        metrics.info("Workspace retained at %s", workspace_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())