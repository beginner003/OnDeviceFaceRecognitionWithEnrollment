from __future__ import annotations

import argparse
import json
import logging
import math
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
from experiments.unknown_identity_eval import evaluate_unknown_identity_rejection
from src.continual.naive_ft import NaiveFTConfig, NaiveFTStrategy
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


def _ensure_clean_workspace(workspace: Path, *, reset: bool) -> None:
    if not workspace.exists():
        return
    if reset:
        for p in sorted(workspace.glob("**/*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            workspace.rmdir()
        except OSError:
            pass


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline classifier-based continual FR experiment.")
    parser.add_argument(
        "--supertask-json",
        type=str,
        default=str(_repo_root() / "data" / "supertask_8_2.json"),
        help="Path to the supertask JSON (default: data/supertask_8_2.json).",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Experiment directory root (default: experiments/baseline_classifier).",
    )
    parser.add_argument(
        "--reset-workspace",
        action="store_true",
        help="Delete the experiment workspace before running (recommended for clean runs).",
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
        help=(
            "Min max-softmax probability to accept a name (else 'unknown'). "
            "For K-way classifiers, max prob is often well below 0.5 even when top-1 is correct; "
            "use 0.0 for closed-set accuracy (always take argmax). Raise toward 1/K for stricter rejection."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of SGD epochs for each incremental update (naive fine-tuning).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="SGD mini-batch size for each incremental update (naive fine-tuning).",
    )
    parser.add_argument(
        "--unknown-identities-count",
        type=int,
        default=10,
        help="Number of unseen identities to evaluate as unknown (0 disables this test).",
    )
    parser.add_argument(
        "--unknown-seed",
        type=int,
        default=20260507,
        help="RNG seed used when sampling unknown identities from data/val.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING)

    supertask_path = Path(args.supertask_json).expanduser().resolve()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    embeddings_root = experiment_root / "embeddings"
    workspace_dir = experiment_root / "workspace"
    logs_dir = experiment_root / "logs"

    loggers = setup_experiment_logging(log_dir=logs_dir, experiment_name="baseline_classifier")
    progress = loggers.progress
    metrics = loggers.metrics
    mem_run_start = snapshot_peak_memory()
    run_started_at = time.perf_counter()
    metrics.info("========================================")
    metrics.info("Run configuration")
    metrics.info("  epochs: %d", int(args.epochs))
    metrics.info("  batch_size: %d", int(args.batch_size))
    metrics.info("  confidence_threshold: %.4f", float(args.confidence_threshold))
    metrics.info("========================================")

    task_order, _, identity_order = _load_supertask(supertask_path)
    all_identities = list(identity_order)
    id_to_task_idx = _identity_task_index_map(task_order)
    task_column_names = [t.name for t in task_order]

    # 1) Embed (train + test), one identity at a time for readable terminal progress.
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

    _ensure_clean_workspace(workspace_dir, reset=bool(args.reset_workspace))
    confidence_threshold = float(args.confidence_threshold)
    system = FaceRecognitionSystem(
        registration_strategy=NaiveFTStrategy(
            config=NaiveFTConfig(
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
            )
        ),
        exemplar_selector=HerdingSelector(),
        recognition_strategy=ClassifierRecognizer(confidence_threshold=confidence_threshold),
        workspace=workspace_dir,
        exemplar_k=5,
        confidence_threshold=confidence_threshold,
    )
    system.load()
    if system.identities() and not args.reset_workspace:
        raise RuntimeError(
            f"Workspace {workspace_dir} already has registered identities: {system.identities()}. "
            "Re-run with --reset-workspace for a clean baseline run."
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
    print_summary_metrics(A, logger=metrics)

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
    evaluate_unknown_identity_rejection(
        system=system,
        supertask_path=supertask_path,
        embeddings_root=embeddings_root,
        known_identities=all_identities,
        unknown_count=int(args.unknown_identities_count),
        seed=int(args.unknown_seed),
        overwrite_embeddings=bool(args.overwrite_embeddings),
        metrics_logger=metrics,
        progress_logger=progress,
    )

    mem_run_end = snapshot_peak_memory()
    log_registration_summary(metrics=metrics, events=registration_events)
    metrics.info(
        "Overall peak memory: %.2f MB (run delta +%.2f MB)",
        mem_run_end.peak_rss_mb,
        peak_delta_mb(mem_run_start, mem_run_end),
    )
    metrics.info("Overall run time seconds: %.3f", time.perf_counter() - run_started_at)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
