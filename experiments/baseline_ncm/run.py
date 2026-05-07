from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
from src.system import FaceRecognitionSystem, SystemConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_sort_key(task_name: str) -> Tuple[int, str]:
    suffix = "".join(ch for ch in str(task_name) if ch.isdigit())
    return (int(suffix) if suffix else 10**9, str(task_name))


def _load_supertask_schema(path: Path) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: Dict[str, List[str]] = {str(k): list(v) for k, v in (raw.get("tasks", {}) or {}).items()}
    task_order = sorted(tasks.keys(), key=_task_sort_key)

    identity_task_map: Dict[str, int] = {}
    for t_idx, task in enumerate(task_order):
        for ident in tasks.get(task, []):
            identity_task_map[str(ident)] = t_idx

    return task_order, tasks, identity_task_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline NCM continual experiment (supertask).")
    parser.add_argument(
        "--supertask-json",
        type=str,
        default="data/supertask_8_2.json",
        help="Path to supertask JSON schema.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=str,
        default="experiments/baseline_ncm/embeddings",
        help="Directory to cache embeddings per identity.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="experiments/baseline_ncm/workspace",
        help="Workspace directory for the FaceRecognitionSystem state.",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Recompute embeddings even if cached files exist.",
    )
    parser.add_argument(
        "--reset-workspace",
        action="store_true",
        help="Delete the workspace directory before running.",
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    repo_root = _repo_root()
    supertask_json = (repo_root / args.supertask_json).resolve() if not Path(args.supertask_json).is_absolute() else Path(args.supertask_json).resolve()
    embeddings_root = (repo_root / args.embeddings_root).resolve() if not Path(args.embeddings_root).is_absolute() else Path(args.embeddings_root).resolve()
    workspace = (repo_root / args.workspace).resolve() if not Path(args.workspace).is_absolute() else Path(args.workspace).resolve()
    experiment_root = embeddings_root.parent
    logs_dir = experiment_root / "logs"

    loggers = setup_experiment_logging(log_dir=logs_dir, experiment_name="baseline_ncm")
    progress = loggers.progress
    metrics = loggers.metrics
    mem_run_start = snapshot_peak_memory()
    run_started_at = time.perf_counter()

    if args.reset_workspace and workspace.exists():
        shutil.rmtree(workspace)

    task_order, tasks, identity_task_map = _load_supertask_schema(supertask_json)
    all_identities = sorted(identity_task_map.keys(), key=lambda k: (identity_task_map[k], k))

    from experiments.embedding_helper import embed_supertask_identities_to_root

    embedding_started_at = time.perf_counter()
    for ident in all_identities:
        progress.info("Embedding started for person %s (train)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=supertask_json,
            output_root=embeddings_root,
            split="train",
            identities_filter=[ident],
            overwrite=bool(args.overwrite_embeddings),
        )
    for ident in all_identities:
        progress.info("Embedding started for person %s (test)", ident)
        embed_supertask_identities_to_root(
            supertask_json_path=supertask_json,
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

    train_embeddings = embed_supertask_identities_to_root(
        supertask_json_path=supertask_json,
        output_root=embeddings_root,
        split="train",
        overwrite=False,
    )
    test_embeddings = embed_supertask_identities_to_root(
        supertask_json_path=supertask_json,
        output_root=embeddings_root,
        split="test",
        overwrite=False,
    )

    config = SystemConfig(
        registration="naive",
        exemplar_selection="herding",
        recognition="ncm",
        exemplar_k=5,
        confidence_threshold=0.5,
    )
    system = FaceRecognitionSystem.from_config(config, workspace=workspace)

    per_task_results: List[Dict[str, float]] = []
    final_predictions = None

    registration_events: List[RegistrationEvent] = []
    for task_name in task_order:
        task_started_at = time.perf_counter()
        mem_task_before = snapshot_peak_memory()
        registration_started_at = time.perf_counter()
        for identity in tasks.get(task_name, []):
            progress.info("Registering person %s", identity)
            emb = np.asarray(train_embeddings[identity], dtype=np.float32)
            storage_before = take_storage_snapshot(workspace)
            reg_result = system.register(identity, emb)
            storage_after = take_storage_snapshot(workspace)
            registration_events.append(
                log_registration_event(
                    metrics=metrics,
                    task_name=task_name,
                    result=reg_result,
                    before=storage_before,
                    after=storage_after,
                )
            )
        registration_seconds = time.perf_counter() - registration_started_at

        registered = system.identities()
        test_subset = {ident: test_embeddings[ident] for ident in registered}
        evaluation_started_at = time.perf_counter()
        per_class_acc, predictions = evaluate_system(
            system,
            test_subset,
            on_identity_start=lambda i: progress.info("Recognising %s on test images", i),
        )
        evaluation_seconds = time.perf_counter() - evaluation_started_at
        final_predictions = predictions
        per_task_results.append(per_class_acc)
        mem_task_after = snapshot_peak_memory()
        metrics.info(
            "Timing after %s: registration/update %.3f s, evaluation %.3f s, task total %.3f s",
            task_name,
            registration_seconds,
            evaluation_seconds,
            time.perf_counter() - task_started_at,
        )
        metrics.info(
            "Peak memory after %s: %.2f MB (delta +%.2f MB)",
            task_name,
            mem_task_after.peak_rss_mb,
            peak_delta_mb(mem_task_before, mem_task_after),
        )

    A, identity_names = compute_accuracy_matrix(per_task_results, task_order, identity_task_map)
    print_per_task_table(A, task_names=task_order, identity_names=identity_names, logger=metrics)
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
        supertask_path=supertask_json,
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

if __name__ == "__main__":
    main()
