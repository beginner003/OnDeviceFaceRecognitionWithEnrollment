from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
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
    for entry in raw.get("identities", []) or []:
        ident = str(entry.get("identity", "")).strip()
        task = str(entry.get("task", "")).strip()
        if not ident or task not in tasks:
            continue
        identity_task_map[ident] = task_order.index(task)

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

    if args.reset_workspace and workspace.exists():
        shutil.rmtree(workspace)

    task_order, tasks, identity_task_map = _load_supertask_schema(supertask_json)
    all_identities = sorted(identity_task_map.keys(), key=lambda k: (identity_task_map[k], k))

    from experiments.embedding_helper import embed_supertask_identities_to_root

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

    for task_name in task_order:
        for identity in tasks.get(task_name, []):
            progress.info("Registering person %s", identity)
            emb = np.asarray(train_embeddings[identity], dtype=np.float32)
            system.register(identity, emb)

        registered = system.identities()
        test_subset = {ident: test_embeddings[ident] for ident in registered}
        per_class_acc, predictions = evaluate_system(
            system,
            test_subset,
            on_identity_start=lambda i: progress.info("Recognising %s on test images", i),
        )
        final_predictions = predictions
        per_task_results.append(per_class_acc)

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

if __name__ == "__main__":
    main()
