from __future__ import annotations

import argparse
import json
import logging
import math
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
from src.system import FaceRecognitionSystem, SystemConfig


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
        default=0.1,
        help=(
            "Min max-softmax probability to accept a name (else 'unknown'). "
            "For K-way classifiers, max prob is often well below 0.5 even when top-1 is correct; "
            "use 0.0 for closed-set accuracy (always take argmax). Raise toward 1/K for stricter rejection."
        ),
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

    task_order, _, identity_order = _load_supertask(supertask_path)
    all_identities = list(identity_order)
    id_to_task_idx = _identity_task_index_map(task_order)
    task_column_names = [t.name for t in task_order]

    # 1) Embed (train + test), one identity at a time for readable terminal progress.
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

    _ensure_clean_workspace(workspace_dir, reset=bool(args.reset_workspace))
    config = SystemConfig(
        registration="naive",
        exemplar_selection="herding",
        recognition="classifier",
        exemplar_k=5,
        confidence_threshold=float(args.confidence_threshold),
    )
    system = FaceRecognitionSystem.from_config(config, workspace=workspace_dir)
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
    for task in task_order:
        for ident in task.identities:
            progress.info("Registering person %s", ident)
            system.register(ident, train_embeddings[ident])
            registered.append(ident)

        test_by_identity = {ident: test_embeddings[ident] for ident in registered}
        per_class_acc, preds = evaluate_system(
            system,
            test_by_identity,
            on_identity_start=lambda i: progress.info("Recognising %s on test images", i),
        )
        per_task_results.append(per_class_acc)
        final_predictions = preds

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
