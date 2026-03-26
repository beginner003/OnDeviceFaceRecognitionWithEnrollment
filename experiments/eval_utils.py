"""Shared evaluation utilities for continual face recognition experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PredictionRecord:
    true_identity: str
    predicted_identity: str
    confidence: float


def evaluate_system(
    system,
    test_embeddings_by_identity: Mapping[str, np.ndarray],
    *,
    on_identity_start: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, float], List[PredictionRecord]]:
    """
    Evaluate a system over a per-identity set of embeddings.

    Args:
        system: Must expose `recognize(embedding)` -> (predicted_name, confidence)
        test_embeddings_by_identity: dict identity -> (N, D) float32 embeddings

    Returns:
        per_class_accuracy: dict identity -> accuracy in [0, 1]
        predictions: list of (true, predicted, confidence) for confusion matrices
    """

    per_class_acc: Dict[str, float] = {}
    predictions: List[PredictionRecord] = []

    for identity, embs in test_embeddings_by_identity.items():
        if on_identity_start is not None:
            on_identity_start(str(identity))
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected embeddings for {identity!r} to be 2D, got shape {arr.shape}")
        if arr.shape[0] == 0:
            per_class_acc[str(identity)] = 0.0
            continue

        correct = 0
        total = int(arr.shape[0])
        for i in range(total):
            pred, conf = system.recognize(arr[i])
            pred_s = str(pred)
            conf_f = float(conf)
            predictions.append(
                PredictionRecord(
                    true_identity=str(identity),
                    predicted_identity=pred_s,
                    confidence=conf_f,
                )
            )
            if pred_s == str(identity):
                correct += 1

        per_class_acc[str(identity)] = float(correct) / float(total)

    return per_class_acc, predictions


def compute_accuracy_matrix(
    per_task_results: Sequence[Mapping[str, float]],
    task_order: Sequence[str],
    identity_task_map: Mapping[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an accuracy matrix A[i, t] = accuracy of identity i after task t.

    Identities that have not been introduced yet (based on `identity_task_map`)
    are filled with NaN for earlier tasks.

    Args:
        per_task_results: list (over tasks) of dict identity -> accuracy
        task_order: ordered task names (len = T)
        identity_task_map: dict identity -> task index (0..T-1) where introduced

    Returns:
        accuracy_matrix: (I, T) float array (NaN for not-yet-registered)
        identity_names: identity ordering used for the rows
    """

    if len(per_task_results) != len(task_order):
        raise ValueError(
            f"per_task_results length ({len(per_task_results)}) must match task_order length ({len(task_order)})"
        )

    identity_names = sorted(identity_task_map.keys(), key=lambda k: (int(identity_task_map[k]), str(k)))
    T = len(task_order)
    I = len(identity_names)
    A = np.full((I, T), np.nan, dtype=np.float32)

    for t in range(T):
        results_t = per_task_results[t]
        for i, identity in enumerate(identity_names):
            intro_t = int(identity_task_map[identity])
            if t < intro_t:
                continue
            if identity in results_t:
                A[i, t] = float(results_t[identity])

    return A, identity_names


def compute_forgetting(accuracy_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-identity forgetting:

        Forgetting(i) = max_{t<=T} A[i,t] - A[i,T]

    NaNs (pre-introduction) are ignored in maxima; identities with no valid values
    return NaN.
    """

    A = np.asarray(accuracy_matrix, dtype=np.float32)
    if A.ndim != 2 or A.shape[1] == 0:
        raise ValueError(f"accuracy_matrix must be (I, T) with T>0, got {A.shape}")

    last = A[:, -1]
    max_over_time = np.nanmax(A, axis=1)
    forgetting = max_over_time - last
    # If an identity never had any valid values, nanmax yields -inf; convert to NaN.
    forgetting[~np.isfinite(max_over_time)] = np.nan
    return forgetting


def print_per_task_table(
    accuracy_matrix: np.ndarray,
    task_names: Sequence[str],
    identity_names: Sequence[str],
    *,
    logger: logging.Logger | None = None,
    title: str = "Per-task per-class accuracy",
) -> None:
    """Log a formatted table of accuracies (rows=identities, cols=tasks)."""

    log = logger or logging.getLogger(__name__)
    A = np.asarray(accuracy_matrix, dtype=np.float32)
    if A.shape != (len(identity_names), len(task_names)):
        raise ValueError(
            f"accuracy_matrix shape {A.shape} must match (len(identity_names), len(task_names)) "
            f"= ({len(identity_names)}, {len(task_names)})"
        )

    col_w = max(6, *(len(str(t)) for t in task_names))
    row_w = max(10, *(len(str(i)) for i in identity_names))

    def fmt(v: float) -> str:
        if np.isnan(v):
            return "  -   "
        return f"{100.0 * float(v):5.1f}"

    header = " " * row_w + " | " + " | ".join(str(t).ljust(col_w) for t in task_names)
    sep = "-" * len(header)

    log.info(title)
    log.info(sep)
    log.info(header)
    log.info(sep)
    for r, identity in enumerate(identity_names):
        vals = " | ".join(fmt(A[r, c]).ljust(col_w) for c in range(len(task_names)))
        log.info(f"{str(identity).ljust(row_w)} | {vals}")
    log.info(sep)


def print_confusion_matrix(
    predictions: Sequence[PredictionRecord] | Sequence[Tuple[str, str] | Tuple[str, str, float]],
    registered_identities: Sequence[str],
    *,
    logger: logging.Logger | None = None,
    title: str = "Confusion matrix (counts)",
) -> None:
    """
    Log a confusion matrix (rows=true, cols=pred) with an explicit 'unknown' column.

    `predictions` can be:
      - PredictionRecord(true_identity, predicted_identity, confidence)
      - (true, pred)
      - (true, pred, conf)
    """

    log = logger or logging.getLogger(__name__)
    true_labels = list(registered_identities)
    pred_labels = list(registered_identities) + ["unknown"]

    idx_true = {lab: i for i, lab in enumerate(true_labels)}
    idx_pred = {lab: j for j, lab in enumerate(pred_labels)}
    M = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)

    for p in predictions:
        if isinstance(p, PredictionRecord):
            t = p.true_identity
            pred = p.predicted_identity
        else:
            t = str(p[0])
            pred = str(p[1])

        if t not in idx_true:
            # Skip predictions for identities we are not reporting.
            continue
        j = idx_pred.get(pred, idx_pred["unknown"])
        M[idx_true[t], j] += 1

    row_w = max(10, *(len(str(x)) for x in true_labels))
    col_w = max(7, *(len(str(x)) for x in pred_labels), 5)

    header = " " * row_w + " | " + " | ".join(str(x).ljust(col_w) for x in pred_labels)
    sep = "-" * len(header)

    log.info(title)
    log.info(sep)
    log.info(header)
    log.info(sep)
    for i, t in enumerate(true_labels):
        vals = " | ".join(str(int(M[i, j])).rjust(col_w) for j in range(len(pred_labels)))
        log.info(f"{str(t).ljust(row_w)} | {vals}")
    log.info(sep)


def print_summary_metrics(
    accuracy_matrix: np.ndarray,
    *,
    logger: logging.Logger | None = None,
    title: str = "Summary metrics",
) -> Dict[str, float]:
    """
    Log and return summary metrics from an (I, T) accuracy matrix:
      - average_accuracy: mean over identities at final task
      - average_forgetting: mean forgetting over identities
      - backward_transfer: mean(A_last - max_prev) over identities (NaN-safe)
    """

    log = logger or logging.getLogger(__name__)
    A = np.asarray(accuracy_matrix, dtype=np.float32)
    if A.ndim != 2 or A.shape[1] == 0:
        raise ValueError(f"accuracy_matrix must be (I, T) with T>0, got {A.shape}")

    last = A[:, -1]
    avg_acc = float(np.nanmean(last)) if np.any(np.isfinite(last)) else float("nan")

    forgetting = compute_forgetting(A)
    avg_forgetting = float(np.nanmean(forgetting)) if np.any(np.isfinite(forgetting)) else float("nan")

    if A.shape[1] >= 2:
        max_prev = np.nanmax(A[:, :-1], axis=1)
        bwt = last - max_prev
        bwt[~np.isfinite(max_prev)] = np.nan
        avg_bwt = float(np.nanmean(bwt)) if np.any(np.isfinite(bwt)) else float("nan")
    else:
        avg_bwt = float("nan")

    metrics = {
        "average_accuracy": avg_acc,
        "average_forgetting": avg_forgetting,
        "backward_transfer": avg_bwt,
    }

    log.info(title)
    log.info("  average_accuracy:   %.4f", metrics["average_accuracy"])
    log.info("  average_forgetting: %.4f", metrics["average_forgetting"])
    log.info("  backward_transfer:  %.4f", metrics["backward_transfer"])
    return metrics

