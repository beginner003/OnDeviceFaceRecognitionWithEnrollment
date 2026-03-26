"""Shared logging for baseline experiments: short terminal progress + file metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentLoggers:
    """`progress` prints minimal stage lines to the terminal; `metrics` writes evaluation output to a file."""

    progress: logging.Logger
    metrics: logging.Logger


def setup_experiment_logging(
    *,
    log_dir: Path,
    experiment_name: str = "experiment",
) -> ExperimentLoggers:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = log_dir / "evaluation.log"

    metrics = logging.getLogger(f"{experiment_name}.metrics")
    metrics.handlers.clear()
    metrics.setLevel(logging.INFO)
    metrics.propagate = False
    fh = logging.FileHandler(metrics_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    metrics.addHandler(fh)

    progress = logging.getLogger(f"{experiment_name}.progress")
    progress.handlers.clear()
    progress.setLevel(logging.INFO)
    progress.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    progress.addHandler(ch)

    # Per-image embedding lines are noisy for terminal; keep warnings visible.
    logging.getLogger("experiments.embedding_helper").setLevel(logging.WARNING)

    return ExperimentLoggers(progress=progress, metrics=metrics)
