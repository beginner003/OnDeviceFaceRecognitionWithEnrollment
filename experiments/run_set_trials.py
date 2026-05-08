"""Run 3 trials for one set and write a consolidated per-set log.

Each trial's full evaluation.log (per-task tables, confusion matrix, registration
summary, etc.) is copied into the set log. A final section averages key metrics
across the three trials.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO_ROOT / "data" / "supertask_suite"
BANNER_WIDTH = 80

SUMMARY_PATTERNS = {
    "average_accuracy": re.compile(r"average_accuracy:\s*([0-9]*\.?[0-9]+)"),
    "average_forgetting": re.compile(r"average_forgetting:\s*([0-9]*\.?[0-9]+)"),
    "backward_transfer": re.compile(r"backward_transfer:\s*([-+]?[0-9]*\.?[0-9]+)"),
    "overall_run_time_seconds": re.compile(r"Overall run time seconds:\s*([0-9]*\.?[0-9]+)"),
}


def _banner(char: str = "=") -> str:
    return char * BANNER_WIDTH


def _parse_metrics(evaluation_log_text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, pat in SUMMARY_PATTERNS.items():
        matches = pat.findall(evaluation_log_text)
        if not matches:
            raise RuntimeError(f"Could not find metric '{key}' in evaluation log.")
        out[key] = float(matches[-1])
    return out


def _resolve_eval_log_path(runner_script: Path, experiment_root: Path | None) -> Path:
    root = experiment_root if experiment_root is not None else runner_script.parent
    return root / "logs" / "evaluation.log"


def _make_header_logger(set_log_path: Path) -> logging.Logger:
    """Logger matching experiments/evaluation.log format for set-level banners only."""
    log = logging.getLogger(f"set_trials.{set_log_path.stem}.{id(set_log_path)}")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fh = logging.FileHandler(set_log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(fh)
    return log


def _flush_set_log(logger: logging.Logger) -> None:
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run set trials (1..3) and consolidate full trial logs into one set log."
    )
    parser.add_argument(
        "--runner",
        type=str,
        required=True,
        help="Experiment runner script path, e.g. experiments/lwf_classifier/run.py",
    )
    parser.add_argument(
        "--set-name",
        type=str,
        default="set1",
        help="Set name to run, e.g. set1 .. set10 (default: set1).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the trial scripts.",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="",
        dest="experiment_root",
        help="Optional experiment root passed to runner via --experiment-root.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse cached embeddings (skip --overwrite-embeddings).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Optional recognition confidence threshold forwarded to runner "
            "(used for unknown-identity rejection tuning)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch count forwarded to runners that support --epochs.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"set\d+", str(args.set_name)):
        raise ValueError(f"Invalid --set-name {args.set_name!r}; expected pattern 'setN'.")
    if args.epochs is not None and int(args.epochs) < 1:
        raise ValueError("--epochs must be >= 1.")

    runner = Path(args.runner).expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"Runner not found: {runner}")

    experiment_root = (
        Path(args.experiment_root).expanduser().resolve()
        if str(args.experiment_root).strip()
        else None
    )
    eval_log_path = _resolve_eval_log_path(runner, experiment_root)

    set_dir = runner.parent / "logs" / "sets"
    set_dir.mkdir(parents=True, exist_ok=True)
    set_log_path = set_dir / f"{args.set_name}.log"

    header_log = _make_header_logger(set_log_path)
    header_log.info(_banner())
    try:
        runner_display = runner.relative_to(REPO_ROOT)
    except ValueError:
        runner_display = runner
    header_log.info(
        "Set suite: %s | runner: %s | trials: 3 (full evaluation.log per trial below)",
        args.set_name,
        runner_display,
    )
    try:
        suite_rel = SUITE_DIR.relative_to(REPO_ROOT)
    except ValueError:
        suite_rel = SUITE_DIR
    header_log.info("Supertask directory: %s", suite_rel)
    header_log.info(_banner())
    _flush_set_log(header_log)

    metrics_per_trial: List[Dict[str, float]] = []

    for trial in (1, 2, 3):
        supertask_path = SUITE_DIR / f"{args.set_name}_trial{trial}.json"
        if not supertask_path.is_file():
            raise FileNotFoundError(f"Missing supertask file: {supertask_path}")

        cmd = [
            args.python,
            str(runner),
            "--supertask-json",
            str(supertask_path),
            "--reset-workspace",
        ]
        if experiment_root is not None:
            cmd.extend(["--experiment-root", str(experiment_root)])
        if args.confidence_threshold is not None:
            cmd.extend(["--confidence-threshold", str(args.confidence_threshold)])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(int(args.epochs))])
        if not args.reuse_embeddings:
            cmd.append("--overwrite-embeddings")

        trial_wall_start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
        )
        elapsed_wall = time.perf_counter() - trial_wall_start
        if proc.returncode != 0:
            raise RuntimeError(
                f"Trial failed ({args.set_name} trial {trial}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not eval_log_path.is_file():
            raise FileNotFoundError(f"Expected evaluation log after trial: {eval_log_path}")

        eval_text = eval_log_path.read_text(encoding="utf-8").rstrip()
        trial_metrics = _parse_metrics(eval_text)
        trial_metrics["elapsed_wall_s"] = float(elapsed_wall)
        metrics_per_trial.append(trial_metrics)

        # Trial section banner + verbatim evaluation.log (same format as per-run logs).
        header_log.info("")
        header_log.info(_banner())
        header_log.info(
            "TRIAL %d / 3  |  %s  |  supertask: %s",
            trial,
            args.set_name,
            supertask_path.name,
        )
        header_log.info("Wall clock (subprocess): %.3f s", elapsed_wall)
        header_log.info("Source: %s", eval_log_path)
        header_log.info(_banner())
        _flush_set_log(header_log)

        with set_log_path.open("a", encoding="utf-8") as f:
            f.write(eval_text)
            f.write("\n\n")

    n = float(len(metrics_per_trial))
    avg = {
        "average_accuracy": sum(m["average_accuracy"] for m in metrics_per_trial) / n,
        "average_forgetting": sum(m["average_forgetting"] for m in metrics_per_trial) / n,
        "backward_transfer": sum(m["backward_transfer"] for m in metrics_per_trial) / n,
        "overall_run_time_seconds": sum(m["overall_run_time_seconds"] for m in metrics_per_trial) / n,
        "elapsed_wall_s": sum(m["elapsed_wall_s"] for m in metrics_per_trial) / n,
    }

    header_log.info("")
    header_log.info(_banner())
    header_log.info("SET SUMMARY  |  %s  |  %d trials averaged", args.set_name, len(metrics_per_trial))
    header_log.info(_banner())
    for i, m in enumerate(metrics_per_trial, start=1):
        header_log.info(
            "Trial %d: average_accuracy=%.4f  average_forgetting=%.4f  "
            "backward_transfer=%.4f  run_time_s=%.3f  wall_s=%.3f",
            i,
            m["average_accuracy"],
            m["average_forgetting"],
            m["backward_transfer"],
            m["overall_run_time_seconds"],
            m["elapsed_wall_s"],
        )
    header_log.info(_banner("-"))
    header_log.info("Means across trials:")
    header_log.info("  average_accuracy:   %.4f", avg["average_accuracy"])
    header_log.info("  average_forgetting: %.4f", avg["average_forgetting"])
    header_log.info("  backward_transfer:  %.4f", avg["backward_transfer"])
    header_log.info("  overall_run_time_s: %.3f (mean of per-trial logged totals)", avg["overall_run_time_seconds"])
    header_log.info("  wall_clock_s:       %.3f (mean subprocess wall time per trial)", avg["elapsed_wall_s"])
    header_log.info(_banner())
    header_log.info("End of set log: %s", set_log_path.name)
    _flush_set_log(header_log)

    print(f"Wrote set log: {set_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
