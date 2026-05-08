"""Run every Raspberry Pi 5 experiment across all set trials."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence


EXPERIMENT_SCRIPTS = (
    Path("experiments/baseline_classifier/run.py"),
    Path("experiments/baseline_ncm/run.py"),
    Path("experiments/replay_classifier/run.py"),
    Path("experiments/replay_lwf_classifier/run.py"),
    Path("experiments/lwf_classifier/run.py"),
    Path("experiments/synthetic_replay_classifier/run.py"),
)
SET_NAMES = tuple(f"set{i}" for i in range(1, 11))
CONTINUAL_LEARNING_EPOCHS = 40
METHOD_CONFIDENCE_THRESHOLDS: dict[Path, float | None] = {
    Path("experiments/baseline_classifier/run.py"): 0.5,
    Path("experiments/baseline_ncm/run.py"): None,  # NCM runner uses internal 0.5.
    Path("experiments/replay_classifier/run.py"): 0.5,
    Path("experiments/replay_lwf_classifier/run.py"): 0.5,
    Path("experiments/lwf_classifier/run.py"): 0.5,
    Path("experiments/synthetic_replay_classifier/run.py"): 0.5,
}
CONTINUAL_LEARNING_METHODS = frozenset(
    script for script in EXPERIMENT_SCRIPTS if script != Path("experiments/baseline_ncm/run.py")
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_pi_linux() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"aarch64", "arm64"}:
        raise RuntimeError(
            "Experiment log renewal must run on Raspberry Pi 5 / 64-bit Linux "
            f"(detected {system} {platform.machine()})."
        )


def _run_command(
    cmd: Sequence[str],
    *,
    repo_root: Path,
    log_path: Path,
    log_lock: threading.Lock | None = None,
    stream_prefix: str = "",
) -> int:
    def _log_line(log_handle, message: str) -> None:
        if log_lock is not None:
            with log_lock:
                log_handle.write(message)
                log_handle.flush()
        else:
            log_handle.write(message)
            log_handle.flush()

    started_at = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        _log_line(log, "\n" + "=" * 80 + "\n")
        if stream_prefix:
            _log_line(log, f"{stream_prefix} ")
        _log_line(log, f"COMMAND: {' '.join(cmd)}\n")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)
        proc = subprocess.Popen(
            list(cmd),
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            prefixed = f"{stream_prefix} {line}" if stream_prefix else line
            print(prefixed, end="")
            _log_line(log, prefixed)
        exit_code = proc.wait()
        elapsed = time.perf_counter() - started_at
        _log_line(log, f"{stream_prefix} EXIT_CODE: {exit_code}\n" if stream_prefix else f"EXIT_CODE: {exit_code}\n")
        _log_line(
            log,
            f"{stream_prefix} ELAPSED_SECONDS: {elapsed:.3f}\n"
            if stream_prefix
            else f"ELAPSED_SECONDS: {elapsed:.3f}\n",
        )
        return int(exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renew all experiment logs on Raspberry Pi 5 (full set-trial suite)."
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse cached embeddings instead of recomputing them for every experiment.",
    )
    parser.add_argument(
        "--single-run-only",
        action="store_true",
        help=(
            "Run each experiment script once with its default supertask JSON (legacy behavior). "
            "By default this script runs full set-trial suites via experiments/run_set_trials.py."
        ),
    )
    parser.add_argument(
        "--start-set",
        type=int,
        default=1,
        help="First set index to run in suite mode (default: 1).",
    )
    parser.add_argument(
        "--end-set",
        type=int,
        default=len(SET_NAMES),
        help="Last set index to run in suite mode (default: 10).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=len(EXPERIMENT_SCRIPTS),
        help=(
            "Maximum concurrent methods per set in suite mode "
            f"(default: {len(EXPERIMENT_SCRIPTS)})."
        ),
    )
    parser.add_argument(
        "--continual-epochs",
        type=int,
        default=CONTINUAL_LEARNING_EPOCHS,
        help=(
            "Epochs forwarded to classifier-based continual learning runners "
            f"(default: {CONTINUAL_LEARNING_EPOCHS})."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if int(args.continual_epochs) < 1:
        raise ValueError("--continual-epochs must be >= 1.")

    _require_pi_linux()
    repo_root = _repo_root()
    logs_dir = repo_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / "pi_experiment_refresh.log"
    run_log.write_text(
        "Raspberry Pi 5 experiment refresh\n"
        f"python: {sys.version.split()[0]}\n"
        f"platform: {platform.platform()}\n",
        encoding="utf-8",
    )

    if args.single_run_only:
        for script in EXPERIMENT_SCRIPTS:
            cmd = [sys.executable, str(script), "--reset-workspace"]
            if not args.reuse_embeddings:
                cmd.append("--overwrite-embeddings")
            threshold = METHOD_CONFIDENCE_THRESHOLDS.get(script)
            if threshold is not None:
                cmd.extend(["--confidence-threshold", str(threshold)])
            if script in CONTINUAL_LEARNING_METHODS:
                cmd.extend(["--epochs", str(int(args.continual_epochs))])
            exit_code = _run_command(cmd, repo_root=repo_root, log_path=run_log)
            if exit_code != 0:
                return exit_code
        return 0

    if (
        int(args.start_set) < 1
        or int(args.end_set) < int(args.start_set)
        or int(args.end_set) > len(SET_NAMES)
    ):
        raise ValueError(
            f"--start-set and --end-set must satisfy 1 <= start <= end <= {len(SET_NAMES)}."
        )
    selected_sets = tuple(f"set{i}" for i in range(int(args.start_set), int(args.end_set) + 1))
    if int(args.max_workers) < 1:
        raise ValueError("--max-workers must be >= 1.")

    suite_runner = Path("experiments/run_set_trials.py")
    log_lock = threading.Lock()
    worker_count = min(int(args.max_workers), len(EXPERIMENT_SCRIPTS))
    for set_name in selected_sets:
        futures: dict[concurrent.futures.Future[int], Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for script in EXPERIMENT_SCRIPTS:
                method_name = script.parent.name
                stream_prefix = f"[{set_name}|{method_name}]"
                cmd = [
                    sys.executable,
                    str(suite_runner),
                    "--runner",
                    str(script),
                    "--set-name",
                    set_name,
                    "--python",
                    sys.executable,
                ]
                threshold = METHOD_CONFIDENCE_THRESHOLDS.get(script)
                if threshold is not None:
                    cmd.extend(["--confidence-threshold", str(threshold)])
                if script in CONTINUAL_LEARNING_METHODS:
                    cmd.extend(["--epochs", str(int(args.continual_epochs))])
                if args.reuse_embeddings:
                    cmd.append("--reuse-embeddings")
                future = executor.submit(
                    _run_command,
                    cmd,
                    repo_root=repo_root,
                    log_path=run_log,
                    log_lock=log_lock,
                    stream_prefix=stream_prefix,
                )
                futures[future] = script

            for future in concurrent.futures.as_completed(futures):
                exit_code = future.result()
                if exit_code != 0:
                    script = futures[future]
                    method_name = script.parent.name
                    with run_log.open("a", encoding="utf-8") as log:
                        with log_lock:
                            log.write(
                                f"[{set_name}|{method_name}] FAILURE detected; "
                                "finishing current set before exit.\n"
                            )
                            log.flush()
                    return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
