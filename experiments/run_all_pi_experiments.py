"""Run every Raspberry Pi 5 experiment and renew evaluation logs."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


EXPERIMENT_SCRIPTS = (
    Path("experiments/baseline_classifier/run.py"),
    Path("experiments/baseline_ncm/run.py"),
    Path("experiments/replay_classifier/run.py"),
    Path("experiments/lwf_classifier/run.py"),
    Path("experiments/synthetic_replay_classifier/run.py"),
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


def _run_command(cmd: Sequence[str], *, repo_root: Path, log_path: Path) -> int:
    started_at = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"COMMAND: {' '.join(cmd)}\n")
        log.flush()

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
            print(line, end="")
            log.write(line)
        exit_code = proc.wait()
        elapsed = time.perf_counter() - started_at
        log.write(f"EXIT_CODE: {exit_code}\n")
        log.write(f"ELAPSED_SECONDS: {elapsed:.3f}\n")
        return int(exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renew all experiment logs on Raspberry Pi 5."
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse cached embeddings instead of recomputing them for every experiment.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

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

    for script in EXPERIMENT_SCRIPTS:
        cmd = [sys.executable, str(script), "--reset-workspace"]
        if not args.reuse_embeddings:
            cmd.append("--overwrite-embeddings")
        exit_code = _run_command(cmd, repo_root=repo_root, log_path=run_log)
        if exit_code != 0:
            return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
