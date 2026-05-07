"""Generate Set1..Set10 multi-trial supertask definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "supertask_suite"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "val"
ORIGINAL_SUPERTASK_PATH = REPO_ROOT / "data" / "supertask_8_2.json"


def _all_identities(dataset_root: Path) -> List[str]:
    ids = [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir() and p.name.startswith("n")]
    if not ids:
        raise RuntimeError(f"No identity folders found in dataset root: {dataset_root}")
    return ids


def _tasks_for_schedule(identities: Sequence[str], *, initial_count: int, chunk_sizes: Sequence[int]) -> Dict[str, List[str]]:
    if initial_count <= 0:
        raise ValueError("initial_count must be > 0")
    total = initial_count + int(sum(chunk_sizes))
    if len(identities) != total:
        raise ValueError(f"Expected {total} identities, got {len(identities)}")

    tasks: Dict[str, List[str]] = {"task0": list(identities[:initial_count])}
    cursor = initial_count
    for task_idx, chunk in enumerate(chunk_sizes, start=1):
        tasks[f"task{task_idx}"] = list(identities[cursor : cursor + chunk])
        cursor += chunk
    return tasks


def _trial_sample(
    *,
    all_ids: Sequence[str],
    total_faces: int,
    rng: np.random.Generator,
    seen: set[tuple[str, ...]],
) -> List[str]:
    for _ in range(200):
        sampled = sorted(rng.choice(np.asarray(all_ids), size=total_faces, replace=False).tolist())
        key = tuple(sampled)
        if key not in seen:
            seen.add(key)
            return sampled
    raise RuntimeError(f"Could not generate a unique trial sample for total_faces={total_faces}.")


def _write_supertask(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Set1..Set10 supertask trial JSONs.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--seed", type=int, default=20260505)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    rng = np.random.default_rng(int(args.seed))

    ids = _all_identities(dataset_root)
    source_payload = json.loads(ORIGINAL_SUPERTASK_PATH.read_text(encoding="utf-8"))

    set_specs = [
        {"set_name": "set1", "total_faces": 10, "chunk_sizes": [1, 1, 1, 2]},
        {"set_name": "set2", "total_faces": 20, "chunk_sizes": [1] * 15},
        {"set_name": "set3", "total_faces": 30, "chunk_sizes": [1] * 25},
        {"set_name": "set4", "total_faces": 40, "chunk_sizes": [1] * 35},
        {"set_name": "set5", "total_faces": 50, "chunk_sizes": [1] * 45},
        {"set_name": "set6", "total_faces": 60, "chunk_sizes": [1] * 55},
        {"set_name": "set7", "total_faces": 70, "chunk_sizes": [1] * 65},
        {"set_name": "set8", "total_faces": 80, "chunk_sizes": [1] * 75},
        {"set_name": "set9", "total_faces": 90, "chunk_sizes": [1] * 85},
        {"set_name": "set10", "total_faces": 100, "chunk_sizes": [1] * 95},
    ]

    for spec in set_specs:
        total_faces = int(spec["total_faces"])
        if total_faces > len(ids):
            raise RuntimeError(f"Need {total_faces} identities but dataset has only {len(ids)}.")

    # Set1 Trial1 keeps the original task split.
    set1_trial1 = {
        "name": "supertask_set1_trial1",
        "source": str(ORIGINAL_SUPERTASK_PATH.relative_to(REPO_ROOT)),
        "scenario": {
            "type": "on_device_evaluation",
            "task_grouping": "set1_trial1_original",
        },
        "set": "set1",
        "trial": 1,
        "tasks": source_payload["tasks"],
        "dataset_root": "data/val",
        "constraints": {
            "train_images_per_identity": 50,
            "test_images_per_identity": 10,
            "test_start_index": 10,
        },
    }
    _write_supertask(output_dir / "set1_trial1.json", set1_trial1)

    for spec in set_specs:
        set_name = str(spec["set_name"])
        total_faces = int(spec["total_faces"])
        chunk_sizes = list(spec["chunk_sizes"])
        seen: set[tuple[str, ...]] = set()
        for trial_idx in (1, 2, 3):
            if set_name == "set1" and trial_idx == 1:
                continue
            sampled = _trial_sample(
                all_ids=ids,
                total_faces=total_faces,
                rng=rng,
                seen=seen,
            )
            tasks = _tasks_for_schedule(sampled, initial_count=5, chunk_sizes=chunk_sizes)
            payload = {
                "name": f"supertask_{set_name}_trial{trial_idx}",
                "source": "experiments/generate_supertask_suite.py",
                "scenario": {
                    "type": "on_device_evaluation",
                    "task_grouping": f"{set_name}_trial{trial_idx}",
                },
                "set": set_name,
                "trial": trial_idx,
                "tasks": tasks,
                "dataset_root": "data/val",
                "constraints": {
                    "train_images_per_identity": 50,
                    "test_images_per_identity": 10,
                    "test_start_index": 10,
                },
            }
            _write_supertask(output_dir / f"{set_name}_trial{trial_idx}.json", payload)

    print(f"Generated supertask suite in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
