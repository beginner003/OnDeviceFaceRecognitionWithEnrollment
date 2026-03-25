"""
Integration test for ExemplarReplayStrategy using FaceRecognitionSystem.

Extracts real embeddings from data/val/ images via embedding_helper,
splits each identity 50/50 into train (registration) and test (evaluation),
registers identities incrementally per supertask_8_2.json task order,
and evaluates recognition accuracy on held-out test embeddings after each task step.

Usage:
    python -m experiments.test_replay_integration
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from experiments.embedding_helper import embed_supertask_identities_to_root
from src.system import FaceRecognitionSystem, SystemConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPERTASK_JSON = REPO_ROOT / "data" / "supertask_8_2.json"
EMBEDDINGS_CACHE = REPO_ROOT / "experiments" / ".cache" / "embeddings"

TRAIN_RATIO = 0.5
MIN_ACCURACY = 0.3
SEED = 42


def _load_task_order() -> OrderedDict[str, list[str]]:
    """Return task_name -> [identity, ...] in insertion order."""
    data = json.loads(SUPERTASK_JSON.read_text(encoding="utf-8"))
    tasks_raw: dict[str, list[str]] = data["tasks"]
    return OrderedDict(sorted(tasks_raw.items()))


def _extract_embeddings(task_order: OrderedDict[str, list[str]]) -> dict[str, np.ndarray]:
    """Extract (or load cached) embeddings for every identity in the supertask."""
    all_ids = [ident for ids in task_order.values() for ident in ids]
    return embed_supertask_identities_to_root(
        supertask_json_path=SUPERTASK_JSON,
        output_root=EMBEDDINGS_CACHE,
        identities_filter=all_ids,
    )


def _train_test_split(
    embeddings: dict[str, np.ndarray],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split each identity's embeddings into train and test sets."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for ident, emb in embeddings.items():
        n = emb.shape[0]
        n_train = max(1, int(n * train_ratio))
        indices = rng.permutation(n)
        train[ident] = emb[indices[:n_train]]
        test[ident] = emb[indices[n_train:]]
    return train, test


def _evaluate(
    system: FaceRecognitionSystem,
    test_embeddings: dict[str, np.ndarray],
    registered: list[str],
) -> tuple[float, int, int]:
    """
    Query held-out test embeddings for every registered identity and return
    (accuracy, correct_count, total_count).
    """
    correct = 0
    total = 0
    for ident in registered:
        emb = test_embeddings[ident].astype(np.float32)
        for i in range(emb.shape[0]):
            pred_name, _conf = system.recognize(emb[i])
            total += 1
            if pred_name == ident:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def main() -> None:
    task_order = _load_task_order()
    print(f"Task order: {dict(task_order)}\n")

    print("Extracting embeddings (cached after first run)...")
    all_embeddings = _extract_embeddings(task_order)
    print(f"  Got embeddings for {len(all_embeddings)} identities.")

    train_emb, test_emb = _train_test_split(all_embeddings)
    for ident in sorted(train_emb):
        print(f"    {ident}: {train_emb[ident].shape[0]} train, {test_emb[ident].shape[0]} test")
    print()

    workspace = Path(tempfile.mkdtemp(prefix="replay_integration_"))
    try:
        config = SystemConfig(
            registration="replay",
            exemplar_selection="herding",
            recognition="classifier",
            confidence_threshold=0.0,
        )
        system = FaceRecognitionSystem.from_config(config, workspace=workspace)

        registered: list[str] = []
        results: list[dict] = []

        for task_name, identities in task_order.items():
            for ident in identities:
                emb = train_emb[ident]
                res = system.register(ident, emb)
                registered.append(ident)
                print(
                    f"  Registered {ident} "
                    f"({res.selected_count} exemplars, {res.elapsed_s:.2f}s)"
                )

            acc, correct, total = _evaluate(system, test_emb, registered)
            results.append({
                "task": task_name,
                "identities": len(registered),
                "accuracy": acc,
                "correct": correct,
                "total": total,
            })
            print(
                f"  -> After {task_name}: "
                f"{len(registered)} identities, "
                f"accuracy={acc:.2f} ({correct}/{total})\n"
            )

        print("=" * 60)
        print("Summary  (evaluated on held-out test split)")
        print("=" * 60)
        for r in results:
            print(
                f"  {r['task']}: {r['identities']} ids, "
                f"acc={r['accuracy']:.2f} ({r['correct']}/{r['total']})"
            )

        final_acc = results[-1]["accuracy"]
        print(f"\nFinal accuracy: {final_acc:.2f}")
        assert final_acc >= MIN_ACCURACY, (
            f"Final accuracy {final_acc:.2f} below minimum {MIN_ACCURACY}"
        )
        print("PASSED")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
