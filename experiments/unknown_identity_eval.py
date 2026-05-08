from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from experiments.embedding_helper import embed_supertask_identities_to_root


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _all_dataset_identities(dataset_root: Path) -> list[str]:
    ids = sorted(p.name for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("n"))
    if not ids:
        raise RuntimeError(f"No identity folders found in dataset root: {dataset_root}")
    return ids


def _sample_unknown_identities(
    *,
    supertask_path: Path,
    known_identities: Sequence[str],
    unknown_count: int,
    seed: int,
) -> tuple[list[str], dict]:
    payload = json.loads(supertask_path.read_text(encoding="utf-8"))
    dataset_root = Path(payload.get("dataset_root", "data/val"))
    if not dataset_root.is_absolute():
        dataset_root = (_repo_root() / dataset_root).resolve()

    all_ids = _all_dataset_identities(dataset_root)
    known_set = set(str(x) for x in known_identities)
    pool = [x for x in all_ids if x not in known_set]
    if len(pool) < unknown_count:
        raise RuntimeError(
            f"Need {unknown_count} unknown identities but only {len(pool)} available "
            f"(known={len(known_set)}, dataset_total={len(all_ids)})."
        )

    rng = np.random.default_rng(int(seed))
    sampled = sorted(rng.choice(np.asarray(pool), size=int(unknown_count), replace=False).tolist())
    compact = {
        "name": f"unknown_eval_{supertask_path.stem}",
        "source": "experiments/unknown_identity_eval.py",
        "scenario": {"type": "unknown_identity_rejection"},
        "tasks": {"task_unknown": sampled},
        "dataset_root": str(dataset_root),
        "constraints": payload.get(
            "constraints",
            {
                "train_images_per_identity": 50,
                "test_images_per_identity": 10,
                "test_start_index": 10,
            },
        ),
    }
    return sampled, compact


def evaluate_unknown_identity_rejection(
    *,
    system,
    supertask_path: Path,
    embeddings_root: Path,
    known_identities: Sequence[str],
    unknown_count: int,
    seed: int,
    overwrite_embeddings: bool,
    metrics_logger: logging.Logger,
    progress_logger: logging.Logger | None = None,
) -> Dict[str, float]:
    if int(unknown_count) <= 0:
        metrics_logger.info("Unknown identity test skipped (unknown_count <= 0).")
        return {}

    sampled, compact_payload = _sample_unknown_identities(
        supertask_path=supertask_path,
        known_identities=known_identities,
        unknown_count=int(unknown_count),
        seed=int(seed),
    )

    unknown_root = embeddings_root / "_unknown_identity_eval"
    unknown_root.mkdir(parents=True, exist_ok=True)
    compact_path = unknown_root / f"{supertask_path.stem}_unknown_ids.json"
    compact_path.write_text(json.dumps(compact_payload, indent=2) + "\n", encoding="utf-8")

    if progress_logger is not None:
        progress_logger.info(
            "Unknown-identity test: embedding %d held-out identities.",
            len(sampled),
        )
    unknown_test_embeddings: Mapping[str, np.ndarray] = embed_supertask_identities_to_root(
        supertask_json_path=compact_path,
        output_root=unknown_root / "embeddings",
        split="test",
        overwrite=bool(overwrite_embeddings),
    )

    total = 0
    rejected = 0
    per_identity: Dict[str, float] = {}
    false_accept_counts: Dict[str, int] = {}
    confidence_sum = 0.0
    confidence_rejected_sum = 0.0
    confidence_accepted_sum = 0.0

    for ident in sampled:
        embs = np.asarray(unknown_test_embeddings[ident], dtype=np.float32)
        if embs.ndim != 2:
            raise RuntimeError(f"Unknown test embeddings for {ident} must be 2D, got {embs.shape}")
        id_total = int(embs.shape[0])
        id_rejected = 0
        for i in range(id_total):
            pred, conf = system.recognize(embs[i])
            pred_s = str(pred)
            conf_f = float(conf)
            total += 1
            confidence_sum += conf_f
            if pred_s == "unknown":
                rejected += 1
                id_rejected += 1
                confidence_rejected_sum += conf_f
            else:
                false_accept_counts[pred_s] = false_accept_counts.get(pred_s, 0) + 1
                confidence_accepted_sum += conf_f
        per_identity[ident] = float(id_rejected) / float(id_total) if id_total > 0 else 0.0

    false_accepts = total - rejected
    rejection_accuracy = float(rejected) / float(total) if total > 0 else 0.0
    false_accept_rate = float(false_accepts) / float(total) if total > 0 else 0.0
    mean_conf = confidence_sum / float(total) if total > 0 else float("nan")
    mean_conf_rej = confidence_rejected_sum / float(rejected) if rejected > 0 else float("nan")
    mean_conf_accept = confidence_accepted_sum / float(false_accepts) if false_accepts > 0 else float("nan")

    metrics_logger.info("Unknown identity evaluation")
    metrics_logger.info("  known identities in run: %d", len(known_identities))
    metrics_logger.info("  unknown identities tested: %d", len(sampled))
    metrics_logger.info("  unknown identities list: %s", ", ".join(sampled))
    for ident in sampled:
        metrics_logger.info("  unknown_rejection_accuracy[%s]: %.4f", ident, per_identity[ident])
    metrics_logger.info("  unknown_total_samples: %d", total)
    metrics_logger.info("  unknown_rejected_samples: %d", rejected)
    metrics_logger.info("  unknown_false_accept_samples: %d", false_accepts)
    metrics_logger.info("  unknown_rejection_accuracy: %.4f", rejection_accuracy)
    metrics_logger.info("  unknown_false_accept_rate: %.4f", false_accept_rate)
    metrics_logger.info("  unknown_mean_confidence: %.4f", mean_conf)
    metrics_logger.info("  unknown_mean_confidence_rejected: %.4f", mean_conf_rej)
    metrics_logger.info("  unknown_mean_confidence_false_accept: %.4f", mean_conf_accept)
    if false_accept_counts:
        for pred, count in sorted(false_accept_counts.items(), key=lambda x: (-x[1], x[0])):
            metrics_logger.info("  unknown_false_accept_as[%s]: %d", pred, count)
    else:
        metrics_logger.info("  unknown_false_accept_as: none")

    return {
        "unknown_total_samples": float(total),
        "unknown_rejected_samples": float(rejected),
        "unknown_false_accept_samples": float(false_accepts),
        "unknown_rejection_accuracy": rejection_accuracy,
        "unknown_false_accept_rate": false_accept_rate,
    }
