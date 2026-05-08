"""Registration capture helpers for fast, capture-first enrollment."""

from __future__ import annotations

import time
from typing import Callable, List, Literal

import numpy as np

from src.capture.realsense import RealSenseCapture
from src.detection.blazeface import BlazeFaceDetector, Detection
from src.ui.pipeline import VisionPipeline

PosePhase = Literal["capture"]
ProgressCb = Callable[[dict], None]
_INSTRUCTION = "Keep your face in view; make slight left/right turns for variety."


def registration_phase_instruction(_phase: PosePhase = "capture") -> str:
    return _INSTRUCTION


def nose_yaw_ratio(det: Detection) -> float:
    """
    Cheap yaw proxy from BlazeFace 6 landmarks: lateral nose offset normalized by eye distance.
    Roughly symmetric at ~0; negative when nose shifts toward image left (viewer perspective).
    """

    lm = det.landmarks_6pt.astype(np.float64, copy=False)
    re, le, nose = lm[0], lm[1], lm[2]
    mid = (re + le) * 0.5
    ipd = float(np.linalg.norm(le - re)) + 1e-6
    return float((nose[0] - mid[0]) / ipd)


def _near_duplicate(embeddings: List[np.ndarray], candidate: np.ndarray, thr: float) -> bool:
    if not embeddings:
        return False
    stacked = np.stack(embeddings, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0)
    nm = np.linalg.norm(mean) + 1e-12
    c = candidate.astype(np.float64)
    sim = float(np.dot(mean / nm, c))
    return sim >= thr


def collect_registration_embeddings(
    *,
    capture: RealSenseCapture,
    detector: BlazeFaceDetector,
    pipeline: VisionPipeline,
    target: int,
    similarity_dup: float,
    stale_limit: int,
    idle_sleep: float = 0.02,
    bump: ProgressCb,
) -> List[np.ndarray]:
    """Gather `target` L2-normalized embeddings before training starts."""

    if target < 1:
        raise ValueError("target must be >= 1")
    instruct = registration_phase_instruction()
    bucket: List[np.ndarray] = []
    stale = 0

    while len(bucket) < target and stale < stale_limit:
        pkt = capture.read()
        if pkt is None:
            stale += 1
            if idle_sleep > 0:
                time.sleep(idle_sleep)
            continue
        detections = detector.detect(pkt.bgr)
        face = pipeline.pick_largest(detections)
        if face is None:
            stale += 1
            if idle_sleep > 0:
                time.sleep(idle_sleep)
            continue

        y = nose_yaw_ratio(face)
        emb = pipeline.embed_detection(pkt.bgr, face).astype(np.float32, copy=False)
        if _near_duplicate(bucket, emb, similarity_dup):
            stale += 1
            if idle_sleep > 0:
                time.sleep(idle_sleep)
            continue

        bucket.append(emb)
        stale = 0
        bump(
            {
                "phase": "capturing",
                "pose": "capture",
                "instruction": instruct,
                "yaw_estimate": round(y, 4),
                "quota_phase_target": target,
                "quota_phase_current": len(bucket),
                "current": len(bucket),
                "target": target,
                "message": instruct,
            }
        )

        if idle_sleep > 0:
            time.sleep(idle_sleep)

    return bucket
