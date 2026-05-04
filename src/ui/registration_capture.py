"""Phased registration capture: frontal + left/right yaw diversity before a single enroll call."""

from __future__ import annotations

import os
import time
from typing import Callable, List, Literal, Tuple

import numpy as np

from src.capture.realsense import RealSenseCapture
from src.detection.blazeface import BlazeFaceDetector, Detection
from src.ui.pipeline import VisionPipeline

PosePhase = Literal["center", "left", "right"]

ProgressCb = Callable[[dict], None]

_PHASE_INSTRUCTIONS: dict[PosePhase, str] = {
    "center": "Look straight at the camera — hold still.",
    "left": "Slowly turn your head to your LEFT (ear toward the shoulder).",
    "right": "Slowly turn your head to your RIGHT (ear toward the shoulder).",
}


def registration_phase_instruction(phase: PosePhase) -> str:
    return _PHASE_INSTRUCTIONS[phase]


def quotas_three_way(total: int) -> Tuple[int, int, int]:
    """Split embeddings across center / left / right; extras go to the frontal phase."""

    if total < 6:
        # Need at least 2 poses each for diversity
        raise ValueError("n_frames must be at least 6 for three-phase registration")
    base = total // 3
    rem = total % 3
    center = base + rem
    left = base
    right = base
    return center, left, right


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


def yaw_matches_pose(
    y: float,
    pose: PosePhase,
    *,
    center_abs_max: float,
    left_below_neg: float,
    right_above_pos: float,
) -> bool:
    if pose == "center":
        return abs(y) <= center_abs_max
    if pose == "left":
        return y <= -left_below_neg
    if pose == "right":
        return y >= right_above_pos
    return False


def _near_duplicate(embeddings: List[np.ndarray], candidate: np.ndarray, thr: float) -> bool:
    if not embeddings:
        return False
    stacked = np.stack(embeddings, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0)
    nm = np.linalg.norm(mean) + 1e-12
    c = candidate.astype(np.float64)
    sim = float(np.dot(mean / nm, c))
    return sim >= thr


def load_yaw_thresholds() -> Tuple[float, float, float]:
    """ENV: center max |norm yaw|, magnitude for left, magnitude for right (positive floats)."""

    cmax = float(os.environ.get("FACE_UI_REG_YAW_CENTER_MAX", "0.22"))
    lm = float(os.environ.get("FACE_UI_REG_YAW_LEFT", "0.07"))
    rm = float(os.environ.get("FACE_UI_REG_YAW_RIGHT", "0.07"))
    return cmax, lm, rm


def collect_embeddings_for_pose(
    *,
    capture: RealSenseCapture,
    detector: BlazeFaceDetector,
    pipeline: VisionPipeline,
    pose: PosePhase,
    quota: int,
    total_target: int,
    baseline_collected: int,
    similarity_dup: float,
    stale_limit: int,
    bump: ProgressCb,
) -> List[np.ndarray]:
    """Gather `quota` L2-normalized embeddings while the user's face matches `pose`."""

    cmax, lmag, rmag = load_yaw_thresholds()
    instruct = registration_phase_instruction(pose)
    bucket: List[np.ndarray] = []
    stale = 0
    idle_sleep = float(os.environ.get("FACE_UI_REG_POLL_S", "0.03"))

    while len(bucket) < quota and stale < stale_limit:
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
        if not yaw_matches_pose(
            y, pose, center_abs_max=cmax, left_below_neg=lmag, right_above_pos=rmag
        ):
            stale += 1
            if idle_sleep > 0:
                time.sleep(idle_sleep)
            continue

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
                "pose": pose,
                "instruction": instruct,
                "yaw_estimate": round(y, 4),
                "quota_phase_target": quota,
                "quota_phase_current": len(bucket),
                "current": baseline_collected + len(bucket),
                "target": total_target,
                "message": instruct,
            }
        )

        if idle_sleep > 0:
            time.sleep(idle_sleep)

    return bucket
