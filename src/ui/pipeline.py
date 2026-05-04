"""Detect → align → embed helpers for live frames."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.alignment import FaceAligner
from src.detection.blazeface import Detection
from src.embedding import MobileFaceNetEmbedder


@dataclass(frozen=True)
class PipelineArtifact:
    """Intermediate result for one detected face in a live demo frame."""

    embedding: np.ndarray
    aligned_bgr: np.ndarray
    timings_ms: dict[str, float]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_mobilefacenet_path(explicit: str | Path | None = None) -> Path:
    """Match experiments/embedding_helper resolution logic without importing experiments."""

    models_dir = _repo_root() / "src" / "models"

    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MobileFaceNet model not found: {p}")
        return p

    env = os.environ.get("MOBILEFACENET_TFLITE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    variant = os.environ.get("MOBILEFACENET_MODEL_VARIANT", "float32").strip().lower()
    if variant not in {"float32", "int8"}:
        raise ValueError("MOBILEFACENET_MODEL_VARIANT must be 'float32' or 'int8'")

    candidates = {
        "float32": ("mobilefacenet_float32.tflite", "mobilefacenet.tflite"),
        "int8": ("mobilefacenet_int8.tflite",),
    }
    for name in candidates[variant]:
        p = models_dir / name
        if p.is_file():
            return p

    hints = sorted(models_dir.glob("*.tflite"))
    hints = [p for p in hints if variant in p.stem.lower()]
    if hints:
        return hints[0]

    fallback = models_dir / "mobilefacenet.tflite"
    if variant == "float32" and fallback.is_file():
        return fallback

    raise FileNotFoundError(
        "No MobileFaceNet .tflite under src/models. Set MOBILEFACENET_TFLITE "
        "or add mobilefacenet.tflite."
    )


class VisionPipeline:
    """Runs alignment + embedding given BlazeFace `Detection` objects."""

    def __init__(
        self,
        embedder: MobileFaceNetEmbedder,
        *,
        expand_ratio: float = 1.5,
    ) -> None:
        self.embedder = embedder
        self.aligner = FaceAligner(output_size=(112, 112), expand_ratio=expand_ratio)

    @staticmethod
    def pick_largest(detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda d: int(d.bbox[2]) * int(d.bbox[3]))

    def embed_detection(self, bgr: np.ndarray, det: Detection) -> np.ndarray:
        """Return L2-normalised (128,) float32 embedding."""

        return self.inspect_detection(bgr, det).embedding

    def inspect_detection(self, bgr: np.ndarray, det: Detection) -> PipelineArtifact:
        """Return embedding plus aligned crop and stage timings for UI display."""

        t0 = time.perf_counter()
        aligned = self.aligner.align(
            bgr_image=bgr,
            landmarks_6pt=det.landmarks_6pt,
            bbox_xywh=det.bbox,
        )
        t1 = time.perf_counter()
        tensor = FaceAligner.to_model_input(aligned.aligned_bgr)
        batch = np.stack([tensor], axis=0).astype(np.float32)
        out = self.embedder.embed_batch(batch)
        t2 = time.perf_counter()
        return PipelineArtifact(
            embedding=out[0].astype(np.float32, copy=False),
            aligned_bgr=aligned.aligned_bgr,
            timings_ms={
                "align": round((t1 - t0) * 1000.0, 2),
                "embed": round((t2 - t1) * 1000.0, 2),
            },
        )

    @staticmethod
    def detection_to_dict(det: Detection, *, index: int, primary: bool = False) -> dict:
        x, y, w, h = [int(v) for v in det.bbox]
        landmarks = np.asarray(det.landmarks_6pt, dtype=np.float32)
        return {
            "index": int(index),
            "primary": bool(primary),
            "confidence": round(float(det.confidence), 4),
            "bbox_xywh": [x, y, w, h],
            "landmarks_6pt": [
                [round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in landmarks
            ],
        }

    @staticmethod
    def annotate_frame(
        bgr: np.ndarray,
        detections: List[Detection],
        labels: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """Draw boxes and optional per-face labels (BGR in/out)."""

        import cv2

        out = bgr.copy()
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = (72, 189, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            for pt in np.asarray(det.landmarks_6pt, dtype=np.int32):
                cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (48, 232, 144), -1)
            label = None
            if labels is not None and i < len(labels):
                label = labels[i]
            if not label:
                label = f"face {i + 1} {float(det.confidence):.2f}"
            if label:
                cv2.putText(
                    out,
                    label,
                    (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return out
