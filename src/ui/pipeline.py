"""Detect → align → embed helpers for live frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.alignment import FaceAligner
from src.detection.blazeface import Detection
from src.embedding import MobileFaceNetEmbedder


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

        aligned = self.aligner.align(
            bgr_image=bgr,
            landmarks_6pt=det.landmarks_6pt,
            bbox_xywh=det.bbox,
        )
        tensor = FaceAligner.to_model_input(aligned.aligned_bgr)
        batch = np.stack([tensor], axis=0).astype(np.float32)
        out = self.embedder.embed_batch(batch)
        return out[0].astype(np.float32, copy=False)

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
            label = None
            if labels is not None and i < len(labels):
                label = labels[i]
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
