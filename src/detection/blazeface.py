"""BlazeFace/MediaPipe face detector wrapper."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

LOG = logging.getLogger(__name__)

# Enable per-call latency + RSS logging without code changes (e.g. on Raspberry Pi):
#   export BLAZEFACE_LOG_PERFORMANCE=1
_PERF_ENV = "BLAZEFACE_LOG_PERFORMANCE"
# Verbose Tasks raw bbox/keypoints (before clipping); off by default:
#   export BLAZEFACE_LOG_RAW=1
_RAW_ENV = "BLAZEFACE_LOG_RAW"
_DEFAULT_TASKS_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "blaze_face_full_range.tflite"
)


def _current_rss_bytes() -> Optional[int]:
    """Best-effort resident set size for this process (Linux / RPi: /proc or psutil)."""
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    except (OSError, TypeError, ValueError):
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # VmRSS is in kB on Linux
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _env_flag_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_tasks_modules():
    """Return (python_module, vision_module) for both MediaPipe task layouts."""
    tasks_root = getattr(mp, "tasks", None)
    if tasks_root is None:
        return None, None

    # Layout A (common docs): mediapipe.tasks.python + mediapipe.tasks.python.vision
    if hasattr(tasks_root, "python"):
        python_mod = tasks_root.python
        vision_mod = getattr(python_mod, "vision", None)
        return python_mod, vision_mod

    # Layout B (some builds): mp.tasks already points to mediapipe.tasks.python
    python_mod = tasks_root
    vision_mod = getattr(tasks_root, "vision", None)
    return python_mod, vision_mod


@dataclass
class Detection:
    """Single detection entry."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks_6pt: np.ndarray  # shape (6, 2), MediaPipe BlazeFace keypoint order
    confidence: float


class BlazeFaceDetector:
    """MediaPipe Tasks BlazeFace detector.

    Set ``log_performance=True`` (or env ``BLAZEFACE_LOG_PERFORMANCE=1``) to emit
    INFO logs after each ``detect()`` with wall-clock latency and best-effort RSS.

    Set ``log_raw_inference=True`` (or env ``BLAZEFACE_LOG_RAW=1``) to also log
    raw Tasks bbox + keypoints per internal inference pass (before clipping).

    RSS is process-wide (not incremental allocation for this call alone). For
    steady-state latency, run a few warm-up ``detect()`` calls before recording.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        *,
        log_performance: Optional[bool] = None,
        log_raw_inference: Optional[bool] = None,
        tasks_model_path: Optional[str] = None,
        min_suppression_threshold: float = 0.5,
    ) -> None:
        self.min_confidence = min_confidence
        self._backend = "tasks"
        self._tasks_detector = None
        if log_performance is None:
            log_performance = _env_flag_truthy(_PERF_ENV)
        self._log_performance = bool(log_performance)
        if log_raw_inference is None:
            log_raw_inference = _env_flag_truthy(_RAW_ENV)
        self._log_raw_inference = bool(log_raw_inference)
        if mp is None:
            raise RuntimeError(
                "MediaPipe is required for BlazeFaceDetector. "
                "Install `mediapipe`."
            )

        python_mod, vision_mod = _resolve_tasks_modules()
        if not (
            python_mod is not None
            and vision_mod is not None
            and hasattr(python_mod, "BaseOptions")
            and hasattr(vision_mod, "FaceDetectorOptions")
            and hasattr(vision_mod, "FaceDetector")
        ):
            raise RuntimeError(
                "Installed `mediapipe` runtime does not expose Tasks FaceDetector API "
                "(`mediapipe.tasks.vision.FaceDetector`)."
            )

        model_path = Path(tasks_model_path) if tasks_model_path else _DEFAULT_TASKS_MODEL
        if not model_path.exists():
            raise RuntimeError(
                "MediaPipe Tasks backend detected, but no detector model found. "
                f"Expected model at: {model_path}. "
                "Pass `tasks_model_path=...` to BlazeFaceDetector."
            )
        self._init_tasks_backend(
            python_mod=python_mod,
            vision_mod=vision_mod,
            model_path=model_path,
            min_detection_confidence=min_confidence,
            min_suppression_threshold=min_suppression_threshold,
        )

    def _init_tasks_backend(
        self,
        python_mod,
        vision_mod,
        model_path: Path,
        min_detection_confidence: float,
        min_suppression_threshold: float,
    ) -> None:
        base_options = python_mod.BaseOptions(model_asset_path=str(model_path))
        options = vision_mod.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision_mod.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
        )
        self._tasks_detector = vision_mod.FaceDetector.create_from_options(options)

    def close(self) -> None:
        if self._tasks_detector is not None:
            self._tasks_detector.close()
            self._tasks_detector = None

    def __enter__(self) -> "BlazeFaceDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(self, bgr_frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        detections: List[Detection] = []
        try:
            if bgr_frame is None or bgr_frame.size == 0:
                pass
            else:
                detections = self._detect_mediapipe_tasks(bgr_frame)
            return detections
        finally:
            if getattr(self, "_log_performance", False):
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                shape = getattr(bgr_frame, "shape", None)
                h_w = (
                    f"{shape[0]}x{shape[1]}"
                    if shape is not None and len(shape) >= 2
                    else "n/a"
                )
                rss = _current_rss_bytes()
                rss_mb = f"{rss / (1024 * 1024):.2f}" if rss is not None else "n/a"
                LOG.info(
                    "blazeface.performance backend=%s detections=%d frame=%s latency_ms=%.3f rss_mb=%s",
                    self._backend,
                    len(detections),
                    h_w,
                    elapsed_ms,
                    rss_mb,
                )

    def _detect_mediapipe_tasks(self, bgr_frame: np.ndarray) -> List[Detection]:
        h, w = bgr_frame.shape[:2]
        detections = self._detect_tasks_on_frame(bgr_frame, inference_pass="native")
        if detections:
            return detections

        # Fallback: retry on downscaled frame for very high-res inputs where
        # face occupies too few pixels at native resolution.
        max_dim = max(h, w)
        if max_dim <= 1280:
            return detections

        scale = 1280.0 / float(max_dim)
        sw, sh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr_frame, (sw, sh), interpolation=cv2.INTER_AREA)
        small_detections = self._detect_tasks_on_frame(
            resized, inference_pass="downscale_1280"
        )
        if not small_detections:
            return small_detections

        inv = 1.0 / scale
        upscaled: List[Detection] = []
        for det in small_detections:
            x, y, bw, bh = det.bbox
            ux = max(0, min(int(round(x * inv)), w - 1))
            uy = max(0, min(int(round(y * inv)), h - 1))
            ubw = max(1, min(int(round(bw * inv)), w - ux))
            ubh = max(1, min(int(round(bh * inv)), h - uy))
            ulm = (det.landmarks_6pt * inv).astype(np.float32)
            upscaled.append(
                Detection(bbox=(ux, uy, ubw, ubh), landmarks_6pt=ulm, confidence=det.confidence)
            )
        return upscaled

    def _log_raw_tasks_result(
        self,
        result,
        frame_hw: Tuple[int, int],
        inference_pass: str,
    ) -> None:
        """Log bbox + keypoints exactly as returned by Tasks (before our clipping)."""
        if not getattr(self, "_log_raw_inference", False):
            return
        h, w = frame_hw
        dets = getattr(result, "detections", None) or []
        LOG.info(
            "blazeface.raw pass=%s frame=%dx%d raw_detection_count=%d",
            inference_pass,
            h,
            w,
            len(dets),
        )
        for i, det in enumerate(dets):
            bbox = det.bounding_box
            score = None
            if getattr(det, "categories", None):
                score = float(det.categories[0].score)
            kps = getattr(det, "keypoints", []) or []
            kps_raw: List[Tuple[float, float, Optional[float]]] = []
            for kp in kps:
                z = getattr(kp, "z", None)
                if z is not None:
                    try:
                        zf = float(z)
                    except (TypeError, ValueError):
                        zf = None
                else:
                    zf = None
                kps_raw.append((float(kp.x), float(kp.y), zf))
            LOG.info(
                "blazeface.raw pass=%s det=%d score=%r bbox_raw=(origin_x=%r,origin_y=%r,width=%r,height=%r) "
                "keypoints_raw=%s",
                inference_pass,
                i,
                score,
                bbox.origin_x,
                bbox.origin_y,
                bbox.width,
                bbox.height,
                kps_raw,
            )

    def _detect_tasks_on_frame(
        self, bgr_frame: np.ndarray, *, inference_pass: str = "frame"
    ) -> List[Detection]:
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tasks_detector.detect(mp_image)
        self._log_raw_tasks_result(result, (h, w), inference_pass)
        detections: List[Detection] = []

        if not result.detections:
            return detections

        for det in result.detections:
            conf = float(det.categories[0].score) if getattr(det, "categories", None) else 0.0
            if conf < self.min_confidence:
                continue

            bbox = det.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            keypoints = getattr(det, "keypoints", []) or []
            landmarks = self._keypoints_to_pixel_array(keypoints, w, h)
            detections.append(Detection(bbox=(x, y, bw, bh), landmarks_6pt=landmarks, confidence=conf))
        return detections

    @staticmethod
    def _keypoints_to_pixel_array(keypoints: Sequence, width: int, height: int) -> np.ndarray:
        """Map Tasks keypoints to pixel coordinates; shape (6, 2) in BlazeFace order."""
        points: List[Tuple[float, float]] = []
        for kp in keypoints[:6]:
            xf, yf = float(kp.x), float(kp.y)
            if 0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0:
                points.append((xf * width, yf * height))
            else:
                points.append((xf, yf))
        if len(points) < 6:
            raise ValueError("BlazeFace returned insufficient keypoints (expected 6).")
        return np.array(points, dtype=np.float32)
