"""Tests for face alignment (eye-line rotation using detector bbox + landmarks)."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason="OpenCV required: pip install opencv-python-headless (use `python -m pytest` so the same env has cv2)",
)

from src.alignment import FaceAligner
from tests.support.alignment_embedding_fixtures import (
    PositiveLandmarkCase,
    iter_positive_landmark_cases,
)

LOG = logging.getLogger(__name__)

DETECTION_POSITIVE_DIR = Path(__file__).parent / "data" / "detection" / "positive"
AE_ARTIFACTS = Path(__file__).parent / "data" / "alignment_embedding" / "artifacts"
AE_VIS_DIR = AE_ARTIFACTS / "visualizations"
AE_LOGS_DIR = AE_ARTIFACTS / "logs"
ALIGNMENT_LOG_PATH = AE_LOGS_DIR / "alignment_results.log"

_LANDMARK_6_LABELS = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right ear",
    "left ear",
)


def _ensure_ae_artifact_dirs() -> None:
    AE_VIS_DIR.mkdir(parents=True, exist_ok=True)
    AE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(prefix: str, index: int, case: PositiveLandmarkCase) -> str:
    raw = f"{prefix}_{index:02d}_{Path(case.image_name).stem}_face{case.face_index}"
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return safe or f"{prefix}_{index:02d}"


def _landmark_log_lines(landmarks: np.ndarray) -> list[str]:
    lines = []
    for i, name in enumerate(_LANDMARK_6_LABELS):
        px, py = float(landmarks[i, 0]), float(landmarks[i, 1])
        lines.append(f"      {name}: ({px:.4f}, {py:.4f})")
    return lines


def _write_image_header(log_file, case: PositiveLandmarkCase, bgr: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    log_file.write("\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(
        f"CASE  file={case.image_name}  face_index={case.face_index}  "
        f"frame={w}x{h}  expected_frame={case.frame_width}x{case.frame_height}\n"
    )
    log_file.write("=" * 80 + "\n\n")


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


@pytest.fixture
def alignment_results_log(request):
    """Append integration run banner to alignment_results.log."""
    _ensure_ae_artifact_dirs()
    ALIGNMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALIGNMENT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n{'#' * 80}\n"
            f"# alignment integration  {datetime.now(timezone.utc).isoformat()}\n"
            f"# test: {request.node.nodeid}\n"
            f"{'#' * 80}\n"
        )
        fh.flush()
        yield fh


def test_estimate_transform_rejects_non_6x2_landmarks() -> None:
    aligner = FaceAligner()
    with pytest.raises(ValueError, match=r"Expected landmarks shape \(6, 2\)"):
        aligner.estimate_transform(np.zeros((5, 2), dtype=np.float32))


def test_align_rejects_empty_image() -> None:
    aligner = FaceAligner()
    pts = np.zeros((6, 2), dtype=np.float32)
    bbox = (0, 0, 10, 10)
    with pytest.raises(ValueError, match="empty"):
        aligner.align(np.array([], dtype=np.uint8), pts, bbox)
    with pytest.raises(ValueError, match="empty"):
        aligner.align(None, pts, bbox)  # type: ignore[arg-type]


def test_align_rejects_invalid_bbox() -> None:
    aligner = FaceAligner()
    pts = np.zeros((6, 2), dtype=np.float32)
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox_xywh"):
        aligner.align(bgr, pts, (1, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bbox_xywh"):
        aligner.align(bgr, pts, None)  # type: ignore[arg-type]


def test_align_output_shape_and_model_input_range() -> None:
    """In-memory frame matching pos1 geometry; landmarks from frozen detection log."""
    case = iter_positive_landmark_cases()[0]
    h, w = case.frame_height, case.frame_width
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:] = (40, 40, 40)

    aligner = FaceAligner()
    t0 = time.perf_counter()
    result = aligner.align(bgr, case.landmarks_6x2, case.bbox_xywh)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert result.aligned_bgr.shape == (112, 112, 3)
    assert result.aligned_bgr.dtype == np.uint8
    assert result.transform.shape == (2, 3)

    tensor = FaceAligner.to_model_input(result.aligned_bgr)
    assert tensor.shape == (112, 112, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0
    LOG.info("alignment_synthetic_frame latency_ms=%.3f", elapsed_ms)


@pytest.mark.integration
def test_positive_fixture_landmarks_align_logs_latency_and_writes_artifacts(
    alignment_results_log,
) -> None:
    cases = iter_positive_landmark_cases()
    missing = [c.image_name for c in cases if not (DETECTION_POSITIVE_DIR / c.image_name).is_file()]
    if missing:
        pytest.skip(f"Missing positive fixture images: {sorted(set(missing))}")

    _ensure_ae_artifact_dirs()
    aligner = FaceAligner()

    for idx, case in enumerate(cases):
        path = DETECTION_POSITIVE_DIR / case.image_name
        bgr = _load_bgr(path)
        h, w = bgr.shape[:2]
        assert w == case.frame_width and h == case.frame_height, (
            f"Fixture geometry mismatch for {case.image_name}: got {w}x{h}, "
            f"expected {case.frame_width}x{case.frame_height}"
        )

        t0 = time.perf_counter()
        result = aligner.align(bgr, case.landmarks_6x2, case.bbox_xywh)
        align_ms = (time.perf_counter() - t0) * 1000.0

        out_path = AE_VIS_DIR / f"{_safe_stem('aligned', idx, case)}.png"
        assert cv2.imwrite(str(out_path), result.aligned_bgr), f"Failed to write {out_path}"

        if alignment_results_log is not None:
            _write_image_header(alignment_results_log, case, bgr)
            alignment_results_log.write("performance:\n")
            alignment_results_log.write(f"  alignment.latency_ms={align_ms:.3f}\n\n")
            bx, by, bw, bh = case.bbox_xywh
            alignment_results_log.write(f"bbox_xywh=({bx},{by},{bw},{bh})\n\n")
            alignment_results_log.write("landmarks_6_px (BlazeFace / MediaPipe order):\n")
            alignment_results_log.write("\n".join(_landmark_log_lines(case.landmarks_6x2)))
            alignment_results_log.write("\n\nsummary:\n")
            alignment_results_log.write(f"  aligned_bgr_shape={result.aligned_bgr.shape}\n")
            alignment_results_log.write(f"  transform_shape={result.transform.shape}\n")
            alignment_results_log.write(f"  artifact_path={out_path.name}\n\n")
            alignment_results_log.flush()

        LOG.info(
            "alignment_fixture file=%s face=%d latency_ms=%.3f artifact=%s",
            case.image_name,
            case.face_index,
            align_ms,
            out_path.name,
        )
