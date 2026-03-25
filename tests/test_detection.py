"""Tests for the face detection module.

These tests intentionally split into:
- unit tests (no MediaPipe runtime required)
- optional integration tests (run only when image fixtures + MediaPipe exist)

On success, ``test_positive_fixture_images_have_face_detections`` regenerates
``tests/support/alignment_embedding_fixtures.py`` from live detections unless
``ALIGNMENT_FIXTURES_NO_AUTO_UPDATE=1`` is set.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason="OpenCV required: pip install opencv-python-headless (use `python -m pytest` so the same env has cv2)",
)

from src.detection import blazeface as blazeface_module
from src.detection.blazeface import BlazeFaceDetector, Detection
from tests.support.alignment_fixtures_writer import (
    alignment_fixtures_auto_update_disabled,
    build_alignment_embedding_fixtures_source,
    write_alignment_embedding_fixtures,
)

LOG = logging.getLogger(__name__)
BLAZEFACE_LOGGER_NAME = blazeface_module.LOG.name

TEST_DATA_DIR = Path(__file__).parent / "data" / "detection"
POSITIVE_DIR = TEST_DATA_DIR / "positive"
NEGATIVE_DIR = TEST_DATA_DIR / "negative"
ARTIFACTS_DIR = TEST_DATA_DIR / "artifacts"
VISUALIZATIONS_DIR = ARTIFACTS_DIR / "visualizations"
LOGS_DIR = ARTIFACTS_DIR / "logs"
RESULTS_LOG_PATH = LOGS_DIR / "detection_results.log"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# MediaPipe BlazeFace keypoint order (``Detection.landmarks_6pt`` rows).
_LANDMARK_6_LABELS = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "right ear",
    "left ear",
)


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    )


def _require_mediapipe() -> None:
    """Skip integration tests gracefully when MediaPipe is unavailable."""
    pytest.importorskip("mediapipe")
    if blazeface_module.mp is None:
        pytest.skip(
            "Installed mediapipe runtime does not expose Tasks FaceDetector API."
        )
    tasks_python, tasks_vision = blazeface_module._resolve_tasks_modules()
    has_tasks = (
        tasks_python is not None
        and tasks_vision is not None
        and hasattr(tasks_python, "BaseOptions")
        and hasattr(tasks_vision, "FaceDetectorOptions")
        and hasattr(tasks_vision, "FaceDetector")
    )
    if not has_tasks:
        pytest.skip(
            "Installed mediapipe runtime does not expose Tasks FaceDetector API."
        )


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _ensure_artifact_dirs() -> None:
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_visualization_stem(prefix: str, index: int, source: Path) -> str:
    """Unique, filesystem-safe base name for saved visualization PNGs."""
    raw = f"{prefix}_{index:02d}_{source.stem}"
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return safe or f"{prefix}_{index:02d}"


def _annotate_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes, confidence labels, and 6 BlazeFace keypoints."""
    out = image_bgr.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{i + 1}: {det.confidence:.3f}"
        cv2.putText(
            out,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        for li, (px, py) in enumerate(det.landmarks_6pt):
            color = (0, 255, 255) if li < 2 else (0, 165, 255)
            cv2.circle(out, (int(px), int(py)), 3, color, -1, lineType=cv2.LINE_AA)
    if not detections:
        cv2.putText(
            out,
            "no detections",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (128, 128, 128),
            2,
            lineType=cv2.LINE_AA,
        )
    return out


def _write_image_log_section_header(log_file, category: str, image_path: Path, bgr: np.ndarray) -> None:
    """Start a clearly separated block in detection_results.log for one image."""
    h, w = bgr.shape[:2]
    log_file.write("\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"IMAGE  category={category}  file={image_path.name}  frame={w}x{h}\n")
    log_file.write("=" * 80 + "\n\n")


def _format_landmark_lines(det: Detection) -> list[str]:
    lines = []
    for i, name in enumerate(_LANDMARK_6_LABELS):
        px, py = float(det.landmarks_6pt[i, 0]), float(det.landmarks_6pt[i, 1])
        lines.append(f"      {name}: ({px:.4f}, {py:.4f})")
    return lines


def _format_detection_file_block(detections: list[Detection]) -> str:
    lines = [f"pipeline_output: count={len(detections)}"]
    for j, det in enumerate(detections):
        x, y, bw, bh = det.bbox
        lines.append(
            f"  [{j + 1}] bbox_xywh=({x},{y},{bw},{bh})  conf={det.confidence:.6f}"
        )
        lines.append(
            "      landmarks_6_px (right_eye, left_eye, nose_tip, mouth_center, right ear, left ear):"
        )
        lines.extend(_format_landmark_lines(det))
    if not detections:
        lines.append("  (no faces above threshold)")
    return "\n".join(lines) + "\n"


def _flush_blazeface_performance_to_file(caplog: pytest.LogCaptureFixture, log_file) -> None:
    """Append only blazeface.performance lines (not raw Tasks dumps) to the log file."""
    if log_file is None:
        return
    log_file.write("performance:\n")
    wrote = False
    for rec in caplog.records:
        if rec.name != BLAZEFACE_LOGGER_NAME:
            continue
        msg = rec.getMessage()
        if "blazeface.performance" in msg:
            log_file.write(f"  {msg}\n")
            wrote = True
    if not wrote:
        log_file.write("  (no blazeface.performance line captured)\n")
    log_file.write("\n")
    log_file.flush()


def _log_integration_image_result(
    caplog: pytest.LogCaptureFixture,
    log_file,
    category: str,
    image_path: Path,
    bgr: np.ndarray,
    detections: list[Detection],
) -> None:
    """Write one separated section to the log file; concise line to pytest logger."""
    if log_file is not None:
        _write_image_log_section_header(log_file, category, image_path, bgr)
        _flush_blazeface_performance_to_file(caplog, log_file)
        log_file.write("summary:\n")
        log_file.write(_format_detection_file_block(detections))
        log_file.write("\n")
        log_file.flush()
    LOG.info(
        "detection_result category=%s file=%s count=%d",
        category,
        image_path.name,
        len(detections),
    )


@pytest.fixture
def detection_results_log(request):
    """Append integration run banner + lines to artifacts/logs/detection_results.log."""
    _ensure_artifact_dirs()
    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n{'#' * 80}\n"
            f"# detection integration  {datetime.now(timezone.utc).isoformat()}\n"
            f"# test: {request.node.nodeid}\n"
            f"{'#' * 80}\n"
        )
        fh.flush()
        yield fh


def test_detector_init_raises_without_mediapipe() -> None:
    with patch("src.detection.blazeface.mp", None):
        with pytest.raises(RuntimeError, match="MediaPipe is required"):
            BlazeFaceDetector()


def test_detect_returns_empty_on_none_or_empty_frame() -> None:
    # Bypass __init__ because this is purely input validation behavior.
    detector = BlazeFaceDetector.__new__(BlazeFaceDetector)
    detector.min_confidence = 0.7
    detector._log_performance = False

    assert detector.detect(None) == []
    assert detector.detect(np.array([], dtype=np.uint8)) == []


def test_keypoints_to_pixel_array_normalized_coords() -> None:
    keypoints = [
        SimpleNamespace(x=0.70, y=0.30),
        SimpleNamespace(x=0.30, y=0.30),
        SimpleNamespace(x=0.50, y=0.50),
        SimpleNamespace(x=0.50, y=0.70),
        SimpleNamespace(x=0.85, y=0.50),
        SimpleNamespace(x=0.15, y=0.50),
    ]
    width, height = 100, 80

    landmarks = BlazeFaceDetector._keypoints_to_pixel_array(keypoints, width, height)

    assert landmarks.shape == (6, 2)
    assert landmarks.dtype == np.float32
    np.testing.assert_allclose(landmarks[0], np.array([70.0, 24.0], dtype=np.float32))
    np.testing.assert_allclose(landmarks[1], np.array([30.0, 24.0], dtype=np.float32))
    np.testing.assert_allclose(landmarks[2], np.array([50.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(landmarks[3], np.array([50.0, 56.0], dtype=np.float32))
    np.testing.assert_allclose(landmarks[4], np.array([85.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(landmarks[5], np.array([15.0, 40.0], dtype=np.float32))


def test_build_alignment_embedding_fixtures_source_is_valid_python() -> None:
    lm = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        dtype=np.float32,
    )
    cases = [("x.png", 0, 640, 480, (10, 20, 30, 40), lm)]
    src = build_alignment_embedding_fixtures_source(cases)
    compile(src, "alignment_embedding_fixtures.py", "exec")


def test_alignment_fixtures_auto_update_disabled_env() -> None:
    import os

    prev = os.environ.get("ALIGNMENT_FIXTURES_NO_AUTO_UPDATE")
    try:
        os.environ["ALIGNMENT_FIXTURES_NO_AUTO_UPDATE"] = "1"
        assert alignment_fixtures_auto_update_disabled() is True
        del os.environ["ALIGNMENT_FIXTURES_NO_AUTO_UPDATE"]
        assert alignment_fixtures_auto_update_disabled() is False
    finally:
        if prev is None:
            os.environ.pop("ALIGNMENT_FIXTURES_NO_AUTO_UPDATE", None)
        else:
            os.environ["ALIGNMENT_FIXTURES_NO_AUTO_UPDATE"] = prev


def test_keypoints_to_pixel_array_raises_on_insufficient_keypoints() -> None:
    keypoints = [SimpleNamespace(x=0.5, y=0.5) for _ in range(5)]
    with pytest.raises(ValueError, match="insufficient keypoints"):
        BlazeFaceDetector._keypoints_to_pixel_array(keypoints, width=100, height=100)


@pytest.mark.integration
def test_positive_fixture_images_have_face_detections(
    caplog: pytest.LogCaptureFixture,
    detection_results_log,
) -> None:
    _require_mediapipe()
    caplog.set_level(logging.INFO, logger=__name__)
    caplog.set_level(logging.INFO, logger=BLAZEFACE_LOGGER_NAME)

    images = _list_images(POSITIVE_DIR)
    if not images:
        pytest.skip(f"No positive fixture images found in {POSITIVE_DIR}")

    _ensure_artifact_dirs()
    fixture_rows: list[
        tuple[str, int, int, int, tuple[int, int, int, int], np.ndarray]
    ] = []
    with BlazeFaceDetector(
        min_confidence=0.5,
        log_performance=True,
        log_raw_inference=False,
    ) as detector:
        for idx, image_path in enumerate(images):
            bgr = _load_bgr(image_path)
            caplog.clear()
            detections = detector.detect(bgr)
            _log_integration_image_result(
                caplog,
                detection_results_log,
                "positive",
                image_path,
                bgr,
                detections,
            )
            assert len(detections) >= 1, f"No face detected in positive fixture: {image_path.name}"
            for det in detections:
                assert isinstance(det, Detection)
                assert len(det.bbox) == 4
                assert det.landmarks_6pt.shape == (6, 2)
                assert 0.0 <= det.confidence <= 1.0

            vis = _annotate_detections(bgr, detections)
            out_name = _safe_visualization_stem("positive", idx, image_path) + ".png"
            out_path = VISUALIZATIONS_DIR / out_name
            assert cv2.imwrite(str(out_path), vis), f"Failed to write visualization: {out_path}"
            LOG.info("detection_visualization written path=%s", out_path)

            h, w = bgr.shape[:2]
            for face_i, det in enumerate(detections):
                x, y, bw, bh = det.bbox
                fixture_rows.append(
                    (
                        image_path.name,
                        face_i,
                        w,
                        h,
                        (int(x), int(y), int(bw), int(bh)),
                        det.landmarks_6pt.astype(np.float32).copy(),
                    )
                )

    if not alignment_fixtures_auto_update_disabled():
        written = write_alignment_embedding_fixtures(fixture_rows)
        LOG.info("alignment_embedding_fixtures updated path=%s cases=%d", written, len(fixture_rows))
    else:
        LOG.info(
            "alignment_embedding_fixtures auto-update skipped (ALIGNMENT_FIXTURES_NO_AUTO_UPDATE set)"
        )


@pytest.mark.integration
def test_negative_fixture_images_have_no_face_detections(
    caplog: pytest.LogCaptureFixture,
    detection_results_log,
) -> None:
    _require_mediapipe()
    caplog.set_level(logging.INFO, logger=__name__)
    caplog.set_level(logging.INFO, logger=BLAZEFACE_LOGGER_NAME)

    images = _list_images(NEGATIVE_DIR)
    if not images:
        pytest.skip(f"No negative fixture images found in {NEGATIVE_DIR}")

    _ensure_artifact_dirs()
    with BlazeFaceDetector(
        min_confidence=0.6,
        log_performance=True,
        log_raw_inference=False,
    ) as detector:
        for idx, image_path in enumerate(images):
            bgr = _load_bgr(image_path)
            caplog.clear()
            detections = detector.detect(bgr)
            _log_integration_image_result(
                caplog,
                detection_results_log,
                "negative",
                image_path,
                bgr,
                detections,
            )
            assert len(detections) == 0, (
                f"Unexpected face detected in negative fixture: {image_path.name}"
            )

            vis = _annotate_detections(bgr, detections)
            out_name = _safe_visualization_stem("negative", idx, image_path) + ".png"
            out_path = VISUALIZATIONS_DIR / out_name
            assert cv2.imwrite(str(out_path), vis), f"Failed to write visualization: {out_path}"
            LOG.info("detection_visualization written path=%s", out_path)
