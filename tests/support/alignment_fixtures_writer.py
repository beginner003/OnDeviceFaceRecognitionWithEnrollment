"""Regenerate ``alignment_embedding_fixtures.py`` from live BlazeFace outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

_FIXTURES_PATH = Path(__file__).resolve().parent / "alignment_embedding_fixtures.py"

_MODULE_PROLOGUE = '''"""Frozen detection outputs synced from detection integration runs.

This file is **auto-generated** when ``tests/test_detection.py`` integration test
``test_positive_fixture_images_have_face_detections`` finishes successfully (same
ordering as sorted positive images × detector face index).

To run detection tests without rewriting this file (e.g. CI), set:
``ALIGNMENT_FIXTURES_NO_AUTO_UPDATE=1``.

BlazeFace keypoint order (pixel rows): right_eye, left_eye, nose_tip, mouth_center,
right_ear_tragion, left_ear_tragion (same as MediaPipe FaceDetector).
Log line ``frame=WxH`` is width × height; OpenCV images are shape (H, W, 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositiveLandmarkCase:
    """One face from a positive fixture image."""

    image_name: str
    """File under tests/data/detection/positive/."""

    face_index: int
    """0-based index among faces for that image (matches log [n] order)."""

    frame_width: int
    frame_height: int

    bbox_xywh: tuple[int, int, int, int]
    """Face box from BlazeFace: (x, y, width, height) in pixels."""

    landmarks_6x2: np.ndarray
    """Shape (6, 2), dtype float32, pixel coordinates (BlazeFace / MediaPipe order)."""


# fmt: off
_POSITIVE_CASES: list[PositiveLandmarkCase] = [
'''

_MODULE_EPILOGUE = '''
]
# fmt: on


def iter_positive_landmark_cases() -> tuple[PositiveLandmarkCase, ...]:
    return tuple(_POSITIVE_CASES)
'''


def alignment_fixtures_auto_update_disabled() -> bool:
    return os.environ.get("ALIGNMENT_FIXTURES_NO_AUTO_UPDATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _format_landmarks_block(lm: np.ndarray) -> str:
    if lm.shape != (6, 2):
        raise ValueError(f"Expected landmarks (6, 2), got {lm.shape}")
    rows = []
    for i in range(6):
        rows.append(f"                [{lm[i, 0]:.4f}, {lm[i, 1]:.4f}]")
    return ",\n".join(rows)


def _format_positive_case(
    image_name: str,
    face_index: int,
    frame_width: int,
    frame_height: int,
    bbox_xywh: tuple[int, int, int, int],
    landmarks_6x2: np.ndarray,
) -> str:
    x, y, w, h = bbox_xywh
    lm = np.asarray(landmarks_6x2, dtype=np.float32)
    inner = _format_landmarks_block(lm)
    return f"""    PositiveLandmarkCase(
        {image_name!r}, {face_index}, {frame_width}, {frame_height},
        ({x}, {y}, {w}, {h}),
        np.array(
            [
{inner}
            ],
            dtype=np.float32,
        ),
    ),"""


def build_alignment_embedding_fixtures_source(
    cases: Sequence[tuple[str, int, int, int, tuple[int, int, int, int], np.ndarray]],
) -> str:
    """Each tuple: image_name, face_index, frame_w, frame_h, bbox_xywh, landmarks (6,2)."""
    blocks = [
        _format_positive_case(name, fi, fw, fh, bbox, lm)
        for name, fi, fw, fh, bbox, lm in cases
    ]
    body = "\n".join(blocks)
    if body:
        body += "\n"
    return _MODULE_PROLOGUE + body + _MODULE_EPILOGUE


def write_alignment_embedding_fixtures(
    cases: Sequence[tuple[str, int, int, int, tuple[int, int, int, int], np.ndarray]],
    *,
    path: Path | None = None,
) -> Path:
    """Write fixtures module; returns path written."""
    target = path or _FIXTURES_PATH
    text = build_alignment_embedding_fixtures_source(cases)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target
