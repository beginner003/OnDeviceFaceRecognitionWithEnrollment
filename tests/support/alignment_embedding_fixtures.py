"""Frozen detection outputs synced from detection integration runs.

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
    PositiveLandmarkCase(
        'pos3.png', 0, 800, 534,
        (438, 72, 109, 109),
        np.array(
            [
                [473.9003, 105.2350],
                [512.8214, 113.9754],
                [485.7554, 136.4137],
                [482.1875, 154.4230],
                [453.0305, 102.3991],
                [539.0872, 120.5364]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos5.png', 0, 390, 280,
        (160, 66, 92, 92),
        np.array(
            [
                [198.1870, 91.8872],
                [234.9343, 95.2644],
                [222.3084, 116.7282],
                [217.8516, 134.0757],
                [164.5861, 95.4644],
                [244.5103, 101.1809]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos7.png', 0, 1500, 1125,
        (454, 272, 554, 554),
        np.array(
            [
                [728.0272, 410.7764],
                [933.6337, 462.3873],
                [868.5729, 578.3254],
                [812.9129, 677.8484],
                [487.0651, 418.2646],
                [936.7510, 519.8812]
            ],
            dtype=np.float32,
        ),
    ),

]
# fmt: on


def iter_positive_landmark_cases() -> tuple[PositiveLandmarkCase, ...]:
    return tuple(_POSITIVE_CASES)
