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
        'pos1.png', 0, 800, 600,
        (279, 198, 38, 38),
        np.array(
            [
                [291.8018, 212.2437],
                [304.1294, 211.7700],
                [303.6722, 219.6011],
                [300.2599, 228.3192],
                [271.3519, 218.6868],
                [302.0854, 217.2312]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos3.png', 0, 800, 534,
        (439, 77, 96, 96),
        np.array(
            [
                [469.7018, 109.0994],
                [507.0393, 109.3145],
                [483.8196, 131.8518],
                [484.3088, 149.5379],
                [454.7067, 112.4252],
                [535.2930, 114.1608]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos3.png', 1, 800, 534,
        (285, 113, 92, 92),
        np.array(
            [
                [318.3176, 141.8734],
                [351.9392, 144.7270],
                [340.4519, 166.1209],
                [330.9605, 183.3974],
                [277.8265, 140.9388],
                [354.3310, 151.2131]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos4.png', 0, 1200, 675,
        (760, 121, 82, 82),
        np.array(
            [
                [786.3879, 152.3719],
                [816.6312, 145.3208],
                [790.6649, 164.5417],
                [791.7501, 182.0549],
                [781.9904, 161.6748],
                [850.2393, 159.4306]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos4.png', 1, 1200, 675,
        (666, 166, 67, 67),
        np.array(
            [
                [687.2492, 194.6424],
                [715.1542, 188.4096],
                [701.2495, 204.1283],
                [703.2174, 217.7101],
                [675.8906, 207.3598],
                [735.8092, 197.1947]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos5.png', 0, 390, 280,
        (167, 59, 96, 96),
        np.array(
            [
                [198.0992, 90.6592],
                [232.4739, 95.2251],
                [221.4799, 115.9909],
                [216.1057, 132.5898],
                [162.8689, 95.3387],
                [238.4860, 103.9206]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos6.png', 0, 1600, 1066,
        (771, 404, 58, 58),
        np.array(
            [
                [790.4712, 425.4170],
                [807.8480, 426.5314],
                [799.0938, 440.0911],
                [805.0622, 451.1582],
                [782.5159, 434.8665],
                [830.7411, 429.7451]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos6.png', 1, 1600, 1066,
        (600, 432, 61, 61),
        np.array(
            [
                [625.6860, 450.2352],
                [634.0280, 454.2203],
                [631.0993, 466.6492],
                [636.8307, 478.6556],
                [618.0126, 459.0305],
                [653.1269, 457.7513]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos6.png', 2, 1600, 1066,
        (1120, 361, 85, 85),
        np.array(
            [
                [1146.0812, 388.7664],
                [1178.2112, 390.7130],
                [1166.0861, 403.4813],
                [1171.7252, 424.2439],
                [1128.4011, 415.1427],
                [1202.1426, 411.3227]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos6.png', 3, 1600, 1066,
        (894, 425, 78, 78),
        np.array(
            [
                [922.2670, 449.1699],
                [949.5994, 455.2545],
                [927.7105, 470.7067],
                [932.6635, 484.4958],
                [919.9765, 452.5531],
                [983.1976, 460.8320]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos6.png', 4, 1600, 1066,
        (601, 406, 61, 61),
        np.array(
            [
                [625.9073, 431.6122],
                [634.3615, 434.5722],
                [633.6837, 448.8594],
                [639.8127, 458.3238],
                [619.3155, 440.6358],
                [652.0824, 437.4376]
            ],
            dtype=np.float32,
        ),
    ),
    PositiveLandmarkCase(
        'pos7.png', 0, 1500, 1125,
        (511, 221, 598, 598),
        np.array(
            [
                [719.5458, 401.6993],
                [923.8112, 459.5229],
                [853.5192, 573.0491],
                [794.5608, 678.8596],
                [482.6940, 412.4265],
                [915.6733, 534.2099]
            ],
            dtype=np.float32,
        ),
    ),

]
# fmt: on


def iter_positive_landmark_cases() -> tuple[PositiveLandmarkCase, ...]:
    return tuple(_POSITIVE_CASES)
