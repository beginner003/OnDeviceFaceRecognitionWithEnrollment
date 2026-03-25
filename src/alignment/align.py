"""Face alignment using eye centers + bbox-guided rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

# MediaPipe BlazeFace FaceDetector keypoint row order in ``landmarks_6pt``:
# 0 right_eye, 1 left_eye, 2 nose_tip, 3 mouth_center, 4 right_ear_tragion, 5 left_ear_tragion
_MP_RIGHT_EYE = 0
_MP_LEFT_EYE = 1


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned face and transform matrix."""

    aligned_bgr: np.ndarray
    transform: np.ndarray


class FaceAligner:
    """Align a face using eye-line rotation on expanded bbox."""

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        expand_ratio: float = 1.5,
    ) -> None:
        self.output_size = output_size
        self.expand_ratio = float(expand_ratio)
        if self.expand_ratio <= 0.0:
            raise ValueError("expand_ratio must be > 0")

    @staticmethod
    def _expand_bbox(
        bbox_xywh: Sequence[int],
        image_shape: Tuple[int, int],
        ratio: float,
    ) -> Tuple[int, int, int, int]:
        """Expand bbox around center; clamp to image bounds."""
        if len(bbox_xywh) != 4:
            raise ValueError(f"Expected bbox (x, y, w, h), got {bbox_xywh}")
        x, y, bw, bh = [int(v) for v in bbox_xywh]
        if bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid bbox size: {(x, y, bw, bh)}")

        h, w = image_shape[:2]
        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)

        ew = max(1, int(round(bw * ratio)))
        eh = max(1, int(round(bh * ratio)))
        ex = int(round(cx - (ew / 2.0)))
        ey = int(round(cy - (eh / 2.0)))
        ex2 = ex + ew
        ey2 = ey + eh

        ex = max(0, ex)
        ey = max(0, ey)
        ex2 = min(w, ex2)
        ey2 = min(h, ey2)
        ew = max(1, ex2 - ex)
        eh = max(1, ey2 - ey)
        return ex, ey, ew, eh

    @staticmethod
    def _eye_upright_rotation_deg(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Rotation (degrees) to make the eye line horizontal.
        """
        dy = float(left_eye[1] - right_eye[1])
        dx = float(left_eye[0] - right_eye[0])
        return float(np.degrees(np.arctan2(dy, dx)))

    def estimate_transform(self, landmarks_6pt: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Estimate 2x3 eye-based rotation matrix in full-image coordinates.

        Args:
            landmarks_6pt: BlazeFace keypoints (6, 2) in MediaPipe order; only the two
                eye centres are used.
        """
        src = np.asarray(landmarks_6pt, dtype=np.float32)
        if src.shape != (6, 2):
            raise ValueError(f"Expected landmarks shape (6, 2), got {src.shape}")

        right_eye = src[_MP_RIGHT_EYE]
        left_eye = src[_MP_LEFT_EYE]
        angle = self._eye_upright_rotation_deg(left_eye, right_eye)
        eyes_center = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0),
        )
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        return matrix.astype(np.float32)

    def align(
        self,
        bgr_image: np.ndarray,
        landmarks_6pt: Sequence[Sequence[float]],
        bbox_xywh: Sequence[int],
    ) -> AlignmentResult:
        """
        Align face by rotating an expanded bbox around the eye midpoint.

        Uses only the left/right eye centres from BlazeFace ``landmarks_6pt``.

        ``bbox_xywh`` must come from the face detector (x, y, width, height in pixels).
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty")

        src = np.asarray(landmarks_6pt, dtype=np.float32)
        if src.shape != (6, 2):
            raise ValueError(f"Expected landmarks shape (6, 2), got {src.shape}")

        if bbox_xywh is None or len(bbox_xywh) != 4:
            raise ValueError(
                f"bbox_xywh must be (x, y, w, h) from detection; got {bbox_xywh!r}"
            )

        ex, ey, ew, eh = self._expand_bbox(
            bbox_xywh=bbox_xywh,
            image_shape=bgr_image.shape,
            ratio=self.expand_ratio,
        )
        face_region = bgr_image[ey : ey + eh, ex : ex + ew]
        if face_region.size == 0:
            raise RuntimeError("Expanded face region is empty after bbox clipping")

        right_eye = src[_MP_RIGHT_EYE]
        left_eye = src[_MP_LEFT_EYE]
        angle = self._eye_upright_rotation_deg(left_eye, right_eye)
        eyes_center_local = (
            float((left_eye[0] + right_eye[0]) / 2.0) - float(ex),
            float((left_eye[1] + right_eye[1]) / 2.0) - float(ey),
        )
        matrix = cv2.getRotationMatrix2D(eyes_center_local, angle, 1.0).astype(np.float32)
        rotated = cv2.warpAffine(
            face_region,
            matrix,
            (face_region.shape[1], face_region.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Re-crop using the rotated original bbox corners inside expanded region.
        x, y, bw, bh = [int(v) for v in bbox_xywh]
        corners_local = np.array(
            [
                [x - ex, y - ey],
                [x + bw - ex, y - ey],
                [x - ex, y + bh - ey],
                [x + bw - ex, y + bh - ey],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        rotated_corners = cv2.transform(corners_local, matrix).reshape(-1, 2)

        rx0 = max(0, int(np.floor(rotated_corners[:, 0].min())))
        ry0 = max(0, int(np.floor(rotated_corners[:, 1].min())))
        rx1 = min(rotated.shape[1], int(np.ceil(rotated_corners[:, 0].max())))
        ry1 = min(rotated.shape[0], int(np.ceil(rotated_corners[:, 1].max())))
        if rx1 <= rx0 or ry1 <= ry0:
            aligned_crop = rotated
        else:
            aligned_crop = rotated[ry0:ry1, rx0:rx1]

        w, h = self.output_size
        aligned = cv2.resize(aligned_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return AlignmentResult(aligned_bgr=aligned, transform=matrix)

    @staticmethod
    def to_model_input(aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Convert aligned BGR image to float32 RGB normalized to [-1, 1].

        Returns:
            Array with shape (112, 112, 3), dtype float32.
        """
        if aligned_bgr is None or aligned_bgr.size == 0:
            raise ValueError("Aligned image is empty")

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32)
        x = (x - 127.5) / 127.5
        return x

