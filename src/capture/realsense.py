"""RealSense camera capture with OpenCV fallback."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


@dataclass
class FramePacket:
    """Container for latest captured frame data."""

    bgr: np.ndarray
    depth: Optional[np.ndarray]
    timestamp: float


class RealSenseCapture:
    """Threaded frame capture from Intel RealSense, with webcam fallback."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_depth: bool = False,
        fallback_camera_index: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.use_depth = use_depth
        self.fallback_camera_index = fallback_camera_index

        self._pipeline = None
        self._capture = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[FramePacket] = None
        self._lock = threading.Lock()
        self._source = "none"

    @property
    def source(self) -> str:
        return self._source

    def start(self) -> "RealSenseCapture":
        if self._running:
            return self
        self._open_device()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._close_device()

    def __enter__(self) -> "RealSenseCapture":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def read(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def read_bgr(self) -> Optional[np.ndarray]:
        packet = self.read()
        return packet.bgr if packet is not None else None

    def _open_device(self) -> None:
        if rs is not None:
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                if self.use_depth:
                    cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                pipeline.start(cfg)
                self._pipeline = pipeline
                self._source = "realsense"
                return
            except Exception:
                self._pipeline = None

        cap = cv2.VideoCapture(self.fallback_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError("Could not open RealSense or fallback camera")
        self._capture = cap
        self._source = "opencv"

    def _close_device(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._source = "none"

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(1, self.fps)
        while self._running:
            packet = self._read_once()
            if packet is not None:
                with self._lock:
                    self._latest = packet
            time.sleep(frame_interval * 0.2)

    def _read_once(self) -> Optional[FramePacket]:
        if self._pipeline is not None:
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            bgr = np.asanyarray(color_frame.get_data())
            depth_arr = None
            if self.use_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_arr = np.asanyarray(depth_frame.get_data())
            return FramePacket(bgr=bgr, depth=depth_arr, timestamp=time.time())

        if self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                return None
            return FramePacket(bgr=frame, depth=None, timestamp=time.time())
        return None
