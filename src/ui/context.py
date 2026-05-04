"""Process-wide resources for the FastAPI app (camera, detector, system)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.capture.realsense import RealSenseCapture
from src.detection.blazeface import BlazeFaceDetector
from src.embedding import MobileFaceNetEmbedder
from src.system import FaceRecognitionSystem

from src.ui.pipeline import VisionPipeline, resolve_mobilefacenet_path
from src.ui.preferences import UiPreferences, load_preferences, save_preferences


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def default_workspace() -> Path:
    raw = os.environ.get("FACE_UI_WORKSPACE", "data/ui_workspace").strip()
    return Path(raw).expanduser().resolve()


@dataclass
class AppContext:
    """Mutable singleton wired in FastAPI lifespan."""

    workspace: Path
    capture: RealSenseCapture
    detector: BlazeFaceDetector
    pipeline: VisionPipeline
    system: FaceRecognitionSystem
    system_lock: threading.Lock = field(default_factory=threading.Lock)
    registration_busy: bool = False
    registration_lock: threading.Lock = field(default_factory=threading.Lock)
    registration_state: dict[str, Any] = field(
        default_factory=lambda: {"phase": "idle", "seq": 0}
    )
    registration_state_lock: threading.Lock = field(default_factory=threading.Lock)
    pipeline_state: dict[str, Any] = field(default_factory=lambda: {"stage": "idle"})
    pipeline_aligned_bgr: Any = None
    pipeline_state_lock: threading.Lock = field(default_factory=threading.Lock)

    def bump_registration(self, **kwargs: Any) -> None:
        with self.registration_state_lock:
            self.registration_state.update(kwargs)
            self.registration_state["seq"] = int(self.registration_state.get("seq", 0)) + 1

    def update_pipeline_state(self, payload: dict[str, Any], aligned_bgr: Any = None) -> None:
        with self.pipeline_state_lock:
            self.pipeline_state = dict(payload)
            if aligned_bgr is not None:
                self.pipeline_aligned_bgr = aligned_bgr

    def read_pipeline_state(self) -> tuple[dict[str, Any], Any]:
        with self.pipeline_state_lock:
            aligned = None if self.pipeline_aligned_bgr is None else self.pipeline_aligned_bgr.copy()
            return dict(self.pipeline_state), aligned

    def settings_locked(self) -> bool:
        return len(self.system.identities()) > 0

    def reload_system(self, prefs: UiPreferences) -> None:
        save_preferences(self.workspace, prefs)
        self.system = FaceRecognitionSystem.from_config(
            prefs.to_system_config(),
            self.workspace,
        )


_ctx: Optional[AppContext] = None
_ctx_lock = threading.Lock()


def get_ctx() -> AppContext:
    if _ctx is None:
        raise RuntimeError("Application context is not initialised")
    return _ctx


def set_ctx(ctx: Optional[AppContext]) -> None:
    global _ctx
    with _ctx_lock:
        _ctx = ctx


def build_context() -> AppContext:
    workspace = default_workspace()
    workspace.mkdir(parents=True, exist_ok=True)

    prefs = load_preferences(workspace)
    system = FaceRecognitionSystem.from_config(prefs.to_system_config(), workspace)

    width = int(os.environ.get("FACE_UI_CAPTURE_WIDTH", "640"))
    height = int(os.environ.get("FACE_UI_CAPTURE_HEIGHT", "480"))
    fps = int(os.environ.get("FACE_UI_CAPTURE_FPS", "30"))
    use_depth = _env_bool("FACE_UI_USE_DEPTH", False)

    capture = RealSenseCapture(
        width=width,
        height=height,
        fps=fps,
        use_depth=use_depth,
        fallback_camera_index=int(os.environ.get("FACE_UI_FALLBACK_CAMERA", "2")),
    )
    capture.start()

    model_path = resolve_mobilefacenet_path()
    threads = int(os.environ.get("FACE_UI_EMBEDDER_THREADS", "2"))
    embedder = MobileFaceNetEmbedder(str(model_path), num_threads=threads)
    pipeline = VisionPipeline(embedder)

    min_conf = float(os.environ.get("FACE_UI_DETECTOR_MIN_CONFIDENCE", "0.7"))
    detector = BlazeFaceDetector(min_confidence=min_conf).__enter__()

    return AppContext(
        workspace=workspace,
        capture=capture,
        detector=detector,
        pipeline=pipeline,
        system=system,
    )
