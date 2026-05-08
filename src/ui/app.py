"""FastAPI application: MJPEG preview, recognition/register modes, enrollment API."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ui.context import build_context, get_ctx, set_ctx

from src.ui.registration_capture import (
    collect_registration_embeddings,
    registration_phase_instruction,
)


_UI_DIR = Path(__file__).resolve().parent
_LOG = logging.getLogger("face_ui.registration")
_LOG.setLevel(logging.INFO)


class RegisterBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    n_frames: int = Field(
        24,
        ge=6,
        le=120,
        description="Collected first, then used for one background registration update.",
    )


class SettingsBody(BaseModel):
    registration: str = "replay_lwf"
    exemplar_selection: str = "herding"
    recognition: str = "classifier"
    exemplar_k: int = 50
    confidence_threshold: float = 0.5


def create_app() -> FastAPI:
    app = FastAPI(title="Continual Face Recognition UI", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        set_ctx(build_context())

    @app.on_event("shutdown")
    def _shutdown() -> None:
        try:
            ctx = get_ctx()
        except RuntimeError:
            return
        ctx.capture.stop()
        ctx.detector.__exit__(None, None, None)
        set_ctx(None)

    app.mount(
        "/static",
        StaticFiles(directory=str(_UI_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        from fastapi.templating import Jinja2Templates

        templates = Jinja2Templates(directory=str(_UI_DIR / "templates"))
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={},
        )

    @app.get("/video_feed")
    def video_feed(mode: str = "recognition") -> StreamingResponse:
        ctx = get_ctx()
        if mode not in ("recognition", "register"):
            mode = "recognition"

        def frames() -> Generator[bytes, None, None]:
            boundary = b"--frame\r\n"
            period = 1.0 / max(1, min(30, int(15)))
            while True:
                t0 = time.perf_counter()
                packet = ctx.capture.read()
                if packet is None:
                    time.sleep(0.05)
                    continue
                bgr = packet.bgr
                detect_t0 = time.perf_counter()
                detections = ctx.detector.detect(bgr)
                detect_ms = round((time.perf_counter() - detect_t0) * 1000.0, 2)
                labels: Optional[List[Optional[str]]] = None
                primary = ctx.pipeline.pick_largest(detections)
                artifact = None
                recognition_payload = None
                pipeline_error = ""

                rh = ""
                active_pose = ""
                busy = False
                if mode == "register":
                    with ctx.registration_state_lock:
                        rh = str(ctx.registration_state.get("instruction", "")).strip()
                        active_pose = str(ctx.registration_state.get("pose", "")).strip()
                        busy = ctx.registration_state.get("phase") == "capturing"

                if mode == "recognition" and detections:
                    labels = []
                    for det in detections:
                        text: Optional[str] = None
                        try:
                            this_artifact = ctx.pipeline.inspect_detection(bgr, det)
                            with ctx.system_lock:
                                name, conf = ctx.system.recognize(this_artifact.embedding)
                            if conf >= ctx.system.confidence_threshold:
                                text = f"{name} ({conf:.2f})"
                            else:
                                text = f"unknown ({conf:.2f})"
                            if primary is not None and det is primary:
                                artifact = this_artifact
                                recognition_payload = {
                                    "name": str(name),
                                    "confidence": round(float(conf), 4),
                                    "accepted": bool(conf >= ctx.system.confidence_threshold),
                                }
                        except Exception:
                            text = "?"
                            pipeline_error = "recognition failed"
                        labels.append(text)
                elif mode == "register" and detections:
                    labels = [None] * len(detections)
                    for i, det in enumerate(detections):
                        if primary is not None and det is primary:
                            labels[i] = "enrollment"

                vis = ctx.pipeline.annotate_frame(bgr, detections, labels)
                pipeline_payload = {
                    "stage": "frame_processed",
                    "mode": mode,
                    "camera_source": ctx.capture.source,
                    "timestamp": packet.timestamp,
                    "frame_size": [int(bgr.shape[1]), int(bgr.shape[0])],
                    "detection_ms": detect_ms,
                    "num_faces": len(detections),
                    "detections": [
                        ctx.pipeline.detection_to_dict(
                            det,
                            index=i,
                            primary=bool(primary is not None and det is primary),
                        )
                        for i, det in enumerate(detections)
                    ],
                    "pipeline_timings_ms": artifact.timings_ms if artifact is not None else {},
                    "recognition": recognition_payload,
                    "error": pipeline_error,
                }
                ctx.update_pipeline_state(
                    pipeline_payload,
                    aligned_bgr=artifact.aligned_bgr if artifact is not None else None,
                )

                if mode == "register" and rh:
                    line1 = rh[:75]
                    cv2.putText(
                        vis,
                        line1,
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (48, 232, 144),
                        2,
                        cv2.LINE_AA,
                    )
                    if active_pose and busy:
                        cv2.putText(
                            vis,
                            f"step: {active_pose}",
                            (8, 46),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )

                src = ctx.capture.source

                cv2.putText(
                    vis,
                    f"{mode} | cam:{src}",
                    (8, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                ok, jpeg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if not ok:
                    time.sleep(period)
                    continue
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, period - elapsed))

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/register")
    def register(body: RegisterBody) -> JSONResponse:
        ctx = get_ctx()
        with ctx.registration_lock:
            if ctx.registration_busy:
                raise HTTPException(status_code=409, detail="Registration already running")

            identity = body.name.strip()
            if not identity:
                raise HTTPException(status_code=400, detail="Empty name")

            with ctx.system_lock:
                if identity in ctx.system.identities():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Identity {identity!r} already enrolled",
                    )

            ctx.registration_busy = True
            ctx.bump_registration(phase="queued", message="Starting capture…")
            start_seq = int(ctx.registration_state.get("seq", 0))
            _LOG.info(
                "register requested identity=%s n_frames=%d start_seq=%d",
                identity,
                int(body.n_frames),
                start_seq,
            )

        def worker() -> None:
            time.sleep(0.15)
            embedding_session: List[np.ndarray] = []
            try:
                total = body.n_frames
                dup = float(os.environ.get("FACE_UI_REG_DEDUP_SIM", "0.995"))
                _LOG.info(
                    "registration worker start identity=%s total=%d capture_first=true dup=%.4f",
                    identity,
                    int(total),
                    float(dup),
                )

                def bump_cb(payload: dict) -> None:
                    ctx.bump_registration(**payload)

                ctx.bump_registration(
                    phase="capturing",
                    current=0,
                    target=total,
                    pose="capture",
                    instruction=registration_phase_instruction(),
                    quota_phase_current=0,
                    quota_phase_target=total,
                    message=registration_phase_instruction(),
                )

                capture_stale_limit = int(
                    os.environ.get(
                        "FACE_UI_REG_STALE_TOTAL",
                        str(max(900, body.n_frames * 130)),
                    )
                )
                _LOG.info(
                    "collect start identity=%s target=%d stale_limit=%d",
                    identity,
                    int(total),
                    int(capture_stale_limit),
                )
                embedding_session = collect_registration_embeddings(
                    capture=ctx.capture,
                    detector=ctx.detector,
                    pipeline=ctx.pipeline,
                    target=total,
                    similarity_dup=dup,
                    stale_limit=capture_stale_limit,
                    idle_sleep=float(os.environ.get("FACE_UI_REG_POLL_S", "0.02")),
                    bump=bump_cb,
                )
                if len(embedding_session) < total:
                    _LOG.warning(
                        "collect incomplete identity=%s collected=%d target=%d",
                        identity,
                        len(embedding_session),
                        int(total),
                    )
                    ctx.bump_registration(
                        phase="error",
                        message=(
                            f"Incomplete capture ({len(embedding_session)}/{total}). "
                            "Improve lighting, move closer, and slightly rotate your head."
                        ),
                        instruction="",
                    )
                    return

                emb_mat = np.stack(embedding_session, axis=0).astype(np.float32, copy=False)
                embedding_session.clear()
                ctx.bump_registration(
                    phase="captured",
                    instruction="Capture done. You can move away from the camera.",
                    message="Capture complete. Processing registration in background…",
                    current=total,
                    target=total,
                    return_to_recognition=True,
                )

                ctx.bump_registration(
                    phase="training",
                    instruction="Processing registration…",
                    message=(
                        "Capture finished. Registering identity and saving workspace artifacts."
                    ),
                    current=total,
                    target=total,
                    return_to_recognition=True,
                )

                try:
                    with ctx.system_lock:
                        result = ctx.system.register(identity, emb_mat)
                finally:
                    del emb_mat
                _LOG.info(
                    "registration training done identity=%s selected=%d total_identities=%d elapsed=%.3fs",
                    result.identity,
                    int(result.selected_count),
                    int(result.total_identities),
                    float(result.elapsed_s),
                )

                ctx.bump_registration(
                    phase="done",
                    message="Done — resumed recognition; session vectors cleared from RAM.",
                    identity=result.identity,
                    selected_count=result.selected_count,
                    total_identities=result.total_identities,
                    elapsed_s=round(result.elapsed_s, 3),
                    exemplar_bytes=result.exemplar_bytes,
                    return_to_recognition=True,
                    instruction="",
                )
            except ValueError as exc:
                ctx.bump_registration(phase="error", message=str(exc), instruction="")
                _LOG.exception("registration value error identity=%s", identity)
            except Exception as exc:  # pragma: no cover
                ctx.bump_registration(
                    phase="error",
                    message=f"{type(exc).__name__}: {exc}",
                    instruction="",
                )
                _LOG.exception("registration failed identity=%s", identity)
            finally:
                embedding_session.clear()
                ctx.registration_busy = False
                _LOG.info("registration worker finished identity=%s", identity)

        threading.Thread(target=worker, daemon=True).start()
        return JSONResponse({"ok": True, "start_seq": start_seq})

    @app.get("/register/stream")
    def register_stream(after_seq: int = -1) -> StreamingResponse:
        ctx = get_ctx()
        _LOG.info("register stream opened after_seq=%d", int(after_seq))

        def gen() -> Generator[str, None, None]:
            last_seq = int(after_seq)
            while True:
                with ctx.registration_state_lock:
                    seq = int(ctx.registration_state.get("seq", 0))
                    payload = dict(ctx.registration_state)
                if seq > last_seq:
                    last_seq = seq
                    yield f"data: {json.dumps(payload)}\n\n"
                    if payload.get("phase") in ("done", "error"):
                        _LOG.info(
                            "register stream terminal phase=%s seq=%d",
                            str(payload.get("phase", "")),
                            int(seq),
                        )
                        break
                else:
                    yield ":\n\n"
                time.sleep(0.12)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/identities")
    def identities() -> JSONResponse:
        ctx = get_ctx()
        with ctx.system_lock:
            names = ctx.system.identities()
            items = []
            store_ids = set(ctx.system.store.identities())
            for n in names:
                ec: Optional[int] = None
                if n in store_ids:
                    ec = int(ctx.system.store.get(n).embeddings.shape[0])
                items.append({"name": n, "exemplar_count": ec})
        return JSONResponse({"items": items})

    @app.get("/pipeline/status")
    def pipeline_status() -> JSONResponse:
        ctx = get_ctx()
        payload, _aligned = ctx.read_pipeline_state()
        return JSONResponse(payload)

    @app.get("/pipeline/aligned_face.jpg")
    def aligned_face() -> StreamingResponse:
        ctx = get_ctx()
        _payload, aligned = ctx.read_pipeline_state()
        if aligned is None:
            canvas = np.zeros((112, 112, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "no face",
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            aligned = canvas
        ok, jpeg = cv2.imencode(".jpg", aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode aligned face")
        return StreamingResponse(
            iter([jpeg.tobytes()]),
            media_type="image/jpeg",
        )

    @app.delete("/identities/{name}")
    def delete_identity(name: str) -> JSONResponse:
        ctx = get_ctx()
        with ctx.system_lock:
            before = set(ctx.system.identities())
            ctx.system.remove_identity(name)
            after = set(ctx.system.identities())
        if name not in before:
            raise HTTPException(status_code=404, detail="Identity not found")
        return JSONResponse({"ok": True, "removed": name, "remaining": sorted(after)})

    @app.get("/status")
    def status() -> JSONResponse:
        ctx = get_ctx()
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        with ctx.system_lock:
            n_id = len(ctx.system.identities())
            prefs_locked = ctx.settings_locked()
        return JSONResponse(
            {
                "workspace": str(ctx.workspace),
                "camera_source": ctx.capture.source,
                "enrolled_count": n_id,
                "registration_busy": ctx.registration_busy,
                "settings_locked": prefs_locked,
                "memory_rss_mb": round(float(rss_mb), 1),
            }
        )

    @app.get("/settings")
    def get_settings() -> JSONResponse:
        from src.ui.preferences import load_preferences

        ctx = get_ctx()
        prefs = load_preferences(ctx.workspace)
        return JSONResponse(
            {
                **prefs.to_json_dict(),
                "locked": ctx.settings_locked(),
            }
        )

    @app.put("/settings")
    def put_settings(body: SettingsBody) -> JSONResponse:
        from src.ui.preferences import UiPreferences, validate_preferences_dict

        ctx = get_ctx()
        if ctx.settings_locked():
            raise HTTPException(
                status_code=409,
                detail="Settings are locked after the first enrollment; remove all identities to change methods.",
            )
        try:
            prefs = validate_preferences_dict(body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with ctx.system_lock:
            ctx.reload_system(prefs)
        return JSONResponse({"ok": True, **prefs.to_json_dict()})

    @app.post("/evaluate")
    def evaluate() -> JSONResponse:
        return JSONResponse(
            {
                "ok": False,
                "message": "Offline benchmarks live under experiments/. "
                "Use /status and /identities for on-device enrollment stats.",
            }
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.ui.app:app",
        host=os.environ.get("FACE_UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("FACE_UI_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
