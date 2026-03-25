from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from src.alignment import FaceAligner
from src.detection.blazeface import BlazeFaceDetector
from src.embedding import MobileFaceNetEmbedder

LOG = logging.getLogger(__name__)

_SelectMode = Literal["max_confidence", "all"]
_OnFail = Literal["skip", "raise"]


def _repo_root() -> Path:
    # experiments/embedding_helper.py -> repo root
    return Path(__file__).resolve().parent.parent


def _resolve_mobilefacenet_model_path(
    *,
    mobilefacenet_tflite_path: str | Path | None,
) -> Path:
    """
    Resolve MobileFaceNet TFLite path.

    If `mobilefacenet_tflite_path` is provided, it is used.
    Otherwise, we follow the same env-driven strategy as `tests/test_embedding.py`.
    """

    models_dir = _repo_root() / "src" / "models"

    if mobilefacenet_tflite_path is not None:
        p = Path(mobilefacenet_tflite_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MobileFaceNet model not found: {p}")
        return p

    env = os.environ.get("MOBILEFACENET_TFLITE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    variant = os.environ.get("MOBILEFACENET_MODEL_VARIANT", "float32").strip().lower()
    if variant not in {"float32", "int8"}:
        raise ValueError(
            "MOBILEFACENET_MODEL_VARIANT must be one of {'float32','int8'}; "
            f"got {variant!r}"
        )

    candidate_names: dict[str, tuple[str, ...]] = {
        "float32": ("mobilefacenet_float32.tflite", "mobilefacenet.tflite"),
        "int8": ("mobilefacenet_int8.tflite",),
    }

    for name in candidate_names[variant]:
        p = models_dir / name
        if p.is_file():
            return p

    # Last resort: any model file that includes the variant hint.
    hint_matches = sorted(p for p in models_dir.glob("*.tflite") if variant in p.stem.lower())
    if hint_matches:
        return hint_matches[0]

    default_float32 = models_dir / "mobilefacenet.tflite"
    if variant == "float32" and default_float32.is_file():
        return default_float32

    raise FileNotFoundError(
        "Could not resolve a MobileFaceNet .tflite model. "
        "Set MOBILEFACENET_TFLITE or MOBILEFACENET_MODEL_VARIANT to float32/int8, "
        "or pass `mobilefacenet_tflite_path=...`."
    )


def _resolve_image_path(path: str | Path, *, base_dir: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _safe_filename(s: str) -> str:
    # Keep it deterministic and filesystem-friendly.
    s2 = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s2 or "item"


def embed_images_to_dir(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
    mobilefacenet_tflite_path: str | Path | None = None,
    tasks_model_path: str | Path | None = None,
    detector_min_confidence: float = 0.7,
    select_face: _SelectMode = "max_confidence",
    on_fail: _OnFail = "skip",
    expand_ratio: float = 1.5,
    embedder_num_threads: int = 2,
    use_cache: bool = True,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Detect -> align -> embed a list of images and save results to `output_dir`.

    Args:
        image_paths: list of paths to images.
        output_dir: directory to create. Writes:
            - `embeddings.npy` (float32, shape (N, 128))
            - `embeddings_meta.json` (per-embedding provenance + bbox/confidence)
        base_dir: base used to resolve relative `image_paths`. Defaults to repo root.
        select_face: `max_confidence` (default) uses the best detector bbox per image.
            `all` embeds all detected faces per image (can produce multiple embeddings
            for a single image).
        on_fail: if detector fails for one image, `skip` continues or `raise` stops.

    Returns:
        embeddings: float32 NumPy array of shape (N, 128).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "embeddings_meta.json"

    if use_cache and embeddings_path.is_file() and meta_path.is_file() and not overwrite:
        emb = np.load(str(embeddings_path))
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 2 and emb.shape[1] == 128:
            return emb

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for embedding extraction. Install "
            "`opencv-python-headless` in the active environment."
        ) from exc

    base_dir = Path(base_dir) if base_dir is not None else _repo_root()

    model_path = _resolve_mobilefacenet_model_path(mobilefacenet_tflite_path=mobilefacenet_tflite_path)
    embedder = MobileFaceNetEmbedder(str(model_path), num_threads=embedder_num_threads)

    with BlazeFaceDetector(
        min_confidence=detector_min_confidence,
        tasks_model_path=str(tasks_model_path) if tasks_model_path is not None else None,
    ) as detector:
        aligner = FaceAligner(output_size=(112, 112), expand_ratio=expand_ratio)

        embeddings: list[np.ndarray] = []
        meta: list[dict] = []

        for img_i, raw_image_path in enumerate(image_paths):
            image_path = _resolve_image_path(raw_image_path, base_dir=base_dir)
            if not image_path.is_file():
                msg = f"Image not found: {image_path}"
                if on_fail == "raise":
                    raise FileNotFoundError(msg)
                LOG.warning(msg)
                continue

            bgr = cv2.imread(str(image_path))
            if bgr is None or bgr.size == 0:
                msg = f"Failed to load image: {image_path}"
                if on_fail == "raise":
                    raise ValueError(msg)
                LOG.warning(msg)
                continue

            detections = detector.detect(bgr)
            if not detections:
                msg = f"No faces detected in image: {image_path}"
                if on_fail == "raise":
                    raise RuntimeError(msg)
                LOG.warning(msg)
                continue

            if select_face == "max_confidence":
                detections_sel = [max(detections, key=lambda d: d.confidence)]
            elif select_face == "all":
                detections_sel = list(detections)
            else:  # pragma: no cover
                raise ValueError(f"Unknown select_face={select_face!r}")

            aligned_tensors: list[np.ndarray] = []
            aligned_meta: list[dict] = []

            for det_face_i, det in enumerate(detections_sel):
                # The aligner expects BlazeFace landmarks (6,2) and a bbox_xywh.
                aligned = aligner.align(
                    bgr_image=bgr,
                    landmarks_6pt=det.landmarks_6pt,
                    bbox_xywh=det.bbox,
                )
                tensor = FaceAligner.to_model_input(aligned.aligned_bgr)
                aligned_tensors.append(tensor)

                aligned_meta.append(
                    {
                        "source_image": str(raw_image_path),
                        "image_index": int(img_i),
                        "face_index": int(det_face_i),
                        "detector_confidence": float(det.confidence),
                        "bbox_xywh": [int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])],
                    }
                )

            # embedder.embed_batch iterates per-sample (still keeps the calling code simple).
            tensor_batch = np.stack(aligned_tensors, axis=0).astype(np.float32)
            batch_emb = embedder.embed_batch(tensor_batch)  # (K, 128)

            for face_emb_i in range(batch_emb.shape[0]):
                embeddings.append(batch_emb[face_emb_i].astype(np.float32, copy=False))
                meta.append(aligned_meta[face_emb_i])

            LOG.info(
                "Embedded image %d/%d: %s -> %d embeddings",
                img_i + 1,
                len(image_paths),
                image_path,
                len(detections_sel),
            )

    if not embeddings:
        raise RuntimeError(
            "No embeddings were produced. "
            "Try lowering `detector_min_confidence`, switch `select_face='all'`, "
            "or set `on_fail='raise'` to see why extraction failed."
        )

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
    if emb_arr.ndim != 2 or emb_arr.shape[1] != 128:
        raise RuntimeError(f"Unexpected embeddings shape: {emb_arr.shape}")

    np.save(str(embeddings_path), emb_arr)
    meta_payload = {
        "count": int(emb_arr.shape[0]),
        "select_face": select_face,
        "on_fail": on_fail,
        "embeddings_meta": meta,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return emb_arr


def embed_supertask_identities_to_root(
    supertask_json_path: str | Path,
    output_root: str | Path,
    *,
    base_dir: str | Path | None = None,
    task_filter: str | None = None,
    identities_filter: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """
    Convenience wrapper for `data/supertask_8_2.json`.

    Creates per-identity subdirectories:
        output_root/<identity>/embeddings.npy
    and returns in-memory embeddings for all selected identities.
    """

    supertask_json_path = Path(supertask_json_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if base_dir is None:
        base_dir = _repo_root()

    data = json.loads(supertask_json_path.read_text(encoding="utf-8"))
    identities = data.get("identities", [])

    ids_set = set(identities_filter) if identities_filter is not None else None

    out: dict[str, np.ndarray] = {}
    for entry in identities:
        identity = str(entry["identity"]).strip()
        task = entry.get("task", None)
        if task_filter is not None and task != task_filter:
            continue
        if ids_set is not None and identity not in ids_set:
            continue

        img_paths = entry.get("image_paths", [])
        if not img_paths:
            raise ValueError(f"Missing image_paths for identity {identity!r}")

        ident_dir = output_root / _safe_filename(identity)
        emb = embed_images_to_dir(
            image_paths=img_paths,
            output_dir=ident_dir,
            base_dir=base_dir,
            overwrite=overwrite,
        )
        out[identity] = emb

    return out

