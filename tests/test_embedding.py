"""Tests for MobileFaceNet TFLite embedding extraction."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason="OpenCV required: pip install opencv-python-headless (use `python -m pytest` so the same env has cv2)",
)

from src.alignment import FaceAligner
from src.embedding import MobileFaceNetEmbedder

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOBILEFACENET_TFLITE = REPO_ROOT / "src" / "models" / "mobilefacenet.tflite"
MODELS_DIR = REPO_ROOT / "src" / "models"

EMBEDDING_POSITIVE_DIR = Path(__file__).parent / "data" / "alignment_embedding" / "positive"
AE_ARTIFACTS = Path(__file__).parent / "data" / "alignment_embedding" / "artifacts"
AE_LOGS_DIR = AE_ARTIFACTS / "logs"
EMBEDDING_LOG_PATH = AE_LOGS_DIR / "embedding_results.log"


def _ensure_ae_artifact_dirs() -> None:
    AE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_mobilefacenet_model_path() -> Path | None:
    env = os.environ.get("MOBILEFACENET_TFLITE", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    variant = os.environ.get("MOBILEFACENET_MODEL_VARIANT", "float32").strip().lower()
    if variant not in {"float32", "int8"}:
        return None

    # Conventional model names in src/models for configurable precision.
    candidate_names = {
        "float32": ("mobilefacenet_float32.tflite", "mobilefacenet.tflite"),
        "int8": ("mobilefacenet_int8.tflite",),
    }[variant]
    for name in candidate_names:
        p = MODELS_DIR / name
        if p.is_file():
            return p

    # Last resort: any model file that includes the variant hint.
    hint_matches = sorted(
        p for p in MODELS_DIR.glob("*.tflite") if variant in p.stem.lower()
    )
    if hint_matches:
        return hint_matches[0]

    return DEFAULT_MOBILEFACENET_TFLITE if variant == "float32" and DEFAULT_MOBILEFACENET_TFLITE.is_file() else None


def _iter_embedding_fixture_images() -> tuple[Path, ...]:
    if not EMBEDDING_POSITIVE_DIR.is_dir():
        return ()
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = [
        p for p in sorted(EMBEDDING_POSITIVE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return tuple(images)


def _has_tflite_runtime() -> bool:
    try:
        import tflite_runtime.interpreter  # type: ignore  # noqa: F401

        return True
    except Exception:
        try:
            import tensorflow.lite  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _safe_stem(prefix: str, index: int, image_name: str, face_index: int) -> str:
    raw = f"{prefix}_{index:02d}_{Path(image_name).stem}_face{face_index}"
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return safe or f"{prefix}_{index:02d}"


@pytest.fixture
def embedding_results_log(request):
    _ensure_ae_artifact_dirs()
    EMBEDDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDING_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n{'#' * 80}\n"
            f"# embedding integration  {datetime.now(timezone.utc).isoformat()}\n"
            f"# test: {request.node.nodeid}\n"
            f"{'#' * 80}\n"
        )
        fh.flush()
        yield fh


@patch("src.embedding.mobilefacenet._load_interpreter_class")
def test_embed_rejects_bad_input_shape(mock_loader: MagicMock) -> None:
    fake = MagicMock()
    fake.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake.get_output_details.return_value = [{"index": 1}]
    fake.get_tensor.return_value = np.random.randn(1, 128).astype(np.float32)
    mock_loader.return_value = MagicMock(return_value=fake)

    emb = MobileFaceNetEmbedder("/tmp/nonexistent_but_mocked.tflite")
    bad = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(112, 112, 3\)"):
        emb.embed(bad)


@patch("src.embedding.mobilefacenet._load_interpreter_class")
def test_embed_returns_l2_normalized_128d(mock_loader: MagicMock) -> None:
    raw_out = np.random.randn(1, 128).astype(np.float32)
    fake = MagicMock()
    fake.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake.get_output_details.return_value = [{"index": 1}]
    fake.get_tensor.return_value = raw_out
    mock_loader.return_value = MagicMock(return_value=fake)

    emb = MobileFaceNetEmbedder("/tmp/nonexistent_but_mocked.tflite")
    x = np.zeros((112, 112, 3), dtype=np.float32)
    out = emb.embed(x)
    assert out.shape == (128,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-5, atol=1e-5)


@patch("src.embedding.mobilefacenet._load_interpreter_class")
def test_embed_batch_matches_per_sample_norm(mock_loader: MagicMock) -> None:
    raw_out = np.ones((1, 128), dtype=np.float32)
    fake = MagicMock()
    fake.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
    fake.get_output_details.return_value = [{"index": 1}]
    fake.get_tensor.return_value = raw_out
    mock_loader.return_value = MagicMock(return_value=fake)

    emb = MobileFaceNetEmbedder("/tmp/nonexistent_but_mocked.tflite")
    batch = np.zeros((2, 112, 112, 3), dtype=np.float32)
    stacked = emb.embed_batch(batch)
    assert stacked.shape == (2, 128)
    for i in range(2):
        np.testing.assert_allclose(stacked[i], emb.embed(batch[i]), rtol=0, atol=0)


@pytest.mark.integration
def test_mobilefacenet_on_aligned_fixture_faces_logs_latency(
    embedding_results_log,
) -> None:
    model_path = _resolve_mobilefacenet_model_path()
    variant = os.environ.get("MOBILEFACENET_MODEL_VARIANT", "float32").strip().lower()
    if model_path is None:
        pytest.skip(
            "MobileFaceNet TFLite not found. Place model under src/models (e.g. "
            "mobilefacenet_float32.tflite or mobilefacenet_int8.tflite), "
            "set MOBILEFACENET_MODEL_VARIANT to float32/int8, "
            "or set MOBILEFACENET_TFLITE to a .tflite path."
        )

    fixture_images = _iter_embedding_fixture_images()
    if not fixture_images:
        pytest.skip(
            "Missing embedding fixture images. Add aligned/cropped faces to "
            "tests/data/alignment_embedding/positive/."
        )
    if not _has_tflite_runtime():
        pytest.skip(
            "TensorFlow Lite runtime not available. Install `tflite-runtime` or `tensorflow` "
            "in the active test environment."
        )

    try:
        embedder = MobileFaceNetEmbedder(str(model_path), num_threads=1)
    except RuntimeError as exc:
        pytest.skip(f"Failed to initialize MobileFaceNet interpreter: {exc}")

    for idx, image_path in enumerate(fixture_images):
        aligned_bgr = _load_bgr(image_path)
        if aligned_bgr.shape[:2] != (112, 112):
            pytest.skip(
                f"Fixture image must be aligned crop of shape (112,112), got {aligned_bgr.shape[:2]} for {image_path.name}"
            )
        tensor = FaceAligner.to_model_input(aligned_bgr)
        t_emb0 = time.perf_counter()
        embedding = embedder.embed(tensor)
        emb_ms = (time.perf_counter() - t_emb0) * 1000.0

        assert embedding.shape == (128,)
        np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, rtol=1e-4, atol=1e-4)

        if embedding_results_log is not None:
            embedding_results_log.write("\n")
            embedding_results_log.write("=" * 80 + "\n")
            embedding_results_log.write(
                f"CASE  file={image_path.name}  index={idx}  "
                f"model={model_path.name}  variant={variant}\n"
            )
            embedding_results_log.write("=" * 80 + "\n\n")
            embedding_results_log.write("performance:\n")
            embedding_results_log.write(f"  embedding.latency_ms={emb_ms:.3f}\n\n")
            embedding_results_log.write("summary:\n")
            embedding_results_log.write(f"  embedding_shape={embedding.shape}\n")
            embedding_results_log.write(f"  l2_norm={float(np.linalg.norm(embedding)):.6f}\n")
            embedding_results_log.write(
                f"  embedding_head=[{', '.join(f'{v:.6f}' for v in embedding[:8])} ...]\n\n"
            )
            embedding_results_log.flush()

        LOG.info(
            "embedding_fixture file=%s idx=%d model=%s variant=%s embed_ms=%.3f",
            image_path.name,
            idx,
            model_path.name,
            variant,
            emb_ms,
        )
