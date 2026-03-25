"""MobileFaceNet TFLite embedding extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

def _load_interpreter_class():
    """Import TFLite Interpreter lazily so importing this module does not require a runtime."""
    try:  # Prefer lightweight runtime on Raspberry Pi.
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter
    except ImportError:  # pragma: no cover
        try:
            import tensorflow as tf  # type: ignore

            return tf.lite.Interpreter
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "TensorFlow Lite runtime is required. Install `tflite-runtime` or `tensorflow`."
            ) from exc


class MobileFaceNetEmbedder:
    """Thin wrapper around a TFLite MobileFaceNet interpreter."""

    def __init__(self, model_path: str, num_threads: int = 2) -> None:
        self.model_path = str(Path(model_path))
        self.num_threads = int(max(1, num_threads))
        self._interpreter: Optional[Interpreter] = None
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._input_dtype = np.float32

        self._init_interpreter()

    def _init_interpreter(self) -> None:
        Interpreter = _load_interpreter_class()
        self._interpreter = Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads,
        )
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        if len(input_details) != 1 or len(output_details) != 1:
            raise RuntimeError("Expected single-input single-output MobileFaceNet model")

        self._input_index = int(input_details[0]["index"])
        self._output_index = int(output_details[0]["index"])
        self._input_dtype = input_details[0]["dtype"]

    def embed(self, aligned_rgb_normalized: np.ndarray) -> np.ndarray:
        """
        Infer one embedding from one aligned face.

        Args:
            aligned_rgb_normalized: (112, 112, 3) float32 values in [-1, 1].

        Returns:
            L2-normalized embedding of shape (128,), dtype float32.
        """
        if aligned_rgb_normalized.shape != (112, 112, 3):
            raise ValueError(
                f"Expected aligned face shape (112, 112, 3), got {aligned_rgb_normalized.shape}"
            )
        if self._interpreter is None or self._input_index is None or self._output_index is None:
            raise RuntimeError("Interpreter is not initialized")

        batched = aligned_rgb_normalized.astype(np.float32, copy=False)[None, ...]
        input_tensor = self._cast_input(batched)

        self._interpreter.set_tensor(self._input_index, input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)

        emb = np.asarray(output, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm

    def embed_batch(self, aligned_rgb_normalized_batch: np.ndarray) -> np.ndarray:
        """Infer embeddings for a batch by iterating single-sample inference."""
        if aligned_rgb_normalized_batch.ndim != 4 or aligned_rgb_normalized_batch.shape[1:] != (112, 112, 3):
            raise ValueError(
                "Expected batch shape (N, 112, 112, 3)"
            )
        embeddings = [self.embed(sample) for sample in aligned_rgb_normalized_batch]
        return np.stack(embeddings, axis=0).astype(np.float32)

    def _cast_input(self, batched_float32: np.ndarray) -> np.ndarray:
        if self._input_dtype == np.float32:
            return batched_float32
        # Simple fallback for quantized models without per-tensor scaling config:
        # clip normalized input to valid int8 range.
        if self._input_dtype == np.int8:
            return np.clip(np.round(batched_float32 * 127.0), -128, 127).astype(np.int8)
        if self._input_dtype == np.uint8:
            return np.clip(np.round((batched_float32 + 1.0) * 127.5), 0, 255).astype(np.uint8)
        return batched_float32.astype(self._input_dtype)

