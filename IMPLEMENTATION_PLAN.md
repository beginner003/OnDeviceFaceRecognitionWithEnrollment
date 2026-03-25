# Implementation Plan: On-Device Continual Face Recognition with Forgetting Prevention

> **Project 8 — COMP4901D**
> Target hardware: Raspberry Pi 5 (8 GB) + Intel RealSense depth camera

---

## Table of Contents

1. [Requirements Checklist](#1-requirements-checklist)
2. [Repository Structure](#2-repository-structure)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Module Breakdown & Implementation Details](#5-module-breakdown--implementation-details)
   - 5.1 [Camera Capture & Preprocessing](#51-camera-capture--preprocessing)
   - 5.2 [Face Detection](#52-face-detection)
   - 5.3 [Face Alignment & Embedding Extraction](#53-face-alignment--embedding-extraction)
   - 5.4 [Exemplar Memory Manager](#54-exemplar-memory-manager)
   - 5.5 [Continual Learning Engine](#55-continual-learning-engine)
   - 5.6 [Face Recognition System (Orchestrator)](#56-face-recognition-system-orchestrator)
   - 5.7 [Web UI (FastAPI)](#57-web-ui-fastapi)
6. [Continual Learning Methods (Detail)](#6-continual-learning-methods-detail)
   - 6.1 [Baseline — Naive Fine-Tuning](#61-baseline--naive-fine-tuning)
   - 6.2 [Method A — Exemplar Replay (iCaRL-style)](#62-method-a--exemplar-replay-icarl-style)
   - 6.3 [Method B — Exemplar Replay + LwF Distillation](#63-method-b--exemplar-replay--lwf-distillation)
7. [Memory-Efficient Strategies & Comparison](#7-memory-efficient-strategies--comparison)
8. [Evaluation Protocol](#8-evaluation-protocol)
   - 8.1 [Offline Evaluation (Laptop / Desktop)](#81-offline-evaluation-laptop--desktop)
   - 8.2 [On-Device Evaluation (Raspberry Pi 5)](#82-on-device-evaluation-raspberry-pi-5)
   - 8.3 [Metrics](#83-metrics)
9. [Performance Targets & Acceptance Criteria](#9-performance-targets--acceptance-criteria)
10. [12-Week Timeline](#10-12-week-timeline)
11. [Risk Mitigation](#11-risk-mitigation)

---

## 1. Requirements Checklist

Each row maps a **project description requirement** to the corresponding implementation section.

| # | Requirement (from project spec) | Plan Section | Status |
|---|---|---|---|
| R1 | Incremental learning pipeline with realistic registration workflow (5 family → +1 housekeeper → +2 neighbors) | §5.6, §6 | To Do |
| R2 | Integrate **at least two** anti-forgetting techniques and compare | §6.2, §6.3 | To Do |
| R3 | Memory-efficient strategies — compare raw images vs. compressed embeddings vs. synthetic replay | §7 | To Do |
| R4 | Intelligent exemplar selection (herding) to maximise retention within memory budget | §5.4 | To Do |
| R5 | Measure memory overhead of each approach | §7, §8.3 | To Do |
| R6 | Deploy on Raspberry Pi 5 and measure real-world performance | §8.2 | To Do |
| R7 | Test on face recognition datasets with class-incremental protocol | §8.1 | To Do |
| R8 | Report accuracy, forgetting rate, training time per new identity, memory usage | §8.3 | To Do |
| R9 | Support **≥10 people** with incremental updates completing **<1 min** per new identity | §9 | To Do |
| R10 | Maintain **>95% accuracy** on previously learned identities | §9 | To Do |
| R11 | Operate within **8 GB memory budget** | §9, §7 | To Do |
| R12 | Working prototype with **user registration interface** | §5.7 | To Do |
| R13 | System demonstration: register 5 → add 3 → verify all 8 | §8.2, §5.7 | To Do |

---

## 2. Repository Structure

```
OnDeviceFaceRecognitionWithEnrollment/
├── IMPLEMENTATION_PLAN.md          # This file
├── README.md
├── requirements.txt                # Python dependencies (pinned)
├── setup_rpi.sh                    # One-shot RPi 5 environment setup script
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All hyperparams, paths, thresholds
│   ├── protocols.py                # Strategy Protocol interfaces
│   ├── system.py                   # FaceRecognitionSystem orchestrator
│   │
│   ├── capture/
│   │   ├── __init__.py
│   │   └── realsense.py            # Intel RealSense RGB+Depth capture
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── blazeface.py            # MediaPipe BlazeFace wrapper
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   └── align.py                # eye-line rotation + bbox crop → 112×112
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── mobilefacenet.py        # TFLite inference wrapper
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── exemplar_store.py       # Exemplar buffer (images / embeddings)
│   │   ├── herding.py              # Herding-based exemplar selection (ExemplarSelector impl)
│   │   ├── random_selector.py      # Random exemplar selection (ExemplarSelector impl)
│   │   └── synthetic_replay.py     # Gaussian generative replay
│   │
│   ├── continual/
│   │   ├── __init__.py
│   │   ├── classifier.py           # CosineLinear classifier head
│   │   ├── naive_ft.py             # Baseline: naive fine-tuning (RegistrationStrategy impl)
│   │   ├── exemplar_replay.py      # Method A: replay only (RegistrationStrategy impl)
│   │   └── replay_lwf.py           # Method B: replay + LwF (RegistrationStrategy impl)
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── ncm.py                  # NCM: cosine sim vs prototypes (RecognitionStrategy impl)
│   │   └── classifier_based.py     # CosineLinear forward pass (RecognitionStrategy impl)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Accuracy, forgetting, BWT
│   │   ├── benchmark.py            # Offline VGGFace2 benchmark runner
│   │   └── profiler.py             # RAM / timing profiler (psutil)
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application
│   │   ├── static/                 # JS, CSS
│   │   └── templates/              # Jinja2 HTML templates
│   │
│   └── models/                     # Pre-trained model files
│       ├── mobilefacenet.tflite
│       └── blaze_face_full_range.tflite
│
├── data/
│   ├── vggface2_subset/            # 20-30 identities for offline eval
│   └── enrolled/                   # Runtime enrollment data (per-person dirs)
│
├── experiments/
│   ├── offline_benchmark.py        # Script: run full offline eval protocol
│   └── compare_methods.py          # Script: generate accuracy/forgetting plots
│
├── notebooks/
│   └── analysis.ipynb              # Result visualisation
│
└── tests/
    ├── test_detection.py
    ├── test_embedding.py
    ├── test_continual.py
    └── test_memory.py
```

---

## 3. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Face detection** | MediaPipe BlazeFace (short-range TFLite model) | ~5 ms on ARM; already have `blaze_face_full_range.tflite` in repo |
| **Face embedding** | MobileFaceNet via TensorFlow Lite interpreter | 0.99 M params, 128-dim output, ~15-30 ms on RPi 5 CPU; designed for edge; same runtime as BlazeFace detector |
| **Continual learning framework** | Custom PyTorch (CPU) + optional Avalanche for offline prototyping | Avalanche for rapid experimentation on laptop; lightweight custom code for RPi |
| **Classifier** | Cosine-normalised `nn.Linear(128, N)` in PyTorch CPU | Tiny, fast to train; cosine normalisation gives better few-shot performance |
| **Camera SDK** | `pyrealsense2` + OpenCV | Required for Intel RealSense depth camera |
| **Web server** | FastAPI + Uvicorn | Async, lightweight; MJPEG streaming support |
| **Profiling** | `psutil`, `/proc/self/status` | RAM and CPU monitoring on RPi 5 |
| **Datasets** | VGGFace2 subset (20-30 identities), optionally MS-Celeb-1M subset | Standard benchmarks specified by project brief |
| **Quantisation (optional)** | TFLite INT8 quantised model | Further speedup on ARM NEON via TFLite XNNPACK delegate |

### Key Python Dependencies

```
tensorflow-lite>=2.14   # or tflite-runtime for RPi 5
opencv-python-headless>=4.9
mediapipe>=0.10
torch>=2.2 (CPU wheel)
numpy>=1.26
scikit-learn>=1.4
fastapi>=0.110
uvicorn[standard]>=0.29
psutil>=5.9
pyrealsense2>=2.55
jinja2>=3.1
Pillow>=10.2
matplotlib>=3.8
avalanche-lib>=0.5   # laptop only, for offline prototyping
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER  (Web UI / CLI / Benchmark scripts)                     │
│  Owns: camera lifecycle, live video overlay, user interaction              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CAMERA INPUT (Intel RealSense — pyrealsense2 + OpenCV)            │    │
│  │  RGB stream @ 640×480, optional depth for liveness                  │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PREPROCESSING                                                      │    │
│  │  ┌──────────────────────┐    ┌────────────────────────────────────┐ │    │
│  │  │ Face Detection       │    │ Face Alignment                     │ │    │
│  │  │ MediaPipe BlazeFace  │───▶│ eye-line rotation + crop → 112×112│ │    │
│  │  │ ~5 ms per frame      │    │ Normalise to [-1, 1]               │ │    │
│  │  └──────────────────────┘    └──────────────────┬─────────────────┘ │    │
│  └─────────────────────────────────────────────────┼───────────────────┘    │
│                                                     ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FEATURE EXTRACTION (Frozen Backbone)                               │    │
│  │  MobileFaceNet TFLite — 0.99 M params, 128-dim embedding           │    │
│  │  TFLite interpreter + XNNPACK delegate — ~15-30 ms per face        │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                  │ 128-dim embeddings                       │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FACE RECOGNITION SYSTEM  (Orchestrator — src/system.py)                   │
│  Facade + Strategy pattern; all APIs operate on 128-d embeddings           │
│                                                                             │
│  ┌────────────── Pluggable Strategies ──────────────────────────────────┐   │
│  │                                                                      │   │
│  │  RegistrationStrategy       ExemplarSelector    RecognitionStrategy  │   │
│  │  ├─ NaiveFT                 ├─ Herding           ├─ NCM (prototype) │   │
│  │  ├─ ExemplarReplay          ├─ Random             └─ Classifier     │   │
│  │  └─ ReplayLwF               └─ (raw images)          (CosineLinear) │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────────────────┐    ┌─────────────────────────────────────┐   │
│  │  register(name, embs)     │    │  recognize(emb) → (name, conf)     │   │
│  │   1. ExemplarSelector     │    │   1. RecognitionStrategy.predict()  │   │
│  │   2. ExemplarStore.upsert │    │   2. Confidence threshold           │   │
│  │   3. RegistrationStrategy │    │   3. Return identity or "Unknown"   │   │
│  │   4. Save checkpoint      │    │                                     │   │
│  └───────────────────────────┘    └─────────────────────────────────────┘   │
│                                                                             │
│  Owns: workspace/                                                           │
│        ├── exemplars/          (ExemplarStore root)                         │
│        ├── checkpoints/        (classifier .pt files)                      │
│        └── logs/               (per-run metrics & timing logs)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  WEB UI (FastAPI + Jinja2 + JS)                                            │
│  /video_feed — MJPEG live stream with bounding boxes + names               │
│  /register  — Start registration (name + N auto-captured frames)           │
│  /status    — Enrolled count, last update time, RAM usage, accuracy        │
│  /evaluate  — Trigger on-demand accuracy check on enrolled set             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Inference (single frame)

1. **Application layer:** RealSense captures RGB frame (640×480)
2. **Application layer:** BlazeFace detects faces → bounding boxes + 6 keypoints
3. **Application layer:** Align each face → 112×112 crop → MobileFaceNet → 128-dim embedding
4. **System:** `system.recognize(embedding)` → delegates to `RecognitionStrategy`
5. **System:** Apply confidence threshold; below threshold → "Unknown"
6. **Application layer:** Draw bounding box + name on frame; stream via MJPEG

### Data Flow — Registration (new person)

1. **Application layer:** User clicks "Register" in web UI, enters name
2. **Application layer:** Capture 20-30 frames → detect + align → MobileFaceNet → `(N, 128)` embeddings
3. **System:** `system.register(name, embeddings)`:
   - `ExemplarSelector` selects K best exemplars (herding / random)
   - `ExemplarStore` persists exemplars under identity name
   - `RegistrationStrategy` expands classifier + trains (naive / replay / replay+LwF)
   - Save classifier checkpoint + exemplar store to workspace
4. **System:** Return `RegistrationResult` (time taken, RAM usage, new identity count)

---

## 5. Module Breakdown & Implementation Details

### 5.1 Camera Capture & Preprocessing

**File:** `src/capture/realsense.py`

| Item | Detail |
|---|---|
| Library | `pyrealsense2` |
| RGB resolution | 640×480 @ 30 FPS |
| Depth stream | Optional; used for liveness (reject flat photos) and distance filtering (ignore faces >2 m) |
| Output | NumPy array `(480, 640, 3)` BGR |
| Fallback | OpenCV `VideoCapture(0)` for laptop development without RealSense |

**Key implementation:**
- Context manager wrapping the RealSense pipeline for clean start/stop
- Thread-based frame grabber to decouple capture from processing
- Configurable resolution and FPS via `src/config.py`

### 5.2 Face Detection

**File:** `src/detection/blazeface.py`

| Item | Detail |
|---|---|
| Model | MediaPipe BlazeFace short-range (already have `blaze_face_full_range.tflite`) |
| Input | RGB frame |
| Output | List of `Detection(bbox, landmarks_6pt, confidence)` |
| Latency target | ~5 ms on RPi 5 |
| Min confidence | 0.7 (configurable) |

**Key implementation:**
- Wrap MediaPipe Tasks `FaceDetector` (BlazeFace TFLite)
- Expose all 6 BlazeFace keypoints in pixel space on each `Detection` (no reduction to a synthetic 5-point set)
- Batch detection not needed (single frame, typically 1-3 faces)

> [!IMPORTANT]
> **IMPORTANT NOTES**
> - Landmark-based alignment is required instead of bbox-only cropping because a bounding box gives location/scale only, while keypoints encode enough geometry to correct in-plane roll before embedding extraction.
> - BlazeFace outputs 6 keypoints (`right_eye`, `left_eye`, `nose_tip`, `mouth_center`, `right_ear_tragion`, `left_ear_tragion`). The in-repo aligner uses **only the two eye centres** plus the detector bounding box; the other keypoints are carried for visualization, logging, and future use.

### 5.3 Face Alignment & Embedding Extraction

**Files:** `src/alignment/align.py`, `src/embedding/mobilefacenet.py`

**Alignment:**
- Use **anatomical left/right eye centres** from the detector (BlazeFace keypoints) to estimate in-plane roll: angle of the eye line w.r.t. horizontal, then rotate the face region so the eyes are level.
- Take the **detector bounding box** `(x, y, w, h)` (required; no landmark-only bbox fallback), **expand** it around its centre (same idea as `area_expand` in the reference), crop that ROI, **rotate about the eye midpoint** with `cv2.getRotationMatrix2D` + `cv2.warpAffine`, then **re-crop** using the transformed original-box corners and **resize** to `112×112`.
- Output: `(112, 112, 3)` aligned face crop; embedding step normalises RGB to `[-1, 1]`.

> [!NOTE]
> **Reference workflow (eye-line alignment)**  
> The in-repo aligner follows the same *idea* as the MakerPRO walkthrough: compute the tilt from the two eyes, work on an **expanded** face region so rotation does not clip the face, then derive a tight aligned crop for recognition. That article uses dlib 5-point landmarks and a second detection pass on the rotated patch; here we reuse **BlazeFace eye centres + bbox** instead of dlib, but the geometry (eye-based angle, expanded ROI, rotate, re-crop) matches the described pipeline.  
> **Reference:** [【人臉辨識】使用5 Facial Landmarks進行臉孔校正 (MakerPRO)](https://makerpro.cc/2019/08/face-alignment-through-5-facial-landmarks/)

**Embedding:**
- Load `mobilefacenet.tflite` with TensorFlow Lite `Interpreter`
- Enable XNNPACK delegate for optimised ARM NEON inference: `tf.lite.experimental.load_delegate('libXNNPACK.so')`
- Input: `(1, 112, 112, 3)` float32 tensor (NHWC format, matching TFLite convention)
- Output: `(1, 128)` embedding → L2-normalise
- Optional: INT8 quantised `.tflite` model for further ARM speedup

### 5.4 Exemplar Memory Manager

**Files:** `src/memory/exemplar_store.py`, `src/memory/herding.py`, `src/memory/synthetic_replay.py`

**Exemplar Store:**
- Data structure: `Dict[str, ExemplarSet]` where each `ExemplarSet` holds:
  - `embeddings: np.ndarray` of shape `(K, 128)` in float16
  - `prototype: np.ndarray` of shape `(128,)` — class mean
  - `image_paths: List[str]` (optional, for raw image strategy)
- Persistence: pickle or numpy `.npz` to `data/enrolled/<person_name>/`
- Memory budget tracking: `total_bytes()` method

**Herding Selection (iCaRL-style):**
- Given N embeddings for a person, select K that best approximate the class mean
- Greedy algorithm: iteratively pick the embedding that brings the running mean closest to the true mean
- This satisfies requirement R4 (intelligent sample selection)

**Synthetic Replay (for comparison):**
- Fit per-class Gaussian: compute mean μ and covariance Σ from stored embeddings
- Sample synthetic embeddings: `np.random.multivariate_normal(μ, Σ, n_samples)`
- L2-normalise synthetic samples before use
- Storage: only μ (128 floats) + Σ (128×128 floats) per class = ~66 KB per person in float16

### 5.5 Continual Learning Engine

**Files:** `src/continual/classifier.py`, `src/continual/naive_ft.py`, `src/continual/exemplar_replay.py`, `src/continual/replay_lwf.py`

**CosineLinear Classifier:**

> [!IMPORTANT]
> **IMPORTANT NOTES**
> - **Matches the embedding geometry:** MobileFaceNet outputs are used with **L2-normalised** 128-d vectors and **cosine** similarity at inference (and for **NCM** prototypes in §5.6). CosineLinear scores each class by cosine similarity between the query and a **learnable per-class direction**, so training optimises the same geometry the system already uses for matching.
> - **Magnitude-invariant decisions:** A standard linear layer mixes direction and (implicit) scale; cosine normalisation makes class logits depend on **direction only**, which tends to be more stable when batch sizes, lighting, or per-person sample counts vary across incremental steps.
> - **Few-shot and incremental steps:** New identities arrive with **small** embedding sets; cosine-style heads often generalise better than unconstrained dot products when each class has few training points.
> - **Learnable scale:** Multiplying cosine logits by a **trainable scalar** (temperature) keeps the head tiny while still letting the optimiser set appropriate softmax sharpness.

```python
class CosineLinear(nn.Module):
    def __init__(self, in_features=128, out_features=0):
        # Weight matrix W: (out_features, 128)
        # Forward: cosine_sim(x, W) * scale
    def expand(self, n_new_classes):
        # Expand weight matrix to accommodate new classes
        # Initialise new rows with mean of new class embeddings
```

**Training loop shared across methods:**
- Optimiser: SGD with momentum 0.9 or Adam, lr=0.01
- Batch size: 16-32 (fits RPi 5 CPU memory)
- Epochs: 5-10 per incremental update
- All training on 128-dim embeddings (backbone frozen, no gradient buffers for backbone)

See §6 for method-specific details.

### 5.6 Face Recognition System (Orchestrator)

**Files:** `src/protocols.py`, `src/system.py`, `src/recognition/ncm.py`, `src/recognition/classifier_based.py`

> [!IMPORTANT]
> **Design Rationale**
>
> The previous plan (v1) had §5.6 Recognition Pipeline and §5.7 Registration Pipeline as separate, standalone modules. This revision unifies them into a single **`FaceRecognitionSystem`** orchestrator class following two established design patterns:
>
> - **Facade Pattern:** one entry point for the entire recognition + registration subsystem. Callers (web UI, CLI, benchmark scripts, tests) interact with a single object instead of wiring up `ExemplarStore`, `CosineLinear`, herding, and training loops manually.
> - **Strategy Pattern:** registration method, exemplar selection, and recognition mode are injected as pluggable strategy objects conforming to Python `Protocol` interfaces. Switching from exemplar replay to LwF distillation is a one-line config change, not a code change.
>
> **Layered architecture:** Camera capture, face detection, and alignment stay in the **application layer** (web UI, CLI, benchmark). The orchestrator operates exclusively on **128-dim embeddings**, keeping the core domain independent of I/O and hardware. This means offline benchmarks (VGGFace2) and unit tests can use the system without a camera or TFLite model.

#### Strategy Protocol Interfaces

**File:** `src/protocols.py`

Three Protocol classes define the pluggable axes. Each existing module implements one of these protocols.

```python
from typing import Protocol, Tuple
import numpy as np
from src.continual.classifier import CosineLinear
from src.memory.exemplar_store import ExemplarStore

class RegistrationStrategy(Protocol):
    """Incremental learning update when a new identity is registered."""
    def update(
        self,
        classifier: CosineLinear,
        store: ExemplarStore,
        new_embeddings: np.ndarray,
        identity: str,
    ) -> CosineLinear: ...

class ExemplarSelector(Protocol):
    """Select K representative exemplars from N candidate embeddings."""
    def select(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (selected_embeddings, selected_indices)."""
        ...

class RecognitionStrategy(Protocol):
    """Predict identity from a single query embedding."""
    def predict(
        self,
        embedding: np.ndarray,
        store: ExemplarStore,
        classifier: CosineLinear,
    ) -> Tuple[str, float]:
        """Return (identity_name, confidence). Return ("unknown", conf) if below threshold."""
        ...
```

#### Strategy Implementations

| Protocol | Implementation | File | Description |
|---|---|---|---|
| `RegistrationStrategy` | `NaiveFTStrategy` | `src/continual/naive_ft.py` | Wraps `incremental_train_naive`; no replay (forgetting baseline) |
| `RegistrationStrategy` | `ExemplarReplayStrategy` | `src/continual/exemplar_replay.py` | Balanced replay from exemplar store (iCaRL-style) |
| `RegistrationStrategy` | `ReplayLwFStrategy` | `src/continual/replay_lwf.py` | Replay + knowledge distillation from frozen teacher |
| `ExemplarSelector` | `HerdingSelector` | `src/memory/herding.py` | Greedy selection minimising distance to class mean |
| `ExemplarSelector` | `RandomSelector` | `src/memory/random_selector.py` | Uniform random baseline for comparison |
| `RecognitionStrategy` | `NCMRecognizer` | `src/recognition/ncm.py` | Cosine similarity against stored class prototypes; argmax |
| `RecognitionStrategy` | `ClassifierRecognizer` | `src/recognition/classifier_based.py` | Forward pass through CosineLinear head; argmax on logits |

#### FaceRecognitionSystem Class

**File:** `src/system.py`

```python
@dataclass(frozen=True)
class SystemConfig:
    registration: str = "replay_lwf"     # "naive" | "replay" | "replay_lwf"
    exemplar_selection: str = "herding"  # "herding" | "random"
    recognition: str = "classifier"      # "ncm" | "classifier"
    exemplar_k: int = 50                 # exemplars per class
    confidence_threshold: float = 0.5
```

```python
class FaceRecognitionSystem:
    """
    Top-level orchestrator for face registration and recognition.

    Operates at the embedding level (128-d vectors). Camera capture,
    face detection, and alignment are the caller's responsibility
    (application layer).
    """

    def __init__(
        self,
        registration_strategy: RegistrationStrategy,
        exemplar_selector: ExemplarSelector,
        recognition_strategy: RecognitionStrategy,
        workspace: Path,
        *,
        exemplar_k: int = 50,
        confidence_threshold: float = 0.5,
        embedding_model: MobileFaceNetWrapper | None = None,
    ) -> None:
        # Creates workspace subdirectories:
        #   workspace/exemplars/
        #   workspace/checkpoints/
        #   workspace/logs/
        ...

    # ── Core API ──────────────────────────────────────────────

    def register(self, name: str, embeddings: np.ndarray) -> RegistrationResult:
        """
        Register a new identity from pre-extracted embeddings.

        Steps:
          1. exemplar_selector.select(embeddings, k)
          2. exemplar_store.upsert_class(name, selected)
          3. registration_strategy.update(classifier, store, embeddings, name)
          4. Save classifier checkpoint + exemplar store
          5. Log timing and memory metrics
        """
        ...

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Identify a person from a single 128-d query embedding.
        Returns ("unknown", confidence) if below threshold.
        """
        ...

    def identities(self) -> List[str]:
        """List all registered identity names."""
        ...

    def remove_identity(self, name: str) -> None:
        """Remove an identity and its exemplars."""
        ...

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Persist classifier checkpoint + exemplar store to workspace."""
        ...

    def load(self) -> None:
        """Restore classifier + exemplar store from workspace."""
        ...

    # ── Factory ───────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: SystemConfig, workspace: Path) -> "FaceRecognitionSystem":
        """
        Convenience constructor: build strategy objects from config strings.

        Example:
            system = FaceRecognitionSystem.from_config(
                SystemConfig(registration="replay_lwf", recognition="ncm"),
                workspace=Path("experiments/run_001"),
            )
        """
        ...
```

#### Workspace Layout

Each `FaceRecognitionSystem` instance owns an isolated workspace directory. This enables running multiple experiments with different configurations side by side.

```
workspace/                          # e.g. experiments/run_001/
├── exemplars/                      # ExemplarStore root (per-identity .npz files)
│   ├── alice/exemplars.npz
│   ├── bob/exemplars.npz
│   └── ...
├── checkpoints/
│   └── classifier.pt               # CosineLinear state_dict
└── logs/
    ├── registration.log             # Per-registration timing & memory
    └── recognition.log              # Per-query predictions (optional)
```

#### Recognition Modes (from old §5.6)

Both recognition strategies receive the same inputs and return `(identity, confidence)`:

- **NCM (Nearest Class Mean):** Compute cosine similarity of the query embedding against each class prototype in `ExemplarStore`. Argmax over similarities. Threshold at `confidence_threshold`; below → `"unknown"`.
- **Classifier-based:** Forward pass through `CosineLinear`; softmax over logits; argmax. Threshold on max probability.

#### Registration Flow (from old §5.7)

The `system.register()` method orchestrates the full flow. Embeddings are provided by the caller (application layer handles capture → detect → align → embed):

```
system.register(name="Alice", embeddings=embeddings)
  │
  ├─ 1. ExemplarSelector.select(embeddings, k=50)
  │     → K exemplar embeddings + indices
  │
  ├─ 2. ExemplarStore.upsert_class("Alice", selected_embeddings)
  │     → persists exemplars; updates prototype
  │
  ├─ 3. RegistrationStrategy.update(classifier, store, embeddings, "Alice")
  │     ├─ Expand classifier head by 1 class
  │     ├─ Build balanced training set (new + old exemplars)
  │     ├─ Train for E epochs with chosen method
  │     └─ (ReplayLwF only) Distill from frozen teacher
  │
  ├─ 4. Save classifier checkpoint + exemplar store to workspace
  │
  └─ 5. Return RegistrationResult(time_s, peak_ram_mb, n_identities)
```

**Timing budget (RPi 5) — registration only (excluding capture/embedding):**
- Herding selection: <0.1 s
- Classifier training (10 epochs, ~200 samples): ~10-30 s
- **Total system.register(): well under 1 minute** (requirement R9)

### 5.7 Web UI (FastAPI)

**Files:** `src/ui/app.py`, `src/ui/templates/`, `src/ui/static/`

**API Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main page with live video + controls |
| `/video_feed` | GET | MJPEG stream (bounding boxes + name overlays) |
| `/register` | POST | Start registration `{name: str, n_frames: int}` |
| `/register/status` | GET | SSE stream for registration progress |
| `/identities` | GET | List of enrolled identities with exemplar count |
| `/identities/{name}` | DELETE | Remove an identity |
| `/status` | GET | System metrics: enrolled count, RAM, last update |
| `/evaluate` | POST | Trigger accuracy check on current enrolled set |

**Frontend features:**
- Live video feed with face bounding boxes and predicted names
- "Register New Person" button → modal with name input + live preview
- Progress bar during registration (frame capture → embedding → training)
- Dashboard showing: enrolled identities, memory usage, last registration time
- Accuracy report after evaluation

---

## 6. Continual Learning Methods (Detail)

All three methods share the same frozen MobileFaceNet backbone. Only the classifier head is updated during incremental learning.

### 6.1 Baseline — Naive Fine-Tuning

**File:** `src/continual/naive_ft.py`

- When a new person arrives, expand classifier and train **only on the new person's embeddings** (no replay)
- Loss: standard cross-entropy
- Expected result: **severe catastrophic forgetting** on old identities
- Purpose: establishes a lower bound to demonstrate the value of anti-forgetting techniques

### 6.2 Method A — Exemplar Replay (iCaRL-style)

**File:** `src/continual/exemplar_replay.py`

**Algorithm:**
```
Input: new_embeddings, new_label, exemplar_store, classifier

1. Expand classifier: add 1 output node for new_label
2. Build exemplars for new person via herding (K samples)
3. Add to exemplar_store
4. Form balanced training set:
     For each class c in exemplar_store:
       sample min(K, available) embeddings
     → balanced dataset D with ~K samples per class
5. Train classifier on D:
     For epoch in 1..E:
       For batch in DataLoader(D, batch_size=32, shuffle=True):
         logits = classifier(batch_embeddings)
         loss = CrossEntropy(logits, batch_labels)
         loss.backward()
         optimizer.step()
6. Update prototypes: for each class, prototype = mean(exemplar_embeddings)
```

**Hyperparameters:**
- K = 30-50 exemplars per class
- E = 5-10 epochs
- lr = 0.01, SGD with momentum 0.9
- Batch size = 32

### 6.3 Method B — Exemplar Replay + LwF Distillation

**File:** `src/continual/replay_lwf.py`

**Algorithm:**
```
Input: new_embeddings, new_label, exemplar_store, classifier

1. Save current classifier as teacher (frozen copy)
2. Expand classifier: add 1 output node for new_label
3. Build exemplars + training set (same as Method A steps 2-4)
4. Train classifier on D:
     For epoch in 1..E:
       For batch in DataLoader(D, batch_size=32, shuffle=True):
         student_logits = classifier(batch_embeddings)          # all classes
         teacher_logits = teacher(batch_embeddings)              # old classes only

         L_CE     = CrossEntropy(student_logits, batch_labels)
         L_distill = KL_Divergence(
                       softmax(student_logits[:, :n_old] / T),
                       softmax(teacher_logits / T)
                     ) * T²

         loss = L_CE + λ * L_distill
         loss.backward()
         optimizer.step()
5. Update prototypes
6. Delete teacher (free memory)
```

**Additional hyperparameters:**
- T = 2.0 (distillation temperature)
- λ = 1.0 (distillation weight)

**Expected outcome:** better retention on old identities than Method A alone, at negligible extra cost (teacher classifier is tiny).

---

## 7. Memory-Efficient Strategies & Comparison

Requirement R3 mandates comparing three memory strategies. Each is implemented and benchmarked.

### Strategy 1: Raw Exemplar Images

- Store K aligned face crops (112×112×3, uint8) per person on disk
- During incremental update: load crops → run through backbone → get embeddings → train
- **Storage per person:** K × 112 × 112 × 3 × 1 byte = K × 37.6 KB
  - K=50 → 1.88 MB per person; 20 people → 37.6 MB
- **RAM during training:** must load images + run backbone forward pass
- **Pro:** can re-extract embeddings if backbone changes
- **Con:** slower update (backbone forward needed), higher storage

### Strategy 2: Compressed Feature Vectors (Embeddings in float16)

- Store K embeddings per person as float16 arrays
- No backbone forward pass needed during update
- **Storage per person:** K × 128 × 2 bytes = K × 256 bytes
  - K=50 → 12.5 KB per person; 20 people → 250 KB
- **RAM during training:** negligible (~250 KB total)
- **Pro:** extremely fast and memory-efficient; backbone stays frozen anyway
- **Con:** cannot re-extract if backbone changes (not a concern for this project)

### Strategy 3: Synthetic Replay (Gaussian Generative)

- Store per-class Gaussian parameters: μ (128 floats) + Σ (128×128 floats)
- Generate synthetic embeddings on-the-fly via `np.random.multivariate_normal`
- **Storage per person:** (128 + 128×128) × 2 bytes ≈ 33 KB (float16)
  - 20 people → 660 KB
- **RAM during training:** generate on-the-fly, no persistent buffer
- **Pro:** fixed storage per class regardless of sample count; novel approach
- **Con:** may lose fine-grained distribution details; quality depends on Gaussian fit

### Comparison Experiment

| Metric | Strategy 1 (Images) | Strategy 2 (Embeddings) | Strategy 3 (Synthetic) |
|---|---|---|---|
| Disk storage (20 ppl) | ~37.6 MB | ~250 KB | ~660 KB |
| RAM overhead (training) | ~50 MB (backbone in RAM) | ~250 KB | ~1 MB (sampling) |
| Update time per person | ~45 s (backbone + train) | ~15 s (train only) | ~15 s (sample + train) |
| Accuracy retention | Baseline reference | Expected: comparable | Expected: slightly lower |

All three strategies will be benchmarked in the offline evaluation (§8.1) and the best-performing one used for the real-world demo.

### RPi 5 Memory Budget

| Component | RAM Usage |
|---|---|
| Raspberry Pi OS + desktop | ~1.5 GB |
| Python + FastAPI + OpenCV | ~200 MB |
| TFLite interpreter + MobileFaceNet loaded | ~30 MB |
| MediaPipe face detection | ~20 MB |
| PyTorch CPU (classifier training) | ~500 MB |
| Exemplar store (20 ppl × 50 embeddings) | ~0.25 MB |
| **Total** | **~2.25 GB** |
| **Remaining headroom** | **~5.75 GB** |

Comfortably within the 8 GB budget (R11).

---

## 8. Evaluation Protocol

### 8.1 Offline Evaluation (Laptop / Desktop)

**Dataset:** VGGFace2 subset — select 20 identities, ~100 images each.

**Task split (class-incremental):**

| Task | Identities | Cumulative Total |
|---|---|---|
| Task 0 (initial) | ID 1-5 (family) | 5 |
| Task 1 | ID 6-7 (housekeeper, neighbor 1) | 7 |
| Task 2 | ID 8-9 (neighbor 2, visitor 1) | 9 |
| Task 3 | ID 10-11 | 11 |
| Task 4 | ID 12-14 | 14 |
| Task 5 | ID 15-17 | 17 |
| Task 6 | ID 18-20 | 20 |

**Protocol per task:**
1. Extract embeddings for new identities (using frozen MobileFaceNet)
2. Run incremental update with each method (Naive, Method A, Method B)
3. Evaluate on held-out test set of **all identities seen so far**
4. Log: accuracy, per-class accuracy, forgetting, training time, peak RAM

**Tool:** Avalanche library for scenario setup and metrics on laptop; custom `src/evaluation/benchmark.py` for the actual runs.

### 8.2 On-Device Evaluation (Raspberry Pi 5)

**Scenario (matching project brief exactly):**
1. Register 5 initial people (you + 4 friends) → Task 0
2. Add person 6 (housekeeper role) → Task 1
3. Add person 7 (neighbor 1) → Task 2
4. Add person 8 (neighbor 2) → Task 3
5. (Stretch) Add persons 9-10 → Task 4

**Per-task evaluation:**
- For each enrolled person, capture 10 test frames under varied conditions (different lighting, angles)
- Run recognition on all test frames
- Record top-1 accuracy per person + overall accuracy
- Generate confusion matrix
- Log: training time, peak RAM via `psutil.Process().memory_info().rss`

### 8.3 Metrics

| Metric | Formula / Method | Target |
|---|---|---|
| **Average accuracy** | Mean of per-class accuracy after all tasks | >95% on old identities |
| **Backward transfer (forgetting)** | `Forgetting(i) = max_{t≤T} A_{i,t} - A_{i,T}` | Minimise |
| **Average forgetting** | Mean forgetting across all old classes | <5% |
| **Incremental training time** | Wall-clock from "start update" to "classifier saved" | <60 s per identity |
| **Peak RAM during training** | `psutil.Process().memory_info().rss` sampled every 0.5 s | <8 GB |
| **Peak RAM during inference** | Same measurement during live recognition | <4 GB |
| **Disk storage** | Size of `data/enrolled/` directory | Report per strategy |
| **Inference latency** | Time from frame capture to identity output | <100 ms (target 10+ FPS) |

---

## 9. Performance Targets & Acceptance Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Supported identities | ≥10 people | Verified in offline eval (20 people) + on-device eval (8-10 people) |
| Accuracy on old identities | >95% after incremental updates | Per-class accuracy averaged; measured after each task |
| Incremental update time | <1 minute per new identity on RPi 5 | Wall-clock timer around `system.register()` |
| Memory budget | <8 GB total RAM on RPi 5 | `psutil` peak measurement during training |
| Anti-forgetting methods | ≥2 methods compared | Naive vs Method A vs Method B; accuracy & forgetting tables |
| Memory strategies | 3 strategies compared with overhead measurements | Images vs embeddings vs synthetic; storage + RAM tables |
| Working demo | Live video + registration UI on RPi 5 | Video recording of full workflow |
| Demo scenario | Register 5 → add 3 → recognise all 8 | Confusion matrix showing all 8 correctly identified |

---

## 10. 12-Week Timeline

### Phase 1: Foundation (Weeks 1-3)

**Week 1 — Environment & Literature**
- [ ] Set up RPi 5: flash OS, install dependencies, test RealSense camera
- [ ] Set up dev environment on laptop (PyTorch, Avalanche, TFLite runtime)
- [ ] Read core papers: iCaRL, LwF, MobileFaceNet, Avalanche
- [ ] Download and curate VGGFace2 subset (20 identities, ~100 images each)

**Week 2 — Baseline Face Recognition Pipeline**
- [ ] Implement `src/capture/realsense.py` (camera capture)
- [ ] Implement `src/detection/blazeface.py` (face detection)
- [ ] Implement `src/alignment/align.py` (landmark-based alignment)
- [ ] Implement `src/embedding/mobilefacenet.py` (TFLite embedding extraction)
- [ ] Build and test non-incremental face recognition on laptop (NCM on 5 people)

**Week 3 — Port to RPi 5**
- [ ] Obtain MobileFaceNet TFLite float32 model for ARM deployment
- [ ] Test full inference pipeline on RPi 5: capture → detect → align → embed → recognise
- [ ] Measure end-to-end latency; target 10-15 FPS for full pipeline
- [ ] Optional: switch to INT8 quantised TFLite model if latency is too high

### Phase 2: Continual Learning — Offline (Weeks 4-6)

**Week 4 — Incremental Protocol & Baseline**
- [ ] Implement `src/continual/classifier.py` (CosineLinear with expand)
- [ ] Implement class-incremental protocol in `experiments/offline_benchmark.py`
- [ ] Implement `src/continual/naive_ft.py` (naive fine-tuning baseline)
- [ ] Run baseline on VGGFace2 subset; measure catastrophic forgetting

**Week 5 — Method A: Exemplar Replay**
- [ ] Implement `src/memory/herding.py` (herding-based exemplar selection)
- [ ] Implement `src/memory/exemplar_store.py` (store/load/manage exemplars)
- [ ] Implement `src/continual/exemplar_replay.py`
- [ ] Run experiments varying K (10, 20, 30, 50 exemplars per class)
- [ ] Compare Method A vs Baseline: accuracy, forgetting, memory

**Week 6 — Method B: Replay + LwF Distillation**
- [ ] Implement `src/continual/replay_lwf.py` (teacher-student distillation)
- [ ] Run experiments: tune λ (0.5, 1.0, 2.0) and T (1.0, 2.0, 4.0)
- [ ] Full comparison: Baseline vs Method A vs Method B
- [ ] Implement `src/memory/synthetic_replay.py` (Gaussian generative replay)
- [ ] Run memory strategy comparison (images vs embeddings vs synthetic)

### Phase 3: On-Device Continual Learning (Weeks 7-9)

**Week 7 — Port Method A to RPi 5**
- [ ] Move exemplar replay to RPi 5 (CPU-only PyTorch training)
- [ ] Implement `src/evaluation/profiler.py` (RAM + timing instrumentation)
- [ ] Measure training time for adding 1-3 identities; tune epochs/lr/batch size
- [ ] Verify <1 minute per identity on RPi 5

**Week 8 — Port Method B + Logging**
- [ ] Implement distillation on RPi 5
- [ ] Add comprehensive logging: time per phase, peak RAM, accuracy after update
- [ ] Compare Method A vs B on RPi 5 with real timing data

**Week 9 — On-Device Experiments**
- [ ] Run full class-incremental protocol on RPi 5 with VGGFace2 subset
- [ ] Run experiments with live-captured data (you + classmates)
- [ ] Collect results: accuracy tables, forgetting curves, timing, memory

### Phase 4: Demo & Report (Weeks 10-12)

**Week 10 — Web UI**
- [ ] Implement `src/ui/app.py` (FastAPI backend)
- [ ] Build frontend: live video, registration form, status dashboard
- [ ] Integrate registration pipeline with UI (SSE progress updates)
- [ ] Test full workflow: open browser → view live feed → register person → see updated recognition

**Week 11 — Real-World Demo & Tuning**
- [ ] Conduct full demo scenario: register 5 → add 3 → verify all 8
- [ ] Generate confusion matrices and accuracy reports
- [ ] Tune hyperparameters if accuracy <95% on old identities
- [ ] Record demo video showing complete workflow

**Week 12 — Report & Presentation**
- [ ] Write final report: motivation, literature, architecture, experiments, analysis, limitations
- [ ] Generate all plots: accuracy curves, forgetting curves, memory comparison, timing
- [ ] Prepare presentation slides
- [ ] Final code cleanup, documentation, and README update

---

## 11. Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| MobileFaceNet TFLite too slow on RPi 5 | Inference >50 ms/face | Switch to INT8 quantised TFLite model; enable XNNPACK delegate; reduce input resolution to 96×96; profile with TFLite benchmark tool |
| Catastrophic forgetting >5% even with Method B | Fails >95% accuracy target | Increase exemplar count K; try EWC as a third method; fine-tune distillation hyperparameters |
| Training time >1 min on RPi 5 | Fails timing target | Reduce epochs (3-5); reduce exemplar count; use NCM (no training) as fallback recognition |
| RealSense driver issues on RPi 5 | Camera unusable | Fall back to Pi Camera Module 3 with `picamera2`; use USB webcam with OpenCV |
| Thermal throttling on RPi 5 | Performance degrades during training | Use active cooler (official RPi 5 Active Cooler); monitor CPU temp; add cooldown period between registrations |
| VGGFace2 download issues | Cannot run offline eval | Use LFW (Labeled Faces in the Wild) as alternative; curate own small dataset from team photos |
| Insufficient test subjects | Cannot demonstrate 8+ people | Use VGGFace2 identities displayed on screen for additional "virtual" registrations; recruit more classmates |
