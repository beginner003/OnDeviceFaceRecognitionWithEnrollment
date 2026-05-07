# OnDeviceFaceRecognitionWithEnrollment

Raspberry Pi 5 on-device continual face recognition with forgetting prevention. The project provides a RealSense-based registration + recognition UI, evaluation scripts, and unit/integration tests for Pi-side validation.

## Read before you start
Please read `IMPLEMENTATION_PLAN.md` (module breakdown, architecture decisions, and how the pieces fit together).

## What’s included
- Pre-trained TFLite models in `src/models/`
- Dataset/enrollment fixtures in `data/`
- FastAPI browser UI in `src/ui/`
- Continual-learning experiments in `experiments/`

## Raspberry Pi 5 setup

This repo is intentionally Pi-only. Use 64-bit Raspberry Pi OS on Raspberry Pi 5.

Install Intel RealSense support before launching the UI. `pyrealsense2` is required at runtime, but it is not installed from `requirements.txt` because Raspberry Pi / Linux aarch64 has no normal PyPI wheel. Build/install Librealsense and its Python bindings for the Pi environment first.

```bash
python3.11 -m venv venv
source venv/bin/activate
python --version  # expect Python 3.11.x on Raspberry Pi OS
pip install -r requirements.txt

python - <<'PY'
import pyrealsense2
import ai_edge_litert.interpreter
print("Pi RealSense + TFLite runtime imports OK")
PY
```

## Launch the UI
Import the RealSense Library
```bash
export LD_LIBRARY_PATH=/home/comp4901d/librealsense/build/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/home/comp4901d/librealsense/wrappers/python${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPATH=/home/comp4901d/OnDeviceFaceRecognitionWithEnrollment:${PYTHONPATH}
```

Quick check:
```bash
python3.11 -c "import pyrealsense2 as rs; print('OK', rs)"
```

Run from the repository root on the Pi:

```bash
export PYTHONPATH=.
export FACE_UI_WORKSPACE=data/ui_workspace
export FACE_UI_CAPTURE_WIDTH=640
export FACE_UI_CAPTURE_HEIGHT=480
export FACE_UI_CAPTURE_FPS=30
python -m src.ui.app
```

Open `http://<pi-ip>:8000/` from a browser on the same network. The UI requires a connected Intel RealSense camera and fails fast if `pyrealsense2` or the device is unavailable.

## How to use `FaceRecognitionSystem`

`FaceRecognitionSystem` is the high-level orchestrator for class-incremental face registration and recognition. It operates on **pre-extracted 128-d embeddings** (so capture/detect/align/embed happen outside this class).

### Minimal example (NCM recognition + naive training)

```python
from pathlib import Path

import numpy as np

from src.system import FaceRecognitionSystem, SystemConfig


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


workspace = Path("experiments/run_demo_001")
cfg = SystemConfig(
    registration="naive",            # "naive" | "replay" | "replay_lwf" | "synthetic_replay"
    exemplar_selection="herding",   # "herding" or "random"
    recognition="ncm",              # "ncm" or "classifier"
    exemplar_k=8,
    confidence_threshold=0.3,
)

system = FaceRecognitionSystem.from_config(cfg, workspace=workspace)

# Embeddings must be shape (N, 128)
np.random.seed(0)
alice_embs = l2_normalize(np.random.randn(20, 128).astype(np.float32))
query = l2_normalize(np.random.randn(1, 128).astype(np.float32))[0]

system.register("alice", alice_embs)
name, conf = system.recognize(query)
print("prediction:", name, "confidence:", conf)
```

### Save / load

The system persists:
- exemplars under `workspace/exemplars/<identity>/exemplars.npz`
- the classifier head under `workspace/checkpoints/classifier.pt`
- identity-to-class mapping under `workspace/system_state.json`

So you can reload the same workspace like this:

```python
system = FaceRecognitionSystem.from_config(cfg, workspace=workspace)
name, conf = system.recognize(query)
```

## Run tests

```bash
PYTHONPATH=. python -m pytest -m "not integration"
PYTHONPATH=. python -m pytest -m integration
```

Notes:
- Integration tests require Pi runtime dependencies, TFLite models under `src/models/`, and real image fixtures.
- Hardware UI validation requires the RealSense camera on the Raspberry Pi 5.


Running Experiments
```bash
cd /home/comp4901d/OnDeviceFaceRecognitionWithEnrollment && export PYTHONPATH=. && for runner in experiments/replay_classifier/run.py experiments/replay_lwf_classifier/run.py experiments/lwf_classifier/run.py experiments/synthetic_replay_classifier/run.py; do for set_idx in {4..10}; do python experiments/run_set_trials.py --runner "$runner" --set-name "set${set_idx}" --python "$(which python)"; done; done
```