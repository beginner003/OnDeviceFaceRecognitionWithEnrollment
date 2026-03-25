# OnDeviceFaceRecognitionWithEnrollment

On-device continual face recognition with forgetting prevention. The project provides an incremental registration + recognition pipeline, plus evaluation code and unit/integration tests.

## Read before you start
Please read `IMPLEMENTATION_PLAN.md` (module breakdown, architecture decisions, and how the pieces fit together).

## What’s included
- Pre-trained TFLite models in `src/models/`
- Dataset/enrollment fixtures in `data/`

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
    registration="naive",            # "naive" works; "replay"/"replay_lwf" are TODO stubs for now
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

## Create a virtual environment (for testing)
```bash
python3.11 -m venv venv
source venv/bin/activate
python --version  # expect Python 3.11.x
pip install -r requirements.txt
```

## Run tests
```bash
pytest -m "not integration"
pytest -m integration
```

Notes:
- `integration` tests may require optional runtime dependencies and real image fixtures.
