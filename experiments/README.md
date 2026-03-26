# Experiments

Baseline continual face-recognition runs use pre-extracted embeddings from [`embedding_helper.py`](embedding_helper.py) (detect → align → MobileFaceNet). Run scripts from the **repository root** with the root on `PYTHONPATH` so `experiments` and `src` resolve.

```bash
cd /path/to/OnDeviceFaceRecognitionWithEnrollment
export PYTHONPATH=.
```

Embedding extraction needs OpenCV (`cv2`) and a MobileFaceNet `.tflite` under `src/models/` (or set `MOBILEFACENET_TFLITE` / `MOBILEFACENET_MODEL_VARIANT` as in the helper).

---

## Classifier baseline (`baseline_classifier`)

Naive registration + herding (`exemplar_k=5`) + classifier recognition. Artifacts default to `experiments/baseline_classifier/embeddings/` and `experiments/baseline_classifier/workspace/`.

```bash
PYTHONPATH=. python experiments/baseline_classifier/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--experiment-root PATH` | Root for embeddings + workspace (default: `experiments/baseline_classifier`). |
| `--reset-workspace` | Delete the experiment workspace before running (recommended for a clean run). |
| `--overwrite-embeddings` | Recompute embeddings even if cache exists. |
| `--confidence-threshold FLOAT` | Recognition threshold (default: `0.5`). |

---

## NCM baseline (`baseline_ncm`)

Same registration setup; recognition uses nearest class mean (prototypes from the exemplar store). Defaults: `data/supertask_8_2.json`, `experiments/baseline_ncm/embeddings`, `experiments/baseline_ncm/workspace`.

```bash
PYTHONPATH=. python experiments/baseline_ncm/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--embeddings-root PATH` | Cache directory for per-identity embeddings (default: `experiments/baseline_ncm/embeddings`). |
| `--workspace PATH` | `FaceRecognitionSystem` workspace (default: `experiments/baseline_ncm/workspace`). |
| `--overwrite-embeddings` | Recompute embeddings even if cache exists. |
| `--reset-workspace` | Delete the workspace directory before running. |

Paths for `--supertask-json`, `--embeddings-root`, and `--workspace` may be absolute or relative to the repo root.

---

## Help

```bash
PYTHONPATH=. python experiments/baseline_classifier/run.py --help
PYTHONPATH=. python experiments/baseline_ncm/run.py --help
```
