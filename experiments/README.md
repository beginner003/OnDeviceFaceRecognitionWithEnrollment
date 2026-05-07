# Experiments

Baseline continual face-recognition runs use pre-extracted embeddings from [`embedding_helper.py`](embedding_helper.py) (detect -> align -> MobileFaceNet). Run scripts on the **Raspberry Pi 5** from the repository root with the root on `PYTHONPATH` so `experiments` and `src` resolve.

```bash
cd /path/to/OnDeviceFaceRecognitionWithEnrollment
export PYTHONPATH=.
```

Embedding extraction needs OpenCV (`cv2`) and a MobileFaceNet `.tflite` under `src/models/` (or set `MOBILEFACENET_TFLITE` / `MOBILEFACENET_MODEL_VARIANT` as in the helper).

Each experiment renews its own `logs/evaluation.log` on every run. Logs report the midterm metrics: overall/per-class accuracy, forgetting/backward transfer, registration/update time, evaluation time, overall runtime, and peak RSS memory.

All experiment runners now also log per-face registration/storage details:
- elapsed registration time per face,
- in-memory exemplar/Gaussian byte counters from `FaceRecognitionSystem.register(...)`,
- on-disk exemplar/Gaussian `.npz` footprint and per-face delta,
- summary averages (including average storage growth per new face).

---

## Renew all Pi experiment logs

After the Pi has the dataset, TFLite models, and dependencies installed:

```bash
PYTHONPATH=. python experiments/run_all_pi_experiments.py
```

This runs every experiment with `--reset-workspace --overwrite-embeddings` and writes a top-level refresh log to `experiments/logs/pi_experiment_refresh.log`. To renew only the evaluation logs while reusing cached embeddings:

```bash
PYTHONPATH=. python experiments/run_all_pi_experiments.py --reuse-embeddings
```

---

## Generate multi-trial supertask suite (Set1..Set10)

Create the full supertask suite (3 trials each for set1..set10):

```bash
PYTHONPATH=. python experiments/generate_supertask_suite.py
```

Generated files:
- `data/supertask_suite/set1_trial1.json` (original 8.2 split)
- `data/supertask_suite/set1_trial2.json` .. `set1_trial3.json` (10 faces total)
- `data/supertask_suite/set2_trial1.json` .. `set2_trial3.json` (20 faces total)
- `data/supertask_suite/set3_trial1.json` .. `set3_trial3.json` (30 faces total)
- `data/supertask_suite/set4_trial1.json` .. `set4_trial3.json` (40 faces total)
- `data/supertask_suite/set5_trial1.json` .. `set5_trial3.json` (50 faces total)
- `data/supertask_suite/set6_trial1.json` .. `set6_trial3.json` (60 faces total)
- `data/supertask_suite/set7_trial1.json` .. `set7_trial3.json` (70 faces total)
- `data/supertask_suite/set8_trial1.json` .. `set8_trial3.json` (80 faces total)
- `data/supertask_suite/set9_trial1.json` .. `set9_trial3.json` (90 faces total)
- `data/supertask_suite/set10_trial1.json` .. `set10_trial3.json` (100 faces total)

Compact supertask JSONs in this suite define `tasks` + split constraints, and are expanded automatically at embedding time.

---

## One consolidated log per set (3 trials + average)

Run one experiment method for an entire set and produce one log file that includes, for each of the three trials, the **full** `logs/evaluation.log` content from that run (configuration, per-face registration lines, per-task timing, per-task accuracy table, forgetting, confusion matrix, unknown-identity rejection section, registration/storage summary, peak memory, etc.). After the third trial, a **SET SUMMARY** section lists per-trial headline metrics and the **mean** across trials.

```bash
PYTHONPATH=. python experiments/run_set_trials.py \
  --runner experiments/lwf_classifier/run.py
```

Default behavior runs `set1` (all 3 trials). To run a different set, pass `--set-name setN`.

Output log path (banner lines use the same `%(asctime)s | INFO |` style as `evaluation.log`):
- `experiments/<method>/logs/sets/setN.log` (e.g. `set4.log`, `set10.log`)

Options:
- add `--reuse-embeddings` to skip `--overwrite-embeddings`,
- add `--confidence-threshold <float>` to override the runner threshold (for unknown-ID rejection tuning),
- add `--experiment-root <path>` if you want a custom experiment root.

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

## Exemplar replay classifier (`replay_classifier`)

Replays stored exemplars from all previous identities during incremental training. Defaults: `data/supertask_8_2.json`, `experiments/replay_classifier/embeddings`, `experiments/replay_classifier/workspace`.

```bash
PYTHONPATH=. python experiments/replay_classifier/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--experiment-root PATH` | Experiment root directory (default: `experiments/replay_classifier`). |
| `--reset-workspace` | Delete and recreate the `workspace/` directory before running. |
| `--overwrite-embeddings` | Recompute embeddings even if cached data exists. |
| `--confidence-threshold FLOAT` | Recognition threshold for classifier-based recognition (default: `0.1`). |
| `--epochs INT` | Number of SGD epochs for each incremental update (default: `10`). |
| `--batch-size INT` | SGD mini-batch size for each incremental update (default: `10`). |
| `--max-new-exemplars INT` | Max sampled train embeddings from the newly registered identity (default: `50`). |
| `--exemplar-k INT` | Number of exemplars stored per identity (default: `5`). |

---

## Replay + LwF classifier (`replay_lwf_classifier`)

Hybrid method: trains the student with both exemplar replay from old identities and LwF distillation from a frozen teacher. Replay uses up to 5 stored exemplars per old identity by default.

```bash
PYTHONPATH=. python experiments/replay_lwf_classifier/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--experiment-root PATH` | Experiment root directory (default: `experiments/replay_lwf_classifier`). |
| `--reset-workspace` | Delete and recreate the `workspace/` directory before running. |
| `--overwrite-embeddings` | Recompute embeddings even if cached data exists. |
| `--confidence-threshold FLOAT` | Recognition threshold for classifier-based recognition (default: `0.1`). |
| `--epochs INT` | Number of SGD epochs for each incremental update (default: `10`). |
| `--batch-size INT` | SGD mini-batch size for each incremental update (default: `10`). |
| `--temperature FLOAT` | Distillation temperature (default: `2.0`). |
| `--distill-weight FLOAT` | Weight of the KL distillation term (default: `1.0`). |
| `--replay-per-identity INT` | Max replay exemplars per previous identity during updates (default: `5`). |
| `--max-new-exemplars INT` | Max sampled train embeddings from the newly registered identity (default: `50`). |
| `--exemplar-k INT` | Number of exemplars stored per identity (default: `5`). |

---

## LwF classifier (`lwf_classifier`)

No-replay LwF baseline: trains only on new identity embeddings while distilling old-class outputs from a frozen teacher.

```bash
PYTHONPATH=. python experiments/lwf_classifier/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--experiment-root PATH` | Experiment root directory (default: `experiments/lwf_classifier`). |
| `--reset-workspace` | Delete and recreate the `workspace/` directory before running. |
| `--overwrite-embeddings` | Recompute embeddings even if cached data exists. |
| `--confidence-threshold FLOAT` | Recognition threshold for classifier-based recognition (default: `0.1`). |
| `--epochs INT` | Number of SGD epochs for each incremental update (default: `10`). |
| `--batch-size INT` | SGD mini-batch size for each incremental update (default: `10`). |
| `--temperature FLOAT` | Distillation temperature (default: `2.0`). |
| `--distill-weight FLOAT` | Weight of the KL distillation term (default: `1.0`). |
| `--max-new-exemplars INT` | Max sampled train embeddings from the newly registered identity (default: `50`). |
| `--exemplar-k INT` | Number of exemplars stored per identity (default: `5`). |

---

## Synthetic replay classifier (`synthetic_replay_classifier`)

Synthetic replay generates synthetic samples for old classes from a per-identity Gaussian model. In the current implementation, the Gaussian parameters are fit from the full set of extracted embeddings for each new identity rather than only the selected `exemplar_k` subset.

Important persistence details:
- Gaussian parameters are persisted under `workspace/gaussians/<identity>/gaussian.npz`.
- If the Gaussian or workspace state is not preserved, synthetic replay for previously enrolled users cannot be reconstructed after restarting the system.

Artifacts default to `experiments/synthetic_replay_classifier/embeddings/` and `experiments/synthetic_replay_classifier/workspace/`.

```bash
PYTHONPATH=. python experiments/synthetic_replay_classifier/run.py --reset-workspace
```

| Flag | Description |
|------|-------------|
| `--supertask-json PATH` | Supertask JSON (default: `data/supertask_8_2.json`). |
| `--experiment-root PATH` | Experiment root directory (default: `experiments/synthetic_replay_classifier`). |
| `--reset-workspace` | Delete and recreate the `workspace/` directory before running. |
| `--overwrite-embeddings` | Recompute embeddings even if cached data exists. |
| `--confidence-threshold FLOAT` | Recognition threshold for classifier-based recognition (default: `0.1`). |
| `--epochs INT` | Number of SGD epochs for each incremental update (default: `10`). |
| `--batch-size INT` | SGD mini-batch size for each incremental update (default: `10`). |
| `--synthetic-samples-per-class INT` | Synthetic replay samples generated per old class (default: `5`). |
| `--exemplar-k INT` | Number of exemplars stored per identity for the system interface (default: `5`). In this experiment, Gaussian parameters are fit from the full embedding set, so `exemplar_k` does not limit the Gaussian fit. |

---

## Help

```bash
PYTHONPATH=. python experiments/baseline_classifier/run.py --help
PYTHONPATH=. python experiments/baseline_ncm/run.py --help
PYTHONPATH=. python experiments/replay_classifier/run.py --help
PYTHONPATH=. python experiments/replay_lwf_classifier/run.py --help
PYTHONPATH=. python experiments/lwf_classifier/run.py --help
PYTHONPATH=. python experiments/synthetic_replay_classifier/run.py --help
```
