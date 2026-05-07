# Final Presentation — Evaluation & Results

**Project:** On-device continual face recognition (COMP4901D)  
**Hardware:** Raspberry Pi 5; embeddings from frozen MobileFaceNet (128D), BlazeFace + alignment upstream.

---

## Evaluation methodology (updated since midterm)

The midterm protocol described a **single** VGGFace2-style class-incremental schedule (Task 0–4, 10 identities) with accuracy, forgetting, registration time, and memory. The **final** suite extends this in three ways:

1. **Pre-extracted embedding cache** — All methods share the same offline pipeline (`embedding_helper.py`: detect → align → MobileFaceNet TFLite). Runs differ only in how the recognition head is trained or queried (classifier fine-tuning, NCM, replay, LwF, synthetic replay), so comparisons isolate continual-learning strategy rather than stochastic embedding noise.

2. **Multi-trial supertasks** — Nine JSON schedules (`data/supertask_suite/`: set1/set2/set3 × trial1–3) vary which identities appear in which incremental task. Each **set log** aggregates **three independent trials** and reports **means** (see `experiments/run_set_trials.py`), reducing sensitivity to one lucky split.

3. **Three difficulty scales (sets)**  
   - **Set 1** — Compact schedule (same family of problems as the classic 8+2 split; 10 identities total over tasks).  
   - **Set 2** — Larger incremental curriculum: **5 initial identities**, grow to **20** (more tasks, more forgetting opportunities).  
   - **Set 3** — **5 initial identities**, grow to **30** (hardest scale in this suite).

**Logged headline metrics** (from `experiments/README.md` and each run’s `evaluation.log` / set rollup):

| Metric | Meaning |
|--------|---------|
| **Average accuracy** | Mean accuracy over tasks and classes in the supertask (as in logs: 1.0 = 100%). |
| **Average forgetting** | Mean drop on previously learned classes after later tasks (higher = worse retention). |
| **Backward transfer** | Aggregate backward transfer (negative values indicate harm to old knowledge as training proceeds). |
| **Overall run time (s)** | Mean per-trial **process** total logged by the runner (sum of task work inside the experiment). |
| **Wall clock (s)** | Mean per-trial **subprocess** wall time (includes startup/OS; slightly larger than process time). |

Logs also record **per-face registration** latency, **on-disk exemplar/Gaussian deltas** (`Storage summary`), and **overall peak RSS** after each full trial (`Overall peak memory`). Tables 4–6 summarize those fields (means of 3 trials per set). Other log detail (per-task confusion matrices, per-task peak lines) is omitted here for space.

**Run date in logs:** 2026-05-05 (Pi refresh consolidated in `experiments/logs/pi_experiment_refresh.log`).

---

## Methods compared

| Label | Description |
|-------|-------------|
| **Naive classifier** | `baseline_classifier` — expandable cosine classifier; train on new identity only (herding exemplars stored; no replay). |
| **NCM** | `baseline_ncm` — nearest class mean on stored exemplars; no classifier fine-tuning. |
| **LwF** | `lwf_classifier` — distillation from frozen teacher; **no** exemplar replay. |
| **Replay** | `replay_classifier` — SGD with replay of stored exemplars from past identities. |
| **Replay + LwF** | `replay_lwf_classifier` — replay **plus** LwF distillation. |
| **Synthetic replay** | `synthetic_replay_classifier` — old classes replayed via per-identity Gaussian samples (fits from extracted embeddings; persists `gaussian.npz`). |

---

## Table 1 — Continual learning metrics by method and set (mean of 3 trials)

Values are **means across trials** from each `experiments/<method>/logs/sets/set{1,2,3}.log` **SET SUMMARY** block. Accuracy and forgetting shown as **percent** for readability.

### Set 1 (10-identity-style compact schedule)

| Method | Avg accuracy ↑ | Avg forgetting ↓ | Backward transfer |
|--------|----------------|------------------|-------------------|
| Naive classifier | 10.7% | 40.7% | −0.508 |
| NCM | **95.0%** | **0.7%** | −0.008 |
| LwF | 19.7% | 46.0% | −0.575 |
| Replay | 92.7% | 5.7% | −0.054 |
| Replay + LwF | 88.7% | 6.0% | −0.008 |
| Synthetic replay | **97.0%** | **2.0%** | −0.025 |

### Set 2 (5 → 20 identities)

| Method | Avg accuracy ↑ | Avg forgetting ↓ | Backward transfer |
|--------|----------------|------------------|-------------------|
| Naive classifier | 8.7% | 71.8% | −0.756 |
| NCM | 94.2% | 2.2% | −0.023 |
| LwF | 13.5% | 68.0% | −0.716 |
| Replay | **97.3%** | **2.5%** | −0.026 |
| Replay + LwF | 96.0% | 3.5% | −0.035 |
| Synthetic replay | **98.3%** | **1.0%** | −0.011 |

### Set 3 (5 → 30 identities)

| Method | Avg accuracy ↑ | Avg forgetting ↓ | Backward transfer |
|--------|----------------|------------------|-------------------|
| Naive classifier | 6.2% | 80.4% | −0.832 |
| NCM | 92.1% | 2.9% | −0.030 |
| LwF | 9.7% | 77.2% | −0.799 |
| Replay | 94.9% | 4.8% | −0.049 |
| Replay + LwF | 93.9% | 5.6% | −0.058 |
| Synthetic replay | **97.4%** | **1.2%** | −0.013 |

---

## Table 2 — Runtime vs scale (mean of 3 trials)

**Overall run time** = mean logged **process** seconds per trial (`overall_run_time_s` in SET SUMMARY). **Wall** = mean **subprocess wall** seconds per trial.

| Method | Set 1<br>proc (s) / wall (s) | Set 2<br>proc (s) / wall (s) | Set 3<br>proc (s) / wall (s) |
|--------|------------------------------|------------------------------|------------------------------|
| Naive classifier | 15.7 / 18.8 | 31.9 / 35.0 | 50.9 / 53.9 |
| NCM | 16.2 / 18.7 | 31.9 / 34.4 | 48.6 / 51.1 |
| LwF | 16.0 / 19.0 | 33.2 / 36.2 | 53.6 / 56.7 |
| Replay | 16.3 / 19.3 | 33.6 / 36.7 | 52.8 / 55.9 |
| Replay + LwF | 16.9 / 20.0 | 34.5 / 37.6 | 54.7 / 57.8 |
| Synthetic replay | 35.2 / 38.4 | 68.2 / 71.4 | **140.5** / **143.5** |

Synthetic replay trades **much longer** registration/update (Gaussian fit + sampling + training) for very **low forgetting** at large scale; NCM and replay-family methods stay fast on Pi for this benchmark.

---

## Table 3 — Macro average across all three sets (unweighted mean of set means)

Useful for a single ranking slide; treats set1–set3 equally (different class counts per set).

| Method | Macro avg<br>accuracy | Macro avg<br>forgetting | Macro avg<br>backward transfer | Macro avg<br>proc time (s) |
|--------|----------------------|-------------------------|-------------------------------|---------------------------|
| Naive classifier | 8.5% | 64.3% | −0.699 | 32.9 |
| NCM | 93.8% | 2.6% | −0.021 | 32.2 |
| LwF | 14.3% | 63.7% | −0.697 | 34.3 |
| Replay | 95.0% | 4.3% | −0.043 | 34.3 |
| Replay + LwF | 92.9% | 4.7% | −0.034 | 35.4 |
| Synthetic replay | **97.6%** | **1.4%** | −0.016 | **81.3** |

---

## Table 3B — Set 4 / Set 5 extension (latest consolidated logs)

These values are from each file's `SET SUMMARY` (mean of 3 trials) and are included as an extension beyond the original set1–set3 comparison.

| Set | Method | Avg accuracy ↑ | Avg forgetting ↓ | Backward transfer | Mean run time / trial (s) |
|-----|--------|----------------|------------------|-------------------|---------------------------|
| Set 4 | Naive classifier | 5.25% | 84.75% | −0.869 | 162.220 |
| Set 4 | NCM | 94.33% | 2.00% | −0.021 | 141.044 |
| Set 4 | LwF | 6.75% | 84.33% | −0.865 | 151.499 |
| Set 4 | Replay | 94.58% | 5.08% | −0.052 | 186.111 |
| Set 4 | Replay + LwF | 92.42% | 6.92% | −0.071 | 211.817 |
| Set 4 | Synthetic replay | **97.50%** | **1.17%** | −0.012 | 334.842 |
| Set 5 | Naive classifier | 3.40% | 88.60% | −0.904 | 224.605 |
| Set 5 | NCM | 92.00% | 2.47% | −0.025 | 181.416 |
| Set 5 | LwF | 5.53% | 87.73% | −0.895 | 203.826 |
| Set 5 | Replay | 92.67% | 6.73% | −0.069 | 259.474 |
| Set 5 | Replay + LwF | 89.93% | 9.20% | −0.094 | 293.393 |
| Set 5 | Synthetic replay | **96.27%** | **1.93%** | −0.020 | 509.907 |

### Table 3C — Unknown identity (Set 4 / Set 5, separate)

Means of the three per-trial totals (`unknown_rejection_accuracy` and `unknown_false_accept_rate`) from each consolidated set log.

| Set | Method | Unknown rejection accuracy ↑ | Unknown false accept rate ↓ |
|-----|--------|------------------------------|-----------------------------|
| Set 4 | Naive classifier | 0.67% | 99.33% |
| Set 4 | NCM | 0.33% | 99.67% |
| Set 4 | LwF | 0.33% | 99.67% |
| Set 4 | Replay | **46.00%** | **54.00%** |
| Set 4 | Replay + LwF | 44.67% | 55.33% |
| Set 4 | Synthetic replay | 41.33% | 58.67% |
| Set 5 | Naive classifier | 1.00% | 99.00% |
| Set 5 | NCM | 1.00% | 99.00% |
| Set 5 | LwF | 0.00% | 100.00% |
| Set 5 | Replay | **38.33%** | **61.67%** |
| Set 5 | Replay + LwF | 37.00% | 63.00% |
| Set 5 | Synthetic replay | 36.33% | 63.67% |

Interpretation (extension):
- Set 4/5 confirm the no-replay methods (Naive/LwF) collapse on known-ID retention and near-zero unknown-ID rejection.
- Synthetic replay keeps top known-ID accuracy at larger scale, but unknown-ID rejection remains below replay-family levels in these runs.
- Replay and Replay+LwF give the strongest unknown-ID rejection on Set 4/5 while staying high on known-ID accuracy.

---

## Table 4 — Average registration time per new face (seconds)

Mean **avg_elapsed** from `Registration summary | … avg_elapsed=… s/face` in each trial, then averaged across the **3 trials** per set. Includes classifier training / replay / distillation time for that face where applicable (not embedding extraction, which is cached separately).

| Method | Set 1 | Set 2 | Set 3 |
|--------|-------|-------|-------|
| Naive classifier | 0.167 | 0.126 | 0.135 |
| NCM | **0.122** | **0.089** | **0.076** |
| LwF | 0.190 | 0.178 | 0.151 |
| Replay | 0.204 | 0.197 | 0.201 |
| Replay + LwF | 0.257 | 0.262 | 0.279 |
| Synthetic replay | 1.926 | 1.916 | **2.923** |

**Macro mean (s/face, unweighted over sets):** Naive 0.143; NCM **0.096**; LwF 0.173; Replay 0.201; Replay + LwF 0.266; Synthetic replay **2.255**.

---

## Table 5 — Peak resident set size (RSS) per run

**Overall peak memory** (MB) from each trial’s `Overall peak memory: … MB` line, averaged over **3 trials** per set. Dominated by loaded MobileFaceNet TFLite + runtime; small method-to-method deltas reflect training tensors and bookkeeping.

| Method | Set 1 | Set 2 | Set 3 |
|--------|-------|-------|-------|
| Naive classifier | 733.9 | 734.8 | 737.9 |
| NCM | **733.4** | 734.8 | 736.7 |
| LwF | 733.6 | 736.7 | 738.1 |
| Replay | **733.5** | 736.0 | 737.7 |
| Replay + LwF | 734.0 | 736.6 | 737.5 |
| Synthetic replay | 735.6 | 739.1 | **742.9** |

**Macro mean (MB):** Naive 735.5; NCM 734.9; LwF 736.1; Replay 735.7; Replay + LwF 736.0; Synthetic replay **739.2**.

---

## Table 6 — On-disk storage growth per enrolled face (bytes)

Means of the averages in **`Storage summary … avg … B/face`** over **3 trials** per set. Exemplar methods persist herded embedding exemplars (~2.5 KB per new face on disk). **Synthetic replay** stores Gaussian stats per identity on disk instead of exemplar `.npz` rows for those classes (exemplar delta 0; Gaussian delta dominates).

### Set 1 (10 faces cumulative)

| Method | Exemplar B/face | Gaussian B/face | Total B/face |
|--------|-----------------|-----------------|--------------|
| Naive classifier | 2508.3 | 0.0 | 2508.3 |
| NCM | 2508.3 | 0.0 | 2508.3 |
| LwF | 2508.3 | 0.0 | 2508.3 |
| Replay | 2508.3 | 0.0 | 2508.3 |
| Replay + LwF | 2508.3 | 0.0 | 2508.3 |
| Synthetic replay | 0.0 | **54003.4** | **54003.4** |

### Set 2 (20 faces cumulative)

| Method | Exemplar B/face | Gaussian B/face | Total B/face |
|--------|-----------------|-----------------|--------------|
| Naive classifier | **2507.0** | 0.0 | **2507.0** |
| NCM | **2507.0** | 0.0 | **2507.0** |
| LwF | **2507.0** | 0.0 | **2507.0** |
| Replay | **2507.0** | 0.0 | **2507.0** |
| Replay + LwF | **2507.0** | 0.0 | **2507.0** |
| Synthetic replay | 0.0 | **53996.2** | **53996.2** |

### Set 3 (30 faces cumulative)

| Method | Exemplar B/face | Gaussian B/face | Total B/face |
|--------|-----------------|-----------------|--------------|
| Naive classifier | **2507.8** | 0.0 | **2507.8** |
| NCM | **2507.8** | 0.0 | **2507.8** |
| LwF | **2507.8** | 0.0 | **2507.8** |
| Replay | **2507.8** | 0.0 | **2507.8** |
| Replay + LwF | **2507.8** | 0.0 | **2507.8** |
| Synthetic replay | 0.0 | **54006.4** | **54006.4** |

**Approximate KiB/face (total disk):** exemplar-based methods ≈ **2.45 KiB**; synthetic replay ≈ **52.7 KiB** on average (Gaussian footprint only in these runs).

---

## Brief interpretation (for speaking notes)

- **Naive classifier and LwF-without-replay** collapse on accuracy at set 2–3 and show **large** forgetting — consistent with catastrophic forgetting when the cosine head is updated only on new classes.
- **NCM** remains a **strong, fast** non-parametric baseline with **low** forgetting in this embedding space; mild drops appear as the number of identities grows (set 3).
- **Replay** matches or beats NCM on **accuracy** in several cells while keeping runtime in the same ballpark as other classifier methods (much faster than synthetic replay).
- **Replay + LwF** is **close to replay alone** here; distillation does not dominate the headline averages in these logs.
- **Synthetic replay** achieves the **best** macro accuracy/forgetting tradeoff in this suite but at **~2–8×** the process time of replay on larger sets (Gaussian updates dominate).
- **Per-face registration latency:** **NCM** is fastest (~0.08–0.12 s/face in these logs) because it skips classifier SGD; **Replay + LwF** is slower than replay alone; **synthetic replay** is ~10–20× slower per face than NCM when Gaussian fitting and long training steps run.
- **Peak RSS** stays in a narrow band (~733–743 MB) across methods; synthetic replay is slightly higher at large set size.
- **Disk per new face:** exemplar-based configs agree at ~**2.5 KB/face**; synthetic replay trades that for ~**53 KB/face** of Gaussian parameters in this implementation.

---

## Source artifacts

- Per-method consolidated logs: `experiments/<method>/logs/sets/set1.log` … `set3.log` (for Tables 4–6: `Registration summary`, `Storage summary`, `Overall peak memory` lines in each embedded trial, **mean of 3 trials** per set)  
- Suite description: `experiments/README.md`  
- Midterm-style problem framing (objectives, pipeline): `midterm_present.md`
