# Midterm Presentation (10 minutes)

## Slide 1 - Title

**Project 8: On-Device Continual Face Recognition with Forgetting Prevention**

- Team / course: COMP4901D
- Hardware target: Raspberry Pi 5 (8 GB) + Intel RealSense
- Core message: Register new people incrementally, keep old people recognizable, run fully on-device

[Title slide image: system photo or project logo]

Speaker notes (20-30s):
- This project targets real home-security use where identities are added over time.
- Main challenge is continual learning under edge-device constraints.

---

## Slide 2 - Problem and Motivation

- Home cameras start with a few family members, then new people are added (housekeeper, neighbors, visitors)
- Standard fine-tuning causes **catastrophic forgetting** (old identities become less accurate)
- Need practical on-device solution: no cloud dependency, low latency, low memory usage
- Hardware change impact: from GPU-oriented expectation to Raspberry Pi CPU runtime, stricter efficiency requirements

[Image: scenario diagram of family + newly added identities over time]

Speaker notes (40-50s):
- Emphasize "on-device" means privacy-preserving and offline-capable.
- Explain why this is harder than static face recognition.

---

## Slide 3 - Project Objectives and Success Criteria

- Build complete face recognition + registration pipeline with simple UI
- Meet key targets:
  - Incremental update time: **< 1 minute per new identity**
  - Accuracy on previously learned identities: **> 95%**
  - Device memory budget: **within 8 GB RAM**
  - Support at least **10 identities**
- Evaluate with accuracy, forgetting rate, training time, memory footprint

[Image: table of success criteria]

Speaker notes (35-45s):
- These metrics directly come from the project specification and drive all design choices.

---

## Slide 4 - End-to-End System Pipeline

Pipeline:
1. Capture frame (RealSense/OpenCV)
2. Face detection (BlazeFace)
3. Alignment + crop (eye-based rotation, 112x112)
4. Embedding extraction (MobileFaceNet, 128D)
5. Recognition / registration update (NCM or expandable cosine classifier)

Why each stage matters:
- Detection + alignment improve embedding quality
- Embedding model size affects latency and memory
- Continual-learning strategy determines forgetting and update speed

[Image: pipeline block diagram with arrows]

Speaker notes (50-60s):
- Briefly connect each module to one benchmark metric (latency, accuracy, memory).

---

## Slide 5 - Architecture and Design Pattern

[Image: architecture diagram ]

Speaker notes (45-55s):
- Highlight that camera/detection/alignment are in application layer, while continual-learning core operates on embeddings.

Current Progress

Completed:
- End-to-end framework and module structure
- Detection, alignment, embedding pipeline
- NCM baseline and classifier baseline flow
- Exemplar replay evaluation pipeline and logs

In progress:
- Replay + LwF distillation module
- Synthetic replay memory strategy comparison
- Final integrated UI polish and full on-device benchmark pass

[Image: timeline with green checks on completed milestones]

Speaker notes (40-50s):
- Be explicit: core system works; remaining work is method extension + deeper comparison.
---

## Slide 6 - Interesting Challenges (and Our Design Choices)

- **Detection challenge:** need fast, reliable face boxes on edge CPU  
  - Solution: BlazeFace (full-range), lightweight and edge-friendly
- **Alignment challenge:** landmark-heavy alignment is too expensive for edge deployment  
  - Solution: simplified eye-line rotation using only two eye landmarks (same core idea, much lighter)
- **Embedding challenge:** keep recognition strong but make inference practical on Raspberry Pi  
  - Solution: MobileFaceNet (128D), compact model; keep backbone **frozen** during incremental updates
- **Choosing exemplars:** it can be random, but we also adopt the way from a research paper iCarl in doing herding (where we choose the n embedding cloest to mean as the exmepler for rpelya method)
- **Incremental update challenge:** when adding a new identity, the classifier must expand without breaking old ones  
  - Solution: cosine-linear classifier head, expandable output for new identities

Why this matters to benchmarks:
- **Latency:** small models + frozen backbone keep enrollment/recognition fast
- **Accuracy/forgetting:** alignment + cosine geometry stabilize recognition under few-shot incremental updates
- **Memory:** storing embeddings/exemplars is far cheaper than storing raw images or retraining big backbones

[Image: “challenge -> choice” table (4 rows: detection/alignment/embedding/classifier)]

Speaker notes (55-65s):
- Frame this slide as tradeoffs we hit on Pi (CPU-only) and how each choice protects latency/accuracy/memory.

---

## Slide 7 - Evaluation Protocol

- Dataset setting: VGGFace2 subset, 10 identities for current super-task
- Class-incremental tasks: Task 0 to Task 4 (identities added over time)
- Per-task evaluation on all seen identities
- Metrics collected:
  - Overall and per-class accuracy
  - Forgetting / backward transfer
  - Registration time
  - Memory overhead

[Image: task timeline chart (Task0->Task4, cumulative identities)]

Speaker notes (45-55s):
- Stress that evaluation is incremental, not one-shot.

---

## Slide 8 - Continual Learning Methods Compared

- **Baseline A (recognition-only): NCM**
  - No classifier retraining; compare query embedding with class prototypes
  - Fast and simple, useful as strong baseline
- **Baseline B (implemented): Naive fine-tuning**
  - Train on new identities without replay; expected forgetting
- **Method C (implemented): Exemplar Replay**
  - Store representative old samples and replay during updates
- **Method D (in progress): Replay + LwF distillation**
  - Add teacher-student regularization to further preserve old knowledge

 - preliminary findings 
  - Baseline and replay evaluations have been run (see experiment logs)
- Early observations:
  - Replay improves retention compared with naive updates
  - NCM gives strong practical performance without retraining, but has memory/search tradeoffs as scale grows
- Current bottleneck: balancing update speed and retention on CPU-only deployment

[Screenshot: key result table from `experiments/baseline_classifier/logs/evaluation.log`]
[Screenshot: key result table from `experiments/baseline_ncm/logs/evaluation.log`]

Speaker notes (50-60s):
- Focus on trend, not just absolute number.
- Mention that final values will be consolidated after LwF module is complete.

---

[Image: method comparison matrix]

Speaker notes (60s):
- Clarify that NCM is a practical baseline with different tradeoffs from classifier-based incremental training.

---



## Slide 11 - Challenges and Mitigation

- **CPU-only edge runtime:** optimize with lightweight models, frozen embeddings, small classifier head
- **Forgetting vs speed tradeoff:** replay buffer + herding selection to maximize retention per memory byte
- **Robustness issues:** lighting/pose variation handled via alignment and incremental data collection

[Image: risk -> mitigation table]

Speaker notes (45-55s):
- Tie each risk directly to a concrete implementation decision.

---

## Slide 12 - Next Steps

Challenges to be aware of 
- **Robustness issues:** lighting/pose variation handled via alignment and incremental data collection
- **Memory budget management:** compare raw images vs compressed embeddings vs synthetic replay

Next steps:
- Finish Replay + LwF and synthetic replay experiments
- Complete full benchmark table (accuracy/forgetting/time/memory)




Speaker notes (30-40s):
- End with readiness: architecture is stable, remaining tasks are focused and measurable.

---

## Backup Slide - Suggested Q&A Answers

- Why freeze MobileFaceNet?  
  -> Reduces compute/memory, stabilizes incremental updates on edge device.
- Why cosine classifier?  
  -> Better geometry for normalized embeddings and few-shot incremental classes.
- Why not cloud training?  
  -> Privacy, offline robustness, and practical deployment constraints.

[Optional image: one-slide technical appendix]