## Detection Test Images

This folder stores image fixtures for detector-only tests in `tests/test_detection.py`.

### Structure

- `positive/`: images that contain at least one visible face.
- `negative/`: images that contain no human face.

### Recommended fixture quality

- Use JPG/PNG files.
- Keep image sizes moderate (for example, 640x480 to 1920x1080).
- Include realistic variation (lighting, pose, distance, cluttered backgrounds).
- Avoid tightly pre-aligned face crops; those are not representative for face detection.

### Naming convention

- Positive: `person_01.jpg`, `group_01.png`, ...
- Negative: `scene_01.jpg`, `object_01.png`, ...

### Generated output (integration tests)

When you run integration tests (`pytest tests/test_detection.py -m integration`), results are written under **`artifacts/`** (git-ignored):

| Path | Purpose |
|------|---------|
| `artifacts/visualizations/` | Annotated PNGs: bounding boxes, confidence labels, 6 keypoint dots (eyes highlighted) |
| `artifacts/logs/detection_results.log` | Append-only log: each image is a **separated block** (`===` header) with **`blazeface.performance`** (indented) and **`summary`** with bbox + **6 BlazeFace keypoint coordinates** (px, MediaPipe order). Integration tests do **not** write raw Tasks dumps. |

### Detector latency & memory (Raspberry Pi / profiling)

The detector (`src/detection/blazeface.py`) can log **one INFO line per `detect()` call**:

- `latency_ms` — wall time for the full `detect()` (BGR→RGB, MediaPipe, postprocess)
- `rss_mb` — process resident set size after the call (via `psutil`, or `/proc/self/status` on Linux/RPi if `psutil` is missing)

Enable in code: `BlazeFaceDetector(..., log_performance=True)`  
Or in the shell: `export BLAZEFACE_LOG_PERFORMANCE=1`

Optional raw Tasks output (per internal pass, before clipping):  
`BlazeFaceDetector(..., log_raw_inference=True)` or `export BLAZEFACE_LOG_RAW=1`

To see log lines on the console as well:

```bash
pytest tests/test_detection.py -m integration -o log_cli=true --log-cli-level=INFO
```

### Notes

- Tests are written to be deterministic and CI-friendly.
- Image-based tests are skipped when no fixture images exist.
