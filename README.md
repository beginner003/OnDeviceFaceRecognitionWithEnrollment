# OnDeviceFaceRecognitionWithEnrollment

On-device continual face recognition with forgetting prevention. The project provides an incremental registration + recognition pipeline, plus evaluation code and unit/integration tests.

## Read before you start
Please read `IMPLEMENTATION_PLAN.md` (module breakdown, architecture decisions, and how the pieces fit together).

## What’s included
- Pre-trained TFLite models in `src/models/`
- Dataset/enrollment fixtures in `data/`

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
