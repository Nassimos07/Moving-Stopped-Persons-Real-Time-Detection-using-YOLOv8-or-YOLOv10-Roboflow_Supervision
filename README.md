# Moving vs Stopped Persons Detection

A cleaned, package-based version of the original notebook project for detecting whether tracked people are moving or stopped in video streams using YOLO and Supervision.

## What changed
- notebook logic moved into `src/movement_detection/`
- config-driven execution via `configs/default.yaml`
- CLI entrypoint for repeatable runs
- assets separated from source code
- tests and dependency files added

## Project structure
```text
.
├── assets/
│   ├── images/
│   └── videos/
├── configs/
│   └── default.yaml
├── notebooks/
│   └── Moving_Stopped_Persons_detection.ipynb
├── scripts/
│   └── run_demo.sh
├── src/
│   └── movement_detection/
├── tests/
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Run
```bash
movement-detection run --config configs/default.yaml --mode relative --filter_mode all
```

Modes:
- `relative`
- `absolute`

Filters:
- `all`
- `moving`
- `stopped`

## Notes
- default config uses `assets/videos/test1.mp4`
- default model path is `yolov8x.pt`
- outputs are written to `outputs/`

## Production risks
- `yolov8x.pt` is heavy, slow, and expensive for real-time use on weak GPUs or CPUs
- current pipeline is inference-only, not a full training pipeline
- movement state is threshold-based, so scene calibration still matters

## Next MLOps steps
- add experiment/config versioning
- add benchmark scripts and metrics logging
- add Docker support
- add CI for lint and tests
