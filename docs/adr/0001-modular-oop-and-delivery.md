# 0001. Modular packages, OOP encapsulation, and Compose delivery

- Status: Accepted
- Date: 2026-09-20
- Deciders: project maintainers
- Reference implementation: `scripts/traffic_analysis/`

## Context

This repository is a Dockerized, learnable fork of a YOLOv8 traffic-density demo. Early code lived in a single script with public globals. Later work split steps into modules, then grouped each step into a cohesive class.

Without a written decision, later changes (including generated code) tend to:

- add a new `.py` at the repo root
- expose YOLO / OpenCV handles and mutable arrays
- mix training and inference in one `docker compose up`
- let README name files that are not in the tree

## Decision

New code MUST follow the practices below. Changing them requires a new ADR that supersedes this one.

### 1. Package layout

- Application Python lives under `scripts/`, not the repository root.
- One **capability** per module; one **primary class** per module (small immutable result types such as `LaneSnapshot` may share the module).
- Packages expose a documented learning / call order in `__init__.py`.
- `__init__.py` MUST NOT import heavy optional stacks (Ultralytics, OpenCV) at module import time. Use lazy `__getattr__` for `run` / facade types.
- Public names go in `__all__`. Entry point: `python -m scripts.<package>` from the repo root. A thin launcher script is optional.

### 2. Object encapsulation

- The class owns both state and behavior for that capability.
- **Public methods** (no leading underscore) are the only supported operations for callers outside the module.
- **Internal methods** (`_name`) implement those operations; orchestrators (e.g. `TrafficAnalyzer`) MUST NOT call another type’s `_` methods.
- **Private attributes** (`_name`) store state. Do not read or write them from outside the class.
- **Public state** is exposed with read-only `@property`. Mutable arrays returned to callers MUST be copies (see `LaneGeometry.left_polygon`).
- Do not leak third-party instances (`YOLO`, `VideoCapture`) as public attributes. Wrap them; return project types (`DetectionOutcome`, `LaneSnapshot`).
- Prefer `classmethod` factories (`AnalysisConfig.from_env()`) over scattering `os.getenv` in every class.
- Domain types MUST NOT import vendor engines when a `Protocol` or duck typing suffices (`LaneCounter` vs Ultralytics `Boxes`).

### 3. Comments for learners

- Module docstrings may include ASCII diagrams **tied to real constants** (coordinates, env vars, class/method names).
- Do not comment every line. Explain non-obvious geometry, BGR vs RGB, and “display-only vs used for counting”.

### 4. Docker Compose delivery

- Use `docker compose` (plugin), never `docker-compose`.
- Each Compose service is a **one-shot stage**. Callers use `docker compose run --rm <service>`.
- Do **not** document or rely on bare `docker compose up`: it would start download, prepare, train, and inference together.
- GPU training is an override file (`compose.gpu.yaml`), not the default `train` service.
- Inference image copies `scripts/` and runs `python -m scripts.traffic_analysis`. Base images MAY use the Huawei mirror prefix `swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/`.
- Paths that differ between host and container (e.g. `data/sample_video.mp4` vs bind-mounted `./sample_video.mp4`) MUST be stated in README.

### 5. Documentation honesty

- README file lists and commands MUST match the tree (license filename, missing notebook/weights/video, actual entry points).
- After moving or renaming code, update README in the same change.

### 6. Existing procedural CLIs

`scripts/download_kaggle_dataset.py`, `prepare_dataset.py`, and `train_model.py` predate this ADR. New features in those files SHOULD move toward the same OOP rules. Do not rewrite them in an unrelated change.

## Consequences

- Inference is testable per class (`LaneCounter.tally` without GPU).
- Callers depend on a narrow public surface; internals can change behind properties.
- Agents and humans have a single checklist (below) plus Cursor rules that point here.

## Compliance checklist (new Python)

- [ ] File is under `scripts/` in a named package or an existing stage script
- [ ] One primary class; public methods + `@property`; state on `self._…`
- [ ] No public third-party engine objects
- [ ] No new root-level `*.py` for features
- [ ] Compose stages remain independently `run --rm`
- [ ] README updated if paths, commands, or files changed

## Example

```python
# BAD: procedural, public YOLO handle, other modules poke internals
model = YOLO("models/best.pt")
frame[:325, :] = 0
print(detector._model)

# GOOD: cohesive type, public infer(), private engine
class VehicleDetector:
    def __init__(self, inference: InferenceConfig) -> None:
        self._model = YOLO(inference.model_path)

    def infer(self, detection_frame: np.ndarray) -> DetectionOutcome:
        result = self._predict(detection_frame)
        return DetectionOutcome(self._draw_boxes(result), result.boxes)

    def _predict(self, detection_frame: np.ndarray):
        return self._model.predict(detection_frame, imgsz=self._imgsz, conf=self._conf)[0]
```
