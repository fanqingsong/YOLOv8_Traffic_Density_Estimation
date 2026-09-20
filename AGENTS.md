# Agent instructions

This is a Dockerized, learnable fork of a YOLOv8 traffic-density demo. Binding practices: [docs/adr/0001-modular-oop-and-delivery.md](docs/adr/0001-modular-oop-and-delivery.md). Cursor also has `.cursor/rules/`. Do not silently weaken ADR 0001; supersede it with a new ADR instead.

Respond to the human in **Chinese** in the chat. Keep new code comments in the same language as the surrounding module (inference package docstrings are Chinese).

## Layout

| Path | Role |
|------|------|
| `scripts/traffic_analysis/` | OOP inference package (one class per module) |
| `scripts/kaggle_download/` | OOP Kaggle download stage |
| `scripts/dataset_preparation/` | OOP data.yaml normalization + sample video stage |
| `scripts/model_training/` | OOP YOLO training + artifact export stage |
| `scripts/model_quantization/` | OOP OpenVINO FP32/INT8 export + comparison stage |
| `scripts/real_time_traffic_analysis.py` | Thin launcher for the package |
| `scripts/download_kaggle_dataset.py` | Compatibility launcher for `kaggle_download` |
| `scripts/prepare_dataset.py` | Compatibility launcher for `dataset_preparation` |
| `scripts/train_model.py` | Compatibility launcher for `model_training` |
| `tests/` | Offline unit tests for functional stages |
| `compose.yaml` | One-shot services: download, prepare, train, quantize, inference |
| `compose.gpu.yaml` | NVIDIA override for `train` only |
| `docs/adr/` | Architecture Decision Records |

Application Python lives under `scripts/`, never new feature files at the repo root. Weights, `sample_video.mp4`, and `data/` are runtime artifacts (often gitignored). Functional stages use separate packages documented by ADR 0002 and ADR 0003; keep compatibility scripts thin.

**Inference learning order** (also in `scripts/traffic_analysis/__init__.py`): `config` → `video_io` → `preprocessor` → `detector` → `counter` → `visualizer` → `pipeline` → `python -m scripts.traffic_analysis`.

## Python (new code)

- One capability per module; one primary class per module. Small immutable results (`LaneSnapshot`, `DetectionOutcome`) may share the module.
- Public API: methods without `_` plus read-only `@property`. Callers (including `TrafficAnalyzer`) must not use another type’s `_` methods or attributes.
- Properties that return mutable arrays must return **copies**.
- Do not expose vendor objects (`YOLO`, `VideoCapture`) as public attributes. Return project types.
- `AnalysisConfig.from_env()` (or similar) owns environment variables. Do not scatter `os.getenv` in collaborators.
- Package `__init__.py` must not import OpenCV/Ultralytics at import time. Keep lazy `__getattr__` for `TrafficAnalyzer` / `run`.
- Domain modules (counting, geometry) should use a `Protocol` or duck typing instead of importing Ultralytics.
- Add download, preparation, and training behavior to their packages, not to the compatibility launchers.

```python
# ❌
best_model.predict(frame)
print(detector._model)

# ✅
outcome = detector.infer(masked_frame)
snapshot = counter.tally(outcome.boxes)
```

Lane counting is **not** point-in-polygon. Intensity uses box `xyxy` left `x` vs `lane_threshold` (609 on the sample 1280×720 video). Green/blue polygons are display-only. Detection band `y ∈ [band_top, band_bottom)` is blacked out outside that range before YOLO.

## Docker Compose

- Use `docker compose`, never `docker-compose`.
- Every service is one-shot: `docker compose run --rm <service>`. Do not treat bare `docker compose up` as the app entry (it would start download, prepare, train, and inference together).
- Default `train` is CPU (`TRAIN_DEVICE=cpu`). GPU: `docker compose -f compose.yaml -f compose.gpu.yaml run --rm train`.
- Quantization is CPU OpenVINO INT8: `docker compose run --rm quantize-model`. Compare FP32 and INT8 under the same runtime.
- Inference image CMD: `python -m scripts.traffic_analysis`. Compose inference is headless (`DISPLAY_VIDEO=false`) and bind-mounts **repo-root** `./sample_video.mp4` (not `data/sample_video.mp4`). After prepare, copy `data/sample_video.mp4` to the repo root when needed.
- Select inference artifacts with `MODEL_VARIANT=pytorch|openvino-fp32|openvino-int8`; an explicit `MODEL_PATH` overrides the variant.
- Base images MAY use the Huawei mirror prefix `swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/`.

## Commands

```bash
docker compose run --rm download-data
docker compose run --rm prepare-data
docker compose run --rm train                                          # CPU
docker compose -f compose.yaml -f compose.gpu.yaml run --rm train      # GPU
docker compose run --rm --build quantize-model                         # OpenVINO FP32 vs INT8
docker compose run --rm --build traffic-analysis                       # needs models/best.pt + ./sample_video.mp4

python -m scripts.traffic_analysis                                     # local; DISPLAY_VIDEO=true by default
DISPLAY_VIDEO=false python -m scripts.traffic_analysis
```

Do not run long GPU training or Kaggle downloads unless the user asked. Prefer unit-style checks on public methods (`LaneCounter.tally`) without GPU.

## Docs

README file names, entry points, and host vs container paths must match the tree. Update README in the same change when you move or rename code. New architectural decisions go in `docs/adr/` (see `docs/adr/README.md` template) and a row in that index.
