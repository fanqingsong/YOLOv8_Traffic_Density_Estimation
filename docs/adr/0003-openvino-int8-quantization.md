# 0003. OpenVINO INT8 quantization and like-for-like comparison

- Status: Accepted
- Date: 2026-09-20
- Deciders: project maintainers
- Extends: ADR 0001 and ADR 0002

## Context

The training stage produces PyTorch weights and an optional ONNX export, but it
does not produce a CPU-quantized model or measure the effect of quantization.
Comparing PyTorch FP32 directly with OpenVINO INT8 would mix two changes: runtime
conversion and numeric precision.

Quantization also needs the prepared dataset for representative calibration and
validation. It must remain independently runnable so that an existing
`models/best.pt` can be quantized without retraining.

## Decision

OpenVINO INT8 post-training quantization is a separate package and one-shot
Compose stage:

- `scripts/model_quantization/`
- `python -m scripts.model_quantization`
- `docker compose run --rm quantize-model`

The stage exports OpenVINO FP32 and INT8 artifacts from the same PyTorch weights.
INT8 calibration uses the prepared YOLO dataset configuration. Both artifacts are
validated with the same OpenVINO runtime, dataset, image size, batch size, and
device.

The comparison report records mAP50-95, mAP50, inference milliseconds per image,
artifact size, accuracy deltas, speedup, and compression ratio. It is written as
JSON for later automation while the CLI prints a short human-readable summary.

Ultralytics, OpenVINO, and NNCF instances remain behind a backend adapter.
Environment access remains in `QuantizationConfig.from_env()`.

## Consequences

- Quantization can be rerun without Kaggle download or model training when the
  prepared dataset and `best.pt` already exist.
- Accuracy and speed deltas isolate precision changes rather than runtime changes.
- INT8 export requires representative calibration data and may be slow on CPU.
- Reported latency is hardware-specific and should only be compared within the
  same environment.
- The CPU inference image includes OpenVINO runtime support so the exported INT8
  model can be selected through `MODEL_PATH`.

## Compliance

- The new feature lives under `scripts/model_quantization/`.
- Each module owns one capability and one primary class.
- Package imports do not eagerly import Ultralytics, OpenVINO, or NNCF.
- Vendor objects are private and public results are project dataclasses.
- Quantization remains a one-shot Compose service and does not run implicitly
  during training.
