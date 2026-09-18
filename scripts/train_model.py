#!/usr/bin/env python3
"""Fine-tune YOLOv8 on the prepared Kaggle vehicle dataset."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(os.getenv("DATA_YAML", "/data/dataset/data.yaml"))
BASE_MODEL = os.getenv("BASE_MODEL", "yolov8n.pt")
RUNS_PROJECT = Path(os.getenv("RUNS_PROJECT", "/data/runs/detect"))
RUN_NAME = os.getenv("RUN_NAME", "train")
MODEL_OUTPUT = Path(os.getenv("MODEL_OUTPUT", "/models/best.pt"))
ONNX_OUTPUT = Path(os.getenv("ONNX_OUTPUT", "/models/best.onnx"))


def main() -> int:
    if not DATA_YAML.is_file():
        print(f"Missing {DATA_YAML}. Run prepare-data first.", file=sys.stderr)
        return 1

    epochs = int(os.getenv("TRAIN_EPOCHS", "100"))
    batch = int(os.getenv("TRAIN_BATCH", "32"))
    device = os.getenv("TRAIN_DEVICE", "0")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=640,
        device=device,
        patience=int(os.getenv("TRAIN_PATIENCE", "50")),
        batch=batch,
        optimizer="auto",
        lr0=float(os.getenv("TRAIN_LR0", "0.0001")),
        lrf=float(os.getenv("TRAIN_LRF", "0.1")),
        dropout=float(os.getenv("TRAIN_DROPOUT", "0.1")),
        seed=int(os.getenv("TRAIN_SEED", "0")),
        project=str(RUNS_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_weights = RUNS_PROJECT / RUN_NAME / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"Training finished but {best_weights} not found.", file=sys.stderr)
        return 1

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, MODEL_OUTPUT)
    print(f"Saved best weights to {MODEL_OUTPUT}")

    if os.getenv("EXPORT_ONNX", "true").lower() in ("1", "true", "yes"):
        best_model = YOLO(str(MODEL_OUTPUT))
        best_model.export(format="onnx")
        exported = MODEL_OUTPUT.with_suffix(".onnx")
        if exported.is_file():
            ONNX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            if exported.resolve() != ONNX_OUTPUT.resolve():
                shutil.copy2(exported, ONNX_OUTPUT)
            print(f"Saved ONNX model to {ONNX_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
