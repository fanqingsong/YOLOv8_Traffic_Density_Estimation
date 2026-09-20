"""模型训练阶段入口。"""

from __future__ import annotations

import sys

from .config import TrainConfig
from .pipeline import ModelTrainingPipeline


def run() -> int:
    try:
        result = ModelTrainingPipeline(TrainConfig.from_env()).run()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Saved best weights to {result.model_output}")
    if result.onnx_output is not None:
        print(f"Saved ONNX model to {result.onnx_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
