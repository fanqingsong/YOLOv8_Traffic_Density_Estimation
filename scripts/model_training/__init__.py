"""模型训练包：config → backend → pipeline → ``python -m scripts.model_training``。"""

from __future__ import annotations

from typing import Any

__all__ = ["ModelTrainingPipeline", "TrainConfig", "TrainingResult", "run"]


def __getattr__(name: str) -> Any:
    if name == "TrainConfig":
        from .config import TrainConfig

        return TrainConfig
    if name in {"ModelTrainingPipeline", "TrainingResult"}:
        from .pipeline import ModelTrainingPipeline, TrainingResult

        return {
            "ModelTrainingPipeline": ModelTrainingPipeline,
            "TrainingResult": TrainingResult,
        }[name]
    if name == "run":
        from .__main__ import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
