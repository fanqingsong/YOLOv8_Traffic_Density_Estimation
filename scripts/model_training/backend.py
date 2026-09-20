"""封装 Ultralytics 训练与导出，不向调用方暴露 YOLO 对象。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .config import TrainConfig

__all__ = ["YoloTrainingBackend"]


class _Model(Protocol):
    def train(self, **kwargs: Any) -> object: ...

    def export(self, **kwargs: Any) -> str | Path: ...


class YoloTrainingBackend:
    """Ultralytics 的窄适配层，可用工厂替身进行无 GPU 测试。"""

    def __init__(
        self,
        model_factory: Callable[[str], _Model] | None = None,
    ) -> None:
        self._model_factory = model_factory or self._default_factory

    def train(self, config: TrainConfig) -> Path:
        model = self._model_factory(config.base_model)
        model.train(
            data=str(config.data_yaml),
            epochs=config.epochs,
            imgsz=640,
            device=config.device,
            patience=config.patience,
            batch=config.batch,
            optimizer="auto",
            lr0=config.lr0,
            lrf=config.lrf,
            dropout=config.dropout,
            seed=config.seed,
            project=str(config.runs_project),
            name=config.run_name,
            exist_ok=True,
        )
        return config.best_weights

    def export_onnx(self, model_path: Path) -> Path:
        model = self._model_factory(str(model_path))
        exported = model.export(format="onnx")
        return Path(exported)

    @staticmethod
    def _default_factory(model_path: str) -> _Model:
        from ultralytics import YOLO

        return YOLO(model_path)
