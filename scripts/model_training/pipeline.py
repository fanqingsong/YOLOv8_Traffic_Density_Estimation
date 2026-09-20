"""编排训练、最佳权重复制与可选 ONNX 导出。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .backend import YoloTrainingBackend
from .config import TrainConfig

__all__ = ["ModelTrainingPipeline", "TrainingResult"]


@dataclass(frozen=True)
class TrainingResult:
    model_output: Path
    onnx_output: Path | None


class _TrainingBackend(Protocol):
    def train(self, config: TrainConfig) -> Path: ...

    def export_onnx(self, model_path: Path) -> Path: ...


class ModelTrainingPipeline:
    """训练阶段门面；文件产物使用项目路径返回。"""

    def __init__(
        self,
        config: TrainConfig,
        backend: _TrainingBackend | None = None,
    ) -> None:
        self._config = config
        self._backend = backend or YoloTrainingBackend()

    def run(self) -> TrainingResult:
        if not self._config.data_yaml.is_file():
            raise FileNotFoundError(
                f"Missing {self._config.data_yaml}. Run prepare-data first."
            )

        best_weights = self._backend.train(self._config)
        if not best_weights.is_file():
            raise FileNotFoundError(
                f"Training finished but {best_weights} was not found."
            )

        self._copy(best_weights, self._config.model_output)
        onnx_output: Path | None = None
        if self._config.export_onnx:
            exported = self._backend.export_onnx(self._config.model_output)
            if not exported.is_file():
                raise FileNotFoundError(
                    f"ONNX export finished but {exported} was not found."
                )
            self._copy(exported, self._config.onnx_output)
            onnx_output = self._config.onnx_output

        return TrainingResult(self._config.model_output, onnx_output)

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
