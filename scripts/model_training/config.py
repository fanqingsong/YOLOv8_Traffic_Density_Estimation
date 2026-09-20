"""模型训练阶段的集中式环境配置。"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["TrainConfig"]


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


class TrainConfig:
    """训练参数与产物路径；所有值均为只读属性。"""

    def __init__(
        self,
        data_yaml: Path,
        base_model: str,
        runs_project: Path,
        run_name: str,
        model_output: Path,
        onnx_output: Path,
        epochs: int,
        batch: int,
        device: str,
        patience: int,
        lr0: float,
        lrf: float,
        dropout: float,
        seed: int,
        export_onnx: bool,
    ) -> None:
        self._data_yaml = data_yaml
        self._base_model = base_model
        self._runs_project = runs_project
        self._run_name = run_name
        self._model_output = model_output
        self._onnx_output = onnx_output
        self._epochs = epochs
        self._batch = batch
        self._device = device
        self._patience = patience
        self._lr0 = lr0
        self._lrf = lrf
        self._dropout = dropout
        self._seed = seed
        self._export_onnx = export_onnx

    @classmethod
    def from_env(cls) -> TrainConfig:
        return cls(
            data_yaml=Path(os.getenv("DATA_YAML", "/data/dataset/data.yaml")),
            base_model=os.getenv("BASE_MODEL", "yolov8n.pt"),
            runs_project=Path(os.getenv("RUNS_PROJECT", "/data/runs/detect")),
            run_name=os.getenv("RUN_NAME", "train"),
            model_output=Path(os.getenv("MODEL_OUTPUT", "/models/best.pt")),
            onnx_output=Path(os.getenv("ONNX_OUTPUT", "/models/best.onnx")),
            epochs=int(os.getenv("TRAIN_EPOCHS", "100")),
            batch=int(os.getenv("TRAIN_BATCH", "32")),
            device=os.getenv("TRAIN_DEVICE", "cpu"),
            patience=int(os.getenv("TRAIN_PATIENCE", "50")),
            lr0=float(os.getenv("TRAIN_LR0", "0.0001")),
            lrf=float(os.getenv("TRAIN_LRF", "0.1")),
            dropout=float(os.getenv("TRAIN_DROPOUT", "0.1")),
            seed=int(os.getenv("TRAIN_SEED", "0")),
            export_onnx=_env_flag("EXPORT_ONNX", "true"),
        )

    @property
    def data_yaml(self) -> Path:
        return self._data_yaml

    @property
    def base_model(self) -> str:
        return self._base_model

    @property
    def runs_project(self) -> Path:
        return self._runs_project

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def model_output(self) -> Path:
        return self._model_output

    @property
    def onnx_output(self) -> Path:
        return self._onnx_output

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def batch(self) -> int:
        return self._batch

    @property
    def device(self) -> str:
        return self._device

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def lr0(self) -> float:
        return self._lr0

    @property
    def lrf(self) -> float:
        return self._lrf

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def export_onnx(self) -> bool:
        return self._export_onnx

    @property
    def best_weights(self) -> Path:
        return self._runs_project / self._run_name / "weights" / "best.pt"
