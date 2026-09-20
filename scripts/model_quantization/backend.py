"""封装 Ultralytics OpenVINO 导出与验证。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import QuantizationConfig

__all__ = ["OpenVinoQuantizationBackend", "ValidationMetrics"]


@dataclass(frozen=True)
class ValidationMetrics:
    """与底层验证结果解耦的检测指标。"""

    map50_95: float
    map50: float
    inference_ms: float


class _BoxMetrics(Protocol):
    map: float
    map50: float


class _ValidationResult(Protocol):
    box: _BoxMetrics
    speed: Mapping[str, float]


class _Model(Protocol):
    def export(self, **kwargs: Any) -> str | Path: ...

    def val(self, **kwargs: Any) -> _ValidationResult: ...


class OpenVinoQuantizationBackend:
    """Ultralytics 的窄适配层，可用工厂替身进行离线测试。"""

    def __init__(
        self,
        model_factory: Callable[[str], _Model] | None = None,
    ) -> None:
        self._model_factory = model_factory or self._default_factory

    def export_fp32(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path:
        model = self._model_factory(str(model_path))
        exported = model.export(
            format="openvino",
            quantize=32,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
        )
        return Path(exported)

    def export_int8(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path:
        model = self._model_factory(str(model_path))
        exported = model.export(
            format="openvino",
            quantize=8,
            data=str(config.data_yaml),
            fraction=config.calibration_fraction,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
        )
        return Path(exported)

    def validate(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> ValidationMetrics:
        model = self._model_factory(str(model_path))
        metrics = model.val(
            data=str(config.data_yaml),
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            plots=False,
            verbose=False,
        )
        return ValidationMetrics(
            map50_95=float(metrics.box.map),
            map50=float(metrics.box.map50),
            inference_ms=float(metrics.speed["inference"]),
        )

    @staticmethod
    def _default_factory(model_path: str) -> _Model:
        from ultralytics import YOLO

        return YOLO(model_path)
