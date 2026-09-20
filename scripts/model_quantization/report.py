"""量化前后指标计算与 JSON 报告。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backend import ValidationMetrics
from .config import QuantizationConfig

__all__ = [
    "ModelBenchmark",
    "QuantizationComparison",
    "QuantizationReporter",
]


@dataclass(frozen=True)
class ModelBenchmark:
    """一个 OpenVINO 产物的精度、延迟与体积。"""

    precision: str
    model_path: str
    map50_95: float
    map50: float
    inference_ms: float
    size_bytes: int


@dataclass(frozen=True)
class QuantizationComparison:
    """同一运行时下 FP32 与 INT8 的可序列化对比。"""

    fp32: ModelBenchmark
    int8: ModelBenchmark
    map50_95_delta: float
    map50_delta: float
    speedup: float | None
    compression_ratio: float | None
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuantizationReporter:
    """构建并持久化量化对比报告。"""

    def compare(
        self,
        fp32_path: Path,
        fp32_metrics: ValidationMetrics,
        int8_path: Path,
        int8_metrics: ValidationMetrics,
        config: QuantizationConfig,
    ) -> QuantizationComparison:
        fp32 = self._benchmark("FP32", fp32_path, fp32_metrics)
        int8 = self._benchmark("INT8", int8_path, int8_metrics)
        return QuantizationComparison(
            fp32=fp32,
            int8=int8,
            map50_95_delta=int8.map50_95 - fp32.map50_95,
            map50_delta=int8.map50 - fp32.map50,
            speedup=self._ratio(fp32.inference_ms, int8.inference_ms),
            compression_ratio=self._ratio(fp32.size_bytes, int8.size_bytes),
            config={
                "source_model": str(config.model_input),
                "data_yaml": str(config.data_yaml),
                "imgsz": config.imgsz,
                "batch": config.batch,
                "device": config.device,
                "calibration_fraction": config.calibration_fraction,
            },
        )

    def write(
        self,
        comparison: QuantizationComparison,
        destination: Path,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination

    def _benchmark(
        self,
        precision: str,
        model_path: Path,
        metrics: ValidationMetrics,
    ) -> ModelBenchmark:
        return ModelBenchmark(
            precision=precision,
            model_path=str(model_path),
            map50_95=metrics.map50_95,
            map50=metrics.map50,
            inference_ms=metrics.inference_ms,
            size_bytes=self._artifact_size(model_path),
        )

    @staticmethod
    def _artifact_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _ratio(numerator: float | int, denominator: float | int) -> float | None:
        if denominator == 0:
            return None
        return float(numerator / denominator)
