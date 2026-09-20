"""编排 OpenVINO 导出、验证与量化对比报告。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .backend import OpenVinoQuantizationBackend, ValidationMetrics
from .config import QuantizationConfig
from .report import QuantizationComparison, QuantizationReporter

__all__ = ["ModelQuantizationPipeline", "QuantizationResult"]


@dataclass(frozen=True)
class QuantizationResult:
    fp32_output: Path
    int8_output: Path
    report_output: Path
    comparison: QuantizationComparison


class _QuantizationBackend(Protocol):
    def export_fp32(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path: ...

    def export_int8(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path: ...

    def validate(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> ValidationMetrics: ...


class ModelQuantizationPipeline:
    """量化阶段门面；只向调用方返回项目类型。"""

    def __init__(
        self,
        config: QuantizationConfig,
        backend: _QuantizationBackend | None = None,
        reporter: QuantizationReporter | None = None,
    ) -> None:
        self._config = config
        self._backend = backend or OpenVinoQuantizationBackend()
        self._reporter = reporter or QuantizationReporter()

    def run(self) -> QuantizationResult:
        self._require_file(self._config.model_input, "source model")
        self._require_file(self._config.data_yaml, "dataset configuration")

        fp32_export = self._backend.export_fp32(
            self._config.model_input,
            self._config,
        )
        self._require_artifact(fp32_export, "FP32 OpenVINO export")
        self._copy_artifact(fp32_export, self._config.fp32_output)

        int8_export = self._backend.export_int8(
            self._config.model_input,
            self._config,
        )
        self._require_artifact(int8_export, "INT8 OpenVINO export")
        self._copy_artifact(int8_export, self._config.int8_output)

        fp32_metrics = self._backend.validate(
            self._config.fp32_output,
            self._config,
        )
        int8_metrics = self._backend.validate(
            self._config.int8_output,
            self._config,
        )
        comparison = self._reporter.compare(
            self._config.fp32_output,
            fp32_metrics,
            self._config.int8_output,
            int8_metrics,
            self._config,
        )
        report_output = self._reporter.write(
            comparison,
            self._config.report_output,
        )
        return QuantizationResult(
            fp32_output=self._config.fp32_output,
            int8_output=self._config.int8_output,
            report_output=report_output,
            comparison=comparison,
        )

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    @staticmethod
    def _require_artifact(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{label} finished but {path} was not found.")

    @staticmethod
    def _copy_artifact(source: Path, destination: Path) -> None:
        if source.resolve() == destination.resolve():
            return
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
