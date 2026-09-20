"""模型量化阶段的集中式环境配置。"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["QuantizationConfig"]


class QuantizationConfig:
    """OpenVINO 量化、验证与报告参数；所有值均为只读属性。"""

    def __init__(
        self,
        model_input: Path,
        data_yaml: Path,
        fp32_output: Path,
        int8_output: Path,
        report_output: Path,
        imgsz: int,
        batch: int,
        device: str,
        calibration_fraction: float,
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz must be > 0")
        if batch <= 0:
            raise ValueError("batch must be > 0")
        if not 0 < calibration_fraction <= 1:
            raise ValueError("calibration_fraction must be in (0, 1]")

        self._model_input = model_input
        self._data_yaml = data_yaml
        self._fp32_output = fp32_output
        self._int8_output = int8_output
        self._report_output = report_output
        self._imgsz = imgsz
        self._batch = batch
        self._device = device
        self._calibration_fraction = calibration_fraction

    @classmethod
    def from_env(cls) -> QuantizationConfig:
        return cls(
            model_input=Path(os.getenv("QUANT_MODEL_INPUT", "/models/best.pt")),
            data_yaml=Path(os.getenv("DATA_YAML", "/data/dataset/data.yaml")),
            fp32_output=Path(
                os.getenv(
                    "OPENVINO_FP32_OUTPUT",
                    "/models/best_openvino_fp32",
                )
            ),
            int8_output=Path(
                os.getenv(
                    "OPENVINO_INT8_OUTPUT",
                    "/models/best_openvino_int8",
                )
            ),
            report_output=Path(
                os.getenv(
                    "QUANT_REPORT_OUTPUT",
                    "/models/quantization_comparison.json",
                )
            ),
            imgsz=int(os.getenv("QUANT_IMGSZ", "640")),
            batch=int(os.getenv("QUANT_BATCH", "1")),
            device=os.getenv("QUANT_DEVICE", "cpu"),
            calibration_fraction=float(
                os.getenv("QUANT_CALIBRATION_FRACTION", "1.0")
            ),
        )

    @property
    def model_input(self) -> Path:
        return self._model_input

    @property
    def data_yaml(self) -> Path:
        return self._data_yaml

    @property
    def fp32_output(self) -> Path:
        return self._fp32_output

    @property
    def int8_output(self) -> Path:
        return self._int8_output

    @property
    def report_output(self) -> Path:
        return self._report_output

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @property
    def batch(self) -> int:
        return self._batch

    @property
    def device(self) -> str:
        return self._device

    @property
    def calibration_fraction(self) -> float:
        return self._calibration_fraction
