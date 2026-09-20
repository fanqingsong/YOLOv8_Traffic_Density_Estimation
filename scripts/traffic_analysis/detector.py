"""YOLOv8 推理：加载权重、predict、把框画到帧上。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from .config import InferenceConfig

if TYPE_CHECKING:
    from ultralytics.engine.results import Boxes

__all__ = ["DetectionOutcome", "VehicleDetector"]


class DetectionOutcome:
    """单帧推理结果。``plotted`` 仍含检测带外的黑边，需再交给 DetectionBand.restore。"""

    def __init__(self, plotted: NDArray[np.uint8], boxes: Boxes) -> None:
        self._plotted = plotted
        self._boxes = boxes

    @property
    def plotted(self) -> NDArray[np.uint8]:
        return self._plotted

    @property
    def boxes(self) -> Boxes:
        return self._boxes


class VehicleDetector:
    """俯视车辆微调后的 Ultralytics YOLO 封装。不对外暴露底层 ``YOLO`` 实例。"""

    def __init__(self, inference: InferenceConfig) -> None:
        self._model_path = inference.model_path
        self._imgsz = inference.imgsz
        self._conf = inference.conf
        self._model = YOLO(self._model_path)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @property
    def conf(self) -> float:
        return self._conf

    def infer(self, detection_frame: NDArray[np.uint8]) -> DetectionOutcome:
        """在已涂黑的画布上推理。画布保持原分辨率，内部缩到 ``imgsz``。"""
        result = self._predict(detection_frame)
        plotted = self._draw_boxes(result)
        return DetectionOutcome(plotted, result.boxes)

    def _predict(self, detection_frame: NDArray[np.uint8]):
        results = self._model.predict(
            detection_frame, imgsz=self._imgsz, conf=self._conf
        )
        return results[0]

    def _draw_boxes(self, result) -> NDArray[np.uint8]:
        return result.plot(line_width=1)
