"""检测带：屏蔽带外行，推理后再把原图像素贴回去。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import LaneGeometry

__all__ = ["DetectionBand"]


class DetectionBand:
    """只保留竖直检测带 ``[top, bottom)``，其余行涂黑。"""

    def __init__(self, geometry: LaneGeometry) -> None:
        self._top = geometry.band_top
        self._bottom = geometry.band_bottom

    @property
    def top(self) -> int:
        return self._top

    @property
    def bottom(self) -> int:
        return self._bottom

    def mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """复制一帧并涂黑带外区域，供 YOLO 使用。不得改写传入的 ``frame``。"""
        detection_frame = frame.copy()
        self._blackout(detection_frame)
        return detection_frame

    def restore(
        self, processed: NDArray[np.uint8], original: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """把 plot() 画在黑边上的内容换成原始像素，框只留在检测带内。"""
        processed[: self._top, :] = original[: self._top, :].copy()
        processed[self._bottom :, :] = original[self._bottom :, :].copy()
        return processed

    def _blackout(self, frame: NDArray[np.uint8]) -> None:
        frame[: self._top, :] = 0
        frame[self._bottom :, :] = 0
