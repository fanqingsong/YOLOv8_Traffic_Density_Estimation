"""在输出帧上画车道 ROI 和顶部统计条。"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import HudStyle, LaneGeometry
from .counter import LaneSnapshot

__all__ = ["FrameAnnotator"]


class FrameAnnotator:
    """绿/蓝梯形只用于显示，不参与 LaneCounter 的分道逻辑。"""

    def __init__(self, geometry: LaneGeometry, hud: HudStyle) -> None:
        self._geometry = geometry
        self._hud = hud

    def annotate(self, frame: NDArray[np.uint8], snapshot: LaneSnapshot) -> NDArray[np.uint8]:
        self._draw_lane_polygons(frame)
        self._draw_hud(frame, snapshot)
        return frame

    def _draw_lane_polygons(self, frame: NDArray[np.uint8]) -> None:
        geometry = self._geometry
        cv2.polylines(
            frame,
            [geometry.left_polygon],
            isClosed=True,
            color=geometry.left_color,
            thickness=2,
        )
        cv2.polylines(
            frame,
            [geometry.right_polygon],
            isClosed=True,
            color=geometry.right_color,
            thickness=2,
        )

    def _draw_hud(self, frame: NDArray[np.uint8], snapshot: LaneSnapshot) -> None:
        hud = self._hud
        self._hud_bar(
            frame, hud.left_count_anchor, f"Vehicles in Left Lane: {snapshot.left_count}"
        )
        self._hud_bar(
            frame, hud.left_intensity_anchor, f"Traffic Intensity: {snapshot.left_intensity}"
        )
        self._hud_bar(
            frame, hud.right_count_anchor, f"Vehicles in Right Lane: {snapshot.right_count}"
        )
        self._hud_bar(
            frame, hud.right_intensity_anchor, f"Traffic Intensity: {snapshot.right_intensity}"
        )

    def _hud_bar(self, frame: NDArray[np.uint8], anchor: tuple[int, int], text: str) -> None:
        x, y = anchor
        hud = self._hud
        cv2.rectangle(
            frame,
            (x - 10, y - hud.bar_ascent),
            (x + hud.bar_width, y + hud.bar_descent),
            hud.background_color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            hud.font_scale,
            hud.font_color,
            2,
            cv2.LINE_AA,
        )
