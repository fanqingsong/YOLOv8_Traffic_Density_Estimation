"""按检测框位置把车辆分到左右车道，并给出拥堵等级。"""

from __future__ import annotations

from typing import Protocol

__all__ = ["LaneCounter", "LaneSnapshot"]


class _HasXyxy(Protocol):
    @property
    def xyxy(self): ...


class LaneSnapshot:
    """一帧的分道计数。强度在构造时算好，之后只读。"""

    def __init__(self, left_count: int, right_count: int, heavy_threshold: int) -> None:
        self._left_count = left_count
        self._right_count = right_count
        self._left_intensity = self._label(left_count, heavy_threshold)
        self._right_intensity = self._label(right_count, heavy_threshold)

    @staticmethod
    def _label(vehicle_count: int, heavy_threshold: int) -> str:
        return "Heavy" if vehicle_count > heavy_threshold else "Smooth"

    @property
    def left_count(self) -> int:
        return self._left_count

    @property
    def right_count(self) -> int:
        return self._right_count

    @property
    def left_intensity(self) -> str:
        return self._left_intensity

    @property
    def right_intensity(self) -> str:
        return self._right_intensity


class LaneCounter:
    """用检测框左上角 x 与分界比较；不做点在多边形内判断。"""

    def __init__(self, lane_threshold: int, heavy_traffic_threshold: int) -> None:
        self._lane_threshold = lane_threshold
        self._heavy_traffic_threshold = heavy_traffic_threshold
        self._last: LaneSnapshot | None = None

    @property
    def lane_threshold(self) -> int:
        return self._lane_threshold

    @property
    def heavy_traffic_threshold(self) -> int:
        return self._heavy_traffic_threshold

    @property
    def last_snapshot(self) -> LaneSnapshot | None:
        return self._last

    def tally(self, boxes: _HasXyxy) -> LaneSnapshot:
        left = 0
        right = 0
        for box in boxes.xyxy:
            if self._is_left_lane(float(box[0])):
                left += 1
            else:
                right += 1
        snapshot = LaneSnapshot(left, right, self._heavy_traffic_threshold)
        self._last = snapshot
        return snapshot

    def _is_left_lane(self, box_left_x: float) -> bool:
        return box_left_x < self._lane_threshold
