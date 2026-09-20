"""运行配置与画面几何。换路口或分辨率时，优先改这里。

画面几何（坐标系原点在左上，x 向右、y 向下）
==========================================

样例视频约 1280×720。检测带、车道线和分界如下::

    x=0                                                          x≈1280
    ┌────────────────────────────────────────────────────────────┐ y=0
    │  顶部 HUD：左道数量 / 强度          右道数量 / 强度          │
    │  (10,50)(10,100)                    (820,50)(820,100)      │
    ├────────────────────────────────────────────────────────────┤ y=band_top=325
    │              ▲ 检测带起点（之上涂黑，不送 YOLO）              │
    │                                                            │
    │     left_lane 左车道(绿)           right_lane 右车道(蓝)    │
    │     (465,350)──(609,350)          (678,350)──(815,350)     │
    │         ╲           ╲                 ╲           ╲        │
    │          ╲           ╲                 ╲           ╲       │
    │   (2,630)────(510,630)          (743,630)──(1203,630)      │
    │                                                            │
    │              分界 x = lane_threshold = 609                  │
    │              │  box[0] < 609 → 左道，否则右道               │
    ├────────────────────────────────────────────────────────────┤ y=band_bottom=635
    │              ▼ 检测带终点（之下涂黑，不送 YOLO）              │
    └────────────────────────────────────────────────────────────┘ y≈720

说明：
  * 绿/蓝四边形只画在输出帧上，计数没有做「点在多边形内」判断。
  * 计数规则是检测框左上角 x（xyxy 的 box[0]）与 609 比大小。
  * band_top / band_bottom 是行号（y），不是图像宽度方向。
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

__all__ = ["AnalysisConfig", "HudStyle", "InferenceConfig", "LaneGeometry"]


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


class LaneGeometry:
    """检测带、分道阈值、可视化梯形。对外只读。"""

    def __init__(
        self,
        band_top: int = 325,
        band_bottom: int = 635,
        lane_threshold: int = 609,
    ) -> None:
        self._band_top = band_top
        self._band_bottom = band_bottom
        self._lane_threshold = lane_threshold
        self._left_polygon = np.array(
            [(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32
        )
        self._right_polygon = np.array(
            [(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32
        )
        self._left_color = (0, 255, 0)
        self._right_color = (255, 0, 0)
        self._validate()

    def _validate(self) -> None:
        if self._band_top >= self._band_bottom:
            raise ValueError("band_top must be < band_bottom")

    @property
    def band_top(self) -> int:
        return self._band_top

    @property
    def band_bottom(self) -> int:
        return self._band_bottom

    @property
    def lane_threshold(self) -> int:
        return self._lane_threshold

    @property
    def left_polygon(self) -> NDArray[np.int32]:
        return self._left_polygon.copy()

    @property
    def right_polygon(self) -> NDArray[np.int32]:
        return self._right_polygon.copy()

    @property
    def left_color(self) -> tuple[int, int, int]:
        return self._left_color

    @property
    def right_color(self) -> tuple[int, int, int]:
        return self._right_color


class InferenceConfig:
    """权重路径与 YOLO 推理超参。"""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.4,
    ) -> None:
        self._model_path = model_path
        self._imgsz = imgsz
        self._conf = conf

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @property
    def conf(self) -> float:
        return self._conf


class HudStyle:
    """顶部统计条的位置与配色（OpenCV BGR）。"""

    def __init__(self) -> None:
        self._left_count = (10, 50)
        self._right_count = (820, 50)
        self._left_intensity = (10, 100)
        self._right_intensity = (820, 100)
        self._font_scale = 1.0
        self._font_color = (255, 255, 255)
        self._background_color = (0, 0, 255)
        self._bar_width = 460
        self._bar_ascent = 25
        self._bar_descent = 10

    @property
    def left_count_anchor(self) -> tuple[int, int]:
        return self._left_count

    @property
    def right_count_anchor(self) -> tuple[int, int]:
        return self._right_count

    @property
    def left_intensity_anchor(self) -> tuple[int, int]:
        return self._left_intensity

    @property
    def right_intensity_anchor(self) -> tuple[int, int]:
        return self._right_intensity

    @property
    def font_scale(self) -> float:
        return self._font_scale

    @property
    def font_color(self) -> tuple[int, int, int]:
        return self._font_color

    @property
    def background_color(self) -> tuple[int, int, int]:
        return self._background_color

    @property
    def bar_width(self) -> int:
        return self._bar_width

    @property
    def bar_ascent(self) -> int:
        return self._bar_ascent

    @property
    def bar_descent(self) -> int:
        return self._bar_descent


class AnalysisConfig:
    """一次分析运行所需的全部配置。请用 ``from_env()`` 构造。"""

    def __init__(
        self,
        video_path: str,
        output_path: str,
        display_video: bool,
        inference: InferenceConfig,
        geometry: LaneGeometry | None = None,
        hud: HudStyle | None = None,
        heavy_traffic_threshold: int = 10,
    ) -> None:
        self._video_path = video_path
        self._output_path = output_path
        self._display_video = display_video
        self._inference = inference
        self._geometry = geometry or LaneGeometry()
        self._hud = hud or HudStyle()
        self._heavy_traffic_threshold = heavy_traffic_threshold

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        """Docker Compose 用环境变量覆盖路径；本机默认相对路径 + 弹窗。"""
        return cls(
            video_path=os.getenv("VIDEO_PATH", "sample_video.mp4"),
            output_path=os.getenv("OUTPUT_PATH", "processed_sample_video.avi"),
            display_video=_env_flag("DISPLAY_VIDEO", "true"),
            inference=InferenceConfig(
                model_path=os.getenv("MODEL_PATH", "models/best.pt"),
            ),
        )

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def output_path(self) -> str:
        return self._output_path

    @property
    def display_video(self) -> bool:
        return self._display_video

    @property
    def inference(self) -> InferenceConfig:
        return self._inference

    @property
    def geometry(self) -> LaneGeometry:
        return self._geometry

    @property
    def hud(self) -> HudStyle:
        return self._hud

    @property
    def heavy_traffic_threshold(self) -> int:
        return self._heavy_traffic_threshold
