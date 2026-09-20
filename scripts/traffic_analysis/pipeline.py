"""把各子模块串成逐帧循环。

::

    sample_video.mp4
            │
            ▼
    VideoSession.read
            │
            ▼
    DetectionBand.mask                 y ∈ [band_top, band_bottom)
            │
            ▼
    VehicleDetector.infer              imgsz / conf 见 InferenceConfig
            │
            ├─ DetectionBand.restore
            ├─ LaneCounter.tally       box[0] vs lane_threshold
            └─ FrameAnnotator.annotate 可视化 ROI + HUD
            │
            ▼
    VideoSession.write（Docker 默认不弹窗）
"""

from __future__ import annotations

from numpy.typing import NDArray

from .config import AnalysisConfig
from .counter import LaneCounter, LaneSnapshot
from .detector import VehicleDetector
from .preprocessor import DetectionBand
from .video_io import VideoSession
from .visualizer import FrameAnnotator

__all__ = ["TrafficAnalyzer", "run"]


class TrafficAnalyzer:
    """一次完整的交通密度分析。对外只需 ``process_frame`` / ``run``。"""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self._config = config or AnalysisConfig.from_env()
        geometry = self._config.geometry
        self._band = DetectionBand(geometry)
        self._detector = VehicleDetector(self._config.inference)
        self._counter = LaneCounter(
            geometry.lane_threshold, self._config.heavy_traffic_threshold
        )
        self._annotator = FrameAnnotator(geometry, self._config.hud)

    @property
    def config(self) -> AnalysisConfig:
        return self._config

    @property
    def last_snapshot(self) -> LaneSnapshot | None:
        return self._counter.last_snapshot

    def process_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        masked = self._band.mask(frame)
        outcome = self._detector.infer(masked)
        drawn = self._band.restore(outcome.plotted, frame)
        snapshot = self._counter.tally(outcome.boxes)
        return self._annotator.annotate(drawn, snapshot)

    def run(self) -> int:
        cfg = self._config
        with VideoSession(cfg.video_path, cfg.output_path, cfg.display_video) as video:
            self._pump(video)
        return 0

    def _pump(self, video: VideoSession) -> None:
        while video.is_open:
            frame = video.read()
            if frame is None:
                break
            processed = self.process_frame(frame)
            video.write(processed)
            if not video.preview(processed):
                break


def run(config: AnalysisConfig | None = None) -> int:
    return TrafficAnalyzer(config).run()
