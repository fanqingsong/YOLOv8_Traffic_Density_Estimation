"""实时交通密度估计：每个文件一个内聚类，建议按下面顺序阅读。

学习路线
========

::

    config.py         AnalysisConfig（组合 LaneGeometry / InferenceConfig / HudStyle）
         │            对外：classmethod from_env + 只读 property
         ▼
    video_io.py       VideoSession  公开 open/read/write/preview/close；内部打开编解码器
         │
         ▼
    preprocessor.py   DetectionBand 公开 mask/restore；内部 _blackout
         │
         ▼
    detector.py       VehicleDetector.infer → DetectionOutcome（不暴露 YOLO 实例）
         │
         ▼
    counter.py        LaneCounter.tally → LaneSnapshot（计数与强度只读）
         │
         ▼
    visualizer.py     FrameAnnotator.annotate（多边形与 HUD 为内部方法）
         │
         ▼
    pipeline.py       TrafficAnalyzer 对外 process_frame / run
         │
         ▼
    python -m scripts.traffic_analysis
"""

from __future__ import annotations

from typing import Any

__all__ = ["TrafficAnalyzer", "run"]


def __getattr__(name: str) -> Any:
    if name in {"TrafficAnalyzer", "run"}:
        from .pipeline import TrafficAnalyzer, run

        return {"TrafficAnalyzer": TrafficAnalyzer, "run": run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
