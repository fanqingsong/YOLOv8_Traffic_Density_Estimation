"""数据集准备阶段的环境配置。"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PrepareConfig"]


class PrepareConfig:
    """准备阶段输入、输出路径；对外只读。"""

    def __init__(
        self,
        raw_dir: Path,
        dataset_dir: Path,
        sample_video_output: Path,
    ) -> None:
        self._raw_dir = raw_dir
        self._dataset_dir = dataset_dir
        self._sample_video_output = sample_video_output

    @classmethod
    def from_env(cls) -> PrepareConfig:
        return cls(
            raw_dir=Path(os.getenv("RAW_DATA_DIR", "/data/raw")),
            dataset_dir=Path(os.getenv("DATASET_DIR", "/data/dataset")),
            sample_video_output=Path(
                os.getenv("SAMPLE_VIDEO_OUTPUT", "/data/sample_video.mp4")
            ),
        )

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    @property
    def data_yaml(self) -> Path:
        return self._dataset_dir / "data.yaml"

    @property
    def marker(self) -> Path:
        return self._dataset_dir / ".prepare_complete"

    @property
    def sample_video_output(self) -> Path:
        return self._sample_video_output
