"""Kaggle 数据集下载阶段的环境配置。"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DownloadConfig"]


class DownloadConfig:
    """下载所需路径与数据集标识；对外只读。"""

    def __init__(
        self,
        dataset_slug: str,
        raw_dir: Path,
        kaggle_username: str = "",
        kaggle_key: str = "",
        kaggle_config_dir: Path | None = None,
    ) -> None:
        self._dataset_slug = dataset_slug
        self._raw_dir = raw_dir
        self._kaggle_username = kaggle_username
        self._kaggle_key = kaggle_key
        self._kaggle_config_dir = kaggle_config_dir or Path.home() / ".kaggle"

    @classmethod
    def from_env(cls) -> DownloadConfig:
        return cls(
            dataset_slug=os.getenv(
                "KAGGLE_DATASET",
                "farzadnekouei/top-view-vehicle-detection-image-dataset",
            ),
            raw_dir=Path(os.getenv("RAW_DATA_DIR", "/data/raw")),
            kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
            kaggle_key=os.getenv("KAGGLE_KEY", ""),
            kaggle_config_dir=Path(
                os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
            ),
        )

    @property
    def dataset_slug(self) -> str:
        return self._dataset_slug

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def marker(self) -> Path:
        return self._raw_dir / ".download_complete"

    @property
    def kaggle_username(self) -> str:
        return self._kaggle_username

    @property
    def kaggle_key(self) -> str:
        return self._kaggle_key

    @property
    def kaggle_json(self) -> Path:
        return self._kaggle_config_dir / "kaggle.json"
