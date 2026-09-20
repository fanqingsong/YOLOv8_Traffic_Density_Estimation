"""Kaggle 下载包：config → downloader → ``python -m scripts.kaggle_download``。"""

from __future__ import annotations

from typing import Any

__all__ = ["DownloadConfig", "DownloadResult", "KaggleDatasetDownloader", "run"]


def __getattr__(name: str) -> Any:
    if name == "DownloadConfig":
        from .config import DownloadConfig

        return DownloadConfig
    if name in {"DownloadResult", "KaggleDatasetDownloader"}:
        from .downloader import DownloadResult, KaggleDatasetDownloader

        return {
            "DownloadResult": DownloadResult,
            "KaggleDatasetDownloader": KaggleDatasetDownloader,
        }[name]
    if name == "run":
        from .__main__ import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
