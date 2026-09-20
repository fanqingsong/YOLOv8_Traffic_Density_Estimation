"""Kaggle 下载阶段入口。"""

from __future__ import annotations

import sys

from .config import DownloadConfig
from .downloader import KaggleDatasetDownloader


def run() -> int:
    try:
        result = KaggleDatasetDownloader(DownloadConfig.from_env()).run()
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if result.skipped:
        print(f"Dataset already present ({result.marker}), skipping download.")
    else:
        print("Download finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
