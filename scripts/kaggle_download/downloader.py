"""通过 Kaggle CLI 下载并标记数据集。"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import DownloadConfig

__all__ = ["DownloadResult", "KaggleDatasetDownloader"]


@dataclass(frozen=True)
class DownloadResult:
    marker: Path
    skipped: bool


class KaggleDatasetDownloader:
    """封装凭证检查、CLI 调用和幂等 marker。"""

    def __init__(
        self,
        config: DownloadConfig,
        command_runner: Callable[[Sequence[str]], object] | None = None,
    ) -> None:
        self._config = config
        self._command_runner = command_runner or subprocess.check_call

    def run(self) -> DownloadResult:
        if self._config.marker.is_file():
            return DownloadResult(self._config.marker, skipped=True)
        if not self._credentials_configured():
            raise RuntimeError(
                "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "or provide kaggle.json."
            )

        self._config.raw_dir.mkdir(parents=True, exist_ok=True)
        self._command_runner(self._command())
        self._config.marker.touch()
        return DownloadResult(self._config.marker, skipped=False)

    def _credentials_configured(self) -> bool:
        has_environment_credentials = bool(
            self._config.kaggle_username and self._config.kaggle_key
        )
        return has_environment_credentials or self._config.kaggle_json.is_file()

    def _command(self) -> list[str]:
        return [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self._config.dataset_slug,
            "-p",
            str(self._config.raw_dir),
            "--unzip",
        ]
