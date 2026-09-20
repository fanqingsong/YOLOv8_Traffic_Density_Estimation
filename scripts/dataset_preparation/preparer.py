"""编排数据集定位、YAML 规范化和样例视频复制。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PrepareConfig
from .locator import DatasetRootLocator
from .video_exporter import SampleVideoExporter
from .yaml_normalizer import DataYamlNormalizer

__all__ = ["DatasetPreparer", "PreparationResult"]


@dataclass(frozen=True)
class PreparationResult:
    data_yaml: Path
    dataset_root: Path | None
    sample_video_copied: bool
    skipped: bool


class DatasetPreparer:
    """对外提供单一准备操作，内部协作者只通过公开方法调用。"""

    def __init__(
        self,
        config: PrepareConfig,
        locator: DatasetRootLocator | None = None,
        normalizer: DataYamlNormalizer | None = None,
        video_exporter: SampleVideoExporter | None = None,
    ) -> None:
        self._config = config
        self._locator = locator or DatasetRootLocator()
        self._normalizer = normalizer or DataYamlNormalizer()
        self._video_exporter = video_exporter or SampleVideoExporter()

    def run(self) -> PreparationResult:
        if self._config.marker.is_file():
            return PreparationResult(
                data_yaml=self._config.data_yaml,
                dataset_root=None,
                sample_video_copied=False,
                skipped=True,
            )

        dataset_root = self._locator.locate(self._config.raw_dir)
        self._normalizer.normalize(dataset_root, self._config.data_yaml)
        copied = self._video_exporter.export(
            dataset_root,
            self._config.sample_video_output,
        )
        self._config.marker.touch()
        return PreparationResult(
            data_yaml=self._config.data_yaml,
            dataset_root=dataset_root,
            sample_video_copied=copied,
            skipped=False,
        )
