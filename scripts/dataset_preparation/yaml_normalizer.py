"""将 YOLO data.yaml 的根路径改为实际解压目录。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["DataYamlNormalizer"]


class DataYamlNormalizer:
    """读取、校验并写出训练使用的 data.yaml。"""

    def normalize(self, dataset_root: Path, output_path: Path) -> None:
        with (dataset_root / "data.yaml").open(encoding="utf-8") as stream:
            document: Any = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"{dataset_root / 'data.yaml'} must contain a mapping")

        document["path"] = str(dataset_root.resolve())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                document,
                stream,
                default_flow_style=False,
                sort_keys=False,
            )
