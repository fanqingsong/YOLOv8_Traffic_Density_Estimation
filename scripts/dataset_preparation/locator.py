"""定位下载内容中的 YOLO 数据集根目录。"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DatasetRootLocator"]


class DatasetRootLocator:
    """优先识别标准目录，并拒绝有歧义的候选集合。"""

    def locate(self, raw_dir: Path) -> Path:
        direct = raw_dir / "Vehicle_Detection_Image_Dataset"
        if (direct / "data.yaml").is_file():
            return direct

        candidates = sorted(raw_dir.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(
                f"No data.yaml found under {raw_dir}. Run download-data first."
            )
        if len(candidates) > 1:
            paths = ", ".join(str(path) for path in candidates)
            raise RuntimeError(f"Multiple data.yaml files found under {raw_dir}: {paths}")
        return candidates[0].parent
