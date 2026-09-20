"""复制数据集随附的样例视频。"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["SampleVideoExporter"]


class SampleVideoExporter:
    """仅在源视频存在时复制，并返回是否产生输出。"""

    def export(self, dataset_root: Path, output_path: Path) -> bool:
        source = dataset_root / "sample_video.mp4"
        if not source.is_file():
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)
        return True
