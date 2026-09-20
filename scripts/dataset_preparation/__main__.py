"""数据集准备阶段入口。"""

from __future__ import annotations

import sys

from .config import PrepareConfig
from .preparer import DatasetPreparer


def run() -> int:
    try:
        result = DatasetPreparer(PrepareConfig.from_env()).run()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if result.skipped:
        print(f"Dataset already prepared ({result.data_yaml.parent}), skipping.")
    else:
        if result.dataset_root is not None:
            print(f"Using dataset root: {result.dataset_root}")
        print(f"Wrote {result.data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
