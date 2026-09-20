#!/usr/bin/env python3
"""兼容入口（从仓库根目录执行）。

推荐: python -m scripts.traffic_analysis
等价: python scripts/real_time_traffic_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.traffic_analysis.pipeline import run

if __name__ == "__main__":
    raise SystemExit(run())
