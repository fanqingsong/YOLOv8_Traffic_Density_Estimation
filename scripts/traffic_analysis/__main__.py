"""仓库根目录执行: python -m scripts.traffic_analysis"""

from .pipeline import run

if __name__ == "__main__":
    raise SystemExit(run())
