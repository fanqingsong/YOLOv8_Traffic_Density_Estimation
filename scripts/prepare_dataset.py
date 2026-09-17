#!/usr/bin/env python3
"""Locate YOLO data.yaml and normalize paths for container training."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "/data/raw"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/data/dataset"))
MARKER = DATASET_DIR / ".prepare_complete"


def find_dataset_root(raw_dir: Path) -> Path:
    direct = raw_dir / "Vehicle_Detection_Image_Dataset"
    if (direct / "data.yaml").is_file():
        return direct

    for yaml_path in raw_dir.rglob("data.yaml"):
        return yaml_path.parent

    raise FileNotFoundError(
        f"No data.yaml found under {raw_dir}. Run download-data first."
    )


def patch_data_yaml(dataset_root: Path, yaml_out: Path) -> None:
    with open(dataset_root / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(dataset_root.resolve())
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main() -> int:
    if MARKER.is_file():
        print(f"Dataset already prepared ({MARKER}), skipping.")
        return 0

    dataset_root = find_dataset_root(RAW_DIR)
    print(f"Using dataset root: {dataset_root}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    patch_data_yaml(dataset_root, DATASET_DIR / "data.yaml")

    sample_video = dataset_root / "sample_video.mp4"
    export_video = Path("/data/sample_video.mp4")
    if sample_video.is_file():
        shutil.copy2(sample_video, export_video)
        print(f"Copied sample video to {export_video}")

    MARKER.touch()
    print(f"Wrote {DATASET_DIR / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
