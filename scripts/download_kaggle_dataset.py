#!/usr/bin/env python3
"""Download the Top-View Vehicle Detection dataset from Kaggle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = os.getenv(
    "KAGGLE_DATASET", "farzadnekouei/top-view-vehicle-detection-image-dataset"
)
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "/data/raw"))
MARKER = RAW_DIR / ".download_complete"


def _credentials_configured() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.is_file()


def main() -> int:
    if MARKER.is_file():
        print(f"Dataset already present ({MARKER}), skipping download.")
        return 0

    if not _credentials_configured():
        print(
            "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY in .env, "
            "or mount ~/.kaggle/kaggle.json to /root/.kaggle/kaggle.json.",
            file=sys.stderr,
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(RAW_DIR),
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    MARKER.touch()
    print("Download finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
