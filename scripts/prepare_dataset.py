#!/usr/bin/env python3
"""Compatibility launcher for :mod:`scripts.dataset_preparation`."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dataset_preparation import run


if __name__ == "__main__":
    raise SystemExit(run())
