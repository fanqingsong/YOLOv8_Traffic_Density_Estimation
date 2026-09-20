"""数据准备包：config → locator → yaml_normalizer → video_exporter → preparer。"""

from __future__ import annotations

from typing import Any

__all__ = ["DatasetPreparer", "PreparationResult", "PrepareConfig", "run"]


def __getattr__(name: str) -> Any:
    if name == "PrepareConfig":
        from .config import PrepareConfig

        return PrepareConfig
    if name in {"DatasetPreparer", "PreparationResult"}:
        from .preparer import DatasetPreparer, PreparationResult

        return {
            "DatasetPreparer": DatasetPreparer,
            "PreparationResult": PreparationResult,
        }[name]
    if name == "run":
        from .__main__ import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
