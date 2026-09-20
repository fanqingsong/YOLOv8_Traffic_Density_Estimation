"""模型量化包：config → backend → report → pipeline → 模块入口。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ModelQuantizationPipeline",
    "QuantizationConfig",
    "QuantizationResult",
    "run",
]


def __getattr__(name: str) -> Any:
    if name == "QuantizationConfig":
        from .config import QuantizationConfig

        return QuantizationConfig
    if name in {"ModelQuantizationPipeline", "QuantizationResult"}:
        from .pipeline import ModelQuantizationPipeline, QuantizationResult

        return {
            "ModelQuantizationPipeline": ModelQuantizationPipeline,
            "QuantizationResult": QuantizationResult,
        }[name]
    if name == "run":
        from .__main__ import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
