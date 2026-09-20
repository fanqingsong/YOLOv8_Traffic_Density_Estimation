"""模型量化阶段入口。"""

from __future__ import annotations

import sys

from .config import QuantizationConfig
from .pipeline import ModelQuantizationPipeline


def run() -> int:
    try:
        result = ModelQuantizationPipeline(QuantizationConfig.from_env()).run()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    comparison = result.comparison
    print(f"Saved FP32 OpenVINO model to {result.fp32_output}")
    print(f"Saved INT8 OpenVINO model to {result.int8_output}")
    print(f"Saved comparison report to {result.report_output}")
    print(
        "mAP50-95: "
        f"{comparison.fp32.map50_95:.4f} -> {comparison.int8.map50_95:.4f} "
        f"({comparison.map50_95_delta:+.4f})"
    )
    print(
        "Inference: "
        f"{comparison.fp32.inference_ms:.3f} ms -> "
        f"{comparison.int8.inference_ms:.3f} ms"
    )
    if comparison.speedup is not None:
        print(f"Speedup: {comparison.speedup:.3f}x")
    if comparison.compression_ratio is not None:
        print(f"Compression: {comparison.compression_ratio:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
