from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.model_quantization.backend import (
    OpenVinoQuantizationBackend,
    ValidationMetrics,
)
from scripts.model_quantization.config import QuantizationConfig
from scripts.model_quantization.pipeline import ModelQuantizationPipeline


def make_config(root: Path) -> QuantizationConfig:
    return QuantizationConfig(
        model_input=root / "models" / "best.pt",
        data_yaml=root / "data" / "data.yaml",
        fp32_output=root / "models" / "best_openvino_fp32",
        int8_output=root / "models" / "best_openvino_int8",
        report_output=root / "models" / "comparison.json",
        imgsz=640,
        batch=1,
        device="cpu",
        calibration_fraction=0.5,
    )


class _FakeBackend:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.validated: list[Path] = []

    def export_fp32(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path:
        output = self._root / "generated_fp32"
        output.mkdir()
        (output / "model.bin").write_bytes(b"12345678")
        return output

    def export_int8(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> Path:
        output = self._root / "generated_int8"
        output.mkdir()
        (output / "model.bin").write_bytes(b"1234")
        return output

    def validate(
        self,
        model_path: Path,
        config: QuantizationConfig,
    ) -> ValidationMetrics:
        self.validated.append(model_path)
        if model_path == config.fp32_output:
            return ValidationMetrics(0.8, 0.9, 20.0)
        return ValidationMetrics(0.78, 0.88, 10.0)


class _BoxMetrics:
    map = 0.7
    map50 = 0.85


class _ValidationResult:
    box = _BoxMetrics()
    speed = {"inference": 12.5}


class _FakeModel:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.export_calls: list[dict[str, object]] = []
        self.val_calls: list[dict[str, object]] = []

    def export(self, **kwargs: object) -> Path:
        self.export_calls.append(kwargs)
        return self._root

    def val(self, **kwargs: object) -> _ValidationResult:
        self.val_calls.append(kwargs)
        return _ValidationResult()


class ModelQuantizationTest(unittest.TestCase):
    def test_pipeline_exports_validates_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            config.model_input.parent.mkdir(parents=True)
            config.model_input.write_bytes(b"weights")
            config.data_yaml.parent.mkdir(parents=True)
            config.data_yaml.write_text("names: [Vehicle]\n", encoding="utf-8")
            backend = _FakeBackend(root)

            result = ModelQuantizationPipeline(config, backend).run()

            self.assertEqual(
                backend.validated,
                [config.fp32_output, config.int8_output],
            )
            self.assertEqual(
                (config.fp32_output / "model.bin").read_bytes(),
                b"12345678",
            )
            self.assertEqual(
                (config.int8_output / "model.bin").read_bytes(),
                b"1234",
            )
            self.assertAlmostEqual(result.comparison.map50_95_delta, -0.02)
            self.assertEqual(result.comparison.speedup, 2.0)
            self.assertEqual(result.comparison.compression_ratio, 2.0)
            report = json.loads(config.report_output.read_text(encoding="utf-8"))
            self.assertEqual(report["fp32"]["precision"], "FP32")
            self.assertEqual(report["int8"]["precision"], "INT8")
            self.assertEqual(report["config"]["calibration_fraction"], 0.5)

    def test_backend_passes_openvino_quantization_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            fake_model = _FakeModel(root / "exported")
            backend = OpenVinoQuantizationBackend(lambda path: fake_model)

            backend.export_fp32(config.model_input, config)
            backend.export_int8(config.model_input, config)
            metrics = backend.validate(config.int8_output, config)

            self.assertEqual(fake_model.export_calls[0]["quantize"], 32)
            self.assertEqual(fake_model.export_calls[1]["quantize"], 8)
            self.assertEqual(
                fake_model.export_calls[1]["data"],
                str(config.data_yaml),
            )
            self.assertEqual(fake_model.export_calls[1]["fraction"], 0.5)
            self.assertEqual(fake_model.val_calls[0]["device"], "cpu")
            self.assertEqual(metrics, ValidationMetrics(0.7, 0.85, 12.5))

    def test_from_env_defaults_to_cpu_and_full_calibration(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = QuantizationConfig.from_env()

        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.calibration_fraction, 1.0)
        self.assertEqual(config.model_input, Path("/models/best.pt"))

    def test_rejects_invalid_calibration_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "calibration_fraction"):
                QuantizationConfig(
                    model_input=root / "best.pt",
                    data_yaml=root / "data.yaml",
                    fp32_output=root / "fp32",
                    int8_output=root / "int8",
                    report_output=root / "report.json",
                    imgsz=640,
                    batch=1,
                    device="cpu",
                    calibration_fraction=0,
                )


if __name__ == "__main__":
    unittest.main()
