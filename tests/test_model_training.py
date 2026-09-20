from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.model_training.backend import YoloTrainingBackend
from scripts.model_training.config import TrainConfig
from scripts.model_training.pipeline import ModelTrainingPipeline


def make_config(root: Path, export_onnx: bool = True) -> TrainConfig:
    return TrainConfig(
        data_yaml=root / "data.yaml",
        base_model="base.pt",
        runs_project=root / "runs",
        run_name="train",
        model_output=root / "models" / "best.pt",
        onnx_output=root / "models" / "best.onnx",
        epochs=2,
        batch=4,
        device="cpu",
        patience=3,
        lr0=0.001,
        lrf=0.1,
        dropout=0.2,
        seed=7,
        export_onnx=export_onnx,
    )


class _FakeBackend:
    def __init__(self, exported_path: Path) -> None:
        self.exported_path = exported_path
        self.trained = False

    def train(self, config: TrainConfig) -> Path:
        self.trained = True
        config.best_weights.parent.mkdir(parents=True)
        config.best_weights.write_bytes(b"weights")
        return config.best_weights

    def export_onnx(self, model_path: Path) -> Path:
        self.exported_path.write_bytes(b"onnx")
        return self.exported_path


class _FakeModel:
    def __init__(self, export_path: Path) -> None:
        self.export_path = export_path
        self.train_kwargs: dict[str, object] = {}

    def train(self, **kwargs: object) -> None:
        self.train_kwargs = kwargs

    def export(self, **kwargs: object) -> Path:
        return self.export_path


class ModelTrainingTest(unittest.TestCase):
    def test_pipeline_copies_weight_and_onnx_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            config.data_yaml.write_text("names: [Vehicle]\n", encoding="utf-8")
            exported = root / "exported.onnx"
            backend = _FakeBackend(exported)

            result = ModelTrainingPipeline(config, backend).run()

            self.assertTrue(backend.trained)
            self.assertEqual(config.model_output.read_bytes(), b"weights")
            self.assertEqual(config.onnx_output.read_bytes(), b"onnx")
            self.assertEqual(result.onnx_output, config.onnx_output)

    def test_backend_passes_typed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            fake_model = _FakeModel(root / "model.onnx")
            backend = YoloTrainingBackend(lambda path: fake_model)

            best = backend.train(config)

            self.assertEqual(best, config.best_weights)
            self.assertEqual(fake_model.train_kwargs["device"], "cpu")
            self.assertEqual(fake_model.train_kwargs["epochs"], 2)
            self.assertEqual(fake_model.train_kwargs["data"], str(config.data_yaml))

    def test_from_env_defaults_to_cpu(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = TrainConfig.from_env()

        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.export_onnx)
