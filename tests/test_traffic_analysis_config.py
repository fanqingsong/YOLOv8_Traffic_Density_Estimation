from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.traffic_analysis.config import InferenceConfig


class InferenceConfigTest(unittest.TestCase):
    def test_defaults_to_pytorch_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = InferenceConfig.from_env()

        self.assertEqual(config.model_variant, "pytorch")
        self.assertEqual(config.model_path, "models/best.pt")

    def test_selects_openvino_int8_model(self) -> None:
        environment = {
            "MODEL_VARIANT": "openvino-int8",
            "MODEL_ROOT": "/app/models",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = InferenceConfig.from_env()

        self.assertEqual(config.model_variant, "openvino-int8")
        self.assertEqual(
            config.model_path,
            "/app/models/best_int8_openvino_model",
        )

    def test_explicit_path_overrides_variant(self) -> None:
        environment = {
            "MODEL_VARIANT": "openvino-fp32",
            "MODEL_PATH": "/models/custom_openvino_model",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = InferenceConfig.from_env()

        self.assertEqual(config.model_path, "/models/custom_openvino_model")
        self.assertEqual(config.model_variant, "openvino-fp32")

    def test_rejects_unknown_variant_without_path_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"MODEL_VARIANT": "unknown"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported MODEL_VARIANT"):
                InferenceConfig.from_env()


if __name__ == "__main__":
    unittest.main()
