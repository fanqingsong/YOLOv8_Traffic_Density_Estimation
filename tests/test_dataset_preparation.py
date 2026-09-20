from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.dataset_preparation.config import PrepareConfig
from scripts.dataset_preparation.locator import DatasetRootLocator
from scripts.dataset_preparation.preparer import DatasetPreparer


class DatasetPreparationTest(unittest.TestCase):
    def test_prepares_yaml_and_sample_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "raw" / "Vehicle_Detection_Image_Dataset"
            dataset_root.mkdir(parents=True)
            (dataset_root / "data.yaml").write_text(
                "train: images/train\nval: images/val\nnames: [Vehicle]\n",
                encoding="utf-8",
            )
            (dataset_root / "sample_video.mp4").write_bytes(b"video")
            config = PrepareConfig(
                raw_dir=root / "raw",
                dataset_dir=root / "prepared",
                sample_video_output=root / "sample_video.mp4",
            )

            result = DatasetPreparer(config).run()

            normalized = yaml.safe_load(config.data_yaml.read_text(encoding="utf-8"))
            self.assertEqual(normalized["path"], str(dataset_root.resolve()))
            self.assertEqual(config.sample_video_output.read_bytes(), b"video")
            self.assertTrue(config.marker.is_file())
            self.assertTrue(result.sample_video_copied)

    def test_marker_makes_preparation_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PrepareConfig(root / "missing", root / "prepared", root / "video")
            config.dataset_dir.mkdir(parents=True)
            config.marker.touch()

            result = DatasetPreparer(config).run()

            self.assertTrue(result.skipped)

    def test_multiple_yaml_candidates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            for name in ("first", "second"):
                candidate = raw_dir / name
                candidate.mkdir()
                (candidate / "data.yaml").touch()

            with self.assertRaisesRegex(RuntimeError, "Multiple data.yaml"):
                DatasetRootLocator().locate(raw_dir)
