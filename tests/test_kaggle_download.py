from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_download.config import DownloadConfig
from scripts.kaggle_download.downloader import KaggleDatasetDownloader


class KaggleDatasetDownloaderTest(unittest.TestCase):
    def test_download_invokes_cli_and_creates_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory) / "raw"
            commands: list[list[str]] = []
            config = DownloadConfig(
                dataset_slug="owner/dataset",
                raw_dir=raw_dir,
                kaggle_username="user",
                kaggle_key="key",
            )

            result = KaggleDatasetDownloader(
                config,
                command_runner=lambda command: commands.append(list(command)),
            ).run()

            self.assertFalse(result.skipped)
            self.assertTrue(config.marker.is_file())
            self.assertEqual(
                commands[0],
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "owner/dataset",
                    "-p",
                    str(raw_dir),
                    "--unzip",
                ],
            )

    def test_existing_marker_skips_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            config = DownloadConfig("owner/dataset", raw_dir)
            config.marker.touch()
            called = False

            def runner(command: object) -> None:
                nonlocal called
                called = True

            result = KaggleDatasetDownloader(config, runner).run()

            self.assertTrue(result.skipped)
            self.assertFalse(called)

    def test_from_env_reads_paths_and_credentials(self) -> None:
        environment = {
            "KAGGLE_DATASET": "owner/custom",
            "RAW_DATA_DIR": "/tmp/raw-data",
            "KAGGLE_USERNAME": "alice",
            "KAGGLE_KEY": "secret",
            "KAGGLE_CONFIG_DIR": "/tmp/kaggle",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = DownloadConfig.from_env()

        self.assertEqual(config.dataset_slug, "owner/custom")
        self.assertEqual(config.raw_dir, Path("/tmp/raw-data"))
        self.assertEqual(config.kaggle_json, Path("/tmp/kaggle/kaggle.json"))
