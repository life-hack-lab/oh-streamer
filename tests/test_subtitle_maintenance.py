"""Validate the remaining subtitle reproduction entry points and packaged resources."""
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SubtitleMaintenanceTests(unittest.TestCase):
    def test_subtitle_package_data_is_valid_and_aligned_with_existing_resources(self):
        text = (ROOT / "pyproject.toml").read_text("utf-8")
        files = tomllib.loads(text)["tool"]["setuptools"]["package-data"]["oh_streamer"]
        for name in files:
            if name.startswith("subtitles/"):
                self.assertTrue((ROOT / "src/oh_streamer" / name).is_file())
                self.assertIn(f'\n    "{name}",', text)


if __name__ == "__main__":
    unittest.main()
