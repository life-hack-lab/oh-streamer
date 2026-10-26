"""Checks for the git identity recorded in the installer build manifest."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
STREAMER_ROOT = PACKAGING_DIR.parent
sys.path.insert(0, str(PACKAGING_DIR))
import build_provenance  # noqa: E402


class BuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=STREAMER_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_git_status_failure_is_unknown_not_clean(self):
        with patch.object(build_provenance, "_run_git",
                          side_effect=[str(self.root.resolve()), "a" * 40, None]):
            identity = build_provenance.git_identity(self.root)
        self.assertEqual(identity["revision"], "a" * 40)
        self.assertIsNone(identity["dirty"])

    def test_nested_non_checkout_does_not_inherit_parent_git_identity(self):
        parent = self.root / "parent-checkout"
        nested_source = parent / ".venv-windows-release" / "rife"
        nested_source.mkdir(parents=True)
        with patch.object(build_provenance, "_run_git",
                          side_effect=[str(parent.resolve())]) as run_git:
            identity = build_provenance.git_identity(nested_source)
        self.assertEqual(identity, {"revision": None, "dirty": None, "exact_tags": []})
        self.assertEqual(run_git.call_count, 1)


if __name__ == "__main__":
    unittest.main()
