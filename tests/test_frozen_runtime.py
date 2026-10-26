"""Frozen resource, bundle lookup, and environment defaults for the portable app."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


_STREAMER_ROOT = Path(__file__).resolve().parents[1]
for _path in (_STREAMER_ROOT / "src", _STREAMER_ROOT / "packaging"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import frozen_entry
from oh_streamer import paths


class FrozenPathTests(unittest.TestCase):
    def test_windows_bundle_is_the_upstream_root_but_macos_keeps_parent_lookup(self):
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory, \
                patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "_MEIPASS", directory, create=True), \
                patch.dict(os.environ, {"OH_STREAMER_PROJECT_ROOT": ""}):
            root = Path(directory).resolve()
            self.assertEqual(paths.bundle_root(), root)
            with patch.object(sys, "platform", "win32"):
                self.assertEqual(paths.checkout_root(), root)
                self.assertEqual(paths.upstream_root(), root)
            with patch.object(sys, "platform", "darwin"):
                self.assertEqual(paths.upstream_root(), root.parent)
            configured = root.parent / "developer-checkout"
            with patch.dict(os.environ, {"OH_STREAMER_PROJECT_ROOT": str(configured)}):
                with patch.object(sys, "platform", "win32"):
                    self.assertEqual(paths.upstream_root(), root)
                with patch.object(sys, "platform", "darwin"):
                    self.assertEqual(paths.upstream_root(), configured.resolve())


class FrozenEnvironmentTests(unittest.TestCase):
    def test_windows_prefers_bundled_tools_and_models_and_keeps_ipc_streams(self):
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory:
            root = Path(directory)
            tools = root / "tools"
            mpv = tools / "mpv"
            mpv.mkdir(parents=True)
            (tools / "ffmpeg.exe").touch()
            (tools / "ffprobe.exe").touch()
            (mpv / "mpv.exe").touch()
            (root / "model_weights").mkdir()
            stdout, stderr = sys.stdout, sys.stderr
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "_MEIPASS", str(root), create=True), \
                    patch.object(sys, "platform", "win32"), \
                    patch.dict(os.environ, {"PATH": "system-tools"}):
                os.environ.pop("LADA_MODEL_WEIGHTS_DIR", None)
                frozen_entry.configure_frozen_environment()
                self.assertEqual(os.environ["PATH"].split(os.pathsep),
                                 [str(tools), str(mpv), "system-tools"])
                self.assertEqual(os.environ["LADA_MODEL_WEIGHTS_DIR"], str(root / "model_weights"))
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)

    def test_windows_respects_explicit_model_override(self):
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory, \
                patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "_MEIPASS", directory, create=True), \
                patch.object(sys, "platform", "win32"), \
                patch.dict(os.environ, {"PATH": "system-tools", "LADA_MODEL_WEIGHTS_DIR": "custom-models"}):
            frozen_entry.configure_frozen_environment()
            self.assertEqual(os.environ["LADA_MODEL_WEIGHTS_DIR"], "custom-models")

    def test_macos_external_tool_and_user_data_defaults_are_unchanged(self):
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory:
            data_dir = Path(directory) / "Application Support" / "oh_streamer"
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "_MEIPASS", directory, create=True), \
                    patch.object(sys, "platform", "darwin"), \
                    patch("oh_streamer.platform.platform_support.data_dir", return_value=data_dir), \
                    patch.dict(os.environ, {"PATH": "system-tools"}):
                os.environ.pop("LADA_MODEL_WEIGHTS_DIR", None)
                frozen_entry.configure_frozen_environment()
                self.assertEqual(os.environ["PATH"].split(os.pathsep),
                                 ["/opt/homebrew/bin", "/usr/local/bin", "system-tools"])
                self.assertEqual(os.environ["LADA_MODEL_WEIGHTS_DIR"], str(data_dir / "model_weights"))


if __name__ == "__main__":
    unittest.main()
