"""Tools are found under <install root>/tools (OH_STREAMER_PROJECT_ROOT), not only beside the code checkout."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import cast_vsr
from oh_streamer.subtitles import runtime


class InstallRootToolsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.code = self.root / "elsewhere"          # stands in for site-packages / the code dir
        self.code.mkdir()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ("OH_STREAMER_NVENC_C", "OH_STREAMER_FFMPEG"):
            os.environ.pop(key, None)
        os.environ["OH_STREAMER_PROJECT_ROOT"] = str(self.root)

    def test_nvencc_under_install_root(self):
        exe = self.root / "tools" / "nvencc" / "NVEncC64.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("x")
        with mock.patch.object(cast_vsr, "checkout_root", return_value=self.code), \
                mock.patch.object(cast_vsr.sys, "platform", "win32"), \
                mock.patch.object(cast_vsr.shutil, "which", return_value=None):
            self.assertEqual(cast_vsr.executable(), str(exe.resolve()))

    def test_ffmpeg_under_install_root(self):
        exe = self.root / "tools" / "ffmpeg.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("x")
        with mock.patch("oh_streamer.paths.checkout_root", return_value=self.code), \
                mock.patch.object(runtime.shutil, "which", return_value=None):
            self.assertEqual(runtime.resolve_ffmpeg(), str(exe))


if __name__ == "__main__":
    unittest.main()
