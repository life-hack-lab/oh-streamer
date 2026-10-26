import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_streamer.runtime_env import launch_env, layout, resetup


def make_install(root: Path, with_python=True, with_launch=True):
    (root / "app").mkdir(parents=True)
    if with_launch:
        (root / "app" / "launch.pyw").write_text("pass", encoding="utf-8")
    if with_python:
        (root / "python").mkdir()
        (root / "python" / "pythonw.exe").write_text("", encoding="utf-8")


class ResetupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "install"
        self.env = {launch_env.ENV_INSTALL_ROOT: str(self.root)}

    def test_dev_checkout_unavailable_and_spawns_nothing(self):
        info = resetup.info({})
        self.assertFalse(info["available"])
        self.assertTrue(info["note"])
        with mock.patch.object(resetup.subprocess, "Popen") as popen:
            result = resetup.start_setup({})
        popen.assert_not_called()
        self.assertFalse(result["ok"])

    def test_info_shows_current_flavor(self):
        make_install(self.root)
        layout.write_current(self.root, layout.Current(lock_sha256="a" * 64, flavor="nvidia", verified=True,
                                                       updated_at="2026-01-01T00:00:00Z"))
        info = resetup.info(self.env)
        self.assertTrue(info["available"])
        self.assertEqual(info["flavor"], "nvidia")
        self.assertIn("NVIDIA", info["label"])
        self.assertTrue(info["verified"])

    def test_info_without_current_json(self):
        make_install(self.root)
        info = resetup.info(self.env)
        self.assertTrue(info["available"])
        self.assertIsNone(info["flavor"])

    def test_command_uses_embedded_python_and_setup_flag(self):
        make_install(self.root)
        command = resetup.setup_launch_command(self.root)
        self.assertEqual(command, [str(self.root / "python" / "pythonw.exe"),
                                   str(self.root / "app" / "launch.pyw"), launch_env.SETUP_FLAG])

    def test_command_none_without_launcher(self):
        make_install(self.root, with_launch=False)
        self.assertIsNone(resetup.setup_launch_command(self.root))

    def test_start_setup_spawns_detached_without_python_vars(self):
        make_install(self.root)
        env = dict(self.env, PYTHONPATH="x", PYTHONHOME="y")
        with mock.patch.object(resetup.subprocess, "Popen") as popen:
            result = resetup.start_setup(env)
        self.assertTrue(result["ok"])
        args, kwargs = popen.call_args
        self.assertEqual(args[0][-1], launch_env.SETUP_FLAG)
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("PYTHONHOME", kwargs["env"])
        self.assertEqual(kwargs["env"][launch_env.ENV_INSTALL_ROOT], str(self.root))
        if sys.platform == "win32":
            self.assertTrue(kwargs["creationflags"] & 0x00000008)           # DETACHED_PROCESS
            self.assertTrue(kwargs["creationflags"] & resetup._CREATE_BREAKAWAY_FROM_JOB)

    @unittest.skipUnless(sys.platform == "win32", "job breakaway is Windows only")
    def test_breakaway_refused_retries_without_it(self):
        make_install(self.root)
        calls = []

        def fake_popen(*args, **kwargs):
            calls.append(kwargs["creationflags"])
            if len(calls) == 1:
                raise PermissionError("breakaway not allowed")

        with mock.patch.object(resetup.subprocess, "Popen", side_effect=fake_popen):
            result = resetup.start_setup(self.env)
        self.assertTrue(result["ok"])
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[1] & resetup._CREATE_BREAKAWAY_FROM_JOB)

    def test_spawn_failure_reports_error(self):
        make_install(self.root)
        with mock.patch.object(resetup.subprocess, "Popen", side_effect=OSError("boom")):
            result = resetup.start_setup(self.env)
        self.assertFalse(result["ok"])
        self.assertIn("boom", result["error"])

    def test_missing_launcher_does_not_spawn(self):
        make_install(self.root, with_launch=False)
        with mock.patch.object(resetup.subprocess, "Popen") as popen:
            result = resetup.start_setup(self.env)
        popen.assert_not_called()
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
