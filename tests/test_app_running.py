import re
import sys
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.runtime_env import app_running

SRC = Path(__file__).resolve().parents[1] / "src" / "oh_streamer"
ENTRIES = {
    "queue/queue_app.py": "def main() -> int:",
    "queue/queue_worker.py": "def main(argv=None) -> int:",
    "player/oh_player.py": "def main():",
    "setup_mode/__main__.py": "def main() -> int:",
    "perf/__main__.py": "def main(argv: list[str] | None = None) -> int:",
}


class AppMutexTest(unittest.TestCase):
    def test_name(self):
        self.assertEqual(app_running.APP_MUTEX_NAME, "oh_streamer-app")

    def test_non_windows_noop(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertFalse(app_running.hold_app_mutex())
            self.assertFalse(app_running.app_mutex_exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_held_mutex_is_visible_from_another_handle(self):
        import subprocess
        code = ("import sys,time;sys.path.insert(0,%r);"
                "from oh_streamer.runtime_env.app_running import hold_app_mutex;"
                "print(hold_app_mutex(),flush=True);time.sleep(30)") % str(SRC.parent)
        base = getattr(sys, "_base_executable", sys.executable)
        proc = subprocess.Popen([base, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "True")
            self.assertTrue(app_running.app_mutex_exists())
        finally:
            proc.kill()
            proc.wait()
        self.assertFalse(app_running.app_mutex_exists())   # released with the process

    def test_every_entry_calls_it(self):
        for rel, signature in ENTRIES.items():
            text = (SRC / rel).read_text(encoding="utf-8")
            i = text.index(signature)
            self.assertIn("hold_app_mutex()", text[i:i + 700], rel)
        launcher = (SRC.parents[1] / "launch.pyw").read_text(encoding="utf-8")
        self.assertIn("hold_app_mutex()", launcher)


if __name__ == "__main__":
    unittest.main()
