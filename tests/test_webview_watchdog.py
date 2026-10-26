import logging
import tempfile
import threading
import unittest
from pathlib import Path

from oh_streamer.branding import APP_NAME
from oh_streamer.platform import webview_watchdog as wd


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def fire(self):
        for fn in list(self.handlers):
            fn()


class _Events:
    def __init__(self):
        self.loaded = _Event()


class _Window:
    def __init__(self):
        self.events = _Events()
        self.destroyed = threading.Event()

    def destroy(self):
        self.destroyed.set()


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = Path(self.tmp.name) / "logs"
        self.dialogs = []
        self.window = _Window()

    def make(self, **kw):
        self.answer = kw.pop("answer", False)
        self.opened = []
        dog = wd.WebviewWatchdog(APP_NAME, self.logs, timeout=kw.pop("timeout", 0.15), enabled=True,
                                 show_dialog=self._dialog, open_url=self.opened.append, **kw)
        self.addCleanup(logging.getLogger("pywebview").removeHandler, dog._handler)
        return dog

    def _dialog(self, title, text):
        self.dialogs.append((title, text))
        return self.answer

    def test_timeout_logs_dialogs_and_destroys(self):
        dog = self.make()
        dog.attach_logging()
        dog.watch(self.window)
        dog.arm(self.window)
        logging.getLogger("pywebview").error("WebView2 initialization failed with exception: boom")
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertTrue(dog.fired)
        files = list(self.logs.glob("webview-*.log"))
        self.assertEqual(len(files), 1)
        self.assertIn("boom", files[0].read_text(encoding="utf-8"))
        title, text = self.dialogs[0]
        self.assertEqual(title, f"{APP_NAME} 無法顯示畫面")
        self.assertIn(wd.WEBVIEW2_DOWNLOAD_URL, text)
        self.assertIn(str(files[0]), text)
        self.assertTrue(text.endswith("要現在開啟 WebView2 下載頁面嗎？"))

    def test_pywebview_error_fires_immediately(self):
        dog = self.make(timeout=60)
        dog.attach_logging()
        dog.watch(self.window)
        dog.arm(self.window)
        logging.getLogger("pywebview").error("WebView2 initialization failed with exception: boom")
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertEqual(len(self.dialogs), 1)
        self.assertIn("boom", list(self.logs.glob("webview-*.log"))[0].read_text(encoding="utf-8"))

    def test_other_pywebview_records_do_not_fire(self):
        dog = self.make(timeout=60)
        dog.attach_logging()
        dog.arm(self.window)
        logging.getLogger("pywebview").error("something else")
        self.assertFalse(self.window.destroyed.wait(0.4))
        dog.disarm()

    def test_immediate_and_timer_fire_once(self):
        dog = self.make(timeout=0.05)
        dog.attach_logging()
        dog.arm(self.window)
        logging.getLogger("pywebview").error("WebView2 initialization failed with exception: x")
        self.assertTrue(self.window.destroyed.wait(3))
        threading.Event().wait(0.4)
        self.assertEqual(len(self.dialogs), 1)

    def test_yes_opens_download_page(self):
        dog = self.make(answer=True)
        dog.arm(self.window)
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertEqual(self.opened, [wd.WEBVIEW2_DOWNLOAD_URL])

    def test_no_does_not_open(self):
        dog = self.make(answer=False)
        dog.arm(self.window)
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertEqual(self.opened, [])

    def test_before_destroy_runs_first(self):
        order = []
        dog = self.make(before_destroy=lambda: order.append("before"))
        self.window.destroy = lambda: (order.append("destroy"), self.window.destroyed.set())
        dog.arm(self.window)
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertEqual(order, ["before", "destroy"])

    def test_loaded_cancels(self):
        dog = self.make()
        dog.watch(self.window)
        dog.arm(self.window)
        self.window.events.loaded.fire()
        self.assertFalse(self.window.destroyed.wait(0.5))
        self.assertFalse(dog.fired)
        self.assertEqual(self.dialogs, [])
        self.assertFalse(self.logs.exists())

    def test_loaded_before_arm_keeps_it_off(self):
        dog = self.make()
        dog.watch(self.window)
        self.window.events.loaded.fire()
        dog.arm(self.window)
        self.assertFalse(self.window.destroyed.wait(0.5))

    def test_arm_twice_fires_once(self):
        dog = self.make()
        dog.arm(self.window)
        dog.arm(self.window)
        self.assertTrue(self.window.destroyed.wait(3))
        self.window.destroyed.clear()
        self.assertFalse(self.window.destroyed.wait(0.4))
        self.assertEqual(len(self.dialogs), 1)

    def test_disabled_is_noop(self):
        dog = wd.WebviewWatchdog(APP_NAME, self.logs, timeout=0.05, enabled=False,
                                 show_dialog=lambda t, x: self.dialogs.append((t, x)))
        dog.watch(self.window)
        dog.arm(self.window)
        self.assertFalse(self.window.destroyed.wait(0.3))
        self.assertEqual(self.dialogs, [])

    def test_unwritable_log_dir_still_dialogs(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        dog = wd.WebviewWatchdog(APP_NAME, blocker / "logs", timeout=0.1, enabled=True,
                                 show_dialog=lambda t, x: self.dialogs.append((t, x)))
        dog.arm(self.window)
        self.assertTrue(self.window.destroyed.wait(3))
        self.assertEqual(len(self.dialogs), 1)


if __name__ == "__main__":
    unittest.main()
