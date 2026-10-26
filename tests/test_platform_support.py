"""OS decisions with no clipboard, shutdown or caffeinate side effects."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import oh_streamer.platform.platform_support as ps
import oh_streamer.queue.queue_app as queue_app
import oh_streamer.queue.queue_service as queue_service


class PlatformTests(unittest.TestCase):
    def test_windows_locations_and_gui(self):
        with patch.object(sys, "platform", "win32"), patch.dict(ps.os.environ, {"LOCALAPPDATA": "C:/data"}):
            self.assertEqual(ps.data_dir(), Path("C:/data/oh_streamer"))
            self.assertEqual(ps.state_dir(), ps.data_dir())
            self.assertEqual(ps.gui_engine(), "edgechromium")
            self.assertTrue(ps.supports_shutdown())

    def test_mac_locations_and_gui(self):
        with patch.object(sys, "platform", "darwin"), patch.object(Path, "home", return_value=Path("/Users/test")):
            self.assertEqual(ps.data_dir(), Path("/Users/test/Library/Application Support/oh_streamer"))
            self.assertEqual(ps.gui_engine(), "cocoa")
            self.assertFalse(ps.supports_shutdown())
            self.assertEqual(ps.detached_process_kwargs(), {"start_new_session": True})

    def test_linux_locations_and_cache(self):
        env = {"XDG_DATA_HOME": "/xdg/data", "XDG_CACHE_HOME": "/xdg/cache"}
        with patch.object(sys, "platform", "linux"), patch.dict(ps.os.environ, env):
            self.assertEqual(ps.data_dir(), Path("/xdg/data/oh_streamer"))
            self.assertEqual(ps.state_dir(), ps.data_dir())
            self.assertEqual(ps.cache_dir(), Path("/xdg/cache/oh_streamer"))

    def test_cache_dir_per_platform(self):
        with patch.object(sys, "platform", "win32"), patch.dict(ps.os.environ, {"LOCALAPPDATA": "C:/data"}):
            self.assertEqual(ps.cache_dir(), Path("C:/data/oh_streamer/cache"))
        with patch.object(sys, "platform", "darwin"), patch.object(Path, "home", return_value=Path("/Users/test")):
            self.assertEqual(ps.cache_dir(), Path("/Users/test/Library/Caches/oh_streamer"))

    def test_windows_state_dir_falls_back_to_tempdir(self):
        with patch.object(sys, "platform", "win32"), patch.dict(ps.os.environ, {"LOCALAPPDATA": ""}), \
                patch.object(ps.tempfile, "gettempdir", return_value="C:/tmp"):
            self.assertEqual(ps.state_dir(), Path("C:/tmp/oh_streamer"))

    def test_caffeinate_is_reused_and_cancelled(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(sys, "platform", "darwin"), patch.object(ps.subprocess, "Popen", return_value=proc) as popen:
            guard = ps.PreventSleep()
            guard.set_active(True)
            guard.set_active(True)
            popen.assert_called_once()
            self.assertEqual(popen.call_args.args[0][:3], ["/usr/bin/caffeinate", "-i", "-w"])
            guard.set_active(False)
            guard.close()
            proc.terminate.assert_called_once()

    def test_open_mac_folder_uses_argv(self):
        with patch.object(sys, "platform", "darwin"), patch.object(ps.subprocess, "Popen") as popen:
            ps.open_folder("/Users/test/a ; b")
            popen.assert_called_once_with(["/usr/bin/open", "/Users/test/a ; b"])

    def test_unsupported_shutdown_settings_and_invocation_are_gated(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ps, "supports_shutdown", return_value=False):
            root = Path(tmp)
            store = queue_service.queue_store.QueueStore(root)
            store.set_settings(shutdown_when_done=True)
            shutdown = Mock()
            svc = queue_service.QueueService(root, run_shutdown=shutdown)
            self.assertFalse(svc.store.get_settings()["shutdown_when_done"])
            self.assertFalse(svc.set_settings({"shutdown_when_done": True})["settings"]["shutdown_when_done"])
            svc._active_seen = True
            svc._check_shutdown()
            svc.cancel_shutdown()
            shutdown.assert_not_called()
            self.assertFalse(svc.get_state()["capabilities"]["shutdown"])
            with patch.object(queue_service.subprocess, "run") as run:
                queue_service._default_run_shutdown(["shutdown"])
                run.assert_not_called()

    def test_tray_failure_allows_normal_window_close(self):
        app = queue_app.QueueApp(Mock(), hidden=True)
        app.window = Mock()
        self.assertTrue(app.on_closing())
        app.window.hide.assert_not_called()

    def test_cocoa_tray_shares_main_application(self):
        tray = Mock()
        appkit = types.SimpleNamespace(NSApplication=Mock())
        native_app = appkit.NSApplication.sharedApplication.return_value
        with patch.object(sys, "platform", "darwin"), patch.dict(sys.modules, {"pystray": tray, "AppKit": appkit}), \
                patch.object(queue_app, "make_icon_image", return_value=Mock()):
            app = queue_app.QueueApp(Mock(), hidden=False)
            app.start_tray()
            self.assertIs(tray.Icon.call_args.kwargs["darwin_nsapplication"], native_app)
            tray.Icon.return_value.run_detached.assert_called_once_with()

    def test_failed_sleep_assertion_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            assertion = Mock(side_effect=[OSError("unavailable"), None])
            svc = queue_service.QueueService(Path(tmp), set_execution_state=assertion)
            svc._dl = Mock()
            svc._update_sleep()
            self.assertFalse(svc._sleep_on)
            svc._update_sleep()
            self.assertTrue(svc._sleep_on)
            self.assertEqual(assertion.call_count, 2)


if __name__ == "__main__":
    unittest.main()
