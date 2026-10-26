"""Device decisions and IPC framing without requiring a Mac GPU or desktop."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import functools
import io
import json
import types
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch

import oh_streamer.processing.device_backend as D
import oh_streamer.player.mpv_ipc as M
import oh_streamer.player.player_settings as P
import oh_streamer.queue.queue_worker as Q
import oh_streamer.queue.queue_store as queue_store
import oh_streamer.queue.queue_service as queue_service
import oh_streamer.queue.queue_app as queue_app


def torch_fake(cuda=False, mps=False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 2,
                                   empty_cache=Mock(), memory_allocated=Mock(return_value=2**30)),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        mps=types.SimpleNamespace(empty_cache=Mock(), current_allocated_memory=Mock(return_value=2 * 2**30)),
        device=lambda name: types.SimpleNamespace(type=name.split(":")[0], name=name))


class DeviceTests(unittest.TestCase):
    def test_auto_cuda_mps_cpu_and_explicit_cpu(self):
        for cuda, mps, expected in [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")]:
            t = torch_fake(cuda, mps)
            self.assertEqual(D.select_device(torch_module=t).name, expected)
            self.assertEqual(D.select_device("cpu", torch_module=t).name, "cpu")

    def test_unavailable_explicit_device_is_an_error(self):
        t = torch_fake()
        for name in ("mps", "cuda:0"):
            with self.assertRaises(RuntimeError):
                D.select_device(name, torch_module=t)
        with self.assertRaises(RuntimeError):
            D.select_device("cuda:2", torch_module=torch_fake(cuda=True))
        for name in ("metal", "cuda:-1", None, "mps:0"):
            with self.assertRaises(ValueError):
                D.validate_device(name)

    def test_memory_calls_only_selected_backend(self):
        t = torch_fake(True, True)
        dev = D.select_device("mps", torch_module=t)
        D.empty_cache(dev, torch_module=t)
        t.mps.empty_cache.assert_called_once()
        t.cuda.empty_cache.assert_not_called()
        self.assertEqual(D.allocated_gib(dev, torch_module=t), 2)
        self.assertIsNone(D.allocated_gib(D.select_device("cpu", torch_module=t), torch_module=t))
        with patch.object(D.sys, "platform", "darwin"):
            self.assertFalse(D.rife_available(D.select_device("cuda", torch_module=t)))

    def test_mac_settings_gates_do_not_rewrite_saved_values(self):
        saved = {"encoder": "h264_nvenc", "vsr": "on", "interpolation": "auto"}
        cfg, notes = P.fix_platform(P.effective(saved, {}), "darwin")
        self.assertEqual(cfg["encoder"], "auto")
        self.assertEqual(cfg["vsr"], "off")
        self.assertEqual(cfg["interpolation"], "off")
        self.assertEqual(saved["vsr"], "on")
        self.assertEqual(len(notes), 3)
        menu = P.build_menu(cfg, True, ["v4-fast"], ["basicvsrpp-v1.2"], False,
                            "h264_videotoolbox", videotoolbox_ok=True, platform="darwin")
        by_key = {row["key"]: row for row in menu}
        self.assertEqual([o["v"] for o in by_key["vsr"]["options"]], ["off"])
        self.assertNotIn("h264_nvenc", [o["v"] for o in by_key["encoder"]["options"]])
        self.assertEqual(P.parse_request('1:{"vsr":"on","interpolation":"auto"}', menu), {})

    def test_worker_validates_forwarded_device(self):
        job = {"source": "input.mp4", "output": "out.mkv", "work_dir": "work", "settings": {"device": "mps"}}
        normalized, _ = Q.normalize_job(job)
        self.assertEqual(normalized["settings"]["device"], "mps")
        job["settings"]["device"] = "bogus"
        with self.assertRaises(Q.WorkerError):
            Q.normalize_job(job)

    def test_explicit_device_separates_cache_and_resume_key(self):
        settings = {"device": "cpu", "fp16": False}
        cpu_key = queue_store.cache_key("video.mp4", None, settings)
        mps_key = queue_store.cache_key("video.mp4", None, {**settings, "device": "mps"})
        self.assertNotEqual(cpu_key, mps_key)
        self.assertEqual(queue_store.cache_key("video.mp4", None, {}),
                         queue_store.cache_key("video.mp4", None, {"device": "auto"}))
        job, _ = Q.normalize_job({"source": "input.mp4", "output": "out.mkv", "work_dir": "work",
                                  "settings": settings})
        info = types.SimpleNamespace(fps=30, frames=120, width=128, height=96, video_start=0)
        plan = Q.SegmentPlan(info.fps, info.frames, 0, 1)
        first = Q.job_key(job, info, plan, "libx264")
        job["settings"]["device"] = "mps"
        self.assertNotEqual(first, Q.job_key(job, info, plan, "libx264"))

    def test_mac_queue_and_explicit_cpu_gate_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            svc = queue_service.QueueService(Path(directory))
            item = queue_store.Item(id="test", target="input.mp4", settings={"device": "cpu"})
            self.assertEqual(svc._build_job(item)["settings"]["interpolation"], "off")
            with patch.object(queue_service.platform_support.sys, "platform", "darwin"):
                svc.set_settings({"interpolation": "full"})
                self.assertEqual(svc.get_state()["settings"]["interpolation"], "off")
                self.assertEqual(svc._build_job(item)["settings"]["interpolation"], "off")


class TrayFallbackTest(unittest.TestCase):
    def test_hidden_window_is_made_visible_before_gui_loop(self):
        window = Mock()
        window.events = types.SimpleNamespace(closing=MagicMock(), loaded=MagicMock())
        # Mock Window.show() would hide the real pre-start blocking problem;
        # make it fail so startup must configure .hidden instead.
        window.show.side_effect = AssertionError("GUI is not running yet")
        webview = types.SimpleNamespace(create_window=Mock(return_value=window), start=Mock())
        svc = Mock()
        svc.start.return_value = True
        svc.on_state, svc.on_show = [], []
        # main() arms a real 45 s timer; left enabled it pops the WebView2 error box later in the test run.
        quiet_watchdog = functools.partial(queue_app.WebviewWatchdog, enabled=False)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"webview": webview}), \
                patch.object(queue_app, "WebviewWatchdog", quiet_watchdog), \
                patch.object(queue_service, "QueueService", return_value=svc), \
                patch.object(queue_app.QueueApp, "start_tray", side_effect=RuntimeError("no tray")), \
                patch.object(queue_app, "kill_children_on_exit", return_value=False), \
                patch.object(queue_app, "clear_web_cache"), \
                patch.object(queue_app.sys, "argv", ["queue_app.py", "--root", directory, "--hidden"]):
            self.assertEqual(queue_app.main(), 0)
        self.assertFalse(window.hidden)
        window.show.assert_not_called()
        webview.start.assert_called_once()


class IpcTests(unittest.TestCase):
    def test_unix_connect_sendall_event_and_split_response(self):
        fake_socket = Mock()
        fake_socket.makefile.return_value = io.BytesIO(
            b'{"event":"start-file"}\n{"request_id":1,"error":"success","data":12}\n')
        with patch.object(M.sys, "platform", "darwin"), patch.object(M.socket, "AF_UNIX", 1, create=True), \
                patch.object(M.socket, "socket", return_value=fake_socket):
            ipc = M.MpvIpc("/tmp/test.sock", Mock())
            self.assertEqual(ipc.get("time-pos"), 12)
            fake_socket.connect.assert_called_once_with("/tmp/test.sock")
            sent = json.loads(fake_socket.sendall.call_args.args[0])
            self.assertEqual(sent["command"], ["get_property", "time-pos"])
            ipc.close()
            fake_socket.close.assert_called_once()

    def test_windows_pipe_and_default_on_property_error(self):
        file = Mock()
        file.read.return_value = b'{"request_id":1,"error":"property unavailable"}\n'
        with patch.object(M.sys, "platform", "win32"), patch("builtins.open", return_value=file):
            ipc = M.MpvIpc(r"\\.\pipe\test", Mock())
            self.assertEqual(ipc.get("time-pos", 99), 99)
            file.write.assert_called_once()
            ipc.close()

    def test_failed_connection_closes_socket(self):
        sock = Mock()
        sock.connect.side_effect = OSError("not ready")
        player = Mock()
        player.poll.return_value = 1
        with patch.object(M.sys, "platform", "darwin"), patch.object(M.socket, "AF_UNIX", 1, create=True), \
                patch.object(M.socket, "socket", return_value=sock):
            with self.assertRaises(RuntimeError):
                M.MpvIpc("missing", player)
        sock.close.assert_called_once()

    def test_endpoint_private_short_path_and_cleanup(self):
        directory = Mock(name="temporary directory")
        directory.name = "/tmp/lada-test"
        with patch.object(M.sys, "platform", "darwin"), \
                patch.object(M.tempfile, "TemporaryDirectory", return_value=directory) as factory:
            endpoint = M.MpvEndpoint()
            self.assertLess(len(endpoint.path.encode()), 104)
            self.assertEqual(factory.call_args.kwargs["dir"], "/tmp")
            endpoint.close()
            directory.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
