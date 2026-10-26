"""Unified entry streaming uses player IPC without inserting queue jobs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.queue.queue_service import QueueService
from oh_streamer.branding import APP_NAME
from oh_streamer.queue.queue_app import Api, WINDOW_TITLE


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.sent, self.launched, self.shutdowns = [], [], []
        self.reply = {"ok": True}
        self.svc = QueueService(
            Path(self.tmp.name) / "queue", clock=lambda: self.now,
            send_to_player=lambda p: self.sent.append(p) or self.reply,
            launch_player=lambda path, payload: self.launched.append((path, payload)),
            run_shutdown=self.shutdowns.append)

    def tearDown(self):
        self.tmp.cleanup()

    def test_curl_headers_and_no_queue_job(self):
        r = self.svc.stream("curl 'https://example.test/video' -H 'Cookie: secret' -H 'Referer: https://example.test/'")
        self.assertTrue(r["ok"])
        self.assertEqual(self.sent[0], {"cmd": "open", "input": "https://example.test/video",
                                     "headers": {"Cookie": "secret", "Referer": "https://example.test/"}})
        self.assertEqual(self.svc.store.items(), [])
        self.assertTrue(self.svc.is_player_busy())

    def test_launch_and_heartbeat_guard(self):
        self.reply = None
        self.assertTrue(self.svc.stream("https://example.test/video")["launched"])
        self.assertEqual(self.launched[0][1], self.sent[0])
        self.svc._on_message({"cmd": "player_busy", "busy": False})
        self.assertTrue(self.svc.is_player_busy())
        self.svc._on_message({"cmd": "player_busy", "busy": True})
        self.svc._on_message({"cmd": "player_busy", "busy": False})
        self.assertFalse(self.svc.is_player_busy())

    def test_failed_launch_and_rejected_forward_clear_guard(self):
        self.reply = {"ok": False, "error": "bad source"}
        self.assertEqual(self.svc.stream("https://example.test/video")["error"], "bad source")
        self.assertFalse(self.svc.is_player_busy())
        self.reply = None
        with patch.object(self.svc, "_launch_player", side_effect=OSError("launch failed")):
            self.assertFalse(self.svc.stream("https://example.test/video")["ok"])
        self.assertFalse(self.svc.is_player_busy())

    def test_invalid_and_missing_file(self):
        self.assertFalse(self.svc.stream("")["ok"])
        self.assertFalse(self.svc.stream(str(Path(self.tmp.name) / "missing.mp4"))["ok"])
        self.assertEqual(self.sent, [])
        clip = Path(self.tmp.name) / "clip.mp4"
        clip.write_bytes(b"test")
        self.assertTrue(self.svc.stream(str(clip))["ok"])

    def test_shutdown_cancel_and_playback_suppression(self):
        with patch("oh_streamer.queue.queue_service.platform_support.supports_shutdown", return_value=True):
            self.svc._shutdown_started = True
            self.assertTrue(self.svc.stream("https://example.test/video")["ok"])
            self.assertEqual(self.shutdowns, [["shutdown", "/a"]])
            self.svc.store.set_settings(shutdown_when_done=True)
            item = self.svc.store.add("https://example.test/finished", "url", {}, None, {})
            self.svc.store.update(item.id, state="done")
            self.svc._active_seen = True
            self.svc._check_shutdown()
            self.assertEqual(len(self.shutdowns), 1)
            self.now += 121
            self.svc._on_message({"cmd": "player_playing", "playing": True})
            self.svc._check_shutdown()
            self.assertEqual(len(self.shutdowns), 1)

    def test_api_and_title(self):
        app = type("App", (), {"service": self.svc})()
        self.assertTrue(Api(app).stream("https://example.test/video")["ok"])
        self.assertEqual(WINDOW_TITLE, APP_NAME)

    def test_failed_shutdown_cancel_restores_guard(self):
        self.svc._shutdown_started = True
        with patch.object(self.svc, "cancel_shutdown", return_value={"ok": False, "error": "abort failed"}):
            self.assertFalse(self.svc.stream("https://example.test/video")["ok"])
        self.assertFalse(self.svc.is_player_busy())
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
