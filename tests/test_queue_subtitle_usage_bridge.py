"""Bridge for the Groq usage snapshot: Api.subtitle_usage -> service -> usage_monitor.usage_snapshot.
It never raises: any failure resolves {version:1, providers:[], error}. Run: python -m unittest tests.test_queue_subtitle_usage_bridge."""
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.queue import queue_app
from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin
from oh_streamer.subtitles import usage_monitor as um


class UsageBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        um.set_db_path(Path(self.tmp.name) / "u.sqlite3")

    def api(self, svc):
        app = types.SimpleNamespace(service=svc)
        return queue_app.Api(app)

    def test_snapshot_flows_through_api_and_service(self):
        snap = self.api(SubtitleQueueMixin()).subtitle_usage()
        self.assertEqual(snap["version"], 1)
        self.assertEqual(snap["providers"], [])
        self.assertNotIn("error", snap)

    def test_service_error_becomes_error_dict(self):
        with mock.patch.object(um, "usage_snapshot", side_effect=RuntimeError("boom")):
            snap = SubtitleQueueMixin().subtitle_usage()
        self.assertEqual((snap["version"], snap["providers"]), (1, []))
        self.assertIn("boom", snap["error"])

    def test_api_error_becomes_error_dict(self):
        svc = types.SimpleNamespace(subtitle_usage=mock.Mock(side_effect=ValueError("bad")))
        snap = self.api(svc).subtitle_usage()
        self.assertEqual((snap["version"], snap["providers"]), (1, []))
        self.assertIn("bad", snap["error"])


if __name__ == "__main__":
    unittest.main()
