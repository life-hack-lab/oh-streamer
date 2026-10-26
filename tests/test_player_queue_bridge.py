"""Focused tests for the bounded player-to-queue bridge (no player engine or models imported)."""
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import unittest
from unittest import mock

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
for _root in (_STREAMER_ROOT, _SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from oh_streamer.player import queue_bridge as QB  # noqa: E402
from oh_streamer.queue import queue_ipc  # noqa: E402


class QueueLinkBoundedTest(unittest.TestCase):
    def test_post_reports_backpressure_at_the_bounded_capacity(self):
        entered_send = threading.Event()
        release_send = threading.Event()
        last_send = threading.Event()
        sent = []

        def send(payload, _timeout):
            sent.append(payload["n"])
            if payload["n"] == 0:
                entered_send.set()
                release_send.wait(3)
            if payload["n"] == 64:
                last_send.set()
            return {"ok": True}

        link = QB.QueueLink(send=send)
        self.assertTrue(link.post({"cmd": "test", "n": 0}))
        self.assertTrue(entered_send.wait(2))
        for n in range(1, 65):
            self.assertTrue(link.post({"cmd": "test", "n": n}))
        self.assertEqual(link._q.maxsize, 64)
        self.assertEqual(link._q.qsize(), 64)
        self.assertFalse(link.post({"cmd": "test", "n": 65}))

        release_send.set()
        self.assertTrue(last_send.wait(3))
        link.close(timeout=3)
        self.assertEqual(sent, list(range(65)))
        self.assertFalse(link._thread.is_alive())


class QueueBridgeImportTest(unittest.TestCase):
    def test_defaults_keep_the_bound_probe_function_identity(self):
        self.assertIs(QB.processed_open.__defaults__[0], QB.probe_compare_setup)
        self.assertIs(QB.processed_from_message.__defaults__[0], QB.probe_compare_setup)
        self.assertEqual(QB.queue_lookup.__defaults__[0], 1.0)
        self.assertIs(QB.queue_lookup.__defaults__[1], QB.probe_compare_setup)
        self.assertEqual(QB.BusyReporter(lambda _busy: None).interval, queue_ipc.BUSY_HEARTBEAT)

    def test_combos_dir_travels_with_processed_message_and_lookup(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "output.mkv")
            open(video, "wb").close()
            probe = lambda p: ("", "", None)  # noqa: E731
            po = QB.processed_from_message({"input": video, "processed": True, "queue_id": "q1",
                                            "subtitle_combos_dir": os.path.join(d, "subtitles", "combos")}, probe)
            self.assertEqual(po.combos_dir, os.path.join(d, "subtitles", "combos"))
            self.assertIsNone(QB.processed_from_message({"input": video, "processed": True}, probe).combos_dir)
            self.assertIsNone(QB.processed_from_message({"input": video, "processed": True,
                                                         "subtitle_combos_dir": 5}, probe).combos_dir)
            link = QB.QueueLink(send=lambda payload, timeout: {"ok": True, "item": {
                "id": "q1", "state": "done", "output": video, "subtitle_combos_dir": d}})
            self.assertEqual(QB.queue_lookup(link, "x", None, probe_layers=probe).combos_dir, d)

    def test_leaf_import_does_not_import_the_player_facade(self):
        code = (f"import sys; sys.path.insert(0, {str(_SOURCE_ROOT)!r}); "
                "import oh_streamer.player.queue_bridge; "
                "assert 'oh_streamer.player.oh_player' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_default_queue_send_uses_the_shared_protocol_module(self):
        with mock.patch.object(queue_ipc, "send_to_queue", return_value={"ok": True}) as send:
            QB.QueueLink(name="player-a").request({"cmd": "show"}, 1.25)
        send.assert_called_once_with({"cmd": "show"}, "player-a", timeout=1.25)

    def test_subtitle_update_accepts_cumulative_path_list_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / "zh.srt", Path(folder) / "ja.srt"]
            for path in paths:
                path.write_text("1\n00:00:01,000 --> 00:00:02,000\n字幕\n", encoding="utf-8")
            tracks = [
                {"path": str(paths[0]), "lang": "chi", "title": "繁體中文（AI）", "default": True},
                {"path": str(paths[1]), "lang": "jpn", "title": "日文（AI）", "default": False},
            ]
            update = QB.subtitle_update_from_message({
                "cmd": "subtitle_update", "item_id": "q7", "srt": [str(path) for path in paths],
                "covered_s": 600.0, "complete": False, "subtitles": tracks, "subtitle_offset": 1.25,
            })

            self.assertEqual(update.item_id, "q7")
            self.assertEqual(update.subtitles, tracks)
            self.assertEqual(update.covered_s, 600.0)
            self.assertFalse(update.complete)
            self.assertEqual(update.subtitle_offset, 1.25)

    def test_subtitle_update_rejects_invalid_coverage_or_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zh.srt"
            path.write_text("", encoding="utf-8")
            base = {"cmd": "subtitle_update", "item_id": "q7", "srt": str(path),
                    "covered_s": 3.0, "complete": False}
            self.assertIsNotNone(QB.subtitle_update_from_message(base))
            for changes in (
                {"covered_s": True}, {"covered_s": -1}, {"covered_s": float("inf")},
                {"complete": 0}, {"item_id": ""}, {"srt": str(Path(folder) / "missing.srt")},
                {"srt": [str(path), str(Path(folder) / "missing.srt")]}, {"subtitle_offset": float("nan")},
            ):
                self.assertIsNone(QB.subtitle_update_from_message({**base, **changes}), changes)

    def test_subtitle_update_validates_and_normalizes_island_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zh.srt"
            path.write_text("", encoding="utf-8")
            base = {"cmd": "subtitle_update", "item_id": "q7", "srt": str(path),
                    "covered_s": 30.0, "complete": False}
            self.assertIsNone(QB.subtitle_update_from_message(base).covered_ranges)
            self.assertEqual(QB.subtitle_update_from_message({**base, "covered_ranges": []}).covered_ranges, [])
            update = QB.subtitle_update_from_message({**base, "covered_ranges": [[2390, 2420], [0, 30], [2410, 2450]]})
            self.assertEqual(update.covered_ranges, [[0.0, 30.0], [2390.0, 2450.0]])
            for invalid in ({}, "0-30", [0, 30], [[0]], [[0, 1, 2]], [[-1, 30]], [[30, 0]],
                            [[True, 30]], [[0, float("nan")]], [[0, float("inf")]], [[0, "30"]]):
                with self.subTest(invalid=invalid):
                    self.assertIsNone(QB.subtitle_update_from_message({**base, "covered_ranges": invalid}))

    def test_processed_open_carries_partial_subtitle_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "output.mkv"
            video.write_bytes(b"video")
            po = QB.processed_from_message({
                "input": str(video), "processed": True, "queue_id": "q8",
                "subtitle_covered_s": 41.0, "subtitle_complete": False,
                "subtitle_covered_ranges": [[0, 41], [2390, 2420]],
            }, probe_layers=lambda _path: ("", "", None))

            self.assertEqual(po.queue_id, "q8")
            self.assertEqual(po.subtitle_covered_s, 41.0)
            self.assertEqual(po.subtitle_covered_ranges, [[0.0, 41.0], [2390.0, 2420.0]])
            self.assertFalse(po.subtitle_complete)


if __name__ == "__main__":
    unittest.main()
