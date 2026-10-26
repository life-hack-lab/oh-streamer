"""Tests for the T1.3 timing log: queue/press_timing.py (按播放 → 第一段完成) and the queue reader that feeds it.

Pure: a fake clock and fake wall time; the reader is run unbound on a stub (no QueueService, no worker, no GPU).
Run: python -P streamer/tests/test_press_timing.py (CUDA_VISIBLE_DEVICES="" with Lada's venv)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import types
import unittest
from unittest.mock import patch

import oh_streamer.queue.queue_service as QS
from oh_streamer.queue.press_timing import PressTiming


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class PressTimingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.wall = FakeClock(1_700_000_000.0)
        self.timing = PressTiming(clock=self.clock, wall=self.wall)

    def test_full_line_with_worker_start_and_models(self):
        t = self.timing
        t.press("q1")                       # 按播放 at monotonic 100
        self.clock.now = 100.5
        t.spawned("q1")                     # worker started 0.5 s later
        t.models_loaded("q1", 2.5)
        self.clock.now = 104.0
        self.assertEqual(t.segment_done("q1"),
                         "first playable q1: 按播放→第一段完成 4.0 s（worker 啟動 0.5 s，模型載入 2.5 s）")

    def test_one_line_per_press(self):
        t = self.timing
        t.press("q1")
        self.clock.now += 3
        self.assertIsNotNone(t.segment_done("q1"))
        self.clock.now += 1
        self.assertIsNone(t.segment_done("q1"))          # the second segment is not logged again

    def test_repress_restarts_the_timer(self):
        t = self.timing
        t.press("q1")
        self.clock.now += 5
        t.press("q1")
        self.clock.now += 2
        self.assertIn("第一段完成 2.0 s", t.segment_done("q1"))

    def test_without_spawn_the_worker_was_already_running(self):
        t = self.timing
        t.press("q1")
        self.clock.now += 1
        line = t.segment_done("q1")
        self.assertIn("worker 已在執行", line)
        self.assertNotIn("模型載入", line)

    def test_unpressed_item_and_bad_models_are_ignored(self):
        t = self.timing
        self.assertIsNone(t.segment_done("nobody"))
        self.assertIsNone(t.pressed_at("nobody"))
        t.models_loaded("nobody", 1.0)                  # no press: nothing to attach to
        t.press("q1")
        for bad in (True, -1.0, "2.5", None):
            t.models_loaded("q1", bad)
        self.assertNotIn("模型載入", t.segment_done("q1"))

    def test_pressed_at_is_the_wall_clock_of_the_last_press(self):
        t = self.timing
        t.press("q1")
        self.assertEqual(t.pressed_at("q1"), 1_700_000_000.0)
        self.wall.now += 10
        t.press("q1")
        self.assertEqual(t.pressed_at("q1"), 1_700_000_010.0)

    def test_spawn_before_press_is_not_counted(self):
        self.timing.spawned("q1")                      # no press yet: nothing stored
        self.timing.press("q1")
        self.clock.now += 1
        self.assertIn("worker 已在執行", self.timing.segment_done("q1"))


class QueueReaderTimingTests(unittest.TestCase):
    """_read_worker_output routes the worker's timing / segment_done lines to the press timing."""

    def run_reader(self, timing, lines):
        proc = types.SimpleNamespace(stdout=[(json.dumps(m) + "\n").encode("utf-8") for m in lines])
        stub = types.SimpleNamespace(_press=timing, _record_perf_obs=lambda msg: None)
        out: dict = {}
        QS.QueueService._read_worker_output(stub, proc, "q9", None, out, None)
        return out

    def test_timing_then_segment_done_logs_once(self):
        timing = PressTiming(clock=FakeClock(), wall=FakeClock())
        timing.press("q9")
        with patch.object(QS, "log") as log:
            self.run_reader(timing, [{"type": "timing", "models_s": 2.5},
                                     {"type": "segment_done", "k": 0, "segments": 1},
                                     {"type": "segment_done", "k": 1, "segments": 2}])
        self.assertEqual(log.call_count, 1)
        self.assertIn("模型載入 2.5 s", log.call_args[0][0])

    def test_segment_done_of_an_unpressed_item_is_silent(self):
        with patch.object(QS, "log") as log:
            self.run_reader(PressTiming(), [{"type": "segment_done", "k": 0, "segments": 1}])
        log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
