"""Offline unit tests for detect_gate.py and its coupling to buffer_progress.py (run: python streamer/tests/test_detect_gate.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import os
import sys
import threading
import time
import unittest
import oh_streamer.player.buffer_progress as BP  # noqa: E402
import oh_streamer.processing.detect_gate as DG  # noqa: E402


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item, *a, **kw):
        self.items.append(item)


class FakeRestorer:
    def __init__(self, max_clip_length=10):
        self.max_clip_length = max_clip_length
        self.stop_requested = False
        self.frame_detection_queue = FakeQueue()
        self.frame_restoration_queue = FakeQueue()
        self.mosaic_clip_queue = FakeQueue()
        self.restored_clip_queue = FakeQueue()
        self.mosaic_clip_queue.get = lambda *a, **kw: None


def detect(r, n, num=0):
    r.frame_detection_queue.put((num, 0))


def output(r):
    r.frame_restoration_queue.put((object(), 0))


def in_thread(fn, *a):
    done = threading.Event()

    def run():
        fn(*a)
        done.set()
    threading.Thread(target=run, daemon=True).start()
    return done


class GateTest(unittest.TestCase):
    def setUp(self):
        self.r = FakeRestorer(max_clip_length=10)        # gate limit 40
        self.g = DG.DetectGate.attach(self.r)

    def fill(self, n):
        for i in range(n):
            detect(self.r, 1, i)

    def test_attach_is_once_and_stored_on_the_restorer(self):
        self.assertIs(DG.DetectGate.attach(self.r), self.g)
        self.assertIs(self.r._lada_detect_gate, self.g)
        self.assertIsNone(DG.DetectGate.attach(object()))

    def test_not_started_gate_is_open(self):
        self.fill(500)
        self.assertEqual(len(self.r.frame_detection_queue.items), 500)

    def test_blocks_when_ahead_and_unblocks_with_output(self):
        self.g.reset(limit=40)
        self.fill(41)                                    # det_n 41: not yet > 40 before the next one
        done = in_thread(detect, self.r, 1, 41)
        self.assertFalse(done.wait(0.3))
        self.assertEqual(len(self.r.frame_detection_queue.items), 41)
        output(self.r)                                   # 41 - 1 is no longer more than the limit
        self.assertTrue(done.wait(1.0))
        self.assertEqual(len(self.r.frame_detection_queue.items), 42)
        self.assertGreater(self.g.held_s, 0.2)
        self.assertEqual(self.g.max_ahead, 41)

    def test_markers_are_never_blocked(self):
        self.g.reset(limit=40)
        self.fill(41)
        marker = object()
        done = in_thread(self.r.frame_detection_queue.put, marker)
        self.assertTrue(done.wait(1.0))
        self.assertIs(self.r.frame_detection_queue.items[-1], marker)
        out_marker = in_thread(self.r.frame_restoration_queue.put, object())
        self.assertTrue(out_marker.wait(1.0))
        self.assertEqual(self.g.out_n, 0)                # non-tuples are not frames

    def test_release_out_n_opens(self):
        self.g.reset(limit=40, release_out_n=5)
        self.fill(41)
        done = in_thread(detect, self.r, 1, 41)
        for _ in range(4):
            output(self.r)
        self.assertFalse(self.g.is_open)
        output(self.r)
        self.assertTrue(self.g.is_open)
        self.assertTrue(done.wait(1.0))
        self.fill(200)                                   # runs freely now
        self.assertIn("opened at out 5 (output reached)", self.g.summary())

    def test_open_wakes_a_blocked_put_promptly(self):
        self.g.reset(limit=40)
        self.fill(41)
        done = in_thread(detect, self.r, 1, 42)
        self.assertFalse(done.wait(0.2))
        t = time.perf_counter()
        self.g.open("target reached")
        self.assertTrue(done.wait(1.0))
        self.assertLess(time.perf_counter() - t, 0.04)
        self.g.open("again")                             # idempotent: the first reason stays
        self.assertEqual(self.g.reason, "target reached")
        self.assertIn("(target reached)", self.g.summary())

    def test_stop_requested_wakes_a_blocked_put(self):
        self.g.reset(limit=40)
        self.fill(41)
        done = in_thread(detect, self.r, 1, 42)
        self.assertFalse(done.wait(0.2))
        self.r.stop_requested = True
        self.assertTrue(done.wait(1.0))

    def test_disabled_gate_never_blocks(self):
        self.g.enabled = False
        self.g.reset(limit=40)
        self.assertTrue(self.g.is_open)
        self.fill(300)
        self.assertEqual(self.g.held_s, 0.0)
        self.assertEqual(self.g.summary(), "detect gate: off")

    def test_limit_is_clamped_to_clip_length_plus_margin(self):
        self.g.reset(limit=5)
        self.assertEqual(self.g.limit, 10 + DG.MARGIN)
        self.g.reset(limit=100)
        self.assertEqual(self.g.limit, 100)

    def test_watchdog_opens_when_the_output_stalls(self):
        self.g.watchdog_s = 0.3
        self.g.reset(limit=40)
        self.fill(41)
        done = in_thread(detect, self.r, 1, 42)
        self.assertTrue(done.wait(2.0))
        self.assertTrue(self.g.is_open)
        self.assertEqual(self.g.reason, "watchdog")
        self.assertIn("(watchdog)", self.g.summary())

    def test_reset_clears_the_counters_and_closes_again(self):
        self.g.reset(limit=40)
        self.fill(10)
        output(self.r)
        self.g.open("x")
        self.g.reset(limit=40)
        self.assertEqual((self.g.det_n, self.g.out_n, self.g.max_ahead, self.g.held_s), (0, 0, 0, 0.0))
        self.assertFalse(self.g.is_open)
        self.assertIn("still closed", self.g.summary())

    def test_on_held_reports_the_blocked_interval(self):
        got = []
        self.g.on_held = lambda a, b: got.append((a, b))
        self.g.reset(limit=40)
        self.fill(41)
        done = in_thread(detect, self.r, 1, 42)
        time.sleep(0.2)
        self.g.open("x")
        self.assertTrue(done.wait(1.0))
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0][1] - got[0][0], 0.15)


class CoexistTest(unittest.TestCase):
    def check(self, gate_first):
        r = FakeRestorer(max_clip_length=10)
        if gate_first:
            g = DG.DetectGate.attach(r)
            p = BP.PipelineProbe.attach(r)
        else:
            p = BP.PipelineProbe.attach(r)
            g = DG.DetectGate.attach(r)
        g.on_held = p.note_detector_held
        p.reset(0.0)
        g.reset(limit=40)
        for i in range(41):
            detect(r, 1, 100 + i)
        done = in_thread(detect, r, 1, 141)
        self.assertFalse(done.wait(0.3))                 # held by the gate
        output(r)
        self.assertTrue(done.wait(1.0))
        self.assertEqual(len(r.frame_detection_queue.items), 42)
        self.assertEqual(p.snapshot()["det"], 42)        # the probe saw every frame
        self.assertEqual(g.det_n, 42)

    def test_gate_attached_first(self):
        self.check(True)

    def test_probe_attached_first(self):
        self.check(False)


class ProbeLearningTest(unittest.TestCase):
    def setUp(self):
        self.saved = BP.RATES.det_fps, BP.RATES.p_mosaic
        BP.RATES.det_fps = None

    def tearDown(self):
        BP.RATES.det_fps, BP.RATES.p_mosaic = self.saved

    def make_probe(self, held):
        """100 detections 10 ms apart (100 fps) with a 2 s hold in the middle."""
        p = BP.PipelineProbe()
        p.reset(0.0, t0=1000.0)
        t = 1000.0
        for i in range(100):
            if i == 50:
                if held:
                    p.note_detector_held(t, t + 2.0)
                t += 2.0
            t += 0.01
            p.base = 0 if p.base is None else p.base
            p.det, p.t_last_det = i + 1, t
            if p.t_first_det is None:
                p.t_first_det = t
        return p

    def test_held_time_is_excluded_from_det_fps(self):
        p = self.make_probe(held=True)
        self.assertAlmostEqual(p.snapshot()["det_span"], 0.99, places=2)
        p.started(10)
        self.assertAlmostEqual(BP.RATES.det_fps, 100.0, delta=1.0)

    def test_without_gate_the_whole_span_counts(self):
        p = self.make_probe(held=False)
        p.started(10)
        self.assertLess(BP.RATES.det_fps, 50.0)

    def test_hold_after_the_last_detection_does_not_count(self):
        p = self.make_probe(held=False)
        p.note_detector_held(p.t_last_det, p.t_last_det + 3.0)   # probe wrapper outside the gate's: block follows
        self.assertAlmostEqual(p.snapshot()["det_span"], p.t_last_det - p.t_first_det, places=6)

    def test_frozen_probe_ignores_holds(self):
        p = BP.PipelineProbe()
        p.note_detector_held(1.0, 2.0)
        self.assertEqual(p.held, [])


if __name__ == "__main__":
    unittest.main()
