"""Offline tests for the restore pause: DetectGate hard hold and Session.pause_restore / resume_restore (fakes, no GPU)."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from fractions import Fraction
from types import SimpleNamespace
import threading
import time
import unittest

import oh_streamer.processing.detect_gate as DG
from oh_streamer.player import pipeline as P


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


def run(fn, *a):
    done = threading.Event()

    def body():
        fn(*a)
        done.set()
    threading.Thread(target=body, daemon=True).start()
    return done


def make_gate():
    r = FakeRestorer()
    g = DG.DetectGate.attach(r)
    g.reset(10)
    g.open("test")           # start/seek limit logic out of the way: only the hard hold is under test
    return r, g


class HoldTest(unittest.TestCase):
    def test_hold_blocks_frames_not_markers_and_release_unblocks(self):
        r, g = make_gate()
        self.assertTrue(g.hold("x"))
        self.assertFalse(g.hold("y"))
        self.assertTrue(g.held)
        self.assertTrue(run(r.frame_detection_queue.put, "EOF").wait(1))      # marker: never waits
        done = run(r.frame_detection_queue.put, (0, 0))
        self.assertFalse(done.wait(0.3))
        self.assertEqual(g.det_n, 0)
        self.assertTrue(g.release())
        self.assertTrue(done.wait(1))
        self.assertEqual(g.det_n, 1)
        self.assertFalse(g.held)

    def test_no_hold_passes_at_once(self):
        r, g = make_gate()
        self.assertTrue(run(r.frame_detection_queue.put, (0, 0)).wait(1))

    def test_stop_while_held_does_not_deadlock(self):
        r, g = make_gate()
        g.hold()
        done = run(r.frame_detection_queue.put, (0, 0))
        self.assertFalse(done.wait(0.2))
        g.open("stopped")
        self.assertTrue(done.wait(1))
        self.assertFalse(g.held)

    def test_reset_releases_hold(self):
        r, g = make_gate()
        g.hold()
        g.reset(10)
        self.assertFalse(g.held)

    def test_watchdog_does_not_open_a_hold(self):
        r = FakeRestorer()
        g = DG.DetectGate.attach(r)
        g.reset(10)
        g.watchdog_s = 0.1
        g.hold()
        done = run(r.frame_detection_queue.put, (0, 0))
        self.assertFalse(done.wait(0.5))
        self.assertFalse(g.is_open)
        self.assertTrue(g.held)
        g.release()
        self.assertTrue(done.wait(1))


class FakeCuda:
    def __init__(self, available=True):
        self.available, self.empty_calls, self.reserved = available, 0, 3 * 2**30

    def is_available(self):
        return self.available

    def memory_reserved(self):
        return self.reserved

    def empty_cache(self):
        self.empty_calls += 1
        self.reserved = 2**30


class SessionPauseTest(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.cuda = FakeCuda()
        self.gate_restorer = FakeRestorer()
        self.gate = DG.DetectGate.attach(self.gate_restorer)
        self.gate.reset(10)
        self.gate.open("test")
        self.restorer = SimpleNamespace(
            attach_detect_gate=lambda: self.gate, attach_progress_probe=lambda: None,
            drain=lambda: None, start=lambda *a, **kw: None, stop=lambda: None, wake=lambda: None,
            max_clip_length=10, frames=lambda: iter(()))
        self.source = SimpleNamespace(
            src="t.mp4", audio_src="t.mp4", has_audio=False, fmt_start=0.0, http=None,
            meta=SimpleNamespace(video_fps_exact=Fraction(10), time_base=Fraction(1, 10), video_width=2, video_height=2))
        self.deps = P.PipelineDependencies(
            log=self.logs.append, start_muxer=lambda *a, **kw: None, stop_process=lambda _m: None,
            free_port=lambda: 1, is_listening=lambda *_: True,
            torch=SimpleNamespace(Tensor=type("Tensor", (), {}), cuda=self.cuda), open_reader=lambda src: None)
        self.sess = P.Session(self.restorer, self.source, "libx264", None, deps=self.deps)
        self.sess.pause_idle_s, self.sess.pause_poll_s = 0.3, 0.02
        self.sess.thread = SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False)
        self.sess.gate = self.gate
        self.sess.stats = SimpleNamespace(summary=lambda: "stats")

    def wait_state(self, state, timeout=3.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end and self.sess.pause_state() != state:
            time.sleep(0.02)
        return self.sess.pause_state()

    def test_pause_goes_pausing_then_paused_while_output_continues_then_stops(self):
        self.assertEqual(self.sess.pause_state(), "running")
        self.assertEqual(self.sess.pause_restore(), "pausing")
        self.assertTrue(self.gate.held)
        for _ in range(8):          # frames already detected still come out: not idle yet
            self.gate_restorer.frame_restoration_queue.put((object(), 0))
            time.sleep(0.1)
            self.assertEqual(self.sess.pause_state(), "pausing")
        self.assertEqual(self.wait_state("paused"), "paused")
        self.assertEqual(self.cuda.empty_calls, 1)
        time.sleep(0.2)
        self.assertEqual(self.cuda.empty_calls, 1)
        self.assertEqual(self.sess.pause_restore(), "paused")    # idempotent
        self.assertEqual(self.cuda.empty_calls, 1)
        self.assertTrue(any("restore paused after" in m and "VRAM" in m for m in self.logs))
        self.assertEqual(self.sess.resume_restore(), "running")
        self.assertEqual(self.sess.pause_state(), "running")
        self.assertFalse(self.gate.held)
        self.assertTrue(any("resumed" in m for m in self.logs))

    def test_reserved_vram_after_the_pause_is_reported_only_while_paused(self):
        self.assertIsNone(self.sess.pause_reserved_mib())
        self.sess.pause_restore()
        self.assertIsNone(self.sess.pause_reserved_mib())                  # still pausing
        self.assertEqual(self.wait_state("paused"), "paused")
        self.assertEqual(self.sess.pause_reserved_mib(), 1024.0)           # reserved after empty_cache (1 GiB)
        self.sess.resume_restore()
        self.assertIsNone(self.sess.pause_reserved_mib())

    def test_no_cuda_means_no_reserved_figure(self):
        self.cuda.available = False
        self.sess.pause_restore()
        self.assertEqual(self.wait_state("paused"), "paused")
        self.assertIsNone(self.sess.pause_reserved_mib())

    def test_resume_while_pausing_cancels_without_empty_cache(self):
        self.sess.pause_restore()
        self.sess.resume_restore()
        time.sleep(0.6)
        self.assertEqual(self.sess.pause_state(), "running")
        self.assertEqual(self.cuda.empty_calls, 0)

    def test_cpu_engine_without_cuda_still_pauses(self):
        self.cuda.available = False
        self.sess.pause_restore()
        self.assertEqual(self.wait_state("paused"), "paused")
        self.assertEqual(self.cuda.empty_calls, 0)

    def test_new_segment_start_clears_pause(self):
        self.sess.pause_restore()
        self.assertEqual(self.wait_state("paused"), "paused")
        self.sess.thread = None
        self.sess.stats = None
        self.sess.start(5.0)
        self.assertEqual(self.sess.pause_state(), "running")
        self.assertFalse(self.gate.held)
        self.sess.thread.join(2)

    def test_stop_clears_pause_and_gate_hold(self):
        self.sess.pause_restore()
        self.sess.stop()
        self.assertEqual(self.sess.pause_state(), "running")
        self.assertFalse(self.gate.held)

    def test_no_gate_reports_paused_and_resumes(self):
        self.sess.gate = None
        self.assertEqual(self.sess.pause_restore(), "paused")
        self.assertEqual(self.sess.pause_state(), "paused")
        self.assertEqual(self.cuda.empty_calls, 0)
        self.assertEqual(self.sess.resume_restore(), "running")
        self.assertEqual(self.sess.pause_state(), "running")


if __name__ == "__main__":
    unittest.main()
