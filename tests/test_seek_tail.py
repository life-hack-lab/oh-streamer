"""Offline unit tests for seek_tail.py: passthrough pacing, ending / discarding a tail, frame layout at 2x (run: python
streamer/tests/test_seek_tail.py). Fakes stand in for the ffmpeg muxer and the source reader; no mpv, PyAV or torch."""
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

import numpy as np
import oh_streamer.player.seek_tail as ST  # noqa: E402

FPS = 10.0



def finished(t):
    """The tail's stream is complete and its muxer has exited."""
    return t.ended and (t._thread is None or not t._thread.is_alive()) and t.muxer.poll() is not None

class FakeStdin:
    def __init__(self):
        self.chunks = []
        self.closed = False
        self.block = threading.Event()   # set: write() blocks until unblocked (a muxer nobody reads)
        self.unblock = threading.Event()

    def write(self, b):
        if self.closed:
            raise ValueError("write to closed file")
        if self.block.is_set():
            self.unblock.wait(10)
            if self.closed:
                raise OSError("broken pipe")
        self.chunks.append(bytes(b))
        return len(b)

    def close(self):
        self.closed = True
        self.unblock.set()


class FakeMuxer:
    def __init__(self):
        self.stdin = FakeStdin()
        self.rc = None
        self.killed = False

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True
        self.rc = -9
        self.stdin.unblock.set()

    def wait(self, timeout=None):
        return self.rc

    def frames(self):
        return [c[0] for c in self.stdin.chunks]   # each fake frame is filled with its index


def source(start_t, n=1000, fail_at=None, opened=None):
    """Frames from start_t: 2x2x3 arrays filled with the frame index (mod 256)."""
    if opened is not None:
        opened.append(start_t)
    k0 = round(start_t * FPS)
    for k in range(k0, k0 + n):
        if fail_at is not None and k >= fail_at:
            raise RuntimeError("decode error")
        yield np.full((2, 2, 3), k % 256, dtype=np.uint8), k / FPS


def wait_for(cond, timeout=3.0):
    t = time.perf_counter() + timeout
    while time.perf_counter() < t:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class PacingTest(unittest.TestCase):
    def test_wait_before_write(self):
        self.assertEqual(ST.wait_before_write(12.0, 10.0, False, 2.0), 0.0)     # exactly at the lead
        self.assertAlmostEqual(ST.wait_before_write(13.5, 10.0, False, 2.0), 1.5)
        self.assertEqual(ST.wait_before_write(50.0, 10.0, True, 2.0), 0.0)      # mpv waits for data: never hold back
        self.assertGreater(ST.wait_before_write(5.0, None, False, 2.0), 0.0)    # no position yet: wait

    def test_switch_mode(self):
        self.assertEqual(ST.switch_mode(2.5), "append")
        self.assertEqual(ST.switch_mode(ST.APPEND_MAX_REMAINING_S + 0.1), "replace")   # much restored left

    def test_clock_extrapolates_only_while_advancing(self):
        c = ST.PlayClock()
        self.assertEqual(c.sample(0.0), (None, False))
        c.update(10.0, True, False, now=100.0)
        self.assertAlmostEqual(c.sample(101.5)[0], 11.5)
        c.update(10.0, False, True, now=100.0)
        self.assertEqual(c.sample(105.0), (10.0, True))
        self.assertEqual(c.last(), 10.0)


class TailTest(unittest.TestCase):
    def make(self, next_t=20.0, start_pos=10.0, **kw):
        self.mux = FakeMuxer()
        self.opened = []
        self.logs = []
        src = kw.pop("src", lambda t: source(t, opened=self.opened))
        self.tail = ST.SeekTail(self.mux, "tcp://127.0.0.1:1", start_pos, next_t, FPS, src, self.logs.append,
                                lead=2.0, prepare_s=4.0, **kw)
        return self.tail

    def tearDown(self):
        if getattr(self, "tail", None) is not None:
            self.tail.close()

    def test_source_opened_only_shortly_before_needed(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        t.observe(10.0, False, False)
        time.sleep(0.3)
        self.assertEqual(self.opened, [])              # 20 - 10 > lead + prepare: restored video still plays
        t.observe(14.5, False, False)                  # within lead + prepare of 20 s: open, but nothing written yet
        self.assertTrue(wait_for(lambda: self.opened == [20.0]))
        time.sleep(0.2)
        self.assertEqual(self.mux.frames(), [])

    def test_writes_only_lead_ahead_of_playback(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        t.observe(19.0, False, False)                  # frames up to 21.0 may be written: 20.0 .. 21.0 = 11 frames
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 11))
        time.sleep(0.3)
        self.assertEqual(len(self.mux.frames()), 11)
        self.assertEqual(self.mux.frames()[0], 200)    # first passthrough frame = next_t (frame 200 at 10 fps)
        self.assertAlmostEqual(t.written_end, 21.1)
        self.assertAlmostEqual(t.remaining(19.0), 2.1)
        t.observe(20.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 21))

    def test_buffering_mpv_gets_frames_beyond_the_lead(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 11))
        t.observe(19.0, False, True)                   # paused-for-cache: it waits for more than the lead allows
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) > 40))
        t.observe(19.0, False, False)
        n = len(self.mux.frames())
        time.sleep(0.3)
        self.assertLessEqual(len(self.mux.frames()), n + 1)   # back to pacing (at most the frame in flight)

    def test_extrapolated_position_paces_while_playing(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        t.observe(19.0, True, False)                   # advancing: about FPS frames per second
        time.sleep(1.0)
        n = len(self.mux.frames())
        self.assertGreaterEqual(n, 11 + 7)
        self.assertLessEqual(n, 11 + 13)

    def test_finish_closes_stdin_and_stops(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 11))
        t.finish("switched")
        self.assertTrue(self.mux.stdin.closed)
        self.assertTrue(t.ended)
        self.assertFalse(self.mux.killed)              # graceful: nothing written is lost
        t.observe(30.0, False, False)
        time.sleep(0.2)
        self.assertEqual(len(self.mux.frames()), 11)
        self.assertFalse(finished(t))                     # the muxer still runs (mpv reads the rest)
        self.mux.rc = 0
        self.assertTrue(wait_for(lambda: finished(t)))
        self.assertIn("switched", t.summary())

    def test_close_kills_the_muxer_even_if_a_write_blocks(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        self.mux.stdin.block.set()
        t.observe(19.0, False, False)
        time.sleep(0.2)
        t0 = time.perf_counter()
        t.close()
        self.assertLess(time.perf_counter() - t0, 1.5)
        self.assertTrue(self.mux.killed)
        self.assertTrue(finished(t))

    def test_finish_with_a_blocked_write_kills_after_the_timeout(self):
        t = self.make(next_t=20.0, start_pos=10.0).start()
        self.mux.stdin.block.set()
        t.observe(19.0, False, False)
        time.sleep(0.2)
        old = ST.WRITE_LOCK_TIMEOUT
        ST.WRITE_LOCK_TIMEOUT = 0.2
        try:
            t.finish()
        finally:
            ST.WRITE_LOCK_TIMEOUT = old
        self.assertTrue(self.mux.killed)
        self.assertTrue(t.ended)

    def test_source_end_closes_the_stream(self):
        t = self.make(next_t=20.0, start_pos=19.0, src=lambda s: source(s, n=5)).start()
        self.assertTrue(wait_for(lambda: t.ended))
        self.assertEqual(len(self.mux.frames()), 5)
        self.assertEqual(t.end_reason, "source ended")
        self.assertTrue(any("source ended" in m for m in self.logs))

    def test_decode_error_closes_the_stream(self):
        t = self.make(next_t=20.0, start_pos=19.0, src=lambda s: source(s, fail_at=203)).start()
        self.assertTrue(wait_for(lambda: t.ended))
        self.assertEqual(len(self.mux.frames()), 3)
        self.assertIn("passthrough failed", t.end_reason)

    def test_passthrough_limit(self):
        t = self.make(next_t=20.0, start_pos=19.0, max_s=0.5).start()
        t.observe(19.0, False, True)                   # buffering: no pacing, only the limit stops it
        self.assertTrue(wait_for(lambda: t.ended))
        self.assertEqual(len(self.mux.frames()), 5)
        self.assertIn("limit", t.end_reason)

    def test_double_rate_writes_previous_then_current(self):
        t = self.make(next_t=20.0, start_pos=19.0, rate_mult=2)
        t.observe(18.25, False, False)                 # (before start: no race with the seeded position)
        t.start()                                      # frames 20.0 .. 20.2
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 6))
        self.assertEqual(self.mux.frames(), [200, 200, 200, 201, 201, 202])
        self.assertEqual(t.frames, 3)

    def test_segment_that_had_ended_has_no_passthrough(self):
        mux = FakeMuxer()
        mux.stdin.close()
        t = ST.SeekTail(mux, "tcp://127.0.0.1:1", 10.0, 20.0, FPS, lambda s: source(s), lambda m: None).start()
        self.assertTrue(t.ended)
        self.assertIsNone(t._thread)
        mux.rc = 0
        self.assertTrue(finished(t))
        t.close()

    def test_summary_counts_restored_and_passthrough(self):
        t = self.make(next_t=20.0, start_pos=12.0).start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 11))
        s = t.summary(final_pos=20.5)
        self.assertIn("8.0s restored played", s)
        self.assertIn("passthrough 1.1s written / 0.5s played", s)


class LayersTest(unittest.TestCase):
    def make(self, **kw):
        self.mux = FakeMuxer()
        self.tail = ST.SeekTail(self.mux, "tcp://127.0.0.1:1", 10.0, 20.0, FPS, lambda t: source(t), lambda m: None,
                                lead=2.0, prepare_s=4.0, **kw)
        return self.tail

    def tearDown(self):
        self.tail.close()

    def test_passthrough_frames_are_written_once_per_layer(self):
        t = self.make(layers=3).start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 33))
        self.assertEqual(self.mux.frames()[:6], [200, 200, 200, 201, 201, 201])
        self.assertEqual(t.frames, 11)
        self.assertAlmostEqual(t.written_end, 21.1)       # accounting counts source frames, not writes

    def test_rate_mult_2_with_layers(self):
        t = self.make(layers=2, rate_mult=2).start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) >= 8))
        # first frame: (arr, arr) x 2 layers; second: (prev=200, 201) x 2 layers
        self.assertEqual(self.mux.frames()[:8], [200, 200, 200, 200, 200, 200, 201, 201])

    def test_default_is_one_layer(self):
        t = self.make().start()
        t.observe(19.0, False, False)
        self.assertTrue(wait_for(lambda: len(self.mux.frames()) == 11))


if __name__ == "__main__":
    unittest.main()
