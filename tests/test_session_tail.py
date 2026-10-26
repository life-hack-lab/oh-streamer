"""Offline unit tests for the Session side of the seek tail (run: python -P streamer/tests/test_session_tail.py).

Imports oh_player (torch + lada, no models / GPU / mpv / ffmpeg needed): a fake FrameRestorer feeds the pump, a fake
muxer stands in for ffmpeg. Covers detach_output() (the muxer survives, next_t = one frame after the last one written),
the normal stop (muxer killed) and output_reached() (Lada's output counted, not what the blocked muxer took)."""
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
import types
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np
import oh_streamer.player.oh_player as L  # noqa: E402
from oh_streamer.engines.lada import engine as lada_engine  # noqa: E402
from lada.utils import threading_utils  # noqa: E402

FPS = 10


class FakeRestorer:
    def __init__(self, maxsize=0):
        self.q = threading_utils.PipelineQueue("frame_restoration_queue", maxsize=maxsize)
        self.mosaic_restoration_model = types.SimpleNamespace(cancel_event=threading.Event())
        self.max_clip_length = 30
        self.eof = False
        self.stop_requested = False
        self.stops = 0

    def get_frame_restoration_queue(self):
        return self.q

    def start(self, start_ns=0):
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True
        self.stops += 1
        threading_utils.empty_out_queue(self.q)

    def __iter__(self):
        return self

    def __next__(self):
        elem = self.q.get()
        return elem

    def feed(self, k0, n):
        for k in range(k0, k0 + n):
            self.q.put((np.full((2, 2, 3), k % 256, dtype=np.uint8), k))   # pts in 1/FPS units


class FakeStdin:
    def __init__(self):
        self.n, self.closed = 0, False

    def write(self, b):
        if self.closed:
            raise ValueError("closed")
        self.n += 1

    def close(self):
        self.closed = True


class FakeMuxer:
    def __init__(self, *a, **kw):
        self.stdin, self.rc, self.killed = FakeStdin(), None, False

    def poll(self):
        return self.rc

    def kill(self):
        self.killed, self.rc = True, -9

    def wait(self, timeout=None):
        return self.rc


def wrap(restorer, source=None):
    """The engine session (what Session / RestorerStream take) around a FakeRestorer."""
    factory = types.SimpleNamespace(max_clip_length=restorer.max_clip_length, make=lambda src: restorer)
    return lada_engine.LadaSession(factory, source or fake_source())


def fake_source():
    meta = types.SimpleNamespace(video_fps_exact=Fraction(FPS), time_base=Fraction(1, FPS), video_width=2,
                                 video_height=2)
    return types.SimpleNamespace(src="x.mp4", audio_src="x.mp4", meta=meta, has_audio=False, fmt_start=0.0, http=None)


def wait_for(cond, timeout=3.0):
    t = time.perf_counter() + timeout
    while time.perf_counter() < t:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class SessionTailTest(unittest.TestCase):
    def setUp(self):
        self.muxers = []

        def start_muxer(*a, **kw):
            m = FakeMuxer()
            self.muxers.append(m)
            return m
        self.patch = mock.patch.object(L, "start_muxer", start_muxer)
        self.patch.start()
        self.reg = mock.patch.object(lada_engine.io_hooks, "register_source")   # no I/O hooks for the fake sources
        self.reg.start()
        self.r = FakeRestorer()
        self.s = L.Session(wrap(self.r), fake_source(), "libx264", None)
        self.s.start(10.0)

    def tearDown(self):
        if self.s.thread is not None:
            self.s.stop()
        self.patch.stop()
        self.reg.stop()

    def test_detach_keeps_the_muxer_and_reports_the_next_frame(self):
        self.r.feed(100, 6)                            # 10.0 .. 10.5 s
        self.assertTrue(wait_for(lambda: self.muxers and self.muxers[0].stdin.n == 6))
        took, out = self.s.detach_output()
        self.assertIsNotNone(out)
        self.assertIs(out["muxer"], self.muxers[0])
        self.assertFalse(self.muxers[0].killed)        # mpv keeps playing it: the tail continues the stream
        self.assertFalse(self.muxers[0].stdin.closed)
        self.assertAlmostEqual(out["next_t"], 10.6)
        self.assertEqual(out["rate_mult"], 1)
        self.assertTrue(out["url"].startswith("tcp://127.0.0.1:"))
        self.assertIsNone(self.s.muxer)
        self.assertIsNone(self.s.thread)
        self.assertTrue(self.r.mosaic_restoration_model.cancel_event.is_set() is False)   # fast stop cleared it
        self.assertEqual(self.r.stops, 1)

    def test_detach_before_the_first_frame_has_nothing_to_continue(self):
        self.r.feed(95, 3)                             # pre-target frames only: skipped, no muxer yet
        time.sleep(0.2)
        took, out = self.s.detach_output()
        self.assertIsNone(out)
        self.assertEqual(self.muxers, [])

    def test_plain_stop_kills_the_muxer(self):
        self.r.feed(100, 3)
        self.assertTrue(wait_for(lambda: self.muxers and self.muxers[0].stdin.n == 3))
        self.s.stop()
        self.assertTrue(self.muxers[0].killed)

    def test_output_reached_counts_lada_output(self):
        self.r.feed(100, 6)
        self.assertTrue(wait_for(lambda: self.s.out_t is not None and abs(self.s.out_t - 10.5) < 1e-9))
        self.assertAlmostEqual(self.s.lada_out_t(), 10.5)
        self.assertTrue(self.s.output_reached(0.5))
        self.assertFalse(self.s.output_reached(1.0))
        self.s.gate = types.SimpleNamespace(out_n=20)  # Lada put out more than the (blocked) pump took
        self.assertAlmostEqual(self.s.lada_out_t(), 11.9)
        self.assertTrue(self.s.output_reached(1.5))
        self.s.gate = None

    def test_output_reached_when_the_output_queue_is_full(self):
        self.s.stop()
        self.r = FakeRestorer(maxsize=4)
        self.s.set_source(wrap(self.r), fake_source())
        self.s.start(10.0)
        self.r.feed(100, 1)
        self.assertTrue(wait_for(lambda: self.s.out_t is not None))
        self.assertFalse(self.s.output_reached(1.5))
        with mock.patch.object(self.r.q, "qsize", lambda: 4):
            self.assertTrue(self.s.output_reached(1.5))    # cannot get further ahead before mpv connects


class RecMuxer(FakeMuxer):
    """Records the bytes written (to check the stacked pipe frames)."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.chunks = []
        self.kw = kw
        stdin = self.stdin

        def write(b):
            if stdin.closed:
                raise ValueError("closed")
            self.chunks.append(bytes(b))
        stdin.write = write


class FakeDoubler:
    """push() -> [middle frame (value 100 + k), frame]; the first frame only [frame]; finish() -> [copy of the last]."""

    def __init__(self):
        self.last = None

    def push(self, frame, arr, t):
        out = [arr] if self.last is None else [np.full_like(arr, 100 + int(arr.flat[0])), arr]
        self.last = arr
        return out

    def finish(self):
        return [self.last]

    def summary(self):
        return "fake"


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.muxers = []

        def start_muxer(*a, **kw):
            m = RecMuxer(**kw)
            self.muxers.append(m)
            return m
        self.patch = mock.patch.object(L, "start_muxer", start_muxer)
        self.patch.start()
        self.reg = mock.patch.object(lada_engine.io_hooks, "register_source")   # no I/O hooks for the fake sources
        self.reg.start()
        self.r = FakeRestorer()
        self.s = L.Session(wrap(self.r), fake_source(), "libx264", None)
        self.s.compare_mode = "on"

    def tearDown(self):
        if self.s.thread is not None:
            self.s.stop()
        self.patch.stop()
        self.reg.stop()

    def feed(self, k0, n, mosaic=()):
        import torch
        for k in range(k0, k0 + n):
            f = torch.full((2, 2, 3), 50 + k - 100, dtype=torch.uint8)
            if k in mosaic:   # Lada blended pixel (0, 0): the original there was 7
                f._lada_orig = [((0, 0, 0, 0), torch.full((1, 1, 3), 7, dtype=torch.uint8))]
            self.r.q.put((f, k))

    def chunks(self):
        return [c for m in self.muxers for c in m.chunks]

    def test_off_is_single_layer(self):
        self.s.compare_mode = "off"
        self.s.start(10.0)
        self.assertEqual(self.s.layers, ("processed",))
        self.assertFalse(self.r.capture_original)
        self.feed(100, 2)
        self.assertTrue(wait_for(lambda: len(self.chunks()) == 2))
        self.assertEqual(self.muxers[0].kw.get("layers"), ("processed",))

    def test_on_writes_layers_in_stack_order_with_rebuilt_original(self):
        self.s.start(10.0)
        self.assertEqual(self.s.layers, ("processed", "original"))
        self.assertTrue(self.r.capture_original)
        self.feed(100, 3, mosaic=(101,))
        self.assertTrue(wait_for(lambda: len(self.chunks()) == 6))
        ch = self.chunks()
        self.assertEqual(self.muxers[0].kw["layers"], ("processed", "original"))
        self.assertEqual([c[0] for c in ch], [50, 50, 51, 7, 52, 52])   # frame 101: original pixel (0,0) is 7
        o = np.frombuffer(ch[3], dtype=np.uint8).reshape(2, 2, 3)
        self.assertEqual(o[1, 1].tolist(), [51] * 3)
        self.assertEqual(ch[2][0], 51)                                   # processed layer untouched
        self.assertEqual(self.s.compare_snapshot()["layers"], ["processed", "original"])
        took, out = self.s.detach_output()
        self.assertEqual(out["layers"], 2)

    def test_rife_pairs_repeat_the_previous_frame_in_the_1x_layers(self):
        self.s.interp_mode = "auto"
        self.s.interp = object()
        with mock.patch.object(L.rife_interp, "eligible", lambda fps: (True, "")), \
                mock.patch.object(L.rife_interp, "FrameDoubler", lambda *a: FakeDoubler()):
            self.s.start(10.0)
        self.assertEqual(self.s.layers, ("processed", "original", "restored"))
        self.feed(100, 2)
        self.assertTrue(wait_for(lambda: len(self.chunks()) == 9))   # frame 100: 1 slot; frame 101: 2 slots
        v = [c[0] for c in self.chunks()]
        self.assertEqual(v[:3], [50, 50, 50])                        # first frame: no middle slot
        self.assertEqual(v[3:6], [100 + 51, 50, 50])                 # middle: interpolated, 1x layers = previous
        self.assertEqual(v[6:9], [51, 51, 51])
        self.s.stop()

    def test_guard_falls_back_to_plain_output(self):
        self.s.meta.video_width, self.s.meta.video_height = 7680, 4320
        self.s.start(10.0)
        self.assertEqual(self.s.layers, ("processed",))
        self.assertIn("too large", self.s.cmp_why)
        self.assertFalse(self.r.capture_original)

    def test_restorer_stream_clears_capture(self):
        self.r.capture_original = True
        with mock.patch.object(L.detect_gate.DetectGate, "attach", return_value=None):
            L.RestorerStream(wrap(self.r), fake_source(), 1.0)
        self.assertFalse(self.r.capture_original)


class StartMuxerArgsTest(unittest.TestCase):
    def argv(self, **kw):
        meta = fake_source().meta
        meta.video_width, meta.video_height = 1920, 1080
        meta.video_fps_exact = Fraction(30)
        with mock.patch.object(L.subprocess, "Popen") as popen:
            L.start_muxer("a.mp4", meta, kw.pop("has_audio", True), 1.0, mock.Mock(cli_args=lambda: []),
                          "tcp://127.0.0.1:1?listen=1", kw.pop("encoder", "h264_nvenc"), ts_offset=1.0, **kw)
        return popen.call_args[0][0]

    def test_single_layer_unchanged(self):
        base = self.argv()
        self.assertEqual(base, self.argv(layers=("processed",)))
        self.assertIn("1920x1080", base)
        self.assertNotIn("-filter_complex", base)
        self.assertEqual(base[base.index("-map") + 1], "0:v:0")
        self.assertIn("format=yuv420p,mpdecimate=hi=0:lo=0:frac=0", self.argv(rate_mult=2))

    def test_layers_stack_input_and_titled_tracks(self):
        a = self.argv(layers=("processed", "original", "restored"), rate_mult=2)
        self.assertEqual(a[a.index("-s") + 1], "1920x3240")
        self.assertIn("-filter_complex", a)
        self.assertIn("title=oh:original", a)
        self.assertIn("title=oh:restored", a)
        self.assertLess(a.index("[v_restored]"), a.index("1:a:0"))
        self.assertEqual(a[a.index("-g") + 1], "15")
        self.assertEqual(a[-1], "tcp://127.0.0.1:1?listen=1")
        na = self.argv(layers=("processed", "original"), has_audio=False)
        self.assertNotIn("1:a:0", na)
        self.assertEqual(na.count("-map"), 2)


if __name__ == "__main__":
    unittest.main()
