"""Deterministic lifecycle checks for restoration segment ownership and shutdown."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oh_streamer.engines import base
from oh_streamer.engines.lada import engine as lada_engine
from oh_streamer.player import pipeline as P


class CancelEvent:
    def __init__(self, events):
        self.events, self.active = events, False

    def set(self):
        self.active = True
        self.events.append("cancel:set")

    def clear(self):
        self.active = False
        self.events.append("cancel:clear")


class PumpThread:
    def __init__(self, events, stuck=False):
        self.events, self.stuck = events, stuck

    def join(self, timeout=None):
        self.events.append(("join", timeout))

    def is_alive(self):
        return self.stuck


class Frame:
    def reshape(self, _size):
        return b"frame"


class FrameList:
    """An engine session that only yields the given RestoredFrames (frames() ends after them: natural end)."""

    def __init__(self, items):
        self.items = items

    def frames(self):
        return iter(self.items)

    def stop(self, **kw):
        pass


class PipelineLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cancel = CancelEvent(self.events)
        self.queue = SimpleNamespace(maxsize=0)
        self.stop_error = None
        self.frames = []

        def stop():
            self.events.append("restorer:stop")
            self.assertTrue(self.cancel.active)
            if self.stop_error is not None:
                raise self.stop_error

        self.restorer = SimpleNamespace(
            mosaic_restoration_model=SimpleNamespace(cancel_event=self.cancel),
            max_clip_length=30, eof=True,
            get_frame_restoration_queue=lambda: self.queue,
            start=lambda **kw: self.events.append(("restorer:start", kw["start_ns"])),
            stop=stop,
        )
        self.source = SimpleNamespace(
            src="test.mp4", audio_src="audio.mp4", has_audio=True, fmt_start=0.0, http=None,
            meta=SimpleNamespace(video_fps_exact=Fraction(10), time_base=Fraction(1, 10),
                                 video_width=2, video_height=2),
        )
        self.muxer = SimpleNamespace(stdin=SimpleNamespace(
            write=lambda _buf: self.events.append("stdin:write"),
            close=lambda: self.events.append("stdin:close")))
        self.gate = SimpleNamespace(
            open=lambda reason: self.events.append(("gate:open", reason)),
            summary=lambda: "gate", reset=lambda *a, **kw: None, is_open=True,
        )
        self.deps = P.PipelineDependencies(
            log=lambda _message: None,
            start_muxer=lambda *a, **kw: self.muxer,
            stop_process=lambda _muxer: self.events.append("muxer:stop"),
            free_port=lambda: 1234, is_listening=lambda _muxer, _port: True,
            torch=SimpleNamespace(Tensor=type("Tensor", (), {}),
                                  cuda=SimpleNamespace(is_available=lambda: False)),
            open_reader=lambda src: self.reader_cls(src),
        )
        # the real engine session (LadaSession) around the fake FrameRestorer: its stop/wake/drain order is what is checked
        for name, event in (("empty_out_queue", "queue:empty"), ("put_queue_stop_marker", "queue:stop")):
            p = patch.object(lada_engine.threading_utils, name, lambda _q, event=event: self.events.append(event))
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(lada_engine.io_hooks, "register_source", lambda _s: self.events.append("source:register"))
        p.start()
        self.addCleanup(p.stop)
        self.esession = lada_engine.LadaSession(SimpleNamespace(max_clip_length=30, make=lambda _s: self.restorer),
                                                self.source)

    def running_session(self, stuck=False):
        sess = P.Session(self.esession, self.source, "libx264", None, deps=self.deps)
        sess.thread = PumpThread(self.events, stuck)
        sess.muxer = self.muxer
        sess.url = "tcp://127.0.0.1:1234"
        sess.gate = self.gate
        sess.probe = SimpleNamespace(freeze=lambda: self.events.append("probe:freeze"))
        sess.stats = SimpleNamespace(summary=lambda: "stats")
        sess.out_t = 10.5
        return sess

    def test_stop_unblocks_output_then_cancels_and_drains_workers(self):
        sess = self.running_session()
        sess.stop()
        self.assertEqual(self.events, [
            "probe:freeze", ("gate:open", "stopped"), "muxer:stop", "cancel:set",
            "restorer:stop", "cancel:clear", "queue:stop", ("join", None), "queue:empty",
        ])
        self.assertIsNone(sess.thread)
        self.assertTrue(sess.stopping)

    def test_detach_transfers_muxer_without_closing_it_and_uses_source_fps(self):
        sess = self.running_session()
        sess.doubler = SimpleNamespace(summary=lambda: "2x")
        sess.layers = ("processed", "restored", "original")
        _elapsed, out = sess.detach_output()
        self.assertIs(out["muxer"], self.muxer)
        self.assertEqual(out["rate_mult"], 2)
        self.assertEqual(out["layers"], 3)
        self.assertAlmostEqual(out["next_t"], 10.6)
        self.assertIsNone(sess.muxer)
        self.assertNotIn("muxer:stop", self.events)
        self.assertNotIn("stdin:close", self.events)

    def test_detach_kills_stuck_output_before_joining_pump(self):
        sess = self.running_session(stuck=True)
        _elapsed, out = sess.detach_output()
        self.assertIsNone(out)
        self.assertIsNone(sess.muxer)
        timeout_join = self.events.index(("join", 2.0))
        kill = self.events.index("muxer:stop")
        final_join = self.events.index(("join", None))
        self.assertLess(timeout_join, kill)
        self.assertLess(kill, final_join)

    def test_stop_clears_cancel_if_restorer_stop_raises(self):
        sess = self.running_session()
        self.stop_error = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            sess.stop()
        self.assertFalse(self.cancel.active)
        self.assertEqual(self.events[-1], "cancel:clear")

    def test_cast_stream_close_is_idempotent_and_wakes_iteration(self):
        with patch.object(P.detect_gate.DetectGate, "attach", return_value=self.gate):
            stream = P.RestorerStream(self.esession, self.source, 3.0, deps=self.deps)
        self.assertFalse(self.restorer.eof)
        self.assertFalse(self.restorer.capture_original)
        self.events.clear()
        stream.close()
        stream.close()
        self.assertEqual(self.events, [
            ("gate:open", "closed"), "cancel:set", "restorer:stop", "cancel:clear", "queue:stop",
        ])

    def test_cast_stream_close_clears_cancel_when_restorer_stop_raises(self):
        with patch.object(P.detect_gate.DetectGate, "attach", return_value=self.gate):
            stream = P.RestorerStream(self.esession, self.source, 3.0, deps=self.deps)
        self.stop_error = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            stream.close()
        self.assertFalse(self.cancel.active)

    def test_natural_pump_end_closes_stdin_after_writing_last_frame(self):
        sess = P.Session(FrameList([base.RestoredFrame(Frame(), 100)]), self.source, "libx264", None, deps=self.deps)
        sess.target = 10.0
        sess.gate = self.gate
        sess.stats = P.Stats(10, 10, deps=self.deps)
        sess._pump()
        self.assertEqual(sess.frames_total, 1)
        self.assertAlmostEqual(sess.out_t, 10.0)
        self.assertLess(self.events.index("stdin:write"), self.events.index("stdin:close"))
        self.assertEqual(self.events[-2:], [("gate:open", "pump ended"), "stdin:close"])

    def test_stopping_pump_leaves_transferred_output_open(self):
        sess = P.Session(FrameList([base.RestoredFrame(Frame(), 100)]), self.source, "libx264", None, deps=self.deps)
        sess.stopping, sess.muxer, sess.gate = True, self.muxer, self.gate
        sess._pump()
        self.assertNotIn("stdin:write", self.events)
        self.assertNotIn("stdin:close", self.events)

    def test_closing_passthrough_generator_closes_reader(self):
        events = self.events

        class Reader:
            def __init__(self, _src):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                events.append("reader:close")

            def seek(self, ns):
                events.append(("reader:seek", ns))

            def frames(self):
                yield Frame(), 100
                yield Frame(), 101

        self.reader_cls = Reader
        frames = P.passthrough_frames(self.source, 10.0, deps=self.deps)
        self.assertEqual(next(frames)[1], 10.0)
        frames.close()
        self.assertEqual(events, [("reader:seek", 10_000_000_000), "reader:close"])

    def test_batch_completion_closes_output_and_waits_after_stopping_restorer(self):
        events = self.events

        class StopThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        class BatchRestorer:
            def start(self, start_s, **kw):
                events.append(("restorer:start", start_s))

            def frames(self):
                yield base.RestoredFrame(Frame(), 100)

            def stop(self, **kw):
                events.append("restorer:stop")

        self.muxer.wait = lambda: events.append("muxer:wait")
        args = SimpleNamespace(start=10.0, output="output.mkv", simulate_player=False)
        with patch.object(P.threading, "Thread", StopThread):
            result = P.run_batch(args, BatchRestorer(), self.source.src, self.source.meta,
                                 True, 0.0, None, "libx264", None, deps=self.deps)
        self.assertEqual(result, 0)
        self.assertEqual(events[-3:], ["restorer:stop", "stdin:close", "muxer:wait"])

    def test_batch_error_marker_returns_failure_and_still_stops_restorer(self):
        events = self.events

        class StopThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        class BatchRestorer:
            def start(self, start_s, **kw):
                pass

            def frames(self):
                yield self_marker

            def stop(self, **kw):
                events.append("restorer:stop")

        self_marker = base.EngineError("decode failed", RuntimeError("decode failed"))
        args = SimpleNamespace(start=0.0, output="null", simulate_player=False)
        with patch.object(P.threading, "Thread", StopThread):
            result = P.run_batch(args, BatchRestorer(), self.source.src, self.source.meta,
                                 True, 0.0, None, "libx264", None, deps=self.deps)
        self.assertEqual(result, 1)
        self.assertEqual(events, ["restorer:stop"])


if __name__ == "__main__":
    unittest.main()
