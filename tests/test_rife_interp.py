"""Offline unit tests for rife_interp.py's pure parts: policy, clocks, frame doubling (run: python streamer/tests/test_rife_interp.py).

No torch / GPU: the RIFE model is replaced by a fake. The GPU path is covered by the end-to-end runs."""
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
import unittest
import oh_streamer.player.buffer_progress as BP  # noqa: E402
import oh_streamer.processing.rife_interp as R  # noqa: E402


class EligibleTest(unittest.TestCase):
    def test_fps(self):
        for fps in (23.976, 24, 25, 29.97, 30, 40):
            self.assertEqual(R.eligible(fps), (True, ""), fps)
        for fps in (50, 59.94, 60, 120, 1000, 90000):
            ok, why = R.eligible(fps)
            self.assertFalse(ok, fps)
            self.assertIn("fps", why)
        for fps in (None, 0, -1, float("nan"), float("inf"), "x", 5):
            self.assertFalse(R.eligible(fps)[0], fps)

    def test_scale(self):
        self.assertEqual(R.model_scale(1080), 0.5)
        self.assertEqual(R.model_scale(721), 0.5)
        self.assertEqual(R.model_scale(720), 1.0)
        self.assertEqual(R.model_scale(480), 1.0)


class PolicyTest(unittest.TestCase):
    def pol(self, **kw):
        return R.InterpPolicy(30.0, start_lead=6.0, stop_lead=3.0, retry_s=15.0, **kw)

    def test_no_lead_never_interpolates(self):
        p = self.pol()
        for i in range(100):
            self.assertFalse(p.decide(None, i * 0.033))
        self.assertEqual(p.state, "waiting")

    def test_hysteresis(self):
        p = self.pol()
        leads = [1, 3, 5.9, 6.0, 7, 5, 3.1, 3.0, 2.99, 4, 5.99, 6.5]
        got = [p.decide(lead, i) for i, lead in enumerate(leads)]
        self.assertEqual(got, [False, False, False, True, True, True, True, True, False, False, False, True])

    def test_lead_lost_resets_to_waiting(self):
        p = self.pol()
        self.assertTrue(p.decide(10, 0))
        self.assertFalse(p.decide(None, 1))      # a seek: the clock was reset
        self.assertFalse(p.decide(5, 2))         # needs the start lead again
        self.assertTrue(p.decide(6, 3))

    def test_not_ready_copies_but_stays_armed(self):
        p = self.pol()
        self.assertFalse(p.decide(10, 0, ready=False))
        self.assertEqual(p.state, "loading")
        self.assertTrue(p.decide(10, 1, ready=True))
        self.assertEqual(p.state, "on")

    def test_budget_stops_and_retries(self):
        p = self.pol()
        self.assertAlmostEqual(p.budget_ms, R.BUDGET_FRAC * 1000 / 30)
        self.assertTrue(p.decide(10, 0))
        for _ in range(R.MIN_SAMPLES - 1):
            p.note_ms(40.0)
        self.assertTrue(p.decide(10, 1))         # too few samples to judge
        p.note_ms(40.0)
        self.assertFalse(p.decide(10, 2))        # 40 ms > the 30 ms budget
        self.assertEqual(p.state, "budget")
        self.assertFalse(p.decide(10, 10))
        self.assertFalse(p.decide(10, 16.9))
        self.assertTrue(p.decide(10, 17.0))      # RETRY_S after it stopped: measured again
        self.assertIsNone(p.rife_ms)
        for _ in range(R.MIN_SAMPLES):
            p.note_ms(12.0)
        self.assertTrue(p.decide(10, 18))        # fits now

    def test_budget_ok_with_fast_rife(self):
        p = self.pol()
        for _ in range(20):
            p.note_ms(15.0)
        self.assertTrue(p.decide(6, 0))
        self.assertAlmostEqual(p.rife_ms, 15.0)

    def test_median_ignores_outliers(self):
        p = self.pol()
        for ms in (12.0, 13.0, 150.0, 12.5, 14.0, 160.0, 13.0):
            p.note_ms(ms)
        self.assertAlmostEqual(p.rife_ms, 13.0)
        self.assertTrue(p.decide(10, 0))          # two stalls behind Lada's kernels do not stop it
        for _ in range(R.MS_WINDOW):
            p.note_ms(35.0)
        self.assertFalse(p.decide(10, 1))         # a lasting slowdown does (35 ms > 30 ms at 30 fps)

    def test_error_disables(self):
        p = self.pol()
        p.fail("boom")
        self.assertFalse(p.decide(100, 0))
        self.assertEqual(p.snapshot()["state"], "error")

    def test_ratio_window(self):
        p = R.InterpPolicy(10.0, window_s=1.0)    # window = 10 pairs
        self.assertIsNone(p.ratio())
        for _ in range(4):
            p.record("interp")
        p.record("scene")
        self.assertAlmostEqual(p.ratio(), 4 / 5)
        for _ in range(10):
            p.record("copy")
        self.assertEqual(p.ratio(), 0.0)          # the old ones fell out of the window
        for _ in range(3):
            p.record("interp")
        self.assertAlmostEqual(p.ratio(), 0.3)
        self.assertEqual(p.counts, {"interp": 7, "copy": 10, "scene": 1})


class ClockTest(unittest.TestCase):
    def test_playback_clock(self):
        c = R.PlaybackClock()
        self.assertIsNone(c.position(5.0))
        c.update(100.0, True, now=10.0)
        self.assertAlmostEqual(c.position(10.5), 100.5)
        self.assertAlmostEqual(c.position(9.0), 100.0)     # never backwards
        c.update(101.0, False, now=11.0)                   # paused / buffering
        self.assertAlmostEqual(c.position(20.0), 101.0)
        c.reset()
        self.assertIsNone(c.position(20.0))
        self.assertIsNone(c.cache_end())
        c.update(50.0, True, now=0.0, cache_end=58.0)
        self.assertEqual(c.cache_end(), 58.0)

    def test_simulated_playback_start_and_stall(self):
        fps = 10.0
        s = R.SimulatedPlayback(initial_s=1.0, rebuffer_s=2.0)
        self.assertIsNone(s.position(0.0))
        now = 0.0
        # output at 2x realtime: 20 frames per wall second
        for i in range(10):
            now = i * 0.05
            s.on_output(50.0 + i / fps, 1 / fps, now)
        self.assertIsNotNone(s.position(now))              # 1 s ready -> playing from 50.0
        self.assertAlmostEqual(s.position(now), 50.0)
        self.assertAlmostEqual(s.position(now + 0.5), 50.5)
        # no more output: runs dry at 51.0 (video ready up to 50.9 + 0.1)
        self.assertAlmostEqual(s.position(now + 5.0), 51.0)
        self.assertEqual(s.stalls, 1)
        # output resumes; playback restarts once 2 s are ready again
        t = now + 5.0
        for i in range(10, 29):
            s.on_output(50.0 + i / fps, 1 / fps, t)
        self.assertFalse(s.playing)
        s.on_output(52.9, 1 / fps, t + 1.0)
        self.assertTrue(s.playing)
        self.assertAlmostEqual(s.stall_s, (t + 1.0) - (now + 1.0), places=6)
        self.assertIn("1 stall", s.summary(t + 1.0))


class Clip:
    def __init__(self, s, e):
        self.frame_start, self.frame_end = s, e


class RestoreMarginTest(unittest.TestCase):
    """buffer_progress.PipelineProbe.restore_margin: how much of the lead the clips ahead will need."""

    def setUp(self):
        self.saved = BP.RATES.res_fps, BP.RATES.det_fps
        BP.RATES.res_fps, BP.RATES.det_fps = 50.0, 200.0
        self.p = BP.PipelineProbe()
        self.p.reset(0.0)

    def tearDown(self):
        BP.RATES.res_fps, BP.RATES.det_fps = self.saved

    def detect(self, n, mosaic=()):
        for i in range(n):
            self.p._on_detection((1000 + self.p.det, 1 if self.p.det in mosaic else 0))

    def test_nothing_pending(self):
        self.assertIsNone(self.p.restore_margin(0, 30.0, 0.0, 300))     # nothing detected yet
        self.detect(100)
        self.assertIsNone(self.p.restore_margin(10, 30.0, 0.0, 300))    # clean so far
        frozen = BP.PipelineProbe()
        self.assertIsNone(frozen.restore_margin(0, 30.0, 0.0, 300))

    def test_cut_clips_in_order(self):
        self.detect(400, mosaic=set(range(100, 250)) | set(range(300, 350)))
        a, b = Clip(1100, 249 + 1000), Clip(1300, 349 + 1000)
        self.p._on_clip(a)
        self.p._on_clip(b)
        # at frame 0: clip a (150 frames = 3 s of work) starts 100 frames = 3.33 s ahead -> +0.33;
        # clip b (50 more = 4 s total) starts 10 s ahead -> +6: the smallest is +0.33
        self.assertAlmostEqual(self.p.restore_margin(0, 30.0, 0.0, 300), 100 / 30 - 3.0)
        # output already at frame 90: clip a is 10 frames away
        self.assertAlmostEqual(self.p.restore_margin(90, 30.0, 0.0, 300), 10 / 30 - 3.0)
        self.p._on_clip_taken(a)
        t = self.p.pending[id(a)][2]
        self.assertAlmostEqual(self.p.restore_margin(90, 30.0, t + 2.0, 300), 10 / 30 - 1.0)   # 100 of 150 done
        self.p._on_clip_done(a)
        self.assertAlmostEqual(self.p.restore_margin(90, 30.0, t + 3.0, 300), 210 / 30 - 1.0)
        self.p._on_clip_done(b)
        self.assertIsNone(self.p.restore_margin(90, 30.0, t + 3.0, 300))

    def test_clips_with_the_same_range(self):
        # one scene with two mosaic regions gives two clips over the same frames; one being restored (ts float),
        # the other not yet (ts None): sorting must not compare the two (it once crashed the pump: TypeError)
        self.detect(300, mosaic=set(range(100, 250)))
        a, b = Clip(1100, 1249), Clip(1100, 1249)
        self.p._on_clip(a)
        self.p._on_clip(b)
        self.p._on_clip_taken(b)
        t = self.p.pending[id(b)][2]
        self.assertIsNotNone(self.p.restore_margin(0, 30.0, t, 300))
        self.p._on_clip_taken(a)
        self.p._on_clip_done(b)
        self.assertIsNotNone(self.p.restore_margin(0, 30.0, t, 300))

    def test_open_scene_worst_case(self):
        self.detect(160, mosaic=set(range(100, 160)))                   # mosaic scene still running at 159
        m = self.p.restore_margin(0, 30.0, 0.0, 300)
        # 300 frames to restore (6 s) + 240 still to detect (1.2 s), starting 100 frames (3.33 s) ahead
        self.assertAlmostEqual(m, 100 / 30 - 6.0 - 240 / 200)
        self.detect(10)                                                 # the scene ended at 159
        self.assertAlmostEqual(self.p.restore_margin(0, 30.0, 0.0, 300), 100 / 30 - 60 / 50)
        self.p._on_clip(Clip(1100, 1159))                               # cut: the open scene is gone
        self.assertAlmostEqual(self.p.restore_margin(0, 30.0, 0.0, 300), 100 / 30 - 60 / 50)
        self.assertIsNone(self.p.open_start if self.p.open_start > self.p.cut_end else None)

    def test_reset_clears(self):
        self.detect(50, mosaic={10})
        self.p._on_clip(Clip(1010, 1010))
        self.p.reset(5.0)
        self.assertEqual(self.p.pending, {})
        self.detect(5)
        self.assertIsNone(self.p.restore_margin(0, 30.0, 0.0, 300))


class FakeRife:
    def __init__(self, scene_every=0):
        self.calls, self.scene_every = 0, scene_every

    def middle(self, f0, f1):
        self.calls += 1
        if self.scene_every and self.calls % self.scene_every == 0:
            return None
        return ("mid", f0, f1)


class FakeInterpolator:
    def __init__(self, rife=None, error=None):
        self.rife, self.error, self.prepared = rife, error, []

    def prepare(self, w, h):
        self.prepared.append((w, h))

    def get(self, w, h):
        return self.rife

    def error_for(self, w, h):
        return self.error


class DoublerTest(unittest.TestCase):
    def run_frames(self, d, n, t0=10.0, fps=30.0, now0=0.0, wall_per_frame=None):
        out = []
        for i in range(n):
            out += d.push(f"f{i}", f"a{i}", t0 + i / fps, now=now0 + i * (wall_per_frame or 0.0))
        return out

    def test_copies_until_lead_then_interpolates(self):
        clock = R.PlaybackClock()
        interp = FakeInterpolator(FakeRife())
        d = R.FrameDoubler(interp, 30.0, 1920, 1080, clock, log=lambda m: None)
        out = self.run_frames(d, 3)
        self.assertEqual(out, ["a0", "a0", "a1", "a1", "a2"])          # no clock: copies
        self.assertEqual(interp.prepared, [])                           # model not even requested yet
        clock.update(0.0, False, now=0.0)                               # playing at 0, frames at 10 s: lead 10 s
        out = d.push("f3", "a3", 10.1, now=1.0)
        self.assertEqual(interp.prepared, [(1920, 1080)])
        self.assertEqual(out, [("mid", "f2", "f3"), "a3"])
        self.assertEqual(d.finish(), ["a3"])                            # 2N frames in total
        self.assertEqual(d.policy.counts, {"interp": 1, "copy": 2, "scene": 0})

    def test_scene_cut_and_not_ready(self):
        clock = R.PlaybackClock()
        clock.update(0.0, False, now=0.0)
        interp = FakeInterpolator(None)
        d = R.FrameDoubler(interp, 30.0, 1280, 720, clock, log=lambda m: None)
        out = self.run_frames(d, 3)
        self.assertEqual(out, ["a0", "a0", "a1", "a1", "a2"])          # model still loading
        self.assertEqual(d.policy.state, "loading")
        interp.rife = FakeRife(scene_every=2)
        out = d.push("f3", "a3", 11.0) + d.push("f4", "a4", 11.1)
        self.assertEqual(out, [("mid", "f2", "f3"), "a3", "a3", "a4"])  # 2nd pair: scene cut -> copy
        self.assertEqual(d.policy.counts["scene"], 1)

    def test_build_error_disables(self):
        clock = R.PlaybackClock()
        clock.update(0.0, False, now=0.0)
        logs = []
        d = R.FrameDoubler(FakeInterpolator(None, error="RuntimeError('x')"), 30.0, 640, 480, clock, log=logs.append)
        out = self.run_frames(d, 3)
        self.assertEqual(out, ["a0", "a0", "a1", "a1", "a2"])
        self.assertEqual(d.policy.state, "error")

    def test_rife_exception_copies(self):
        class Boom:
            def middle(self, f0, f1):
                raise RuntimeError("cuda")
        clock = R.PlaybackClock()
        clock.update(0.0, False, now=0.0)
        logs = []
        d = R.FrameDoubler(FakeInterpolator(Boom()), 30.0, 640, 480, clock, log=logs.append)
        out = self.run_frames(d, 4)
        self.assertEqual(out, ["a0", "a0", "a1", "a1", "a2", "a2", "a3"])
        self.assertEqual(len(logs), 1)                                  # logged once, then off for the segment

    def test_lead_from_frame_time(self):
        clock = R.PlaybackClock()
        clock.update(100.0, True, now=0.0)
        d = R.FrameDoubler(FakeInterpolator(FakeRife()), 30.0, 640, 480, clock, log=lambda m: None)
        d.push("f0", "a0", 104.0, now=0.0)
        d.push("f1", "a1", 104.0 + 1 / 30, now=0.0)
        self.assertAlmostEqual(d.policy.lead, 4.0 + 1 / 30)
        self.assertFalse(d.policy.active)                               # 4 s < 6 s start lead
        d.push("f2", "a2", 110.0, now=1.0)                              # playing: pos 101 -> lead 9
        self.assertAlmostEqual(d.policy.lead, 9.0)
        self.assertTrue(d.policy.active)
        self.assertIn("interp", d.summary())

    def test_restoration_ahead_reduces_the_lead(self):
        clock = R.PlaybackClock()
        clock.update(0.0, False, now=0.0)
        margin = [None]
        d = R.FrameDoubler(FakeInterpolator(FakeRife()), 30.0, 640, 480, clock, log=lambda m: None,
                           margin=lambda now: margin[0])
        d.push("f0", "a0", 10.0, now=0.0)
        margin[0] = -5.0                                                # a big clip ahead: 10 s lead -> 5 s
        d.push("f1", "a1", 10.0, now=0.0)
        self.assertAlmostEqual(d.policy.lead, 5.0)
        self.assertFalse(d.policy.active)
        margin[0] = 2.0                                                 # restored in time: the lead counts as is
        d.push("f2", "a2", 10.0, now=0.0)
        self.assertAlmostEqual(d.policy.lead, 10.0)
        self.assertTrue(d.policy.active)
        self.assertIn("restore +2.0s", d.summary())

    def test_lead_capped_by_mpv_cache(self):
        """Frames written but held up before mpv (encoder / muxer) do not count beyond IN_FLIGHT_S."""
        clock = R.PlaybackClock()
        clock.update(100.0, False, now=0.0, cache_end=102.0)
        d = R.FrameDoubler(FakeInterpolator(FakeRife()), 30.0, 640, 480, clock, log=lambda m: None)
        d.push("f0", "a0", 112.0, now=0.0)
        d.push("f1", "a1", 112.0, now=0.0)                              # written 12 s ahead, mpv has 2 s
        self.assertAlmostEqual(d.policy.lead, 2.0 + R.IN_FLIGHT_S)
        self.assertFalse(d.policy.active)
        clock.update(100.0, False, now=1.0, cache_end=111.0)
        d.push("f2", "a2", 112.0, now=1.0)
        self.assertAlmostEqual(d.policy.lead, 12.0)
        self.assertTrue(d.policy.active)

    def test_missing_pts(self):
        clock = R.PlaybackClock()
        d = R.FrameDoubler(FakeInterpolator(FakeRife()), 25.0, 640, 480, clock, log=lambda m: None)
        d.push("f0", "a0", 5.0)
        d.push("f1", "a1", None)
        self.assertAlmostEqual(d.t_last, 5.04)

    def test_simulated_end_to_end(self):
        """Restoration at 1.5x realtime with RIFE costing 30 % of the wall time: interpolation kicks in after the lead
        builds up, the viewer never stalls."""
        fps = 30.0
        sim = R.SimulatedPlayback(1.0, 5.0)
        d = R.FrameDoubler(FakeInterpolator(FakeRife()), fps, 1920, 1080, sim, log=lambda m: None)
        now, n = 0.0, 0
        while n < 3000:
            t = 100.0 + n / fps
            interp_before = d.policy.counts["interp"]
            d.push(f"f{n}", f"a{n}", t, now=now)
            now += 1 / (1.5 * fps)
            if d.policy.counts["interp"] > interp_before:
                d.policy.note_ms(15.0)
                now += 0.015
            n += 1
        self.assertEqual(sim.stalls, 0)
        self.assertGreater(d.policy.counts["interp"], 1000)
        self.assertGreater(d.policy.counts["copy"], 100)


class RemoteClockTest(unittest.TestCase):
    def test_stale_means_nobody_watching(self):
        c = R.RemoteClock()
        self.assertIsNone(c.position(0.0))
        self.assertFalse(c.fresh(0.0))
        c.update(100.0, True, now=10.0)
        self.assertAlmostEqual(c.position(12.0), 102.0)
        self.assertTrue(c.fresh(10.0 + R.WATCH_STALE_S))
        self.assertIsNone(c.position(10.0 + R.WATCH_STALE_S + 0.1))
        self.assertFalse(c.fresh(10.0 + R.WATCH_STALE_S + 0.1))
        c.update(110.0, False, now=20.0)                                # a new update revives it; paused: no advance
        self.assertEqual(c.position(23.0), 110.0)
        self.assertEqual(R.RemoteClock(stale_s=1.0).stale_s, 1.0)


class WatchedDoublerTest(unittest.TestCase):
    FPS = 30.0

    def make(self, clock, scene_every=0, **kw):
        self.logs = []
        self.rife = FakeRife(scene_every=scene_every)
        return R.WatchedDoubler(FakeInterpolator(self.rife), self.FPS, 640, 480, clock, log=self.logs.append, **kw)

    def push(self, d, i, t, now=0.0):
        """Frame i at source time t; returns what is written for it (the pair's middle slot, then the frame)."""
        return d.push(f"f{i}", f"a{i}", t, now=now)

    def test_no_clock_all_interpolated(self):
        d = self.make(R.RemoteClock())                                  # never fed
        out = []
        for i in range(5):
            out += self.push(d, i, 10.0 + i / self.FPS, now=100.0 + i)
        self.assertEqual(out[:3], ["a0", ("mid", "f0", "f1"), "a1"])
        self.assertEqual(sum(1 for o in out if isinstance(o, tuple)), 4)
        self.assertEqual(d.copied, 0)
        self.assertEqual(d.finish(), ["a4"])
        self.assertEqual(self.logs, [])
        self.assertNotIn("copied", d.summary())

    def test_viewer_close_copies_and_hysteresis(self):
        clock = R.RemoteClock()
        clock.update(100.0, False, now=0.0)                             # paused viewer at 100 s
        d = self.make(clock)
        self.push(d, 0, 125.0)
        self.assertEqual(self.push(d, 1, 125.0), [("mid", "f0", "f1"), "a1"])   # lead 25 s: interpolate
        self.assertEqual(d.copied, 0)
        self.assertEqual(self.push(d, 2, 112.0), [("mid", "f1", "f2"), "a2"])   # 12 s: between stop and start
        out = self.push(d, 3, 109.9)                                    # lead 9.9 s < 10: copy
        self.assertEqual(out, ["a2", "a3"])
        self.assertEqual(d.copied, 1)
        self.assertEqual(len(self.logs), 1)
        self.assertIn("viewer close (lead 9.9s) -> copies", self.logs[0])
        self.assertEqual(self.push(d, 4, 115.0), ["a3", "a4"])          # 15 s: still copying (below start_lead)
        self.assertEqual(self.push(d, 5, 119.99), ["a4", "a5"])
        self.assertEqual(d.copied, 3)
        self.assertEqual(len(self.logs), 1)                             # no log without a state change
        out = self.push(d, 6, 120.0)                                    # back at start_lead (20 s) exactly
        self.assertEqual(out, [("mid", "f5", "f6"), "a6"])
        self.assertEqual(d.copied, 3)
        self.assertEqual(len(self.logs), 2)
        self.assertIn("lead 20.0s -> every frame again", self.logs[1])
        self.assertEqual(self.push(d, 7, 112.0), [("mid", "f6", "f7"), "a7"])   # hysteresis: stays interpolating
        self.assertIn("3 copied", d.summary())

    def test_custom_leads(self):
        clock = R.RemoteClock()
        clock.update(0.0, False, now=0.0)
        d = self.make(clock, start_lead=5.0, stop_lead=2.0)
        self.push(d, 0, 10.0)
        self.assertEqual(self.push(d, 1, 1.9), ["a0", "a1"])
        self.assertEqual(self.push(d, 2, 4.9), ["a1", "a2"])
        self.assertEqual(self.push(d, 3, 5.0), [("mid", "f2", "f3"), "a3"])
        self.assertEqual(d.copied, 2)

    def test_behind_s(self):
        clock = R.RemoteClock()
        clock.update(100.0, False, now=0.0)
        d = self.make(clock, behind_s=8.0)
        self.push(d, 0, 105.0)
        self.assertEqual(self.push(d, 1, 105.0), ["a0", "a1"])          # lead 5 s: copy
        self.assertEqual(d.copied, 1)
        self.assertEqual(self.push(d, 2, 92.0), ["a1", "a2"])           # lead -8.0 is not < -8.0: viewer is close
        self.assertEqual(d.copied, 2)
        out = self.push(d, 3, 91.9)                                     # lead -8.1: the viewer is past this part
        self.assertEqual(out, [("mid", "f2", "f3"), "a3"])
        self.assertEqual(d.copied, 2)
        self.assertIn("every frame again", self.logs[-1])
        self.assertEqual(self.push(d, 4, 95.0), ["a3", "a4"])           # back within reach (lead -5 s): copies again
        self.assertEqual(d.copied, 3)

    def test_stale_clock_interpolates(self):
        clock = R.RemoteClock()
        clock.update(100.0, True, now=0.0)
        d = self.make(clock)
        self.push(d, 0, 105.0, now=0.0)
        self.assertEqual(self.push(d, 1, 105.0, now=0.0), ["a0", "a1"])     # lead 5 s: copy
        self.assertEqual(self.push(d, 2, 105.0, now=4.9)[0], "a1")          # still fresh (pos 104.9)
        self.assertEqual(d.copied, 2)
        out = self.push(d, 3, 105.0, now=5.1)                           # the player stopped reporting
        self.assertEqual(out, [("mid", "f2", "f3"), "a3"])
        self.assertEqual(d.copied, 2)
        self.assertIn("nobody watching", self.logs[-1])
        clock.update(104.0, True, now=6.0)                              # a viewer is back, close again
        self.assertEqual(self.push(d, 4, 105.0, now=6.0), ["a3", "a4"])
        self.assertEqual(d.copied, 3)

    def test_scene_cut_repeats_are_not_copied(self):
        clock = R.RemoteClock()
        clock.update(0.0, False, now=0.0)
        d = self.make(clock, scene_every=2)                             # every 2nd interpolated pair is a scene cut
        out = []
        for i in range(5):
            out += self.push(d, i, 100.0 + i / self.FPS)                # lead 100 s: always interpolating
        self.assertEqual(out, ["a0", ("mid", "f0", "f1"), "a1", "a1", "a2", ("mid", "f2", "f3"), "a3", "a3", "a4"])
        self.assertEqual(d.copied, 0)
        self.assertEqual(d.policy.counts["scene"], 2)
        self.assertNotIn("copied", d.summary())
        clock.update(99.0, False, now=1.0)                              # a close viewer: policy copies are counted
        self.push(d, 5, 100.0 + 5 / self.FPS, now=1.0)
        self.assertEqual(d.copied, 1)
        self.assertEqual(d.policy.counts["scene"], 2)

    def test_first_push_has_no_pair(self):
        clock = R.RemoteClock()
        clock.update(0.0, False, now=0.0)
        d = self.make(clock)
        self.assertEqual(self.push(d, 0, 1.0), ["a0"])
        self.assertEqual(d.copied, 0)

    def test_same_contract_as_full_doubler(self):
        d = self.make(R.RemoteClock())
        self.assertEqual((d.rate_mult, d.layers), (2, 1))
        self.assertIsInstance(d, R.FullDoubler)
        self.assertIn("interp", d.summary())


if __name__ == "__main__":
    unittest.main(verbosity=1)
