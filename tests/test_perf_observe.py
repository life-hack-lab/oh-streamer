"""PerfObserver (queue worker throughput / density observations): pure logic, fake clock, no GPU."""
import types
import unittest

from oh_streamer.queue.perf_observe import PerfObserver


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(**kw):
    clock = Clock()
    kw.setdefault("warmup_s", 15.0)
    kw.setdefault("min_frames", 1500)
    kw.setdefault("interval_s", 300.0)
    return PerfObserver(clock=clock, **kw), clock


def run_frames(obs, clock, n, fps, detect=None):
    """Push n frames at `fps`; `detect` = mosaics per consumed detection entry. Returns the emitted observations."""
    out = []
    for _ in range(n):
        clock.t += 1.0 / fps
        if detect is not None:
            obs.note_detection(detect)
        got = obs.frame()
        if got:
            out.append(got)
    return out


class MeasureTests(unittest.TestCase):
    def test_fps_and_density_after_warmup(self):
        obs, clock = make()
        obs.begin_run("full", 720, 29.97)
        run_frames(obs, clock, 15 * 20, 20.0, detect=1)          # warm-up (15 s at 20 fps) + a bit
        run_frames(obs, clock, 2000, 20.0, detect=2)
        result = obs.finish()
        self.assertEqual(result["type"], "perf_obs")
        self.assertEqual((result["mode"], result["height"], result["src_fps"]), ("full", 720, 29.97))
        self.assertAlmostEqual(result["fps"], 20.0, places=1)
        self.assertGreaterEqual(result["frames"], 1900)
        self.assertAlmostEqual(result["density"], 2.0, places=1)       # the warm-up's density is not counted

    def test_first_seconds_are_not_measured(self):
        obs, clock = make()
        obs.begin_run("off", 480, 30.0)
        run_frames(obs, clock, 3000, 400.0, detect=0)             # 7.5 s: all warm-up
        self.assertIsNone(obs.finish())

    def test_too_few_frames_gives_nothing(self):
        obs, clock = make()
        obs.begin_run("off", 480, 30.0)
        run_frames(obs, clock, 15 * 50, 50.0, detect=1)
        run_frames(obs, clock, 1000, 50.0, detect=1)
        self.assertIsNone(obs.finish())
        run_frames(obs, clock, 600, 50.0, detect=1)               # the window keeps growing past the minimum
        self.assertIsNotNone(obs.finish())

    def test_periodic_emit_every_interval_of_steady_processing(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        got = run_frames(obs, clock, 15 * 10 + 10 * 300 * 2 + 50, 10.0, detect=1)
        self.assertEqual(len(got), 2)                              # 300 s windows at 10 fps = 3000 frames each
        for o in got:
            self.assertAlmostEqual(o["fps"], 10.0, places=1)
            self.assertAlmostEqual(o["frames"], 3000, delta=2)
        self.assertIsNone(obs.finish())                            # the window was cleared after each emit

    def test_stall_is_left_out_of_the_elapsed_time(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        run_frames(obs, clock, 15 * 30, 30.0, detect=1)
        run_frames(obs, clock, 1000, 30.0, detect=1)
        clock.t += 600.0                                           # held / system asleep
        run_frames(obs, clock, 1000, 30.0, detect=1)
        result = obs.finish()
        self.assertAlmostEqual(result["fps"], 30.0, delta=0.5)

    def test_discard_restarts_warmup(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 2000, 30.0, detect=1)
        obs.discard()
        self.assertIsNone(obs.finish())
        run_frames(obs, clock, 15 * 30 + 1000, 30.0, detect=1)    # warm-up again, then too few
        self.assertIsNone(obs.finish())

    def test_taint_drops_the_rest_of_the_run_but_not_the_next_one(self):
        obs, clock = make()
        obs.begin_run("full", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 2000, 30.0, detect=1)
        obs.taint()
        self.assertEqual(run_frames(obs, clock, 20000, 30.0, detect=1), [])
        self.assertIsNone(obs.finish())
        obs.begin_run("full", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 2000, 30.0, detect=1)
        self.assertIsNotNone(obs.finish())

    def test_each_run_has_its_own_warmup_and_window(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 2000, 30.0, detect=1)
        first = obs.finish()
        obs.begin_run("off", 720, 30.0)
        clock.t += 5.0
        run_frames(obs, clock, 15 * 30 + 2000, 15.0, detect=3)
        second = obs.finish()
        self.assertAlmostEqual(first["fps"], 30.0, delta=0.5)
        self.assertAlmostEqual(second["fps"], 15.0, delta=0.5)
        self.assertAlmostEqual(second["density"], 3.0, places=1)

    def test_unobserved_modes_and_bad_parameters(self):
        for mode, h, fps in (("auto", 720, 30.0), (None, 720, 30.0), ("full", 0, 30.0), ("off", 720, 0.0),
                             ("off", 720, float("nan"))):
            obs, clock = make()
            obs.begin_run(mode, h, fps)
            run_frames(obs, clock, 15 * 30 + 3000, 30.0, detect=1)
            self.assertIsNone(obs.finish(), (mode, h, fps))

    def test_no_detection_counts_no_observation(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 3000, 30.0)               # nothing ever counted
        self.assertIsNone(obs.finish())

    def test_zero_density_is_a_valid_observation(self):
        obs, clock = make()
        obs.begin_run("off", 720, 30.0)
        run_frames(obs, clock, 15 * 30 + 3000, 30.0, detect=0)
        self.assertEqual(obs.finish()["density"], 0.0)


class QueueHookTests(unittest.TestCase):
    def test_wrapped_get_counts_tuples_and_passes_everything_through(self):
        marker = object()
        entries = iter([(0, 2), (1, 0), marker, (2, 3), (3, "bad")])
        q = types.SimpleNamespace(get=lambda *a, **kw: next(entries))
        obs, clock = make(warmup_s=0.0, min_frames=1)
        self.assertTrue(obs.wrap_queue_get(q))
        obs.begin_run("off", 720, 30.0)
        got = [q.get() for _ in range(5)]
        self.assertIs(got[2], marker)
        self.assertEqual(got[0], (0, 2))
        run_frames(obs, clock, 3, 30.0)                            # frame 1 = run start, frame 2 starts measuring
        obs._det0 = (0, 0)
        result = obs.finish()
        self.assertEqual(result["density"], round(5 / 3, 4))      # 3 valid entries, 5 mosaics

    def test_wrap_fails_softly(self):
        obs, _ = make()
        self.assertFalse(obs.wrap_queue_get(None))
        self.assertFalse(obs.wrap_queue_get(types.SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
