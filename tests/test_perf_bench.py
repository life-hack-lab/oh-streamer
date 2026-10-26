"""Perf bench: pure parts only (arg parsing, result parsing, profile assembly, synthetic box schedule). No GPU/torch."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import json
import unittest

from oh_streamer.perf import bench as B
from oh_streamer.perf import profile as P


def parse(argv):
    ap = B.build_run_parser(argparse.ArgumentParser())
    return B.options_from_args(ap.parse_args(argv))


class ArgsTests(unittest.TestCase):
    def test_defaults(self):
        o = B.normalize_options(parse([]))
        self.assertEqual(o["mode"], "default")
        self.assertEqual(o["heights"], [480, 720, 1080])
        self.assertEqual(o["device"], "auto")

    def test_quick_and_full(self):
        self.assertEqual(B.normalize_options(parse(["--quick"]))["heights"], [720])
        self.assertEqual(parse(["--full"])["mode"], "full")
        with self.assertRaises(SystemExit):
            parse(["--quick", "--full"])

    def test_options(self):
        o = B.normalize_options(parse(["--heights", "720p,1080", "--timeout", "12", "--keep-temp", "--out", "x.json",
                                       "--device", "cuda:1"]))
        self.assertEqual(o["heights"], [720, 1080])
        self.assertEqual(o["timeout"], 12.0)
        self.assertTrue(o["keep_temp"])
        self.assertEqual((o["out"], o["device"]), ("x.json", "cuda:1"))

    def test_bad_values(self):
        for bad in ("abc", "", "721", "100", "720,,x"):
            with self.assertRaises(ValueError, msg=bad):
                B.parse_heights(bad)
        with self.assertRaises(ValueError):
            B.normalize_options({"timeout": 0})
        with self.assertRaises(ValueError):
            B.normalize_options({"mode": "bogus"})

    def test_heights_dedupe_and_widths(self):
        self.assertEqual(B.parse_heights("720,720,480"), [720, 480])
        self.assertEqual([B.width_for(h) for h in (480, 720, 1080)], [854, 1280, 1920])
        self.assertEqual(B.width_for(360) % 2, 0)

    def test_plan_steps(self):
        def kinds(m):
            return [(s["kind"], s.get("family"), s.get("densities")) for s in B.plan_steps({"mode": m})]
        self.assertEqual(kinds("quick"), [("restore", None, [0, 1]), ("rife", None, None), ("encode", "hevc", None)])
        self.assertEqual(B.plan_steps({"mode": "quick"})[0]["d0_heights"], [1080])
        self.assertNotIn("d0_heights", B.plan_steps({"mode": "quick", "heights": [1080]})[0])
        self.assertTrue(all("detector" not in s for s in B.plan_steps({"mode": "quick"})))
        default = B.plan_steps({"mode": "default", "detector": "v4-accurate"})
        restores = [s for s in default if s["kind"] == "restore"]
        self.assertEqual([(s["height"], s["densities"], s["d0_heights"], s.get("detector")) for s in restores],
                         [(720, [0, 1], [480, 1080], None), (720, [0], [480, 1080], "v4-fast")])
        self.assertEqual(default[-1]["kind"], "combined")
        self.assertEqual(B.plan_steps({"mode": "default", "heights": [480, 1080]})[0]["height"], 1080)
        self.assertTrue(all(s.get("detector") != "v4-fast"
                            for s in B.plan_steps({"mode": "default", "detector": "v4-fast"})))
        full = B.plan_steps({"mode": "full"})
        self.assertEqual(full[0]["densities"], [0, 1, 2])
        self.assertEqual(len([s for s in full if s["kind"] == "restore" and not s.get("detector")]), 3)
        self.assertEqual({s.get("family") for s in full if s["kind"] == "encode"}, {"hevc", "h264", "libx264"})
        steps = B.plan_steps({"mode": "default", "heights": [480]})
        self.assertEqual([s for s in steps if s["kind"] == "combined"][0]["height"], 480)

    def test_source_and_child_commands(self):
        cmd = B.source_cmd(1280, 720, "o.mp4", 122)
        self.assertIn("testsrc2=size=1280x720:rate=30:duration=122", cmd[cmd.index("-i") + 1])
        c = B.child_command("rife", {"heights": [720]})
        self.assertEqual(c[1:4], ["-m", "oh_streamer.perf", "child"])
        self.assertEqual(json.loads(c[4]), {"heights": [720], "kind": "rife"})


class SourceLengthTests(unittest.TestCase):
    def test_detector_lead_follows_lada_clip_queue(self):
        self.assertEqual(B.detector_lead_frames(180), (11 + 2) * 180)    # 512 MiB // (180*256*256*4) = 11 clips
        self.assertEqual(B.detector_lead_frames(360), (5 + 2) * 360)
        self.assertEqual(B.detector_lead_frames(100000), (1 + 2) * 100000)

    def test_source_outlasts_window_plus_lead(self):
        for L in (30, 180, 240, 360):
            frames = B.source_seconds(L) * B.FPS
            window_end = (B.WARMUP_CLIPS + B.WINDOW_MIN_PERIODS) * L
            self.assertGreaterEqual(frames, window_end + B.detector_lead_frames(L) + B.SOURCE_MARGIN_FRAMES)
        self.assertGreater(B.source_seconds(360), B.source_seconds(180))

    def test_detector_done_early(self):
        w = B.SteadyWindow(0, 10, min_s=0.0, min_periods=1)
        self.assertFalse(B.detector_done_early(w, {"t": 1.0}))           # window not finished
        for i in range(11):
            w.update(i, float(i))
        self.assertTrue(B.detector_done_early(w, {"t": 5.0}))
        self.assertFalse(B.detector_done_early(w, {"t": 10.5}))
        self.assertFalse(B.detector_done_early(w, {"t": None}))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ProgressTests(unittest.TestCase):
    def test_step_estimates_grow_with_clip_length(self):
        def secs(L):
            return {s["kind"]: s["est_s"] for s in B.plan_steps({"mode": "default", "clip_length": L})
                    if not s.get("detector")}
        short, long = secs(180), secs(240)
        self.assertGreater(long["restore"], short["restore"])
        self.assertGreater(long["combined"], short["combined"])
        self.assertEqual(long["rife"], short["rife"])
        quick = B.plan_steps({"mode": "quick"})
        self.assertGreater(quick[0]["est_s"], B.plan_steps({"mode": "quick", "heights": [1080]})[0]["est_s"])

    def test_running_step_advances_and_holds_below_its_end(self):
        c = FakeClock()
        p = B.ProgressClock([10, 30], clock=c)
        self.assertEqual(p.status(), (0.0, None))
        p.start(0)
        c.t = 5
        self.assertAlmostEqual(p.status()[0], 5 / 40)
        c.t = 50                                      # far over its estimate: stops at PROGRESS_HOLD of it
        self.assertAlmostEqual(p.status()[0], B.PROGRESS_HOLD * 10 / 40)
        self.assertIsNone(p.status()[1])              # nothing finished yet: no remaining time

    def test_finished_steps_rescale_the_rest(self):
        c = FakeClock()
        p = B.ProgressClock([10, 30], clock=c)
        p.start(0)
        c.t = 20                                      # took twice the estimate
        self.assertEqual(p.finish(0), 20)
        self.assertEqual(p.scale(), 2.0)
        frac, left = p.status()
        self.assertAlmostEqual(frac, 20 / 80)
        self.assertAlmostEqual(left, 60)
        p.start(1)
        c.t = 50
        frac, left = p.status()
        self.assertAlmostEqual(frac, 50 / 80)
        self.assertAlmostEqual(left, 30)

    def test_skipped_steps_do_not_count_as_measured(self):
        c = FakeClock()
        p = B.ProgressClock([10, 5, 10], clock=c)
        p.start(0)
        c.t = 10
        p.finish(0)
        p.start(1)
        p.finish(1, skipped=True)                     # e.g. the same encoder as an earlier family
        self.assertEqual(p.scale(), 1.0)
        self.assertAlmostEqual(p.status()[0], 10 / 20)

    def test_scale_is_limited(self):
        c = FakeClock()
        p = B.ProgressClock([10, 10], clock=c)
        p.start(0)
        c.t = 1000
        p.finish(0)
        self.assertEqual(p.scale(), B.PROGRESS_SCALE_LIMITS[1])

    def test_run_reports_time_based_progress_and_remaining(self):
        import tempfile
        from unittest import mock
        calls = []

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(B, "run_child", lambda kind, payload, **_kw: (None, "stub")), \
                    mock.patch.object(B, "ensure_source", lambda h, tmp, *a, **k: tmp / f"src{h}.mp4"), \
                    mock.patch.object(B, "read_system_conditions",
                                      lambda device="auto": {"conditions": {}, "fingerprint": {}}), \
                    mock.patch.object(B, "_resolve_family", lambda family: ("libx265", None)):
                B.run_benchmark({"mode": "quick", "workdir": str(d)},
                                progress=lambda msg, frac, remaining=None: calls.append((msg, frac, remaining)))
        self.assertEqual(calls[0][2], None)                       # before the first step is timed
        done = [c for c in calls if c[0].startswith("restore done")]
        self.assertEqual(len(done), 1)
        self.assertIsNotNone(done[0][2])
        self.assertEqual(calls[-1][:2], ("finished", 1.0))
        fracs = [f for _, f, _ in calls]
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fracs))


class ParseResultTests(unittest.TestCase):
    def test_last_valid_line_wins(self):
        text = 'noise\nPERF_RESULT {"a": 1}\nPERF_RESULT not json\nlog PERF_RESULT {"a": 2}\nmore\n'
        self.assertEqual(B.parse_result(text), {"a": 2})

    def test_none(self):
        self.assertIsNone(B.parse_result(""))
        self.assertIsNone(B.parse_result("PERF_RESULT [1,2]\nPERF_RESULT {bad"))


class BoxScheduleTests(unittest.TestCase):
    def test_overlap_and_bounds(self):
        for (w, h) in ((854, 480), (1280, 720), (1920, 1080)):
            for d in (1, 2):
                for f in range(0, 400, 3):
                    boxes = B.region_boxes(f, d, w, h)
                    self.assertEqual(len(boxes), d)
                    for t, l, b, r in boxes:
                        self.assertTrue(0 <= t < b <= h and 0 <= l < r <= w, (w, h, d, f, (t, l, b, r)))
                        self.assertAlmostEqual((b - t) / h, 0.30, delta=0.01)
                    for i in range(d):
                        for j in range(i + 1, d):
                            self.assertFalse(B.boxes_overlap(boxes[i], boxes[j]), (w, h, d, f))

    def test_density_zero_and_too_dense(self):
        self.assertEqual(B.region_boxes(5, 0, 1280, 720), [])
        with self.assertRaises(ValueError):
            B.region_boxes(0, 40, 1280, 720)

    def test_slow_movement_keeps_scene_continuity(self):
        for (w, h) in ((854, 480), (1920, 1080)):
            for d in (1, 2):
                prev = B.region_boxes(0, d, w, h)
                moved = False
                for f in range(1, 800):
                    cur = B.region_boxes(f, d, w, h)
                    for p, c in zip(prev, cur):
                        self.assertTrue(B.boxes_overlap(p, c))
                        self.assertLess(abs(c[1] - p[1]), 0.02 * w)
                        moved |= c != p
                    prev = cur
                self.assertTrue(moved)

    def test_triangle(self):
        self.assertEqual([B._triangle(i, 3) for i in range(8)], [0, 1, 2, 3, 2, 1, 0, 1])
        self.assertEqual(B._triangle(5, 0), 0)


class SteadyWindowTests(unittest.TestCase):
    def test_aligned_window(self):
        w = B.SteadyWindow(10, 10, min_s=1.0, cap_s=100.0, aligned=True, min_frames=5)
        done = False
        for i in range(200):
            done = w.update(i, i * 0.1)
            if done:
                break
        r = w.result()
        self.assertTrue(done)
        self.assertEqual(r["frames"] % 10, 0)
        self.assertAlmostEqual(r["fps"], 10.0)
        self.assertFalse(r["capped"])

    def test_aligned_window_spans_min_periods(self):
        # min_s is long past after one period; the window still waits for min_periods whole clips
        w = B.SteadyWindow(10, 10, min_s=0.1, cap_s=100.0, aligned=True, min_frames=5, min_periods=3)
        for i in range(200):
            if w.update(i, i * 0.1):
                break
        r = w.result()
        self.assertEqual(r["frames"], 30)
        self.assertAlmostEqual(r["fps"], 10.0)
        # an unaligned window (d=0: no clips) is not held back by min_periods
        w = B.SteadyWindow(10, 10, min_s=0.5, cap_s=100.0, aligned=False, min_frames=5, min_periods=3)
        for i in range(200):
            if w.update(i, i * 0.1):
                break
        self.assertEqual(w.result()["frames"], 5)

    def test_aligned_window_uses_median_period(self):
        # clips arrive as bursts of 10 frames every 1 s, except one clip that a transient slowed to 3 s
        w = B.SteadyWindow(10, 10, min_s=0.1, cap_s=100.0, aligned=True, min_frames=5, min_periods=3)
        t, idx = 0.0, 0
        done = False
        for clip in range(1, 10):
            t += 3.0 if clip == 3 else 1.0       # clip 2 starts the window (start_idx 10)
            for _ in range(10):
                done = w.update(idx, t)
                idx += 1
                if done:
                    break
            if done:
                break
        r = w.result()
        self.assertEqual(r["frames"], 30)
        self.assertAlmostEqual(r["fps"], 10.0)                 # median clip: 10 frames per 1 s
        self.assertAlmostEqual(r["fps_mean"], 30 / 5.0)        # the plain rate is dragged down by the slow clip

    def test_cap_and_eof(self):
        w = B.SteadyWindow(0, 10, min_s=100, cap_s=2.0)
        for i in range(100):
            if w.update(i, i * 0.5):
                break
        self.assertTrue(w.result()["capped"])
        w = B.SteadyWindow(5, 10, min_s=100, cap_s=100, min_frames=20)
        for i in range(30):
            w.update(i, i * 0.1)
        self.assertIsNone(w.result())
        w.finish_eof(29, 2.9)
        self.assertAlmostEqual(w.result()["fps"], 10.0)

    def test_capped_aligned_window_ends_on_a_whole_clip(self):
        # a clip of 10 frames starts every 1 s and takes 0.72 s to deliver; the cap hits inside a burst
        w = B.SteadyWindow(10, 10, min_s=0.1, cap_s=2.5, aligned=True, min_frames=5, min_periods=3)
        idx, done = 0, False
        for clip in range(1, 10):
            for k in range(10):
                done = w.update(idx, clip + k * 0.08)
                idx += 1
                if done:
                    break
            if done:
                break
        r = w.result()
        self.assertTrue(r["capped"])
        self.assertEqual(r["frames"], 20)                       # not 20 + part of the third clip
        self.assertAlmostEqual(r["fps"], 10.0)
        self.assertEqual(r["period_s"], [1.0, 1.0])

    def test_measure_clip_length_is_capped(self):
        self.assertEqual(B.measure_clip_length(180), 180)
        self.assertEqual(B.measure_clip_length(300), B.MEASURE_CLIP_MAX)
        self.assertEqual(P.MEASURE_CLIP_MAX, B.MEASURE_CLIP_MAX)
        self.assertEqual(B.plan_steps({"clip_length": 360}), B.plan_steps({"clip_length": B.MEASURE_CLIP_MAX}))

    def test_too_short(self):
        w = B.SteadyWindow(5, 10, min_frames=90)
        for i in range(20):
            w.update(i, i * 0.1)
        w.finish_eof(19, 1.9)
        self.assertIsNone(w.result())


class AssemblyTests(unittest.TestCase):
    ENV = {"gpu": "RTX 5060", "backend": "cuda", "device": "cuda:0", "torch": "2.9.1", "lada": "0.9", "fp16": True,
           "vram_total_gib": 8.0}

    def test_full_profile_validates_and_formats(self):
        p = B.new_profile()
        B.merge_restore(p, 720, {"env": self.ENV, "densities": {
            "0": {"fps": 140.04, "vram_peak_gib": 1.5, "frames": 180, "window_s": 1.3, "capped": False},
            "1": {"fps": 48.26, "vram_peak_gib": 2.12, "frames": 180, "window_s": 3.7, "capped": False}}},
            densities=(0, 1))
        B.merge_rife(p, [720, 1080], {"env": self.ENV, "heights": {
            "720": {"pair_ms": 9.456, "mode": "CUDA graph", "pairs": 60},
            "1080": {"pair_ms": None, "error": "OOM"}}})
        B.merge_encode(p, "hevc_nvenc", [720], {"heights": {"720": {"fps": 420.04, "error": None}}})
        B.merge_combined(p, 720, "hevc_nvenc", {"env": self.ENV, "fps": 30.14, "frames": 180})
        self.assertEqual(p["restore"]["720"], {"d0_fps": 140.0, "d1_fps": 48.3, "d2_fps": None,
                                               "vram_peak_gib": 2.12, "error": None})
        self.assertEqual(p["rife"]["720"], {"pair_ms": 9.46, "error": None})
        self.assertEqual(p["rife"]["1080"], {"pair_ms": None, "error": "OOM"})
        self.assertEqual(p["encode"]["hevc_nvenc"]["720"]["fps"], 420.0)
        self.assertEqual(p["combined"]["fps"], 30.1)
        self.assertEqual((p["fingerprint"]["gpu"], p["settings"]["fp16"]), ("RTX 5060", True))
        self.assertEqual(P.validate(json.loads(json.dumps(p))), [])
        self.assertIn("720p", P.format_table(p))

    def test_failures_become_nulls_with_error(self):
        p = B.new_profile()
        B.merge_restore(p, 1080, None, "timed out after 5s", (0, 1))
        self.assertEqual(p["restore"]["1080"]["d1_fps"], None)
        self.assertEqual(p["restore"]["1080"]["error"], "timed out after 5s")
        B.merge_restore(p, 720, {"densities": {"0": {"fps": 100.0}, "1": {"fps": None, "error": "boom"}}}, None, (0, 1))
        e = p["restore"]["720"]
        self.assertEqual((e["d0_fps"], e["d1_fps"]), (100.0, None))
        self.assertIn("d1: boom", e["error"])
        B.merge_restore(p, 480, {"densities": {"0": {"fps": 1.0}}}, None, (0, 1))
        self.assertIn("d1: no measurement", p["restore"]["480"]["error"])
        B.merge_rife(p, [720], None, "child exited with status 1")
        self.assertEqual(p["rife"]["720"], {"pair_ms": None, "error": "child exited with status 1"})
        B.merge_encode(p, "libx264", [720, 1080], {"heights": {"720": {"fps": 50.0}}})
        self.assertIsNone(p["encode"]["libx264"]["1080"]["fps"])
        self.assertTrue(p["encode"]["libx264"]["1080"]["error"])
        B.merge_combined(p, 720, "libx265", None, "timed out after 5s")
        self.assertEqual(p["combined"]["fps"], None)
        self.assertEqual(p["combined"]["error"], "timed out after 5s")
        self.assertEqual(P.validate(json.loads(json.dumps(p))), [])
        P.format_table(p)                       # must not raise on a mostly failed profile

    def test_detector_done_early_becomes_warning(self):
        p = B.new_profile()
        B.merge_restore(p, 720, {"densities": {"0": {"fps": 300.0, "detector_done_early": False},
                                               "1": {"fps": 55.0, "detector_done_early": True}}}, None, (0, 1))
        B.merge_combined(p, 720, "hevc_nvenc", {"fps": 36.0, "frames": 540, "detector_done_early": True})
        self.assertEqual(p["restore"]["720"]["d1_fps"], 55.0)
        self.assertTrue(p["detail"]["restore"]["720"]["d1"]["detector_done_early"])
        self.assertEqual(len(p["warnings"]), 2)
        self.assertTrue(all("too high" in w for w in p["warnings"]))
        self.assertIn("720p d=1", p["warnings"][0])

    def test_warnings(self):
        self.assertEqual(B.build_warnings({"on_ac_power": True, "gpu_util_before_pct": 3.0}), [])
        w = B.build_warnings({"on_ac_power": False, "gpu_util_before_pct": 55.0})
        self.assertEqual(len(w), 2)
        self.assertIn("battery", w[0])
        self.assertIn("busy", w[1])
        self.assertEqual(B.build_warnings({"on_ac_power": None, "gpu_util_before_pct": None}), [])

    def test_summary(self):
        p = B.new_profile()
        B.merge_restore(p, 720, None, "x", (0, 1))
        self.assertIn("720", B.format_summary(p))


class RunChildTests(unittest.TestCase):
    """run_child against tiny stand-in child commands (monkeypatched child_command)."""

    def _run(self, code, timeout=20.0, cancel=None):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.object(B, "child_command", lambda kind, payload: [sys.executable, "-c", code]):
                return B.run_child("x", {}, timeout=timeout, log_dir=d, workdir=d, cancel=cancel)

    def test_ok(self):
        res, err = self._run('import json;print("PERF_RESULT "+json.dumps({"fps":1.5}))')
        self.assertEqual((res, err), ({"fps": 1.5}, None))

    def test_crash(self):
        res, err = self._run('import sys;sys.stderr.write("boom\\n");sys.exit(3)')
        self.assertIsNone(res)
        self.assertIn("status 3", err)
        self.assertIn("boom", err)

    def test_timeout(self):
        res, err = self._run("import time;time.sleep(60)", timeout=1.0)
        self.assertIsNone(res)
        self.assertIn("timed out", err)

    def test_cancel(self):
        import threading
        ev = threading.Event()
        threading.Timer(0.5, ev.set).start()
        with self.assertRaises(B.Cancelled):
            self._run("import time;time.sleep(60)", cancel=ev)


class ModelSettingsTests(unittest.TestCase):
    """detector / restoration / clip_length: module defaults unless the caller (queue app or CLI) overrides them."""

    DEFAULTS = {"detector": B.DETECTOR, "restoration": B.RESTORATION, "clip_length": B.CLIP_LENGTH}

    def test_normalize_keeps_defaults(self):
        for opts in (None, {}, {"detector": None, "restoration": None, "clip_length": None}):
            o = B.normalize_options(opts)
            self.assertEqual({k: o[k] for k in self.DEFAULTS}, self.DEFAULTS)

    def test_normalize_accepts_overrides(self):
        o = B.normalize_options({"detector": " det-x ", "restoration": "res-x", "clip_length": 90.0})
        self.assertEqual((o["detector"], o["restoration"], o["clip_length"]), ("det-x", "res-x", 90))
        self.assertIsInstance(o["clip_length"], int)
        self.assertEqual(B.normalize_options({"clip_length": 1})["clip_length"], 1)

    def test_normalize_rejects_bad_values(self):
        for bad in ({"clip_length": 0}, {"clip_length": -5}, {"clip_length": True}, {"clip_length": "180"},
                    {"clip_length": 1.5}, {"detector": "  "}, {"restoration": ""}):
            with self.assertRaises(ValueError, msg=str(bad)):
                B.normalize_options(bad)

    def test_cli_flags(self):
        o = parse(["--detector", "det-x", "--restoration", "res-x", "--clip-length", "90"])
        self.assertEqual((o["detector"], o["restoration"], o["clip_length"]), ("det-x", "res-x", 90))
        o = B.normalize_options(parse([]))
        self.assertEqual({k: o[k] for k in self.DEFAULTS}, self.DEFAULTS)

    def test_new_profile_records_settings(self):
        p = B.new_profile(B.normalize_options({"detector": "det-x", "restoration": "res-x", "clip_length": 60}))
        self.assertEqual(p["settings"], {"detector": "det-x", "restoration": "res-x", "clip_length": 60, "fp16": None})
        self.assertEqual(B.new_profile()["settings"], {**self.DEFAULTS, "fp16": None})
        self.assertEqual(P.validate(json.loads(json.dumps(p))), [])

    def test_child_settings_fall_back_to_constants(self):
        self.assertEqual(B._model_settings({"height": 720}), self.DEFAULTS)
        self.assertEqual(B._model_settings({"detector": None, "clip_length": None}), self.DEFAULTS)
        self.assertEqual(B._model_settings({"detector": "det-x", "restoration": "res-x", "clip_length": 30}),
                         {"detector": "det-x", "restoration": "res-x", "clip_length": 30})

    def test_child_payloads_carry_model_settings(self):
        import tempfile
        from unittest import mock
        seen = []

        def fake_run_child(kind, payload, **_kw):
            seen.append((kind, payload))
            return None, "stub"

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.object(B, "run_child", fake_run_child), \
                    mock.patch.object(B, "ensure_source", lambda h, tmp, *a, **k: tmp / f"src{h}.mp4"), \
                    mock.patch.object(B, "read_system_conditions",
                                      lambda device="auto": {"conditions": {}, "fingerprint": {}}), \
                    mock.patch.object(B, "_resolve_family", lambda family: ("libx265", None)):
                profile = B.run_benchmark({"mode": "quick", "workdir": str(d), "detector": "det-x",
                                           "restoration": "res-x", "clip_length": 42})
        self.assertEqual([k for k, _ in seen], ["restore", "rife", "encode"])
        for kind, payload in seen:
            self.assertEqual({k: payload[k] for k in self.DEFAULTS},
                             {"detector": "det-x", "restoration": "res-x", "clip_length": 42}, msg=kind)
        self.assertEqual({k: profile["settings"][k] for k in self.DEFAULTS},
                         {"detector": "det-x", "restoration": "res-x", "clip_length": 42})

    def test_quick_merges_the_extra_d0_height(self):
        import tempfile
        from unittest import mock
        seen = []

        def fake_run_child(kind, payload, **_kw):
            seen.append((kind, payload))
            if kind == "restore":
                return {"densities": {"0": {"fps": 340.0}, "1": {"fps": 45.0}},
                        "d0_extra": {"1080": {"fps": 290.0, "detector_done_early": False}}}, None
            return None, "stub"

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.object(B, "run_child", fake_run_child), \
                    mock.patch.object(B, "ensure_source", lambda h, tmp, *a, **k: tmp / f"src{h}.mp4"), \
                    mock.patch.object(B, "read_system_conditions",
                                      lambda device="auto": {"conditions": {}, "fingerprint": {}}), \
                    mock.patch.object(B, "_resolve_family", lambda family: ("libx265", None)):
                profile = B.run_benchmark({"mode": "quick", "workdir": str(d)})
        payload = seen[0][1]
        self.assertEqual([x["height"] for x in payload["d0_sources"]], [1080])
        self.assertTrue(payload["d0_sources"][0]["source"].endswith("src1080.mp4"))
        self.assertEqual(profile["restore"]["720"]["d1_fps"], 45.0)
        self.assertEqual(profile["restore"]["1080"], {"d0_fps": 290.0, "d1_fps": None, "d2_fps": None,
                                                      "vram_peak_gib": None, "error": None})
        self.assertAlmostEqual(P.restore_frame_time(profile, 1080, 1), 1 / 290 + (1 / 45 - 1 / 340))

    def test_standard_measures_the_other_detector_at_density_0(self):
        import tempfile
        from unittest import mock
        seen = []

        def fake_run_child(kind, payload, **_kw):
            seen.append((kind, payload))
            if kind == "restore":
                fast = payload["detector"] == "v4-fast"
                d0 = {"0": {"fps": 330.0 if fast else 170.0}}
                if 1 in payload["densities"]:
                    d0["1"] = {"fps": 40.0}
                extra = {str(x["height"]): {"fps": (300.0 if fast else 175.0)} for x in payload["d0_sources"]}
                return {"densities": d0, "d0_extra": extra}, None
            return None, "stub"

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(B, "run_child", fake_run_child), \
                    mock.patch.object(B, "ensure_source", lambda h, tmp, *a, **k: tmp / f"src{h}.mp4"), \
                    mock.patch.object(B, "read_system_conditions",
                                      lambda device="auto": {"conditions": {}, "fingerprint": {}}), \
                    mock.patch.object(B, "_resolve_family", lambda family: ("libx265", None)):
                profile = B.run_benchmark({"workdir": str(d), "detector": "v4-accurate", "clip_length": 300})
        restores = [p for k, p in seen if k == "restore"]
        self.assertEqual([(p["detector"], p["densities"], p["clip_length"]) for p in restores],
                         [("v4-accurate", [0, 1], 240), ("v4-fast", [0], 240)])
        self.assertEqual(profile["settings"]["clip_length"], 300)
        self.assertEqual(profile["detail"]["measure_clip_length"], 240)
        self.assertEqual({h: e["d1_fps"] for h, e in profile["restore"].items()}, {"720": 40.0, "480": None, "1080": None})
        self.assertEqual({h: e["d0_fps"] for h, e in profile["restore_alt"]["v4-fast"].items()},
                         {"720": 330.0, "480": 300.0, "1080": 300.0})
        self.assertEqual(P.detectors(profile), ["v4-accurate", "v4-fast"])
        self.assertEqual(P.validate(profile), [])


if __name__ == "__main__":
    unittest.main()
