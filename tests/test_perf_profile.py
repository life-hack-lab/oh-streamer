"""Perf profile estimation model (stdlib only, no GPU)."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import tempfile
import unittest
from unittest import mock

from oh_streamer.perf import profile as P


def fixture():
    return {
        "schema": 1,
        "fingerprint": {"gpu": "RTX 5060", "driver": "617.42", "backend": "cuda",
                        "torch": "2.9.1", "lada": "0.x"},
        "restore": {
            "480": {"d0_fps": 220.0, "d1_fps": 70.0, "d2_fps": None, "error": None},
            "720": {"d0_fps": 140.0, "d1_fps": 48.0, "d2_fps": None, "error": None},
            "1080": {"d0_fps": 80.0, "d1_fps": 30.0, "d2_fps": None, "error": None},
        },
        "rife": {"720": {"pair_ms": 9.5, "error": None}, "1080": {"pair_ms": 20.0, "error": None}},
        "encode": {"hevc_nvenc": {"720": {"fps": 420.0, "error": None},
                                  "1080": {"fps": 200.0, "error": None}}},
        "combined": {"height": 720, "density": 1.0, "encoder": "hevc_nvenc", "rife": "full",
                     "fps": 30.0, "error": None},
        "calibration": {},
        "warnings": ["on battery power"],
    }


def flat(d0, d1, h="720", enc=None, rife_ms=None):
    """Single-height profile for exact threshold tests."""
    p = {"schema": 1, "fingerprint": {}, "restore": {h: {"d0_fps": d0, "d1_fps": d1}},
         "rife": {}, "encode": {}}
    if enc:
        p["encode"] = {"hevc_nvenc": {h: {"fps": enc}}}
    if rife_ms:
        p["rife"] = {h: {"pair_ms": rife_ms}}
    return p


class RestoreModelTests(unittest.TestCase):
    def test_linear_model(self):
        p = fixture()
        self.assertAlmostEqual(P.restore_frame_time(p, 720, 0), 1 / 140)
        self.assertAlmostEqual(P.restore_frame_time(p, 720, 1), 1 / 48)
        self.assertAlmostEqual(P.restore_frame_time(p, 720, "typical"),
                               1 / 140 + 0.7 * (1 / 48 - 1 / 140))
        self.assertAlmostEqual(P.restore_frame_time(p, 720, 2), 1 / 140 + 2 * (1 / 48 - 1 / 140))

    def test_d2_least_squares(self):
        p = fixture()
        p["restore"]["720"] = {"d0_fps": 100.0, "d1_fps": 50.0, "d2_fps": 25.0}
        # points (0,.01) (1,.02) (2,.04): slope .015, intercept 0.0083333
        self.assertAlmostEqual(P.restore_frame_time(p, 720, 0), 0.01 / 1.2, places=6)
        self.assertAlmostEqual(P.restore_frame_time(p, 720, 1), 0.0083333 + 0.015, places=5)

    def test_height_interpolation_by_pixels(self):
        p = fixture()
        t480, t720 = (P.restore_frame_time(p, h, 1) for h in (480, 720))
        w = (540 ** 2 - 480 ** 2) / (720 ** 2 - 480 ** 2)
        self.assertAlmostEqual(P.restore_frame_time(p, 540, 1), t480 + w * (t720 - t480))
        self.assertTrue(t480 < P.restore_frame_time(p, 540, 1) < t720)

    def test_height_extrapolation_extends_fixed_plus_per_pixel_line(self):
        # a = 1/d0 and b = 1/d1 - 1/d0 are each extended along the line through the two nearest heights
        p = {"restore": {"720": {"d0_fps": 1 / 0.003, "d1_fps": 1 / 0.022},       # a 3 ms, b 19 ms
                         "1080": {"d0_fps": 1 / 0.0035, "d1_fps": 1 / 0.0235}}}  # a 3.5 ms, b 20 ms
        px = lambda h: h * h
        slope_a = 0.0005 / (px(1080) - px(720))
        slope_b = 0.001 / (px(1080) - px(720))
        a = 0.0035 + slope_a * (px(2160) - px(1080))
        b = 0.020 + slope_b * (px(2160) - px(1080))
        self.assertAlmostEqual(P.restore_frame_time(p, 2160, 0), a)
        self.assertAlmostEqual(P.restore_frame_time(p, 2160, 1), a + b)
        a360 = 0.003 - slope_a * (px(720) - px(360))
        self.assertAlmostEqual(P.restore_frame_time(p, 360, 0), a360)

    def test_height_extrapolation_stays_between_fixed_and_per_pixel(self):
        # a cost that falls with resolution (noise) is held at the nearest value; one that would grow faster
        # than the pixel count is capped at pixel scaling
        p = {"restore": {"720": {"d0_fps": 1 / 0.004, "d1_fps": 1 / 0.024},
                         "1080": {"d0_fps": 1 / 0.003, "d1_fps": 1 / 0.023}}}
        self.assertAlmostEqual(P.restore_frame_time(p, 2160, 0), 0.003)
        p = {"restore": {"720": {"d0_fps": 1 / 0.001, "d1_fps": 1 / 0.020},
                         "1080": {"d0_fps": 1 / 0.010, "d1_fps": 1 / 0.029}}}
        self.assertAlmostEqual(P.restore_frame_time(p, 2160, 0), 0.010 * 4)
        self.assertAlmostEqual(P.restore_frame_time(p, 360, 0), 0.001 / 4)

    def test_single_height_scales_d0_by_pixels_keeps_clip_cost(self):
        p = flat(1 / 0.003, 1 / 0.022)
        self.assertAlmostEqual(P.restore_frame_time(p, 1080, 0), 0.003 * 2.25)
        self.assertAlmostEqual(P.restore_frame_time(p, 1080, 1), 0.003 * 2.25 + 0.019)

    def test_d0_only_height_gives_the_per_frame_part(self):
        p = flat(1 / 0.003, 1 / 0.022)
        p["restore"]["1080"] = {"d0_fps": 1 / 0.0035, "d1_fps": None}
        self.assertAlmostEqual(P.restore_frame_time(p, 1080, 1), 0.0035 + 0.019)
        self.assertIsNone(P.restore_frame_time({"restore": {"1080": {"d0_fps": 300.0}}}, 1080, 1))

    def test_none_without_data(self):
        p = fixture()
        p["restore"] = {"720": {"d0_fps": None, "d1_fps": None, "error": "OOM"}}
        self.assertIsNone(P.restore_frame_time(p, 720, 1))
        self.assertIsNone(P.restore_frame_time({}, 720, 1))
        self.assertIsNone(P.restore_frame_time({"restore": None}, 720, 1))
        self.assertIsNone(P.estimate({}, height=720, src_fps=30)["fps"])

    def test_failed_height_is_skipped(self):
        p = fixture()
        p["restore"]["1080"] = {"d0_fps": None, "d1_fps": None, "error": "OOM"}
        q = {"restore": {h: p["restore"][h] for h in ("480", "720")}}
        self.assertAlmostEqual(P.restore_frame_time(p, 1080, 1), P.restore_frame_time(q, 1080, 1))

    def test_bad_density(self):
        with self.assertRaises(ValueError):
            P.restore_frame_time(fixture(), 720, "huge")


def two_detectors():
    """Measured with v4-accurate (d1 at 720 only, as plan_steps does), v4-fast at density 0."""
    p = fixture()
    p["settings"] = {"detector": "v4-accurate", "restoration": "basicvsrpp-v1.2", "clip_length": 300}
    p["detail"] = {"measure_clip_length": 240}
    p["restore"] = {"480": {"d0_fps": 173.0, "d1_fps": None}, "720": {"d0_fps": 171.0, "d1_fps": 40.0},
                    "1080": {"d0_fps": 181.0, "d1_fps": None}}
    p["restore_alt"] = {"v4-fast": {"480": {"d0_fps": 366.0}, "720": {"d0_fps": 331.0},
                                    "1080": {"d0_fps": 286.0}}}
    return p


def rtx5060_accurate():
    """The 2026-10-13 standard check (RTX 5060 Laptop, v4-accurate, v4-fast at density 0, clip 300 -> 240).
    Nothing is live at heavy density, every 30 fps cell plays after a short prebuffer, 60 fps does not keep up,
    and both models get the same verdict everywhere."""
    p = fixture()
    p["settings"] = {"detector": "v4-accurate", "restoration": "basicvsrpp-v1.2", "clip_length": 300, "fp16": True}
    p["detail"] = {"measure_clip_length": 240}
    p["restore"] = {"480": {"d0_fps": 143.2}, "720": {"d0_fps": 171.1, "d1_fps": 33.6}, "1080": {"d0_fps": 177.2}}
    p["restore_alt"] = {"v4-fast": {"480": {"d0_fps": 351.8}, "720": {"d0_fps": 338.3}, "1080": {"d0_fps": 287.5}}}
    p["rife"] = {"480": {"pair_ms": 4.85}, "720": {"pair_ms": 10.77}, "1080": {"pair_ms": 15.64}}
    p["encode"] = {"hevc_nvenc": {"480": {"fps": 2207.4}, "720": {"fps": 1023.5}, "1080": {"fps": 448.2}}}
    p["combined"] = {"height": 720, "density": 1.0, "encoder": "hevc_nvenc", "rife": "full", "fps": 27.7}
    return p


class DetectorTests(unittest.TestCase):
    def test_recommends_the_accurate_model_when_no_verdict_changes(self):
        r = P.recommend_detector(rtx5060_accurate())
        self.assertEqual(r, {"detector": "v4-accurate", "differs": []})

    def test_no_model_advice_when_a_verdict_changes(self):
        r = P.recommend_detector(two_detectors())
        self.assertIsNone(r["detector"])
        self.assertIn((60, 480, "off", {"v4-accurate": "queue", "v4-fast": "prebuffer"}), r["differs"])
        self.assertFalse([d for d in r["differs"] if d[0] == 60 and d[2] != "off"])   # 60 fps: off rows only
        self.assertIsNone(P.recommend_detector(fixture())["detector"])                  # one model measured


    def test_detectors_lists_the_measured_and_usable_alternatives(self):
        p = two_detectors()
        self.assertEqual(P.detectors(p), ["v4-accurate", "v4-fast"])
        p["restore_alt"]["v4-x"] = {"720": {"d0_fps": None, "error": "missing model"}}
        self.assertEqual(P.detectors(p), ["v4-accurate", "v4-fast"])
        self.assertEqual(P.detectors(fixture()), [])

    def test_other_detector_keeps_the_clip_cost_and_swaps_the_frame_cost(self):
        p = two_detectors()
        v = P.for_detector(p, "v4-fast")
        self.assertIsNot(v, p)
        self.assertEqual(v["settings"]["detector"], "v4-fast")
        b = 1 / 40.0 - 1 / 171.0
        self.assertAlmostEqual(P.restore_frame_time(v, 720, 1.0), 1 / 331.0 + b)
        self.assertAlmostEqual(P.restore_frame_time(v, 1080, 1.0), 1 / 286.0 + b)
        self.assertAlmostEqual(P.restore_frame_time(p, 1080, 1.0), 1 / 181.0 + b)
        self.assertGreater(P.estimate(v, height=1080, src_fps=30)["fps"], P.estimate(p, height=1080, src_fps=30)["fps"])
        self.assertAlmostEqual(P.rife_k(v), P.rife_k(p))           # the combined run used the main model
        self.assertEqual(p["restore"]["720"]["d1_fps"], 40.0)       # the original is not touched

    def test_same_unknown_or_missing_detector_is_the_profile_itself(self):
        p = two_detectors()
        for det in ("v4-accurate", None, "", "v9"):
            self.assertIs(P.for_detector(p, det), p)
        f = fixture()                                # older profile: no restore_alt
        self.assertIs(P.for_detector(f, "v4-fast"), f)

    def test_calibration_compares_with_the_model_the_worker_used(self):
        p = two_detectors()
        pred = P.estimate(P.for_detector(p, "v4-fast"), height=720, src_fps=30, density=0.5, calibrate=False)["fps"]
        out = P.record_observation(p, mode="off", height=720, observed_fps=pred * 1.1, density=0.5, detector="v4-fast")
        self.assertAlmostEqual(out["calibration"]["off:720"]["factor"], 1.1)
        self.assertNotIn("restore_alt", {k for k in out if out[k] != p[k]})

    def test_clip_length_coverage(self):
        p = two_detectors()                          # setting 300, measured with 240
        self.assertEqual(P.measured_clip_length(p), 240)
        for cur in (240, 300, 360, None):
            self.assertTrue(P.covers_clip_length(p, cur), cur)
        self.assertFalse(P.covers_clip_length(p, 180))
        old = fixture()
        old["settings"] = {"clip_length": 180}
        self.assertEqual(P.measured_clip_length(old), 180)
        self.assertFalse(P.covers_clip_length(old, 240))


class InterpRangeTests(unittest.TestCase):
    def test_range_matches_the_interpolator(self):
        from oh_streamer.processing import rife_interp
        self.assertEqual(P.INTERP_FPS_RANGE, (rife_interp.MIN_SOURCE_FPS, rife_interp.MAX_SOURCE_FPS))
        self.assertTrue(P.interp_eligible(29.97) and P.interp_eligible(40))
        self.assertFalse(P.interp_eligible(50) or P.interp_eligible(60) or P.interp_eligible(8) or P.interp_eligible(None))

    def test_sixty_fps_auto_and_full_are_estimated_as_off(self):
        p = fixture()
        off = P.estimate(p, height=720, src_fps=60, mode="off")
        for mode in ("auto", "full"):
            e = P.estimate(p, height=720, src_fps=60, mode=mode)
            self.assertAlmostEqual(e["fps"], off["fps"])
            self.assertEqual((e["interp_fraction"], e["interp_ok"], e["mode"]), (0.0, False, mode))
        self.assertAlmostEqual(P.job_seconds(p, duration_s=600, height=720, src_fps=60, mode="full"),
                               600 * 60 / off["fps"])
        self.assertTrue(P.estimate(p, height=720, src_fps=30, mode="full")["interp_ok"])

    def test_sixty_fps_uses_the_off_calibration(self):
        p = P.record_observation(fixture(), mode="off", height=720, observed_fps=50.0, predicted_fps=100.0)
        self.assertAlmostEqual(P.estimate(p, height=720, src_fps=60, mode="full")["factor"], 0.5)


class EstimateTests(unittest.TestCase):
    def test_off_ratio(self):
        e = P.estimate(fixture(), height=720, src_fps=30, density=1.0)
        self.assertAlmostEqual(e["fps"], 48.0)
        self.assertAlmostEqual(e["ratio"], 48 / 30)
        self.assertEqual(e["bottleneck"], "gpu")
        self.assertEqual(e["interp_fraction"], 0.0)
        self.assertFalse(e["calibrated"])

    def test_k_derivation(self):
        p = fixture()
        k = (1 / 30 - 1 / 48) / 0.0095
        self.assertAlmostEqual(P.rife_k(p), k)
        e = P.estimate(p, height=720, src_fps=30, mode="full", density=1.0)
        self.assertAlmostEqual(e["fps"], 30.0)  # reproduces the combined measurement
        self.assertAlmostEqual(e["k"], k)

    def test_k_defaults_and_clamp(self):
        p = fixture()
        p["combined"]["error"] = "failed"
        self.assertEqual(P.rife_k(p), 1.0)
        p["combined"] = {"height": 720, "density": 1.0, "fps": 1.0}
        self.assertEqual(P.rife_k(p), 3.0)
        p["combined"]["fps"] = 47.9  # barely slower than plain restore -> tiny k -> clamp low
        self.assertEqual(P.rife_k(p), 0.5)
        p["combined"] = None
        self.assertEqual(P.rife_k(p), 1.0)

    def test_k_ignored_when_encode_limited(self):
        p = fixture()
        p["encode"]["hevc_nvenc"]["720"]["fps"] = 60.0  # full cap = 30 = observed
        self.assertEqual(P.rife_k(p), 1.0)

    def test_full_slower_than_off(self):
        p = fixture()
        for h in P.HEIGHTS:
            off = P.estimate(p, height=h, src_fps=30)["fps"]
            full = P.estimate(p, height=h, src_fps=30, mode="full")["fps"]
            self.assertLess(full, off)

    def test_encode_bottleneck(self):
        p = fixture()
        p["encode"]["hevc_nvenc"]["720"]["fps"] = 20.0
        e = P.estimate(p, height=720, src_fps=30, density=1.0)
        self.assertEqual(e["bottleneck"], "encode")
        self.assertAlmostEqual(e["fps"], 20.0)
        full = P.estimate(p, height=720, src_fps=30, density=1.0, mode="full")
        self.assertAlmostEqual(full["fps"], 10.0)  # two outputs per source frame

    def test_full_encode_cap_halves(self):
        p = flat(1000, 1000, enc=100.0, rife_ms=0.1)
        self.assertAlmostEqual(P.estimate(p, height=720, src_fps=30, mode="full")["fps"], 50.0)

    def test_compare_encode_halves_headroom(self):
        p = flat(1000, 1000, enc=100.0)
        self.assertAlmostEqual(P.estimate(p, height=720, src_fps=30)["fps"], 100.0)
        self.assertAlmostEqual(P.estimate(p, height=720, src_fps=30, compare="encode")["fps"], 50.0)
        self.assertAlmostEqual(P.estimate(p, height=720, src_fps=30, compare="copy")["fps"], 100.0)

    def test_encoder_selection(self):
        p = fixture()
        p["encode"]["h264_nvenc"] = {"720": {"fps": 900.0}}
        p["encode"]["libx265"] = {"720": {"fps": 5.0}}
        self.assertEqual(P.estimate(p, height=720, src_fps=30)["encoder"], "hevc_nvenc")
        self.assertEqual(P.estimate(p, height=720, src_fps=30, encoder="h264_nvenc")["encoder"],
                         "h264_nvenc")
        self.assertEqual(P.estimate(p, height=720, src_fps=30, encoder="h264")["encoder"],
                         "h264_nvenc")
        e = P.estimate(p, height=720, src_fps=30, encoder="hevc_amf")
        self.assertEqual(e["encoder"], "hevc_nvenc")
        self.assertTrue(e["notes"])

    def test_encode_error_entries_ignored(self):
        p = fixture()
        p["encode"] = {"hevc_nvenc": {"720": {"fps": None, "error": "unsupported"}}}
        e = P.estimate(p, height=720, src_fps=30)
        self.assertIsNone(e["encoder"])
        self.assertEqual(e["bottleneck"], "gpu")

    def test_auto_interp_fraction(self):
        p = fixture()
        e = P.estimate(p, height=720, src_fps=30, mode="auto", density=1.0)
        k = P.rife_k(p)
        self.assertAlmostEqual(e["fps"], 48.0)
        self.assertAlmostEqual(e["interp_fraction"], (1 / 30 - 1 / 48) / (k * 0.0095))
        # slower than realtime: no spare time at all
        self.assertEqual(P.estimate(p, height=1080, src_fps=60, mode="auto")["interp_fraction"], 0.0)
        # very fast: fraction saturates at 1
        self.assertEqual(P.estimate(flat(1000, 1000, rife_ms=1.0), height=720, src_fps=30,
                                    mode="auto")["interp_fraction"], 1.0)

    def test_full_without_rife_data(self):
        e = P.estimate(flat(100, 100), height=720, src_fps=30, mode="full")
        self.assertIsNone(e["fps"])
        self.assertTrue(e["notes"])

    def test_calibration_factor_applied(self):
        p = fixture()
        base = P.estimate(p, height=720, src_fps=30, density=1.0)["fps"]
        p["calibration"]["off:720"] = {"factor": 0.8, "n": 3}
        e = P.estimate(p, height=720, src_fps=30, density=1.0)
        self.assertTrue(e["calibrated"])
        self.assertAlmostEqual(e["fps"], base * 0.8)
        self.assertFalse(P.estimate(p, height=1080, src_fps=30)["calibrated"])
        self.assertFalse(P.estimate(p, height=720, src_fps=30, calibrate=False)["calibrated"])


class VerdictTests(unittest.TestCase):
    # t(d) = .01 + .015 d ; typical .0205 (48.8 fps), heavy .028 (35.7 fps)
    P = flat(100, 40)

    def test_live(self):
        v = P.verdict(self.P, height=720, src_fps=30)
        self.assertEqual(v["level"], "live")
        self.assertEqual(v["prebuffer_s"], 0)
        self.assertAlmostEqual(v["ratio_heavy"], (1 / 0.028) / 30)

    def test_prebuffer_math(self):
        v = P.verdict(self.P, height=720, src_fps=40)
        rh = (1 / 0.028) / 40
        self.assertEqual(v["level"], "prebuffer")
        self.assertGreater(v["ratio_typical"], 1.0)
        self.assertLess(rh, 1.1)
        self.assertAlmostEqual(v["prebuffer_s"], max(P.MIN_PREBUFFER_S, 60 * (1 - rh) / rh))

    def test_prebuffer_minimum(self):
        with mock.patch.object(P, "HEAVY_SEGMENT_S", 1):
            v = P.verdict(self.P, height=720, src_fps=40)
        self.assertEqual(v["prebuffer_s"], P.MIN_PREBUFFER_S)

    def test_prebuffer_clamped(self):
        with mock.patch.object(P, "MAX_PREBUFFER_S", 5), mock.patch.object(P, "MIN_PREBUFFER_S", 1):
            v = P.verdict(self.P, height=720, src_fps=40)
        self.assertEqual(v["prebuffer_s"], 5)

    def test_queue(self):
        v = P.verdict(self.P, height=720, src_fps=60)
        self.assertEqual(v["level"], "queue")
        self.assertIsNone(v["prebuffer_s"])

    def test_boundaries(self):
        # heavy ratio exactly at LIVE_RATIO is live; just below falls to prebuffer
        f = 1.1 * 30 * 1.0000001
        self.assertEqual(P.verdict(flat(f, f), height=720, src_fps=30)["level"], "live")
        f = 1.09 * 30
        self.assertEqual(P.verdict(flat(f, f), height=720, src_fps=30)["level"], "prebuffer")
        f = 0.99 * 30
        self.assertEqual(P.verdict(flat(f, f), height=720, src_fps=30)["level"], "queue")

    def test_no_data_is_queue(self):
        self.assertEqual(P.verdict({}, height=720, src_fps=30)["level"], "queue")

    def test_kwargs_passthrough(self):
        p = flat(1000, 1000, enc=100.0)
        v = P.verdict(p, height=720, src_fps=30, compare="encode")
        self.assertAlmostEqual(v["ratio_typical"], 50 / 30)
        self.assertEqual(v["bottleneck"], "encode")
        self.assertAlmostEqual(P.verdict(p, height=720, src_fps=30)["ratio_typical"], 100 / 30)


class MatrixRecommendJobTests(unittest.TestCase):
    def test_matrix_shape(self):
        rows = P.matrix(fixture())
        self.assertEqual(len(rows), len(P.HEIGHTS) * len(P.MODES))
        self.assertEqual({(r["height"], r["mode"]) for r in rows},
                         {(h, m) for h in P.HEIGHTS for m in P.MODES})
        for key in ("level", "ratio_typical", "ratio_heavy", "prebuffer_s", "interp_fraction"):
            self.assertIn(key, rows[0])

    def test_recommend_consistent_with_matrix(self):
        p = fixture()
        rec = P.recommend(p)
        live = [r["height"] for r in P.matrix(p) if r["mode"] == "off" and r["level"] == "live"]
        self.assertEqual(rec["live_height"], max(live, default=None))
        self.assertEqual(rec["use_queue"], not live)
        self.assertTrue(rec["summary"])

    def test_recommend_fast_machine(self):
        p = fixture()
        for e in p["restore"].values():
            e["d0_fps"], e["d1_fps"] = 2000.0, 1000.0
        p["rife"] = {h: {"pair_ms": 2.0} for h in ("480", "720", "1080")}
        p["encode"]["hevc_nvenc"] = {h: {"fps": 3000.0} for h in ("480", "720", "1080")}
        p["combined"] = None
        rec = P.recommend(p)
        self.assertEqual(rec["live_height"], 1080)
        self.assertEqual(rec["full_height"], 1080)
        self.assertTrue(rec["auto_interp"])
        self.assertFalse(rec["use_queue"])
        self.assertIn("1080p", rec["summary"])

    def test_recommend_auto_needs_enough_interpolation(self):
        # live at every height, but spare time is small: interp_fraction ~0.21 < 0.3
        p = flat(100, 40, rife_ms=60.0)
        p["restore"] = {h: p["restore"]["720"] for h in ("480", "720", "1080")}
        p["rife"] = {h: {"pair_ms": 60.0} for h in ("480", "720", "1080")}
        rec = P.recommend(p)
        self.assertEqual(rec["live_height"], 1080)
        self.assertLess(rec["auto_fraction"], P.AUTO_INTERP_MIN)
        self.assertFalse(rec["auto_interp"])
        self.assertTrue(P.recommend({**p, "rife": {h: {"pair_ms": 30.0} for h in p["rife"]}})["auto_interp"])

    def test_recommend_auto_when_playback_needs_a_prebuffer(self):
        # auto never slows playback (its speed is off's), so a prebuffer-only machine still gets it
        rec = P.recommend(rtx5060_accurate())
        self.assertIsNone(rec["live_height"])
        self.assertEqual((rec["prebuffer_height"], rec["full_watch_height"]), (1080, 1080))
        self.assertTrue(rec["auto_interp"])
        self.assertIn("先緩衝約 10 秒後可播放，最高 1080p", rec["summary"])
        self.assertEqual(rec["restoration_fps"], "30")
        self.assertIn("60 fps 影片原始幀率跟不上；修復幀率設「最高 30fps」後，先緩衝約 10 秒即可播放，最高 1080p，"
                      "自動補幀約補 100% 回到 60 fps", rec["summary"])
        self.assertNotIn("60 fps", P.recommend(rtx5060_accurate(), 60)["summary"])

    def test_recommend_nothing_live(self):
        p = fixture()
        for e in p["restore"].values():
            e["d0_fps"], e["d1_fps"] = 10.0, 5.0
        rec = P.recommend(p)
        self.assertIsNone(rec["live_height"])
        self.assertTrue(rec["use_queue"])
        self.assertIn("佇列", rec["summary"])

    def test_recommend_empty_profile(self):
        rec = P.recommend({})
        self.assertTrue(rec["use_queue"])
        self.assertTrue(rec["summary"])

    def test_job_seconds(self):
        p = fixture()
        e = P.estimate(p, height=720, src_fps=30, mode="full")
        self.assertAlmostEqual(P.job_seconds(p, duration_s=600, height=720, src_fps=30),
                               600 * 30 / e["fps"])
        slow = P.job_seconds(p, duration_s=600, height=720, src_fps=30, compare="encode")
        self.assertGreaterEqual(slow, P.job_seconds(p, duration_s=600, height=720, src_fps=30))
        self.assertIsNone(P.job_seconds({}, duration_s=600, height=720, src_fps=30))
        self.assertIsNone(P.job_seconds(p, duration_s=0, height=720, src_fps=30))


class CalibrationTests(unittest.TestCase):
    def test_first_observation_and_ema(self):
        p = fixture()
        pred = P.estimate(p, height=720, src_fps=30, calibrate=False)["fps"]
        q = P.record_observation(p, mode="off", height=720, observed_fps=pred * 0.5)
        self.assertEqual(p["calibration"], {})  # input not mutated
        self.assertAlmostEqual(q["calibration"]["off:720"]["factor"], 0.5)
        self.assertEqual(q["calibration"]["off:720"]["n"], 1)
        q = P.record_observation(q, mode="off", height=720, observed_fps=pred * 1.0)
        self.assertAlmostEqual(q["calibration"]["off:720"]["factor"], 0.8 * 0.5 + 0.2 * 1.0)
        self.assertEqual(q["calibration"]["off:720"]["n"], 2)

    def test_ratio_uses_uncalibrated_prediction(self):
        p = fixture()
        pred = P.estimate(p, height=720, src_fps=30, calibrate=False)["fps"]
        p["calibration"]["off:720"] = {"factor": 0.5, "n": 1}
        q = P.record_observation(p, mode="off", height=720, observed_fps=pred * 0.5)
        self.assertAlmostEqual(q["calibration"]["off:720"]["factor"], 0.5)

    def test_explicit_predicted_and_alpha(self):
        q = P.record_observation(fixture(), mode="full", height=1080, observed_fps=20,
                                 predicted_fps=40)
        self.assertAlmostEqual(q["calibration"]["full:1080"]["factor"], 0.5)
        q = P.record_observation(q, mode="full", height=1080, observed_fps=40, predicted_fps=40,
                                 alpha=0.5)
        self.assertAlmostEqual(q["calibration"]["full:1080"]["factor"], 0.75)

    def test_insane_ratios_ignored(self):
        p = fixture()
        for obs in (1.0, 500.0, 0, None, -3):
            q = P.record_observation(p, mode="off", height=720, observed_fps=obs, predicted_fps=40)
            self.assertEqual(q["calibration"], {}, obs)

    def test_boundary_ratios_accepted(self):
        q = P.record_observation(fixture(), mode="off", height=720, observed_fps=8, predicted_fps=40)
        self.assertAlmostEqual(q["calibration"]["off:720"]["factor"], 0.2)
        q = P.record_observation(fixture(), mode="off", height=720, observed_fps=200, predicted_fps=40)
        self.assertAlmostEqual(q["calibration"]["off:720"]["factor"], 5.0)

    def test_missing_calibration_section(self):
        p = fixture()
        del p["calibration"]
        q = P.record_observation(p, mode="off", height=720, observed_fps=40, predicted_fps=40)
        self.assertEqual(q["calibration"]["off:720"]["n"], 1)

    def test_calibration_feeds_back_into_estimate(self):
        p = fixture()
        raw = P.estimate(p, height=720, src_fps=30)["fps"]
        q = P.record_observation(p, mode="off", height=720, observed_fps=raw * 0.7)
        self.assertAlmostEqual(P.estimate(q, height=720, src_fps=30)["fps"], raw * 0.7)


class StaleTests(unittest.TestCase):
    def test_same_and_missing(self):
        p = fixture()
        cur = dict(p["fingerprint"])
        self.assertEqual(P.stale_reasons(p, cur), [])
        self.assertEqual(P.stale_reasons(p, {}), [])
        self.assertEqual(P.stale_reasons(p, {"gpu": None, "driver": None}), [])
        self.assertEqual(P.stale_reasons(p, {"os": "other", "vram_total_gib": 4}), [])

    def test_differences(self):
        p = fixture()
        r = P.stale_reasons(p, {"gpu": "RTX 9090", "driver": "700.0", "torch": "2.9.1"})
        self.assertEqual(len(r), 2)
        self.assertTrue(r[0].startswith("gpu"))
        self.assertTrue(r[1].startswith("driver"))

    def test_profile_without_fingerprint(self):
        self.assertEqual(len(P.stale_reasons({}, {"gpu": "x"})), 1)


class StorageTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "perf_profile.json"
            p = fixture()
            P.save(p, path)
            self.assertEqual(P.load(path), p)
            self.assertEqual([f.name for f in path.parent.iterdir()], ["perf_profile.json"])
            p["warnings"] = ["中文"]
            P.save(p, path)  # overwrite
            self.assertEqual(P.load(path)["warnings"], ["中文"])

    def test_load_missing_and_corrupt(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(P.load(Path(d) / "nope.json"))
            bad = Path(d) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertIsNone(P.load(bad))
            bad.write_text("[1, 2]", encoding="utf-8")
            self.assertIsNone(P.load(bad))

    def test_failed_save_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            with self.assertRaises(TypeError):
                P.save({"x": object()}, path)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_default_path(self):
        self.assertEqual(P.default_path().name, "perf_profile.json")
        self.assertEqual(P.default_path().parent.name, "oh_streamer")


class ValidateTests(unittest.TestCase):
    def test_fixture_ok(self):
        self.assertEqual(P.validate(fixture()), [])

    def test_broken_input(self):
        self.assertTrue(P.validate(None))
        self.assertTrue(P.validate([]))
        self.assertTrue(any("schema" in x for x in P.validate({"schema": 2})))
        probs = P.validate({"schema": 1})
        for name in ("fingerprint", "restore", "rife", "encode"):
            self.assertTrue(any(name in x for x in probs), name)
        p = fixture()
        p["restore"] = []
        p["encode"]["hevc_nvenc"] = 5
        probs = P.validate(p)
        self.assertTrue(any("restore" in x for x in probs))
        self.assertTrue(any("hevc_nvenc" in x for x in probs))

    def test_nulls_tolerated(self):
        p = fixture()
        p["rife"] = None
        p["combined"] = None
        p["restore"]["1080"] = {"d0_fps": None, "d1_fps": None, "error": "OOM"}
        p["encode"]["hevc_nvenc"]["1080"] = {"fps": None, "error": "x"}
        self.assertEqual(P.validate(p), [])
        self.assertIsNotNone(P.estimate(p, height=720, src_fps=30)["fps"])
        self.assertIsNone(P.estimate(p, height=720, src_fps=30, mode="full")["fps"])
        P.matrix(p)
        P.format_table(p)


class FormatTableTests(unittest.TestCase):
    def test_table(self):
        t = P.format_table(fixture())
        for s in ("480p", "720p", "1080p", "30fps 不補幀", "自動補幀", "全補幀", "60fps", "on battery power"):
            self.assertIn(s, t)
        self.assertTrue("✅" in t or "⏳" in t or "📥" in t)
        self.assertIn(P.recommend(fixture())["summary"], t)
        head = t.splitlines()[1]
        self.assertLess(head.index("30fps 不補幀"), head.index("自動補幀"))
        self.assertLess(head.index("全補幀"), head.index("60fps"))

    def test_cells(self):
        p = flat(100, 40)
        t = P.format_table(p)
        self.assertIn("1.6x ✅", t)  # typical 48.8 fps / 30
        v = {"level": "prebuffer", "ratio_typical": 0.9, "prebuffer_s": 45.4,
             "mode": "off", "interp_fraction": 0.0}
        self.assertEqual(P._cell(v), "0.9x ⏳ 45s")
        self.assertEqual(P._cell({**v, "level": "queue", "ratio_typical": 0.6}), "0.6x 📥")

    def test_auto_cell_shows_the_interpolated_share_not_the_speed(self):
        v = {"level": "live", "ratio_typical": 2.1, "prebuffer_s": 0.0, "mode": "auto", "interp_fraction": 0.77}
        self.assertEqual(P._cell(v), "補77% ✅")
        self.assertEqual(P._cell({**v, "interp_fraction": 0.0}), "不補 ✅")

    def test_empty_profile(self):
        self.assertIn("效能", P.format_table({}))


class ImportHygieneTests(unittest.TestCase):
    def test_no_heavy_imports(self):
        import os
        import subprocess
        code = ("import sys; import oh_streamer.perf.profile as p; "
                "bad = [m for m in ('torch', 'numpy') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        env = {**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
