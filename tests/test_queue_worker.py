"""Tests for queue_worker.py (run: lada\\.venv\\Scripts\\python.exe streamer\\tests\\test_queue_worker.py).

No GPU, no torch: the segmenting / resume / finalise path runs for real with ffmpeg + libx264 on synthetic frames
(SegmentRun is fed numpy frames instead of Lada's restorer); the rest is pure logic.
"""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
from fractions import Fraction

import numpy as np
import oh_streamer.processing.compare_stack as compare_stack  # noqa: E402
import oh_streamer.queue.queue_worker as Q  # noqa: E402
import oh_streamer.processing.rife_interp as R  # noqa: E402

TMP_ROOT = tempfile.mkdtemp(prefix="qwtest_")
W, H = 128, 96


def probe(path):
    return Q.ffprobe_json(path)


def make_source(path, seconds=4, fps=30, audio=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=size={W}x{H}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


class StrRife:
    """Stand-in for Rife on string frames "f<i>": a scene cut (None) when i is a multiple of 10."""

    def middle(self, f0, f1):
        return None if int(f1[1:]) % 10 == 0 else ("mid", f0, f1)


class AvgRife:
    """Stand-in for Rife on numpy frames: the average, never a scene cut."""

    def middle(self, f0, f1):
        f0, f1 = getattr(f0, "arr", f0), getattr(f1, "arr", f1)      # Fr stand-ins carry their pixels in .arr
        return ((f0.astype(np.int16) + f1) // 2).astype(np.uint8)


class FakeInterpolator:
    def __init__(self, rife=None, error=None):
        self.rife, self.error = rife, error

    def prepare(self, w, h):
        pass

    def get(self, w, h):
        return self.rife

    def error_for(self, w, h):
        return self.error


class Fr:
    """Stand-in for a restored frame plus the ROI patches the engine hands out with it (RestoredFrame.original_patches)."""

    def __init__(self, arr, patches):
        self.arr = arr
        self.patches = patches


def push(stage, fr, arr, t=None):
    """What the restoring loop does: compare stages also get the frame's original patches."""
    if isinstance(stage, Q.CompareStage):
        return stage.push(fr, arr, t, fr.patches)
    return stage.push(fr, arr, t)


def types_none():
    """A stand-in for sys.stdin being None / unusable."""
    return None


def frame(i):
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :, 1] = (np.arange(W)[None, :] + i * 3) % 256
    f[:8, :, 2] = 255 if i % 2 else 0
    return f


class Pure(unittest.TestCase):
    def test_plan_30fps(self):
        p = Q.SegmentPlan(Fraction(30), 300, 0.0, 4)
        self.assertEqual((p.n, p.count), (120, 3))
        self.assertEqual([p.frames_in(k) for k in range(3)], [120, 120, 60])
        self.assertEqual(p.start_ns(0), 0)
        self.assertAlmostEqual(p.source_time(2), 8.0)

    def test_plan_ntsc_start_ns_not_truncated(self):
        p = Q.SegmentPlan(Fraction(30000, 1001), 1000, 0.0, 4)
        self.assertEqual(p.n, 120)
        # exact start of segment 1 is 4.004 s; Lada does int(ns/1e9 * fps) and must land on frame 120, not 119
        ns = p.start_ns(1)
        self.assertGreater(ns, 4_004_000_000)
        self.assertEqual(int(Fraction(ns, 10**9) * p.fps), 120)
        self.assertEqual(int(Fraction(p.start_ns(7), 10**9) * p.fps), 840)

    def test_plan_video_start_and_last_partial(self):
        p = Q.SegmentPlan(Fraction(25), 251, 1.4, 4)
        self.assertEqual((p.n, p.count, p.frames_in(2)), (100, 3, 51))
        self.assertAlmostEqual(p.source_time(1), 5.4)
        self.assertEqual(p.start_ns(1), 5_400_000_000 + 1000)

    def test_normalize_job_defaults_and_notes(self):
        job, notes = Q.normalize_job({"source": "a.mp4", "work_dir": "w", "output": "o.mkv",
                                      "settings": {"interpolation": "full", "compare": "on", "max_clip_length": 30}})
        st = job["settings"]
        self.assertEqual((st["interpolation"], st["compare"], st["max_clip_length"]), ("full", "on", 30))
        self.assertEqual((st["detection_model"], st["restoration_model"], st["encoder"]),
                         ("v4-fast", "basicvsrpp-v1.2", "hevc"))
        self.assertEqual(job["segment_seconds"], 4.0)
        self.assertEqual(len(notes), 0)
        job, notes = Q.normalize_job({"source": "a.mp4", "work_dir": "w", "output": "o.mkv",
                                      "settings": {"interpolation": "auto"}})
        self.assertEqual((job["settings"]["interpolation"], len(notes)), ("off", 1))
        with self.assertRaises(Q.WorkerError):
            Q.normalize_job({"source": "a", "work_dir": "w", "output": "o", "settings": {"interpolation": "x"}})

    def test_cached_job_validates_loopback_url_and_uses_full_probe(self):
        raw = {"source": "stable.m3u8", "work_dir": "w", "output": "o.mkv", "source_url": "http://127.0.0.1:4321/vod.m3u8",
               "source_identity": "media-7", "source_generation": "gen-2", "source_status_path": "status.json",
               "source_probe": {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 128,
                                                "height": 96, "r_frame_rate": "30/1", "time_base": "1/90000",
                                                "start_time": "1.4", "duration": "2", "nb_frames": "60"}],
                                "format": {"duration": "2", "start_time": "1.4"}, "duration": 10.0}}
        job, _ = Q.normalize_job(raw)
        self.assertEqual(job["source_url"], raw["source_url"])
        self.assertEqual(job["source_probe"]["format"]["duration"], "10.0")
        self.assertNotIn("nb_frames", job["source_probe"]["streams"][0])
        info = Q.parse_probe(job["source_probe"])
        self.assertEqual((info.frames, info.duration, info.video_start), (300, 10.0, 1.4))
        for url in ("https://127.0.0.1:4321/vod.m3u8", "http://localhost:4321/vod.m3u8",
                    "http://example.com/vod.m3u8", "http://127.0.0.1/vod.m3u8"):
            with self.assertRaises(Q.WorkerError, msg=url):
                Q.normalize_job({**raw, "source_url": url})

    def test_cache_job_key_uses_identity_not_path_proxy_port_or_generation(self):
        info = Q.SourceInfo(Fraction(30), 300, 128, 96, 1.4, 10.0, True)
        plan = Q.SegmentPlan(info.fps, info.frames, info.video_start, 1.0)
        base = {"source": "one/local.m3u8", "source_identity": "media-7", "source_generation": 2,
                "settings": {"detection_model": "d", "restoration_model": "r", "fp16": None,
                              "max_clip_length": 30, "interpolation": "off", "compare": "off"}}
        a = Q.job_key({**base, "source_url": "http://127.0.0.1:40001/a"}, info, plan, "libx264")
        b = Q.job_key({**base, "source": "other/path.m3u8", "source_url": "http://127.0.0.1:40002/b"},
                      info, plan, "libx264")
        c = Q.job_key({**base, "source_generation": 3}, info, plan, "libx264")
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertNotIn("source_id", a)

    def test_cached_source_completion_gate_rejects_incomplete_generation_short_eof_and_pts_gap(self):
        status = {"version": 1, "identity": "media-7", "generation": "gen-2", "status": "complete",
                  "error": None, "ready_segments": 5, "segment_count": 5}
        fps, start, duration = Fraction(10), 1.4, 1.0
        pts = [start + i / 10 for i in range(10)]
        self.assertIsNone(Q.source_completion_error(status, "media-7", "gen-2", pts, fps, duration, start))
        for bad_status in ({**status, "status": "downloading"}, {**status, "status": "failed", "error": "disk full"},
                           {**status, "generation": "new"}):
            self.assertIsNotNone(Q.source_completion_error(bad_status, "media-7", "gen-2", pts,
                                                           fps, duration, start), bad_status)
        self.assertIn("expected VOD end", Q.source_completion_error(status, "media-7", "gen-2", pts[:-1],
                                                                      fps, duration, start))
        self.assertIn("PTS gap", Q.cache_pts_error([start, start + 0.1, start + 0.3], fps))

    def test_cached_finalize_uses_runtime_url_and_compares_timeline_span(self):
        with tempfile.TemporaryDirectory() as td:
            seg_dir = os.path.join(td, "segments")
            os.makedirs(seg_dir)
            seg_path = os.path.join(seg_dir, Q.segment_name(0))
            _Path(seg_path).write_bytes(b"segment")
            status_path = os.path.join(td, "source-status.json")
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "identity": "media-7", "generation": "gen-2", "status": "complete",
                           "error": None, "ready_segments": 1, "segment_count": 1}, f)
            job = {"source": os.path.join(td, "stable.m3u8"), "source_url": "http://127.0.0.1:4321/vod.m3u8",
                   "source_identity": "media-7", "source_generation": "gen-2", "source_status_path": status_path,
                   "output": os.path.join(td, "out.mkv")}
            info = Q.SourceInfo(Fraction(10), 100, 128, 96, 1.4, 10.0, True)
            plan = Q.SegmentPlan(info.fps, info.frames, info.video_start, 10)
            manifest = Q.Manifest(os.path.join(td, "segments.json"), {})
            manifest.add(0, 100, file=Q.segment_name(0))
            manifest.mark_complete(100, 0)
            probe = {"streams": [{"codec_type": "video", "start_time": "1.466"},
                                 {"codec_type": "audio", "start_time": "1.466"}],
                     "format": {"duration": "11.466", "start_time": "1.466"}}
            self.assertTrue(Q.finalized_duration_ok(probe, 10.0, job))
            self.assertFalse(Q.finalized_duration_ok(probe, 10.0, {"source": "legacy.mp4"}))
            commands = []

            def fake_run(cmd, **kwargs):
                commands.append(cmd)
                _Path(cmd[-1]).write_bytes(b"finalized")
                return types.SimpleNamespace(returncode=0, stderr="")

            with unittest.mock.patch.object(Q.subprocess, "run", side_effect=fake_run), \
                    unittest.mock.patch.object(Q, "ffprobe_json", return_value=probe), \
                    unittest.mock.patch.object(Q, "check_tracks", return_value=None):
                Q.finalize(job, info, plan, manifest, td, seg_dir)
            self.assertIn(job["source_url"], commands[0])
            self.assertNotIn(job["source"], commands[0])

    def test_effective_interpolation(self):
        e = Q.effective_interpolation
        self.assertEqual(e("full", Fraction(30)), ("full", ""))
        self.assertEqual(e("full", Fraction(30000, 1001))[0], "full")
        self.assertEqual(e("full", Fraction(40))[0], "full")
        for bad in (Fraction(60), Fraction(50), Fraction(5), Fraction(0)):
            mode, why = e("full", bad)
            self.assertEqual(mode, "off")
            self.assertTrue(why)
        self.assertEqual(e("off", Fraction(30)), ("off", ""))

    def test_job_key_has_interpolation(self):
        def key(interp, fps):
            info = Q.SourceInfo(Fraction(fps), fps * 10, 128, 96, 0.0, 10.0, True)
            job = {"source": "nonexistent.mp4",
                   "settings": {"detection_model": "d", "restoration_model": "r", "fp16": None, "max_clip_length": 30,
                                "interpolation": interp, "compare": "off"}}
            return Q.job_key(job, info, Q.SegmentPlan(info.fps, info.frames, 0, 1.0), "libx264")
        self.assertNotEqual(key("off", 30), key("full", 30))
        self.assertEqual(key("full", 30)["interpolation"], "full")
        self.assertEqual(key("off", 60), key("full", 60))     # not eligible: 1x either way, segments stay reusable

    def test_closed_source_frames(self):
        f = Q.closed_source_frames
        self.assertEqual(f(1, 0, 100, 2, 401, False), 100)    # segment 1 closes when pipe frame 400 (first of seg 2) is in
        self.assertIsNone(f(2, 0, 100, 2, 401, False))        # segment 2 only just started
        self.assertEqual(f(3, 0, 100, 2, 2 * 330, True), 30)  # last segment: 30 source frames = 60 pipe frames
        self.assertIsNone(f(3, 0, 100, 2, 2 * 330, False))
        self.assertIsNone(f(3, 0, 100, 2, 2 * 330 - 1, True))  # odd pipe count: not a whole number of source frames
        self.assertEqual(f(5, 5, 100, 1, 100, True), 100)      # 1x unchanged

    def run_doubler(self, first, last, finish):
        d = R.FullDoubler(FakeInterpolator(StrRife()), 30.0, 8, 8, log=lambda m: None)
        out = []
        for i in range(first, last):
            out += d.push(f"f{i}", f"a{i}", None)
        return out + (d.finish() if finish else [])

    def test_full_doubler_boundary_split_equals_continuous(self):
        """Resume at a segment boundary: the previous run's output + the resumed run's output == one run's output."""
        total, n = 100, 30
        full = self.run_doubler(0, total, True)
        self.assertEqual(len(full), 2 * total)                 # exactly 2N output frames
        self.assertEqual(full[:3], ["a0", ("mid", "f0", "f1"), "a1"])
        self.assertEqual(full[-3:], [("mid", "f98", "f99"), "a99", "a99"])   # ... and the closing copy of the last frame
        for k in (1, 2, 3):                                    # killed once the first frame of segment k was pushed
            a = self.run_doubler(0, k * n + 1, False)[:2 * k * n]
            b = self.run_doubler(k * n, total, True)           # resumed: fresh stage, first frame alone
            self.assertEqual(len(a), 2 * k * n)
            self.assertEqual(len(b), 2 * (total - k * n))
            self.assertEqual(a + b, full)
        # per-segment output frame counts: 2N, 2N, 2N and 2 * 10 for the short last segment
        self.assertEqual([len(full[i:i + 2 * n]) for i in range(0, 2 * total, 2 * n)], [60, 60, 60, 20])
        # the pair f59/f60 straddles the boundary; f60 is a "scene cut" here: the segment before ends with a repeat of f59
        self.assertEqual((full[2 * 60 - 1], full[2 * 60]), ("a59", "a60"))
        # f58/f59 is interpolated and right after f58
        self.assertEqual(full[2 * 58 + 1], ("mid", "f58", "f59"))

    def test_full_doubler_errors_are_not_hidden(self):
        d = R.FullDoubler(FakeInterpolator(None, error="boom"), 30.0, 8, 8, log=lambda m: None)
        d.push("f0", "a0", None)
        with self.assertRaises(RuntimeError):
            d.push("f1", "a1", None)

    def test_normalize_job_rejects(self):
        for bad in ({}, {"source": "a", "work_dir": "w"},
                    {"source": "a", "work_dir": "w", "output": "o", "settings": {"encoder": "nope"}},
                    {"source": "a", "work_dir": "w", "output": "o", "settings": {"max_clip_length": 0}},
                    {"source": "a", "work_dir": "w", "output": "o", "segment_seconds": 0}):
            with self.assertRaises(Q.WorkerError):
                Q.normalize_job(bad)

    def test_resolve_encoder(self):
        yes, no = (lambda e: True), (lambda e: False)
        self.assertEqual(Q.resolve_encoder("hevc", yes), "hevc_nvenc")
        self.assertEqual(Q.resolve_encoder("hevc_nvenc", no), "libx265")
        self.assertEqual(Q.resolve_encoder("auto", yes), "h264_nvenc")
        self.assertEqual(Q.resolve_encoder("auto", no), "libx264")
        self.assertEqual(Q.resolve_encoder("h264_nvenc", no), "libx264")
        self.assertEqual(Q.resolve_encoder("libx264", no), "libx264")
        with self.assertRaises(Q.WorkerError):
            Q.resolve_encoder("av1", yes)

    def test_encode_cmd(self):
        p = Q.SegmentPlan(Fraction(30000, 1001), 1000, 0.0, 4)
        cmd = Q.build_encode_cmd(p, 3, 1920, 1080, "hevc_nvenc", "S", 1)
        s = " ".join(cmd)
        self.assertIn("-c:v hevc_nvenc", s)
        self.assertIn("expr:eq(mod(n,120),0)", s)
        self.assertIn("-segment_start_number 3", s)
        self.assertIn("-reset_timestamps 1", s)
        self.assertIn("-s 1920x1080", s)
        self.assertIn("-framerate 30000/1001", s)
        self.assertEqual(cmd[-1], os.path.join("S", "seg_%06d.mkv"))
        # a doubler writes 2 pipe frames per source frame: twice the rate, keyframe every 2N pipe frames
        cmd2 = " ".join(Q.build_encode_cmd(p, 0, 640, 360, "libx264", "S", 2))
        self.assertIn("-framerate 60000/1001", cmd2)
        self.assertIn("expr:eq(mod(n,240),0)", cmd2)
        # segment_time is a hair under the exact length (4.004 s)
        st = float(cmd[cmd.index("-segment_time") + 1])
        self.assertTrue(4.0039 < st < 4.004)
        with self.assertRaises(Q.WorkerError):
            Q.build_encode_cmd(p, 0, 64, 64, "bogus", "S")

    def settings(self, **kw):
        return dict({"detection_model": "d", "restoration_model": "r", "fp16": None, "max_clip_length": 30,
                     "interpolation": "off", "compare": "on", "compare_original": "copy", "compare_quality": "small"}, **kw)

    def test_compare_plan(self):
        e = Q.compare_plan
        info = Q.SourceInfo(Fraction(30), 300, 1920, 1080, 0.0, 10.0, True)
        self.assertEqual(e(self.settings(compare="off"), info, "hevc_nvenc"), Q.NO_COMPARE)
        cp = e(self.settings(), info, "hevc_nvenc")                       # copy: plain single-track pipe
        self.assertEqual((cp.on, cp.original, cp.layers, cp.out_layers, cp.notes),
                         (True, "copy", ("processed",), ("processed", "original"), ()))
        cp = e(self.settings(compare_original="encode"), info, "hevc_nvenc")
        self.assertEqual((cp.on, cp.original, cp.layers, cp.out_layers),
                         (True, "encode", ("processed", "original"), ("processed", "original")))
        # never a stored restored track, whatever the interpolation
        self.assertNotIn("restored", e(self.settings(compare_original="encode", interpolation="full"), info,
                                       "hevc_nvenc").layers)
        # not copyable -> encode, with the reason
        bad = Q.SourceInfo(Fraction(30), 300, 1920, 1080, 0.0, 10.0, True, "codec vp9 (copy needs h264 / hevc)")
        cp = e(self.settings(), bad, "hevc_nvenc")
        self.assertEqual((cp.on, cp.original, cp.layers), (True, "encode", ("processed", "original")))
        self.assertIn("vp9", cp.notes[0])
        # encode refused by compare_ok (4K60, software 1080p30): compare off; copy has no such limit
        big = Q.SourceInfo(Fraction(60), 600, 3840, 2160, 0.0, 10.0, False)
        cp = e(self.settings(compare_original="encode"), big, "hevc_nvenc")
        self.assertEqual((cp.on, cp.original, cp.layers), (False, None, ("processed",)))
        self.assertTrue(cp.notes)
        self.assertFalse(e(self.settings(compare_original="encode"), info, "libx265").on)
        self.assertEqual(e(self.settings(), info, "libx265").original, "copy")
        self.assertEqual(e(self.settings(), big, "hevc_nvenc").original, "copy")

    def test_job_key_has_compare(self):
        def key(compare, interp="off", w=1280, h=720, fps=30, orig="copy", q="small", why=None):
            info = Q.SourceInfo(Fraction(fps), fps * 10, w, h, 0.0, 10.0, True, why)
            job = {"source": "nonexistent.mp4",
                   "settings": self.settings(compare=compare, interpolation=interp, compare_original=orig, compare_quality=q)}
            return Q.job_key(job, info, Q.SegmentPlan(info.fps, info.frames, 0, 1.0), "hevc_nvenc")
        self.assertEqual(key("off")["compare"], "off")
        self.assertNotIn("original", key("off"))
        self.assertEqual(key("off", orig="encode", q="tiny"), key("off"))        # compare off: the new settings are ignored
        self.assertEqual((key("on")["compare"], key("on")["original"]), ("on", "copy"))
        self.assertNotIn("compare_quality", key("on"))                          # a copy has no quality
        self.assertEqual(key("on", q="tiny"), key("on", q="same"))
        self.assertEqual((key("on", orig="encode")["original"], key("on", orig="encode")["compare_quality"]),
                         ("encode", "small"))
        self.assertNotEqual(key("on", orig="encode"), key("on", orig="copy"))
        self.assertNotEqual(key("on", orig="encode", q="small"), key("on", orig="encode", q="tiny"))
        self.assertNotEqual(key("on"), key("off"))
        self.assertNotEqual(key("on", "full"), key("on", "off"))
        # a copy that falls back to encode has the encode key (the effective mode counts)
        self.assertEqual(key("on", why="codec vp9"), key("on", orig="encode"))
        self.assertEqual(key("on", orig="encode", w=3840, h=2160, fps=60), key("off", w=3840, h=2160, fps=60))   # refused
        # an old 3-layer compare key ({"compare": "on"} only) can never match
        old = dict(key("on"))
        old.pop("original")
        self.assertNotEqual(old, key("on"))

    def test_normalize_job_compare_settings(self):
        base = {"source": "a", "work_dir": "w", "output": "o"}
        st = Q.normalize_job(dict(base))[0]["settings"]
        self.assertEqual((st["compare_original"], st["compare_quality"]), ("copy", "small"))
        st = Q.normalize_job(dict(base, settings={"compare_original": "encode", "compare_quality": "tiny"}))[0]["settings"]
        self.assertEqual((st["compare_original"], st["compare_quality"]), ("encode", "tiny"))
        for bad in ({"compare_original": "x"}, {"compare_quality": "huge"}, {"compare_original": None}):
            with self.assertRaises(Q.WorkerError):
                Q.normalize_job(dict(base, settings=bad))

    def test_encode_cmd_compare(self):
        p = Q.SegmentPlan(Fraction(30), 300, 0.0, 4)
        layers = ("processed", "original")
        cmd = Q.build_encode_cmd(p, 2, 1280, 720 * 2, "hevc_nvenc", "S", 2, layers, "", 8)
        s = " ".join(cmd)
        self.assertIn("-s 1280x1440 -framerate 60 ", s)
        i = cmd.index("-filter_complex")
        # exactly compare_stack's split / crop, the original reduced to the real (even) slots at the source rate
        self.assertEqual(cmd[i:i + 2], compare_stack.queue_layer_args(1280, 720, layers, 2, p.fps)[0])
        self.assertIn("[s0]crop=1280:720:0:0,format=yuv420p[v_processed]", cmd[i + 1])
        self.assertIn("[s1]crop=1280:720:0:720,select='not(mod(n,2))',fps=30,format=yuv420p[v_original]", cmd[i + 1])
        self.assertNotIn("mpdecimate", s)
        self.assertNotIn("restored", s)
        self.assertEqual([cmd[j + 1] for j, a in enumerate(cmd) if a == "-map"], ["[v_processed]", "[v_original]"])
        self.assertEqual([cmd[j + 1] for j, a in enumerate(cmd) if a.startswith("-metadata:s:v:")],
                         ["title=oh:processed", "title=oh:original"])
        self.assertEqual([cmd[j + 1] for j, a in enumerate(cmd) if a.startswith("-disposition:v:")], ["default", "0"])
        self.assertIn("-c:v hevc_nvenc", s)                      # the main encoder family, not compare_stack's h264
        self.assertNotIn("h264_nvenc", s)
        # GOP = one segment on both tracks (no short compare GOP): processed 2N pipe frames, original N frames
        self.assertIn("-g:v:0 240", s)
        self.assertIn("-g:v:1 120", s)
        self.assertNotIn("-g 15", s)
        self.assertIn("-force_key_frames:v:0 expr:eq(mod(n,240),0)", s)
        self.assertIn("-force_key_frames:v:1 expr:eq(mod(n,120),0)", s)
        self.assertIn("-cq:v:0 23", s)                           # CQ = main + offset on the original only
        self.assertIn("-cq:v:1 31", s)
        self.assertIn("-segment_start_number 2", s)
        st = float(cmd[cmd.index("-segment_time") + 1])
        self.assertTrue(3.9999 < st < 4.0)
        # 1x: no select / fps, both tracks N frames
        c1 = " ".join(Q.build_encode_cmd(p, 0, 1280, 1440, "libx264", "S", 1, layers))
        self.assertNotIn("select", c1)
        self.assertNotIn("fps=", c1)
        self.assertIn("-g:v:0 120 ", c1 + " ")
        self.assertIn("-g:v:1 120", c1)
        self.assertIn("-crf:v:0 20", c1)
        self.assertIn("-crf:v:1 20", c1)
        x265 = " ".join(Q.build_encode_cmd(p, 0, 1280, 1440, "libx265", "S", 2, layers, "", 4))
        self.assertIn("-crf:v:1 28", x265)
        self.assertIn("-x265-params:v:0 keyint=240:min-keyint=240", x265)
        self.assertIn("-x265-params:v:1 keyint=120:min-keyint=120", x265)
        with self.assertRaises(Q.WorkerError):
            Q.build_encode_cmd(p, 0, 1280, 1441, "hevc_nvenc", "S", 1, layers)
        one = Q.build_encode_cmd(p, 0, 1280, 720, "hevc_nvenc", "S", 1)       # no compare / copy: plain single track
        self.assertNotIn("-filter_complex", one)
        self.assertIn("-g 120", " ".join(one))
        self.assertIn("-cq 23", " ".join(one))
        self.assertNotIn("-cq:v", " ".join(one))
        self.assertEqual(Q.quality_args("libx264", "", 8), ["-crf", "28"])
        self.assertEqual(Q.quality_args("h264_nvenc", ":v:1", 4), ["-cq:v:1", "23"])

    def test_finalize_cmd_compare(self):
        tags = compare_stack.lada_tags(Fraction(30000, 1001), 2, 0.033367, True, "copy")
        # copy: the source's video next to the (single track) segments, no -itsoffset on the source input
        cmd = Q.build_finalize_cmd("l.txt", "src.mp4", True, 0.033367, "o.mkv", ("processed",), True, tags)
        s = " ".join(cmd)
        self.assertEqual(cmd.count("-itsoffset"), 1)
        self.assertLess(cmd.index("-itsoffset"), cmd.index("-f"))                       # before the concat input
        self.assertIn("-itsoffset 0.033367 -f concat -safe 0 -i l.txt -i src.mp4 -map 0:v:0 -map 1:v:0 -map 1:a -c copy", s)
        self.assertEqual(cmd[cmd.index("-c") + 2:cmd.index("-metadata")],
                         compare_stack.layer_meta_args(("processed", "original")))
        self.assertIn("-metadata OH_SOURCE_FPS=30000/1001", s)
        self.assertIn("-metadata OH_ORIGINAL=copy", s)
        # copy without audio still needs the source input
        s = " ".join(Q.build_finalize_cmd("l.txt", "src.mp4", False, 0.0, "o.mkv", ("processed",), True, tags))
        self.assertIn("-i src.mp4 -map 0:v:0 -map 1:v:0 -c copy", s)
        self.assertNotIn("1:a", s)
        # encode: both video tracks come out of the segments
        layers = ("processed", "original")
        cmd = Q.build_finalize_cmd("l.txt", "src.mp4", True, 0.0, "o.mkv", layers, False, tags)
        s = " ".join(cmd)
        self.assertIn("-map 0:v -map 1:a", s)
        self.assertNotIn("0:v:0", s)
        self.assertNotIn("1:v", s)
        self.assertEqual(cmd[cmd.index("-c") + 2:cmd.index("-metadata")], compare_stack.layer_meta_args(layers))
        self.assertIsNone(Q.check_tracks({"streams": [
            {"codec_type": "video", "tags": {"title": "oh:processed"}, "disposition": {"default": 1}},
            {"codec_type": "video", "tags": {"title": "oh:original"}, "disposition": {"default": 0}}]},
            ("processed", "original")))
        self.assertIn("titles", Q.check_tracks({"streams": [{"codec_type": "video", "tags": {}},
                                                            {"codec_type": "video", "tags": {}}]},
                                               ("processed", "original")))
        self.assertIn("1 video", Q.check_tracks({"streams": [{"codec_type": "video"}]}, ("processed", "original")))
        self.assertIsNone(Q.check_tracks({"streams": [{"codec_type": "video"}]}, ("processed",)))
        # tags: checked case-insensitively on the key, exact on the value
        pr = {"streams": [{"codec_type": "video"}], "format": {"tags": {"oh_source_fps": "30000/1001"}}}
        self.assertIsNone(Q.check_tracks(pr, ("processed",), {"OH_SOURCE_FPS": "30000/1001"}))
        self.assertIn("tags", Q.check_tracks(pr, ("processed",), {"OH_SOURCE_FPS": "25"}))
        self.assertIn("tags", Q.check_tracks(pr, ("processed",), {"OH_RATE_MULT": "1"}))

    def test_lada_tags(self):
        t = compare_stack.lada_tags(Fraction(30000, 1001), 2, 0.0333667, True, "encode")
        self.assertEqual(t, {"OH_SOURCE_FPS": "30000/1001", "OH_RATE_MULT": "2", "OH_VIDEO_START": "0.033367",
                             "OH_COMPARE": "1", "OH_ORIGINAL": "encode"})
        t = compare_stack.lada_tags(Fraction(25), 1, 0, False, None)
        self.assertEqual((t["OH_SOURCE_FPS"], t["OH_COMPARE"], t["OH_ORIGINAL"], t["OH_VIDEO_START"]),
                         ("25", "0", "none", "0.000000"))
        self.assertEqual(compare_stack.tag_args({"A": "1"}), ["-metadata", "A=1"])

    def test_compare_stage_frames(self):
        """Frame-count / sync logic: the stage writes exactly what its inner stage would, one stacked frame each."""
        a, b, c = frame(1), frame(2), frame(3)
        pa, pc = np.full((2, 2, 3), 7, np.uint8), np.full((2, 2, 3), 9, np.uint8)
        fa = Fr(a, [((0, 0, 0, 0), np.array([[[100, 100, 100]]], np.uint8))])
        names = ("processed", "original", "restored")
        st = Q.CompareStage(R.FullDoubler(FakeInterpolator(AvgRife()), 30.0, W, H, log=lambda m: None), names)
        self.assertEqual((st.rate_mult, st.layers, st.layer_names), (2, 3, names))
        o1 = push(st, fa, a)
        self.assertEqual(len(o1), 1)                                   # first frame alone
        self.assertEqual(o1[0].shape, (3 * H, W, 3))
        o1 = [x.copy() for x in o1]
        o2 = push(st, Fr(b, None), b)
        self.assertEqual(len(o2), 2)                                   # middle, frame
        parts = compare_stack.split_stacked(o2[0], names)
        self.assertTrue((parts["processed"] == AvgRife().middle(a, b)).all())      # processed: the RIFE frame
        self.assertTrue((parts["restored"] == a).all())                # 1x layers repeat the previous frame ...
        self.assertTrue((parts["original"][0, 0] == [100, 100, 100]).all())       # ... with its original patch
        self.assertTrue((parts["original"][1:] == a[1:]).all())
        parts = compare_stack.split_stacked(o2[1], names)
        self.assertTrue((parts["processed"] == b).all() and (parts["restored"] == b).all()
                        and (parts["original"] == b).all())
        o3 = st.finish()
        self.assertEqual(len(o3), 1)
        parts = compare_stack.split_stacked(o3[0], names)
        self.assertTrue((parts["processed"] == b).all() and (parts["restored"] == b).all())
        # N source frames in = 2N stacked frames out (1 + 2 + finish 1 = 4 for N = 2)
        self.assertEqual(len(o1) + len(o2) + len(o3), 4)
        # plain stage + compare (interpolation off): processed + original, one frame per frame
        st = Q.CompareStage(Q.PlainStage(), ("processed", "original"))
        self.assertEqual((st.rate_mult, st.layers), (1, 2))
        out = push(st, fa, a)
        self.assertEqual((len(out), out[0].shape), (1, (2 * H, W, 3)))
        parts = compare_stack.split_stacked(out[0], ("processed", "original"))
        self.assertTrue((parts["processed"] == a).all())
        self.assertTrue((parts["original"][0, 0] == [100, 100, 100]).all())
        self.assertEqual(st.finish(), [])

    def test_finalize_cmd(self):
        cmd = Q.build_finalize_cmd("l.txt", "src.mp4", True, 1.4, "o.mkv")
        s = " ".join(cmd)
        self.assertIn("-itsoffset 1.400000 -f concat", s)
        self.assertIn("-map 0:v:0 -map 1:a", s)
        self.assertIn("-c copy", s)
        s2 = " ".join(Q.build_finalize_cmd("l.txt", "src.mp4", False, 0.0, "o.mkv"))
        self.assertNotIn("-itsoffset", s2)
        self.assertNotIn("1:a", s2)
        self.assertNotIn("src.mp4", s2)
        cached = Q.build_finalize_cmd("l.txt", "http://127.0.0.1:4321/vod.m3u8", True, 1.4, "o.mkv",
                                      preserve_source_clock=True)
        self.assertIn("-copyts", cached)
        self.assertLess(cached.index("-copyts"), cached.index("-i"))

    def test_concat_list(self):
        t = Q.concat_list_text([r"C:\a b\seg_000000.mkv", "it's.mkv"])
        self.assertIn("file '", t)
        self.assertNotIn("\\a", t.split("\n")[0].replace("\\'", ""))
        self.assertIn("'\\''", t)

    def test_duration_ok(self):
        self.assertTrue(Q.duration_ok(10.3, 10.0))
        self.assertFalse(Q.duration_ok(10.6, 10.0))

    def test_closed_segment_frames(self):
        f = Q.closed_segment_frames
        self.assertEqual(f(1, 0, 100, 201, False), 100)        # mid-run close of segment 1 (frame 201 starts segment 2)
        self.assertIsNone(f(2, 0, 100, 201, False))             # segment 2 only just started: cannot be closed
        self.assertEqual(f(1, 0, 100, 250, True), 100)          # late-processed mid-run close after the end
        self.assertEqual(f(3, 0, 100, 330, True), 30)           # last, short segment at a natural end
        self.assertIsNone(f(3, 0, 100, 330, False))             # short without natural end: truncated, not trusted
        self.assertEqual(f(5, 5, 100, 100, True), 100)          # resumed run, last segment exactly full
        self.assertIsNone(f(6, 5, 100, 100, True))              # nothing fed for it

    def test_progress_throttle(self):
        now = [0.0]
        t = Q.ProgressThrottle(0.5, clock=lambda: now[0])
        emitted = 0
        for i in range(1000):             # a frame every 5 ms for 5 s
            now[0] = i * 0.005
            if t.ready():
                emitted += 1
        self.assertEqual(emitted, 10)     # 2 per second
        now[0] = 5.0001
        self.assertTrue(t.ready(force=True))

    def test_rate_meter(self):
        now = [0.0]
        m = Q.RateMeter(5.0, clock=lambda: now[0])
        self.assertEqual(m.fps(), 0.0)
        self.assertIsNone(m.eta(0, 100))
        for i in range(0, 41):
            now[0] = i * 0.5
            m.tick(i * 10)                # 20 fps
        self.assertAlmostEqual(m.fps(), 20.0, places=1)
        self.assertAlmostEqual(m.eta(400, 600), 10.0, places=0)

    def test_progress_line(self):
        d = Q.progress_line(12, 10, 30.123, None, 2)
        self.assertEqual((d["type"], d["frame"], d["frames"], d["eta"], d["k"]), ("progress", 12, 12, None, None))
        self.assertNotIn("source_fps", d)
        self.assertNotIn("duration", d)
        json.dumps(d)
        d = Q.progress_line(100, 500, 20.0, 20.0, 3, "restore", 4, source_fps=29.97, duration=16.68)
        self.assertEqual((d["segments"], d["k"], d["phase"], d["source_fps"], d["duration"]),
                         (3, 4, "restore", 29.97, 16.68))

    def test_run_job_finalize_progress_uses_completed_manifest_frames(self):
        from oh_streamer.processing import device_backend

        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as td:
            src = os.path.join(td, "source.mp4")
            _Path(src).touch()
            work = os.path.join(td, "work")
            output = os.path.join(td, "output.mkv")
            info = Q.SourceInfo(Fraction(30), 30, 128, 96, 0.0, 1.0, False)
            manifest = unittest.mock.Mock()
            manifest.segments = [{"k": 0, "frames": 30}]
            manifest.frames_done.return_value = 30
            manifest.next_k = 1
            manifest.partial_ks.return_value = set()
            manifest.is_complete.return_value = True
            emitted = []
            adapter_name = "oh_streamer.engines.lada.adapter"
            fake_adapter = types.ModuleType(adapter_name)
            fake_adapter.upstream_gpu_has_fp16_acceleration = lambda device: False
            job = {"source": src, "work_dir": work, "output": output, "segment_seconds": 1.0,
                   "settings": {"device": "cpu", "fp16": False, "interpolation": "off", "encoder": "libx264",
                                "engine": "lada"}}

            with unittest.mock.patch.dict(_sys.modules, {adapter_name: fake_adapter}), \
                 unittest.mock.patch.object(Q, "upstream_root", return_value=_Path(td)), \
                 unittest.mock.patch.object(device_backend, "select_device", return_value="cpu"), \
                 unittest.mock.patch.object(device_backend, "rife_available", return_value=True), \
                 unittest.mock.patch.object(Q, "ffprobe_json", return_value={}), \
                 unittest.mock.patch.object(Q, "parse_probe", return_value=info), \
                 unittest.mock.patch.object(Q, "resolve_encoder", return_value="libx264"), \
                 unittest.mock.patch.object(Q, "effective_interpolation", return_value=("off", "")), \
                 unittest.mock.patch.object(Q, "compare_plan", return_value=Q.ComparePlan(False, None,
                                                                                           ("processed",), [])), \
                 unittest.mock.patch.object(Q, "job_key", return_value={}), \
                 unittest.mock.patch.object(Q.Manifest, "load", return_value=manifest), \
                 unittest.mock.patch.object(Q, "engine_stamp", return_value=("lada", "test")), \
                 unittest.mock.patch.object(Q, "clean_work_dir", return_value=[]), \
                 unittest.mock.patch.object(Q, "play_block", return_value={}), \
                 unittest.mock.patch.object(Q.THROTTLE, "ready", return_value=True), \
                 unittest.mock.patch.object(Q, "finalize"), \
                 unittest.mock.patch.object(Q, "emit", side_effect=emitted.append), \
                 unittest.mock.patch.object(Q, "log"):
                self.assertEqual(Q.run_job(job, []), output)

            final_progress = next(msg for msg in emitted if msg.get("phase") == "finalize")
            self.assertEqual((final_progress["frame"], final_progress["frames"]), (30, 30))

    def test_start_k(self):
        base = {"source": "a.mp4", "work_dir": "w", "output": "o.mkv"}
        self.assertEqual(Q.normalize_job(dict(base))[0]["start_k"], 0)
        self.assertEqual(Q.normalize_job(dict(base, start_k=7))[0]["start_k"], 7)
        for bad in (-1, 1.5, "3", True, None):
            with self.assertRaises(Q.WorkerError):
                Q.normalize_job(dict(base, start_k=bad))
        # start_k is not part of the key: it never invalidates segments
        info = Q.SourceInfo(Fraction(30), 300, 128, 96, 0.0, 10.0, True)
        plan = Q.SegmentPlan(info.fps, info.frames, 0, 1.0)
        st = {"detection_model": "d", "restoration_model": "r", "fp16": None, "max_clip_length": 30,
              "interpolation": "off", "compare": "off"}

        def key(sk):
            return Q.job_key({"source": "nonexistent.mp4", "settings": st, "start_k": sk}, info, plan, "libx264")
        self.assertEqual(key(0), key(5))

    def test_first_gap(self):
        g = Q.first_gap
        self.assertEqual(g(set(), 0, 5), 0)
        self.assertEqual(g({0, 1, 3}, 0, 5), 2)
        self.assertEqual(g({0, 1, 3}, 3, 5), 4)           # first gap >= 3
        self.assertEqual(g({0, 1, 3, 4}, 3, 5), 2)        # none ahead: wraps to the first gap from 0
        self.assertEqual(g({1, 2, 3, 4}, 3, 5), 0)
        self.assertIsNone(g({0, 1, 2, 3, 4}, 2, 5))
        self.assertEqual(g({0, 1}, 9, 5), 2)              # start beyond the limit: wraps
        self.assertEqual(g({2}, -3, 5), 0)
        self.assertEqual(Q.run_end_k({0, 1, 5, 7}, 2), 5)
        self.assertEqual(Q.run_end_k({0, 1, 5, 7}, 5), 7)
        self.assertIsNone(Q.run_end_k({0, 1, 5, 7}, 7))
        self.assertEqual(Q.run_frame_limit(2, 5, 120), 360)
        self.assertIsNone(Q.run_frame_limit(2, None, 120))

    def test_gap_limit(self):
        m = Q.Manifest("unused.json", {"n": 100})
        self.assertEqual(m.gap_limit(10), 10)
        m.segments = [{"k": 0, "file": "a", "frames": 100}, {"k": 9, "file": "b", "frames": 100}]
        self.assertEqual(m.gap_limit(10), 11)             # estimate may be low: probe past the last finished segment
        m.last_k = 9
        self.assertEqual(m.gap_limit(10), 10)             # end known

    def test_goto_action(self):
        a = Q.goto_action
        fin = {0, 1, 5, 6}
        # run from 2 producing segment 2, stopping before 5
        self.assertEqual(a(0, fin, 2, 2, 5, 12), "next")             # finished: only steers the next gap search
        self.assertEqual(a(6, fin, 2, 2, 5, 12), "next")
        self.assertEqual(a(2, fin, 2, 2, 5, 12), "keep")             # being produced
        self.assertEqual(a(3, fin, 2, 2, 5, 12), "keep")
        self.assertEqual(a(4, fin, 2, 2, 5, 12), "keep")             # 2 ahead
        self.assertEqual(a(4, fin, 2, 3, 5, 12), "keep")
        self.assertEqual(a(4, fin, 2, 2, None, 12), "keep")
        self.assertEqual(a(8, fin, 2, 2, 5, 12), "abort")            # far ahead
        self.assertEqual(a(7, fin, 2, 4, None, 12), "abort")         # 3 ahead of 4
        self.assertEqual(a(7, fin, 2, 5, None, 12), "keep")
        self.assertEqual(a(4, fin, 2, 3, 4, 12), "abort")            # inside the 2-segment window but the run stops at 4
        self.assertEqual(a(3, {0, 1}, 4, 4, None, 12), "abort")      # behind the run's start: not produced by it
        self.assertEqual(a(2, {0, 1}, 4, 4, None, 12), "abort")
        self.assertEqual(a(12, fin, 2, 2, 5, 12), "ignore")          # outside the video
        self.assertEqual(a(-1, fin, 2, 2, 5, 12), "ignore")
        self.assertEqual(a(3, fin, None, None, None, 12), "abort")   # no run going

    def test_goto_during_redo(self):
        a = Q.goto_action
        fin = {0, 1, 2, 3, 4, 5}                  # no gaps during a redo
        # the same call that aborts a normal run (far ahead of the segment being produced) is just "next" in a redo
        self.assertEqual(a(5, fin - {5}, 1, 1, 3, 6), "abort")
        for k in (0, 1, 2, 5):
            self.assertEqual(a(k, fin, 1, 1, 3, 6, redo=True), "next")
        self.assertEqual(a(6, fin, 1, 1, 3, 6, redo=True), "ignore")
        self.assertEqual(a(-1, fin, 1, 1, 3, 6, redo=True), "ignore")

    def test_segment_names(self):
        self.assertEqual(Q.segment_name(5), "seg_000005.mkv")
        self.assertEqual(Q.segment_name(5, "_f"), "seg_000005_f.mkv")
        self.assertEqual((Q.file_suffix("seg_000005.mkv"), Q.file_suffix("seg_000005_f.mkv")), ("", "_f"))
        self.assertEqual((Q.alt_suffix("seg_000005.mkv"), Q.alt_suffix("seg_000005_f.mkv")), ("_f", ""))
        self.assertEqual(Q.parse_segment_name("seg_000012.mkv"), (12, ""))
        self.assertEqual(Q.parse_segment_name("seg_000012_f.mkv"), (12, "_f"))
        for bad in ("seg_x.mkv", "seg_000012.mp4", "foo.mkv", "seg__f.mkv", "seg_000012_g.mkv"):
            self.assertIsNone(Q.parse_segment_name(bad), bad)
        cmd = Q.build_encode_cmd(Q.SegmentPlan(Fraction(30), 300, 0.0, 4), 3, 128, 96, "libx264", "S", 1, ("processed",), "_f")
        self.assertTrue(cmd[-1].endswith("seg_%06d_f.mkv"))
        self.assertTrue(Q.build_encode_cmd(Q.SegmentPlan(Fraction(30), 300, 0.0, 4), 3, 128, 96, "libx264", "S")[-1]
                        .endswith("seg_%06d.mkv"))

    def test_redo_plan(self):
        def seg(k, partial=False, suffix=""):
            d = {"k": k, "file": Q.segment_name(k, suffix), "frames": 100}
            if partial:
                d["partial"] = True
            return d
        segs = [seg(0), seg(1, True), seg(2, True), seg(3), seg(4, True), seg(5)]
        r = Q.redo_plan
        self.assertEqual(r(segs, 0, 5), (1, 3, "_f"))             # consecutive partials 1..2, stops before the final 3
        self.assertEqual(r(segs, 2, 5), (2, 3, "_f"))             # a goto into the middle of a group starts there
        self.assertEqual(r(segs, 3, 5), (4, 5, "_f"))             # first partial >= start
        self.assertEqual(r(segs, 5, 5), (1, 3, "_f"))             # none ahead: wraps
        self.assertIsNone(r(segs, 0, None))                        # the video's end is unknown
        self.assertIsNone(r([seg(0), seg(1)], 0, 1))               # nothing partial
        self.assertIsNone(r([seg(0), seg(2, True)], 0, 2))         # a real gap (1) comes first
        # the group reaching the video's end runs "to the end"
        self.assertEqual(r([seg(0), seg(1, True), seg(2, True)], 0, 2), (1, None, "_f"))
        # the alternate name is the other one than the current file; a run does not mix the two variants
        mixed = [seg(0), seg(1, True), seg(2, True, "_f"), seg(3, True)]
        self.assertEqual(r(mixed, 0, 3), (1, 2, "_f"))
        self.assertEqual(r(mixed, 2, 3), (2, 3, ""))
        self.assertEqual(r([seg(0, True, "_f")], 0, 0), (0, None, ""))
        # segments beyond last_k are ignored
        self.assertIsNone(r([seg(0), seg(1), seg(2, True)], 0, 1))

    def test_policy_copies_and_make_doubler(self):
        class W:
            def __init__(self, *a):
                self.args, self.copied, self.rate_mult = a, 3, 2
        class F:
            def __init__(self, *a):
                self.args = a
        with unittest.mock.patch.object(Q.rife_interp, "WatchedDoubler", W, create=True), \
                unittest.mock.patch.object(Q.rife_interp, "FullDoubler", F):
            clock = object()
            w = Q.make_doubler("I", 30.0, 64, 48, clock, False, "L")
            self.assertIsInstance(w, W)
            self.assertEqual(w.args, ("I", 30.0, 64, 48, clock, "L"))
            self.assertIsInstance(Q.make_doubler("I", 30.0, 64, 48, clock, True, "L"), F)      # redo: always full
            self.assertIsInstance(Q.make_doubler("I", 30.0, 64, 48, None, False, "L"), F)
            self.assertEqual(Q.policy_copies(w), 3)
            self.assertEqual(Q.policy_copies(Q.CompareStage(w, ("processed", "original"))), 3)
        self.assertEqual(Q.policy_copies(Q.PlainStage()), 0)
        self.assertEqual(Q.policy_copies(F()), 0)

    def test_segment_speed_line(self):
        self.assertEqual(Q.segment_speed_line(41, 120, 3.9, 30.0, 37),
                         "segment 41: 120 frames in 3.9s (30.8 fps, 1.03x realtime, 37 copies)")
        self.assertEqual(Q.segment_speed_line(2, 120, 4.0, 30.0), "segment 2: 120 frames in 4.0s (30.0 fps, 1.00x realtime)")
        self.assertIn("(redone)", Q.segment_speed_line(2, 120, 4.0, 30.0, redo=True))
        self.assertEqual(Q.segment_speed_line(2, 120, 0.0, 30.0, 4), "segment 2: 120 frames in 0.0s (4 copies)")

    def test_parse_command(self):
        p = Q.parse_command
        self.assertEqual(p('{"cmd":"goto","k":7}'), ("goto", 7))
        self.assertEqual(p('{"cmd": "goto", "k": 0}\n'), ("goto", 0))
        for bad in ("", "nope", "[1]", '{"cmd":"goto"}', '{"cmd":"goto","k":-1}', '{"cmd":"goto","k":1.5}',
                    '{"cmd":"goto","k":"3"}', '{"cmd":"goto","k":true}', '{"cmd":"stop","k":3}', None):
            self.assertIsNone(p(bad), bad)

    def test_parse_command_clock(self):
        p = Q.parse_command
        self.assertEqual(p('{"cmd":"clock","pos":12.5,"playing":true}'), ("clock", 12.5, True))
        self.assertEqual(p('{"cmd": "clock", "pos": 0, "playing": false}\n'), ("clock", 0.0, False))
        self.assertEqual(p('{"cmd":"clock","pos":-0.5,"playing":true,"id":3}'), ("clock", -0.5, True))
        for bad in ('{"cmd":"clock"}', '{"cmd":"clock","pos":1.0}', '{"cmd":"clock","playing":true}',
                    '{"cmd":"clock","pos":"3","playing":true}', '{"cmd":"clock","pos":true,"playing":true}',
                    '{"cmd":"clock","pos":1.0,"playing":1}', '{"cmd":"clock","pos":null,"playing":true}',
                    '{"cmd":"clock","pos":NaN,"playing":true}', '{"cmd":"clock","pos":Infinity,"playing":true}'):
            self.assertIsNone(p(bad), bad)

    def test_command_reader(self):
        class Stream:
            def __init__(self, data):
                self.buffer = io.BytesIO(data)

            def isatty(self):
                return False
        q = Q.start_command_reader(Stream(b'garbage\n{"cmd":"goto","k":3}\n\xff\xfe\n{"cmd":"x"}\n{"cmd":"goto","k":9}'))
        self.assertEqual([q.get(timeout=5), q.get(timeout=5)], [3, 9])       # malformed lines skipped, EOF ends it quietly
        self.assertTrue(q.empty())
        self.assertTrue(Q.start_command_reader(Stream(b"")).empty())
        self.assertTrue(Q.start_command_reader(types_none()).empty())

    def test_command_reader_clock(self):
        class Stream:
            def __init__(self, data):
                self.buffer = io.BytesIO(data)

            def isatty(self):
                return False

        class Clock:
            def __init__(self):
                self.updates = []

            def update(self, pos, playing):
                self.updates.append((pos, playing))
        data = (b'{"cmd":"clock","pos":1.5,"playing":true}\n{"cmd":"goto","k":4}\n{"cmd":"clock","pos":2.5,"playing":false}\n'
                b'{"cmd":"clock","pos":"x","playing":true}\n')
        clock = Clock()
        q = Q.start_command_reader(Stream(data), clock=clock)
        self.assertEqual(q.get(timeout=5), 4)                 # clock lines never reach the goto queue
        for _ in range(100):
            if len(clock.updates) >= 2:
                break
            time.sleep(0.02)
        self.assertEqual(clock.updates, [(1.5, True), (2.5, False)])
        self.assertTrue(q.empty())
        q2 = Q.start_command_reader(Stream(data))             # no clock: clock lines are dropped quietly
        self.assertEqual(q2.get(timeout=5), 4)

    def test_play_block(self):
        plan = Q.SegmentPlan(Fraction(30000, 1001), 1000, 0.021, 4)
        self.assertEqual(Q.play_block(plan, "full", ("processed", "original"), "encode"),
                         {"fps": "30000/1001", "n": 120, "video_start": 0.021, "rate_mult": 2,
                          "layers": ["processed", "original"], "compare": True, "original": "encode", "count": plan.count})
        pb = Q.play_block(plan, "off", ("processed",))
        self.assertEqual((pb["rate_mult"], pb["compare"], pb["original"], pb["layers"]), (1, False, None, ["processed"]))
        pb = Q.play_block(plan, "full", ("processed",), "copy")      # copy: only processed in the segments
        self.assertEqual((pb["compare"], pb["original"], pb["layers"]), (True, "copy", ["processed"]))


class Bookkeeping(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP_ROOT)
        self.seg = os.path.join(self.dir, "segments")
        os.makedirs(self.seg)
        self.path = os.path.join(self.dir, "segments.json")
        self.key = {"n": 120, "encoder": "hevc_nvenc"}

    def touch(self, name, data=b"x"):
        with open(os.path.join(self.seg, name), "wb") as f:
            f.write(data)

    def test_roundtrip_and_resume(self):
        m = Q.Manifest(self.path, self.key)
        for k in range(3):
            self.touch(Q.segment_name(k))
            m.add(k, 120)
        self.touch(Q.segment_name(3))          # partial segment of a killed run: not listed
        self.touch("junk.tmp")
        m2 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m2.next_k, m2.frames_done(), m2.complete), (3, 360, False))
        removed = Q.clean_work_dir(self.seg, m2)
        self.assertEqual(sorted(removed), sorted([Q.segment_name(3), "junk.tmp"]))
        self.assertEqual(sorted(os.listdir(self.seg)), [Q.segment_name(k) for k in range(3)])
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_add_any_k_sorted_and_replace(self):
        m = Q.Manifest(self.path, self.key)
        for k, fr in ((5, 120), (1, 120), (3, 120), (1, 99)):
            self.touch(Q.segment_name(k))
            m.add(k, fr)
        self.assertEqual([(s["k"], s["frames"]) for s in m.segments], [(1, 99), (3, 120), (5, 120)])
        self.assertEqual(m.next_k, 0)
        self.assertEqual(m.finished(), {1, 3, 5})
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual((raw["version"], [s["k"] for s in raw["segments"]], raw["last_k"], raw["complete"]),
                         (2, [1, 3, 5], None, False))

    def test_v2_doc_and_play(self):
        play = {"fps": "30/1", "n": 120, "video_start": 0.0, "rate_mult": 1, "layers": ["processed"], "count": 3}
        m = Q.Manifest(self.path, self.key, play)
        for k, fr in ((2, 40), (0, 120)):
            self.touch(Q.segment_name(k))
            m.add(k, fr)
        d = m.doc()
        self.assertEqual((d["last_k"], d["complete"], d["play"]), (2, False, play))   # short top segment = the end; 1 missing
        self.touch(Q.segment_name(1))
        m.add(1, 120)
        d = m.doc()
        self.assertEqual((d["last_k"], d["complete"], d["total_frames"]), (2, True, 280))
        m2 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m2.complete, m2.is_complete(120), m2.end_k()), (True, True, 2))

    def test_v1_manifest_loads(self):
        for k in range(3):
            self.touch(Q.segment_name(k))
        v1 = {"version": 1, "key": self.key, "n": 120,
              "segments": [{"k": k, "file": Q.segment_name(k), "frames": 120} for k in range(3)],
              "complete": False, "total_frames": None}
        with open(self.path, "w") as f:
            json.dump(v1, f)
        m = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m.next_k, m.frames_done(), m.complete, m.last_k), (3, 360, False, None))
        v1.update(complete=True, total_frames=360)
        with open(self.path, "w") as f:
            json.dump(v1, f)
        m = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m.complete, m.last_k, m.total_frames, m.is_complete(120)), (True, 2, 360, True))
        m.save()                                   # re-saved as version 2
        with open(self.path) as f:
            self.assertEqual(json.load(f)["version"], 2)
        self.assertTrue(Q.Manifest.load(self.path, self.key, self.seg).complete)
        # a version 1 file that is complete but lost a file is not complete any more
        os.remove(os.path.join(self.seg, Q.segment_name(1)))
        with open(self.path, "w") as f:
            json.dump(v1, f)
        m = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual(m.next_k, 1)
        self.assertEqual(m.finished(), {0, 2})
        self.assertFalse(m.complete)
        # an unknown version is ignored
        with open(self.path, "w") as f:
            json.dump(dict(v1, version=3), f)
        self.assertEqual(Q.Manifest.load(self.path, self.key, self.seg).segments, [])

    def test_non_contiguous_load(self):
        m = Q.Manifest(self.path, self.key)
        for k in (0, 1, 4, 5):
            self.touch(Q.segment_name(k))
            m.add(k, 120)
        self.touch(Q.segment_name(2))                  # present but never listed: a leftover
        self.touch(Q.segment_name(6), b"")             # listed below with an empty file: dropped
        m.segments.append({"k": 6, "file": Q.segment_name(6), "frames": 120})
        m.save()
        m2 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual([s["k"] for s in m2.segments], [0, 1, 4, 5])
        self.assertEqual(m2.next_k, 2)
        self.assertEqual(m2.frames_done(), 480)
        self.assertEqual(Q.first_gap(m2.finished(), 0, m2.gap_limit(8)), 2)
        # clean_work_dir keeps every listed file, whatever the gaps, and removes the rest
        self.assertEqual(sorted(Q.clean_work_dir(self.seg, m2)), [Q.segment_name(2), Q.segment_name(6)])
        self.assertEqual(sorted(os.listdir(self.seg)), [Q.segment_name(k) for k in (0, 1, 4, 5)])

    def test_changed_job_invalidates(self):
        m = Q.Manifest(self.path, self.key)
        self.touch(Q.segment_name(0))
        m.add(0, 120)
        m2 = Q.Manifest.load(self.path, dict(self.key, encoder="libx264"), self.seg)
        self.assertEqual(m2.next_k, 0)
        self.assertEqual(Q.clean_work_dir(self.seg, m2), [Q.segment_name(0)])

    def test_missing_file_leaves_a_gap(self):
        m = Q.Manifest(self.path, self.key)
        for k in range(3):
            self.touch(Q.segment_name(k))
            m.add(k, 120)
        os.remove(os.path.join(self.seg, Q.segment_name(1)))
        m2 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m2.next_k, m2.finished()), (1, {0, 2}))
        self.assertFalse(m2.complete)

    def test_corrupt_manifest(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(Q.Manifest.load(self.path, self.key, self.seg).next_k, 0)

    def test_complete_flags(self):
        m = Q.Manifest(self.path, self.key)
        self.touch(Q.segment_name(0))
        m.add(0, 120)
        self.assertFalse(m.is_complete(120))
        self.touch(Q.segment_name(1))
        m.add(1, 40)                            # short last segment: complete even if 'complete' was never written
        self.assertTrue(m.is_complete(120))
        m3 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertTrue(m3.is_complete(120))
        m.mark_complete(160)
        m4 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertTrue(m4.complete)
        self.assertEqual((m4.total_frames, m4.last_k), (160, 1))
        m5 = Q.Manifest(self.path + "5", self.key)          # last_k known but segments before it missing
        self.touch(Q.segment_name(2))
        m5.add(2, 50)
        m5.mark_complete(290, 2)
        self.assertFalse(m5.complete)
        self.assertFalse(m5.is_complete(120))
        self.touch(Q.segment_name(0))
        self.touch(Q.segment_name(1))
        m5.add(0, 120)
        m5.add(1, 120)
        self.assertTrue(m5.complete)


class PartialBookkeeping(Bookkeeping):
    """The "partial" flag of manifest entries, the alternate file of a redone segment, completeness."""

    setUp = Bookkeeping.setUp
    touch = Bookkeeping.touch

    def listed(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_partial_flag_set_kept_and_omitted(self):
        m = Q.Manifest(self.path, self.key)
        for k, partial in ((0, False), (1, True), (2, False)):
            self.touch(Q.segment_name(k))
            m.add(k, 120, partial=partial)
        raw = self.listed()
        self.assertEqual([s.get("partial") for s in raw["segments"]], [None, True, None])
        self.assertNotIn("partial", raw["segments"][0])               # omitted, not false
        self.assertNotIn("partial", raw["segments"][2])
        self.assertEqual(raw["version"], 2)
        m2 = Q.Manifest.load(self.path, self.key, self.seg)           # kept on load
        self.assertEqual([s.get("partial") for s in m2.segments], [None, True, None])
        self.assertEqual(m2.partial_ks(), {1})
        # replacing the entry without the flag (the redo) drops it again
        self.touch(Q.segment_name(1, "_f"))
        self.assertEqual(m2.add(1, 120, file=Q.segment_name(1, "_f")), Q.segment_name(1))
        self.assertNotIn("partial", next(s for s in m2.segments if s["k"] == 1))
        self.assertEqual(m2.partial_ks(), set())
        # loose values in an old / foreign manifest are not flags
        raw = self.listed()
        raw["segments"][0]["partial"] = False
        raw["segments"][2]["partial"] = "yes"
        with open(self.path, "w") as f:
            json.dump(raw, f)
        self.assertEqual(Q.Manifest.load(self.path, self.key, self.seg).partial_ks(), set())

    def test_old_manifest_has_no_partial(self):
        for k in range(2):
            self.touch(Q.segment_name(k))
        v1 = {"version": 1, "key": self.key, "n": 120, "complete": True, "total_frames": 240,
              "segments": [{"k": k, "file": Q.segment_name(k), "frames": 120} for k in range(2)]}
        with open(self.path, "w") as f:
            json.dump(v1, f)
        m = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertEqual((m.partial_ks(), m.complete), (set(), True))

    def test_complete_blocked_while_partial(self):
        m = Q.Manifest(self.path, self.key)
        self.touch(Q.segment_name(0))
        self.touch(Q.segment_name(1))
        m.add(0, 120)
        m.add(1, 40, partial=True)
        m.mark_complete(160, 1)                       # the end is recorded (a redo needs it) ...
        self.assertEqual((m.last_k, m.total_frames), (1, 160))
        self.assertFalse(m.complete)                  # ... but the job is not complete while a segment is partial
        self.assertFalse(m.is_complete(120))
        d = self.listed()
        self.assertEqual((d["last_k"], d["complete"], d["total_frames"]), (1, False, 160))
        self.assertFalse(Q.Manifest.load(self.path, self.key, self.seg).complete)
        self.touch(Q.segment_name(1, "_f"))
        m.add(1, 40, file=Q.segment_name(1, "_f"))
        self.assertTrue(m.complete)
        self.assertTrue(m.is_complete(120))
        self.assertTrue(self.listed()["complete"])
        m3 = Q.Manifest.load(self.path, self.key, self.seg)
        self.assertTrue(m3.complete)
        self.assertEqual(m3.segments[1]["file"], Q.segment_name(1, "_f"))

    def test_progress_counts_partial_as_unfinished(self):
        m = Q.Manifest(self.path, self.key)
        for k, fr, partial in ((0, 120, False), (1, 120, True), (2, 60, False)):
            self.touch(Q.segment_name(k))
            m.add(k, fr, partial=partial)
        self.assertEqual((m.frames_done(), m.frames_done(partial=False)), (300, 180))
        self.assertEqual((len(m.segments), m.finished_count()), (3, 2))
        # run progress is base + pushed; the card only reaches 100 % when the base is everything
        line = Q.progress_line(m.frames_done(partial=False), 300, 20.0, 1.0, m.finished_count(), "restore", 1)
        self.assertLess(line["frame"], line["frames"])
        self.assertEqual(line["segments"], 2)

    def test_clean_work_dir_removes_unlisted_alternate(self):
        m = Q.Manifest(self.path, self.key)
        self.touch(Q.segment_name(0))
        m.add(0, 120, partial=True)
        self.touch(Q.segment_name(0, "_f"))           # a redo killed before its segment closed: not listed
        self.assertEqual(Q.clean_work_dir(self.seg, m), [Q.segment_name(0, "_f")])
        self.assertEqual(os.listdir(self.seg), [Q.segment_name(0)])
        self.touch(Q.segment_name(0, "_f"))           # redo done: the alternate is listed, the old file is the leftover
        m.add(0, 120, file=Q.segment_name(0, "_f"))
        self.assertEqual(Q.clean_work_dir(self.seg, m), [Q.segment_name(0)])
        self.assertEqual(os.listdir(self.seg), [Q.segment_name(0, "_f")])

    def test_finalize_refuses_partial(self):
        m = Q.Manifest(self.path, self.key)
        self.touch(Q.segment_name(0))
        m.add(0, 40, partial=True)
        m.mark_complete(40, 0)
        info = Q.SourceInfo(Fraction(30), 40, 128, 96, 0.0, 1.3, False)
        plan = Q.SegmentPlan(30, 40, 0.0, 4)
        with self.assertRaises(Q.WorkerError):
            Q.finalize({"source": "x", "output": os.path.join(self.dir, "o.mkv")}, info, plan, m, self.dir, self.seg)


class RealFfmpeg(unittest.TestCase):
    """Segmenting -> kill -> resume -> finalise with libx264 and synthetic frames."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(dir=TMP_ROOT)
        cls.src = os.path.join(cls.dir, "src.mp4")
        make_source(cls.src, seconds=4, fps=30, audio=True)          # 120 frames, N = 30 with 1 s segments
        cls.info = Q.parse_probe(Q.ffprobe_json(cls.src))

    def job(self, work, out):
        return {"source": self.src, "work_dir": work, "output": out, "segment_seconds": 1.0,
                "settings": {"encoder": "libx264", "detection_model": "v4-fast", "restoration_model": "basicvsrpp-v1.2",
                             "fp16": None, "max_clip_length": 30, "interpolation": "off", "compare": "off"},
                "headers": {}}

    def run_frames(self, plan, manifest, work, seg, first, last, natural, interp=False, layers=("processed",)):
        """Feed source frames first..last-1 (k0 = first / N) like the restoring loop does."""
        stage = R.FullDoubler(FakeInterpolator(AvgRife()), 30.0, W, H, log=lambda m: None) if interp else Q.PlainStage()
        if len(layers) > 1:
            stage = Q.CompareStage(stage, layers)
        run = Q.SegmentRun(plan, first // plan.n, self.info, "libx264", seg, manifest, work, stage, cq_offset=4)
        for i in range(first, last):
            # the "original" differs from the restored frame in a 4x4 box, so the tracks can be told apart
            patch = [((0, 0, 3, 3), np.full((4, 4, 3), 200, np.uint8))]
            self.assertTrue(run.write(push(stage, Fr(frame(i), patch), frame(i))))
        if natural:
            self.assertTrue(run.write(stage.finish()))
            self.assertEqual(run.finish(), 0)
        else:
            import time
            time.sleep(0.5)
            run.kill()
        return run

    def setup_job(self, name):
        work = os.path.join(self.dir, name)
        seg = os.path.join(work, "segments")
        os.makedirs(seg)
        job = self.job(work, os.path.join(self.dir, name + ".mkv"))
        plan = Q.SegmentPlan(self.info.fps, self.info.frames, self.info.video_start, 1.0)
        key = Q.job_key(job, self.info, plan, "libx264")
        return job, plan, work, seg, key

    def frame_count(self, path, sel="v:0"):
        p = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", sel, "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True)
        return int(p.stdout.strip())

    def test_plan_matches_source(self):
        self.assertEqual((self.info.frames, float(self.info.fps), self.info.has_audio), (120, 30.0, True))
        self.assertEqual(Q.SegmentPlan(self.info.fps, self.info.frames, 0, 1.0).n, 30)

    def test_cached_hls_finalize_preserves_nonzero_source_clock_for_audio_and_original(self):
        """A real muxed TS VOD with a nonzero MPEGTS start stays aligned through the cache-only finalizer."""
        fixture = os.path.join(self.dir, "cached-hls")
        os.makedirs(fixture)
        playlist = os.path.join(fixture, "source.m3u8")
        ts_pattern = os.path.join(fixture, "part_%03d.ts")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size={W}x{H}:rate=30:duration=4", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-f", "hls", "-hls_time", "4",
                        "-hls_playlist_type", "vod", "-hls_segment_filename", ts_pattern, playlist], check=True)
        source_probe = Q.ffprobe_json(playlist)
        with open(playlist, encoding="utf-8") as f:
            extinf = sum(float(line.split(":", 1)[1].split(",", 1)[0])
                         for line in f if line.startswith("#EXTINF:"))
        source_probe["duration"] = extinf
        source_probe["format"]["duration"] = str(extinf)
        source_probe = Q.full_vod_probe(source_probe)
        info = Q.parse_probe(source_probe)
        self.assertGreater(info.video_start, 1.0)
        self.assertAlmostEqual(info.duration, 4.0, delta=0.1)

        work, seg_dir = os.path.join(fixture, "work"), os.path.join(fixture, "work", "segments")
        os.makedirs(seg_dir)
        status_path = os.path.join(fixture, "source-status.json")
        segment_count = len([n for n in os.listdir(fixture) if n.endswith(".ts")])
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "identity": "fixture-hls", "generation": "run-1", "status": "complete",
                       "error": None, "ready_segments": segment_count, "segment_count": segment_count}, f)
        job = {"source": playlist, "source_url": playlist, "source_identity": "fixture-hls",
               "source_generation": "run-1", "source_status_path": status_path,
               "output": os.path.join(fixture, "final.mkv")}
        plan = Q.SegmentPlan(info.fps, info.frames, info.video_start, info.duration)
        manifest = Q.Manifest(os.path.join(work, "segments.json"), {"n": plan.n})
        run = Q.SegmentRun(plan, 0, info, "libx264", seg_dir, manifest, work, Q.PlainStage())
        for i in range(plan.frames_total):
            self.assertTrue(run.write([frame(i)]))
        self.assertEqual(run.finish(), 0)
        manifest.mark_complete(plan.frames_total, plan.count - 1)
        Q.finalize(job, info, plan, manifest, work, seg_dir, original="copy")

        output = Q.ffprobe_json(job["output"])
        streams = output["streams"]
        videos = [s for s in streams if s["codec_type"] == "video"]
        audio = next(s for s in streams if s["codec_type"] == "audio")
        self.assertEqual(len(videos), 2)
        starts = [float(s["start_time"]) for s in videos] + [float(audio["start_time"])]
        self.assertLess(max(starts) - min(starts), 0.1, starts)
        self.assertTrue(Q.finalized_duration_ok(output, info.duration, job), output["format"])
        self.assertAlmostEqual(float(output["format"]["duration"]), info.video_start + info.duration, delta=0.5)

    def test_kill_resume_finalise(self):
        job, plan, work, seg, key = self.setup_job("resume")
        m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        # first run: 75 frames then killed -> segments 0 and 1 closed (60 frames), 2 is partial
        self.run_frames(plan, m, work, seg, 0, 75, natural=False)
        self.assertEqual([s["frames"] for s in m.segments], [30, 30])
        self.assertTrue(any(n not in {s["file"] for s in m.segments} for n in os.listdir(seg)))   # partial leftover
        # restart: manifest from disk, leftovers removed, resume at frame 60
        m2 = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.assertEqual(m2.next_k, 2)
        Q.clean_work_dir(seg, m2)
        self.assertEqual(len(os.listdir(seg)), 2)
        self.run_frames(plan, m2, work, seg, 60, 120, natural=True)
        m2.mark_complete(120)
        self.assertEqual([s["frames"] for s in m2.segments], [30, 30, 30, 30])
        for s in m2.segments:
            self.assertEqual(self.frame_count(os.path.join(seg, s["file"])), 30)
        Q.finalize(job, self.info, plan, m2, work, seg)
        self.assertTrue(os.path.isdir(seg))                # the queue service deletes work_dir, not the worker
        out = probe(job["output"])
        self.assertTrue(Q.duration_ok(float(out["format"]["duration"]), self.info.duration))
        self.assertEqual(sorted(s["codec_type"] for s in out["streams"]), ["audio", "video"])
        self.assertEqual(self.frame_count(job["output"]), 120)

    def test_interpolated_kill_resume_finalise(self):
        """2x: 120 source frames, 30 per segment -> 60 pipe frames per segment, 240 output frames, same duration."""
        job, plan, work, seg, _ = self.setup_job("interp")
        job["settings"]["interpolation"] = "full"
        key = Q.job_key(job, self.info, plan, "libx264")
        self.assertEqual(key["interpolation"], "full")
        m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.run_frames(plan, m, work, seg, 0, 75, natural=False, interp=True)
        self.assertEqual([s["frames"] for s in m.segments], [30, 30])          # manifest counts source frames
        m2 = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.assertEqual(m2.next_k, 2)
        Q.clean_work_dir(seg, m2)
        self.run_frames(plan, m2, work, seg, 60, 120, natural=True, interp=True)
        m2.mark_complete(120)
        self.assertEqual([s["frames"] for s in m2.segments], [30, 30, 30, 30])
        for s in m2.segments:
            self.assertEqual(self.frame_count(os.path.join(seg, s["file"])), 60)
        Q.finalize(job, self.info, plan, m2, work, seg)
        out = probe(job["output"])
        vs = next(s for s in out["streams"] if s["codec_type"] == "video")
        self.assertEqual(vs["r_frame_rate"], "60/1")
        self.assertTrue(Q.duration_ok(float(out["format"]["duration"]), self.info.duration))
        self.assertEqual(self.frame_count(job["output"]), 240)
        # a changed mode never reuses segments
        job["settings"]["interpolation"] = "off"
        self.assertEqual(Q.Manifest.load(os.path.join(self.dir, "none.json"), Q.job_key(job, self.info, plan, "libx264"),
                                         seg).next_k, 0)

    def pkt(self, path, sel, what="pts_time"):
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", f"packet={what}",
                            "-of", "csv=p=0", path], capture_output=True, text=True)
        return [x for x in p.stdout.split()]

    def keyframes(self, path, sel):
        """pts (4 decimals) of the key packets of a stream."""
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "packet=pts_time,flags",
                            "-of", "csv=p=0", path], capture_output=True, text=True)
        return [round(float(l.split(",")[0]), 4) for l in p.stdout.split() if "K" in l.split(",")[1]]

    def compare_encode_roundtrip(self, name, interp):
        """compare on, original=encode: processed at the pipe rate, original at the SOURCE rate, same time span."""
        mult = 2 if interp else 1
        layers = ("processed", "original")
        job, plan, work, seg, _ = self.setup_job(name)
        job["settings"]["interpolation"] = "full" if interp else "off"
        job["settings"]["compare"] = "on"
        job["settings"]["compare_original"] = "encode"
        key = {"n": plan.n, "compare": "on"}
        m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        # killed after 100 of 120 frames: the encoder's lookahead on the half-rate track may hold the last closed segment
        # back, so only "at least segments 0 and 1" is certain; the run resumes at the first missing segment
        self.run_frames(plan, m, work, seg, 0, 100, natural=False, interp=interp, layers=layers)
        self.assertIn([s["frames"] for s in m.segments], ([30, 30], [30, 30, 30]))
        m2 = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.assertEqual(m2.next_k, len(m.segments))
        Q.clean_work_dir(seg, m2)
        self.run_frames(plan, m2, work, seg, m2.next_k * 30, 120, natural=True, interp=interp, layers=layers)
        m2.mark_complete(120)
        self.assertEqual([s["frames"] for s in m2.segments], [30, 30, 30, 30])
        m2.set_play(Q.play_block(plan, "full" if interp else "off", layers, "encode"))
        for s in m2.segments:
            p = os.path.join(seg, s["file"])
            self.assertEqual(self.frame_count(p, "v:0"), 30 * mult)         # processed: the pipe rate
            self.assertEqual(self.frame_count(p, "v:1"), 30)                # original: source frames only
            for v in (0, 1):                                                # every track starts with a keyframe
                self.assertEqual(self.keyframes(p, f"v:{v}"), [0.0])        # ... and has no other (GOP = segment)
        Q.finalize(job, self.info, plan, m2, work, seg, layers=layers, original="encode")
        out = probe(job["output"])
        vs = [s for s in out["streams"] if s["codec_type"] == "video"]
        self.assertEqual([s["tags"]["title"] for s in vs], ["oh:processed", "oh:original"])
        self.assertEqual([s["disposition"]["default"] for s in vs], [1, 0])
        self.assertEqual([s["r_frame_rate"] for s in vs], ["30/1" if not interp else "60/1", "30/1"])
        self.assertTrue(Q.duration_ok(float(out["format"]["duration"]), self.info.duration))
        self.assertEqual(self.frame_count(job["output"], "v:0"), 120 * mult)
        self.assertEqual(self.frame_count(job["output"], "v:1"), 120)
        self.assertEqual(sum(1 for s in out["streams"] if s["codec_type"] == "audio"), 1)
        tags = {k.upper(): v for k, v in out["format"]["tags"].items()}
        self.assertEqual((tags["OH_SOURCE_FPS"], tags["OH_RATE_MULT"], tags["OH_COMPARE"], tags["OH_ORIGINAL"]),
                         ("30", str(mult), "1", "encode"))
        # same time span, same grid: the original's timestamps are the processed track's even slots; keyframes at the
        # segment starts (every 1 s) on both
        pts0, pts1 = [round(float(x), 4) for x in self.pkt(job["output"], "v:0")], \
                     [round(float(x), 4) for x in self.pkt(job["output"], "v:1")]
        self.assertEqual(sorted(pts1), sorted(pts0)[::mult])
        starts = [float(k) for k in range(4)]
        self.assertEqual(self.keyframes(job["output"], "v:0"), starts)
        self.assertEqual(self.keyframes(job["output"], "v:1"), starts)
        return job["output"]

    def test_compare_encode_kill_resume_finalise(self):
        self.compare_encode_roundtrip("cmp1", False)

    def test_compare_encode_interpolated_kill_resume_finalise(self):
        self.compare_encode_roundtrip("cmp2", True)

    def test_compare_copy_finalise(self):
        """original=copy: single-track segments; the source's own video stream is added at finalize on its own timeline."""
        for interp in (False, True):
            mult = 2 if interp else 1
            job, plan, work, seg, _ = self.setup_job("copy%d" % mult)
            job["settings"]["interpolation"] = "full" if interp else "off"
            job["settings"]["compare"] = "on"
            key = {"n": plan.n, "compare": "on"}
            m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
            self.run_frames(plan, m, work, seg, 0, 120, natural=True, interp=interp)
            m.mark_complete(120)
            m.set_play(Q.play_block(plan, "full" if interp else "off", ("processed",), "copy"))
            self.assertEqual(self.frame_count(os.path.join(seg, m.segments[0]["file"]), "v:0"), 30 * mult)
            Q.finalize(job, self.info, plan, m, work, seg, layers=("processed",), original="copy")
            out = probe(job["output"])
            vs = [s for s in out["streams"] if s["codec_type"] == "video"]
            self.assertEqual([s["tags"]["title"] for s in vs], ["oh:processed", "oh:original"])
            self.assertEqual([s["disposition"]["default"] for s in vs], [1, 0])
            self.assertEqual(vs[1]["codec_name"], "h264")
            self.assertEqual(self.frame_count(job["output"], "v:0"), 120 * mult)
            self.assertEqual(self.frame_count(job["output"], "v:1"), 120)
            # the copy keeps the source timeline = where the shifted processed track sits
            self.assertAlmostEqual(float(vs[0]["start_time"]), float(vs[1]["start_time"]), delta=0.002)
            self.assertAlmostEqual(float(vs[1]["start_time"]), self.info.video_start, delta=0.002)
            self.assertEqual(sum(1 for s in out["streams"] if s["codec_type"] == "audio"), 1)
            tags = {k.upper(): v for k, v in out["format"]["tags"].items()}
            self.assertEqual((tags["OH_RATE_MULT"], tags["OH_COMPARE"], tags["OH_ORIGINAL"], tags["OH_SOURCE_FPS"]),
                             (str(mult), "1", "copy", "30"))
            self.assertAlmostEqual(float(tags["OH_VIDEO_START"]), self.info.video_start, delta=1e-6)

    def test_compare_copy_no_audio_source(self):
        src = os.path.join(self.dir, "noaudio.mp4")
        make_source(src, seconds=4, fps=30, audio=False)
        info = Q.parse_probe(Q.ffprobe_json(src))
        self.assertIsNone(info.copy_why)
        work = os.path.join(self.dir, "noaudio_w")
        seg = os.path.join(work, "segments")
        os.makedirs(seg)
        job = self.job(work, os.path.join(self.dir, "noaudio.mkv"))
        job["source"] = src
        plan = Q.SegmentPlan(info.fps, info.frames, info.video_start, 1.0)
        m = Q.Manifest.load(os.path.join(work, "segments.json"), {"n": plan.n}, seg)
        stage = Q.PlainStage()
        run = Q.SegmentRun(plan, 0, info, "libx264", seg, m, work, stage)
        for i in range(120):
            self.assertTrue(run.write(push(stage, Fr(frame(i), None), frame(i))))
        self.assertTrue(run.write(stage.finish()))
        self.assertEqual(run.finish(), 0)
        m.mark_complete(120)
        Q.finalize(job, info, plan, m, work, seg, original="copy")
        out = probe(job["output"])
        self.assertEqual(sorted(s["codec_type"] for s in out["streams"]), ["video", "video"])

    def test_compare_off_tags(self):
        job, plan, work, seg, key = self.setup_job("offtags")
        m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.run_frames(plan, m, work, seg, 0, 120, natural=True)
        m.mark_complete(120)
        Q.finalize(job, self.info, plan, m, work, seg)
        out = probe(job["output"])
        tags = {k.upper(): v for k, v in out["format"]["tags"].items()}
        self.assertEqual({k: v for k, v in tags.items() if k.startswith("OH_")},
                         {"OH_SOURCE_FPS": "30", "OH_RATE_MULT": "1", "OH_VIDEO_START": "%.6f" % self.info.video_start,
                          "OH_COMPARE": "0", "OH_ORIGINAL": "none"})
        self.assertEqual(sum(1 for s in out["streams"] if s["codec_type"] == "video"), 1)

    def run_gap(self, plan, manifest, work, seg, first, last, k_end, interp=False, suffix="", copies=None,
                layers=("processed",)):
        """A run over a gap the way restore_segments does it: feed source frames first..(k_end*N or the end), then finish.
        suffix: a redo run's file name variant; copies {k: n}: policy copies noted for those segments up front."""
        stage = R.FullDoubler(FakeInterpolator(AvgRife()), 30.0, W, H, log=lambda m: None) if interp else Q.PlainStage()
        if len(layers) > 1:
            stage = Q.CompareStage(stage, layers)
        k0 = first // plan.n
        run = Q.SegmentRun(plan, k0, self.info, "libx264", seg, manifest, work, stage, suffix, cq_offset=4)
        for ck, cn in (copies or {}).items():
            run.add_copies(ck, cn)
        limit = Q.run_frame_limit(k0, k_end, plan.n)
        pushed = 0
        for i in range(first, last):
            self.assertTrue(run.write(push(stage, Fr(frame(i), None), frame(i))))
            pushed += 1
            if limit is not None and pushed >= limit:
                break
        self.assertTrue(run.write(stage.finish()))
        self.assertEqual(run.finish(), 0)
        return pushed

    def test_compare_encode_out_of_order_runs(self):
        """original=encode, 2x: a run to the end from segment 2, then a run over the gap 0..1 that stops before segment 2:
        every segment has 2N processed and N original frames and a keyframe on both tracks at its start."""
        layers = ("processed", "original")
        job, plan, work, seg, _ = self.setup_job("cmp_ooo")
        job["settings"].update(interpolation="full", compare="on", compare_original="encode")
        m = Q.Manifest.load(os.path.join(work, "segments.json"), {"n": plan.n, "compare": "on"}, seg)
        self.assertEqual(self.run_gap(plan, m, work, seg, 60, 120, None, True, layers=layers), 60)
        m.mark_complete(120, 3)
        self.assertEqual(self.run_gap(plan, m, work, seg, 0, 120, Q.run_end_k(m.finished(), 0), True, layers=layers), 60)
        self.assertTrue(m.complete)
        m.set_play(Q.play_block(plan, "full", layers, "encode"))
        for s in m.segments:
            p = os.path.join(seg, s["file"])
            self.assertEqual((self.frame_count(p, "v:0"), self.frame_count(p, "v:1")), (60, 30))
            self.assertEqual((self.keyframes(p, "v:0"), self.keyframes(p, "v:1")), ([0.0], [0.0]))
        Q.finalize(job, self.info, plan, m, work, seg, layers=layers, original="encode")
        self.assertEqual((self.frame_count(job["output"], "v:0"), self.frame_count(job["output"], "v:1")), (240, 120))

    def out_of_order(self, name, interp):
        mult = 2 if interp else 1
        job, plan, work, seg, _ = self.setup_job(name)
        job["settings"]["interpolation"] = "full" if interp else "off"
        key = Q.job_key(job, self.info, plan, "libx264")
        path = os.path.join(work, "segments.json")
        m = Q.Manifest.load(path, key, seg)
        m.set_play(Q.play_block(plan, "full" if interp else "off", ("processed",)))
        # run 1: the user jumped to segment 2 -> a run to the end (natural end: last_k known, 0 and 1 still missing)
        self.assertEqual(self.run_gap(plan, m, work, seg, 60, 120, None, interp), 60)
        m.mark_complete(120, 3)
        self.assertEqual([s["k"] for s in m.segments], [2, 3])
        self.assertFalse(m.complete)
        self.assertEqual(Q.first_gap(m.finished(), 0, m.gap_limit(plan.count)), 0)
        self.assertEqual(Q.run_end_k(m.finished(), 0), 2)
        # reload (restart): the gap is still there
        m = Q.Manifest.load(path, key, seg)
        self.assertEqual((m.next_k, m.last_k, m.complete), (0, 3, False))
        # run 2 over the gap 0..1: stops before the finished segment 2, every segment full with N source frames
        pushed = self.run_gap(plan, m, work, seg, 0, 120, Q.run_end_k(m.finished(), 0), interp)
        self.assertEqual(pushed, 60)
        self.assertEqual([(s["k"], s["frames"]) for s in m.segments], [(0, 30), (1, 30), (2, 30), (3, 30)])
        self.assertTrue(m.complete)
        self.assertFalse(os.path.exists(os.path.join(seg, "seg_000004.mkv")))
        for s in m.segments:
            self.assertEqual(self.frame_count(os.path.join(seg, s["file"])), 30 * mult)
        self.assertEqual(Q.clean_work_dir(seg, m), [])
        # finalize concatenates 0..last_k in order although 2, 3 were produced first
        Q.finalize(job, self.info, plan, m, work, seg)
        self.assertTrue(os.path.isdir(work))
        out = probe(job["output"])
        self.assertTrue(Q.duration_ok(float(out["format"]["duration"]), self.info.duration))
        self.assertEqual(self.frame_count(job["output"]), 120 * mult)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual((raw["version"], raw["last_k"], raw["complete"], raw["total_frames"], raw["play"]["n"]),
                         (2, 3, True, 120, 30))

    def test_out_of_order_gap_runs_finalise(self):
        self.out_of_order("ooo1", False)

    def test_out_of_order_gap_runs_interpolated(self):
        self.out_of_order("ooo2", True)

    def redo_scenario(self, name, interp):
        mult = 2 if interp else 1
        job, plan, work, seg, _ = self.setup_job(name)
        job["settings"]["interpolation"] = "full" if interp else "off"
        key = Q.job_key(job, self.info, plan, "libx264")
        path = os.path.join(work, "segments.json")
        m = Q.Manifest.load(path, key, seg)
        # normal run over everything; segments 1 and 2 got policy copies
        self.run_gap(plan, m, work, seg, 0, 120, None, interp, copies={1: 5, 2: 1})
        m.mark_complete(120, 3)
        self.assertEqual([bool(s.get("partial")) for s in m.segments], [False, True, True, False])
        self.assertFalse(m.complete)
        self.assertIsNone(Q.first_gap(m.finished(), 0, m.gap_limit(plan.count)))        # no real gaps ...
        plan_r = Q.redo_plan(list(m.segments), 0, m.end_k(plan.n))                      # ... so the redo is next
        self.assertEqual(plan_r, (1, 3, "_f"))
        old_files = {s["k"]: s["file"] for s in m.segments}
        # a restart in between keeps the flags
        m = Q.Manifest.load(path, key, seg)
        self.assertEqual(m.partial_ks(), {1, 2})
        self.assertEqual(Q.clean_work_dir(seg, m), [])
        # the redo run: segments 1..2 under the alternate names, always full
        before = {s["k"]: s["file"] for s in m.segments}
        self.run_gap(plan, m, work, seg, 30, 120, 3, interp, suffix="_f")
        by_k = {s["k"]: s for s in m.segments}
        self.assertEqual(sorted(by_k), [0, 1, 2, 3])
        for k in (1, 2):
            self.assertEqual(by_k[k]["file"], Q.segment_name(k, "_f"))
            self.assertNotIn("partial", by_k[k])
            self.assertEqual(by_k[k]["frames"], 30)
            self.assertFalse(os.path.exists(os.path.join(seg, old_files[k])))            # the old file is gone
            self.assertEqual(self.frame_count(os.path.join(seg, by_k[k]["file"])), 30 * mult)
        self.assertEqual((by_k[0]["file"], by_k[3]["file"]), (before[0], before[3]))     # others untouched
        self.assertTrue(m.complete)
        self.assertIsNone(Q.redo_plan(list(m.segments), 0, m.end_k(plan.n)))
        self.assertEqual(Q.clean_work_dir(seg, m), [])
        self.assertEqual(Q.Manifest.load(path, key, seg).partial_ks(), set())
        Q.finalize(job, self.info, plan, m, work, seg)                                   # concat uses the new names
        self.assertEqual(self.frame_count(job["output"]), 120 * mult)

    def test_redo_replaces_partial_segments(self):
        self.redo_scenario("redo1", False)

    def test_redo_replaces_partial_segments_interpolated(self):
        self.redo_scenario("redo2", True)

    def test_redo_old_file_removal_failure_tolerated(self):
        job, plan, work, seg, key = self.setup_job("redo3")
        path = os.path.join(work, "segments.json")
        m = Q.Manifest.load(path, key, seg)
        self.run_gap(plan, m, work, seg, 0, 120, None, False, copies={0: 2})
        m.mark_complete(120, 3)
        self.assertEqual(m.partial_ks(), {0})
        seen = []
        real_remove = os.remove

        def locked(p, *a, **kw):
            if os.path.basename(p) == Q.segment_name(0):
                seen.append(p)
                raise PermissionError("in use by the player")
            return real_remove(p, *a, **kw)
        with unittest.mock.patch.object(os, "remove", locked):
            self.run_gap(plan, m, work, seg, 0, 120, 1, False, suffix="_f")
        self.assertEqual(len(seen), 1)
        entry = next(s for s in m.segments if s["k"] == 0)
        self.assertEqual((entry["file"], entry.get("partial")), (Q.segment_name(0, "_f"), None))
        self.assertTrue(os.path.exists(os.path.join(seg, Q.segment_name(0))))           # still there
        self.assertTrue(m.complete)
        # the next resume's cleanup removes the leftover
        self.assertEqual(Q.clean_work_dir(seg, Q.Manifest.load(path, key, seg)), [Q.segment_name(0)])
        # and a second redo goes back to the regular name
        self.assertEqual(Q.alt_suffix(entry["file"]), "")

    def test_aborted_run_keeps_closed_segments_only(self):
        job, plan, work, seg, key = self.setup_job("abort")
        m = Q.Manifest.load(os.path.join(work, "segments.json"), key, seg)
        self.run_frames(plan, m, work, seg, 30, 105, natural=False)        # run at segment 1, killed in segment 3
        self.assertEqual([s["k"] for s in m.segments], [1, 2])
        self.assertEqual(len(os.listdir(seg)), 3)                           # + the partial segment 3
        self.assertEqual(Q.clean_work_dir(seg, m), [Q.segment_name(3)])
        self.assertEqual(Q.goto_action(0, m.finished(), 1, 3, None, 4), "abort")      # gap 0 is behind the run
        self.assertEqual(Q.goto_action(2, m.finished(), 1, 3, None, 4), "next")

    def test_short_last_segment_and_no_audio(self):
        src = os.path.join(self.dir, "noaudio.mp4")
        make_source(src, seconds=3, fps=30, audio=False)           # 90 frames, N = 40 (round(1.3 * 30)): 40, 40, 10
        info = Q.parse_probe(Q.ffprobe_json(src))
        self.assertFalse(info.has_audio)
        work = os.path.join(self.dir, "short")
        seg = os.path.join(work, "segments")
        os.makedirs(seg)
        job = {"source": src, "work_dir": work, "output": os.path.join(self.dir, "short.mkv"),
               "segment_seconds": 1.3, "settings": {"encoder": "libx264"}, "headers": {}}
        plan = Q.SegmentPlan(info.fps, info.frames, info.video_start, 1.3)
        self.assertEqual(plan.n, 39)
        m = Q.Manifest(os.path.join(work, "segments.json"), {"n": plan.n})
        stage = Q.PlainStage()
        run = Q.SegmentRun(plan, 0, info, "libx264", seg, m, work, stage)
        for i in range(90):
            self.assertTrue(run.write(stage.push(None, frame(i), None)))
        self.assertEqual(run.finish(), 0)
        self.assertEqual([s["frames"] for s in m.segments], [39, 39, 12])
        self.assertTrue(m.is_complete(plan.n))
        Q.finalize(job, info, plan, m, work, seg)
        out = probe(job["output"])
        self.assertEqual([s["codec_type"] for s in out["streams"]], ["video"])
        self.assertEqual(self.frame_count(job["output"]), 90)


if __name__ == "__main__":
    try:
        unittest.main(exit=False, verbosity=2)
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
