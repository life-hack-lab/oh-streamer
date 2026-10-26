"""Tests for progressive_stream.py (watching a queue item while it is processed; run: python -P
streamer/tests/test_progressive_stream.py with Lada's venv: PyAV + ffmpeg / ffprobe on PATH).

Pure parts: manifest v1 / v2 parsing, ranges merging, segment lookup, progress text, the queue message, want / watching
timing with a fake clock. Real streams: small libx264 segments made with ffmpeg the way queue_worker writes them (3 titled
video tracks at 2x, keyframes every 15 frames, forced IDR at every segment start, timestamps reset to 0) + a source with
AAC audio; the Feeder's output is read from its TCP socket like mpv would and checked with ffprobe (tracks, absolute
timestamps, continuity across segments, waiting for a segment that appears later, start at the keyframe before a seek
target, every file released on close). Last: run_player (test_queue_player's fakes) in progressive mode."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import collections
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from unittest import mock
import oh_streamer.player.info_panel as info_panel  # noqa: E402
import oh_streamer.player.progressive_stream as P  # noqa: E402

FPS = Fraction(30000, 1001)
N = 30                       # source frames per test segment (1.001 s)
LAST = 18                    # source frames of the last (4th) test segment
TITLES = ["oh:processed", "oh:original", "oh:restored"]


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def ffprobe_json(path, *args):
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", *args, path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise AssertionError(p.stderr)
    return json.loads(p.stdout)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------------------------------------------
# pure parts
# ---------------------------------------------------------------------------------------------------------------

class PlayInfoTest(unittest.TestCase):
    def test_segment_at_round_trips_segment_starts(self):
        p = P.PlayInfo(FPS, 120, 0.021333, 2, count=1805)
        self.assertEqual(p.segment_at(0.0), 0)
        self.assertEqual(p.segment_at(-5.0), 0)
        for k in range(0, 1805, 7):
            self.assertEqual(p.segment_at(p.seg_start(k)), k, k)
            if k:
                self.assertEqual(p.segment_at(p.seg_start(k) - 0.001), k - 1, k)
        self.assertEqual(p.segment_at(1e6), 1804)              # clamped to the count
        self.assertAlmostEqual(p.seg_start(1), 0.021333 + 4.004, places=6)
        self.assertAlmostEqual(p.seg_end(1, 120), p.seg_start(2), places=9)
        self.assertEqual(P.PlayInfo(FPS, 120).segment_at(1e6), int(1e6 * 30000 / 1001 / 120))   # no count: no clamp


class ManifestTest(unittest.TestCase):
    fallback = P.PlayInfo(FPS, 120, 0.021333, 1, ("processed",), 1805)

    def test_v2_non_contiguous_with_play_block(self):
        doc = {"version": 2, "segments": [{"k": 0, "file": "seg_000000.mkv", "frames": 96},
                                          {"k": 1, "file": "seg_000001.mkv", "frames": 96},
                                          {"k": 5, "file": "seg_000005.mkv", "frames": 96},
                                          {"k": 6, "file": "seg_000006.mkv", "frames": 77}],
               "last_k": 6, "complete": False, "total_frames": 797,
               "play": {"fps": "24000/1001", "n": 96, "video_start": 0.5, "rate_mult": 2,
                        "layers": ["processed", "original", "restored"], "count": 9}}
        st = P.parse_manifest(doc, self.fallback)
        self.assertEqual(st.version, 2)
        self.assertEqual(sorted(st.segments), [0, 1, 5, 6])
        self.assertEqual((st.play.fps, st.play.n, st.play.video_start, st.play.rate_mult),
                         (Fraction(24000, 1001), 96, 0.5, 2))
        self.assertEqual(st.play.layers, ("processed", "original", "restored"))
        self.assertEqual(st.play.count, 7)                      # last_k known: it wins over the estimate
        self.assertEqual(st.final_k(), 6)
        self.assertFalse(st.complete)
        d = 96 * 1001 / 24000
        r = st.ranges()
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(r[0][0], 0.5)
        self.assertAlmostEqual(r[0][1], 0.5 + 2 * d)
        self.assertAlmostEqual(r[1][0], 0.5 + 5 * d)
        self.assertAlmostEqual(r[1][1], 0.5 + (6 * 96 + 77) * 1001 / 24000)
        self.assertEqual(P.ranges_text(r), f"0.500-{0.5 + 2 * d:.3f},{0.5 + 5 * d:.3f}-{r[1][1]:.3f}")

    def test_v2_without_last_k_keeps_the_estimate(self):
        doc = {"version": 2, "segments": [{"k": 3, "file": "seg_000003.mkv", "frames": 120}], "last_k": None,
               "complete": False, "play": {"fps": "30000/1001", "n": 120, "video_start": 0.0, "rate_mult": 1,
                                           "layers": ["processed"], "count": 40}}
        st = P.parse_manifest(doc, self.fallback)
        self.assertIsNone(st.final_k())
        self.assertEqual(st.play.count, 40)
        self.assertEqual(st.play.segment_at(1e5), 39)

    def test_v1_uses_key_and_source_probe(self):
        doc = {"version": 1, "key": {"fps": "30000/1001", "n": 120, "interpolation": "full", "compare": "on"},
               "n": 120, "segments": [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": 120} for k in range(3)],
               "complete": False, "total_frames": None}
        st = P.parse_manifest(doc, self.fallback)
        self.assertEqual(st.version, 1)
        self.assertEqual(st.play.video_start, 0.021333)        # from the source probe (v1 has no play block)
        self.assertEqual(st.play.rate_mult, 2)
        self.assertEqual(st.play.layers, ("processed", "original", "restored"))
        self.assertEqual(st.play.count, 1805)
        self.assertIsNone(st.final_k())
        self.assertEqual(P.ranges_text(st.ranges()), f"0.021-{0.021333 + 3 * 4.004:.3f}")

    def test_v1_complete_and_short_last_segment_mark_the_end(self):
        segs = [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": 120} for k in range(4)]
        st = P.parse_manifest({"version": 1, "n": 120, "segments": segs, "complete": True, "total_frames": 480},
                              self.fallback)
        self.assertEqual(st.final_k(), 3)
        self.assertEqual(st.play.count, 4)
        segs[-1]["frames"] = 50
        st = P.parse_manifest({"version": 1, "n": 120, "segments": segs, "complete": False}, self.fallback)
        self.assertEqual(st.final_k(), 3)                       # shorter than N: the video's last segment

    def test_bad_entries_are_skipped(self):
        doc = {"version": 2, "segments": [{"k": 0, "file": "../evil.mkv", "frames": 120},
                                          {"k": 1, "file": "sub\\x.mkv", "frames": 120},
                                          {"k": "x", "file": "a.mkv", "frames": 1}, {"k": 2, "frames": 120},
                                          {"k": 3, "file": "seg_000003.mkv", "frames": 0},
                                          {"k": 4, "file": "seg_000004.mkv", "frames": 120}]}
        st = P.parse_manifest(doc, self.fallback)
        self.assertEqual(sorted(st.segments), [4])
        with self.assertRaises(ValueError):
            P.parse_manifest([1, 2], self.fallback)

    def test_read_manifest_missing_or_broken_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "segments.json")
            self.assertIsNone(P.read_manifest(path, self.fallback))
            with open(path, "w") as f:
                f.write('{"version": 2, "segm')
            self.assertIsNone(P.read_manifest(path, self.fallback))


class RangesTest(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(P.merge_ranges([]), [])
        self.assertEqual(P.merge_ranges([(4, 8), (0, 4.001), (10, 12), (11, 13), (20, 20)]),
                         [(0.0, 8.0), (10.0, 13.0)])
        self.assertEqual(P.merge_ranges([(0, 4), (4.01, 8)]), [(0.0, 4.0), (4.01, 8.0)])   # a real gap stays
        self.assertEqual(P.ranges_text([(0.021333, 8.029333), (20, 24.5)]), "0.021-8.029,20.000-24.500")
        self.assertEqual(P.ranges_text([]), "")


class SourceCopyHelpersTest(unittest.TestCase):
    def test_aac_config_for_adts_sources(self):
        cc = mock.Mock(sample_rate=48000, profile="LC", layout=mock.Mock(nb_channels=2))
        cc.name = "aac"
        self.assertEqual(P.aac_config(cc), bytes([0x11, 0x90]))       # LC, 48 kHz, stereo
        cc.sample_rate = 44100
        self.assertEqual(P.aac_config(cc), bytes([0x12, 0x10]))
        cc.sample_rate = 12345
        self.assertIsNone(P.aac_config(cc))
        cc.name = "opus"
        self.assertIsNone(P.aac_config(cc))

    def test_seek_never_before_stream_start(self):
        s = mock.Mock(time_base=Fraction(1, 90000), start_time=130086)   # MPEG-TS starts at ~1.445 s
        self.assertEqual(P._seek_ts(s, 0.967), 130086)
        self.assertEqual(P._seek_ts(s, 33.0), 33 * 90000)
        s.start_time = None
        self.assertEqual(P._seek_ts(s, -1.0), 0)


class ProgressTextTest(unittest.TestCase):
    def test_waiting_and_loading(self):
        msg, pct = P.progress_json("init3", True)
        self.assertEqual(json.loads(msg), {"req": "init3", "pct": 0, "stage": P.WAIT_STAGE, "eta": None})
        self.assertIn("處理中", P.WAIT_STAGE)
        msg, pct = P.progress_json("5:12.000", False, 0.5, 1.0)
        self.assertEqual((json.loads(msg)["pct"], json.loads(msg)["stage"], pct), (50, P.LOAD_STAGE, 50))
        self.assertEqual(P.progress_json("x", False, 9.0, 1.0)[1], 99)
        self.assertEqual(P.progress_json("x", False, 0.1, 1.0, floor_pct=40)[1], 40)   # never goes down
        self.assertEqual(P.progress_json("x", True, floor_pct=40)[1], 40)

    def test_hold_stage_while_waiting(self):
        for hold, text in (("paused", "已暫停"), ("failed", "失敗")):
            msg, pct = P.progress_json("r1", True, hold=hold)
            self.assertEqual(json.loads(msg), {"req": "r1", "pct": 0, "stage": P.HOLD_STAGES[hold], "eta": None})
            self.assertIn(text, P.HOLD_STAGES[hold])
            self.assertNotEqual(P.HOLD_STAGES[hold], P.WAIT_STAGE)
        self.assertEqual(json.loads(P.progress_json("r1", True, hold=None)[0])["stage"], P.WAIT_STAGE)
        self.assertEqual(json.loads(P.progress_json("r1", True, hold="bogus")[0])["stage"], P.WAIT_STAGE)
        self.assertEqual(P.progress_json("r1", True, floor_pct=40, hold="paused")[1], 40)
        loading = json.loads(P.progress_json("r1", False, 0.5, 1.0, hold="paused")[0])   # segment ready: hold is moot
        self.assertEqual(loading["stage"], P.LOAD_STAGE)


class StatusTest(unittest.TestCase):
    """ProgressivePlayback.status / speed (info panel) and info_panel.progressive_rows, on a fake manifest."""

    def playback(self, segs, duration=100.0, complete=False):
        play = P.PlayInfo(Fraction(30), 120, 0.0, 2, ("processed",), 25)      # 4 s segments, RIFE 2x
        pb = P.ProgressivePlayback.__new__(P.ProgressivePlayback)
        pb.po = mock.Mock(duration=duration)
        pb.watcher = mock.Mock(state=P.ManifestState(play, {k: mock.Mock(frames=120) for k in segs}, complete=complete))
        pb._now, pb._samples = time.monotonic, collections.deque()
        return pb

    def test_status_speed_lead_and_rows(self):
        pb = self.playback(range(0, 5))                                      # 0-20 s processed
        st = pb.status(6.0)
        self.assertEqual((st["src_fps"], st["out_fps"], st["processed"], st["ahead"], st["speed"], st["done"]),
                         (30.0, 60.0, 20.0, 14.0, None, False))
        self.assertEqual(pb.status(30.0)["ahead"], 0.0)                     # not processed there
        pb.sample(0.0)
        pb.watcher.state.segments.update({5: mock.Mock(frames=120), 6: mock.Mock(frames=120)})
        pb.sample(5.0)
        self.assertIsNone(pb.speed())                                        # span too short
        pb.sample(10.0)
        self.assertAlmostEqual(pb.speed(), 0.8)                              # 8 s of video in 10 s
        rows = {r["k"]: r for r in info_panel.progressive_rows(pb.status(6.0), 6.0, 3.0, None, {}, "邊處理邊看")}
        self.assertEqual(rows["背景處理"]["v"], "28%（00:28 / 01:40）")
        self.assertEqual(rows["處理速度"]["v"], "0.80x 即時")
        self.assertEqual(rows["已處理領先"]["v"], "00:22")
        self.assertEqual(rows["預計"]["v"], "約 01:50 後追上（全部 01:30）")   # 22 / (1 - 0.8) s
        rows = {r["k"]: r for r in info_panel.progressive_rows(pb.status(50.0), 50.0, 0.0, None, {}, "")}
        self.assertEqual(rows["預計"]["v"], "播放位置還沒處理，背景處理中")
        pb.sample(40.0)                                                       # 30 s without a new segment
        self.assertEqual(pb.speed(), 0.0)
        self.assertIn("沒有進度", info_panel.progressive_rows(pb.status(6.0), 6.0, 3.0, None, {}, "")[3]["v"])

    def test_done_and_lead_to_the_end(self):
        pb = self.playback(range(0, 25), complete=True)
        rows = {r["k"]: r["v"] for r in info_panel.progressive_rows(pb.status(10.0), 10.0, 5.0, None, {}, "")}
        self.assertEqual((rows["背景處理"], rows["已處理領先"], rows["預計"]), ("全部處理完成", "到結尾", "—"))
        pb = self.playback(range(20, 25))                                    # worker jumped to the end part
        rows = {r["k"]: r["v"] for r in info_panel.progressive_rows(pb.status(90.0), 90.0, 5.0, None, {}, "")}
        self.assertEqual(rows["已處理領先"], "到結尾")


class MessageTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.dir.name, "source.mp4")
        with open(self.src, "wb") as f:
            f.write(b"x")
        self.info = {"duration": 100.0, "fps": FPS, "video_start": 0.02, "frames": 2997, "has_audio": True}

    def tearDown(self):
        self.dir.cleanup()

    def test_valid_message(self):
        po = P.progressive_from_message({"input": self.src, "progressive": True, "queue_id": 7,
                                         "work_dir": os.path.join(self.dir.name, "work"), "title": "T"},
                                        probe=lambda p: self.info)
        self.assertEqual((po.path, po.queue_id, po.title, po.duration, po.has_audio),
                         (os.path.abspath(self.src), "7", "T", 100.0, True))
        self.assertEqual(po.work_dir, os.path.abspath(os.path.join(self.dir.name, "work")))
        self.assertEqual((po.fallback.n, po.fallback.video_start, po.fallback.count), (120, 0.02, 25))

    def test_start_is_parsed_and_clamped(self):
        base = {"input": self.src, "progressive": True, "work_dir": self.dir.name}
        probe = lambda p: self.info   # noqa: E731
        po = P.progressive_from_message(base, probe=probe)
        self.assertEqual((po.start, po.start_requested), (0.0, False))
        for raw, want in ((42.5, 42.5), ("7", 7.0), (0, 0.0), (1e9, 100.0)):
            po = P.progressive_from_message({**base, "start": raw}, probe=probe)
            self.assertEqual((po.start, po.start_requested), (want, True), raw)
        for bad in (-1, "x", float("nan"), float("inf"), True):
            po = P.progressive_from_message({**base, "start": bad}, probe=probe)
            self.assertEqual((po.start, po.start_requested), (0.0, False), bad)

    def test_pressed_at_is_parsed_for_the_first_playable_log(self):   # T1.3
        base = {"input": self.src, "progressive": True, "work_dir": self.dir.name}
        probe = lambda p: self.info   # noqa: E731
        self.assertIsNone(P.progressive_from_message(base, probe=probe).pressed_at)
        po = P.progressive_from_message({**base, "pressed_at": 1700000000.5}, probe=probe)
        self.assertEqual(po.pressed_at, 1700000000.5)
        for bad in ("x", True, float("nan"), None):
            po = P.progressive_from_message({**base, "pressed_at": bad}, probe=probe)
            self.assertIsNone(po.pressed_at, bad)

    def test_rejected(self):
        ok = {"input": self.src, "progressive": True, "work_dir": self.dir.name}
        probe = lambda p: self.info   # noqa: E731
        self.assertIsNotNone(P.progressive_from_message(ok, probe=probe))
        self.assertEqual(P.progressive_from_message(ok, probe=probe).title, "source.mp4")
        for bad in ({**ok, "progressive": False}, {**ok, "input": self.src + ".gone"}, {**ok, "work_dir": None},
                    {**ok, "input": 3}):
            self.assertIsNone(P.progressive_from_message(bad, probe=probe), bad)

        def boom(_p):
            raise RuntimeError("ffprobe failed")
        self.assertIsNone(P.progressive_from_message(ok, probe=boom))

    def test_cached_message_uses_fullprobe_and_loopback_url(self):
        source_probe = {"streams": [{"codec_type": "video", "r_frame_rate": "30000/1001", "start_time": "0.02",
                                     "nb_frames": "30", "duration": "1"},
                                    {"codec_type": "audio"}],
                        "format": {"duration": "1", "start_time": "0.02"},
                        "metadata": {"duration": 100.0}}
        msg = {"input": self.src, "progressive": True, "work_dir": self.dir.name, "queue_id": "q7",
               "source_url": "http://127.0.0.1:4321/vod.m3u8", "source_probe": source_probe}
        po = P.progressive_from_message(msg, probe=lambda _p: self.fail("must use the shared fullprobe"))
        self.assertIsNotNone(po)
        self.assertEqual(po.path, os.path.abspath(self.src))
        self.assertEqual(po.source_url, msg["source_url"])
        self.assertEqual((po.duration, po.info["frames"], po.has_audio), (100.0, 2997, True))
        self.assertEqual(po.fallback.count, 25)
        self.assertIs(po.source_probe, source_probe)
        for url in ("https://127.0.0.1:4321/vod.m3u8", "http://localhost:4321/vod.m3u8",
                    "http://example.com/vod.m3u8"):
            self.assertIsNone(P.progressive_from_message({**msg, "source_url": url}), url)

    def test_cached_feeder_source_errors_do_not_become_silent_audio_eof(self):
        url = "http://127.0.0.1:4321/vod.m3u8"

        class BrokenAv:
            @staticmethod
            def open(_path):
                raise RuntimeError("proxy source failed")

        f = P.Feeder(None, url, 0.0, True)
        try:
            with self.assertRaisesRegex(RuntimeError, "cached source audio could not be opened"):
                f._add_audio(None, BrokenAv())
        finally:
            f.sink.close()

    def test_info_from_probe(self):
        info = P.info_from_probe({"streams": [{"codec_type": "audio"}, {"codec_type": "video", "r_frame_rate": "30000/1001",
                                                                       "start_time": "0.021333", "nb_frames": "216504"}],
                                  "format": {"duration": "7224.045"}})
        self.assertEqual((info["fps"], info["video_start"], info["frames"], info["duration"], info["has_audio"]),
                         (FPS, 0.021333, 216504, 7224.045, True))
        with self.assertRaises(RuntimeError):
            P.info_from_probe({"streams": [{"codec_type": "audio"}]})


# ---------------------------------------------------------------------------------------------------------------
# real segments
# ---------------------------------------------------------------------------------------------------------------

def make_source(path: str, seconds: float):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k", path], check=True)


def make_segments(seg_dir: str, source_frames: int, single: bool = False):
    """queue_worker-like segments: 2x (60000/1001), 3 titled tracks, GOP 15, forced IDR every 2N frames, segment
    muxer cutting there with timestamps reset to 0. single: one untitled track (compare with original=copy: the
    worker's single-layer pipe writes no title)."""
    os.makedirs(seg_dir, exist_ok=True)
    out_fps = FPS * 2
    total = 2 * source_frames
    seg_time = (round(float(Fraction(2 * N) / out_fps) * 1e6) - 50) / 1e6
    meta = []
    for i, t in enumerate([] if single else TITLES):
        meta += [f"-metadata:s:v:{i}", f"title={t}", f"-disposition:v:{i}", "default" if i == 0 else "0"]
    tracks = (["-vf", f"trim=end_frame={total}"] if single else
              ["-filter_complex", f"[0:v]trim=end_frame={total},split=3[a][b][c];[b]hue=s=0[b2];[c]negate[c2]",
               "-map", "[a]", "-map", "[b2]", "-map", "[c2]"])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={out_fps}", *tracks, "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-bf", "0", "-g", "15", "-sc_threshold", "0",
                    "-forced-idr", "1", "-force_key_frames", f"expr:eq(mod(n,{2 * N}),0)", "-pix_fmt", "yuv420p", *meta,
                    "-f", "segment", "-segment_format", "matroska", "-segment_time", f"{seg_time:.6f}",
                    "-reset_timestamps", "1", os.path.join(seg_dir, "seg_%06d.mkv")], check=True)


class StreamReader:
    """Plays mpv's part: connects to the feeder's port and reads the stream to EOF into a file."""

    def __init__(self, port: int, path: str):
        self.path, self.eof, self.bytes = path, False, 0
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with open(self.path, "wb") as f:
            while True:
                try:
                    b = self.sock.recv(1 << 16)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not b:
                    self.eof = True
                    break
                f.write(b)
                self.bytes += len(b)
        self.sock.close()

    def join(self, timeout=20):
        self.thread.join(timeout)
        return not self.thread.is_alive()


class SegmentsBase(unittest.TestCase):
    """Worker-like segments made once per class, a fresh work dir per test, helpers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_progressive_")
        cls.src = os.path.join(cls.tmp, "source.mp4")
        frames = 3 * N + LAST
        make_source(cls.src, float(frames / FPS) + 0.5)
        cls.seg_store = os.path.join(cls.tmp, "store")
        make_segments(cls.seg_store, frames)
        cls.seg_names = sorted(os.listdir(cls.seg_store))
        cls.info = P.probe_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="work_", dir=self.tmp)
        self.seg_dir = os.path.join(self.work, "segments")
        os.makedirs(self.seg_dir)
        self.logs = []
        self.closers = []

    def tearDown(self):
        for c in reversed(self.closers):
            c()

    def play(self, **kw) -> P.PlayInfo:
        return P.PlayInfo(FPS, N, self.info["video_start"], 2, ("processed", "original", "restored"), 4, **kw)

    def publish(self, ks, complete=False):
        """Copy segments ks into work/segments and write a v2 manifest listing them."""
        for k in ks:
            dst = os.path.join(self.seg_dir, f"seg_{k:06d}.mkv")
            if not os.path.exists(dst):
                shutil.copy(os.path.join(self.seg_store, f"seg_{k:06d}.mkv"), dst)
        self.write_manifest({}, ks=ks, complete=complete)

    def write_manifest(self, files=None, partial=(), ks=(0, 1, 2, 3), complete=True):
        """Rewrite segments.json for segments ks; files {k: name} overrides a segment's file name, partial = ks
        marked "partial": true (the worker's not fully interpolated segments)."""
        segs = []
        for k in sorted(ks):
            s = {"k": k, "file": (files or {}).get(k, f"seg_{k:06d}.mkv"), "frames": LAST if k == 3 else N}
            if k in partial:
                s["partial"] = True
            segs.append(s)
        p = self.play()
        write_json(os.path.join(self.work, "segments.json"), {
            "version": 2, "segments": segs, "last_k": 3, "complete": complete, "total_frames": 3 * N + LAST,
            "play": {"fps": str(p.fps), "n": p.n, "video_start": p.video_start, "rate_mult": 2,
                     "layers": list(p.layers), "count": 4}})

    def watcher(self) -> P.ManifestWatcher:
        w = P.ManifestWatcher(self.work, P.play_from_probe(self.info), interval=0.1, log=self.logs.append).start()
        self.closers.append(w.close)
        return w

    def feeder(self, w, target) -> P.Feeder:
        f = P.Feeder(w, self.src, target, True, self.logs.append).start()
        self.closers.append(f.close)
        return f

    def video_packets(self, path, stream=0):
        pk = ffprobe_json(path, "-select_streams", f"v:{stream}", "-show_entries", "packet=pts_time,flags")["packets"]
        return [(float(p["pts_time"]), "K" in p["flags"]) for p in pk]


class FeederCopyOriginalTest(SegmentsBase):
    """compare on with original=copy: single untitled segment track + the source video copied in as oh:original."""
    single = True

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_progressive_copy_")
        cls.src = os.path.join(cls.tmp, "source.mp4")
        frames = 3 * N + LAST
        make_source(cls.src, float(frames / FPS) + 0.5)
        cls.seg_store = os.path.join(cls.tmp, "store")
        make_segments(cls.seg_store, frames, single=True)
        cls.info = P.probe_source(cls.src)

    def play(self, **kw) -> P.PlayInfo:
        return P.PlayInfo(FPS, N, self.info["video_start"], 2, ("processed",), 4, compare=True, original="copy")

    def write_manifest(self, files=None, partial=(), ks=(0, 1, 2, 3), complete=True):
        segs = [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": LAST if k == 3 else N} for k in sorted(ks)]
        write_json(os.path.join(self.work, "segments.json"), {
            "version": 2, "segments": segs, "last_k": 3, "complete": complete, "total_frames": 3 * N + LAST,
            "play": {"fps": str(FPS), "n": N, "video_start": self.info["video_start"], "rate_mult": 2,
                     "layers": ["processed"], "compare": True, "original": "copy", "count": 4}})

    def test_untitled_segment_track_is_processed_and_the_original_is_added(self):
        self.publish([0, 1, 2, 3], complete=True)
        f = self.feeder(self.watcher(), 0.0)
        self.assertTrue(f.ready.wait(10), self.logs)
        self.assertEqual(f.compare_titles(), "oh:processed,oh:original")      # was "" (no compare at all)
        self.assertTrue(f.derive_1x())                                          # the 1x stage is derived from it
        out = os.path.join(self.work, "stream.mkv")
        r = StreamReader(f.sink.port, out)
        self.assertTrue(r.join(), self.logs)
        streams = ffprobe_json(out, "-show_streams")["streams"]
        self.assertEqual([s.get("tags", {}).get("title") for s in streams if s["codec_type"] == "video"],
                         ["oh:processed", "oh:original"])


class FeederTest(SegmentsBase):
    def test_segments_look_like_the_workers(self):
        self.assertEqual(self.seg_names, [f"seg_{k:06d}.mkv" for k in range(4)])
        pr = ffprobe_json(os.path.join(self.seg_store, "seg_000001.mkv"), "-show_streams")
        self.assertEqual([s["tags"]["title"] for s in pr["streams"]], TITLES)
        pk = self.video_packets(os.path.join(self.seg_store, "seg_000001.mkv"))
        self.assertEqual(len(pk), 2 * N)
        self.assertEqual(pk[0], (0.0, True))
        self.assertEqual(len(self.video_packets(os.path.join(self.seg_store, "seg_000003.mkv"))), 2 * LAST)

    def test_remux_waits_for_a_missing_segment_then_runs_to_the_end(self):
        self.publish([0, 1, 3])
        w = self.watcher()
        self.assertEqual(P.ranges_text(w.state.ranges()).count(","), 1)        # 0-1 and 3: two ranges
        f = self.feeder(w, 0.0)
        self.assertTrue(f.ready.wait(10), self.logs)
        self.assertEqual(f.compare_titles(), ",".join(TITLES))
        out = os.path.join(self.work, "stream.mkv")
        r = StreamReader(f.sink.port, out)
        self.assertTrue(wait_for(lambda: f.waiting_k == 2), self.logs)
        time.sleep(0.6)
        self.assertFalse(r.eof)                                  # the stream stays open while waiting
        self.assertTrue(f.alive)
        self.assertEqual(f.waiting_k, 2)
        self.publish([0, 1, 2, 3], complete=True)               # the worker got there
        self.assertTrue(r.join(), self.logs)
        self.assertTrue(r.eof)
        self.assertTrue(wait_for(lambda: not f.alive))
        self.assertTrue(f.ended, self.logs)
        self.assertIsNone(f.error)

        pr = ffprobe_json(out, "-show_streams")
        streams = pr["streams"]
        self.assertEqual([s["codec_type"] for s in streams], ["video"] * 3 + ["audio"])
        self.assertEqual([s["tags"].get("title") for s in streams[:3]], TITLES)
        self.assertEqual([s["codec_name"] for s in streams], ["h264"] * 3 + ["aac"])   # copied, not re-encoded
        self.assertEqual(int(streams[0]["disposition"]["default"]), 1)
        play = self.play()
        frame = float(1 / (FPS * 2))
        for i in range(3):
            pk = self.video_packets(out, i)
            self.assertEqual(len(pk), 2 * (3 * N + LAST), i)    # every frame of every segment, once
            self.assertTrue(pk[0][1])
            self.assertAlmostEqual(pk[0][0], play.video_start, delta=0.001)          # absolute source time
            gaps = [b[0] - a[0] for a, b in zip(pk, pk[1:])]
            self.assertTrue(all(0 < g < frame + 0.002 for g in gaps), (i, max(gaps), min(gaps)))
            self.assertAlmostEqual(pk[-1][0], play.seg_end(3, LAST) - frame, delta=0.002)
            self.assertAlmostEqual(pk[2 * 2 * N][0], play.seg_start(2), delta=0.001)  # segment 2 at its own time
            self.assertTrue(pk[2 * 2 * N][1])
        apk = ffprobe_json(out, "-select_streams", "a", "-show_entries", "packet=pts_time")["packets"]
        self.assertLessEqual(float(apk[0]["pts_time"]), play.video_start + 0.03)
        self.assertGreater(float(apk[-1]["pts_time"]), play.seg_end(3, LAST) - 0.1)
        # (audio is muxed up to AUDIO_LEAD past the video packet being written)
        self.assertLess(float(apk[-1]["pts_time"]), play.seg_end(3, LAST) + P.AUDIO_LEAD + 0.03)
        dec = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-map", "0", "-f", "null", "-"], capture_output=True,
                             text=True)
        self.assertEqual((dec.returncode, dec.stderr.strip()), (0, ""))

    def test_feeder_uses_the_current_manifest_file_of_a_segment(self):
        # a redo swapped segment 1's file: the manifest lists the new name, the old file is gone
        self.publish([0, 1, 2, 3], complete=True)
        os.rename(os.path.join(self.seg_dir, "seg_000001.mkv"), os.path.join(self.seg_dir, "seg_000001_f.mkv"))
        self.write_manifest({1: "seg_000001_f.mkv"}, partial=[2])
        w = self.watcher()
        self.assertEqual(w.state.segments[1].file, "seg_000001_f.mkv")
        self.assertEqual(sorted(w.state.segments), [0, 1, 2, 3])             # "partial" is tolerated, still playable
        f = self.feeder(w, 0.0)
        out = os.path.join(self.work, "renamed.mkv")
        self.assertTrue(f.ready.wait(10), self.logs)
        r = StreamReader(f.sink.port, out)
        self.assertTrue(r.join(), self.logs)
        self.assertTrue(f.ended, self.logs)
        self.assertIsNone(f.error)
        self.assertEqual(len(self.video_packets(out, 0)), 2 * (3 * N + LAST))

    def test_feeder_retries_when_the_segment_file_vanishes_under_it(self):
        self.publish([0, 1, 2, 3], complete=True)
        w = self.watcher()
        calls = []
        real = P.av_open

        def swapping_open(path):
            if os.path.basename(path) == "seg_000001.mkv" and not calls:
                calls.append(path)                                           # the worker's redo lands right now
                os.rename(path, os.path.join(self.seg_dir, "seg_000001_f.mkv"))
                self.write_manifest({1: "seg_000001_f.mkv"})
                raise FileNotFoundError(path)
            return real(path)
        P.av_open = swapping_open
        self.closers.append(lambda: setattr(P, "av_open", real))
        f = self.feeder(w, 0.0)
        out = os.path.join(self.work, "retry.mkv")
        self.assertTrue(f.ready.wait(10), self.logs)
        r = StreamReader(f.sink.port, out)
        self.assertTrue(r.join(), self.logs)
        self.assertEqual(len(calls), 1)
        self.assertTrue(f.ended, self.logs)
        self.assertIsNone(f.error)
        self.assertTrue(any("seg_000001_f.mkv" in m for m in self.logs), self.logs)
        self.assertEqual(len(self.video_packets(out, 0)), 2 * (3 * N + LAST))

    def test_seek_starts_at_the_keyframe_before_the_target(self):
        self.publish([0, 1, 2, 3], complete=True)
        w = self.watcher()
        play = self.play()
        target = play.seg_start(2) + 0.4                         # keyframes every 15 frames at 2x (0.25 s)
        f = self.feeder(w, target)
        self.assertTrue(f.ready.wait(10), self.logs)
        frame = float(1 / (FPS * 2))
        self.assertLessEqual(f.first_s, target)
        self.assertGreater(f.first_s, target - 15 * frame - 0.001)
        self.assertAlmostEqual(f.first_s, play.seg_start(2) + 15 * frame, delta=0.001)
        out = os.path.join(self.work, "seek.mkv")
        r = StreamReader(f.sink.port, out)
        self.assertTrue(r.join(), self.logs)
        for i in range(3):
            pk = self.video_packets(out, i)
            self.assertEqual(pk[0], (round(f.first_s, 3), True), i)
            self.assertEqual(len(pk), 2 * N - 15 + 2 * LAST)
        apk = ffprobe_json(out, "-select_streams", "a", "-show_entries", "packet=pts_time")["packets"]
        self.assertAlmostEqual(float(apk[0]["pts_time"]), f.first_s, delta=0.025)   # one AAC frame
        self.assertEqual([s["codec_type"] for s in ffprobe_json(out, "-show_streams")["streams"]].count("audio"), 1)

    def test_target_in_an_unprocessed_part_waits_and_close_releases_every_file(self):
        self.publish([0, 1])
        w = self.watcher()
        play = self.play()
        f = self.feeder(w, play.seg_start(3) + 0.2)
        self.assertTrue(wait_for(lambda: f.waiting_k == 3), self.logs)
        self.assertFalse(f.ready.is_set())
        f.close()
        self.assertFalse(f.alive)
        # a streaming feeder (mpv connected, reading slowly) stops promptly and lets go of segment + source files
        f2 = self.feeder(w, 0.0)
        self.assertTrue(f2.ready.wait(10), self.logs)
        s = socket.create_connection(("127.0.0.1", f2.sink.port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.assertTrue(wait_for(lambda: f2.out_t is not None), self.logs)
        t = time.time()
        f2.close()
        self.assertLess(time.time() - t, 2.0)
        self.assertFalse(f2.alive)
        s.close()
        w.close()
        for name in os.listdir(self.seg_dir) + ["../segments.json"]:
            p = os.path.normpath(os.path.join(self.seg_dir, name))
            os.rename(p, p + ".x")                               # fails on Windows while a handle is open
            os.rename(p + ".x", p)
        os.rename(self.src, self.src + ".x")
        os.rename(self.src + ".x", self.src)

    def test_playback_messages(self):
        self.publish([0, 1])
        po = P.ProgressiveOpen(self.src, self.work, "q1", "T", self.info)
        clock, sent = FakeClock(), []
        pb = P.ProgressivePlayback(po, send=sent.append, log=self.logs.append, clock=clock, watching_interval=10.0)
        self.closers.append(pb.close)
        target = self.play().seg_start(3) + 0.1
        pb.start(target)
        foreground_want = {"cmd": "want", "id": "q1", "k": 3, "pos": round(target, 3), "foreground": True}
        background_want = {"cmd": "want", "id": "q1", "k": 3, "pos": round(target, 3), "foreground": False}
        self.assertEqual(sent, [foreground_want])                                  # explicit start position/segment
        self.assertTrue(wait_for(lambda: pb.waiting_k == 3), self.logs)
        pb.tick()
        self.assertEqual(sent[1:], [{"cmd": "watching", "id": "q1"}])           # want for 3 was just sent
        clock.t += 4.9
        pb.tick()
        self.assertEqual(len(sent), 2)
        clock.t += 0.2
        pb.tick()
        self.assertEqual(sent[2:], [background_want])                              # prefetch repeat, not a user seek
        clock.t += 5.0
        pb.tick()
        self.assertEqual(sent[3:], [{"cmd": "watching", "id": "q1"}, background_want])
        self.assertEqual(pb.ranges_text(), f"{self.play().video_start:.3f}-{self.play().seg_start(2):.3f}")
        old = pb.feeder
        pb.start(0.0)                                            # a seek into the processed part: new stream
        self.assertFalse(old.alive)
        self.assertEqual(sent[-1], {"cmd": "want", "id": "q1", "k": 0, "pos": 0.0, "foreground": True})
        self.assertTrue(pb.feeder.ready.wait(10))
        self.assertTrue(pb.ready())
        self.assertNotEqual(pb.feeder.url, old.url)
        pb.close()
        self.assertFalse(pb.feeder is not None and pb.feeder.alive)
        # without a queue id nothing is sent
        quiet = P.ProgressivePlayback(P.ProgressiveOpen(self.src, self.work, None, None, self.info), send=sent.append)
        n = len(sent)
        quiet.start(0.0)
        quiet.tick()
        quiet.close()
        self.assertEqual(len(sent), n)

    def test_progressive_playback_passes_runtime_url_to_feeder(self):
        self.publish([0, 1])
        po = P.ProgressiveOpen(self.src, self.work, "q1", "T", self.info,
                               source_url="http://127.0.0.1:4321/vod.m3u8", source_probe={"shared": True})
        captured = []

        class CaptureFeeder:
            def __init__(self, watcher, source, target, has_audio, log):
                self.source = source
                self.waiting_k = None
                self.error = RuntimeError("cached audio source failed")
                captured.append(source)

            def start(self):
                return self

            def close(self):
                pass

            @staticmethod
            def _first_k(state, target):
                return 0

        pb = P.ProgressivePlayback(po, send=lambda _p: None, log=self.logs.append)
        self.closers.append(pb.close)
        with mock.patch.object(P, "Feeder", CaptureFeeder):
            pb.start(0.0)
        pb.tick()
        self.assertEqual(captured, [po.source_url])
        self.assertEqual(pb.hold, "failed")

    def test_clock_is_rate_limited_and_carries_position_and_playing(self):
        self.publish([0, 1])
        po = P.ProgressiveOpen(self.src, self.work, "q1", "T", self.info)
        clock, sent = FakeClock(), []
        pb = P.ProgressivePlayback(po, send=sent.append, log=self.logs.append, clock=clock)
        self.closers.append(pb.close)
        self.assertTrue(pb.clock_due())
        self.assertTrue(pb.clock(12.3456, True, subtitle_pos=10.2))
        self.assertEqual(sent, [{"cmd": "clock", "id": "q1", "pos": 12.346, "playing": True,
                                 "subtitle_pos": 10.2}])
        self.assertFalse(pb.clock_due())
        self.assertFalse(pb.clock(13.0, True))                               # same instant: dropped
        clock.t += P.CLOCK_INTERVAL - 0.1
        self.assertFalse(pb.clock(13.5, False))
        self.assertEqual(len(sent), 1)
        clock.t += 0.2
        self.assertTrue(pb.clock_due())
        self.assertTrue(pb.clock(14.0, False))
        self.assertEqual(sent[1], {"cmd": "clock", "id": "q1", "pos": 14.0, "playing": False})
        self.assertTrue(pb.clock(15.0, True, now=clock.t + 5.0))              # explicit time
        self.assertEqual(len(sent), 3)
        pb.want(2)
        self.assertEqual(sent[-1], {"cmd": "want", "id": "q1", "k": 2, "pos": 15.0, "foreground": False})
        self.assertEqual(len(sent), 4)
        # without a queue id nothing is sent (and nothing is ever due)
        quiet = P.ProgressivePlayback(P.ProgressiveOpen(self.src, self.work, None, None, self.info), send=sent.append,
                                      clock=FakeClock())
        self.closers.append(quiet.close)
        self.assertFalse(quiet.clock_due())
        self.assertFalse(quiet.clock(1.0, True))
        self.assertEqual(len(sent), 4)

    def test_want_reply_sets_hold(self):
        self.publish([0])
        po = P.ProgressiveOpen(self.src, self.work, "q1", "T", self.info)
        sent, replies = [], []

        def send(payload, callback):
            sent.append(payload)
            callback(replies.pop(0))
        pb = P.ProgressivePlayback(po, send=send, log=self.logs.append, clock=FakeClock(), replies=True)
        self.closers.append(pb.close)
        self.assertIsNone(pb.hold)
        for reply, hold in (({"ok": False, "error": "paused"}, "paused"),
                            ({"ok": False, "error": "failed", "message": "boom"}, "failed"),
                            ({"ok": False, "error": "paused"}, "paused"),
                            ({"ok": True}, None),                                  # 繼續 / 重試: the next want clears it
                            ({"ok": False, "error": "paused"}, "paused"),
                            (None, None),                                          # no queue app answered
                            ({"ok": False, "error": "source not downloaded"}, None)):
            replies.append(reply)
            pb.want(1)
            self.assertEqual(pb.hold, hold, reply)
        self.assertEqual(sent, [{"cmd": "want", "id": "q1", "k": 1, "foreground": False}] * 7)
        # without replies=True the callback style is not used (send takes the payload only) and hold stays None
        plain = P.ProgressivePlayback(po, send=sent.append, log=self.logs.append, clock=FakeClock())
        self.closers.append(plain.close)
        plain.want(2)
        self.assertIsNone(plain.hold)
        self.assertEqual(sent[-1], {"cmd": "want", "id": "q1", "k": 2, "foreground": False})



# ---------------------------------------------------------------------------------------------------------------
# run_player in progressive mode (oh_player with test_queue_player's fake mpv / IPC / queue link)
# ---------------------------------------------------------------------------------------------------------------

class RunPlayerProgressiveTest(SegmentsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tests import test_queue_player as TQ   # imports oh_player (torch + lada, no models / GPU / mpv)
        cls.TQ = TQ

    def test_run_player_progressive(self):
        TQ = self.TQ
        self.publish([0, 1, 3])
        rt = TQ.RunPlayerQueueTest("test_processed_message_for_a_missing_file_is_refused")
        rt.setUp()
        self.closers.append(rt.doCleanups)     # (closers run in reverse: player stopped first, then the patches)
        self.closers.append(rt.tearDown)
        inst = TQ.FakeInstance()
        ipc, link = rt.run_player(instance=inst)
        self.assertTrue(wait_for(lambda: inst.on_message is not None))
        reply = inst.on_message({"input": self.src, "progressive": True, "queue_id": "q5", "work_dir": self.work,
                                 "title": "Vid"})
        self.assertEqual(reply, {"ok": True})
        self.assertEqual(inst.on_message({"input": self.src + ".gone", "progressive": True, "work_dir": self.work}),
                         {"ok": False, "error": "file not found"})
        loads = lambda: [c for c in ipc.commands if c[0] == "loadfile"]   # noqa: E731
        self.assertTrue(wait_for(lambda: loads()), rt.logs)
        url = loads()[-1][1]
        self.assertTrue(url.startswith("tcp://127.0.0.1:"), url)
        self.assertEqual(loads()[-1][2], "replace")
        self.assertEqual(ipc.props["user-data/oh/served"], "init1")
        self.assertEqual(ipc.props["user-data/oh/expect-load"], url)
        self.assertEqual(ipc.props["user-data/oh/compare_layers"], ",".join(TITLES))
        self.assertEqual(ipc.props["user-data/oh/active"], "1")
        self.assertAlmostEqual(ipc.props["user-data/oh/duration"], self.info["duration"])
        self.assertEqual(ipc.props["force-media-title"], "Vid")
        self.assertNotEqual(ipc.props.get("user-data/oh/processed"), "1")       # not processed mode: seeks via us
        self.assertTrue(wait_for(lambda: ipc.props.get("user-data/oh/ranges")), rt.logs)
        play = self.play()
        self.assertEqual(ipc.props["user-data/oh/ranges"],
                         f"{play.video_start:.3f}-{play.seg_start(2):.3f},{play.seg_start(3):.3f}-"
                         f"{play.seg_end(3, LAST):.3f}")
        self.assertIn({"cmd": "want", "id": "q5", "k": 0, "pos": 0.0, "foreground": True}, link.posts)
        self.assertTrue(wait_for(lambda: {"cmd": "watching", "id": "q5"} in link.posts))
        # "mpv" reads the stream; the feeder then waits for segment 2: want(2)
        r = StreamReader(int(url.rsplit(":", 1)[1]), os.path.join(self.work, "rp.mkv"))
        self.assertTrue(wait_for(lambda: any(p.get("cmd") == "want" and p.get("id") == "q5" and
                                           p.get("k") == 2 for p in link.posts)), rt.logs)
        ipc.props["user-data/oh/resumed"] = "init1"
        self.assertTrue(wait_for(lambda: ipc.props.get("user-data/oh/progress") == ""), rt.logs)
        # mid-play stall at the unprocessed part: centred 處理中 status
        ipc.props["paused-for-cache"] = True
        self.assertTrue(wait_for(lambda: "處理中" in (ipc.props.get("user-data/oh/status") or "")), rt.logs)
        ipc.props["paused-for-cache"] = False
        # A seek into segment 2 plays the source immediately, without waiting for the worker.
        req = f"1:{play.seg_start(2) + 0.3:.3f}"
        ipc.props["user-data/oh/seek-request"] = req
        self.assertTrue(wait_for(lambda: len(loads()) == 2), rt.logs)
        self.assertEqual(loads()[-1][1], self.src)
        self.assertEqual(ipc.props["user-data/oh/served"], req)
        self.assertEqual(ipc.props["user-data/oh/passthrough"], "1")
        ipc.props["time-pos"] = float(req.split(":")[1])
        ipc.props["user-data/oh/resumed"] = req
        self.assertTrue(r.join(), rt.logs)                       # the old stream was stopped
        self.publish([0, 1, 2, 3], complete=True)               # the worker got there
        self.assertTrue(wait_for(lambda: len(loads()) >= 3), rt.logs)
        # The repaired startup is a seekable short file: start= works here, unlike a live TCP stream.
        self.assertTrue(loads()[-1][1].endswith(".mkv"), loads())
        restored_clip = loads()[-1][1]
        handoff = json.loads(ipc.props["user-data/oh/handoff"])
        ipc.props["user-data/oh/handoff-resumed"] = handoff["token"]
        self.assertTrue(wait_for(lambda: len(loads()) == 4), rt.logs)
        self.assertEqual(loads()[-1][2], "append")
        continuation_url = loads()[-1][1]
        continuation = StreamReader(int(continuation_url.rsplit(":", 1)[1]), os.path.join(self.work, "rp-next.mkv"))
        self.closers.append(continuation.sock.close)
        handoff = json.loads(ipc.props["user-data/oh/handoff"])
        ipc.props["path"] = continuation_url
        ipc.props["user-data/oh/handoff-resumed"] = handoff["token"]
        self.assertTrue(wait_for(lambda: not os.path.exists(restored_clip)), rt.logs)
        self.assertEqual(ipc.props["user-data/oh/served"], req)
        self.assertTrue(wait_for(lambda: ipc.props["user-data/oh/ranges"] ==
                                 f"{play.video_start:.3f}-{play.seg_end(3, LAST):.3f}"), ipc.props["user-data/oh/ranges"])
        # info panel: the source frame rate with the RIFE output rate, the worker's real progress
        ipc.props.update({"user-data/oh/info-visible": True, "time-pos": play.seg_start(1), "container-fps": 59.94})

        def panel():
            vals = ipc.props.get("user-data/oh/info")
            if not vals:
                return {}
            return {r["k"]: r["v"] for sec in json.loads(vals)["sections"] for r in sec["rows"]}

        self.assertTrue(wait_for(lambda: panel().get("背景處理") == "全部處理完成"), panel())
        fps = float(play.fps)
        self.assertEqual(panel()["影格率"], f"{fps:.3f} fps".replace(".000 ", " ") +
                         (f"（插幀輸出 {fps * 2:.5g}）" if play.rate_mult == 2 else ""))
        self.assertEqual(panel()["已處理領先"], "到結尾")
        self.assertEqual(panel()["標題"], "Vid")
        # the viewer's position goes to the queue about once a second (clock), playing = mpv's core is not idle
        ipc.props["core-idle"] = False
        ipc.props["user-data/oh/resumed"] = req                    # (the clock waits for the seek's resume)
        self.assertTrue(wait_for(lambda: any(
            p["cmd"] == "clock" and p["id"] == "q5" and abs(p["pos"] - play.seg_start(1)) < 0.01 and p["playing"]
            for p in list(link.posts))), [p for p in link.posts if p["cmd"] == "clock"][-3:])
        self.assertLess(len([p for p in link.posts if p["cmd"] == "clock"]), 40)    # rate limited
        # quality requests: answered without a load
        ipc.props["user-data/oh/quality-request"] = "2:720"
        self.assertTrue(wait_for(lambda: ipc.props.get("user-data/oh/request-done") == "2:720"), rt.logs)
        # 95 % -> watched
        ipc.props["time-pos"] = self.info["duration"] * 0.95
        self.assertTrue(wait_for(lambda: {"cmd": "watched", "id": "q5"} in link.posts), rt.logs)
        self.assertFalse([p for p in link.posts if p["cmd"] == "player_busy"])   # never: the GPU is the worker's
        # a processed file replaces it: feeder stopped, ranges cleared, segment files released
        inst.on_message({"input": rt.out, "processed": True})
        self.assertTrue(wait_for(lambda: loads()[-1][1] == rt.out), rt.logs)
        self.assertEqual(ipc.props["user-data/oh/ranges"], "")
        for name in os.listdir(self.seg_dir):
            p = os.path.join(self.seg_dir, name)
            os.rename(p, p + ".x")                               # fails on Windows while a handle is open
            os.rename(p + ".x", p)
        rt.stop()


class DeriveParamTest(unittest.TestCase):
    """derive_1x_param: the user-data/oh/derive-1x value for a rate_mult-2 compare output."""

    TWO = ["oh:processed", "oh:original"]

    def test_derives_for_rate_mult_two_compare(self):
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 2, True, self.TWO),
                         "0.033367,30000/1001")

    def test_no_derive_for_rate_mult_one(self):
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 1, True, self.TWO), "")

    def test_no_derive_without_compare(self):
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 2, False, self.TWO), "")

    def test_no_derive_when_restored_track_exists(self):
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 2, True,
                                           self.TWO + ["oh:restored"]), "")

    def test_no_derive_without_processed_track(self):
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 2, True, ["oh:original"]), "")
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), 0.033367, 2, True, None), "")

    def test_float_fps(self):
        self.assertEqual(P.derive_1x_param(29.97, 0.033367, 2, True, self.TWO), "0.033367,2997/100")

    def test_fraction_string_fps(self):
        self.assertEqual(P.derive_1x_param("30000/1001", "0.033367", 2, True, self.TWO), "0.033367,30000/1001")

    def test_unusable_numbers(self):
        self.assertEqual(P.derive_1x_param(0, 0.033367, 2, True, self.TWO), "")
        self.assertEqual(P.derive_1x_param("abc", 0.033367, 2, True, self.TWO), "")
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), "abc", 2, True, self.TWO), "")
        self.assertEqual(P.derive_1x_param(Fraction(30000, 1001), float("nan"), 2, True, self.TWO), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
