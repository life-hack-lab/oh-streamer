"""Tests for hls_cast.py (run: lada\\.venv\\Scripts\\python.exe streamer\\tests\\test_hls_cast.py).

No GPU and no network beyond 127.0.0.1: frames are synthetic (each carries its frame number as a bit pattern), encoded
with libx264; audio comes from testdata\\timecode_10min.mp4 when present.
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
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from fractions import Fraction

import av
import numpy as np
import oh_streamer.platform.hls_cast as H  # noqa: E402

ROOT = _STREAMER_ROOT.parent
AUDIO_SRC = str(ROOT / "testdata" / "timecode_10min.mp4")
W, HGT = 320, 192
TMP_ROOT = tempfile.mkdtemp(prefix="hlstest_")      # removed at the end (see __main__)
BITS = 16


def make_frame(i: int) -> np.ndarray:
    """Frame number i as 16 black/white blocks (survives lossy coding), plus a moving gradient."""
    f = np.full((HGT, W, 3), 40, np.uint8)
    f[:, :, 1] = (np.arange(W)[None, :] + i) % 256
    for b in range(BITS):
        x0, y0 = (b % 8) * 40, (b // 8) * 40
        f[y0 + 4:y0 + 36, x0 + 4:x0 + 36] = 255 if (i >> b) & 1 else 0
    return f


def read_frame_number(img: np.ndarray) -> int:
    n = 0
    for b in range(BITS):
        x0, y0 = (b % 8) * 40, (b // 8) * 40
        if img[y0 + 12:y0 + 28, x0 + 12:x0 + 28].mean() > 128:
            n |= 1 << b
    return n


class FinishedFileTest(unittest.TestCase):
    def test_processed_track_audio_and_seek(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as d:
            path = os.path.join(d, "finished.mkv")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x48:r=10:d=12",
                            "-f", "lavfi", "-i", "color=blue:s=64x48:r=10:d=12",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-map", "0:v", "-map", "1:v", "-map", "2:a",
                            "-c:v", "libx264", "-c:a", "aac", "-metadata:s:v:0", "title=oh:original",
                            "-metadata:s:v:1", "title=oh:processed", path], check=True)
            source = H.FileCastSource(path)
            self.assertEqual(source.stream_index, 1)
            self.assertTrue(source.has_audio)
            inp = source.run_input(4.0)
            try:
                frame, timestamp = next(iter(inp.frames))
                self.assertGreater(frame[:, :, 0].mean(), 200)
                self.assertLess(frame[:, :, 2].mean(), 30)
                self.assertEqual(timestamp, 4.0)
            finally:
                inp.frames.close()
            m = source.meta
            plan = H.SegmentPlan(m.video_fps_exact, m.frames_count, m.video_width, m.video_height,
                                 video_start=float(m.start_pts * m.time_base), has_audio=True)
            vod = H.HlsVod(plan, source.run_input, "libx264", start_k=1, cache_root=d, log=lambda msg: None)
            try:
                for k in (1, 0, 2):
                    data, reason = vod.get_segment(k)
                    self.assertIsNotNone(data, reason)
                    with av.open(io.BytesIO(data), format="mpegts") as container:
                        self.assertEqual(len(container.streams.audio), 1)
                        frame = next(container.decode(video=0)).to_ndarray(format="bgr24")
                        self.assertGreater(frame[:, :, 0].mean(), 200)
            finally:
                vod.close()


class SynthStream:
    """Frames from start_s on, at most `fps_limit` frames per second (None = as fast as possible)."""

    def __init__(self, fps: Fraction, start_s: float, total: int, fps_limit=None, stats=None, fail_at=None):
        self.fps, self.i, self.total, self.limit = fps, round(start_s * fps), total, fps_limit
        self.closed = threading.Event()
        self.stats, self.fail_at = stats, fail_at
        if stats is not None:
            stats["opened"] = stats.get("opened", 0) + 1
            stats.setdefault("starts", []).append(start_s)

    def __iter__(self):
        t0 = time.perf_counter()
        n = 0
        while self.i < self.total and not self.closed.is_set():
            if self.fail_at is not None and self.i >= self.fail_at:
                raise RuntimeError("synthetic read error")
            yield make_frame(self.i), self.i / float(self.fps)
            if self.stats is not None:
                self.stats["frames"] = self.stats.get("frames", 0) + 1
            self.i += 1
            n += 1
            if self.limit:
                ahead = n / self.limit - (time.perf_counter() - t0)
                if ahead > 0:
                    self.closed.wait(ahead)

    def close(self):
        self.closed.set()
        if self.stats is not None:
            self.stats["closed"] = self.stats.get("closed", 0) + 1


def make_vod(fps=Fraction(30), seconds=120, fps_limit=None, window=300.0, cap=H.DEFAULT_CACHE_BYTES, start_k=0,
             audio=True, stats=None, timeout=25.0, fail_at=None, logs=None):
    total = int(seconds * fps)
    has_audio = audio and os.path.exists(AUDIO_SRC)
    plan = H.SegmentPlan(fps, total, W, HGT, has_audio=has_audio)

    def provider(t):
        return H.RunInput(SynthStream(fps, t, total, fps_limit, stats, fail_at), W, HGT,
                          {"src": AUDIO_SRC, "args": [], "fmt_start": 0.0} if has_audio else None)
    log = (lambda m: logs.append(m)) if logs is not None else (lambda m: None)
    vod = H.HlsVod(plan, provider, "libx264", cache_bytes=cap, start_k=start_k, log=log, window_seconds=window,
                   request_timeout=timeout, cache_root=TMP_ROOT)
    return vod


def segment_info(data: bytes) -> dict:
    """First video/audio pts, keyframe flag, first decoded frame number, video frame count."""
    c = av.open(io.BytesIO(data), format="mpegts")
    try:
        out = {"v_pts": None, "a_pts": None, "key": None, "n0": None, "frames": 0, "a_end": None}
        vs = c.streams.video[0]
        has_a = bool(c.streams.audio)
        for pkt in c.demux(vs):
            if pkt.pts is not None:
                t = float(pkt.pts * pkt.time_base)
                if out["v_pts"] is None:
                    out["v_pts"], out["key"] = t, pkt.is_keyframe
            for fr in pkt.decode():        # (the final empty packet flushes the decoder)
                if out["n0"] is None:
                    out["n0"] = read_frame_number(fr.to_ndarray(format="bgr24"))
                out["frames"] += 1
        if has_a:
            c.close()
            c = av.open(io.BytesIO(data), format="mpegts")
            for pkt in c.demux(c.streams.audio[0]):
                if pkt.pts is None:
                    continue
                t = float(pkt.pts * pkt.time_base)
                if out["a_pts"] is None:
                    out["a_pts"] = t
                out["a_end"] = t + float(pkt.duration * pkt.time_base) if pkt.duration else t
        out["has_audio"] = has_a
        return out
    finally:
        c.close()


def get(url, timeout=40):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------------------------

class PlanTest(unittest.TestCase):
    def test_frame_rates(self):
        for fps, n, d in ((Fraction(30000, 1001), 120, Fraction(4004, 1000)), (Fraction(30), 120, Fraction(4)),
                          (Fraction(60), 240, Fraction(4)), (Fraction(60000, 1001), 240, Fraction(4004, 1000)),
                          (Fraction(24000, 1001), 96, Fraction(4004, 1000)), (Fraction(25), 100, Fraction(4))):
            p = H.SegmentPlan(fps, 10_000, 1920, 1080)
            self.assertEqual(p.frames_per_segment, n, fps)
            self.assertEqual(p.seg_duration, d, fps)
            self.assertEqual(p.target_duration(), 4)
            segs = H.parse_media_playlist(p.media_playlist())
            self.assertEqual(len(segs), p.count)
            self.assertAlmostEqual(sum(s[0] for s in segs), float(Fraction(10_000) / fps), places=4)
            self.assertEqual(p.start(7), 7 * d)
            self.assertEqual(p.index_at(float(p.start(7))), 7)
            self.assertEqual(p.index_at(float(p.start(7)) - 0.001), 6)

    def test_playlist_tags(self):
        p = H.SegmentPlan(Fraction(30), 18000, 1920, 1080)
        text = p.media_playlist()
        lines = text.splitlines()
        self.assertEqual(lines[0], "#EXTM3U")
        for tag in ("#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:4", "#EXT-X-PLAYLIST-TYPE:VOD",
                    "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-INDEPENDENT-SEGMENTS"):
            self.assertIn(tag, lines)
        self.assertEqual(lines[-1], "#EXT-X-ENDLIST")
        self.assertEqual(p.count, 150)
        self.assertIn("#EXTINF:4.000000,", lines)
        m = p.master_playlist()
        self.assertIn('CODECS="avc1.64002a,mp4a.40.2"', m)
        self.assertIn("RESOLUTION=1920x1080", m)
        self.assertTrue(m.rstrip().endswith("index.m3u8"))

    def test_short_last_segment_and_target_duration(self):
        p = H.SegmentPlan(Fraction(30), 4500 + 15, 1280, 720)     # 150.5 s
        self.assertEqual(p.count, 38)
        self.assertEqual(p.frames_in(37), 15 + 60)
        self.assertEqual(p.duration(37), Fraction(75, 30))
        self.assertEqual(p.total_duration, Fraction(4515, 30))
        p2 = H.SegmentPlan(Fraction(30), 1000, 1280, 720, target=6.6)   # 198 frames = 6.6 s -> rounds to 7
        self.assertEqual(p2.target_duration(), 7)

    def test_level_and_even_size(self):
        self.assertEqual(H.SegmentPlan(Fraction(60), 100, 1920, 1080).level, "4.2")
        p = H.SegmentPlan(Fraction(30), 100, 3840, 2160)
        self.assertEqual(p.level, "5.2")
        self.assertIn("avc1.640034", p.codecs)
        p = H.SegmentPlan(Fraction(30), 100, 853, 481, has_audio=False)
        self.assertEqual((p.width, p.height), (852, 480))
        self.assertEqual(p.codecs, "avc1.64002a")

    def test_ffmpeg_cmd(self):
        p = H.SegmentPlan(Fraction(30000, 1001), 18000, 1920, 1080)
        cmd = H.build_ffmpeg_cmd(p, 10, 1280, 720, "h264_nvenc", "D", {"src": "a.mp4", "args": ["-headers", "x"],
                                                                     "ss": 40.04})
        s = " ".join(cmd)
        self.assertIn("-initial_offset 50.040000", s)            # TS_BASE + 10 * 4.004
        self.assertIn("-segment_start_number 10", s)
        self.assertIn("-segment_time 4.003950", s)
        self.assertIn("expr:eq(mod(n,120),0)", s)
        self.assertIn("-bf 0", s)
        self.assertIn("scale=1920:1080", s)                       # other renditions are scaled to the plan size
        self.assertIn("-ss 40.040000", s)
        self.assertIn("-headers x -ss", s)
        self.assertNotIn("-output_ts_offset", s)
        self.assertNotIn("scale=", " ".join(H.build_ffmpeg_cmd(p, 0, 1920, 1080, "libx264", "D", None)))


class CacheTest(unittest.TestCase):
    def test_lru_eviction_protects_range(self):
        d = tempfile.mkdtemp(dir=TMP_ROOT)
        c = H.SegmentCache(d, cap_bytes=5000)
        for k in range(10):
            src = os.path.join(d, f"tmp{k}")
            with open(src, "wb") as f:
                f.write(b"x" * 1000)
            self.assertTrue(c.add(k, src))
            time.sleep(0.002)
        self.assertEqual(c.bytes, 10000)
        c.path(1)                               # recently used
        n = c.evict(range(7, 10))
        self.assertEqual(n, 5)
        self.assertEqual(sorted(c.items), [1, 6, 7, 8, 9])
        self.assertEqual(c.ranges(), [(1, 1), (6, 9)])
        self.assertEqual(c.bytes, 5000)
        c.clear()
        self.assertEqual((c.bytes, c.items), (0, {}))
        self.assertEqual(os.listdir(d), [])


class ServerRoutingTest(unittest.TestCase):
    def setUp(self):
        self.vod = make_vod(seconds=30, audio=False)
        self.srv = H.HlsServer(self.vod, "127.0.0.1").start()

    def tearDown(self):
        self.srv.stop()
        self.vod.close()

    def test_token_and_404(self):
        base = f"http://127.0.0.1:{self.srv.port}"
        self.assertEqual(get(f"{base}/index.m3u8")[0], 404)
        self.assertEqual(get(f"{base}/wrongtoken/index.m3u8")[0], 404)
        self.assertEqual(get(f"{base}/{self.srv.token}/")[0], 404)
        self.assertEqual(get(f"{base}/{self.srv.token}/../etc")[0], 404)
        self.assertEqual(get(f"{base}/{self.srv.token}/seg_999.ts")[0], 404)
        self.assertEqual(get(f"{base}/")[0], 404)
        code, body, _ = get(self.srv.url("index.m3u8"))
        self.assertEqual(code, 200)
        self.assertTrue(body.startswith(b"#EXTM3U"))
        self.assertEqual(get(self.srv.url("master.m3u8"))[0], 200)
        self.assertGreaterEqual(len(self.srv.token), 20)

    def test_range_request(self):
        code, full, _ = get(self.srv.url("seg_0.ts"))
        self.assertEqual(code, 200)
        req = urllib.request.Request(self.srv.url("seg_0.ts"), headers={"Range": "bytes=188-375"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(r.read(), full[188:376])
            self.assertEqual(r.headers["Content-Range"], f"bytes 188-375/{len(full)}")


class TsFirstPtsTest(unittest.TestCase):
    @staticmethod
    def packet(pid, sid, pts, adaptation=0):
        ticks = round(pts * 90000)
        pes = bytes([0, 0, 1, sid, 0, 0, 0x80, 0x80, 5,
                     0x21 | ((ticks >> 30) & 7) << 1, (ticks >> 22) & 0xFF, 1 | ((ticks >> 15) & 0x7F) << 1,
                     (ticks >> 7) & 0xFF, 1 | (ticks & 0x7F) << 1])
        head = bytes([0x47, 0x40 | pid >> 8, pid & 0xFF, 0x30 if adaptation else 0x10])
        af = bytes([adaptation]) + b"\xff" * adaptation if adaptation else b""
        return (head + af + pes).ljust(188, b"\xff")

    def test_reads_first_video_and_audio_pts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seg.ts")
            with open(path, "wb") as f:
                f.write(self.packet(256, 0xC0, 19.421, adaptation=7) + self.packet(257, 0xE0, 19.4)
                        + self.packet(257, 0xE0, 19.44))
            video, audio = H.ts_first_pts(path)
            self.assertAlmostEqual(video, 19.4, places=4)
            self.assertAlmostEqual(audio, 19.421, places=4)
            self.assertEqual(H.ts_first_pts(os.path.join(tmp, "missing.ts")), (None, None))


class VodBehaviourTest(unittest.TestCase):
    """Production / request heuristics with real ffmpeg segments (libx264, small frames)."""

    def test_prefetch_far_seek_without_http_request(self):
        vod = make_vod(seconds=200, fps_limit=90)
        try:
            self.assertIsNotNone(vod.get_segment(0)[0])
            vod.prefetch(30)
            deadline = time.monotonic() + 15
            while not vod.is_cached(30) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(vod.is_cached(30))
            self.assertTrue(vod.is_cached(0))
            data, how = vod.get_segment(30)
            self.assertEqual(how, "cached")
            self.check_segment(vod, 30, data)
        finally:
            vod.close()

    def check_segment(self, vod, k, data):
        info = segment_info(data)
        plan = vod.plan
        want = H.TS_BASE + H.MPEGTS_DELAY + float(plan.start(k))
        self.assertAlmostEqual(info["v_pts"], want, delta=0.0012, msg=f"segment {k} video pts")
        self.assertTrue(info["key"], f"segment {k} starts with a keyframe")
        self.assertEqual(info["n0"], k * plan.frames_per_segment, f"segment {k} first frame number")
        self.assertEqual(info["frames"], plan.frames_in(k), f"segment {k} frame count")
        if info["has_audio"]:
            self.assertLess(abs(info["a_pts"] - info["v_pts"]), 0.0215 * 1.01, f"segment {k} A/V start offset")
        return info

    def test_sequential_jump_back_and_probe(self):
        stats, logs = {}, []
        vod = make_vod(seconds=200, fps_limit=90, stats=stats, logs=logs)     # producer ~3x realtime
        srv = H.HlsServer(vod, "127.0.0.1").start()
        try:
            plan = vod.plan
            # sequential start (AVPlayer: playlist, then segment 0, 1, 2 ...)
            code, body, _ = get(srv.url("index.m3u8"))
            self.assertEqual(code, 200)
            for k in range(3):
                code, data, took = get(srv.url(f"seg_{k}.ts"))
                self.assertEqual(code, 200)
                self.check_segment(vod, k, data)
            # prefetch burst: 3 parallel requests just ahead
            res = {}
            ths = [threading.Thread(target=lambda k=k: res.__setitem__(k, get(srv.url(f"seg_{k}.ts"))))
                   for k in (3, 4, 5)]
            [t.start() for t in ths]
            [t.join() for t in ths]
            for k in (3, 4, 5):
                self.assertEqual(res[k][0], 200)
                self.check_segment(vod, k, res[k][1])
            restarts0 = vod.restarts
            self.assertEqual(restarts0, 0)
            # far jump
            code, data, took = get(srv.url("seg_30.ts"))
            self.assertEqual(code, 200)
            self.check_segment(vod, 30, data)
            self.assertEqual(vod.restarts, 1)
            self.assertLess(took, 10)
            # back to cached
            code, data, took2 = get(srv.url("seg_2.ts"))
            self.assertEqual(code, 200)
            self.assertLess(took2, 0.5)
            self.assertEqual(vod.restarts, 1)
            # end probe while "playing" at 31..: does not restart
            get(srv.url("seg_31.ts"))
            code, _, took3 = [None, None, None]
            probe = {}
            th = threading.Thread(target=lambda: probe.setdefault("r", get(srv.url(f"seg_{plan.count - 1}.ts"))))
            th.start()
            time.sleep(0.3)
            self.assertEqual(get(srv.url("seg_32.ts"))[0], 200)      # playback continues near production
            th.join()
            self.assertEqual(probe["r"][0], 503)
            self.assertEqual(vod.restarts, 1)
            self.assertEqual(get(srv.url("seg_33.ts"))[0], 200)
            # an isolated request near the end (a real seek there) restarts after the grace period
            time.sleep(H.PROBE_ACTIVE_WINDOW + 0.2)
            code, data, took4 = get(srv.url(f"seg_{plan.count - 1}.ts"))
            self.assertEqual(code, 200)
            self.check_segment(vod, plan.count - 1, data)
            self.assertEqual(vod.restarts, 2)
            st = vod.status()
            self.assertGreater(st["cached_segments"], 10)
            json.dumps(st)
        finally:
            srv.stop()
            vod.close()
        self.assertEqual(stats["opened"], stats["closed"])
        self.assertFalse(os.path.exists(vod.dir))

    def test_coalesced_rapid_jumps(self):
        stats = {}
        vod = make_vod(seconds=200, fps_limit=120, stats=stats, audio=False)
        srv = H.HlsServer(vod, "127.0.0.1").start()
        try:
            get(srv.url("seg_0.ts"))
            res = {}
            ths = []
            for k in (10, 20, 30, 40):     # scrubbing: each newer one supersedes the previous
                t = threading.Thread(target=lambda k=k: res.__setitem__(k, get(srv.url(f"seg_{k}.ts"))))
                t.start()
                ths.append(t)
                time.sleep(0.03)
            [t.join() for t in ths]
            self.assertEqual(res[40][0], 200)
            self.check_segment(vod, 40, res[40][1])
            for k in (10, 20, 30):
                self.assertEqual(res[k][0], 503, k)
                self.assertLess(res[k][2], 3.0, k)          # answered quickly, not after the timeout
            self.assertEqual(vod.restarts, 1)                # one restart for the whole burst
        finally:
            srv.stop()
            vod.close()
        self.assertEqual(stats["opened"], stats["closed"])

    def test_idle_when_far_ahead_then_resume(self):
        stats, logs = {}, []
        vod = make_vod(seconds=300, stats=stats, window=20.0, audio=False, logs=logs)   # 5 segments window
        try:
            data, how = vod.get_segment(0)
            self.assertIsNotNone(data)
            deadline = time.time() + 20
            while not vod.status()["paused"] and time.time() < deadline:
                time.sleep(0.05)
            st = vod.status()
            self.assertTrue(st["paused"], st)
            f1 = stats["frames"]
            time.sleep(1.0)
            self.assertEqual(stats["frames"] - f1, 0)       # nothing produced while far ahead
            self.assertLessEqual(st["prod_pos"], float(vod.plan.start(vod.window + 2)) + 0.01)
            data, how = vod.get_segment(3)                   # playback approaches: production continues
            self.assertEqual(how, "cached")
            deadline = time.time() + 20
            while not vod.cache.has(vod.window + 3) and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(vod.cache.has(vod.window + 3))
        finally:
            vod.close()

    def test_cache_cap_evicts_old_segments(self):
        vod = make_vod(seconds=200, cap=400_000, audio=False, window=30.0)
        try:
            for k in range(0, 40):
                data, how = vod.get_segment(k)
                self.assertIsNotNone(data, k)
            st = vod.status()
            self.assertLessEqual(st["cached_bytes"], 400_000 + 200_000)
            self.assertGreater(st["evicted"], 0)
            for k in range(37, 40):
                self.assertTrue(vod.cache.has(k), k)          # never evicted near the playhead
            self.assertFalse(vod.cache.has(0))
            data, how = vod.get_segment(0)                    # evicted: produced again
            self.assertEqual((data is not None, how), (True, "跳轉"))
            self.check_segment(vod, 0, data)
        finally:
            vod.close()

    def test_invalidate_restarts_at_segment(self):
        stats = {}
        vod = make_vod(seconds=100, stats=stats, audio=False)
        try:
            for k in range(3):
                self.assertIsNotNone(vod.get_segment(k)[0])
            self.assertTrue(vod.suspend())
            n_closed = stats["closed"]
            self.assertEqual(stats["opened"], n_closed)       # the frame stream is released while suspended
            vod.resume(invalidate=True, at_k=vod.restart_point())
            self.assertEqual(vod.restart_point(), 3)
            self.assertFalse(vod.cache.has(0))
            data, how = vod.get_segment(3)
            self.assertIsNotNone(data)
            self.check_segment(vod, 3, data)
            self.assertEqual(stats["starts"][-1], float(vod.plan.start(3)))
            vod.invalidate(at_k=10)
            data, how = vod.get_segment(10)
            self.check_segment(vod, 10, data)
        finally:
            vod.close()

    def test_timeout_and_error(self):
        vod = make_vod(seconds=100, fps_limit=5, audio=False, timeout=1.0)    # far too slow
        try:
            data, how = vod.get_segment(0)
            self.assertEqual((data is None, how), (True, "timeout"))
        finally:
            vod.close()
        vod = make_vod(seconds=100, audio=False, fail_at=130)
        try:
            self.assertIsNotNone(vod.get_segment(0)[0])
            data, how = vod.get_segment(1)
            self.assertEqual((data is None, how), (True, "error"))
            self.assertIn("synthetic", vod.status()["error"])
        finally:
            vod.close()

    def test_locked_segment_file_is_retried_and_never_404(self):
        """Windows: the finished segment can still be open elsewhere for a moment (WinError 32 on rename)."""
        real_replace = os.replace
        fails = {"left": 5, "hits": 0}

        def flaky_replace(src, dst):
            if fails["left"] > 0 and os.path.basename(src).startswith("seg_"):
                fails["left"] -= 1
                fails["hits"] += 1
                raise PermissionError(32, "The process cannot access the file because it is being used")
            return real_replace(src, dst)
        H.os.replace = flaky_replace
        try:
            vod = make_vod(seconds=20, audio=False)
            try:
                data, how = vod.get_segment(0)
                self.assertIsNotNone(data, how)            # retried until the rename worked
                self.assertEqual(fails["hits"], 5)
            finally:
                vod.close()
        finally:
            H.os.replace = real_replace
        # the move fails completely for the last segment: it is produced again, never reported as missing
        real_move = H._move
        state = {"failed": False}

        def failing_move(src, dst, *a, **kw):
            if not state["failed"] and os.path.basename(src) == "seg_4.ts":
                state["failed"] = True
                raise PermissionError(32, "locked")
            return real_move(src, dst, *a, **kw)
        H._move = failing_move
        try:
            vod = make_vod(seconds=20, audio=False)      # 5 segments; the run reaches the end of the source
            try:
                deadline = time.time() + 30
                while not state["failed"] and time.time() < deadline:
                    time.sleep(0.05)
                self.assertTrue(state["failed"])
                time.sleep(0.5)
                data, how = vod.get_segment(4)               # (re-produced by the idle manager or on request)
                self.assertTrue(data is not None and how in ("cached", "跳轉", "等待"), how)
                self.assertGreaterEqual(vod.runs_started, 2)
                self.check_segment(vod, 4, data)
            finally:
                vod.close()
        finally:
            H._move = real_move

    def test_other_rendition_scaled_to_plan_size(self):
        fps = Fraction(30)
        plan = H.SegmentPlan(fps, 600, W, HGT, has_audio=False)

        class Half(SynthStream):
            def __iter__(self):
                for f, t in super().__iter__():
                    yield np.ascontiguousarray(f[::2, ::2]), t      # a lower rendition (160x96)
        vod = H.HlsVod(plan, lambda t: H.RunInput(Half(fps, t, 600), W // 2, HGT // 2, None), "libx264",
                       log=lambda m: None, cache_root=TMP_ROOT)
        try:
            data, how = vod.get_segment(0)
            c = av.open(io.BytesIO(data), format="mpegts")
            st = c.streams.video[0]
            self.assertEqual((st.codec_context.width, st.codec_context.height), (W, HGT))
            c.close()
        finally:
            vod.close()

    def test_source_shorter_than_planned(self):
        fps = Fraction(30)
        plan = H.SegmentPlan(fps, 600, W, HGT, has_audio=False)      # says 20 s, has 10 s

        def provider(t):
            return H.RunInput(SynthStream(fps, t, 300), W, HGT, None)
        vod = H.HlsVod(plan, provider, "libx264", log=lambda m: None, cache_root=TMP_ROOT)
        try:
            self.assertIsNotNone(vod.get_segment(2)[0])
            data, how = vod.get_segment(3)                    # 300 frames = segments 0..2
            self.assertEqual(how, "missing")
        finally:
            vod.close()


if __name__ == "__main__":
    import shutil
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
