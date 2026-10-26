"""Offline tests for the progressive single-file (byte-range cache) path of queue_download.download.

A local HTTP server with Range support stands in for the media host; yt-dlp format selection is faked.
Run from streamer/: python -m unittest tests.test_queue_download_range"""
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import oh_streamer.queue.queue_download as QD  # noqa: E402
from oh_streamer.sources import range_cache  # noqa: E402
from tests.test_queue_download import Base, FakeYDL, writes_file  # noqa: E402

BLOCK = range_cache.DEFAULT_BLOCK_SIZE     # queue_download uses the module default
SIZE = BLOCK * 2 + 123457            # 3 blocks, the last one short


def payload(size=SIZE) -> bytes:
    return bytes((i * 31 + i // 251) & 0xFF for i in range(size))


class RangeServer:
    """Serves `data` at /v1.bin and /v2.bin with Range support. Knobs: gate_from (requests starting at or after this
    offset wait until `release` is set), ignore_range (always 200), expire_v1 (v1 answers 403 once set)."""

    def __init__(self, data: bytes):
        self.data = data
        self.release = threading.Event()
        self.gate_from: int | None = None
        self.ignore_range = False
        self.expire_v1 = threading.Event()
        self.requests: list[tuple[str, int, int]] = []        # (path, start, end)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path not in ("/v1.bin", "/v2.bin"):
                    self.send_error(404)
                    return
                if path == "/v1.bin" and outer.expire_v1.is_set():
                    self.send_error(403)
                    return
                total = len(outer.data)
                m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
                if not m or outer.ignore_range:
                    body = outer.data
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass
                    return
                start = int(m.group(1))
                end = min(total - 1, int(m.group(2))) if m.group(2) else total - 1
                outer.requests.append((path, start, end))
                if outer.gate_from is not None and start >= outer.gate_from:
                    if not outer.release.wait(20):
                        self.send_error(504)
                        return
                body = outer.data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


FAKE_PROBE = {
    "format": {"format_name": "mov,mp4", "duration": "12.5", "start_time": "0.000000"},
    "streams": [{"codec_type": "video", "codec_name": "h264", "width": 640, "height": 360,
                 "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "duration": "12.480000",
                 "start_time": "0.000000", "nb_frames": "312"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "12.5"}],
}


class RangeBase(Base):
    def setUp(self):
        super().setUp()
        self.data = payload()
        self.server = RangeServer(self.data)
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: range_cache.close_session(self.dest / "range-cache" / range_cache.MARKER_NAME))
        self.logs: list[str] = []
        patches = [mock.patch.object(QD, "log", self.logs.append),
                   mock.patch.object(QD, "_resolve", lambda target, headers: (target, dict(headers), "Resolved")),
                   mock.patch.object(QD, "_extract_selected", self._extract),
                   mock.patch.object(QD, "_probe_range_url", lambda url, cancel, timeout=90.0: json.loads(
                       json.dumps(FAKE_PROBE))),
                   # the payload is not real media: no tail probe (covered by tests.test_queue_download_span)
                   mock.patch.object(QD, "_tail_span", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.extract_calls = 0
        self.info = self.progressive_info("/v1.bin")

    def progressive_info(self, path):
        return {"title": "Picked", "url": self.server.base + path, "protocol": "https", "ext": "mp4",
                "vcodec": "h264", "acodec": "aac"}

    def _extract(self, url, headers, fmt):
        self.extract_calls += 1
        return self.info

    def start_download(self, on_ready):
        box = {}

        def run():
            try:
                box["result"] = QD.download("https://example.com/video", {}, None, self.dest,
                                            self.progress.append, self.cancel, on_source_ready=on_ready)
            except BaseException as exc:      # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, box

    def marker(self) -> Path:
        return self.dest / "range-cache" / range_cache.MARKER_NAME


class RangeDownloadTest(RangeBase):
    def test_source_ready_fires_before_download_completes(self):
        self.server.gate_from = BLOCK            # blocks 1.. are held back by the server
        ready, handles = threading.Event(), []

        def on_ready(session):
            handles.append(session)
            ready.set()

        t, box = self.start_download(on_ready)
        self.assertTrue(ready.wait(5), "source should be announced once the first block + metadata exist")
        self.assertTrue(t.is_alive(), "the download must still be running")
        session = handles[0]
        self.assertEqual(session.status, "downloading")
        self.assertTrue(session.playlist_path.endswith("source.rcache"))
        self.assertTrue(session.runtime_url.startswith("http://127.0.0.1:"))
        self.assertEqual(session.metadata["format"]["duration"], "12.480000")      # video span, not the longer audio
        self.assertEqual(session.snapshot()["total_bytes"], SIZE)
        self.server.release.set()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertNotIn("error", box)
        result = box["result"]
        self.assertEqual((result.path, result.title, result.duration), (session.playlist_path, "Resolved", 12.48))
        self.assertEqual(session.status, "complete")
        self.assertEqual((self.dest / "range-cache" / "source.data").read_bytes(), self.data)
        self.assertEqual(len(handles), 1)
        # exactly one upstream read per block (plus the 0-0 probe): the worker/player read the local server only
        self.assertEqual(sorted(s for _p, s, e in self.server.requests if e > 0), [0, BLOCK, 2 * BLOCK])

    def test_progress_reports_exact_bytes_and_blocks(self):
        t, box = self.start_download(lambda s: None)
        t.join(15)
        self.assertNotIn("error", box)
        downloading = [p for p in self.progress if p["stage"] == "downloading"]
        last = self.progress[-1]
        self.assertEqual(last["stage"], "finished")
        self.assertEqual((last["downloaded_bytes"], last["total_bytes"]), (SIZE, SIZE))
        self.assertEqual((last["fragment_index"], last["fragment_count"]), (3, 3))
        self.assertFalse(last["total_approx"])
        for p in downloading:
            self.assertEqual(p["total_bytes"], SIZE)
            self.assertLessEqual(p["downloaded_bytes"], SIZE)

    def test_cancel_pauses_session_and_resume_skips_ready_blocks(self):
        self.server.gate_from = BLOCK
        ready = threading.Event()
        handles = []
        t, box = self.start_download(lambda s: (handles.append(s), ready.set()))
        self.assertTrue(ready.wait(5))
        deadline = time.monotonic() + 5            # the faked ffprobe does not wait for block 0 like the real one
        while not handles[0].snapshot()["ready_segments"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.cancel.set()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertIsInstance(box.get("error"), QD.DownloadCancelled)
        self.assertEqual(handles[0].status, "paused")
        side = json.loads((self.dest / "range-cache" / "range-source.json").read_text("utf-8"))
        self.assertEqual(side["status"], "paused")
        self.assertEqual([x["index"] for x in side["ready_segments"]], [0])
        self.server.release.set()

        # restart: same item folder, nothing but the missing blocks is requested, block 0 is not fetched again
        self.server.requests.clear()
        self.server.gate_from = None
        self.cancel.clear()
        ready2, handles2 = threading.Event(), []
        t2, box2 = self.start_download(lambda s: (handles2.append(s), ready2.set()))
        t2.join(15)
        self.assertNotIn("error", box2)
        self.assertEqual(handles2[0].status, "complete")
        self.assertEqual(sorted(s for _p, s, e in self.server.requests if e > 0), [BLOCK, 2 * BLOCK])
        self.assertEqual((self.dest / "range-cache" / "source.data").read_bytes(), self.data)
        self.assertEqual(handles2[0].identity, handles[0].identity)

    def test_complete_cache_is_reused_without_network(self):
        t, box = self.start_download(lambda s: None)
        t.join(15)
        self.assertNotIn("error", box)
        range_cache.close_session(self.marker())
        self.server.requests.clear()
        self.extract_calls = 0
        seen = []
        result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                             self.cancel, on_source_ready=seen.append)
        self.assertEqual(self.extract_calls, 0)
        self.assertEqual(self.server.requests, [])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].status, "complete")
        self.assertEqual(result.path, seen[0].playlist_path)
        self.assertEqual(result.duration, 12.48)

    def test_expired_url_is_re_resolved_and_download_continues(self):
        self.server.gate_from = BLOCK
        ready = threading.Event()
        t, box = self.start_download(lambda s: ready.set())
        self.assertTrue(ready.wait(5))
        calls_before = self.extract_calls
        self.info = self.progressive_info("/v2.bin")        # what the re-resolve will return
        self.server.expire_v1.set()                         # the old signed URL now answers 403
        self.server.release.set()
        t.join(15)
        self.assertNotIn("error", box)
        self.assertGreater(self.extract_calls, calls_before)
        self.assertTrue(any(p == "/v2.bin" for p, _s, _e in self.server.requests))
        self.assertEqual((self.dest / "range-cache" / "source.data").read_bytes(), self.data)

    def test_headers_of_the_selected_format_reach_the_upstream(self):
        seen = []
        orig = range_cache.prepare_session

        def spy(*a, **kw):
            seen.append(kw.get("headers"))
            return orig(*a, **kw)

        self.info["http_headers"] = {"Referer": "https://example.com/"}
        with mock.patch.object(range_cache, "prepare_session", spy):
            t, box = self.start_download(lambda s: None)
            t.join(15)
        self.assertNotIn("error", box)
        self.assertEqual(seen[0].get("Referer"), "https://example.com/")


class RangeFallbackTest(RangeBase):
    def run_fallback(self):
        FakeYDL.script = [lambda y: writes_file(y)]
        callbacks = []
        result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append, self.cancel,
                             on_source_ready=callbacks.append)
        self.assertEqual(callbacks, [])
        self.assertTrue(result.path.endswith("source.mkv"))
        self.assertEqual(len(FakeYDL.instances), 1)
        return result

    def test_server_without_range_support_falls_back(self):
        self.server.ignore_range = True
        self.run_fallback()
        self.assertTrue(any("does not support byte ranges" in m for m in self.logs), self.logs)
        self.assertFalse((self.dest / "range-cache").exists())

    def test_separate_audio_and_video_falls_back_and_says_second_phase(self):
        self.info = {"title": "T", "requested_formats": [self.progressive_info("/v1.bin"),
                                                         self.progressive_info("/v2.bin")]}
        self.run_fallback()
        self.assertTrue(any("separate" in m and "second phase" in m for m in self.logs), self.logs)

    def test_non_progressive_protocols_fall_back(self):
        for protocol, ext in (("http_dash_segments", "mp4"), ("rtmp", "flv")):
            with self.subTest(protocol=protocol):
                self.logs.clear()
                FakeYDL.instances = []
                self.info = {**self.progressive_info("/v1.bin"), "protocol": protocol, "ext": ext}
                self.run_fallback()
                self.assertTrue(any("Range cache not used" in m for m in self.logs), self.logs)

    def test_unreadable_media_falls_back_and_removes_the_cache(self):
        def bad_probe(url, cancel, timeout=90.0):
            raise range_cache.RangeCacheError("ffprobe failed: invalid data")

        with mock.patch.object(QD, "_probe_range_url", bad_probe):
            self.run_fallback()
        self.assertTrue(any("ffprobe failed" in m for m in self.logs), self.logs)
        self.assertFalse((self.dest / "range-cache").exists())
        self.assertIsNone(range_cache.get_session(self.marker()))

    def test_candidate_rules(self):
        ok = self.progressive_info("/v1.bin")
        self.assertIsNotNone(QD._progressive_candidate(ok)[1])
        self.assertIsNotNone(QD._progressive_candidate({"requested_formats": [ok]})[1])
        self.assertIsNone(QD._progressive_candidate({**ok, "is_live": True})[1])
        self.assertIsNone(QD._progressive_candidate({**ok, "url": "ftp://x/y.mp4"})[1])
        self.assertIsNone(QD._progressive_candidate({**ok, "url": "https://x/v.m3u8?a=1"})[1])
        self.assertIsNone(QD._progressive_candidate({**ok, "vcodec": "none"})[1])
        self.assertIsNone(QD._progressive_candidate({**ok, "has_drm": True})[1])


def _ffmpeg_ok():
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg/ffprobe not installed")
class RealMp4Test(Base):
    """A real mp4 with the moov atom at the END (ffmpeg default) and audio longer than video, through the whole
    path: tail-moov metadata over the local range server, then the worker's own PTS / completion gates."""

    def setUp(self):
        super().setUp()
        self.log = []
        patcher = mock.patch.object(QD, "log", self.log.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = Path(self.tmp.name) / "src.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=30",
                        "-f", "lavfi", "-t", "2.3", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264",
                        "-bf", "2", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
        self.data = src.read_bytes()
        self.assertGreater(self.data.rfind(b"moov"), len(self.data) // 2, "fixture must have a trailing moov")
        self.server = RangeServer(self.data)
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: range_cache.close_session(self.dest / "range-cache" / range_cache.MARKER_NAME))

    def test_tail_moov_metadata_and_worker_gates(self):
        import av
        from oh_streamer.queue import queue_worker as W

        info = {"title": "Real", "url": self.server.base + "/v1.bin", "protocol": "https", "ext": "mp4",
                "vcodec": "h264", "acodec": "aac"}
        orig = range_cache.prepare_session
        small = lambda *a, **kw: orig(*a, **{**kw, "block_size": 2048})       # noqa: E731
        handles = []
        with mock.patch.object(QD, "_resolve", lambda t, h: (t, dict(h), None)), \
                mock.patch.object(QD, "_extract_selected", lambda *a: info), \
                mock.patch.object(range_cache, "prepare_session", small):
            result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                                 self.cancel, on_source_ready=handles.append)
        session = handles[0]
        self.assertEqual(session.status, "complete", self.log)
        self.assertGreater(session.snapshot()["segment_count"], 3)
        meta = session.metadata
        video = next(s for s in meta["streams"] if s["codec_type"] == "video")
        self.assertEqual(meta["format"]["duration"], video["duration"])            # video span (2.0 s), not the audio's
        self.assertAlmostEqual(float(meta["format"]["duration"]), 2.0, places=2)
        self.assertAlmostEqual(result.duration, 2.0, places=2)

        # worker side: probe normalisation, then every decoded frame through the same PTS gate the worker applies
        job_probe = W.full_vod_probe(meta)
        parsed = W.parse_probe(job_probe)
        self.assertEqual(parsed.frames, 60)
        with av.open(session.runtime_url) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            times = [float(f.pts * tb) for f in container.decode(stream)]
        self.assertEqual(len(times), parsed.frames)
        for i, t in enumerate(times):
            self.assertIsNone(W.cache_pts_error((t,), parsed.fps, parsed.video_start, i), f"frame {i}")
        status = json.loads(Path(session.status_path).read_text("utf-8"))
        self.assertIsNone(W.source_status_error(status, session.identity, session.generation))
        self.assertIsNone(W.source_completion_error(
            status, session.identity, session.generation, (times[-1],), parsed.fps, parsed.duration,
            parsed.video_start, natural_end=True, frame_count=len(times), expected_frames=parsed.frames))


if __name__ == "__main__":
    unittest.main()
