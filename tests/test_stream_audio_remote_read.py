"""T2.3 task B: reading a remote audio marker (``audio.rstream``) with ffmpeg.

Small HLS + mp4 fixtures (120 s, a 100 ms beep at every x.5 s multiple of 5) are served by a loopback
ThreadingHTTPServer that records every request, answers 403 unless the custom header is present, and can be told to
403 one path or answer 410 on an expired path."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from oh_streamer.sources import media_input as mi
from oh_streamer.subtitles import cloud_chunks, runtime

HEADER_NAME, HEADER_VALUE = "X-Test", "hello"
_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z", re.IGNORECASE)
DURATION = 120


class RecordingServer:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.requests: list[dict] = []
        self.sent_bytes = 0
        self.deny: set[str] = set()  # paths answered 403
        self.expired_prefix = "/expired/"  # answered 410
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                return

            def handle(self):
                try:
                    super().handle()
                except (ConnectionError, OSError):
                    return

            def do_HEAD(self):
                self._serve(False)

            def do_GET(self):
                self._serve(True)

            def _reply(self, status, body=b""):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _serve(self, body):
                path = self.path.split("?", 1)[0]
                rng = self.headers.get("Range")
                with owner._lock:
                    owner.requests.append({"path": path, "range": rng, "header": self.headers.get(HEADER_NAME),
                                           "ua": self.headers.get("User-Agent")})
                if self.headers.get(HEADER_NAME) != HEADER_VALUE or path in owner.deny:
                    return self._reply(403, b"forbidden")
                if path.startswith(owner.expired_prefix):
                    return self._reply(410, b"gone")
                file = owner.root / path.lstrip("/")
                if not file.is_file():
                    return self._reply(404, b"nf")
                size = file.stat().st_size
                start, end, status = 0, size - 1, 200
                if rng:
                    m = _RANGE.fullmatch(rng.strip())
                    if m and m.group(1):
                        start = int(m.group(1))
                        end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
                        status = 206
                    elif m and m.group(2):
                        start, end, status = max(0, size - int(m.group(2))), size - 1, 206
                length = max(0, end - start + 1)
                self.send_response(status)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not body:
                    return
                # Write in chunks and count what was really sent: an open-ended Range (bytes=N-) that the
                # client abandons early must not be counted as the whole remainder of the file.
                try:
                    with file.open("rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining:
                            data = f.read(min(16384, remaining))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                            with owner._lock:
                                owner.sent_bytes += len(data)
                except (ConnectionError, OSError):
                    pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"

    def reset(self):
        with self._lock:
            self.requests.clear()
            self.sent_bytes = 0
            self.deny.clear()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def _ffmpeg() -> str:
    return runtime.resolve_ffmpeg()


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def make_media(root: Path) -> None:
    ff = _ffmpeg()
    src = root / "src.wav"
    _run([ff, "-y", "-v", "error", "-f", "lavfi", "-i",
          f"aevalsrc=if(between(mod(t\\,5)\\,0.5\\,0.6)\\,0.8*sin(2*PI*1000*t)\\,0)+0.02*(random(0)-0.5)+0.1*sin(2*PI*(300*t+2*t*t)):s=44100:d={DURATION}",
          "-c:a", "pcm_s16le", str(src)])
    _run([ff, "-y", "-v", "error", "-i", str(src), "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(root / "src.mp4")])
    hls = root / "hls"
    hls.mkdir()
    _run([ff, "-y", "-v", "error", "-i", str(src), "-c:a", "aac", "-b:a", "64k", "-f", "hls", "-hls_time", "6",
          "-hls_playlist_type", "vod", "-hls_segment_type", "mpegts", "-hls_segment_filename",
          str(hls / "seg%03d.ts"), str(hls / "index.m3u8")])
    src.unlink()


def beep_onset(wav_path) -> float:
    """Time (s) of the first sample above half scale in a PCM WAV."""
    with wave.open(str(wav_path), "rb") as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype="<i2")
    hits = np.nonzero(np.abs(data) > 8000)[0]
    if not len(hits):
        raise AssertionError("no beep found in output")
    return float(hits[0]) / rate


def flac_to_wav(flac: Path, wav: Path) -> Path:
    _run([_ffmpeg(), "-y", "-v", "error", "-i", str(flac), "-c:a", "pcm_s16le", str(wav)])
    return wav


class RemoteReadCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="stream-audio-remote-")
        cls.root = Path(cls._tmp.name)
        cls.server = None
        try:
            cls.media = cls.root / "media"
            cls.media.mkdir()
            make_media(cls.media)
            cls.server = RecordingServer(cls.media)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.server is not None:
            cls.server.close()
            cls.server = None
        cls._tmp.cleanup()

    def setUp(self):
        self.server.reset()
        self.work = Path(tempfile.mkdtemp(prefix="case-", dir=self.root))
        self.addCleanup(shutil.rmtree, self.work, True)
        self._markers: list[Path] = []
        self.addCleanup(lambda: [mi.unregister_refresher(m) for m in self._markers])

    def marker(self, path: str, name="audio.rstream", *, headers=None) -> Path:
        marker = self.work / "streams" / name
        mi.write_remote_marker(marker, {
            "url": f"{self.server.base}{path}", "kind": "lowest", "hls": path.endswith(".m3u8"),
            "headers": headers if headers is not None else {HEADER_NAME: HEADER_VALUE, "User-Agent": "UA-Test/1"},
            "identity": "id1"})
        self._markers.append(marker)
        return marker

    def paths(self):
        return [r["path"] for r in self.server.requests]


class HlsRemoteTest(RemoteReadCase):
    def test_window_reads_only_nearby_segments_with_headers(self):
        marker = self.marker("/hls/index.m3u8")
        out = self.work / "w.wav"
        runtime.resample_window_to_wav(marker, out, 60.0, 70.0)
        paths = self.paths()
        print("HLS window requests:", paths)
        self.assertIn("/hls/index.m3u8", paths)
        segs = sorted({int(re.search(r"seg(\d+)\.ts", p).group(1)) for p in paths if p.endswith(".ts")})
        print("HLS segments touched:", segs)
        self.assertTrue(set(segs) & {10, 11}, segs)
        self.assertFalse(set(segs) & set(range(2, 9)), segs)
        self.assertLessEqual(len([s for s in segs if s < 2]), 2)
        for r in self.server.requests:
            self.assertEqual(r["header"], HEADER_VALUE, r)
            self.assertEqual(r["ua"], "UA-Test/1", r)
        local = self.work / "local.wav"
        runtime.resample_window_to_wav(self.media / "hls" / "index.m3u8", local, 60.0, 70.0)
        remote_t, local_t = beep_onset(out), beep_onset(local)
        print(f"HLS beep onset remote={remote_t:.5f} local(prefix-atrim)={local_t:.5f} diff={abs(remote_t-local_t):.5f}")
        self.assertLess(abs(remote_t - local_t), 0.03)
        self.assertAlmostEqual(remote_t, 60.5, delta=0.1)

    def test_offsets_inside_a_segment_stay_exact(self):
        # Plain input -ss on audio-only remote HLS lands on a segment boundary (up to ~6 s early); -copyts fixes it.
        marker = self.marker("/hls/index.m3u8")
        local_src = self.media / "hls" / "index.m3u8"
        for start in (30.0, 63.3, 107.9):
            with self.subTest(start=start):
                remote, local = self.work / f"r{start}.wav", self.work / f"l{start}.wav"
                runtime.resample_window_to_wav(marker, remote, start, start + 8.0)
                runtime.resample_window_to_wav(local_src, local, start, start + 8.0)
                self.assertLess(abs(beep_onset(remote) - beep_onset(local)), 0.03)
                self.assertAlmostEqual(runtime.wav_duration(remote), runtime.wav_duration(local), delta=0.05)

    def test_extract_chunk_audio_flac(self):
        marker = self.marker("/hls/index.m3u8")
        out = cloud_chunks.extract_chunk_audio(marker, self.work / "c.flac", 60.0, 10.0, fmt="flac")
        self.assertTrue(out.is_file())
        segs = sorted({int(re.search(r"seg(\d+)\.ts", p).group(1)) for p in self.paths() if p.endswith(".ts")})
        self.assertFalse(set(segs) & set(range(2, 9)), segs)
        for r in self.server.requests:
            self.assertEqual(r["header"], HEADER_VALUE)
        local = cloud_chunks.extract_chunk_audio(self.media / "hls" / "index.m3u8", self.work / "l.flac", 60.0, 10.0,
                                                 fmt="flac")
        remote_t = beep_onset(flac_to_wav(out, self.work / "c.wav"))
        local_t = beep_onset(flac_to_wav(local, self.work / "l.wav"))
        print(f"HLS chunk beep remote={remote_t:.5f} local={local_t:.5f} diff={abs(remote_t-local_t):.5f} segs={segs}")
        self.assertLess(abs(remote_t - local_t), 0.03)
        self.assertAlmostEqual(remote_t, 0.5, delta=0.1)
        self.assertFalse([p for p in self.work.iterdir() if ".partial" in p.name])

    def test_middle_segment_403_raises_not_short_audio(self):
        marker = self.marker("/hls/index.m3u8")
        self.server.deny.add("/hls/seg011.ts")
        out = self.work / "w.wav"
        with self.assertRaises(RuntimeError) as ctx:
            runtime.resample_window_to_wav(marker, out, 60.0, 78.0)
        self.assertNotIn(self.server.base, str(ctx.exception))
        self.assertFalse(out.exists())
        with self.assertRaises(RuntimeError):
            cloud_chunks.extract_chunk_audio(marker, self.work / "c.flac", 60.0, 18.0, fmt="flac")
        self.assertFalse((self.work / "c.flac").exists())

    def test_probe_audio_duration(self):
        marker = self.marker("/hls/index.m3u8")
        dur = runtime._probe_audio_duration(marker, runtime.resolve_ffmpeg(), 30.0)
        print("HLS probe duration:", dur)
        self.assertIsNotNone(dur)
        self.assertAlmostEqual(dur, DURATION, delta=2.5)
        for r in self.server.requests:
            self.assertEqual(r["header"], HEADER_VALUE)

    def test_missing_header_gives_failure_without_url(self):
        marker = self.marker("/hls/index.m3u8", headers={})
        with self.assertRaises(RuntimeError) as ctx:
            runtime.resample_window_to_wav(marker, self.work / "w.wav", 60.0, 70.0)
        self.assertNotIn("127.0.0.1", str(ctx.exception))


class Mp4RemoteTest(RemoteReadCase):
    def assert_partial_read(self, size):
        # How much of an abandoned open-ended Range (bytes=0-) the server manages to push before ffmpeg hangs up
        # depends on socket buffers and load, so a byte budget is flaky; assert the seek itself instead: one request
        # starts well into the file (the 60 s window of 120 s) and the file was not transferred whole.
        starts = [int(m.group(1)) for r in self.server.requests
                  if r["range"] and (m := _RANGE.fullmatch(r["range"].strip())) and m.group(1)]
        self.assertTrue(any(s >= size * 0.3 for s in starts), self.server.requests)
        self.assertLess(self.server.sent_bytes, size)

    def test_window_uses_range_requests(self):
        marker = self.marker("/src.mp4")
        out = self.work / "w.wav"
        runtime.resample_window_to_wav(marker, out, 60.0, 70.0)
        size = (self.media / "src.mp4").stat().st_size
        print(f"mp4 window: requests={[(r['path'], r['range']) for r in self.server.requests]} "
              f"sent={self.server.sent_bytes} of {size}")
        self.assertTrue(self.server.requests)
        for r in self.server.requests:
            self.assertEqual(r["header"], HEADER_VALUE)
            self.assertIsNotNone(r["range"], r)
        self.assert_partial_read(size)
        local = self.work / "local.wav"
        runtime.resample_window_to_wav(self.media / "src.mp4", local, 60.0, 70.0)
        remote_t, local_t = beep_onset(out), beep_onset(local)
        print(f"mp4 beep onset remote={remote_t:.5f} local={local_t:.5f} diff={abs(remote_t-local_t):.5f}")
        self.assertLess(abs(remote_t - local_t), 0.03)

    def test_extract_chunk_audio_flac(self):
        marker = self.marker("/src.mp4")
        out = cloud_chunks.extract_chunk_audio(marker, self.work / "c.flac", 60.0, 10.0, fmt="flac")
        size = (self.media / "src.mp4").stat().st_size
        for r in self.server.requests:
            self.assertEqual(r["header"], HEADER_VALUE)
            self.assertIsNotNone(r["range"], r)
        self.assert_partial_read(size)
        local = cloud_chunks.extract_chunk_audio(self.media / "src.mp4", self.work / "l.flac", 60.0, 10.0, fmt="flac")
        remote_t = beep_onset(flac_to_wav(out, self.work / "c.wav"))
        local_t = beep_onset(flac_to_wav(local, self.work / "l.wav"))
        print(f"mp4 chunk beep remote={remote_t:.5f} local={local_t:.5f} diff={abs(remote_t-local_t):.5f} "
              f"sent={self.server.sent_bytes}/{size}")
        self.assertLess(abs(remote_t - local_t), 0.03)

    def test_probe_audio_duration(self):
        marker = self.marker("/src.mp4")
        dur = runtime._probe_audio_duration(marker, runtime.resolve_ffmpeg(), 30.0)
        self.assertIsNotNone(dur)
        self.assertAlmostEqual(dur, DURATION, delta=1.0)


class ExpiryTest(RemoteReadCase):
    def _expired_marker(self):
        marker = self.marker("/expired/src.mp4")
        calls = []

        def refresher(path):
            calls.append(path)
            doc = mi.read_remote_marker(path)
            mi.write_remote_marker(path, dict(doc, url=f"{self.server.base}/src.mp4", generation=doc["generation"] + 1))

        mi.register_refresher(marker, refresher)
        return marker, calls

    def test_resample_410_refreshes_once_and_succeeds(self):
        marker, calls = self._expired_marker()
        out = self.work / "w.wav"
        runtime.resample_window_to_wav(marker, out, 60.0, 70.0)
        print("410 requests:", [(r["path"], r["range"]) for r in self.server.requests][:4], "refresher calls:", len(calls))
        self.assertEqual(len(calls), 1)
        self.assertEqual(mi.read_remote_marker(marker)["generation"], 1)
        self.assertEqual(mi.read_remote_marker(marker)["identity"], "id1")
        self.assertTrue(any(p.startswith("/expired/") for p in self.paths()))
        self.assertAlmostEqual(beep_onset(out), 60.5, delta=0.1)

    def test_extract_chunk_410_refreshes_once_and_succeeds(self):
        marker, calls = self._expired_marker()
        out = cloud_chunks.extract_chunk_audio(marker, self.work / "c.flac", 60.0, 10.0, fmt="flac")
        self.assertTrue(out.is_file())
        self.assertEqual(len(calls), 1)

    def test_still_expired_after_refresh_fails(self):
        marker = self.marker("/expired/src.mp4")
        mi.register_refresher(marker, lambda path: mi.write_remote_marker(
            path, dict(mi.read_remote_marker(path), generation=mi.read_remote_marker(path)["generation"] + 1)))
        with self.assertRaises(RuntimeError) as ctx:
            runtime.resample_window_to_wav(marker, self.work / "w.wav", 60.0, 70.0)
        self.assertNotIn("127.0.0.1", str(ctx.exception))

    def test_no_refresher_fails(self):
        marker = self.marker("/expired/src.mp4")
        with self.assertRaises(RuntimeError):
            runtime.resample_window_to_wav(marker, self.work / "w.wav", 60.0, 70.0)


class LocalRegressionTest(unittest.TestCase):
    def test_probe_input_args_plain_and_rcache(self):
        self.assertEqual(runtime._probe_input_args("a/b.wav"), ["a/b.wav"])
        arg, opts = mi.ffmpeg_input("x.rcache", "http://127.0.0.1:1/r")
        self.assertEqual(arg, "http://127.0.0.1:1/r")
        self.assertTrue(opts)


if __name__ == "__main__":
    unittest.main()
