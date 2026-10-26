"""Offline tests for queue_download.py (no network; yt_dlp.YoutubeDL and the resolvers are faked).
Run from streamer/: python run_tests.py (or python -m unittest tests.test_queue_download)."""
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
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen
from unittest import mock
import oh_streamer.queue.queue_download as QD  # noqa: E402
from oh_streamer.sources import hls_cache  # noqa: E402
import yt_dlp  # noqa: E402


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL. `script` has one entry per extract_info call: a callable(ydl) -> info (it may
    call the progress hooks or raise) or an Exception instance to raise."""
    instances = []
    script = []

    def __init__(self, opts):
        self.opts = opts
        FakeYDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        self.url = url
        step = FakeYDL.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(self)


def writes_file(ydl, ext="mkv", progress=(), title="T", duration=61.5):
    for d in progress:
        for h in ydl.opts["progress_hooks"]:
            h(d)
    out = Path(ydl.opts["outtmpl"].replace("%(ext)s", ext))
    out.write_bytes(b"x")
    return {"title": title, "duration": duration, "requested_downloads": [{"filepath": str(out)}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "item"
        FakeYDL.instances = []
        FakeYDL.script = []
        self.progress = []
        self.cancel = threading.Event()
        patches = [mock.patch.object(QD.yt_dlp, "YoutubeDL", FakeYDL),
                   mock.patch.object(QD, "log", lambda m: None),
                   mock.patch("oh_streamer.sources.http_transport.configure"),
                   mock.patch.object(QD.source_resolve, "find_resolver", lambda s, *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_dl(self, target="https://example.com/v", headers=None, quality=None):
        return QD.download(target, headers or {}, quality, self.dest, self.progress.append, self.cancel)


class LocalFileTest(Base):
    def test_passthrough_no_copy(self):
        f = Path(self.tmp.name) / "movie.mp4"
        f.write_bytes(b"abc")
        r = self.run_dl(str(f))
        self.assertEqual(r.path, str(f))
        self.assertEqual(r.title, "movie")
        self.assertFalse(self.dest.exists())
        self.assertEqual(FakeYDL.instances, [])

    def test_missing_file(self):
        with self.assertRaises(QD.DownloadFailed):
            self.run_dl(str(Path(self.tmp.name) / "nope.mp4"))


class FormatSelectorTest(unittest.TestCase):
    def test_specs(self):
        self.assertEqual(QD.format_selector(None), "bv*+ba/b")
        self.assertEqual(QD.format_selector("best"), "bv*+ba/b")
        self.assertEqual(QD.format_selector("worst"), "wv*+wa/w")
        self.assertIn("height<=720", QD.format_selector("720"))
        self.assertIn("height<=720", QD.format_selector(720))
        self.assertIn("height<=480", QD.format_selector("480p"))
        with self.assertRaises(ValueError):
            QD.format_selector("huge")


class DownloadTest(Base):
    def test_options_and_result(self):
        FakeYDL.script = [lambda y: writes_file(y)]
        r = self.run_dl(headers={"Referer": "https://x/"}, quality=720)
        o = FakeYDL.instances[0].opts
        self.assertEqual(o["http_headers"], {"Referer": "https://x/"})
        self.assertTrue(o["continuedl"])
        self.assertTrue(o["noprogress"])
        self.assertEqual(o["merge_output_format"], "mkv")
        self.assertIn("height<=720", o["format"])
        self.assertTrue(o["outtmpl"].endswith("source.%(ext)s"))
        self.assertEqual(Path(r.path), self.dest / "source.mkv")
        self.assertEqual((r.title, r.duration), ("T", 61.5))

    def test_bad_quality(self):
        with self.assertRaises(QD.DownloadFailed):
            self.run_dl(quality="huge")

    def test_progress_mapping(self):
        prog = [
            {"status": "downloading", "downloaded_bytes": 100, "total_bytes": 1000, "speed": 50.0, "eta": 18},
            {"status": "downloading", "downloaded_bytes": 300, "total_bytes_estimate": 900.0, "speed": 60.0, "eta": 10,
             "fragment_index": 3, "fragment_count": 10},
            {"status": "finished", "downloaded_bytes": 900, "total_bytes": 900, "filename": "source.mp4"},
        ]
        FakeYDL.script = [lambda y: writes_file(y, progress=prog)]
        self.run_dl()
        a, b, c = self.progress
        self.assertEqual((a["stage"], a["downloaded_bytes"], a["total_bytes"], a["speed"], a["eta"]),
                         ("downloading", 100, 1000, 50.0, 18))
        self.assertIsNone(a["fragment_index"])
        self.assertEqual((b["total_bytes"], b["fragment_index"], b["fragment_count"]), (900.0, 3, 10))
        self.assertEqual((c["stage"], c["filename"]), ("finished", "source.mp4"))

    def test_progress_callback_errors_do_not_abort(self):
        FakeYDL.script = [lambda y: writes_file(y, progress=[{"status": "downloading", "downloaded_bytes": 1}])]

        def boom(_d):
            raise RuntimeError("ui")
        r = QD.download("https://example.com/v", {}, None, self.dest, boom, self.cancel)
        self.assertTrue(r.path)

    def test_external_resolver_used(self):
        FakeYDL.script = [lambda y: writes_file(y, title="generic")]
        with mock.patch.object(QD.source_resolve, "find_resolver",
                               lambda s, *a, **k: {"match": "x", "command": ["c"]}), \
                mock.patch.object(QD.source_resolve, "resolve_external",
                                  lambda page, h, r, *a: ("https://cdn/i.m3u8", {**h, "X": "1"}, "Resolved")):
            r = self.run_dl(headers={"Referer": "r"})
        y = FakeYDL.instances[0]
        self.assertEqual(y.url, "https://cdn/i.m3u8")
        self.assertEqual(y.opts["http_headers"], {"Referer": "r", "X": "1"})
        self.assertEqual(r.title, "Resolved")


class CancelTest(Base):
    def test_cancel_from_progress_hook(self):
        def script(y):
            y.opts["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 1})
            self.cancel.set()
            y.opts["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 2})
            self.fail("hook should have raised")
        FakeYDL.script = [script]
        with self.assertRaises(QD.DownloadCancelled):
            self.run_dl()
        self.assertEqual(len(self.progress), 1)

    def test_cancel_wrapped_in_download_error(self):
        def script(y):
            self.cancel.set()
            raise yt_dlp.utils.DownloadError("ERROR: something wrapped the cancel")
        FakeYDL.script = [script]
        with self.assertRaises(QD.DownloadCancelled):
            self.run_dl()

    def test_cancel_before_start(self):
        self.cancel.set()
        with self.assertRaises(QD.DownloadCancelled):
            self.run_dl()
        self.assertEqual(FakeYDL.instances, [])


class WatcherTest(Base):
    """Downloaders that never call a progress hook (ffmpeg): the watcher polls the growing files."""

    def setUp(self):
        super().setUp()
        for name in ("IDLE_AFTER", "POLL_EVERY", "TICK"):
            p = mock.patch.object(QD._ProgressWatcher, name, {"IDLE_AFTER": 0.15, "POLL_EVERY": 0.1, "TICK": 0.03}[name])
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def growing(ydl, steps=8, chunk=1000, pause=0.1, info=None):
        part = Path(ydl.opts["outtmpl"].replace("%(ext)s", "mkv.part"))
        part.parent.mkdir(parents=True, exist_ok=True)
        if info is not None:
            ydl.opts["match_filter"](info)
        with open(part, "wb") as f:
            for _ in range(steps):
                f.write(b"x" * chunk)
                f.flush()
                time.sleep(pause)
        out = part.with_suffix("")
        part.rename(out)
        return {"title": "T", "duration": 10, "requested_downloads": [{"filepath": str(out)}]}

    def test_reports_bytes_speed_and_estimate(self):
        info = {"duration": 8.0, "tbr": 8.0}      # 8 kbit/s x 8 s = 8000 bytes
        FakeYDL.script = [lambda y: self.growing(y, info=info)]
        self.run_dl()
        polled = [p for p in self.progress if p["downloaded_bytes"]]
        self.assertTrue(polled, "watcher should have reported")
        self.assertTrue(all(p["stage"] == "downloading" for p in polled))
        self.assertEqual([p["downloaded_bytes"] for p in polled], sorted(p["downloaded_bytes"] for p in polled))
        self.assertTrue(any(p["speed"] and p["speed"] > 0 for p in polled))
        with_total = [p for p in polled if p["total_bytes"]]
        self.assertTrue(with_total)
        self.assertEqual(with_total[0]["total_bytes"], 8000)
        self.assertTrue(with_total[0]["total_approx"])

    def test_no_estimate_means_no_total(self):
        FakeYDL.script = [lambda y: self.growing(y)]
        self.run_dl()
        polled = [p for p in self.progress if p["downloaded_bytes"]]
        self.assertTrue(polled)
        self.assertTrue(all(p["total_bytes"] is None and p["eta"] is None for p in polled))

    def test_stops_after_finish(self):
        FakeYDL.script = [lambda y: self.growing(y, steps=4)]
        self.run_dl()
        n = len(self.progress)
        time.sleep(0.4)
        self.assertEqual(len(self.progress), n)
        self.assertFalse([t for t in threading.enumerate() if t.name == "queue-dl-watch"])

    def test_silent_while_hooks_fire(self):
        def script(y):
            for i in range(1, 6):
                y.opts["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": i})
                time.sleep(0.05)
            return writes_file(y)
        FakeYDL.script = [script]
        self.run_dl()
        self.assertEqual([p["downloaded_bytes"] for p in self.progress], [1, 2, 3, 4, 5])

    def test_stops_on_cancel_and_kills_new_ffmpeg(self):
        killed = []

        class Proc:
            pid = 4242

            def name(self):
                return "ffmpeg.exe"

            def kill(self):
                killed.append(self.pid)
                release.set()

        release = threading.Event()

        class Me:
            def children(self, recursive=False):
                return [Proc()] if started.is_set() else []

        started = threading.Event()

        def script(y):
            self.growing(y, steps=2)            # some bytes appear, no hooks
            started.set()
            threading.Timer(0.2, self.cancel.set).start()
            # like FFmpegFD: block in proc.wait() until the process is killed, then fail
            self.assertTrue(release.wait(5), "watcher should kill the ffmpeg child")
            raise yt_dlp.utils.DownloadError("ERROR: ffmpeg exited with code 1")

        FakeYDL.script = [script]
        fake_psutil = mock.Mock(Process=lambda: Me())
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with self.assertRaises(QD.DownloadCancelled):
                self.run_dl()
        self.assertEqual(killed, [4242])
        self.assertFalse([t for t in threading.enumerate() if t.name == "queue-dl-watch"])

    def test_baseline_ffmpeg_not_killed(self):
        class Proc:
            pid = 7

            def name(self):
                return "ffmpeg.exe"

            def kill(self):
                raise AssertionError("pre-existing ffmpeg must not be killed")

        fake_psutil = mock.Mock(Process=lambda: mock.Mock(children=lambda recursive=False: [Proc()]))
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            w = QD._ProgressWatcher(Path(self.tmp.name), lambda d: None, self.cancel)
            self.cancel.set()
            w._kill_new_ffmpeg()

    def test_estimate_total_bytes(self):
        est = QD.estimate_total_bytes
        self.assertEqual(est({"filesize": 500}), 500)
        self.assertEqual(est({"filesize_approx": 700.0}), 700)
        self.assertEqual(est({"tbr": 800, "duration": 10}), 800 * 1000 / 8 * 10)
        self.assertEqual(est({"requested_formats": [{"filesize": 100}, {"abr": 8, "duration": 10}]}), 100 + 10000)
        self.assertEqual(est({"requested_formats": [{"vbr": 8}, {"abr": 8}], "duration": 10}), 20000)
        self.assertIsNone(est({"duration": 10}))
        self.assertIsNone(est({"requested_formats": [{"filesize": 100}, {}], "duration": 10}))
        self.assertIsNone(est(None))

    def test_options_include_concurrency(self):
        FakeYDL.script = [lambda y: writes_file(y)]
        self.run_dl()
        self.assertEqual(FakeYDL.instances[0].opts["concurrent_fragment_downloads"], 4)

    def test_hook_flags_approximate_total(self):
        prog = [{"status": "downloading", "downloaded_bytes": 1, "total_bytes": 10},
                {"status": "downloading", "downloaded_bytes": 2, "total_bytes_estimate": 10.0}]
        FakeYDL.script = [lambda y: writes_file(y, progress=prog)]
        self.run_dl()
        self.assertEqual([p["total_approx"] for p in self.progress], [False, True])


class RetryTest(Base):
    def expired(self, code=403):
        return yt_dlp.utils.DownloadError(f"ERROR: unable to download video data: HTTP Error {code}: Forbidden")

    def test_retry_once_on_expired_then_ok(self):
        for code in (403, 404, 410):
            FakeYDL.instances = []
            FakeYDL.script = [self.expired(code), lambda y: writes_file(y)]
            resolved = []

            def fake_resolve(t, h):
                resolved.append(t)
                return f"{t}?sig={len(resolved)}", h, None
            with mock.patch.object(QD, "_resolve", fake_resolve):
                r = self.run_dl()
            self.assertEqual(len(FakeYDL.instances), 2)
            self.assertEqual(len(resolved), 2, "re-resolved for the retry")
            self.assertTrue(r.path.endswith("source.mkv"))

    def test_second_expiry_fails(self):
        FakeYDL.script = [self.expired(), self.expired(410)]
        with self.assertRaises(QD.DownloadFailed) as cm:
            self.run_dl()
        self.assertIn("410", str(cm.exception))
        self.assertEqual(len(FakeYDL.instances), 2)

    def test_underlying_http_error(self):
        class HTTPErr(Exception):
            status = 403
        inner = HTTPErr("denied")
        outer = yt_dlp.utils.DownloadError("ERROR: fetch failed", exc_info=(HTTPErr, inner, None))
        FakeYDL.script = [outer, lambda y: writes_file(y)]
        self.assertTrue(self.run_dl().path)
        self.assertEqual(len(FakeYDL.instances), 2)

    def test_other_error_not_retried(self):
        FakeYDL.script = [yt_dlp.utils.DownloadError("ERROR: HTTP Error 500: Server Error")]
        with self.assertRaises(QD.DownloadFailed):
            self.run_dl()
        self.assertEqual(len(FakeYDL.instances), 1)


class HLSProgressiveDownloadTest(Base):
    def setUp(self):
        super().setUp()
        self.release_second = threading.Event()
        self.second_started = threading.Event()
        self.counts = {"/source.m3u8": 0, "/s0.ts": 0, "/s1.ts": 0}
        outer = self
        media = (b"#EXTM3U\n#EXT-X-PLAYLIST-TYPE:VOD\n"
                 b"#EXTINF:1.25,\ns0.ts\n#EXTINF:2.75,\ns1.ts\n#EXT-X-ENDLIST\n")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                outer.counts[self.path] = outer.counts.get(self.path, 0) + 1
                if self.path == "/source.m3u8":
                    body = media
                elif self.path == "/s0.ts":
                    body = b"first complete TS"
                elif self.path == "/s1.ts":
                    outer.second_started.set()
                    if not outer.release_second.wait(5):
                        self.send_error(504)
                        return
                    body = b"second complete TS"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self._stop_httpd)

    def _stop_httpd(self):
        self.release_second.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2)
        hls_cache.close_session(self.dest / "hls-cache" / hls_cache.PLAYLIST_NAME)

    def test_queue_downloader_owns_one_upstream_read_for_shared_waiting_readers(self):
        base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        FakeYDL.script = [lambda y: {
            "title": "selected title", "url": base + "/source.m3u8", "protocol": "m3u8_native",
            "ext": "ts", "vcodec": "h264", "acodec": "aac",
        }]
        probe = {
            "format": {"format_name": "mpegts", "duration": "1.0", "start_time": "1.4"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "1", "nb_frames": "10",
                 "start_pts": 127890, "time_base": "1/90000", "start_time": "1.421"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
        ready = threading.Event()
        handles = []
        results, errors = [], []

        def on_ready(handle):
            handles.append(handle)
            ready.set()

        def do_download():
            try:
                results.append(QD.download(
                    "https://example.com/video", {}, None, self.dest, self.progress.append, self.cancel,
                    on_source_ready=on_ready,
                ))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(QD, "_resolve", lambda target, headers: (base + "/source.m3u8", headers, "Resolved")), \
                mock.patch.object(QD, "_probe_ts", lambda path: probe):
            worker = threading.Thread(target=do_download)
            worker.start()
            self.assertTrue(ready.wait(3), "first complete segment should publish metadata")
            self.assertTrue(self.second_started.wait(3), "downloader should begin the next segment")
            handle = handles[0]
            with urlopen(handle.runtime_url, timeout=3) as response:
                playlist = response.read().decode("utf-8")
            self.assertIn("#EXT-X-ENDLIST", playlist)
            self.assertIn("#EXTINF:1.250000,", playlist)
            self.assertIn("#EXTINF:2.750000,", playlist)
            self.assertNotIn("http://", playlist)

            local_segment = handle.runtime_url.rsplit("/", 1)[0] + "/segment-000001.ts"
            outputs, read_errors = [], []

            def read_local():
                try:
                    with urlopen(local_segment, timeout=5) as response:
                        outputs.append(response.read())
                except Exception as exc:
                    read_errors.append(exc)

            readers = [threading.Thread(target=read_local) for _ in range(2)]
            for reader in readers:
                reader.start()
            time.sleep(0.15)
            self.assertTrue(all(reader.is_alive() for reader in readers))
            self.assertEqual(self.counts["/s1.ts"], 1)
            self.release_second.set()
            for reader in readers:
                reader.join(timeout=4)
            worker.join(timeout=4)

        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertFalse(read_errors)
        self.assertEqual(outputs, [b"second complete TS", b"second complete TS"])
        self.assertEqual(self.counts, {"/source.m3u8": 1, "/s0.ts": 1, "/s1.ts": 1})
        self.assertEqual(len(handles), 1)
        self.assertEqual((results[0].path, results[0].title, results[0].duration),
                         (handles[0].playlist_path, "Resolved", 4.0))
        meta = handles[0].metadata
        self.assertEqual(meta["format"]["duration"], "4.000000")
        video = next(s for s in meta["streams"] if s["codec_type"] == "video")
        self.assertEqual(video["duration"], "4.000000")
        self.assertNotIn("nb_frames", video)
        self.assertEqual((video["start_pts"], video["time_base"], video["start_time"]),
                         (127890, "1/90000", "1.421"))

    def test_malformed_hls_uses_existing_complete_download_fallback(self):
        base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        # The HTTP fixture responds with a media playlist, but selected format metadata says a non-HLS mux.
        FakeYDL.script = [
            lambda y: {"title": "selected", "url": base + "/s0.ts", "protocol": "https",
                       "ext": "mp4", "vcodec": "h264", "acodec": "aac"},
            lambda y: writes_file(y),
        ]
        callbacks = []
        with mock.patch.object(QD, "_resolve", lambda target, headers: (target, headers, None)):
            result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                                 self.cancel, on_source_ready=callbacks.append)
        self.assertTrue(result.path.endswith("source.mkv"))
        self.assertEqual(callbacks, [])
        self.assertEqual(len(FakeYDL.instances), 2)


def _ffmpeg_ok():
    import shutil
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _encrypt_aes128(plain: bytes, key: bytes, iv: bytes) -> bytes:
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    padder = padding.PKCS7(128).padder()
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return enc.update(padder.update(plain) + padder.finalize()) + enc.finalize()


def _ffprobe_stream(path_or_url: str) -> tuple[int, float]:
    import json as _json
    import subprocess
    p = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames:format=duration", "-of", "json", path_or_url],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    data = _json.loads(p.stdout)
    return int(data["streams"][0]["nb_read_frames"]), float(data["format"]["duration"])


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg/ffprobe not available")
class HLSEncryptedDownloadTest(Base):
    """AES-128 HLS served from loopback with a required header on playlist, key and segment requests.
    The cache must hold decrypted TS, expose a key-free playlist and decode like the unencrypted source."""
    HEADERS = {"X-Test-Token": "ok"}
    KEY_A = bytes(range(16))
    KEY_B = bytes(range(100, 116))

    @classmethod
    def setUpClass(cls):
        import subprocess
        cls.root = tempfile.TemporaryDirectory()
        r = Path(cls.root.name)
        # Reference: unencrypted HLS from the same encode; ffmpeg's own AES-128 output (explicit IV).
        base = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=4",
                "-f", "lavfi", "-i", "sine=duration=4", "-c:v", "libx264", "-g", "10", "-c:a", "aac",
                "-shortest", "-hls_time", "1", "-hls_playlist_type", "vod"]
        (r / "plain").mkdir()
        subprocess.run(base + ["-hls_segment_filename", str(r / "plain" / "s%d.ts"), str(r / "plain" / "p.m3u8")],
                       check=True, timeout=120)
        cls.plain_segments = [(r / "plain" / f"s{i}.ts").read_bytes() for i in range(4)]
        cls.reference = _ffprobe_stream(str(r / "plain" / "p.m3u8"))
        (r / "ff").mkdir()
        (r / "ff" / "key.bin").write_bytes(cls.KEY_A)
        iv = (0x1234).to_bytes(16, "big").hex()
        (r / "ff" / "ki.txt").write_text(f"http://127.0.0.1:1/key.bin\n{r / 'ff' / 'key.bin'}\n{iv}\n")
        subprocess.run(base + ["-hls_key_info_file", str(r / "ff" / "ki.txt"),
                               "-hls_segment_filename", str(r / "ff" / "s%d.ts"), str(r / "ff" / "p.m3u8")],
                       check=True, timeout=120)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def setUp(self):
        super().setUp()
        self.files = {}
        self.counts = {}
        self.rejected = []
        self.fail_paths = set()
        self.gates = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                if self.headers.get("X-Test-Token") != "ok":
                    outer.rejected.append(self.path)
                    self.send_error(403)
                    return
                outer.counts[self.path] = outer.counts.get(self.path, 0) + 1
                gate = outer.gates.get(self.path)
                if gate is not None and not gate.wait(10):
                    self.send_error(504)
                    return
                if self.path in outer.fail_paths or self.path not in outer.files:
                    self.send_error(500 if self.path in outer.fail_paths else 404)
                    return
                body = outer.files[self.path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self._stop_httpd)

    def _stop_httpd(self):
        for gate in self.gates.values():
            gate.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2)
        hls_cache.close_session(self.dest / "hls-cache" / hls_cache.PLAYLIST_NAME)

    # fixtures -----------------------------------------------------------------------------------------
    def publish_ffmpeg(self):
        root = Path(self.root.name) / "ff"
        text = (root / "p.m3u8").read_text().replace("http://127.0.0.1:1/key.bin", f"{self.base}/key.bin")
        self.files["/source.m3u8"] = text.encode()
        self.files["/key.bin"] = self.KEY_A
        for i in range(4):
            self.files[f"/s{i}.ts"] = (root / f"s{i}.ts").read_bytes()
        return text

    def publish_python(self, *, media_sequence=0, keys=None, ivs=None):
        """Encrypt the plain segments ourselves; keys[i]/ivs[i] choose per-segment key file and explicit IV."""
        keys = keys or [("k.bin", self.KEY_A)] * 4
        ivs = ivs or [None] * 4
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:VOD", f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}"]
        last = None
        for i, plain in enumerate(self.plain_segments):
            name, key = keys[i]
            self.files["/" + name] = key
            if (name, ivs[i]) != last:
                iv_attr = f",IV=0x{ivs[i]:032x}" if ivs[i] is not None else ""
                lines.append(f'#EXT-X-KEY:METHOD=AES-128,URI="{name}"{iv_attr}')
                last = (name, ivs[i])
            iv = (ivs[i] if ivs[i] is not None else media_sequence + i).to_bytes(16, "big")
            self.files[f"/s{i}.ts"] = _encrypt_aes128(plain, key, iv)
            lines += ["#EXTINF:1.000000,", f"s{i}.ts"]
        lines.append("#EXT-X-ENDLIST")
        self.files["/source.m3u8"] = ("\n".join(lines) + "\n").encode()

    # runner -------------------------------------------------------------------------------------------------
    def selected_info(self):
        return lambda y: {"title": "enc", "url": self.base + "/source.m3u8", "protocol": "m3u8_native",
                          "ext": "ts", "vcodec": "h264", "acodec": "aac"}

    def run_progressive(self, *, gate_last=True):
        """Download in a thread; return (result, handle, seen_before_last). The last segment is held until the
        source-ready callback has fired and the local playlist was fetched, proving overlap."""
        FakeYDL.script = [self.selected_info()]
        if gate_last:
            self.gates["/s3.ts"] = threading.Event()
        ready, handles, results, errors = threading.Event(), [], [], []

        def on_ready(handle):
            handles.append(handle)
            ready.set()

        def work():
            try:
                results.append(QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                                           self.cancel, on_source_ready=on_ready))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(QD, "_resolve",
                               lambda target, headers: (self.base + "/source.m3u8", dict(self.HEADERS), "Resolved")):
            worker = threading.Thread(target=work)
            worker.start()
            self.assertTrue(ready.wait(30), f"source-ready did not fire (errors={errors})")
            if gate_last:
                self.assertFalse(results, "download must still be running while the last segment is held")
                with urlopen(handles[0].runtime_url, timeout=5) as response:
                    self.assertNotIn("EXT-X-KEY", response.read().decode())
                self.gates["/s3.ts"].set()
            worker.join(60)
        self.assertFalse(worker.is_alive())
        return results, errors, handles

    def assert_decrypted_cache(self, result, handles):
        cache = self.dest / "hls-cache"
        local = (cache / hls_cache.PLAYLIST_NAME).read_text()
        self.assertNotIn("EXT-X-KEY", local)
        self.assertNotIn("key", local.lower())
        for i in range(4):
            data = (cache / f"segment-{i:06d}.ts").read_bytes()
            self.assertEqual(data[0], 0x47)
            self.assertEqual(len(data) % 188, 0)
        frames, duration = _ffprobe_stream(str(cache / hls_cache.PLAYLIST_NAME))
        self.assertEqual(frames, self.reference[0])
        self.assertAlmostEqual(duration, self.reference[1], delta=0.05)
        self.assertEqual(result.path, handles[0].playlist_path)
        self.assertEqual(self.rejected, [], "every playlist/key/segment request must carry the header")

    # tests --------------------------------------------------------------------------------------------------
    def test_ffmpeg_aes128_explicit_iv_decrypts_and_overlaps(self):
        text = self.publish_ffmpeg()
        self.assertIn("IV=0x", text)
        results, errors, handles = self.run_progressive()
        self.assertEqual(errors, [])
        self.assert_decrypted_cache(results[0], handles)
        self.assertEqual(self.counts["/key.bin"], 1, "one key fetch per session")
        self.assertEqual(self.counts["/source.m3u8"], 1)

    def test_no_iv_attribute_uses_media_sequence_number(self):
        self.publish_python(media_sequence=7)
        self.assertNotIn(b"IV=", self.files["/source.m3u8"])
        results, errors, handles = self.run_progressive()
        self.assertEqual(errors, [])
        self.assert_decrypted_cache(results[0], handles)
        for i in range(4):
            self.assertEqual((self.dest / "hls-cache" / f"segment-{i:06d}.ts").read_bytes(), self.plain_segments[i])
        self.assertEqual(self.counts["/k.bin"], 1)

    def test_rotating_keys_and_ivs_fetch_each_distinct_key_once(self):
        keys = [("a.bin", self.KEY_A), ("b.bin", self.KEY_B), ("a.bin", self.KEY_A), ("b.bin", self.KEY_B)]
        self.publish_python(media_sequence=3, keys=keys, ivs=[None, 0x99, None, 0xFF])
        results, errors, handles = self.run_progressive()
        self.assertEqual(errors, [])
        self.assert_decrypted_cache(results[0], handles)
        for i in range(4):
            self.assertEqual((self.dest / "hls-cache" / f"segment-{i:06d}.ts").read_bytes(), self.plain_segments[i])
        self.assertEqual((self.counts["/a.bin"], self.counts["/b.bin"]), (1, 1))

    def test_resume_after_interruption_does_not_redownload_or_double_decrypt(self):
        self.publish_python(media_sequence=0)
        self.fail_paths.add("/s2.ts")
        FakeYDL.script = [self.selected_info(), self.selected_info()]
        with mock.patch.object(QD, "_resolve",
                               lambda target, headers: (self.base + "/source.m3u8", dict(self.HEADERS), "Resolved")):
            with self.assertRaises(QD.DownloadFailed):
                QD.download("https://example.com/video", {}, None, self.dest, self.progress.append, self.cancel,
                            on_source_ready=lambda h: None)
            cache = self.dest / "hls-cache"
            first = [(cache / f"segment-{i:06d}.ts").read_bytes() for i in range(2)]
            self.assertEqual(first, self.plain_segments[:2])
            hls_cache.close_session(cache / hls_cache.PLAYLIST_NAME)
            identity = hls_cache._load_sidecar(hls_cache.sidecar_path(cache / hls_cache.PLAYLIST_NAME))["identity"]
            self.fail_paths.clear()
            handles = []
            result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append, self.cancel,
                                 on_source_ready=handles.append)
        self.assertEqual((self.counts["/s0.ts"], self.counts["/s1.ts"]), (1, 1))
        self.assertEqual(self.counts["/s2.ts"], 2)
        self.assertEqual(self.counts["/key.bin" if "/key.bin" in self.counts else "/k.bin"], 2)  # one per session
        for i in range(4):
            self.assertEqual((cache / f"segment-{i:06d}.ts").read_bytes(), self.plain_segments[i])
        self.assertEqual(handles[0].identity, identity)
        self.assertEqual(result.path, handles[0].playlist_path)
        self.assertEqual(_ffprobe_stream(str(cache / hls_cache.PLAYLIST_NAME))[0], self.reference[0])

    def test_sample_aes_falls_back_to_full_download(self):
        self.publish_python()
        self.files["/source.m3u8"] = self.files["/source.m3u8"].replace(b"METHOD=AES-128", b"METHOD=SAMPLE-AES")
        FakeYDL.script = [self.selected_info(), lambda y: writes_file(y)]
        callbacks = []
        with mock.patch.object(QD, "_resolve",
                               lambda target, headers: (self.base + "/source.m3u8", dict(self.HEADERS), None)):
            result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                                 self.cancel, on_source_ready=callbacks.append)
        self.assertTrue(result.path.endswith("source.mkv"))
        self.assertEqual(callbacks, [])
        self.assertNotIn("/k.bin", self.counts)

    def test_wrong_key_falls_back_to_full_download_without_caching(self):
        self.publish_python()
        self.files["/k.bin"] = self.KEY_B
        FakeYDL.script = [self.selected_info(), lambda y: writes_file(y)]
        callbacks = []
        with mock.patch.object(QD, "_resolve",
                               lambda target, headers: (self.base + "/source.m3u8", dict(self.HEADERS), None)):
            result = QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                                 self.cancel, on_source_ready=callbacks.append)
        self.assertTrue(result.path.endswith("source.mkv"))
        self.assertEqual(callbacks, [])
        self.assertFalse(list((self.dest / "hls-cache").glob("segment-*.ts")))
        self.assertFalse(list((self.dest / "hls-cache").glob("*.enc")))

    def test_key_rotated_to_wrong_key_midstream_fails_cleanly(self):
        self.publish_python(keys=[("a.bin", self.KEY_A), ("a.bin", self.KEY_A), ("b.bin", self.KEY_B), ("b.bin", self.KEY_B)])
        self.files["/b.bin"] = bytes(16)
        FakeYDL.script = [self.selected_info()]
        callbacks = []
        with mock.patch.object(QD, "_resolve",
                               lambda target, headers: (self.base + "/source.m3u8", dict(self.HEADERS), None)):
            with self.assertRaises(QD.DownloadFailed):
                QD.download("https://example.com/video", {}, None, self.dest, self.progress.append,
                            self.cancel, on_source_ready=callbacks.append)
        self.assertEqual(len(callbacks), 1)
        self.assertFalse(list((self.dest / "hls-cache").glob("*.enc")))
        self.assertFalse((self.dest / "hls-cache" / "segment-000002.ts").exists())


if __name__ == "__main__":
    unittest.main()
