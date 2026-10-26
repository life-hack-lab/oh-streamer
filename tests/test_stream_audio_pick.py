"""T2.3 task C: choosing the subtitle audio source, the start-time probe, the url refresher and attach().

ffprobe/ffmpeg run against a loopback HTTP server that demands a custom header; yt-dlp and the network are faked.
Run from streamer/: PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_audio_pick."""
from __future__ import annotations

import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from oh_streamer.player.open_input import OpenInput
from oh_streamer.queue import queue_store, stream_audio as sa
from oh_streamer.sources import media_input as mi
from oh_streamer.sources import qualities, source_info

HEADER = ("X-Test", "hello")
TARGET = "https://site.test/watch/42"


# ----------------------------------------------------------------------------------------------------------
# fake yt-dlp results
# ----------------------------------------------------------------------------------------------------------

def vfmt(fid, height, url, **kw):
    return {"format_id": fid, "url": url, "height": height, "width": height * 16 // 9, "fps": 30, "tbr": height,
            "vcodec": "avc1.64001f", "acodec": "none", "protocol": "m3u8_native", **kw}


def mfmt(fid, height, url, **kw):          # muxed
    return {**vfmt(fid, height, url), "acodec": "mp4a.40.2", **kw}


def afmt(fid, url, **kw):
    return {"format_id": fid, "url": url, "vcodec": "none", "acodec": "mp4a.40.2", "abr": 128, "tbr": 128,
            "protocol": "m3u8_native", **kw}


def resolve_info(formats, headers=None, info_headers=None):
    info = {"title": "t", "duration": 100.0, "formats": formats, "http_headers": info_headers or {}}
    quals = qualities.build_quality_list(info)

    def ytdlp(url, hdrs, with_info=False):
        return quals, "t", None, info

    def probe(src, hdrs):
        return {"height": 360, "fps": 30.0, "duration": 100.0, "vcodec": "h264", "acodec": "aac"}

    return source_info.resolve(OpenInput(TARGET, dict(headers or {})), ytdlp=ytdlp, find_resolver=lambda u: None,
                               probe=probe)


class PickTest(unittest.TestCase):
    def test_hls_master_with_audio_group_is_audio_only(self):
        r = resolve_info([vfmt("v720", 720, "https://cdn.test/720.m3u8"), vfmt("v240", 240, "https://cdn.test/240.m3u8"),
                          afmt("a1", "https://cdn.test/audio.m3u8", http_headers={"Referer": "https://site.test/"})],
                         headers={"Cookie": "c=1"})
        self.assertEqual(r.audio_format["id"], "a1")
        self.assertEqual(r.audio_format["headers"], {"Referer": "https://site.test/", "Cookie": "c=1"})
        pick = sa.pick_subtitle_audio(r)
        self.assertEqual((pick["kind"], pick["url"], pick["hls"]), ("audio_only", "https://cdn.test/audio.m3u8", True))
        self.assertEqual(pick["headers"]["Cookie"], "c=1")

    def test_muxed_plus_audio_only_prefers_audio_only(self):
        r = resolve_info([mfmt("m720", 720, "https://cdn.test/720.m3u8"), mfmt("m240", 240, "https://cdn.test/240.m3u8"),
                          afmt("a1", "https://cdn.test/a.m4a", protocol="https", ext="m4a")])
        pick = sa.pick_subtitle_audio(r)
        self.assertEqual((pick["kind"], pick["url"], pick["hls"]), ("audio_only", "https://cdn.test/a.m4a", False))

    def test_dash_only_audio_is_skipped(self):
        r = resolve_info([mfmt("m720", 720, "https://cdn.test/720.m3u8"), mfmt("m240", 240, "https://cdn.test/240.m3u8"),
                          afmt("dash", "https://cdn.test/dash.mpd", protocol="http_dash_segments")])
        self.assertIsNone(r.audio_format)
        self.assertEqual(sa.pick_subtitle_audio(r)["kind"], "lowest")

    def test_dash_audio_does_not_hide_a_usable_one(self):
        r = resolve_info([mfmt("m720", 720, "https://cdn.test/720.m3u8"),
                          afmt("dash", "https://cdn.test/dash.mpd", protocol="http_dash_segments", abr=320),
                          afmt("a1", "https://cdn.test/a.m3u8", abr=64)])
        self.assertEqual(r.audio_format["id"], "a1")

    def test_several_muxed_renditions_use_the_lowest(self):
        r = resolve_info([mfmt("m1080", 1080, "https://cdn.test/1080.m3u8", http_headers={"Referer": "R"}),
                          mfmt("m480", 480, "https://cdn.test/480.m3u8", http_headers={"Referer": "R"}),
                          mfmt("m240", 240, "https://cdn.test/240.m3u8", http_headers={"Referer": "R"})],
                         headers={"X-User": "u"})
        pick = sa.pick_subtitle_audio(r)
        self.assertEqual((pick["kind"], pick["url"]), ("lowest", "https://cdn.test/240.m3u8"))
        self.assertEqual(pick["headers"], {"Referer": "R", "X-User": "u"})
        self.assertEqual(pick["rendition"], {"id": "m240", "height": 240, "label": "240p"})

    def test_single_rendition_is_same(self):
        r = resolve_info([mfmt("m720", 720, "https://cdn.test/720.m3u8")])
        pick = sa.pick_subtitle_audio(r)
        self.assertEqual((pick["kind"], pick["url"], pick["rendition"]["id"]), ("same", "https://cdn.test/720.m3u8", "m720"))

    def test_direct_media_url_uses_media_src(self):
        target = "https://cdn.test/video.mp4"
        r = source_info.resolve(OpenInput(target, {"Referer": "R"}), find_resolver=lambda u: None,
                                probe=lambda s, h: {"height": 360, "fps": 30.0, "duration": 5.0, "vcodec": "h264",
                                                    "acodec": "aac"})
        self.assertIsNone(r.quals)
        self.assertEqual(r.media_src, (target, {"Referer": "R"}))
        pick = sa.pick_subtitle_audio(r)
        self.assertEqual((pick["kind"], pick["url"], pick["headers"], pick["rendition"], pick["hls"]),
                         ("same", target, {"Referer": "R"}, None, False))

    def test_local_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            path.write_bytes(b"x")
            r = source_info.resolve(OpenInput(str(path), {}, kind="file"),
                                    probe=lambda s, h: {"height": 360, "fps": 30.0, "duration": 5.0,
                                                        "vcodec": "h264", "acodec": "aac"})
            pick = sa.pick_subtitle_audio(r)
            self.assertEqual((pick["kind"], pick["url"], pick["headers"]), ("file", str(path.resolve()), {}))

    def test_prefer_matches_id_then_height(self):
        r = resolve_info([mfmt("m1080", 1080, "https://cdn.test/1080.m3u8"), mfmt("m480", 480, "https://cdn.test/480.m3u8"),
                          mfmt("m240", 240, "https://cdn.test/240.m3u8")])
        by_id = sa.pick_subtitle_audio(r, prefer={"id": "m480", "height": 999})
        self.assertEqual(by_id["rendition"]["id"], "m480")
        by_height = sa.pick_subtitle_audio(r, prefer={"id": "gone", "height": 1080})
        self.assertEqual(by_height["rendition"]["id"], "m1080")
        none = sa.pick_subtitle_audio(r, prefer={"id": "gone", "height": 720})
        self.assertEqual(none["rendition"]["id"], "m240")             # normal rule; the caller notices the mismatch
        self.assertFalse(sa._same_rendition({"id": "gone", "height": 720}, none["rendition"]))


# ----------------------------------------------------------------------------------------------------------
# loopback server + generated audio
# ----------------------------------------------------------------------------------------------------------

def make_ts(path: Path, offset: float):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
           "-c:a", "aac", "-output_ts_offset", str(offset), "-f", "mpegts", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


class Server:
    """Serves files under root; 403 without the X-Test header, /gone/* -> 410 (after the header check), /garbage.ts
    is not media."""

    def __init__(self, root: Path):
        owner = self
        self.requests: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                return

            def do_GET(self):
                owner.requests.append(self.path)
                if self.headers.get(HEADER[0]) != HEADER[1]:
                    return self._reply(403, b"forbidden")
                if self.path.startswith("/gone/"):
                    return self._reply(410, b"gone")
                if self.path == "/garbage.ts":
                    return self._reply(200, b"this is not media" * 100)
                f = root / self.path.lstrip("/")
                if f.is_file():
                    return self._reply(200, f.read_bytes())
                self._reply(404, b"missing")

            do_HEAD = do_GET

            def _reply(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


class FakeService:
    """The three things attach() may use: store, _lock, _subtitle_update."""

    def __init__(self, root: Path):
        self.store = queue_store.QueueStore(root)
        self._lock = threading.RLock()

    def _subtitle_update(self, item_id, **fields):
        with self._lock:
            it = self.store.get(item_id)
            if it is not None:
                self.store.update(item_id, subtitle={**it.subtitle, **fields})


def qual(fid, height, url):
    return {"id": fid, "height": height, "width": height * 16 // 9, "fps": 30.0, "tbr": height, "url": url,
            "audio_url": None, "http_headers": {}, "label": f"{height}p"}


def resolved(quals, headers=None, kind="url", media_src=None):
    return source_info.Resolved(TARGET, kind, dict(headers or {}), "t", "site.test", duration=4.0, quals=quals,
                                media_src=media_src)


class LoopbackCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._media = tempfile.TemporaryDirectory()
        cls.media_dir = Path(cls._media.name)
        make_ts(cls.media_dir / "hi.ts", 0.0)
        make_ts(cls.media_dir / "lo.ts", 1.0)           # this rendition starts 1.0 s later
        (cls.media_dir / "gone").mkdir()

    @classmethod
    def tearDownClass(cls):
        cls._media.cleanup()

    def setUp(self):
        self.server = Server(self.media_dir)
        self.addCleanup(self.server.close)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.service = FakeService(Path(self._tmp.name))
        self.logs: list[str] = []
        patcher = mock.patch.object(sa, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.headers = {HEADER[0]: HEADER[1]}

    def url(self, name):
        return f"{self.server.base}/{name}"

    def new_item(self):
        item = self.service.store.add_stream(TARGET, "url", self.headers, None, title="t")
        marker = self.service.store.item_dir(item.id) / sa.MARKER_NAME
        self.addCleanup(mi.unregister_refresher, marker)
        return item

    def two_renditions(self, lo="lo.ts"):
        return resolved([qual("h", 720, self.url("hi.ts")), qual("l", 240, self.url(lo))], self.headers)

    def run_attach(self, r, played_height=720):
        item = self.new_item()
        pick = sa.pick_subtitle_audio(r)
        thread = sa.attach(self.service, item.id, r, pick, played_height)
        thread.join(60)
        self.assertFalse(thread.is_alive())
        return self.service.store.get(item.id)

    def assertNoSecrets(self, *texts):
        for text in texts:
            self.assertNotIn("127.0.0.1", text)
            self.assertNotIn(HEADER[1], text)


class ProbeTest(LoopbackCase):
    def test_start_time_needs_header(self):
        low = sa.probe_audio_start(self.url("lo.ts"), self.headers)
        high = sa.probe_audio_start(self.url("hi.ts"), self.headers)
        self.assertAlmostEqual(low - high, 1.0, delta=0.05)
        with self.assertRaises(mi.RemoteExpired):                  # 403 without the header
            sa.probe_audio_start(self.url("hi.ts"), {})
        with self.assertRaises(mi.RemoteExpired):
            sa.probe_audio_start(self.url("gone/lo.ts"), self.headers)
        with self.assertRaises(RuntimeError) as ctx:
            sa.probe_audio_start(self.url("garbage.ts"), self.headers)
        self.assertNoSecrets(str(ctx.exception))

    def test_local_file(self):
        self.assertAlmostEqual(sa.probe_audio_start(str(self.media_dir / "lo.ts")),
                               sa.probe_audio_start(self.url("lo.ts"), self.headers), delta=0.01)


class AttachTest(LoopbackCase):
    def test_lowest_rendition_offset_follows_play_start(self):
        item = self.run_attach(self.two_renditions())
        marker = self.service.store.item_dir(item.id) / sa.MARKER_NAME
        self.assertEqual(Path(item.source_path), marker)
        s = item.stream
        self.assertEqual((s["audio_kind"], s["audio_status"], s["audio_error"], s["audio_marker"], s["audio_generation"]),
                         ("lowest", "ready", "", str(marker), 0))
        self.assertEqual(s["audio_rendition"], {"id": "l", "height": 240, "label": "240p"})
        self.assertEqual(s["source_kind"], "url")                    # earlier fields survive the merge
        self.assertAlmostEqual(s["audio_start"] - s["play_start"], 1.0, delta=0.05)
        self.assertEqual(s["audio_offset"], s["play_start"])
        self.assertEqual(item.subtitle["audio_offset"], s["play_start"])
        data = mi.read_remote_marker(marker)
        self.assertEqual((data["kind"], data["generation"], data["url"], data["headers"]),
                         ("lowest", 0, self.url("lo.ts"), self.headers))
        self.assertTrue(data["identity"])
        diff_lines = [m for m in self.logs if "diff=" in m]
        self.assertEqual(len(diff_lines), 1)
        self.assertRegex(diff_lines[0], r"diff=-(0\.9|1\.0)")        # play rendition starts earlier than the audio
        self.assertNoSecrets(*self.logs)

    def test_same_rendition_probes_once_and_logs_nothing(self):
        r = resolved([qual("h", 720, self.url("hi.ts"))], self.headers)
        item = self.run_attach(r)
        s = item.stream
        self.assertEqual((s["audio_kind"], s["audio_status"]), ("same", "ready"))
        self.assertEqual(s["play_start"], s["audio_start"])
        self.assertEqual(s["audio_offset"], s["audio_start"])
        self.assertFalse([m for m in self.logs if "diff=" in m])

    def test_playing_the_lowest_needs_no_second_probe(self):
        item = self.run_attach(self.two_renditions(), played_height=240)
        self.assertEqual(item.stream["play_start"], item.stream["audio_start"])

    def test_local_file_has_no_marker(self):
        path = str((self.media_dir / "lo.ts").resolve())
        r = resolved(None, kind="file", media_src=(path, {}))
        item = self.run_attach(r)
        s = item.stream
        self.assertEqual((item.source_path, s["audio_kind"], s["audio_marker"], s["audio_status"], s["audio_rendition"]),
                         (path, "file", "", "ready", None))
        self.assertEqual(s["audio_offset"], s["audio_start"])
        self.assertEqual(item.subtitle["audio_offset"], s["audio_offset"])
        self.assertAlmostEqual(s["audio_offset"], sa.probe_audio_start(path), delta=0.001)
        self.assertEqual([p.name for p in self.service.store.item_dir(item.id).iterdir()], [])

    def test_unreadable_audio_fails_with_unsupported_message(self):
        r = resolved([qual("h", 720, self.url("hi.ts")), qual("l", 240, self.url("garbage.ts"))], self.headers)
        item = self.run_attach(r)
        self.assertEqual((item.stream["audio_status"], item.stream["audio_error"]), ("failed", sa.ERR_UNSUPPORTED))
        self.assertNotIn("audio_offset", item.subtitle)
        self.assertNoSecrets(item.stream["audio_error"], *self.logs)

    def test_missing_item_stops_quietly(self):
        r = self.two_renditions()
        with self.assertRaises(KeyError):
            sa.attach(self.service, "nope", r, sa.pick_subtitle_audio(r), 720)


class RefreshTest(LoopbackCase):
    def fresh(self, lo="lo.ts", lo_id="l", lo_height=240):
        return resolved([qual("h", 720, self.url("hi.ts")), qual(lo_id, lo_height, self.url(lo))], self.headers)

    def test_expired_url_is_replaced_once_and_the_probe_retries(self):
        expired = self.two_renditions(lo="gone/lo.ts")
        with mock.patch.object(source_info, "resolve", return_value=self.fresh()) as resolve:
            item = self.run_attach(expired)
        self.assertEqual(resolve.call_count, 1)
        called = resolve.call_args[0][0]
        self.assertEqual((called.target, called.headers), (TARGET, self.headers))
        marker = self.service.store.item_dir(item.id) / sa.MARKER_NAME
        data = mi.read_remote_marker(marker)
        self.assertEqual((data["generation"], data["url"]), (1, self.url("lo.ts")))
        self.assertEqual(data["identity"], sa._identity(expired, sa.pick_subtitle_audio(expired)))
        self.assertEqual((item.stream["audio_status"], item.stream["audio_generation"]), ("ready", 1))
        self.assertAlmostEqual(item.stream["audio_start"] - item.stream["play_start"], 1.0, delta=0.05)
        self.assertNoSecrets(*self.logs)

    def test_rendition_not_offered_any_more_fails(self):
        expired = self.two_renditions(lo="gone/lo.ts")
        with mock.patch.object(source_info, "resolve", return_value=self.fresh(lo_id="x", lo_height=360)):
            item = self.run_attach(expired)
        self.assertEqual((item.stream["audio_status"], item.stream["audio_error"]), ("failed", sa.ERR_EXPIRED))
        marker = self.service.store.item_dir(item.id) / sa.MARKER_NAME
        self.assertEqual(mi.read_remote_marker(marker)["generation"], 0)      # not switched to another source
        self.assertNoSecrets(*self.logs)

    def test_same_height_with_new_id_is_accepted(self):
        expired = self.two_renditions(lo="gone/lo.ts")
        with mock.patch.object(source_info, "resolve", return_value=self.fresh(lo_id="renamed")):
            item = self.run_attach(expired)
        self.assertEqual(item.stream["audio_status"], "ready")

    def test_resolve_error_and_same_url_fail_without_leaking(self):
        for name, effect in (("resolve error", source_info.ResolveError("連線失敗", f"boom {self.url('x')}")),
                             ("same url", None)):
            with self.subTest(name):
                expired = self.two_renditions(lo="gone/lo.ts")
                kwargs = {"side_effect": effect} if effect else {"return_value": expired}
                with mock.patch.object(source_info, "resolve", **kwargs):
                    item = self.run_attach(expired)
                self.assertEqual((item.stream["audio_status"], item.stream["audio_error"]),
                                 ("failed", sa.ERR_EXPIRED))
                self.assertNoSecrets(*self.logs)

    def test_refresher_directly(self):
        r = self.two_renditions()
        item = self.run_attach(r)
        marker = str(self.service.store.item_dir(item.id) / sa.MARKER_NAME)
        before = mi.read_remote_marker(marker)
        fresh = resolved([qual("h", 720, self.url("hi.ts")), qual("l", 240, self.url("lo.ts") + "?sig=2")], self.headers)
        with mock.patch.object(source_info, "resolve", return_value=fresh):
            sa.make_refresher(self.service, item.id)(marker)
        after = mi.read_remote_marker(marker)
        self.assertEqual((after["generation"], after["identity"], after["kind"], after["rendition"]),
                         (1, before["identity"], "lowest", before["rendition"]))
        self.assertTrue(after["url"].endswith("?sig=2"))
        self.assertEqual(self.service.store.get(item.id).stream["audio_generation"], 1)

    def test_refresher_for_removed_item_does_not_recreate_the_folder(self):
        r = self.two_renditions()
        item = self.run_attach(r)
        marker = str(self.service.store.item_dir(item.id) / sa.MARKER_NAME)
        refresher = sa.make_refresher(self.service, item.id)
        self.service.store.remove_stream(item.id)
        with mock.patch.object(source_info, "resolve", return_value=r):
            with self.assertRaises(sa.RefreshError):
                refresher(marker)
        self.assertFalse(Path(marker).parent.exists())


if __name__ == "__main__":
    unittest.main()
