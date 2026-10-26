"""Offline tests for sources/source_info.py (intake panel: resolve one input to its facts + a thumbnail).
yt-dlp, the external resolver and the network are replaced by fakes / a fake-formats JSON; ffprobe / ffmpeg run on a
tiny generated clip. Run from streamer/: python -m unittest tests.test_source_info."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.player import open_input
from oh_streamer.queue import queue_thumbs
from oh_streamer.sources import source_info as SI
from oh_streamer.sources import source_resolve


def make_clip(path: Path, seconds=3, fps=30, size="640x360") -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}",
           "-f", "lavfi", "-i", f"sine=duration={seconds}",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def fmt(fid, height, fps, url, **kw):
    return {"format_id": fid, "url": url, "height": height, "width": height * 16 // 9, "fps": fps,
            "vcodec": "avc1.64001f", "acodec": "mp4a.40.2", "protocol": "https", "ext": "mp4", **kw}


class SourceInfoBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="srcinfo-"))
        cls.clip = cls.tmp / "clip one.mp4"
        make_clip(cls.clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fake_formats(self, **extra) -> str:
        data = {"title": "Fake page", "duration": 3.0,
                "formats": [fmt("720p60", 720, 60.0, str(self.clip), tbr=4000, filesize=1_500_000),
                            fmt("360", 360, 30.0, str(self.clip), tbr=800)], **extra}
        p = self.tmp / "fake_formats.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def ytdlp_from(self, path):
        calls = []

        def ytdlp(url, headers, with_info=False):
            calls.append((url, dict(headers), with_info))
            return source_resolve.resolve_with_ytdlp(url, headers, path, with_info=with_info)
        ytdlp.calls = calls
        return ytdlp


class ResolveTests(SourceInfoBase):
    def test_browser_fallback_keeps_media_credentials_for_sample_probe(self):
        fake = self.ytdlp_from(self.fake_formats())
        media = "https://media.example/master.m3u8"
        def ytdlp(*args, **kwargs):
            quals, title, thumb, info = fake(*args, **kwargs)
            quals = [{**q, "url": media, "http_headers": {"Cookie": "media=session"}} for q in quals]
            return quals, title, None, {**info, "duration": None, "browser_resolved": True,
                                       "browser_media_url": media}
        facts = {"height": 720, "fps": 30, "duration": 20, "vcodec": "h264", "acodec": "aac"}
        with mock.patch.object(SI, "_probe_hls_sample", return_value=facts) as sample:
            SI.resolve(open_input.OpenInput("https://page.example/watch", {"Cookie": "page=secret"}),
                       ytdlp=ytdlp, find_resolver=lambda _: None)
        self.assertEqual(sample.call_args.args[1]["Cookie"], "media=session")

    def test_redirected_sample_does_not_restore_original_credentials(self):
        from oh_streamer.queue import queue_download as QD
        playlist = b"#EXTM3U\n#EXTINF:4,\none.ts\n#EXT-X-ENDLIST\n"
        with mock.patch.object(QD, "_fetch_bytes", side_effect=[
                (playlist, "https://redirect.example/stream.m3u8"),
                (b"sample", "https://redirect.example/one.ts")]) as fetch:
            SI._probe_hls_sample("https://media.example/master.m3u8",
                {"Cookie": "original=secret", "Authorization": "token", "Referer": "page"},
                lambda *_: {"duration": 4, "fps": 30})
        self.assertEqual(fetch.call_args.args[1], {"Referer": "page"})

    def test_page_url_goes_through_ytdlp(self):
        ytdlp = self.ytdlp_from(self.fake_formats())
        oi = open_input.OpenInput("https://www.example.test/watch/1", {"Referer": "https://r/"})
        r = SI.resolve(oi, ytdlp=ytdlp, find_resolver=lambda u: None, probe=lambda *a: self.fail("no probe needed"))
        self.assertEqual(ytdlp.calls, [("https://www.example.test/watch/1", {"Referer": "https://r/"}, True)])
        self.assertEqual((r.kind, r.title, r.site, r.duration), ("url", "Fake page", "www.example.test", 3.0))
        self.assertEqual([(h["height"], h["fps"], h["label"]) for h in r.heights],
                         [(720, 60.0, "720p60"), (360, 30.0, "360p")])
        self.assertEqual([h["size_bytes"] for h in r.heights], [1_500_000, 300_000])   # filesize, else tbr * duration
        self.assertEqual((r.default_height, r.fps, r.vcodec, r.acodec, r.hls), (720, 60.0, "h264", "aac", False))
        self.assertEqual(r.fps_at(360), 30.0)
        self.assertEqual(len(r.quals), 2)
        self.assertEqual(r.headers, {"Referer": "https://r/"})
        url, headers = r.thumb_src                                   # lightest rendition, headers passed on
        self.assertEqual(url, str(self.clip))
        self.assertEqual(headers["Referer"], "https://r/")

    def test_missing_duration_is_probed(self):
        formats = self.fake_formats()
        data = json.loads(Path(formats).read_text(encoding="utf-8"))
        del data["duration"]
        Path(formats).write_text(json.dumps(data), encoding="utf-8")
        probed = []
        probe = lambda src, headers: probed.append(src) or {"height": 720, "fps": 60.0, "duration": 99.0,   # noqa: E731
                                                            "vcodec": "h264", "acodec": "aac"}
        r = SI.resolve(open_input.OpenInput("https://www.example.test/w"), ytdlp=self.ytdlp_from(formats),
                       find_resolver=lambda u: None, probe=probe)
        self.assertEqual((r.duration, probed), (99.0, [str(self.clip)]))

    def test_ytdlp_website_cover_is_kept_separate_from_frame_source(self):
        formats = self.fake_formats()
        def ytdlp(url, headers, with_info=False):
            quals, title, thumb, info = self.ytdlp_from(formats)(url, headers, with_info=True)
            info["thumbnail"] = "https://images.test/cover.jpg"
            return quals, title, thumb, info
        r = SI.resolve(open_input.OpenInput("https://www.example.test/w", {"User-Agent": "UA"}),
                       ytdlp=ytdlp, find_resolver=lambda u: None)
        self.assertEqual(r.cover_src[0], "https://images.test/cover.jpg")
        self.assertEqual(r.cover_src[1]["Referer"], "https://www.example.test/w")
        self.assertEqual(r.cover_src[1]["User-Agent"], "UA")
        self.assertIsNotNone(r.thumb_src)

    def test_external_resolver_beats_ytdlp(self):
        seen = {}
        probe = lambda src, headers: seen.update(src=src, headers=headers) or {    # noqa: E731
            "height": 1080, "fps": 29.97, "duration": 600.0, "vcodec": "hevc", "acodec": None}
        r = SI.resolve(open_input.OpenInput("https://site.test/page", {"Cookie": "a=b"}),
                       ytdlp=lambda *a, **k: self.fail("yt-dlp must not run"),
                       find_resolver=lambda u: {"match": "site.test", "command": ["x"]},
                       resolve_external=lambda url, headers, resolver: ("https://cdn.test/index.m3u8",
                                                                       {"Referer": "https://site.test/"}, "Resolved title"),
                       probe=probe)
        self.assertEqual(seen["src"], "https://cdn.test/index.m3u8")
        self.assertEqual(seen["headers"], {"Referer": "https://site.test/", "Cookie": "a=b"})   # user headers win/merge
        self.assertEqual((r.title, r.default_height, r.hls, r.vcodec, r.acodec), ("Resolved title", 1080, True, "hevc", None))
        self.assertEqual([h["label"] for h in r.heights], ["1080p"])
        self.assertIsNone(r.quals)
        self.assertEqual(r.target, "https://site.test/page")           # the page URL is what is queued / streamed

    def test_direct_media_url_is_probed(self):
        r = SI.resolve(open_input.OpenInput("https://cdn.test/dir/movie.mp4?sig=1", {"Referer": "x"}),
                       ytdlp=lambda *a, **k: self.fail("yt-dlp must not run"), find_resolver=lambda u: None,
                       probe=lambda src, headers: {"height": 720, "fps": 25.0, "duration": 60.0, "vcodec": "h264",
                                                   "acodec": "aac"})
        self.assertEqual((r.kind, r.title, r.site, r.default_height, r.fps), ("url", "movie", "cdn.test", 720, 25.0))
        self.assertIsNone(r.heights[0]["size_bytes"])

    def test_browser_master_uses_ytdlp_and_local_sample_probe(self):
        formats = self.fake_formats()
        data = json.loads(Path(formats).read_text(encoding="utf-8"))
        del data["duration"]
        Path(formats).write_text(json.dumps(data), encoding="utf-8")
        cover = "data:image/jpeg;base64,/9j/aW1hZ2U="
        facts = {"height": 720, "fps": 60.0, "duration": 3985.0, "vcodec": "h264", "acodec": "aac"}
        with mock.patch.object(SI, "_probe_hls_sample", return_value=facts) as sample:
            r = SI.resolve(open_input.OpenInput("https://video.example/watch/1"),
                find_resolver=lambda u: {"builtin": "browser"},
                resolve_external=lambda *a: ("https://media.example/example/playlist.m3u8",
                                             {"Referer": "https://video.example/watch/1"}, "Site title", {"cover": cover}),
                ytdlp=self.ytdlp_from(formats), probe=lambda *a: self.fail("no remote ffprobe"))
        sample.assert_called_once()
        self.assertEqual(r.target, "https://video.example/watch/1")
        self.assertEqual((r.title, r.site, r.duration, r.cover), ("Site title", "video.example", 3985.0, cover))
        self.assertEqual(len(r.quals), 2)

    def test_hls_sample_uses_full_playlist_duration(self):
        from oh_streamer.queue import queue_download as QD
        playlist = b"#EXTM3U\n#EXTINF:4,\none.jpeg\n#EXTINF:5,\ntwo.jpeg\n#EXT-X-ENDLIST\n"
        def probe(path, headers):
            self.assertTrue(Path(path).is_file())
            self.assertEqual(Path(path).read_bytes(), b"sample")
            self.assertEqual(headers, {})
            return {"duration": 4.0, "fps": 30.0}
        with mock.patch.object(QD, "_fetch_bytes", side_effect=[
                (playlist, "https://cdn.test/video.m3u8"), (b"sample", "https://cdn.test/one.jpeg")]) as fetch:
            facts = SI._probe_hls_sample("https://cdn.test/video.m3u8", {"Referer": "page"}, probe)
        self.assertEqual(facts["duration"], 9.0)
        self.assertEqual(fetch.call_args.args[0], "https://cdn.test/one.jpeg")

    def test_ytdlp_browser_fallback_uses_local_sample_and_cover(self):
        fake = self.ytdlp_from(self.fake_formats())
        cover = "data:image/png;base64,iVBORw0KGgo="
        def fallback(*args, **kwargs):
            quals, title, thumb, info = fake(*args, **kwargs)
            for q in quals:
                q["url"] = "https://media.example/variant.m3u8"
            for f in info["formats"]:
                f["url"] = "https://media.example/variant.m3u8"
                f["protocol"] = "m3u8_native"
            info["duration"] = None
            info["browser_resolved"] = True
            info["cover"] = cover
            return quals, title, thumb, info
        with mock.patch.object(SI, "_probe_hls_sample", return_value={
                "duration": 20.0, "fps": 30.0, "height": 720, "vcodec": "h264", "acodec": "aac"}) as sample:
            result = SI.resolve(open_input.OpenInput("https://video.example/watch/3"),
                                find_resolver=lambda _: None, ytdlp=fallback,
                                probe=lambda *_: self.fail("remote ffprobe called"))
        self.assertEqual(result.cover, cover)
        self.assertEqual(result.duration, 20.0)
        sample.assert_called_once()

    def test_opaque_hls_sample_fills_only_selected_unknown_quality(self):
        selected = {"id": "opaque", "height": 0, "width": 0, "fps": None, "label": "source",
                    "url": "https://media.example/variant.m3u8", "http_headers": {}}
        other = {"id": "other", "height": 480, "width": 854, "fps": 25.0, "label": "480p",
                 "url": "https://media.example/other.m3u8", "http_headers": {}}
        def ytdlp(*_args, **_kwargs):
            return [selected, other], "Title", None, {"duration": 10, "formats": [], "browser_resolved": True}
        with mock.patch.object(SI, "_probe_hls_sample", return_value={
                "height": 720, "fps": 60.0, "duration": 10, "vcodec": "h264", "acodec": "aac"}) as sample:
            result = SI.resolve(open_input.OpenInput("https://page.example/watch/4"),
                                find_resolver=lambda _: None, ytdlp=ytdlp,
                                probe=lambda *_: self.fail("remote probe called"))
        self.assertEqual(result.default_height, 720)
        self.assertEqual(result.heights[0]["height"], 720)
        self.assertEqual(result.heights[0]["label"], "720p60")
        self.assertEqual(result.fps_by_height[720], 60.0)
        self.assertEqual(result.quals[0]["height"], 720)
        self.assertEqual(result.quals[1], other)
        sample.assert_called_once()

    def test_hls_sample_does_not_replace_known_quality_height_or_fps(self):
        quality = {"id": "known", "height": 1080, "width": 1920, "fps": 24.0, "label": "1080p",
                   "url": "https://media.example/variant.m3u8", "http_headers": {}}
        with mock.patch.object(SI, "_probe_hls_sample", return_value={
                "height": 720, "fps": 30.0, "duration": 12, "vcodec": "h264", "acodec": "aac"}):
            result = SI.resolve(open_input.OpenInput("https://page.example/watch/5"),
                find_resolver=lambda _: None,
                ytdlp=lambda *_a, **_kw: ([quality], "Title", None,
                    {"duration": None, "formats": [], "browser_resolved": True}),
                probe=lambda *_: self.fail("remote probe called"))
        self.assertEqual((result.default_height, result.fps), (1080, 24.0))
        self.assertEqual((quality["height"], quality["fps"]), (1080, 24.0))

    def test_hls_sample_decrypts_aes128_first_segment_with_scoped_key_headers(self):
        from oh_streamer.queue import queue_download as QD
        from oh_streamer.sources import hls_cache
        playlist = (b'#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="https://keys.example/key"\n'
                    b'#EXTINF:4,\nhttps://segments.example/one.jpeg\n#EXT-X-ENDLIST\n')
        fetched = []
        def fetch(url, headers, limit=0):
            fetched.append((url, dict(headers), limit))
            payload = {"https://media.example/master.m3u8": playlist,
                       "https://segments.example/one.jpeg": b"encrypted",
                       "https://keys.example/key": b"0123456789abcdef"}[url]
            return payload, url
        def probe(path, headers):
            self.assertEqual(Path(path).read_bytes(), b"decrypted-ts")
            return {"duration": 4.0, "fps": 30.0}
        with (mock.patch.object(QD, "_fetch_bytes", side_effect=fetch),
              mock.patch.object(hls_cache, "decrypt_segment", return_value=b"decrypted-ts") as decrypt):
            facts = SI._probe_hls_sample("https://media.example/master.m3u8",
                {"Cookie": "media=1", "Authorization": "token", "Referer": "page"}, probe)
        self.assertEqual(facts["duration"], 4.0)
        self.assertEqual(fetched[1], ("https://segments.example/one.jpeg", {"Referer": "page"}, 16 * 1024 * 1024))
        self.assertEqual(fetched[2], ("https://keys.example/key", {"Referer": "page"}, 4096))
        decrypt.assert_called_once_with(b"encrypted", b"0123456789abcdef", "00" * 16)

    def test_hls_sample_strips_media_credentials_for_cross_host_segment(self):
        from oh_streamer.queue import queue_download as QD
        playlist = b"#EXTM3U\n#EXTINF:4,\nhttps://segments.example/one.jpeg\n#EXT-X-ENDLIST\n"
        with mock.patch.object(QD, "_fetch_bytes", side_effect=[
                (playlist, "https://media.example/master.m3u8"),
                (b"sample", "https://segments.example/one.jpeg")]) as fetch:
            SI._probe_hls_sample("https://media.example/master.m3u8",
                {"Cookie": "media=1", "Authorization": "media-token", "Referer": "page"},
                lambda *_: {"duration": 4, "fps": 30})
        self.assertEqual(fetch.call_args.args[0], "https://segments.example/one.jpeg")
        self.assertEqual(fetch.call_args.args[1], {"Referer": "page"})

    def test_external_hls_metadata_uses_quality_path_and_requires_sample_fps(self):
        facts = {"height": 720, "duration": None, "fps": None, "vcodec": "h264", "acodec": "aac"}
        fake = self.ytdlp_from(self.fake_formats())
        def incomplete(*args, **kwargs):
            quals, title, thumb, info = fake(*args, **kwargs)
            info["duration"] = None
            return quals, title, thumb, info
        with mock.patch.object(SI, "_probe_hls_sample", return_value=facts):
            with self.assertRaisesRegex(SI.ResolveError, "無法讀取影片資訊"):
                SI.resolve(open_input.OpenInput("https://video.example/watch/2"),
                    find_resolver=lambda _u: {"command": ["external"]},
                    resolve_external=lambda *_: ("https://media.example/master.m3u8", {}, "Title",
                                                 {"media_type": "hls"}),
                    ytdlp=incomplete,
                    probe=lambda *_: self.fail("remote probe called"))

    def test_local_file_via_ffprobe(self):
        r = SI.resolve(open_input.OpenInput(str(self.clip), {}, "file"))
        self.assertEqual((r.kind, r.title, r.site), ("file", "clip one", str(self.tmp)))
        self.assertEqual((r.default_height, r.vcodec, r.acodec, r.hls), (360, "h264", "aac", False))
        self.assertAlmostEqual(r.fps, 30.0, places=1)
        self.assertAlmostEqual(r.duration, 3.0, delta=0.3)
        self.assertEqual(r.heights[0]["label"], "360p")
        self.assertEqual(r.heights[0]["size_bytes"], self.clip.stat().st_size)
        self.assertEqual(r.thumb_src, (str(self.clip.resolve()), {}))

    def test_failures_become_short_errors(self):
        with self.assertRaises(SI.ResolveError) as cm:
            SI.resolve(open_input.OpenInput(str(self.tmp / "nope.mp4"), {}, "file"))
        self.assertEqual(cm.exception.short, "找不到影片檔案")
        bad = self.tmp / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(SI.ResolveError) as cm:
            SI.resolve(open_input.OpenInput(str(bad), {}, "file"))
        self.assertIn("無法讀取影片", cm.exception.short)

        def boom(url, headers, with_info=False):
            raise RuntimeError("ERROR: [generic] Unsupported URL: https://x.test/a\nlast line: https://x.test/a")
        with self.assertRaises(SI.ResolveError) as cm:
            SI.resolve(open_input.OpenInput("https://x.test/a"), ytdlp=boom, find_resolver=lambda u: None)
        self.assertEqual(cm.exception.detail, "last line: https://x.test/a")
        self.assertEqual(cm.exception.short, "無法解析這個來源")
        with self.assertRaises(SI.ResolveError) as cm:
            SI.resolve(open_input.OpenInput("https://x.test/a"), find_resolver=lambda u: None,
                       ytdlp=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ERROR: Unsupported URL: https://x.test/a")))
        self.assertEqual(cm.exception.short, "不支援這個網站或網址")


class ThumbnailTests(SourceInfoBase):
    def test_real_grab_is_a_320px_jpeg(self):
        uri = SI.grab_thumbnail(str(self.clip), {}, 3.0)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        data = base64.b64decode(uri.split(",", 1)[1])
        self.assertEqual(data[:2], b"\xff\xd8")
        jpg = self.tmp / "t.jpg"
        jpg.write_bytes(data)
        self.assertEqual(queue_thumbs.jpeg_size(jpg), (320, 180))

    def test_failures_give_none(self):
        self.assertIsNone(SI.grab_thumbnail(str(self.tmp / "nope.mp4"), {}, 3.0))
        with mock.patch.object(SI.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 8)) as run:
            self.assertIsNone(SI.grab_thumbnail("https://h/v.mp4", {}, 100.0))
            self.assertEqual(run.call_args.kwargs["timeout"], SI.THUMB_TIMEOUT)
        with mock.patch.object(SI.subprocess, "run", side_effect=FileNotFoundError("ffmpeg")):
            self.assertIsNone(SI.grab_thumbnail(str(self.clip), {}, 3.0))

    def test_command_seeks_10_percent_and_passes_headers(self):
        cmd = SI.thumb_cmd("https://h/v.m3u8", {"Referer": "https://r/", "User-Agent": "UA/1"}, 200.0, "out.jpg")
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertEqual(cmd[cmd.index("-ss") + 1], "20.000")
        self.assertEqual(cmd[cmd.index("-user_agent") + 1], "UA/1")
        self.assertIn("Referer: https://r/", cmd[cmd.index("-headers") + 1])
        self.assertIn("-rw_timeout", cmd)
        self.assertIn("scale=320:-2", cmd)
        self.assertEqual(cmd[cmd.index("-allowed_extensions") + 1], "ALL")
        signed = SI.thumb_cmd("https://h/v.m3u8?token=example", {}, 200.0, "out.jpg")
        self.assertIn("-allowed_extensions", signed)
        local = SI.thumb_cmd(str(self.clip), {"Referer": "x"}, None, "out.jpg")           # no headers for local files
        self.assertNotIn("-headers", local)
        self.assertNotIn("-allowed_extensions", local)
        self.assertEqual(local[local.index("-ss") + 1], f"{SI.THUMB_FALLBACK_AT:.3f}")


class TokenCacheTests(unittest.TestCase):
    def test_lru_eviction_and_touch(self):
        cache = SI.TokenCache(size=3)
        mk = lambda n: SI.Resolved(f"t{n}", "url", {}, "", "")      # noqa: E731
        tokens = [cache.put(mk(n)) for n in range(3)]
        self.assertEqual(cache.get(tokens[0]).target, "t0")        # touch: t1 is now the oldest
        tokens.append(cache.put(mk(3)))
        self.assertIsNone(cache.get(tokens[1]))
        self.assertEqual([cache.get(t).target for t in (tokens[0], tokens[2], tokens[3])], ["t0", "t2", "t3"])
        self.assertEqual(len(cache), 3)
        self.assertEqual(len({cache.put(mk(9)) for _ in range(5)}), 5)
        self.assertIsNone(cache.get("missing"))

    def test_default_size_is_50(self):
        cache = SI.TokenCache()
        for n in range(60):
            cache.put(SI.Resolved(str(n), "url", {}, "", ""))
        self.assertEqual(len(cache), 50)


class HelperTests(unittest.TestCase):
    def test_short_codec(self):
        self.assertEqual([SI.short_codec(c) for c in ("avc1.640028", "hev1.1.6.L93", "vp09.00.40.08", "av01.0.05M.08",
                                                      "mp4a.40.2", "opus", "none", None, "h264", "flac")],
                         ["h264", "hevc", "vp9", "av1", "aac", "opus", None, None, "h264", "flac"])

    def test_resolve_with_ytdlp_default_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.json"
            p.write_text(json.dumps({"title": "T", "duration": 5, "formats": [fmt("a", 720, 30.0, "https://h/a.mp4")]}))
            plain = source_resolve.resolve_with_ytdlp("https://p/", {}, str(p))
            self.assertEqual(len(plain), 3)
            full = source_resolve.resolve_with_ytdlp("https://p/", {}, str(p), with_info=True)
            self.assertEqual(len(full), 4)
            self.assertEqual(full[3]["duration"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
