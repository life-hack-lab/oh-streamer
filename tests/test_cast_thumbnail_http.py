"""Preview HTTP routing and cast integration without a receiver or GPU."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform.hls_cast import HlsServer, HlsVod, SegmentCache, SegmentPlan


class CastThumbnailHttpTests(unittest.TestCase):
    def setUp(self):
        self.plan = SegmentPlan(30, 360, 64, 48)
        self.vod = Mock(plan=self.plan)
        self.thumbs = Mock(width=64, height=48, bandwidth=512000, codecs="avc1.42c01e")
        self.thumbs.playlist.return_value = "#EXTM3U\n#EXT-X-I-FRAMES-ONLY\n"
        self.thumbs.get_segment.return_value = b"preview-bytes"
        self.server = HlsServer(self.vod, "127.0.0.1", thumbnails=self.thumbs).start()
        self.addCleanup(self.server.stop)

    def fetch(self, name, **kwargs):
        request = urllib.request.Request(self.server.url(name), **kwargs)
        return urllib.request.urlopen(request)

    def test_master_and_preview_routes_do_not_request_production(self):
        with self.fetch("master.m3u8") as response:
            master = response.read().decode()
        self.assertIn('#EXT-X-VERSION:4', master)
        self.assertIn('#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH=512000', master)
        self.assertIn('URI="thumbnails.m3u8"', master)
        with self.fetch("thumbnails.m3u8") as response:
            self.assertIn(b"I-FRAMES-ONLY", response.read())
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with self.fetch("thumb_2.ts") as response:
            self.assertEqual(response.read(), b"preview-bytes")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.thumbs.get_segment.assert_called_once_with(2)
        self.vod.get_segment.assert_not_called()

    def test_ranges_head_and_invalid_ranges(self):
        with self.fetch("thumb_0.ts", headers={"Range": "bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"evie")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/13")
        with self.fetch("thumb_0.ts", headers={"Range": "bytes=-5"}) as response:
            self.assertEqual(response.read(), b"bytes")
        with self.fetch("thumb_0.ts", method="HEAD") as response:
            self.assertEqual(response.read(), b"")
            self.assertEqual(response.headers["Content-Length"], "13")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.fetch("thumb_0.ts", headers={"Range": "bytes=99-"})
        self.assertEqual(raised.exception.code, 416)

    def test_allowlist_token_and_failure_isolation(self):
        for name in ("thumb_3.ts", "thumb_-1.ts", "../thumb_0.ts", "thumb_0.ts/extra"):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.fetch(name)
            self.assertEqual(raised.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(self.server.url("thumb_0.ts").replace(self.server.token, "wrong"))
        self.assertEqual(raised.exception.code, 404)
        for error in (None, RuntimeError("private source details")):
            self.thumbs.get_segment.side_effect = error
            self.thumbs.get_segment.return_value = None
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.fetch("thumb_0.ts")
            self.assertEqual(raised.exception.code, 503)
            self.assertEqual(raised.exception.read(), b"preview unavailable\n")
        with self.fetch("index.m3u8") as response:
            self.assertEqual(response.status, 200)
        self.vod.get_segment.assert_not_called()

    def test_snapshot_does_not_move_anchor_or_cache_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            vod = object.__new__(HlsVod)
            vod.cond, vod.gen, vod.closed = threading.Condition(), 7, False
            vod.cache = SegmentCache(directory, 4096)
            path = Path(directory) / "segment.ts"
            path.write_bytes(b"restored")
            vod.cache.add(1, str(path))
            vod.anchor, vod.requests = 2, 0
            stamp = vod.cache.items[1][2]
            self.assertEqual(vod.peek_segment(1), (7, b"restored"))
            self.assertEqual(vod.peek_segment(0), (7, None))
            self.assertEqual((vod.anchor, vod.requests), (2, 0))
            self.assertEqual(vod.cache.items[1][2], stamp)

    def test_manager_forces_master_and_honors_disabled_previews(self):
        from test_progressive_cast_controller import manager_type
        from oh_streamer.platform import hls_cast as hls
        vod, server = Mock(), Mock()
        server.start.return_value = server
        fake_hls = SimpleNamespace(FileCastSource=hls.FileCastSource, SegmentPlan=SegmentPlan,
                                   HlsVod=Mock(return_value=vod), HlsServer=Mock(return_value=server))
        for enabled in (True, False):
            manager = object.__new__(manager_type(fake_hls))
            manager.active = None
            manager.args = SimpleNamespace(cast_bind="127.0.0.1", cast_port=0, thumbnails=enabled,
                                           cast_cache_gb=1, cast_playlist="media")
            manager._ipc, manager._osd, manager.publish_menu = Mock(), Mock(), Mock()
            session = SimpleNamespace(meta=SimpleNamespace(video_fps_exact=30, frames_count=360,
                                      duration=12, video_width=64, video_height=48,
                                      start_pts=0, time_base=1), fmt_start=0,
                                      has_audio=True, encoder="libx264", target=0, stop=Mock(return_value=0))
            server.url.reset_mock()
            with patch("oh_streamer.platform.cast_subtitles.selected_subtitles", return_value=None), \
                 patch("oh_streamer.player.cast_thumbnail_sources.create_thumbnails", return_value=self.thumbs) as factory:
                self.assertTrue(manager.start("hls", session, 0))
            server.url.assert_called_once_with("master.m3u8" if enabled else "index.m3u8")
            if enabled:
                factory.assert_called_once_with(vod, session)
                self.assertIs(fake_hls.HlsServer.call_args.kwargs["thumbnails"], self.thumbs)
            else:
                factory.assert_not_called()
                self.assertNotIn("thumbnails", fake_hls.HlsServer.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
