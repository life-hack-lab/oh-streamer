"""Offline cover extraction, browser download, and intake handoff regressions."""
import base64
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_streamer.sources.resolvers import covers, resolve_browser as RB
from oh_streamer.sources import source_info as SI, source_resolve as SR
from oh_streamer.player.open_input import OpenInput
from oh_streamer.queue.queue_service import QueueService


class SourceCoverTests(unittest.TestCase):
    def test_metadata_priority_and_relative_url(self):
        html = '''<meta content="https://cdn.test/preview.jpg" name="other">
            <video poster="/poster.jpg"></video>
            <meta name="twitter:image" content="/twitter.jpg">
            <meta content="//cdn.test/cover.jpg?a=1&amp;b=2" property="og:image">'''
        self.assertEqual(covers.cover_url(html, "https://source.test/videos/example/"),
                         "https://cdn.test/cover.jpg?a=1&b=2")

    def test_preview_fallback_and_missing_cover(self):
        self.assertEqual(covers.cover_url('<meta content="/media/preview.jpg">', "https://source.test/"),
                         "https://source.test/media/preview.jpg")
        self.assertEqual(covers.cover_url('<video poster="/poster.jpg">', "https://source.test/"),
                         "https://source.test/poster.jpg")
        self.assertIsNone(covers.cover_url('<meta property="og:image" content="javascript:bad">',
                                           "https://source.test/"))
        self.assertIsNone(covers.cover_url("<html></html>", "https://source.test/"))

    def test_download_uses_browser_session_and_referer(self):
        context = MagicMock()
        response = context.request.get.return_value
        response.ok = True
        data = b"\xff\xd8\xffimage"
        response.body.return_value = data
        self.assertEqual(covers.fetch_cover(context, "https://cdn.test/preview.jpg", "https://source.test/videos/x/"),
                         "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        context.request.get.assert_called_once_with("https://cdn.test/preview.jpg",
            headers={"Referer": "https://source.test/videos/x/"}, timeout=8000)
        response.dispose.assert_called_once()

    def test_failed_cover_download_is_optional(self):
        context = MagicMock()
        context.request.get.side_effect = RuntimeError("timeout")
        self.assertIsNone(covers.fetch_cover(context, "https://cdn.test/cover.jpg", "https://source.test/"))
        context.request.get.side_effect = None
        response = context.request.get.return_value
        response.ok = False
        self.assertIsNone(covers.fetch_cover(context, "https://cdn.test/cover.jpg", "https://source.test/"))
        response.ok = True
        for data in (b"<html>Challenge</html>", b"", b"x" * (5 * 1024 * 1024 + 1)):
            response.body.return_value = data
            self.assertIsNone(covers.fetch_cover(context, "https://cdn.test/cover.jpg", "https://source.test/"))

    def test_browser_resolver_reads_cover_before_closing_context(self):
        manager, browser, context, page = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        manager.__enter__.return_value = object()
        browser.new_context.return_value = context
        context.new_page.return_value = page
        page.evaluate.return_value = "UA"
        page.url = "https://source.test/videos/x/"
        result_from_page = {"url": "https://cdn.test/video.m3u8", "title": "Example",
                            "headers": {"Referer": page.url}, "cover": "cover-data"}

        def resolve_page(actual_page, actual_context, page_url, user_agent, deadline):
            self.assertIs(actual_page, page)
            self.assertIs(actual_context, context)
            self.assertEqual((page_url, user_agent), (page.url, "UA"))
            self.assertFalse(browser.close.called)
            return result_from_page

        fake_api = types.SimpleNamespace(sync_playwright=lambda: manager)
        with patch.dict(sys.modules, {"playwright.sync_api": fake_api}), \
                patch.object(RB, "launch_browser", return_value=browser), \
                patch.object(RB, "_resolve_page", side_effect=resolve_page):
            result = RB.resolve_browser(page.url)
        self.assertEqual(result, result_from_page)
        browser.close.assert_called_once()

    def test_result_extracts_cover_from_a_candidate(self):
        page, context = MagicMock(), MagicMock()
        page.url = "https://source.test/videos/x/"
        page.content.return_value = '<meta property="og:image" content="/preview.jpg">'
        page.locator.return_value.first.get_attribute.return_value = "Example"
        context.cookies.return_value = []
        candidate = RB.Candidate("https://cdn.test/video.m3u8", page.url, 1, 2,
                                 {"Referer": page.url}, verified=True)
        with patch.object(RB, "fetch_cover", return_value="cover-data") as fetch:
            result = RB._result(page, context, candidate, page.url)
        self.assertEqual((result["url"], result["title"], result["cover"]),
                         (candidate.url, "Example", "cover-data"))
        fetch.assert_called_once_with(context, "https://source.test/preview.jpg", page.url)

    def test_cover_survives_resolver_and_intake_without_ffmpeg(self):
        cover = "data:image/jpeg;base64,/9j/aW1hZ2U="
        payload = {"url": "https://cdn.test/video.m3u8", "title": "Example", "cover": cover}
        proc = MagicMock(returncode=0, stdout=json.dumps(payload))
        facts = {"height": 720, "fps": 30.0, "duration": 120.0, "vcodec": "h264", "acodec": "aac"}
        rule = {"command": ["example-resolver"]}
        with patch.object(SR.subprocess, "run", return_value=proc):
            self.assertEqual(len(SR.resolve_external("https://source.test/videos/x/", {}, rule)), 3)
            resolved = SI.resolve(OpenInput("https://source.test/videos/x/"),
                                  find_resolver=lambda url: rule, probe=lambda *args: facts)
        self.assertEqual(resolved.cover, cover)
        with patch.object(SI, "grab_thumbnail") as grab:
            self.assertEqual(QueueService._grab_thumb(resolved), cover)
            grab.assert_not_called()

    def test_no_cover_falls_back_to_video_frame(self):
        resolved = SI.Resolved("https://source.test/videos/x/", "url", {}, "Example", "source.test",
                               duration=120.0, thumb_src=("https://cdn.test/video.m3u8", {"Referer": "r"}))
        with patch.object(SI, "grab_thumbnail", return_value="frame") as grab:
            self.assertEqual(QueueService._grab_thumb(resolved), "frame")
            grab.assert_called_once_with("https://cdn.test/video.m3u8", {"Referer": "r"}, 120.0)

    def test_extractor_cover_preferred_and_failure_falls_back(self):
        resolved = SI.Resolved("https://example.test/video", "url", {}, "Example", "example.test",
                               duration=120.0, thumb_src=("https://cdn.test/video.mp4", {}),
                               cover_src=("https://images.test/cover.jpg", {"Referer": "page"}))
        with patch.object(covers, "fetch_http_cover", return_value="cover") as fetch, \
                patch.object(SI, "grab_thumbnail", return_value="frame") as grab:
            self.assertEqual(QueueService._grab_thumb(resolved), "cover")
            fetch.assert_called_once_with(*resolved.cover_src)
            grab.assert_not_called()

            resolved_fail = SI.Resolved("https://example.test/video", "url", {}, "Example", "example.test",
                                        duration=120.0, thumb_src=("https://cdn.test/video.mp4", {}),
                                        cover_src=("https://images.test/cover.jpg", {"Referer": "page"}))
            fetch.return_value = None
            self.assertEqual(QueueService._grab_thumb(resolved_fail), "frame")
            grab.assert_called_once()

    def test_http_cover_headers_limits_and_html_rejection(self):
        with patch.object(covers, "urlopen") as opened:
            response = opened.return_value.__enter__.return_value
            response.read.return_value = b"\xff\xd8\xffimage"
            self.assertTrue(covers.fetch_http_cover("https://images.test/cover.jpg", {"Referer": "page"})
                            .startswith("data:image/jpeg;base64,"))
            self.assertEqual(opened.call_args.args[0].get_header("Referer"), "page")
            self.assertEqual(opened.call_args.kwargs["timeout"], 8)
            response.read.assert_called_with(5 * 1024 * 1024 + 1)
            response.read.return_value = b"<html>Challenge</html>"
            self.assertIsNone(covers.fetch_http_cover("https://images.test/cover.jpg", {}))
        self.assertIsNone(covers.fetch_http_cover("file:///cover.jpg", {}))


if __name__ == "__main__":
    unittest.main()
