"""Offline tests for capability based browser HLS discovery."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.sources.resolvers import resolve_browser as rb

PAGE = "https://pages.example.test/watch/1"
EMBED = "https://embed.example.test/player"
VIDEO = "https://media.example.test/master.m3u8"
AD = "https://ads.example.test/roll.m3u8"


class FakeFrame:
    def __init__(self, page, url, sources=(), content=""):
        self.page, self.url = page, url
        self.sources, self.html = list(sources), content

    def evaluate(self, _script):
        return self.sources

    def content(self):
        return self.html


def fake_page():
    page = MagicMock()
    page.url = PAGE
    page.main_frame = FakeFrame(page, PAGE)
    page.frames = [page.main_frame]
    page.content.return_value = '<meta property="og:image" content="/poster.jpg">'
    page.locator.return_value.first.get_attribute.return_value = "Example title"
    context = MagicMock()
    context.cookies.return_value = []
    return page, context


def response(page, frame, url, headers=None, body="\ufeff\n#EXTM3U\n#EXTINF:2,", content_type="application/vnd.apple.mpegurl"):
    item = MagicMock()
    item.url, item.status = url, 200
    item.headers = {"content-type": content_type}
    item.text.return_value = body
    item.request.frame = frame
    item.request.all_headers.return_value = headers or {}
    return item


class BrowserResolverTests(unittest.TestCase):
    def test_url_and_playlist_rules(self):
        self.assertTrue(rb._is_hls(VIDEO + "?key=x"))
        self.assertTrue(rb._is_hls("https://media.example.test/manifest", "application/x-mpegurl"))
        self.assertFalse(rb._is_hls("https://media.example.test/image.png?next=foo.m3u8"))
        self.assertTrue(rb._is_playlist("\ufeff \n#EXTM3U"))
        self.assertFalse(rb._is_playlist("<html>blocked</html>"))
        self.assertFalse(rb._http_url("blob:https://media.example.test/master.m3u8"))

    def test_player_beats_early_network_ad_and_keeps_actual_headers(self):
        page, context = fake_page()
        page.main_frame.sources = [VIDEO]
        collector = rb.Collector(page, context, "UA", 1000)
        collector.observe(response(page, page.main_frame, AD, {"referer": PAGE}))
        collector.observe(response(page, page.main_frame, VIDEO,
                                   {"referer": EMBED, "origin": "https://embed.example.test",
                                    "authorization": "Bearer token", "range": "bytes=0-"}))
        collector.scan()
        chosen = collector.choose()
        self.assertEqual(chosen.url, VIDEO)
        self.assertEqual(chosen.headers["Referer"], EMBED)
        self.assertEqual(chosen.headers["Origin"], "https://embed.example.test")
        self.assertEqual(chosen.headers["Authorization"], "Bearer token")
        self.assertNotIn("range", chosen.headers)

    def test_ambiguous_players_fail(self):
        page, context = fake_page()
        page.main_frame.sources = [VIDEO, AD]
        collector = rb.Collector(page, context, "UA", 1000)
        collector.scan()
        for item in collector.candidates:
            item.verified = True
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            collector.choose()

    def test_replaced_player_source_does_not_remain_active(self):
        page, context = fake_page()
        page.main_frame.sources = [AD]
        collector = rb.Collector(page, context, "UA", 1000)
        collector.scan()
        collector.candidates[0].verified = True
        self.assertEqual(collector.choose().url, AD)
        page.main_frame.sources = [VIDEO]
        collector.scan()
        for candidate in collector.candidates:
            if candidate.url == VIDEO:
                candidate.verified = True
        self.assertEqual(collector.choose().url, VIDEO)

    def test_child_frame_relative_player_and_main_html_fallback(self):
        page, context = fake_page()
        page.main_frame.html = '<script>var stream="/backup.m3u8"; var second="https:\\/\\/media.example.test\\/other.m3u8"</script>'
        child = FakeFrame(page, EMBED, ["/main.m3u8"])
        page.frames.append(child)
        collector = rb.Collector(page, context, "UA", 1000)
        collector.scan()
        self.assertEqual({c.url for c in collector.candidates},
                         {"https://pages.example.test/backup.m3u8",
                          "https://embed.example.test/main.m3u8",
                          "https://media.example.test/other.m3u8"})
        self.assertEqual(max(c.rank for c in collector.candidates), 3)

    def test_mime_only_and_invalid_response(self):
        page, context = fake_page()
        page.main_frame.sources = ["blob:https://pages.example.test/opaque"]
        collector = rb.Collector(page, context, "UA", 1000)
        collector.observe(response(page, page.main_frame, "https://media.example.test/manifest"))
        collector.observe(response(page, page.main_frame, AD, body="<html>error</html>"))
        collector.scan()
        self.assertEqual(len(collector.candidates), 1)
        self.assertEqual(collector.choose().url, "https://media.example.test/manifest")

    def test_ad_only_network_playlist_is_not_selected(self):
        page, context = fake_page()
        collector = rb.Collector(page, context, "UA", 1000)
        collector.observe(response(page, page.main_frame, AD))
        collector.scan()
        self.assertIsNone(collector.choose())

    def test_mime_only_player_manifest_verifies_by_body(self):
        page, context = fake_page()
        manifest = "https://media.example.test/manifest?id=7"
        page.main_frame.sources = [manifest]
        reply = context.request.get.return_value
        reply.ok, reply.url = True, manifest
        reply.text.return_value = "\ufeff#EXTM3U\n#EXTINF:1,"
        collector = rb.Collector(page, context, "UA", 1000)
        collector.scan()
        with patch.object(rb.time, "monotonic", return_value=1):
            collector.verify()
        self.assertEqual(collector.choose().url, manifest)
        self.assertEqual(collector.choose().rank, 3)

    def test_verification_does_not_forward_auth_on_redirect(self):
        page, context = fake_page()
        collector = rb.Collector(page, context, "UA", 1000)
        collector.add(VIDEO, PAGE, id(page.main_frame), 3,
                      {"authorization": "Bearer scoped"})
        reply = context.request.get.return_value
        reply.ok = False
        reply.status = 302
        reply.headers = {"location": "https://other.example.test/manifest"}
        with patch.object(rb.time, "monotonic", return_value=1):
            collector.verify()
        self.assertFalse(collector.candidates[0].verified)
        self.assertEqual(context.request.get.call_args.kwargs["max_redirects"], 0)

    def test_browser_redirect_keeps_player_relation_and_final_request_headers(self):
        page, context = fake_page()
        original = "https://media.example.test/entry"
        final = "https://other.example.test/master.m3u8"
        page.main_frame.sources = [original]
        collector = rb.Collector(page, context, "UA", 1000)
        seen = response(page, page.main_frame, final,
                        {"referer": EMBED, "authorization": "Bearer final"})
        prior = MagicMock(url=original, redirected_from=None)
        seen.request.redirected_from = prior
        collector.observe(seen)
        collector.scan()
        chosen = collector.choose()
        self.assertEqual(chosen.url, final)
        self.assertEqual(chosen.rank, 3)
        self.assertEqual(chosen.headers["Referer"], EMBED)
        self.assertEqual(chosen.headers["Authorization"], "Bearer final")

    def test_same_url_can_retry_with_later_valid_headers(self):
        page, context = fake_page()
        collector = rb.Collector(page, context, "UA", 1000)
        collector.add(VIDEO, PAGE, id(page.main_frame), 3)
        first = context.request.get.return_value
        first.ok = False
        with patch.object(rb.time, "monotonic", return_value=1):
            collector.verify()
        collector.add(VIDEO, EMBED, 12345, 3, {"authorization": "Bearer valid"})
        second = MagicMock(ok=True, url=VIDEO)
        second.text.return_value = "#EXTM3U"
        context.request.get.return_value = second
        with patch.object(rb.time, "monotonic", return_value=1):
            collector.verify()
        self.assertTrue(any(c.verified and c.headers.get("Authorization") == "Bearer valid"
                            for c in collector.candidates))
        self.assertEqual(context.request.get.call_count, 2)

    def test_result_scopes_cookies_and_tolerates_bad_metadata(self):
        page, context = fake_page()
        context.cookies.return_value = [{"name": "cdn", "value": "yes"}]
        candidate = rb.Candidate(VIDEO, EMBED, 1, 3,
                                 {"User-Agent": "UA", "Referer": EMBED}, True)
        page.content.side_effect = RuntimeError("detached")
        page.locator.side_effect = RuntimeError("navigating")
        result = rb._result(page, context, candidate, PAGE)
        self.assertEqual(result["url"], VIDEO)
        self.assertEqual(result["headers"]["Cookie"], "cdn=yes")
        self.assertEqual(result["title"], PAGE)
        self.assertIsNone(result["cover"])
        context.cookies.assert_called_once_with([VIDEO])
        self.assertNotIn("Origin", result["headers"])

    def test_navigation_collects_without_clicking_and_removes_observer(self):
        page, context = fake_page()
        page.main_frame.sources = [VIDEO]
        observed = response(page, page.main_frame, VIDEO, {"referer": EMBED})
        clock = [0]
        def navigate(*_args, **_kwargs):
            page.on.call_args.args[1](observed)
        def advance(ms):
            clock[0] += ms / 1000
        page.goto.side_effect = navigate
        page.wait_for_timeout.side_effect = advance
        with patch.object(rb.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(rb, "fetch_cover", return_value=None):
            result = rb._resolve_page(page, context, PAGE, "UA", 5)
        self.assertEqual(result["url"], VIDEO)
        self.assertEqual(result["headers"]["Referer"], EMBED)
        context.request.get.assert_not_called()
        page.click.assert_not_called()
        page.remove_listener.assert_called_once()

    def test_late_main_player_displaces_early_ad(self):
        page, context = fake_page()
        clock = [0]
        observed_ad = response(page, page.main_frame, AD)
        reply = context.request.get.return_value
        reply.ok, reply.url = True, VIDEO
        reply.text.return_value = "#EXTM3U\n#EXTINF:1,"
        def navigate(*_args, **_kwargs):
            page.on.call_args.args[1](observed_ad)
        def advance(ms):
            clock[0] += ms / 1000
            if clock[0] >= 3:
                page.main_frame.sources = [VIDEO]
        page.goto.side_effect = navigate
        page.wait_for_timeout.side_effect = advance
        with patch.object(rb.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(rb, "fetch_cover", return_value=None):
            result = rb._resolve_page(page, context, PAGE, "UA", 8)
        self.assertEqual(result["url"], VIDEO)

    def test_ad_time_does_not_count_toward_new_player_stability(self):
        page, context = fake_page()
        page.main_frame.sources = [AD]
        clock = [0]
        def fetched(url, **_kwargs):
            reply = MagicMock(ok=True, url=url)
            reply.text.return_value = "#EXTM3U\n#EXTINF:1,"
            return reply
        context.request.get.side_effect = fetched
        def advance(ms):
            clock[0] += ms / 1000
            if clock[0] >= 10:
                page.main_frame.sources = [VIDEO]
        page.wait_for_timeout.side_effect = advance
        with patch.object(rb.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(rb, "fetch_cover", return_value=None):
            result = rb._resolve_page(page, context, PAGE, "UA", 25)
        self.assertEqual(result["url"], VIDEO)
        self.assertGreaterEqual(clock[0], 22)


if __name__ == "__main__":
    unittest.main()
