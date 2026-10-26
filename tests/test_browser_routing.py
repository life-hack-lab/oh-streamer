"""Offline routing from arbitrary video pages to the shared browser resolver."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from unittest import mock

import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oh_streamer.sources import source_resolve


class BrowserRoutingTests(unittest.TestCase):
    def test_browser_result_does_not_inherit_page_credentials(self):
        rule = {"command": ["{python}", "-m", "oh_streamer.sources.resolvers.resolve_browser"]}
        result = {"url": "https://media.example/master.m3u8", "headers": {
            "Cookie": "media=session", "Authorization": "media-token", "referer": "https://embed.example/"}}
        process = MagicMock(returncode=0, stdout=json.dumps(result))
        with patch.object(source_resolve.subprocess, "run", return_value=process):
            _, headers, _ = source_resolve.resolve_external("https://page.example/watch", {
                "Cookie": "page=secret", "authorization": "page-token", "Referer": "user-reference",
                "User-Agent": "user-agent"}, rule)
        self.assertEqual(headers["Cookie"], "media=session")
        self.assertEqual(headers["Authorization"], "media-token")
        self.assertEqual(headers["Referer"], "user-reference")
        self.assertEqual(headers["User-Agent"], "user-agent")
        self.assertEqual(len([k for k in headers if k.lower() == "referer"]), 1)

    def test_direct_media_and_empty_command_opt_out(self):
        browser_rule = {"match": "*", "builtin": "browser", "command": ["{python}", "-m",
                        "oh_streamer.sources.resolvers.resolve_browser"]}
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "resolvers.json"
            defaults = Path(directory) / "defaults.json"
            defaults.write_text(json.dumps([browser_rule]), encoding="utf-8")
            with patch.object(source_resolve, "_has_page_extractor", return_value=False):
                self.assertIsNone(source_resolve.find_resolver(
                    "https://media.example/video.mp4", primary, defaults, migrate=False))
                self.assertIsNone(source_resolve.find_resolver(
                    "https://media.example/master.m3u8", primary, defaults, migrate=False))
            primary.write_text(json.dumps([{"match": "optout.example", "command": []}]), encoding="utf-8")
            self.assertIsNone(source_resolve.find_resolver(
                "https://optout.example/watch/1", primary, defaults, migrate=False))

    def test_failed_page_falls_back_once_and_failed_media_stops(self):
        calls = []
        class FakeYDL:
            def __init__(self, _opts):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def extract_info(self, url, download=False):
                calls.append(url)
                raise yt_dlp.utils.DownloadError("offline failure")

        with (patch.object(yt_dlp, "YoutubeDL", FakeYDL),
              patch("oh_streamer.sources.http_transport.configure"),
              patch.object(source_resolve, "resolve_external", return_value=(
                  "https://media.example/master.m3u8", {}, "Title", {})) as browser,
              patch.object(source_resolve.shutil, "which", return_value=None),
              patch.object(source_resolve, "_browser_opted_out", return_value=False)):
            with self.assertRaises(yt_dlp.utils.DownloadError):
                source_resolve.resolve_with_ytdlp("https://page.example/watch/1", {})
            self.assertEqual(calls, ["https://page.example/watch/1", "https://media.example/master.m3u8"])
            browser.assert_called_once()
            calls.clear()
            browser.reset_mock()
            with self.assertRaises(yt_dlp.utils.DownloadError):
                source_resolve.resolve_with_ytdlp("https://media.example/movie.mp4", {})
            browser.assert_not_called()

    def test_failed_page_with_empty_command_does_not_fall_back(self):
        class FakeYDL:
            def __init__(self, _opts):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def extract_info(self, *_args, **_kwargs):
                raise yt_dlp.utils.DownloadError("offline failure")

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "resolvers.json"
            config.write_text(json.dumps([{"match": "optout.example", "command": []}]), encoding="utf-8")
            with (patch.object(yt_dlp, "YoutubeDL", FakeYDL),
                  patch.object(source_resolve, "RESOLVERS_FILE", config),
                  patch.object(source_resolve, "resolve_external") as browser):
                with self.assertRaises(yt_dlp.utils.DownloadError):
                    source_resolve.resolve_with_ytdlp("https://optout.example/watch/1", {})
                browser.assert_not_called()

    def test_resolved_opaque_media_with_referer_uses_scoped_transport(self):
        configured = []
        class FakeYDL:
            def __init__(self, _opts):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def extract_info(self, url, download=False):
                if url.startswith("https://page.example/"):
                    raise yt_dlp.utils.DownloadError("page failed")
                return {"title": "Media", "duration": 10, "formats": [{
                    "format_id": "720", "url": url, "height": 720, "width": 1280, "fps": 30,
                    "vcodec": "avc1", "acodec": "mp4a", "protocol": "m3u8_native", "ext": "mp4"}]}

        media = "https://media.example/opaque-token"
        with (patch.object(yt_dlp, "YoutubeDL", FakeYDL),
              patch("oh_streamer.sources.http_transport.configure",
                    side_effect=lambda _ydl, url, headers: configured.append((url, dict(headers)))),
              patch.object(source_resolve, "resolve_external", return_value=(
                  media, {"Referer": "https://page.example/watch/1"}, "Page", {})),
              patch.object(source_resolve.shutil, "which", return_value=None),
              patch.object(source_resolve, "_browser_opted_out", return_value=False)):
            quals, title, _ = source_resolve.resolve_with_ytdlp("https://page.example/watch/1", {})
        self.assertEqual(title, "Page")
        self.assertEqual(quals[0]["url"], media)
        self.assertEqual(configured, [(media, {"Referer": "https://page.example/watch/1"})])
    def test_wildcard_browser_applies_only_to_unclaimed_page_urls(self):
        page = "https://pages.example.test/watch/1"
        browser_rule = {
            "match": "*",
            "builtin": "browser",
            "command": ["{python}", "-m", "oh_streamer.sources.resolvers.resolve_browser"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "resolvers.json"
            defaults = root / "resolvers.default.json"
            defaults.write_text(json.dumps([browser_rule]), encoding="utf-8")
            with patch.object(source_resolve, "_has_page_extractor", return_value=False):
                self.assertEqual(source_resolve.find_resolver(page, primary, defaults, migrate=False), browser_rule)
                self.assertIsNone(source_resolve.find_resolver(
                    "https://media.example.test/master.m3u8", primary, defaults, migrate=False))
            with patch.object(source_resolve, "_has_page_extractor", return_value=True):
                self.assertIsNone(source_resolve.find_resolver(page, primary, defaults, migrate=False))

    def test_reserved_resolver_module_is_marked_as_builtin_browser(self):
        page = "https://pages.example.test/watch/1"
        rule = {
            "match": "pages.example.test",
            "command": ["{python}", "-m", "oh_streamer.sources.resolvers.resolve_retired"],
        }
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "resolvers.json"
            defaults = Path(directory) / "resolvers.default.json"
            primary.write_text(json.dumps([rule]), encoding="utf-8")
            defaults.write_text("[]", encoding="utf-8")
            found = source_resolve.find_resolver(page, primary, defaults, migrate=False)
        self.assertEqual(found["command"], rule["command"])
        self.assertEqual(found["builtin"], "browser")


if __name__ == "__main__":
    unittest.main()
