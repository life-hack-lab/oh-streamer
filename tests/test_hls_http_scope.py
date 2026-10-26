"""HLS urllib redirects and segment/key credential boundaries."""

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest import mock
import urllib.request

from oh_streamer.queue import queue_download as QD
from oh_streamer.sources.hls_cache import Segment


class HLSHTTPBoundaryTests(unittest.TestCase):
    def test_redirect_keeps_same_origin_and_strips_cross_origin_credentials(self):
        handler = QD._ScopedHLSRedirect()
        initial = urllib.request.Request("https://media.example:443/master.m3u8",
            headers={"Cookie": "media=1", "AUTHORIZATION": "token", "Referer": "page"})
        same = handler.redirect_request(initial, None, 302, "Found", {},
                                        "https://media.example/variant.m3u8")
        self.assertEqual(same.get_header("Cookie"), "media=1")
        self.assertEqual(same.get_header("Authorization"), "token")
        different = handler.redirect_request(initial, None, 302, "Found", {},
                                             "https://media.example:8443/variant.m3u8")
        self.assertIsNone(different.get_header("Cookie"))
        self.assertIsNone(different.get_header("Authorization"))
        self.assertEqual(different.get_header("Referer"), "page")
        later = handler.redirect_request(different, None, 302, "Found", {},
                                         "https://other.example/variant.m3u8")
        self.assertIsNone(later.get_header("Cookie"))

    def test_segment_and_key_requests_scope_against_initial_media_origin(self):
        segment = Segment(index=0, duration=4.0, url="https://segments.example/one.jpeg",
                          filename="000000.ts", key_uri="https://keys.example/key", iv="00" * 16)
        fetched = []
        downloaded = []
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".test-tmp") as directory:
            path = Path(directory) / "segment.ts"
            def fake_fetch(url, headers, limit=0):
                fetched.append((url, dict(headers)))
                return b"0123456789abcdef", url
            def fake_download(url, headers, path, *_args, **_kwargs):
                downloaded.append((url, dict(headers)))
                path.write_bytes(b"encrypted")
                return 9
            with (mock.patch.object(QD, "_fetch_bytes", side_effect=fake_fetch),
                  mock.patch.object(QD, "_download_segment", side_effect=fake_download),
                  mock.patch.object(QD.hls_cache, "decrypt_segment", return_value=b"plain")):
                QD._fetch_segment_plain(segment,
                    {"Cookie": "media=1", "authorization": "token", "Referer": "page"},
                    path, {}, threading.Event(), None, 1, source_url="https://media.example/master.m3u8")
            self.assertEqual(path.read_bytes(), b"plain")
        self.assertEqual(fetched, [("https://keys.example/key", {"Referer": "page"})])
        self.assertEqual(downloaded, [("https://segments.example/one.jpeg", {"Referer": "page"})])


if __name__ == "__main__":
    unittest.main()
