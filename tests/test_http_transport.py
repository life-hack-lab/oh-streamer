import unittest

import requests
import yt_dlp
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import UnsupportedRequest

from oh_streamer.sources.http_transport import configure


class ScopedHTTPTransportTests(unittest.TestCase):
    def test_exact_host_and_idempotence(self):
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            configure(ydl, "file:///video.m3u8", {})
            self.assertNotIn("ScopedHTTPCompatibility", ydl._request_director.handlers)
            configure(ydl, "https://media.example/video.m3u8", {"Authorization": "secret"})
            configure(ydl, "https://media.example/other.m3u8", {})
            handler = ydl._request_director.handlers["ScopedHTTPCompatibility"]
            handler.validate(Request("https://media.example/video.m3u8"))
            for url in ("https://media.example.evil.test/a", "https://sub.media.example/a", "https://other.example/a"):
                with self.assertRaises(UnsupportedRequest):
                    handler.validate(Request(url))
            self.assertEqual(sum(h is handler for h in ydl._request_director.handlers.values()), 1)

    def test_session_keeps_verification_proxy_and_cookie_contract(self):
        with yt_dlp.YoutubeDL({"quiet": True, "proxy": "http://proxy.example:8080"}) as ydl:
            configure(ydl, "https://media.example/video.m3u8", {})
            handler = ydl._request_director.handlers["ScopedHTTPCompatibility"]
            session = handler._create_instance(ydl.cookiejar)
            try:
                self.assertTrue(session.verify)
                self.assertFalse(session.trust_env)
                self.assertIs(type(session.adapters["https://"]), requests.adapters.HTTPAdapter)
                self.assertIs(session.cookies, ydl.cookiejar)
                self.assertEqual(handler._get_proxies(Request("https://media.example/a"))["all"], "http://proxy.example:8080")
                response = requests.Response()
                response.status_code = 302
                response.url = "https://media.example/video.m3u8"
                response.headers["Location"] = "https://other.example/key"
                self.assertIsNone(session.get_redirect_target(response))
                response.headers["Location"] = "/variant.m3u8"
                self.assertEqual(session.get_redirect_target(response), "/variant.m3u8")
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
