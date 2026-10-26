import unittest

from oh_streamer.sources.http_headers import merge_for_media, scope_for_url


class MediaHeaderTests(unittest.TestCase):
    def test_cross_origin_preserves_observed_media_credentials_and_user_non_sensitive(self):
        result = merge_for_media("https://page.example/watch", "https://media.example/master",
                                 {"authorization": "media-token", "Cookie": "media=1", "referer": "browser",
                                  "user-agent": "browser-agent"},
                                 {"Authorization": "page-token", "cookie": "page=1", "Referer": "user-ref",
                                  "User-Agent": "user-agent", "X-Test": "user"})
        self.assertEqual(result, {"authorization": "media-token", "Cookie": "media=1",
                                  "Referer": "user-ref", "User-Agent": "user-agent", "X-Test": "user"})
        self.assertEqual(len([k for k in result if k.lower() == "referer"]), 1)

    def test_same_origin_allows_user_credentials_and_cross_origin_segment_drops_them(self):
        media = "https://media.example:443/master.m3u8"
        headers = merge_for_media("https://media.example/watch", media,
                                  {"cookie": "observed=1"}, {"Cookie": "user=1", "Authorization": "user"})
        self.assertEqual(headers, {"Cookie": "user=1", "Authorization": "user"})
        self.assertEqual(scope_for_url(media, "https://segments.example/one.jpeg", headers), {})
        self.assertEqual(scope_for_url(media, "https://media.example/one.jpeg", headers), headers)


if __name__ == "__main__":
    unittest.main()
