"""Offline unit tests for qualities.py (run: python streamer/tests/test_qualities.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import os
import sys
import tempfile
import unittest
import oh_streamer.sources.qualities as Q  # noqa: E402


def fmt(fid, h, w=None, fps=30, tbr=1000, v="avc1.64001f", a="mp4a.40.2", proto="https", ext="mp4", **kw):
    d = {"format_id": fid, "url": f"https://cdn.example/{fid}", "height": h, "width": w or (h * 16 // 9 if h else None),
         "fps": fps, "tbr": tbr, "vcodec": v, "acodec": a, "protocol": proto, "ext": ext,
         "http_headers": {"User-Agent": "UA", "Referer": "https://example/"}}
    d.update(kw)
    return d


# A realistic mixed bag, in yt-dlp's worst -> best order
INFO = {
    "title": "t",
    "http_headers": {"User-Agent": "TOP"},
    "formats": [
        {"format_id": "sb0", "url": "https://cdn.example/sb0", "ext": "mhtml", "format_note": "storyboard",
         "height": 45, "width": 80, "vcodec": "none", "acodec": "none", "protocol": "mhtml"},
        fmt("audio-low", None, None, fps=None, tbr=48, v="none", a="mp4a.40.5", ext="m4a"),
        fmt("audio-high", None, None, fps=None, tbr=128, v="none", a="opus", ext="webm"),
        fmt("hls-240", 240, tbr=300, proto="m3u8_native", ext="mp4"),
        fmt("hls-480", 480, tbr=900, proto="m3u8_native", ext="mp4"),
        fmt("hls-720", 720, tbr=2400, proto="m3u8_native", ext="mp4"),
        fmt("hls-1080", 1080, tbr=5200, proto="m3u8_native", ext="mp4"),
        fmt("v-720-only", 720, tbr=9999, a="none", ext="webm"),                  # video-only: excluded
        fmt("v-2160-only", 2160, tbr=20000, a="none", ext="webm"),               # video-only: excluded
        fmt("http-360", 360, tbr=600),
        fmt("http-720-a", 720, tbr=2000),                                        # duplicate height, lower tbr
        fmt("http-720p60", 720, fps=60, tbr=3600),                               # duplicate height, best tbr
        fmt("http-1080-lowbr", 1080, tbr=4000),                                  # duplicate height, lower tbr
        fmt("drm-1440", 1440, tbr=8000, has_drm=True),                           # DRM: excluded
        fmt("nourl-900", 900, url=None),                                         # no url: excluded
        fmt("unknown-codec-540", 540, v=None, a=None, tbr=1200, proto="m3u8_native"),  # codecs unknown: kept
    ],
}


class SeparateStreams(unittest.TestCase):
    """YouTube-style info: no muxed formats, so video-only renditions are paired with the best audio-only one."""

    @staticmethod
    def info(with_audio=True):
        v = lambda i, h, vc, tbr, proto="https": {"format_id": i, "url": f"u-{i}", "height": h, "width": h * 16 // 9,  # noqa: E731
                                                  "vcodec": vc, "acodec": "none", "tbr": tbr, "protocol": proto}
        a = lambda i, ext, abr, note="": {"format_id": i, "url": f"a-{i}", "vcodec": "none", "acodec": "x", "ext": ext,  # noqa: E731
                                          "abr": abr, "format_note": note, "protocol": "https"}
        fmts = [v("hls1080", 1080, "avc1.64", 4000, "m3u8"), v("vp9-1080", 1080, "vp09", 1200), v("avc-1080", 1080, "avc1.64", 1600),
                v("avc-720", 720, "avc1.4d", 700)]
        if with_audio:
            fmts += [a("251", "webm", 137), a("140", "m4a", 129), a("140-drc", "m4a", 129, "DRC"), a("250", "webm", 300)]
        return {"formats": fmts}

    def test_pairs_video_with_best_audio(self):
        q = Q.build_quality_list(self.info())
        self.assertEqual([(x["id"], x["audio_url"]) for x in q], [("avc-1080", "a-140"), ("avc-720", "a-140")])

    def test_video_only_without_audio_is_unplayable(self):
        self.assertEqual(Q.build_quality_list(self.info(with_audio=False)), [])

    def test_muxed_formats_have_no_audio_url(self):
        self.assertTrue(all(x["audio_url"] is None for x in Q.build_quality_list(INFO)))


class BuildList(unittest.TestCase):
    def setUp(self):
        self.q = Q.build_quality_list(INFO)

    def test_heights_sorted_desc_one_per_height(self):
        self.assertEqual([x["height"] for x in self.q], [1080, 720, 540, 480, 360, 240])

    def test_excluded_formats(self):
        ids = {x["id"] for x in self.q}
        for bad in ("sb0", "audio-low", "audio-high", "v-720-only", "v-2160-only", "drm-1440", "nourl-900"):
            self.assertNotIn(bad, ids)

    def test_best_per_height(self):
        by_h = {x["height"]: x for x in self.q}
        self.assertEqual(by_h[1080]["id"], "hls-1080")      # tbr 5200 > 4000
        self.assertEqual(by_h[720]["id"], "http-720p60")    # tbr 3600 beats 2400 / 2000
        self.assertEqual(by_h[480]["id"], "hls-480")

    def test_labels(self):
        by_h = {x["height"]: x for x in self.q}
        self.assertEqual(by_h[1080]["label"], "1080p")
        self.assertEqual(by_h[720]["label"], "720p60")
        self.assertEqual(Q.quality_label(1080, 29.97), "1080p")
        self.assertEqual(Q.quality_label(480, None), "480p")
        self.assertEqual(Q.quality_label(1080, 59.94), "1080p60")

    def test_fields(self):
        top = self.q[0]
        self.assertEqual(set(top), {"id", "height", "width", "fps", "tbr", "url", "audio_url", "http_headers", "label"})
        self.assertEqual(top["http_headers"]["Referer"], "https://example/")

    def test_header_fallback_to_info(self):
        q = Q.build_quality_list({"http_headers": {"X": "1"}, "formats": [
            {"format_id": "a", "url": "u", "height": 480, "vcodec": "h264", "acodec": "aac"}]})
        self.assertEqual(q[0]["http_headers"], {"X": "1"})

    def test_empty(self):
        self.assertEqual(Q.build_quality_list({}), [])
        self.assertEqual(Q.build_quality_list({"formats": []}), [])

    def test_single_quality_fallback(self):
        q = Q.single_quality_from_info({"url": "https://x/y.mp4", "height": 360, "format_id": "18"})
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["label"], "360p")
        self.assertEqual(Q.single_quality_from_info({}), [])

    def test_public_list_has_no_urls(self):
        data = json.loads(Q.public_list(self.q))
        self.assertEqual(data[0], {"id": "hls-1080", "label": "1080p", "height": 1080})
        self.assertNotIn("cdn.example", Q.public_list(self.q))


class Pick(unittest.TestCase):
    def setUp(self):
        self.q = Q.build_quality_list(INFO)

    def h(self, *a, **k):
        return Q.pick_quality(self.q, *a, **k)["height"]

    def test_default_is_best_up_to_1080(self):
        self.assertEqual(self.h(), 1080)
        self.assertEqual(self.h(None, 720), 720)      # --max-height 720 (alias)
        self.assertEqual(self.h(None, 700), 540)
        self.assertEqual(self.h(None, 100), 240)      # nothing that small: the smallest

    def test_numeric_spec(self):
        self.assertEqual(self.h(720), 720)
        self.assertEqual(self.h(600), 540)
        self.assertEqual(self.h(4320), 1080)
        self.assertEqual(self.h(720, 480), 480)       # --max-height still caps

    def test_best_worst(self):
        self.assertEqual(self.h("best"), 1080)
        self.assertEqual(self.h("worst"), 240)
        self.assertEqual(self.h("best", 480), 480)

    def test_best_exceeds_default_cap(self):
        tall = [{"id": "a", "height": 2160}, {"id": "b", "height": 1440}, {"id": "c", "height": 720}]
        self.assertEqual(Q.pick_quality(tall, "best")["height"], 2160)     # best ignores the 1080 default
        self.assertEqual(Q.pick_quality(tall)["height"], 720)              # default keeps <= 1080
        self.assertEqual(Q.pick_quality(tall[:2])["height"], 1440)         # all taller: the smallest

    def test_parse_spec(self):
        self.assertEqual(Q.parse_quality_spec("720"), 720)
        self.assertEqual(Q.parse_quality_spec("720p"), 720)
        self.assertEqual(Q.parse_quality_spec("BEST"), "best")
        self.assertEqual(Q.parse_quality_spec("worst"), "worst")
        self.assertIsNone(Q.parse_quality_spec(None))
        for bad in ("", "hd", "-5", "1p"):
            with self.assertRaises(ValueError):
                Q.parse_quality_spec(bad)

    def test_find(self):
        self.assertEqual(Q.find_quality(self.q, "hls-480")["height"], 480)
        self.assertIsNone(Q.find_quality(self.q, "nope"))
        self.assertEqual(Q.find_quality(self.q, "nope", 360)["id"], "http-360")

    def test_table(self):
        t = Q.format_table(self.q, self.q[1])
        self.assertIn("720p60", t)
        self.assertEqual([l[0] for l in t.splitlines()[1:]].count("*"), 1)


class Fake(unittest.TestCase):
    def test_relative_urls_resolve_and_refresh_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"formats": [fmt("a", 720, url="x/a.mp4"), fmt("b", 480, url="https://h/b.mp4")],
                           "refresh_formats": [fmt("a", 720, url="y/a.mp4")]}, f)
            info = Q.load_fake_info(p)
            self.assertEqual(info["formats"][0]["url"], os.path.join(os.path.realpath(d), "x", "a.mp4"))
            self.assertEqual(info["formats"][1]["url"], "https://h/b.mp4")
            self.assertEqual(len(Q.load_fake_info(p, "refresh_formats")["formats"]), 1)
            self.assertEqual(len(Q.build_quality_list(info)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
