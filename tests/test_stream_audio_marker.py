"""T2.3 task A: remote audio marker (``audio.rstream``), signed-url expiry mechanism, media_fingerprint."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from oh_streamer.sources import media_input as mi
from oh_streamer.subtitles import slice_results as sr

SECRET_URL = "https://cdn.example.test/v/240p.m3u8?sig=SECRET123&exp=1"

# Real stderr strings captured with ffmpeg 9.0.2 against a loopback server (design doc 1.4).
STDERR_410 = "[http @ 000001] HTTP error 410 Gone\n[in#0 @ 000002] Error opening input: Server returned 4XX Client Error, but not one of 40{0,1,3,4}\n"
STDERR_403 = "[http @ 000001] HTTP error 403 Forbidden\n[in#0 @ 000002] Error opening input: Server returned 403 Forbidden (access denied)\n"
STDERR_404 = "[in#0 @ 000002] Error opening input: Server returned 404 Not Found\n"
STDERR_SEG_SKIP = ("[http @ 0001] HTTP error 403 Forbidden\n[hls @ 0002] Failed to open segment 10 of playlist 0\n"
                   "[hls @ 0002] Segment 10 of playlist 0 failed too many times, skipping\n")
STDERR_SEG_SKIP_NO_HTTP = ("[hls @ 0002] Failed to open segment 10 of playlist 0\n"
                           "[hls @ 0002] Segment 10 of playlist 0 failed too many times, skipping\n")
STDERR_500 = "[http @ 0001] HTTP error 503 Service Unavailable\n"
STDERR_4XX = "[in#0 @ 000002] Error opening input: Server returned 4XX Client Error, but not one of 40{0,1,3,4}\n"


def marker_doc(url=SECRET_URL, generation=0, identity="abc123", headers=None):
    return {"format": mi.REMOTE_FORMAT, "kind": "lowest", "url": url,
            "headers": headers if headers is not None else {"Referer": "https://site.test/", "User-Agent": "UA/1"},
            "hls": True, "generation": generation, "identity": identity,
            "rendition": {"id": "hls-240p", "height": 240, "label": "240p"}}


class MarkerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.marker = self.dir / "streams" / "s1" / "audio.rstream"
        self.addCleanup(mi.unregister_refresher, self.marker)


class MarkerIoTest(MarkerCase):
    def test_roundtrip_and_atomic(self):
        mi.write_remote_marker(self.marker, marker_doc())
        got = mi.read_remote_marker(self.marker)
        self.assertEqual(got["url"], SECRET_URL)
        self.assertEqual(got["generation"], 0)
        self.assertEqual(got["identity"], "abc123")
        self.assertEqual(sorted(p.name for p in self.marker.parent.iterdir()), ["audio.rstream"])  # no tmp left
        mi.write_remote_marker(self.marker, marker_doc(generation=1, url="https://x.test/new.m3u8"))
        self.assertEqual(mi.read_remote_marker(self.marker)["generation"], 1)
        self.assertEqual(sorted(p.name for p in self.marker.parent.iterdir()), ["audio.rstream"])

    def test_failed_write_keeps_old_file(self):
        mi.write_remote_marker(self.marker, marker_doc())
        with self.assertRaises(mi.RemoteMarkerError):
            mi.write_remote_marker(self.marker, {"url": ""})
        self.assertEqual(mi.read_remote_marker(self.marker)["url"], SECRET_URL)
        self.assertEqual(sorted(p.name for p in self.marker.parent.iterdir()), ["audio.rstream"])

    def test_defaults_filled(self):
        mi.write_remote_marker(self.marker, {"url": "https://x.test/a.m3u8"})
        got = mi.read_remote_marker(self.marker)
        self.assertEqual((got["format"], got["headers"], got["generation"]), (mi.REMOTE_FORMAT, {}, 0))

    def test_bad_markers_give_clear_errors_without_url(self):
        self.marker.parent.mkdir(parents=True)
        cases = {
            "not json": "{" + SECRET_URL,
            "array": "[1]",
            "wrong format": json.dumps(dict(marker_doc(), format="other-1")),
            "no url": json.dumps({"format": mi.REMOTE_FORMAT}),
            "bad headers": json.dumps(dict(marker_doc(), headers=[1])),
            "bad generation": json.dumps(dict(marker_doc(), generation="1")),
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.marker.write_text(text, encoding="utf-8")
                with self.assertRaises(mi.RemoteMarkerError) as cm:
                    mi.read_remote_marker(self.marker)
                self.assertNotIn("SECRET123", str(cm.exception))
                self.assertIn("audio.rstream", str(cm.exception))
        self.marker.unlink()
        with self.assertRaises(mi.RemoteMarkerError):
            mi.read_remote_marker(self.marker)

    def test_suffix_detection(self):
        self.assertTrue(mi.is_remote_marker(self.marker))
        self.assertTrue(mi.is_remote_marker(str(self.marker).upper()))
        self.assertFalse(mi.is_remote_marker(self.dir / "source.rcache"))
        self.assertFalse(mi.is_remote_marker(self.dir / "a.mp4"))
        self.assertFalse(mi.is_remote_marker(None))
        self.assertFalse(mi.needs_runtime_url(self.marker))  # must not go through the range-cache lookup


class FfmpegInputTest(MarkerCase):
    def test_marker_gives_url_headers_and_timeout(self):
        mi.write_remote_marker(self.marker, marker_doc())
        url, opts = mi.ffmpeg_input(self.marker)
        self.assertEqual(url, SECRET_URL)
        self.assertEqual(opts[:2], ["-user_agent", "UA/1"])
        self.assertEqual(opts[2], "-headers")
        self.assertEqual(opts[3], "Referer: https://site.test/\r\n")
        self.assertEqual(opts[-4:], ["-rw_timeout", "30000000", "-extension_picky", "0"])   # hls marker
        self.assertNotIn("-reconnect", opts)

    def test_marker_without_headers(self):
        mi.write_remote_marker(self.marker, marker_doc(headers={}))
        self.assertEqual(mi.ffmpeg_input(self.marker)[1], ["-rw_timeout", "30000000", "-extension_picky", "0"])

    def test_extension_picky_only_for_hls_markers(self):
        # Some media servers serve MPEG-TS as video0.jpeg; ffmpeg >= 7.1 refuses that unless extension_picky is off. A plain
        # (mp4) marker must not get the hls-only option: ffmpeg errors on an unknown demuxer option.
        mi.write_remote_marker(self.marker, {**marker_doc(headers={}), "hls": False})
        self.assertEqual(mi.ffmpeg_input(self.marker)[1], ["-rw_timeout", "30000000"])

    def test_plain_path_unchanged(self):
        plain = self.dir / "movie.mp4"
        self.assertEqual(mi.ffmpeg_input(plain), (os.fspath(plain), []))
        self.assertEqual(mi.ffmpeg_input("https://x.test/a.mp4"), ("https://x.test/a.mp4", []))

    def test_rcache_unchanged(self):
        rc = self.dir / "range-cache" / "source.rcache"
        url, opts = mi.ffmpeg_input(rc, "http://127.0.0.1:1/x")
        self.assertEqual(url, "http://127.0.0.1:1/x")
        self.assertEqual(opts, list(mi.HTTP_INPUT_ARGS))


class RemoteFailureTest(unittest.TestCase):
    def test_expired(self):
        for name, text in (("410", STDERR_410), ("403", STDERR_403), ("404", STDERR_404), ("seg 403", STDERR_SEG_SKIP)):
            with self.subTest(name):
                self.assertEqual(mi.remote_failure(text), "expired")

    def test_failed(self):
        for name, text in (("seg skip", STDERR_SEG_SKIP_NO_HTTP), ("503", STDERR_500), ("4XX", STDERR_4XX)):
            with self.subTest(name):
                self.assertEqual(mi.remote_failure(text), "failed")

    def test_clean(self):
        self.assertIsNone(mi.remote_failure(""))
        self.assertIsNone(mi.remote_failure(None))
        self.assertIsNone(mi.remote_failure("[hls @ 0001] Opening 'seg001.ts' for reading\nsize=N/A\n"))
        self.assertIsNone(mi.remote_failure("frame 4030 bitrate 1403 kbit/s"))  # numbers alone are not errors


class RunRemoteTest(MarkerCase):
    def setUp(self):
        super().setUp()
        mi.write_remote_marker(self.marker, marker_doc())
        self.clock = [1000.0]
        old = mi._monotonic
        mi._monotonic = lambda: self.clock[0]
        self.addCleanup(setattr, mi, "_monotonic", old)
        self.refresh_calls = []

        def refresher(path):
            self.refresh_calls.append(path)
            doc = mi.read_remote_marker(path)
            mi.write_remote_marker(path, dict(doc, url="https://cdn.example.test/new.m3u8?sig=NEW",
                                              generation=doc["generation"] + 1))
        self.refresher = refresher

    def test_non_marker_runs_once(self):
        calls = []
        self.assertEqual(mi.run_remote(self.dir / "a.mp4", lambda: calls.append(1) or "ok"), "ok")
        self.assertEqual(len(calls), 1)
        with self.assertRaises(mi.RemoteExpired):  # no retry for ordinary files
            mi.run_remote(self.dir / "a.mp4", lambda: (_ for _ in ()).throw(mi.RemoteExpired()))

    def test_success_without_expiry(self):
        mi.register_refresher(self.marker, self.refresher)
        self.assertEqual(mi.run_remote(self.marker, lambda: 7), 7)
        self.assertEqual(self.refresh_calls, [])

    def test_expired_then_ok_refreshes_once(self):
        mi.register_refresher(self.marker, self.refresher)
        urls = []

        def attempt():
            url = mi.ffmpeg_input(self.marker)[0]
            urls.append(url)
            if "SECRET123" in url:
                raise mi.RemoteExpired()
            return "done"
        self.assertEqual(mi.run_remote(self.marker, attempt), "done")
        self.assertEqual(len(self.refresh_calls), 1)
        self.assertEqual(len(urls), 2)
        self.assertEqual(mi.read_remote_marker(self.marker)["generation"], 1)

    def test_expired_twice_raises_runtime_error_without_url(self):
        mi.register_refresher(self.marker, self.refresher)

        def attempt():
            raise mi.RemoteExpired("x")
        with self.assertRaises(RuntimeError) as cm:
            mi.run_remote(self.marker, attempt)
        self.assertNotIsInstance(cm.exception, mi.RemoteExpired)
        self.assertIn("重新解析後仍無法讀取", str(cm.exception))
        self.assertNotIn("SECRET123", str(cm.exception))
        self.assertEqual(len(self.refresh_calls), 1)

    def test_no_refresher(self):
        with self.assertRaises(RuntimeError) as cm:
            mi.run_remote(self.marker, lambda: (_ for _ in ()).throw(mi.RemoteExpired()))
        self.assertNotIsInstance(cm.exception, mi.RemoteExpired)

    def test_two_threads_expire_together_refresh_once(self):
        gate = threading.Barrier(2)
        in_refresh = threading.Event()
        release = threading.Event()

        def slow_refresher(path):
            self.refresh_calls.append(path)
            in_refresh.set()
            release.wait(5)
            doc = mi.read_remote_marker(path)
            mi.write_remote_marker(path, dict(doc, url="https://cdn.example.test/new.m3u8", generation=doc["generation"] + 1))
        mi.register_refresher(self.marker, slow_refresher)
        results, errors = [], []

        def worker():
            def attempt():
                url = mi.ffmpeg_input(self.marker)[0]
                if "SECRET123" in url:
                    try:
                        gate.wait(5)  # both threads have read the old url before either refreshes
                    except threading.BrokenBarrierError:
                        pass
                    raise mi.RemoteExpired()
                return url
            try:
                results.append(mi.run_remote(self.marker, attempt))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        self.assertTrue(in_refresh.wait(5))
        release.set()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.refresh_calls), 1)
        self.assertEqual(results, ["https://cdn.example.test/new.m3u8"] * 2)

    def test_second_refresh_within_interval_rejected(self):
        mi.register_refresher(self.marker, self.refresher)
        mi.refresh_remote(self.marker, 0)
        self.assertEqual(mi.read_remote_marker(self.marker)["generation"], 1)
        self.clock[0] += mi.REFRESH_MIN_INTERVAL_S - 1
        with self.assertRaises(RuntimeError):
            mi.refresh_remote(self.marker, 1)
        self.assertEqual(len(self.refresh_calls), 1)
        self.clock[0] += 2
        mi.refresh_remote(self.marker, 1)
        self.assertEqual(len(self.refresh_calls), 2)
        self.assertEqual(mi.read_remote_marker(self.marker)["generation"], 2)

    def test_stale_generation_returns_without_calling_refresher(self):
        mi.register_refresher(self.marker, self.refresher)
        mi.refresh_remote(self.marker, 0)
        mi.refresh_remote(self.marker, 0)  # someone already swapped: no second call, no rate-limit error
        self.assertEqual(len(self.refresh_calls), 1)

    def test_unregister_forgets_refresher_and_rate_limit(self):
        mi.register_refresher(self.marker, self.refresher)
        mi.refresh_remote(self.marker, 0)
        mi.unregister_refresher(self.marker)
        with self.assertRaises(RuntimeError):
            mi.refresh_remote(self.marker, 1)
        mi.register_refresher(self.marker, self.refresher)
        mi.refresh_remote(self.marker, 1)  # rate-limit state was dropped with the item
        self.assertEqual(len(self.refresh_calls), 2)

    def test_refresher_error_propagates_and_rate_limits(self):
        def boom(path):
            raise RuntimeError("解析失敗")
        mi.register_refresher(self.marker, boom)
        with self.assertRaises(RuntimeError) as cm:
            mi.run_remote(self.marker, lambda: (_ for _ in ()).throw(mi.RemoteExpired()))
        self.assertIn("解析失敗", str(cm.exception))
        with self.assertRaises(RuntimeError) as cm2:  # retried at once: refused, resolver not hammered
            mi.refresh_remote(self.marker, 0)
        self.assertNotIn("解析失敗", str(cm2.exception))


class FingerprintTest(MarkerCase):
    def test_stable_across_refresh(self):
        mi.write_remote_marker(self.marker, marker_doc())
        before = sr.media_fingerprint(self.marker)
        self.assertEqual(before, "stream:abc123")
        mi.write_remote_marker(self.marker, marker_doc(url="https://other.test/new.m3u8?sig=2", generation=3))
        self.assertEqual(sr.media_fingerprint(self.marker), before)

    def test_different_identity_differs(self):
        mi.write_remote_marker(self.marker, marker_doc(identity="one"))
        other = self.dir / "streams" / "s2" / "audio.rstream"
        mi.write_remote_marker(other, marker_doc(identity="two"))
        self.assertNotEqual(sr.media_fingerprint(self.marker), sr.media_fingerprint(other))

    def test_missing_identity_falls_back_to_path_hash(self):
        doc = marker_doc()
        del doc["identity"]
        other = self.dir / "streams" / "s2" / "audio.rstream"
        mi.write_remote_marker(self.marker, doc)
        mi.write_remote_marker(other, doc)
        a, b = sr.media_fingerprint(self.marker), sr.media_fingerprint(other)
        self.assertTrue(a.startswith("stream-path:"))
        self.assertNotEqual(a, b)
        mi.write_remote_marker(self.marker, dict(doc, generation=2, url="https://x.test/n.m3u8"))
        self.assertEqual(sr.media_fingerprint(self.marker), a)

    def test_unreadable_marker_does_not_raise(self):
        self.marker.parent.mkdir(parents=True)
        self.marker.write_text("garbage", encoding="utf-8")
        self.assertTrue(sr.media_fingerprint(self.marker).startswith("stream-path:"))


if __name__ == "__main__":
    unittest.main()
