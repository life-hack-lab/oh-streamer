"""Local-only HLS cache tests; all upstream resources use loopback or in-memory playlists."""
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import json
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request

from oh_streamer.sources import hls_cache as HC


class PlaylistParserTest(unittest.TestCase):
    BASE = "http://127.0.0.1:31001/media/index.m3u8"

    def parse(self, body):
        return HC.parse_media_playlist(body, self.BASE)

    def test_creates_complete_local_relative_playlist(self):
        p = self.parse("""#EXTM3U
#EXT-X-VERSION:3
#EXT-X-PLAYLIST-TYPE:VOD
#EXTINF:1.25,title
one.ts
#EXTINF:2.75,
two.ts?sig=x
#EXT-X-ENDLIST
""")
        self.assertEqual(p.duration, 4.0)
        self.assertEqual([s.filename for s in p.segments], ["segment-000000.ts", "segment-000001.ts"])
        self.assertEqual(p.segments[0].url, "http://127.0.0.1:31001/media/one.ts")
        self.assertIn("#EXT-X-ENDLIST", p.local_text)
        self.assertIn("segment-000001.ts", p.local_text)
        self.assertNotIn("http://", p.local_text)

    def test_rejects_unsupported_or_incomplete_playlists(self):
        rejected = [
            "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nchild.m3u8\n#EXT-X-ENDLIST\n",
            "#EXTM3U\n#EXTINF:1,\na.ts\n",
            "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:EVENT\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n",
            "#EXTM3U\n#EXTINF:0,\na.ts\n#EXT-X-ENDLIST\n",
            '#EXTM3U\n#EXT-X-KEY:METHOD=SAMPLE-AES,URI="key"\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-KEY:METHOD=SAMPLE-AES-CTR,URI="key"\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="key",KEYFORMAT="com.apple.streamingkeydelivery"\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="key",IV=0xZZ\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-BYTERANGE:10@0\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-DISCONTINUITY\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
            '#EXTM3U\n#EXT-X-SESSION-KEY:METHOD=AES-128,URI="key"\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n',
        ]
        for body in rejected:
            with self.subTest(body=body):
                with self.assertRaises(HC.HLSCacheError):
                    self.parse(body)

    def test_aes128_keys_ivs_and_none(self):
        p = self.parse("""#EXTM3U
#EXT-X-MEDIA-SEQUENCE:7
#EXTINF:1,
plain.ts
#EXT-X-KEY:METHOD=AES-128,URI="k1.bin"
#EXTINF:1,
a.ts
#EXTINF:1,
b.ts
#EXT-X-KEY:METHOD=AES-128,URI="/abs/k2.bin?t=1",IV=0xAB,KEYFORMAT="identity",KEYFORMATVERSIONS="1"
#EXTINF:1,
c.ts
#EXT-X-KEY:METHOD=NONE
#EXTINF:1,
d.ts
#EXT-X-ENDLIST
""")
        k = [(s.key_uri, s.iv) for s in p.segments]
        self.assertEqual(k[0], (None, None))
        self.assertEqual(k[1], ("http://127.0.0.1:31001/media/k1.bin", (8).to_bytes(16, "big").hex()))
        self.assertEqual(k[2], ("http://127.0.0.1:31001/media/k1.bin", (9).to_bytes(16, "big").hex()))
        self.assertEqual(k[3], ("http://127.0.0.1:31001/abs/k2.bin?t=1", (0xAB).to_bytes(16, "big").hex()))
        self.assertEqual(k[4], (None, None))
        self.assertNotIn("EXT-X-KEY", p.local_text)
        self.assertNotIn("k1.bin", p.local_text)
        plain = p.descriptor["segments"]
        self.assertNotIn("key_uri", plain[0])
        self.assertEqual(plain[3]["iv"], k[3][1])

    def test_descriptor_of_unencrypted_playlist_has_no_key_fields(self):
        p = self.parse('#EXTM3U\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST\n')
        self.assertEqual(set(p.descriptor["segments"][0]), {"index", "duration", "url", "filename"})


class DescriptorMatchTest(unittest.TestCase):
    @staticmethod
    def _desc(token, rk="k", n=2, dur=4.0):
        return {"request_key": rk, "source_url": f"https://a/x.m3u8?t={token}",
                "final_url": f"https://cdn{token}/x.m3u8?t={token}", "manifest_sha256": token,
                "segments": [{"index": i, "duration": dur, "url": f"https://cdn{token}/p{token}/s{i}.ts?sig={token}",
                              "filename": f"segment-{i:06d}.ts"} for i in range(n)]}

    def test_rotated_tokens_still_match(self):
        self.assertTrue(HC._descriptors_match(self._desc("1"), self._desc("2")))

    def test_real_changes_do_not_match(self):
        old = self._desc("1")
        self.assertFalse(HC._descriptors_match(old, self._desc("2", rk="other")))
        self.assertFalse(HC._descriptors_match(old, self._desc("2", n=3)))
        self.assertFalse(HC._descriptors_match(old, self._desc("2", dur=5.0)))
        changed = self._desc("2")
        changed["segments"][1]["url"] = "https://cdn2/p2/other.ts?sig=2"
        self.assertFalse(HC._descriptors_match(old, changed))


class DecryptSegmentTest(unittest.TestCase):
    KEY = bytes(range(16))
    IV = (5).to_bytes(16, "big")

    @staticmethod
    def encrypt(plain, key, iv):
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        padder = padding.PKCS7(128).padder()
        enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
        return enc.update(padder.update(plain) + padder.finalize()) + enc.finalize()

    def test_round_trip_with_every_backend(self):
        plain = b"G" + bytes(range(256)) * 3
        blob = self.encrypt(plain, self.KEY, self.IV)
        self.assertEqual(HC.decrypt_segment(blob, self.KEY, self.IV.hex()), plain)
        import builtins
        real_import = builtins.__import__
        for blocked in (("cryptography",), ("cryptography", "Cryptodome")):
            def fake_import(name, globals=None, *a, **k):
                # Only hide the libraries from hls_cache itself; yt-dlp's own imports must keep working.
                if (globals or {}).get("__name__") == HC.__name__ and name.split(".")[0] in blocked:
                    raise ImportError(name)
                return real_import(name, globals, *a, **k)
            with self.subTest(blocked=blocked), unittest.mock.patch("builtins.__import__", fake_import):
                self.assertEqual(HC.decrypt_segment(blob, self.KEY, self.IV.hex()), plain)

    def test_wrong_key_iv_or_shape_is_rejected(self):
        blob = self.encrypt(b"G" * 400, self.KEY, self.IV)
        for args in ((blob, bytes(16), self.IV.hex()), (blob[:-1], self.KEY, self.IV.hex()),
                     (blob, self.KEY[:8], self.IV.hex()), (b"", self.KEY, self.IV.hex())):
            with self.subTest(args=args[1:]):
                with self.assertRaises(HC.HLSCacheError):
                    HC.decrypt_segment(*args)
        # right padding but not TS
        with self.assertRaises(HC.HLSCacheError):
            HC.decrypt_segment(self.encrypt(b"not ts", self.KEY, self.IV), self.KEY, self.IV.hex())


class CacheServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name) / "hls-cache"
        self.raw = ("#EXTM3U\n#EXT-X-TARGETDURATION:2\n"
                    "#EXTINF:1.25,\none.ts\n#EXTINF:1.75,\ntwo.ts\n#EXT-X-ENDLIST\n")
        self.manifest = HC.parse_media_playlist(self.raw, "http://127.0.0.1:31002/source.m3u8")
        self.metadata = {
            "format": {"format_name": "mpegts", "duration": "9.0", "start_time": "1.4"},
            "streams": [{"codec_type": "video", "duration": "9.0", "nb_frames": "15",
                         "start_pts": 128000, "time_base": "1/90000", "start_time": "1.421"},
                        {"codec_type": "audio", "duration": "8.0"}],
        }
        self.first = self.cache_dir / "first-stage.tmp"
        self.cache_dir.mkdir(parents=True)
        self.first.write_bytes(b"first complete TS")
        self.session, self.ready = HC.prepare_session(
            self.manifest, self.cache_dir, metadata=self.metadata, request_key="request-a", seed_path=self.first)
        self.addCleanup(HC.close_session, self.session.playlist_path)

    def _get(self, segment, headers=None):
        req = urllib.request.Request(self.session.runtime_url.rsplit("/", 1)[0] + "/" + segment, headers=headers or {})
        return urllib.request.urlopen(req, timeout=5)

    def test_missing_segment_waits_then_returns_bytes_to_multiple_readers(self):
        url = self.session.runtime_url.rsplit("/", 1)[0] + "/segment-000001.ts"
        outputs, errors = [], []

        def read_one():
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    outputs.append(response.read())
            except Exception as exc:
                errors.append(exc)

        readers = [threading.Thread(target=read_one) for _ in range(2)]
        for thread in readers:
            thread.start()
        time.sleep(0.15)
        self.assertTrue(all(thread.is_alive() for thread in readers), "reader should wait for downloader commit")
        self.assertFalse(outputs)

        stage = self.cache_dir / "second-stage.tmp"
        stage.write_bytes(b"second complete TS")
        HC.commit_segment(self.session, 1, stage)
        for thread in readers:
            thread.join(timeout=3)
        self.assertFalse(errors)
        self.assertEqual(outputs, [b"second complete TS", b"second complete TS"])

    def test_waiting_reader_gets_non_200_when_cancelled_or_failed(self):
        for state in ("paused", "failed"):
            with self.subTest(state=state):
                # A new generation binds a new ephemeral port after the pause case.
                url = self.session.runtime_url.rsplit("/", 1)[0] + "/segment-000001.ts"
                got = []

                def read_one():
                    try:
                        with urllib.request.urlopen(url, timeout=5) as response:
                            got.append(response.status)
                    except urllib.error.HTTPError as exc:
                        got.append(exc.code)
                    except Exception as exc:
                        got.append(type(exc).__name__)

                reader = threading.Thread(target=read_one)
                reader.start()
                time.sleep(0.08)
                if state == "paused":
                    self.session.cancel()
                else:
                    self.session.set_status("failed", "fixture failure")
                reader.join(timeout=2)
                self.assertFalse(reader.is_alive())
                self.assertEqual(got, [503])
                # Prepare a fresh incomplete generation for the second subcase.
                if state == "paused":
                    HC.close_session(self.session.playlist_path)
                    first = self.cache_dir / "again.tmp"
                    first.write_bytes(b"first complete TS")
                    self.session, _ = HC.prepare_session(self.manifest, self.cache_dir,
                                                          metadata=self.metadata, request_key="request-a",
                                                          seed_path=first)

    def test_range_head_metadata_and_complete_reopen(self):
        stage = self.cache_dir / "second-stage.tmp"
        stage.write_bytes(b"abcdef")
        HC.commit_segment(self.session, 1, stage)
        HC.complete_session(self.session)
        seg_url = self.session.runtime_url.rsplit("/", 1)[0] + "/segment-000001.ts"
        req = urllib.request.Request(seg_url, headers={"Range": "bytes=2-4"})
        with urllib.request.urlopen(req, timeout=3) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-4/6")
            self.assertEqual(response.read(), b"cde")
        req = urllib.request.Request(seg_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as response:
            self.assertEqual(response.headers["Content-Length"], "6")

        sidecar = json.loads(Path(self.session.status_path).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["version"], 1)
        self.assertEqual(sidecar["status"], "complete")
        self.assertEqual(sidecar["segment_count"], 2)
        self.assertEqual(sidecar["metadata"]["format"]["duration"], "3.000000")
        video = sidecar["metadata"]["streams"][0]
        self.assertEqual(video["duration"], "3.000000")
        self.assertNotIn("nb_frames", video)
        self.assertEqual((video["start_pts"], video["time_base"], video["start_time"]),
                         (128000, "1/90000", "1.421"))
        identity, generation = sidecar["identity"], sidecar["generation"]
        path = self.session.playlist_path
        HC.close_session(path)
        reopened = HC.open_session(path)
        self.assertIsNotNone(reopened)
        self.assertEqual(reopened.identity, identity)
        self.assertEqual(reopened.generation, generation)
        self.assertEqual(reopened.status, "complete")

    def test_resume_keeps_identity_changes_generation_and_rejects_corruption(self):
        old_identity, old_generation = self.session.identity, self.session.generation
        HC.close_session(self.session.playlist_path)
        metadata, ready = HC.inspect_cache(self.cache_dir, self.manifest, "request-a")
        self.assertEqual(ready, [0])
        self.assertIsNotNone(metadata)
        resumed, ready = HC.prepare_session(self.manifest, self.cache_dir, metadata=metadata,
                                            request_key="request-a")
        self.assertEqual(ready, [0])
        self.assertEqual(resumed.identity, old_identity)
        self.assertNotEqual(resumed.generation, old_generation)
        HC.close_session(resumed.playlist_path)

        # A re-signed URL for the same media still resumes.
        resigned = HC.parse_media_playlist(self.raw, "http://127.0.0.1:31002/new-signed-url.m3u8")
        self.assertEqual(HC.inspect_cache(self.cache_dir, resigned, "request-a")[1], [0])
        # Different segments or a different request are refused.
        changed = HC.parse_media_playlist(self.raw.replace("two.ts", "other.ts"),
                                          "http://127.0.0.1:31002/source.m3u8")
        with self.assertRaises(HC.HLSCacheError):
            HC.inspect_cache(self.cache_dir, changed, "request-a")
        with self.assertRaises(HC.HLSCacheError):
            HC.inspect_cache(self.cache_dir, self.manifest, "request-b")
        self.assertTrue((self.cache_dir / "segment-000000.ts").is_file())

    def test_live_paused_session_is_not_reopened_as_complete(self):
        # Pause then resume in one process: the paused registry session must not pass for a finished cache.
        HC.fail_session(self.session, "download cancelled", paused=True)
        self.assertIsNone(HC.open_session(self.session.playlist_path))

    def test_complete_cache_with_modified_file_is_never_reopened(self):
        stage = self.cache_dir / "second-stage.tmp"
        stage.write_bytes(b"second complete TS")
        HC.commit_segment(self.session, 1, stage)
        HC.complete_session(self.session)
        path = self.session.playlist_path
        HC.close_session(path)
        (self.cache_dir / "segment-000001.ts").write_bytes(b"tampered")
        self.assertIsNone(HC.open_session(path))

    def test_ready_segment_served_even_when_session_failed_or_paused(self):
        # Segment 0 was seeded and ready in setUp.
        self.session.set_status("failed", "simulated network failure")
        url = self.session.runtime_url.rsplit("/", 1)[0] + "/segment-000000.ts"
        with urllib.request.urlopen(url, timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"first complete TS")

        self.session.set_status("paused", "simulated pause")
        with urllib.request.urlopen(url, timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"first complete TS")

    def test_replace_atomic_retries_on_permission_error(self):
        src = self.cache_dir / "test-src.tmp"
        dst = self.cache_dir / "test-dst.tmp"
        src.write_text("hello", encoding="utf-8")
        dst.write_text("old", encoding="utf-8")

        attempts = [0]
        real_replace = HC.os.replace

        def mock_replace(s, d):
            attempts[0] += 1
            if attempts[0] < 3:
                raise PermissionError(13, "Access is denied")
            return real_replace(s, d)

        orig = HC.os.replace
        HC.os.replace = mock_replace
        try:
            HC._replace_atomic(src, dst, max_attempts=5, backoff=0.01)
        finally:
            HC.os.replace = orig

        self.assertEqual(attempts[0], 3)
        self.assertEqual(dst.read_text(encoding="utf-8"), "hello")

    def test_dynamic_token_in_comments_allows_cache_resume(self):
        # Some CDNs insert a dynamic comment/tag like #EXT-X-TOKEN=xyz on every request.
        # Descriptors should match and allow resume as long as segments, URLs, and request_key match.
        raw_with_token = ("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-TOKEN=random123\n"
                          "#EXTINF:1.25,\none.ts\n#EXTINF:1.75,\ntwo.ts\n#EXT-X-ENDLIST\n")
        manifest2 = HC.parse_media_playlist(raw_with_token, "http://127.0.0.1:31002/source.m3u8")
        metadata, ready = HC.inspect_cache(self.cache_dir, manifest2, "request-a")
        self.assertEqual(ready, [0])
        self.assertIsNotNone(metadata)


if __name__ == "__main__":
    unittest.main()
