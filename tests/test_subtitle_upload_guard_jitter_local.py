"""Upload-size guard, backoff jitter, and Ollama / LM Studio (keyless localhost) behaviour. No network."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.subtitles import cloud_chunks, fallback, slice_results
from oh_streamer.subtitles import model_adapters as adapters
from oh_streamer.subtitles import model_registry as registry


class UploadSizeGuardTests(unittest.TestCase):
    def _run(self, limit, slice_len=300.0):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls = []

        def fake_extract(source_path, out_path, start_sec, duration_sec, fmt="flac", ffmpeg=None):
            calls.append((Path(out_path).name, start_sec, duration_sec))
            Path(out_path).write_bytes(b"x" * int(duration_sec))  # 1 byte per second
            return Path(out_path)

        with patch.object(cloud_chunks, "extract_chunk_audio", side_effect=fake_extract),                 patch.object(slice_results, "CLOUD_UPLOAD_MAX_BYTES", limit):
            pieces = slice_results._audio_pieces("a.mp4", Path(tmp.name) / "slice_001.flac", 100.0, slice_len)
        names = [Path(f).name for f, _ in pieces]
        # one fake ASR segment per piece at t=1, shifted by the piece offset like the scheduler does
        segs = [slice_results._shift_segment({"start": 1.0, "end": 2.0}, off) for _, off in pieces]
        return segs, calls, names

    def test_normal_slice_unchanged(self):
        segs, calls, seen = self._run(limit=10_000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(seen, ["slice_001.flac"])
        self.assertEqual(segs[0]["start"], 1.0)

    def test_oversized_slice_split_with_correct_offsets(self):
        segs, calls, seen = self._run(limit=150, slice_len=300.0)  # 300 B > 150 -> halves of 150 B (not > limit)
        self.assertEqual(seen, ["slice_001_p0.flac", "slice_001_p1.flac"])
        self.assertEqual([s["start"] for s in segs], [1.0, 151.0])  # offsets relative to slice start
        starts = {name: start for name, start, _ in calls}
        self.assertEqual(starts["slice_001_p1.flac"], 250.0)  # absolute source time = 100 + 150

    def test_recursive_split_stops_at_minimum(self):
        segs, calls, seen = self._run(limit=1, slice_len=200.0)  # always oversized
        self.assertTrue(all(c[2] >= slice_results.CLOUD_UPLOAD_MIN_SPLIT_SECONDS for c in calls))
        self.assertEqual(len(seen), 4)
        offsets = [s["start"] for s in segs]
        self.assertEqual(offsets, sorted(offsets))

    def test_shift_segment_shifts_words_and_ignores_zero_offset(self):
        seg = {"start": 1.0, "end": 2.0, "words": [{"start": 1.0, "end": 1.5}]}
        self.assertIs(slice_results._shift_segment(seg, 0.0), seg)
        shifted = slice_results._shift_segment(seg, 10.0)
        self.assertEqual((shifted["start"], shifted["words"][0]["end"]), (11.0, 11.5))
        self.assertEqual(seg["start"], 1.0)


class BackoffJitterTests(unittest.TestCase):
    def test_equal_jitter_bounds(self):
        with patch.object(fallback, "_random", lambda: 0.0):
            self.assertEqual(fallback.backoff_delay(0), 1.0)
            self.assertEqual(fallback.backoff_delay(2), 4.0)
        with patch.object(fallback, "_random", lambda: 0.999999):
            self.assertAlmostEqual(fallback.backoff_delay(0), 2.0, places=3)
            self.assertLessEqual(fallback.backoff_delay(10), 30.0)
        with patch.object(fallback, "_random", lambda: 0.0):
            self.assertEqual(fallback.backoff_delay(10), 15.0)  # base capped at 30

    def test_retry_after_is_not_jittered(self):
        # Existing PolicyTests assert exact Retry-After waits; here make sure the helper is not on that path.
        import inspect
        src = inspect.getsource(fallback)
        self.assertIn("float(exc.retry_after) if exc.retry_after is not None else backoff_delay(retry)", src)


class LocalPresetTests(unittest.TestCase):
    def test_local_base_url_detection(self):
        for url in ("http://localhost:11434/v1", "http://127.0.0.1:1234/v1", "http://[::1]:1234/v1"):
            self.assertTrue(registry.is_local_base_url(url), url)
        for url in ("https://api.groq.com/openai/v1", "http://192.168.1.5:11434/v1", "http://localhost.evil.com/v1", ""):
            self.assertFalse(registry.is_local_base_url(url), url)

    def test_http_only_allowed_for_loopback(self):
        row = adapters.validate_spec(dict(kind="translation", provider="openai-compatible",
                                          base_url="http://localhost:11434/v1", remote_model="qwen2.5:7b"))
        self.assertEqual(row["base_url"], "http://localhost:11434/v1")
        with self.assertRaises(ValueError):
            adapters.validate_spec(dict(kind="translation", provider="openai-compatible",
                                        base_url="http://192.168.1.5:11434/v1", remote_model="m"))

    def test_empty_key_sends_no_authorization_header(self):
        row = dict(provider="openai-compatible", base_url="http://localhost:1234/v1", remote_model="m")
        captured = {}

        def fake_post(url, payload, headers, transport=None, **_kwargs):
            captured.update(url=url, headers=headers)
            return {"choices": []}

        with patch.object(adapters, "http_post", fake_post):
            adapters._request(row, "/chat/completions", b"{}", "", "application/json")
            self.assertEqual(captured["url"], "http://localhost:1234/v1/chat/completions")
            self.assertNotIn("Authorization", captured["headers"])
            adapters._request(row, "/chat/completions", b"{}", "k", "application/json")
            self.assertEqual(captured["headers"]["Authorization"], "Bearer k")

    def _save(self, base_url, api_key=""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = {"model_registry_path": str(Path(tmp.name) / "models.json"), "subtitle_cloud_enabled": True}
        spec = dict(api_family="openai-compatible", base_url=base_url,
                    models={"translation": "qwen2.5:7b"}, api_key=api_key)
        with patch.object(registry, "_env_key_for_model", return_value=""), \
                patch.object(adapters, "test_model") as tm:
            return registry.test_save_connection(spec, cfg), tm

    def test_connection_with_empty_key_ok_for_localhost(self):
        res, tm = self._save("http://localhost:11434/v1")
        self.assertTrue(res["ok"], res)
        tm.assert_called_once()
        self.assertEqual(tm.call_args.kwargs["api_key"], "")
        self.assertEqual(res["connection"]["auth_source"], "keyless")

    def test_connection_with_empty_key_still_rejected_for_cloud(self):
        res, tm = self._save("https://api.example.com/v1")
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "missing_credential")
        tm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
