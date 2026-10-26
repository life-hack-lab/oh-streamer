"""Unit and integration tests for Milestone 1 & 2:
- Milestone 1: HTTP header parsing, CloudAsrRatePacer (RPM + ASH ledger), Groq dual-model relay pool.
- Milestone 2: Silence stripping chunk planning, 16kHz mono FLAC compression and timeline reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import wave

from oh_streamer.subtitles import cloud_chunks, model_adapters as adapters
from oh_streamer.subtitles.cloud_chunks import (
    ChunkCheckpoint,
    ChunkSpec,
    extract_chunk_audio,
    plan_chunks,
    reconstruct_timeline,
)
from oh_streamer.subtitles.model_adapters import GROQ_SIBLING_MODELS, ModelAdapterError
from oh_streamer.subtitles.providers.http import (
    CloudAsrRatePacer,
    ProviderError,
    _parse_retry_after,
    get_cloud_asr_pacer,
    http_post,
    parse_retry_after,
)


class Milestone1HttpHeaderAndRatePacerTests(unittest.TestCase):
    def test_parse_retry_after_and_ratelimit_reset_headers(self):
        # Plain numeric values
        self.assertEqual(_parse_retry_after("30"), 30.0)
        self.assertEqual(_parse_retry_after("2.5"), 2.5)
        self.assertEqual(_parse_retry_after(10), 10.0)
        self.assertIsNone(_parse_retry_after(""))
        self.assertIsNone(_parse_retry_after(None))

        # Groq compound duration strings
        self.assertEqual(_parse_retry_after("2.5s"), 2.5)
        self.assertEqual(_parse_retry_after("200ms"), 0.2)
        self.assertEqual(_parse_retry_after("1m30s"), 90.0)
        self.assertEqual(_parse_retry_after("6m0s"), 360.0)
        self.assertEqual(_parse_retry_after("1h2m3s"), 3723.0)

        # Header mappings
        self.assertEqual(_parse_retry_after({"retry-after": "15"}), 15.0)
        self.assertEqual(_parse_retry_after({"x-ratelimit-reset-requests": "2.5s"}), 2.5)
        self.assertEqual(_parse_retry_after({"x-ratelimit-reset-tokens": "1m30s"}), 90.0)
        self.assertEqual(_parse_retry_after({"x-ratelimit-reset-audio": "3600s"}), 3600.0)

        # parse_retry_after alias
        self.assertEqual(parse_retry_after({"x-ratelimit-reset-requests": "6m0s"}), 3600.0 / 10.0)

    def test_http_post_429_retry_after_threshold(self):
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        payload = b"{}"

        # 429 with retry_after <= 90s is retryable
        def t_429_short(req, to):
            raise urllib.error.HTTPError(url, 429, "Rate limit", {"x-ratelimit-reset-requests": "30s"}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_429_short)
        self.assertEqual(ctx.exception.kind, "rate_limit")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.retry_after, 30.0)

        # 429 with retry_after > 90s raises non-retryable error to trigger sibling model fallback
        def t_429_long(req, to):
            raise urllib.error.HTTPError(url, 429, "Rate limit", {"x-ratelimit-reset-requests": "120s"}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_429_long)
        self.assertEqual(ctx.exception.kind, "rate_limit")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(ctx.exception.retry_after, 120.0)

    def test_cloud_asr_rate_pacer_ash_ledger_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_file = Path(tmp_dir) / "test_ledger.json"
            now = 10000.0
            pacer = CloudAsrRatePacer(
                "https://api.groq.com/openai/v1",
                ledger_path=ledger_file,
                clock=lambda: now,
                safe_rpm=18.0,
            )

            # Initially 0 audio seconds used
            self.assertEqual(pacer.get_hourly_ash("whisper-large-v3"), 0.0)
            afford, wait_s = pacer.can_afford("whisper-large-v3", 300.0, ash_limit=7000.0)
            self.assertTrue(afford)
            self.assertEqual(wait_s, 0.0)

            # Record 6800s of audio usage
            pacer.record_usage("whisper-large-v3", 6800.0, ts=now)
            self.assertEqual(pacer.get_hourly_ash("whisper-large-v3"), 6800.0)

            # Now requesting 300s exceeds 7000s limit (6800 + 300 = 7100 > 7000)
            afford, wait_s = pacer.can_afford("whisper-large-v3", 300.0, ash_limit=7000.0)
            self.assertFalse(afford)
            self.assertAlmostEqual(wait_s, 3600.0, delta=1.0)

            # Verify persistent ledger was written to disk
            self.assertTrue(ledger_file.exists())
            with open(ledger_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("https://api.groq.com/openai/v1::whisper-large-v3", saved)

            # A new pacer instance pointing to same file restores the state
            pacer2 = CloudAsrRatePacer(
                "https://api.groq.com/openai/v1",
                ledger_path=ledger_file,
                clock=lambda: now,
            )
            self.assertEqual(pacer2.get_hourly_ash("whisper-large-v3"), 6800.0)

            # After 3600 seconds, the old entries expire
            now += 3601.0
            self.assertEqual(pacer2.get_hourly_ash("whisper-large-v3"), 0.0)
            afford, wait_s = pacer2.can_afford("whisper-large-v3", 300.0, ash_limit=7000.0)
            self.assertTrue(afford)


class Milestone1DualModelRelayTests(unittest.TestCase):
    def test_cloud_asr_rate_pacer_and_fallback(self):
        """Verify:
        1. When whisper-large-v3 reaches 6,900s ASH limit, it automatically switches to whisper-large-v3-turbo.
        2. When receiving 429 with retry_after > 30s, it automatically switches to sibling model.
        3. Both models share the exact same checkpoint directory.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "input.wav"
            # 605s dummy wav (produces 3 chunks)
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000 * 605)

            ledger_file = root / "ledger.json"
            pacer = CloudAsrRatePacer(
                "https://api.groq.com/openai/v1",
                ledger_path=ledger_file,
            )

            # Pre-load whisper-large-v3 to 6800s.
            # Next 300s chunk would push it to 7100s >= 6900s safe limit.
            pacer.record_usage("whisper-large-v3", 6800.0)

            row = {
                "kind": "asr",
                "provider": "openai-compatible",
                "remote_model": "whisper-large-v3",
                "base_url": "https://api.groq.com/openai/v1",
            }

            models_called = []
            phases = []

            def mock_request(req_row, route, payload, key, content_type):
                models_called.append(req_row["remote_model"])
                return {"segments": [{"start": 0.0, "end": 2.0, "text": "測試"}]}

            with patch.object(adapters, "_request", side_effect=mock_request):
                result = adapters._cloud_asr(
                    row,
                    source,
                    root / "out",
                    pacer=pacer,
                    on_phase=lambda *args: phases.append(args),
                )

            # Verified: primary was full (6800 + 300 >= 6900), automatically switched to turbo!
            self.assertIn("whisper-large-v3-turbo", models_called)
            self.assertEqual(models_called[0], "whisper-large-v3-turbo")

            # Check that detail message notified UI of switch
            switch_phases = [p for p in phases if "自動切換至 whisper-large-v3-turbo" in p[2]]
            self.assertTrue(len(switch_phases) > 0)

            # Verify checkpoint directory was populated and shared
            ckpt_dir = root / "out/chunks"
            self.assertTrue((ckpt_dir / "chunk_0000.json").exists())
            self.assertTrue((ckpt_dir / "chunk_0001.json").exists())

    def test_fallback_on_429_rate_limit_error(self):
        """When whisper-large-v3 receives 429 (retry_after > 30s), seamlessly fall back to turbo."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "input.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000 * 100)

            row = {
                "kind": "asr",
                "provider": "openai-compatible",
                "remote_model": "whisper-large-v3",
                "base_url": "https://api.groq.com/openai/v1",
            }

            calls = []

            def mock_request(req_row, route, payload, key, content_type):
                calls.append(req_row["remote_model"])
                if req_row["remote_model"] == "whisper-large-v3":
                    raise ProviderError("rate_limit", "Rate limit", retryable=False, retry_after=60.0, http_status=429)
                return {"segments": [{"start": 0.0, "end": 2.0, "text": "由備援模型辨識成功"}]}

            with patch.object(adapters, "_request", side_effect=mock_request):
                result = adapters._cloud_asr(row, source, root / "out")

            self.assertEqual(calls, ["whisper-large-v3", "whisper-large-v3-turbo"])
            self.assertEqual(result["segments"][0]["text"], "由備援模型辨識成功")


class Milestone2SilenceStrippingAndFlacTests(unittest.TestCase):
    def test_cloud_chunks_silence_skipping_and_flac(self):
        """Verify:
        1. plan_chunks with skip_long_silence_sec=10.0 strips silence >= 10s with 0.4s breathing buffer.
        2. reconstruct_timeline aligns subtitles accurately with sub-millisecond precision.
        3. extract_chunk_audio outputs a valid 16kHz mono FLAC file.
        """
        duration = 100.0
        # 30-second continuous silence from 30.0s to 60.0s
        silences = [(30.0, 60.0)]
        chunks = plan_chunks(
            duration,
            target_chunk_sec=300.0,
            silence_intervals=silences,
            skip_long_silence_sec=10.0,
        )

        # Expected:
        # Long silence from 30.0 to 60.0 excluded with 0.4s padding: [30.4, 59.6]
        # Chunk 0: [0.0, 30.4]
        # Chunk 1: [59.6, 100.0]
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0][1], 0.0, places=3)
        self.assertAlmostEqual(chunks[0][2], 30.4, places=3)
        self.assertAlmostEqual(chunks[1][1], 59.6, places=3)
        self.assertAlmostEqual(chunks[1][2], 100.0, places=3)

        # Verify timeline reconstruction
        c0_segments = [{"start": 5.0, "end": 8.0, "text": "前段對白"}]
        c1_segments = [{"start": 1.5, "end": 4.5, "text": "後段對白"}]

        reconstructed = reconstruct_timeline(chunks, [c0_segments, c1_segments])
        self.assertEqual(len(reconstructed), 2)

        # Segment 0 was in Chunk 0 (start offset = 0.0) -> start: 5.0, end: 8.0
        self.assertAlmostEqual(reconstructed[0]["start"], 5.0, places=3)
        self.assertAlmostEqual(reconstructed[0]["end"], 8.0, places=3)
        self.assertEqual(reconstructed[0]["text"], "前段對白")

        # Segment 1 was in Chunk 1 (start offset = 59.6) -> start: 59.6 + 1.5 = 61.1, end: 59.6 + 4.5 = 64.1
        self.assertAlmostEqual(reconstructed[1]["start"], 61.1, places=3)
        self.assertAlmostEqual(reconstructed[1]["end"], 64.1, places=3)
        self.assertEqual(reconstructed[1]["text"], "後段對白")

        # Verify FLAC extraction
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_wav = tmp / "test.wav"
            out_flac = tmp / "test.flac"

            # Generate 1.0s tone in WAV
            import math, struct
            rate = 16000
            samples = int(rate * 1.0)
            data = bytearray(samples * 2)
            for i in range(samples):
                val = int(8000 * math.sin(2 * math.pi * 440.0 * (i / rate)))
                struct.pack_into("<h", data, i * 2, val)
            with wave.open(str(source_wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(bytes(data))

            flac_path = extract_chunk_audio(source_wav, out_flac, 0.0, 1.0, fmt="flac")
            self.assertTrue(flac_path.exists())
            header = flac_path.read_bytes()[:4]
            # Valid FLAC stream marker is b"fLaC"
            self.assertEqual(header, b"fLaC")


if __name__ == "__main__":
    unittest.main()
