"""Unit and contract tests for M1: HTTP error classification, RequestGate, and cache/storm protection."""

import json
from pathlib import Path
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from oh_streamer.subtitles.providers.http import (
    ProviderError,
    RequestGate,
    http_post,
    parse_retry_after,
)
from oh_streamer.subtitles import core, cli
from oh_streamer.subtitles.core import Segment


class HttpTransportAndGateTests(unittest.TestCase):
    def test_parse_retry_after_numeric_and_http_date(self):
        # Numeric seconds
        self.assertEqual(parse_retry_after("42"), 42.0)
        self.assertEqual(parse_retry_after("0"), 0.0)
        self.assertIsNone(parse_retry_after(""))
        self.assertIsNone(parse_retry_after(None))

        # HTTP Date
        fixed_now = 1700000000.0  # Tue, 14 Nov 2023 22:13:20 GMT
        date_str = "Tue, 14 Nov 2023 22:14:20 GMT"  # +60 seconds
        parsed = parse_retry_after(date_str, now_ts=fixed_now)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed, 60.0, delta=1.0)

    def test_http_post_error_classification(self):
        url = "https://api.example.com/v1/chat"
        payload = b'{"test": 1}'

        # 401 Auth
        def t_401(req, to):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_401)
        self.assertEqual(ctx.exception.kind, "auth")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(ctx.exception.http_status, 401)

        # 403 Auth
        def t_403(req, to):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_403)
        self.assertEqual(ctx.exception.kind, "auth")
        self.assertFalse(ctx.exception.retryable)

        # 429 Rate limit with short Retry-After
        def t_429_short(req, to):
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"retry-after": "15"}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_429_short)
        self.assertEqual(ctx.exception.kind, "rate_limit")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.retry_after, 15.0)
        self.assertEqual(ctx.exception.scope, "minute")

        # 429 Daily quota with long Retry-After (> 300s)
        def t_429_long(req, to):
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"retry-after": "3600"}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_429_long)
        self.assertEqual(ctx.exception.kind, "quota_daily")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(ctx.exception.scope, "day")

        # 500 / 503 Server error
        def t_503(req, to):
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_503)
        self.assertEqual(ctx.exception.kind, "server")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.http_status, 503)

        # Network / DNS error
        def t_net(req, to):
            raise urllib.error.URLError("getaddrinfo failed")

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_net)
        self.assertEqual(ctx.exception.kind, "network")
        self.assertTrue(ctx.exception.retryable)

        # Timeout
        def t_timeout(req, to):
            raise TimeoutError("timed out")

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_timeout)
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertTrue(ctx.exception.retryable)

        # Malformed non-JSON
        def t_non_json(req, to):
            return 200, {}, b"<html>Not JSON</html>"

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, payload, transport=t_non_json)
        self.assertEqual(ctx.exception.kind, "bad_response")
        self.assertFalse(ctx.exception.retryable)

    def test_no_credential_leak_in_error_message(self):
        url = "https://api.example.com/v1/chat?secret_param=SUPER_SECRET_TOKEN"
        secret_key = "sk-super-secret-key-12345"

        def t_fail(req, to):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        with self.assertRaises(ProviderError) as ctx:
            http_post(url, b"payload", headers={"Authorization": f"Bearer {secret_key}"}, transport=t_fail)

        err_str = str(ctx.exception)
        self.assertNotIn(secret_key, err_str)
        self.assertNotIn("SUPER_SECRET_TOKEN", err_str)

    def test_request_gate_rpm_and_token_sliding_window(self):
        simulated_time = [1000.0]

        def fake_clock():
            return simulated_time[0]

        sleep_calls = []

        def fake_sleep(sec):
            sleep_calls.append(sec)
            simulated_time[0] += sec

        gate = RequestGate(
            rpm=2,
            clock=fake_clock,
            sleep_fn=fake_sleep,
            identity="test_gate",
        )

        # 1st request at 1000.0 -> OK
        gate.acquire(requests=1)
        self.assertEqual(len(sleep_calls), 0)

        # 2nd request at 1000.0 -> OK (reached rpm=2)
        gate.acquire(requests=1)
        self.assertEqual(len(sleep_calls), 0)

        # 3rd request at 1000.0 -> Must wait 60s to clear the first request
        # max_wait defaults to 30.0s, so 60.0s wait will exceed max_wait and raise ProviderError('rate_limit')
        with self.assertRaises(ProviderError) as ctx:
            gate.acquire(requests=1, max_wait=30.0)
        self.assertEqual(ctx.exception.kind, "rate_limit")
        self.assertEqual(ctx.exception.scope, "minute")

        # But if max_wait is 65.0s, it will sleep for 60.0s and succeed
        gate.acquire(requests=1, max_wait=65.0)
        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 60.0)
        self.assertAlmostEqual(simulated_time[0], 1060.0)

    def test_request_gate_daily_quota_persistence_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            quota_file = Path(tmp_dir) / "quota.json"
            simulated_time = [1700000000.0]

            def fake_clock():
                return simulated_time[0]

            # 1st gate instance with rpd=2
            gate1 = RequestGate(
                rpd=2,
                quota_file=quota_file,
                identity="user_model_1",
                clock=fake_clock,
            )
            gate1.acquire(requests=1)
            gate1.acquire(requests=1)

            # Re-instantiate (simulating process restart)
            gate2 = RequestGate(
                rpd=2,
                quota_file=quota_file,
                identity="user_model_1",
                clock=fake_clock,
            )
            # 3rd request should immediately hit daily limit
            with self.assertRaises(ProviderError) as ctx:
                gate2.acquire(requests=1)
            self.assertEqual(ctx.exception.kind, "quota_daily")
            self.assertEqual(ctx.exception.scope, "day")

            # Advance time by 1 day (86400 seconds) -> new day resets count
            simulated_time[0] += 86400.0
            gate2.acquire(requests=1)  # should succeed now!


class RequestStormAndBadCacheRegressionTests(unittest.TestCase):
    def test_request_storm_prevented_on_fatal_401_and_429(self):
        """Verifies that an auth error or fatal quota does NOT cause fallback retry storms."""
        segments = [
            Segment(start=float(i), end=float(i + 1), text=f"句子{i}")
            for i in range(25)
        ]
        call_count = [0]

        def batch_fail(b):
            call_count[0] += 1
            raise ProviderError("auth", "Auth failed", retryable=False, http_status=401)

        def one_fail(text):
            call_count[0] += 1
            raise ProviderError("auth", "Auth failed", retryable=False, http_status=401)

        with self.assertRaises(ProviderError):
            core.translate_segments(segments, batch_fail, one_fail, batch_size=20)

        # If request storm were happening, it would call batch_fail once, then one_fail 20 times!
        # Because we abort immediately on fatal ProviderError, total calls must be exactly 1!
        self.assertEqual(call_count[0], 1)

    def test_bad_cache_solidification_prevented(self):
        """Verifies that when translation fails with [原文], result is not cached and memo is preserved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            source_file = out_dir / "ja.json"
            # 40 segments -> 2 batches with batch_size=20
            source_data = [
                {"start": float(i), "end": float(i + 1), "text": f"こんにちは_{i}"}
                for i in range(40)
            ]
            with open(source_file, "w", encoding="utf-8") as f:
                json.dump(source_data, f)

            result_path = out_dir / "sakura.zh-TW.json"

            # Create a mock client that succeeds on the 1st batch but fails with ProviderError on 2nd
            batch_calls = [0]

            class MockClient:
                model = "sakura"
                translate_model = "sakura-v1"
                metrics = dict(requests=0, failures=0, prompt_tokens=0, completion_tokens=0, total_tokens=0, tokens=0, elapsed_seconds=0.0)

                def translate(self, prompt, max_tokens=None):
                    batch_calls[0] += 1
                    if batch_calls[0] == 1:
                        # 1st batch of 20 items succeeds
                        return "\n".join(f"{i}. 你好_{i}" for i in range(1, 21))
                    raise ProviderError("auth", "Token expired", retryable=False, http_status=401)

            # Running translate_file should raise and NOT write result_path
            with self.assertRaises(Exception):
                cli.translate_file(
                    source_file,
                    out_dir,
                    model="sakura",
                    model_path=None,
                    llama_server=None,
                    batch_size=20,
                    gpu_managed=True,
                    shared_client=MockClient(),
                    model_identity="test_remote",
                )

            # Check that result_path was NOT written!
            # The bad translation must NOT be solidified into cache.
            self.assertFalse(result_path.exists())

            # Check that memo still preserved the 1st successful translation
            memo_file = out_dir / "sakura.translate.partial.json"
            self.assertTrue(memo_file.exists())
            with open(memo_file, "r", encoding="utf-8") as f:
                memo_data = json.load(f)
            self.assertGreater(len(memo_data.get("calls", {})), 0)


class TooLargeAndMismatchTests(unittest.TestCase):
    def post(self, status, body):
        return http_post('https://example.org/v1/test', b'{}', transport=lambda *_: (status, {}, json.dumps(body).encode()))

    def test_request_too_large_429_is_bad_response_with_flag(self):
        body = {"error": {"message": "Request too large for model `x` in organization `o` service tier `on_demand` "
                          "on tokens per minute (TPM): Limit 6000, Requested 9000, please reduce your message size",
                          "type": "tokens", "code": "rate_limit_exceeded"}}
        with self.assertRaises(ProviderError) as raised:
            self.post(429, body)
        exc = raised.exception
        self.assertEqual(exc.kind, "bad_response")
        self.assertTrue(exc.too_large)
        self.assertEqual(exc.http_status, 429)

    def test_normal_429_stays_rate_limit(self):
        body = {"error": {"message": "Rate limit reached for model `x` on tokens per minute (TPM): Limit 6000, "
                          "Used 5900, Requested 500. Please try again in 3s.", "type": "tokens",
                          "code": "rate_limit_exceeded"}}
        with self.assertRaises(ProviderError) as raised:
            self.post(429, body)
        self.assertEqual(raised.exception.kind, "rate_limit")
        self.assertFalse(getattr(raised.exception, "too_large", False))

    def test_numbered_mismatch_counts(self):
        out = "\n".join(f"{i}. 行{i}" for i in range(1, 32))
        m = core.numbered_mismatch(out, 30)
        self.assertEqual((m["expected"], m["got"], m["first_bad"], m["unnumbered"]), (30, 31, 31, 0))
        m = core.numbered_mismatch("1. a\n(note)\n2. b", 2)
        self.assertEqual((m["got"], m["unnumbered"], m["first_bad"]), (2, 1, 0))


if __name__ == "__main__":
    unittest.main()
