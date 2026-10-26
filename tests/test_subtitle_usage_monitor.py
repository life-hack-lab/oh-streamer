import os
import tempfile
import unittest
import urllib.request
from pathlib import Path

from oh_streamer.subtitles import usage_monitor as um
from oh_streamer.subtitles.providers.http import ProviderError, http_post

T0 = 1_800_000_000.0  # fixed fake clock
ASR_H = {'x-ratelimit-limit-requests': '2000', 'x-ratelimit-remaining-requests': '1990',
         'x-ratelimit-reset-requests': '43.2s'}
CHAT_H = {'x-ratelimit-limit-requests': '1000', 'x-ratelimit-remaining-requests': '900',
          'x-ratelimit-reset-requests': '1m26.4s', 'x-ratelimit-limit-tokens': '10000',
          'x-ratelimit-remaining-tokens': '9000', 'x-ratelimit-reset-tokens': '134ms'}
ROW = {'api_family': 'groq', 'base_url': 'https://api.groq.com/openai/v1', 'connection_id': 'c1',
       'remote_model': 'whisper-large-v3'}


class UsageMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        um.set_db_path(Path(self.tmp.name) / 'u.sqlite3')
        um.set_audio_limits(7200.0, 28800.0)

    def tearDown(self):
        um.set_db_path(None)
        um._audio_limits.clear()
        self.tmp.cleanup()

    def model(self, snap, name):
        return next(m for p in snap['providers'] for m in p['models'] if m['model'] == name)

    def test_header_parsing(self):
        p = um.parse_rate_limit_headers({k.upper(): v for k, v in CHAT_H.items()})
        self.assertEqual(p['requests_limit'], 1000)
        self.assertEqual(p['tokens_remaining'], 9000)
        self.assertAlmostEqual(p['requests_reset_s'], 86.4)
        self.assertAlmostEqual(p['tokens_reset_s'], 0.134)
        self.assertAlmostEqual(um.parse_rate_limit_headers(ASR_H)['requests_reset_s'], 43.2)
        self.assertIsNone(um.parse_rate_limit_headers(ASR_H)['tokens_limit'])
        self.assertIsNone(um.parse_rate_limit_headers(None)['requests_limit'])

    def test_min_billing_and_windows(self):
        um.record_response('c1', 'w', 'asr', 200, ASR_H, {}, audio_s=3.0, now=T0)
        um.record_response('c1', 'w', 'asr', 200, ASR_H, {}, audio_s=100.0, now=T0 + 10)
        m = self.model(um.usage_snapshot(T0 + 20), 'w')
        self.assertEqual(m['audio']['hour']['used_s'], 110.0)
        # hour window rolls off, day window keeps it
        m = self.model(um.usage_snapshot(T0 + 3600 + 120), 'w')
        self.assertEqual(m['audio']['hour']['used_s'], 0.0)
        self.assertEqual(m['audio']['day']['used_s'], 110.0)
        m = self.model(um.usage_snapshot(T0 + 86400 + 120), 'w')
        self.assertEqual(m['audio']['day']['used_s'], 0.0)

    def test_pruning(self):
        um.record_response('c1', 'w', 'asr', 200, ASR_H, {}, audio_s=20, now=T0)
        um.record_response('c1', 'w', 'asr', 200, ASR_H, {}, audio_s=20, now=T0 + um.HOUR_RETENTION_S + 7200)
        con = um._connect()
        self.assertEqual(con.execute('SELECT COUNT(*) FROM minute_buckets').fetchone()[0], 1)
        self.assertEqual(con.execute('SELECT COUNT(*) FROM hour_buckets').fetchone()[0], 1)
        con.close()

    def test_snapshot_contract_chat_and_series(self):
        um.record_response('c1', 'qwen', 'chat', 200, CHAT_H, {'usage': {'total_tokens': 123}}, now=T0)
        um.record_response('c1', 'qwen', 'chat', 429, CHAT_H, None, now=T0 + 5)
        um.record_response('c1', 'whisper-large-v3', 'asr', 200, ASR_H, {}, audio_s=30, now=T0 + 7)
        snap = um.usage_snapshot(T0 + 10)
        self.assertEqual(set(snap), {'version', 'generated_at', 'providers'})
        self.assertEqual(snap['version'], 1)
        prov = snap['providers'][0]
        self.assertEqual(set(prov), {'provider', 'connection_id', 'models', 'series'})
        q = self.model(snap, 'qwen')
        self.assertEqual(set(q), {'model', 'kind', 'requests', 'tokens', 'tokens_day', 'requests_minute',
                                  'audio', 'limits_source', 'warning', 'updated_at'})
        self.assertEqual(q['kind'], 'chat')
        self.assertEqual(q['tokens']['limit'], 10000)
        self.assertAlmostEqual(q['requests']['reset_in_s'], 86.4 - 10 + 5, places=2)
        self.assertEqual(q['warning'], 'ok')
        self.assertEqual(q['tokens_day']['used'], 123)
        self.assertEqual(q['requests_minute']['used'], 2)
        self.assertIsNone(q['audio']['hour']['limit_s'])
        self.assertIs(q['audio']['limits_verified'], False)
        w = self.model(snap, 'whisper-large-v3')
        self.assertEqual(w['audio']['hour'], {'used_s': 30.0, 'limit_s': 7200.0})
        self.assertEqual(w['audio']['day'], {'used_s': 30.0, 'limit_s': 28800.0})
        self.assertIs(w['audio']['limits_verified'], True)
        self.assertIsNone(w['tokens']['limit'])
        self.assertEqual(w['requests_minute']['limit'], 20)
        self.assertEqual(w['requests_minute']['used'], 1)
        self.assertEqual(w['limits_source'], 'mixed')
        pts = prov['series']['points']
        self.assertEqual(prov['series']['granularity'], 'hour')
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0], {'t': int(T0 // 3600) * 3600, 'tokens': 123, 'audio_s': 30.0, 'requests': 3})

    def test_official_limits_and_cached_tokens(self):
        # cached tokens deduction
        body = {
            'usage': {
                'total_tokens': 100,
                'prompt_tokens_details': {'cached_tokens': 40}
            }
        }
        um.record_response('c1', 'qwen/qwen3.8-27b', 'chat', 200, None, body, now=T0)
        snap = um.usage_snapshot(T0 + 1)
        m = self.model(snap, 'qwen/qwen3.8-27b')
        self.assertEqual(m['tokens_day']['used'], 60)
        self.assertEqual(m['tokens_day']['limit'], 200000)
        self.assertEqual(m['tokens']['limit'], 8000)  # from official table
        self.assertEqual(m['requests']['limit'], 1000)  # from official table
        self.assertEqual(m['requests_minute']['limit'], 30)  # from official table
        self.assertEqual(m['limits_source'], 'table')

        # override official limits
        um.set_official_limits('qwen/qwen3.8-27b', {'rpm': 50, 'tpd': 300000})
        snap2 = um.usage_snapshot(T0 + 1)
        m2 = self.model(snap2, 'qwen/qwen3.8-27b')
        self.assertEqual(m2['requests_minute']['limit'], 50)
        self.assertEqual(m2['tokens_day']['limit'], 300000)
        um.set_official_limits('qwen/qwen3.8-27b', None)

    def test_warning_levels(self):
        h = dict(CHAT_H, **{'x-ratelimit-remaining-tokens': '1500'})
        um.record_response('c1', 'a', 'chat', 200, h, {}, now=T0)
        h = dict(CHAT_H, **{'x-ratelimit-remaining-tokens': '400'})
        um.record_response('c1', 'b', 'chat', 200, h, {}, now=T0)
        um.record_response('c1', 'c', 'chat', 200, {}, {}, now=T0)
        um.record_response('c1', 'w', 'asr', 200, {}, {}, audio_s=6900, now=T0)  # 4% hour left
        snap = um.usage_snapshot(T0)
        self.assertEqual(self.model(snap, 'a')['warning'], 'low')
        self.assertEqual(self.model(snap, 'b')['warning'], 'critical')
        self.assertEqual(self.model(snap, 'c')['warning'], 'unknown')
        self.assertEqual(self.model(snap, 'w')['warning'], 'critical')

    def test_empty_snapshot(self):
        self.assertEqual(um.usage_snapshot(T0)['providers'], [])

    def test_never_raises(self):
        um.set_db_path(Path(self.tmp.name))  # a directory: sqlite cannot open it
        um.safe_record('c1', 'm', 'chat', 200, {}, {}, now=T0)
        um.record_for_row(ROW, '/audio/transcriptions', 200, None, object())
        um.record_for_row(None, '/x', 200, None, None)
        self.assertEqual(um.usage_snapshot(T0)['providers'], [])

    def test_non_groq_ignored(self):
        um.record_for_row({'base_url': 'https://example.com/v1', 'connection_id': 'x'}, '/chat/completions', 200, {}, {})
        self.assertEqual(um.usage_snapshot(T0)['providers'], [])

    def test_http_hook_success_429_and_broken_observer(self):
        calls = []

        def ok(req, timeout):
            return 200, ASR_H, b'{"usage": {"total_tokens": 5}}'

        def limited(req, timeout):
            return 429, {'retry-after': '1'}, b'{}'

        def boom(*a):
            raise RuntimeError('observer failure')

        data = http_post('https://api.groq.com/x', b'', transport=ok, on_response=lambda *a: calls.append(a))
        self.assertEqual(data['usage']['total_tokens'], 5)
        self.assertEqual(calls[0][0], 200)
        self.assertEqual(http_post('https://api.groq.com/x', b'', transport=ok, on_response=boom)['usage']['total_tokens'], 5)
        with self.assertRaises(ProviderError) as ctx:
            http_post('https://api.groq.com/x', b'', transport=limited, on_response=lambda *a: calls.append(a))
        self.assertEqual(ctx.exception.http_status, 429)
        self.assertEqual(calls[-1][0], 429)

    def test_request_hook_records_asr_and_chat(self):
        from oh_streamer.subtitles import model_adapters as ma
        seen = {}

        def transport(req, timeout):
            return 200, ASR_H if 'audio' in req.full_url else CHAT_H, b'{"usage":{"total_tokens":42}}'
        ma._request(dict(ROW, _usage_audio_s=4), '/audio/transcriptions', b'', 'k', 'multipart/form-data', transport=transport)
        ma._request(dict(ROW, remote_model='qwen'), '/chat/completions', b'', 'k', 'application/json', transport=transport)
        snap = um.usage_snapshot()
        self.assertEqual(self.model(snap, 'whisper-large-v3')['audio']['hour']['used_s'], 10.0)
        self.assertEqual(self.model(snap, 'qwen')['kind'], 'chat')
        self.assertEqual(snap['providers'][0]['series']['points'][-1]['tokens'], 42)


if __name__ == '__main__':
    unittest.main()
