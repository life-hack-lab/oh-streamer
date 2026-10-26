"""S3 isolated integration. No real network, credentials, models or GPU needed."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from oh_streamer.subtitles import fallback, model_adapters as adapters
from oh_streamer.subtitles.providers.http import ProviderError
from oh_streamer.subtitles.providers.http import http_post
from oh_streamer.queue.queue_service import QueueService, _Run
from test_subtitle_cloud_first_runtime import settings


def binding(name, kind='translation', local=False):
    return dict(id='user-' + name, kind=kind, provider='local' if local else 'openai-compatible',
                connection_id='connection-' + name, remote_model=name, base_url='https://example.org/v1',
                recipe='sakura-v1' if kind == 'translation' else 'whisper-large-v3', model_path='missing')


class Clock:
    def __init__(self):
        self.now = 10000.
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.ledger = fallback.Ledger(self.root / 'ledger.sqlite3', clock=self.clock)
        self.plan = dict(policy=dict(max_attempts=4, retries=1, cooldown_seconds=60),
                         whisper_policy='fixed', whisper_models=[], mode='custom',
                         fallback=dict(asr=[], translation=[binding('backup')]))
        self.events, self.calls = [], []

    def policy(self, **kw):
        return fallback.Policy(self.plan, 'translation', binding('primary'), ledger=self.ledger,
            clock=self.clock, sleep=self.clock.sleep, on_event=self.events.append, **kw)

    def run_error(self, kind, retryable=False, retry_after=None):
        def operation(row, key):
            self.calls.append(row['id'])
            if row['id'] == 'user-primary':
                raise ProviderError(kind, retryable=retryable, retry_after=retry_after)
            return 'ok'
        return self.policy().run(operation, lambda row: 'dummy'), operation

    def test_transient_errors_retry_then_ordered_backup(self):
        for kind in ('rate_limit', 'network', 'timeout', 'server'):
            with self.subTest(kind=kind):
                self.ledger = fallback.Ledger(self.root / (kind + '.db'), clock=self.clock)
                self.calls.clear()
                result, _ = self.run_error(kind, retryable=True, retry_after=.2)
                self.assertEqual(result, 'ok')
                self.assertEqual(self.calls, ['user-primary', 'user-primary', 'user-backup'])
                self.assertTrue(all(wait <= .10001 for wait in self.clock.waits))

    def test_transport_status_classification(self):
        for status, kind, body in [(429, 'rate_limit', b'{}'), (503, 'server', b'{}'),
                                   (401, 'auth', b'{}'), (404, 'unsupported', b'{}'),
                                   (400, 'content_blocked', b'{"error":{"code":"content_policy_violation"}}')]:
            with self.subTest(status=status), self.assertRaises(ProviderError) as raised:
                http_post('https://example.org/v1/test', b'{}', transport=lambda *_: (status, {}, body))
            self.assertEqual(raised.exception.kind, kind)

    def test_auth_invalid_model_and_bad_response_do_not_retry(self):
        for kind in ('auth', 'unsupported', 'bad_response'):
            self.ledger = fallback.Ledger(self.root / (kind + '.db'), clock=self.clock)
            self.calls.clear()
            self.run_error(kind)
            self.assertEqual(self.calls, ['user-primary', 'user-backup'])

    def test_retry_after_long_window_skips_wait(self):
        self.run_error('quota_daily', retry_after=3600)
        self.assertEqual(self.calls, ['user-primary', 'user-backup'])
        self.assertFalse(self.clock.waits)

    def test_content_blocked_never_resends(self):
        with self.assertRaises(ProviderError) as raised:
            self.run_error('content_blocked')
        self.assertEqual(raised.exception.kind, 'content_blocked')
        self.assertEqual(self.calls, ['user-primary'])

    def test_cooldown_then_one_probe_and_recovery(self):
        _, operation = self.run_error('auth')
        self.calls.clear()
        self.policy().run(operation, lambda row: 'dummy')
        self.assertEqual(self.calls, ['user-backup'])
        self.clock.now += 61
        def recovered(row, key):
            self.calls.append(row['id'])
            return 'recovered'
        self.assertEqual(self.policy().run(recovered, lambda row: 'dummy'), 'recovered')
        self.assertEqual(self.events[-1]['state'], 'recovered')
        self.assertEqual(self.calls[-1], 'user-primary')

    def test_probe_is_exclusive_across_ledger_instances(self):
        self.ledger.fail('candidate', ProviderError('auth'), 60)
        self.clock.now += 61
        self.assertEqual(self.ledger.claim('candidate'), 'probe')
        other = fallback.Ledger(self.ledger.path, clock=self.clock)
        self.assertIsInstance(other.claim('candidate'), ProviderError)
        self.ledger.success('candidate')
        self.assertIsNone(other.claim('candidate'))

    def test_attempt_budget_and_deduplication(self):
        self.plan['policy'].update(max_attempts=3, retries=3)
        self.plan['fallback']['translation'] *= 5
        with self.assertRaises(ProviderError):
            self.policy().run(lambda row, key: (self.calls.append(row['id']),
                (_ for _ in ()).throw(ProviderError('network', retryable=True)))[1], lambda row: 'dummy')
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.events[-1]['state'], 'exhausted')
        self.assertTrue(self.events[-1]['recoverable'])

    def test_different_ids_with_same_account_and_model_are_not_retried(self):
        duplicate = dict(binding('primary'), id='other-id', connection_id='other-connection')
        self.plan['fallback']['translation'] = [duplicate, binding('backup')]
        self.run_error('auth')
        self.assertEqual(self.calls, ['user-primary', 'user-backup'])

    def test_no_fallback_does_not_switch(self):
        self.plan['fallback']['translation'] = []
        with self.assertRaises(ProviderError):
            self.run_error('auth')
        self.assertEqual(self.calls, ['user-primary'])

    def test_cancellation_interrupts_retry_wait(self):
        def check():
            if self.clock.waits:
                raise InterruptedError()
        with self.assertRaises(InterruptedError):
            self.policy(check_cancel=check).run(lambda row, key: (_ for _ in ()).throw(
                ProviderError('rate_limit', retryable=True, retry_after=10)), lambda row: 'dummy')
        self.assertLessEqual(sum(self.clock.waits), .101)
        self.assertFalse(any(e['state'] == 'fallback' for e in self.events))

    def test_cancellation_releases_expired_primary_probe(self):
        row = binding('primary')
        key = self.ledger.account(row, 'dummy') + ':' + json.dumps({k: row.get(k) for k in
            ('remote_model', 'recipe', 'model_path', 'kind')}, sort_keys=True)
        self.ledger.fail(key, ProviderError('network'), 60)
        self.clock.now += 61
        def check():
            if self.clock.waits:
                raise InterruptedError()
        with self.assertRaises(InterruptedError):
            self.policy(check_cancel=check).run(lambda row, key: (_ for _ in ()).throw(
                ProviderError('network', retryable=True, retry_after=10)), lambda row: 'dummy')
        self.assertEqual(self.ledger.claim(key), 'probe')

    def test_silence_empty_success_does_not_fallback(self):
        self.assertEqual(self.policy().run(lambda row, key: [], lambda row: 'dummy'), [])
        self.assertEqual(self.events[-1]['model']['id'], 'user-primary')

    def groq(self, name):
        return dict(binding(name), api_family='groq', base_url='https://api.groq.com/openai/v1')

    def test_groq_only_admission_wait_is_waited_out_not_failed(self):
        # S6 F1: our own pacing used to raise for waits > 30 s and the queue marked the subtitles failed for good.
        self.plan['fallback']['translation'] = []
        policy = fallback.Policy(self.plan, 'translation', self.groq('primary'), ledger=self.ledger,
                                 clock=self.clock, sleep=self.clock.sleep, on_event=self.events.append)
        replies = [policy.run(lambda row, key: 'batch', lambda row: 'dummy', tokens=30 * 128) for _ in range(6)]
        self.assertEqual(replies, ['batch'] * 6)
        waits = [e for e in self.events if e['state'] == 'waiting']
        self.assertTrue(waits and all(e.get('pacing') for e in waits), 'internal pacing is flagged as such')
        self.assertTrue(any(e['retry_after'] > 30 for e in waits), 'a full-window wait is waited out')
        self.assertFalse(any(e['state'] in ('failed', 'exhausted') for e in self.events))

    def test_max_tokens_bound_is_weighted_so_batches_are_not_one_per_minute(self):
        account = self.ledger.account(self.groq('primary'), 'dummy')
        for _ in range(4):
            self.ledger.reserve(account, tokens=30 * 128)
            self.clock.now += 2.0
        with self.assertRaises(ProviderError):
            self.ledger.reserve(account, tokens=30 * 128)

    def test_last_candidate_honours_repeated_provider_retry_after(self):
        # Real Groq-only run: 429s with Retry-After 19/2/8/6 s on one batch; retries=1 used to give up.
        self.plan['fallback']['translation'] = []
        statuses = iter([429, 429, 429, 200])
        def operation(row, key):
            if next(statuses) == 429:
                raise ProviderError('rate_limit', retryable=True, retry_after=19)
            return 'ok'
        policy = fallback.Policy(self.plan, 'translation', self.groq('primary'), ledger=self.ledger,
                                 clock=self.clock, sleep=self.clock.sleep, on_event=self.events.append)
        self.assertEqual(policy.run(operation, lambda row: 'dummy'), 'ok')
        with self.assertRaises(ProviderError):     # still bounded by max_attempts
            policy.run(lambda row, key: (_ for _ in ()).throw(
                ProviderError('rate_limit', retryable=True, retry_after=19)), lambda row: 'dummy')
        self.assertEqual(self.events[-1]['state'], 'exhausted')
        self.assertTrue(self.events[-1]['recoverable'])

    def test_shared_account_reservations_visible_to_other_process(self):
        row = binding('primary')
        account = self.ledger.account(row, 'dummy')
        self.assertEqual(account, self.ledger.account(dict(row, connection_id='alias'), 'dummy'))
        self.assertNotEqual(account, self.ledger.account(row, 'different'))
        self.ledger.reserve(account, audio=100)
        script = ('from oh_streamer.subtitles.fallback import Ledger; '
                  'from oh_streamer.subtitles.providers.http import ProviderError; '
                  'import sys; l=Ledger(sys.argv[1],clock=lambda:10000); '
                  '\ntry: l.reserve(sys.argv[2],audio=100)'
                  '\nexcept ProviderError as e: print(e.kind)')
        # The child must import oh_streamer without relying on the parent's PYTHONPATH: run_tests.py only edits
        # sys.path in-process, so under the full suite the child had no src path and died with ModuleNotFoundError.
        src = str(Path(fallback.__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (src, os.environ.get('PYTHONPATH')) if p))
        result = subprocess.run([sys.executable, '-c', script, str(self.ledger.path), account],
                                text=True, capture_output=True, env=env)
        # flaky once in a full run with no output captured: keep the child's stderr in the failure message
        self.assertEqual(result.returncode, 0, f'child exit {result.returncode}: {result.stderr[-1500:]}')
        self.assertEqual(result.stdout.strip(), 'rate_limit')
        self.assertNotIn('dummy', self.ledger.path.read_bytes().decode('latin1'))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = settings()
        self.plan = self.cfg['subtitle_resolved']
        for kind in ('asr', 'translation'):
            self.plan['models'][kind] = binding('primary-' + kind, kind)
            self.plan['fallback'][kind] = [binding('backup-' + kind, kind)]
        self.plan['policy'] = dict(max_attempts=4, retries=0, cooldown_seconds=60)
        self.ledger = fallback.Ledger(self.root / 'state.db')
        for target, value in [('oh_streamer.subtitles.fallback.Ledger', self.ledger),
                              ('oh_streamer.subtitles.model_registry.connection_api_key', 'dummy')]:
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def client(self, **kwargs):
        return adapters.make_translation_client('user-primary-translation', settings=self.cfg, **kwargs)

    def test_translation_real_http_retry_does_not_multiply_and_records_source(self):
        self.plan['policy']['retries'] = 1
        def request(row, *args, **kwargs):
            if row['remote_model'] == 'primary-translation':
                raise ProviderError('rate_limit', retryable=True, retry_after=0)
            return dict(choices=[dict(message=dict(content='1. 測試'))])
        with patch.object(adapters, '_request', side_effect=request) as transport:
            client = self.client()
            self.assertEqual(client.translate('1. テスト'), '1. 測試')
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(client.last_source['id'], 'user-backup-translation')

    def test_invalid_line_mapping_uses_backup(self):
        def request(row, *args):
            text = '2. 錯誤' if row['remote_model'] == 'primary-translation' else '1. 測試'
            return dict(choices=[dict(message=dict(content=text))])
        with patch.object(adapters, '_request', side_effect=request) as transport:
            self.assertEqual(self.client().translate('1. テスト'), '1. 測試')
        self.assertEqual(transport.call_count, 2)

    def test_local_candidate_needs_owned_worker_and_is_closed(self):
        local = binding('local', local=True)
        local['model_path'] = str(self.root / 'model.gguf')
        Path(local['model_path']).write_bytes(b'fixture')
        self.plan['fallback']['translation'] = [local]
        def request(*args):
            raise ProviderError('auth')
        local_client = Mock()
        local_client.translate.return_value = '1. 備援'
        with patch.object(adapters, '_request', side_effect=request), \
             patch('oh_streamer.subtitles.llm.LlamaServerClient', return_value=local_client) as factory:
            with self.assertRaises(ProviderError):
                self.client().translate('1. テスト')
            factory.assert_not_called()
            # Clear only the test ledger; production cooldown remains persistent.
            with self.ledger.connect() as db:
                db.execute('DELETE FROM state')
            client = self.client(gpu_managed=True)
            self.assertEqual(client.translate('1. テスト'), '1. 備援')
        local_client.close.assert_called_once()
        self.assertEqual(client.last_source['provider'], 'local')

    def test_local_exception_closes_server_before_next_candidate(self):
        local = binding('local', local=True)
        local['model_path'] = str(self.root / 'model.gguf')
        Path(local['model_path']).write_bytes(b'fixture')
        self.plan['models']['translation'] = local
        local_client = Mock()
        local_client.translate.side_effect = RuntimeError('fixture')
        cloud = Mock()
        cloud.translate.return_value = '1. 備援'
        def factory(row, key, **kwargs):
            if row['provider'] == 'local':
                return local_client
            local_client.close.assert_called_once()
            return cloud
        with patch.object(adapters, '_translation_client', side_effect=factory):
            result = adapters.make_translation_client(local['id'], settings=self.cfg,
                                                       gpu_managed=True).translate('1. テスト')
        self.assertEqual(result, '1. 備援')
        cloud.close.assert_called_once()

    def test_pure_local_cloud_candidate_cannot_send(self):
        self.plan['mode'] = 'local'
        with patch.object(adapters, '_request') as request:
            with self.assertRaises(PermissionError):
                self.client().translate('1. テスト')
        request.assert_not_called()

    def test_local_missing_is_skipped_without_download(self):
        self.plan['models']['translation'] = binding('local', local=True)
        with patch.object(adapters, '_request', return_value=dict(choices=[dict(message=dict(content='1. 備援'))])), \
             patch('oh_streamer.subtitles.llm.LlamaServerClient') as local:
            client = adapters.make_translation_client('user-local', settings=self.cfg, gpu_managed=True)
            self.assertEqual(client.translate('1. テスト'), '1. 備援')
        local.assert_not_called()


    def test_global_disable_stops_owned_fallback_worker_but_preserves_restore(self):
        qs = QueueService(self.root / 'queue')
        item = qs.store.add('fixture.mp4', 'file', None, '1080p', self.cfg)
        run = _Run(item.id, 0)
        qs._wk = run
        with patch.object(qs, '_model_options', return_value={}), patch.object(qs, '_request_stop') as stop:
            run.stage = 'restore'
            self.assertTrue(qs.set_settings(dict(subtitle_cloud_enabled=False))['ok'])
            stop.assert_not_called()
            qs._wk = None                       # M7: the cloud executor thread lives in its own slot
            run.stage = 'subtitle_cloud'
            qs._subtitle_cloud = run
            with patch.object(qs, '_subtitle_stop_executors') as stop_executors:
                self.assertTrue(qs.set_settings(dict(subtitle_cloud_enabled=False))['ok'])
            stop.assert_called_once_with(run, 'cancel')
            stop_executors.assert_called_once_with(item.id, 'cancel')

    def test_local_model_revision_changes_cache_identity(self):
        local = binding('local', local=True)
        local['model_path'] = str(self.root / 'model.gguf')
        path = Path(local['model_path'])
        path.write_bytes(b'first')
        self.plan['fallback']['translation'] = [local]
        before = fallback.cache_identity(self.plan)
        path.write_bytes(b'second version')
        self.assertNotEqual(before, fallback.cache_identity(self.plan))
        self.assertNotIn('local_revisions', self.plan)


    def test_sibling_shares_policy_budget_and_fixed_disables_it(self):
        primary = self.plan['models']['asr']
        primary.update(api_family='groq', remote_model='whisper-large-v3')
        self.plan.update(whisper_policy='auto', whisper_models=['whisper-large-v3-turbo'])
        policy = fallback.Policy(self.plan, 'asr', primary, ledger=self.ledger)
        self.assertEqual([r['remote_model'] for r in policy.candidates],
                         ['whisper-large-v3', 'whisper-large-v3-turbo', 'backup-asr'])
        self.plan['whisper_policy'] = 'fixed'
        policy = fallback.Policy(self.plan, 'asr', primary, ledger=self.ledger)
        self.assertEqual([r['remote_model'] for r in policy.candidates], ['whisper-large-v3', 'backup-asr'])

    def test_fixed_whisper_cannot_reenable_sibling_through_alias_connection(self):
        primary = self.plan['models']['asr']
        primary.update(api_family='groq', remote_model='whisper-large-v3')
        sibling = dict(primary, id='user-sibling', connection_id='alias', remote_model='whisper-large-v3-turbo')
        self.plan['fallback']['asr'] = [sibling, binding('other', 'asr')]
        used = []
        def operation(row, key):
            used.append(row['remote_model'])
            if row['id'] == primary['id']:
                raise ProviderError('auth')
            return []
        with patch.object(self.ledger, 'reserve'):
            policy = fallback.Policy(self.plan, 'asr', primary, ledger=self.ledger)
            policy.run(operation, lambda row: 'dummy')
        self.assertEqual(used, ['whisper-large-v3', 'other'])









if __name__ == '__main__':
    unittest.main()

