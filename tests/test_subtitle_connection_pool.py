"""Connection pools: encrypted registry management, frozen plans, and real quota decisions."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oh_streamer.subtitles import configuration as config, model_registry as registry, fallback as fb
from oh_streamer.subtitles.slice_plan import TaskState
from oh_streamer.subtitles.providers.http import ProviderError
from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin, subtitle_configuration, snapshot_subtitle_settings


class ConnectionPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = dict(model_registry_path=str(self.root / 'models.json'), subtitle_cloud_enabled=True)
        for mocked in [patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('OH_STREAMER_SUBTITLE_')}, clear=True),
                       patch.object(registry, '_secret', side_effect=lambda v, decrypt=False: v[7:][::-1] if decrypt else 'opaque:' + v[::-1]),
                       patch('oh_streamer.subtitles.model_adapters.test_model')]:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.connections = [self.connection('a'), self.connection('b'), self.connection('c')]
        self.profile = config.migrate({}, subtitle_configuration({})['subtitle_primary'], fresh=True)
        self.profile['profiles']['cloud']['connection_pool'] = {
            'auto_switch': True, 'connections': [dict(id=c['connection']['id'], enabled=True) for c in self.connections]}

    def connection(self, key, **extra):
        result = registry.test_save_connection(dict(api_family='groq', label='連線 ' + key,
            api_key='fake-key-' + key, test_audio_path='speech-fixture.wav', **extra), self.settings)
        self.assertIn('connection', result, result)
        return result

    def test_order_enable_and_disable_auto_switch(self):
        cloud = self.profile['profiles']['cloud']
        cloud['connection_pool']['connections'].reverse()
        cloud['connection_pool']['connections'][1]['enabled'] = False
        projected = config.project_connection_pool(self.profile, self.settings)
        self.assertEqual(projected['profiles']['cloud']['asr'], self.connections[2]['models'][0]['id'])
        plan = config.resolve(self.profile, self.settings)
        self.assertTrue(plan['ready'])
        self.assertEqual([m['connection_id'] for m in plan['fallback']['translation']], [self.connections[0]['connection']['id']])
        cloud['connection_pool']['auto_switch'] = False
        plan = config.resolve(self.profile, self.settings)
        self.assertEqual(plan['fallback'], dict(asr=[], translation=[]))
        self.assertEqual(plan['models']['translation']['connection_id'], self.connections[2]['connection']['id'])
        self.assertNotIn('fake-key-', json.dumps(plan))

    def test_roles_use_only_validated_models_and_empty_pool_is_not_ready(self):
        cid = self.connections[0]['connection']['id']
        with patch('oh_streamer.subtitles.model_adapters.test_model', side_effect=lambda row, **kw: (_ for _ in ()).throw(ValueError()) if row['kind'] == 'translation' else None):
            registry.test_save_connection(dict(id=cid, api_family='groq', test_audio_path='speech-fixture.wav'), self.settings)
        plan = config.resolve(self.profile, self.settings)
        self.assertEqual(plan['models']['asr']['connection_id'], cid)
        self.assertEqual(plan['models']['translation']['connection_id'], self.connections[1]['connection']['id'])
        for c in self.profile['profiles']['cloud']['connection_pool']['connections']:
            c['enabled'] = False
        plan = config.resolve(self.profile, self.settings)
        self.assertFalse(plan['ready'])
        self.assertIsNone(plan['models']['asr'])
        self.assertEqual(plan['fallback']['translation'], [])

    def test_bad_pool_rejected_and_legacy_profiles_unchanged(self):
        legacy = copy.deepcopy(self.profile)
        del legacy['profiles']['cloud']['connection_pool']
        self.assertEqual(config.project_connection_pool(legacy, self.settings), config.normalize(legacy))
        for bad in [dict(auto_switch=1, connections=[]), dict(auto_switch=True, connections=[dict(id='x', enabled='yes')]),
                    dict(auto_switch=True, connections=[dict(id='x', enabled=True)] * 2)]:
            profile = copy.deepcopy(self.profile)
            profile['profiles']['cloud']['connection_pool'] = bad
            with self.assertRaises(config.ConfigurationError):
                config.normalize(profile)
        profile = copy.deepcopy(self.profile)
        profile['profiles']['cloud']['connection_pool']['connections'][0]['id'] = 'missing'
        with self.assertRaises(config.ConfigurationError):
            config.resolve(profile, self.settings)

    def test_rename_blank_key_retained_and_duplicate_key_rejected(self):
        cid = self.connections[0]['connection']['id']
        original = registry.connection_api_key(cid, self.settings)
        result = registry.test_save_connection(dict(id=cid, api_family='groq', label='新名稱'), self.settings)
        self.assertEqual(result['connection']['label'], '新名稱')
        self.assertEqual(registry.connection_api_key(cid, self.settings), original)
        ids = [m['id'] for m in result['models']]
        self.assertEqual(ids, [m['id'] for m in self.connections[0]['models']])
        before = registry.registry_path(self.settings).read_bytes()
        duplicate = registry.test_save_connection(dict(api_family='groq', api_key=original), self.settings)
        self.assertEqual(duplicate['code'], 'duplicate_connection')
        self.assertEqual(before, registry.registry_path(self.settings).read_bytes())

    def test_resolved_pool_runs_through_quota_fallback_and_waits_when_all_cool(self):
        plan = config.resolve(self.profile, self.settings)
        chain = fb.CandidateChain(plan, 'translation')
        first = chain.next(TaskState(), 1000).candidate
        error = ProviderError('quota_daily', 'quota', retryable=True, retry_after=500)
        decision = chain.decide(TaskState(candidate=first.id), first, error, 1000)
        self.assertEqual(decision.action, fb.SWITCH)
        self.assertEqual(decision.candidate.row['connection_id'], self.connections[1]['connection']['id'])
        cooling = fb.CandidateChain(plan, 'translation', cooldown=lambda _: 1500)
        pick = cooling.next(TaskState(), 1000)
        self.assertFalse(pick.exhausted)
        self.assertEqual(pick.retry_at, 1500)
        self.assertEqual(fb.CandidateChain(plan, 'asr').next(TaskState(), 1000).candidate.row['connection_id'], self.connections[0]['connection']['id'])

    def service(self):
        store = QueueStore(self.root / 'queue')
        projected = config.project_connection_pool(self.profile, self.settings)
        store.set_settings(**self.settings, subtitle_profile=projected)
        return SimpleNamespace(store=store, _lock=threading.RLock(), _subtitle_thaw_all=lambda **kw: None)

    def test_delete_removes_pool_and_models_but_blocks_queue_or_custom_references(self):
        svc = self.service()
        cid = self.connections[0]['connection']['id']
        model = self.connections[0]['models'][0]['id']
        profile = svc.store.get_settings()['subtitle_profile']
        profile['profiles']['custom']['asr'] = model
        svc.store.set_settings(subtitle_profile=profile)
        self.assertEqual(SubtitleQueueMixin.remove_subtitle_pool_connection(svc, cid)['code'], 'connection_in_use')
        profile['profiles']['custom']['asr'] = None
        svc.store.set_settings(subtitle_profile=profile)
        result = SubtitleQueueMixin.remove_subtitle_pool_connection(svc, cid)
        self.assertTrue(result['ok'], result)
        self.assertNotIn(cid, [c['id'] for c in registry.list_connections(self.settings)])
        self.assertFalse(any(m.get('connection_id') == cid for m in registry._read(self.settings)))
        cid = self.connections[1]['connection']['id']
        svc.store.add('video', 'file', None, None, {'subtitles': False, 'frozen_binding': {'connection_id': cid}})
        before = svc.store.get_settings()['subtitle_profile']
        result = SubtitleQueueMixin.remove_subtitle_pool_connection(svc, cid)
        self.assertEqual(result['code'], 'connection_in_use')
        self.assertEqual(svc.store.get_settings()['subtitle_profile'], before)

    def test_snapshot_is_frozen_and_delete_failure_rolls_back_profile(self):
        svc = self.service()
        snapshot = snapshot_subtitle_settings(svc.store.get_settings())
        before = copy.deepcopy(snapshot['subtitle_resolved'])
        profile = svc.store.get_settings()['subtitle_profile']
        profile['profiles']['cloud']['connection_pool']['connections'].reverse()
        self.assertTrue(SubtitleQueueMixin.set_subtitle_profile(svc, profile)['ok'])
        self.assertEqual(snapshot['subtitle_resolved'], before)
        saved = svc.store.get_settings()['subtitle_profile']
        with patch.object(registry, 'remove_connection', side_effect=OSError('disk full')):
            result = SubtitleQueueMixin.remove_subtitle_pool_connection(svc, self.connections[0]['connection']['id'])
        self.assertFalse(result['ok'])
        self.assertEqual(svc.store.get_settings()['subtitle_profile'], saved)


if __name__ == '__main__':
    unittest.main()
