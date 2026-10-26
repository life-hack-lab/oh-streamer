"""S1 contract tests: no network, GPU, model download, or real credentials."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oh_streamer.subtitles import configuration as config, model_registry as registry
from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.subtitle_jobs import SubtitleQueueMixin, snapshot_subtitle_settings, subtitle_configuration


class CloudFirstContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = dict(model_registry_path=str(self.root / 'models.json'), subtitle_cloud_enabled=True)
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith('OH_STREAMER_SUBTITLE_')}
        self.env = patch.dict(os.environ, clean_env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.secret = patch.object(registry, '_secret', side_effect=lambda v, decrypt=False: v[7:][::-1] if decrypt else 'opaque:' + v[::-1])
        self.secret.start()
        self.addCleanup(self.secret.stop)
        self.inference = patch('oh_streamer.subtitles.model_adapters.test_model')
        self.test_model = self.inference.start()
        self.addCleanup(self.inference.stop)

    def save(self, **kwargs):
        return registry.test_save_connection(dict(api_family='groq', api_key='fake-key-a', test_audio_path='speech-fixture.wav', **kwargs), self.settings)

    def profile(self):
        primary = subtitle_configuration({})['subtitle_primary']
        return config.migrate({}, primary)

    def test_groq_one_secret_two_bindings_and_atomic_commit(self):
        with patch.object(registry.runtime, 'atomic_write_json', wraps=registry.runtime.atomic_write_json) as write:
            result = self.save()
        self.assertTrue(result['ok'])
        self.assertEqual(write.call_count, 1)
        models = result['models']
        self.assertEqual(len(models), 2)
        self.assertEqual(len({m['connection_id'] for m in models}), 1)
        doc = registry._document(self.settings)
        self.assertNotIn('fake-key-a', json.dumps(doc))
        self.assertTrue(all('credential' not in m for m in doc['models']))
        config.assert_secret_free(result)
        config.assert_secret_free(registry.list_models(self.settings))
        config.assert_secret_free(registry.list_connections(self.settings))
        for model in models:
            self.assertEqual(registry.model_api_key(registry.get_model(model['id'], self.settings)), 'fake-key-a')

    def test_partial_failure_and_repair_keep_success_ids(self):
        self.test_model.side_effect = [None, RuntimeError('fake-key-a')]
        partial = self.save()
        self.assertFalse(partial['ok'])
        self.assertEqual(partial['code'], 'partial_validation')
        self.assertEqual(partial['connection']['validation']['asr']['status'], 'verified')
        self.assertEqual(partial['connection']['validation']['translation']['status'], 'failed')
        self.assertNotIn('fake-key-a', json.dumps(partial))
        asr_id = partial['models'][0]['id']
        self.test_model.side_effect = None
        repaired = registry.test_save_connection(dict(id=partial['connection']['id'], api_family='groq', test_audio_path='speech-fixture.wav'), self.settings)
        self.assertTrue(repaired['ok'])
        self.assertIn(asr_id, [m['id'] for m in repaired['models']])
        self.assertEqual(len(repaired['models']), 2)

    def test_rotating_shared_secret_and_endpoint_does_not_change_saved_plan(self):
        result = self.save()
        models = {m['kind']: m['id'] for m in result['models']}
        profile = self.profile()
        profile['active_mode'] = 'cloud'
        profile['profiles']['cloud'].update(asr=models['asr'], translation=models['translation'])
        before = config.resolve(profile, self.settings)
        identity = registry.model_identity(models['asr'], self.settings)
        rotated = registry.test_save_connection(dict(id=result['connection']['id'], api_family='groq', api_key='fake-key-b', test_audio_path='speech-fixture.wav'), self.settings)
        self.assertTrue(rotated['ok'])
        self.assertEqual(identity, registry.model_identity(models['asr'], self.settings))
        self.assertEqual(before, config.resolve(profile, self.settings))
        for mid in models.values():
            self.assertEqual(registry.model_api_key(registry.get_model(mid, self.settings)), 'fake-key-b')
        self.assertEqual(registry.connection_api_key(result['connection']['id'], self.settings), 'fake-key-b')

    def test_rotation_failure_marks_capability_unverified_without_deleting_models(self):
        result = self.save()
        self.test_model.side_effect = [None, RuntimeError('fake-key-b')]
        rotated = registry.test_save_connection(dict(id=result['connection']['id'], api_family='groq', api_key='fake-key-b', test_audio_path='speech-fixture.wav'), self.settings)
        self.assertFalse(rotated['ok'])
        self.assertEqual({m['id'] for m in result['models']}, {m['id'] for m in rotated['models']})
        profile = self.profile()
        profile['active_mode'] = 'cloud'
        profile['profiles']['cloud'].update({m['kind']: m['id'] for m in result['models']})
        resolved = config.resolve(profile, self.settings)
        self.assertFalse(resolved['ready'])
        self.assertIn('capability_unverified', {e['code'] for e in resolved['errors']})
        catalog = {m['id']: m for m in registry.list_models(self.settings)}
        translation = next(m for m in result['models'] if m['kind'] == 'translation')
        self.assertEqual(catalog[translation['id']]['status'], 'unverified')

    def test_legacy_key_update_rotates_siblings_with_one_commit(self):
        result = self.save()
        asr = next(m for m in result['models'] if m['kind'] == 'asr')
        with patch.object(registry.runtime, 'atomic_write_json', wraps=registry.runtime.atomic_write_json) as write:
            rotated = registry.update_model_key(asr['id'], 'fake-key-b', self.settings)
        self.assertTrue(rotated['ok'])
        self.assertEqual(write.call_count, 1)
        self.assertEqual(registry.connection_api_key(result['connection']['id'], self.settings), 'fake-key-b')
        validation = registry.list_connections(self.settings)[0]['validation']
        self.assertEqual(validation['asr']['status'], 'reachable')
        self.assertEqual(validation['translation']['status'], 'unverified')

    def test_connection_write_interruption_preserves_previous_key(self):
        result = self.save()
        before = registry.registry_path(self.settings).read_bytes()
        with patch.object(registry.runtime, 'atomic_write_json', side_effect=OSError('interrupted')):
            failed = registry.test_save_connection(dict(id=result['connection']['id'], api_family='groq', api_key='fake-key-b', test_audio_path='speech-fixture.wav'), self.settings)
        self.assertFalse(failed['ok'])
        self.assertEqual(before, registry.registry_path(self.settings).read_bytes())
        self.assertEqual(registry.connection_api_key(result['connection']['id'], self.settings), 'fake-key-a')

    def test_different_accounts_never_merge(self):
        a = self.save()
        b = registry.test_save_connection(dict(api_family='groq', api_key='fake-key-b', test_audio_path='speech-fixture.wav'), self.settings)
        self.assertNotEqual(a['connection']['id'], b['connection']['id'])
        spec = dict(kind='translation', provider='openai-compatible', base_url='https://example.org/v1', remote_model='test', api_key='first')
        first = registry.test_and_add_model(spec, self.settings)
        second = registry.test_and_add_model({**spec, 'api_key': 'second'}, self.settings)
        self.assertTrue(first['ok'] and second['ok'])
        self.assertNotEqual(first['model']['id'], second['model']['id'])

    def test_legacy_migration_idempotent_backup_and_env_model_precedence(self):
        rows = [dict(id='user-old', kind='asr', provider='openai-compatible', base_url='https://example.org/v1', remote_model='test', api_family='groq', tested=True),
                dict(id='user-other', kind='translation', provider='openai-compatible', base_url='https://example.org/v1', remote_model='test', credential='opaque:a')]
        registry.runtime.atomic_write_json(registry.registry_path(self.settings), rows)
        before = registry.model_identity(rows[0])
        registry.list_models(self.settings)
        first = registry.registry_path(self.settings).read_bytes()
        registry.list_models(self.settings)
        self.assertEqual(first, registry.registry_path(self.settings).read_bytes())
        self.assertEqual(registry._document(self.settings)['legacy_backup'], rows)
        self.assertEqual(before, registry.model_identity('user-old', self.settings))
        connections = registry.list_connections(self.settings)
        self.assertEqual(len({c['id'] for c in connections}), 2)
        with patch.dict(os.environ, {'OH_STREAMER_SUBTITLE_KEY_USER_OLD': 'model-env', 'OH_STREAMER_SUBTITLE_KEY_GROQ': 'family-env'}):
            self.assertEqual(registry.model_api_key(registry.get_model('user-old', self.settings)), 'model-env')
        self.assertNotIn('model-env', first.decode())

    def test_migration_interrupted_replace_leaves_original_recoverable(self):
        rows = [dict(id='user-old', kind='asr', provider='openai-compatible')]
        registry.runtime.atomic_write_json(registry.registry_path(self.settings), rows)
        with patch.object(registry.runtime, 'atomic_write_json', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                registry.list_connections(self.settings)
        self.assertEqual(json.loads(registry.registry_path(self.settings).read_text()), rows)
        self.assertEqual(len(registry.list_connections(self.settings)), 1)

    def test_env_and_azure_region_preserved(self):
        with patch.dict(os.environ, {'OH_STREAMER_SUBTITLE_KEY_AZURE_TRANSLATOR': 'fake-env'}):
            result = registry.test_save_connection(dict(api_family='azure-translator', azure_region='eastasia'), self.settings)
            self.assertTrue(result['ok'])
            row = registry.get_model(result['models'][0]['id'], self.settings)
            self.assertEqual(row['azure_region'], 'eastasia')
            self.assertEqual(registry.model_api_key(row), 'fake-env')
        self.assertNotIn('fake-env', registry.registry_path(self.settings).read_text())

    def test_connection_delete_refused_until_models_removed(self):
        result = self.save()
        cid = result['connection']['id']
        self.assertEqual(registry.remove_connection(cid, self.settings)['code'], 'connection_in_use')
        for model in result['models']:
            self.assertTrue(registry.remove_model(model['id'], self.settings)['ok'])
        self.assertTrue(registry.remove_connection(cid, self.settings)['ok'])

    def test_cloud_guard_prevents_test_calls_and_storage(self):
        result = registry.test_save_connection(dict(api_family='groq', api_key='fake'), {**self.settings, 'subtitle_cloud_enabled': False})
        self.assertEqual(result['code'], 'cloud_disabled')
        self.test_model.assert_not_called()
        self.assertFalse(registry.registry_path(self.settings).exists())

    def test_builtin_probe_is_reachable_not_speech_validation(self):
        result = registry.test_save_connection(dict(api_family='groq', api_key='fake-key-a'), self.settings)
        self.assertFalse(result['ok'])
        self.assertEqual(result['connection']['validation']['asr']['status'], 'reachable')
        self.assertEqual(result['connection']['validation']['translation']['status'], 'verified')

    def test_real_sample_validation_disables_sibling_fallback(self):
        from oh_streamer.subtitles import model_adapters
        source = self.root / 'speech.wav'
        source.write_bytes(b'fixture')
        # Use the actual validator; isolate only the ASR transport/pipeline.
        self.inference.stop()
        with patch.object(model_adapters, '_cloud_asr', return_value={'segments': [{'start': 0, 'end': 1, 'text': 'speech'}]}) as asr:
            model_adapters.test_model(dict(kind='asr', provider='openai-compatible'), test_audio_path=str(source))
        self.assertFalse(asr.call_args.kwargs['allow_model_fallback'])

    def test_translation_validation_rejects_duplicate_line_mapping(self):
        from oh_streamer.subtitles import model_adapters
        self.inference.stop()
        row = dict(kind='translation', provider='openai-compatible')
        with patch.object(model_adapters.CloudTranslationClient, 'translate', return_value='1. 一\n1. 重複\n2. 二'):
            with self.assertRaises(model_adapters.ModelAdapterError):
                model_adapters.test_model(row)

    def test_validation_does_not_hold_registry_lock_and_detects_stale_writes(self):
        import threading
        observed = []
        def probe(*args, **kwargs):
            def reader():
                observed.extend(registry.list_connections(self.settings))
                observed.append('read_completed')
            thread = threading.Thread(target=reader)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            doc = registry._document(self.settings)
            doc['connections'].append(dict(id='concurrent', provider='openai-compatible'))
            registry.runtime.atomic_write_json(registry.registry_path(self.settings), doc)
        self.test_model.side_effect = probe
        result = self.save()
        self.assertIn('read_completed', observed)
        self.assertEqual(result['code'], 'concurrent_update')
        self.assertTrue(any(c['id'] == 'concurrent' for c in registry.list_connections(self.settings)))

    def test_full_preload_migrates_and_legacy_edits_preserve_other_modes(self):
        primary = subtitle_configuration({})['subtitle_primary']
        migrated = config.migrate(dict(subtitle_cloud_strategy='full_preload'), primary)
        self.assertEqual(migrated['profiles']['local']['processing'], 'full_preload')
        store = QueueStore(self.root / 'queue')
        initial_cloud = store.get_settings()['subtitle_profile']['profiles']['cloud']
        saved = store.get_settings()['subtitle_profile']
        saved['profiles']['local']['fallback']['asr'] = ['anime-whisper']
        store.set_settings(subtitle_profile=saved)
        store.set_settings(subtitle_cloud_strategy='full_preload')
        profile = store.get_settings()['subtitle_profile']
        self.assertEqual(profile['profiles']['local']['processing'], 'full_preload')
        self.assertEqual(profile['profiles']['cloud'], initial_cloud)
        self.assertEqual(profile['profiles']['local']['fallback']['asr'], ['anime-whisper'])

    def test_recovery_history_is_not_a_live_model_reference(self):
        result = self.save()
        mid = result['models'][0]['id']
        store = QueueStore(self.root / 'queue')
        store.set_settings(**self.settings)
        profile = self.profile()
        profile['migration']['original_primary']['asr']['model'] = mid
        store.set_settings(subtitle_profile=profile)
        self.assertTrue(SubtitleQueueMixin.remove_subtitle_model(SimpleNamespace(store=store), mid)['ok'])

    def test_three_modes_roundtrip_and_fresh_cloud_has_no_consent(self):
        store = QueueStore(self.root / 'queue')
        initial = store.get_settings()
        self.assertEqual(initial['subtitle_profile']['active_mode'], 'cloud')
        self.assertFalse(initial['subtitle_cloud_enabled'])
        self.assertFalse(config.resolve(initial['subtitle_profile'], self.settings)['ready'])
        profile = self.profile()
        profile['profiles']['custom']['fallback']['asr'] = ['anime-whisper']
        for mode in ('custom', 'cloud', 'local', 'custom'):
            profile['active_mode'] = mode
            store.set_settings(subtitle_profile=profile)
        loaded = QueueStore(self.root / 'queue').get_settings()['subtitle_profile']
        self.assertEqual(loaded, profile)
        self.assertEqual(config.normalize(loaded), loaded)

    def test_fresh_install_pending_cloud_blocks_subtitles_with_readable_message(self):
        # S6 F2/F3: a new install's unconfigured cloud profile must not fall back to the legacy local pipeline
        # (a release may ship without local models), and the rejection is user-facing text, not the bare code.
        store = QueueStore(self.root / 'queue')
        fresh = store.get_settings()
        self.assertEqual(fresh['subtitle_runtime_contract'], 'cloud-first-1')
        with self.assertRaises(config.ConfigurationError) as raised:
            snapshot_subtitle_settings(fresh)
        self.assertEqual(raised.exception.code, 'configuration_not_ready')
        self.assertEqual(str(raised.exception), 'configuration_not_ready')
        self.assertIn('設定 → 字幕', raised.exception.message)
        root = self.root / 'existing'
        root.mkdir()
        (root / 'queue.json').write_text(json.dumps(dict(settings=dict(subtitle_japanese=True), items=[])))
        self.assertNotIn('subtitle_runtime_contract', QueueStore(root).get_settings(),
                         'existing users keep the legacy contract until they save a profile')

    def test_legacy_settings_and_old_job_remain_readable_and_disabled(self):
        root = self.root / 'queue'
        root.mkdir()
        legacy = dict(settings=dict(subtitle_japanese=True, subtitle_cloud_enabled=False),
                      items=[dict(id='old', settings=dict(subtitles=True, subtitle_configs=[dict(id='primary')]), subtitle=dict(status='partial', covered_s=30))])
        (root / 'queue.json').write_text(json.dumps(legacy))
        store = QueueStore(root)
        first = (root / 'queue.json').read_bytes()
        self.assertEqual(store.get('old').settings, legacy['items'][0]['settings'])
        self.assertEqual(store.get_settings()['subtitle_profile']['active_mode'], 'local')
        self.assertTrue(store.get_settings()['subtitle_primary']['japanese_track'])
        self.assertFalse(store.get_settings()['subtitle_cloud_enabled'])
        QueueStore(root)
        self.assertEqual(first, (root / 'queue.json').read_bytes())

    def test_snapshot_freezes_bindings_and_fallback_without_secrets(self):
        result = self.save()
        models = {m['kind']: m['id'] for m in result['models']}
        profile = self.profile()
        profile['active_mode'] = 'custom'
        profile['profiles']['custom'].update(asr=models['asr'], translation=models['translation'])
        profile['profiles']['custom']['fallback']['asr'] = ['anime-whisper', 'anime-whisper', models['asr']]
        settings = {**self.settings, 'subtitle_profile': profile}
        frozen = snapshot_subtitle_settings(settings)
        config.assert_secret_free(frozen)
        self.assertEqual([m['id'] for m in frozen['subtitle_resolved']['fallback']['asr']], ['anime-whisper'])
        self.assertEqual(frozen['subtitle_resolved']['flow'], 'direct')
        before = copy.deepcopy(frozen)
        profile['profiles']['custom']['asr'] = 'anime-whisper'
        self.assertEqual(before, frozen)
        self.assertEqual(before, snapshot_subtitle_settings(frozen))
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            store = QueueStore(Path(tmp))
            store.set_settings(subtitle_profile=before['subtitle_profile'])
            item = store.add('source', 'file', None, None, {**settings, 'subtitle_profile': before['subtitle_profile'], 'subtitles': True})
            self.assertEqual(store.get(item.id).settings['subtitle_resolved'], frozen['subtitle_resolved'])

    def test_guard_all_fallback_candidates_and_fixed_whisper(self):
        result = self.save()
        mid = next(m['id'] for m in result['models'] if m['kind'] == 'asr')
        profile = self.profile()
        profile['profiles']['local']['fallback']['asr'] = [mid]
        profile['profiles']['local']['whisper_policy'] = 'fixed'
        resolved = config.resolve(profile, {**self.settings, 'subtitle_cloud_enabled': False})
        self.assertEqual(resolved['whisper_models'], [])
        self.assertEqual({e['code'] for e in resolved['errors']}, {'cloud_disabled', 'local_mode_cloud_candidate'})
        self.test_model.reset_mock()
        config.resolve(profile, self.settings)
        self.test_model.assert_not_called()

    def test_secret_and_invalid_schema_rejected_before_save(self):
        profile = self.profile()
        for value in ({**profile, 'api_key': 'fake'}, {**profile, 'schema_version': 99}):
            with self.assertRaises(config.ConfigurationError):
                config.normalize(value)
        profile['profiles']['cloud']['fallback']['max_attempts'] = 999
        with self.assertRaises(config.ConfigurationError):
            config.normalize(profile)

    def test_model_delete_checks_inactive_profiles_and_frozen_jobs(self):
        result = self.save()
        mid = result['models'][0]['id']
        store = QueueStore(self.root / 'queue')
        store.set_settings(**self.settings)
        profile = store.get_settings()['subtitle_profile']
        profile['profiles']['custom']['asr'] = mid
        store.set_settings(subtitle_profile=profile)
        svc = SimpleNamespace(store=store)
        self.assertFalse(SubtitleQueueMixin.remove_subtitle_model(svc, mid)['ok'])
        store.set_settings(subtitle_profile=self.profile())
        store.add('source', 'file', None, None, dict(subtitles=True, subtitle_resolved=dict(models={'asr': dict(id=mid, connection_id=result['connection']['id'])})))
        self.assertFalse(SubtitleQueueMixin.remove_subtitle_model(svc, mid)['ok'])
        self.assertEqual(SubtitleQueueMixin.remove_subtitle_connection(svc, result['connection']['id'])['code'], 'connection_in_use')


if __name__ == '__main__':
    unittest.main()
