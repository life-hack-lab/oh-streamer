"""S2 integration: frozen configuration through queue/actions and combo workers.

All audio, providers and muxing are mocked; no real credentials are read.
"""
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch, Mock

from oh_streamer.subtitles import configuration, combos, model_adapters
from oh_streamer.queue import queue_service
from oh_streamer.queue.subtitle_jobs import snapshot_subtitle_settings, validate_combo_ids


class Client:
    def __init__(self, text='譯文', fail=False, after=None):
        self.calls = []
        self.text, self.fail, self.after = text, fail, after

    def translate(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError('fixture provider failure')
        if self.after:
            self.after()
        text = prompt[-1]['content'] if isinstance(prompt, list) else prompt
        ids = re.findall(r'^(\d+)\.\s+', text, re.M)
        return '\n'.join(f'{i}. {self.text}' for i in ids)

    def close(self):
        pass


def settings(processing='smart_horizon'):
    def binding(name, kind, family='groq'):
        return dict(id='user-' + name, kind=kind, provider='openai-compatible',
                    connection_id='connection-' + family, base_url='https://fixture.invalid/v1',
                    api_family=family, remote_model=name)
    plan = dict(schema_version=1, plan_version='cloud-first-1', mode='cloud',
                flow='direct',
                models=dict(asr=binding('whisper-large-v3', 'asr'),
                            translation=binding('translation', 'translation')),
                processing=processing, whisper_policy='fixed', whisper_models=[],
                fallback=dict(asr=[], translation=[]), policy={}, ready=True, errors=[])
    row = combos.normalize_combo(dict(id='primary', label='主字幕', sentence=dict(sentence_mode=False)))
    result = dict(subtitles=True, subtitle_cloud_enabled=True, subtitle_runtime_contract='cloud-first-1',
                  subtitle_resolved=plan, subtitle_configs=[row], subtitle_combos=['primary'],
                  subtitle_cloud_strategy=processing)
    result['subtitle_configs'][0] = configuration.runtime_primary(result, row)
    return result


class CloudFirstRuntime(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('OH_STREAMER_SUBTITLE_')}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)









    def test_frozen_endpoint_rotated_key_and_cloud_guard(self):
        cfg = settings()
        with patch('oh_streamer.subtitles.model_registry.get_model', side_effect=AssertionError('live metadata read')), \
             patch('oh_streamer.subtitles.model_registry.connection_api_key', return_value='fixture-rotated'):
            client = model_adapters.make_translation_client('user-translation', settings=cfg)
            self.assertEqual(client.base_url, 'https://fixture.invalid/v1')
            self.assertEqual(client._key, 'fixture-rotated')
            cfg['subtitle_cloud_enabled'] = False
            with self.assertRaises(model_adapters.ModelAdapterError):
                model_adapters.make_translation_client('user-translation', settings=cfg)

    def test_fixed_whisper_never_enables_sibling(self):
        cfg = settings()
        with patch('oh_streamer.subtitles.model_registry.connection_api_key', return_value='fixture'), \
             patch.object(model_adapters, '_cloud_asr', return_value={}) as run:
            model_adapters.run_registered_asr('user-whisper-large-v3', 'source', self.root, settings=cfg)
        self.assertFalse(run.call_args.kwargs['allow_model_fallback'])



    def test_legacy_snapshot_stays_legacy(self):
        legacy = snapshot_subtitle_settings(dict(subtitles=False))
        self.assertEqual(legacy['subtitle_runtime_contract'], 'legacy')
        self.assertEqual(legacy, snapshot_subtitle_settings(legacy))

    def test_profile_api_activates_only_new_jobs_and_persists(self):
        qs = queue_service.QueueService(self.root / 'queue')
        # Existing user: stored legacy contract; the profile API must flip only NEW jobs to cloud-first.
        qs.store.set_settings(subtitle_runtime_contract='legacy')
        old = qs.store.add('old.mp4', 'file', None, '1080p', dict(subtitles=True))
        before = copy.deepcopy(old.settings)
        profile = qs.store.get_settings()['subtitle_profile']
        profile['active_mode'] = 'cloud'
        profile['profiles']['cloud'].update(asr='user-whisper-large-v3', translation='user-translation')
        with patch.object(configuration, 'resolve', return_value=settings()['subtitle_resolved']):
            reply = qs.set_subtitle_profile(profile)
            self.assertEqual(reply['runtime_contract'], 'cloud-first-1')
            new = qs.store.add('new.mp4', 'file', None, '1080p', dict(subtitles=True))
            from oh_streamer.queue.subtitle_jobs import derive_subtitle_snapshot
            derived = derive_subtitle_snapshot(qs.store.get_settings(), new.settings)
        self.assertEqual(qs.store.get(old.id).settings, before)
        self.assertEqual(new.settings, dict(subtitles=True))                   # nothing frozen at add
        self.assertEqual(derived['subtitle_configs'][0]['translation']['model_id'], 'user-translation')
        restored = queue_service.QueueService(self.root / 'queue')
        self.assertEqual(restored.store.get(new.id).settings, new.settings)

    def test_legacy_primary_mirrors_profile_after_model_change(self):
        qs = queue_service.QueueService(self.root / 'queue')
        profile = qs.store.get_settings()['subtitle_profile']
        profile['active_mode'] = 'cloud'
        profile['profiles']['cloud'].update(asr='user-whisper-large-v3', translation='user-translation')
        from oh_streamer.queue.subtitle_jobs import derive_subtitle_snapshot
        resolved = settings()['subtitle_resolved']
        with patch.object(configuration, 'resolve', return_value=resolved):
            self.assertTrue(qs.set_subtitle_profile(profile)['ok'])
            first = qs.store.get_settings()
            self.assertEqual(first['subtitle_primary']['translation']['model_id'], 'user-translation')
            changed = copy.deepcopy(resolved)
            changed['models']['translation'] = dict(changed['models']['translation'], id='user-other')
            with patch.object(configuration, 'resolve', return_value=changed):
                profile['profiles']['cloud']['translation'] = 'user-other'
                self.assertTrue(qs.set_subtitle_profile(profile)['ok'])
                saved = qs.store.get_settings()
                derived = derive_subtitle_snapshot(saved, {})
        for snap in (saved, derived):
            self.assertEqual(snap['subtitle_primary']['translation']['model_id'], 'user-other')
        self.assertEqual(derived['subtitle_primary'], derived['subtitle_configs'][0])
        self.assertEqual(derived['subtitle_resolved']['models']['translation']['id'], 'user-other')








if __name__ == '__main__':
    unittest.main()
