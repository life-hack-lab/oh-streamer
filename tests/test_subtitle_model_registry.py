import json
from pathlib import Path
from unittest.mock import patch
import unittest
import tempfile
from oh_streamer.subtitles import model_registry as registry
from oh_streamer.subtitles import model_adapters as adapters

def settings(tmp_path):
    return {'model_registry_path':str(tmp_path/'models.json')}

class ModelRegistryTests(unittest.TestCase):
    def test_failed_inference_never_registers(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        cfg=settings(tmp_path)
        with patch.object(adapters,'test_model',side_effect=RuntimeError('secret-key')):
            result=registry.test_and_add_model({'kind':'translation','provider':'openai-compatible','base_url':'https://example.org/v1','remote_model':'translator','api_key':'secret-key'},cfg)
        assert not result['ok']
        assert 'secret-key' not in json.dumps(result)
        assert not Path(cfg['model_registry_path']).exists()
    
    def test_registered_model_safe_removable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        cfg=settings(tmp_path)
        spec={'kind':'translation','provider':'openai-compatible','base_url':'https://example.org/v1','remote_model':'translator','api_key':'secret-key'}
        with patch.object(adapters,'test_model'), patch.object(registry,'_secret',return_value='opaque'):
            result=registry.test_and_add_model(spec,cfg)
        assert result['ok']
        assert 'credential' not in result['model']
        assert 'secret-key' not in Path(cfg['model_registry_path']).read_text()
        assert 'credential' not in json.dumps(registry.list_models(cfg))
        assert registry.remove_model(result['model']['id'],cfg)['ok']
        assert not registry.remove_model('anime-whisper',cfg)['ok']
    
    def test_reject_unsupported_and_credential_urls(self):
        with self.assertRaises(ValueError):
            adapters.validate_spec({'kind':'translation','provider':'local','recipe':'arbitrary','model_path':'anything'})
        with self.assertRaises(ValueError):
            adapters.validate_spec({'kind':'asr','provider':'openai-compatible','base_url':'https://key@example.org/v1','remote_model':'a'})
    
    def test_cloud_translation_numbered_test_required(self):
        row=adapters.validate_spec({'kind':'translation','provider':'openai-compatible','base_url':'https://example.org/v1','remote_model':'a'})
        with patch.object(adapters,'_request',return_value={'choices':[{'message':{'content':'one line'}}]}):
            with self.assertRaises(adapters.ModelAdapterError):
                adapters.test_model(row)
        with patch.object(adapters,'_request',return_value={'choices':[{'message':{'content':'1. 你好\n2. 謝謝'}}]}):
            adapters.test_model(row)
            client=adapters.CloudTranslationClient(row)
            assert client.model=='sakura'
    
    def test_cloud_asr_requires_real_timestamp_segments(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        import wave
        source=tmp_path/'source.wav'
        with wave.open(str(source),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\0\0'*16000)
        row={'remote_model':'whisper','base_url':'https://example.org/v1'}
        with patch.object(adapters,'_request',return_value={'text':'hello'}):
            with self.assertRaises(adapters.ModelAdapterError):
                adapters._cloud_asr(row,source,tmp_path/'bad')
        with patch.object(adapters,'_request',return_value={'segments':[{'start':0,'end':0.9,'text':'こんにちは'}]}):
            result=adapters._cloud_asr(row,source,tmp_path/'good')
            assert result['segments'][0]['end']==0.9
            assert (tmp_path/'good/ja.srt').is_file()
    def test_cloud_asr_chunks_preserve_offsets(self):
        import wave
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = root/'long.wav'
        with wave.open(str(source),'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\0\0' * 16000 * 301)
        row={'remote_model':'whisper','base_url':'https://example.org/v1'}
        phases=[]
        with patch.object(adapters,'_request',return_value={'segments':[{'start':0,'end':0.9,'text':'こんにちは'}]}) as request:
            result=adapters._cloud_asr(row,source,root/'output',on_phase=lambda *args: phases.append(args))
        self.assertEqual([p[1] for p in phases],[0,99,100])
        self.assertTrue(all(p[0]=='asr' and '雲端語音辨識' in p[2] for p in phases))
        self.assertEqual(request.call_count,2)
        self.assertEqual([s['start'] for s in result['segments']],[0,300])
    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows DPAPI')
    def test_windows_credential_protection_roundtrip(self):
        protected=registry._secret('test-only-secret')
        self.assertNotIn('test-only-secret',protected)
        self.assertEqual(registry._secret(protected,True),'test-only-secret')
    def test_local_inference_uses_managed_runtime_subprocess(self):
        import sys
        from types import SimpleNamespace
        row={'kind':'translation','provider':'local','recipe':'sakura-v1','model_path':'model.gguf','llama_server_path':'llama-server'}
        with patch('oh_streamer.subtitles.environment.detect',return_value={'python':sys.executable}), patch.object(adapters.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            adapters.test_model(row,settings={'subtitle_python':sys.executable})
        self.assertEqual(run.call_args.args[0][0],sys.executable)
        self.assertIn('PYTHONPATH',run.call_args.kwargs['env'])
        self.assertNotIn('api_key',run.call_args.kwargs['input'])
    def test_builtin_status_uses_configured_models_and_real_files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        cfg={'model_registry_path':str(root/'registry.json'),'subtitle_models':str(root/'models')}
        folder=root/'models/anime-whisper'
        folder.mkdir(parents=True)
        (folder/'config.json').write_text('{}')
        catalog={m['id']:m for m in registry.list_models(cfg)}
        self.assertEqual(catalog['anime-whisper']['status'],'missing')
        (folder/'model.safetensors').mkdir()
        self.assertEqual(next(m for m in registry.list_models(cfg) if m['id']=='anime-whisper')['status'],'missing')
        (folder/'model.bin').write_bytes(b'weights')
        catalog={m['id']:m for m in registry.list_models(cfg)}
        self.assertEqual(catalog['anime-whisper']['status'],'ready')
        self.assertEqual(Path(catalog['anime-whisper']['model_path']),folder)
        gguf=Path(catalog['sakura-v1']['model_path'])
        gguf.mkdir(parents=True)
        self.assertEqual(next(m for m in registry.list_models(cfg) if m['id']=='sakura-v1')['status'],'missing')

    def test_cloud_silence_valid_in_production_but_not_registration(self):
        import wave
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        source=root/'silent.wav'
        with wave.open(str(source),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\0\0'*16000*301)
        row={'kind':'asr','provider':'openai-compatible','remote_model':'whisper','base_url':'https://example.org/v1'}
        with patch.object(adapters,'_request',side_effect=[{'segments':[]},{'segments':[{'start':0,'end':0.9,'text':'こんにちは'}]}]):
            result=adapters._cloud_asr(row,source,root/'partial-silence')
        self.assertEqual(result['segments'][0]['start'],300)
        with patch.object(adapters,'_request',return_value={'segments':[]}):
            self.assertEqual(adapters._cloud_asr(row,source,root/'silence')['segments'],[])
            with self.assertRaises(adapters.ModelAdapterError):
                adapters.test_model(row,test_audio_path=source)

    def test_cloud_clamps_overlapping_timestamps(self):
        import wave
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        source=root/'sample.wav'
        with wave.open(str(source),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\0\0'*32000)
        row={'remote_model':'whisper','base_url':'https://example.org/v1'}
        response={'segments':[{'start':0,'end':1,'text':'はい'},{'start':0.5,'end':1.5,'text':'こんにちは'}]}
        # sample video A: one overlap no longer fails the whole chunk; the later segment starts where the earlier ends
        with patch.object(adapters,'_request',return_value=response):
            segments=adapters._cloud_asr(row,source,root/'out')['segments']
        texts=[s['text'] for s in segments]
        starts=[s['start'] for s in segments]
        self.assertEqual(texts,['はい','こんにちは'])
        self.assertEqual(starts,[0,1])

    def test_http_post_rejects_query_credentials_and_redacts_errors(self):
        from oh_streamer.subtitles.providers.http import ProviderError, http_post, redact_sensitive_text
        fake_groq = 'gsk_1234567890abcdefghij123456'
        fake_google = 'AIzaSyA1234567890abcdefghijk123456789'
        # 1. URL query secrets must be rejected before any transport is called
        called = False
        def _boom_transport(req, timeout):
            nonlocal called
            called = True
            return b'{}'
        with self.assertRaises(ProviderError) as ctx:
            http_post(f'https://example.org/v1/chat?key={fake_google}', b'{}', transport=_boom_transport)
        self.assertEqual(ctx.exception.kind, 'unsupported')
        self.assertFalse(called)
        self.assertNotIn(fake_google, str(ctx.exception))

        # 2. ProviderError message redaction
        err = ProviderError('auth', f'Failed with Bearer {fake_groq} and api_key={fake_google}')
        self.assertNotIn(fake_groq, str(err))
        self.assertNotIn(fake_google, str(err))
        self.assertIn('[REDACTED]', str(err))
        self.assertNotIn(fake_groq, redact_sensitive_text(f'token is {fake_groq}'))

    def test_readonly_env_var_fallback_never_persists_to_disk(self):
        import os
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        cfg = settings(tmp_path)
        env_secret = 'gsk_env_only_secret_value_1234567890'
        spec = {'kind': 'translation', 'provider': 'openai-compatible', 'base_url': 'https://example.org/v1', 'remote_model': 'translator'}
        with patch.dict(os.environ, {'OH_STREAMER_SUBTITLE_API_KEY': env_secret}, clear=False), \
             patch.object(adapters, 'test_model') as mock_test:
            result = registry.test_and_add_model(spec, cfg)
            self.assertTrue(result['ok'])
            self.assertEqual(mock_test.call_args.kwargs['api_key'], env_secret)
            model_id = result['model']['id']
            # Disk file must not contain the env secret or any credential field
            disk_text = Path(cfg['model_registry_path']).read_text(encoding='utf-8')
            self.assertNotIn(env_secret, disk_text)
            self.assertNotIn('credential', disk_text)
            # Runtime lookup resolves from env var
            row = registry.get_model(model_id, cfg)
            self.assertEqual(registry.model_api_key(row), env_secret)
        # Once env var is removed, key is empty
        row = registry.get_model(model_id, cfg)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(registry.model_api_key(row), '')

    def test_secret_scanner_detects_leaked_keys(self):
        import importlib.util
        script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'check_no_secrets.py'
        spec = importlib.util.spec_from_file_location('check_no_secrets', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        leaked = 'API_KEY = "' + 'gsk_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4' + '"'
        self.assertTrue(mod.scan_text(leaked, 'dummy.py'))
        safe = 'API_KEY = os.environ.get("OH_STREAMER_SUBTITLE_API_KEY")'
        self.assertEqual(mod.scan_text(safe, 'dummy.py'), [])

    def test_update_model_key_and_builtin_cloud_asr_probe(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_path = Path(tmp.name)
        cfg = settings(tmp_path)
        first_key = 'gsk_initial_secret_value_111111111'  # pragma: allowlist secret - test key
        second_key = 'gsk_rotated_secret_value_222222222'  # pragma: allowlist secret - test key
        spec = {
            'kind': 'asr',
            'provider': 'openai-compatible',
            'label': 'Groq Whisper',
            'base_url': 'https://api.groq.com/openai/v1',
            'remote_model': 'whisper-large-v3-turbo',
            'api_family': 'groq',
            'api_key': first_key,
            'test_audio_path': '__builtin__',
        }
        response = {'segments': [{'start': 0.0, 'end': 1.0, 'text': 'こんにちは'}]}
        with patch.object(adapters, '_request', return_value=response) as mock_req, \
             patch.object(registry, '_secret', side_effect=lambda v, read=False: v.removeprefix('dpapi:enc:')[::-1] if read else f'dpapi:enc:{v[::-1]}'):
            added = registry.test_and_add_model(spec, cfg)
            self.assertTrue(added['ok'])
            self.assertEqual(added['model']['auth_source'], 'dpapi')
            self.assertEqual(added['model']['api_family'], 'groq')
            model_id = added['model']['id']
            self.assertEqual(mock_req.call_args.args[3], first_key)

            # Rotate the key with update_model_key
            rotated = registry.update_model_key(model_id, second_key, cfg)
            self.assertTrue(rotated['ok'])
            self.assertEqual(rotated['model']['auth_source'], 'dpapi')
            self.assertEqual(mock_req.call_args.args[3], second_key)
            row = registry.get_model(model_id, cfg)
            self.assertEqual(registry.model_api_key(row), second_key)
            disk_text = Path(cfg['model_registry_path']).read_text(encoding='utf-8')
            self.assertNotIn(first_key, disk_text)
            self.assertNotIn(second_key, disk_text)
            self.assertNotIn('credential', json.dumps(registry.list_models(cfg)))


