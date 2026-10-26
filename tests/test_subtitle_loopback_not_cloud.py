import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.subtitles import model_adapters as adapters
from oh_streamer.subtitles import model_registry as registry


class LoopbackNotCloudTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = {'model_registry_path': str(Path(tmp.name) / 'models.json'), 'subtitle_cloud_enabled': False}
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('OH_STREAMER_SUBTITLE_CLOUD', None)

    def test_is_cloud_model(self):
        for url in ('http://localhost:11434/v1', 'http://127.0.0.1:1234/v1', 'http://[::1]:1234/v1'):
            self.assertFalse(registry.is_cloud_model({'provider': 'openai-compatible', 'base_url': url}))
        self.assertFalse(registry.is_cloud_model({'provider': 'local'}))
        self.assertTrue(registry.is_cloud_model({'provider': 'openai-compatible', 'base_url': 'https://api.groq.com/openai/v1'}))
        self.assertTrue(registry.is_cloud_model({'provider': 'openai-compatible', 'base_url': 'https://localhost.evil.com/v1'}))

    def test_save_connection_loopback_works_with_cloud_disabled(self):
        spec = {'api_family': 'openai-compatible', 'base_url': 'http://localhost:11434/v1', 'models': {'translation': 'qwen'}}
        with patch.object(adapters, 'test_model'):
            result = registry.test_save_connection(spec, self.cfg)
        self.assertNotEqual(result.get('code'), 'cloud_disabled', result)
        self.assertNotEqual(result.get('code'), 'missing_credential', result)

    def test_save_connection_cloud_host_still_blocked(self):
        spec = {'api_family': 'openai-compatible', 'base_url': 'https://api.example.com/v1',
                'models': {'translation': 'qwen'}, 'api_key': 'k'}
        with patch.object(adapters, 'test_model') as probe:
            result = registry.test_save_connection(spec, self.cfg)
        self.assertEqual(result.get('code'), 'cloud_disabled')
        probe.assert_not_called()

    def test_runtime_model_gate(self):
        local = {'provider': 'openai-compatible', 'base_url': 'http://127.0.0.1:1234/v1', 'kind': 'translation', 'id': 'x'}
        cloud = dict(local, base_url='https://api.example.com/v1')
        with patch('oh_streamer.subtitles.configuration.frozen_binding', side_effect=lambda s, i, k: dict(local if i == 'l' else cloud)), \
             patch.object(registry, 'model_api_key', return_value=''), \
             patch('oh_streamer.subtitles.configuration.runtime_plan', return_value=None):
            row, key = adapters._runtime_model('l', 'translation', self.cfg)[:2]
            self.assertEqual(row['base_url'], local['base_url'])
            with self.assertRaises(adapters.ModelAdapterError):
                adapters._runtime_model('c', 'translation', self.cfg)

    def test_binding_credentials_gate(self):
        local = {'provider': 'openai-compatible', 'base_url': 'http://localhost:11434/v1'}
        cloud = {'provider': 'openai-compatible', 'base_url': 'https://api.example.com/v1'}
        with patch.object(registry, 'model_api_key', return_value=''):
            self.assertEqual(adapters._binding_credentials(local, self.cfg), '')
            with self.assertRaises(PermissionError):
                adapters._binding_credentials(cloud, self.cfg)

    def test_update_model_key_empty_for_loopback(self):
        row = {'id': 'm1', 'kind': 'translation', 'provider': 'openai-compatible',
               'base_url': 'http://localhost:11434/v1', 'remote_model': 'q'}
        cloud = dict(row, id='m2', base_url='https://api.example.com/v1')
        with patch.object(registry, '_read', return_value=[row, cloud]), \
             patch.object(registry, '_write', create=True), \
             patch.object(adapters, 'test_model') as probe:
            registry.update_model_key('m2', '', self.cfg)
            probe.assert_not_called()  # cloud model still needs a key
            try:
                registry.update_model_key('m1', '', self.cfg)
            except Exception:
                pass
            probe.assert_called_once()


if __name__ == '__main__':
    unittest.main()
