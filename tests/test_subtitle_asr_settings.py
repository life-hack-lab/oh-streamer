"""Persisted production ASR modes through the existing global settings API."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.queue_service import QueueService


class SubtitleAsrSettingsTests(unittest.TestCase):
    def test_old_settings_default_dual_and_mode_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            service = QueueService.__new__(QueueService)
            service.store = QueueStore(Path(folder))
            service._model_options = Mock(return_value={})
            self.assertEqual(service.store.get_settings()['subtitle_asr_mode'], 'dual')
            for mode in ('single', 'dual'):
                response = service.set_settings({'subtitle_asr_mode': mode})
                self.assertTrue(response['ok'])
                self.assertEqual(QueueStore(Path(folder)).get_settings()['subtitle_asr_mode'], mode)

    def test_invalid_mode_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            service = QueueService.__new__(QueueService)
            service.store = QueueStore(Path(folder))
            for value in (None, True, 1, 'invalid', ['dual'], {'mode': 'dual'}):
                response = service.set_settings({'subtitle_asr_mode': value, 'subtitle_japanese': True})
                self.assertFalse(response['ok'])
                self.assertEqual(service.store.get_settings()['subtitle_asr_mode'], 'dual')
                self.assertFalse(service.store.get_settings()['subtitle_japanese'])


if __name__ == '__main__':
    unittest.main()
