"""Unit and integration tests for M8: subtitle_cloud_enabled global gatekeeper and consent protection."""

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.subtitle_jobs import (
    SubtitleQueueMixin,
    validate_combo_ids,
)
from oh_streamer.subtitles.combos import (
    ComboError,
    is_cloud_combo,
    is_cloud_enabled,
)


class CloudGatekeeperTests(unittest.TestCase):
    def setUp(self):
        self._orig_env = os.environ.get("OH_STREAMER_SUBTITLE_CLOUD")

    def tearDown(self):
        if self._orig_env is not None:
            os.environ["OH_STREAMER_SUBTITLE_CLOUD"] = self._orig_env
        else:
            os.environ.pop("OH_STREAMER_SUBTITLE_CLOUD", None)

    def test_default_cloud_disabled(self):
        os.environ.pop("OH_STREAMER_SUBTITLE_CLOUD", None)
        self.assertFalse(is_cloud_enabled({}))
        self.assertFalse(is_cloud_enabled(None))
        self.assertFalse(is_cloud_enabled({"subtitle_cloud_enabled": False}))
        self.assertTrue(is_cloud_enabled({"subtitle_cloud_enabled": True}))

    def test_env_var_override(self):
        os.environ["OH_STREAMER_SUBTITLE_CLOUD"] = "1"
        self.assertTrue(is_cloud_enabled({"subtitle_cloud_enabled": False}))
        os.environ["OH_STREAMER_SUBTITLE_CLOUD"] = "0"
        self.assertFalse(is_cloud_enabled({"subtitle_cloud_enabled": True}))

    def test_is_cloud_combo_detection(self):
        local_combo = {
            "id": "c1",
            "asr": {"model": "whisper-ja-1.5B"},
            "translation": {"profile": "sakura-v1"},
        }
        cloud_asr_combo = {
            "id": "c2",
            "asr": {"model": "user-remote-whisper"},
            "translation": {"profile": "sakura-v1"},
        }
        cloud_trans_combo = {
            "id": "c3",
            "asr": {"model": "whisper-ja-1.5B"},
            "translation": {"model_id": "user-remote-gpt"},
        }
        self.assertFalse(is_cloud_combo(local_combo))
        self.assertTrue(is_cloud_combo(cloud_asr_combo))
        self.assertTrue(is_cloud_combo(cloud_trans_combo))

    def test_validate_combo_ids_rejects_cloud_when_disabled(self):
        os.environ.pop("OH_STREAMER_SUBTITLE_CLOUD", None)
        presets = {
            "local_preset": {"id": "local_preset", "label": "Local", "asr": {"model": "anime-whisper"}, "translation": {}},
            "cloud_preset": {"id": "cloud_preset", "label": "Cloud", "asr": {"model": "user-whisper"}, "translation": {}},
        }
        # When cloud is disabled
        checked, err = validate_combo_ids(["cloud_preset"], presets=presets, settings={"subtitle_cloud_enabled": False})
        self.assertIsNone(checked)
        self.assertIn("雲端字幕功能未開啟", err)

        # When cloud is enabled
        checked, err = validate_combo_ids(["cloud_preset"], presets=presets, settings={"subtitle_cloud_enabled": True})
        self.assertEqual(checked, ["cloud_preset"])
        self.assertEqual(err, "")

    def test_set_subtitle_configuration_blocks_cloud_when_disabled(self):
        os.environ.pop("OH_STREAMER_SUBTITLE_CLOUD", None)
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            svc = SimpleNamespace(store=store, _subtitle_thaw_all=lambda **kw: None, _subtitle_thaw=lambda *a, **kw: None)

            # Try setting cloud translation as primary when disabled
            primary_cloud = {
                "id": "primary",
                "label": "主字幕",
                "translation": {"model_id": "user-groq"},
            }
            res = SubtitleQueueMixin.set_subtitle_configuration(svc, primary_cloud, False, [])
            self.assertFalse(res["ok"])
            self.assertIn("雲端字幕功能未開啟", res["error"])

            # Enable cloud in store settings
            store.set_settings(subtitle_cloud_enabled=True)
            res = SubtitleQueueMixin.set_subtitle_configuration(svc, primary_cloud, False, [])
            self.assertTrue(res["ok"])



if __name__ == "__main__":
    unittest.main()
