"""Block-mode head settings are gone (M7 D5); old partial subtitles survive a reload without legacy fields."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.queue import queue_store


class SubtitleHeadStoreTests(unittest.TestCase):
    def test_head_settings_are_not_defaults_and_dropped_on_load(self):
        self.assertNotIn("subtitle_head_minutes", queue_store.DEFAULT_SETTINGS)
        self.assertNotIn("subtitle_head_seconds", queue_store.DEFAULT_SETTINGS)
        with tempfile.TemporaryDirectory() as tmp:
            store = queue_store.QueueStore(tmp)
            store.set_settings(subtitle_head_minutes=2, subtitle_head_seconds=4)
            reloaded = queue_store.QueueStore(tmp)
            self.assertNotIn("subtitle_head_minutes", reloaded.get_settings())
            self.assertNotIn("subtitle_head_seconds", reloaded.get_settings())

    def test_cloud_horizon_setting_is_gone_and_old_values_ignored(self):
        self.assertNotIn("subtitle_cloud_horizon_minutes", queue_store.DEFAULT_SETTINGS)
        with tempfile.TemporaryDirectory() as tmp:
            store = queue_store.QueueStore(tmp)
            store.set_settings(subtitle_cloud_horizon_minutes=45)       # an older queue.json carried this key
            reloaded = queue_store.QueueStore(tmp)
            self.assertNotIn("subtitle_cloud_horizon_minutes", reloaded.get_settings())
        from oh_streamer.subtitles import combos
        self.assertNotIn("cloud_horizon_minutes", combos.COMBO_TEMPLATE)
        combo = combos.normalize_combo({"id": "old-combo", "label": "Old", "cloud_horizon_minutes": 45})
        self.assertNotIn("cloud_horizon_minutes", combo)
        combos.normalize_combo({"id": "old-combo", "label": "Old", "cloud_horizon_minutes": 5})   # no longer validated

    def test_partial_survives_store_reload_without_legacy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = queue_store.QueueStore(tmp)
            item = store.add("sample.mp4", "file", {}, None, {})
            store.update(item.id, state="processing", subtitle={"status": "partial", "covered_s": 120,
                         "blocks_done": 1, "blocks_total": 5, "block_mode": True})
            reloaded = queue_store.QueueStore(tmp).get(item.id)
            self.assertEqual(reloaded.state, "queued")
            self.assertEqual(reloaded.subtitle["status"], "partial")
            self.assertEqual(reloaded.subtitle["covered_s"], 120)
            for key in ("blocks_done", "blocks_total", "block_mode"):
                self.assertNotIn(key, reloaded.subtitle)
            self.assertIn("partial", queue_store.SUBTITLE_STATES)


if __name__ == "__main__":
    unittest.main()
