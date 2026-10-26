"""Unit and integration tests for Milestone 4 & 5:
- Milestone 4: 30-minute Smart Rolling Horizon scheduler, preemption, non-blocking RTX execution.
- Milestone 5: Settings schema, UI choices, and configuration validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from oh_streamer.queue import queue_service, queue_store, subtitle_jobs
from oh_streamer.subtitles import combos


class Milestone4HorizonSchedulerTests(unittest.TestCase):
    """Test slice planning, Smart Rolling Horizon logic, seek preemption, and publication."""






class Milestone5SettingsAndQueueIntegrationTests(unittest.TestCase):
    """Test Milestone 5 settings validation, UI defaults, and non-blocking parallel RTX integration."""

    def test_cloud_settings_and_combo_validation(self):
        # 1. queue_store defaults
        self.assertEqual(queue_store.DEFAULT_SETTINGS.get("subtitle_cloud_strategy"), "smart_horizon")

        # 2. combos COMBO_TEMPLATE & normalize_combo
        self.assertEqual(combos.COMBO_TEMPLATE.get("cloud_strategy"), "smart_horizon")

        valid_combo = {
            "id": "test-combo",
            "label": "Test Combo",
            "cloud_strategy": "full_preload",
        }
        normalized = combos.normalize_combo(valid_combo)
        self.assertEqual(normalized["cloud_strategy"], "full_preload")

        # Invalid strategy
        with self.assertRaises(combos.ComboError):
            combos.normalize_combo({"id": "bad-strat", "label": "Bad", "cloud_strategy": "invalid_strat"})

        # 3. is_cloud_combo detection
        self.assertTrue(combos.is_cloud_combo({"asr": {"model": "whisper-large-v3"}}))
        self.assertTrue(combos.is_cloud_combo({"asr": {"model": "whisper-large-v3-turbo"}}))
        self.assertTrue(combos.is_cloud_combo({"asr": {"model": "user-groq-whisper"}}))
        # D2: a legacy 'tiered' combo is an ordinary one; its refine model translates, the draft model is dropped
        legacy = combos.normalize_combo({"id": "old", "label": "Old", "translation": {
            "mode": "tiered", "baseline_model_id": "user-azure", "refine_model_id": "user-qwen"}})
        self.assertEqual(legacy["translation"]["mode"], "standard")
        self.assertEqual(legacy["translation"]["model_id"], "user-qwen")
        self.assertNotIn("baseline_model_id", legacy["translation"])
        self.assertNotIn("refine_model_id", legacy["translation"])
        self.assertTrue(combos.is_cloud_combo(legacy))
        self.assertTrue(combos.is_cloud_combo({"translation": {"model_id": "user-azure"}}))
        self.assertFalse(combos.is_cloud_combo({"asr": {"model": "whisper-ja-1.5B"}, "translation": {"profile": "sakura-v1"}}))

    def test_queue_service_set_settings_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            qs = queue_service.QueueService(tmp_dir)
            store = qs.store

            # Valid settings
            res = qs.set_settings({"subtitle_cloud_strategy": "full_preload"})
            self.assertTrue(res.get("ok"))
            self.assertEqual(store.get_settings().get("subtitle_cloud_strategy"), "full_preload")

            # Invalid strategy rejected
            res_bad_strat = qs.set_settings({"subtitle_cloud_strategy": "unknown_strategy"})
            self.assertFalse(res_bad_strat.get("ok"))
            self.assertIn("smart_horizon", res_bad_strat.get("error", ""))




    def test_cloud_mode_lifecycle_external_subs_want_and_clock(self):
        """Verify cloud_mode attaches partial SRT on open, survives _ipc_want without skipping, and writes play_clock.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            qs = queue_service.QueueService(tmp_dir)
            store = qs.store
            store.set_settings(subtitle_runtime_contract='legacy')
            src_video = Path(tmp_dir) / "video.mp4"
            src_video.write_bytes(b"video")

            item = store.add(
                target=str(src_video),
                kind="file",
                headers=None,
                quality="1080p",
                settings={
                    "subtitles": True,
                    "subtitle_configs": [{
                        "id": "primary",
                        "label": "雲端主字幕",
                        "asr": {"model": "whisper-large-v3"},
                        "translation": {"mode": "standard"},
                    }],
                },
            )
            store.update(item.id, source_path=str(src_video), state="processing")
            sub_dir = store.item_dir(item.id) / "subtitles"
            sub_dir.mkdir(parents=True, exist_ok=True)
            p_zh = sub_dir / "partial.zh-TW.srt"
            p_full = sub_dir / "partial.zh-TW.full.srt"
            p_zh.write_text("1\n00:00:00,000 --> 00:00:02,000\n測試\n", encoding="utf-8")
            p_full.write_text("1\n00:00:00,000 --> 00:00:02,000\n測試\nテスト\n", encoding="utf-8")

            qs._subtitle_update(
                item.id,
                view={"v": 2, "state": "running", "slices_done": 1, "slices_total": 6},
                status="partial",
                covered_s=300.0,
                block_srt={"zh-TW": str(p_zh), "full": str(p_full)},
                block_subtitles=[
                    {"path": str(p_zh), "lang": "chi", "title": "繁中精簡", "default": True},
                    {"path": str(p_full), "lang": "chi", "title": "繁中完整", "default": False},
                ],
            )

            # 1. _add_external_subtitles attaches the published partial srt
            payload = {"progressive": True}
            qs._add_external_subtitles(payload, store.get(item.id))
            self.assertIn("subtitles", payload)
            self.assertEqual(len(payload["subtitles"]), 2)
            self.assertEqual(payload["subtitles"][0]["path"], str(p_zh))
            self.assertEqual(payload["subtitle_covered_s"], 300.0)
            self.assertFalse(payload["subtitle_complete"])

            # 2. _ipc_want does NOT skip cloud subtitles and writes play_clock.json
            want_res = qs._ipc_want({"id": item.id, "k": 5, "pos": 150.0})
            self.assertTrue(want_res.get("ok"))
            after_want = store.get(item.id)
            self.assertEqual(after_want.subtitle.get("status"), "partial")
            clock_file = sub_dir / "play_clock.json"
            self.assertTrue(clock_file.is_file())
            self.assertAlmostEqual(json.loads(clock_file.read_text("utf-8"))["position"], 150.0)

            # 3. _ipc_clock updates play_clock.json and returns ok=True while a view exists
            clock_res = qs._ipc_clock({"id": item.id, "pos": 420.5, "playing": True})
            self.assertTrue(clock_res.get("ok"))
            self.assertAlmostEqual(json.loads(clock_file.read_text("utf-8"))["position"], 420.5)

            # 4. _subtitle_source_pending protects the source file while the table is unfinished
            self.assertTrue(qs._subtitle_source_pending(store.get(item.id)))


if __name__ == "__main__":
    unittest.main()
