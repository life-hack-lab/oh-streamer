"""derive_subtitle_snapshot / content_signature / asr_signature, lean item settings at add, load-time migration."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.subtitle_jobs import (
    asr_signature, content_signature, derive_subtitle_snapshot, snapshot_subtitle_settings)
from oh_streamer.queue.subtitle_recovery import refresh_snapshot_settings

PATH_KEYS = ("subtitle_python", "subtitle_models", "subtitle_server")


def legacy_store(tmp, **settings):
    store = QueueStore(Path(tmp))
    store.set_settings(subtitle_runtime_contract="legacy", subtitle_combos_default=["anime-gt7b"], **settings)
    return store


class DeriveTests(unittest.TestCase):
    def test_overrides_apply_and_paths_are_never_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            glob = legacy_store(tmp, subtitle_python="py", subtitle_models=str(Path(tmp) / "m"),
                                subtitle_server="srv").get_settings()
            on = derive_subtitle_snapshot(glob, {"subtitles": True})
            off = derive_subtitle_snapshot(glob, {"subtitles": True, "subtitle_overrides": {"subtitle_comparison_enabled": False}})
            self.assertEqual([r["id"] for r in on["subtitle_configs"]], ["primary", "anime-gt7b"])
            self.assertEqual([r["id"] for r in off["subtitle_configs"]], ["primary"])
            self.assertEqual(sorted(off["subtitle_track_plans"]), ["primary"])
            for snap in (on, off):
                for key in PATH_KEYS:
                    self.assertNotIn(key, snap)
            for key in PATH_KEYS:
                self.assertNotIn(key, snapshot_subtitle_settings(dict(glob, subtitles=True)))

    def test_refresh_applies_overrides_and_keeps_item_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            glob = legacy_store(tmp).get_settings()
            item = {"subtitles": True, "detection_model": "chosen", "subtitle_python": "stale",
                    "subtitle_overrides": {"subtitle_comparison_enabled": False}}
            fresh = refresh_snapshot_settings(item, glob)
            self.assertEqual([r["id"] for r in fresh["subtitle_configs"]], ["primary"])
            self.assertEqual(fresh["detection_model"], "chosen")
            self.assertEqual(fresh["subtitle_overrides"], {"subtitle_comparison_enabled": False})
            self.assertNotIn("subtitle_python", fresh)

    def test_signatures_ignore_how_and_model_folder_but_not_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            glob = legacy_store(tmp).get_settings()
            base = derive_subtitle_snapshot(glob, {})
            sig, asr = content_signature(base), asr_signature(base)
            self.assertEqual(sig, content_signature(copy.deepcopy(base)))
            # model folder moved: same ids, different absolute paths
            moved = copy.deepcopy(base)
            moved["subtitle_resolved"] = json.loads(json.dumps(moved["subtitle_resolved"]).replace(
                '"model_path": "', '"model_path": "D:\\\\elsewhere\\\\'))
            self.assertEqual(content_signature(moved), sig)
            self.assertEqual(asr_signature(moved), asr)
            # how: fallback lists, processing, cloud consent
            how = copy.deepcopy(base)
            how["subtitle_resolved"]["fallback"] = {"asr": [{"id": "x"}], "translation": []}
            how["subtitle_resolved"]["policy"] = {"max_attempts": 9}
            how["subtitle_resolved"]["processing"] = "full_preload"
            how["subtitle_cloud_strategy"] = "full_preload"
            how["subtitle_cloud_enabled"] = not how.get("subtitle_cloud_enabled")
            self.assertEqual(content_signature(how), sig)
            # content: comparison rows, primary ASR, MT
            rows = derive_subtitle_snapshot(glob, {"subtitle_overrides": {"subtitle_comparison_enabled": False}})
            self.assertNotEqual(content_signature(rows), sig)
            self.assertEqual(asr_signature(rows), asr)               # rows do not change the primary ASR
            asr_changed = copy.deepcopy(base)
            asr_changed["subtitle_configs"][0]["asr"]["model"] = "anime-whisper"
            self.assertNotEqual(content_signature(asr_changed), sig)
            self.assertNotEqual(asr_signature(asr_changed), asr)
            mt_changed = copy.deepcopy(base)
            mt_changed["subtitle_configs"][0]["translation"]["profile"] = "other-profile"
            self.assertNotEqual(content_signature(mt_changed), sig)
            self.assertEqual(asr_signature(mt_changed), asr)
            resolved_changed = copy.deepcopy(base)
            resolved_changed["subtitle_resolved"]["models"]["translation"] = {"id": "user-other", "provider": "local"}
            self.assertNotEqual(content_signature(resolved_changed), sig)


class AddAndMigrationTests(unittest.TestCase):
    def test_add_and_add_stream_store_only_subtitles_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = legacy_store(tmp, subtitle_python="py")
            item = store.add("a", "file", None, None, {"subtitles": True, "subtitle_overrides": {"subtitle_comparison_enabled": False}})
            self.assertEqual(item.settings, {"subtitles": True, "subtitle_overrides": {"subtitle_comparison_enabled": False}})
            stream = store.add_stream("s", "url", None, {"subtitles": True})
            self.assertEqual(stream.settings, {"subtitles": True})

    def test_add_still_validates_the_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))            # fresh store: cloud-first profile is not configured
            with self.assertRaises(ValueError):
                store.add("a", "file", None, None, {"subtitles": True})

    def test_migration_pops_paths_and_unfreezes_only_unstarted_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = {"subtitles": True, "subtitle_python": "p", "subtitle_models": "m", "subtitle_server": "s",
                    "subtitle_configs": [{"id": "primary"}], "subtitle_combos": ["primary"],
                    "subtitle_overrides": {"subtitle_comparison_enabled": False}, "detection_model": "d"}
            raw = [dict(id="new", settings=copy.deepcopy(snap), subtitle=dict(status="queued")),
                   dict(id="run", settings=copy.deepcopy(snap), subtitle=dict(status="partial", covered_s=10)),
                   dict(id="planned", settings=copy.deepcopy(snap), subtitle=dict(status="queued")),
                   dict(id="off", settings={"subtitles": False, "subtitle_python": "p"}, subtitle={})]
            root = Path(tmp)
            (root / "queue.json").write_text(json.dumps(dict(settings=dict(subtitle_python="keep"), items=raw)))
            plan = root / "items" / "planned" / "subtitles"
            plan.mkdir(parents=True)
            (plan / "slice-plan.json").write_text("{}")
            store = QueueStore(root)
            by_id = {it.id: it for it in store.items()}
            for it in by_id.values():
                for key in PATH_KEYS:
                    self.assertNotIn(key, it.settings)
            self.assertEqual(by_id["new"].settings, {"subtitles": True, "subtitle_overrides": {"subtitle_comparison_enabled": False}, "detection_model": "d"})
            self.assertEqual(by_id["run"].settings["subtitle_configs"], [{"id": "primary"}])
            self.assertIn("subtitle_configs", by_id["planned"].settings)      # a slice plan exists: it has started
            self.assertEqual(store.get_settings()["subtitle_python"], "keep")  # global setting untouched

    def test_model_delete_guard_covers_items_that_follow_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = legacy_store(tmp)
            store.set_settings(subtitle_comparison_rows=[{"id": "extra", "label": "e", "translation": {"model_id": "user-test"}}],
                               subtitle_comparison_enabled=False)
            item = store.add("a", "file", None, None, {"subtitles": True})
            from oh_streamer.queue.subtitle_jobs import _guard_settings
            from oh_streamer.subtitles.configuration import references
            self.assertTrue(references(_guard_settings(item, store.get_settings()), "user-test"))
            self.assertFalse(references(item.settings, "user-test"))


if __name__ == "__main__":
    unittest.main()
