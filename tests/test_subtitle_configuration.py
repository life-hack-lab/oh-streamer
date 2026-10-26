"""Editor configuration persistence and immutable queue snapshots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import threading
import json
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import unittest
from oh_streamer.queue.queue_store import QueueStore
from oh_streamer.queue.subtitle_jobs import subtitle_configuration, snapshot_subtitle_settings, SubtitleQueueMixin
from oh_streamer.subtitles.cli import translation_key

class ConfigurationTests(unittest.TestCase):
    def test_toggle_preserves_rows_and_snapshot(self):
        config = subtitle_configuration({"subtitle_combos_default": ["anime-gt7b"]})
        self.assertTrue(config["subtitle_comparison_enabled"])
        queued = snapshot_subtitle_settings(config)
        config["subtitle_comparison_enabled"] = False
        disabled = snapshot_subtitle_settings(config)
        self.assertEqual([r["id"] for r in disabled["subtitle_configs"]], ["primary"])
        self.assertEqual(len(disabled["subtitle_comparison_rows"]), 1)
        config["subtitle_comparison_rows"][0]["label"] = "changed"
        self.assertNotEqual(queued["subtitle_configs"][1]["label"], "changed")

    def test_store_add_freezes_nothing_and_follows_the_global_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            store.set_settings(subtitle_runtime_contract='legacy')
            item = store.add("source", "file", None, None, {"subtitles": True, "subtitle_overrides": {"subtitle_comparison_enabled": True}})
            store.set_settings(subtitle_asr="anime-whisper")
            saved = QueueStore(Path(tmp)).get(item.id)
            subtitle_keys = {k for k in saved.settings if k.startswith("subtitle_")}
            self.assertEqual(subtitle_keys, {"subtitle_overrides"})
            self.assertTrue(saved.settings["subtitles"])
            self.assertEqual(saved.subtitle["status"], "queued")

    def test_snapshot_freezes_the_track_plans_and_ignores_a_stale_copy(self):
        from oh_streamer.subtitles import configuration
        config = subtitle_configuration({"subtitle_combos_default": ["anime-gt7b"]})
        queued = snapshot_subtitle_settings(config)
        plans = queued["subtitle_track_plans"]
        self.assertEqual(list(plans), ["primary", "anime-gt7b"])
        self.assertIsNone(plans["primary"], "legacy snapshot: no frozen primary plan")
        self.assertEqual(plans["anime-gt7b"]["track"], "anime-gt7b")
        # the freeze is what the scheduler reads, whatever happens to the rows afterwards
        queued["subtitle_comparison_rows"][0]["label"] = "changed later"
        queued["subtitle_configs"][1]["asr"]["model"] = "changed-later"
        self.assertEqual(configuration.track_runtime_plans(queued), plans)
        # a snapshot derived from an older snapshot recomputes, it does not copy the stale plans
        queued["subtitle_track_plans"] = {"primary": None, "stale": {"track": "stale"}}
        again = snapshot_subtitle_settings(queued)
        self.assertEqual(list(again["subtitle_track_plans"]), ["primary", "anime-gt7b"])
        # comparison rows switched off: only the primary is frozen
        config["subtitle_comparison_enabled"] = False
        self.assertEqual(list(snapshot_subtitle_settings(config)["subtitle_track_plans"]), ["primary"])

    def test_refreshing_a_snapshot_refreezes_the_track_plans(self):
        from oh_streamer.queue.subtitle_recovery import refresh_snapshot_settings, SNAPSHOT_KEYS
        self.assertIn("subtitle_track_plans", SNAPSHOT_KEYS)
        item = snapshot_subtitle_settings(subtitle_configuration({"subtitle_combos_default": ["anime-gt7b"]}))
        item["subtitle_track_plans"] = {"primary": None, "stale": {"track": "stale"}}
        glob = subtitle_configuration({"subtitle_combos_default": []})
        glob["subtitle_comparison_enabled"] = False
        refreshed = refresh_snapshot_settings(item, glob)
        self.assertEqual(list(refreshed["subtitle_track_plans"]), ["primary"])

    def test_registry_translation_cache_uses_explicit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ja.json"
            source.write_text("{}")
            a = translation_key(source, "sakura", None, False, 30, model_identity={"id": "one"})
            b = translation_key(source, "sakura", None, False, 30, model_identity={"id": "two"})
            self.assertNotEqual(a, b)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            subtitle_configuration({"subtitle_comparison_rows": [{"id": "primary", "label": "duplicate"}]})

    def test_model_removal_blocks_saved_rows_and_queued_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            store.set_settings(subtitle_runtime_contract='legacy')
            svc = SimpleNamespace(store=store, _subtitle_thaw_all=lambda **kw: None, _subtitle_thaw=lambda *a, **kw: None)
            store.set_settings(subtitle_comparison_rows=[{"id": "extra", "label": "extra", "translation": {"model_id": "user-test"}}], subtitle_comparison_enabled=False)
            result = SubtitleQueueMixin.remove_subtitle_model(svc, "user-test")
            self.assertFalse(result["ok"])
            store.set_settings(subtitle_comparison_rows=[])
            store.add("file", "file", None, None, {"subtitles": True, "subtitle_configs": [{"id": "primary", "translation": {"model_id": "user-test"}}]})
            self.assertFalse(SubtitleQueueMixin.remove_subtitle_model(svc, "user-test")["ok"])

    def test_runtime_paths_are_never_stored_and_follow_the_global_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "old-models"
            models.mkdir()
            store = QueueStore(root / "queue")
            store.set_settings(subtitle_runtime_contract='legacy', subtitle_python=sys.executable, subtitle_models=str(models), detection_model="global-model")
            item = store.add("source", "file", None, None, {"subtitles": True, "detection_model": "chosen-model"})
            for key in ("subtitle_python", "subtitle_models", "subtitle_server"):
                self.assertNotIn(key, item.settings)
            snapshot = snapshot_subtitle_settings({**store.get_settings(), "subtitles": True})
            for key in ("subtitle_python", "subtitle_models", "subtitle_server"):
                self.assertNotIn(key, snapshot)
            svc = SimpleNamespace(_subtitle_environment=lambda: dict(python=sys.executable, models="new", server="new", runtime_available=False))
            item.settings["subtitle_configs"] = snapshot["subtitle_configs"]
            item.settings["subtitle_models"] = "stale"       # a stray stored value is ignored
            env = SubtitleQueueMixin._subtitle_item_environment(svc, item)
            self.assertEqual(env["models"], "new")
            self.assertTrue(env["runtime_available"])
            self.assertEqual(item.settings["detection_model"], "chosen-model")

    def test_local_model_test_reservation_rejects_concurrent_work_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            svc = SimpleNamespace(store=store, _lock=threading.RLock(), _wk=object(), is_player_busy=lambda: False,
                                  perf=SimpleNamespace(holding=False), _wake=threading.Event())
            spec = {"provider": "local"}
            with patch("oh_streamer.subtitles.model_registry.test_and_add_model") as test:
                self.assertFalse(SubtitleQueueMixin.test_add_subtitle_model(svc, spec)["ok"])
                test.assert_not_called()
                svc._wk = None
                def inference(*args):
                    self.assertTrue(svc._subtitle_model_test_busy)
                    self.assertFalse(SubtitleQueueMixin.test_add_subtitle_model(svc, spec)["ok"])
                    raise RuntimeError("test interrupted")
                test.side_effect = inference
                with self.assertRaises(RuntimeError):
                    SubtitleQueueMixin.test_add_subtitle_model(svc, spec)
                self.assertFalse(svc._subtitle_model_test_busy)
                self.assertTrue(svc._wake.is_set())

    def test_cloud_model_test_does_not_reserve_queue_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = SimpleNamespace(store=QueueStore(Path(tmp)))
            with patch("oh_streamer.subtitles.model_registry.test_and_add_model", return_value={"ok": True}) as test:
                self.assertTrue(SubtitleQueueMixin.test_add_subtitle_model(svc, {"provider": "openai-compatible"})["ok"])
                test.assert_called_once()

    def test_set_subtitle_configuration_cloud_first_contract_updates_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            svc = SimpleNamespace(store=store, _subtitle_thaw_all=lambda **kw: None, _subtitle_thaw=lambda *a, **kw: None)
            profile = {
                "schema_version": 1,
                "plan_version": "cloud-first-1",
                "active_mode": "cloud",
                "profiles": {
                    "cloud": {
                        "asr": None, "translation": None, "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": ["whisper-large-v3-turbo"],
                        "fallback": {"asr": [], "translation": [], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {"asr": {"mode": "dual", "model": "whisper-ja-1.5B"}},
                    },
                    "local": {
                        "asr": "whisper-ja-1.5B", "translation": "sakura-13b", "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": [],
                        "fallback": {"asr": [], "translation": [], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {"asr": {"mode": "single", "model": "whisper-ja-1.5B"}},
                    },
                    "custom": {
                        "asr": "whisper-ja-1.5B", "translation": "sakura-13b", "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": [],
                        "fallback": {"asr": [], "translation": [], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {"asr": {"mode": "single", "model": "whisper-ja-1.5B"}},
                    },
                },
                "migration": {"source": "test"},
            }
            store.set_settings(subtitle_runtime_contract="cloud-first-1", subtitle_profile=profile, subtitle_cloud_enabled=True)
            primary = {
                "id": "primary",
                "label": "主字幕",
                "asr": {"mode": "single", "model": "whisper-ja-1.5B"},
                "translation": {"profile": "sakura-v1"},
                "sentence": {},
            }
            res = SubtitleQueueMixin.set_subtitle_configuration(svc, primary, False, [])
            self.assertTrue(res["ok"])
            self.assertIn("subtitle_profile", res)
            cloud_p = res["subtitle_profile"]["profiles"]["cloud"]
            self.assertIsNone(cloud_p["asr"])
            self.assertIsNone(cloud_p["translation"])
            self.assertEqual(res["subtitle_profile"]["active_mode"], "cloud")

            profile["active_mode"] = "local"
            store.set_settings(subtitle_profile=profile)
            primary_cloud = {
                "id": "primary",
                "label": "主字幕",
                "asr": {"mode": "single", "model": "whisper-ja-1.5B"},
                "translation": {"model_id": "user-groq-trans"},
            }
            with patch("oh_streamer.subtitles.model_registry.resolve_model", return_value={"id": "user-groq-trans", "provider": "openai-compatible", "kind": "translation"}):
                res = SubtitleQueueMixin.set_subtitle_configuration(svc, primary_cloud, False, [])
                self.assertFalse(res["ok"])
                self.assertEqual(res.get("code"), "local_mode_cloud_candidate")

    def test_remove_subtitle_connection_blocked_when_referenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QueueStore(Path(tmp))
            svc = SimpleNamespace(store=store, _subtitle_thaw_all=lambda **kw: None, _subtitle_thaw=lambda *a, **kw: None)
            reg_path = Path(tmp) / "registry.json"
            reg_doc = {
                "schema_version": 2,
                "connections": [
                    {"id": "conn-groq", "api_family": "groq", "base_url": "https://api.groq.com/openai/v1"}
                ],
                "models": [
                    {"id": "whisper-ja-1.5B", "kind": "asr", "provider": "local"},
                    {"id": "sakura-13b", "kind": "translation", "provider": "local"},
                    {"id": "groq-qwen-custom", "kind": "translation", "provider": "openai-compatible", "connection_id": "conn-groq"}
                ]
            }
            reg_path.write_text(json.dumps(reg_doc), encoding="utf-8")

            profile = {
                "schema_version": 1,
                "plan_version": "cloud-first-1",
                "active_mode": "custom",
                "profiles": {
                    "cloud": {
                        "asr": "whisper-ja-1.5B", "translation": "sakura-13b", "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": [],
                        "fallback": {"asr": [], "translation": [], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {},
                    },
                    "local": {
                        "asr": "whisper-ja-1.5B", "translation": "sakura-13b", "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": [],
                        "fallback": {"asr": [], "translation": [], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {},
                    },
                    "custom": {
                        "asr": "whisper-ja-1.5B", "translation": "sakura-13b", "draft": None, "processing": "smart_horizon",
                        "whisper_policy": "auto", "whisper_models": [],
                        "fallback": {"asr": [], "translation": ["groq-qwen-custom"], "max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
                        "legacy_primary": {},
                    },
                },
                "migration": {"source": "test"},
            }
            from oh_streamer.subtitles.configuration import resolve
            store.set_settings(model_registry_path=str(reg_path))
            resolved = resolve(profile, store.get_settings())
            store.set_settings(subtitle_profile=profile, subtitle_resolved=resolved, subtitle_runtime_contract="cloud-first-1")

            # 1. Blocked when referenced in settings (custom fallback translation)
            res = SubtitleQueueMixin.remove_subtitle_connection(svc, "conn-groq")
            self.assertFalse(res["ok"])
            self.assertEqual(res["code"], "connection_in_use")
            self.assertIn("字幕設定引用", res["error"])

            # Clear setting reference
            profile["profiles"]["custom"]["fallback"]["translation"] = []
            resolved = resolve(profile, store.get_settings())
            store.set_settings(subtitle_profile=profile, subtitle_resolved=resolved)

            # 2. Blocked when referenced in queue item
            item = store.add("source", "file", None, None, {
                "subtitles": True,
                "subtitle_configs": [
                    {"id": "primary"},
                    {"id": "comp-groq", "translation": {"model_id": "groq-qwen-custom"}}
                ]
            })
            res = SubtitleQueueMixin.remove_subtitle_connection(svc, "conn-groq")
            self.assertFalse(res["ok"])
            self.assertEqual(res["code"], "connection_in_use")
            self.assertIn("佇列引用", res["error"])

            # Remove item from store
            store.remove(item.id)
            # Remove model from registry doc so connection is not referenced by remaining models
            reg_doc["models"] = [m for m in reg_doc["models"] if m.get("connection_id") != "conn-groq"]
            reg_path.write_text(json.dumps(reg_doc), encoding="utf-8")

            # 3. Can successfully remove unused connection
            res = SubtitleQueueMixin.remove_subtitle_connection(svc, "conn-groq")
            self.assertTrue(res["ok"])


