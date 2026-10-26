"""Queue model defaults and explicit application, without GPU processing."""
from tests.test_queue_service import ServiceTestBase
from unittest.mock import patch
from oh_streamer.player import player_settings as PS
from oh_streamer.queue import queue_store
import os


class QueueModelTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc.pause_all()
        self.options = {"detection_model": [{"value": "v4-fast"}, {"value": "v4-accurate"}],
                        "restoration_model": [{"value": "basicvsrpp-v1.2"}]}
        self.catalog = patch.object(self.svc, "_model_options", return_value=self.options)
        self.catalog.start()
        self.addCleanup(self.catalog.stop)

    def test_defaults_snapshot_and_player_add(self):
        with patch.object(PS, "load_file", return_value=({"detection_model": "v4-fast"}, None)):
            old = self.svc.add("https://example.com/old")["id"]
            self.assertTrue(self.svc.set_settings({"detection_model": "v4-accurate"})["ok"])
            new = self.svc.add("https://example.com/new")["id"]
            player = self.svc._ipc_add({"input": "https://example.com/player",
                                        "settings": {"detection_model": "v4-fast", "fp16": False}})["id"]
        self.assertEqual(self.svc.store.get(old).settings["detection_model"], "v4-fast")
        for item_id in (new, player):
            self.assertEqual(self.svc.store.get(item_id).settings["detection_model"], "v4-accurate")
        self.assertFalse(self.svc.store.get(player).settings["fp16"])
        loaded = queue_store.QueueStore(self.svc.store.root)
        self.assertEqual(loaded.get_settings()["detection_model"], "v4-accurate")

    def test_validation_and_apply_changes_cache_identity(self):
        item_id = self.add("https://example.com/item", detection_model="v4-fast", fp16=False)
        before = self.svc.store.get(item_id)
        self.assertFalse(self.svc.set_settings({"detection_model": "missing"})["ok"])
        self.svc.set_settings({"detection_model": "v4-accurate", "restoration_model": "basicvsrpp-v1.2"})
        self.assertTrue(self.svc.apply_model_settings(item_id)["ok"])
        after = self.svc.store.get(item_id)
        self.assertNotEqual(before.cache_key, after.cache_key)
        self.assertFalse(after.settings["fp16"])
        self.assertEqual(after.cache_key, queue_store.cache_key(after.target, after.quality, after.settings))
        for state in ("processing", "done"):
            self.svc.store.update(item_id, state=state)
            self.assertFalse(self.svc.apply_model_settings(item_id)["ok"])
        self.assertFalse(self.svc.apply_model_settings("missing")["ok"])

    def test_installed_catalog_includes_custom_models(self):
        weights = self.tmp / "weights"
        weights.mkdir()
        for filename in ("lada_mosaic_detection_model_v4_fast.pt", "lada_mosaic_detection_model_custom.pt",
                         "lada_mosaic_restoration_model_generic_v1.2.pth", "lada_mosaic_restoration_model_custom.pth"):
            (weights / filename).touch()
        with patch.dict(os.environ, {"LADA_MODEL_WEIGHTS_DIR": str(weights)}):
            options = type(self.svc)._model_options(self.svc)
        self.assertEqual({o["value"] for o in options["detection_model"]}, {"v4-fast", "custom"})
        self.assertEqual({o["value"] for o in options["restoration_model"]}, {"basicvsrpp-v1.2", "basicvsrpp-custom"})
