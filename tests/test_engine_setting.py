"""The `engine` setting: precedence, fallback, old files, model lists from the engine, the conditional selector.

No GPU, no weights: a fake engine is registered for the cases that need a second visible engine."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_STREAMER_ROOT, _STREAMER_ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer import engines
from oh_streamer.engines import base
from oh_streamer.player import player_settings as PS
from oh_streamer.queue import queue_worker as Q
from tests.test_queue_service import ServiceTestBase

FAKE_MODELS = base.ModelCatalog(("fake-det",), ("fake-res-a", "fake-res-b"))


def register_fake(test, *, visible=True):
    engines.register_engine("fake", lambda: mock.Mock(list_models=lambda: FAKE_MODELS), user_visible=visible,
                            label="Fake", catalog=lambda: FAKE_MODELS)
    test.addCleanup(engines.unregister_engine, "fake")


class EngineStateTestCase(unittest.TestCase):
    def setUp(self):
        saved = (list(engines._selected), list(engines._configured), set(engines._logged))
        self.addCleanup(self._restore, saved)
        engines._selected.clear()
        engines._configured.clear()
        engines._logged.clear()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("OH_STREAMER_ENGINE", None)

    @staticmethod
    def _restore(saved):
        engines._selected[:] = saved[0]
        engines._configured[:] = saved[1]
        engines._logged.clear()
        engines._logged.update(saved[2])


class PrecedenceTests(EngineStateTestCase):
    def test_default_is_lada(self):
        self.assertEqual(PS.DEFAULTS["engine"], "lada")
        self.assertEqual(engines.default_engine_name(), "lada")
        self.assertEqual(engines.effective_engine_name(None), "lada")

    def test_flag_beats_env_beats_settings_beats_default(self):
        register_fake(self)
        self.assertEqual(engines.effective_engine_name("fake"), "fake")                  # settings
        os.environ["OH_STREAMER_ENGINE"] = "passthrough"
        self.assertEqual(engines.effective_engine_name("fake"), "passthrough")           # env
        engines.set_default_engine("lada")
        self.assertEqual(engines.effective_engine_name("fake"), "lada")                  # --engine flag
        self.assertEqual(engines.explicit_engine_name(), "lada")

    def test_configured_engine_is_the_process_default_unless_overridden(self):
        register_fake(self)
        self.assertEqual(engines.set_configured_engine("fake"), "fake")
        self.assertEqual(engines.default_engine_name(), "fake")
        os.environ["OH_STREAMER_ENGINE"] = "passthrough"
        self.assertEqual(engines.default_engine_name(), "passthrough")

    def test_cli_flag_is_a_player_override_in_cli_overrides(self):
        args = mock.Mock(**{k: None for k in PS.KEYS})
        args.engine = "passthrough"
        self.assertEqual(PS.cli_overrides(args), {"engine": "passthrough"})
        self.assertEqual(PS.effective({"engine": "lada"}, {"engine": "passthrough"})["engine"], "passthrough")


class FallbackTests(EngineStateTestCase):
    def test_unknown_engine_falls_back_with_one_log_line(self):
        lines = []
        self.assertEqual(engines.resolve_engine_name("gone", lines.append), "lada")
        self.assertEqual(engines.resolve_engine_name("gone", lines.append), "lada")
        self.assertEqual(len(lines), 1)
        self.assertIn("gone", lines[0])
        self.assertEqual(engines.effective_engine_name("gone", lines.append), "lada")
        self.assertEqual(engines.set_configured_engine("gone"), "lada")

    def test_unknown_name_in_the_settings_file_does_not_crash_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text('{"engine": "gone", "fp16": true}', encoding="utf-8")
            saved, warning = PS.load_file(path)
        self.assertIsNone(warning)
        self.assertEqual(saved, {"engine": "gone", "fp16": True})
        self.assertEqual(engines.effective_engine_name(saved.get("engine")), "lada")

    def test_bad_value_type_is_dropped_by_the_file_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text('{"engine": 5}', encoding="utf-8")
            saved, warning = PS.load_file(path)
        self.assertEqual(saved, {})
        self.assertIn("engine", warning)

    def test_worker_job_with_unknown_engine_runs_lada_and_says_so(self):
        job, notes = Q.normalize_job({"source": "a.mp4", "work_dir": "w", "output": "o.mkv",
                                      "settings": {"engine": "gone"}})
        self.assertEqual(job["settings"]["engine"], "lada")
        self.assertEqual(len(notes), 1)


class OldFilesTests(EngineStateTestCase):
    def test_file_without_the_key_behaves_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text('{"detection_model": "v4-accurate"}', encoding="utf-8")
            saved, warning = PS.load_file(path)
        self.assertIsNone(warning)
        cfg = PS.effective(saved, {})
        self.assertEqual(cfg["engine"], "lada")
        self.assertEqual({k: v for k, v in cfg.items() if k != "engine"},
                         {**{k: v for k, v in PS.DEFAULTS.items() if k != "engine"}, "detection_model": "v4-accurate"})
        self.assertNotIn("engine", saved)

    def test_saving_only_writes_what_was_picked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            PS.save_file(path, {"detection_model": "v4-fast"})
            self.assertNotIn("engine", path.read_text(encoding="utf-8"))
            PS.save_file(path, {"engine": "passthrough"})
            self.assertEqual(PS.load_file(path)[0], {"engine": "passthrough"})

    def test_worker_job_without_the_key_is_lada(self):
        job, notes = Q.normalize_job({"source": "a.mp4", "work_dir": "w", "output": "o.mkv", "settings": {}})
        self.assertEqual(job["settings"]["engine"], "lada")
        self.assertEqual(notes, [])

    def test_worker_job_names_another_engine(self):
        job, _ = Q.normalize_job({"source": "a.mp4", "work_dir": "w", "output": "o.mkv",
                                  "settings": {"engine": "passthrough"}})
        self.assertEqual(job["settings"]["engine"], "passthrough")

    def test_engine_setting_travels_to_the_queue_with_the_player_settings(self):
        from oh_streamer.player import queue_bridge
        self.assertIn("engine", queue_bridge.QUEUE_SETTING_KEYS)
        payload = queue_bridge.build_add_payload("C:\\a.mp4", {}, None, None, dict(PS.DEFAULTS, engine="passthrough"), False)
        self.assertEqual(payload["settings"]["engine"], "passthrough")


class ModelListTests(EngineStateTestCase):
    def test_catalog_comes_from_the_engine(self):
        register_fake(self)
        self.assertEqual(engines.model_catalog("fake"), FAKE_MODELS)
        self.assertEqual(engines.model_catalog("passthrough").detection, ("passthrough",))

    def test_saved_model_not_offered_by_the_engine_uses_the_engines_default(self):
        cfg, notes = PS.fix_models(dict(PS.DEFAULTS, engine="fake"), {}, list(FAKE_MODELS.detection),
                                   list(FAKE_MODELS.restoration))
        self.assertEqual((cfg["detection_model"], cfg["restoration_model"]), ("fake-det", "fake-res-a"))
        self.assertEqual(len(notes), 2)

    def test_lada_defaults_unchanged(self):
        cfg, notes = PS.fix_models(dict(PS.DEFAULTS), {}, ["v4-accurate", "v4-fast"], ["basicvsrpp-v1.2"])
        self.assertEqual((cfg["detection_model"], cfg["restoration_model"]), ("v4-fast", "basicvsrpp-v1.2"))
        self.assertEqual(notes, [])


class SelectorTests(EngineStateTestCase):
    def test_passthrough_is_hidden_and_lada_is_the_only_visible_engine(self):
        self.assertEqual(engines.visible_engine_names(), ("lada",))
        self.assertIn("passthrough", engines.engine_names())

    def _menu(self, engine_list):
        cfg = dict(PS.DEFAULTS)
        return PS.build_menu(cfg, False, ["v4-fast"], ["basicvsrpp-v1.2"], False, "libx264", platform="win32",
                             engines=engine_list)

    def test_player_menu_has_no_selector_with_one_visible_engine(self):
        keys = [it["key"] for it in self._menu([(n, engines.engine_label(n)) for n in engines.visible_engine_names()])]
        self.assertNotIn("engine", keys)
        self.assertEqual(keys, [it["key"] for it in self._menu(())])        # identical to the menu without the parameter

    def test_player_menu_has_the_selector_with_two_visible_engines(self):
        register_fake(self)
        names = engines.visible_engine_names()
        self.assertEqual(names, ("lada", "fake"))
        menu = self._menu([(n, engines.engine_label(n)) for n in names])
        item = menu[0]
        self.assertEqual((item["key"], item["label"], item["value"]), ("engine", "修復引擎", "lada"))
        self.assertEqual([o["v"] for o in item["options"]], ["lada", "fake"])
        self.assertEqual(PS.parse_value("engine", "fake", menu), "fake")
        with self.assertRaises(ValueError):
            PS.parse_value("engine", "passthrough", menu)                    # hidden engines are never offered

    def test_hidden_engine_does_not_count(self):
        register_fake(self, visible=False)
        self.assertEqual(engines.visible_engine_names(), ("lada",))


class QueueServiceTests(ServiceTestBase, EngineStateTestCase):
    def setUp(self):
        EngineStateTestCase.setUp(self)
        ServiceTestBase.setUp(self)
        self.svc.pause_all()
        player_file = mock.patch.object(PS, "load_file", return_value=({}, None))
        player_file.start()
        self.addCleanup(player_file.stop)

    def test_state_offers_one_engine_today(self):
        state = self.svc.get_state()
        self.assertEqual(state["engine_options"], [{"value": "lada", "label": "Lada"}])
        self.assertEqual(state["settings"]["engine"], "lada")

    def test_state_offers_the_selector_data_with_a_second_visible_engine(self):
        register_fake(self)
        self.assertEqual([o["value"] for o in self.svc.get_state()["engine_options"]], ["lada", "fake"])

    def test_model_options_follow_the_selected_engine(self):
        register_fake(self)
        self.assertTrue(self.svc.set_settings({"engine": "fake"})["ok"])
        options = self.svc._model_options()
        self.assertEqual([o["value"] for o in options["detection_model"]], ["fake-det"])
        self.assertEqual([o["value"] for o in options["restoration_model"]], ["fake-res-a", "fake-res-b"])
        self.assertEqual(self.svc.get_state()["model_options"], options)
        self.assertTrue(self.svc.set_settings({"detection_model": "fake-det"})["ok"])
        self.assertFalse(self.svc.set_settings({"detection_model": "v4-fast"})["ok"])

    def test_hidden_or_unknown_engine_cannot_be_chosen(self):
        self.assertFalse(self.svc.set_settings({"engine": "passthrough"})["ok"])
        self.assertFalse(self.svc.set_settings({"engine": "gone"})["ok"])
        self.assertTrue(self.svc.set_settings({"engine": None})["ok"])

    def test_new_items_and_jobs_carry_the_engine(self):
        register_fake(self)
        old = self.svc.add("https://example.com/old")["id"]
        self.assertEqual(self.svc.store.get(old).settings["engine"], "lada")
        self.assertTrue(self.svc.set_settings({"engine": "fake"})["ok"])
        new = self.svc.add("https://example.com/new")["id"]
        self.assertEqual(self.svc.store.get(new).settings["engine"], "fake")
        self.assertEqual(self.svc._build_job(self.svc.store.get(new))["settings"]["engine"], "fake")
        self.assertEqual(self.svc._build_job(self.svc.store.get(old))["settings"]["engine"], "lada")   # pinned at add

    def test_item_without_the_key_follows_the_queue_engine_lada_by_default(self):
        item = self.add("https://example.com/legacy")
        self.assertNotIn("engine", self.svc.store.get(item).settings)
        self.assertEqual(self.svc._build_job(self.svc.store.get(item))["settings"]["engine"], "lada")

    def test_unknown_engine_in_the_player_file_falls_back_to_lada(self):
        with mock.patch.object(PS, "load_file", return_value=({"engine": "gone"}, None)):
            self.assertEqual(self.svc._queue_engine(), "lada")


if __name__ == "__main__":
    unittest.main()
