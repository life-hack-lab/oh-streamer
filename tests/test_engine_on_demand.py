"""The player's models load on demand and leave the GPU when idle (Engine.ensure_loaded / unload, SettingsControl
without models, PlayerController.release_idle_engine)."""
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
for _root in (_STREAMER_ROOT, _SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from oh_streamer.player import player_settings as PS  # noqa: E402
from oh_streamer.player import oh_player as L  # noqa: E402  (imports torch + lada, no GPU work)
from tests.test_player_overrides import DETECTORS, RESTORERS, FakeModels, OverrideTestBase  # noqa: E402


def engine_args():
    return SimpleNamespace(device="auto", output="null", settings_file=None)


class SettingsWithoutModelsTests(unittest.TestCase):
    def make(self, **cfg):
        return L.SettingsControl(None, {}, {**PS.DEFAULTS, **cfg}, None, DETECTORS, RESTORERS, False)

    def test_menu_builds_without_models(self):
        with patch.object(L.device_backend, "rife_available", lambda _device: True):
            sc = self.make(fp16=True)
            keys = [it["key"] for it in sc.menu()]
        self.assertIn("fp16", keys)
        self.assertIn("vsr", keys)
        self.assertTrue(sc.fp16_on)

    def test_fp16_setting_decides_before_the_models_load(self):
        self.assertFalse(self.make(fp16=False).fp16_on)
        self.assertTrue(self.make(fp16=True).fp16_on)

    def test_attach_and_detach_keep_the_resolved_fp16(self):
        sc = self.make(fp16=False)
        factory = FakeModels()
        factory.models = SimpleNamespace(detection="v4-fast", restoration="basicvsrpp-v1.2", fp16=True)
        sc.attach(factory, None)
        self.assertTrue(sc.fp16_on)
        sc.detach()
        self.assertIsNone(sc.lada)
        self.assertTrue(sc.fp16_on)        # what the models used stays what the menu shows

    def test_commit_works_and_persists_without_models(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as d:
            path = Path(d) / "oh_settings.json"
            sc = L.SettingsControl(str(path), {}, dict(PS.DEFAULTS), None, DETECTORS, RESTORERS, False)
            sc.commit({"detection_model": "v4-accurate", "volume": 40.0})
            self.assertEqual(sc.cfg["detection_model"], "v4-accurate")
            self.assertTrue(path.exists())


class EngineOnDemandTests(unittest.TestCase):
    def make(self):
        return L.Engine(engine_args(), {}, dict(PS.DEFAULTS), DETECTORS, RESTORERS)

    def test_nothing_is_loading_until_asked(self):
        e = self.make()
        self.assertFalse(e.loading)
        self.assertFalse(e.ready.is_set())
        self.assertIsNone(e.lada)

    def test_ensure_loaded_starts_one_load(self):
        e = self.make()
        starts = []
        e.load_async = lambda: starts.append(1)
        e.ensure_loaded()
        e.ensure_loaded()
        self.assertEqual(starts, [1])
        self.assertTrue(e.loading)

    def test_load_uses_the_settings_in_use_and_attaches_the_models(self):
        e = self.make()
        sc = e._ensure_sc()
        sc.commit({"detection_model": "v4-accurate"}, write=False)   # changed while no models were loaded
        factory = FakeModels()
        with patch.object(L, "get_engine", lambda: factory), \
                patch.object(L.device_backend, "select_device", lambda _r: SimpleNamespace(type="cpu")), \
                patch.object(L, "gpu_has_fp16_acceleration", lambda _d: False), \
                patch.object(L, "log", lambda *_a: None):
            e.load()
        self.assertEqual(factory.loads[-1][0], "v4-accurate")
        self.assertIs(e.sc, sc)
        self.assertIs(sc.lada, factory)
        self.assertTrue(e.ready.is_set())

    def test_unload_frees_the_models_and_allows_a_new_load(self):
        e = self.make()
        sc = e._ensure_sc()
        factory = FakeModels()
        factory.models = SimpleNamespace(detection="v4-fast", restoration="basicvsrpp-v1.2", fp16=False)
        released = []
        factory.release = lambda: released.append(1)
        sc.attach(factory, None)
        e.lada, e._load_started = factory, True
        e.ready.set()
        with patch.object(L, "log", lambda *_a: None):
            self.assertTrue(e.unload())
        self.assertEqual(released, [1])
        self.assertIsNone(e.lada)
        self.assertIsNone(sc.lada)
        self.assertFalse(e.ready.is_set())
        self.assertFalse(e.loading)
        starts = []
        e.load_async = lambda: starts.append(1)
        with patch.object(L, "log", lambda *_a: None):
            e.ensure_loaded()
        self.assertEqual(starts, [1])

    def test_unload_does_nothing_without_models_or_while_loading(self):
        e = self.make()
        self.assertFalse(e.unload())
        e._load_started = True            # loading, not ready
        self.assertFalse(e.unload())


class IdleReleaseTests(OverrideTestBase):
    def setUp(self):
        super().setUp()
        p = self.player
        self.unloads = []
        p.engine.unload = lambda: self.unloads.append(1) or True
        p.engine.loading = False
        p.engine_idle_since = None
        p.loaded, p.tail = False, None
        p.pending_open = None
        p.opener = SimpleNamespace(busy=False)
        p.update_status = lambda: None

    def test_loaded_models_are_released_after_the_idle_time(self):
        p = self.player
        p.release_idle_engine(100.0)
        p.release_idle_engine(100.0 + p.ENGINE_IDLE_UNLOAD_S - 1)
        self.assertEqual(self.unloads, [])
        p.release_idle_engine(100.0 + p.ENGINE_IDLE_UNLOAD_S)
        self.assertEqual(self.unloads, [1])

    def test_a_live_session_keeps_the_models(self):
        p = self.player
        p.session = object()
        p.release_idle_engine(0.0)
        p.release_idle_engine(10_000.0)
        self.assertEqual(self.unloads, [])

    def test_the_clock_restarts_after_a_live_session(self):
        p = self.player
        p.release_idle_engine(0.0)
        p.session = object()
        p.release_idle_engine(500.0)         # busy: the idle clock is dropped
        p.session = None
        p.release_idle_engine(510.0)         # idle again: a new clock starts
        p.release_idle_engine(510.0 + p.ENGINE_IDLE_UNLOAD_S - 1)
        self.assertEqual(self.unloads, [])

    def test_a_pending_or_resolving_open_keeps_the_models(self):
        p = self.player
        p.pending_open = object()
        p.release_idle_engine(0.0)
        p.release_idle_engine(10_000.0)
        p.pending_open, p.opener.busy = None, True
        p.release_idle_engine(20_000.0)
        p.release_idle_engine(30_000.0)
        self.assertEqual(self.unloads, [])

    def ended_live_source(self):
        p = self.player
        self.left = []
        p.session, p.loaded, p.tail = object(), True, None
        p.cast = SimpleNamespace(active=None)
        p.ipc.props["eof-reached"] = True
        p.leave_current = lambda: self.left.append(1) or 0.0
        return p

    def test_an_ended_live_video_counts_as_idle_and_is_closed_before_the_unload(self):
        p = self.ended_live_source()
        p.release_idle_engine(0.0)
        p.release_idle_engine(p.ENGINE_IDLE_UNLOAD_S - 1)
        self.assertEqual((self.left, self.unloads), ([], []))
        p.release_idle_engine(p.ENGINE_IDLE_UNLOAD_S)
        self.assertEqual((self.left, self.unloads), ([1], [1]))
        self.assertEqual(p.ipc.props["user-data/oh/active"], "0")

    def test_a_live_video_that_has_not_ended_is_not_idle(self):
        p = self.ended_live_source()
        p.ipc.props["eof-reached"] = False
        p.release_idle_engine(0.0)
        p.release_idle_engine(10_000.0)
        self.assertEqual((self.left, self.unloads), ([], []))

    def test_seeking_back_from_the_end_restarts_the_idle_clock(self):
        p = self.ended_live_source()
        p.release_idle_engine(0.0)
        p.ipc.props["eof-reached"] = False        # the user seeked back: playing again
        p.release_idle_engine(100.0)
        p.ipc.props["eof-reached"] = True
        p.release_idle_engine(150.0)
        p.release_idle_engine(150.0 + p.ENGINE_IDLE_UNLOAD_S - 1)
        self.assertEqual((self.left, self.unloads), ([], []))

    def test_an_ended_video_while_casting_or_with_a_tail_is_not_idle(self):
        p = self.ended_live_source()
        p.cast = SimpleNamespace(active=object())
        p.release_idle_engine(0.0)
        p.release_idle_engine(10_000.0)
        p.cast, p.tail = SimpleNamespace(active=None), object()
        p.release_idle_engine(20_000.0)
        p.release_idle_engine(30_000.0)
        self.assertEqual((self.left, self.unloads), ([], []))

    def test_nothing_loaded_nothing_to_release(self):
        p = self.player
        p.engine.lada = None
        p.release_idle_engine(0.0)
        p.release_idle_engine(10_000.0)
        self.assertEqual(self.unloads, [])


class ModelBoundSettingsWithoutModelsTests(OverrideTestBase):
    def setUp(self):
        super().setUp()
        self.sc.detach()      # the player's models are not loaded

    def test_model_change_is_remembered_not_loaded(self):
        took, rebuild = self.player.apply_settings({"detection_model": "v4-accurate", "max_clip_length": 120.0,
                                                    "restoration_fps": "30"})
        self.assertEqual(self.factory.loads, [])
        self.assertEqual(self.sc.cfg["detection_model"], "v4-accurate")
        self.assertEqual(self.sc.cfg["max_clip_length"], 120.0)
        self.assertEqual(took["detection_model"], "v4-accurate")
        self.assertEqual(self.saved_file()["detection_model"], "v4-accurate")


if __name__ == "__main__":
    unittest.main()
