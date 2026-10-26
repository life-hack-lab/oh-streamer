"""Per-source "open" extras: a quality height and setting overrides that never reach the settings file."""
from pathlib import Path
import json
import queue
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
for _root in (_STREAMER_ROOT, _SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from oh_streamer.engines.base import ModelChoice  # noqa: E402
from oh_streamer.player import controller as C  # noqa: E402
from oh_streamer.player import player_settings as PS  # noqa: E402
from oh_streamer.player import oh_player as L  # noqa: E402  (SettingsControl; imports torch + lada, no GPU work)
from oh_streamer.sources import qualities  # noqa: E402
from tests.test_player_controller import FakeEngine, FakeHistory, FakeIpc, controller_args, controller_deps  # noqa: E402

DETECTORS = ["v4-fast", "v4-accurate", "v4-broken"]
RESTORERS = ["basicvsrpp-v1.2", "deepmosaics"]


class FakeModels:
    """The shared LadaEngine: records loads; "v4-broken" cannot be loaded."""

    device = None
    max_clip_length = 240

    def __init__(self):
        self.models, self.restoration_fps = ModelChoice("v4-fast", "basicvsrpp-v1.2", False), "original"
        self.loads = []

    def release(self):
        self.models = None

    def load(self, device, models, max_clip_length):
        self.loads.append((models.detection, models.restoration, models.fp16))
        if models.detection == "v4-broken":
            raise RuntimeError("cannot load")
        self.models = models


class OverrideTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_STREAMER_ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.settings_file = Path(self.tmp.name) / "oh_settings.json"
        self.logs = []
        self.factory = FakeModels()
        patcher = patch.object(L.device_backend, "rife_available", lambda _device: True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.sc = L.SettingsControl(str(self.settings_file), {}, dict(PS.DEFAULTS), self.factory, DETECTORS, RESTORERS,
                                    False)
        self.player = C.PlayerController(controller_args(), FakeEngine(),
                                         deps=controller_deps(PS=PS, qualities=qualities, log=self.logs.append))
        p = self.player
        p.engine.interp, p.sc, p.cast, p.session, p.qc = None, self.sc, None, None, None
        p.ipc, p.hist, p.proc_samples, p.proc_fps_now = FakeIpc([]), FakeHistory(), [], None
        p.info_visible, p.loading_models = False, False
        p.published_settings, p.published_opts = None, {}
        p.inbox = queue.Queue()
        p.reload_box, p.reload_pending, p.reload_requested, p.reload_restart_at = queue.Queue(), {}, False, -10.0

    def saved_file(self):
        return json.loads(self.settings_file.read_text(encoding="utf-8")) if self.settings_file.exists() else None

    def row(self, key):
        return next(it for it in self.sc.menu() if it["key"] == key)


class OpenMessageTests(OverrideTestBase):
    def message(self, **extra):
        self.assertEqual(self.player.on_message({"cmd": "open", "input": "https://example.test/v", "headers": {"A": 1},
                                                 **extra}), {"ok": True})
        return self.player.inbox.get_nowait()

    def test_plain_open_has_no_quality_or_overrides(self):
        self.assertEqual(self.message(), ("https://example.test/v", {"A": "1"}, "轉送", 0.0, None, {}, None))

    def test_quality_and_valid_overrides_are_queued_for_the_controller_loop(self):
        item = self.message(quality=720, overrides={"detection_model": "v4-accurate", "interpolation": "auto",
                                                    "restoration_fps": "30", "restoration_model": "deepmosaics"})
        self.assertEqual(item[4:6], (720, {"detection_model": "v4-accurate", "interpolation": "auto",
                                          "restoration_fps": "30", "restoration_model": "deepmosaics"}))

    def test_quality_forms(self):
        self.assertEqual(self.message(quality="480p")[4], 480)
        self.assertEqual(self.message(quality=1080.0)[4], 1080)
        for bad in (True, "high", [720], {"h": 1}):
            self.assertIsNone(self.message(quality=bad)[4], bad)

    def test_invalid_overrides_are_ignored_with_a_log_line(self):
        item = self.message(overrides={"interpolation": "full", "restoration_fps": "30", "max_clip_length": 60,
                                       "detection_model": "  ", "fp16": True})
        self.assertEqual(item[5], {"restoration_fps": "30"})
        self.assertEqual(len([m for m in self.logs if m.startswith("open: ")]), 4)
        self.assertEqual(self.message(overrides="auto")[5], {})

    def test_listener_thread_only_queues(self):
        self.player.ipc = None            # on_message must not need the mpv IPC (or the settings object)
        self.player.sc = None
        self.message(quality=720, overrides={"interpolation": "auto"})


class SubmitTests(OverrideTestBase):
    def test_extras_reach_the_opener_and_plain_opens_keep_the_old_call(self):
        calls = []
        self.player.opener = SimpleNamespace(submit=lambda *a, **k: calls.append((a, k)))
        self.player.submit_text("https://example.test/v", {}, "轉送", 0.0)
        self.player.submit_text("https://example.test/v", {}, "轉送", 0.0, 720, {"interpolation": "auto"})
        self.assertEqual(len(calls[0][0]), 3)
        self.assertEqual(calls[0][1], {})
        self.assertEqual(calls[1][1], {"quality": 720, "overrides": {"interpolation": "auto"}})


class ApplyOverridesTests(OverrideTestBase):
    def test_override_applies_but_is_not_persisted(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto", "restoration_fps": "30"})
        self.assertEqual(self.factory.loads, [("v4-accurate", "basicvsrpp-v1.2", False)])
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["interpolation"], self.sc.cfg["restoration_fps"]),
                         ("v4-accurate", "auto", "30"))
        self.assertEqual(self.factory.restoration_fps, "30")
        self.assertIsNone(self.saved_file())                      # nothing was committed
        self.assertEqual(self.sc.saved, {})
        self.assertEqual(self.sc.override_base,
                         {"detection_model": "v4-fast", "interpolation": "off", "restoration_fps": "original"})

    def test_menu_shows_the_effective_values(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto"})
        self.assertEqual((self.row("detection_model")["value"], self.row("interpolation")["value"]),
                         ("v4-accurate", "auto"))
        self.assertIn("此影片", self.row("detection_model")["hint"])
        self.assertNotIn("此影片", self.row("restoration_model")["hint"])

    def test_next_open_without_overrides_reverts_to_the_saved_values(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto", "restoration_fps": "30"})
        self.factory.loads.clear()
        self.player.apply_overrides({})
        self.assertEqual(self.factory.loads, [("v4-fast", "basicvsrpp-v1.2", False)])
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["interpolation"], self.sc.cfg["restoration_fps"]),
                         ("v4-fast", "off", "original"))
        self.assertEqual(self.factory.restoration_fps, "original")
        self.assertEqual(self.sc.override_base, {})
        self.assertNotIn("此影片", self.row("detection_model")["hint"])
        self.assertIsNone(self.saved_file())

    def test_plain_open_with_nothing_overridden_does_nothing(self):
        self.player.apply_overrides({})
        self.assertEqual(self.factory.loads, [])
        self.assertEqual(self.logs, [])

    def test_different_overrides_replace_the_previous_ones(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto"})
        self.factory.loads.clear()
        self.player.apply_overrides({"interpolation": "auto", "restoration_fps": "30"})
        self.assertEqual(self.factory.loads, [("v4-fast", "basicvsrpp-v1.2", False)])   # detector back, models reloaded
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["interpolation"], self.sc.cfg["restoration_fps"]),
                         ("v4-fast", "auto", "30"))
        self.assertEqual(self.sc.override_base, {"interpolation": "off", "restoration_fps": "original"})

    def test_same_overrides_again_do_not_reload(self):
        self.player.apply_overrides({"detection_model": "v4-accurate"})
        self.factory.loads.clear()
        self.player.apply_overrides({"detection_model": "v4-accurate"})
        self.assertEqual(self.factory.loads, [])
        self.assertEqual(self.sc.override_base, {"detection_model": "v4-fast"})

    def test_override_equal_to_the_current_value_is_not_an_override(self):
        self.player.apply_overrides({"detection_model": "v4-fast", "interpolation": "off"})
        self.assertEqual((self.factory.loads, self.sc.override_base), ([], {}))

    def test_values_the_player_cannot_use_are_ignored(self):
        self.player.apply_overrides({"detection_model": "not-installed", "restoration_fps": "30"})
        self.assertEqual(self.factory.loads, [])
        self.assertEqual(self.sc.cfg["restoration_fps"], "30")
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")
        self.assertTrue(any(m.startswith("override ignored") for m in self.logs))

    def test_interpolation_is_ignored_where_the_menu_does_not_offer_it(self):
        with patch.object(L.device_backend, "rife_available", lambda _device: False):
            self.sc.invalidate_menu()
            self.player.apply_overrides({"interpolation": "auto"})
        self.assertEqual(self.sc.cfg["interpolation"], "off")
        self.assertEqual(self.sc.override_base, {})

    def test_a_failed_model_load_leaves_no_override_behind(self):
        self.player.apply_overrides({"detection_model": "v4-broken", "restoration_fps": "30"})
        self.assertEqual(self.factory.loads[-1], ("v4-fast", "basicvsrpp-v1.2", False))   # previous models restored
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")
        self.assertEqual(self.sc.cfg["restoration_fps"], "30")
        self.assertEqual(self.sc.override_base, {"restoration_fps": "original"})


class MenuChangeDuringOverrideTests(OverrideTestBase):
    def test_menu_change_of_an_overridden_key_is_committed_and_stops_being_an_override(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto"})
        self.player.apply_settings({"detection_model": "v4-fast"})           # what handle_requests does for the menu
        self.assertEqual(self.saved_file()["detection_model"], "v4-fast")
        self.assertEqual(self.sc.override_base, {"interpolation": "off"})   # the other override stays
        self.assertEqual(self.sc.saved["detection_model"], "v4-fast")
        self.assertNotIn("interpolation", self.saved_file())                # ... and still is not persisted
        self.assertNotIn("此影片", self.row("detection_model")["hint"])
        self.assertIn("此影片", self.row("interpolation")["hint"])
        self.player.apply_overrides({})                                       # next plain open
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")           # not reverted to the pre-override value
        self.assertEqual(self.sc.cfg["interpolation"], "off")

    def test_menu_change_to_another_value_keeps_that_value_for_later_sources(self):
        self.player.apply_overrides({"detection_model": "v4-accurate"})
        self.player.apply_settings({"detection_model": "v4-fast", "restoration_fps": "30"})
        self.assertEqual(self.sc.override_base, {})
        self.player.apply_overrides({})
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["restoration_fps"]), ("v4-fast", "30"))
        self.assertEqual(self.saved_file()["restoration_fps"], "30")

    def test_changing_a_setting_that_is_not_overridden_does_not_persist_the_overrides(self):
        self.player.apply_overrides({"interpolation": "auto"})
        self.player.apply_settings({"encoder": "libx264"})
        self.assertEqual(self.saved_file(), {"encoder": "libx264"})
        self.assertEqual(self.sc.override_base, {"interpolation": "off"})


class SwitchToTests(OverrideTestBase):
    def test_overrides_are_applied_after_the_old_source_left_and_before_the_new_one_begins(self):
        order = []
        p = self.player
        p.leave_current = lambda: order.append("leave") or 0.0
        p.apply_overrides = lambda raw: order.append(("overrides", raw))
        p.begin_source = lambda opened: order.append("begin")
        p.publish_source = lambda: order.append("publish")
        p.session = SimpleNamespace(source=SimpleNamespace(describe=lambda: "src"))
        p.epoch, p.proc_samples = 0, __import__("collections").deque()
        p.engine.lada = None
        p.cast = None
        opened = SimpleNamespace(title="t", overrides={"interpolation": "auto"})
        p.switch_to(opened)
        self.assertEqual(order, ["leave", ("overrides", {"interpolation": "auto"}), "begin", "publish"])
        order.clear()
        p.switch_to(SimpleNamespace(title="plain"))     # an Opened without overrides (or a CLI first source)
        self.assertEqual(order[1], ("overrides", {}))


class RecordingIpc(FakeIpc):
    def __init__(self):
        super().__init__([])
        self.commands = []

    def command(self, *args):
        self.commands.append(args)


class SettingsReloadTests(OverrideTestBase):
    """{"cmd": "settings_reload"}: the queue app rewrote the settings file; the player adopts it, never writes it back."""

    def setUp(self):
        super().setUp()
        p = self.player
        p.reload_box, p.reload_pending, p.reload_requested, p.reload_restart_at = queue.Queue(), {}, False, -10.0
        p.ipc = RecordingIpc()
        p.last_sreq = p.last_qreq = p.last_creq = None
        self.write(detection_model="v4-fast", seek_prebuffer=2.0)
        self.sc.saved = {"detection_model": "v4-fast", "seek_prebuffer": 2.0}
        self.sc.cfg.update(seek_prebuffer=2.0)

    def write(self, **values):
        self.settings_file.write_text(json.dumps(values), encoding="utf-8")

    def reload(self, now=0.0):
        self.player.settings_reload()
        self.player.apply_reloaded_settings(now)

    def test_message_is_queued_for_the_loop_and_answered_at_once(self):
        self.player.ipc = None            # the listener thread must not need the mpv IPC (or the settings object)
        self.player.sc = None
        self.assertEqual(self.player.on_message({"cmd": "settings_reload"}), {"ok": True})
        self.assertFalse(self.player.reload_box.empty())
        self.assertTrue(self.player.inbox.empty())

    def test_changed_keys_are_applied_like_a_menu_change_without_writing_the_file(self):
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0, interpolation="auto", restoration_fps="30")
        before = self.settings_file.read_text(encoding="utf-8")
        self.reload()
        self.assertEqual(self.factory.loads, [("v4-accurate", "basicvsrpp-v1.2", False)])
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["interpolation"], self.sc.cfg["restoration_fps"]),
                         ("v4-accurate", "auto", "30"))
        self.assertEqual(self.factory.restoration_fps, "30")
        self.assertEqual(self.sc.saved, {"detection_model": "v4-accurate", "seek_prebuffer": 2.0,
                                         "interpolation": "auto", "restoration_fps": "30"})
        self.assertEqual(self.settings_file.read_text(encoding="utf-8"), before)       # not rewritten
        self.assertEqual(self.sc.override_base, {})
        self.assertEqual(self.player.reload_pending, {})
        self.assertEqual(self.row("detection_model")["value"], "v4-accurate")

    def test_unchanged_file_does_nothing(self):
        self.reload()
        self.assertEqual((self.factory.loads, self.player.reload_pending), ([], {}))
        self.assertTrue(any("no changes" in m for m in self.logs))

    def test_the_next_menu_change_keeps_the_reloaded_values(self):
        """The bug: the player rewrote the file from its own stale copy on its next menu change."""
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0, restoration_fps="30")
        self.reload()
        self.player.apply_settings({"encoder": "libx264"})                              # what the menu does
        self.assertEqual(self.saved_file(), {"detection_model": "v4-accurate", "seek_prebuffer": 2.0,
                                             "restoration_fps": "30", "encoder": "libx264"})

    def test_a_value_removed_from_the_file_goes_back_to_the_default(self):
        self.write(seek_prebuffer=2.0)
        self.sc.saved["detection_model"] = "v4-accurate"
        self.sc.cfg["detection_model"] = "v4-accurate"
        self.factory.models = ModelChoice("v4-accurate", self.factory.models.restoration, self.factory.models.fp16)
        self.reload()
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")
        self.assertEqual(self.saved_file(), {"seek_prebuffer": 2.0})                     # (not rewritten)

    def test_an_overridden_key_keeps_its_value_and_only_what_it_reverts_to_changes(self):
        self.player.apply_overrides({"detection_model": "v4-accurate", "interpolation": "auto"})
        self.factory.loads.clear()
        self.write(detection_model="v4-broken", seek_prebuffer=2.0, interpolation="auto", restoration_fps="30")
        self.reload()
        self.assertEqual(self.factory.loads, [])                                         # nothing reloaded
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["interpolation"]), ("v4-accurate", "auto"))
        # detection_model: the default after this video is the file's; interpolation: the override equals the new
        # default, so it is none any more; restoration_fps was not overridden: applied now
        self.assertEqual(self.sc.override_base, {"detection_model": "v4-broken"})
        self.assertEqual(self.sc.cfg["restoration_fps"], "30")
        self.assertEqual(self.sc.saved["detection_model"], "v4-broken")
        self.assertIn("此影片", self.row("detection_model")["hint"])
        self.assertNotIn("此影片", self.row("interpolation")["hint"])
        self.assertEqual(self.player.reload_pending, {})

    def test_the_override_ends_with_the_next_source_and_the_new_default_comes_back(self):
        self.player.apply_overrides({"restoration_fps": "30"})
        self.write(detection_model="v4-fast", seek_prebuffer=2.0, restoration_fps="original", interpolation="auto")
        before = self.settings_file.read_text(encoding="utf-8")
        self.reload()
        self.assertEqual((self.sc.cfg["restoration_fps"], self.sc.cfg["interpolation"]), ("30", "auto"))
        self.assertEqual(self.sc.override_base, {"restoration_fps": "original"})
        self.player.apply_overrides({})
        self.assertEqual((self.sc.cfg["restoration_fps"], self.sc.cfg["interpolation"]), ("original", "auto"))
        self.assertEqual(self.settings_file.read_text(encoding="utf-8"), before)         # the player never wrote the file

    def test_unusable_files_change_nothing(self):
        self.settings_file.write_text("{not json", encoding="utf-8")
        self.reload()
        self.settings_file.unlink()
        self.reload()
        self.assertEqual((self.factory.loads, self.sc.cfg["detection_model"], self.player.reload_pending), ([], "v4-fast", {}))
        self.assertEqual(self.sc.saved, {"detection_model": "v4-fast", "seek_prebuffer": 2.0})

    def test_a_model_the_player_cannot_use_is_remembered_but_not_applied(self):
        self.write(detection_model="not-installed", seek_prebuffer=2.0, restoration_fps="30")
        self.reload()
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["restoration_fps"]), ("v4-fast", "30"))
        self.assertEqual(self.factory.loads, [])
        self.assertEqual(self.sc.saved["detection_model"], "not-installed")
        self.assertTrue(any("not applied" in m for m in self.logs))

    def test_a_failed_model_load_restores_the_models_in_use(self):
        self.write(detection_model="v4-broken", seek_prebuffer=2.0)
        self.reload()
        self.assertEqual(self.factory.loads[-1], ("v4-fast", "basicvsrpp-v1.2", False))
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")

    def test_a_running_session_is_restarted_through_lada_seek_and_changes_wait_for_that_request(self):
        p = self.player
        p.session, p.cast = SimpleNamespace(target=3.0), SimpleNamespace(active=None)
        p.loaded, p.resume_wait, p.restore_deadline, p.tail = True, False, None, None
        p.ipc.props.update({"time-pos": 12.5, "pause": False})
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0)
        self.reload(now=100.0)
        self.assertEqual(p.ipc.commands, [("script-message", "oh-restart", "12.500", "0")])
        self.assertEqual(p.reload_pending, {"detection_model": "v4-accurate"})           # applied by handle_requests
        self.assertEqual(self.factory.loads, [])
        p.apply_reloaded_settings(101.0)                                                 # not asked again at once
        self.assertEqual(len(p.ipc.commands), 1)
        p.apply_reloaded_settings(106.0)                                                 # ... but again if lost
        self.assertEqual(len(p.ipc.commands), 2)

    def test_the_paused_state_goes_with_the_restart_and_a_starting_segment_waits(self):
        p = self.player
        p.session, p.cast = SimpleNamespace(target=3.0), SimpleNamespace(active=None)
        p.loaded, p.resume_wait, p.restore_deadline, p.tail = False, False, None, None
        p.ipc.props.update({"time-pos": 4.0, "pause": True})
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0)
        self.reload(now=100.0)
        self.assertEqual(p.ipc.commands, [])
        p.loaded = True
        p.apply_reloaded_settings(100.5)
        self.assertEqual(p.ipc.commands, [("script-message", "oh-restart", "4.000", "1")])

    def test_a_request_on_its_way_takes_the_changes_along_without_a_restart(self):
        p = self.player
        p.session, p.cast = SimpleNamespace(target=3.0), SimpleNamespace(active=None)
        p.loaded, p.resume_wait, p.restore_deadline, p.tail = True, False, None, None
        p.ipc.props["user-data/oh/seek-request"] = "7:30.000"
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0)
        self.reload(now=100.0)
        self.assertEqual(p.ipc.commands, [])
        self.assertEqual(p.reload_pending, {"detection_model": "v4-accurate"})

    def test_pending_changes_are_applied_when_the_next_source_opens(self):
        p = self.player
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0)
        p.settings_reload()
        order = []
        running = SimpleNamespace(source=SimpleNamespace(describe=lambda: "src"))
        p.session, p.cast = running, None                # a segment runs when the file changes, then the user opens another
        p.leave_current = lambda: order.append("leave") or setattr(p, "session", None) or 0.0
        p.begin_source = lambda opened: (order.append(("begin", self.sc.cfg["detection_model"])), setattr(p, "session", running))
        p.publish_source = lambda: None
        p.epoch, p.proc_samples = 0, __import__("collections").deque()
        p.engine.lada = None
        p.switch_to(SimpleNamespace(title="t", overrides={}))
        self.assertEqual(order, ["leave", ("begin", "v4-accurate")])
        self.assertEqual(self.factory.loads, [("v4-accurate", "basicvsrpp-v1.2", False)])
        self.assertEqual(p.reload_pending, {})

    def test_a_reload_before_the_models_exist_waits_for_them(self):
        p = self.player
        p.sc = None                                    # the loop's self.sc is None until the models have loaded
        self.assertEqual(p.on_message({"cmd": "settings_reload"}), {"ok": True})
        p.reload_box.get_nowait()
        p.reload_requested = True
        self.assertTrue(p.reload_requested and p.sc is None)   # the loop leaves the flag set
        p.sc = self.sc
        self.write(detection_model="v4-accurate", seek_prebuffer=2.0)
        p.settings_reload()
        p.apply_reloaded_settings(0.0)
        self.assertEqual(self.sc.cfg["detection_model"], "v4-accurate")


class HandleRequestsReloadTests(OverrideTestBase):
    """handle_requests takes the pending file changes along with the request that restarts the segment."""

    def setUp(self):
        super().setUp()
        p = self.player
        self.restarts = []
        p.reload_box, p.reload_pending, p.reload_restart_at = queue.Queue(), {}, -10.0
        p.ipc = RecordingIpc()
        p.last_sreq = p.last_qreq = p.last_creq = None
        self.settings_file.write_text(json.dumps({"detection_model": "v4-fast"}), encoding="utf-8")
        self.sc.saved = {"detection_model": "v4-fast"}
        session = SimpleNamespace(
            target=12.5, source=SimpleNamespace(meta=SimpleNamespace(video_fps_exact=30)), restorer=object(), url=None,
            stop=lambda: 0.0, release_restorer=lambda: None, set_source=lambda *a: None,
            start=lambda pos, t, **k: self.restarts.append(pos), encoder="libx264")
        p.session, p.cast, p.qc = session, SimpleNamespace(active=None), SimpleNamespace(
            lada=SimpleNamespace(open=lambda source: object()), current={"id": "q"})
        p.loaded, p.resume_wait, p.restore_deadline, p.tail = True, False, None, None
        p.stop_for_request = lambda *a: 0.0
        p.served, p.n_req = "", 0

    def request(self, sreq, creq=""):
        p = self.player
        p.ipc.props["user-data/oh/seek-request"] = sreq
        p.ipc.props["user-data/oh/settings-request"] = creq
        p.handle_requests(sreq, "", creq)

    def test_the_restart_applies_the_file_changes_without_writing_them(self):
        p = self.player
        self.settings_file.write_text(json.dumps({"detection_model": "v4-accurate", "restoration_fps": "30"}),
                                      encoding="utf-8")
        before = self.settings_file.read_text(encoding="utf-8")
        p.settings_reload()
        self.request("1:12.500")
        self.assertEqual(self.factory.loads, [("v4-accurate", "basicvsrpp-v1.2", False)])
        self.assertEqual((self.sc.cfg["detection_model"], self.sc.cfg["restoration_fps"]), ("v4-accurate", "30"))
        self.assertEqual(self.settings_file.read_text(encoding="utf-8"), before)
        self.assertEqual((p.reload_pending, self.restarts), ({}, [12.5]))

    def test_a_menu_choice_in_the_same_round_wins_and_is_written(self):
        p = self.player
        self.settings_file.write_text(json.dumps({"detection_model": "v4-accurate"}), encoding="utf-8")
        p.settings_reload()
        self.request("1:12.500", "2:" + json.dumps({"detection_model": "v4-fast", "encoder": "libx264"}))
        self.assertEqual(self.sc.cfg["detection_model"], "v4-fast")
        self.assertEqual(self.saved_file()["encoder"], "libx264")

    def test_while_casting_the_changes_are_applied_through_handle_requests_directly(self):
        p = self.player
        calls = []
        p.cast = SimpleNamespace(active={"mode": "hls"}, suspend=lambda: calls.append("suspend"),
                                 resume=lambda enc, changed=False: calls.append(("resume", changed)))
        self.settings_file.write_text(json.dumps({"detection_model": "v4-accurate"}), encoding="utf-8")
        p.settings_reload()
        p.apply_reloaded_settings(100.0)
        self.assertFalse([c for c in p.ipc.commands if c[0] == "script-message"])         # no oh_seek.lua restart
        self.assertEqual(self.sc.cfg["detection_model"], "v4-accurate")
        self.assertEqual(calls, ["suspend", ("resume", True)])
        self.assertEqual(p.reload_pending, {})


class SettingsLabelTests(unittest.TestCase):
    def test_max_clip_length_is_called_the_restoration_window(self):
        menu = PS.build_menu(dict(PS.DEFAULTS), False, DETECTORS, RESTORERS, False, "libx264", platform="win32")
        self.assertEqual(next(it for it in menu if it["key"] == "max_clip_length")["label"], "修復時間窗")

    def test_clean_overrides(self):
        self.assertEqual(PS.clean_overrides(None), ({}, []))
        out, notes = PS.clean_overrides({"interpolation": "auto", "detection_model": " v4-fast ", "volume": 5, "x": 1,
                                         "restoration_fps": "60"})
        self.assertEqual(out, {"interpolation": "auto", "detection_model": "v4-fast"})
        self.assertEqual(len(notes), 3)


if __name__ == "__main__":
    unittest.main()
