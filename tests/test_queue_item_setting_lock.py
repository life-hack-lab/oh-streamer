"""Queue items (processed file / progressive) lock quality, restart-kind settings and interpolation=auto.

Only the live stream (a Session) may change them; VSR / smooth playback / seek prebuffer stay live.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_player_controller import controller_args, controller_deps, FakeEngine, FakeIpc
from oh_streamer.player import player_settings
from oh_streamer.player.controller import PlayerController

QUALITY = "3:720"
SETTING = '4:{"max_clip_length": 120}'
INTERP = '5:{"interpolation": "auto"}'


class QueueItemSettingLockTest(unittest.TestCase):
    def controller(self, mode):
        c = PlayerController(controller_args(), FakeEngine(), deps=controller_deps())
        c.ipc = FakeIpc([])
        c.ipc.command = Mock()
        c.osd = Mock()
        c.apply_settings = Mock()
        c.load_processed = Mock()
        c.progressive_open_source = Mock()
        c.last_sreq, c.last_qreq, c.last_creq = None, None, None
        if mode == "processed":
            c.processed = SimpleNamespace(path="out.mkv", queue_id="q1")
        else:
            c.progressive = Mock()
        return c

    def run_request(self, mode, qreq=None, creq=None):
        c = self.controller(mode)
        (c.processed_requests if mode == "processed" else c.progressive_requests)(None, qreq, creq)
        return c

    def assert_locked(self, c, expected_osd, done):
        c.osd.assert_called_once_with(expected_osd)
        self.assertEqual(c.ipc.get("user-data/oh/request-done"), done)
        c.apply_settings.assert_not_called()
        c.ipc.command.assert_not_called()          # no loadfile / restart
        c.load_processed.assert_not_called()
        c.progressive_open_source.assert_not_called()

    def test_processed_blocks_quality_switch(self):
        self.assert_locked(self.run_request("processed", qreq=QUALITY), "已處理的影片", QUALITY)

    def test_processed_blocks_restart_kind_setting(self):
        self.assert_locked(self.run_request("processed", creq=SETTING), "已處理的影片", SETTING)

    def test_processed_blocks_interpolation_auto(self):
        self.assert_locked(self.run_request("processed", creq=INTERP), "已處理的影片", INTERP)

    def test_progressive_blocks_quality_switch(self):
        self.assert_locked(self.run_request("progressive", qreq=QUALITY), "邊處理邊看：無法切換", QUALITY)

    def test_progressive_blocks_restart_kind_setting(self):
        self.assert_locked(self.run_request("progressive", creq=SETTING), "邊處理邊看：無法切換", SETTING)

    def test_progressive_blocks_interpolation_auto(self):
        self.assert_locked(self.run_request("progressive", creq=INTERP), "邊處理邊看：無法切換", INTERP)

    def test_interpolation_and_restart_keys_are_not_live_keys(self):
        for key in ("interpolation", "max_clip_length", "restoration_fps", "compare"):
            self.assertNotIn(key, player_settings.LIVE_KEYS)

    def test_apply_live_drops_non_live_keys_in_queue_modes(self):
        for mode in ("processed", "progressive"):
            with self.subTest(mode=mode):
                c = PlayerController(
                    controller_args(), FakeEngine(),
                    deps=controller_deps(PS=SimpleNamespace(LIVE_KEYS=player_settings.LIVE_KEYS,
                                                            parse_request=lambda req, _menu: req)))
                c.sc = SimpleNamespace(menu=lambda: None,
                                       diff=lambda req: {"interpolation": "auto", "vsr": "off"})
                c.apply_settings, c.publish_settings = Mock(), Mock()
                c.apply_live("5:x")
                c.apply_settings.assert_called_once_with({"vsr": "off"})


if __name__ == "__main__":
    unittest.main()
