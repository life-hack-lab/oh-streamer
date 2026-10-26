"""AMD player settings and cached hardware selection integration."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import oh_streamer.player.player_settings as PS
import oh_streamer.player.oh_player as L

class AMDPlayerTest(unittest.TestCase):
    def test_setting_and_menu(self):
        self.assertIn("h264_amf", PS.ENCODERS)
        cfg = dict(PS.DEFAULTS, encoder="h264_amf")
        menu = PS.build_menu(cfg, False, [], [], False, "h264_amf", amf_ok=True, platform="win32")
        options = next(row["options"] for row in menu if row["key"] == "encoder")
        self.assertIn("h264_amf", [row["v"] for row in options])

    def test_amf_settings_migration(self):
        saved = dict(PS.DEFAULTS, encoder="h264_amf")
        for platform in ("darwin", "linux"):
            cfg, notes = PS.fix_platform(saved, platform=platform)
            self.assertEqual(cfg["encoder"], "libx264")
            self.assertTrue(notes)
            self.assertEqual(saved["encoder"], "h264_amf")

    def test_cached_probe_selects_amf(self):
        factory = types.SimpleNamespace(fp16=False, device=types.SimpleNamespace(type="cpu"))
        with patch.object(L.encoding_backend.sys, "platform", "win32"):
            settings = L.SettingsControl("unused", {}, dict(PS.DEFAULTS), factory, [], [], False, amf_ok=True)
            self.assertEqual(settings.auto_encoder, "h264_amf")
            self.assertEqual(settings.resolve_encoder("h264_amf"), "h264_amf")
            self.assertEqual(settings.resolve_encoder("libx264"), "libx264")

if __name__ == "__main__":
    unittest.main()
