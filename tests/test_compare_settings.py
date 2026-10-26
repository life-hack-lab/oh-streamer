"""Offline unit tests for the compare setting in player_settings.py (run: python streamer/tests/test_compare_settings.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import json
import os
import sys
import tempfile
import unittest
import oh_streamer.player.player_settings as PS  # noqa: E402


def cli(**kw):
    ns = argparse.Namespace(**{k: None for k in PS.KEYS})
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


DET = ["v2", "v4-fast", "v4-accurate"]
RES = ["basicvsrpp-v1.2"]


class CompareSettingTest(unittest.TestCase):
    """compare: off / on (extra original-frame track), restarts the segment (the stream layout changes)."""

    def test_default_off_and_validate(self):
        self.assertEqual(PS.DEFAULTS["compare"], "off")
        self.assertEqual(PS.validate("compare", "on"), "on")
        self.assertEqual(PS.validate("compare", "off"), "off")
        for bad in ("auto", True, False, None, "ON", "", 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                PS.validate("compare", bad)

    def test_kind_is_restart(self):
        self.assertIn("compare", PS.RESTART_KEYS)
        self.assertNotIn("compare", PS.LIVE_KEYS)
        self.assertNotIn("compare", PS.RELOAD_KEYS)

    def test_persist_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            PS.save_file(path, PS.updated_file({}, {"compare": "on"}))
            saved, warn = PS.load_file(path)
            self.assertIsNone(warn)
            self.assertEqual(saved, {"compare": "on"})
            self.assertEqual(PS.effective(saved, {})["compare"], "on")
            self.assertEqual(PS.effective(saved, PS.cli_overrides(cli(compare="off")))["compare"], "off")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"compare": "auto"}, f)
            saved, warn = PS.load_file(path)
            self.assertEqual(saved, {})
            self.assertIn("compare", warn)
        with self.assertRaises(ValueError):
            PS.cli_overrides(cli(compare="yes"))

    def test_menu_item_and_request(self):
        m = PS.build_menu(dict(PS.DEFAULTS), True, DET, RES, True, "h264_nvenc")
        it = next(i for i in m if i["key"] == "compare")
        self.assertEqual(it["label"], "對照原始畫面")
        self.assertEqual(it["kind"], "restart")
        self.assertIn("o", it["hint"])
        self.assertIn("w", it["hint"])
        self.assertEqual([(o["v"], o["label"]) for o in it["options"]], [("off", "關"), ("on", "開")])
        self.assertEqual((it["value"], it["value_label"]), ("off", "關"))
        wanted = PS.parse_request("5:" + json.dumps({"compare": "on"}), m)
        self.assertEqual(wanted, {"compare": "on"})
        self.assertEqual(PS.diff(dict(PS.DEFAULTS), True, wanted), {"compare": "on"})
        self.assertEqual(PS.parse_request("6:" + json.dumps({"compare": "auto"}), m), {})
        json.dumps(m, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main(verbosity=1)
