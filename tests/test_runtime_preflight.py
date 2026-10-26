r"""preflight.webview2_installed and diagnostics.export_diagnostics: WebView2 check and support zip contents."""
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import preflight
from oh_streamer.runtime_env.diagnostics import LOG_LIMIT, export_diagnostics
from oh_streamer.runtime_env.preflight import webview2_installed

HKLM = "HKLM"
HKCU = "HKCU"


def fake_winreg(values):
    """values: {(hive, subkey): pv}; a missing key raises OSError the way winreg does."""
    module = types.ModuleType("winreg")
    module.HKEY_LOCAL_MACHINE = HKLM
    module.HKEY_CURRENT_USER = HKCU

    class Key:
        def __init__(self, pv):
            self.pv = pv

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(hive, subkey):
        if (hive, subkey) not in values:
            raise FileNotFoundError(subkey)
        return Key(values[(hive, subkey)])

    module.OpenKey = open_key
    module.QueryValueEx = lambda key, name: (key.pv, 1)
    return module


class WebView2InstalledTests(unittest.TestCase):
    def check(self, values):
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(sys.modules, {"winreg": fake_winreg(values)}):
            return webview2_installed()

    def test_non_windows_is_true(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIs(webview2_installed(), True)

    def test_missing_keys_is_false(self):
        self.assertIs(self.check({}), False)

    def test_zero_or_empty_pv_is_false(self):
        self.assertIs(self.check({(HKLM, preflight._HKLM_CLIENT): "0.0.0.0"}), False)
        self.assertIs(self.check({(HKCU, preflight._HKCU_CLIENT): ""}), False)

    def test_valid_pv_in_either_hive_is_true(self):
        self.assertIs(self.check({(HKLM, preflight._HKLM_CLIENT): "130.0.2849.80"}), True)
        self.assertIs(self.check({(HKCU, preflight._HKCU_CLIENT): "130.0.2849.80"}), True)

    @unittest.skipUnless(sys.platform == "win32", "real registry check only on Windows")
    def test_real_call_returns_bool(self):
        self.assertIsInstance(webview2_installed(), bool)


class DiagnosticsExportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.install = self.root / "install"
        self.dest = self.root / "out" / "nested"
        self.install.mkdir()
        self.logs = self.install / "logs"
        self.logs.mkdir()
        (self.install / "oh_settings.json").write_text('{"token": "secret"}', encoding="utf-8")
        (self.install / "queue.json").write_text("[]", encoding="utf-8")

    def names(self, zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            return sorted(zf.namelist())

    def test_contains_only_listed_items(self):
        (self.logs / "oh_player 日誌 1.log").write_text("log one", encoding="utf-8")
        (self.logs / "oh_player 2.log").write_text("log two", encoding="utf-8")
        (self.install / "current.json").write_text('{"version": "1"}', encoding="utf-8")
        detection = {"gpus": [{"name": "RTX 5060"}]}

        zip_path = export_diagnostics(self.install, self.dest, detection=detection, now=datetime(2026, 10, 24, 2, 3, 4))

        self.assertEqual(zip_path, self.dest / "oh_streamer-diagnostics-20261024-020304.zip")
        self.assertTrue(zip_path.is_file())
        self.assertEqual(
            self.names(zip_path),
            ["current.json", "detection.json", "logs/oh_player 2.log", "logs/oh_player 日誌 1.log", "preflight.json"],
        )
        with zipfile.ZipFile(zip_path) as zf:
            self.assertEqual(json.loads(zf.read("detection.json")), detection)
            preflight_info = json.loads(zf.read("preflight.json"))
            self.assertIsInstance(preflight_info["webview2_installed"], bool)
            self.assertEqual(preflight_info["sys_platform"], sys.platform)
            self.assertIn("python_version", preflight_info)

    def test_missing_current_json_still_builds_zip(self):
        zip_path = export_diagnostics(self.install, self.dest, now=datetime(2026, 1, 2, 3, 4, 5))
        self.assertEqual(self.names(zip_path), ["detection.json", "preflight.json"])
        with zipfile.ZipFile(zip_path) as zf:
            self.assertEqual(json.loads(zf.read("detection.json")), None)

    def test_logs_limited_to_newest_five(self):
        base = 1_700_000_000
        for i in range(LOG_LIMIT + 2):
            path = self.logs / f"oh_player {i} 紀錄.log"
            path.write_text(str(i), encoding="utf-8")
            os.utime(path, (base + i, base + i))

        zip_path = export_diagnostics(self.install, self.dest, now=datetime(2026, 1, 2, 3, 4, 5))

        expected_logs = [f"logs/oh_player {i} 紀錄.log" for i in range(2, LOG_LIMIT + 2)]
        self.assertEqual(self.names(zip_path), sorted(expected_logs + ["detection.json", "preflight.json"]))


if __name__ == "__main__":
    unittest.main()
