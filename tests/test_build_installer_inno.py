"""Tests for the Inno Setup side of packaging/build_installer.py (no ISCC run needed)."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))

import build_installer as bi  # noqa: E402

ISS = ROOT / "packaging" / "installer" / "oh_streamer.iss"


class BrandingTest(unittest.TestCase):
    def test_real_branding_has_name(self):
        self.assertTrue(bi.read_brand_name())

    def test_parse_and_errors(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "branding.js"
            p.write_text('// c\nwindow.BRAND = {"name": " My App ", "logo": "M"};\n', encoding="utf-8")
            self.assertEqual(bi.read_brand_name(p), "My App")
            p.write_text("window.BRAND = {name: 'x'};", encoding="utf-8")
            with self.assertRaises(bi.BuildError):
                bi.read_brand_name(p)
            p.write_text("nothing", encoding="utf-8")
            with self.assertRaises(bi.BuildError):
                bi.read_brand_name(p)

    def test_basename(self):
        self.assertEqual(bi.installer_basename("Oh Streamer", "0.1.0"), "OhStreamer-Setup-0.1.0")


class IsccTest(unittest.TestCase):
    def test_find_iscc_env_and_missing(self):
        with tempfile.TemporaryDirectory() as t:
            exe = Path(t) / "ISCC.exe"
            exe.write_bytes(b"x")
            self.assertEqual(bi.find_iscc({bi.ISCC_ENV: str(exe)}, []), exe)
            self.assertEqual(bi.find_iscc({}, [Path(t) / "no.exe", exe]), exe)
            with self.assertRaises(bi.BuildError):
                bi.find_iscc({bi.ISCC_ENV: str(Path(t) / "gone.exe")}, [])
            with self.assertRaises(bi.BuildError):
                bi.find_iscc({}, [Path(t) / "no.exe"])

    def test_command(self):
        cmd = bi.iscc_command(Path("ISCC.exe"), Path("a.iss"), stage=Path("st age"), out=Path("out"),
                              app_name="Oh Streamer", version="1.2.3")
        self.assertIn("/DAppName=Oh Streamer", cmd)
        self.assertIn("/DAppVersion=1.2.3", cmd)
        self.assertIn("/DOutputBaseName=OhStreamer-Setup-1.2.3", cmd)
        self.assertTrue(any(c.startswith("/DStageDir=") and c.endswith("st age") for c in cmd))
        self.assertEqual(cmd[-1], "a.iss")


class IssTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = ISS.read_bytes()
        cls.has_bom = raw.startswith(b"\xef\xbb\xbf")
        cls.text = raw.decode("utf-8-sig")

    def section(self, name):
        m = re.search(rf"^\[{name}\]\s*$(.*?)(?=^\[|\Z)", self.text, re.M | re.S)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def test_bom(self):
        self.assertTrue(self.has_bom)

    def test_setup_values(self):
        s = self.section("Setup")
        self.assertIn("AppId={{759A8B7A-6BC3-4A79-BA93-D39701E922CA}", s)
        self.assertIn("PrivilegesRequired=lowest", s)
        self.assertNotIn("PrivilegesRequiredOverridesAllowed", s)
        self.assertIn("DisableDirPage=no", s)
        self.assertIn(r"DefaultDirName={userpf}\oh_streamer", s)
        self.assertIn("SolidCompression=yes", s)
        self.assertIn("LZMAUseSeparateProcess=yes", s)
        self.assertIn("Compression=lzma2/ultra64", s)
        m = re.search(r"LZMADictionarySize=(\d+)", s)
        self.assertGreater(int(m.group(1)), 228 * 1024)
        self.assertTrue(re.search(r"ExtraDiskSpaceRequired=\d{10}", s))

    def test_publisher_urls_and_no_license_page(self):
        s = self.section("Setup")
        self.assertIn("AppPublisher=life-hack-lab", s)
        self.assertRegex(s, r"(?m)^AppPublisherURL=https://github.com/life-hack-lab/oh-streamer\s*$")
        self.assertIn("AppSupportURL=https://github.com/life-hack-lab/oh-streamer/issues", s)
        self.assertIn("AppUpdatesURL=https://github.com/life-hack-lab/oh-streamer/releases", s)
        self.assertNotIn("LicenseFile", re.sub(r";.*", "", self.text))

    def test_no_hardcoded_product_name(self):
        self.assertNotIn("Oh Streamer", re.sub(r";.*", "", self.text))

    def test_install_delete_protects_user_state(self):
        body = "\n".join(l for l in self.section("InstallDelete").splitlines() if not l.strip().startswith(";"))
        self.assertIn(r"{app}\app", body)
        self.assertIn(r"{app}\python", body)
        for protected in ("envs", "uv-cache", "logs", "current.json"):
            self.assertNotIn(protected, body)
        for line in body.splitlines():
            if line.strip():
                self.assertRegex(line, r'Name: "\{app\}\\[a-z]+"')

    def test_files_and_icons(self):
        self.assertIn(r'DestDir: "{app}"', self.section("Files"))
        self.assertIn("recursesubdirs", self.section("Files"))
        icons = self.section("Icons")
        self.assertEqual(icons.count('AppUserModelID: "oh_streamer.Desktop"'), 2)
        self.assertIn(r'Filename: "{app}\python\pythonw.exe"', icons)
        self.assertIn("launch.pyw", icons)
        run = self.section("Run")
        self.assertIn("postinstall", run)
        self.assertIn("nowait", run)

    def test_webview2_bootstrapper_is_embedded_not_installed_and_detected(self):
        files = self.section("Files")
        self.assertIn(r'Excludes: "redist\*"', files)
        line = [l for l in files.splitlines() if "MicrosoftEdgeWebview2Setup.exe" in l and not l.startswith(";")][0]
        self.assertIn("dontcopy", line)
        self.assertIn(r"\redist\MicrosoftEdgeWebview2Setup.exe", line)
        code = self.section("Code")
        for needle in ("{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients",
                       "HKCU", "'0.0.0.0'", "FORCEWEBVIEW2", "ExtractTemporaryFile", "/silent /install",
                       "ssPostInstall", "正在安裝 Microsoft Edge WebView2 執行階段"):
            self.assertIn(needle, code)
        self.assertIn("/FORCEWEBVIEW2", self.text.split("[Setup]")[0])

    def test_code_checks(self):
        code = self.section("Code")
        for needle in ("{commonpf}", "{commonpf32}", "{win}", "MaxInstallRootLength = 110", "wpSelectDir"):
            self.assertIn(needle, code)

    def test_app_user_model_id_matches_source(self):
        src = (ROOT / "src" / "oh_streamer" / "platform" / "windows_identity.py").read_text(encoding="utf-8")
        self.assertIn('APP_USER_MODEL_ID = "oh_streamer.Desktop"', src)

    def test_app_mutex_matches_python_constant(self):
        sys.path.insert(0, str(ROOT / "src"))
        from oh_streamer.runtime_env.app_running import APP_MUTEX_NAME
        self.assertIn(f'#define AppMutexName "{APP_MUTEX_NAME}"', self.text)
        s = self.section("Setup")
        self.assertIn("AppMutex={#AppMutexName}", s)
        self.assertIn("CloseApplications=no", s)

    def test_uninstall_delete_stays_under_app(self):
        body = [l.strip() for l in self.section("UninstallDelete").splitlines() if l.strip() and not l.strip().startswith(";")]
        self.assertTrue(body)
        for line in body:
            self.assertRegex(line, r'Name: "\{app\}(\\[a-z]+)?"$')
            self.assertNotIn("..", line)
        text = " ".join(body)
        for kept in ("envs", "uv-cache", "current.json"):    # only deleted by the user's choice, in [Code]
            self.assertNotIn(kept, text)

    def test_purge_code_is_guarded(self):
        code = self.section("Code")
        for needle in ("function SafeBase", "InitializeUninstall", "CurUninstallStepChanged", "{param:PURGE|}",
                       "{param:USERDATADIR|}", "usPostUninstall", "AttrReparsePoint"):
            self.assertIn(needle, code)
        # every folder deletion goes through PurgeChild/PurgeTree; no direct DelTree of a variable base
        self.assertEqual(code.count("DelTree("), 1)
        self.assertIn("DelTree(P, False, True, False)", code)         # single FILE with deleteFiles only
        # defaults: nothing is purged unless asked
        self.assertIn("HasToken(List, 'envs')", code)
        self.assertNotRegex(code, r"Purge(Envs|Models|UserData) := True")


if __name__ == "__main__":
    unittest.main()
