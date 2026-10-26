"""Source privacy/integrity checks and a real FFmpeg A/V fixture smoke test."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

from pathlib import Path
import json
import hashlib
import shutil
import subprocess
import tempfile
import unittest
import zipfile

from borrow_mac_bundle import REQUIRED, create_bundle, source_allowed, verify_bundle
from portability_fixtures import generate_fixture


class BundleTests(unittest.TestCase):
    def setUp(self):
        scratch = _STREAMER_ROOT / ".test-tmp"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="borrow-bundle-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "streamer"
        self.root.mkdir()
        for name in REQUIRED:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"public test stand-in: {name}\n", encoding="utf-8")
        self.archive = Path(self.temporary.name) / "source.zip"

    def add_source(self, name, text="DO NOT SHARE PRIVATE SENTINEL"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def mutate(self, callback):
        with zipfile.ZipFile(self.archive) as archive:
            data = {name: archive.read(name) for name in archive.namelist()}
        callback(data)
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, content in data.items():
                archive.writestr(name, content)

    def test_allowlist_excludes_private_files_and_includes_runtime_new_tools(self):
        blocked = ["settings.json", "queue.json", "resolvers.json", "logs/player.log",
                   "credentials.json", "secret.py", ".env", ".git/config", ".venv/site.py",
                   "testdata/private.mp4", "model_weights/weight.pt", ".test-tmp/private.py",
                   ".port_baseline/oh_player.py",
                   "src/oh_streamer/private.py", "src/oh_streamer/player/private.py",
                   "src/oh_streamer/resources/resolvers.json",
                   "src/oh_streamer/resources/queue_ui/private.json",
                   "benchmarks/bench_startup.py", "queue_ui/private.json", "resolvers/private.json"]
        for name in blocked:
            self.add_source(name)
        public = ["new_preflight_tool.py", "docs/docs_new_tool.md", "resolvers/resolve_public.py",
                  "packaging/build_macos.py", "streamer_macos.spec", "run_tests.py",
                  "src/oh_streamer/player/oh_player.py",
                  "src/oh_streamer/player/player_runtime.py",
                  "src/oh_streamer/platform/player_branding.py",
                  "src/oh_streamer/platform/windows_identity.py",
                  "src/oh_streamer/resources/oh_subtitle_preferences.lua",
                  "src/oh_streamer/resources/icons/app.icns"]
        for name in public:
            self.add_source(name, "public source")
        self.add_source("tests/test_public.py", "public test")
        self.add_source("tests/__init__.py", "test package")
        manifest = create_bundle(self.archive, self.root)
        self.assertTrue({"streamer/" + name for name in public}.issubset(manifest["members"]))
        for name in blocked + ["tests/test_public.py", "tests/__init__.py"]:
            self.assertNotIn("streamer/" + name, manifest["members"])
        self.assertFalse(manifest["model_weights_included"])
        self.assertEqual(verify_bundle(self.archive), manifest)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertFalse(any(b"DO NOT SHARE PRIVATE SENTINEL" in archive.read(name) for name in archive.namelist()))
            launcher = archive.getinfo("streamer/setup_macos.command")
            self.assertTrue((launcher.external_attr >> 16) & 0o111)
        manifest = create_bundle(self.archive, self.root.parent, include_tests=True)
        self.assertIn("streamer/tests/test_public.py", manifest["members"])
        self.assertIn("streamer/tests/__init__.py", manifest["members"])
        extracted = Path(self.temporary.name) / "extracted"
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(extracted)  # only after verify_bundle in create_bundle
        for name, expected in manifest["members"].items():
            self.assertEqual(hashlib.sha256((extracted / name).read_bytes()).hexdigest(), expected["sha256"])

    def test_incomplete_runtime_is_rejected(self):
        for name in ("src/oh_streamer/resources/queue_ui/app.js",
                     "src/oh_streamer/resources/queue_ui/branding.js",
                     "src/oh_streamer/resources/icons/app.icns",
                     "src/oh_streamer/branding.py", "Oh Streamer.command",
                     "src/oh_streamer/queue/queue_worker.py", "resolvers/resolve_browser.py",
                     "src/oh_streamer/sources/resolvers/resolve_browser.py",
                     "src/oh_streamer/player/player_runtime.py",
                     "src/oh_streamer/platform/player_branding.py",
                     "src/oh_streamer/platform/windows_identity.py"):
            with self.subTest(missing=name):
                path = self.root / name
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    create_bundle(self.archive, self.root)
                self.assertFalse(self.archive.exists())
                path.write_text(f"public test stand-in: {name}\n", encoding="utf-8")

    def test_branding_and_renamed_launchers_survive_source_bundle(self):
        from oh_streamer.branding import APP_NAME, LOGO_TEXT, parse_branding

        branding_path = "src/oh_streamer/resources/queue_ui/branding.js"
        self.add_source(branding_path, (_STREAMER_ROOT / branding_path).read_text(encoding="utf-8"))
        manifest = create_bundle(self.archive, self.root)
        for name in (branding_path, "src/oh_streamer/branding.py",
                     "Oh Player.command", "Oh Streamer.command"):
            self.assertIn("streamer/" + name, manifest["members"])
        with zipfile.ZipFile(self.archive) as archive:
            parsed = parse_branding(archive.read("streamer/" + branding_path).decode("utf-8"))
            self.assertEqual(parsed, {"name": APP_NAME, "logo": LOGO_TEXT})
            for launcher in ("Oh Player.command", "Oh Streamer.command"):
                self.assertTrue((archive.getinfo("streamer/" + launcher).external_attr >> 16) & 0o111)

    def test_corruption_missing_extra_duplicate_and_manifest_mismatch(self):
        mutations = [
            lambda data: data.update({"streamer/oh_player.py": b"corrupted source"}),
            lambda data: data.pop("streamer/oh_player.py"),
            lambda data: data.update({"streamer/surprise.py": b"unexpected"}),
            lambda data: data.pop("manifest.json"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                create_bundle(self.archive, self.root)
                self.mutate(mutation)
                with self.assertRaises(ValueError):
                    verify_bundle(self.archive)
        create_bundle(self.archive, self.root)
        with zipfile.ZipFile(self.archive, "a") as archive:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("streamer/oh_player.py", "duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_bundle(self.archive)
        create_bundle(self.archive, self.root)
        def change_manifest(data):
            manifest = json.loads(data["manifest.json"])
            manifest["members"]["streamer/oh_player.py"]["sha256"] = "0" * 64
            data["manifest.json"] = json.dumps(manifest).encode()
        self.mutate(change_manifest)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_bundle(self.archive)

    def test_archive_paths_are_not_extractable_traversals(self):
        for name in ["../private.py", "/private.py", "streamer/../private.py",
                     "streamer/C:/private.py", "streamer/..\\private.py", "streamer//private.py"]:
            with self.subTest(name=name):
                create_bundle(self.archive, self.root)
                self.mutate(lambda data: data.update({name: b"danger"}))
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    verify_bundle(self.archive)
        self.assertFalse(source_allowed("../escape.py"))

    def test_symlink_source_is_rejected(self):
        target = Path(self.temporary.name) / "outside.py"
        target.write_text("private", encoding="utf-8")
        try:
            (self.root / "linked.py").symlink_to(target)
        except OSError:
            self.skipTest("Creating symlinks requires permission on this host")
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_bundle(self.archive, self.root)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
    def test_real_fixture_has_video_audio_timebase_and_moving_frames(self):
        output = Path(self.temporary.name) / "synthetic.mp4"
        report = generate_fixture(output, duration=0.5)
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)], timeout=20))
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
        self.assertEqual((video["width"], video["height"], video["r_frame_rate"]), (320, 180, "30/1"))
        self.assertEqual(int(video["nb_frames"]), 15)
        self.assertEqual(audio["codec_name"], "aac")
        self.assertEqual(int(audio["sample_rate"]), 48000)
        self.assertAlmostEqual(float(info["format"]["duration"]), 0.5, delta=0.06)
        md5 = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:v:0",
                                       "-f", "framemd5", "-"], timeout=20).decode()
        hashes = [line.rsplit(",", 1)[1].strip() for line in md5.splitlines() if line and not line.startswith("#")]
        self.assertGreater(len(set(hashes)), 1)
        self.assertIn("not a real mosaic", report["purpose"])
        self.assertEqual(report["frame_count"], 15)
        repeated = generate_fixture(Path(self.temporary.name) / "repeated.mp4", duration=0.5)
        self.assertEqual(report["sha256"], repeated["sha256"])

    def test_invalid_fixture_options_fail_before_encoding(self):
        for options in [{"duration": 0}, {"duration": float("nan")}, {"width": 321}, {"fps": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_fixture(Path(self.temporary.name) / "bad.mp4", **options)


if __name__ == "__main__":
    unittest.main()
