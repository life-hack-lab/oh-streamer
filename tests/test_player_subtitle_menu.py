"""Real mpv/Lua integration checks; set OH_STREAMER_TEST_MPV to enable."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oh_streamer.player.mpv_ipc import MpvEndpoint, MpvIpc

MPV = os.environ.get("OH_STREAMER_TEST_MPV") or shutil.which("mpv")


@unittest.skipUnless(MPV, "Set OH_STREAMER_TEST_MPV to run real player UI checks")
class SubtitleMenuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subtitle-menu-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.endpoint = MpvEndpoint()
        self.addCleanup(self.endpoint.close)
        self.log = self.directory / "mpv.log"
        self.process = subprocess.Popen([
            MPV, "--no-config", "--idle=yes", "--pause=yes", "--keep-open=yes", "--osc=no",
            "--vo=" + os.environ.get("OH_STREAMER_TEST_MPV_VO", "null"), "--ao=null", "--force-window=yes",
            "--geometry=960x540", "--input-ipc-server=" + self.endpoint.path,
            "--log-file=" + str(self.log), "--script=" + str(ROOT / "src/oh_streamer/resources/oh_seek.lua"),
            "--script=" + str(ROOT / "src/oh_streamer/resources/oh_subtitle_preferences.lua"),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.stop)
        self.ipc = MpvIpc(self.endpoint.path, self.process, timeout=5)
        self.addCleanup(self.ipc.close)
        self.wait(lambda: any("oh-key-subtitles" in b.get("cmd", "")
                              for b in self.ipc.get("input-bindings", [])))
        self.ipc.set("user-data/oh/active", "1")
        self.ipc.set("user-data/oh/duration", 60)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        log = self.log.read_text(encoding="utf-8", errors="replace") if self.log.exists() else ""
        errors = "\n".join(line for line in log.splitlines() if "[e][lada_seek]" in line or "[f][lada_seek]" in line)
        self.assertFalse(errors, errors)

    def wait(self, condition):
        until = time.monotonic() + 5
        while time.monotonic() < until:
            if condition():
                return
            time.sleep(0.05)
        log = self.log.read_text(encoding="utf-8", errors="replace")
        self.fail("mpv condition timed out: " + "\n".join(line for line in log.splitlines() if "[lada_seek]" in line)[-2500:])

    def key(self, key):
        self.ipc.command("keypress", key)
        time.sleep(0.1)

    def setting(self, key, value):
        self.ipc.command("script-message", "oh-setting", key, value)
        time.sleep(0.1)

    def menu_open(self):
        return any("oh-smenu-ENTER" in b.get("cmd", "") for b in self.ipc.get("input-bindings", []))

    def test_empty_tracks_keyboard_and_local_adjustments(self):
        self.key("u")
        self.assertTrue(self.menu_open())
        self.key("DOWN")  # track list, with only the off option
        self.key("ENTER")
        self.key("ESC")  # back to subtitle controls
        self.assertTrue(self.menu_open())
        self.key("ESC")
        self.assertFalse(self.menu_open())

        self.setting("subtitle_visibility", "no")
        self.assertFalse(self.ipc.get("sub-visibility"))
        self.setting("subtitle_visibility", "yes")
        self.setting("subtitle_scale", "1.5")
        self.setting("subtitle_delay", "0.1")
        self.assertEqual(self.ipc.get("sub-scale"), 1.5)
        self.assertAlmostEqual(self.ipc.get("sub-delay"), 0.1)
        self.setting("subtitle_delay", "0")
        self.assertEqual(self.ipc.get("sub-delay"), 0)
        self.key("u")
        for _ in range(3):
            self.key("DOWN")
        self.key("ENTER")
        self.key("ENTER")
        self.assertAlmostEqual(self.ipc.get("sub-delay"), -0.1)
        self.assertIsNone(self.ipc.get("user-data/oh/settings-request"))
        self.assertIsNone(self.ipc.get("user-data/oh/settings-set"))

    def test_track_updates_hide_and_reselect(self):
        source = "av://lavfi:color=c=black:s=960x540:r=1"
        self.ipc.set("user-data/oh/expect-load", source)
        self.ipc.command("loadfile", source)
        self.wait(lambda: self.ipc.get("path") == source and self.ipc.get("video-params") is not None)
        self.key("u")
        self.key("DOWN")
        self.key("ENTER")  # keep track submenu open while tracks arrive
        for index in range(2):
            sub = self.directory / f"track{index}.srt"
            sub.write_text("1\n00:00:00,000 --> 00:01:00,000\nSubtitle test\n", encoding="utf-8")
            self.ipc.command("sub-add", str(sub), "auto", f"Subtitle {index}", "eng")
        self.wait(lambda: len([t for t in self.ipc.get("track-list", []) if t["type"] == "sub"]) == 2)
        tracks = [t for t in self.ipc.get("track-list") if t["type"] == "sub"]
        self.setting("subtitle_track", str(tracks[0]["id"]))
        self.setting("subtitle_visibility", "no")
        self.setting("subtitle_track", str(tracks[0]["id"]))
        self.assertTrue(self.ipc.get("sub-visibility"))
        self.setting("subtitle_visibility", "no")
        self.setting("subtitle_track", str(tracks[-1]["id"]))
        self.assertEqual(self.ipc.get("sid"), tracks[-1]["id"])
        self.assertTrue(self.ipc.get("sub-visibility"))
        self.setting("subtitle_track", "no")
        self.assertFalse(self.ipc.get("sid"))
        self.setting("subtitle_track", str(tracks[0]["id"]))
        self.assertEqual(self.ipc.get("sid"), tracks[0]["id"])
        self.ipc.command("sub-remove", tracks[-1]["id"])
        self.key("ESC")
        self.assertTrue(self.menu_open())
        self.assertEqual(self.ipc.get("path"), source)
        self.ipc.set("user-data/oh/casting", "1")
        time.sleep(0.1)
        self.setting("subtitle_scale", "2")
        self.assertEqual(self.ipc.get("sub-scale"), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for embedded track preference")
    def test_embedded_concise_preferred_over_default_full(self):
        subs = []
        for index in range(2):
            path = self.directory / f"embedded{index}.srt"
            path.write_text("1\n00:00:00,000 --> 00:00:02,000\n字幕\n", encoding="utf-8")
            subs.append(str(path))
        source = self.directory / "two-tracks.mkv"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=64x64:r=1:d=2", "-i", subs[0], "-i", subs[1],
                        "-map", "0:v", "-map", "1:0", "-map", "2:0", "-c:v", "ffv1", "-c:s", "srt",
                        "-metadata:s:s:0", "language=chi", "-metadata:s:s:1", "language=chi",
                        "-metadata:s:s:0", "title=繁體中文（AI・完整）",
                        "-metadata:s:s:1", "title=繁體中文（AI・精簡）",
                        "-disposition:s:0", "default", "-disposition:s:1", "0", str(source)],
                       check=True, capture_output=True)
        self.ipc.set("user-data/oh/expect-load", str(source))
        self.ipc.command("loadfile", str(source))
        self.wait(lambda: len([t for t in self.ipc.get("track-list", []) if t["type"] == "sub"]) == 2)
        tracks = [t for t in self.ipc.get("track-list") if t["type"] == "sub"]
        concise = next(t for t in tracks if "精簡" in t["title"])
        full = next(t for t in tracks if "完整" in t["title"])
        self.wait(lambda: self.ipc.get("sid") == concise["id"])
        self.ipc.set("sid", full["id"])
        self.assertEqual(self.ipc.get("sid"), full["id"])

    @unittest.skipUnless(os.environ.get("OH_STREAMER_TEST_MPV_VO") == "gpu", "Mouse checks need a rendered window")
    def test_mouse_button_and_settings_switching(self):
        source = "av://lavfi:color=c=black:s=960x540:r=1"
        self.ipc.set("user-data/oh/expect-load", source)
        self.ipc.command("loadfile", source)
        self.wait(lambda: self.ipc.get("video-params") is not None)
        self.wait(lambda: self.ipc.get("osd-dimensions", {}).get("w", 0) > 0)
        size = self.ipc.get("osd-dimensions")
        scale = max(1, min(3, self.ipc.get("display-hidpi-scale", 1)))
        self.ipc.command("mouse", int(size["w"] - 120 * scale), int(size["h"] - 38 * scale))
        time.sleep(0.2)
        self.key("MBTN_LEFT")
        self.assertTrue(self.menu_open())
        self.key("ENTER")  # display options
        self.key("DOWN")  # hide (back, show, hide)
        self.key("ENTER")
        self.assertFalse(self.ipc.get("sub-visibility"))
        self.ipc.set("user-data/oh/settings", '[{"key":"vsr","label":"VSR","kind":"live","value":"off","options":[{"v":"on","label":"On"},{"v":"off","label":"Off"}]}]')
        self.key("s")  # switch from subtitles to settings
        self.key("ENTER")
        self.key("UP")  # choose On instead of Off
        self.key("ENTER")
        self.assertIn('"vsr":"on"', self.ipc.get("user-data/oh/settings-set"))
        self.key("u")
        self.key("ESC")
        self.assertFalse(self.menu_open())
        self.ipc.set("window-scale", 0.5)
        self.wait(lambda: self.ipc.get("osd-dimensions", {}).get("w", 960) < 700)
        size = self.ipc.get("osd-dimensions")
        scale = min(scale, size["w"] / 340)
        self.ipc.command("mouse", int(size["w"] - 120 * scale), int(size["h"] - 38 * scale))
        time.sleep(0.2)
        self.key("MBTN_LEFT")
        self.assertTrue(self.menu_open())


if __name__ == "__main__":
    unittest.main()
