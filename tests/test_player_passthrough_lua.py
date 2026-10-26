"""mpv integration checks for the progressive source handoff in oh_seek.lua."""
import json
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
SOURCE = "av://lavfi:color=c=black:s=64x64:r=10:d=8"


@unittest.skipUnless(MPV, "Set OH_STREAMER_TEST_MPV to run Lua handoff integration checks")
class PassthroughLuaTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / ".tmp-t12-lua"
        temp_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="mpv-", dir=temp_root)
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.endpoint = MpvEndpoint()
        self.addCleanup(self.endpoint.close)
        self.log = self.directory / "mpv.log"
        self.process = subprocess.Popen([
            MPV, "--no-config", "--idle=yes", "--pause=yes", "--keep-open=yes", "--osc=no",
            "--vo=" + os.environ.get("OH_STREAMER_TEST_MPV_VO", "null"), "--ao=null", "--force-window=yes",
            "--input-ipc-server=" + self.endpoint.path,
            "--log-file=" + str(self.log),
            "--script=" + str(ROOT / "src/oh_streamer/resources/oh_seek.lua"),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.stop)
        self.ipc = MpvIpc(self.endpoint.path, self.process, timeout=5)
        self.addCleanup(self.ipc.close)
        self.wait(lambda: any("oh-seek-LEFT" in b.get("cmd", "")
                             for b in self.ipc.get("input-bindings", [])))

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        log = self.log.read_text(encoding="utf-8", errors="replace") if self.log.exists() else ""
        errors = "\n".join(line for line in log.splitlines()
                            if "[e][lada_seek]" in line or "[f][lada_seek]" in line)
        self.assertFalse(errors, errors)

    def wait(self, condition):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            if condition():
                return
            time.sleep(0.04)
        log = self.log.read_text(encoding="utf-8", errors="replace") if self.log.exists() else ""
        self.fail("mpv condition timed out: " + "\n".join(
            line for line in log.splitlines() if "[lada_seek]" in line)[-2500:])

    def prepare(self):
        self.ipc.set("user-data/oh/active", "1")
        self.ipc.set("user-data/oh/duration", 8)
        self.ipc.set("user-data/oh/handoff-resumed", "")
        self.ipc.set("user-data/oh/resumed", "previous-user-resume")
        self.ipc.set("user-data/oh/served", "previous-user-request")
        # Let mpv deliver the active observer before invoking the scripted seek/pause bindings.
        time.sleep(0.1)

    def set_handoff(self, *, token, url=SOURCE, paused=False, position=0.0):
        self.ipc.set("user-data/oh/handoff", json.dumps({
            "token": token, "url": url, "paused": paused, "position": position,
        }))

    def test_playing_handoff_waits_for_position_and_keeps_user_ack_properties(self):
        self.prepare()
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.set_handoff(token="handoff-1", position=2.5)
        self.ipc.command("loadfile", SOURCE, "replace")

        self.wait(lambda: self.ipc.get("path") == SOURCE
                  and self.ipc.get("core-idle") is False
                  and 0.2 < (self.ipc.get("time-pos") or 0) < 1.8)
        self.assertEqual(self.ipc.get("user-data/oh/handoff-resumed"), "")
        self.wait(lambda: self.ipc.get("user-data/oh/handoff-resumed") == "handoff-1")
        self.assertGreaterEqual(self.ipc.get("time-pos"), 2.5)
        self.assertEqual(self.ipc.get("user-data/oh/served"), "previous-user-request")
        self.assertEqual(self.ipc.get("user-data/oh/resumed"), "previous-user-resume")

    def test_paused_handoff_acks_at_file_loaded(self):
        self.prepare()
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.set_handoff(token="handoff-paused", paused=True, position=3.0)
        self.ipc.command("loadfile", SOURCE, "replace")

        self.wait(lambda: self.ipc.get("path") == SOURCE
                  and self.ipc.get("user-data/oh/handoff-resumed") == "handoff-paused")
        self.assertTrue(self.ipc.get("pause"))
        self.assertLess(self.ipc.get("time-pos") or 0, 3.0)

    def test_user_resume_during_handoff_updates_pause_intent(self):
        self.prepare()
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.set_handoff(token="handoff-resume", paused=True, position=2.0)
        # The user resumes between the controller's publish and loadfile; file-loaded must honor that change.
        until = time.monotonic() + 2
        published = json.loads(self.ipc.get("user-data/oh/handoff"))
        while published["paused"] and time.monotonic() < until:
            self.ipc.command("script-binding", "lada_seek/oh-pause-p")
            time.sleep(0.05)
            published = json.loads(self.ipc.get("user-data/oh/handoff"))
        self.assertFalse(published["paused"])
        self.ipc.command("loadfile", SOURCE, "replace")

        self.wait(lambda: self.ipc.get("path") == SOURCE
                  and self.ipc.get("user-data/oh/handoff-resumed") == "handoff-resume")
        self.assertFalse(self.ipc.get("pause"))
        self.assertGreaterEqual(self.ipc.get("time-pos") or 0, 2.0)

    def test_pending_user_seek_cannot_be_completed_by_stale_handoff_or_served(self):
        self.prepare()
        self.set_handoff(token="stale-handoff", position=1.0)
        self.ipc.command("script-message", "oh-seek", "4")
        self.wait(lambda: self.ipc.get("user-data/oh/seek-request"))
        request = self.ipc.get("user-data/oh/seek-request")
        self.assertTrue(request)
        self.assertEqual(self.ipc.get("user-data/oh/handoff"), "")

        # Simulate a controller race: an old handoff and served value arrive after the new user request.
        self.set_handoff(token="stale-handoff", position=1.0)
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.ipc.set("user-data/oh/served", request)
        self.ipc.command("loadfile", SOURCE, "replace")
        self.wait(lambda: self.ipc.get("path") == SOURCE)
        time.sleep(0.2)

        self.assertEqual(self.ipc.get("user-data/oh/seek-request"), request)
        self.assertEqual(self.ipc.get("user-data/oh/handoff-resumed"), "")
        self.assertEqual(self.ipc.get("user-data/oh/resumed"), "previous-user-resume")

    def test_internal_source_urls_are_not_drops_and_stale_entries_are_removed(self):
        self.prepare()
        source_a = "av://lavfi:color=c=black:s=64x64:r=5:d=8"
        source_b = "av://lavfi:color=c=white:s=64x64:r=5:d=8"
        self.ipc.set("user-data/oh/expect-load", source_a)
        self.ipc.set("user-data/oh/internal-urls", json.dumps([source_a, source_b]))
        self.ipc.command("loadfile", source_a, "replace")
        self.wait(lambda: self.ipc.get("path") == source_a)
        self.ipc.command("loadfile", source_b, "append")

        self.wait(lambda: self.ipc.get("playlist-count") == 1)
        self.assertIn(source_a, [e.get("filename") for e in self.ipc.get("playlist", [])])
        self.assertIn(self.ipc.get("user-data/oh/open-request"), (None, ""))

    def test_dropped_loads_are_discarded_not_played_or_forwarded(self):
        dropped = "av://lavfi:color=c=red:s=64x64:r=10:d=8"
        # idle: a foreign load (what a drag & drop does) is cancelled before mpv plays it
        self.ipc.command("loadfile", dropped, "replace")
        time.sleep(0.5)
        self.assertIn(self.ipc.get("path"), (None, ""))
        self.assertEqual(self.ipc.get("playlist-count"), 0)
        # playing: a dropped entry appended behind the segment is taken out again, the segment keeps playing
        self.prepare()
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.ipc.command("loadfile", SOURCE, "replace")
        self.wait(lambda: self.ipc.get("path") == SOURCE)
        self.ipc.command("loadfile", dropped, "append")
        self.wait(lambda: self.ipc.get("playlist-count") == 1)
        time.sleep(0.4)
        self.assertEqual(self.ipc.get("path"), SOURCE)
        self.assertNotIn(dropped, [e.get("filename") for e in self.ipc.get("playlist", [])])
        self.assertIn(self.ipc.get("user-data/oh/open-request"), (None, ""))

    def test_passthrough_keeps_compare_intent_without_installing_a_graph(self):
        self.prepare()
        self.ipc.set("user-data/oh/passthrough", "1")
        self.ipc.set("user-data/oh/compare_layers", "original,processed")
        self.ipc.set("user-data/oh/expect-load", SOURCE)
        self.ipc.command("loadfile", SOURCE, "replace")

        self.wait(lambda: self.ipc.get("path") == SOURCE)
        self.assertEqual(self.ipc.get("user-data/oh/compare_layers"), "original,processed")
        self.assertIn(self.ipc.get("lavfi-complex"), (None, ""))
        self.assertNotEqual(self.ipc.get("user-data/oh/processed"), "1")

        # Leaving passthrough asks Lua to apply the saved compare intent on the next restored load.
        self.ipc.set("user-data/oh/passthrough", "")
        self.wait(lambda: self.ipc.get("user-data/oh/passthrough") == "")

    def test_controller_advances_a_keep_open_clip_that_ended_before_append(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock
        from tests.test_player_controller import controller_args, controller_deps, FakeEngine
        from oh_streamer.player.controller import PlayerController

        short = "av://lavfi:color=c=black:s=64x64:r=10:d=0.2"
        next_url = "av://lavfi:color=c=white:s=64x64:r=10:d=8,setpts=PTS+0.2/TB"
        self.prepare()
        self.ipc.set("rebase-start-time", False)
        self.ipc.set("user-data/oh/expect-load", short)
        self.ipc.set("pause", False)
        self.ipc.command("loadfile", short, "replace")
        self.wait(lambda: self.ipc.get("eof-reached") is True and self.ipc.get("pause") is True)
        ready = threading.Event()
        ready.set()
        clip = SimpleNamespace(url=short, end_s=0.2, retire=Mock())
        c = PlayerController(controller_args(), FakeEngine(), deps=controller_deps())
        c.ipc, c.epoch = self.ipc, 1
        c.progressive = SimpleNamespace(po=SimpleNamespace(source_url=None, path=short),
            feeder=SimpleNamespace(url=next_url, ready=ready, error=None))
        c.progressive_source, c.progressive_clip, c.progressive_active_url = False, clip, short
        c.progressive_continuation_tick(time.perf_counter())
        token = c.progressive_handoff["token"]
        self.assertFalse(c.progressive_handoff["paused"])
        self.wait(lambda: self.ipc.get("path") == next_url and
                  self.ipc.get("user-data/oh/handoff-resumed") == token)
        c.progressive_check_handoff(time.perf_counter())
        self.assertFalse(self.ipc.get("pause"))
        self.assertFalse(self.ipc.get("eof-reached"))
        self.assertEqual(c.progressive_active_url, next_url)
        clip.retire.assert_called_once_with(keep_continuation=True)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for source subtitle preservation")
    def test_source_embedded_subtitle_is_restored_by_stream_identity(self):
        from types import SimpleNamespace
        from tests.test_player_controller import controller_args, controller_deps, FakeEngine
        from oh_streamer.player.controller import PlayerController

        subtitle = self.directory / "native.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n來源內嵌字幕\n", encoding="utf-8")
        source = self.directory / "source.mkv"
        repaired = self.directory / "repaired.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=s=64x64:r=10:d=4", "-i", str(subtitle), "-map", "0:v", "-map", "1:s",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:s", "srt",
                        "-metadata:s:s:0", "title=繁體中文", "-output_ts_offset", "12", str(source)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(source),
                        "-map", "0:v", "-c", "copy", "-avoid_negative_ts", "disabled", str(repaired)], check=True)
        self.prepare()
        self.ipc.set("rebase-start-time", False)
        self.ipc.set("user-data/oh/expect-load", str(source))
        self.ipc.command("loadfile", str(source), "replace", -1, "start=13")
        self.wait(lambda: any(t.get("type") == "sub" for t in self.ipc.get("track-list", [])))
        native = next(t for t in self.ipc.get("track-list") if t.get("type") == "sub")
        self.ipc.set("sid", native["id"])
        self.wait(lambda: "來源內嵌字幕" in (self.ipc.get("sub-text") or ""))
        c = PlayerController(controller_args(), FakeEngine(), deps=controller_deps())
        c.ipc = self.ipc
        c.progressive = SimpleNamespace(po=SimpleNamespace(path=str(source), source_url=None, subtitle_offset=0,
            subtitles=[dict(path=str(subtitle), title=native.get("title"), lang=native.get("lang") or "", default=True)]))
        c.processed = None
        c.progressive_sub_dir = None
        c.progressive_sub_loaded = None
        c.progressive_sub_selection = None
        c.progressive_sub_visibility = None
        c.remember_progressive_subtitle()
        self.assertEqual(c.progressive_sub_selection[:2], ("source", native["ff-index"]))
        self.ipc.set("user-data/oh/expect-load", str(repaired))
        self.ipc.command("loadfile", str(repaired), "replace", -1, "start=13")
        self.wait(lambda: self.ipc.get("path") == str(repaired) and not any(
            t.get("type") == "sub" for t in self.ipc.get("track-list", [])))
        c.load_progressive_subtitles(str(repaired))
        tracks = self.ipc.get("track-list")
        chosen = next(t for t in tracks if t.get("type") == "sub" and t.get("id") == self.ipc.get("sid"))
        self.assertEqual(chosen["external-filename"], str(source))
        self.assertEqual(chosen["ff-index"], native["ff-index"])
        self.assertTrue(any(t.get("type") == "sub" and t.get("external-filename") == str(subtitle) for t in tracks))
        self.assertFalse(any(t.get("type") == "video" and t.get("external") for t in tracks))
        self.wait(lambda: "來源內嵌字幕" in (self.ipc.get("sub-text") or ""))


if __name__ == "__main__":
    unittest.main()
