"""VSR control and pipeline tests; GPU integration is opt-in (OH_STREAMER_TEST_VSR=1)."""
import ast
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import cast_vsr as V, hls_cast as H, airplay_cast as A
from oh_streamer.player import player_settings as PS


def manager():
    source = Path(__file__).resolve().parents[1] / "src/oh_streamer/player/oh_player.py"
    node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == "CastManager")
    ns = dict(airplay_cast=A, json=json, queue=queue, time=time, threading=threading, log=lambda *_: None)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "CastManager", "exec"), ns)
    obj = object.__new__(ns["CastManager"])
    obj.active = None
    obj.vsr_enabled = obj.vsr_checking = False
    obj.events = queue.Queue()
    obj._osd = Mock()
    obj._ipc = Mock()
    obj.publish_menu = Mock()
    obj.vsr_commit = Mock()
    return obj


class VsrTest(unittest.TestCase):
    def test_resolution_and_copy_mux(self):
        self.assertEqual(V.output_size(1280, 720), (2560, 1440))
        self.assertEqual(V.output_size(1920, 1080), (3840, 2160))
        self.assertEqual(V.output_size(1280, 720, "4k"), (3840, 2160))
        self.assertEqual(V.output_size(2560, 1080, "4k"), (3840, 1620))
        self.assertEqual(V.output_size(720, 1280, "4k"), (1214, 2160))
        with self.assertRaises(ValueError):
            V.output_size(3840, 2160, "4k")
        with self.assertRaises(ValueError):
            V.output_size(1280, 720, "invalid")
        self.assertEqual(V.output_size(2560, 1080), (3840, 1620))
        with self.assertRaises(ValueError):
            V.output_size(3840, 2160)
        p = H.SegmentPlan(30, 600, 3840, 2160)
        cmd = H.build_ffmpeg_cmd(p, 2, 1920, 1080, "libx264", "out", None, encoded_video=True)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertNotIn("-vf", cmd)
        self.assertIn("18.000000", cmd)  # TS_BASE + segment start
        self.assertIn("RESOLUTION=3840x2160", p.master_playlist())

    def test_setting_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            PS.save_file(path, {"cast_vsr": "on", "cast_vsr_size": "4k", "vsr": "off", "volume": 42})
            saved, warning = PS.load_file(path)
            self.assertIsNone(warning)
            self.assertEqual(saved["cast_vsr"], "on")
            self.assertEqual(saved["cast_vsr_size"], "4k")
            self.assertEqual(saved["vsr"], "off")
            self.assertEqual(saved["volume"], 42)
        with self.assertRaises(ValueError):
            PS.validate("cast_vsr", True)
        with self.assertRaises(ValueError):
            PS.validate("cast_vsr_size", "3x")

    def test_interrupted_encoder_drain_does_not_mark_source_end(self):
        vod = object.__new__(H.HlsVod)
        vod.cond = threading.Condition()
        vod.gen, vod.eof_k, vod.last_error = 0, None, None
        vod.plan = H.SegmentPlan(30, 360, 640, 360)
        vod.log = Mock()
        run = SimpleNamespace(eof=True, stopping=True, error=None, gen=0, next_k=2, reported_k=1)
        vod._run_ended(run, 0)
        self.assertIsNone(vod.eof_k)
        run.stopping = False
        run.next_k, run.reported_k = 3, 2
        vod._run_ended(run, 0)
        self.assertEqual(vod.eof_k, 3)

    def test_failed_pipeline_start_reaps_already_started_process(self):
        child = Mock()
        child.poll.return_value = None
        with patch.object(V.subprocess, "Popen", side_effect=[child, FileNotFoundError("NVEncC")]):
            with self.assertRaises(FileNotFoundError):
                V.Pipeline("missing.exe", H.SegmentPlan(30, 120, 640, 360), 320, 180, ["ffmpeg"])
        child.kill.assert_called_once()
        child.wait.assert_called_once()
        child.stdin.close.assert_called_once()
        child.stdout.close.assert_called_once()

    def test_probe_failure_does_not_enable_or_persist(self):
        m = manager()
        with patch.object(V, "probe", side_effect=RuntimeError("missing DLL")):
            m.handle_request("1:vsr", None, None)
            kind, data = m.events.get(timeout=3)
        m.events.put((kind, data))
        m.poll()
        self.assertFalse(m.vsr_enabled)
        m.vsr_commit.assert_not_called()
        self.assertIn("missing DLL", m._osd.call_args.args[0])

    def test_toggle_success_and_active_guard(self):
        m = manager()
        with patch.object(V, "probe", return_value="NVEncC64.exe"):
            m.handle_request("1:vsr", None, None)
            event = m.events.get(timeout=3)
        m.events.put(event)
        m.poll()
        self.assertTrue(m.vsr_enabled)
        m.vsr_commit.assert_called_once_with({"cast_vsr": "on"})
        m.active = {"mode": "hls"}
        m.handle_request("2:vsr", None, None)
        self.assertTrue(m.vsr_enabled)
        m.active = None
        m.handle_request("3:vsr", None, None)
        self.assertFalse(m.vsr_enabled)

    def test_size_toggle_persistence_and_guards(self):
        m = manager()
        m.handle_request("1:vsr-size", None, None)
        self.assertEqual(m.vsr_size, "4k")
        self.assertFalse(m.vsr_enabled)
        m.vsr_commit.assert_called_once_with({"cast_vsr_size": "4k"})
        for active, checking in (({"mode": "hls"}, False), (None, True)):
            m.active, m.vsr_checking = active, checking
            m.handle_request("2:vsr-size", None, None)
            self.assertEqual(m.vsr_size, "4k")
        m.active, m.vsr_checking = None, False
        m.handle_request("3:vsr-size", None, None)
        self.assertEqual(m.vsr_size, "2x")

    def test_menu_switch_present_and_locked_during_cast(self):
        m = manager()
        del m.publish_menu
        m.devices, m.scanning, m.scan_error, m.pyatv_ok, m.published_menu = [], False, None, True, None
        m.publish_menu()
        rows = json.loads(m._ipc.call_args.args[-1])
        self.assertEqual(rows[-2]["id"], "vsr")
        self.assertEqual(rows[-1]["id"], "vsr-size")
        self.assertEqual(rows[-1]["kind"], "item")
        m.active = {"device": "TV", "url": "http://local"}
        m.publish_menu()
        self.assertEqual(json.loads(m._ipc.call_args.args[-1])[-1]["kind"], "info")


@unittest.skipUnless(os.environ.get("OH_STREAMER_TEST_VSR") == "1", "requires NVIDIA RTX + NVEncC")
class VsrGpuTest(unittest.TestCase):
    def test_720p_to_4k(self):
        self._check_4k(1280, 720)

    def test_4k_fractional_fps_without_audio(self):
        self._check_4k(1920, 1080)

    def _check_4k(self, width, height):
        import av
        import numpy as np
        from fractions import Fraction
        fps = Fraction(30000, 1001)
        plan = H.SegmentPlan(fps, 150, *V.output_size(width, height, "4k"), has_audio=False)
        class Frames:
            closed = False
            def __iter__(self):
                # Spatial detail (not a uniform frame) exercises the VSR filter.
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 1] = np.arange(width, dtype=np.uint16)[None, :] % 256
                for i in range(150):
                    if self.closed:
                        break
                    yield frame, float(i / fps)
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            vod = H.HlsVod(plan, lambda _: H.RunInput(Frames(), width, height, None), "libx264",
                           vsr_exe=V.executable(), cache_root=tmp, request_timeout=60,
                           stderr=subprocess.DEVNULL)
            try:
                for k, count in ((0, 120), (1, 30)):
                    data, reason = vod.get_segment(k)
                    self.assertIsNotNone(data, reason)
                    with av.open(io.BytesIO(data), format="mpegts") as container:
                        frames = container.decode(video=0)
                        first = next(frames)
                        self.assertEqual((first.width, first.height), (3840, 2160))
                        self.assertTrue(first.key_frame)
                        self.assertAlmostEqual(float(first.pts * first.time_base),
                                               11.4 + float(plan.start(k)), places=3)
                        self.assertEqual(1 + sum(1 for _ in frames), count)
                        self.assertFalse(container.streams.audio)
                self.assertIsNone(vod.last_error)
            finally:
                vod.close()
                self.assertFalse(vod.child_pids)

    def test_real_vsr_hls_audio_seek_and_cleanup(self):
        import av
        import numpy as np
        exe = V.probe()
        with tempfile.TemporaryDirectory() as tmp:
            audio = str(Path(tmp) / "audio.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=12",
                            "-y", audio], check=True)
            # 12 sec, start in the middle, then restart before the first cached segment.
            fps = 30
            plan = H.SegmentPlan(fps, 360, 640, 360, has_audio=True)
            class Frames:
                def __init__(self, t):
                    self.start = round(t * fps)
                    self.closed = False
                def __iter__(self):
                    for i in range(self.start, 360):
                        if self.closed:
                            break
                        yield np.full((180, 320, 3), [40, 100, 180], dtype=np.uint8), i / fps
                def close(self):
                    self.closed = True
            def provider(t):
                return H.RunInput(Frames(t), 320, 180, {"src": audio, "fmt_start": 0})
            with open(Path(tmp) / "encoder.log", "wb") as logs:
                vod = H.HlsVod(plan, provider, "libx264", start_k=1, cache_root=tmp,
                               request_timeout=60, vsr_exe=exe, stderr=logs)
                try:
                    for k in (1, 0, 2):
                        data, reason = vod.get_segment(k)
                        self.assertIsNotNone(data, reason)
                        with av.open(io.BytesIO(data), format="mpegts") as container:
                            frames = list(container.decode(video=0))
                            self.assertEqual(len(frames), 120)
                            self.assertEqual((frames[0].width, frames[0].height), (640, 360))
                            self.assertTrue(frames[0].key_frame)
                            self.assertAlmostEqual(float(frames[0].pts * frames[0].time_base),
                                                   H.TS_BASE + H.MPEGTS_DELAY + k * 4, places=2)
                            self.assertEqual(len(container.streams.audio), 1)
                        with av.open(io.BytesIO(data), format="mpegts") as container:
                            a = next(container.decode(audio=0))
                            self.assertLess(abs(float(a.pts * a.time_base) - (11.4 + k * 4)), .1)
                    self.assertIsNone(vod.last_error)
                finally:
                    vod.close()
                    self.assertFalse(vod.child_pids)


if __name__ == "__main__":
    unittest.main()
