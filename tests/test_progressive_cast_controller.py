"""Queue casting lifecycle without loading restoration models or a real receiver."""
import ast
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_player_controller import controller_args, controller_deps, FakeEngine
from oh_streamer.player.controller import PlayerController
from oh_streamer.player.progressive_cast import ProgressiveCastSource
from oh_streamer.platform import hls_cast as H, airplay_cast as A
from oh_streamer.processing import encoding_backend


def manager_type(hls=H):
    source = (ROOT / "src/oh_streamer/player/oh_player.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CastManager")
    ns = dict(hls_cast=hls, airplay_cast=A, json=json, queue=queue, time=time, log=lambda *_: None,
              LOG_TO_FILE=False)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "CastManager", "exec"), ns)
    return ns["CastManager"]


def cold_engine(encoder="auto"):
    """Use the real lazy engine without importing torch or loading any models."""
    source = (ROOT / "src/oh_streamer/player/oh_player.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Engine")
    ns = dict(encoding_backend=encoding_backend, threading=threading, log=lambda *_: None,
              PS=SimpleNamespace(fix_platform=lambda cfg: (cfg, [])))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "Engine", "exec"), ns)
    engine = ns["Engine"](SimpleNamespace(), {}, {"encoder": encoder}, [], [])
    engine.load = Mock(side_effect=AssertionError("casting must not load AI models"))
    return engine


class ProgressiveCastLifecycleTest(unittest.TestCase):
    def controller(self):
        c = PlayerController(controller_args(), FakeEngine(), deps=controller_deps())
        c.ipc, c.cast, c.link = Mock(), SimpleNamespace(active=None), Mock()
        c.progressive = Mock()
        c.progressive.po.queue_id = "job"
        c.progressive.ranges_text.return_value = "0.000-8.000"
        c.published_duration, c.progressive_next = 20, 0
        c.progressive_sent, c.ranges_shown = False, None
        return c

    def test_menu_does_not_probe_and_repeated_cast_reuses_source(self):
        c = self.controller()
        self.assertIsNone(c.cast_source("menu"))
        source = object()
        c.cast.active = {"source": source}
        self.assertIs(c.cast_source("dev:tv"), source)

    def test_cast_source_uses_existing_playback_and_encoder(self):
        c = self.controller()
        with patch("oh_streamer.player.progressive_cast.ProgressiveCastSource") as factory:
            self.assertIs(c.cast_source("hls"), factory.return_value)
            factory.assert_called_once_with(c.progressive, encoder="libx264")

    def test_cast_from_cold_engine_resolves_encoder_without_loading_models(self):
        c = self.controller()
        c.engine = cold_engine()
        with patch.object(encoding_backend, "encoder_usable", return_value=True) as probe, \
                patch("oh_streamer.player.progressive_cast.ProgressiveCastSource") as factory:
            self.assertIs(c.cast_source("dev:tv"), factory.return_value)
            factory.assert_called_once_with(c.progressive, encoder="h264_nvenc")
            probe.assert_called_once_with("h264_nvenc")
        c.engine.load.assert_not_called()
        self.assertIsNone(c.engine.sc)
        self.assertIsNone(c.engine.lada)
        self.assertFalse(c.engine.ready.is_set())

    def test_cold_encoder_resolution_is_cached_and_tracks_setting_changes(self):
        engine = cold_engine()
        with patch.object(encoding_backend, "encoder_usable", return_value=True) as probe:
            self.assertEqual(engine.encoder(), "h264_nvenc")
            self.assertEqual(engine.encoder(), "h264_nvenc")
            probe.assert_called_once()
            engine.cfg["encoder"] = "libx264"
            self.assertEqual(engine.encoder(), "libx264")
            probe.assert_called_once()
        engine.load.assert_not_called()

    def test_loaded_engine_retains_settings_encoder_resolution(self):
        engine = cold_engine()
        engine.sc = SimpleNamespace(cfg={"encoder": "auto"}, resolve_encoder=Mock(return_value="h264_amf"))
        with patch.object(encoding_backend, "encoder_usable") as probe:
            self.assertEqual(engine.encoder(), "h264_amf")
            engine.sc.resolve_encoder.assert_called_once_with("auto")
            probe.assert_not_called()

    def test_cast_tick_keeps_protection_and_receiver_clock_without_local_load(self):
        c = self.controller()
        c.cast.active = dict(mode="airplay", phase="playing", local_pos=12)
        c.ipc.get.return_value = "1"
        c.progressive_tick(3)
        c.progressive.tick.assert_called_once()
        c.progressive.clock.assert_called_once_with(12, False, subtitle_pos=12.0)
        c.ipc.command.assert_not_called()
        c.progressive.feeder.ready.is_set.assert_not_called()
        c.ipc.set.assert_called_once_with("user-data/oh/ranges", "0.000-8.000")

    def test_hls_tick_does_not_report_transcoder_position_as_viewer_clock(self):
        c = self.controller()
        c.cast.active = dict(mode="hls", phase="hls", local_pos=12)
        c.progressive_tick(3)
        c.progressive.tick.assert_called_once()
        c.progressive.clock.assert_not_called()
        c.ipc.command.assert_not_called()

    def test_start_stop_preserves_source_and_resumes_through_progressive_seek(self):
        vod, server = Mock(), Mock()
        vod.status.return_value = dict(requests=1, restarts=0, cached_segments=2, cached_bytes=100, evicted=0)
        server.start.return_value = server
        server.url.return_value = "http://local/index.m3u8"
        fake_h = SimpleNamespace(FileCastSource=H.FileCastSource, SegmentPlan=H.SegmentPlan,
                                 HlsVod=Mock(return_value=vod), HlsServer=Mock(return_value=server))
        cls = manager_type(fake_h)
        m = object.__new__(cls)
        m.active = None
        m.args = SimpleNamespace(cast_bind="127.0.0.1", cast_port=0, cast_cache_gb=1, cast_playlist="master")
        m._ipc, m._osd, m.publish_menu = Mock(), Mock(), Mock()
        m.caster = SimpleNamespace(tv_position=lambda: 9.5, stop=lambda: True)
        source = SimpleNamespace(is_progressive_cast=True, meta=SimpleNamespace(video_fps_exact=60,
                                 frames_count=1200, duration=20, video_width=64, video_height=48,
                                 start_pts=5000000, time_base=1/1000000), fmt_start=5,
                                 has_audio=True, encoder="libx264", target=5, playback=Mock())
        # Exercise the real adapter contract rather than mocking its return value.
        source.stop = lambda: ProgressiveCastSource.stop(source)
        self.assertTrue(m.start("airplay", source, 7, "tv", "TV"))
        self.assertIs(m.active["source"], source)
        self.assertFalse(m.active["finished_file"])
        self.assertEqual(m.active["plan"].fps, 60)
        source.playback.stop_feeder.assert_called_once()
        m._ipc.assert_any_call("set", "user-data/oh/casting", "1")
        m.stop()
        m._ipc.assert_any_call("command", "script-message", "oh-restart", "14.500", "1")
        vod.close.assert_called_once()

    def test_preparation_waits_for_queue_instead_of_timing_out_into_airplay(self):
        cls = manager_type()
        m = object.__new__(cls)
        m.events, m.caster = queue.Queue(), None
        m._ipc, m._osd, m.publish_menu, m._caster = Mock(), Mock(), Mock(), Mock()
        vod = Mock(last_error=None)
        vod.is_cached.return_value = False
        source = SimpleNamespace(is_progressive_cast=True, waiting_k=3, playback=SimpleNamespace(hold="paused"))
        m.active = dict(mode="airplay", phase="prepare", vod=vod, source=source, start_k=3,
                        plan=H.SegmentPlan(30, 600, 64, 48), t_start=time.monotonic()-100,
                        v0=0, local_pos=12, device="TV")
        m.poll()
        m._caster.assert_not_called()
        self.assertEqual(m.active["phase"], "prepare")
        self.assertEqual(m.status_text(), "背景處理已暫停")


if __name__ == "__main__":
    unittest.main()
