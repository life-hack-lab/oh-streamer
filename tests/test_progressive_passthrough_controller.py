"""Seek passthrough routing and handoff races, without mpv or restoration models."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_player_controller import controller_args, controller_deps, FakeEngine, FakeIpc
from oh_streamer.player.controller import PlayerController


class PassthroughControllerTest(unittest.TestCase):
    def controller(self):
        c = PlayerController(controller_args(), FakeEngine(), deps=controller_deps())
        c.ipc = FakeIpc([])
        c.ipc.command = Mock()
        c.ipc.props.update({"pause": False, "core-idle": False, "time-pos": 612.0,
                            "path": "http://127.0.0.1:1/source.mp4"})
        p = SimpleNamespace(
            po=SimpleNamespace(source_url="http://127.0.0.1:1/source.mp4", path="source.rcache",
                               queue_id="job", has_audio=True, pressed_at=None),
            feeder=None, target=600.0, hold=None,
            segment_available=Mock(return_value=False),
            restore_range=Mock(return_value=(600.0, 640.0)),
            start_source=Mock(return_value="http://127.0.0.1:1/source.mp4"),
            start=Mock(), prepare_restore=Mock(), activate_restore=Mock(),
            tick=Mock(), ranges_text=Mock(return_value="600.000-640.000"),
            clock_due=Mock(return_value=True), clock=Mock())
        c.progressive, c.cast, c.link = p, SimpleNamespace(active=None), Mock()
        p.activate_restore.side_effect = lambda pos: setattr(p, "feeder", p.feeder.continuation)
        c.epoch, c.loaded, c.resume_wait = 1, True, False
        c.progressive_source, c.progressive_active_url = True, p.po.source_url
        c.progressive_next, c.progressive_sent, c.ranges_shown = 0.0, False, None
        c.last_sreq, c.last_qreq, c.last_creq = None, None, None
        c.served, c.t_req, c.published_duration = "1:600.000", 0.0, 1000.0
        c.thumbs, c.prefetch_started = None, False
        c.set_cache_wait = Mock()
        c.publish_compare_layers = Mock()
        c.remember_progressive_subtitle = Mock()
        c.load_progressive_subtitles = Mock()
        c.osd = Mock()
        return c

    def candidate(self, c, *, prepared=True):
        event = threading.Event()
        if prepared:
            event.set()
        f = SimpleNamespace(url="repaired-startup.mkv", target=610.0, first_s=608.0, end_s=616.0,
                            continuation=None,
                            error=None, retire=Mock(), prepared=event,
                            compare_titles=lambda: "", derive_1x=lambda: "", ended=False)
        c.progressive.feeder = f
        return f

    def test_unprocessed_seek_loads_original_without_pausing_for_worker(self):
        c = self.controller()
        c.progressive_requests("2:600.000", None, None)
        c.progressive.start_source.assert_called_once_with(600.0)
        c.progressive.start.assert_not_called()
        self.assertFalse(c.ipc.get("pause"))
        c.ipc.command.assert_called_once_with("loadfile", c.progressive.po.source_url,
                                             "replace", -1, "start=600.000000")
        self.assertEqual(c.ipc.get("user-data/oh/served"), "2:600.000")
        self.assertEqual(c.ipc.get("user-data/oh/passthrough"), "1")

    def test_processed_seek_keeps_existing_feeder_path(self):
        c = self.controller()
        c.progressive.segment_available.return_value = True
        c.progressive_requests("2:600.000", None, None)
        c.progressive.start.assert_called_once_with(600.0)
        c.progressive.start_source.assert_not_called()
        self.assertTrue(c.ipc.get("pause"))
        self.assertFalse(c.progressive_source)

    def test_old_handoff_invalidated_by_new_user_seek(self):
        c = self.controller()
        c.progressive_begin_handoff("tcp://old", 612, 10)
        generation = c.progressive_generation
        c.progressive_requests("2:800.000", None, None)
        self.assertGreater(c.progressive_generation, generation)
        self.assertIsNone(c.progressive_handoff)
        self.assertEqual(c.ipc.get("user-data/oh/handoff"), "")

    def test_candidate_uses_current_position_and_does_not_request_old_target(self):
        c = self.controller()
        c.progressive_source_tick(10)
        c.progressive.prepare_restore.assert_called_once_with(612.0)
        c.progressive.start.assert_not_called()
        c.progressive.start_source.assert_not_called()

    def test_only_prepared_candidate_can_replace_original(self):
        c = self.controller()
        self.candidate(c, prepared=False)
        c.progressive_source_tick(10)
        c.ipc.command.assert_not_called()

    def test_handoff_uses_latest_cursor_and_independent_ack(self):
        c = self.controller()
        f = self.candidate(c)
        c.progressive_source_tick(10)
        c.ipc.command.assert_called_once_with("loadfile", f.url, "replace", -1, "start=612.000000")
        handoff = json.loads(c.ipc.get("user-data/oh/handoff"))
        self.assertEqual(handoff["position"], 612)
        self.assertEqual(c.served, "1:600.000")
        c.ipc.props.update({"path": f.url, "user-data/oh/handoff-resumed": handoff["token"]})
        c.progressive_source_tick(11)
        self.assertFalse(c.progressive_source)
        self.assertFalse(c.resume_wait)
        self.assertIsNone(c.progressive_handoff)

    def test_wrong_url_or_token_cannot_complete_handoff(self):
        c = self.controller()
        self.candidate(c)
        c.progressive_source_tick(10)
        c.ipc.props["user-data/oh/handoff-resumed"] = c.progressive_handoff["token"]
        c.progressive_source_tick(11)
        self.assertTrue(c.progressive_source)  # path is still the original
        self.assertIsNotNone(c.progressive_handoff)

    def test_native_seek_before_clip_load_cancels_candidate_and_keeps_new_cursor(self):
        c = self.controller()
        clip = self.candidate(c)
        c.progressive_source_tick(10)
        c.ipc.props["user-data/oh/handoff"] = ""
        c.ipc.props["time-pos"] = 720.0  # source cache seek while the clip load was being prepared
        c.progressive_check_handoff(11)
        clip.retire.assert_called_once()
        self.assertEqual(c.progressive_handoff["kind"], "source")
        c.ipc.command.assert_called_with("loadfile", c.progressive.po.source_url,
                                        "replace", -1, "start=720.000000")

    def test_new_user_request_cancels_handoff_without_recovering_old_position(self):
        c = self.controller()
        self.candidate(c)
        c.progressive_source_tick(10)
        c.ipc.command.reset_mock()
        c.ipc.props.update({"user-data/oh/handoff": "", "user-data/oh/seek-request": "5:900.000"})
        c.progressive_check_handoff(11)
        self.assertIsNone(c.progressive_handoff)
        c.ipc.command.assert_not_called()

    def test_old_generation_cannot_switch_source(self):
        c = self.controller()
        self.candidate(c)
        c.progressive_source_tick(10)
        c.progressive_generation += 1
        c.ipc.props.update({"path": c.progressive_handoff["url"],
                            "user-data/oh/handoff-resumed": c.progressive_handoff["token"]})
        c.progressive_source_tick(11)
        self.assertTrue(c.progressive_source)

    def test_native_seek_backwards_retires_candidate_without_rewind(self):
        c = self.controller()
        f = self.candidate(c)
        c.ipc.props["time-pos"] = 603.0
        c.progressive_source_tick(10)
        f.retire.assert_called_once()
        self.assertIsNone(c.progressive.feeder)
        c.ipc.command.assert_not_called()

    def test_gap_or_slow_worker_keeps_original_without_restarts(self):
        c = self.controller()
        c.progressive.restore_range.return_value = None
        for now in range(10, 20):
            c.progressive_source_tick(now)
        c.progressive.prepare_restore.assert_not_called()
        c.progressive.start_source.assert_not_called()
        c.ipc.command.assert_not_called()

    def test_paused_viewer_never_auto_switches(self):
        c = self.controller()
        self.candidate(c)
        c.ipc.props["pause"] = True
        c.progressive_source_tick(10)
        c.ipc.command.assert_not_called()

    def test_candidate_failure_is_nonblocking_and_preserves_original(self):
        c = self.controller()
        f = self.candidate(c)
        f.error = OSError("input vanished")
        c.progressive_source_tick(10)
        f.retire.assert_called_once()
        self.assertEqual(c.progressive_active_url, c.progressive.po.source_url)
        c.ipc.command.assert_not_called()

    def test_candidate_file_creation_failure_keeps_original_and_backs_off(self):
        c = self.controller()
        c.progressive.prepare_restore.side_effect = OSError("disk full")
        c.progressive_source_tick(10)
        self.assertEqual(c.progressive_active_url, c.progressive.po.source_url)
        self.assertTrue(c.progressive_source)
        self.assertGreater(c.progressive_retry_at, 10)
        c.ipc.command.assert_not_called()

    def test_cast_source_construction_failure_keeps_local_clip_and_continuation(self):
        c = self.controller()
        clip = self.candidate(c)
        c.progressive_source, c.progressive_clip = False, clip
        with patch("oh_streamer.player.progressive_cast.ProgressiveCastSource", side_effect=OSError("probe failed")):
            with self.assertRaisesRegex(OSError, "probe failed"):
                c.cast_source("hls")
        clip.retire.assert_not_called()
        self.assertIs(c.progressive_clip, clip)

    def test_candidate_timeout_retires_without_stopping_original(self):
        c = self.controller()
        f = self.candidate(c, prepared=False)
        c.progressive_candidate_at = 1.0
        c.progressive_source_tick(12)
        f.retire.assert_called_once()
        c.ipc.command.assert_not_called()

    def test_candidate_already_played_past_clip_end_is_cancelled(self):
        c = self.controller()
        f = self.candidate(c)
        c.ipc.props["time-pos"] = f.end_s
        c.progressive_source_tick(10)
        f.retire.assert_called_once()
        c.ipc.command.assert_not_called()

    def test_startup_clip_appends_continuation_at_exact_boundary(self):
        c = self.controller()
        clip = self.candidate(c)
        ready = threading.Event()
        ready.set()
        live = SimpleNamespace(url="tcp://127.0.0.1:1234", ready=ready, error=None, retire=Mock(), ended=False)
        clip.continuation = live
        c.progressive_source_tick(10)
        handoff = c.progressive_handoff
        c.ipc.props.update({"path": clip.url, "user-data/oh/handoff-resumed": handoff["token"]})
        c.progressive_source_tick(11)
        self.assertIs(c.progressive.feeder, live)
        self.assertIs(c.progressive_clip, clip)
        c.progressive_continuation_tick(11)
        self.assertEqual(c.progressive_handoff["kind"], "continuation")
        self.assertEqual(c.progressive_handoff["position"], clip.end_s)
        c.ipc.command.assert_called_with("loadfile", live.url, "append")
        self.assertEqual(c.progressive_active_url, clip.url)
        token = c.progressive_handoff["token"]
        c.ipc.props.update({"path": live.url, "user-data/oh/handoff-resumed": token})
        c.progressive_check_handoff(12)
        self.assertEqual(c.progressive_active_url, live.url)
        clip.retire.assert_called_once_with(keep_continuation=True)
        self.assertIsNone(c.progressive_clip)
        live.retire.assert_not_called()

    def test_paused_startup_clip_does_not_time_out_appended_continuation(self):
        c = self.controller()
        c.progressive_source = False
        c.progressive_begin_handoff("tcp://next", 616, 10, "continuation")
        c.ipc.props["pause"] = True
        c.progressive_check_handoff(26)
        self.assertGreater(c.progressive_handoff["deadline"], 26)
        c.ipc.command.assert_not_called()

    def test_keep_open_pause_at_clip_eof_advances_and_does_not_become_a_user_pause(self):
        c = self.controller()
        clip = self.candidate(c)
        ready = threading.Event()
        ready.set()
        live = SimpleNamespace(url="tcp://next", ready=ready, error=None)
        c.progressive.feeder, c.progressive_clip, c.progressive_source = live, clip, False
        c.ipc.props.update({"path": clip.url, "pause": True, "eof-reached": True})
        c.progressive_continuation_tick(10)
        self.assertFalse(c.progressive_handoff["paused"])
        c.ipc.command.assert_any_call("loadfile", live.url, "append")
        c.ipc.command.assert_any_call("playlist-next")
        c.ipc.command.reset_mock()
        c.progressive_check_handoff(11)
        c.ipc.command.assert_not_called()  # never skip twice while file-loaded is still in flight

    def test_continuation_watchdog_leaves_a_clip_that_reached_eof_after_append(self):
        c = self.controller()
        clip = self.candidate(c)
        c.progressive_clip, c.progressive_source = clip, False
        c.ipc.props.update({"path": clip.url, "pause": True, "eof-reached": True})
        c.progressive_begin_handoff("tcp://next", clip.end_s, 10, "continuation", paused=False)
        c.progressive_check_handoff(11)
        c.ipc.command.assert_called_once_with("playlist-next")

    def test_native_seek_inside_clip_rearms_existing_appended_continuation(self):
        c = self.controller()
        clip = self.candidate(c)
        c.progressive_clip, c.progressive_source = clip, False
        c.ipc.props["path"] = clip.url
        c.progressive_begin_handoff("tcp://next", clip.end_s, 10, "continuation")
        old = c.progressive_handoff["token"]
        c.ipc.props["user-data/oh/handoff"] = ""
        c.progressive_check_handoff(11)
        self.assertEqual(c.progressive_handoff["kind"], "continuation")
        self.assertNotEqual(c.progressive_handoff["token"], old)
        c.ipc.command.assert_not_called()  # do not append the same stream twice

    def test_clip_eof_is_not_the_end_of_the_video(self):
        c = self.controller()
        clip = self.candidate(c)
        ready = threading.Event()
        ready.set()
        live = SimpleNamespace(url="tcp://next", ended=True, ready=ready, error=None)
        c.progressive.feeder, c.progressive_clip = live, clip
        c.progressive_active_url, c.progressive_source = clip.url, False
        c.ipc.props.update({"path": clip.url, "eof-reached": True})
        c.progressive_begin_handoff(live.url, clip.end_s, 10, "continuation")
        watched = Mock(return_value=False)
        c._deps = controller_deps(processed_watched=watched)
        c.progressive_tick(11)
        watched.assert_called_once_with(612.0, 1000.0, False)

    def test_failed_handoff_recovers_original_at_switch_position(self):
        c = self.controller()
        f = self.candidate(c)
        c.progressive_source_tick(10)
        c.progressive_source_tick(26)
        f.retire.assert_called_once()
        self.assertEqual(c.progressive_handoff["kind"], "source")
        c.ipc.command.assert_called_with("loadfile", c.progressive.po.source_url,
                                        "replace", -1, "start=612.000000")
        c.progressive.start_source.assert_not_called()  # no foreground goto on recovery

    def test_original_sends_real_clock_and_loads_subtitles(self):
        c = self.controller()
        c.progressive.restore_range.return_value = None
        c.ipc.props["track-list"] = [{"type": "video", "id": 1}]
        c.progressive_tick(10)
        c.progressive.clock.assert_called_once_with(612.0, True, subtitle_pos=612.0)
        c.load_progressive_subtitles.assert_called_once_with(c.progressive.po.source_url)

    def test_original_reaching_eof_can_mark_watched(self):
        c = self.controller()
        c.progressive.restore_range.return_value = None
        c.ipc.props["eof-reached"] = True
        watched = Mock(return_value=True)
        c._deps = controller_deps(processed_watched=watched)
        c.progressive_tick(10)
        watched.assert_called_once_with(612.0, 1000.0, True)
        c.link.post.assert_called_once_with({"cmd": "watched", "id": "job"})

    def test_paused_original_reports_a_stationary_clock_even_if_core_is_active(self):
        c = self.controller()
        c.ipc.props["pause"] = True
        c.progressive_tick(10)
        c.progressive.clock.assert_called_once_with(612.0, False, subtitle_pos=612.0)


if __name__ == "__main__":
    unittest.main()
