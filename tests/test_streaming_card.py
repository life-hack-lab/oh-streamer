"""T3.5: the 「串流中」 card. The live-mode player adds a "live" report to its player_busy heartbeat (queue_ipc.py, "Live
stream report"); the queue turns it into get_state()["streaming"] (title, position / length, restore lead and speed, the
stream item's subtitle view). Run from streamer/: CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_streaming_card."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import unittest
import threading
from types import SimpleNamespace

try:
    from tests.test_queue_service import ServiceTestBase
    from tests.test_player_controller import FakeIpc
except ImportError:
    from test_queue_service import ServiceTestBase
    from test_player_controller import FakeIpc

from oh_streamer.player import controller as C
from oh_streamer.player.queue_bridge import BusyReporter
from oh_streamer.queue import queue_ipc, queue_store


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------------------------------------------
# player side: BusyReporter carries the report
# ---------------------------------------------------------------------------------------------------------------

class BusyReporterLiveTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.sent, self.report = Clock(), [], None
        self.rep = BusyReporter(lambda busy, **extra: self.sent.append((busy, extra)), self.clock, interval=10.0,
                                live_interval=2.0)
        self.rep.live = lambda: self.report

    def test_report_goes_out_every_live_interval_busy_or_not(self):
        self.report = {"title": "t"}
        self.rep.update(False)
        self.assertEqual(self.sent, [(False, {"live": {"title": "t"}})])
        self.clock.t += 1.9
        self.rep.update(False)
        self.assertEqual(len(self.sent), 1)                       # not due yet
        self.clock.t += 0.2
        self.rep.update(False)
        self.assertEqual(len(self.sent), 2)

    def test_busy_heartbeat_between_reports_has_no_live_key(self):
        self.report = {"title": "t"}
        self.rep.update(True)                                     # busy start + the first report in one message
        self.assertEqual(self.sent, [(True, {"live": {"title": "t"}})])
        self.clock.t += 2.1
        self.rep.update(True)
        self.assertEqual(self.sent[-1], (True, {"live": {"title": "t"}}))
        self.rep.live = None                                      # reports off: a plain heartbeat has no key at all
        self.clock.t += 10
        self.rep.update(True)
        self.assertEqual(self.sent[-1], (True, {}))

    def test_a_vanished_report_is_cleared_once(self):
        self.report = {"title": "t"}
        self.rep.update(False)
        self.report = None
        self.clock.t += 2.1
        self.rep.update(False)
        self.assertEqual(self.sent[-1], (False, {"live": None}))
        self.clock.t += 2.1
        self.rep.update(False)
        self.assertEqual(len(self.sent), 2)                       # nothing more once cleared

    def test_without_a_report_it_behaves_as_before(self):
        for _ in range(3):
            self.rep.update(False)
            self.clock.t += 5
        self.assertEqual(self.sent, [])
        self.rep.update(True)
        self.rep.update(False)
        self.assertEqual(self.sent, [(True, {}), (False, {})])

    def test_final_clear_after_next_live_reset(self):
        self.report = {"title": "t"}
        self.rep.update(False)
        self.report = None
        self.rep.next_live = 0.0                                  # what the controller does on exit
        self.rep.update(False)
        self.assertEqual(self.sent[-1], (False, {"live": None}))


# ---------------------------------------------------------------------------------------------------------------
# player side: the controller's report
# ---------------------------------------------------------------------------------------------------------------

def make_controller(**over):
    ipc = FakeIpc([])
    ipc.props.update({"time-pos": 125.5, "demuxer-cache-time": 170.5, "pause": False, "paused-for-cache": False})
    session = SimpleNamespace(fps=30.0, processing_done=lambda: False)
    values = dict(session=session, processed=None, progressive=None, loaded=True, resume_wait=False,
                  cast=SimpleNamespace(active=None), ipc=ipc, title="示範影片", published_duration=3600.0,
                  proc_fps_now=21.0, stream_item="abc", _deps=SimpleNamespace(log=lambda _m: None))
    values.update(over)
    return SimpleNamespace(**values)


def report(**over):
    return C.PlayerController.live_report(make_controller(**over))


class ControllerLiveReportTests(unittest.TestCase):
    def test_playing_report(self):
        self.assertEqual(report(), {
            "title": "示範影片", "pos": 125.5, "duration": 3600.0, "playing": True, "buffering": False,
            "ahead_s": 45.0, "speed": 0.7, "restore_done": False, "stream_item": "abc"})

    def test_no_stream_item_means_none_not_missing(self):
        self.assertIsNone(report(stream_item=None)["stream_item"])

    def test_paused_and_cache_wait(self):
        ipc = make_controller().ipc
        ipc.props["pause"] = True
        self.assertEqual((report(ipc=ipc)["playing"], report(ipc=ipc)["buffering"]), (False, False))
        ipc.props["pause"], ipc.props["paused-for-cache"] = False, True
        self.assertEqual((report(ipc=ipc)["playing"], report(ipc=ipc)["buffering"]), (False, True))

    def test_not_loaded_yet_has_a_title_but_no_position(self):
        r = report(loaded=False)
        self.assertEqual((r["title"], r["pos"], r["ahead_s"], r["buffering"], r["restore_done"]),
                         ("示範影片", None, None, True, False))

    def test_restore_done_has_no_speed(self):
        r = report(session=SimpleNamespace(fps=30.0, processing_done=lambda: True))
        self.assertEqual((r["restore_done"], r["speed"]), (True, None))

    def test_speed_unknown_until_measured_and_ahead_never_negative(self):
        self.assertIsNone(report(proc_fps_now=None)["speed"])
        ipc = make_controller().ipc
        ipc.props["demuxer-cache-time"] = 100.0
        self.assertEqual(report(ipc=ipc)["ahead_s"], 0.0)
        ipc.props["demuxer-cache-time"] = None
        self.assertIsNone(report(ipc=ipc)["ahead_s"])

    def test_source_switch_does_not_report_the_previous_restore_speed(self):
        source = SimpleNamespace(describe=lambda: "new source")
        player = SimpleNamespace(
            _deps=SimpleNamespace(time=SimpleNamespace(perf_counter=lambda: 10.0),
                                  device_backend=SimpleNamespace(allocated_gib=lambda _device: 0.0),
                                  threading=threading, log=lambda _message: None),
            engine=SimpleNamespace(device=None), epoch=0, reload_pending=False,
            leave_current=lambda: 0.1, apply_overrides=lambda _overrides: None,
            begin_source=lambda opened: setattr(player, "session", SimpleNamespace(source=opened.source)),
            publish_source=lambda: None, req_kind="old", last_latency=1.0, last_kind="old",
            proc_samples=[], proc_fps_now=21.0, hist=SimpleNamespace(clear=lambda *_keys: None))

        C.PlayerController.switch_to(player, SimpleNamespace(source=source, title="new", overrides=None))

        self.assertIsNone(player.proc_fps_now)

    def test_unknown_duration_is_none(self):
        self.assertIsNone(report(published_duration=0.0)["duration"])

    def test_gone_when_not_live_mode_or_at_the_end(self):
        self.assertIsNone(report(session=None))
        self.assertIsNone(report(processed=object()))
        self.assertIsNone(report(progressive=object()))
        ipc = make_controller().ipc
        ipc.props["eof-reached"] = True
        self.assertIsNone(report(ipc=ipc))

    def test_casting_has_no_position(self):
        r = report(cast=SimpleNamespace(active={"mode": "airplay"}))
        self.assertEqual((r["pos"], r["ahead_s"], r["playing"]), (None, None, False))

    def test_an_mpv_error_is_swallowed(self):
        class Broken(FakeIpc):
            def get(self, name, default=None):
                raise OSError("mpv gone")
        logs = []
        self.assertIsNone(report(ipc=Broken([]), _deps=SimpleNamespace(log=logs.append)))
        self.assertTrue(logs)


# ---------------------------------------------------------------------------------------------------------------
# queue side: get_state()["streaming"]
# ---------------------------------------------------------------------------------------------------------------

LIVE = {"title": "示範影片", "pos": 125.5, "duration": 3600.0, "playing": True, "buffering": False,
        "ahead_s": 45.0, "speed": 0.7, "restore_done": False, "stream_item": None}


class StreamingStateTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc.pause_all()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        self.svc._subtitle_view_tick = lambda: None           # the views here are set by hand (no slice table behind them)

    def beat(self, live=LIVE, busy=True, **extra):
        msg = {"cmd": "player_busy", "busy": busy, "pid": 1, **extra}
        if live is not False:
            msg["live"] = live
        return self.svc._on_message(msg)

    def streaming(self):
        return self.svc.get_state()["streaming"]

    def stream_item(self, subtitles=True):
        settings = {"subtitles": True} if subtitles else {}
        return self.svc.store.add_stream("https://x.test/v", "url", {}, settings, title="Clip")

    def test_no_report_no_card(self):
        self.assertIsNone(self.streaming())
        self.beat(live=False)                                           # an ordinary heartbeat
        self.assertIsNone(self.streaming())

    def test_stream_without_subtitles_still_shows_playback(self):
        self.assertEqual(self.beat(), {"ok": True})
        s = self.streaming()
        self.assertEqual((s["title"], s["pos"], s["duration"], s["ahead_s"], s["speed"], s["playing"]),
                         ("示範影片", 125.5, 3600.0, 45.0, 0.7, True))
        self.assertIsNone(s["subtitle"])
        self.assertIsNone(s["stream_item"])

    def test_stream_with_subtitles_carries_the_view_and_the_lead(self):
        item = self.stream_item()
        view = {"v": 2, "state": "running", "covered_s": 845.5, "progress": 0.4, "running": {"executor": "cloud"}}
        self.svc._subtitle_update(item.id, view=view, audio_offset=0.0)
        self.beat({**LIVE, "stream_item": item.id})
        sub = self.streaming()["subtitle"]
        self.assertEqual(sub["view"]["state"], "running")
        self.assertEqual(sub["lead_s"], 720.0)                          # 845.5 - 125.5 s = 12 minutes
        self.assertEqual(self.streaming()["stream_item"], item.id)

    def test_lead_uses_audio_offset(self):
        item = self.stream_item()
        self.svc._subtitle_update(item.id, view={"v": 2, "state": "queued", "covered_s": 100.0}, audio_offset=5.0)
        self.beat({**LIVE, "stream_item": item.id, "pos": 20.0})
        self.assertEqual(self.streaming()["subtitle"]["lead_s"], 85.0)  # audio clock = 20 - 5

    def test_subtitles_on_but_no_view_yet(self):
        item = self.stream_item()
        self.beat({**LIVE, "stream_item": item.id})
        sub = self.streaming()["subtitle"]
        self.assertIsNone(sub["view"])
        self.assertIsNone(sub["lead_s"])
        self.assertEqual(sub["status"], "queued")

    def test_stream_item_without_subtitle_setting_or_unknown_or_regular_gives_no_subtitle(self):
        plain = self.stream_item(subtitles=False)
        regular = self.add("http://h/a")
        for item_id in (plain.id, "nope", regular):
            self.beat({**LIVE, "stream_item": item_id})
            self.assertIsNone(self.streaming()["subtitle"], item_id)

    def test_a_heartbeat_without_live_keeps_the_card_and_null_clears_it(self):
        self.beat()
        self.beat(live=False)
        self.assertIsNotNone(self.streaming())
        self.beat(live=None, busy=False)                                # playback ended / closed
        self.assertIsNone(self.streaming())

    def test_card_survives_restore_finishing(self):
        self.beat()
        self.beat({**LIVE, "restore_done": True, "speed": None, "ahead_s": 300.0}, busy=False)
        s = self.streaming()
        self.assertEqual((s["restore_done"], s["speed"], s["ahead_s"]), (True, None, 300.0))
        self.assertFalse(self.svc.is_player_busy())

    def test_card_vanishes_when_the_reports_stop(self):
        self.beat()
        self.now[0] += self.svc.busy_expire - 1
        self.assertIsNotNone(self.streaming())
        self.now[0] += 2
        self.assertIsNone(self.streaming())

    def test_a_fresh_report_renews_the_card(self):
        self.beat()
        self.now[0] += self.svc.busy_expire - 1
        self.beat()
        self.now[0] += self.svc.busy_expire - 1
        self.assertIsNotNone(self.streaming())

    def test_junk_fields_are_cleaned_and_junk_live_clears(self):
        self.beat({"title": 5, "pos": "x", "duration": -1, "ahead_s": float("nan"), "speed": True,
                   "playing": "yes", "stream_item": 7})
        s = self.streaming()
        self.assertEqual((s["title"], s["pos"], s["duration"], s["ahead_s"], s["speed"], s["playing"], s["stream_item"]),
                         ("5", None, None, None, None, False, None))
        self.beat(live="junk")
        self.assertIsNone(self.streaming())

    def test_live_does_not_change_the_busy_rules(self):
        self.beat(busy=False)
        self.assertFalse(self.svc.is_player_busy())
        self.beat(busy=True)
        self.assertTrue(self.svc.is_player_busy())

    def test_stream_end_does_not_clear_the_card_by_itself(self):
        item = self.stream_item()
        self.beat({**LIVE, "stream_item": item.id})
        self.svc._on_message({"cmd": "stream_end", "id": item.id})
        s = self.streaming()
        self.assertIsNotNone(s)
        self.assertIsNone(s["subtitle"])                                # the item is gone; the playback info stays


if __name__ == "__main__":
    unittest.main()
