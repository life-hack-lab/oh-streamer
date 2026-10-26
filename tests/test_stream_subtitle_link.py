"""T2.4: the live player reports position to a stream item (watching / clock / want) and the queue records it for the
subtitle scheduling only. Run from streamer/: CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_subtitle_link."""
import dataclasses
import json
import threading
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest import mock

try:                                    # run_tests.py imports tests.* ; "-s tests" imports flat names
    from tests.test_queue_service import ServiceTestBase
    from tests.test_player_controller import (
        FakeCast, FakeEngine, FakeIpc, FakeSession, controller_args, controller_deps)
except ImportError:
    from test_queue_service import ServiceTestBase
    from test_player_controller import FakeCast, FakeEngine, FakeIpc, FakeSession, controller_args, controller_deps

from oh_streamer.player import controller as C
from oh_streamer.player import opening
from oh_streamer.player.stream_subtitle_link import StreamSubtitleLink
from oh_streamer.queue import queue_store


# ---------------------------------------------------------------------------------------------------------------
# player side
# ---------------------------------------------------------------------------------------------------------------

class StreamSubtitleLinkTests(unittest.TestCase):
    def setUp(self):
        self.sent, self.logs = [], []
        self.link = StreamSubtitleLink("abc", lambda p: self.sent.append(p) or True, self.logs.append)

    def test_watching_every_ten_seconds(self):
        self.assertTrue(self.link.tick(100.0))
        self.assertFalse(self.link.tick(105.0))
        self.assertTrue(self.link.tick(110.0))
        self.assertEqual(self.sent, [{"cmd": "watching", "id": "abc"}] * 2)

    def test_clock_is_throttled_and_carries_position_and_playing(self):
        self.assertTrue(self.link.clock(12.34567, True, 50.0))
        self.assertFalse(self.link.clock_due(50.5))
        self.assertFalse(self.link.clock(13.0, True, 50.5))
        self.assertTrue(self.link.clock(13.5, False, 51.0))
        self.assertEqual(self.sent, [{"cmd": "clock", "id": "abc", "pos": 12.346, "playing": True},
                                     {"cmd": "clock", "id": "abc", "pos": 13.5, "playing": False}])

    def test_clock_ignores_invalid_positions(self):
        for bad in (None, True, "3", -1.0, float("nan"), float("inf")):
            self.assertFalse(self.link.clock(bad, True, 1.0), bad)
        self.assertEqual(self.sent, [])

    def test_want_has_position_no_k_and_lets_the_next_clock_through(self):
        self.link.clock(1.0, True, 10.0)
        self.assertTrue(self.link.want(600.25, 10.2))
        self.assertEqual(self.sent[-1], {"cmd": "want", "id": "abc", "pos": 600.25, "foreground": True})
        self.assertNotIn("k", self.sent[-1])
        self.assertTrue(self.link.clock_due(10.2))
        self.assertFalse(self.link.want(-5, 11.0))
        self.assertFalse(self.link.want(None, 11.0))

    def test_a_failing_sender_never_raises(self):
        def boom(_payload):
            raise OSError("queue gone")
        link = StreamSubtitleLink("abc", boom, self.logs.append)
        self.assertTrue(link.tick(0.0))
        self.assertTrue(any("failed" in line for line in self.logs))

    def test_full_backlog_is_logged(self):
        link = StreamSubtitleLink("abc", lambda _p: False, self.logs.append)
        link.tick(0.0)
        self.assertTrue(any("dropped" in line for line in self.logs))


class OpenerStreamItemTests(unittest.TestCase):
    def make(self):
        opened = opening.Opened(input="u")
        opener = opening.Opener(SimpleNamespace(), None, open_source=lambda *a, **k: opened, log=lambda _m: None)
        return opener

    def poll(self, opener):
        for _ in range(200):
            done = opener.poll()
            if done is not None:
                return done[0]
            threading.Event().wait(0.01)
        self.fail("opener did not finish")

    def test_stream_item_comes_back_on_the_opened_source(self):
        opener = self.make()
        opener.submit(SimpleNamespace(kind="url", target="u", headers={}), "r", 0.0, stream_item="abc")
        self.assertEqual(self.poll(opener).stream_item, "abc")

    def test_plain_submit_has_none(self):
        opener = self.make()
        opener.submit(SimpleNamespace(kind="url", target="u", headers={}), "r", 0.0)
        self.assertIsNone(self.poll(opener).stream_item)


class RecordingOpener:
    def __init__(self, *_a):
        self.calls, self.busy = [], False

    def submit(self, item, label, start, **extra):
        self.calls.append((label, start, extra))

    def poll(self):
        return None


class PlayerSide(unittest.TestCase):
    def make_player(self, *, link=True):
        self.posted, self.logs = [], []
        events = []

        class Link:
            def post(inner, payload, *_a, **_k):
                self.posted.append(payload)
                return True

            def close(inner):
                pass

        deps = controller_deps(events, QueueLink=lambda *a, **k: Link(), Opener=RecordingOpener,
                               log=self.logs.append)
        player = C.PlayerController(controller_args(), FakeEngine(), deps=deps)
        player.link = Link() if link else None
        player.opener = RecordingOpener()
        player.ipc, player.cast, player.loaded, player.resume_wait = FakeIpc(events), FakeCast(events), True, False
        player.session = None
        player.tail = player.progressive = player.thumbs = player.processed = None
        player.last_sreq = player.last_qreq = player.last_creq = None
        player.t_req = player.stop_s = 0.0
        player.n_req, player.switch_ctx, player.restore_deadline, player.sw_note, player.req_kind = 0, None, None, "", ""
        return player

    # --- the open message -----------------------------------------------------------------------------------------
    def test_open_message_stream_item_is_queued_with_the_input(self):
        player = self.make_player()
        player.inbox = __import__("queue").Queue()
        self.assertEqual(player.on_message({"cmd": "open", "input": "https://x.test/v", "stream_item": "abc12345"}),
                         {"ok": True})
        self.assertEqual(player.inbox.get_nowait()[6], "abc12345")

    def test_open_message_without_or_with_bad_stream_item_has_none(self):
        player = self.make_player()
        player.inbox = __import__("queue").Queue()
        for extra in ({}, {"stream_item": ""}, {"stream_item": 7}, {"stream_item": None}, {"stream_item": ["a"]}):
            player.on_message({"cmd": "open", "input": "https://x.test/v", **extra})
            self.assertIsNone(player.inbox.get_nowait()[6], extra)

    def test_submit_text_passes_stream_item_only_when_present(self):
        player = self.make_player()
        player.submit_text("https://x.test/v", {}, "轉送", 0.0, 720, {}, "abc")
        player.submit_text("https://x.test/v", {}, "轉送", 0.0, None, {}, None)
        self.assertEqual(player.opener.calls, [("轉送", 0.0, {"quality": 720, "stream_item": "abc"}), ("轉送", 0.0, {})])

    # --- source lifecycle -----------------------------------------------------------------------------------------
    def begin(self, player, **extra):
        source = SimpleNamespace(meta=SimpleNamespace(duration=30.0, video_fps_exact=30))
        opened = SimpleNamespace(input="clip", user_headers={}, source=source, quals=[{"id": "s"}],
                                 chosen={"id": "s", "label": "1080p"}, refresh=None, title="clip", source_text="clip",
                                 start=0.0, thumb_src=None, **extra)
        player.session = None
        player.epoch = 0
        player.sc = player.engine.sc
        player.begin_source(opened)

    def test_begin_source_remembers_the_stream_item_and_a_new_source_without_it_clears_it(self):
        player = self.make_player()
        self.begin(player, stream_item="abc")
        self.assertEqual((player.stream_item, player.stream_link.item_id), ("abc", "abc"))
        player.end_source()                                   # what leave_current does before the next source
        self.assertEqual((player.stream_item, player.stream_link), (None, None))
        self.begin(player, stream_item="abc")
        self.begin(player)                                    # next source: an open without stream_item
        self.assertEqual((player.stream_item, player.stream_link), (None, None))

    def test_no_queue_means_no_reporting(self):
        player = self.make_player(link=False)
        self.begin(player, stream_item="abc")
        self.assertIsNone(player.stream_link)

    def test_starting_past_zero_sends_a_want(self):
        player = self.make_player()
        source = SimpleNamespace(meta=SimpleNamespace(duration=30.0, video_fps_exact=30))
        opened = SimpleNamespace(input="clip", user_headers={}, source=source, quals=[{"id": "s"}],
                                 chosen={"id": "s", "label": "1080p"}, refresh=None, title="clip", source_text="clip",
                                 start=90.0, thumb_src=None, stream_item="abc")
        player.epoch, player.sc = 0, player.engine.sc
        player.begin_source(opened)
        self.assertEqual(self.posted, [{"cmd": "want", "id": "abc", "pos": 90.0, "foreground": True}])

    # --- the loop tick --------------------------------------------------------------------------------------------
    def ready(self, **props):
        player = self.make_player()
        self.begin(player, stream_item="abc")
        self.posted.clear()
        player.session.url = "http://127.0.0.1/seg"
        player.ipc.props.update({"path": "http://127.0.0.1/seg", "time-pos": 42.5, "core-idle": False, "pause": False,
                                 "eof-reached": False, **props})
        return player

    def test_tick_sends_watching_then_clock_when_playing(self):
        player = self.ready()
        player.stream_link_tick(100.0)
        self.assertEqual(self.posted, [{"cmd": "watching", "id": "abc"},
                                       {"cmd": "clock", "id": "abc", "pos": 42.5, "playing": True}])
        self.posted.clear()
        player.stream_link_tick(100.5)                        # throttled
        self.assertEqual(self.posted, [])
        player.ipc.props["pause"] = True
        player.stream_link_tick(101.0)
        self.assertEqual(self.posted, [{"cmd": "clock", "id": "abc", "pos": 42.5, "playing": False}])

    def test_no_clock_before_the_stream_plays_or_for_another_path(self):
        player = self.ready()
        player.loaded = False
        player.stream_link_tick(100.0)
        player.loaded, player.resume_wait = True, True
        player.stream_link_tick(101.0)
        player.resume_wait = False
        player.ipc.props["path"] = "tail-url"
        player.stream_link_tick(102.0)
        player.ipc.props["path"] = player.session.url
        player.ipc.props["time-pos"] = None
        player.stream_link_tick(103.0)
        self.assertEqual([m["cmd"] for m in self.posted], ["watching"])

    def test_nothing_is_sent_without_a_stream_item(self):
        player = self.make_player()
        self.begin(player)
        player.ipc.props.update({"path": "x", "time-pos": 1.0})
        player.stream_link_tick(100.0)
        player.stream_link_tick(200.0)
        self.assertEqual(self.posted, [])

    # --- seeks ----------------------------------------------------------------------------------------------------
    def seek(self, player, request="5:123.5"):
        player.ipc.props["user-data/oh/seek-request"] = request
        player.ipc.props["user-data/oh/quality-request"] = ""
        player.ipc.props["user-data/oh/settings-request"] = ""
        player.stop_for_request = lambda *_a: 0.0
        player.sc = SimpleNamespace(diff=lambda _w: {}, menu=lambda: [])
        player.session.restorer = object()
        player.session.target = 0.0
        starts = []
        player.session.start = lambda pos, *_a, **_k: starts.append(pos)
        player.args.seek_prebuffer = 0.2
        player.reload_pending = {}
        player.handle_requests(request, "", "")
        return starts

    def test_a_seek_sends_want_with_the_position_before_restarting(self):
        player = self.ready()
        self.assertEqual(self.seek(player), [123.5])
        self.assertEqual(self.posted, [{"cmd": "want", "id": "abc", "pos": 123.5, "foreground": True}])

    def test_a_seek_without_stream_item_sends_nothing(self):
        player = self.make_player()
        self.begin(player)
        player.session.url = "u"
        self.seek(player)
        self.assertEqual(self.posted, [])

    def test_a_settings_restart_is_not_a_seek(self):
        player = self.ready()
        player.ipc.props["user-data/oh/seek-request"] = ""
        player.ipc.props["user-data/oh/settings-request"] = "6:x"
        player.ipc.props["user-data/oh/settings-pos"] = "50"
        player.stop_for_request = lambda *_a: 0.0
        player.sc = SimpleNamespace(diff=lambda _w: {}, menu=lambda: [])
        player.session.restorer = object()
        player.session.start = lambda *_a, **_k: None
        player.reload_pending = {}
        with mock.patch.object(player._deps.PS, "parse_request", create=True, return_value={}):
            player.handle_requests("", "", "6:x")
        self.assertEqual(self.posted, [])
        self.assertEqual(player.stream_item, "abc")           # restarting the segment keeps the stream item

    # --- T2.5: quality / settings restarts keep the item, the link and the clock ---------------------------------
    def restart_ready(self, player):
        """Stubs for handle_requests with a settings / quality request (no real Session or mpv)."""
        player.stop_for_request = lambda *_a: 0.0
        player.reload_pending = {}
        player.session.restorer = object()
        player.session.target = 12.0
        player.session.source = SimpleNamespace()
        player.session.set_source = lambda *_a: None
        player.session.start = lambda *_a, **_k: None
        player.qc = SimpleNamespace(current={"id": "s", "label": "1080p"}, lada=SimpleNamespace(open=lambda _s: object()))
        player.max_frames_for = lambda _s: None

    def quality_switch(self, player, *, seek=""):
        self.restart_ready(player)
        player.sc = SimpleNamespace(diff=lambda _w: {}, menu=lambda: [])
        player.prepare = lambda fid, _state: (
            {"id": fid, "label": "480p"},
            SimpleNamespace(describe=lambda: "480p", meta=SimpleNamespace(video_fps_exact=30, duration=30.0)))
        player.apply_source = lambda q, source, restorer: None
        player.ipc.props["user-data/oh/seek-request"] = seek
        player.ipc.props["user-data/oh/quality-request"] = "7:low"
        player.ipc.props["user-data/oh/quality-pos"] = "40"
        player.ipc.props["user-data/oh/settings-request"] = ""
        player.args.seek_prebuffer = 0.2
        player.handle_requests(seek, "7:low", "")

    def settings_restart(self, player, changes, rebuild=True):
        self.restart_ready(player)
        player.sc = SimpleNamespace(diff=lambda _w: dict(changes), menu=lambda: [])
        player.apply_settings = lambda ch, persist=True: (ch, rebuild)
        player.ipc.props["user-data/oh/seek-request"] = ""
        player.ipc.props["user-data/oh/quality-request"] = ""
        player.ipc.props["user-data/oh/settings-request"] = "6:x"
        player.ipc.props["user-data/oh/settings-pos"] = "50"
        player.args.seek_prebuffer = 0.2
        with mock.patch.object(player._deps.PS, "parse_request", create=True, return_value=dict(changes)), \
                mock.patch.object(player._deps.PS, "RELOAD_KEYS", frozenset(), create=True):
            player.handle_requests("", "", "6:x")

    def segment_loaded(self, player, pos):
        """What the loop does once the restarted segment plays: loaded, resume done, mpv reports `pos`."""
        player.loaded, player.resume_wait = True, False
        player.ipc.props["time-pos"] = pos

    def test_a_quality_switch_keeps_the_link_and_sends_nothing(self):
        player = self.ready()
        link = player.stream_link
        self.quality_switch(player)
        self.assertIs(player.stream_link, link)                # not rebuilt: same object, same id
        self.assertEqual((player.stream_item, link.item_id), ("abc", "abc"))
        self.assertEqual(self.posted, [])                      # no stream_end, no want
        self.assertFalse(player.loaded)                        # (the segment really restarted)

    def test_a_settings_restart_with_auto_interpolation_keeps_the_link_and_sends_nothing(self):
        player = self.ready()
        link = player.stream_link
        self.settings_restart(player, {"interpolation": "auto"})
        self.assertIs(player.stream_link, link)
        self.assertEqual(player.stream_item, "abc")
        self.assertEqual(self.posted, [])
        self.assertFalse(player.loaded)

    def test_a_quality_switch_together_with_a_seek_sends_exactly_one_want(self):
        player = self.ready()
        link = player.stream_link
        self.quality_switch(player, seek="8:300")              # newest request decides the position: the seek
        self.assertIs(player.stream_link, link)
        self.assertEqual(self.posted, [{"cmd": "want", "id": "abc", "pos": 300.0, "foreground": True}])

    def test_the_clock_continues_after_a_restart_from_the_new_position(self):
        player = self.ready()
        player.stream_link_tick(100.0)                         # watching + clock 42.5 before the restart
        self.posted.clear()
        self.settings_restart(player, {"interpolation": "auto"})
        player.stream_link_tick(100.5)                         # restart not loaded yet: nothing, no watching burst
        self.assertEqual(self.posted, [])
        self.segment_loaded(player, 50.25)                     # the settings restart resumed at settings-pos
        player.stream_link_tick(101.0)
        self.assertEqual(self.posted, [{"cmd": "clock", "id": "abc", "pos": 50.25, "playing": True}])
        self.quality_switch(player)                            # and again for a quality switch
        self.posted.clear()
        self.segment_loaded(player, 40.5)
        player.stream_link_tick(102.0)
        self.assertEqual(self.posted, [{"cmd": "clock", "id": "abc", "pos": 40.5, "playing": True}])
        player.stream_link_tick(109.9)                         # the 10 s heartbeat did not restart with the segment
        self.assertEqual([m["cmd"] for m in self.posted], ["clock", "clock"])
        player.stream_link_tick(110.0)
        self.assertEqual(self.posted[-1], {"cmd": "watching", "id": "abc"})

    # --- T2.5 / T3.2: an external subtitle track on a live restart --------------------------------------------------
    def test_a_live_restart_request_itself_never_touches_subtitle_tracks(self):
        """handle_requests only restarts the Lada segment. mpv drops external tracks with the old file at the loop's
        loadfile; re-adding them is stream_subtitle_tick's job (T3.2 tests below), not the request handler's."""
        player = self.ready()
        commands = []
        player.ipc.command = lambda *a: commands.append(a)
        self.settings_restart(player, {"interpolation": "auto"})
        self.quality_switch(player)
        self.assertEqual([c for c in commands if c and c[0] in ("sub-add", "sub-remove", "sub-reload")], [])

    # --- T2.5b: casting keeps the `watching` heartbeat ------------------------------------------------------------
    def test_watching_continues_while_casting_but_clock_does_not(self):
        player = self.ready()
        player.stream_link_tick(100.0)
        self.assertEqual([m["cmd"] for m in self.posted], ["watching", "clock"])
        self.posted.clear()
        player.cast.active = {"mode": "hls", "local_pos": 42.5}
        for now in (101.0, 105.0, 109.9):
            player.stream_link_tick(now)
        self.assertEqual(self.posted, [])                      # throttled like everywhere else
        player.stream_link_tick(110.0)
        player.stream_link_tick(111.0)                         # clock is due, but casting: P5 decides what to send
        player.stream_link_tick(120.0)
        self.assertEqual(self.posted, [{"cmd": "watching", "id": "abc"}, {"cmd": "watching", "id": "abc"}])

    def test_watching_does_not_depend_on_the_local_session_or_mpv(self):
        player = self.ready()
        player.cast.active = {"mode": "airplay"}
        player.ipc.get = mock.Mock(side_effect=RuntimeError("mpv gone"))
        player.stream_link_tick(100.0)
        self.assertEqual(self.posted, [{"cmd": "watching", "id": "abc"}])
        player.session = None                                  # (a link without a session cannot normally exist)
        player.stream_link_tick(110.0)
        self.assertEqual([m["cmd"] for m in self.posted], ["watching", "watching"])

    def test_a_restart_while_casting_keeps_the_item_and_sends_neither_stream_end_nor_want(self):
        player = self.ready()
        link = player.stream_link
        player.cast.active = {"mode": "hls", "plan": SimpleNamespace(fps=30), "local_pos": 42.5}
        calls = []
        player.cast.suspend = lambda: calls.append("suspend")
        player.cast.resume = lambda *_a, **_k: calls.append("resume")
        player.session.encoder = object()
        player._deps = dataclasses.replace(player._deps, Fraction=Fraction)                     # (the fake deps stub it out)
        self.quality_switch(player)
        self.assertEqual(calls, ["suspend", "resume"])         # the cast path really ran
        self.assertIs(player.stream_link, link)
        self.assertEqual(self.posted, [])
        player.stream_link_tick(200.0)
        self.assertEqual(self.posted, [{"cmd": "watching", "id": "abc"}])

    # --- P5: casting subtitle linkage (clock & seek want) ---------------------------------------------------------
    def test_airplay_casting_reports_clock_and_respects_cast_paused(self):
        player = self.ready()
        player.cast.active = {"mode": "airplay", "phase": "playing", "local_pos": 88.5}
        player.stream_link_tick(100.0)
        self.assertEqual(self.posted, [
            {"cmd": "watching", "id": "abc"},
            {"cmd": "clock", "id": "abc", "pos": 88.5, "playing": True},
        ])
        self.posted.clear()
        player.stream_link_tick(100.5)                        # throttled
        self.assertEqual(self.posted, [])
        player.ipc.props["user-data/oh/cast-paused"] = "1"
        player.stream_link_tick(101.0)
        self.assertEqual(self.posted, [
            {"cmd": "clock", "id": "abc", "pos": 88.5, "playing": False},
        ])
        self.posted.clear()
        player.ipc.props["user-data/oh/cast-paused"] = "0"
        player.stream_link_tick(102.0)
        self.assertEqual(self.posted, [
            {"cmd": "clock", "id": "abc", "pos": 88.5, "playing": True},
        ])

    def test_hls_casting_reports_clock_from_v0_and_vod_last_request(self):
        player = self.ready()
        vod = SimpleNamespace(last_request={"pos": 32.0}, restarts=0)
        player.cast.active = {"mode": "hls", "v0": 10.0, "vod": vod}
        player.stream_link_tick(100.0)
        self.assertEqual(self.posted, [
            {"cmd": "watching", "id": "abc"},
            {"cmd": "clock", "id": "abc", "pos": 42.0, "playing": True},
        ])
        self.posted.clear()
        vod.last_request = {"pos": 35.5}
        player.stream_link_tick(101.0)
        self.assertEqual(self.posted, [
            {"cmd": "clock", "id": "abc", "pos": 45.5, "playing": True},
        ])

    def test_airplay_cast_seek_calls_cast_seek_and_sends_want(self):
        player = self.ready()
        player.cast.active = {
            "mode": "airplay",
            "phase": "playing",
            "plan": SimpleNamespace(index_at=mock.Mock(return_value=12)),
            "v0": 10.0,
        }
        player.cast.seek = mock.Mock()
        player.handle_cast_seek("1:120.5", 105.0)
        player.cast.seek.assert_called_once_with(120.5)
        self.assertEqual(self.posted, [
            {"cmd": "want", "id": "abc", "pos": 120.5, "foreground": True},
        ])
        self.assertEqual(player.cast.active["plan"].index_at.call_args[0][0], 110.5)
        self.assertEqual(player._cast_stream_want_k, 12)
        self.assertEqual(player.ipc.props.get("user-data/oh/request-done"), "1:120.5")

    def test_hls_receiver_jump_sends_want_and_deduplicates_same_anchor(self):
        player = self.ready()
        plan = SimpleNamespace(start=lambda k: float(k * 2))
        vod = SimpleNamespace(restarts=1, anchor=15, last_request={"pos": 30.0})
        player.cast.active = {"mode": "hls", "v0": 10.0, "plan": plan, "vod": vod}

        player.stream_link_tick(100.0)
        self.assertEqual(self.posted, [
            {"cmd": "watching", "id": "abc"},
            {"cmd": "want", "id": "abc", "pos": 40.0, "foreground": True},
            {"cmd": "clock", "id": "abc", "pos": 40.0, "playing": True},
        ])
        self.assertEqual(player._cast_stream_restarts, 1)
        self.assertEqual(player._cast_stream_want_k, 15)
        self.posted.clear()

        player.stream_link_tick(101.0)
        self.assertEqual([m for m in self.posted if m["cmd"] == "want"], [])
        self.posted.clear()

        vod.restarts = 2
        player.stream_link_tick(102.0)
        self.assertEqual([m for m in self.posted if m["cmd"] == "want"], [])
        self.assertEqual(player._cast_stream_restarts, 2)
        self.posted.clear()

        vod.anchor = 20
        vod.restarts = 3
        player.stream_link_tick(103.0)
        self.assertEqual([m for m in self.posted if m["cmd"] == "want"], [
            {"cmd": "want", "id": "abc", "pos": 50.0, "foreground": True},
        ])
        self.assertEqual(player._cast_stream_want_k, 20)


class LiveMpv(FakeIpc):
    """Just enough mpv for external subtitle tracks: loadfile drops them, sub-add / sub-remove keep a track list,
    "select" selects the new track, "auto" does not, and sid / sub-visibility are plain properties."""
    BASE = [{"type": "video", "id": 1}, {"type": "audio", "id": 1}]

    def __init__(self, events):
        super().__init__(events)
        self.commands, self.next_id = [], 1
        self.props.update({"track-list": list(self.BASE), "sid": False, "sub-visibility": True})

    def command(self, *args):
        self.commands.append(args)
        tracks = self.props["track-list"]
        if args[0] == "loadfile":
            self.props.update({"path": args[1], "track-list": list(self.BASE), "sid": False})
        elif args[0] == "sub-add":
            track = {"type": "sub", "id": self.next_id, "external": True, "external-filename": args[1],
                     "title": args[3] if len(args) > 3 else None, "lang": args[4] if len(args) > 4 else None}
            self.next_id += 1
            tracks.append(track)
            if args[2] == "select":
                self.props["sid"] = track["id"]
        elif args[0] == "sub-remove":
            tracks[:] = [t for t in tracks if not (t["type"] == "sub" and t["id"] == args[1])]
            if self.props["sid"] == args[1]:
                self.props["sid"] = False

    def adds(self):
        return [c for c in self.commands if c[0] == "sub-add"]

    def select(self, title):
        self.props["sid"] = next(t["id"] for t in self.props["track-list"] if t.get("title") == title)

    def selected_title(self):
        return next((t["title"] for t in self.props["track-list"]
                     if t["type"] == "sub" and t["id"] == self.props["sid"]), None)


class LiveSubtitleTests(PlayerSide):
    """T3.2: the live-mode player loads the stream item's growing external subtitles, keeps the user's track and
    visibility across segment restarts, and warns when the viewer is past the covered range. (Subclasses PlayerSide for
    its helpers; the inherited tests run again against the same code, which is harmless.)"""
    URL = "tcp://127.0.0.1:5001"
    ZH, JA = "繁體中文（AI）", "日文（AI）"
    GAP = "此處字幕尚未產生"

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path
        self.tmp = Path(tempfile.mkdtemp(prefix="t32-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tracks(self, *names):
        meta = {"zh": (self.ZH, "chi", True), "ja": (self.JA, "jpn", False)}
        out = []
        for n in names:
            path = self.tmp / f"{n}.srt"
            path.write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
            out.append({"path": str(path), "title": meta[n][0], "lang": meta[n][1], "default": meta[n][2]})
        return out

    def update(self, item="abc", names=("zh", "ja"), covered=60.0, complete=False, offset=0.0):
        return SimpleNamespace(item_id=item, subtitles=self.tracks(*names), covered_s=covered,
                               complete=complete, subtitle_offset=offset)

    def live(self):
        player = self.ready()
        player.ipc = LiveMpv(player.ipc.events)
        player.ipc.props.update({"path": self.URL, "time-pos": 42.5})
        player.session.url = self.URL
        player.pending_subtitle_updates = {}
        player.progressive_sub_loaded = player.progressive_sub_dir = None
        player.progressive_sub_selection = player.progressive_sub_visibility = None
        return player

    def restart(self, player, url):
        """The loop's replace-loadfile of a restarted segment: stream_subtitles_before_load, then mpv's loadfile."""
        player.stream_subtitles_before_load()
        player.session.url = url
        player.ipc.command("loadfile", url, "replace")
        player.loaded, player.resume_wait = True, False

    def osd_texts(self, player):
        return [c[1] for c in player.ipc.commands if c[0] == "show-text"]

    # --- receiving and loading ----------------------------------------------------------------------------------
    def test_begin_source_creates_the_holder_and_end_source_drops_it(self):
        player = self.make_player()
        self.begin(player)
        self.assertIsNone(player.stream_subs)
        self.begin(player, stream_item="abc")
        self.assertEqual((player.stream_subs.queue_id, player.stream_subs.subtitles), ("abc", []))
        player.end_source()
        self.assertIsNone(player.stream_subs)

    def test_a_snapshot_for_the_stream_item_is_stored_and_loads_into_the_playing_segment(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        self.assertEqual([t["title"] for t in player.stream_subs.subtitles], [self.ZH, self.JA])
        self.assertEqual((player.stream_subs.subtitle_covered_s, player.stream_subs.subtitle_complete), (60.0, False))
        self.assertEqual(len(player.ipc.adds()), 2)
        self.assertEqual(player.ipc.selected_title(), self.ZH)           # the default track, as for progressive
        self.assertEqual(player.progressive_sub_loaded, self.URL)

    def test_a_snapshot_for_another_item_is_kept_pending_and_not_loaded(self):
        player = self.live()
        player.apply_subtitle_update(self.update(item="other"))
        self.assertEqual(player.stream_subs.subtitles, [])
        self.assertEqual(player.ipc.adds(), [])
        self.assertIn("other", player.pending_subtitle_updates)

    def test_a_snapshot_that_came_before_the_open_is_applied_at_begin_source(self):
        player = self.make_player()
        player.pending_subtitle_updates = {}
        player.apply_subtitle_update(self.update())                      # nothing plays yet: parked
        self.assertIn("abc", player.pending_subtitle_updates)
        self.begin(player, stream_item="abc")
        self.assertEqual(len(player.stream_subs.subtitles), 2)
        self.assertNotIn("abc", player.pending_subtitle_updates)

    def test_the_tick_loads_tracks_once_per_file_and_only_when_that_file_plays(self):
        player = self.live()
        player.stream_subs.subtitles = self.tracks("zh", "ja")           # a snapshot while the segment was not loaded yet
        player.loaded = False
        player.stream_subtitle_tick()
        player.loaded = True
        player.ipc.props["path"] = "tcp://old"                           # mpv still on another file (a tail)
        player.stream_subtitle_tick()
        self.assertEqual(player.ipc.adds(), [])
        player.ipc.props["path"] = self.URL
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 2)
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 2)                      # once per file load

    def test_the_tick_does_nothing_without_tracks_or_while_casting(self):
        player = self.live()
        player.stream_subtitle_tick()                                    # no snapshot yet
        player.stream_subs.subtitles = self.tracks("zh")
        player.cast.active = {"source": object()}
        player.stream_subtitle_tick()
        self.assertEqual(player.ipc.adds(), [])
        player.cast.active = None
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 1)

    def test_a_growing_snapshot_reloads_the_existing_tracks_and_keeps_the_choice(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.select(self.JA)
        player.ipc.commands.clear()
        player.apply_subtitle_update(self.update(covered=120.0))         # same files, more cues
        self.assertEqual([c[0] for c in player.ipc.commands if c[0].startswith("sub-")], ["sub-reload", "sub-reload"])
        self.assertEqual(player.ipc.selected_title(), self.JA)
        self.assertEqual(player.stream_subs.subtitle_covered_s, 120.0)

    def test_the_snapshot_offset_shifts_the_tracks_like_progressive(self):
        import dataclasses
        from oh_streamer.player import progressive_stream
        player = self.live()
        player._deps = dataclasses.replace(player._deps, progressive_stream=progressive_stream)
        player.apply_subtitle_update(self.update(names=("zh",), offset=3.5))
        added = player.ipc.adds()[0][1]
        self.assertNotEqual(added, player.stream_subs.subtitles[0]["path"])     # a temporary shifted copy
        with open(added, encoding="utf-8") as f:
            self.assertIn("00:00:04,500", f.read())                      # the 1.0 s cue + 3.5 s
        self.assertIsNotNone(player.progressive_sub_dir)
        player.end_source()
        self.assertIsNone(player.progressive_sub_dir)                    # temp folder removed with the source

    # --- restarts keep the chosen track and visibility ------------------------------------------------------------
    def test_a_restart_re_adds_the_tracks_and_restores_the_chosen_one(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.select(self.JA)
        player.ipc.commands.clear()
        self.restart(player, "tcp://127.0.0.1:5002")                     # quality / settings restart: tracks gone with the file
        self.assertEqual([t for t in player.ipc.props["track-list"] if t["type"] == "sub"], [])
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 2)
        self.assertEqual(player.ipc.selected_title(), self.JA)           # not the default
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 2)

    def test_subtitles_turned_off_stay_off_after_a_restart(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.props["sid"] = False
        player.ipc.props["sub-visibility"] = False
        self.restart(player, "tcp://127.0.0.1:5002")
        player.ipc.props["sub-visibility"] = True                        # (stand-in for anything that reset it)
        player.stream_subtitle_tick()
        self.assertIsNone(player.ipc.selected_title())
        self.assertIn(player.ipc.props["sid"], (False, "no"))
        self.assertIs(player.ipc.props["sub-visibility"], False)

    def test_the_choice_survives_several_restarts(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.select(self.JA)
        for port in (5002, 5003, 5004):
            self.restart(player, f"tcp://127.0.0.1:{port}")
            player.stream_subtitle_tick()
            self.assertEqual(player.ipc.selected_title(), self.JA, port)

    def test_a_restart_before_any_track_was_loaded_does_not_invent_a_choice(self):
        player = self.live()                                             # no snapshot yet: sid=no is not the user's choice
        self.restart(player, "tcp://127.0.0.1:5002")
        self.assertIsNone(player.progressive_sub_selection)
        player.apply_subtitle_update(self.update())                      # the first snapshot arrives after the restart
        self.assertEqual(player.ipc.selected_title(), self.ZH)

    def test_a_snapshot_during_a_restart_is_loaded_when_the_new_segment_plays(self):
        player = self.live()
        player.apply_subtitle_update(self.update(names=("zh",)))
        player.loaded = False                                            # restart in progress: old file still current
        player.apply_subtitle_update(self.update(covered=90.0))
        self.assertEqual(len(player.stream_subs.subtitles), 2)
        self.restart(player, "tcp://127.0.0.1:5002")
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 1 + 2)                  # one before, both on the new file
        self.assertEqual(player.ipc.selected_title(), self.ZH)

    def test_a_tail_append_keeps_the_choice_for_the_file_that_follows(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.select(self.JA)
        player.stream_subtitles_before_load()                            # via == "append": mpv keeps playing the tail
        player.session.url = "tcp://127.0.0.1:5002"
        player.loaded = True                                             # (mpv's path is still the tail's)
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 2)                      # nothing added to the tail file
        player.ipc.command("loadfile", "tcp://127.0.0.1:5002", "replace")   # mpv reaches the appended entry
        player.stream_subtitle_tick()
        self.assertEqual(len(player.ipc.adds()), 4)
        self.assertEqual(player.ipc.selected_title(), self.JA)

    def test_a_quality_switch_request_then_loadfile_loses_nothing(self):
        player = self.live()
        player.apply_subtitle_update(self.update())
        player.ipc.select(self.JA)
        subs = player.stream_subs
        self.quality_switch(player)                                      # the request path of the UI's quality button
        self.assertIs(player.stream_subs, subs)
        self.assertEqual(self.posted, [])
        self.restart(player, "tcp://127.0.0.1:5009")
        player.stream_subtitle_tick()
        self.assertEqual(player.ipc.selected_title(), self.JA)

    # --- jumping past the covered range ---------------------------------------------------------------------------
    def test_a_jump_past_the_covered_range_shows_the_gap_osd_once_until_caught_up(self):
        player = self.live()
        player.apply_subtitle_update(self.update(covered=60.0))
        player.ipc.props["time-pos"] = 30.0
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [])
        player.ipc.props["time-pos"] = 900.0                             # seek far ahead: no subtitles there yet
        player.subtitle_gap_tick()
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [self.GAP])
        player.apply_subtitle_update(self.update(covered=960.0))         # the queue caught up (want = foreground)
        player.subtitle_gap_tick()
        player.ipc.props["time-pos"] = 2000.0                            # and a later jump warns again
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [self.GAP] * 2)

    def test_empty_first_srt_failure_does_not_disable_the_first_nonempty_publication(self):
        player = self.live()
        command = player.ipc.command
        def fail_add(*args):
            if args[0] == "sub-add":
                raise RuntimeError("sub-add: error loading file")
            return command(*args)
        with mock.patch.object(player.ipc, "command", side_effect=fail_add):
            player.apply_subtitle_update(self.update())
        self.assertFalse(any(t.get("type") == "sub" for t in player.ipc.get("track-list")))
        self.assertIsNone(player.progressive_sub_selection)
        player.apply_subtitle_update(self.update(covered=120.0))
        self.assertIsNone(player.progressive_sub_selection)
        selected = player.ipc.get("sid")
        track = next(t for t in player.ipc.get("track-list") if t.get("type") == "sub" and t.get("id") == selected)
        self.assertEqual(track.get("title"), self.ZH)
        # Once a real track exists, an explicit user sid=no still survives publication updates.
        player.ipc.set("sid", "no")
        player.apply_subtitle_update(self.update(covered=180.0))
        self.assertIs(player.progressive_sub_selection, False)
        self.assertIn(player.ipc.get("sid"), (False, "no"))

    def test_empty_srt_failure_across_live_restart_preserves_default_selection(self):
        player = self.live()
        with mock.patch.object(player.ipc, "command", side_effect=RuntimeError("empty SRT")):
            player.apply_subtitle_update(self.update())
        self.restart(player, "tcp://new-segment")
        self.assertIsNone(player.progressive_sub_selection)
        player.stream_subtitle_tick()
        selected = player.ipc.get("sid")
        track = next(t for t in player.ipc.get("track-list") if t.get("type") == "sub" and t.get("id") == selected)
        self.assertEqual(track.get("title"), self.ZH)

    def test_live_gap_stays_quiet_in_seek_island_then_warns_in_the_gap(self):
        player = self.live()
        update = self.update(covered=0.0, offset=2.0)
        update.covered_ranges = [[2390.0, 2420.0]]
        player.apply_subtitle_update(update)
        self.assertEqual(player.stream_subs.subtitle_covered_ranges, update.covered_ranges)
        player.ipc.props["time-pos"] = 2402.0
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [])
        player.ipc.props["time-pos"] = 1200.0
        player.subtitle_gap_tick()
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), ["此處字幕尚未產生"])
        player.ipc.props["time-pos"] = 2402.0
        player.subtitle_gap_tick()
        self.assertFalse(player.subtitle_gap_warned)
        # An older queue snapshot replaces ranges with the legacy continuous prefix.
        player.apply_subtitle_update(self.update(covered=2450.0))
        self.assertIsNone(player.stream_subs.subtitle_covered_ranges)

    def test_the_gap_uses_the_snapshot_offset_and_stays_quiet_before_the_first_snapshot(self):
        player = self.live()
        player.ipc.props["time-pos"] = 500.0
        player.subtitle_gap_tick()                                       # no snapshot yet: nothing to compare with
        self.assertEqual(self.osd_texts(player), [])
        player.apply_subtitle_update(self.update(covered=100.0, offset=450.0))   # covers up to 550 on the mpv clock
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [])
        player.ipc.props["time-pos"] = 551.0
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [self.GAP])

    def test_a_complete_snapshot_never_warns(self):
        player = self.live()
        player.apply_subtitle_update(self.update(covered=60.0, complete=True))
        player.ipc.props["time-pos"] = 900.0
        player.subtitle_gap_tick()
        self.assertEqual(self.osd_texts(player), [])

    def test_without_a_stream_item_nothing_loads_and_no_gap_is_shown(self):
        player = self.make_player()
        self.begin(player)
        player.ipc = LiveMpv(player.ipc.events)
        player.session.url = self.URL
        player.ipc.props["path"] = self.URL
        player.stream_subtitle_tick()
        player.stream_subtitles_before_load()
        player.subtitle_gap_tick()
        self.assertEqual(player.ipc.commands, [])

    # --- P5: casting subtitle update ------------------------------------------------------------------------------
    def test_apply_subtitle_update_while_casting_updates_cast_subtitles(self):
        player = self.live()
        plan = SimpleNamespace(
            video_start=0.0,
            total_duration=120.0,
            index_at=lambda t: int(t // 2),
            start=lambda k: float(k * 2),
        )
        player.cast.active = {"plan": plan}
        player.cast.update_subtitles = mock.Mock()
        update = self.update()
        player.apply_subtitle_update(update)
        player.cast.update_subtitles.assert_called_once()
        subs = player.cast.update_subtitles.call_args[0][0]
        self.assertEqual([r.title for r in subs.renditions], [self.ZH, self.JA])
        self.assertIs(subs.renditions[0].plan, plan)

    def test_apply_subtitle_update_while_casting_without_plan_does_not_update_cast(self):
        player = self.live()
        player.cast.active = {"mode": "airplay"}
        player.cast.update_subtitles = mock.Mock()
        player.apply_subtitle_update(self.update())
        player.cast.update_subtitles.assert_not_called()


# ---------------------------------------------------------------------------------------------------------------
# queue side
# ---------------------------------------------------------------------------------------------------------------

class QueueSideTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc.pause_all()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")      # a fresh store's cloud profile is unconfigured
        self.stream = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": True}, title="Clip")
        self.sid = self.stream.id

    def send(self, **msg):
        return self.svc._on_message(msg)

    def test_watching_refreshes_the_watch_lease_only(self):
        with mock.patch.object(self.svc, "goto") as goto, \
                mock.patch.object(self.svc, "_session_for") as session, \
                mock.patch.object(self.svc.store, "mark_seen") as seen:
            self.assertEqual(self.send(cmd="watching", id=self.sid), {"ok": True})
        self.assertIn(self.sid, self.svc._watch)
        self.assertIsNone(self.svc._player_item)
        for mocked in (goto, session, seen):
            mocked.assert_not_called()
        self.assertEqual(self.svc._priority_item(self.svc._watched_ids()), None)   # no worker priority from a stream

    def test_clock_records_position_and_playing(self):
        sub_dir = self.svc.store.item_dir(self.sid) / "subtitles"
        sub_dir.mkdir()
        self.assertEqual(self.send(cmd="clock", id=self.sid, pos=61.5, playing=True), {"ok": True})
        self.assertEqual(self.svc._subtitle_clocks[self.sid], (61.5, True))
        clock = json.loads((sub_dir / "play_clock.json").read_text(encoding="utf-8"))
        self.assertEqual((clock["position"], clock["playing"]), (61.5, True))
        self.send(cmd="clock", id=self.sid, pos=62.5, playing=False)
        self.assertEqual(self.svc._subtitle_clocks[self.sid], (62.5, False))
        self.assertIsNone(self.svc._player_item)

    def test_clock_ignores_subtitle_pos_and_rejects_bad_positions(self):
        self.send(cmd="clock", id=self.sid, pos=10.0, playing=True, subtitle_pos=3.0)
        self.assertEqual(self.svc._subtitle_clocks[self.sid][0], 10.0)
        for bad in (None, "x", float("nan"), -2):
            self.assertEqual(self.send(cmd="clock", id=self.sid, pos=bad, playing=True), {"ok": False, "error": "bad pos"})
        self.assertEqual(self.svc._subtitle_clocks[self.sid][0], 10.0)

    def test_want_records_the_position_without_regular_item_logic(self):
        with mock.patch.object(self.svc, "goto") as goto, \
                mock.patch.object(self.svc, "_session_for") as session, \
                mock.patch.object(self.svc, "_skip_subtitles_for_play") as skip, \
                mock.patch.object(self.svc.store, "mark_seen") as seen:
            reply = self.send(cmd="want", id=self.sid, pos=300.25, foreground=True)
        self.assertEqual(reply, {"ok": True})
        self.assertEqual(self.svc._subtitle_clocks[self.sid], (300.25, False))
        self.assertEqual(self.svc._subtitle_wants[self.sid][0], 300.25)
        self.assertIn(self.sid, self.svc._watch)
        self.assertIsNone(self.svc._player_item)
        for mocked in (goto, session, skip, seen):
            mocked.assert_not_called()
        self.send(cmd="want", id=self.sid, pos=5, k=9)                       # a stray k changes nothing
        self.assertEqual(self.svc._subtitle_wants[self.sid][0], 5.0)

    def test_want_without_a_valid_position_is_refused(self):
        for msg in ({}, {"pos": None}, {"pos": -1}, {"pos": "x"}):
            self.assertEqual(self.send(cmd="want", id=self.sid, **msg), {"ok": False, "error": "bad pos"})
        self.assertNotIn(self.sid, self.svc._subtitle_wants)

    def test_unknown_ids_still_get_an_error(self):
        self.assertEqual(self.send(cmd="watching", id="nope"), {"ok": False, "error": "unknown id"})
        self.assertEqual(self.send(cmd="want", id="nope", pos=1.0), {"ok": False, "error": "unknown id"})
        self.assertEqual(self.send(cmd="want", id="nope", k=1), {"ok": False, "error": "unknown id"})
        self.svc.store.remove_stream(self.sid)                               # the stream item is gone
        self.assertEqual(self.send(cmd="watching", id=self.sid), {"ok": False, "error": "unknown id"})
        self.assertEqual(self.send(cmd="want", id=self.sid, pos=1.0), {"ok": False, "error": "unknown id"})

    def test_player_state_is_unaffected(self):
        self.send(cmd="watching", id=self.sid)
        self.send(cmd="clock", id=self.sid, pos=1.0, playing=True)
        self.send(cmd="want", id=self.sid, pos=2.0)
        self.assertFalse(self.svc.is_player_busy())
        self.assertFalse(self.svc.is_player_playing())
        self.assertEqual(self.svc.store.items(), [])
        self.assertEqual(self.svc.store.get(self.sid).state, queue_store.STREAM_STATE)
        self.assertFalse(self.svc.store.get(self.sid).seen)

    def test_regular_items_keep_their_behaviour(self):
        item = self.ready()
        self.assertEqual(self.send(cmd="watching", id=item), {"ok": True})
        self.assertEqual(self.svc._player_item, item)
        self.send(cmd="clock", id=item, pos=7.0, playing=True)
        self.assertEqual(self.svc._subtitle_clocks[item], (7.0, True))
        self.assertEqual(self.send(cmd="want", id="nope", pos=1.0), {"ok": False, "error": "unknown id"})

    # --- T3.2: the subtitle snapshot of a stream item reaches the player like a progressive item's ----------------
    def test_a_subtitle_snapshot_of_a_stream_item_is_sent_to_the_player(self):
        folder = self.svc.store.item_dir(self.sid) / "subtitles"
        folder.mkdir(exist_ok=True)
        srt = folder / "subtitle.zh.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
        self.assertIsNone(self.svc._player_item)                  # a stream item is never the player's queue item
        self.player_reply = {"ok": True}
        self.svc._subtitle_publish_update(self.svc.store.get(self.sid), {
            "srt": [str(srt)], "covered_s": 120.0, "complete": False,
            "subtitles": [{"path": str(srt), "lang": "chi", "title": "繁體中文（AI）", "default": True}]})
        sent = [p for p in self.players if p.get("cmd") == "subtitle_update"]
        self.assertEqual(len(sent), 1)
        self.assertEqual((sent[0]["item_id"], sent[0]["covered_s"], sent[0]["complete"]), (self.sid, 120.0, False))
        self.assertEqual(sent[0]["srt"], [str(srt)])
        from oh_streamer.player import queue_bridge             # the player accepts exactly this message
        update = queue_bridge.subtitle_update_from_message(sent[0])
        self.assertEqual((update.item_id, update.covered_s), (self.sid, 120.0))

    def test_a_replaced_stream_item_forgets_what_the_player_said(self):
        self.send(cmd="clock", id=self.sid, pos=1.0, playing=True)
        self.send(cmd="want", id=self.sid, pos=2.0)
        self.svc._forget_stream_playback(self.sid)
        self.assertEqual((self.sid in self.svc._watch, self.sid in self.svc._subtitle_clocks,
                          self.sid in self.svc._subtitle_wants), (False, False, False))


if __name__ == "__main__":
    unittest.main()
