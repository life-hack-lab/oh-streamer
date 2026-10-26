"""T2.7: a stream item lives for one play. The player sends {"cmd":"stream_end","id"} when it is done with the item
(new source, close, end_source, or the item will never be used); the queue removes the item and its streams/<id>/ folder,
times out items of a player that died, and clears all of them when it stops. Run from streamer/:
CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_lifecycle."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

try:                                    # run_tests.py imports tests.* ; "-s tests" imports flat names
    from tests import test_queue_service as TQS
    from tests.test_queue_service import ServiceTestBase
    from tests.test_player_controller import (
        FakeCast, FakeEngine, FakeIpc, FakeOpener, FakePlayer, FakeEndpoint, FakeGpu, FakeHistory, FakeSession,
        controller_args, controller_deps)
    from tests import test_stream_subtitle_link as LINK
except ImportError:
    import test_queue_service as TQS
    from test_queue_service import ServiceTestBase
    from test_player_controller import (
        FakeCast, FakeEngine, FakeIpc, FakeOpener, FakePlayer, FakeEndpoint, FakeGpu, FakeHistory, FakeSession,
        controller_args, controller_deps)
    import test_stream_subtitle_link as LINK

from oh_streamer.player import controller as C
from oh_streamer.player import opening
from oh_streamer.player.queue_bridge import QueueLink
from oh_streamer.queue import queue_service as QS
from oh_streamer.queue import queue_store


def stream_ends(posted):
    return [m["id"] for m in posted if m.get("cmd") == "stream_end"]


# ---------------------------------------------------------------------------------------------------------------
# player side
# ---------------------------------------------------------------------------------------------------------------

class PlayerStreamEndTests(unittest.TestCase):
    make_player = LINK.PlayerSide.make_player
    begin = LINK.PlayerSide.begin
    ready = LINK.PlayerSide.ready
    seek = LINK.PlayerSide.seek

    def test_end_source_sends_stream_end_once(self):
        player = self.make_player()
        self.begin(player, stream_item="abc")
        self.posted.clear()
        player.end_source()
        self.assertEqual(self.posted, [{"cmd": "stream_end", "id": "abc"}])
        player.end_source()                                    # nothing left to end
        self.assertEqual(stream_ends(self.posted), ["abc"])

    def test_leaving_for_another_source_ends_the_old_stream_item_only(self):
        player = self.make_player()
        player.engine.interp, player.thumb_dir, player.lua_reset = None, None, lambda: None
        player._deps.device_backend.empty_cache = lambda _device: None
        self.begin(player, stream_item="old")
        self.posted.clear()
        player.leave_current()                                 # what switch_to / switch_to_processed do first
        self.assertEqual(self.posted, [{"cmd": "stream_end", "id": "old"}])
        self.begin(player, stream_item="new")
        self.assertEqual(stream_ends(self.posted), ["old"])    # starting the new source does not end its own item

    def test_a_source_without_stream_item_or_without_queue_sends_nothing(self):
        player = self.make_player()
        self.begin(player)
        player.end_source()
        self.assertEqual(self.posted, [])
        player = self.make_player(link=False)
        self.begin(player, stream_item="abc")
        player.end_source()
        self.assertEqual(self.posted, [])

    def test_a_failing_sender_never_raises(self):
        player = self.make_player()
        player.link.post = mock.Mock(side_effect=RuntimeError("boom"))
        player.release_stream_item("abc")
        player.link.post = mock.Mock(return_value=False)
        player.release_stream_item("abc")
        self.assertTrue(any("stream_end" in line for line in self.logs))

    def test_a_pending_open_that_never_plays_is_ended(self):
        player = self.make_player()
        player.pending_open = SimpleNamespace(stream_item="waiting")
        player.drop_pending_open()
        self.assertIsNone(player.pending_open)
        self.assertEqual(stream_ends(self.posted), ["waiting"])
        player.drop_pending_open()                              # nothing pending
        player.pending_open = SimpleNamespace()                 # an Opened without stream item
        player.drop_pending_open()
        self.assertEqual(stream_ends(self.posted), ["waiting"])

    def test_switch_to_processed_ends_the_item_of_the_source_it_replaces(self):
        player = self.make_player()
        player.engine.interp, player.thumb_dir, player.lua_reset = None, None, lambda: None
        player._deps.device_backend.empty_cache = lambda _device: None
        self.begin(player, stream_item="abc")
        player.pending_open = SimpleNamespace(stream_item="waiting")
        player.epoch = 0
        player.publish_settings = lambda: None
        player.apply_pending_subtitle_update = lambda *_a: None
        player.publish_compare_layers = lambda *a, **k: None
        po = SimpleNamespace(path="done.mkv", title="t", queue_id=None, layers="", derive="", combos_dir=None,
                             tags=None, engine="lada", start=0.0, subtitles=[], from_lookup=False)
        try:
            player.switch_to_processed(po)
        except Exception:    # noqa: BLE001 - the rest of the switch needs a real mpv; the ending happened before it
            pass
        self.assertEqual(sorted(stream_ends(self.posted)), ["abc", "waiting"])

    # --- not on restarts -----------------------------------------------------------------------------------------
    def test_a_seek_and_a_settings_restart_keep_the_item(self):
        player = self.ready()
        self.seek(player)
        player.ipc.props["user-data/oh/seek-request"] = ""
        player.ipc.props["user-data/oh/settings-request"] = "6:x"
        player.ipc.props["user-data/oh/settings-pos"] = "50"
        player.session.restorer = object()
        player.session.start = lambda *_a, **_k: None
        player.reload_pending = {}
        with mock.patch.object(player._deps.PS, "parse_request", create=True, return_value={}):
            player.handle_requests("", "", "6:x")
        self.assertEqual(stream_ends(self.posted), [])
        self.assertEqual(player.stream_item, "abc")

    def test_a_quality_switch_keeps_the_item(self):
        player = self.ready()
        player.sc = SimpleNamespace(diff=lambda _w: {}, menu=lambda: [])
        player.stop_for_request = lambda *_a: 0.0
        player.reload_pending = {}
        player.session.restorer = object()
        player.session.target = 12.0
        player.session.source = SimpleNamespace()
        player.session.start = lambda *_a, **_k: None
        player.qc = SimpleNamespace(current={"id": "s", "label": "1080p"}, lada=SimpleNamespace(open=lambda _s: object()))
        new_source = SimpleNamespace(describe=lambda: "480p")
        player.prepare = lambda fid, _state: ({"id": fid, "label": "480p"}, new_source)
        applied = []
        player.apply_source = lambda q, source, restorer: applied.append(q["id"])
        player.ipc.props["user-data/oh/quality-request"] = "7:low"
        player.ipc.props["user-data/oh/quality-pos"] = "40"
        player.handle_requests("", "7:low", "")
        self.assertEqual(applied, ["low"])                       # the quality really switched
        self.assertEqual(stream_ends(self.posted), [])
        self.assertEqual((player.stream_item, player.stream_link.item_id), ("abc", "abc"))

    # --- the opener: items no source will use --------------------------------------------------------------------
    def opener(self, *, open_source=None, lookup=None):
        unused = []
        opened = opening.Opened(input="u")
        opener = opening.Opener(SimpleNamespace(), None, open_source=open_source or (lambda *a, **k: opened),
                                log=lambda _m: None)
        opener.on_unused_stream_item = unused.append
        opener.lookup = lookup
        return opener, unused

    def wait(self, opener):
        for _ in range(300):
            if not opener.busy:
                return
            threading.Event().wait(0.01)
        self.fail("opener did not finish")

    def submit(self, opener, **extra):
        opener.submit(SimpleNamespace(kind="url", target="u", headers={}), "r", 0.0, **extra)

    def test_a_live_source_uses_its_item(self):
        opener, unused = self.opener()
        self.submit(opener, stream_item="abc")
        self.wait(opener)
        self.assertEqual(opener.poll()[0].stream_item, "abc")
        self.assertEqual(unused, [])

    def test_processed_path_drops_the_item_at_once(self):
        processed = SimpleNamespace(path="done.mkv")            # lookup found the queue's output: no Opened, no item
        opener, unused = self.opener(lookup=lambda _oi: processed)
        self.submit(opener, stream_item="abc")
        self.wait(opener)
        self.assertIs(opener.poll()[0], processed)
        self.assertEqual(unused, ["abc"])

    def test_a_failed_open_drops_the_item(self):
        def fail(*_a, **_k):
            raise RuntimeError("no such video")
        opener, unused = self.opener(open_source=fail)
        self.submit(opener, stream_item="abc")
        self.wait(opener)
        self.assertIsNotNone(opener.poll()[1])
        self.assertEqual(unused, ["abc"])

    def test_a_superseded_request_drops_its_item(self):
        gate = threading.Event()

        def slow(*_a, **_k):
            gate.wait(5)
            return opening.Opened(input="u")
        opener, unused = self.opener(open_source=slow)
        self.submit(opener, stream_item="first")
        opener.cancel()                                          # a processed file / newer open supersedes it
        gate.set()
        for _ in range(300):
            if unused:
                break
            threading.Event().wait(0.01)
        self.assertEqual(unused, ["first"])
        self.assertIsNone(opener.poll())

    def test_a_finished_result_nobody_picked_up_is_dropped_by_cancel_and_by_a_new_submit(self):
        opener, unused = self.opener()
        self.submit(opener, stream_item="a")
        self.wait(opener)
        opener.cancel()
        self.assertEqual(unused, ["a"])
        self.submit(opener, stream_item="b")
        self.wait(opener)
        self.submit(opener, stream_item="c")                     # b was never polled
        self.wait(opener)
        self.assertEqual(unused, ["a", "b"])
        self.assertEqual(opener.poll()[0].stream_item, "c")

    def test_no_stream_item_means_no_callback(self):
        opener, unused = self.opener(lookup=lambda _oi: SimpleNamespace(path="x"))
        self.submit(opener)
        self.wait(opener)
        opener.cancel()
        self.assertEqual(unused, [])

    # --- closing the player --------------------------------------------------------------------------------------
    def test_run_wires_the_opener_and_sends_stream_end_before_the_link_closes(self):
        events = []
        posted = []

        class Link:
            def post(_self, payload, *_a, **_k):
                posted.append(payload)
                events.append(("post", payload["cmd"]))
                return True

            def close(_self):
                events.append("link:close")

        source = SimpleNamespace(meta=SimpleNamespace(duration=30.0, video_fps_exact=30))
        opened = SimpleNamespace(input="clip.mp4", user_headers={}, source=source, quals=[{"id": "source"}],
                                 chosen={"id": "source", "label": "1080p"}, refresh=None, title="clip",
                                 source_text="clip.mp4", start=0.0, thumb_src=None, stream_item="abc")
        FakeOpener.instances.clear()
        deps = controller_deps(
            events, MpvEndpoint=lambda: FakeEndpoint(events), start_mpv=lambda *_a, **_k: FakePlayer(events),
            QueueLink=lambda *_a, **_k: Link(), Session=lambda *args: FakeSession(events, *args),
            info_panel=SimpleNamespace(GpuMonitor=lambda: FakeGpu(events), History=lambda: FakeHistory(),
                                       build_info=lambda snap: snap))
        player = C.PlayerController(controller_args(), FakeEngine(), first=opened, deps=deps)
        self.assertEqual(player.run(), 0)                       # the mpv window closing
        self.assertEqual(stream_ends(posted), ["abc"])
        self.assertLess(events.index(("post", "stream_end")), events.index("link:close"))
        self.assertEqual(FakeOpener.instances[0].on_unused_stream_item, player.release_stream_item)

    def test_close_flushes_stream_end_through_the_real_link_without_blocking(self):
        sent = []
        gate = threading.Event()

        def send(payload, _timeout):
            sent.append(payload["cmd"])
            if payload["cmd"] == "stream_end":
                gate.set()
            return {"ok": True}

        player = self.make_player()
        player.link = QueueLink(send=send)
        self.begin(player, stream_item="abc")
        player.end_source()
        player.link.close()
        self.assertTrue(gate.wait(2))
        self.assertIn("stream_end", sent)

        player = self.make_player()                               # a queue that does not answer: close stays bounded
        release = threading.Event()
        player.link = QueueLink(send=lambda payload, _t: release.wait(5))
        self.begin(player, stream_item="abc")
        player.end_source()
        started = threading.Event()
        t = threading.Thread(target=lambda: (started.set(), player.link.close(timeout=0.2)))
        t.start()
        t.join(2)
        release.set()
        self.assertFalse(t.is_alive())


# ---------------------------------------------------------------------------------------------------------------
# queue side
# ---------------------------------------------------------------------------------------------------------------

class QueueStreamLifecycleTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc.pause_all()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        self.stream = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": True}, title="Clip")
        self.sid = self.stream.id
        self.folder = self.svc.store.item_dir(self.sid)

    def send(self, **msg):
        return self.svc._on_message(msg)

    # --- stream_end -----------------------------------------------------------------------------------------------
    def test_stream_end_removes_the_record_the_folder_and_the_player_data(self):
        self.send(cmd="watching", id=self.sid)
        self.send(cmd="clock", id=self.sid, pos=3.0, playing=True)
        self.send(cmd="want", id=self.sid, pos=9.0)
        (self.folder / "audio.wav").write_bytes(b"x")
        self.assertTrue(self.folder.is_dir())
        self.assertEqual(self.send(cmd="stream_end", id=self.sid), {"ok": True})
        self.assertFalse(self.folder.exists())
        self.assertEqual(self.svc.store.stream_items(), [])
        self.assertIsNone(self.svc.store.get(self.sid))
        for table in (self.svc._watch, self.svc._subtitle_clocks, self.svc._subtitle_wants, self.svc._stream_created):
            self.assertNotIn(self.sid, table)

    def test_a_second_stream_end_and_unknown_ids_are_errors(self):
        self.send(cmd="stream_end", id=self.sid)
        self.assertEqual(self.send(cmd="stream_end", id=self.sid), {"ok": False, "error": "unknown id"})
        self.assertEqual(self.send(cmd="stream_end", id="nope"), {"ok": False, "error": "unknown id"})
        self.assertEqual(self.send(cmd="stream_end"), {"ok": False, "error": "unknown id"})

    def test_stream_end_never_touches_a_regular_item(self):
        item = self.add("http://h/regular")
        folder = self.svc.store.item_dir(item)
        folder.mkdir(parents=True, exist_ok=True)
        self.assertEqual(self.send(cmd="stream_end", id=item), {"ok": False, "error": "not a stream item"})
        self.assertIsNotNone(self.svc.store.get(item))
        self.assertTrue(folder.is_dir())
        self.assertIsNotNone(self.svc.store.get(self.sid))       # and not the stream item either

    def test_stream_end_of_the_old_item_after_a_new_stream_is_harmless(self):
        newer = self.svc.store.add_stream("https://x.test/w", "url", {}, {}, title="Newer")
        self.send(cmd="stream_end", id=self.sid)
        self.assertEqual([it.id for it in self.svc.store.stream_items()], [newer.id])

    # --- the heartbeat timeout ------------------------------------------------------------------------------------
    def test_constants(self):
        self.assertEqual((QS.STREAM_WATCH_EXPIRE, QS.STREAM_FIRST_WATCH_GRACE), (60.0, 120.0))

    def test_an_item_whose_watching_stopped_is_dropped_after_60_seconds(self):
        self.send(cmd="watching", id=self.sid)                  # at now = 1000
        self.svc._expire_streams()
        self.now[0] += 60.0
        self.svc._expire_streams()
        self.assertIsNotNone(self.svc.store.get(self.sid), "exactly 60 s is still alive")
        self.now[0] += 0.5
        self.svc._expire_streams()
        self.assertIsNone(self.svc.store.get(self.sid))
        self.assertFalse(self.folder.exists())
        self.assertNotIn(self.sid, self.svc._watch)

    def test_a_heartbeat_keeps_the_item_alive(self):
        for _ in range(5):
            self.send(cmd="watching", id=self.sid)
            self.now[0] += 50.0
            self.svc._expire_streams()
        self.assertIsNotNone(self.svc.store.get(self.sid))

    def test_an_item_that_never_got_a_watching_is_dropped_after_120_seconds(self):
        self.svc._expire_streams()                                # the queue first sees it now
        self.now[0] += 119.0
        self.svc._expire_streams()
        self.assertIsNotNone(self.svc.store.get(self.sid))
        self.now[0] += 2.0
        self.svc._expire_streams()
        self.assertIsNone(self.svc.store.get(self.sid))
        self.assertFalse(self.folder.exists())

    def test_the_first_watching_ends_the_creation_grace(self):
        self.svc._expire_streams()
        self.now[0] += 100.0
        self.send(cmd="watching", id=self.sid)
        self.now[0] += 100.0                                      # 200 s after creation, 100 s after the heartbeat
        self.svc._expire_streams()
        self.assertIsNone(self.svc.store.get(self.sid), "the 60 s rule applies now, not the 120 s creation grace")

    def test_the_timeout_uses_the_services_own_constants(self):
        self.svc.stream_watch_expire, self.svc.stream_first_watch_grace = 5.0, 7.0
        self.svc._expire_streams()
        self.now[0] += 8.0
        self.svc._expire_streams()
        self.assertIsNone(self.svc.store.get(self.sid))

    def test_the_scheduler_step_runs_the_timeout_and_spares_regular_items(self):
        item = self.add("http://h/regular")
        self.svc._watch[item] = self.now[0]
        self.svc._step()                                          # registers the stream item's creation time
        self.now[0] += 500.0
        self.svc._step()
        self.assertIsNone(self.svc.store.get(self.sid))
        self.assertIsNotNone(self.svc.store.get(item))

    # --- T4.2 B1: the busy player does not stop the stream item's own subtitle lease ------------------------------
    def _lease(self, item_id):
        from oh_streamer.queue.subtitle_host_gpu import RUN_STAGE_LEASE
        lease = QS._Run(item_id, 0)
        lease.stage, lease.subtitle_only = RUN_STAGE_LEASE, True
        return lease

    def _step_with_lease(self, lease, live_stream_item):
        svc = self.svc
        svc._stopping.set()                                       # park the service thread; _step is called by hand
        svc._wake.set()
        svc._thread.join(5)
        svc._stopping.clear()
        svc._paused_all = False
        svc._wk = lease
        live = {"title": "t", "pos": 1.0, "duration": 100.0, "stream_item": live_stream_item}
        self.send(cmd="player_busy", busy=True, live=live if live_stream_item else None)
        with mock.patch.object(svc, "_subtitle_executor_tick"), mock.patch.object(svc, "_maybe_start_download"):
            svc._step()
        svc._wk = None

    def test_step_keeps_the_stream_items_own_lease_while_the_player_is_busy(self):
        lease = self._lease(self.sid)
        self._step_with_lease(lease, self.sid)
        self.assertFalse(lease.cancel.is_set())

    def test_step_still_stops_other_leases_and_leases_without_a_live_report(self):
        other = self._lease("someone-else")
        self._step_with_lease(other, self.sid)
        self.assertTrue(other.cancel.is_set())
        own = self._lease(self.sid)
        self._step_with_lease(own, None)                          # busy player, no live report naming the stream
        self.assertTrue(own.cancel.is_set())

    # --- queue shutdown -------------------------------------------------------------------------------------------
    def test_stop_clears_every_stream_item_and_the_streams_folder(self):
        other = self.svc.store.add_stream("https://x.test/w", "url", {}, {}, title="Other")
        (self.folder / "audio.wav").write_bytes(b"x")
        streams_root = self.folder.parent
        self.assertTrue(streams_root.is_dir())
        self.svc.stop()
        self.assertEqual(self.svc.store.stream_items(), [])
        self.assertFalse(self.folder.exists())
        self.assertFalse(self.svc.store.item_dir(other.id).exists())
        self.assertFalse(streams_root.exists())

    def test_leftovers_of_a_crashed_run_are_cleared_at_the_next_start(self):
        leftover = self.svc.store.root / queue_store.STREAMS_DIR / "deadbeef"
        leftover.mkdir(parents=True)
        (leftover / "audio.wav").write_bytes(b"x")
        self.svc.stop()                                           # (a crash would have skipped this)
        leftover.mkdir(parents=True)
        (leftover / "audio.wav").write_bytes(b"x")
        reopened = queue_store.QueueStore(self.svc.store.root)
        self.assertFalse(leftover.exists())
        self.assertEqual(reopened.stream_items(), [])



# ---------------------------------------------------------------------------------------------------------------
# T3.1-C: removing a stream item stops its subtitle executor first
# ---------------------------------------------------------------------------------------------------------------

class FakeStreamExecutor:
    """A subtitle executor thread of a stream item: runs until stop() (or until `release` when it is `stubborn`)."""

    def __init__(self):
        self.stopped = threading.Event()
        self.release = threading.Event()
        self.stubborn = False

    def stop(self):
        self.stopped.set()

    def loop(self):
        if self.stubborn:
            self.release.wait(30)
        else:
            self.stopped.wait(30)


class QueueStreamRemovalStopsSubtitlesTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc.pause_all()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        self.stream = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": True}, title="Clip")
        self.sid = self.stream.id
        self.folder = self.svc.store.item_dir(self.sid)
        self.executors = []

    def tearDown(self):
        for ex in self.executors:
            ex.stopped.set()
            ex.release.set()
        super().tearDown()

    def start_executor(self, slot="_subtitle_cloud", item_id=None, stubborn=False, executor=None):
        ex = executor or FakeStreamExecutor()
        ex.stubborn = stubborn
        run = QS._Run(item_id or self.sid)
        run.executor = ex
        run.thread = threading.Thread(target=ex.loop, daemon=True)
        run.thread.start()
        setattr(self.svc, slot, run)
        self.executors.append(ex)
        return run, ex

    def host(self):
        from oh_streamer.queue.subtitle_host import QueueSubtitleHost
        return QueueSubtitleHost(self.svc, self.sid)

    RESULT = {"complete": False, "covered_s": 30.0, "srt": ["a.srt"], "cues": [1], "done_indices": [0]}

    def updates(self):
        return [m for m in self.players if m.get("cmd") == "subtitle_update"]

    def test_stream_end_stops_the_executor_thread_then_deletes_the_folder(self):
        run, ex = self.start_executor()
        (self.folder / "subtitles").mkdir()
        (self.folder / "subtitles" / "slice-plan.json").write_text("{}", encoding="utf-8")
        generation = self.svc._generations.get(self.sid, 0)
        self.assertEqual(self.svc._on_message({"cmd": "stream_end", "id": self.sid}), {"ok": True})
        self.assertTrue(ex.stopped.is_set())
        self.assertFalse(run.thread.is_alive())
        self.assertTrue(run.cancel.is_set())
        self.assertEqual(run.reason, "remove")
        self.assertIn(self.sid, self.svc._tombstones)
        self.assertEqual(self.svc._generations[self.sid], generation + 1)
        self.assertFalse(self.folder.exists())
        self.assertIsNone(self.svc.store.get(self.sid))

    def test_every_removal_path_goes_through_it(self):
        for how in ("heartbeat", "replaced by a new stream", "refused open"):
            with self.subTest(how=how):
                stream = self.svc.store.add_stream("https://x.test/w", "url", {}, {"subtitles": True}, title=how)
                run, ex = self.start_executor(item_id=stream.id)
                if how == "heartbeat":
                    self.svc._stream_created[stream.id] = self.svc._clock() - QS.STREAM_FIRST_WATCH_GRACE - 1
                    self.svc._expire_streams()
                else:
                    self.svc._drop_stream_item(stream.id)         # what _send_open's replace / refuse paths call
                self.assertTrue(ex.stopped.is_set())
                self.assertFalse(run.thread.is_alive())
                self.assertIsNone(self.svc.store.get(stream.id))
                self.assertIn(stream.id, self.svc._tombstones)
                self.svc._subtitle_cloud = None

    def test_stop_stops_the_stream_executor_before_clearing_streams(self):
        run, ex = self.start_executor()
        self.svc.stop()
        self.assertTrue(ex.stopped.is_set())
        self.assertFalse(run.thread.is_alive())
        self.assertEqual(self.svc.store.stream_items(), [])
        self.assertFalse(self.folder.exists())

    def test_the_stream_slot_is_stopped_on_removal(self):
        """T3.1-A: the dedicated stream slot `_subtitle_stream_cloud` is listed by _subtitle_runs(), so removal stops it."""
        run, ex = self.start_executor(slot="_subtitle_stream_cloud")
        self.assertIn(run, self.svc._subtitle_runs())
        self.assertTrue(self.svc._drop_stream_item(self.sid))
        self.assertTrue(ex.stopped.is_set())
        self.assertFalse(run.thread.is_alive())
        self.assertEqual(run.reason, "remove")

    def test_an_executor_of_another_item_is_left_alone(self):
        other = self.svc.store.add_stream("https://x.test/w", "url", {}, {"subtitles": True}, title="Other")
        run, ex = self.start_executor(item_id=other.id)
        self.svc._drop_stream_item(self.sid)
        self.assertFalse(ex.stopped.is_set())
        self.assertTrue(run.thread.is_alive())
        self.assertNotIn(other.id, self.svc._tombstones)
        self.assertIsNotNone(self.svc.store.get(other.id))

    def test_the_executor_results_and_caches_of_the_item_are_dropped(self):
        self.svc._subtitle_results = {self.sid: {"cloud": object()}, "keep": {"cloud": object()}}
        self.svc._subtitle_plan_files[self.sid] = ((1, 1), object())
        self.svc._subtitle_sent_updates = {self.sid: "sig", "keep": "sig"}
        self.svc._subtitle_play_offsets = {self.sid: 1.0}
        self.svc._drop_stream_item(self.sid)
        self.assertNotIn(self.sid, self.svc._subtitle_results)
        self.assertIn("keep", self.svc._subtitle_results)
        self.assertNotIn(self.sid, self.svc._subtitle_plan_files)
        self.assertEqual(self.svc._subtitle_sent_updates, {"keep": "sig"})
        self.assertEqual(self.svc._subtitle_play_offsets, {})

    def test_an_executor_that_does_not_stop_in_time_does_not_keep_the_item(self):
        run, ex = self.start_executor(stubborn=True)
        with mock.patch.object(QS.QueueService, "STREAM_REMOVE_JOIN", 0.1):
            self.assertTrue(self.svc._drop_stream_item(self.sid))
        self.assertTrue(run.thread.is_alive())                    # it was told to stop, it just has not yet
        self.assertTrue(ex.stopped.is_set())
        self.assertIsNone(self.svc.store.get(self.sid))
        self.assertFalse(self.folder.exists())

    def test_a_failing_executor_stop_does_not_keep_the_item(self):
        with mock.patch.object(self.svc, "_subtitle_stop_executors", side_effect=RuntimeError("boom")):
            self.assertTrue(self.svc._drop_stream_item(self.sid))
        self.assertIsNone(self.svc.store.get(self.sid))

    def test_removing_an_unknown_id_is_false_and_leaves_no_tombstone(self):
        self.assertFalse(self.svc._remove_stream("nope"))
        self.assertNotIn("nope", self.svc._tombstones)

    # --- nothing is written or sent after the tombstone ----------------------------------------------------------
    def test_publish_before_the_tombstone_writes_and_tells_the_player(self):
        self.host().publish(dict(self.RESULT))
        self.assertEqual(self.svc.store.get(self.sid).subtitle["status"], "partial")
        self.assertEqual(len(self.updates()), 1)
        self.assertEqual(self.updates()[0]["item_id"], self.sid)

    def test_publish_after_the_tombstone_writes_nothing_and_sends_nothing(self):
        host = self.host()
        before = dict(self.svc.store.get(self.sid).subtitle)
        with self.svc._lock:
            self.svc._tombstones.add(self.sid)                   # the first thing _remove_stream does
        host.publish(dict(self.RESULT))
        self.assertEqual(self.svc.store.get(self.sid).subtitle, before)
        self.assertEqual(self.updates(), [])

    def test_a_publish_racing_the_removal_is_dropped(self):
        """The executor thread publishes while _remove_stream waits for it: no write, no subtitle_update."""
        host = self.host()
        published = threading.Event()
        test = self

        class Racing(FakeStreamExecutor):
            def stop(self):
                host.publish(dict(test.RESULT))                  # a last result lands during the stop
                published.set()
                super().stop()

        run, ex = self.start_executor(executor=Racing())
        before = dict(self.svc.store.get(self.sid).subtitle)
        self.svc._drop_stream_item(self.sid)
        self.assertTrue(published.is_set())
        self.assertEqual(self.updates(), [])
        self.assertEqual(before.get("status"), "queued")

    def test_publish_after_the_item_is_gone_sends_nothing(self):
        host = self.host()
        self.svc._drop_stream_item(self.sid)
        host.publish(dict(self.RESULT))
        self.assertEqual(self.updates(), [])

    def test_a_regular_item_is_still_updated_and_announced(self):
        item_id = self.add("http://h/regular")
        from oh_streamer.queue.subtitle_host import QueueSubtitleHost
        self.svc.store.update(item_id, subtitle={"status": "queued"})
        QueueSubtitleHost(self.svc, item_id).publish(dict(self.RESULT))
        self.assertEqual(self.svc.store.get(item_id).subtitle["status"], "partial")
        self.assertEqual([m["item_id"] for m in self.updates()], [item_id])


if __name__ == "__main__":
    unittest.main()
