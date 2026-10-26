"""Offline unit tests for the player side of the background queue (run: python -P streamer/tests/test_queue_player.py).

Imports oh_player (torch + lada, no models / GPU / mpv): the pure parts (add payload, busy heartbeat with a fake
clock, queue link, lookup, processed-message handling, watched threshold) and run_player driven with fake mpv / IPC /
queue link objects (processed mode, add to queue, busy reports)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import argparse
import os
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from unittest import mock
from oh_streamer.engines.base import ModelChoice  # noqa: E402
import oh_streamer.player.oh_player as L  # noqa: E402
import oh_streamer.player.open_input as open_input  # noqa: E402
import oh_streamer.player.player_settings as PS  # noqa: E402
import oh_streamer.queue.queue_ipc as queue_ipc  # noqa: E402


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class BusyReporterTest(unittest.TestCase):
    def setUp(self):
        self.clock, self.sent = FakeClock(), []
        self.rep = L.BusyReporter(self.sent.append, self.clock, interval=10.0)

    def test_quiet_while_idle(self):
        for _ in range(5):
            self.rep.update(False)
            self.clock.t += 20
        self.assertEqual(self.sent, [])

    def test_true_once_then_heartbeat_then_false_once(self):
        self.rep.update(True)
        self.assertEqual(self.sent, [True])
        self.clock.t += 9.9
        self.rep.update(True)
        self.assertEqual(self.sent, [True])            # not due yet
        self.clock.t += 0.2
        self.rep.update(True)
        self.assertEqual(self.sent, [True, True])      # heartbeat after 10 s
        self.clock.t += 1
        self.rep.update(True)
        self.assertEqual(len(self.sent), 2)
        self.rep.update(False)
        self.rep.update(False)
        self.assertEqual(self.sent, [True, True, False])

    def test_restart_sends_true_at_once(self):
        self.rep.update(True)
        self.clock.t += 1
        self.rep.update(False)
        self.clock.t += 1
        self.rep.update(True)
        self.assertEqual(self.sent, [True, False, True])

    def test_default_interval_is_the_protocol_heartbeat(self):
        self.assertEqual(L.BusyReporter(lambda b: None).interval, queue_ipc.BUSY_HEARTBEAT)


class QueueLinkTest(unittest.TestCase):
    def test_none_reply_logs_once_and_does_not_raise(self):
        logs = []
        link = L.QueueLink(send=lambda p, t: None, log=logs.append)
        for _ in range(3):
            self.assertIsNone(link.request({"cmd": "lookup"}))
        self.assertEqual(len(logs), 1)

    def test_logs_again_after_the_queue_was_back(self):
        logs, replies = [], iter([None, {"ok": True}, None])
        link = L.QueueLink(send=lambda p, t: next(replies), log=logs.append)
        link.request({"cmd": "x"})
        link.request({"cmd": "x"})
        link.request({"cmd": "x"})
        self.assertEqual(len(logs), 2)

    def test_exception_in_send_is_swallowed(self):
        def boom(p, t):
            raise OSError("x")
        link = L.QueueLink(send=boom)
        self.assertIsNone(link.request({"cmd": "x"}))

    def test_post_runs_off_thread_in_order_and_close_flushes(self):
        main, sent, got = threading.get_ident(), [], []
        gate = threading.Event()

        def send(payload, timeout):
            sent.append((payload["n"], threading.get_ident()))
            gate.wait(2)
            return {"ok": True, "n": payload["n"]}

        link = L.QueueLink(send=send)
        for n in range(3):
            link.post({"cmd": "x", "n": n}, got.append)       # returns at once although send blocks
        gate.set()
        link.close(timeout=3)
        self.assertEqual([n for n, _ in sent], [0, 1, 2])
        self.assertTrue(all(tid != main for _, tid in sent))
        self.assertEqual([r["n"] for r in got], [0, 1, 2])

    def test_close_without_a_sender_is_a_noop(self):
        L.QueueLink(send=lambda p, t: None).close()

    def test_default_send_uses_queue_ipc(self):
        with mock.patch.object(queue_ipc, "send_to_queue", return_value={"ok": True}) as m:
            L.QueueLink(name="abc").request({"cmd": "show"}, 1.5)
        m.assert_called_once_with({"cmd": "show"}, "abc", timeout=1.5)


class AddPayloadTest(unittest.TestCase):
    def test_settings_are_the_restart_relevant_ones_with_fp16_resolved(self):
        cfg = dict(PS.DEFAULTS, interpolation="auto", compare="on", vsr="off", seek_prebuffer=2.0)
        p = L.build_add_payload("https://x/page", {"Cookie": "a=1"}, 720, "T", cfg, True)
        self.assertEqual(p["cmd"], "add")
        self.assertEqual((p["input"], p["headers"], p["quality"], p["title"]),
                         ("https://x/page", {"Cookie": "a=1"}, 720, "T"))
        self.assertEqual(p["settings"]["fp16"], True)
        self.assertEqual(p["settings"]["interpolation"], "auto")
        self.assertEqual(p["settings"]["compare"], "on")
        for live in PS.LIVE_KEYS:
            self.assertNotIn(live, p["settings"])
        self.assertEqual(set(p["settings"]), set(PS.KEYS) - PS.LIVE_KEYS)

    def test_file_input_is_made_absolute(self):
        p = L.build_add_payload("some.mp4", {}, None, None, dict(PS.DEFAULTS), False)
        self.assertEqual(p["input"], os.path.abspath("some.mp4"))
        self.assertIs(p["settings"]["fp16"], False)

    def test_quality_is_the_playing_height_only_when_there_is_a_choice(self):
        quals = [{"height": 1080}, {"height": 720}]
        self.assertEqual(L.queue_quality(quals, quals[1]), 720)
        self.assertIsNone(L.queue_quality(quals[:1], quals[0]))
        self.assertIsNone(L.queue_quality(None, None))


class ProcessedMessageTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "output.mkv")
        with open(self.path, "wb") as f:
            f.write(b"x")

    def tearDown(self):
        self.dir.cleanup()

    def test_valid_message(self):
        po = L.processed_from_message({"input": self.path, "headers": {}, "processed": True, "queue_id": "q7",
                                       "title": "My video"}, probe_layers=lambda p: "oh:processed,oh:original")
        self.assertEqual((po.path, po.queue_id, po.title), (self.path, "q7", "My video"))
        self.assertEqual(po.layers, "oh:processed,oh:original")
        self.assertFalse(po.from_lookup)

    def test_start_option(self):
        probe = lambda p: ""  # noqa: E731
        self.assertEqual(L.processed_from_message({"input": self.path, "processed": True}, probe).start, 0.0)
        self.assertEqual(L.processed_from_message({"input": self.path, "processed": True, "start": "31.5"}, probe).start, 31.5)
        for bad in (-2, "x", float("nan"), None):
            self.assertEqual(L.processed_from_message({"input": self.path, "processed": True, "start": bad}, probe).start, 0.0)

    def test_parse_start(self):
        self.assertIsNone(queue_ipc.parse_start(None))
        self.assertEqual(queue_ipc.parse_start(" 5 "), 5.0)
        self.assertEqual(queue_ipc.parse_start(50, 20), 20.0)
        self.assertEqual(queue_ipc.parse_start(50, 0), 50.0)
        self.assertIsNone(queue_ipc.parse_start(float("-inf")))

    def test_defaults_title_and_id(self):
        po = L.processed_from_message({"input": self.path, "processed": True}, probe_layers=lambda p: "")
        self.assertEqual((po.title, po.queue_id, po.layers), ("output.mkv", None, ""))

    def test_rejects_missing_file_wrong_flag_and_bad_types(self):
        probe = lambda p: ""  # noqa: E731
        self.assertIsNone(L.processed_from_message({"input": self.path + ".nope", "processed": True}, probe))
        self.assertIsNone(L.processed_from_message({"input": self.path}, probe))
        self.assertIsNone(L.processed_from_message({"input": self.path, "processed": "yes"}, probe))
        self.assertIsNone(L.processed_from_message({"input": 5, "processed": True}, probe))
        self.assertIsNone(L.processed_from_message({"input": "  ", "processed": True}, probe))

    def test_compare_titles(self):
        def probe(*titles):
            return {"streams": [{"codec_type": "video", "tags": {"title": t} if t else {}} for t in titles]
                    + [{"codec_type": "audio", "tags": {"title": "oh:original"}}]}
        self.assertEqual(L.compare_titles_from_probe(probe("oh:processed", "oh:original")),
                         "oh:processed,oh:original")
        self.assertEqual(L.compare_titles_from_probe(probe("oh:processed")), "")       # one track: no compare
        self.assertEqual(L.compare_titles_from_probe(probe(None, "other")), "")
        self.assertEqual(L.compare_titles_from_probe({}), "")

    def test_watched_threshold(self):
        self.assertFalse(L.processed_watched(None, None, False))
        self.assertFalse(L.processed_watched(80, 100, False))
        self.assertTrue(L.processed_watched(90, 100, False))
        self.assertTrue(L.processed_watched(10, 100, True))     # eof
        self.assertFalse(L.processed_watched(5, 0, False))


class LookupTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.dir.name, "output.mkv")
        with open(self.out, "wb") as f:
            f.write(b"x")
        self.probe = lambda p: ""  # noqa: E731

    def tearDown(self):
        self.dir.cleanup()

    def link(self, reply, seen=None):
        def send(payload, timeout):
            if seen is not None:
                seen.append((payload, timeout))
            return reply
        return L.QueueLink(send=send)

    def test_done_item_with_output_plays_processed(self):
        seen = []
        link = self.link({"ok": True, "item": {"id": "a1", "state": "done", "output": self.out, "progress": {}}}, seen)
        po = L.queue_lookup(link, "https://x/v", 720, probe_layers=self.probe)
        self.assertEqual((po.path, po.queue_id, po.from_lookup), (self.out, "a1", True))
        self.assertEqual(seen[0][0], {"cmd": "lookup", "input": "https://x/v", "quality": 720})
        self.assertLessEqual(seen[0][1], 1.0)               # short timeout: never delays opening

    def test_other_states_queue_not_running_or_missing_output_play_live(self):
        for item in ({"id": "a", "state": "processing", "output": self.out},
                     {"id": "a", "state": "done", "output": self.out + ".gone"},
                     {"id": "a", "state": "done"}, None):
            self.assertIsNone(L.queue_lookup(self.link({"ok": True, "item": item}), "https://x/v", None,
                                             probe_layers=self.probe), item)
        self.assertIsNone(L.queue_lookup(self.link(None), "https://x/v", None, probe_layers=self.probe))

    def test_opener_plays_the_processed_version_without_resolving(self):
        po = L.ProcessedOpen(self.out, "T", "a1", from_lookup=True)
        opener = L.Opener(argparse.Namespace(), None)
        opener.lookup = lambda oi: po
        with mock.patch.object(L, "open_source", side_effect=AssertionError("must not resolve")):
            opener.submit(open_input.OpenInput("https://x/v"), "貼上")
            self.assertTrue(wait_for(lambda: not opener.busy))
        opened, err, _took, _label = opener.poll()
        self.assertIs(opened, po)
        self.assertIsNone(err)

    def test_opener_resolves_normally_when_lookup_finds_nothing_or_fails(self):
        for lookup in (lambda oi: None, mock.Mock(side_effect=RuntimeError("boom"))):
            opener = L.Opener(argparse.Namespace(), None)
            opener.lookup = lookup
            sentinel = L.Opened(title="live")
            with mock.patch.object(L, "open_source", return_value=sentinel), mock.patch.object(L, "log"):
                opener.submit(open_input.OpenInput("https://x/v"), "貼上")
                self.assertTrue(wait_for(lambda: not opener.busy))
            self.assertIs(opener.poll()[0], sentinel)


# ---------------------------------------------------------------------------------------------------------------
# run_player with fakes
# ---------------------------------------------------------------------------------------------------------------

class FakePlayer:
    def __init__(self):
        self.pid, self.exit = 4242, False

    def poll(self):
        return 0 if self.exit else None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.exit = True


class FakeIpc:
    def __init__(self, *_a):
        self.props, self.sets, self.commands = {}, [], []

    def get(self, name, default=None):
        v = self.props.get(name, default)
        return default if v is None else v

    def set(self, name, value):
        self.sets.append((name, value))
        self.props[name] = value

    def command(self, *a):
        self.commands.append(a)
        if a[:2] == ("script-message", "oh-reset"):
            self.props["user-data/oh/reset-ack"] = a[2]
        elif a[0] == "loadfile":
            self.props["path"] = a[1]
        elif a[0] == "stop":
            self.props["path"] = None

    def close(self):
        pass


class FakeCast:
    active = None

    def info_snapshot(self):
        return None

    def __init__(self, *_a):
        pass

    def poll(self):
        pass

    def close(self):
        pass


class FakeGpu:
    def start(self):
        pass

    def set_active(self, _on):
        pass

    def snapshot(self):
        return None

    def close(self):
        pass


class FakeInstance:
    def __init__(self):
        self.on_message, self.closed = None, False

    def serve(self, on_message, log):
        self.on_message = on_message

    def close(self):
        self.closed = True


class FakeLink:
    """QueueLink stand-in: records what the controller posts; `reply` answers the callbacks (None = queue off)."""
    reply = {"ok": True, "id": "n1"}
    instances = []

    def __init__(self, *a, **k):
        self.posts, self.closed = [], False
        FakeLink.instances.append(self)

    def post(self, payload, callback=None):
        self.posts.append(payload)
        if callback is not None:
            callback(FakeLink.reply)
        return True

    def request(self, payload, timeout=2.0):
        return None

    def close(self, timeout=2.0):
        self.closed = True


class AliveThread:
    def is_alive(self):
        return True


class FakeMeta:
    duration, video_width, video_height, video_fps_exact = 60.0, 1920, 1080, 30


class FakeSource:
    meta = FakeMeta()

    def describe(self):
        return "fake source"


class FakeSession:
    def __init__(self, restorer, source, encoder, max_frames):
        self.source, self.meta, self.encoder = source, source.meta, encoder
        self.target, self.url, self.stats, self.doubler = 0.0, "tcp://127.0.0.1:1", None, None
        self.frames_total, self.fps = 0, 30.0
        self.thread = None

    def start(self, start, gate_release_s=0.0):
        self.target, self.thread = start, AliveThread()

    def ready(self):
        return False

    def stop(self, fast=False):
        self.thread = None
        return 0.0

    def release_restorer(self):
        pass


class FakeQualityControl:
    def __init__(self, quals, chosen, *_a):
        self.qualities, self.current = quals, chosen


class FakeLada:
    models, max_clip_length = ModelChoice("det", "res", True), 180

    def open(self, source):
        return object()


class FakeSettings:
    cfg = dict(PS.DEFAULTS, encoder="libx264", interpolation="off", compare="off")
    fp16_on = True

    def menu(self):
        return [{"key": "fake"}]

    def resolve_encoder(self, name):
        return name


class RunPlayerQueueTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.dir.name, "output.mkv")
        with open(self.out, "wb") as f:
            f.write(b"x")
        FakeLink.instances.clear()
        FakeLink.reply = {"ok": True, "id": "n1"}
        self.args = argparse.Namespace(mpv=None, initial_prebuffer=1.0, prebuffer=3.0, seek_prebuffer=1.0, duration=0.0,
                                       thumbnails=False, settings_file=None, output="play", instance="unit-test",
                                       cast=None, hls_only=False, processed=False, queue_id=None)
        cfg = {"interpolation": "off", "vsr": "off", "smooth_playback": "off"}
        self.engine = L.Engine(self.args, {}, cfg, ["det"], ["res"])
        self.engine.lada, self.engine.sc = FakeLada(), FakeSettings()
        self.engine.ready.set()
        self.player, self.ipcs, self.logs = FakePlayer(), [], []
        self.thread = None

    def tearDown(self):
        self.player.exit = True
        if self.thread is not None:
            self.thread.join(5)
        self.dir.cleanup()

    def run_player(self, **kw):
        def make_ipc(*a):
            self.ipcs.append(FakeIpc(*a))
            return self.ipcs[-1]

        patches = [mock.patch.object(L, "find_mpv", lambda _m: "mpv"),
                   mock.patch.object(L, "start_mpv", lambda *a, **k: self.player),
                   mock.patch.object(L, "MpvIpc", make_ipc), mock.patch.object(L, "CastManager", FakeCast),
                   mock.patch.object(L.info_panel, "GpuMonitor", FakeGpu), mock.patch.object(L, "Session", FakeSession),
                   mock.patch.object(L, "QualityControl", FakeQualityControl),
                   mock.patch.object(L, "QueueLink", FakeLink), mock.patch.object(L, "log", self.logs.append)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.result = []
        self.thread = threading.Thread(target=lambda: self.result.append(L.run_player(self.args, self.engine, **kw)),
                                       daemon=True)
        self.thread.start()
        self.assertTrue(wait_for(lambda: self.ipcs and FakeLink.instances), "run_player did not start")
        return self.ipcs[0], FakeLink.instances[0]

    def stop(self):
        self.player.exit = True
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.result, [0], self.logs)

    def test_processed_message_plays_the_file_without_a_session_and_reports_watched(self):
        inst = FakeInstance()
        ipc, link = self.run_player(instance=inst)
        self.assertTrue(wait_for(lambda: inst.on_message is not None))
        reply = inst.on_message({"input": self.out, "headers": {}, "processed": True, "queue_id": "q9", "title": "Vid"})
        self.assertEqual(reply, {"ok": True})
        self.assertTrue(wait_for(lambda: any(c[0] == "loadfile" for c in ipc.commands)), self.logs)
        load = [c for c in ipc.commands if c[0] == "loadfile"][-1]
        self.assertEqual(load[1:3], (self.out, "replace"))
        self.assertIn(("user-data/oh/processed", "1"), ipc.sets)
        self.assertIn(("user-data/oh/expect-load", self.out), ipc.sets)
        self.assertEqual([v for k, v in ipc.sets if k == "user-data/oh/compare_layers"][-1], "")
        self.assertEqual([v for k, v in ipc.sets if k == "force-media-title"][-1], "Vid")
        self.assertEqual([v for k, v in ipc.sets if k == "user-data/oh/active"][-1], "1")
        self.assertFalse([p for p in link.posts if p["cmd"] == "watched"])
        ipc.props.update({"time-pos": 50.0, "duration": 100.0})
        time.sleep(0.8)
        self.assertFalse([p for p in link.posts if p["cmd"] == "watched"])       # 50 %: not yet
        ipc.props["time-pos"] = 91.0
        self.assertTrue(wait_for(lambda: any(p["cmd"] == "watched" for p in link.posts)), self.logs)
        time.sleep(0.8)
        self.assertEqual([p for p in link.posts if p["cmd"] == "watched"], [{"cmd": "watched", "id": "q9"}])   # once
        # no restoring happened, so the queue was never told "busy"
        self.assertFalse([p for p in link.posts if p["cmd"] == "player_busy"])
        # the add key on a processed file: only a note
        ipc.props["user-data/oh/queue-add"] = "1"
        self.assertTrue(wait_for(lambda: ("show-text", "已是處理過的影片", 2500) in ipc.commands))
        self.stop()
        self.assertTrue(link.closed)
        self.assertTrue(inst.closed)

    def test_processed_file_info_panel_shows_the_video(self):
        inst = FakeInstance()
        ipc, _link = self.run_player(instance=inst)
        self.assertTrue(wait_for(lambda: inst.on_message is not None))
        inst.on_message({"input": self.out, "headers": {}, "processed": True, "queue_id": "q9", "title": "Vid"})
        self.assertTrue(wait_for(lambda: any(c[0] == "loadfile" for c in ipc.commands)), self.logs)
        ipc.props.update({"time-pos": 12.5, "duration": 7224.0, "video-params/w": 1280, "video-params/h": 720,
                          "container-fps": 59.94, "video-format": "hevc", "demuxer-cache-time": 40.0,
                          "user-data/oh/info-visible": True})

        def panel():
            vals = [v for k, v in ipc.sets if k == "user-data/oh/info"]
            if not vals:
                return None
            info = L.json.loads(vals[-1])
            return {r["k"]: r["v"] for s in info["sections"] for r in s["rows"] if s["title"] in ("影片", "處理狀態")}

        self.assertTrue(wait_for(lambda: (panel() or {}).get("解析度") == "1280x720"), panel())
        rows = panel()
        self.assertEqual(rows["標題"], "Vid")
        self.assertEqual(rows["影格率"], "59.940 fps")
        self.assertEqual(rows["來源編碼"], "hevc")
        self.assertEqual(rows["長度"], "2:00:24")
        self.assertEqual(rows["處理速度"], "已處理到結尾")
        self.assertEqual(rows["預處理緩衝"], "27.5 秒")
        self.stop()

    def test_processed_message_for_a_missing_file_is_refused(self):
        inst = FakeInstance()
        ipc, _link = self.run_player(instance=inst)
        self.assertTrue(wait_for(lambda: inst.on_message is not None))
        self.assertEqual(inst.on_message({"input": self.out + ".gone", "processed": True}),
                         {"ok": False, "error": "file not found"})
        time.sleep(0.2)
        self.assertFalse([c for c in ipc.commands if c[0] == "loadfile"])
        self.stop()

    def test_processed_cli_flag_plays_the_initial_input(self):
        self.args.processed, self.args.queue_id = True, "q3"
        ipc, link = self.run_player(instance=FakeInstance(), initial_inputs=[(self.out, {}, 0.0)])
        self.assertTrue(wait_for(lambda: any(c[0] == "loadfile" for c in ipc.commands)), self.logs)
        ipc.props.update({"time-pos": 1.0, "duration": 100.0, "eof-reached": True})
        self.assertTrue(wait_for(lambda: any(p == {"cmd": "watched", "id": "q3"} for p in link.posts)), self.logs)
        self.stop()

    def test_seek_request_with_nothing_loaded_reloads_the_processed_file(self):
        inst = FakeInstance()
        ipc, _link = self.run_player(instance=inst)
        self.assertTrue(wait_for(lambda: inst.on_message is not None))
        inst.on_message({"input": self.out, "processed": True})
        self.assertTrue(wait_for(lambda: any(c[0] == "loadfile" for c in ipc.commands)))
        ipc.props["path"] = None
        ipc.props["idle-active"] = True              # a dropped link had replaced the file
        ipc.props["user-data/oh/seek-request"] = "4:37.500"
        self.assertTrue(wait_for(lambda: len([c for c in ipc.commands if c[0] == "loadfile"]) == 2), self.logs)
        again = [c for c in ipc.commands if c[0] == "loadfile"][-1]
        self.assertEqual(again, ("loadfile", self.out, "replace", -1, "start=37.500"))
        self.assertEqual(ipc.props["user-data/oh/served"], "4:37.500")
        self.stop()

    def test_live_session_reports_busy_and_adds_to_the_queue(self):
        opened = L.Opened(input="https://example.com/watch?v=1", user_headers={"Cookie": "a=1"},
                          quals=[{"id": "source", "label": "1080p", "height": 1080},
                                 {"id": "t", "label": "720p", "height": 720}],
                          chosen={"id": "t", "label": "720p", "height": 720}, source=FakeSource(), title="Live video",
                          source_text="x")
        ipc, link = self.run_player(first=opened)
        self.assertTrue(wait_for(lambda: any({k: v for k, v in p.items() if k != "live"} ==
                                             {"cmd": "player_busy", "busy": True, "pid": os.getpid()} for p in link.posts)),
                        self.logs)
        # T3.5: the same heartbeat carries the live-stream report (a stream without a subtitle item: stream_item None)
        self.assertTrue(wait_for(lambda: any((p.get("live") or {}).get("title") == "Live video" for p in link.posts)), self.logs)
        ipc.props["user-data/oh/queue-add"] = "1"
        self.assertTrue(wait_for(lambda: any(p["cmd"] == "add" for p in link.posts)), self.logs)
        add = [p for p in link.posts if p["cmd"] == "add"][0]
        self.assertEqual((add["input"], add["headers"], add["quality"], add["title"]),
                         ("https://example.com/watch?v=1", {"Cookie": "a=1"}, 720, "Live video"))
        self.assertEqual(add["settings"]["fp16"], True)
        self.assertNotIn("vsr", add["settings"])
        self.assertTrue(wait_for(lambda: ("show-text", "已加入處理佇列", 2500) in ipc.commands), self.logs)
        FakeLink.reply = None                         # queue app not running
        ipc.props["user-data/oh/queue-add"] = "2"
        self.assertTrue(wait_for(lambda: ("show-text", "處理佇列未開啟", 2500) in ipc.commands), self.logs)
        self.stop()
        busy = [p["busy"] for p in link.posts if p["cmd"] == "player_busy"]
        self.assertEqual(busy[-1], False)             # exit: busy=false goes out before the link closes
        lives = [p["live"] for p in link.posts if p["cmd"] == "player_busy" and "live" in p]
        self.assertIsNone(lives[-1])                  # ... and the report is cleared (the 串流中 card goes)
        self.assertTrue(link.closed)


class LadaTagsProbeTest(unittest.TestCase):
    """lada_tags_from_probe / compare_setup_from_probe / processed_open on hand-built ffprobe results."""

    FULL = {"OH_SOURCE_FPS": "30000/1001", "OH_RATE_MULT": "2", "OH_VIDEO_START": "0.033367",
            "OH_COMPARE": "1", "OH_ORIGINAL": "copy"}

    @staticmethod
    def probe(*titles, tags=None):
        streams = [{"codec_type": "video", "tags": {"title": t}} for t in titles]
        fmt = {"tags": dict(tags)} if tags is not None else {}
        return {"streams": streams, "format": fmt}

    def test_lada_tags_full_set(self):
        tags = L.lada_tags_from_probe(self.probe(tags=self.FULL))
        self.assertEqual(tags, {"source_fps": Fraction(30000, 1001), "rate_mult": 2, "video_start": 0.033367,
                                "compare": True, "original": "copy", "engine": "lada", "engine_version": None})

    def test_lada_tags_lowercase_names(self):
        lower = {k.lower(): v for k, v in self.FULL.items()}
        self.assertEqual(L.lada_tags_from_probe(self.probe(tags=lower)), L.lada_tags_from_probe(self.probe(tags=self.FULL)))

    def test_lada_tags_absent_is_none(self):
        self.assertIsNone(L.lada_tags_from_probe(self.probe(tags={"title": "movie"})))
        self.assertIsNone(L.lada_tags_from_probe(self.probe()))
        self.assertIsNone(L.lada_tags_from_probe({}))

    def test_lada_tags_bad_numbers_are_none(self):
        for key, bad in (("OH_SOURCE_FPS", "0"), ("OH_SOURCE_FPS", "abc"), ("OH_RATE_MULT", "3"),
                         ("OH_RATE_MULT", "abc"), ("OH_VIDEO_START", "nan")):
            with self.subTest(key=key, value=bad):
                self.assertIsNone(L.lada_tags_from_probe(self.probe(tags={**self.FULL, key: bad})))

    def test_lada_tags_unknown_original(self):
        tags = L.lada_tags_from_probe(self.probe(tags={**self.FULL, "OH_ORIGINAL": "weird"}))
        self.assertIsNone(tags["original"])
        self.assertEqual(tags["rate_mult"], 2)

    def test_compare_setup_two_tracks_with_tags(self):
        probe = self.probe("oh:processed", "oh:original", tags=self.FULL)
        layers, derive, tags = L.compare_setup_from_probe(probe)
        self.assertEqual(layers, "oh:processed,oh:original")
        self.assertEqual(derive, "0.033367,30000/1001")
        self.assertEqual(tags, L.lada_tags_from_probe(probe))

    def test_compare_setup_one_track_derived(self):
        layers, derive, _ = L.compare_setup_from_probe(self.probe("oh:processed", tags=self.FULL))
        self.assertEqual((layers, derive), ("oh:processed", "0.033367,30000/1001"))

    def test_compare_setup_old_file_three_tracks(self):
        layers, derive, tags = L.compare_setup_from_probe(
            self.probe("oh:processed", "oh:original", "oh:restored"))
        self.assertEqual((layers, derive, tags), ("oh:processed,oh:original,oh:restored", "", None))

    def test_compare_setup_rate_mult_one_no_derive(self):
        layers, derive, tags = L.compare_setup_from_probe(
            self.probe("oh:processed", "oh:original", tags={**self.FULL, "OH_RATE_MULT": "1"}))
        self.assertEqual((layers, derive), ("oh:processed,oh:original", ""))
        self.assertEqual(tags["rate_mult"], 1)

    def test_compare_setup_compare_off(self):
        tags = {**self.FULL, "OH_COMPARE": "0"}
        layers, derive, _ = L.compare_setup_from_probe(self.probe("oh:processed", "oh:original", tags=tags))
        self.assertEqual((layers, derive), ("oh:processed,oh:original", ""))
        layers, derive, _ = L.compare_setup_from_probe(self.probe("oh:processed", tags=tags))
        self.assertEqual((layers, derive), ("", ""))

    def test_processed_open_with_tuple_hook(self):
        probe = self.probe("oh:processed", "oh:original", tags=self.FULL)
        seen = []

        def hook(path):
            seen.append(path)
            return L.compare_setup_from_probe(probe)

        po = L.processed_open("C:/out/output.mkv", "My video", "q9", probe=hook, from_lookup=True)
        self.assertEqual(seen, ["C:/out/output.mkv"])
        self.assertEqual((po.path, po.title, po.queue_id, po.from_lookup), ("C:/out/output.mkv", "My video", "q9", True))
        self.assertEqual(po.layers, "oh:processed,oh:original")
        self.assertEqual(po.derive, "0.033367,30000/1001")
        self.assertEqual(po.tags["source_fps"], Fraction(30000, 1001))

    def test_processed_open_with_string_hook(self):
        po = L.processed_open("C:/out/output.mkv", None, None, probe=lambda p: "oh:processed,oh:original")
        self.assertEqual(po.layers, "oh:processed,oh:original")
        self.assertEqual(po.derive, "")
        self.assertIsNone(po.tags)
        self.assertEqual(po.title, "output.mkv")
        self.assertIsNone(po.queue_id)
        self.assertFalse(po.from_lookup)


if __name__ == "__main__":
    unittest.main()
