"""Focused lifecycle and event-routing checks for the extracted player controller."""
from pathlib import Path
import collections
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
for _root in (_STREAMER_ROOT, _SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from oh_streamer.player import controller as C  # noqa: E402


class FakeEndpoint:
    def __init__(self, events):
        self.events, self.path = events, "fake-mpv-endpoint"

    def close(self):
        self.events.append("endpoint:close")


class FakePlayer:
    def __init__(self, events, polls=(0,)):
        self.events, self.polls, self.poll_count = events, tuple(polls), 0
        self.pid = 4242

    def poll(self):
        value = self.polls[min(self.poll_count, len(self.polls) - 1)]
        self.poll_count += 1
        return value

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.events.append("player:kill")


class FakeIpc:
    def __init__(self, events, *_args):
        self.events, self.props = events, {}
        events.append("ipc:create")

    def get(self, name, default=None):
        return self.props.get(name, default)

    def set(self, name, value):
        self.props[name] = value

    def command(self, *_args):
        pass

    def close(self):
        self.events.append("ipc:close")


class FakeCast:
    def __init__(self, events, *_args):
        self.events, self.active = events, None
        events.append("cast:create")

    def poll(self):
        pass

    def close(self):
        self.events.append("cast:close")


class FakeGpu:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("gpu:start")

    def set_active(self, _active):
        pass

    def snapshot(self):
        return None

    def close(self):
        self.events.append("gpu:close")


class FakeSettings:
    cfg = {"interpolation": "off", "compare": "off", "smooth_playback": False, "vsr": "off"}

    def menu(self):
        return []


class FakeLada:
    def open(self, _source):
        return object()


class FakeEngine:
    def __init__(self):
        self.cfg = {"vsr": "off", "smooth_playback": False, "compare": "off", "interpolation": "off"}
        self.saved, self.sc, self.lada = {}, FakeSettings(), FakeLada()
        self.ready = threading.Event()
        self.error = None
        self.loading = False
        self.loads = 0

    def ensure_loaded(self):
        self.loads += 1

    def encoder(self):
        return "libx264"

    def release_interp(self):
        pass


class FakeSession:
    instances = []

    def __init__(self, events, _restorer, source, _encoder, _max_frames):
        self.events, self.source, self.meta = events, source, source.meta
        self.target = 0.0
        self.started = False
        events.append("session:create")
        FakeSession.instances.append(self)

    def start(self, _start, **_kwargs):
        self.started = True
        self.events.append("session:start")

    def stop(self):
        self.events.append("session:stop")
        return 0.0

    def release_restorer(self):
        self.events.append("session:release")


class FakeQualityControl:
    def __init__(self, _quals, chosen, *_args):
        self.qualities, self.current = [chosen], chosen


class FakeOpener:
    instances = []

    def __init__(self, *_args):
        self.busy = False
        self.submissions = []
        FakeOpener.instances.append(self)

    def submit(self, item, label, start):
        self.submissions.append((item, label, start, threading.get_ident()))
        self.busy = True

    def poll(self):
        return None

    def close(self):
        pass


class FakeLink:
    def __init__(self, events, *_args, **_kwargs):
        self.events = events

    def post(self, *_args, **_kwargs):
        return True

    def close(self):
        self.events.append("link:close")


class FakeBusyReporter:
    def __init__(self, *_args):
        pass

    def update(self, _busy):
        pass


def controller_args(**overrides):
    values = {
        "mpv": None,
        "initial_prebuffer": 0.1,
        "prebuffer": 1.0,
        "seek_prebuffer": 0.2,
        "duration": 0.0,
        "thumbnails": False,
        "settings_file": None,
        "instance": "test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def controller_deps(events=None, **overrides):
    """Supply inert adapters so these tests never import the Lada/torch facade or start GPU work."""
    events = events if events is not None else []
    values = {
        "BusyReporter": lambda *_args, **_kwargs: FakeBusyReporter(),
        "CastManager": lambda *_args, **_kwargs: FakeCast(events),
        "Fraction": object,
        "IDLE_HINT": "idle",
        "MpvEndpoint": lambda: FakeEndpoint(events),
        "MpvError": OSError,
        "MpvIpc": lambda *args: FakeIpc(events, *args),
        "Opener": FakeOpener,
        "PS": SimpleNamespace(ui_state=lambda _saved: {"volume": 70, "mute": False}, smooth_props=lambda _on: {}),
        "Path": Path,
        "ProcessedOpen": type("ProcessedOpen", (), {}),
        "QualityControl": FakeQualityControl,
        "QueueLink": lambda *_args, **_kwargs: FakeLink(events),
        "Session": lambda *args: FakeSession(events, *args),
        "ThumbnailWorker": lambda *_args: None,
        "_float_or": lambda value, default=0.0: default if value is None else float(value),
        "airplay_cast": SimpleNamespace(CastError=RuntimeError, short_reason=lambda error: str(error)),
        "buffer_progress": SimpleNamespace(estimate=lambda *_args, **_kwargs: (0.0, "", 0.0)),
        "build_add_payload": lambda *_args, **_kwargs: {},
        "collections": collections,
        "compare_stack": SimpleNamespace(LAYER_TITLES={}),
        "device_backend": SimpleNamespace(allocated_gib=lambda _device: 0.0),
        "find_mpv": lambda _requested: "mpv",
        "forget_sources": lambda: None,
        "gc": SimpleNamespace(collect=lambda: None),
        "info_panel": SimpleNamespace(GpuMonitor=lambda: FakeGpu(events), History=lambda: FakeHistory(),
                                       build_info=lambda snap: snap),
        "json": json,
        "log": lambda _message: None,
        "mpv_playback_snapshot": lambda *_args: {},
        "open_input": SimpleNamespace(parse=lambda text: SimpleNamespace(
            kind="url", target=text, headers={}, notes="", describe=lambda: "url")),
        "os": os,
        "passthrough_frames": lambda *_args: iter(()),
        "platform_support": SimpleNamespace(),
        "processed_from_message": lambda *_args: None,
        "processed_watched": lambda *_args: False,
        "subtitle_update_from_message": lambda *_args: None,
        "progressive_stream": SimpleNamespace(ProgressiveOpen=type("ProgressiveOpen", (), {})),
        "qualities": SimpleNamespace(public_list=lambda _qualities: "[]", find_quality=lambda *_args: None),
        "queue": queue,
        "queue_lookup": lambda *_args: None,
        "queue_quality": lambda *_args: None,
        "register_source": lambda *_args: None,
        "req_counter": lambda request: int(request.split(":", 1)[0]),
        "seek_tail": SimpleNamespace(SeekTail=object),
        "short_error": str,
        "shutil": shutil,
        "start_mpv": lambda *_args, **_kwargs: FakePlayer(events),
        "tempfile": tempfile,
        "threading": threading,
        "time": time,
    }
    values.update(overrides)
    return C.PlayerDependencies(**values)


class FakeHistory:
    def add(self, **_values):
        pass

    def snapshot(self):
        return {}

    def clear(self, *_keys):
        pass


class PlayerWindowReadyTests(unittest.TestCase):
    def test_window_ready_waits_for_mpv_ipc(self):
        controller = C.PlayerController.__new__(C.PlayerController)
        controller.ipc = None
        self.assertEqual(controller.on_message({"cmd": "window_ready"}), {"ok": True, "ready": False})
        controller.ipc = object()
        self.assertEqual(controller.on_message({"cmd": "window_ready"}), {"ok": True, "ready": True})


class PlayerControllerLifecycleTests(unittest.TestCase):
    def test_processed_previews_wait_for_loaded_file_then_start_once(self):
        events = []
        worker = SimpleNamespace(start=lambda: events.append("start"),
                                 start_prefetch=lambda interval: events.append(("prefetch", interval)),
                                 close=lambda: events.append("close"))
        calls = []
        with tempfile.TemporaryDirectory(dir=_STREAMER_ROOT) as directory:
            deps = controller_deps(
                ThumbnailWorker=lambda *args: (calls.append(args) or worker),
                tempfile=SimpleNamespace(mkdtemp=lambda **kwargs: directory))
            player = C.PlayerController(controller_args(thumbnails=True, thumb_width=320, thumb_interval=10),
                                        FakeEngine(), deps=deps)
            player.ipc = FakeIpc([])
            player.thumbs = None
            player.tail, player.session, player.progressive = None, None, None
            player.processed = SimpleNamespace(path="finished.mkv")
            player.ipc.props.update(path="old.mkv", duration=60.0)
            player.processed_thumbnails_tick()
            self.assertEqual(calls, [])
            player.ipc.props.update(path="finished.mkv", duration=None)
            player.processed_thumbnails_tick()
            self.assertEqual(calls, [])
            player.ipc.props["duration"] = 60.0
            player.processed_thumbnails_tick()
            player.processed_thumbnails_tick()
            self.assertEqual(calls, [("finished.mkv", {}, 60.0, 320, directory)])
            self.assertEqual(events, ["start", ("prefetch", 10)])
            player.end_source()
            self.assertEqual(events[-1], "close")
            self.assertIsNone(player.thumbs)

    def test_processed_previews_can_be_disabled(self):
        player = C.PlayerController(controller_args(thumbnails=False), FakeEngine(), deps=controller_deps())
        player.processed = SimpleNamespace(path="finished.mkv")
        player.thumbs = None
        player.processed_thumbnails_tick()
        self.assertIsNone(player.thumbs)

    def setUp(self):
        FakeSession.instances.clear()
        FakeOpener.instances.clear()

    def test_startup_failure_closes_endpoint_before_propagating(self):
        events = []
        endpoint = FakeEndpoint(events)
        failure = RuntimeError("mpv could not start")
        def fail_start(*_args, **_kwargs):
            raise failure

        deps = controller_deps(events, MpvEndpoint=lambda: endpoint, start_mpv=fail_start)
        player = C.PlayerController(controller_args(), FakeEngine(), deps=deps)
        with self.assertRaisesRegex(RuntimeError, "mpv could not start"):
            player.run()

        self.assertEqual(events, ["endpoint:close"])

    def test_end_source_closes_tail_then_progressive_session_and_thumbnails(self):
        events = []

        class Tail:
            def close(self, reason):
                events.append(("tail:close", reason))

            def summary(self):
                return "tail summary"

        class Progressive:
            def close(self):
                events.append("progressive:close")

        class Session:
            def stop(self):
                events.append("session:stop")
                return 1.25

            def release_restorer(self):
                events.append("session:release")

        class Thumbnails:
            def close(self):
                events.append("thumbnails:close")

        class Ipc:
            def set(self, name, value):
                events.append(("ipc:set", name, value))
                raise OSError("mpv exited")

        deps = controller_deps(events, log=lambda message: events.append(("log", message)))
        player = C.PlayerController(controller_args(), FakeEngine(), deps=deps)
        player.tail, player.ipc = Tail(), Ipc()
        player.progressive, player.session, player.thumbs = Progressive(), Session(), Thumbnails()

        took = player.end_source()

        self.assertEqual(took, 1.25)
        self.assertEqual(events[0], ("tail:close", "source closed"))
        self.assertEqual(events[1], ("log", "tail summary | old muxer closed"))
        self.assertEqual(events[2], ("ipc:set", "user-data/oh/tail", ""))
        self.assertEqual(events[3:], ["progressive:close", "session:stop", "session:release", "thumbnails:close"])
        self.assertIsNone(player.tail)
        self.assertIsNone(player.progressive)
        self.assertIsNone(player.session)
        self.assertIsNone(player.thumbs)

    def test_run_teardown_releases_player_resources_in_order(self):
        events = []
        endpoint = FakeEndpoint(events)
        player_proc = FakePlayer(events)
        source = SimpleNamespace(meta=SimpleNamespace(duration=30.0, video_fps_exact=30))
        opened = SimpleNamespace(
            input="clip.mp4", user_headers={}, source=source, quals=[{"id": "source"}],
            chosen={"id": "source", "label": "1080p"}, refresh=None, title="clip", source_text="clip.mp4",
            start=2.0, thumb_src=None,
        )

        deps = controller_deps(
            events,
            MpvEndpoint=lambda: endpoint,
            start_mpv=lambda *_args, **_kwargs: player_proc,
            Session=lambda *args: FakeSession(events, *args),
            info_panel=SimpleNamespace(GpuMonitor=lambda: FakeGpu(events), History=lambda: FakeHistory(),
                                       build_info=lambda snap: snap),
        )
        player = C.PlayerController(controller_args(), FakeEngine(), first=opened, deps=deps)
        self.assertEqual(player.run(), 0)

        order = [
            "gpu:start", "session:create", "session:start", "ipc:create", "cast:create",
            "cast:close", "session:stop", "session:release", "link:close", "gpu:close", "ipc:close",
            "endpoint:close",
        ]
        positions = [events.index(event) for event in order]
        self.assertEqual(positions, sorted(positions), events)

    def test_forwarded_open_is_queued_and_dispatched_by_controller_thread(self):
        events = []
        endpoint = FakeEndpoint(events)
        player_proc = FakePlayer(events, polls=(None, 0))
        main_thread = threading.get_ident()

        class FakeInstance:
            def serve(self, handler, _log):
                self.listener_thread = None
                self.reply = None

                def dispatch():
                    self.listener_thread = threading.get_ident()
                    self.reply = handler({"cmd": "open", "input": "https://example.test/video",
                                          "headers": {"X-Test": 7}})

                worker = threading.Thread(target=dispatch)
                worker.start()
                worker.join(2)
                self.assert_stopped = not worker.is_alive()

            def close(self):
                pass

        instance = FakeInstance()
        deps = controller_deps(
            events,
            MpvEndpoint=lambda: endpoint,
            start_mpv=lambda *_args, **_kwargs: player_proc,
            info_panel=SimpleNamespace(GpuMonitor=lambda: FakeGpu(events), History=lambda: FakeHistory(),
                                       build_info=lambda snap: snap),
        )
        player = C.PlayerController(controller_args(), FakeEngine(), instance=instance, deps=deps)
        self.assertEqual(player.run(), 0)

        self.assertTrue(instance.assert_stopped)
        self.assertEqual(instance.reply, {"ok": True})
        self.assertEqual(len(FakeOpener.instances), 1)
        self.assertEqual(len(FakeOpener.instances[0].submissions), 1)
        opened, label, start, submit_thread = FakeOpener.instances[0].submissions[0]
        self.assertEqual((label, start), ("轉送", 0.0))
        self.assertEqual(opened.headers, {"X-Test": "7"})
        self.assertNotEqual(instance.listener_thread, main_thread)
        self.assertEqual(submit_thread, main_thread)


class ProgressiveSourceRefreshTests(unittest.TestCase):
    def test_same_queue_item_reopens_at_current_position_after_proxy_changes(self):
        old = SimpleNamespace(queue_id="item-a", work_dir="work", source_url="http://127.0.0.1:30101/playlist.m3u8")
        new = SimpleNamespace(queue_id="item-a", work_dir="work", source_url="http://127.0.0.1:30102/playlist.m3u8",
                              duration=100.0, start=0.0)
        player = object.__new__(C.PlayerController)
        player.progressive = SimpleNamespace(po=old)
        player.ipc = SimpleNamespace(get=lambda prop: 37.25)
        player._deps = controller_deps()
        switched, notices = [], []
        player.switch_to_progressive = switched.append
        player.osd = notices.append
        player.play_progressive(new)
        self.assertEqual(switched, [new])
        self.assertEqual(new.start, 37.25)
        self.assertEqual(notices, [])

    def test_repeated_play_with_same_proxy_preserves_existing_feeder(self):
        source = SimpleNamespace(queue_id="item-a", work_dir="work", source_url="http://127.0.0.1:30101/playlist.m3u8")
        player = object.__new__(C.PlayerController)
        player.progressive = SimpleNamespace(po=source)
        switched, notices = [], []
        player.switch_to_progressive = switched.append
        player.osd = notices.append
        player.play_progressive(source)
        self.assertEqual(switched, [])
        self.assertEqual(notices, ["已在邊處理邊看"])


class QueueStartTimeTests(unittest.TestCase):
    def test_requested_start_reopens_same_progressive_item_there(self):
        src = "http://127.0.0.1:30101/playlist.m3u8"
        old = SimpleNamespace(queue_id="item-a", work_dir="work", source_url=src)
        new = SimpleNamespace(queue_id="item-a", work_dir="work", source_url=src, duration=100.0, start=55.0,
                              start_requested=True)
        player = object.__new__(C.PlayerController)
        player.progressive = SimpleNamespace(po=old)
        player.ipc = SimpleNamespace(get=lambda prop: self.fail("must not use the current position"))
        switched, notices = [], []
        player.switch_to_progressive = switched.append
        player.osd = notices.append
        player.play_progressive(new)
        self.assertEqual((switched, notices, new.start), ([new], [], 55.0))

    def test_processed_file_loads_at_requested_start(self):
        loads = []
        player = object.__new__(C.PlayerController)
        player.opener, player.pending_open, player.epoch = SimpleNamespace(), None, 0
        player.leave_current = lambda: None
        player._deps = controller_deps()
        player.publish_settings = lambda: None
        player.ipc = SimpleNamespace(set=lambda *a: None)
        player.load_processed = lambda po, pos, tag: loads.append(pos)
        for start, want in ((0.0, 0.0), (12.5, 12.5)):
            po = SimpleNamespace(title="t", path="p.mkv", queue_id="q", layers="", start=start)
            C.PlayerController.switch_to_processed(player, po)
            self.assertEqual(loads[-1], want)


class CastProcessingStatusTests(unittest.TestCase):
    def test_cast_speed_uses_hls_frames_and_resets_after_restart(self):
        c = C.PlayerController.__new__(C.PlayerController)
        status = dict(restarts=0, run_start=0.0, frames_written=30,
                      tv_pos=2.0, cached_ranges=[(0.0, 8.0)])
        c.cast = SimpleNamespace(active=dict(vod=object(), plan=SimpleNamespace(fps=30), v0=10.0),
                                 info_snapshot=lambda: status)
        c.proc_samples = collections.deque(maxlen=8)
        c.gpu = SimpleNamespace(snapshot=lambda: {})
        samples = []
        c.hist = SimpleNamespace(add=lambda **kw: samples.append(kw))
        c.sample_history(1.0)
        status['frames_written'] = 90
        c.sample_history(2.0)
        self.assertEqual(c.proc_fps_now, 60.0)
        self.assertEqual(samples[-1]['ratio'], 2.0)
        self.assertEqual(samples[-1]['buffer'], 6.0)
        status.update(restarts=1, run_start=20.0, frames_written=0)
        c.sample_history(3.0)
        self.assertIsNone(c.proc_fps_now)


if __name__ == "__main__":
    unittest.main()
