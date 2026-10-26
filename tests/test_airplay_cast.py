"""Tests for airplay_cast.py with a fake pyatv module (no network, no real devices).

Run: lada\\.venv\\Scripts\\python.exe streamer\\tests\\test_airplay_cast.py
"""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import asyncio
import ast
import enum
import json
import logging
import os
import plistlib
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
import oh_streamer.platform.airplay_cast as AC  # noqa: E402


class Protocol(enum.Enum):
    MRP = 2
    AirPlay = 4
    Companion = 5


class GuiMenuTest(unittest.TestCase):
    def test_seek_prefetches_source_time_and_reuses_cast(self):
        from unittest.mock import Mock
        import oh_streamer.platform.hls_cast as H
        source = (_SOURCE_ROOT / "oh_streamer/player/oh_player.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CastManager")
        namespace = dict(airplay_cast=AC, json=json, queue=queue, time=time, log=lambda *_: None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), "CastManager", "exec"), namespace)
        manager = object.__new__(namespace["CastManager"])
        vod, caster = Mock(), Mock()
        plan = H.SegmentPlan(30, 1800, 64, 48, video_start=5)
        manager.active = dict(mode="airplay", v0=5, local_pos=5, plan=plan, vod=vod,
                              phase="playing", device="TV", dev_id="tv", url="http://local/master.m3u8")
        manager._ipc, manager._osd, manager.publish_menu = Mock(), Mock(), Mock()
        manager.caster, manager._caster = caster, lambda: caster
        manager.events = queue.Queue()
        manager.seek(35)
        self.assertEqual(manager.active["local_pos"], 5)
        self.assertEqual(manager.active["phase"], "playing")
        vod.prefetch.assert_not_called()
        caster.control.assert_called_once_with("seek", 30)
        caster.stop.assert_not_called()
        caster.play.assert_not_called()
        caster.tv_position.return_value = 32
        caster.tv_state.return_value = {"rate": 0}
        manager.events.put(("control-done", {"action": "seek", "position": 30}))
        manager.events.put(("playing", "TV"))
        manager.poll()
        manager._ipc.assert_any_call("set", "user-data/oh/cast-position", 37)
        self.assertEqual(manager.active["local_pos"], 37)
        vod.prefetch.assert_called_with(plan.index_at(30))
        manager._ipc.assert_any_call("set", "user-data/oh/cast-paused", "1")
        manager.handle_request("1:play_pause", None, None)
        caster.control.assert_called_with("play_pause")
        manager.events.put(("control-error", "unsupported"))
        manager.poll()
        manager._osd.assert_called_with("unsupported", 6000)
        self.assertEqual(manager.active["phase"], "playing")
        manager.seek(999)
        caster.control.assert_called_with("seek", 59)
        manager.seek(-10)
        caster.control.assert_called_with("seek", 0)
        with self.assertRaises(ValueError):
            manager.seek(float("nan"))

    def test_finished_file_start_and_resume_at_receiver_position(self):
        from unittest.mock import Mock
        import oh_streamer.platform.hls_cast as H
        source = (_SOURCE_ROOT / "oh_streamer/player/oh_player.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CastManager")
        vod = Mock()
        vod.status.return_value = dict(requests=3, restarts=0, cached_segments=2, cached_bytes=0, evicted=0)
        server = Mock()
        server.start.return_value = server
        server.url.return_value = "http://localhost/token/master.m3u8"
        file_source = types.SimpleNamespace(meta=types.SimpleNamespace(video_fps_exact=30, frames_count=300,
                           duration=10, video_width=64, video_height=48, start_pts=0, time_base=1),
                           fmt_start=0, has_audio=True, encoder="libx264", target=0, stop=lambda: 0)
        class FakeFileSource:
            def __new__(cls, path):
                return file_source
        fake_hls = types.SimpleNamespace(FileCastSource=FakeFileSource, SegmentPlan=H.SegmentPlan,
                                        HlsVod=Mock(return_value=vod), HlsServer=Mock(return_value=server))
        namespace = dict(airplay_cast=AC, hls_cast=fake_hls, json=json, queue=queue, time=time,
                         log=lambda *_: None, LOG_TO_FILE=False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), "CastManager", "exec"), namespace)
        manager = object.__new__(namespace["CastManager"])
        manager.active = None
        manager.args = types.SimpleNamespace(cast_bind="127.0.0.1", cast_port=0, cast_cache_gb=1, cast_playlist="master")
        manager.caster = types.SimpleNamespace(tv_position=lambda: 7.5, stop=lambda: True)
        manager._ipc, manager._osd, manager.publish_menu = Mock(), Mock(), Mock()
        manager._resume_local = Mock()
        self.assertTrue(manager.start("airplay", types.SimpleNamespace(path="finished.mkv"), 4.5, "TV", "TV"))
        self.assertTrue(manager.active["finished_file"])
        self.assertEqual(manager.active["local_pos"], 4.5)
        manager.stop()
        manager._ipc.assert_any_call("command", "seek", 7.5, "absolute+exact")
        manager._resume_local.assert_not_called()
        vod.close.assert_called_once()

    def test_pin_requests_and_success_refresh(self):
        # Load just CastManager: the player module's model imports are unrelated to pairing.
        source = (_SOURCE_ROOT / "oh_streamer/player/oh_player.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CastManager")
        namespace = {"airplay_cast": AC, "json": json, "queue": queue, "time": time,
                     "log": lambda *_: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), "CastManager", "exec"), namespace)
        manager = object.__new__(namespace["CastManager"])
        manager.active, manager.devices = None, []
        manager.scanning, manager.scan_error, manager.pyatv_ok = False, None, True
        manager.published_menu = None
        manager.events = queue.Queue()
        published, pins = [], []
        manager._ipc = lambda *args: published.append(args)
        manager._osd = lambda *_: None
        manager._caster = lambda: types.SimpleNamespace(submit_pin=pins.append)
        manager.pairing = {"status": "輸入 PIN", "input": True, "pin": ""}
        for digit in "04321":
            manager.handle_request("1:pin:" + digit, None, None)
        manager.handle_request("2:pin:back", None, None)
        self.assertEqual(manager.pairing["pin"], "0432")
        manager.handle_request("3:pin:submit", None, None)
        self.assertEqual(pins, ["0432"])
        self.assertFalse(manager.pairing["input"])
        self.assertTrue(any("確認 PIN" in str(row) for row in published))
        manager.store = types.SimpleNamespace(load=lambda: {"TV": {"name": "Living Room", "airplay": "secret"}})
        manager.events.put(("paired", AC.Device("TV", "Living Room", paired=True)))
        manager.poll()
        self.assertIsNone(manager.pairing)
        self.assertTrue(manager.devices[0].paired)
        self.assertIn("投放到 Living Room", str(published[-1]))
        manager.active = {"mode": "airplay", "source": "finished-source", "local_pos": 12.0,
                          "dev_id": "TV", "device": "Living Room"}
        requests = []
        def stop(*args, **kwargs):
            manager.active = None
        manager.stop = stop
        manager.handle_request = lambda *args: requests.append(args)
        manager.events.put(("auth-error", {"device_id": "TV", "reason": "驗證失敗"}))
        manager.poll()
        self.assertEqual(requests[0][0], "auth:pair:TV")
        retries = []
        manager.start = lambda *args: retries.append(args)
        manager.events.put(("paired", AC.Device("TV", "Living Room", paired=True)))
        manager.poll()
        self.assertEqual(retries[0], ("airplay", "finished-source", 12.0, "TV", "Living Room"))
        self.assertIsNone(manager.pair_retry)


class FakeService:
    def __init__(self, protocol, identifier):
        self.protocol, self.identifier, self.enabled, self.credentials = protocol, identifier, True, None


class FakeConf:
    def __init__(self, name, ident, address, airplay=True):
        self.name, self.identifier, self.address = name, ident, address
        self.services = [FakeService(Protocol.Companion, ident + "-c")]
        if airplay:
            self.services.append(FakeService(Protocol.AirPlay, ident + "-ap"))
        self.device_info = types.SimpleNamespace(model_str="Apple TV 4K (gen 3)")

    def get_service(self, proto):
        return next((s for s in self.services if s.protocol == proto), None)

    def set_credentials(self, proto, creds):
        self.get_service(proto).credentials = creds
        return True


class FakePairing:
    def __init__(self, fake, conf):
        self.fake, self.conf = fake, conf
        self.service = conf.get_service(Protocol.AirPlay)
        self.device_provides_pin = True
        self.has_paired = False
        self.pin_value = None
        self.closed = False

    async def begin(self):
        self.fake.calls.append("begin")

    def pin(self, pin):
        self.pin_value = pin

    async def finish(self):
        if self.pin_value == self.fake.correct_pin:
            self.has_paired = True
            self.service.credentials = "CREDS:" + self.conf.identifier

    async def close(self):
        self.closed = True
        self.fake.calls.append("close")


class FakeStream:
    def __init__(self, fake):
        self.fake = fake

    async def play_url(self, url, position=0):
        self.fake.calls.append(("play_url", url, position))
        if self.fake.play_error:
            raise self.fake.play_error
        self.fake.playing.set()
        try:
            # like pyatv: poll /playback-info once a second (logged at DEBUG) until the TV ends playback
            log = logging.getLogger("pyatv.protocols.airplay.player")
            while not self.fake.tv_ended.is_set():
                body = plistlib.dumps({"duration": 600.0, "position": 42.0, "rate": 1.0}, fmt=plistlib.FMT_BINARY)
                log.debug("Playback-info: %s", types.SimpleNamespace(body=body))
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            self.fake.calls.append("play_cancelled")   # pyatv closes the AirPlay connection here
            raise


class FakeRemoteControl:
    def __init__(self, fake):
        self.fake = fake

    async def set_position(self, pos):
        self.fake.calls.append(("set_position", pos))

    async def pause(self):
        self.fake.calls.append("pause")

    async def play(self):
        self.fake.calls.append("play")

    async def play_pause(self):
        self.fake.calls.append("play_pause")


class FakeAtv:
    def __init__(self, fake):
        self.fake = fake
        self.stream = FakeStream(fake)
        self.remote_control = FakeRemoteControl(fake)

    def close(self):
        self.fake.calls.append("atv_close")
        return set()


def make_fake(devices):
    fake = types.SimpleNamespace(calls=[], correct_pin=4321, play_error=None, connect_error=None,
                                 playing=threading.Event(), tv_ended=threading.Event(), last_conf=None)
    fake.const = types.SimpleNamespace(Protocol=Protocol)
    fake.protocols = types.SimpleNamespace(airplay=types.SimpleNamespace(player=types.SimpleNamespace(WAIT_RETRIES=5)))

    async def scan(loop, timeout=5, hosts=None, identifier=None, **kw):
        fake.calls.append(("scan", hosts, identifier))
        out = [d for d in devices if (not hosts or d.address in hosts)
               and (not identifier or identifier == d.identifier)]
        return out

    async def pair(conf, protocol, loop, **kw):
        assert protocol == Protocol.AirPlay
        return FakePairing(fake, conf)

    async def connect(conf, loop, **kw):
        fake.last_conf = conf
        if fake.connect_error:
            raise fake.connect_error
        return FakeAtv(fake)

    fake.scan, fake.pair, fake.connect = scan, pair, connect
    return fake


DEVICES = lambda: [FakeConf("Living Room", "AA:BB:CC:00:00:01", "192.0.2.10"),   # noqa: E731
                   FakeConf("Bedroom TV", "AA:BB:CC:00:00:02", "192.0.2.11"),
                   FakeConf("HomePod", "AA:BB:CC:00:00:03", "192.0.2.12", airplay=False)]


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = AC.CredentialStore(os.path.join(self.dir.name, "airplay_credentials.json"))

    def tearDown(self):
        self.dir.cleanup()

    def test_roundtrip_and_remove(self):
        self.assertEqual(self.store.load(), {})
        self.store.save_device("ID1", "Living Room", "192.0.2.10", "secret")
        self.assertEqual(self.store.get("ID1")["airplay"], "secret")
        self.store.update_address("ID1", "192.0.2.99")
        self.assertEqual(self.store.get("ID1")["address"], "192.0.2.99")
        with open(self.store.path, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(raw["version"], 1)
        self.assertFalse(os.path.exists(str(self.store.path) + ".tmp"))
        self.assertTrue(self.store.remove("ID1"))
        self.assertFalse(self.store.remove("ID1"))
        self.assertEqual(self.store.load(), {})

    def test_corrupt_file(self):
        for text in ("{bad", "[]", '{"devices": 3}', '{"devices": {"x": {"name": "n"}}}'):
            with open(self.store.path, "w", encoding="utf-8") as f:
                f.write(text)
            self.assertEqual(self.store.load(), {}, text)


class DeviceTest(unittest.TestCase):
    def test_match(self):
        devs = [AC.Device("AA:01", "Living Room"), AC.Device("AA:02", "Bedroom TV"), AC.Device("AA:03", "Bedroom")]
        self.assertEqual(AC.match_device(devs, "aa:01").name, "Living Room")
        self.assertEqual(AC.match_device(devs, "living room").id, "AA:01")
        self.assertEqual(AC.match_device(devs, "Bedroom").id, "AA:03")       # exact beats substring
        self.assertEqual(AC.match_device(devs, "iving").id, "AA:01")         # unique substring
        with self.assertRaises(AC.CastError):
            AC.match_device(devs, "bed")                                     # ambiguous
        with self.assertRaises(AC.CastError):
            AC.match_device(devs, "kitchen")
        with self.assertRaises(AC.CastError):
            AC.match_device([], "x")

    def test_merged_order(self):
        with tempfile.TemporaryDirectory() as d:
            store = AC.CredentialStore(os.path.join(d, "c.json"))
            store.save_device("B", "Bedroom", None, "c1")
            store.save_device("Z", "Old TV", None, "c2")          # paired, not found by this scan
            devs = AC.merged_devices([AC.Device("A", "Attic"), AC.Device("B", "Bedroom")], store)
            self.assertEqual([(d.id, d.paired, d.found) for d in devs],
                             [("B", True, True), ("Z", True, False), ("A", False, True)])

    def test_menu_rows(self):
        devs = [AC.Device("B", "Bedroom", paired=True), AC.Device("A", "Attic")]
        rows = AC.build_cast_menu(None, devs, False, None)
        self.assertEqual(rows[0], {"id": "dev:B", "kind": "item", "label": "投放到 Bedroom"})
        self.assertEqual(rows[2]["kind"], "item")
        self.assertIn("播放器", rows[2]["hint"])
        self.assertEqual([r["id"] for r in rows if r["kind"] == "item"], ["dev:B", "pair:B", "pair:A", "scan", "hls"])
        rows = AC.build_cast_menu(None, None, True, None)
        self.assertIn("搜尋裝置中…", [r["label"] for r in rows])
        self.assertNotIn("scan", [r["id"] for r in rows])
        rows = AC.build_cast_menu(None, [], False, None)
        self.assertIn("找不到 AirPlay 裝置", [r["label"] for r in rows])
        rows = AC.build_cast_menu({"device": "Bedroom"}, devs, False, None)
        self.assertEqual([r["id"] for r in rows if r["kind"] == "item"], ["stop"])
        self.assertEqual(rows[0]["label"], "投放中：Bedroom")
        rows = AC.build_cast_menu({"device": None, "url": "http://x/t/master.m3u8"}, devs, False, None)
        self.assertEqual(rows[2]["label"], "停止區網串流")
        rows = AC.build_cast_menu(None, None, False, None, pyatv_ok=False)
        self.assertEqual([r["id"] for r in rows if r["kind"] == "item"], ["hls"])
        json.dumps(rows, ensure_ascii=False)


class PairTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = AC.CredentialStore(os.path.join(self.dir.name, "c.json"))
        self.fake = make_fake(DEVICES())
        self.loop = asyncio.new_event_loop()
        self.said = []

    def tearDown(self):
        self.loop.close()
        self.dir.cleanup()

    def pair(self, query, pin):
        return self.loop.run_until_complete(AC.pair_device(self.loop, query, self.store, lambda prompt: pin,
                                                           self.said.append, mod=self.fake))

    def test_pin_flow_saves_credentials(self):
        dev = self.pair("living", "4321")
        self.assertEqual(dev.id, "AA:BB:CC:00:00:01")
        self.assertEqual(self.store.get(dev.id)["airplay"], "CREDS:AA:BB:CC:00:00:01")
        self.assertEqual(self.store.get(dev.id)["address"], "192.0.2.10")
        self.assertIn("close", self.fake.calls)
        self.assertTrue(any("本機機密" in s for s in self.said))

    def test_wrong_pin_saves_nothing(self):
        with self.assertRaises(AC.CastError):
            self.pair("Bedroom TV", "1111")
        with self.assertRaises(AC.CastError):
            self.pair("Bedroom TV", "abc")
        self.assertEqual(self.store.load(), {})
        self.assertEqual(self.fake.calls.count("close"), 2)          # the pairing is always closed

    def test_no_airplay_device_is_not_offered(self):
        with self.assertRaises(AC.CastError):
            self.pair("HomePod", "4321")


class CasterTest(unittest.TestCase):
    def test_remote_commands_keep_existing_stream_and_connection(self):
        dev = "AA:BB:CC:00:00:02"
        url = "http://192.0.2.1:5000/tok/master.m3u8"
        self.caster.play(dev, url)
        self.next_event("playing")
        self.assertTrue(self.fake.playing.wait(3))
        original_task = self.caster.play_task
        self.caster.control("seek", 95.5).result(timeout=5)
        for action in ("pause", "play", "play_pause"):
            self.caster.control(action).result(timeout=5)
            self.assertIn(action, self.fake.calls)
        self.assertIn(("set_position", 95), self.fake.calls)
        self.assertIs(self.caster.play_task, original_task)
        self.assertEqual(sum(isinstance(c, tuple) and c[0] == "play_url" for c in self.fake.calls), 1)
        self.assertNotIn("play_cancelled", self.fake.calls)
        self.assertNotIn("atv_close", self.fake.calls)
        self.assertTrue(self.caster.stop())

    def test_control_error_keeps_stream_alive(self):
        from unittest.mock import AsyncMock
        self.caster.play("AA:BB:CC:00:00:02", "http://local/video.m3u8")
        self.next_event("playing")
        self.assertTrue(self.fake.playing.wait(3))
        self.caster.atv.remote_control.pause = AsyncMock(side_effect=RuntimeError("unsupported"))
        self.caster.control("pause").result(timeout=5)
        self.assertIn("unsupported", self.next_event("control-error"))
        self.assertTrue(self.caster.busy)
        self.assertNotIn("play_cancelled", self.fake.calls)

    def test_pause_icon_updates_without_receiver_push_and_ignores_stale_clock(self):
        self.caster.play("AA:BB:CC:00:00:02", "http://local/video.m3u8")
        self.next_event("playing")
        self.assertTrue(self.fake.playing.wait(3))
        self.caster.control("pause").result(timeout=5)
        self.assertEqual(self.caster.tv_state()["rate"], 0)
        AC._TAP.set_state({"position": 42, "rate": 1, "t": time.monotonic() - 2, "source": "clock"})
        self.assertEqual(self.caster.tv_state()["rate"], 0)
        self.assertEqual(self.caster.tv_position(), 42)
        self.caster.control("play_pause").result(timeout=5)
        self.assertEqual(self.caster.tv_state()["rate"], 1)
        self.caster.control("play_pause").result(timeout=5)
        self.assertEqual(self.caster.tv_state()["rate"], 0)
        self.caster.playstatus_update(None, types.SimpleNamespace(device_state=types.SimpleNamespace(name="Playing"),
                                                                  position=45, total_time=60))
        self.assertEqual(self.caster.tv_state()["rate"], 1)

    def test_receiver_pause_state_freezes_position(self):
        self.caster.playstatus_update(None, types.SimpleNamespace(device_state=types.SimpleNamespace(name="Paused"),
                                                                  position=17, total_time=60))
        self.assertEqual(self.caster.tv_position(), 17)
        self.assertEqual(self.caster.tv_state()["rate"], 0)

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = AC.CredentialStore(os.path.join(self.dir.name, "c.json"))
        self.store.save_device("AA:BB:CC:00:00:02", "Bedroom TV", "192.0.2.11", "CREDS2")
        self.fake = make_fake(DEVICES())
        self.events = queue.Queue()
        self.caster = AC.AirPlayCaster(self.store, lambda k, d: self.events.put((k, d)), mod=self.fake,
                                       log=lambda m: None)

    def tearDown(self):
        self.caster.close()
        self.dir.cleanup()

    def next_event(self, kind, timeout=10):
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            try:
                k, d = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            seen.append(k)
            if k == kind:
                return d
        self.fail(f"no {kind} event (saw {seen})")

    def test_play_then_stop(self):
        self.caster.play("AA:BB:CC:00:00:02", "http://192.0.2.1:5000/tok/master.m3u8", position=123.7)
        self.assertEqual(self.next_event("playing"), "Bedroom TV")
        self.assertTrue(self.fake.playing.wait(3))
        self.assertIn(("play_url", "http://192.0.2.1:5000/tok/master.m3u8", 123), self.fake.calls)
        conf = self.fake.last_conf
        self.assertEqual(conf.get_service(Protocol.AirPlay).credentials, "CREDS2")
        self.assertFalse(conf.get_service(Protocol.Companion).enabled)        # only AirPlay is used
        self.assertGreaterEqual(self.fake.protocols.airplay.player.WAIT_RETRIES, 20)
        self.assertTrue(self.caster.busy)
        time.sleep(0.2)
        self.assertAlmostEqual(self.caster.tv_position(), 42.0, delta=1.0)    # from the playback-info tap
        self.assertTrue(self.caster.stop())
        self.next_event("stopped")
        self.assertIn("play_cancelled", self.fake.calls)
        self.assertIn("atv_close", self.fake.calls)
        self.assertFalse(self.caster.busy)
        self.assertEqual(self.fake.calls[0], ("scan", ["192.0.2.11"], None))     # unicast scan at the stored address

    def test_gui_pair_pin_and_save(self):
        self.caster.pair("Living Room")
        self.next_event("pair-pin")
        self.caster.submit_pin("4321")
        dev = self.next_event("paired")
        self.assertTrue(dev.paired)
        self.assertIsNotNone(self.store.get(dev.id))
        self.assertIn("close", self.fake.calls)

    def test_gui_pair_wrong_pin_and_retry(self):
        self.caster.pair("Living Room")
        self.next_event("pair-pin")
        self.caster.submit_pin("1111")
        self.assertIn("配對失敗", self.next_event("pair-error"))
        self.caster.pair_future.result(timeout=3)
        self.caster.pair("Living Room")
        self.next_event("pair-pin")
        self.caster.submit_pin("4321")
        self.next_event("paired")

    def test_gui_pair_cancel_closes_connection(self):
        self.caster.pair("Living Room")
        self.next_event("pair-pin")
        self.caster.cancel_pair()
        self.next_event("pair-cancelled")
        self.assertIn("close", self.fake.calls)

    def test_tv_ends_playback(self):
        self.caster.play("AA:BB:CC:00:00:02", "http://x/m.m3u8")
        self.next_event("playing")
        self.fake.tv_ended.set()
        self.assertEqual(self.next_event("ended"), "Bedroom TV")
        self.assertIn("atv_close", self.fake.calls)

    def test_errors_are_events_not_exceptions(self):
        self.caster.play("NOT:PAIRED", "http://x/m.m3u8")
        self.assertIn("未配對", self.next_event("error"))
        AuthenticationError = type("AuthenticationError", (Exception,), {})
        self.fake.connect_error = AuthenticationError("bad creds")
        self.caster.play("AA:BB:CC:00:00:02", "http://x/m.m3u8")
        self.assertIn("重新配對", self.next_event("auth-error")["reason"])
        self.fake.connect_error = None
        self.fake.play_error = RuntimeError("boom")
        self.caster.play("AA:BB:CC:00:00:02", "http://x/m.m3u8")
        self.assertIn("boom", self.next_event("error"))
        self.assertFalse(self.caster.busy)

    def test_device_moved_found_by_identifier(self):
        self.store.save_device("AA:BB:CC:00:00:02", "Bedroom TV", "192.0.2.250", "CREDS2")   # stale address
        self.caster.play("AA:BB:CC:00:00:02", "http://x/m.m3u8")
        self.next_event("playing")
        self.assertEqual(self.store.get("AA:BB:CC:00:00:02")["address"], "192.0.2.11")
        self.caster.stop()

    def test_device_not_found(self):
        self.store.save_device("AA:BB:CC:00:00:09", "Gone", "192.0.2.99", "C")
        self.caster.play("AA:BB:CC:00:00:09", "http://x/m.m3u8")
        self.assertIn("找不到", self.next_event("error"))

    def test_scan_event(self):
        self.caster.scan()
        devs = self.next_event("scan")
        self.assertEqual([(d.name, d.paired) for d in devs], [("Bedroom TV", True), ("Living Room", False)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
