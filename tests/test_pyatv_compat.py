"""pyatv_compat: PR #2899 video flow patched into pyatv 0.18.0 (real pyatv classes, fake connection, no network).

Run: lada\\.venv\\Scripts\\python.exe -m unittest streamer.tests.test_pyatv_compat
"""
import asyncio
import plistlib
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import airplay_cast, pyatv_compat

try:
    import pyatv  # noqa: F401
    from pyatv import exceptions
    from pyatv.protocols.airplay import channels, player
    from pyatv.protocols.raop.protocols import StreamContext, airplayv2
    from pyatv.support.http import HttpResponse
except ImportError:  # pragma: no cover - casting is optional
    pyatv = None


def bplist(value) -> bytes:
    return plistlib.dumps(value, fmt=plistlib.FMT_BINARY)


def response(body=None, code=200):
    return HttpResponse("RTSP", "1.0", code, "OK", {}, bplist(body) if body is not None else b"")


def event_body(event: dict) -> bytes:
    return bplist({"params": {"data": bplist(event)}})


class FakeConnection:
    local_ip, remote_ip = "192.168.1.2", "192.168.1.3"

    def __init__(self, log, info_error=None):
        self.log, self.info_error = log, info_error
        self.transport = object()
        self.commands = []

    async def send_and_receive(self, method, uri, user_agent=None, **_kw):
        self.log.append(f"{method} {uri} ua={user_agent}")
        if uri == "/info":
            if self.info_error:
                raise self.info_error
            return response({"psi": "PSI"})
        raise AssertionError(uri)

    async def post(self, uri, headers=None, body=None, allow_error=False):
        self.log.append(f"POST {uri}")
        if uri == "/command":
            self.commands.append(plistlib.loads(plistlib.loads(body)["params"]["data"]))
        return response()


class FakeRtsp:
    session_id = 1

    def __init__(self, info_error=None):
        self.log = []
        self.connection = FakeConnection(self.log, info_error)

    async def setup(self, body):
        if "timingProtocol" in body:
            self.log.append(f"SETUP {body['timingProtocol']}")
            return response({"eventPort": 0})
        self.log.append(f"SETUP type={body['streams'][0]['type']}")
        return response({"streams": [{"streamID": 7}]})

    async def record(self):
        self.log.append("RECORD")

    async def feedback(self):
        self.log.append("FEEDBACK")

    async def exchange(self, method, uri, body=None):
        self.log.append(f"{method} {uri}")


@unittest.skipIf(pyatv is None, "pyatv not installed")
class PyatvCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.applied = pyatv_compat.apply()

    def setUp(self):
        self.states, self.diags = [], []
        pyatv_compat.listener.on_state = self.states.append
        pyatv_compat.listener.on_diag = self.diags.append
        self.verify = mock.patch.object(airplayv2, "verify_connection", mock.AsyncMock(return_value=object()))
        self.verify.start()

    def tearDown(self):
        self.verify.stop()
        pyatv_compat.listener.on_state = pyatv_compat.listener.on_diag = None

    def protocol(self, **kw):
        rtsp = FakeRtsp(**kw)
        return airplayv2.AirPlayV2(StreamContext(), rtsp), rtsp

    def test_patch_applies_once(self):
        self.assertTrue(self.applied, pyatv_compat.status())
        self.assertTrue(pyatv_compat.apply())
        self.assertTrue(hasattr(airplayv2.AirPlayV2, "_setup_remote_control_session"))

    def test_command_flow_order_position_and_user_agent(self):
        proto, rtsp = self.protocol()

        async def run():
            resp = await proto.play_url(0, "http://192.168.1.2:8000/x/index.m3u8", 9.8)
            await asyncio.sleep(0)
            proto.teardown()
            return resp
        self.assertEqual(asyncio.run(run()).code, 200)
        log = rtsp.log
        self.assertEqual(log[:5], ["SETUP PTP", "RECORD", "GET /info ua=AirPlay/960.13.1", "SETUP type=130",
                                   "POST /command"])
        self.assertNotIn("POST /play", log)
        self.assertLess(max(i for i, x in enumerate(log) if x == "POST /command"), log.index("FEEDBACK"))
        types_ = [c["type"] for c in rtsp.connection.commands]
        self.assertEqual(types_, ["insertPlayQueueItem", "setProperty", "setRate"])
        item = rtsp.connection.commands[0]["item"]
        self.assertEqual(item["Start-Position"]["value"], 9)
        self.assertLessEqual(len(item["playerLoggingID"]), 6)
        self.assertEqual(self.states[0]["position"], 9.0)

    def test_info_rejected_falls_back_to_post_play(self):
        proto, rtsp = self.protocol(info_error=exceptions.AuthenticationError("status code: 403"))
        asyncio.run(proto.play_url(0, "http://h/x.m3u8", 0))
        self.assertIn("POST /play", rtsp.log)
        self.assertFalse(rtsp.connection.commands)
        self.assertTrue(any("GET /info failed" in d for d in self.diags))
        self.assertFalse(asyncio.run(proto.wait_for_media_end()))    # -> player polls /playback-info as before

    def test_event_channel_dispatch(self):
        got = []
        chan = channels.EventChannel.__new__(channels.EventChannel)
        chan.listener = type("L", (), {"handle_event": lambda _s, e: got.append(e)})()
        body = event_body({"type": "playbackState", "name": "playing"})
        raw = (b"POST /command RTSP/1.0\r\nCSeq: 3\r\nContent-Type: application/x-apple-binary-plist\r\n"
               b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        request, consumed, rest = chan.parse_request(raw + b"POST /x")
        self.assertEqual(request.headers["CSeq"], "3")
        self.assertEqual((consumed, rest), (raw, b"POST /x"))
        self.assertEqual(got, [{"type": "playbackState", "name": "playing"}])
        chan.listener = type("L", (), {"handle_event": lambda _s, e: 1 / 0})()
        self.assertIsNotNone(chan.parse_request(raw)[0])     # a listener failure must not break the channel

    def test_stopped_event_ends_wait_and_player(self):
        proto, _rtsp = self.protocol()

        async def run():
            await proto.play_url(0, "http://h/x.m3u8", 0)
            waiter = asyncio.ensure_future(player.AirPlayPlayer(proto.rtsp, proto)._wait_for_media_to_end())
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            proto.handle_event({"type": "playbackState", "name": "playing"})
            proto.handle_event({"type": "playbackState", "name": "stopped"})
            await asyncio.wait_for(waiter, 2)
            proto.teardown()
        asyncio.run(run())

    def test_dropped_connection_ends_wait(self):
        proto, rtsp = self.protocol()

        async def run():
            await proto.play_url(0, "http://h/x.m3u8", 0)
            rtsp.connection.transport = None
            await asyncio.wait_for(player.AirPlayPlayer(rtsp, proto)._wait_for_media_to_end(), 3)
            proto.teardown()
        asyncio.run(run())


class PlaybackClockTest(unittest.TestCase):
    def setUp(self):
        self.states, self.diags = [], []
        pyatv_compat.listener.on_state = self.states.append
        pyatv_compat.listener.on_diag = self.diags.append

    def tearDown(self):
        pyatv_compat.listener.on_state = pyatv_compat.listener.on_diag = None

    def test_counts_only_while_playing(self):
        clock = pyatv_compat.PlaybackClock()
        clock.start(30.7)
        self.assertEqual((clock.position, clock.rate), (30.0, 0.0))
        clock.update({"type": "playbackState", "name": "playing"})
        clock.t -= 5                                     # 5 s of playing
        clock.update({"type": "playbackState", "name": "paused"})
        self.assertAlmostEqual(clock.position, 35.0, delta=0.1)
        clock.t -= 100                                   # paused: does not advance
        self.assertAlmostEqual(clock.now(), 35.0, delta=0.1)
        self.assertEqual(self.states[-1]["source"], "clock")

    def test_reported_position_wins(self):
        clock = pyatv_compat.PlaybackClock()
        clock.start(0)
        clock.update({"type": "playbackState", "name": "playing", "params": {"position": 12.5, "rate": 1.0,
                                                                              "duration": 600.0}})
        self.assertEqual((clock.position, clock.duration, self.states[-1]["source"]), (12.5, 600.0, "events"))
        clock.update({"type": "notification", "name": "timeJumped",
                      "currentTime": {"value": 9000, "timescale": 100, "flags": 1, "epoch": 0}})
        self.assertEqual(clock.position, 90.0)

    def test_tvos27_playing_event(self):
        # Shape logged by a tvOS 27 Apple TV (playbackState/playing): CMTime position and duration under params.
        clock = pyatv_compat.PlaybackClock()
        clock.start(0)
        clock.update({"type": "playbackState", "name": "playing", "item": {"uuid": "U"}, "params": {
            "duration": {"epoch": 0, "flags": 1, "timescale": 1000, "value": 600000}, "playbackState": "playing",
            "position": {"epoch": 0, "flags": 1, "timescale": 1, "value": 20}, "rate": 1.0, "stallCount": 0}})
        self.assertEqual((clock.position, clock.rate, clock.duration), (20.0, 1.0, 600.0))
        clock.update({"type": "playbackState", "name": "paused"})
        self.assertEqual(clock.rate, 0.0)
        self.assertEqual(self.states[-1]["source"], "events")

    def test_diagnostic_is_capped(self):
        pyatv_compat.PlaybackClock().describe({"type": "updateInfo", "value": {f"k{i}": i for i in range(200)}})
        self.assertLess(len(self.diags[0]), 450)

    def test_diagnostic_shape_hides_strings(self):
        clock = pyatv_compat.PlaybackClock()
        clock.describe({"type": "playbackState", "name": "playing", "url": "http://SECRET/token", "rate": 1.0})
        clock.describe({"type": "playbackState", "name": "playing"})        # once per type/name
        self.assertEqual(len(self.diags), 1)
        self.assertNotIn("SECRET", self.diags[0])
        self.assertIn("rate: 1.0", self.diags[0])

    def test_event_state_does_not_go_stale_for_the_player(self):
        try:
            airplay_cast._TAP.set_state({"position": 50.0, "rate": 1.0, "duration": None,
                                         "t": time.monotonic() - 60, "source": "clock"})
            self.assertAlmostEqual(airplay_cast.AirPlayCaster.tv_position(), 110.0, delta=1.0)
            airplay_cast._TAP.state = {"position": 50.0, "rate": 1.0, "duration": None, "t": time.monotonic() - 60}
            self.assertIsNone(airplay_cast.AirPlayCaster.tv_position())    # /playback-info polling: stale
        finally:
            airplay_cast._TAP.state = None


if __name__ == "__main__":
    unittest.main()
