"""
pyatv_compat: runtime patch so pyatv 0.18.0 can play video (play_url) on tvOS 26 / 27.

tvOS 26+ no longer serves the legacy AirPlay video flow pyatv 0.18.0 uses (POST /play, then polling
GET /playback-info, which answers 500 on tvOS 26 and 403 on tvOS 27). The upstream fix is pyatv PR #2899 (not
merged / released yet): a PTP timed video session, a remote control session, then a play queue over POST /command,
with playback state pushed on the event channel. tvOS 27 additionally rejects GET /info (needed for that) unless it
carries an AirPlay User-Agent.

apply() installs that logic into the already imported pyatv classes (the installed package is never modified). It
only patches pyatv versions in PATCHED_VERSIONS and skips itself once pyatv ships the fix. Delete this module when
a pyatv release contains PR #2899 (and the /info User-Agent fix).

Playback state from the event channel is handed to `listener` (set by airplay_cast): on_state(dict) with the
playback position (playlist seconds), rate, duration and state; on_diag(str) for one-line diagnostics.
"""

import asyncio
import logging
import plistlib
import time
from uuid import uuid4

PATCHED_VERSIONS = {"0.18.0"}

# Modern receivers drive video through POST /command rather than POST /play.
COMMAND_HEADERS = {
    "User-Agent": "AirPlay/960.13.1",
    "Content-Type": "application/x-apple-binary-plist",
}
# Asks for a remote control session (type 130, controlType 1); the MRP data channel uses controlType 2.
RCS_CLIENT_TYPE_UUID = "A6B27562-B43A-4F2D-B75F-82391E250194"

_LOGGER = logging.getLogger("pyatv.protocols.raop.protocols.airplayv2")

_applied: bool | None = None
_reason = "not applied"


class _Listener:
    on_state = None     # (dict) -> None
    on_diag = None      # (str) -> None


listener = _Listener()


def status() -> str:
    return _reason


def _diag(msg: str):
    cb = listener.on_diag
    if cb is not None:
        try:
            cb(msg)
        except Exception:  # noqa: BLE001 - informational only
            pass


def apply() -> bool:
    """Patch pyatv once. True when the PR #2899 flow is active (patched now or earlier)."""
    global _applied, _reason
    if _applied is not None:
        return _applied
    _applied = False
    try:
        from pyatv import const  # noqa: PLC0415
        from pyatv.protocols.airplay import channels, player  # noqa: PLC0415
        from pyatv.protocols.raop import protocols as raop_protocols  # noqa: PLC0415
        from pyatv.protocols.raop.protocols import airplayv2  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 - fake module in tests / incompatible pyatv
        _reason = f"pyatv internals unavailable ({type(e).__name__})"
        return False
    version = getattr(const, "__version__", "unknown")
    if hasattr(airplayv2.AirPlayV2, "_setup_remote_control_session"):
        _reason = f"pyatv {version} already has the /command video flow; not patched"
        return False
    if version not in PATCHED_VERSIONS:
        _reason = f"pyatv {version} is not a version this patch was written for; not patched"
        return False
    _patch_event_channel(channels)
    _patch_player(player, raop_protocols)
    _patch_airplayv2(airplayv2)
    _applied = True
    _reason = f"pyatv {version} patched (PR #2899 + /info User-Agent)"
    return True


# ---------------------------------------------------------------------------------------------------------------
# event channel: decode pushed events and hand them to the stream protocol
# ---------------------------------------------------------------------------------------------------------------

def decode_event(body) -> dict | None:
    """The payload is a binary plist whose params.data member is itself a binary plist."""
    if not isinstance(body, bytes) or not body.startswith(b"bplist00"):
        return None
    outer = plistlib.loads(body)
    if not isinstance(outer, dict):
        return None
    params = outer.get("params")
    data = params.get("data") if isinstance(params, dict) else None
    event = plistlib.loads(data) if isinstance(data, bytes) and data.startswith(b"bplist00") else outer
    return event if isinstance(event, dict) else None


def _patch_event_channel(channels):
    cls = channels.EventChannel
    original = cls.parse_request    # staticmethod on BaseEventChannel; handle_received calls self.parse_request

    def parse_request(self, data):
        request, raw, rest = original(data)
        target = getattr(self, "listener", None)
        if request is not None and target is not None:
            try:
                event = decode_event(request.body)
                if event is not None:
                    target.handle_event(event)
            except Exception:  # noqa: BLE001 - never break the channel (it still has to acknowledge)
                _LOGGER.debug("Failed to dispatch event", exc_info=True)
        return request, raw, rest

    cls.listener = None
    cls.parse_request = parse_request


# ---------------------------------------------------------------------------------------------------------------
# player: wait for the receiver's own end-of-media report instead of polling /playback-info
# ---------------------------------------------------------------------------------------------------------------

def _patch_player(player, raop_protocols):
    from pyatv import exceptions  # noqa: PLC0415

    async def wait_for_media_end(self) -> bool:
        """False: this protocol cannot report it; the caller polls GET /playback-info."""
        return False

    if not hasattr(raop_protocols.StreamProtocol, "wait_for_media_end"):
        raop_protocols.StreamProtocol.wait_for_media_end = wait_for_media_end

    original = player.AirPlayPlayer._wait_for_media_to_end

    async def _wait_for_media_to_end(self) -> None:
        try:
            if await self.stream_protocol.wait_for_media_end():
                _LOGGER.debug("media playback ended")
                return
        except exceptions.ConnectionLostError:
            _LOGGER.debug("Connection was lost, assuming video playback stopped")
            return
        await original(self)

    player.AirPlayPlayer._wait_for_media_to_end = _wait_for_media_to_end


# ---------------------------------------------------------------------------------------------------------------
# AirPlayV2: PR #2899 video flow
# ---------------------------------------------------------------------------------------------------------------

def _patch_airplayv2(m):
    from pyatv import exceptions  # noqa: PLC0415
    cls = m.AirPlayV2
    orig_init, orig_teardown = cls.__init__, cls.teardown

    def __init__(self, context, rtsp):
        orig_init(self, context, rtsp)
        self._rcs_stream_id = None              # set once a remote control session makes /command available
        self._item_uuid = str(uuid4()).upper()
        self._media_ended = asyncio.Event()     # the receiver reports state on the event channel
        self._playback = PlaybackClock()

    async def _setup_base(self, timing_server_port: int) -> None:
        # The video path verifies up front; do not verify twice on the same connection when falling back.
        if self._verifier is None:
            self._verifier = await m.verify_connection(self.context.credentials, self.rtsp.connection)
        setup_resp = await self.rtsp.setup(body={
            "deviceID": "AA:BB:CC:DD:EE:FF", "sessionUUID": str(uuid4()).upper(),
            "timingPort": timing_server_port, "timingProtocol": "NTP", "isMultiSelectAirPlay": True,
            "groupContainsGroupLeader": False, "macAddress": "AA:BB:CC:DD:EE:FF", "model": "iPhone14,3",
            "name": "pyatv", "osBuildVersion": "20F66", "osName": "iPhone OS", "osVersion": "16.5",
            "senderSupportsRelay": False, "sourceVersion": "690.7.1", "statsCollectionEnabled": False,
        })
        resp = m.decode_bplist_from_body(setup_resp)
        event_port = resp.get("eventPort", 0)
        retries = 5      # airplay2-receiver opens the event port a little after answering
        transport = None
        while transport is None:
            try:
                transport, _ = await m.setup_channel(
                    m.EventChannel, self._verifier, self.rtsp.connection.remote_ip, event_port,
                    m.EVENTS_SALT, m.EVENTS_READ_INFO, m.EVENTS_WRITE_INFO)
            except ConnectionRefusedError:
                retries -= 1
                if retries == 0:
                    raise
                await asyncio.sleep(1.0)
        self.event_channel = transport

    def teardown(self) -> None:
        self._media_ended.set()
        orig_teardown(self)

    async def _setup_video_session(self) -> None:
        """PTP timed session with a peer list. NTP is accepted but sends no events and dies after ~20 s."""
        peer = {"ID": self.uuid.upper(), "Addresses": [self.rtsp.connection.local_ip], "DeviceType": 0,
                "SupportsClockPortMatchingOverride": True}
        setup_resp = await self.rtsp.setup(body={
            "timingProtocol": "PTP", "timingPeerInfo": peer, "timingPeerList": [peer],
            "sessionUUID": str(uuid4()).upper(), "sessionCorrelationUUID": self.uuid.upper(),
            "updateSessionRequest": False, "statsCollectionEnabled": False, "isMultiSelectAirPlay": False,
            "deviceID": "AA:BB:CC:DD:EE:FF", "macAddress": "AA:BB:CC:DD:EE:FF", "model": "iPhone14,3",
            "name": "pyatv", "osName": "iPhone OS", "osVersion": "16.5", "osBuildVersion": "20F66",
            "sourceVersion": "690.7.1",
        })
        resp = m.decode_bplist_from_body(setup_resp)
        event_port = resp.get("eventPort", 0)
        if event_port:
            transport, protocol = await m.setup_channel(
                m.EventChannel, self._verifier, self.rtsp.connection.remote_ip, event_port,
                m.EVENTS_SALT, m.EVENTS_READ_INFO, m.EVENTS_WRITE_INFO)
            protocol.listener = self
            self.event_channel = transport

    async def _setup_remote_control_session(self) -> bool:
        """Register a remote control session (enables /command). True when the receiver accepted it."""
        try:
            # tvOS 27 answers 403 to GET /info without an AirPlay User-Agent (pyatv sends "pyatv/<version>").
            info = m.decode_bplist_from_body(await self.rtsp.connection.send_and_receive(
                "GET", "/info", user_agent=COMMAND_HEADERS["User-Agent"]))
            psi = info.get("psi")
        except Exception as e:  # noqa: BLE001
            _diag(f"GET /info failed ({type(e).__name__}); using POST /play")
            return False
        if not psi:
            _diag("receiver did not report psi; using POST /play")
            return False
        try:
            setup_resp = await self.rtsp.setup(body={"streams": [{
                "type": 130, "controlType": 1, "channelID": f"{psi}-RCS-1",
                "clientUUID": str(uuid4()).upper(), "clientTypeUUID": RCS_CLIENT_TYPE_UUID}]})
        except exceptions.HttpError as e:
            _diag(f"remote control session rejected ({type(e).__name__}); using POST /play")
            return False
        streams = m.decode_bplist_from_body(setup_resp).get("streams") or []
        if not streams:
            _diag("remote control session: no streams in answer; using POST /play")
            return False
        self._rcs_stream_id = streams[0].get("streamID", 1)
        _LOGGER.debug("Remote control session established (id=%s)", self._rcs_stream_id)
        return True

    async def send_command(self, command: dict):
        """POST /command; the body is a plist whose params.data member is itself a serialized plist."""
        headers = dict(COMMAND_HEADERS)
        headers["X-Apple-StreamID"] = str(self._rcs_stream_id or 1)
        inner = plistlib.dumps(command, fmt=plistlib.FMT_BINARY)
        return await self.rtsp.connection.post(
            "/command", headers=headers,
            body=plistlib.dumps({"params": {"data": inner}}, fmt=plistlib.FMT_BINARY), allow_error=True)

    def handle_event(self, event) -> None:
        kind, name = event.get("type"), event.get("name")
        self._playback.describe(event)
        if kind == "playbackState":
            _LOGGER.debug("Playback state: %s", name)
            # A replacement goes playing -> loading -> playing, so "stopped" means nothing plays any more.
            if name == "stopped":
                self._media_ended.set()
        elif kind == "notification":
            _LOGGER.debug("Playback notification: %s", name)
        self._playback.update(event)

    async def wait_for_media_end(self) -> bool:
        if self._rcs_stream_id is None:
            return False
        # Also end when the connection drops (TV switched off, network gone): no "stopped" event comes then.
        while not self._media_ended.is_set():
            if self.rtsp.connection.transport is None:
                raise exceptions.ConnectionLostError("AirPlay connection closed")
            try:
                await asyncio.wait_for(self._media_ended.wait(), 1.0)
            except asyncio.TimeoutError:
                pass
        return True

    async def play_url(self, timing_server_port: int, url: str, position: float = 0.0):
        """Play queue over POST /command; receivers without a remote control session still get POST /play.

        Order matters: /info before pair-verify, or feedback before RECORD / alongside the /command requests, makes
        the receiver drop the connection part way through the play queue commands.
        """
        self._verifier = await m.verify_connection(self.context.credentials, self.rtsp.connection)
        self._playback.start(position)
        try:
            await self._setup_video_session()
        except exceptions.HttpError as e:
            _diag(f"video session setup rejected ({type(e).__name__}); using legacy setup")
            await self._setup_base(timing_server_port)
            await self.start_feedback()
            await self.rtsp.record()
            return await self._play_url_legacy(url, position)

        await self.rtsp.record()
        if await self._setup_remote_control_session():
            _diag("video flow: POST /command play queue")
            resp = await self._play_url_command(url, position)
        else:
            resp = await self._play_url_legacy(url, position)
        await self.start_feedback()
        return resp

    def _queue_item(self, url: str, position: float) -> dict:
        return {
            "uuid": self._item_uuid, "Content-Location": url, "mediaType": "file",
            "IsTLSEnabled": url.startswith("https://"), "playbackRestrictions": 0, "referenceRestrictions": 2,
            "supportsIntegratedTimeline": False, "snapTimeToPausePlayback": False,
            "clientBundleID": "dev.pyatv", "clientProcName": "pyatv",
            # At most 6 characters, or the receiver closes the connection without an error response.
            "playerLoggingID": "P/ATV", "playerItemLoggingID": "I/PYATV.01",
            "Start-Position": {"flags": 1, "value": int(position), "epoch": 0, "timescale": 1},
        }

    async def _play_url_command(self, url: str, position: float):
        resp = await self.send_command({"type": "insertPlayQueueItem", "item": self._queue_item(url, position)})
        if resp.code != 200:
            return resp
        await self.send_command({"type": "setProperty", "property": "isInterestedInDateRange", "value": True,
                                 "item": {"uuid": self._item_uuid}})
        await self.send_command({"type": "setRate", "rate": 1.0})     # playback starts paused without this
        return resp

    cls._play_url_legacy = _legacy(m)
    for fn in (__init__, _setup_base, teardown, _setup_video_session, _setup_remote_control_session,
               send_command, handle_event, wait_for_media_end, play_url, _queue_item, _play_url_command):
        setattr(cls, fn.__name__, fn)


def _legacy(m):
    """POST /play for receivers without /command: 0.18.0's play_url minus its own setup (done by the caller)."""

    async def _play_url_legacy(self, url: str, position: float):
        body = {
            "Content-Location": url, "Start-Position-Seconds": position, "uuid": self.uuid, "streamType": 1,
            "mediaType": "file", "mightSupportStorePastisKeyRequests": True, "playbackRestrictions": 0,
            "secureConnectionMs": 22, "volume": 1.0, "infoMs": 122, "connectMs": 18, "authMs": 0, "bonjourMs": 0,
            "referenceRestrictions": 3, "SenderMACAddress": "AA:BB:CC:DD:EE:FF", "model": "iPhone14,3",
            "postAuthMs": 0, "clientBundleID": "dev.pyatv.GPU", "clientProcName": "dev.pyatv.GPU",
            "osBuildVersion": "20G1116", "rate": 1.0,
        }
        resp = await self.rtsp.connection.post(
            "/play", headers=m.HEADERS, body=plistlib.dumps(body, fmt=plistlib.FMT_BINARY), allow_error=True)
        await self.rtsp.exchange("PUT", uri="/setProperty?isInterestedInDateRange", body={"value": True})
        await self.rtsp.exchange("PUT", uri="/setProperty?actionAtItemEnd", body={"value": 0})
        await self.rtsp.exchange("POST", uri="/rate?value=1.000000")       # starts paused without this
        await self.rtsp.exchange("PUT", uri="/setProperty?forwardEndTime",
                                 body={"value": {"flags": 0, "value": 0, "epoch": 0, "timescale": 0}})
        await self.rtsp.exchange("PUT", uri="/setProperty?reverseEndTime",
                                 body={"value": {"flags": 0, "value": 0, "epoch": 0, "timescale": 0}})
        return resp

    return _play_url_legacy


# ---------------------------------------------------------------------------------------------------------------
# playback position from event channel state
# ---------------------------------------------------------------------------------------------------------------

_POSITION_KEYS = ("position", "elapsedtime", "currenttime")
_DURATION_KEYS = ("duration",)
_RATE_KEYS = ("rate",)


def _find_number(event, keys, depth=0):
    """A numeric value under one of `keys` (case-insensitive) in the event or a nested dict; CMTime dicts too."""
    if not isinstance(event, dict) or depth > 3:
        return None
    for k, v in event.items():
        if isinstance(k, str) and k.casefold() in keys:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict) and isinstance(v.get("value"), (int, float)) and v.get("timescale"):
                return float(v["value"]) / float(v["timescale"])
    for v in event.values():
        if isinstance(v, dict):
            found = _find_number(v, keys, depth + 1)
            if found is not None:
                return found
    return None


def _shape(value, depth=0):
    """Keys and value types only (numbers shown): never strings, which can hold the stream URL."""
    if isinstance(value, dict) and depth < 3:
        return "{" + ", ".join(f"{k}: {_shape(v, depth + 1)}" for k, v in sorted(value.items(), key=str)) + "}"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return repr(value)
    return type(value).__name__


class PlaybackClock:
    """Playback position for the player's resume point. Uses a position the receiver reports when there is one;
    otherwise counts the time spent in the "playing" state from the start position (a seek on the TV itself is then
    not followed)."""

    def __init__(self):
        self.position, self.rate, self.duration, self.state = 0.0, 0.0, None, None
        self.reported = False
        self.t = time.monotonic()
        self._described = set()

    def start(self, position: float):
        self.position, self.rate, self.state, self.reported = float(int(position)), 0.0, "loading", False
        self.t = time.monotonic()
        self._publish()

    def now(self) -> float:
        return self.position + self.rate * (time.monotonic() - self.t)

    def describe(self, event):
        key = (event.get("type"), event.get("name"))
        if key in self._described or len(self._described) > 40:
            return
        self._described.add(key)
        shape = _shape(event)
        _diag(f"event {key[0]}/{key[1]}: " + (shape if len(shape) <= 400 else shape[:400] + "…"))

    def update(self, event):
        pos = _find_number(event, _POSITION_KEYS)
        rate = _find_number(event, _RATE_KEYS)
        dur = _find_number(event, _DURATION_KEYS)
        state = event.get("name") if event.get("type") == "playbackState" else None
        if pos is None and rate is None and dur is None and state is None:
            return
        now = time.monotonic()
        if pos is not None:
            self.position, self.reported = pos, True
        else:
            self.position = self.now()
        self.t = now
        if dur is not None and dur > 0:
            self.duration = dur
        if state is not None:
            self.state = state
        if rate is not None:
            self.rate = rate
        elif state is not None:
            self.rate = 1.0 if state == "playing" else 0.0
        self._publish()

    def _publish(self):
        cb = listener.on_state
        if cb is not None:
            try:
                cb({"position": self.position, "rate": self.rate, "duration": self.duration, "t": self.t,
                    "state": self.state, "source": "events" if self.reported else "clock"})
            except Exception:  # noqa: BLE001 - informational only
                pass
