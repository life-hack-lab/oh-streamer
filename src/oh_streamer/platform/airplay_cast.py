"""
airplay_cast: AirPlay glue (pyatv) for casting oh_player's HLS (hls_cast.py) to an Apple TV's native player.

  * scan: mDNS discovery of AirPlay receivers (only on request: --airplay-scan, or opening the cast menu).
  * pair: from the player's cast menu, or `oh_player.py --airplay-pair <name|id>`; enter the Apple TV PIN. The
    resulting AirPlay credentials are stored in the platform user-data directory. THAT FILE IS A LOCAL SECRET: anyone
    who has it can control / stream to that Apple TV from this network. Delete it (or the device's entry) to unpair.
  * cast: AirPlayCaster runs one asyncio loop on a background thread. play() connects with the stored credentials and
    calls atv.stream.play_url(url, position), which only returns when playback on the Apple TV ends (the receiver
    needs the AirPlay connection open the whole time). stop() cancels it, which closes the connection; the Apple TV
    then stops playing. Errors never propagate to the caller: they are reported as events.
  * tvOS 26+ needs the play queue video flow pyatv 0.18.0 lacks: pyatv_compat.py patches it in at runtime, and the
    playback position then comes from the receiver's event channel instead of /playback-info polling.

pyatv is imported lazily (and can be injected: tests use a fake module), so importing this file is cheap.
"""

import asyncio
import json
import logging
import os
import sys
import plistlib
import threading
import time
from pathlib import Path

from oh_streamer import user_data
from oh_streamer.paths import checkout_root
from oh_streamer.platform import platform_support, pyatv_compat

CRED_FILE = platform_support.data_dir() / "airplay_credentials.json"
LEGACY_CRED_FILE = checkout_root() / "airplay_credentials.json"
SCAN_TIMEOUT = 5


class CastError(Exception):
    """A user-facing casting problem (message in Traditional Chinese)."""


# ---------------------------------------------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------------------------------------------

class CredentialStore:
    """Platform user data credentials: {"version": 1, "devices": {id: {name, address, airplay, paired_at}}}."""

    def __init__(self, path=None):
        self.path = Path(CRED_FILE if path is None else path)
        self.lock = threading.Lock()
        self._migration_lock = threading.Lock()
        self._migration_attempted = False
        self._should_migrate_legacy = user_data.same_path(self.path, CRED_FILE)

    @staticmethod
    def _valid_legacy(payload: bytes) -> bool:
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeError, ValueError):
            return False
        devices = raw.get("devices") if isinstance(raw, dict) else None
        return isinstance(devices, dict)

    def _ensure_migrated(self) -> None:
        if not self._should_migrate_legacy:
            return
        with self._migration_lock:
            if self._migration_attempted:
                return
            self._migration_attempted = True
            _copied, warning = user_data.migrate_legacy_file(
                LEGACY_CRED_FILE, self.path, self._valid_legacy)
            if warning:
                logging.getLogger(__name__).warning(warning)

    def load(self) -> dict:
        self._ensure_migrated()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return {}
        devs = raw.get("devices") if isinstance(raw, dict) else None
        if not isinstance(devs, dict):
            return {}
        return {str(k): v for k, v in devs.items()
                if isinstance(v, dict) and isinstance(v.get("airplay"), str) and v["airplay"]}

    def get(self, dev_id: str) -> dict | None:
        return self.load().get(dev_id)

    def save_device(self, dev_id: str, name: str, address: str | None, credentials: str):
        with self.lock:
            devs = self.load()
            devs[dev_id] = {"name": name, "address": address, "airplay": credentials,
                            "paired_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._write(devs)

    def update_address(self, dev_id: str, address: str):
        with self.lock:
            devs = self.load()
            if dev_id in devs and devs[dev_id].get("address") != address:
                devs[dev_id]["address"] = address
                self._write(devs)

    def remove(self, dev_id: str) -> bool:
        with self.lock:
            devs = self.load()
            if devs.pop(dev_id, None) is None:
                return False
            self._write(devs)
            return True

    def _write(self, devs: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "note": "AirPlay credentials (local secret, do not share)", "devices": devs},
                      f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------------------------------------------

class Device:
    def __init__(self, dev_id: str, name: str, address: str | None = None, model: str | None = None,
                 paired: bool = False, found: bool = True):
        self.id, self.name, self.address, self.model = dev_id, name, address, model
        self.paired, self.found = paired, found

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "address": self.address, "model": self.model,
                "paired": self.paired, "found": self.found}

    def __repr__(self):
        return f"Device({self.name!r}, {self.id}, {self.address}, paired={self.paired}, found={self.found})"


def _pyatv(mod=None):
    if mod is not None:
        return mod
    import pyatv  # noqa: PLC0415 - lazy: heavy, and only needed when casting
    return pyatv


def _protocol(mod):
    return mod.const.Protocol


def device_from_config(conf, mod, store: CredentialStore | None) -> Device:
    proto = _protocol(mod)
    svc = conf.get_service(proto.AirPlay)
    dev_id = str(conf.identifier or getattr(svc, "identifier", None))   # what atvremote --id shows
    model = None
    try:
        model = str(conf.device_info.model_str) if conf.device_info else None
    except Exception:  # noqa: BLE001
        pass
    paired = bool(store and store.get(dev_id))
    return Device(dev_id, str(conf.name), str(conf.address), model, paired)


async def scan_configs(loop, mod, timeout=SCAN_TIMEOUT, hosts=None, identifier=None) -> list:
    proto = _protocol(mod)
    confs = await mod.scan(loop, timeout=timeout, hosts=hosts, identifier=identifier)
    return [c for c in confs if c.get_service(proto.AirPlay) is not None]


def merged_devices(scanned: list[Device], store: CredentialStore) -> list[Device]:
    """Paired devices first (also remembered ones the scan did not find), then unpaired ones; by name."""
    stored = store.load()
    out = {d.id: d for d in scanned}
    for dev_id, d in out.items():
        d.paired = dev_id in stored
    for dev_id, e in stored.items():
        if dev_id not in out:
            out[dev_id] = Device(dev_id, str(e.get("name") or dev_id), e.get("address"), None, True, found=False)
    return sorted(out.values(), key=lambda d: (not d.paired, not d.found, d.name.casefold()))


def match_device(devices: list[Device], query: str) -> Device:
    """By identifier, exact name (case-insensitive) or a unique name prefix / substring."""
    q = (query or "").strip()
    if not q:
        raise CastError("請指定裝置名稱或 ID")
    for d in devices:
        if d.id == q or d.id.casefold() == q.casefold():
            return d
    exact = [d for d in devices if d.name.casefold() == q.casefold()]
    if len(exact) == 1:
        return exact[0]
    part = [d for d in devices if q.casefold() in d.name.casefold()]
    if len(part) == 1:
        return part[0]
    if not part and not exact:
        raise CastError(f"找不到裝置「{q}」" + ("（可用：" + "、".join(d.name for d in devices) + "）" if devices
                                              else "（沒有找到任何 AirPlay 裝置）"))
    raise CastError(f"「{q}」符合多個裝置：" + "、".join(f"{d.name}（{d.id}）" for d in (exact or part)) + "，請改用 ID")


async def pair_device(loop, query: str, store: CredentialStore, ask_pin, say, mod=None,
                      timeout=SCAN_TIMEOUT) -> Device:
    """Pair with an async GUI PIN callback, or a synchronous CLI callback run in an executor."""
    mod = _pyatv(mod)
    proto = _protocol(mod)
    say("搜尋 AirPlay 裝置中…")
    confs = await scan_configs(loop, mod, timeout)
    devices = [device_from_config(c, mod, store) for c in confs]
    dev = match_device(devices, query)
    conf = next(c for c, d in zip(confs, devices) if d.id == dev.id)
    say(f"配對 {dev.name}（{dev.address}，ID {dev.id}）")
    pairing = await mod.pair(conf, proto.AirPlay, loop)
    try:
        await pairing.begin()
        if pairing.device_provides_pin:
            prompt = "請輸入 Apple TV 畫面上的 PIN 碼："
            pin = (await ask_pin(prompt) if asyncio.iscoroutinefunction(ask_pin)
                   else await loop.run_in_executor(None, ask_pin, prompt)).strip()
            if not pin.isdigit():
                raise CastError("PIN 碼必須是數字")
            pairing.pin(int(pin))
        else:
            pairing.pin(1234)
            say("請在 Apple TV 上輸入 PIN 碼 1234")
        await pairing.finish()
        if not pairing.has_paired:
            raise CastError("配對失敗（PIN 碼錯誤或已取消）")
        creds = pairing.service.credentials
        if not creds:
            raise CastError("配對失敗：裝置沒有回傳憑證")
        store.save_device(dev.id, dev.name, dev.address, str(creds))
        dev.paired = True
        say(f"已配對 {dev.name}；憑證存於 {store.path}（本機機密檔案，請勿分享）")
        return dev
    finally:
        await pairing.close()


# ---------------------------------------------------------------------------------------------------------------
# casting thread
# ---------------------------------------------------------------------------------------------------------------

def short_reason(e: BaseException) -> str:
    if isinstance(e, CastError):
        return str(e)
    if playback_status_rejected(e):
        return "Apple TV 拒絕播放狀態查詢（HTTP 403）；請查看 AirPlay 相容性診斷紀錄"
    name = type(e).__name__
    msg = (str(e).strip().splitlines() or [""])[0][:80]
    if name in ("AuthenticationError", "InvalidCredentialsError", "NoCredentialsError"):
        return "驗證失敗，請在投放選單選擇重新配對"
    if name in ("ConnectionFailedError", "ConnectionRefusedError", "TimeoutError", "OperationTimeoutError"):
        return "無法連線到 Apple TV"
    if name == "PlaybackError":
        return "Apple TV 無法播放" + (f"（{msg}）" if msg else "")
    return f"{name}: {msg}" if msg else name


def failure_details(error: BaseException) -> str:
    """Extract only allowlisted HTTP metadata; never dump frame locals or secrets."""
    endpoints = {"/pair-verify", "/pair-setup", "/pair-pin-start", "/auth-setup",
                 "/info", "/play", "/playback-info", "/feedback", "/reverse", "/command",
                 "/fp-setup", "/stop", "/rate", "/scrub"}
    methods = {"GET", "POST", "PUT", "SETUP", "RECORD", "TEARDOWN",
               "OPTIONS", "GET_PARAMETER", "SET_PARAMETER", "FLUSH"}
    details, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        tb = current.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if (frame.f_globals.get("__name__") == "pyatv.support.http"
                    and frame.f_code.co_name == "send_and_receive"):
                values = frame.f_locals
                method = values.get("method")
                uri = values.get("uri")
                code = getattr(values.get("response"), "code", None)
                details.append("request=" + (method if isinstance(method, str) and method in methods else "other")
                               + " " + (uri if isinstance(uri, str) and uri in endpoints else "<redacted>")
                               + " status=" + (str(code) if type(code) is int else "unknown"))
            tb = tb.tb_next
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return "; ".join(details) or "HTTP metadata unavailable"


def playback_status_rejected(error: BaseException) -> bool:
    return "request=GET /playback-info status=403" in failure_details(error).split("; ")


class _PlaybackInfoTap(logging.Handler):
    """pyatv polls the Apple TV's /playback-info every second while play_url waits, and logs each answer at DEBUG
    ("Playback-info: %s"). This handler (on that logger only, not propagated) keeps the latest position / rate.
    With pyatv_compat's video flow nothing is polled: the event channel state is stored here instead (set_state)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.state = None
        self.remote_paused = None

    def emit(self, record):
        try:
            if record.msg != "Playback-info: %s" or not record.args:
                return
            body = getattr(record.args[0], "body", None)
            if not body:
                return
            info = plistlib.loads(body if isinstance(body, bytes) else str(body).encode("utf-8"))
            if "position" in info:
                self.state = {"position": float(info["position"]), "rate": float(info.get("rate") or 0.0),
                              "duration": float(info["duration"]) if info.get("duration") else None,
                              "t": time.monotonic()}
        except Exception:  # noqa: BLE001 - purely informational
            pass

    def set_state(self, state: dict):
        self.state = dict(state)


_TAP = _PlaybackInfoTap()


def _install_tap():
    lg = logging.getLogger("pyatv.protocols.airplay.player")
    if _TAP not in lg.handlers:
        lg.addHandler(_TAP)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False       # its once-a-second debug lines stay out of our log


class AirPlayCaster:
    """One background asyncio loop for scanning and playing. on_event(kind, data) is called on that thread:
    ("scan", [Device]), ("scan-error", msg), ("connecting", name), ("playing", name), ("ended", name),
    ("stopped", name), ("error", msg)."""

    def __init__(self, store: CredentialStore, on_event, mod=None, log=print, wait_retries: int = 20):
        self.store, self.on_event, self.log = store, on_event, log
        self._mod_in, self.wait_retries = mod, wait_retries
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name="airplay", daemon=True)
        self.thread.start()
        self.play_task = None
        self.atv = None
        self.control_lock = asyncio.Lock()
        self.scan_future = None
        self.pair_future = None
        self.pin_future = None
        self.playing_name = None
        self.closed = False
        _install_tap()

    @staticmethod
    def tv_position() -> float | None:
        """The Apple TV's playback position (seconds, playlist time) from the latest playback-info, if recent, or
        from the event channel state (pushed only on changes, so it does not go stale)."""
        st = _TAP.state
        if not st:
            return None
        age = time.monotonic() - st["t"]
        if age > 15 and st.get("source") not in ("events", "clock"):
            return None
        rate = 0.0 if _TAP.remote_paused is True else st["rate"]
        return st["position"] + rate * age

    @staticmethod
    def tv_state() -> dict | None:
        state = dict(_TAP.state) if _TAP.state else None
        if _TAP.remote_paused is not None:
            state = state or {"position": 0.0, "t": time.monotonic()}
            state["rate"] = 0.0 if _TAP.remote_paused else 1.0
        return state

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _emit(self, kind, data=None):
        try:
            self.on_event(kind, data)
        except Exception as e:  # noqa: BLE001
            self.log(f"airplay event handler failed: {e!r}")

    def _mod(self):
        mod = _pyatv(self._mod_in)
        if self._mod_in is None:
            pyatv_compat.apply()
        try:   # the TV may take a while until the first segment plays (pyatv's default gives up after 5 s)
            mod.protocols.airplay.player.WAIT_RETRIES = max(5, self.wait_retries)
        except AttributeError:
            pass
        return mod

    @property
    def busy(self) -> bool:
        return self.play_task is not None and not self.play_task.done()

    # -- scan ----------------------------------------------------------------------------------------------------

    def scan(self, timeout=SCAN_TIMEOUT):
        if self.closed or (self.scan_future is not None and not self.scan_future.done()):
            return
        self.scan_future = asyncio.run_coroutine_threadsafe(self._scan(timeout), self.loop)

    async def _scan(self, timeout):
        try:
            mod = self._mod()
            confs = await scan_configs(self.loop, mod, timeout)
            devs = [device_from_config(c, mod, self.store) for c in confs]
            for d in devs:
                if d.paired and d.address:
                    self.store.update_address(d.id, d.address)
            self._emit("scan", merged_devices(devs, self.store))
        except Exception as e:  # noqa: BLE001
            self.log(f"airplay scan failed: {e!r}")
            self._emit("scan-error", short_reason(e))

    # -- play ----------------------------------------------------------------------------------------------------

    def pair(self, dev_id):
        if self.closed or (self.pair_future is not None and not self.pair_future.done()):
            return
        self.pair_future = asyncio.run_coroutine_threadsafe(self._pair(dev_id), self.loop)

    async def _pair(self, dev_id):
        async def ask_pin(prompt):
            self.pin_future = self.loop.create_future()
            self._emit("pair-pin", prompt)
            return await asyncio.wait_for(self.pin_future, timeout=120)

        try:
            dev = await pair_device(self.loop, dev_id, self.store, ask_pin,
                                    lambda msg: self._emit("pair-status", msg), self._mod())
            self._emit("paired", dev)
        except asyncio.CancelledError:
            self._emit("pair-cancelled")
        except Exception as e:
            self._emit("pair-error", short_reason(e))
        finally:
            self.pin_future = None

    def submit_pin(self, pin):
        def submit():
            if self.pin_future is not None and not self.pin_future.done():
                self.pin_future.set_result(pin)
        self.loop.call_soon_threadsafe(submit)

    def cancel_pair(self):
        if self.pair_future is not None:
            self.pair_future.cancel()

    def play(self, dev_id: str, url: str, position: float = 0.0):
        if self.closed:
            return
        self.stop()

        def start():
            self.play_task = self.loop.create_task(self._play(dev_id, url, position))
        done = threading.Event()
        self.loop.call_soon_threadsafe(lambda: (start(), done.set()))
        done.wait(5)

    async def _find(self, mod, dev_id, entry):
        addr = entry.get("address")
        if addr:
            try:
                confs = await scan_configs(self.loop, mod, 3, hosts=[addr])
                for c in confs:
                    if device_from_config(c, mod, None).id == dev_id:
                        return c
            except Exception as e:  # noqa: BLE001
                self.log(f"airplay unicast scan of {addr} failed: {e!r}")
        confs = await scan_configs(self.loop, mod, SCAN_TIMEOUT, identifier=dev_id)
        for c in confs:
            d = device_from_config(c, mod, None)
            if d.id == dev_id:
                self.store.update_address(dev_id, d.address)
                return c
        return None

    def control(self, action: str, position: float | None = None):
        """Send a remote command on the existing connection; never restart the stream."""
        if action not in ("seek", "pause", "play", "play_pause"):
            raise ValueError(f"unknown remote action: {action}")
        if self.closed:
            return
        atv = self.atv

        async def send():
            async with self.control_lock:
                try:
                    if atv is None or atv is not self.atv:
                        raise CastError("遙控通道尚未連線")
                    rc = atv.remote_control
                    previous = self.tv_state()
                    paused = bool(previous and previous.get("rate") == 0)
                    if action == "seek":
                        await asyncio.wait_for(rc.set_position(int(max(0, position))), timeout=10)
                    else:
                        await asyncio.wait_for(getattr(rc, action)(), timeout=10)
                        # Some receivers apply commands without pushing a new playback state.
                        # A successful command must still update the local play/pause icon.
                        target_paused = not paused if action == "play_pause" else action == "pause"
                        current_position = self.tv_position()
                        if _TAP.state and current_position is not None:
                            _TAP.state = {**_TAP.state, "position": current_position,
                                          "rate": 0.0 if target_paused else 1.0, "t": time.monotonic()}
                        _TAP.remote_paused = target_paused
                    self._emit("control-done", {"action": action, "position": position})
                except Exception as e:
                    self.log(f"airplay remote {action} failed: {e!r}")
                    self._emit("control-error", f"Apple TV 遙控失敗：{short_reason(e)}")

        return asyncio.run_coroutine_threadsafe(send(), self.loop)

    def playstatus_update(self, updater, playing):
        """MRP supplies actual receiver position and pause state."""
        state = getattr(playing.device_state, "name", "")
        if state in ("Playing", "Paused"):
            _TAP.remote_paused = state == "Paused"
        _TAP.set_state({"position": float(playing.position or 0),
                        "rate": 1.0 if state == "Playing" else 0.0,
                        "duration": playing.total_time, "t": time.monotonic(), "source": "events"})

    def playstatus_error(self, updater, exception):
        self.log(f"airplay remote status failed: {exception!r}")

    async def _play(self, dev_id, url, position):
        name = dev_id
        atv = None
        stage = "credentials"
        try:
            entry = self.store.get(dev_id)
            if not entry:
                raise CastError("此裝置尚未配對，請在投放選單選擇配對")
            name = entry.get("name") or dev_id
            self.playing_name = name
            _TAP.state = None
            _TAP.remote_paused = None
            self._emit("connecting", name)
            mod = self._mod()
            from importlib.metadata import version, PackageNotFoundError
            try:
                pyatv_version = version("pyatv")
            except PackageNotFoundError:
                pyatv_version = "unknown"
            self.log(f"airplay diagnostic: pyatv={pyatv_version}; {pyatv_compat.status()}")
            pyatv_compat.listener.on_state = _TAP.set_state
            pyatv_compat.listener.on_diag = lambda msg: self.log(f"airplay diagnostic: {msg}")
            proto = _protocol(mod)
            stage = "discovery"
            conf = await self._find(mod, dev_id, entry)
            if conf is None:
                raise CastError(f"找不到 {name}（請確認 Apple TV 已開機並在同一個網路）")
            conf.set_credentials(proto.AirPlay, entry["airplay"])
            for svc in conf.services:            # play_url only needs AirPlay
                if svc.protocol != proto.AirPlay:
                    svc.enabled = False
            storage = None
            try:
                from pyatv.settings import MrpTunnel  # noqa: PLC0415
                from pyatv.storage.memory_storage import MemoryStorage  # noqa: PLC0415
                storage = MemoryStorage()
                settings = await storage.get_settings(conf)
                settings.protocols.airplay.mrp_tunnel = MrpTunnel.Auto
            except Exception as settings_error:  # noqa: BLE001 - fake module in tests / older pyatv
                self.log(f"airplay diagnostic: MRP tunnel settings unavailable ({type(settings_error).__name__})")
                storage = None
            stage = "connect"
            atv = await (mod.connect(conf, self.loop, storage=storage) if storage is not None
                         else mod.connect(conf, self.loop))
            self.atv = atv
            try:
                atv.push_updater.listener = self
                atv.push_updater.start()
            except Exception as e:
                self.log(f"airplay remote push updates unavailable: {e!r}")
            self._emit("playing", name)
            self.log(f"airplay: play_url on {name} (position {position:.1f}s)")
            stage = "play_url"
            await atv.stream.play_url(url, position=int(position))
            self._emit("ended", name)
        except asyncio.CancelledError:
            self._emit("stopped", name)
        except Exception as e:  # noqa: BLE001 - reported, never raised
            self.log(f"airplay diagnostic: stage={stage}; {failure_details(e)}")
            self.log(f"airplay: playing on {name} failed: {e!r}")
            import traceback
            frames = traceback.extract_tb(e.__traceback__)
            self.log("airplay failure location: " + " -> ".join(f"{Path(f.filename).name}:{f.name}:{f.lineno}" for f in frames))
            if (type(e).__name__ in ("AuthenticationError", "InvalidCredentialsError", "NoCredentialsError")
                    and not playback_status_rejected(e)):
                self._emit("auth-error", {"device_id": dev_id, "reason": short_reason(e)})
            else:
                self._emit("error", short_reason(e))
        finally:
            self.playing_name = None
            if self.atv is atv:
                self.atv = None
            if atv is not None:
                try:
                    atv.push_updater.stop()
                except Exception:
                    pass
                try:
                    tasks = atv.close()
                    if tasks:
                        await asyncio.wait(tasks, timeout=3)
                except Exception as e:  # noqa: BLE001
                    self.log(f"airplay: close failed: {e!r}")

    def stop(self, wait: float = 5.0) -> bool:
        """Cancel playback (the Apple TV stops when the connection closes). True when nothing is left running."""
        task = self.play_task
        if task is None or self.closed and not self.thread.is_alive():
            return True
        done = threading.Event()

        def cancel():
            if task.done():
                done.set()
            else:
                task.add_done_callback(lambda _t: done.set())
                task.cancel()
        self.loop.call_soon_threadsafe(cancel)
        ok = done.wait(wait)
        if ok and self.play_task is task:
            self.play_task = None
        return ok

    def close(self):
        if self.closed:
            return
        self.stop()
        self.cancel_pair()
        # Drain cancellation so pairing and discovery release their connections.
        async def drain():
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if self.thread.is_alive():
            try:
                asyncio.run_coroutine_threadsafe(drain(), self.loop).result(timeout=5)
            except TimeoutError:
                self.log("airplay: connection cleanup timed out")
        self.closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        if not self.thread.is_alive():
            self.loop.close()


# ---------------------------------------------------------------------------------------------------------------
# cast menu (published to oh_seek.lua as user-data/oh/cast-menu)
# ---------------------------------------------------------------------------------------------------------------

PAIR_HINT = "選擇裝置即可在播放器內配對"


def build_cast_menu(casting: dict | None, devices: list[Device] | None, scanning: bool, scan_error: str | None,
                    pyatv_ok: bool = True) -> list[dict]:
    """Rows [{id, label, kind: item|info, hint?, active?}]. ids: dev:<id>, stop, hls, scan."""
    rows = []
    if casting:
        label = f"投放中：{casting['device']}" if casting.get("device") else "區網串流中"
        rows.append({"id": "", "label": label, "kind": "info", "active": True})
        if casting.get("url") and not casting.get("device"):
            rows.append({"id": "", "label": casting["url"], "kind": "info"})
        rows.append({"id": "stop", "label": "停止投放" if casting.get("device") else "停止區網串流", "kind": "item"})
        return rows
    if not pyatv_ok:
        rows.append({"id": "", "label": "AirPlay 無法使用（未安裝 pyatv）", "kind": "info"})
    else:
        for d in devices or []:
            if d.paired:
                rows.append({"id": f"dev:{d.id}", "kind": "item",
                             "label": f"投放到 {d.name}" + ("" if d.found else "（未找到）")})
                rows.append({"id": f"pair:{d.id}", "kind": "item", "label": f"重新配對 {d.name}"})
            else:
                rows.append({"id": f"pair:{d.id}", "kind": "item", "label": f"配對 {d.name}", "hint": PAIR_HINT})
        if scanning:
            rows.append({"id": "", "label": "搜尋裝置中…", "kind": "info"})
        elif scan_error:
            rows.append({"id": "", "label": f"搜尋失敗：{scan_error}", "kind": "info"})
        elif devices is not None and not devices:
            rows.append({"id": "", "label": "找不到 AirPlay 裝置", "kind": "info"})
        if not scanning:
            rows.append({"id": "scan", "label": "重新搜尋裝置", "kind": "item"})
        if devices and not any(d.paired for d in devices):
            rows.append({"id": "", "label": PAIR_HINT, "kind": "info"})
    rows.append({"id": "hls", "label": "僅開啟區網串流", "kind": "item"})
    return rows
