"""
Single-instance support for the persistent player (oh_player.py --player).

The running player owns a named mutex (Local\\oh-player-<name>) and listens on 127.0.0.1:<random port>. Its port
and a random token are written to %LOCALAPPDATA%\\oh_streamer\\player-<name>.json (user-private folder). A second
`--player <input>` finds the mutex taken, reads that file, connects to 127.0.0.1:<port> and sends one JSON line
{"token": ..., "input": "...", "headers": {...}}; the player answers {"ok": true} and opens the input, the
second process exits. Connections from anything but 127.0.0.1 and messages without the right token are dropped.

<name> is "default" unless --instance NAME or the OH_STREAMER_PLAYER_INSTANCE environment variable says otherwise (tests use
their own name so they never talk to a player the user is running). Stdlib only, no torch import: forwarding stays
fast.
"""

import ctypes
import hmac
import json
import os
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

from oh_streamer.platform import platform_support


ERROR_ALREADY_EXISTS = 183
MAX_MESSAGE = 1_000_000


def instance_name(cli_value: str | None = None) -> str:
    name = cli_value or os.environ.get("OH_STREAMER_PLAYER_INSTANCE") or "default"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64] or "default"


def state_dir() -> Path:
    return platform_support.state_dir()


def _private_state_dir() -> Path:
    p = state_dir()
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    if sys.platform != "win32":
        if p.is_symlink() or p.stat().st_uid != os.getuid():
            raise OSError("instance state directory must be owned by the current user")
        p.chmod(0o700)
    return p


def info_path(name: str) -> Path:
    return state_dir() / f"player-{name}.json"


class Instance:
    """The primary instance: owns the mutex; serve() starts the localhost listener."""

    def __init__(self, name: str, mutex_handle, lock_file=None):
        self.name, self._mutex = name, mutex_handle
        self._lock_file = lock_file
        self.sock = None
        self.thread = None
        self.token = secrets.token_hex(32)
        self.closing = threading.Event()

    def serve(self, on_message, log=lambda m: None):
        """on_message(dict) -> dict reply; called on the listener thread (must not touch mpv IPC)."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        port = self.sock.getsockname()[1]
        p = info_path(self.name)
        _private_state_dir()
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "port": port, "token": self.token}, f)
        os.replace(tmp, p)
        self.thread = threading.Thread(target=self._run, args=(on_message, log), name="instance listener",
                                       daemon=True)
        self.thread.start()
        log(f"single instance '{self.name}': listening on 127.0.0.1:{port} ({p})")

    def _run(self, on_message, log):
        while not self.closing.is_set():
            try:
                conn, addr = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    if addr[0] != "127.0.0.1":
                        log(f"single instance: rejected connection from {addr[0]}")
                        continue
                    conn.settimeout(5)
                    data = b""
                    while b"\n" not in data and len(data) < MAX_MESSAGE:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    line, separator, _ = data.partition(b"\n")
                    if not separator or len(line) > MAX_MESSAGE:
                        log("single instance: rejected incomplete or oversized message")
                        continue
                    msg = json.loads(line.decode("utf-8"))
                    if not isinstance(msg, dict) or not hmac.compare_digest(str(msg.get("token", "")), self.token):
                        log("single instance: rejected message with a bad token")
                        continue
                    msg.pop("token", None)
                    reply = on_message(msg) or {"ok": True}
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                except Exception as e:  # noqa: BLE001 - a bad client must never take the listener down
                    log(f"single instance: bad message ({e!r})")

    def close(self):
        self.closing.set()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
        p = info_path(self.name)
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("pid") == os.getpid():
                p.unlink()
        except (OSError, ValueError):
            pass
        if self._mutex:
            ctypes.windll.kernel32.CloseHandle(self._mutex)
            self._mutex = None
        if self._lock_file is not None:
            import fcntl
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None


def acquire(name: str) -> Instance | None:
    """Instance if we are the first player with this name, None if one is already running."""
    if sys.platform != "win32":
        import fcntl
        p = _private_state_dir() / f"player-{name}.lock"
        fd = os.open(p, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        f = os.fdopen(fd, "a+")
        try:
            os.fchmod(f.fileno(), 0o600)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            return None
        except BaseException:
            f.close()
            raise
        return Instance(name, None, f)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    k32.CreateMutexW.restype = ctypes.c_void_p
    h = k32.CreateMutexW(None, False, f"Local\\oh-player-{name}")
    err = ctypes.get_last_error()
    if not h:
        return Instance(name, None)       # cannot tell: behave as the primary
    if err == ERROR_ALREADY_EXISTS:
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(h))
        return None
    return Instance(name, h)


def forward(name: str, payload: dict, timeout: float = 20.0, log=lambda m: None) -> dict | None:
    """Send payload to the running instance; retries while it is still starting. Reply dict or None."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            info = json.loads(info_path(name).read_text(encoding="utf-8"))
            with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=3) as s:
                s.sendall((json.dumps({**payload, "token": info["token"]}) + "\n").encode("utf-8"))
                s.settimeout(10)
                data = b""
                while b"\n" not in data:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            if data:
                return json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
            last = "no reply"
        except (OSError, ValueError, KeyError) as e:
            last = repr(e)
        time.sleep(0.3)
    log(f"could not reach the running player '{name}': {last}")
    return None
