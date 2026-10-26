"""mpv JSON IPC: synchronous Windows pipe or Unix domain socket."""

import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path


class MpvEndpoint:
    def __init__(self):
        self._directory = None
        if sys.platform == "win32":
            self.path = rf"\\.\pipe\oh-mpv-{os.getpid()}"
        else:
            # macOS's per-user temporary path can exceed sockaddr_un's limit.
            self._directory = tempfile.TemporaryDirectory(prefix="oh-", dir="/tmp" if sys.platform == "darwin" else None)
            self.path = str(Path(self._directory.name) / "mpv.sock")

    def close(self):
        if self._directory is not None:
            self._directory.cleanup()


class MpvError(Exception):
    pass


class MpvIpc:
    """One thread owns reads/writes; event lines are discarded until the reply."""

    def __init__(self, path, player, timeout=15.0):
        self._socket = None
        deadline = time.perf_counter() + timeout
        while True:
            candidate = None
            try:
                if sys.platform == "win32":
                    self.f = open(path, "r+b", buffering=0)
                else:
                    candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    candidate.settimeout(max(0.05, deadline - time.perf_counter()))
                    candidate.connect(path)
                    candidate.settimeout(None)
                    self.f = candidate.makefile("rwb", buffering=0)
                    self._socket = candidate
                break
            except OSError as error:
                if candidate is not None:
                    candidate.close()
                if player.poll() is not None or time.perf_counter() > deadline:
                    raise RuntimeError(f"could not connect to mpv IPC at {path}") from error
                time.sleep(0.05)
        self.buf = b""
        self.request_id = 0

    def command(self, *args):
        self.request_id += 1
        payload = (json.dumps({"command": list(args), "request_id": self.request_id}) + "\n").encode()
        if self._socket is not None:
            self._socket.sendall(payload)
        else:
            self.f.write(payload)
        while True:
            msg = json.loads(self._readline())
            if msg.get("request_id") == self.request_id and "error" in msg:
                if msg["error"] != "success":
                    raise MpvError(f"{args[0]}: {msg['error']}")
                return msg.get("data")

    def get(self, name, default=None):
        try:
            return self.command("get_property", name)
        except MpvError:
            return default

    def set(self, name, value):
        return self.command("set_property", name, value)

    def _readline(self):
        while b"\n" not in self.buf:
            chunk = self.f.read(65536)
            if not chunk:
                raise EOFError("mpv IPC closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def close(self):
        try:
            self.f.close()
        finally:
            if self._socket is not None:
                self._socket.close()
