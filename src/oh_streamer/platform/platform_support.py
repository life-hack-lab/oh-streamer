"""Small OS services; optional Cocoa imports stay out of Windows startup."""

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path


APP_DIR_NAME = "oh_streamer"  # technical identifier: user-data folder name on every platform


def cache_dir() -> Path:
    """Disposable caches. Windows keeps them under data_dir(); macOS/Linux use the OS cache locations."""
    if sys.platform == "win32":
        return data_dir() / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP_DIR_NAME
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / APP_DIR_NAME


def data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_DIR_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP_DIR_NAME


def state_dir() -> Path:
    # Keep the existing Windows instance discovery location, including its fallback.
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / APP_DIR_NAME
    return data_dir()


def supports_shutdown() -> bool:
    return sys.platform == "win32"


def capabilities() -> dict:
    return {"shutdown": supports_shutdown(), "vsr": sys.platform == "win32",
            "rife": sys.platform == "win32"}


def gui_engine() -> str | None:
    return {"win32": "edgechromium", "darwin": "cocoa"}.get(sys.platform)


def detached_process_kwargs() -> dict:
    if sys.platform == "win32":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0) |
                 getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                 getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"creationflags": flags}
    return {"start_new_session": True}


def open_folder(path: str | Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(["/usr/bin/open" if sys.platform == "darwin" else "xdg-open", str(path)])


class PreventSleep:
    """Hold a cancellable sleep assertion. Windows calls must stay on one thread."""

    def __init__(self):
        self._proc = None

    def set_active(self, active: bool) -> None:
        if sys.platform == "win32":
            flags = 0x80000000 | (0x1 if active else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        elif sys.platform == "darwin":
            if active and (self._proc is None or self._proc.poll() is not None):
                self._proc = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            elif not active:
                self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
