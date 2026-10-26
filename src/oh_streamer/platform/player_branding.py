"""Best-effort Windows branding for the external mpv window."""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from oh_streamer.paths import resource_root
from oh_streamer.platform.windows_identity import set_window_app_id


_WINDOW_LOOKUP_TIMEOUT = 8.0
_WINDOW_LOOKUP_INTERVAL = 0.1
_PROCESS_POLL_INTERVAL = 0.25


class _WindowsApi:
    """Small, pointer-safe wrapper around the Win32 calls used here."""

    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1
    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x0010
    LR_DEFAULTSIZE = 0x0040
    SMTO_BLOCK = 0x0001
    SMTO_ABORTIFHUNG = 0x0002

    def __init__(self):
        import ctypes

        self._ctypes = ctypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)

        # Win32 BOOL is a 32-bit integer even on 64-bit Windows.
        self._enum_callback_type = ctypes.WINFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_ssize_t)
        self._user32.EnumWindows.argtypes = [self._enum_callback_type, ctypes.c_ssize_t]
        self._user32.EnumWindows.restype = ctypes.c_int32
        self._user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self._user32.GetWindowThreadProcessId.restype = ctypes.c_uint32
        self._user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
        self._user32.IsWindowVisible.restype = ctypes.c_int32
        self._user32.SendMessageTimeoutW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t,
            ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t),
        ]
        self._user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
        self._user32.LoadImageW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int,
            ctypes.c_int, ctypes.c_uint,
        ]
        self._user32.LoadImageW.restype = ctypes.c_void_p
        try:
            self._user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]
            self._user32.GetDpiForWindow.restype = ctypes.c_uint
        except AttributeError:
            pass

        self._user32.DestroyIcon.argtypes = [ctypes.c_void_p]
        self._user32.DestroyIcon.restype = ctypes.c_int32

    def find_window(self, target_pid: int) -> int | None:
        ctypes = self._ctypes
        found: list[int] = []

        def visit(hwnd: int, _lparam: int) -> bool:
            owner_pid = ctypes.c_uint32()
            self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
            if owner_pid.value == target_pid and self._user32.IsWindowVisible(hwnd):
                found.append(int(hwnd))
                return False
            return True

        callback = self._enum_callback_type(visit)
        self._user32.EnumWindows(callback, 0)
        return found[0] if found else None

    def window_dpi(self, hwnd: int) -> int:
        try:
            get_dpi = self._user32.GetDpiForWindow
            dpi = int(get_dpi(hwnd))
            return dpi if dpi > 0 else 96
        except (AttributeError, OSError, ValueError):
            return 96

    def load_icon(self, path: Path, width: int, height: int) -> int | None:
        flags = self.LR_LOADFROMFILE
        handle = self._user32.LoadImageW(None, str(path), self.IMAGE_ICON, width, height, flags)
        if not handle:
            # A valid ICO can still be usable when the requested DPI size is absent.
            handle = self._user32.LoadImageW(
                None, str(path), self.IMAGE_ICON, 0, 0, flags | self.LR_DEFAULTSIZE,
            )
        return int(handle) if handle else None

    def set_icon(self, hwnd: int, which: int, icon: int, timeout_ms: int = 500) -> bool:
        result = self._ctypes.c_size_t()
        sent = self._user32.SendMessageTimeoutW(
            hwnd,
            self.WM_SETICON,
            which,
            icon,
            self.SMTO_BLOCK | self.SMTO_ABORTIFHUNG,
            timeout_ms,
            self._ctypes.byref(result),
        )
        return bool(sent)

    def destroy_icon(self, icon: int) -> None:
        self._user32.DestroyIcon(icon)


def _icon_sizes(api: Any, hwnd: int) -> tuple[int, int]:
    """Return (small, large) icon sizes scaled for the target window's DPI."""
    try:
        dpi = int(api.window_dpi(hwnd))
    except Exception:
        dpi = 96
    if dpi <= 0:
        dpi = 96
    scale = dpi / 96.0
    return max(1, math.floor(16 * scale + 0.5)), max(1, math.floor(32 * scale + 0.5))


def _wait_for_window(
    process: Any,
    pid: int,
    api: Any,
    timeout: float,
    poll_interval: float,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
) -> int | None:
    deadline = clock() + max(0.0, timeout)
    while True:
        try:
            if process.poll() is not None:
                return None
            hwnd = api.find_window(pid)
        except Exception:
            return None
        if hwnd:
            return int(hwnd)
        now = clock()
        if now >= deadline:
            return None
        sleep(min(max(0.0, poll_interval), deadline - now))


def _run_branding_worker(
    process: Any,
    *,
    api: Any = None,
    icon_path: Path | None = None,
    lookup_timeout: float = _WINDOW_LOOKUP_TIMEOUT,
    lookup_interval: float = _WINDOW_LOOKUP_INTERVAL,
    process_poll_interval: float = _PROCESS_POLL_INTERVAL,
    clock=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Wait for mpv's own HWND, set both icons, and own HICONs until exit."""
    try:
        pid = process.pid
    except (AttributeError, TypeError, ValueError):
        return
    if type(pid) is not int or pid <= 0:
        return

    icons: set[int] = set()
    try:
        if api is None:
            api = _WindowsApi()
        hwnd = _wait_for_window(process, pid, api, lookup_timeout, lookup_interval,
                                clock=clock, sleep=sleep)
        if hwnd is None:
            return

        # Group the external process alongside the queue app even if icon loading
        # fails. Identity is independent of window decoration.
        set_window_app_id(hwnd)

        if icon_path is None:
            icon_path = resource_root() / "icons" / "app.ico"
        small_size, large_size = _icon_sizes(api, hwnd)
        small_icon = api.load_icon(icon_path, small_size, small_size)
        if small_icon:
            icons.add(int(small_icon))
        large_icon = api.load_icon(icon_path, large_size, large_size)
        if large_icon:
            icons.add(int(large_icon))

        if not icons:
            return
        # Keep each successfully loaded icon alive while mpv owns the window.
        if small_icon:
            try:
                api.set_icon(hwnd, _WindowsApi.ICON_SMALL, int(small_icon))
            except Exception:
                pass
        if large_icon:
            try:
                api.set_icon(hwnd, _WindowsApi.ICON_BIG, int(large_icon))
            except Exception:
                pass
        # If only one size loaded, use it for both slots.
        if small_icon and not large_icon:
            try:
                api.set_icon(hwnd, _WindowsApi.ICON_BIG, int(small_icon))
            except Exception:
                pass
        elif large_icon and not small_icon:
            try:
                api.set_icon(hwnd, _WindowsApi.ICON_SMALL, int(large_icon))
            except Exception:
                pass
    except Exception:
        # Window discovery and icon changes are cosmetic and must not affect playback.
        return
    finally:
        if api is not None:
            # If any HICON was loaded, keep it alive until mpv exits, even if a
            # later cosmetic Win32 operation failed.
            while icons:
                try:
                    if process.poll() is not None:
                        break
                except Exception:
                    break
                sleep(max(0.01, process_poll_interval))
            for icon in icons:
                try:
                    api.destroy_icon(icon)
                except Exception:
                    pass


def apply_player_branding(process: Any) -> None:
    """Apply the app icon to an external mpv window in a daemon thread.

    This is intentionally a no-op away from Windows. All lookup, loading, and
    Win32 messaging happen asynchronously and failures are ignored so playback
    remains independent of window decoration.
    """
    if sys.platform != "win32" or process is None:
        return
    try:
        pid = process.pid
        if type(pid) is not int or pid <= 0:
            return
        thread = threading.Thread(
            target=_run_branding_worker,
            args=(process,),
            name="mpv-window-branding",
            daemon=True,
        )
        thread.start()
    except Exception:
        pass
