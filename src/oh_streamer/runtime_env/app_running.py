r"""Named mutex that tells the installer "the program is running". Stdlib only (ctypes); launch.pyw uses it too.

Every process that can lock the install folder (queue app, tray, background worker, player, setup mode, launcher)
calls hold_app_mutex() once at startup. The handle is never closed explicitly: Windows releases it when the process
ends, even after a crash. packaging/installer/oh_streamer.iss sets AppMutex to APP_MUTEX_NAME (keep them equal; a test
checks it), so Setup and the uninstaller ask the user to close the program first.
The name has no "Local\" or "Global\" prefix on purpose: that is the session namespace, the same as "Local\", and it is
what Inno Setup's AppMutex check also finds. Outside Windows nothing is done.
"""

from __future__ import annotations

import sys

APP_MUTEX_NAME = "oh_streamer-app"

_held = []                                      # keeps the handle referenced for the life of the process


def hold_app_mutex() -> bool:
    """Take the mutex for the rest of this process. True when held (also when another process already holds it, since
    every process opens the same name); False outside Windows or when it cannot be created. Never raises."""
    if sys.platform != "win32":
        return False
    if _held:
        return True
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateMutexW.restype = ctypes.c_void_p
        handle = k32.CreateMutexW(None, False, APP_MUTEX_NAME)
        if not handle:
            return False
        _held.append(handle)
        return True
    except Exception:
        return False


def app_mutex_exists() -> bool:
    """True when some process currently holds the mutex. Never raises; False outside Windows."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        k32.OpenMutexW.restype = ctypes.c_void_p
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.OpenMutexW(0x00100000, False, APP_MUTEX_NAME)      # SYNCHRONIZE
        if handle:
            k32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False
