"""Stable Windows taskbar identity shared by the queue and external player."""
from __future__ import annotations

import sys

# Technical identifier: keep stable across display-name changes and upgrades.
APP_USER_MODEL_ID = "oh_streamer.Desktop"


def set_process_app_id() -> bool:
    """Set identity before creating UI, without making startup depend on Shell."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        shell = ctypes.WinDLL("shell32")
        setter = shell.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_int32
        return setter(APP_USER_MODEL_ID) >= 0
    except Exception:
        return False


def set_window_app_id(hwnd: int) -> bool:
    """Set the Shell property on a PID-verified external mpv window.

    Window properties override the owning process identity, so no injection or
    changes to the user's mpv binary/configuration are needed.
    """
    if sys.platform != "win32" or not hwnd:
        return False
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

        def __init__(self, value):
            super().__init__((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", ctypes.c_uint32)]

    class VALUE(ctypes.Union):
        _fields_ = [("pointer", ctypes.c_void_p), ("wide", ctypes.c_int64),
                    ("counted", ctypes.c_void_p * 2)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3),
                    ("value", VALUE)]

    store = ctypes.c_void_p()
    initialized = False
    variant = PROPVARIANT()
    release = None
    ole = None
    try:
        ole = ctypes.WinDLL("ole32")
        ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ole.CoInitializeEx.restype = ctypes.c_int32
        ole.CoUninitialize.argtypes = []
        ole.CoUninitialize.restype = None
        result = ole.CoInitializeEx(None, 0)  # MTA on the branding worker thread.
        initialized = result >= 0
        if result < 0 and result != -2147417850:  # RPC_E_CHANGED_MODE: already STA.
            return False
        shell = ctypes.WinDLL("shell32")
        shell.SHGetPropertyStoreForWindow.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
        shell.SHGetPropertyStoreForWindow.restype = ctypes.c_int32
        iid = GUID("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")  # IPropertyStore
        if shell.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid), ctypes.byref(store)) < 0:
            return False
        vtable = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        release = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(vtable[2])
        set_value = ctypes.WINFUNCTYPE(
            ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY),
            ctypes.POINTER(PROPVARIANT))(vtable[6])
        # VT_LPWSTR; SetValue copies the borrowed string synchronously.
        text = ctypes.create_unicode_buffer(APP_USER_MODEL_ID)
        variant.vt = 31
        variant.value.pointer = ctypes.cast(text, ctypes.c_void_p).value
        key = PROPERTYKEY(GUID("9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3"), 5)
        # Window property stores apply SetValue immediately; Commit is unnecessary.
        return set_value(store, ctypes.byref(key), ctypes.byref(variant)) >= 0
    except Exception:
        return False
    finally:
        if release is not None:
            release(store)
        if initialized and ole is not None:
            ole.CoUninitialize()
