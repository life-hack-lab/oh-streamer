"""Windows Shell identity behavior, including real per-window property readback."""
import ctypes
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import windows_identity as identity


class IdentityTests(unittest.TestCase):
    def test_other_platforms_are_noop(self):
        with patch.object(identity.sys, "platform", "darwin"):
            self.assertFalse(identity.set_process_app_id())
            self.assertFalse(identity.set_window_app_id(123))

    def test_process_api_failure_is_nonfatal(self):
        with patch.object(identity.sys, "platform", "win32"), \
                patch.object(ctypes, "WinDLL", create=True, side_effect=OSError("unavailable")):
            self.assertFalse(identity.set_process_app_id())
            self.assertFalse(identity.set_window_app_id(123))

    @unittest.skipUnless(sys.platform == "win32", "Windows Shell integration")
    def test_process_id_roundtrip_in_isolated_child(self):
        # Do not alter the test runner's taskbar identity.
        script = '''
import ctypes
from oh_streamer.platform.windows_identity import APP_USER_MODEL_ID, set_process_app_id
assert set_process_app_id()
shell = ctypes.WinDLL("shell32")
shell.GetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
shell.GetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_int32
value = ctypes.c_void_p()
assert shell.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(value)) >= 0
try:
    assert ctypes.wstring_at(value) == APP_USER_MODEL_ID
finally:
    ole = ctypes.WinDLL("ole32")
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.CoTaskMemFree(value)
'''
        import os
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=15)

    @unittest.skipUnless(sys.platform == "win32", "Windows Shell integration")
    def test_external_window_identity_roundtrip(self):
        import uuid
        script = '''
import ctypes, sys
user = ctypes.WinDLL("user32")
user.CreateWindowExW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_wchar_p,
    ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
user.CreateWindowExW.restype = ctypes.c_void_p
user.DestroyWindow.argtypes = [ctypes.c_void_p]
hwnd = user.CreateWindowExW(0, "STATIC", "Identity test", 0, 0, 0, 1, 1,
    None, None, None, None)
print(hwnd, flush=True)
try:
    sys.stdin.readline()
finally:
    user.DestroyWindow(hwnd)
'''
        child = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        def stop_child():
            try:
                child.communicate("\n", timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
        self.addCleanup(stop_child)
        hwnd = int(child.stdout.readline())
        self.assertTrue(hwnd)

        class KEY(ctypes.Structure):
            _fields_ = [("guid", ctypes.c_ubyte * 16), ("pid", ctypes.c_uint32)]

        class VALUE(ctypes.Union):
            _fields_ = [("pointer", ctypes.c_void_p), ("wide", ctypes.c_int64),
                        ("counted", ctypes.c_void_p * 2)]

        class VARIANT(ctypes.Structure):
            _fields_ = [("vt", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3),
                        ("value", VALUE)]

        store = ctypes.c_void_p()
        release = None
        value = VARIANT()
        try:
            self.assertTrue(identity.set_window_app_id(hwnd))
            iid = (ctypes.c_ubyte * 16).from_buffer_copy(
                uuid.UUID("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99").bytes_le)
            shell = ctypes.WinDLL("shell32")
            shell.SHGetPropertyStoreForWindow.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            shell.SHGetPropertyStoreForWindow.restype = ctypes.c_int32
            self.assertGreaterEqual(shell.SHGetPropertyStoreForWindow(hwnd, iid, ctypes.byref(store)), 0)
            table = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            release = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(table[2])
            get_value = ctypes.WINFUNCTYPE(ctypes.c_int32, ctypes.c_void_p,
                                           ctypes.POINTER(KEY), ctypes.POINTER(VARIANT))(table[5])
            key = KEY((ctypes.c_ubyte * 16).from_buffer_copy(
                uuid.UUID("9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3").bytes_le), 5)
            self.assertGreaterEqual(get_value(store, ctypes.byref(key), ctypes.byref(value)), 0)
            self.assertEqual(value.vt, 31)
            self.assertEqual(ctypes.wstring_at(value.value.pointer), identity.APP_USER_MODEL_ID)
        finally:
            ole = ctypes.WinDLL("ole32")
            ole.PropVariantClear.argtypes = [ctypes.POINTER(VARIANT)]
            ole.PropVariantClear(ctypes.byref(value))
            if release is not None:
                release(store)
