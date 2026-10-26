r"""Setup single-instance lock and "is an env in use" detection. Stdlib only (ctypes): launch.pyw uses it on the embedded Python.

SetupLock: a named mutex Local\oh_streamer-setup-<hash of the install root>. The setup process holds it for its whole
life, so a second setup refuses to start and launch.pyw does not start the app while a setup window is open. The
handle is not inheritable and Windows releases it when the process ends, even after a crash. Outside Windows an
fcntl lock on <install_root>/setup.lock does the same.

processes_under(folder): processes whose executable lives under a folder (normally <install_root>\envs). The caller,
its descendants (uv, the self-check python) and its ancestors (a venv's Scripts\python.exe is a small launcher that
waits for the real interpreter, so the setup process's parent is an env executable too) are left out.
Windows: Toolhelp snapshot + QueryFullProcessImageNameW. Linux: /proc/<pid>/exe. Elsewhere: nothing is reported.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ERROR_ALREADY_EXISTS = 183
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TH32CS_SNAPPROCESS = 0x00000002
_MAX_DEPTH = 64                                 # parent-chain walk limit (pid reuse can form a cycle)


def mutex_name(install_root) -> str:
    """Per install root, so two separate installs (e.g. a dev tree and a test root) never block each other."""
    key = os.path.normcase(os.path.abspath(str(install_root))).encode("utf-8")
    return "Local\\oh_streamer-setup-" + hashlib.sha256(key).hexdigest()[:16]


def _lock_file(install_root) -> Path:
    return Path(install_root) / "setup.lock"


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k32.OpenMutexW.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.CloseHandle.restype = wintypes.BOOL
    k32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
    k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    k32.WaitForSingleObject.restype = wintypes.DWORD
    return k32


class SetupLock:
    """acquire() -> True when this process now holds the lock; False when another setup holds it."""

    def __init__(self, install_root) -> None:
        self._root = Path(install_root)
        self._handle = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        if sys.platform == "win32":
            import ctypes
            k32 = _kernel32()
            handle = k32.CreateMutexW(None, False, mutex_name(self._root))
            if not handle:
                return True                                             # cannot create a mutex at all: do not block the user
            if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
                k32.CloseHandle(handle)
                return False
            self._handle = handle
            return True
        try:
            import fcntl
            self._root.mkdir(parents=True, exist_ok=True)
            fh = open(_lock_file(self._root), "a+")
        except (ImportError, OSError):
            return True
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._handle = fh
        return True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                _kernel32().CloseHandle(handle)
            else:
                handle.close()
        except Exception:
            pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def setup_running(install_root) -> bool:
    """True while some process (not necessarily this one) holds the SetupLock of install_root. Never raises."""
    try:
        if sys.platform == "win32":
            k32 = _kernel32()
            handle = k32.OpenMutexW(_SYNCHRONIZE, False, mutex_name(install_root))
            if handle:
                k32.CloseHandle(handle)
                return True
            return False
        probe = SetupLock(install_root)
        if probe.acquire():
            probe.release()
            return False
        return True
    except Exception:
        return False


# ---- processes ----

@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    exe: str | None                             # full path; None when it cannot be read (other user, already gone)

    @property
    def name(self) -> str:
        return Path(self.exe).name if self.exe else ""


def _windows_processes() -> list[ProcessInfo]:
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    k32 = _kernel32()
    k32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32FirstW.restype = wintypes.BOOL
    k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.restype = wintypes.BOOL
    snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    pairs = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pairs.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)

    out = []
    buf = ctypes.create_unicode_buffer(32768)
    for pid, ppid in pairs:
        exe = None
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid) if pid else None
        if handle:
            try:
                size = wintypes.DWORD(len(buf))
                if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    exe = buf.value
            finally:
                k32.CloseHandle(handle)
        out.append(ProcessInfo(pid, ppid, exe))
    return out


def _proc_processes() -> list[ProcessInfo]:
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                ppid = int(fh.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        try:
            exe = os.readlink(f"/proc/{entry}/exe")
        except OSError:
            exe = None
        out.append(ProcessInfo(int(entry), ppid, exe))
    return out


def list_processes() -> list[ProcessInfo]:
    """Every process we can see; [] when the platform gives us no way to list them. Never raises."""
    try:
        if sys.platform == "win32":
            return _windows_processes()
        if os.path.isdir("/proc"):
            return _proc_processes()
    except Exception:
        pass
    return []


def _related(processes: list[ProcessInfo], pid: int) -> set[int]:
    """pid, its ancestors and its descendants."""
    parent = {p.pid: p.ppid for p in processes}
    keep = {pid}
    cur = pid
    for _ in range(_MAX_DEPTH):
        cur = parent.get(cur)
        if not cur or cur in keep:
            break
        keep.add(cur)
    children: dict[int, list[int]] = {}
    for p in processes:
        children.setdefault(p.ppid, []).append(p.pid)
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in keep:
                keep.add(child)
                stack.append(child)
    return keep


def _under(path: str, folder: str) -> bool:
    path = os.path.normcase(os.path.abspath(path))
    return path.startswith(folder.rstrip("\\/") + os.sep)


def processes_under(folder, *, exclude_pid: int | None = None, processes: list[ProcessInfo] | None = None) -> list[ProcessInfo]:
    """Processes whose executable is inside folder, minus exclude_pid (default: this process) and its ancestors and
    descendants. processes is for tests; by default the live list."""
    folder_key = os.path.normcase(os.path.abspath(str(folder)))
    procs = list_processes() if processes is None else processes
    skip = _related(procs, os.getpid() if exclude_pid is None else exclude_pid)
    return [p for p in procs if p.exe and p.pid not in skip and _under(p.exe, folder_key)]


def wait_for_exit(pid: int, timeout_s: float) -> bool:
    """True once pid is gone (or was never there), False on timeout."""
    import time
    if sys.platform == "win32":
        try:
            k32 = _kernel32()
            handle = k32.OpenProcess(_SYNCHRONIZE, False, pid)
        except Exception:
            return True
        if not handle:
            return True
        try:
            return k32.WaitForSingleObject(handle, int(timeout_s * 1000)) == 0
        finally:
            k32.CloseHandle(handle)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return True
        time.sleep(0.2)
    return False
