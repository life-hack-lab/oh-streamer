"""Small standard-library report/command helpers for the portability tools."""
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA_VERSION = 1


def environment():
    return {"os": sys.platform, "os_release": platform.release(),
            "architecture": platform.machine(), "python": platform.python_version()}


def project_root(value=None):
    return Path(value or os.environ.get("OH_STREAMER_PROJECT_ROOT") or Path(__file__).resolve().parent.parent).resolve()


def child_environment(root=None):
    env = dict(os.environ)
    root = project_root(root)
    env["OH_STREAMER_PROJECT_ROOT"] = str(root)
    env.setdefault("LADA_MODEL_WEIGHTS_DIR", str(root / "lada" / "model_weights"))
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def last_json(output):
    try:
        value = json.loads(output.strip())
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    raise ValueError("command did not produce a JSON object")


class _WindowsJob:
    """Keep descendants owned even after their immediate parent exits."""
    def __init__(self, process):
        self.handle = None
        if os.name != "nt" or not isinstance(getattr(process, "_handle", None), int):
            return
        import ctypes
        from ctypes import wintypes as W
        class Basic(ctypes.Structure):
            _fields_ = [("user", ctypes.c_int64), ("job_user", ctypes.c_int64),
                        ("flags", W.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                        ("active", W.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", W.DWORD), ("scheduling", W.DWORD)]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
        self.api.CreateJobObjectW.restype = W.HANDLE
        self.api.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        self.api.CloseHandle.argtypes = [W.HANDLE]
        handle = self.api.CreateJobObjectW(None, None)
        if not handle:
            return
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (self.api.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info))
                and self.api.AssignProcessToJobObject(handle, process._handle)):
            self.handle = handle
        else:
            self.api.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _kill_owned(process):
    try:
        import psutil
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
    except Exception:
        # The diagnostic must still run before psutil is installed.
        if os.name == "nt" and process.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                               timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.TimeoutExpired):
                pass
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()


def run_command(argv, *, timeout=60, cwd=None, env=None, log_path=None):
    """Bound a process and its owned descendants; do not launch visible consoles."""
    start = time.perf_counter()
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    try:
        process = subprocess.Popen([str(x) for x in argv], cwd=cwd, env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", **kwargs)
    except OSError as error:
        result = {"returncode": None, "timed_out": False, "elapsed_s": time.perf_counter() - start,
                  "stdout": "", "stderr": str(error)}
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(error) + "\n", encoding="utf-8")
        return result
    job = _WindowsJob(process)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        job.close()
        _kill_owned(process)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired as error:
            stdout = error.output or ""
            stderr = error.stderr or ""
            stdout = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
            stderr += "\nOwned child pipe did not close after timeout cleanup."
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=5)
    except BaseException:
        job.close()
        _kill_owned(process)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.stdout.close()
            process.stderr.close()
        raise
    finally:
        job.close()
    result = {"returncode": 124 if timed_out else process.returncode,
              "process_returncode": process.returncode, "timed_out": timed_out,
              "elapsed_s": round(time.perf_counter() - start, 4), "stdout": stdout, "stderr": stderr}
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    return result


def positive_seconds(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be a finite positive number")
    return number
