"""uv sync runner: arguments, environment, log, line callback, failure message, cancel, torch prune after the GPU step.

uv is never run: a small Python script stands in for it and behaves according to STUB_* environment variables.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import installer, layout, torch_prune

LOCK_BYTES = b'version = 1\nrequires-python = "==3.13.*"\n'

STUB = r'''
import json, os, sys, time

def emit(text, stream=sys.stdout):
    stream.buffer.write(text.encode("utf-8") + b"\n")
    stream.buffer.flush()

pidfile = os.environ.get("STUB_PIDFILE")
if pidfile:
    with open(pidfile, "w") as fh:
        fh.write(str(os.getpid()))
for key in ("UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_NO_CONFIG", "UV_SYSTEM_CERTS"):
    emit(key + "=" + os.environ.get(key, "<unset>"))
emit("ARGS=" + json.dumps(sys.argv[1:]))
emit("CWD=" + os.getcwd())
for line in filter(None, os.environ.get("STUB_OUT", "").split("|")):
    emit(line)
for line in filter(None, os.environ.get("STUB_ERR", "").split("|")):
    emit(line, sys.stderr)
deadline = time.monotonic() + float(os.environ.get("STUB_SLEEP", "0"))
while time.monotonic() < deadline:
    time.sleep(0.05)
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
'''


def pid_alive(pid: int) -> bool:
    """Read-only liveness check (never signals the process: on Windows os.kill would terminate it)."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)               # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InstallerTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.root = self.tmp / "install"
        self.app = layout.app_dir(self.root)
        self.app.mkdir(parents=True)
        (self.app / "uv.lock").write_bytes(LOCK_BYTES)
        self.sha = layout.lock_sha256(self.app / "uv.lock")
        self.env_dir = layout.env_dir(self.root, self.sha)
        self.stub = self.tmp / "stub_uv.py"
        self.stub.write_text(STUB, encoding="utf-8")
        self.uv_cmd = [sys.executable, str(self.stub)]
        self.pidfile = self.tmp / "stub.pid"

    def stub_env(self, **values):
        return mock.patch.dict(os.environ, {"STUB_PIDFILE": str(self.pidfile), **{k: str(v) for k, v in values.items()}})

    def run_sync(self, args=("sync",), phase="base", **kw):
        return installer.run_uv_sync(self.uv_cmd, self.app, self.env_dir, self.root, list(args), phase=phase, **kw)

    @staticmethod
    def output_of(result) -> dict:
        """KEY=value lines the stub printed, from the log."""
        out = {}
        for line in result.log_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.startswith("#"):
                out[key] = value
        return out


class ArgsTests(unittest.TestCase):
    def test_base_args(self):
        args = installer.base_args()
        self.assertEqual(args[0], "sync")
        self.assertIn("--frozen", args)
        self.assertIn("--no-install-project", args)
        self.assertIn("--inexact", args)                                # keeps an installed GPU torch on re-runs
        skipped = [args[i + 1] for i, a in enumerate(args) if a == "--no-install-package"]
        self.assertEqual(skipped, ["torch", "torchvision", "torchaudio"])
        self.assertEqual(args[args.index("--extra") + 1], "subtitles")
        self.assertNotIn("nvidia", args)

    def test_gpu_args(self):
        args = installer.gpu_args("nvidia")
        self.assertEqual(args[0], "sync")
        self.assertIn("--frozen", args)
        self.assertIn("--no-install-project", args)
        self.assertNotIn("--no-install-package", args)
        extras = [args[i + 1] for i, a in enumerate(args) if a == "--extra"]
        self.assertEqual(extras, ["subtitles", "nvidia"])               # subtitles every time, or uv removes its packages

    def test_cpu_flavor_is_allowed_for_dev_tests(self):
        self.assertEqual(installer.gpu_args("cpu")[-2:], ["--extra", "cpu"])

    def test_unsupported_flavor(self):
        for flavor in ("rocm", "cu128", "", "NVIDIA"):
            with self.assertRaises(ValueError):
                installer.gpu_args(flavor)


class RunUvSyncTests(InstallerTestCase):
    def test_environment_reaches_uv(self):
        with self.stub_env():
            result = self.run_sync()
        self.assertTrue(result.ok, result.message)
        out = self.output_of(result)
        self.assertEqual(out["UV_PROJECT_ENVIRONMENT"], str(self.env_dir))
        self.assertEqual(out["UV_CACHE_DIR"], str(layout.cache_dir(self.root)))
        self.assertEqual(out["UV_NO_CONFIG"], "1")
        self.assertEqual(out["UV_SYSTEM_CERTS"], "1")
        self.assertEqual(out["CWD"], str(self.app))

    def test_env_and_cache_dirs_are_created(self):
        self.assertFalse(self.env_dir.exists())
        with self.stub_env():
            self.run_sync()
        self.assertTrue(self.env_dir.is_dir())
        self.assertTrue(layout.cache_dir(self.root).is_dir())

    def test_relative_env_dir_becomes_absolute(self):
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.tmp)
        with self.stub_env():
            result = installer.run_uv_sync(self.uv_cmd, self.app, Path("rel_env"), self.root, ["sync"], phase="base")
        self.assertEqual(self.output_of(result)["UV_PROJECT_ENVIRONMENT"], str(self.tmp / "rel_env"))

    def test_args_are_passed_without_python_by_default(self):
        with self.stub_env():
            result = self.run_sync(args=installer.base_args())
        self.assertEqual(json.loads(self.output_of(result)["ARGS"]), installer.base_args())

    def test_python_exe_adds_python_argument(self):
        python = self.tmp / "python" / "python.exe"
        with self.stub_env():
            result = self.run_sync(args=["sync", "--frozen"], python_exe=python)
        self.assertEqual(json.loads(self.output_of(result)["ARGS"]), ["sync", "--frozen", "--python", str(python)])

    @unittest.skipUnless(sys.platform == "win32", "CREATE_NO_WINDOW is Windows only")
    def test_no_console_window_on_windows(self):
        with self.stub_env(), mock.patch.object(installer.subprocess, "Popen", wraps=subprocess.Popen) as popen:
            self.run_sync()
        self.assertEqual(popen.call_args.kwargs["creationflags"] & 0x08000000, 0x08000000)

    def test_log_file(self):
        with self.stub_env(STUB_OUT="resolving|installed 3 packages|裝好了", STUB_ERR="a warning"):
            result = self.run_sync()
        self.assertTrue(result.log_path.is_file())
        self.assertEqual(result.log_path.parent, layout.logs_dir(self.root))
        self.assertRegex(result.log_path.name, r"^uv-\d{8}-\d{6}-base\.log$")
        text = result.log_path.read_text(encoding="utf-8")
        for expected in ("resolving", "installed 3 packages", "裝好了", "a warning", "ARGS="):
            self.assertIn(expected, text)
        self.assertLess(text.index("resolving"), text.index("a warning"))   # stdout and stderr share one ordered stream

    def test_log_name_uses_phase_and_never_collides(self):
        with self.stub_env():
            first = self.run_sync(phase="gpu-nvidia")
            second = self.run_sync(phase="gpu-nvidia")
        self.assertRegex(first.log_path.name, r"^uv-\d{8}-\d{6}-gpu-nvidia(-\d+)?\.log$")
        self.assertNotEqual(first.log_path, second.log_path)
        self.assertTrue(first.log_path.is_file() and second.log_path.is_file())

    def test_on_line_receives_every_line(self):
        lines = []
        with self.stub_env(STUB_OUT="one|two|三", STUB_ERR="err1"):
            result = self.run_sync(on_line=lines.append)
        self.assertTrue(result.ok)
        for expected in ("one", "two", "三", "err1"):
            self.assertIn(expected, lines)
        self.assertTrue(all("\n" not in line and "\r" not in line for line in lines))
        logged = [l for l in result.log_path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
        self.assertEqual(lines, logged)                                 # callback and log see the same lines

    def test_failing_on_line_does_not_break_the_run(self):
        def boom(_line):
            raise RuntimeError("ui gone")
        with self.stub_env(STUB_OUT="x"):
            result = self.run_sync(on_line=boom)
        self.assertTrue(result.ok)
        self.assertIn("x", result.log_path.read_text(encoding="utf-8"))

    def test_success_result(self):
        with self.stub_env():
            result = self.run_sync()
        self.assertEqual((result.ok, result.returncode, result.cancelled, result.prune_bytes_freed), (True, 0, False, 0))

    def test_nonzero_exit_is_failure_with_tail_in_message(self):
        out = "|".join(f"line{i}" for i in range(40))
        with self.stub_env(STUB_OUT=out, STUB_ERR="error: no matching distribution", STUB_EXIT=2):
            result = self.run_sync()
        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.cancelled)
        self.assertIn("error: no matching distribution", result.message)
        self.assertIn("line39", result.message)
        self.assertNotIn("line0\n", result.message + "\n")              # only the last lines, not the whole output
        self.assertIn("2", result.message)

    def test_uv_that_cannot_start(self):
        missing = str(self.tmp / "no_such_uv.exe")
        result = installer.run_uv_sync([missing], self.app, self.env_dir, self.root, ["sync"], phase="base")
        self.assertFalse(result.ok)
        self.assertIsNone(result.returncode)
        self.assertFalse(result.cancelled)
        self.assertTrue(result.log_path.is_file())
        self.assertIn("無法啟動 uv", result.message)

    def test_cancel_before_start_does_not_launch_uv(self):
        event = threading.Event()
        event.set()
        with self.stub_env():
            result = self.run_sync(cancel_event=event)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)
        self.assertFalse(self.pidfile.exists())

    def test_cancel_stops_the_process(self):
        event = threading.Event()

        def cancel_when_started():
            deadline = time.monotonic() + 15
            while not self.pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.5)
            event.set()

        thread = threading.Thread(target=cancel_when_started, daemon=True)
        thread.start()
        started = time.monotonic()
        with self.stub_env(STUB_OUT="working", STUB_SLEEP=30):
            result = self.run_sync(cancel_event=event)
        elapsed = time.monotonic() - started
        thread.join(5)

        self.assertLess(elapsed, 15)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)
        self.assertTrue(result.log_path.is_file())
        self.assertIn("working", result.log_path.read_text(encoding="utf-8"))
        pid = int(self.pidfile.read_text())
        self.assertFalse(pid_alive(pid), "stub uv process is still running after cancel")

    def test_finished_process_is_not_reported_as_cancelled(self):
        event = threading.Event()                                       # never set
        with self.stub_env():
            result = self.run_sync(cancel_event=event)
        self.assertTrue(result.ok)
        self.assertFalse(result.cancelled)


class InstallStepsTests(InstallerTestCase):
    def fake_prune(self, **result_fields):
        calls = []

        def prune(env_dir, cache_dir=None, dry_run=False):
            calls.append((Path(env_dir), Path(cache_dir) if cache_dir is not None else None))
            return torch_prune.PruneResult(**result_fields)
        return calls, prune

    def test_install_base(self):
        lines = []
        with self.stub_env(STUB_OUT="base ok"):
            result = installer.install_base(self.uv_cmd, self.root, on_line=lines.append)
        self.assertTrue(result.ok, result.message)
        self.assertIn("base ok", lines)
        self.assertRegex(result.log_path.name, r"-base\.log$")
        out = self.output_of(result)
        self.assertEqual(out["UV_PROJECT_ENVIRONMENT"], str(self.env_dir))
        self.assertEqual(out["CWD"], str(self.app))
        self.assertEqual(json.loads(out["ARGS"]), installer.base_args())

    def test_install_base_passes_python_exe(self):
        python = self.tmp / "embedded" / "python.exe"
        with self.stub_env():
            result = installer.install_base(self.uv_cmd, self.root, python_exe=python)
        self.assertEqual(json.loads(self.output_of(result)["ARGS"])[-2:], ["--python", str(python)])

    def test_install_gpu_prunes_in_the_same_env_as_base(self):
        calls, prune = self.fake_prune(files=["a.lib", "b.lib"], bytes_freed=3 * 2**30)
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            base = installer.install_base(self.uv_cmd, self.root)
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertTrue(base.ok and gpu.ok, gpu.message)
        base_env = Path(self.output_of(base)["UV_PROJECT_ENVIRONMENT"])
        self.assertEqual(Path(self.output_of(gpu)["UV_PROJECT_ENVIRONMENT"]), base_env)
        self.assertEqual(calls, [(base_env, layout.cache_dir(self.root))])
        self.assertEqual(base_env, self.env_dir)

    def test_install_gpu_result_and_log(self):
        _, prune = self.fake_prune(files=["a.lib", "b.lib"], bytes_freed=3 * 2**30)
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertTrue(gpu.ok)
        self.assertEqual(gpu.prune_bytes_freed, 3 * 2**30)
        self.assertRegex(gpu.log_path.name, r"-gpu-nvidia\.log$")
        self.assertEqual(json.loads(self.output_of(gpu)["ARGS"]), installer.gpu_args("nvidia"))
        self.assertIn("nvidia", gpu.message)
        self.assertIn("3.00 GiB", gpu.message)
        log = gpu.log_path.read_text(encoding="utf-8")
        self.assertIn("2 個檔案", log)
        self.assertIn("3.00 GiB", log)

    def test_prune_errors_are_logged_but_do_not_fail_the_install(self):
        _, prune = self.fake_prune(files=["a.lib"], bytes_freed=0, errors=[("b.lib", "access denied")])
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertTrue(gpu.ok)
        self.assertIn("1 個", gpu.message)
        log = gpu.log_path.read_text(encoding="utf-8")
        self.assertIn("b.lib", log)
        self.assertIn("access denied", log)

    def test_prune_exception_does_not_fail_the_install(self):
        def prune(*_args, **_kwargs):
            raise PermissionError("locked by antivirus")
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertTrue(gpu.ok)
        self.assertEqual(gpu.prune_bytes_freed, 0)
        self.assertIn("locked by antivirus", gpu.message)
        self.assertIn("locked by antivirus", gpu.log_path.read_text(encoding="utf-8"))

    def test_failed_gpu_install_does_not_prune(self):
        calls, prune = self.fake_prune(bytes_freed=1)
        with self.stub_env(STUB_ERR="download failed", STUB_EXIT=1), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertFalse(gpu.ok)
        self.assertEqual(gpu.returncode, 1)
        self.assertIn("download failed", gpu.message)
        self.assertEqual(calls, [])

    def test_cancelled_gpu_install_does_not_prune(self):
        calls, prune = self.fake_prune(bytes_freed=1)
        event = threading.Event()
        event.set()
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            gpu = installer.install_gpu(self.uv_cmd, self.root, "nvidia", cancel_event=event)
        self.assertTrue(gpu.cancelled)
        self.assertFalse(gpu.ok)
        self.assertEqual(calls, [])

    def test_unsupported_flavor_fails_before_running_anything(self):
        with self.stub_env():
            with self.assertRaises(ValueError):
                installer.install_gpu(self.uv_cmd, self.root, "rocm")
        self.assertFalse(self.pidfile.exists())
        self.assertFalse(layout.logs_dir(self.root).exists())

    def test_missing_lock_raises(self):
        (self.app / "uv.lock").unlink()
        with self.assertRaises(OSError):
            installer.install_base(self.uv_cmd, self.root)

    def test_install_does_not_write_current_json(self):
        calls, prune = self.fake_prune()
        with self.stub_env(), mock.patch.object(torch_prune, "prune_torch_static_libs", prune):
            installer.install_base(self.uv_cmd, self.root)
            installer.install_gpu(self.uv_cmd, self.root, "nvidia")
        self.assertFalse((self.root / layout.CURRENT_FILE).exists())


if __name__ == "__main__":
    unittest.main()
