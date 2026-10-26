"""setup_mode.bridge：detect()、安裝流程（runtime_env.provision 經由 bridge 跑：成功、失敗、取消、重複開始）、launch_main、診斷包。

installer／selfcheck／gpu_detect 全部換成假的：不下載、不跑 uv、不啟動 GUI、不開真的子行程。
uv 與 Python 的尋找規則在 test_launch_env 測。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform import platform_support
from oh_streamer.runtime_env import diagnostics, flavors, gpu_detect, installer, launch_env, layout, preflight, selfcheck
from oh_streamer.runtime_env.gpu_detect import Detection, GpuInfo
from oh_streamer.runtime_env.installer import SyncResult
from oh_streamer.runtime_env.selfcheck import SelfCheckResult
from oh_streamer.setup_mode import __main__ as setup_main
from oh_streamer.setup_mode import bridge
from oh_streamer.setup_mode.bridge import SetupApi

RTX_5060 = GpuInfo("NVIDIA GeForce RTX 5060 Laptop GPU", "nvidia", "576.02", (12, 0))
INTEL_ARC = GpuInfo("Intel(R) Arc(TM) Graphics", "intel", "32.0.101.6078", None)
LOG = Path("fake-uv.log")
WAIT_S = 10


def ok_result(message="uv sync 完成"):
    return SyncResult(True, 0, LOG, False, message)


def fail_result(message="uv 結束碼 2"):
    return SyncResult(False, 2, LOG, False, message)


def cancelled_result():
    return SyncResult(False, None, LOG, True, "已取消")


def check_ok(device="NVIDIA GeForce RTX 5060 Laptop GPU"):
    return SelfCheckResult(True, True, device, True, f"自我檢查通過：{device}")


def check_failed(message="CUDA 不可用"):
    return SelfCheckResult(False, False, None, False, message)


class BridgeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="setup_bridge_")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop("OH_STREAMER_UV", None)

        self.root = Path(self._tmp) / "install"
        (self.root / "app").mkdir(parents=True)
        self.lock = self.root / "app" / "uv.lock"
        self.lock.write_text("version = 1\n", encoding="utf-8")
        (self.root / "uv.exe").write_bytes(b"")
        self.sha = layout.lock_sha256(self.lock)
        self.env = layout.env_dir(self.root, self.sha)

        self.api = SetupApi(self.root)
        self.addCleanup(self._stop_install)

    def _stop_install(self):
        thread = self.api._thread
        if thread is not None and thread.is_alive():
            self.api._cancel_event.set()
            thread.join(WAIT_S)

    def make_env_python(self, env=None):
        python = layout.python_in_env(env or self.env)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")
        return python

    def write_current(self, *, sha=None, flavor="nvidia", verified=True):
        layout.write_current(self.root, layout.Current(
            lock_sha256=sha or self.sha, flavor=flavor, verified=verified, updated_at="2026-10-24T00:00:00+00:00"))

    def current_json(self):
        return self.root / "current.json"

    def run_install(self, flavor="nvidia"):
        result = self.api.start_install(flavor)
        self.assertTrue(result["ok"], result)
        self.api._thread.join(WAIT_S)
        self.assertFalse(self.api._thread.is_alive(), "安裝執行緒沒有結束")
        return self.api.poll()


class SurfaceTests(BridgeTestCase):
    def test_only_the_six_api_methods_are_public(self):
        """pywebview 會暴露所有不以底線開頭的屬性；window 與內部狀態都不能出現在 dir() 裡。"""
        self.api.window = object()
        public = sorted(name for name in dir(self.api) if not name.startswith("_"))
        self.assertEqual(public, ["cancel", "detect", "export_diagnostics", "launch_main", "poll", "start_install"])

    def test_initial_poll(self):
        self.assertEqual(self.api.poll(), {"phase": "idle", "lines": [], "message": "", "device_name": None,
                                           "phase_progress": None})


class DetectTests(BridgeTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(preflight, "webview2_installed", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def detect_with(self, detection):
        with patch.object(gpu_detect, "detect", return_value=detection):
            return self.api.detect()

    def test_no_gpu(self):
        result = self.detect_with(Detection([], False))
        json.dumps(result)
        self.assertTrue(result["ok"])
        self.assertEqual(result["gpus"], [])
        self.assertIsNone(result["recommendation"]["flavor"])
        self.assertEqual(result["recommendation"]["status"], "cpu_only")
        self.assertEqual(result["recommendation"]["warnings"], [])
        self.assertEqual(result["options"], [
            {"flavor": "nvidia", "label": "NVIDIA（CUDA 12.8）", "supported": True},
            {"flavor": "intel", "label": "Intel（Arc／Core Ultra）", "supported": False},
            {"flavor": "amd", "label": "AMD（ROCm，實驗性）", "supported": False},
        ])
        self.assertEqual(result["supported_cards"], list(flavors.supported_cards()))
        self.assertEqual(len(result["supported_cards"]), len(flavors.offered()))
        self.assertTrue(all(isinstance(c, str) and c for c in result["supported_cards"]))
        self.assertIsNone(result["current"])
        self.assertFalse(result["base_ready"])
        self.assertEqual(result["estimated_download_gb"], 4.0)
        self.assertGreater(result["required_disk_gb"], result["estimated_download_gb"])   # peak use is env + cache, not the download
        self.assertTrue(result["webview2_installed"])
        self.assertIsInstance(result["disk_free_gb"], float)
        self.assertEqual(result["disk_free_gb"], round(result["disk_free_gb"], 1))

    def test_rtx_recommends_nvidia(self):
        result = self.detect_with(Detection([RTX_5060], True))
        json.dumps(result)
        self.assertEqual(result["recommendation"]["flavor"], "nvidia")
        self.assertEqual(result["recommendation"]["status"], "ok")
        self.assertEqual(result["gpus"], [{
            "name": RTX_5060.name, "vendor": "nvidia", "driver_version": "576.02", "compute_capability": "12.0"}])

    def test_gpu_without_compute_capability_is_null(self):
        result = self.detect_with(Detection([INTEL_ARC], False))
        self.assertIsNone(result["gpus"][0]["compute_capability"])

    def test_intel_is_not_offered_because_only_nvidia_is_supported_now(self):
        result = self.detect_with(Detection([INTEL_ARC], False))
        self.assertIsNone(result["recommendation"]["flavor"])
        self.assertEqual(result["recommendation"]["status"], "intel_not_offered")

    def test_old_driver_warning_is_passed_through_as_codes(self):
        old = GpuInfo(RTX_5060.name, "nvidia", "551.23", (12, 0))
        result = self.detect_with(Detection([old], True))
        self.assertEqual(result["recommendation"]["warnings"], ["driver_below_570"])

    def test_current_and_base_ready_reflect_the_install_root(self):
        self.write_current(flavor="nvidia", verified=True)
        self.make_env_python()
        result = self.detect_with(Detection([RTX_5060], True))
        self.assertEqual(result["current"], {"flavor": "nvidia", "verified": True})
        self.assertTrue(result["base_ready"])

    def test_base_only_current_has_null_flavor(self):
        self.write_current(flavor=None, verified=False)
        result = self.detect_with(Detection([], False))
        self.assertEqual(result["current"], {"flavor": None, "verified": False})

    def test_missing_install_root_still_reports_disk_space(self):
        api = SetupApi(Path(self._tmp) / "does" / "not" / "exist")
        with patch.object(gpu_detect, "detect", return_value=Detection([], False)):
            result = api.detect()
        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["disk_free_gb"])
        self.assertFalse(result["base_ready"])

    def test_detect_failure_becomes_ok_false(self):
        with patch.object(gpu_detect, "detect", side_effect=RuntimeError("boom")):
            result = self.api.detect()
        self.assertFalse(result["ok"])
        self.assertIn("boom", result["message"])
        json.dumps(result)

    def test_last_detection_is_kept_for_diagnostics(self):
        result = self.detect_with(Detection([RTX_5060], True))
        self.assertEqual(self.api._last_detection, result)
        self.assertIsNot(self.api._last_detection, result)


class InstallTests(BridgeTestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.phases = []
        self.base = MagicMock(side_effect=self.fake_base)
        self.gpu = MagicMock(side_effect=self.fake_gpu)
        self.check = MagicMock(side_effect=self.fake_check)
        for target, mock in ((installer, ("install_base", self.base)), (installer, ("install_gpu", self.gpu)),
                             (selfcheck, ("run_self_check", self.check))):
            patcher = patch.object(target, mock[0], mock[1])
            patcher.start()
            self.addCleanup(patcher.stop)

    # 預設的假步驟：成功。個別測試覆寫 side_effect。
    def fake_base(self, uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
        self.phases.append(self.api.poll()["phase"])
        self.make_env_python(env_dir)
        return ok_result()

    def fake_gpu(self, uv_cmd, install_root, flavor, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
        self.phases.append(self.api.poll()["phase"])
        return ok_result(f"GPU 版（{flavor}）安裝完成")

    def fake_check(self, python_exe, timeout_s=300):
        self.phases.append(self.api.poll()["phase"])
        return check_ok()

    def test_rejects_unsupported_flavor_without_starting_a_thread(self):
        for flavor in ("rocm", "intel", "amd", "cpu", "", None):
            result = self.api.start_install(flavor)
            self.assertFalse(result["ok"], flavor)
            self.assertTrue(result["message"])
        self.assertIsNone(self.api._thread)
        self.assertEqual(self.api.poll()["phase"], "idle")
        self.base.assert_not_called()

    def test_success_path(self):
        started = self.api.start_install("nvidia")
        self.assertEqual(started["ok"], True)
        json.dumps(started)
        self.api._thread.join(WAIT_S)
        self.assertFalse(self.api._thread.is_alive())
        self.assertTrue(self.api._thread.daemon)

        self.assertEqual(self.phases, ["base", "gpu", "selfcheck"])
        polled = self.api.poll()
        json.dumps(polled)
        self.assertEqual(polled["phase"], "done")
        self.assertEqual(polled["device_name"], "NVIDIA GeForce RTX 5060 Laptop GPU")

        data = json.loads(self.current_json().read_text(encoding="utf-8"))
        self.assertEqual(data["flavor"], "nvidia")
        self.assertIs(data["verified"], True)
        self.assertEqual(data["lock_sha256"], self.sha)
        self.assertEqual(data["schema"], layout.SCHEMA)
        self.assertTrue(data["updated_at"])
        self.assertEqual(layout.status(self.root, self.lock), layout.READY)

    def test_steps_get_the_expected_arguments(self):
        python = self.root / "python" / "python.exe"
        python.parent.mkdir()
        python.write_bytes(b"")
        self.run_install()

        uv_cmd = [str(self.root / "uv.exe")]
        base_args = self.base.call_args.args
        self.assertEqual(base_args[:3], (uv_cmd, self.root, python))
        gpu_args = self.gpu.call_args.args
        self.assertEqual(gpu_args[:4], (uv_cmd, self.root, "nvidia", python))
        cancel_event = base_args[4]
        self.assertIsInstance(cancel_event, threading.Event)
        self.assertIs(gpu_args[5], cancel_event)
        self.check.assert_called_once_with(layout.python_in_env(self.root / "envs" / f"{self.sha[:8]}-nvidia-1"))

    def test_without_embedded_python_uv_picks_one(self):
        self.run_install()
        self.assertIsNone(self.base.call_args.args[2])

    def test_selfcheck_failure(self):
        self.check.side_effect = lambda *a, **k: check_failed("CUDA 不可用")
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertTrue(polled["message"].startswith("自我檢查失敗："))
        self.assertIn("CUDA 不可用", polled["message"])
        self.assertIsNone(polled["device_name"])
        self.assertFalse(self.current_json().exists())

    def test_selfcheck_failure_keeps_an_existing_current_json_untouched(self):
        self.write_current(sha="a" * 64, flavor="nvidia", verified=True)
        before = self.current_json().read_bytes()
        self.check.side_effect = lambda *a, **k: check_failed()
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertEqual(self.current_json().read_bytes(), before)

    def test_base_failure_skips_the_rest(self):
        self.base.side_effect = lambda *a, **k: fail_result("uv 結束碼 2：\nnetwork down")
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertTrue(polled["message"].startswith("基礎環境安裝失敗："))
        self.assertIn("network down", polled["message"])
        self.gpu.assert_not_called()
        self.check.assert_not_called()
        self.assertFalse(self.current_json().exists())

    def test_gpu_failure_skips_selfcheck(self):
        self.gpu.side_effect = lambda *a, **k: fail_result("uv 結束碼 1")
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertTrue(polled["message"].startswith("GPU 版安裝失敗："))
        self.check.assert_not_called()
        self.assertFalse(self.current_json().exists())

    def test_unexpected_exception_becomes_failed(self):
        self.base.side_effect = RuntimeError("kaboom")
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertIn("kaboom", polled["message"])
        self.assertFalse(self.current_json().exists())
        self.assertFalse(self.api.cancel()["ok"])           # 執行緒死掉後不能卡在「安裝中」

    def test_write_current_failure_becomes_failed(self):
        with patch.object(layout, "write_current", side_effect=OSError("disk full")):
            polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertIn("disk full", polled["message"])

    def test_missing_lock_file_fails_before_any_step(self):
        self.lock.unlink()
        polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertIn("uv.lock", polled["message"])
        self.base.assert_not_called()

    def test_missing_uv_fails_before_any_step(self):
        (self.root / "uv.exe").unlink()
        with patch.object(shutil, "which", return_value=None):
            polled = self.run_install()
        self.assertEqual(polled["phase"], "failed")
        self.assertIn("uv", polled["message"])
        self.base.assert_not_called()
        self.assertFalse(self.current_json().exists())

    def test_cancel_during_base(self):
        in_base = threading.Event()

        def waiting_base(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            in_base.set()
            self.assertTrue(cancel_event.wait(WAIT_S), "cancel_event 沒有被設定")
            return cancelled_result()

        self.base.side_effect = waiting_base
        self.assertTrue(self.api.start_install("nvidia")["ok"])
        self.assertTrue(in_base.wait(WAIT_S))
        self.assertEqual(self.api.poll()["phase"], "base")

        cancelled = self.api.cancel()
        json.dumps(cancelled)
        self.assertTrue(cancelled["ok"])
        self.api._thread.join(WAIT_S)
        self.assertFalse(self.api._thread.is_alive())

        polled = self.api.poll()
        self.assertEqual(polled["phase"], "cancelled")
        self.gpu.assert_not_called()
        self.check.assert_not_called()
        self.assertFalse(self.current_json().exists())

    def test_cancel_during_gpu_keeps_existing_current_json(self):
        self.write_current(sha="b" * 64, flavor="nvidia", verified=True)
        before = self.current_json().read_bytes()
        in_gpu = threading.Event()

        def waiting_gpu(uv_cmd, install_root, flavor, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            in_gpu.set()
            cancel_event.wait(WAIT_S)
            return cancelled_result()

        self.gpu.side_effect = waiting_gpu
        self.assertTrue(self.api.start_install("nvidia")["ok"])
        self.assertTrue(in_gpu.wait(WAIT_S))
        self.assertTrue(self.api.cancel()["ok"])
        self.api._thread.join(WAIT_S)
        self.assertEqual(self.api.poll()["phase"], "cancelled")
        self.check.assert_not_called()
        self.assertEqual(self.current_json().read_bytes(), before)

    def test_cancel_during_selfcheck_wins_and_writes_nothing(self):
        def check_then_cancel(python_exe, timeout_s=300):
            self.assertTrue(self.api.cancel()["ok"])
            return check_ok()

        self.check.side_effect = check_then_cancel
        polled = self.run_install()
        self.assertEqual(polled["phase"], "cancelled")
        self.assertFalse(self.current_json().exists())

    def test_cancel_without_an_install_is_refused(self):
        result = self.api.cancel()
        self.assertFalse(result["ok"])
        self.assertTrue(result["message"])
        self.run_install()
        self.assertFalse(self.api.cancel()["ok"])           # 已結束的安裝也不能取消

    def test_second_start_while_running_is_refused(self):
        release = threading.Event()
        in_base = threading.Event()

        def blocking_base(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            in_base.set()
            release.wait(WAIT_S)
            self.make_env_python(env_dir)
            return ok_result()

        self.base.side_effect = blocking_base
        self.assertTrue(self.api.start_install("nvidia")["ok"])
        self.assertTrue(in_base.wait(WAIT_S))
        first_thread = self.api._thread

        second = self.api.start_install("nvidia")
        self.assertFalse(second["ok"])
        self.assertTrue(second["message"])
        self.assertIs(self.api._thread, first_thread)

        release.set()
        first_thread.join(WAIT_S)
        self.assertEqual(self.api.poll()["phase"], "done")
        self.assertEqual(self.base.call_count, 1)

    def test_can_retry_after_a_failure(self):
        self.base.side_effect = lambda *a, **k: fail_result()
        self.assertEqual(self.run_install()["phase"], "failed")
        self.base.side_effect = self.fake_base
        polled = self.run_install()
        self.assertEqual(polled["phase"], "done")
        self.assertTrue(self.current_json().exists())

    def test_retry_clears_the_previous_run(self):
        def noisy_failure(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            on_line("old line")
            return fail_result("壞掉了")

        self.base.side_effect = noisy_failure
        first = self.run_install()
        self.assertEqual(first["lines"], ["old line"])
        self.base.side_effect = self.fake_base
        second = self.run_install()
        self.assertEqual(second["lines"], [])
        self.assertEqual(second["phase"], "done")

    def test_poll_returns_only_the_last_80_lines(self):
        def chatty_base(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            for n in range(100):
                on_line(f"line {n}")
            self.make_env_python(env_dir)
            return ok_result()

        self.base.side_effect = chatty_base
        polled = self.run_install()
        self.assertEqual(len(polled["lines"]), 80)
        self.assertEqual(polled["lines"], [f"line {n}" for n in range(20, 100)])

    def test_lines_are_visible_to_poll_while_the_step_is_still_running(self):
        pushed = threading.Event()
        release = threading.Event()

        def streaming_base(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            on_line("Resolved 123 packages")
            pushed.set()
            release.wait(WAIT_S)
            self.make_env_python(env_dir)
            return ok_result()

        self.base.side_effect = streaming_base
        self.api.start_install("nvidia")
        self.assertTrue(pushed.wait(WAIT_S))
        self.assertEqual(self.api.poll()["lines"], ["Resolved 123 packages"])
        release.set()
        self.api._thread.join(WAIT_S)

    def test_shutdown_cancels_a_running_install_and_waits_for_it(self):
        in_base = threading.Event()

        def waiting_base(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            in_base.set()
            cancel_event.wait(WAIT_S)
            return cancelled_result()

        self.base.side_effect = waiting_base
        self.api.start_install("nvidia")
        self.assertTrue(in_base.wait(WAIT_S))
        self.api._shutdown(WAIT_S)
        self.assertFalse(self.api._thread.is_alive())
        self.assertEqual(self.api.poll()["phase"], "cancelled")
        self.assertFalse(self.current_json().exists())

    def test_shutdown_without_an_install_is_a_noop(self):
        self.api._shutdown(0.1)
        self.assertEqual(self.api.poll()["phase"], "idle")

    def test_phase_progress_is_visible_during_the_step_and_cleared_after(self):
        reported = threading.Event()
        release = threading.Event()
        seen = {}

        def downloading_gpu(uv_cmd, install_root, flavor, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            seen["callback"] = on_progress is not None
            on_progress({"downloaded_bytes": 1_200_000_000, "total_bytes": 3_100_000_000, "speed_bps": 12_300_000.4, "eta_s": 150})
            reported.set()
            release.wait(WAIT_S)
            return ok_result()

        self.gpu.side_effect = downloading_gpu
        self.assertIsNone(self.api.poll()["phase_progress"])
        self.api.start_install("nvidia")
        self.assertTrue(reported.wait(WAIT_S))
        polled = self.api.poll()
        json.dumps(polled)
        self.assertEqual(polled["phase"], "gpu")
        self.assertEqual(polled["phase_progress"],
                         {"downloaded_bytes": 1_200_000_000, "total_bytes": 3_100_000_000, "speed_bps": 12_300_000, "eta_s": 150})
        self.assertTrue(seen["callback"])
        release.set()
        self.api._thread.join(WAIT_S)
        self.assertIsNone(self.api.poll()["phase_progress"])                    # done: no stale download figures

    def test_base_progress_does_not_leak_into_the_gpu_phase(self):
        def base_with_progress(uv_cmd, install_root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            on_progress({"downloaded_bytes": 5, "total_bytes": 10, "speed_bps": None, "eta_s": None})
            self.make_env_python(env_dir)
            return ok_result()

        def gpu_checks(uv_cmd, install_root, flavor, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
            self.phases.append(self.api.poll()["phase_progress"])
            return ok_result()

        self.base.side_effect = base_with_progress
        self.gpu.side_effect = gpu_checks
        self.run_install()
        self.assertEqual(self.phases[0], None)

    def test_progress_dict_rejects_bad_snapshots(self):
        from oh_streamer.setup_mode.bridge import _progress_dict
        self.assertIsNone(_progress_dict(None))
        self.assertIsNone(_progress_dict({"downloaded_bytes": 1}))
        self.assertIsNone(_progress_dict({"downloaded_bytes": 0, "total_bytes": 0}))
        self.assertIsNone(_progress_dict({"downloaded_bytes": "x", "total_bytes": 9}))
        self.assertEqual(_progress_dict({"downloaded_bytes": 20, "total_bytes": 9, "speed_bps": -1, "eta_s": "soon"}),
                         {"downloaded_bytes": 9, "total_bytes": 9, "speed_bps": None, "eta_s": None})


class LaunchTests(BridgeTestCase):
    def launch(self):
        with patch.object(subprocess, "Popen") as popen:
            result = self.api.launch_main()
        return result, popen

    def test_refused_without_current_json(self):
        result, popen = self.launch()
        self.assertFalse(result["ok"])
        self.assertTrue(result["message"])
        popen.assert_not_called()

    def test_refused_when_not_verified(self):
        self.write_current(verified=False)
        self.make_env_python()
        result, popen = self.launch()
        self.assertFalse(result["ok"])
        popen.assert_not_called()

    def test_refused_when_the_lock_changed(self):
        self.write_current(sha="c" * 64)
        self.make_env_python()
        result, popen = self.launch()
        self.assertFalse(result["ok"])
        popen.assert_not_called()

    def test_refused_when_the_env_python_is_missing(self):
        self.write_current()
        result, popen = self.launch()
        self.assertFalse(result["ok"])
        popen.assert_not_called()

    def test_launches_the_main_app_detached(self):
        self.write_current()
        self.make_env_python()
        self.assertIsNone(self.api.window)
        result, popen = self.launch()                       # window 為 None 也不能出錯
        json.dumps(result)
        self.assertEqual(result, {"ok": True, "message": "已啟動"})

        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], launch_env.main_command(self.env))
        self.assertEqual(kwargs["env"]["OH_STREAMER_PROJECT_ROOT"], str(layout.app_dir(self.root)))
        self.assertEqual(kwargs["env"]["OH_STREAMER_INSTALL_ROOT"], str(self.root))
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(kwargs[stream], subprocess.DEVNULL, stream)
        if sys.platform == "win32":
            flags = kwargs["creationflags"]
            self.assertTrue(flags & subprocess.DETACHED_PROCESS)
            self.assertTrue(flags & subprocess.CREATE_NEW_PROCESS_GROUP)

    def test_closes_the_setup_window_shortly_after_launch(self):
        self.write_current()
        self.make_env_python()
        self.api.window = MagicMock()
        with patch.object(subprocess, "Popen"), patch.object(bridge.threading, "Timer") as timer:
            result = self.api.launch_main()
        self.assertTrue(result["ok"])
        timer.assert_called_once_with(0.5, self.api.window.destroy)
        timer.return_value.start.assert_called_once_with()

    def test_failed_launch_keeps_the_window_open(self):
        self.write_current()
        self.make_env_python()
        self.api.window = MagicMock()
        with patch.object(subprocess, "Popen", side_effect=OSError("no pythonw")), \
                patch.object(bridge.threading, "Timer") as timer:
            result = self.api.launch_main()
        self.assertFalse(result["ok"])
        self.assertIn("no pythonw", result["message"])
        timer.assert_not_called()


class DiagnosticsTests(BridgeTestCase):
    def setUp(self):
        super().setUp()
        self.data = Path(self._tmp) / "data"
        for patcher in (patch.object(platform_support, "data_dir", return_value=self.data),
                        patch.object(preflight, "webview2_installed", return_value=True),
                        patch.object(diagnostics, "webview2_installed", return_value=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_exports_a_zip_with_the_last_detection(self):
        with patch.object(gpu_detect, "detect", return_value=Detection([RTX_5060], True)):
            self.api.detect()
        result = self.api.export_diagnostics()
        json.dumps(result)
        self.assertTrue(result["ok"], result)
        self.assertIsInstance(result["path"], str)
        path = Path(result["path"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, self.data / "diagnostics")
        with zipfile.ZipFile(path) as zf:
            detection = json.loads(zf.read("detection.json"))
        self.assertEqual(detection["gpus"][0]["name"], RTX_5060.name)
        self.assertEqual(detection["recommendation"]["flavor"], "nvidia")

    def test_exports_even_if_detect_never_ran(self):
        result = self.api.export_diagnostics()
        self.assertTrue(result["ok"], result)
        with zipfile.ZipFile(result["path"]) as zf:
            self.assertIsNone(json.loads(zf.read("detection.json")))

    def test_passes_install_root_and_detection_to_the_exporter(self):
        with patch.object(gpu_detect, "detect", return_value=Detection([], False)):
            detected = self.api.detect()
        with patch.object(diagnostics, "export_diagnostics", return_value=self.data / "x.zip") as export:
            result = self.api.export_diagnostics()
        self.assertTrue(result["ok"])
        export.assert_called_once_with(self.root, self.data / "diagnostics", detection=detected)

    def test_failure_becomes_ok_false(self):
        with patch.object(diagnostics, "export_diagnostics", side_effect=OSError("read-only disk")):
            result = self.api.export_diagnostics()
        self.assertEqual(result["ok"], False)
        self.assertEqual(result["path"], "")
        self.assertIn("read-only disk", result["message"])
        json.dumps(result)


class MainModuleTests(unittest.TestCase):
    """__main__ 只測不開視窗的部分：網址與安裝根目錄環境變數。"""

    def test_setup_url_is_a_file_uri_next_to_the_queue_ui_branding(self):
        url = setup_main.setup_url()
        self.assertTrue(url.startswith("file:///"), url)
        path = unquote(urlparse(url).path)
        index = Path(path[1:] if os.name == "nt" else path)          # "/C:/x/y" -> "C:/x/y"
        self.assertTrue(index.is_file(), index)
        self.assertEqual((index.parent.name, index.name), ("setup_ui", "index.html"))
        self.assertTrue((index.parent.parent / "queue_ui" / "branding.js").is_file())   # index.html 用 ../queue_ui/branding.js

    @unittest.skipUnless(find_spec("webview"), "pywebview 不在這個環境")
    def test_setup_url_does_not_make_pywebview_start_its_http_server(self):
        """純路徑會被 pywebview 當本地檔案，自動開根目錄在 setup_ui/ 的 http 伺服器，../queue_ui/branding.js 就 404。"""
        from webview.util import is_local_url
        self.assertFalse(is_local_url(setup_main.setup_url()))

    def test_install_root_comes_from_the_environment(self):
        with patch.dict(os.environ, {"OH_STREAMER_INSTALL_ROOT": r"C:\oh\root"}):
            self.assertEqual(setup_main._install_root_from_env(), Path(r"C:\oh\root"))

    def test_missing_install_root_raises_with_an_explanation(self):
        with patch.dict(os.environ):
            os.environ.pop("OH_STREAMER_INSTALL_ROOT", None)
            with patch.object(sys, "stderr", StringIO()) as err:
                with self.assertRaises(RuntimeError) as ctx:
                    setup_main._install_root_from_env()
        self.assertIn("OH_STREAMER_INSTALL_ROOT", str(ctx.exception))
        self.assertIn("OH_STREAMER_INSTALL_ROOT", err.getvalue())

    def test_missing_install_root_without_stderr_still_raises(self):
        with patch.dict(os.environ):
            os.environ.pop("OH_STREAMER_INSTALL_ROOT", None)
            with patch.object(sys, "stderr", None):
                with self.assertRaises(RuntimeError):
                    setup_main._install_root_from_env()


if __name__ == "__main__":
    unittest.main()
