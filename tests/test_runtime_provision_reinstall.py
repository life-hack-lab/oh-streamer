"""重新安裝／換版本：provision.run 裝進新資料夾、成功才切換 current.json 並刪舊環境、失敗或取消時刪新資料夾；
launch_env 的設定模式環境選擇；bridge 的「程式還開著」與「先刪除舊環境再安裝」。

installer／selfcheck 是假的：不跑 uv、不下載。假的 uv sync 只在 env_dir 建 python 與一個假的 torch 套件。
"""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import env_cleanup, env_guard, installer, launch_env, layout, provision, selfcheck
from oh_streamer.runtime_env.env_guard import ProcessInfo
from oh_streamer.runtime_env.installer import SyncResult
from oh_streamer.runtime_env.selfcheck import SelfCheckResult
from oh_streamer.setup_mode import bridge
from oh_streamer.setup_mode.bridge import SetupApi

LOG = Path("fake.log")
OK = SyncResult(True, 0, LOG, False, "ok")
WAIT_S = 10


def make_python(env: Path) -> Path:
    exe = layout.python_in_env(env)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    return exe


def make_torch(env: Path) -> None:
    site = env / "Lib" / "site-packages"
    (site / "torch").mkdir(parents=True, exist_ok=True)
    (site / "torch" / "__init__.py").write_text("t")
    dist = site / "torch-2.8.0+cu128.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "RECORD").write_text("torch/__init__.py,,\n")


class ReinstallCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / "install"
        (self.root / "app").mkdir(parents=True)
        (self.root / "app" / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (self.root / "uv.exe").write_bytes(b"")
        self.sha = layout.lock_sha256(self.root / "app" / "uv.lock")
        self.lid = self.sha[:8]
        self.envs = layout.envs_dir(self.root)
        self.base_env = self.envs / self.lid
        self.seen = {}
        self.base = mock.MagicMock(side_effect=self.fake_base)
        self.gpu = mock.MagicMock(side_effect=self.fake_gpu)
        self.check = mock.MagicMock(return_value=SelfCheckResult(True, True, "RTX", True, "ok"))
        for target, name, fake in ((installer, "install_base", self.base), (installer, "install_gpu", self.gpu),
                                   (selfcheck, "run_self_check", self.check)):
            patcher = mock.patch.object(target, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_base(self, uv_cmd, root, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
        self.seen["base_env"] = env_dir
        self.seen["old_torch_at_base"] = (getattr(self, "watch", self.base_env) / "Lib" / "site-packages" / "torch").exists()
        make_python(env_dir)
        return OK

    def fake_gpu(self, uv_cmd, root, flavor, python_exe=None, on_line=None, cancel_event=None, on_progress=None, env_dir=None):
        self.seen["gpu_env"] = env_dir
        make_torch(env_dir)
        return OK

    def installed(self, name=None, flavor="nvidia", verified=True, torch=True) -> Path:
        """An env that a previous install left behind, selected in current.json (name None: legacy, no env field)."""
        env = self.envs / (name or self.lid)
        make_python(env)
        if torch:
            make_torch(env)
        layout.write_current(self.root, layout.Current(
            lock_sha256=self.sha, flavor=flavor, verified=verified, updated_at="t", env=name))
        return env

    def run_provision(self, *, own_env=None, cancel=None, **kw):
        cancel = cancel or threading.Event()
        own = own_env if own_env is not None else Path(tempfile.gettempdir()) / "not-an-env"
        return provision.run(self.root, "nvidia", cancel, own_env=own, **kw)


class ProvisionTests(ReinstallCase):
    def test_first_install_goes_to_a_new_folder_and_retires_the_base_env(self):
        make_python(self.base_env)                                  # launch.pyw built it; setup runs from it
        lines = []
        out = self.run_provision(own_env=self.base_env, on_line=lines.append)
        self.assertEqual(out.phase, provision.DONE)
        target = self.envs / f"{self.lid}-nvidia-1"
        self.assertEqual(self.seen["base_env"], target)
        self.assertEqual(self.seen["gpu_env"], target)
        self.check.assert_called_once_with(layout.python_in_env(target))
        cur = layout.read_current(self.root)
        self.assertEqual((cur.env, cur.flavor, cur.verified), (target.name, "nvidia", True))
        self.assertEqual(cur.retired, (self.lid,))                  # setup still runs from it: launch.pyw deletes it
        self.assertTrue(self.base_env.exists())
        self.assertEqual(layout.status(self.root, self.root / "app" / "uv.lock"), layout.READY)
        self.assertEqual(launch_env.main_env_dir(self.root), target)
        env_cleanup.run_after_setup(self.root)                      # what launch.pyw does after setup exits
        self.assertFalse(self.base_env.exists())
        self.assertEqual(layout.read_current(self.root).retired, ())

    def test_reinstall_same_lock_deletes_the_old_env_in_the_same_session(self):
        old = self.installed()                                      # legacy layout: envs/<lock_id>, no env field
        out = self.run_provision()                                  # setup not running from old (e.g. tests / helper)
        self.assertEqual(out.phase, provision.DONE)
        self.assertFalse(old.exists())
        cur = layout.read_current(self.root)
        self.assertEqual((cur.env, cur.retired), (f"{self.lid}-nvidia-1", ()))

    def test_reinstall_from_the_running_env_removes_its_torch_now_and_the_rest_later(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        phases = []
        out = self.run_provision(own_env=old, on_phase=phases.append)
        self.assertEqual(out.phase, provision.DONE)
        self.assertEqual(phases, ["base", "gpu", "selfcheck", "cleanup"])
        new = self.envs / f"{self.lid}-nvidia-2"                    # never the folder in use
        self.assertEqual(layout.read_current(self.root).env, new.name)
        self.assertTrue(layout.python_in_env(old).is_file())
        self.assertFalse((old / "Lib" / "site-packages" / "torch").exists())
        self.assertEqual(layout.read_current(self.root).retired, (old.name,))

    def test_failure_deletes_the_new_folder_and_keeps_the_old_env(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        before = (self.root / "current.json").read_bytes()
        self.check.return_value = SelfCheckResult(False, False, None, False, "CUDA 不可用")
        out = self.run_provision(own_env=old)
        self.assertEqual(out.phase, provision.FAILED)
        self.assertFalse((self.envs / f"{self.lid}-nvidia-2").exists())
        self.assertEqual((self.root / "current.json").read_bytes(), before)
        self.assertTrue((old / "Lib" / "site-packages" / "torch").exists())
        self.assertEqual(layout.status(self.root, self.root / "app" / "uv.lock"), layout.READY)

    def test_gpu_failure_and_cancel_also_clean_up(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        self.gpu.side_effect = None
        for result in (SyncResult(False, 2, LOG, False, "uv 結束碼 2"), SyncResult(False, None, LOG, True, "已取消")):
            self.gpu.return_value = result
            out = self.run_provision(own_env=old)
            self.assertIn(out.phase, (provision.FAILED, provision.CANCELLED))
            self.assertEqual(env_cleanup.env_folders(self.root), [old.name])
        self.assertEqual(layout.read_current(self.root).env, old.name)

    def test_cancel_during_selfcheck_deletes_the_new_folder(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        cancel = threading.Event()

        def check_then_cancel(python):
            cancel.set()
            return SelfCheckResult(True, True, "RTX", True, "ok")

        self.check.side_effect = check_then_cancel
        out = self.run_provision(own_env=old, cancel=cancel)
        self.assertEqual(out.phase, provision.CANCELLED)
        self.assertEqual(env_cleanup.env_folders(self.root), [old.name])

    def test_unexpected_error_still_removes_the_half_env(self):
        self.installed(f"{self.lid}-nvidia-1")
        self.gpu.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.run_provision()
        self.assertFalse((self.envs / f"{self.lid}-nvidia-2").exists())

    def test_delete_old_first(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        other = self.envs / self.lid
        make_python(other)
        self.watch = old
        self.check.return_value = SelfCheckResult(False, False, None, False, "CUDA 不可用")
        out = self.run_provision(own_env=old, delete_old_first=True)
        self.assertFalse(self.seen["old_torch_at_base"])           # torch was gone before the install started
        self.assertEqual(out.phase, provision.FAILED)
        self.assertFalse((old / "Lib" / "site-packages" / "torch").exists())
        self.assertTrue(layout.python_in_env(old).is_file())
        self.assertFalse(other.exists())
        self.assertFalse(layout.read_current(self.root).verified)   # failed after deleting: back to setup next time
        self.assertEqual(layout.status(self.root, self.root / "app" / "uv.lock"), layout.NEED_SETUP)
        self.assertEqual(launch_env.setup_env_dir(self.root), old)  # its base still runs setup mode


class LaunchEnvTests(ReinstallCase):
    def test_setup_env_dir(self):
        self.assertEqual(launch_env.setup_env_dir(self.root), self.base_env)          # nothing installed
        old = self.installed(f"{self.lid}-nvidia-1")
        self.assertEqual(launch_env.setup_env_dir(self.root), old)                     # reinstall: no base env build
        layout.write_current(self.root, layout.Current(lock_sha256="f" * 64, flavor="nvidia", verified=True,
                                                       updated_at="t", env=old.name))
        self.assertEqual(launch_env.setup_env_dir(self.root), self.base_env)          # new uv.lock: fresh base env
        self.assertEqual(launch_env.main_env_dir(self.root), old)

    def test_main_env_dir_without_current(self):
        self.assertIsNone(launch_env.main_env_dir(self.root))


class BridgeReinstallTests(ReinstallCase):
    def setUp(self):
        super().setUp()
        self.api = SetupApi(self.root)
        self.addCleanup(self._stop)
        self.procs = mock.patch.object(env_guard, "processes_under", return_value=[])
        self.procs_mock = self.procs.start()
        self.addCleanup(self.procs.stop)

    def _stop(self):
        thread = self.api._thread
        if thread is not None and thread.is_alive():
            self.api._cancel_event.set()
            thread.join(WAIT_S)

    def wait(self):
        self.api._thread.join(WAIT_S)
        self.assertFalse(self.api._thread.is_alive())
        return self.api.poll()

    def test_running_app_blocks_the_install(self):
        self.installed(f"{self.lid}-nvidia-1")
        self.procs_mock.return_value = [ProcessInfo(4242, 1, str(self.envs / "x" / "Scripts" / "pythonw.exe"))]
        res = self.api.start_install("nvidia")
        json.dumps(res)
        self.assertEqual((res["ok"], res["reason"]), (False, "in_use"))
        self.assertEqual(res["processes"], [{"pid": 4242, "name": "pythonw.exe"}])
        self.assertIn(bridge.APP_NAME, res["message"])
        self.assertIsNone(self.api._thread)
        self.base.assert_not_called()
        self.procs_mock.return_value = []                           # 使用者關掉了，按「重試」
        self.assertTrue(self.api.start_install("nvidia")["ok"])
        self.assertEqual(self.wait()["phase"], "done")

    def test_detect_reports_in_use_and_plans(self):
        self.installed(f"{self.lid}-nvidia-1")
        with mock.patch.object(bridge.gpu_detect, "detect", return_value=bridge.gpu_detect.Detection([], False)), \
                mock.patch.object(bridge.preflight, "webview2_installed", return_value=True):
            info = self.api.detect()
        json.dumps(info)
        self.assertEqual(info["in_use"], [])
        plan = info["disk_plans"]["nvidia"]
        self.assertTrue(plan["reuses_installed"])
        self.assertEqual(plan["required_gb"], bridge.REUSE_DISK_GB)
        self.assertEqual(info["required_disk_gb"], bridge.REUSE_DISK_GB)

    def low_disk(self, free_gb, freeable_gib):
        old = self.installed("ffffffff-nvidia-1")                   # an env for another uv.lock: new torch needed
        layout.write_current(self.root, layout.Current(lock_sha256="f" * 64, flavor="nvidia", verified=True,
                                                       updated_at="t", env=old.name))
        patches = [mock.patch.object(bridge, "_disk_free_gb", return_value=free_gb),
                   mock.patch.object(env_cleanup, "freeable_torch_bytes", return_value=int(freeable_gib * 2**30))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return old

    def test_low_space_offers_delete_old_first(self):
        old = self.low_disk(5.0, 4.5)
        res = self.api.start_install("nvidia")
        self.assertEqual((res["ok"], res["reason"]), (False, "low_space"))
        self.assertTrue(res["plan"]["can_delete_old_first"])
        self.assertEqual(res["plan"]["required_gb"], bridge.REQUIRED_DISK_GB)
        self.assertIsNone(self.api._thread)
        with mock.patch.object(env_cleanup, "free_old_envs_first", wraps=env_cleanup.free_old_envs_first) as free:
            self.assertTrue(self.api.start_install("nvidia", {"delete_old_first": True})["ok"])
            self.assertEqual(self.wait()["phase"], "done")
        free.assert_called_once()
        self.assertFalse(old.exists())

    def test_delete_old_first_refused_when_it_would_not_be_enough(self):
        self.low_disk(2.0, 1.0)
        res = self.api.start_install("nvidia", {"delete_old_first": True})
        self.assertEqual((res["ok"], res["reason"]), (False, "low_space"))
        self.assertFalse(res["plan"]["can_delete_old_first"])

    def test_delete_old_first_is_ignored_when_space_is_enough(self):
        old = self.low_disk(50.0, 4.5)
        with mock.patch.object(env_cleanup, "free_old_envs_first") as free:
            self.assertTrue(self.api.start_install("nvidia", {"delete_old_first": True})["ok"])
            self.assertEqual(self.wait()["phase"], "done")
        free.assert_not_called()

    def test_same_version_reinstall_never_offers_delete_first(self):
        self.installed(f"{self.lid}-nvidia-1")
        plan = bridge.disk_plan(self.root, "nvidia", 0.5, current=layout.read_current(self.root), lock_sha=self.sha)
        self.assertTrue(plan["low_space"])
        self.assertFalse(plan["can_delete_old_first"])

    def test_launch_main_uses_the_current_env(self):
        old = self.installed(f"{self.lid}-nvidia-1")
        with mock.patch.object(launch_env, "start_main") as start:
            self.assertTrue(self.api.launch_main()["ok"])
        self.assertEqual(start.call_args.args[1], old)


if __name__ == "__main__":
    unittest.main()
