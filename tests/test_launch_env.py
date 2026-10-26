"""launch_env: environment and commands for the app, and the pure decisions launch.pyw relies on (never runs launch.pyw)."""
import ast
import hashlib
import os
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from oh_streamer.runtime_env import launch_env, layout

LOCK_BYTES = b'version = 1\r\nrequires-python = "==3.13.*"\r\n'
LOCK_SHA = hashlib.sha256(LOCK_BYTES).hexdigest()


class InstallRootTestCase(unittest.TestCase):
    """A fake install root: <root>/app/uv.lock, nothing else."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.root = self.tmp / "install"
        self.app = layout.app_dir(self.root)
        self.app.mkdir(parents=True)
        self.lock = self.app / "uv.lock"
        self.lock.write_bytes(LOCK_BYTES)
        self.env_dir = layout.env_dir(self.root, LOCK_SHA)

    def make_env(self, sha: str = LOCK_SHA) -> Path:
        python = layout.python_in_env(layout.env_dir(self.root, sha))
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")
        return python

    def write_current(self, **overrides):
        fields = dict(lock_sha256=LOCK_SHA, flavor="nvidia", verified=True, updated_at="2026-10-24T02:00:00+00:00")
        fields.update(overrides)
        layout.write_current(self.root, layout.Current(**fields))

    def make_ready(self):
        self.make_env()
        self.write_current()


class MainEnvTests(InstallRootTestCase):
    def test_project_and_install_root(self):
        env = launch_env.main_env(self.root, self.env_dir, base={})
        self.assertEqual(env["OH_STREAMER_PROJECT_ROOT"], str(self.app))
        self.assertEqual(env["OH_STREAMER_INSTALL_ROOT"], str(self.root))

    def test_pythonpath_has_app_src(self):
        env = launch_env.main_env(self.root, self.env_dir, base={"PYTHONPATH": "ignored"})
        self.assertIn(str(self.app / "src"), env["PYTHONPATH"].split(os.pathsep))

    def test_path_starts_with_tools_mpv_and_env_scripts(self):
        env = launch_env.main_env(self.root, self.env_dir, base={"PATH": "C:\\existing"})
        parts = env["PATH"].split(os.pathsep)
        self.assertEqual(parts[:3], [str(self.app / "tools"), str(self.app / "tools" / "mpv"), str(self.env_dir / "Scripts")])
        self.assertEqual(parts[-1], "C:\\existing")                      # the old PATH is kept, after ours

    def test_path_without_base_path(self):
        env = launch_env.main_env(self.root, self.env_dir, base={})
        self.assertEqual(env["PATH"].split(os.pathsep)[:3],
                         [str(self.app / "tools"), str(self.app / "tools" / "mpv"), str(self.env_dir / "Scripts")])

    def test_model_weights_dir(self):
        env = launch_env.main_env(self.root, self.env_dir, base={})
        self.assertEqual(env["LADA_MODEL_WEIGHTS_DIR"], str(self.app / "model_weights"))

    def test_base_is_copied_not_modified(self):
        base = {"PATH": "x", "KEEP": "1"}
        snapshot = dict(base)
        env = launch_env.main_env(self.root, self.env_dir, base=base)
        self.assertEqual(base, snapshot)
        self.assertIsNot(env, base)
        self.assertEqual(env["KEEP"], "1")

    def test_default_base_is_os_environ_and_leaves_it_alone(self):
        with mock.patch.dict(os.environ, {"OH_STREAMER_TEST_MARK": "yes"}):
            before = dict(os.environ)
            env = launch_env.main_env(self.root, self.env_dir)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(env["OH_STREAMER_TEST_MARK"], "yes")
        self.assertEqual(env["OH_STREAMER_PROJECT_ROOT"], str(self.app))


class CommandTests(InstallRootTestCase):
    def test_main_command(self):
        command = launch_env.main_command(self.env_dir)
        self.assertEqual(command[0], str(launch_env.pythonw_in_env(self.env_dir)))
        self.assertEqual(command[-3:], ["-m", "oh_streamer", "queue"])
        if sys.platform == "win32":
            self.assertEqual(command[0], str(self.env_dir / "Scripts" / "pythonw.exe"))

    def test_setup_command_runs_setup_mode_in_the_env(self):
        command = launch_env.setup_command(self.env_dir)
        self.assertEqual(command[0], str(launch_env.pythonw_in_env(self.env_dir)))
        self.assertEqual(command[1:], ["-m", "oh_streamer.setup_mode"])

    def test_wants_setup(self):
        self.assertTrue(launch_env.wants_setup(["--setup"]))
        self.assertTrue(launch_env.wants_setup(["--hidden", "--setup"]))
        self.assertFalse(launch_env.wants_setup([]))
        self.assertFalse(launch_env.wants_setup(["--hidden", "--setup-later"]))

    def test_app_args_drops_only_the_launcher_flag(self):
        self.assertEqual(launch_env.app_args(["--setup", "--hidden", "--instance", "a"]), ["--hidden", "--instance", "a"])
        self.assertEqual(launch_env.app_args([]), [])

    def test_detached_creationflags(self):
        with mock.patch.object(launch_env.sys, "platform", "win32"):
            self.assertEqual(launch_env.detached_creationflags(), 0x00000008 | 0x00000200)
        with mock.patch.object(launch_env.sys, "platform", "linux"):
            self.assertEqual(launch_env.detached_creationflags(), 0)


class LocateTests(InstallRootTestCase):
    def test_paths_follow_the_shipped_lock(self):
        install = layout.locate(self.root)
        self.assertEqual((install.root, install.lock_sha256), (self.root, LOCK_SHA))
        self.assertEqual((install.app, install.lock, install.env), (self.app, self.lock, self.env_dir))
        self.assertEqual(install.python, layout.python_in_env(self.env_dir))
        self.lock.write_bytes(LOCK_BYTES + b"# changed\n")
        self.assertNotEqual(layout.locate(self.root).env, self.env_dir)

    def test_missing_lock_raises_oserror(self):
        self.lock.unlink()
        with self.assertRaises(OSError):
            layout.locate(self.root)


class StartMainTests(InstallRootTestCase):
    def test_starts_the_queue_window_detached_with_the_app_env(self):
        with mock.patch.object(launch_env.subprocess, "Popen") as popen:
            launch_env.start_main(self.root, self.env_dir, ["--hidden"])
        args, kwargs = popen.call_args
        self.assertEqual(args[0], launch_env.main_command(self.env_dir) + ["--hidden"])
        self.assertEqual(kwargs["env"]["OH_STREAMER_PROJECT_ROOT"], str(self.app))
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(kwargs[stream], launch_env.subprocess.DEVNULL, stream)
        if sys.platform == "win32":
            self.assertEqual(kwargs["creationflags"], launch_env.detached_creationflags())
        else:
            self.assertTrue(kwargs["start_new_session"])


class PlanLaunchTests(InstallRootTestCase):
    def test_ready_goes_to_main(self):
        self.make_ready()
        self.assertEqual(layout.status(self.root, self.lock), layout.READY)
        self.assertEqual(launch_env.plan_launch(self.root, force_setup=False), launch_env.MODE_MAIN)
        self.assertEqual(launch_env.plan_launch(self.root), "main")

    def test_force_setup_wins_even_when_ready(self):
        self.make_ready()
        self.assertEqual(launch_env.plan_launch(self.root, force_setup=True), launch_env.MODE_SETUP)
        self.assertEqual(launch_env.plan_launch(self.root, force_setup=True), "setup")

    def test_no_current_goes_to_setup(self):
        self.make_env()
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_changed_lock_goes_to_setup(self):
        self.make_ready()
        self.lock.write_bytes(LOCK_BYTES + b"# new dependency set\n")
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_base_only_or_unverified_current_goes_to_setup(self):
        self.make_env()
        self.write_current(flavor=None)
        self.assertEqual(launch_env.plan_launch(self.root), "setup")
        self.write_current(verified=False)
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_missing_env_python_goes_to_setup(self):
        self.write_current()
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_missing_lock_goes_to_setup(self):
        self.make_ready()
        self.lock.unlink()
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_corrupt_current_goes_to_setup(self):
        self.make_env()
        (self.root / layout.CURRENT_FILE).write_text("{not json", encoding="utf-8")
        self.assertEqual(launch_env.plan_launch(self.root), "setup")

    def test_does_not_write_current_json(self):
        launch_env.plan_launch(self.root)
        self.assertFalse((self.root / layout.CURRENT_FILE).exists())
        self.make_ready()
        before = (self.root / layout.CURRENT_FILE).read_bytes()
        launch_env.plan_launch(self.root)
        launch_env.plan_launch(self.root, force_setup=True)
        self.assertEqual((self.root / layout.CURRENT_FILE).read_bytes(), before)


class UvCommandTests(InstallRootTestCase):
    def setUp(self):
        super().setUp()
        which = mock.patch.object(launch_env.shutil, "which", return_value=None)   # this machine's own uv must not leak in
        self.which = which.start()
        self.addCleanup(which.stop)

    def test_falls_back_to_uv_on_path(self):
        self.which.return_value = r"C:\path\uv.exe"
        self.assertEqual(launch_env.uv_command(self.root, {}), Path(r"C:\path\uv.exe"))
        self.which.assert_called_once_with("uv")

    def test_bundled_uv_beats_path(self):
        (self.root / "uv.exe").write_bytes(b"")
        self.which.return_value = r"C:\path\uv.exe"
        self.assertEqual(launch_env.uv_command(self.root, {}), self.root / "uv.exe")

    def test_env_var_wins_over_bundled_uv(self):
        (self.root / "uv.exe").write_bytes(b"")
        custom = self.tmp / "tools" / "uv.exe"
        found = launch_env.uv_command(self.root, {launch_env.ENV_UV: str(custom)})
        self.assertEqual(found, custom)

    def test_bundled_uv_when_no_env_var(self):
        (self.root / "uv.exe").write_bytes(b"")
        self.assertEqual(launch_env.uv_command(self.root, {}), self.root / "uv.exe")

    def test_blank_env_var_is_ignored(self):
        (self.root / "uv.exe").write_bytes(b"")
        self.assertEqual(launch_env.uv_command(self.root, {launch_env.ENV_UV: "   "}), self.root / "uv.exe")

    def test_none_when_nothing_found(self):
        self.assertIsNone(launch_env.uv_command(self.root, {}))
        (self.root / "uv.exe").mkdir()                                  # a folder named uv.exe is not a uv
        self.assertIsNone(launch_env.uv_command(self.root, {}))

    def test_default_environ_is_os_environ(self):
        custom = self.tmp / "uv-from-env.exe"
        with mock.patch.dict(os.environ, {launch_env.ENV_UV: str(custom)}):
            self.assertEqual(launch_env.uv_command(self.root), custom)
        with mock.patch.dict(os.environ):
            os.environ.pop(launch_env.ENV_UV, None)
            self.assertIsNone(launch_env.uv_command(self.root))


class BasePythonTests(InstallRootTestCase):
    def setUp(self):
        super().setUp()
        self.embedded = self.tmp / "python"
        self.embedded.mkdir()

    def test_install_python_prefers_the_embedded_python(self):
        bundled = self.root / "python" / "python.exe"
        (self.embedded / "python.exe").write_bytes(b"")
        self.assertIsNone(launch_env.install_python(self.root))                            # no embedded, no hint: uv picks
        self.assertEqual(launch_env.install_python(self.root, self.embedded / "python.exe"), self.embedded / "python.exe")
        bundled.parent.mkdir()
        bundled.write_bytes(b"")
        self.assertEqual(launch_env.install_python(self.root), bundled)
        self.assertEqual(launch_env.install_python(self.root, self.embedded / "python.exe"), bundled)

    def test_pythonw_maps_to_python_exe_beside_it(self):
        (self.embedded / "python.exe").write_bytes(b"")
        (self.embedded / "pythonw.exe").write_bytes(b"")
        self.assertEqual(launch_env.base_python(self.embedded / "pythonw.exe"), self.embedded / "python.exe")
        self.assertEqual(launch_env.base_python(self.embedded / "PYTHONW.EXE"), self.embedded / "python.exe")

    def test_python_exe_is_used_as_is(self):
        (self.embedded / "python.exe").write_bytes(b"")
        self.assertEqual(launch_env.base_python(self.embedded / "python.exe"), self.embedded / "python.exe")

    def test_none_when_the_console_python_is_missing(self):
        (self.embedded / "pythonw.exe").write_bytes(b"")
        self.assertIsNone(launch_env.base_python(self.embedded / "pythonw.exe"))

    def test_none_for_empty_executable(self):
        self.assertIsNone(launch_env.base_python(""))
        with mock.patch.object(sys, "executable", ""):
            self.assertIsNone(launch_env.base_python())

    def test_defaults_to_sys_executable(self):
        (self.embedded / "python.exe").write_bytes(b"")
        (self.embedded / "pythonw.exe").write_bytes(b"")
        with mock.patch.object(sys, "executable", str(self.embedded / "pythonw.exe")):
            self.assertEqual(launch_env.base_python(), self.embedded / "python.exe")


class LaunchScriptTests(unittest.TestCase):
    script = REPO / "launch.pyw"

    def test_compiles(self):
        with tempfile.TemporaryDirectory() as tmp:                      # cfile: keep __pycache__ out of the repo
            py_compile.compile(str(self.script), cfile=str(Path(tmp) / "launch.pyc"), doraise=True)

    def test_imports_are_stdlib_or_lazy_oh_streamer(self):
        tree = ast.parse(self.script.read_text(encoding="utf-8"))
        stdlib = set(sys.stdlib_module_names)
        top_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root == "oh_streamer":
                    self.assertNotIn(id(node), top_level, "oh_streamer must be imported after app/src is on sys.path")
                else:
                    self.assertIn(root, stdlib, f"launch.pyw may only use the standard library, not {name!r}")

    def test_product_name_is_not_hard_coded(self):
        from oh_streamer.branding import APP_NAME
        self.assertNotIn(APP_NAME, self.script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
