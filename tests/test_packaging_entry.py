"""Bounded packaging contract checks; no PyInstaller, GPU, browser or build required."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import patch


ROOT = _STREAMER_ROOT


def load_helper(name):
    spec = importlib.util.spec_from_file_location("streamer_packaging_" + name, ROOT / "packaging" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load_helper("frozen_entry")
build = load_helper("build_macos")
package_entry = importlib.import_module("oh_streamer.__main__")


class DispatchTests(unittest.TestCase):
    def test_source_resolver_uses_browser_route_only(self):
        expected = ("oh_streamer.sources.resolvers.resolve_browser", ["https://pages.example.test/watch"])
        self.assertEqual(package_entry.route(["--resolver", "browser", *expected[1]]), expected)
        self.assertEqual(package_entry.route(["resolver", "browser", *expected[1]]), expected)
        for name in ("retired", "unsupported", "other"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package_entry.route(["--resolver", name, "https://pages.example.test/watch"])

    def test_public_modes_and_existing_child_argv(self):
        cases = [(["--player", "clip.mkv"], "oh_streamer.player.oh_player", ["--player", "clip.mkv"]),
                 (["--queue", "--root", "/tmp/queue"], "oh_streamer.queue.queue_app", ["--root", "/tmp/queue"]),
                 (["--worker", "--job", "job.json"], "oh_streamer.queue.queue_worker", ["--job", "job.json"]),
                 (["--perf", "child", "{}"], "oh_streamer.perf.__main__", ["child", "{}"]),
                 ([r"C:\bundle\queue_worker.py", "--job", "j"], "oh_streamer.queue.queue_worker", ["--job", "j"]),
                 (["/bundle/oh_player.py", "--player"], "oh_streamer.player.oh_player", ["--player"]),
                 (["/bundle/queue_app.py"], "oh_streamer.queue.queue_app", []),
                 (["--resolver", "browser", "https://example.invalid/"], "oh_streamer.sources.resolvers.resolve_browser", ["https://example.invalid/"]),
                 (["/bundle/resolvers/resolve_browser.py", "url"], "oh_streamer.sources.resolvers.resolve_browser", ["url"])]
        for argv, module, args in cases:
            with self.subTest(argv=argv):
                original = list(argv)
                self.assertEqual(entry.route(argv), (module, args))
                self.assertEqual(argv, original)

    def test_no_arguments_default_to_queue(self):
        self.assertEqual(entry.route([]), (entry.QUEUE_MODULE, []))

    def test_frozen_bench_child_command_uses_perf_route(self):
        from oh_streamer.perf import bench
        with patch.object(sys, "frozen", True, create=True):
            cmd = bench.child_command("rife", {"heights": [720]})
        self.assertEqual(cmd[:3], [sys.executable, "--perf", "child"])
        module, args = entry.route(cmd[1:])
        self.assertEqual(module, "oh_streamer.perf.__main__")
        self.assertEqual(args, cmd[2:])
        self.assertIn("-m", bench.child_command("rife", {}))

    def test_unknown_resolver_or_script_fails_without_running(self):
        for argv in (["--resolver"], ["--resolver", "unknown"], ["--resolver", "retired"],
                     ["--resolver", "unsupported"], ["/custom/script.py"],
                     ["/custom/resolve_browser.py"], ["/bundle/resolvers/resolve_retired.py"]):
            with self.subTest(argv=argv), patch.object(entry.runpy, "run_module") as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(entry.main(argv), 2)
                run.assert_not_called()

    def test_lazy_main_restores_argv_even_when_target_exits(self):
        original = sys.argv
        observed = []

        def fake_run(module, **kwargs):
            observed.append((module, list(sys.argv), kwargs))
            raise SystemExit(7)

        with patch.object(entry.runpy, "run_module", side_effect=fake_run), patch.object(entry, "configure_frozen_environment"):
            with self.assertRaises(SystemExit) as error:
                entry.main(["--worker", "--job", "with spaces.json"])
        self.assertEqual(error.exception.code, 7)
        self.assertIs(sys.argv, original)
        self.assertEqual(observed[0][0], "oh_streamer.queue.queue_worker")
        self.assertEqual(observed[0][1][1:], ["--job", "with spaces.json"])
        self.assertEqual(observed[0][2], {"run_name": "__main__", "alter_sys": True})

    def test_frozen_resources_and_models_are_separate(self):
        platform = types.SimpleNamespace(data_dir=lambda: Path("/user/application-support"))
        platform_package = __import__("oh_streamer.platform", fromlist=["platform_support"])
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", "/bundle/resources", create=True), \
                patch.object(sys, "platform", "darwin"), \
                patch.object(platform_package, "platform_support", platform), patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            entry.configure_frozen_environment()
            self.assertEqual(entry.resource_root(), Path("/bundle/resources/resources").resolve())
            self.assertEqual(os.environ["LADA_MODEL_WEIGHTS_DIR"], str(Path("/user/application-support/model_weights")))
            self.assertIn("/opt/homebrew/bin", os.environ["PATH"])
            os.environ["LADA_MODEL_WEIGHTS_DIR"] = "/external/models"
            entry.configure_frozen_environment()
            self.assertEqual(os.environ["LADA_MODEL_WEIGHTS_DIR"], "/external/models")


class BuildTests(unittest.TestCase):
    def test_non_mac_guard_precedes_pyinstaller_or_source_checks(self):
        with patch.object(sys, "platform", "win32"), patch.object(build.subprocess, "run") as run, \
                patch.object(build, "resource_inventory") as inventory, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(build.main(["--lada-source", "/missing"]), 2)
        self.assertIn("require macOS", err.getvalue())
        run.assert_not_called()
        inventory.assert_not_called()

    def test_resource_inventory_is_adjacent_to_frozen_modules(self):
        entries = build.resource_inventory()
        mapping = {Path(path).relative_to(ROOT).as_posix(): dest for path, dest in entries}
        for name in build.RESOURCE_FILES:
            self.assertIn(name, mapping)
        self.assertEqual(mapping["src/oh_streamer/resources/queue_ui/index.html"], "resources/queue_ui")
        self.assertEqual(mapping["src/oh_streamer/resources/resolvers.default.json"], "resources")
        self.assertEqual(mapping["resolvers/resolve_browser.py"], "resolvers")
        self.assertTrue(any(name.startswith("docs/docs_") for name in mapping))
        self.assertFalse(any(name.endswith((".pth", ".pt", ".exe", ".dylib")) for name in mapping))
        self.assertNotIn("resolvers.json", mapping)
        self.assertNotIn("oh_settings.json", mapping)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Missing packaging resource"):
                build.resource_inventory(Path(tmp))

    def test_mac_build_invokes_selected_python_without_installing(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp) / "upstream"
            (source / "lada").mkdir(parents=True)
            (source / "lada" / "__init__.py").write_text("")
            mpv = Path(tmp) / "mpv"
            mpv.write_bytes(b"prepared Mach-O")
            notices = Path(tmp) / "notices"
            notices.mkdir()
            (notices / "COPYING").write_text("mpv and dependency notices")
            with patch.object(sys, "platform", "darwin"), patch.object(build.subprocess, "run", return_value=types.SimpleNamespace(returncode=9)) as run:
                result = build.main(["--lada-source", str(source), "--dist", str(Path(tmp) / "dist"),
                                     "--work", str(Path(tmp) / "work"), "--mpv", str(mpv), "--mpv-notices", str(notices)])
            self.assertEqual(result, 9)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, "-m", "PyInstaller"])
            self.assertEqual(run.call_args.kwargs["env"]["OH_STREAMER_PACKAGING_LADA_SOURCE"], str(source.resolve()))
            self.assertEqual(run.call_args.kwargs["env"]["OH_STREAMER_PACKAGING_MPV"], str(mpv.resolve()))
            self.assertEqual(run.call_args.kwargs["env"]["OH_STREAMER_PACKAGING_MPV_NOTICES"], str(notices.resolve()))
            self.assertIn("--clean", command)
            self.assertEqual(command[-1], str(ROOT / "streamer_macos.spec"))

    def test_spec_is_streamer_specific_and_has_platform_guard(self):
        text = (ROOT / "streamer_macos.spec").read_text(encoding="utf-8")
        compile(text, "streamer_macos.spec", "exec")
        self.assertIn('sys.platform != "darwin"', text)
        self.assertIn('"frozen_entry.py"', text)
        self.assertIn('"oh_streamer.queue.queue_app"', text)
        self.assertIn('"oh_streamer.queue.queue_worker"', text)
        import ast
        tree = ast.parse(text, filename="streamer_macos.spec")
        hidden_node = next(node for node in tree.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "hidden" for target in node.targets))
        self.assertEqual(ast.literal_eval(hidden_node.value), [
            "oh_streamer.player.oh_player",
            "oh_streamer.queue.queue_app",
            "oh_streamer.queue.queue_worker",
            "oh_streamer.perf.__main__",
            "oh_streamer.sources.resolvers.resolve_browser",
            "oh_streamer.sources.http_transport",
            "oh_streamer.platform.platform_support",
            "webview.platforms.cocoa",
        ])
        self.assertIn('excludes=["gi", "lada.gui"]', text)
        self.assertIn("console=True", text)


class ConsoleEntryPointTests(unittest.TestCase):
    def test_installed_scripts_target_lazy_package_dispatchers(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["scripts"], {
            "oh-streamer": "oh_streamer.__main__:main",
            "oh-streamer-queue": "oh_streamer.__main__:queue_main",
            "oh-streamer-worker": "oh_streamer.__main__:worker_main",
            "oh-streamer-resolve-browser": "oh_streamer.__main__:resolver_main",
        })

    def test_wrappers_dispatch_and_restore_argv_when_execution_fails(self):
        modules = {
            "main": "oh_streamer.player.oh_player",
            "queue_main": "oh_streamer.queue.queue_app",
            "worker_main": "oh_streamer.queue.queue_worker",
            "resolver_main": "oh_streamer.sources.resolvers.resolve_browser",
        }
        cases = (
            ("main", "oh-streamer", ["clip.mkv", "--quality", "720"]),
            ("queue_main", "oh-streamer-queue", ["--root", "queue data"]),
            ("worker_main", "oh-streamer-worker", ["--job", "job data.json"]),
            ("resolver_main", "oh-streamer-resolve-browser", ["https://pages.example.test/watch"]),
        )
        for entrypoint, command, args in cases:
            module = modules[entrypoint]
            argv_before = sys.argv
            supplied_argv = [command, *args]
            observed = []

            def fail_before_application_import(name, **kwargs):
                observed.append((name, list(sys.argv), kwargs))
                raise RuntimeError("stop before launching the application")

            try:
                sys.argv = supplied_argv
                with self.subTest(entrypoint=entrypoint), \
                        patch.object(package_entry.runpy, "run_module", side_effect=fail_before_application_import):
                    with self.assertRaisesRegex(RuntimeError, "stop before launching"):
                        getattr(package_entry, entrypoint)()
                    self.assertIs(sys.argv, supplied_argv)
                    self.assertEqual(observed, [(module, [module, *args],
                                                 {"run_name": "__main__", "alter_sys": True})])
            finally:
                sys.argv = argv_before


if __name__ == "__main__":
    unittest.main()
