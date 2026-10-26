"""env_cleanup（新環境資料夾名稱、刪舊環境、刪 torch 套件、uv 快取清理）與 env_guard（設定模式單一實例、使用中程序）。

快取與環境都是 tmp 底下的假樹：環境裡的檔案用 os.link 硬連結到假的 uv-cache/archive-v0/<id>/，跟 uv 真的安裝方式一樣。
"""
import ast
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from oh_streamer.runtime_env import env_cleanup, env_guard, layout
from oh_streamer.runtime_env.env_guard import ProcessInfo

SHA = "cdcedc9f" + "0" * 56


class Tree(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve() / "install"
        self.envs = layout.envs_dir(self.root)
        self.cache = layout.cache_dir(self.root)
        self.envs.mkdir(parents=True)
        self.archives = self.cache / "archive-v0"
        self.archives.mkdir(parents=True)

    def archive(self, archive_id: str, files: dict) -> Path:
        base = self.archives / archive_id
        for rel, data in files.items():
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return base

    def env(self, name: str, python=True) -> Path:
        env = self.envs / name
        site = env / "Lib" / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
        if python:
            exe = layout.python_in_env(env)
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_bytes(b"")
        return env

    def link(self, archive: Path, env: Path) -> None:
        """Hardlink every archive file into env's site-packages, like uv does."""
        site = env / "Lib" / "site-packages"
        for path in archive.rglob("*"):
            if path.is_file():
                target = site / path.relative_to(archive)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)

    def current(self, **fields):
        data = dict(lock_sha256=SHA, flavor="nvidia", verified=True, updated_at="2026-10-24T00:00:00+00:00")
        data.update(fields)
        layout.write_current(self.root, layout.Current(**data))


def torch_archive(tree: Tree, archive_id="TORCH", version="2.8.0+cu128") -> Path:
    dist = f"torch-{version}.dist-info"
    record = "\n".join([
        "torch/__init__.py,sha256=x,1", "torch/lib/c10.dll,sha256=x,4", "functorch/__init__.py,,",
        "torchgen/x.py,,", f"{dist}/RECORD,,", "../../Scripts/torchrun.exe,,",
    ])
    return tree.archive(archive_id, {
        "torch/__init__.py": b"t", "torch/lib/c10.dll": b"dlld", "functorch/__init__.py": b"f",
        "torchgen/x.py": b"g", f"{dist}/RECORD": record.encode(), f"{dist}/METADATA": b"Name: torch\n",
    })


class NamesTests(Tree):
    def test_first_name_and_skipping(self):
        self.assertEqual(env_cleanup.new_env_name(self.root, SHA, "nvidia"), "cdcedc9f-nvidia-1")
        (self.envs / "cdcedc9f-nvidia-1").mkdir()
        self.assertEqual(env_cleanup.new_env_name(self.root, SHA, "nvidia"), "cdcedc9f-nvidia-2")
        self.current(env="cdcedc9f-nvidia-1", retired=("cdcedc9f-nvidia-2",))
        self.assertEqual(env_cleanup.new_env_name(self.root, SHA, "nvidia"), "cdcedc9f-nvidia-3")
        self.assertEqual(env_cleanup.new_env_name(self.root, SHA, "intel"), "cdcedc9f-intel-1")

    def test_env_folders_lists_only_valid_real_folders(self):
        self.env("cdcedc9f")
        self.env("cdcedc9f-nvidia-1")
        (self.envs / "stray.txt").write_text("x")
        (self.envs / ".hidden").mkdir()
        self.assertEqual(env_cleanup.env_folders(self.root), ["cdcedc9f", "cdcedc9f-nvidia-1"])

    def test_is_deletable_env(self):
        self.current(env="cdcedc9f-nvidia-1")
        self.assertTrue(env_cleanup.is_deletable_env(self.root, self.envs / "cdcedc9f"))
        self.assertFalse(env_cleanup.is_deletable_env(self.root, self.envs / "cdcedc9f-nvidia-1"))   # current
        self.assertFalse(env_cleanup.is_deletable_env(self.root, self.root / "app"))
        self.assertFalse(env_cleanup.is_deletable_env(self.root, self.envs / "a" / "b"))
        self.assertFalse(env_cleanup.is_deletable_env(self.root, self.envs))


class LayoutFieldTests(Tree):
    def test_env_and_retired_round_trip_and_status_follows_env(self):
        self.root.joinpath("app").mkdir()
        lock = self.root / "app" / "uv.lock"
        lock.write_bytes(b"lock")
        sha = layout.lock_sha256(lock)
        self.current(lock_sha256=sha, env="x-nvidia-1", retired=("old",))
        cur = layout.read_current(self.root)
        self.assertEqual((cur.env, cur.retired), ("x-nvidia-1", ("old",)))
        self.assertEqual(layout.status(self.root, lock), layout.NEED_SETUP)             # env folder has no python
        self.env("x-nvidia-1")
        self.assertEqual(layout.status(self.root, lock), layout.READY)
        self.assertEqual(layout.current_env_dir(self.root, cur), self.envs / "x-nvidia-1")

    def test_missing_env_means_the_legacy_folder(self):
        self.current()
        cur = layout.read_current(self.root)
        self.assertIsNone(cur.env)
        self.assertEqual(cur.retired, ())
        self.assertEqual(layout.current_env_dir(self.root, cur), self.envs / "cdcedc9f")

    def test_unsafe_names_make_current_invalid(self):
        import json
        for bad in ("..", "../x", "a\\b", "a/b", "", "C:x", ".x", 5):
            for field, value in (("env", bad), ("retired", [bad])):
                data = dict(schema=1, lock_sha256=SHA, flavor="nvidia", verified=True, updated_at="t")
                data[field] = value
                (self.root / "current.json").write_text(json.dumps(data), encoding="utf-8")
                self.assertIsNone(layout.read_current(self.root), f"{field}={bad!r}")


class GcCacheTests(Tree):
    def test_only_archives_no_env_links_go(self):
        a = self.archive("A", {"pkg_a/x.py": b"aaaa"})
        b = self.archive("B", {"pkg_b/x.py": b"bb"})
        self.archive("C", {"pkg_c/x.py": b"c" * 10, "pkg_c/y.py": b"c"})
        env1, env2 = self.env("e1"), self.env("e2")
        self.link(a, env1)
        self.link(b, env2)

        dry = env_cleanup.gc_cache(self.cache, dry_run=True)
        self.assertEqual((dry.archives, dry.bytes_freed, dry.kept), (["C"], 11, 2))
        self.assertTrue((self.archives / "C").exists())

        result = env_cleanup.gc_cache(self.cache)
        self.assertEqual(result.archives, ["C"])
        self.assertEqual(sorted(p.name for p in self.archives.iterdir()), ["A", "B"])
        self.assertFalse((self.cache / ".oh-trash").exists())

        env_cleanup.remove_tree(env1)
        result = env_cleanup.gc_cache(self.cache)
        self.assertEqual(result.archives, ["A"])
        self.assertEqual([p.name for p in self.archives.iterdir()], ["B"])
        self.assertEqual((env2 / "Lib" / "site-packages" / "pkg_b" / "x.py").read_bytes(), b"bb")

    def test_partly_linked_archive_stays(self):
        a = self.archive("A", {"p/x.py": b"1", "p/y.py": b"2"})
        env = self.env("e1")
        os.link(a / "p" / "x.py", env / "x.py")
        self.assertEqual(env_cleanup.gc_cache(self.cache).archives, [])
        self.assertTrue(a.exists())

    def test_nothing_linked_at_all_means_copy_mode_and_nothing_is_touched(self):
        self.archive("A", {"p/x.py": b"1"})
        result = env_cleanup.gc_cache(self.cache)
        self.assertEqual(result.archives, [])
        self.assertTrue(result.skipped)
        self.assertTrue((self.archives / "A").exists())

    def test_empty_archive_and_missing_cache(self):
        (self.archives / "EMPTY").mkdir()
        a = self.archive("A", {"p/x.py": b"1"})
        self.link(a, self.env("e1"))
        self.assertEqual(env_cleanup.gc_cache(self.cache).archives, [])
        self.assertTrue((self.archives / "EMPTY").exists())
        self.assertTrue(env_cleanup.gc_cache(self.root / "nope").skipped)

    def test_failed_rename_keeps_the_archive(self):
        a = self.archive("A", {"p/x.py": b"1"})
        self.archive("B", {"p/x.py": b"2"})
        self.link(a, self.env("e1"))
        with mock.patch.object(os, "rename", side_effect=PermissionError("locked")):
            result = env_cleanup.gc_cache(self.cache)
        self.assertEqual(result.archives, [])
        self.assertTrue(result.errors)
        self.assertTrue((self.archives / "B").exists())


class CacheTempTests(Tree):
    def test_only_top_level_tmp_folders_go(self):
        half = self.cache / ".tmpAb12" / "torch" / "lib"
        half.mkdir(parents=True)
        (half / "dnnl.lib").write_bytes(b"x" * 100)
        a = self.archive("A", {"pkg_a/.tmpkeep.py": b"a"})
        (self.cache / ".oh-trash").mkdir()
        self.assertEqual(env_cleanup.clear_cache_temp(self.root), 100)
        self.assertFalse((self.cache / ".tmpAb12").exists())
        self.assertTrue((a / "pkg_a" / ".tmpkeep.py").is_file())
        self.assertTrue((self.cache / ".oh-trash").is_dir())
        self.assertIn("cache temp: deleted .tmpAb12", (layout.logs_dir(self.root) / "env-cleanup.log").read_text("utf-8"))

    def test_missing_cache_is_fine(self):
        shutil.rmtree(self.cache)
        self.assertEqual(env_cleanup.clear_cache_temp(self.root), 0)


class TorchPackageTests(Tree):
    def test_remove_torch_packages_uses_record_and_leaves_the_rest(self):
        env = self.env("e1")
        self.link(torch_archive(self), env)
        site = env / "Lib" / "site-packages"
        (site / "numpy").mkdir()
        (site / "numpy" / "__init__.py").write_text("n")
        (site / "torchvision").mkdir()
        (site / "torchvision" / "__init__.py").write_text("v")
        (site / "torchvision-0.23.0+cu128.dist-info").mkdir()
        (site / "torchvision-0.23.0+cu128.dist-info" / "RECORD").write_text("torchvision/__init__.py,,\n")
        (site / "torch_tensorrt_like").mkdir()                      # not a torch package: name only starts with torch
        names = {p.name for p in env_cleanup.torch_paths(env)}
        self.assertEqual(names, {"torch", "functorch", "torchgen", "torch-2.8.0+cu128.dist-info", "torchvision",
                                 "torchvision-0.23.0+cu128.dist-info"})
        self.assertEqual(env_cleanup.remove_torch_packages(env), [])
        self.assertEqual(sorted(p.name for p in site.iterdir()), ["numpy", "torch_tensorrt_like"])
        self.assertTrue(layout.python_in_env(env).is_file())

    def test_freeable_counts_files_only_this_env_and_the_cache_hold(self):
        archive = torch_archive(self)
        env1, env2 = self.env("e1"), self.env("e2")
        self.link(archive, env1)
        total = sum(p.stat().st_size for p in archive.rglob("*") if p.is_file())
        self.assertEqual(env_cleanup.freeable_torch_bytes(env1), total)
        self.link(archive, env2)                                    # a second env keeps every file alive
        self.assertEqual(env_cleanup.freeable_torch_bytes(env1), 0)


class DeleteRetiredTests(Tree):
    def setUp(self):
        super().setUp()
        self.torch = torch_archive(self)
        self.base = self.archive("BASE", {"numpy/__init__.py": b"n"})
        self.old = self.env("cdcedc9f")                             # legacy env, setup ran from it
        self.other = self.env("cdcedc9f-nvidia-1")
        self.new = self.env("cdcedc9f-nvidia-2")
        for env in (self.old, self.other, self.new):
            self.link(self.base, env)
        self.link(self.torch, self.old)
        self.link(self.torch, self.other)
        self.current(env="cdcedc9f-nvidia-2", retired=("cdcedc9f", "cdcedc9f-nvidia-1", "cdcedc9f-nvidia-2", "gone"))

    def test_running_env_loses_torch_others_go_and_launcher_finishes(self):
        report = env_cleanup.delete_retired(self.root, running_env=self.old, delay_s=0)
        self.assertEqual(report.deleted, ["cdcedc9f-nvidia-1"])
        self.assertEqual(report.torch_removed, ["cdcedc9f"])
        self.assertEqual(report.remaining, ["cdcedc9f"])
        self.assertFalse(self.other.exists())
        self.assertTrue(layout.python_in_env(self.old).is_file())
        self.assertFalse((self.old / "Lib" / "site-packages" / "torch").exists())
        self.assertTrue(self.new.exists())                          # current.json points here: never deleted
        self.assertEqual(layout.read_current(self.root).retired, ("cdcedc9f",))
        self.assertEqual(report.cache_archives, 1)                  # TORCH: no env links it any more
        self.assertFalse(self.torch.exists())
        self.assertTrue(self.base.exists())

        report = env_cleanup.delete_retired(self.root, delay_s=0)   # launch.pyw, after setup exited
        self.assertEqual(report.deleted, ["cdcedc9f"])
        self.assertFalse(self.old.exists())
        self.assertEqual(layout.read_current(self.root).retired, ())
        self.assertTrue(self.base.exists())                         # still linked by the new env
        self.assertTrue((self.new / "Lib" / "site-packages" / "numpy" / "__init__.py").is_file())
        self.assertTrue(layout.read_current(self.root).verified)

    def test_nothing_retired_is_a_noop(self):
        self.current(env="cdcedc9f-nvidia-2")
        report = env_cleanup.delete_retired(self.root)
        self.assertEqual((report.deleted, report.errors), ([], []))
        self.assertTrue(self.old.exists())

    def test_failed_delete_stays_retired_and_is_logged(self):
        with mock.patch.object(env_cleanup, "remove_tree", return_value=["locked: x"]):
            report = env_cleanup.delete_retired(self.root, delay_s=0)
        self.assertEqual(report.deleted, [])
        self.assertIn("cdcedc9f", layout.read_current(self.root).retired)
        log = (layout.logs_dir(self.root) / "env-cleanup.log").read_text(encoding="utf-8")
        self.assertIn("locked: x", log)

    def test_a_new_retired_entry_written_meanwhile_is_kept(self):
        real_remove = env_cleanup.remove_tree

        def remove_and_switch(path, **kw):
            cur = layout.read_current(self.root)
            layout.write_current(self.root, layout.Current(**{**cur.__dict__, "retired": cur.retired + ("later",)}))
            return real_remove(path, **kw)

        with mock.patch.object(env_cleanup, "remove_tree", side_effect=remove_and_switch):
            env_cleanup.delete_retired(self.root, running_env=self.old, delay_s=0)
        self.assertEqual(layout.read_current(self.root).retired, ("cdcedc9f", "later"))

    @unittest.skipUnless(sys.platform == "win32", "Windows file locking")
    def test_remove_tree_retries_a_locked_file(self):
        locked = self.other / "Lib" / "site-packages" / "held.bin"
        locked.write_bytes(b"x")
        fh = open(locked, "rb")
        try:
            errors = env_cleanup.remove_tree(self.other, attempts=2, delay_s=0)
            self.assertTrue(errors)
            self.assertTrue(locked.exists())
        finally:
            fh.close()
        self.assertEqual(env_cleanup.remove_tree(self.other, attempts=2, delay_s=0), [])
        self.assertFalse(self.other.exists())

    def test_free_old_envs_first(self):
        self.current(env="cdcedc9f-nvidia-1")
        report = env_cleanup.free_old_envs_first(self.root, running_env=self.old)
        self.assertEqual(sorted(report.deleted), ["cdcedc9f-nvidia-1", "cdcedc9f-nvidia-2"])
        self.assertEqual(report.torch_removed, ["cdcedc9f"])
        self.assertTrue(layout.python_in_env(self.old).is_file())   # setup keeps running from it
        self.assertFalse(layout.read_current(self.root).verified)
        self.assertFalse(self.torch.exists())                       # cache copy freed too
        self.assertTrue(self.base.exists())

    def test_run_after_setup_waits_for_the_lock(self):
        holder = env_guard.SetupLock(self.root)
        self.assertTrue(holder.acquire())
        try:
            self.assertIsNone(env_cleanup.run_after_setup(self.root, lock_timeout_s=0))
            self.assertTrue(self.other.exists())
        finally:
            holder.release()
        report = env_cleanup.run_after_setup(self.root, lock_timeout_s=0)
        self.assertIn("cdcedc9f-nvidia-1", report.deleted)

    def test_helper_python_is_outside_envs(self):
        python = env_cleanup.helper_python(self.root)
        self.assertIsNotNone(python)
        self.assertFalse(str(python).lower().startswith(str(self.envs).lower()))
        with mock.patch("subprocess.Popen") as popen:
            self.assertTrue(env_cleanup.spawn_cleanup_helper(self.root, [123, 456]))
        args = popen.call_args.args[0]
        self.assertEqual(args[args.index("--root") + 1], str(self.root))
        self.assertEqual([a for i, a in enumerate(args) if args[i - 1] == "--wait-pid"], ["123", "456"])

    def test_main_waits_then_cleans(self):
        with mock.patch.object(env_guard, "wait_for_exit", return_value=True) as wait:
            self.assertEqual(env_cleanup.main(["--root", str(self.root), "--wait-pid", "7"]), 0)
        wait.assert_called_once_with(7, 600)
        self.assertFalse(self.other.exists())


class GuardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "install"
        self.root.mkdir()

    def test_lock_is_exclusive_per_root(self):
        first, second = env_guard.SetupLock(self.root), env_guard.SetupLock(self.root)
        self.assertFalse(env_guard.setup_running(self.root))
        self.assertTrue(first.acquire())
        self.addCleanup(first.release)
        self.assertTrue(env_guard.setup_running(self.root))
        self.assertFalse(second.acquire())
        other_root = env_guard.SetupLock(self.root.parent / "other")
        self.assertTrue(other_root.acquire())
        other_root.release()
        first.release()
        self.assertFalse(env_guard.setup_running(self.root))
        self.assertTrue(second.acquire())
        second.release()

    def test_mutex_name(self):
        name = env_guard.mutex_name(self.root)
        self.assertTrue(name.startswith("Local\\oh_streamer-setup-"))
        self.assertEqual(name.count("\\"), 1)
        self.assertNotEqual(name, env_guard.mutex_name(self.root / "x"))

    def test_processes_under_skips_self_ancestors_and_descendants(self):
        envs = str(self.root / "envs")
        exe = lambda name: os.path.join(envs, name, "Scripts", "pythonw.exe")
        procs = [
            ProcessInfo(1, 0, None),
            ProcessInfo(9, 1, exe("old")),                          # venv launcher of the setup process
            ProcessInfo(10, 9, r"C:\py\pythonw.exe"),               # setup itself
            ProcessInfo(11, 10, exe("new")),                        # self-check child
            ProcessInfo(12, 11, exe("new")),                        # grandchild
            ProcessInfo(20, 1, exe("old")),                         # the app: in use
            ProcessInfo(21, 1, r"C:\other\python.exe"),
            ProcessInfo(22, 1, envs + "-not-envs\\x.exe"),          # a sibling folder with the same prefix
            ProcessInfo(23, 1, None),
        ]
        found = env_guard.processes_under(envs, exclude_pid=10, processes=procs)
        self.assertEqual([p.pid for p in found], [20])
        self.assertEqual(found[0].name, "pythonw.exe")

    def test_live_process_list_has_this_process(self):
        procs = env_guard.list_processes()
        if not procs:
            self.skipTest("no process listing on this platform")
        self.assertIn(os.getpid(), {p.pid for p in procs})
        found = {p.pid for p in env_guard.processes_under(Path(sys.prefix))}  # other processes of this venv may exist
        self.assertNotIn(os.getpid(), found)
        self.assertNotIn(os.getppid(), found)                       # our own venv launcher is an ancestor

    def test_wait_for_exit_of_a_missing_pid(self):
        self.assertTrue(env_guard.wait_for_exit(2**31 - 5, 0.1))


class StdlibOnlyTests(unittest.TestCase):
    def test_launcher_side_modules_import_only_stdlib(self):
        stdlib = set(sys.stdlib_module_names)
        for mod in ("env_cleanup", "env_guard", "layout"):
            path = REPO / "src" / "oh_streamer" / "runtime_env" / f"{mod}.py"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:                                  # relative: sibling stdlib-only modules
                        continue
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    self.assertIn(name.split(".")[0], stdlib, f"{mod}: {name}")


if __name__ == "__main__":
    unittest.main()
