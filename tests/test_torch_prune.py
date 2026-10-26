r"""prune_torch_static_libs: removes torch\lib\*.lib from an env and only the hardlinked copies from the uv cache."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env.torch_prune import prune_torch_static_libs


class TorchPruneTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.env = self.root / "env"
        self.cache = self.root / "cache"
        self.env_lib = self.env / "Lib" / "site-packages" / "torch" / "lib"
        self.cache_lib = self.cache / "archive-v0" / "aaa" / "torch" / "lib"
        self.env_lib.mkdir(parents=True)
        self.cache_lib.mkdir(parents=True)

    def linked(self, name, data=b"x" * 100):
        """A file in the cache archive with a hardlink in the env, like uv installs it."""
        (self.cache_lib / name).write_bytes(data)
        os.link(self.cache_lib / name, self.env_lib / name)

    def test_env_only_prune_frees_nothing_while_cache_links_it(self):
        self.linked("dnnl.lib")
        res = prune_torch_static_libs(self.env)
        self.assertEqual(res.files, [str(self.env_lib / "dnnl.lib")])
        self.assertFalse((self.env_lib / "dnnl.lib").exists())
        self.assertTrue((self.cache_lib / "dnnl.lib").exists())
        self.assertEqual(res.bytes_freed, 0)                    # the cache copy still holds the data

    def test_env_only_prune_of_unshared_file_frees_bytes(self):
        (self.env_lib / "c10.lib").write_bytes(b"x" * 50)
        res = prune_torch_static_libs(self.env)
        self.assertEqual(res.bytes_freed, 50)
        self.assertEqual(res.errors, [])

    def test_cache_prune_removes_only_linked_copies(self):
        self.linked("dnnl.lib")
        other = self.cache / "archive-v0" / "bbb" / "torch" / "lib"
        other.mkdir(parents=True)
        (other / "dnnl.lib").write_bytes(b"x" * 100)            # same name and content, different file
        (self.cache_lib / "stray.lib").write_bytes(b"y")        # same archive, not in the env
        res = prune_torch_static_libs(self.env, self.cache)
        self.assertFalse((self.cache_lib / "dnnl.lib").exists())
        self.assertTrue((other / "dnnl.lib").exists())
        self.assertTrue((self.cache_lib / "stray.lib").exists())
        self.assertEqual(sorted(res.files), sorted([str(self.env_lib / "dnnl.lib"), str(self.cache_lib / "dnnl.lib")]))

    def test_bytes_freed_counted_once_per_inode(self):
        self.linked("dnnl.lib", b"x" * 100)
        self.linked("c10.lib", b"y" * 30)
        res = prune_torch_static_libs(self.env, self.cache)
        self.assertEqual(len(res.files), 4)
        self.assertEqual(res.bytes_freed, 130)

    def test_dry_run_deletes_nothing(self):
        self.linked("dnnl.lib")
        res = prune_torch_static_libs(self.env, self.cache, dry_run=True)
        self.assertTrue(res.dry_run)
        self.assertEqual(len(res.files), 2)
        self.assertEqual(res.bytes_freed, 100)
        self.assertTrue((self.env_lib / "dnnl.lib").exists())
        self.assertTrue((self.cache_lib / "dnnl.lib").exists())

    def test_other_files_untouched(self):
        for name in ("torch_cuda.dll", "dnnl.lib.txt", "notes.h", "libfoo.a"):
            self.linked(name)
        sub = self.env_lib / "sub"
        sub.mkdir()
        (sub / "nested.lib").write_bytes(b"z")                  # only torch\lib itself is a target
        res = prune_torch_static_libs(self.env, self.cache)
        self.assertEqual(res.files, [])
        for name in ("torch_cuda.dll", "dnnl.lib.txt", "notes.h", "libfoo.a"):
            self.assertTrue((self.env_lib / name).exists())
            self.assertTrue((self.cache_lib / name).exists())
        self.assertTrue((sub / "nested.lib").exists())

    def test_missing_dirs_are_fine(self):
        res = prune_torch_static_libs(self.root / "nope", self.root / "nocache")
        self.assertEqual((res.files, res.bytes_freed, res.errors), ([], 0, []))

    def test_symlinked_torch_dir_is_not_followed(self):
        outside = self.root / "outside" / "lib"
        outside.mkdir(parents=True)
        (outside / "keep.lib").write_bytes(b"k")
        env2 = self.root / "env2"
        site = env2 / "Lib" / "site-packages"
        site.mkdir(parents=True)
        try:
            os.symlink(outside.parent, site / "torch", target_is_directory=True)
        except (OSError, NotImplementedError):
            # no symlink privilege: a junction needs none and is the reparse point pip/uv envs can really contain
            made = sys.platform == "win32" and subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(site / "torch"), str(outside.parent)],
                capture_output=True).returncode == 0
            if not made:
                self.skipTest("cannot create symlinks or junctions here")
        res = prune_torch_static_libs(env2)
        self.assertEqual(res.files, [])
        self.assertTrue((outside / "keep.lib").exists())

    def test_read_only_file_is_removed(self):
        path = self.env_lib / "ro.lib"
        path.write_bytes(b"x")
        os.chmod(path, 0o444)
        res = prune_torch_static_libs(self.env)
        self.assertFalse(path.exists())
        self.assertEqual(res.errors, [])

    def test_failure_is_collected_not_raised(self):
        self.linked("a.lib")
        self.linked("b.lib")
        real_remove = os.remove
        target = str(self.env_lib / "a.lib")

        def flaky(path):
            if str(path) == target:
                raise OSError("in use")
            real_remove(path)

        import oh_streamer.runtime_env.torch_prune as mod
        orig, mod.os.remove = mod.os.remove, flaky
        try:
            res = prune_torch_static_libs(self.env, self.cache)
        finally:
            mod.os.remove = orig
        self.assertEqual(res.errors, [(target, "in use")])
        self.assertFalse((self.env_lib / "b.lib").exists())
        self.assertEqual(res.bytes_freed, 100)                  # only b.lib was fully removed


if __name__ == "__main__":
    unittest.main()
