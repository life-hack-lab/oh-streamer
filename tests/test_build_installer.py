"""Tests for packaging/build_installer.py with a tiny fake project and a fake inputs manifest (no network)."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))

import build_installer as bi  # noqa: E402
import installer_inputs as ii  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write(path: Path, data: bytes | str = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


class DefaultRifeDirTest(unittest.TestCase):
    def test_default_is_in_repo_copy(self):
        d = bi._default_rife_dir()
        self.assertEqual(d, ii.ROOT / "third_party" / "rife")
        self.assertTrue((d / "vsrife_models" / "LICENSE").is_file())
        self.assertTrue((d / "README.md").is_file())


class StageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        t = Path(self._tmp.name)
        self.project, self.cache, self.out, self.rife = t / "proj", t / "cache", t / "stage", t / "rife"
        p = self.project
        write(p / "launch.pyw", "print(1)")
        write(p / "pyproject.toml", "[project]\n")
        write(p / "uv.lock", "lock")
        write(p / "LICENSE", "AGPL")
        write(p / "THIRD_PARTY_NOTICES.md", "notices")
        write(p / "src/oh_streamer/version.py", '__version__ = "9.8.7"\n')
        shutil_layout = ROOT / "src/oh_streamer/runtime_env/layout.py"
        write(p / "src/oh_streamer/runtime_env/layout.py", shutil_layout.read_bytes())
        write(p / "src/oh_streamer/resources/a.lua", "lua")
        write(p / "src/oh_streamer/__pycache__/x.cpython-313.pyc", "pyc")
        write(p / "src/oh_streamer/old.py.bak", "bak")
        write(p / "tests/test_x.py", "no")
        write(p / "docs/d.md", "no")
        (p / "packaging/wheels").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p / "packaging/wheels/lada-1-py3-none-any.whl", "w") as z:
            z.writestr("lada-1.dist-info/licenses/LICENSE.md", "lada lic")
        write(self.rife / "vsrife_models/m.py", "m")
        write(self.rife / "vsrife_models/LICENSE", "mit")
        write(self.rife / "vsrife_models/ref.py.txt", "no")
        write(self.rife / "README.md", "r")
        write(self.rife / "weights/w.pkl", "no")
        # cache laid out like installer_inputs
        tool = b"tool-bytes"
        write(self.cache / "files/app/tools/t.exe", tool)
        write(self.cache / "files/uv.exe", b"uv")
        write(self.cache / "trees/py/python.exe", b"py")
        write(self.cache / "trees/py/LICENSE.txt", b"pylic")
        write(self.cache / "trees/py/.complete", "ARCH\n")
        (self.cache / "archives").mkdir(parents=True, exist_ok=True)
        self.inputs = {
            "archives": [{"id": "pyarch", "filename": "py.tgz", "url": "u", "sha256": "ARCH", "size": 1}],
            "files": [
                {"id": "t", "dest": "app/tools/t.exe", "sha256": sha(tool), "size": len(tool), "source": {"type": "url", "url": "u"}},
                {"id": "uv", "dest": "uv.exe", "sha256": sha(b"uv"), "size": 2, "source": {"type": "url", "url": "u"}},
            ],
            "trees": [{"id": "py", "dest": "python", "source": {"type": "archive", "archive": "pyarch"}}],
        }
        write(self.cache / "archives/py.tgz", "a")
        self.inputs["archives"][0]["sha256"] = "ARCH"

    def stage(self):
        orig = ii.verify_all
        ii.verify_all = lambda m, c: []          # cache integrity is installer_inputs' own tested job
        try:
            return bi.stage_build(project=self.project, out=self.out, cache=self.cache, inputs=self.inputs,
                                  rife_dir=self.rife, fetch=False, log=lambda *_: None)
        finally:
            ii.verify_all = orig

    def test_app_whitelist_and_exclusions(self):
        m = self.stage()
        paths = {r["path"] for r in m["files"]}
        for want in ("LICENSE", "THIRD_PARTY_NOTICES.md", "app/launch.pyw", "app/pyproject.toml", "app/uv.lock", "app/src/oh_streamer/resources/a.lua",
                     "app/src/oh_streamer/version.py", "app/packaging/wheels/lada-1-py3-none-any.whl",
                     "app/tools/t.exe", "uv.exe", "python/python.exe", "python/LICENSE.txt",
                     "app/third_party/rife/vsrife_models/m.py", "app/third_party/rife/vsrife_models/LICENSE",
                     "app/third_party/rife/README.md"):
            self.assertIn(want, paths)
        for bad in paths:
            self.assertFalse(bad.startswith(("app/tests", "app/docs")), bad)
            self.assertNotIn("__pycache__", bad)
            self.assertFalse(bad.endswith((".pyc", ".bak", ".complete", ".txt")) and bad.startswith("app/third_party/rife/vsrife_models/ref"))
        self.assertNotIn("app/third_party/rife/weights/w.pkl", paths)
        self.assertNotIn("app/src/oh_streamer/old.py.bak", paths)
        self.assertNotIn("python/.complete", paths)

    def test_manifest_content(self):
        m = self.stage()
        self.assertEqual(m["version"], "9.8.7")
        self.assertEqual(m["uv_lock"]["sha256"], sha(b"lock"))
        self.assertEqual(m["uv_lock"]["lock_id"], sha(b"lock")[:8])
        self.assertIn("revision", m["git"])
        self.assertTrue(m["built_at"])
        self.assertEqual({i["id"] for i in m["inputs"]["files"]}, {"t", "uv"})
        rec = next(r for r in m["files"] if r["path"] == "uv.exe")
        self.assertEqual((rec["size"], rec["sha256"]), (2, sha(b"uv")))
        self.assertIn("oh-streamer-LICENSE.txt", m["licenses"])
        self.assertIn("lada/LICENSE.md", m["licenses"])
        self.assertTrue(any("uv" in x for x in m["missing_licenses"]))
        on_disk = json.loads((self.out / bi.BUILD_MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["files"], m["files"])

    def test_restage_clears_old_files(self):
        self.stage()
        write(self.out / "stale.txt")
        self.stage()
        self.assertFalse((self.out / "stale.txt").exists())

    def test_verify_ok_and_detects_problems(self):
        self.stage()
        self.assertEqual(bi.verify_stage(self.out), [])
        (self.out / "uv.exe").write_bytes(b"UV")
        self.assertTrue(any("sha256 differs: uv.exe" in p for p in bi.verify_stage(self.out)))
        (self.out / "uv.exe").write_bytes(b"uvuv")
        self.assertTrue(any("size differs: uv.exe" in p for p in bi.verify_stage(self.out)))
        (self.out / "uv.exe").unlink()
        self.assertIn("missing: uv.exe", bi.verify_stage(self.out))
        write(self.out / "extra.dll")
        self.assertIn("unexpected: extra.dll", bi.verify_stage(self.out))

    def test_missing_required_app_file_fails_before_touching_out(self):
        (self.project / "uv.lock").unlink()
        with self.assertRaises(bi.BuildError):
            self.stage()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
