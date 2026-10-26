"""Tests for packaging/installer_inputs.py (no network: fake opener only)."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))

import installer_inputs as ii  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeOpener:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return FakeResponse(self.routes[url])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.quiet = lambda _msg: None

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_then_cache_hit(self):
        data = b"hello world"
        opener = FakeOpener({"https://x/a": data})
        dest = self.dir / "a.bin"
        self.assertTrue(ii.download_verified("https://x/a", dest, sha(data), opener=opener, log=self.quiet))
        self.assertEqual(dest.read_bytes(), data)
        self.assertFalse(dest.with_name("a.bin.part").exists())
        self.assertFalse(ii.download_verified("https://x/a", dest, sha(data), opener=opener, log=self.quiet))
        self.assertEqual(len(opener.calls), 1)

    def test_hash_mismatch_removes_part_and_raises(self):
        opener = FakeOpener({"https://x/a": b"evil"})
        dest = self.dir / "a.bin"
        with self.assertRaises(ii.InputsError):
            ii.download_verified("https://x/a", dest, sha(b"good"), opener=opener, log=self.quiet)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("a.bin.part").exists())

    def test_corrupt_cached_file_is_replaced(self):
        data = b"payload"
        dest = self.dir / "a.bin"
        dest.write_bytes(b"corrupt")
        opener = FakeOpener({"https://x/a": data})
        self.assertTrue(ii.download_verified("https://x/a", dest, sha(data), opener=opener, log=self.quiet))
        self.assertEqual(dest.read_bytes(), data)

    def test_failed_transfer_cleans_part(self):
        def broken(url):
            raise OSError("boom")

        dest = self.dir / "a.bin"
        with self.assertRaises(ii.InputsError):
            ii.download_verified("https://x/a", dest, "0" * 64, opener=broken, log=self.quiet, retries=0)
        self.assertFalse(dest.with_name("a.bin.part").exists())


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_zip_member_extracted(self):
        z = self.dir / "a.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("top/bin/tool.exe", b"exe")
        out = self.dir / "out" / "tool.exe"
        ii.extract_zip_member(z, "top/bin/tool.exe", out)
        self.assertEqual(out.read_bytes(), b"exe")

    def test_zip_missing_member(self):
        z = self.dir / "a.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("a.txt", b"x")
        with self.assertRaises(ii.InputsError):
            ii.extract_zip_member(z, "b.txt", self.dir / "b.txt")

    def test_zip_member_name_traversal_rejected(self):
        z = self.dir / "a.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../evil.txt", b"x")
        for name in ("../evil.txt", "/abs.txt", "C:/x.txt"):
            with self.assertRaises(ii.InputsError):
                ii.extract_zip_member(z, name, self.dir / "o.txt")
        with self.assertRaises(ii.InputsError):
            ii.extract_zip_all(z, self.dir / "all")
        self.assertFalse((self.dir / "evil.txt").exists())

    def test_tar_tree_strip_and_traversal(self):
        good = self.dir / "good.tar.gz"
        with tarfile.open(good, "w:gz") as tf:
            info = tarfile.TarInfo("python/python.exe")
            info.size = 3
            tf.addfile(info, io.BytesIO(b"abc"))
        dest = self.dir / "tree"
        self.assertEqual(ii.extract_tar_tree(good, dest, "python"), 1)
        self.assertEqual((dest / "python.exe").read_bytes(), b"abc")

        bad = self.dir / "bad.tar.gz"
        with tarfile.open(bad, "w:gz") as tf:
            info = tarfile.TarInfo("python/../../evil.txt")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(ii.InputsError):
            ii.extract_tar_tree(bad, self.dir / "tree2", "python")
        self.assertFalse((self.dir / "evil.txt").exists())

    def test_tar_symlink_rejected(self):
        link = self.dir / "link.tar.gz"
        with tarfile.open(link, "w:gz") as tf:
            info = tarfile.TarInfo("python/l")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        with self.assertRaises(ii.InputsError):
            ii.extract_tar_tree(link, self.dir / "t", "python")


FAKE_7Z_TOOL = (
    "import sys, pathlib\n"
    "out = pathlib.Path(sys.argv[1][2:]); member = sys.argv[3]\n"
    "if member == 'missing.bin': sys.exit(2)\n"
    "p = out / member; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'SEVENZ')\n"
)


class SevenZipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        script = self.dir / "fake7z.py"
        script.write_text(FAKE_7Z_TOOL, encoding="utf-8")
        self.tool = [sys.executable, str(script)]
        self.archive = self.dir / "a.7z"
        self.archive.write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_member_extracted_through_external_tool(self):
        out = self.dir / "out" / "mpv.exe"
        ii.extract_7z_member(self.archive, "mpv.exe", out, tools=[self.tool])
        self.assertEqual(out.read_bytes(), b"SEVENZ")
        self.assertFalse(out.with_name("mpv.exe.part").exists())

    def test_failing_tool_falls_back_to_next(self):
        broken = [sys.executable, "-c", "import sys; sys.exit(3)"]
        out = self.dir / "o.bin"
        ii.extract_7z_member(self.archive, "m.bin", out, tools=[broken, self.tool])
        self.assertEqual(out.read_bytes(), b"SEVENZ")

    def test_missing_member_and_no_tools(self):
        with self.assertRaises(ii.InputsError):
            ii.extract_7z_member(self.archive, "missing.bin", self.dir / "o.bin", tools=[self.tool])
        with self.assertRaises(ii.InputsError):
            ii.extract_7z_member(self.archive, "a.bin", self.dir / "o.bin", tools=[])

    def test_traversal_member_rejected_before_running_anything(self):
        for name in ("../evil.exe", "/abs.exe", "C:/x.exe"):
            with self.assertRaises(ii.InputsError):
                ii.extract_7z_member(self.archive, name, self.dir / "o.bin", tools=[self.tool])
        self.assertFalse((self.dir / "o.bin").exists())

    def test_archive_format_detection(self):
        self.assertEqual(ii.archive_format({"filename": "a.7z"}), "7z")
        self.assertEqual(ii.archive_format({"filename": "a.zip"}), "zip")
        self.assertEqual(ii.archive_format({"filename": "a.tar.gz"}), "tar")
        self.assertEqual(ii.archive_format({"filename": "vc_redist.x64.exe"}), "burn")
        self.assertEqual(ii.archive_format({"filename": "vc.exe", "format": "zip"}), "zip")


def _tool(name):
    root = os.environ.get("SystemRoot")
    if root and (Path(root) / "System32" / name).is_file():
        return str(Path(root) / "System32" / name)
    return None


@unittest.skipUnless(sys.platform == "win32" and _tool("expand.exe") and _tool("makecab.exe"), "needs Windows cab tools")
class BurnBundleTests(unittest.TestCase):
    def test_nested_cabinet_member_extracted_from_exe_like_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            work.mkdir()
            makecab = _tool("makecab.exe")

            def make_cab(cab_name, files):
                # two or more files per cabinet, like the real bundle (a one-file cabinet makes expand.exe rename its output)
                lines = [f".Set CabinetNameTemplate={cab_name}", ".Set DiskDirectoryTemplate=" + str(work),
                         ".Set Cabinet=on", ".Set Compress=on"] + ['"' + str(work / f) + '"' for f in files]
                ddf = work / (cab_name + ".ddf")
                ddf.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self.assertEqual(subprocess.run([makecab, "/F", str(ddf)], cwd=work, capture_output=True).returncode, 0)

            (work / "msvcp140.dll_amd64").write_bytes(b"DLLDATA" * 50)
            (work / "vcruntime140.dll_amd64").write_bytes(b"OTHER" * 50)
            make_cab("a12.cab", ["msvcp140.dll_amd64", "vcruntime140.dll_amd64"])
            # the container "a12" is itself a cabinet; wrap it into an outer cabinet under that name
            (work / "a12").write_bytes((work / "a12.cab").read_bytes())
            (work / "a0").write_bytes(b"other payload")
            make_cab("outer.cab", ["a0", "a12"])
            exe = tmp / "bundle.exe"
            exe.write_bytes(b"MZ" + b"\0" * 300 + (work / "outer.cab").read_bytes() + b"tail")
            self.assertEqual(len(list(ii.find_embedded_cabs(exe.read_bytes()))), 1)
            out = tmp / "out" / "msvcp140.dll"
            ii.extract_burn_member(exe, "a12/msvcp140.dll_amd64", out)
            self.assertEqual(out.read_bytes(), b"DLLDATA" * 50)
            with self.assertRaises(ii.InputsError):
                ii.extract_burn_member(exe, "a12/nothing.dll", tmp / "x.dll")
            with self.assertRaises(ii.InputsError):
                ii.extract_burn_member(exe, "zz/msvcp140.dll_amd64", tmp / "y.dll")
            with self.assertRaises(ii.InputsError):
                ii.extract_burn_member(exe, "../a12/x", tmp / "z.dll")
            with self.assertRaises(ii.InputsError):
                ii.extract_burn_member(exe, "onlyname", tmp / "z.dll")


class BurnHeaderTests(unittest.TestCase):
    def test_find_embedded_cabs_checks_header(self):
        def header(size):
            return b"MSCF\0\0\0\0" + size.to_bytes(4, "little") + b"\0" * 12 + b"\x03\x01" + b"\0" * 10

        good = header(100) + b"\0" * 64
        data = b"junk" + good + b"MSCF\0\0\0\0" + b"\xff" * 40
        self.assertEqual(list(ii.find_embedded_cabs(data)), [(4, 100)])
        self.assertEqual(list(ii.find_embedded_cabs(b"MSCF\0\0\0\0")), [])


class FetchAllTests(unittest.TestCase):
    def test_fetch_and_verify_with_all_source_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as zf:
                zf.writestr("p/tool.exe", b"TOOL")
            zdata = zbuf.getvalue()
            tbuf = io.BytesIO()
            with tarfile.open(fileobj=tbuf, mode="w:gz") as tf:
                info = tarfile.TarInfo("python/python.exe")
                info.size = 2
                tf.addfile(info, io.BytesIO(b"py"))
            tdata = tbuf.getvalue()
            local = tmp / "src" / "m.dll"
            local.parent.mkdir()
            local.write_bytes(b"DLL")
            manifest = {
                "format": ii.MANIFEST_FORMAT,
                "archives": [
                    {"id": "z", "filename": "z.zip", "url": "https://x/z.zip", "sha256": sha(zdata), "size": len(zdata)},
                    {"id": "t", "filename": "t.tar.gz", "url": "https://x/t.tar.gz", "sha256": sha(tdata), "size": len(tdata)},
                ],
                "files": [
                    {"id": "tool", "dest": "app/tools/tool.exe", "sha256": sha(b"TOOL"), "size": 4,
                     "source": {"type": "archive", "archive": "z", "member": "p/tool.exe"}},
                    {"id": "raw", "dest": "app/raw.bin", "sha256": sha(b"RAW"), "size": 3,
                     "source": {"type": "url", "url": "https://x/raw"}},
                    {"id": "dll", "dest": "app/m.dll", "sha256": sha(b"DLL"), "size": 3,
                     "source": {"type": "local", "candidates": [str(local)]}},
                ],
                "trees": [{"id": "python", "dest": "python", "source": {"archive": "t", "strip": "python"}}],
            }
            self.assertEqual(ii.validate_manifest(manifest), [])
            opener = FakeOpener({"https://x/z.zip": zdata, "https://x/t.tar.gz": tdata, "https://x/raw": b"RAW"})
            cache = tmp / "cache"
            ii.fetch_all(manifest, cache, opener=opener, log=lambda _m: None)
            self.assertEqual(ii.verify_all(manifest, cache), [])
            n = len(opener.calls)
            ii.fetch_all(manifest, cache, opener=opener, log=lambda _m: None)
            self.assertEqual(len(opener.calls), n)  # all cache hits
            # tamper -> verify reports, fetch repairs
            (cache / "files" / "app" / "raw.bin").write_bytes(b"BAD")
            self.assertTrue(ii.verify_all(manifest, cache))
            ii.fetch_all(manifest, cache, opener=opener, log=lambda _m: None)
            self.assertEqual(ii.verify_all(manifest, cache), [])

    def test_local_source_wrong_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "m.dll"
            local.write_bytes(b"other")
            entry = {"id": "dll", "dest": "app/m.dll", "sha256": sha(b"DLL"),
                     "source": {"type": "local", "candidates": [str(local)]}}
            with self.assertRaises(ii.InputsError):
                ii.resolve_local(entry, [])


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(ii.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_real_manifest_valid(self):
        self.assertEqual(ii.validate_manifest(self.manifest), [])
        ii.load_manifest()

    def test_every_item_has_sha256_and_relative_unique_dest(self):
        dests = []
        for entry in self.manifest["files"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", entry["id"])
            self.assertGreater(entry["size"], 0)
            self.assertFalse(Path(entry["dest"]).is_absolute(), entry["id"])
            self.assertNotIn("..", entry["dest"].split("/"))
            dests.append(entry["dest"].lower())
        for entry in self.manifest["archives"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", entry["id"])
        self.assertEqual(len(dests), len(set(dests)))

    def test_mpv_is_downloadable_plain_x86_64_and_vc_dlls_are_staged_next_to_python(self):
        by_id = {e["id"]: e for e in self.manifest["files"]}
        archives = {a["id"]: a for a in self.manifest["archives"]}
        self.assertEqual(by_id["mpv"]["source"]["type"], "archive")
        mpv_archive = archives[by_id["mpv"]["source"]["archive"]]
        self.assertTrue(mpv_archive["url"].startswith("https://github.com/shinchiro/mpv-winbuild-cmake/releases/download/"))
        self.assertNotIn("-v3-", mpv_archive["filename"])
        self.assertNotIn("mpv-d3dcompiler", by_id)
        self.assertFalse([e for e in self.manifest["files"] if e["source"]["type"] == "local"])
        for name in ("msvcp140", "msvcp140_1", "vcomp140"):
            entry = by_id["vcrt-" + name]
            self.assertEqual(entry["dest"], f"python/{name}.dll")
            self.assertEqual(archives[entry["source"]["archive"]]["format"], "burn")
        self.assertNotIn("vcrt-vcruntime140", by_id)  # shipped by the embedded Python

    def test_webview2_bootstrapper_is_pinned_and_staged_outside_app(self):
        entry = {e["id"]: e for e in self.manifest["files"]}["webview2-bootstrapper"]
        self.assertEqual(entry["dest"], "redist/MicrosoftEdgeWebview2Setup.exe")
        self.assertEqual(entry["source"]["type"], "url")
        self.assertTrue(entry["source"]["url"].startswith("https://"))
        self.assertIn("microsoft.com", entry["source"]["url"])
        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(entry["dest"].startswith(("app/", "python/")))

    def test_files_may_sit_inside_a_tree_but_not_overlap_each_other(self):
        m = json.loads(json.dumps(self.manifest))
        self.assertEqual(ii.validate_manifest(m), [])
        extra = dict(m["files"][-1])
        extra["id"] = "dup"
        m["files"].append(extra)
        self.assertTrue(any("overlaps" in p for p in ii.validate_manifest(m)))
        m = json.loads(json.dumps(self.manifest))
        m["archives"][0]["format"] = "rar"
        self.assertTrue(any("archive format" in p for p in ii.validate_manifest(m)))

    def test_hf_models_pinned_to_commit(self):
        for entry in self.manifest["files"]:
            url = entry["source"].get("url", "")
            if "huggingface.co" in url:
                self.assertNotIn("/resolve/main/", url)
                self.assertRegex(url, r"/resolve/[0-9a-f]{40}/")

    def test_validation_catches_problems(self):
        bad = json.loads(json.dumps(self.manifest))
        bad["files"][0]["sha256"] = "xyz"
        bad["files"][1]["dest"] = bad["files"][0]["dest"]
        bad["files"][2]["dest"] = "/abs/path"
        problems = ii.validate_manifest(bad)
        self.assertTrue(any("sha256" in p for p in problems))
        self.assertTrue(any("overlaps" in p for p in problems))
        self.assertTrue(any("dest" in p and "absolute" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
