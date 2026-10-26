import hashlib
import io
import re
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from oh_streamer.subtitles import model_download as md

FILES = {}      # url path -> bytes
SLOW = set()    # url paths served slowly (for cancel)
NO_RANGE = set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        data = FILES.get(self.path)
        if data is None:
            self.send_error(404)
            return
        start = 0
        match = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        if match and self.path not in NO_RANGE:
            start = int(match.group(1))
            if start >= len(data):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if self.path in SLOW:
                for i in range(0, len(body), 1000):
                    self.wfile.write(body[i:i + 1000])
                    self.wfile.flush()
                    time.sleep(0.05)
            else:
                self.wfile.write(body)
        except OSError:
            pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FILES.clear(); SLOW.clear(); NO_RANGE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name) / "models"
        self.addCleanup(self._tmp.cleanup)

    def spec(self, name, data, dest, **kw):
        FILES["/" + name] = data
        return md.FileSpec(self.base + "/" + name, dest, len(data), sha(data), **kw)

    def item(self, item_id, *files, marker="", default=True):
        return md.CatalogItem(item_id, item_id, "r1", default, tuple(files), marker=marker)

    def run_job(self, items, ids, **kw):
        job = md.DownloadJob(self.models, ids, items, backoff=0.01, **kw).start()
        job.join(30)
        return job

    def test_full_download_and_status(self):
        data = bytes(range(256)) * 2000
        item = self.item("a", self.spec("m.bin", data, "sakura/m.gguf"))
        self.assertFalse(md.status(self.models, [item])[0]["installed"])
        job = self.run_job([item], ["a"])
        snap = job.snapshot()
        self.assertEqual(snap["state"], "done", snap)
        self.assertEqual(snap["bytes_done"], snap["bytes_total"])
        self.assertEqual((self.models / "sakura/m.gguf").read_bytes(), data)
        self.assertFalse((self.models / "sakura/m.gguf.part").exists())
        row = md.status(self.models, [item])[0]
        self.assertTrue(row["installed"])
        self.assertEqual(row["remaining_bytes"], 0)
        record = (self.models / "downloads.json").read_text(encoding="utf-8")
        self.assertIn('"a"', record)

    def test_resume_from_part(self):
        data = b"0123456789" * 50000
        spec = self.spec("r.bin", data, "asr/model.safetensors")
        item = self.item("a", spec)
        part = self.models / "asr/model.safetensors.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(data[:123456])
        row = md.status(self.models, [item])[0]
        self.assertEqual(row["partial_bytes"], 123456)
        self.assertEqual(row["remaining_bytes"], len(data) - 123456)
        job = self.run_job([item], ["a"])
        self.assertEqual(job.snapshot()["state"], "done", job.snapshot())
        self.assertEqual((self.models / "asr/model.safetensors").read_bytes(), data)

    def test_server_without_range_restarts(self):
        data = b"abcdefgh" * 10000
        spec = self.spec("n.bin", data, "x/n.bin")
        NO_RANGE.add("/n.bin")
        part = self.models / "x/n.bin.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(data[:5000])
        job = self.run_job([self.item("a", spec)], ["a"])
        self.assertEqual(job.snapshot()["state"], "done", job.snapshot())
        self.assertEqual((self.models / "x/n.bin").read_bytes(), data)
        self.assertEqual(job.snapshot()["bytes_done"], len(data))

    def test_sha_mismatch_fails_and_not_published(self):
        data = b"good data" * 1000
        bad = md.FileSpec(self.base + "/bad.bin", "x/bad.bin", len(data), sha(b"other"))
        FILES["/bad.bin"] = data
        job = self.run_job([self.item("a", bad)], ["a"])
        snap = job.snapshot()
        self.assertEqual(snap["state"], "failed")
        self.assertIn("校驗碼", snap["error"])
        self.assertFalse((self.models / "x/bad.bin").exists())
        self.assertFalse((self.models / "x/bad.bin.part").exists())
        self.assertFalse(md.status(self.models, [self.item("a", bad)])[0]["installed"])

    def test_http_error_fails_with_message(self):
        spec = md.FileSpec(self.base + "/missing", "x/missing", 10, "0" * 64)
        job = self.run_job([self.item("a", spec)], ["a"])
        self.assertEqual(job.snapshot()["state"], "failed")
        self.assertIn("404", job.snapshot()["error"])

    def test_network_error_retries_then_fails(self):
        spec = md.FileSpec("http://127.0.0.1:1/none", "x/none", 10, "0" * 64)
        job = self.run_job([self.item("a", spec)], ["a"], )
        snap = job.snapshot()
        self.assertEqual(snap["state"], "failed")
        self.assertIn("網路", snap["error"])

    def test_cancel_keeps_part_then_resume(self):
        data = b"z" * 400000
        spec = self.spec("c.bin", data, "x/c.bin")
        SLOW.add("/c.bin")
        item = self.item("a", spec)
        job = md.DownloadJob(self.models, ["a"], [item], backoff=0.01).start()
        deadline = time.time() + 10
        while job.snapshot()["bytes_done"] < 20000 and time.time() < deadline:
            time.sleep(0.02)
        job.cancel()
        job.join(10)
        self.assertEqual(job.snapshot()["state"], "cancelled")
        part = self.models / "x/c.bin.part"
        self.assertTrue(part.exists())
        self.assertFalse((self.models / "x/c.bin").exists())
        SLOW.clear()
        job2 = self.run_job([item], ["a"])
        self.assertEqual(job2.snapshot()["state"], "done", job2.snapshot())
        self.assertEqual((self.models / "x/c.bin").read_bytes(), data)

    def test_zip_extract(self):
        archive = make_zip({"llama-server.exe": b"EXE", "sub/lib.dll": b"DLL"})
        spec = self.spec("z.zip", archive, ".downloads/z.zip", extract_to="llama-cpp", unpacked=100)
        item = self.item("srv", spec, marker="llama-cpp/llama-server.exe")
        job = self.run_job([item], ["srv"])
        self.assertEqual(job.snapshot()["state"], "done", job.snapshot())
        self.assertEqual((self.models / "llama-cpp/llama-server.exe").read_bytes(), b"EXE")
        self.assertEqual((self.models / "llama-cpp/sub/lib.dll").read_bytes(), b"DLL")
        self.assertFalse((self.models / ".downloads").exists())
        self.assertEqual([p.name for p in self.models.glob(".extract-*")], [])
        self.assertTrue(md.status(self.models, [item])[0]["installed"])

    def _two_zip_item(self, revision="r1"):
        a = make_zip({"cudart.dll": b"CUDA"})
        b = make_zip({"llama-server.exe": b"EXE"})
        s1 = self.spec("z1.zip", a, ".downloads/z1.zip", extract_to="llama-cpp", unpacked=100)
        s2 = self.spec("z2.zip", b, ".downloads/z2.zip", extract_to="llama-cpp", unpacked=100)
        item = md.CatalogItem("srv", "srv", revision, True, (s1, s2), marker="llama-cpp/llama-server.exe")
        return item, s1, s2, b

    def test_zip_resume_after_failure_between_zips(self):
        item, s1, s2, b = self._two_zip_item()
        FILES.pop("/z2.zip")                                   # zip 2 fails (404) after zip 1 was extracted
        job = self.run_job([item], ["srv"])
        self.assertEqual(job.snapshot()["state"], "failed", job.snapshot())
        self.assertEqual((self.models / "llama-cpp/cudart.dll").read_bytes(), b"CUDA")
        self.assertFalse((self.models / ".downloads/z1.zip").exists())
        row = md.status(self.models, [item])[0]
        self.assertFalse(row["installed"])
        self.assertEqual(row["remaining_bytes"], s2.size)
        self.assertEqual(row["partial_bytes"], s1.size)
        self.assertEqual(md.plan(self.models, ["srv"], [item], free_bytes=10**12)["download_bytes"], s2.size)
        # resume: zip 1 is gone from the server, so any attempt to re-download it would fail
        FILES.pop("/z1.zip")
        FILES["/z2.zip"] = b
        job2 = self.run_job([item], ["srv"])
        snap = job2.snapshot()
        self.assertEqual(snap["state"], "done", snap)
        self.assertEqual(snap["bytes_done"], snap["bytes_total"])
        self.assertEqual((self.models / "llama-cpp/llama-server.exe").read_bytes(), b"EXE")
        self.assertEqual((self.models / "llama-cpp/cudart.dll").read_bytes(), b"CUDA")
        self.assertTrue(md.status(self.models, [item])[0]["installed"])
        self.assertNotIn("srv", md._read_record(self.models).get("partial", {}))

    def test_zip_partial_record_invalidated_by_revision_change(self):
        item, s1, s2, b = self._two_zip_item()
        FILES.pop("/z2.zip")
        self.run_job([item], ["srv"])
        newer = md.CatalogItem(item.id, item.label, "r2", True, item.files, marker=item.marker)
        self.assertEqual(md.status(self.models, [newer])[0]["remaining_bytes"], s1.size + s2.size)
        FILES.pop("/z1.zip", None)
        job = self.run_job([newer], ["srv"])
        self.assertEqual(job.snapshot()["state"], "failed")      # zip 1 must be fetched again (now 404)

    def test_zip_path_escape_rejected(self):
        archive = make_zip({"../evil.txt": b"x", "ok.txt": b"y"})
        spec = self.spec("e.zip", archive, ".downloads/e.zip", extract_to="llama-cpp", unpacked=10)
        item = self.item("srv", spec, marker="llama-cpp/ok.txt")
        job = self.run_job([item], ["srv"])
        self.assertEqual(job.snapshot()["state"], "failed")
        self.assertIn("不安全", job.snapshot()["error"])
        self.assertFalse((self.models.parent / "evil.txt").exists())
        self.assertFalse((self.models / "evil.txt").exists())
        self.assertFalse((self.models / "llama-cpp/ok.txt").exists())
        self.assertFalse(md.status(self.models, [item])[0]["installed"])

    def test_plan_refuses_low_disk(self):
        data = b"q" * 1000
        item = self.item("a", self.spec("p.bin", data, "x/p.bin"))
        ok = md.plan(self.models, ["a"], [item], free_bytes=10**9, margin=1)
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["download_bytes"], 1000)
        low = md.plan(self.models, ["a"], [item], free_bytes=500, margin=1)
        self.assertFalse(low["ok"])
        self.assertIn("磁碟空間不足", low["message"])
        job = md.DownloadJob(self.models, ["a"], [item], free_bytes=500).start()
        self.assertEqual(job.snapshot()["state"], "failed")
        self.assertIn("磁碟空間不足", job.snapshot()["error"])
        self.assertFalse((self.models / "x/p.bin").exists())

    def test_plan_counts_extraction_space(self):
        archive = make_zip({"llama-server.exe": b"E"})
        item = self.item("srv", self.spec("w.zip", archive, ".downloads/w.zip", extract_to="llama-cpp", unpacked=5000),
                         marker="llama-cpp/llama-server.exe")
        result = md.plan(self.models, ["srv"], [item], free_bytes=10**9, margin=0)
        self.assertEqual(result["needed_bytes"], len(archive) + 5000)

    def test_catalog_complete(self):
        ids = [i.id for i in md.catalog()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(i.default for i in md.catalog()), 4)
        for item in md.catalog():
            self.assertTrue(item.label and item.revision, item.id)
            self.assertTrue(item.files, item.id)
            for f in item.files:
                self.assertTrue(f.url.startswith("https://"), f)
                self.assertGreater(f.size, 0, f)
                self.assertRegex(f.sha256, r"^[0-9a-f]{64}$")
                self.assertNotIn("..", f.dest)
                self.assertNotIn("TODO", f.url + f.dest)
                if f.extract_to:
                    self.assertGreater(f.unpacked, 0)
        layout = {f.dest for i in md.catalog() for f in i.files}
        self.assertIn("whisper-ja-1.5B/config.json", layout)
        self.assertIn("anime-whisper/model.safetensors", layout)
        self.assertIn("sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf", layout)


if __name__ == "__main__":
    unittest.main()
