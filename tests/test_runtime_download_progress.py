"""Download progress for uv sync: lock sizes, dry-run plan, zip index mapping, speed/ETA, tracker, sampler, installer hook.

No network and no real uv: ranged HTTP reads are served from an in-memory zip, and a Python script stands in for uv.
"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.runtime_env import download_progress as dp
from oh_streamer.runtime_env import installer, layout

PY = (3, 13)
PLAT = "win_amd64"

LOCK = r'''
version = 1
requires-python = "==3.13.*"

[[package]]
name = "numpy"
version = "2.2.6"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://x/numpy-2.2.6.tar.gz", size = 900 }
wheels = [
    { url = "https://x/numpy-2.2.6-cp313-cp313-win_amd64.whl", size = 100 },
    { url = "https://x/numpy-2.2.6-cp313-cp313t-win_amd64.whl", size = 101 },
    { url = "https://x/numpy-2.2.6-cp313-cp313-macosx_11_0_arm64.whl", size = 102 },
    { url = "https://x/numpy-2.2.6-cp312-cp312-win_amd64.whl", size = 103 },
]

[[package]]
name = "Charset_Normalizer"
version = "3.4.4"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://x/charset_normalizer-3.4.4-py3-none-any.whl", size = 50 },
    { url = "https://x/charset_normalizer-3.4.4-cp313-cp313-win_amd64.whl", size = 60 },
]

[[package]]
name = "cryptography"
version = "50.0.2"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://x/cryptography-50.0.2-cp39-abi3-win_amd64.whl", size = 39 },
    { url = "https://x/cryptography-50.0.2-cp311-abi3-win_amd64.whl", size = 311 },
    { url = "https://x/cryptography-50.0.2-cp314-abi3-win_amd64.whl", size = 314 },
]

[[package]]
name = "proxy-tools"
version = "0.1.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://x/proxy_tools-0.1.0.tar.gz", size = 7 }

[[package]]
name = "torch"
version = "2.8.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }
wheels = [
    { url = "https://r2/torch-2.8.0%2Bcu128-cp313-cp313-win_amd64.whl" },
]

[[package]]
name = "lada"
version = "0.11.1"
source = { path = "packaging/wheels/lada-0.11.1-py3-none-any.whl" }
wheels = [
    { filename = "lada-0.11.1-py3-none-any.whl" },
]

[[package]]
name = "maconly"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://x/maconly-1.0-cp313-cp313-macosx_11_0_arm64.whl", size = 5 },
]
'''


class WheelChoiceTests(unittest.TestCase):
    def test_rank_order(self):
        rank = lambda name: dp.wheel_rank(name, PY, PLAT)
        self.assertLess(rank("a-1-cp313-cp313-win_amd64.whl"), rank("a-1-cp311-abi3-win_amd64.whl"))
        self.assertLess(rank("a-1-cp311-abi3-win_amd64.whl"), rank("a-1-cp39-abi3-win_amd64.whl"))
        self.assertLess(rank("a-1-cp39-abi3-win_amd64.whl"), rank("a-1-py3-none-win_amd64.whl"))
        self.assertLess(rank("a-1-py3-none-win_amd64.whl"), rank("a-1-py3-none-any.whl"))
        for bad in ("a-1-cp313-cp313t-win_amd64.whl", "a-1-cp312-cp312-win_amd64.whl", "a-1-cp314-abi3-win_amd64.whl",
                    "a-1-cp313-cp313-macosx_11_0_arm64.whl", "a-1-py2-none-any.whl", "a-1.tar.gz", "short-1.whl"):
            self.assertIsNone(rank(bad), bad)
        self.assertIsNotNone(rank("a-1-py2.py3-none-any.whl"))

    def test_lock_artifacts(self):
        arts = dp.lock_artifacts(LOCK, PY, PLAT)
        self.assertEqual(arts[("numpy", "2.2.6")], dp.Artifact("https://x/numpy-2.2.6-cp313-cp313-win_amd64.whl", 100))
        self.assertEqual(arts[("charset-normalizer", "3.4.4")].size, 60)            # name normalized, platform wheel wins
        self.assertEqual(arts[("cryptography", "50.0.2")].size, 311)               # newest abi3 baseline this python allows
        self.assertEqual(arts[("proxy-tools", "0.1.0")].size, 7)                   # sdist only
        self.assertIsNone(arts[("torch", "2.8.0+cu128")].size)                     # pytorch index: no size in the lock
        self.assertEqual(arts[("lada", "0.11.1")], dp.Artifact(None, 0))           # local path source
        self.assertIsNone(arts[("maconly", "1.0")])                                # nothing usable on this platform

    def test_target_python_from_lock(self):
        self.assertEqual(dp.target_python({"requires-python": "==3.12.*"}), (3, 12))
        self.assertEqual(dp.target_python({}), sys.version_info[:2])


DRY_COLD = """Using CPython 3.13.16
DEBUG Identified uncached distribution: numpy==2.2.6
DEBUG Identified uncached distribution: charset-normalizer==3.4.4
DEBUG Identified uncached distribution: lada @ file:///C:/app/packaging/wheels/lada-0.11.1-py3-none-any.whl
DEBUG Identified uncached distribution: torch==2.8.0+cu128
Would download 4 packages
Would install 4 packages
 + numpy==2.2.6
"""


class DryRunTests(unittest.TestCase):
    def test_parse_cold(self):
        plan = dp.parse_dry_run(DRY_COLD)
        self.assertEqual(plan, [("numpy", "2.2.6"), ("charset-normalizer", "3.4.4"), ("lada", None), ("torch", "2.8.0+cu128")])

    def test_parse_warm(self):
        self.assertEqual(dp.parse_dry_run("Checked 93 packages in 4ms\nWould make no changes\n"), [])

    def test_count_mismatch_is_unknown(self):
        self.assertIsNone(dp.parse_dry_run(DRY_COLD.replace("Would download 4", "Would download 5")))
        self.assertIsNone(dp.parse_dry_run("DEBUG Identified uncached distribution: numpy==2.2.6\n"))

    def test_build_tracker(self):
        tracker = dp.build_tracker(DRY_COLD, LOCK, "cache")
        self.assertEqual(len(tracker), 3)                                           # lada is a local file
        self.assertTrue(dp.build_tracker("Would make no changes", LOCK, "cache").empty)
        self.assertIsNone(dp.build_tracker(None, LOCK, "cache"))
        unknown = "DEBUG Identified uncached distribution: nope==1\nWould download 1 package\n"
        self.assertIsNone(dp.build_tracker(unknown, LOCK, "cache"))
        self.assertIsNone(dp.build_tracker(DRY_COLD, "not = [toml", "cache"))

    def test_run_dry_run_appends_flags_and_handles_failure(self):
        out = dp.run_dry_run([sys.executable, "-c", "import sys; print(' '.join(sys.argv[1:]))", "sync"], ".", dict(os.environ), None)
        self.assertEqual(out.strip(), "sync --dry-run -v")
        self.assertIsNone(dp.run_dry_run([sys.executable, "-c", "raise SystemExit(2)"], ".", dict(os.environ), None))
        self.assertIsNone(dp.run_dry_run([str(Path(tempfile.gettempdir()) / "no-such-uv.exe")], ".", dict(os.environ), None))

    def test_run_dry_run_cancel_and_timeout(self):
        sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
        event = threading.Event()
        event.set()
        t0 = time.monotonic()
        self.assertIsNone(dp.run_dry_run(sleeper, ".", dict(os.environ), event))
        self.assertIsNone(dp.run_dry_run(sleeper, ".", dict(os.environ), None, timeout_s=0.3))
        self.assertLess(time.monotonic() - t0, 10)


def make_wheel(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


ENTRIES = [("pkg/a.bin", os.urandom(4000)), ("pkg/sub/", b""), ("pkg/sub/b.txt", b"x" * 20000),
           ("pkg/c.bin", os.urandom(3000)), ("pkg-1.0.dist-info/RECORD", b"r" * 100)]


def ranged(blob: bytes):
    def fake_range(url, spec, timeout):
        if spec.startswith("-"):
            body = blob[-int(spec[1:]):]
        else:
            a, b = (int(x) for x in spec.split("-"))
            body = blob[a:b + 1]
        return body, len(blob)
    return fake_range


class ZipIndexTests(unittest.TestCase):
    def setUp(self):
        self.blob = make_wheel(ENTRIES)
        with mock.patch.object(dp, "_range", ranged(self.blob)):
            self.index = dp.fetch_zip_index("https://x/pkg-1.0-py3-none-any.whl")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_index_matches_zipfile(self):
        infos = [i for i in zipfile.ZipFile(io.BytesIO(self.blob)).infolist() if not i.is_dir()]
        self.assertEqual(self.index.size, len(self.blob))
        self.assertEqual(self.index.names, [i.filename for i in infos])
        self.assertEqual(self.index.csizes, [i.compress_size for i in infos])
        self.assertEqual(self.index.usizes, [i.file_size for i in infos])
        self.assertEqual(self.index.offsets, [i.header_offset for i in infos])

    def test_central_directory_outside_the_tail_is_fetched(self):
        calls = []
        inner = ranged(self.blob)

        def spy(url, spec, timeout):
            calls.append(spec)
            return inner(url, spec, timeout)
        with mock.patch.object(dp, "_EOCD_TAIL", 40), mock.patch.object(dp, "_range", spy):
            index = dp.fetch_zip_index("u")
        self.assertEqual(len(calls), 2)
        self.assertEqual(index.names, self.index.names)

    def test_not_a_zip(self):
        with mock.patch.object(dp, "_range", ranged(b"hello" * 100)):
            with self.assertRaises(ValueError):
                dp.fetch_zip_index("u")

    def write(self, name, data):
        path = self.root.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_position_follows_unpacked_files(self):
        idx = self.index
        self.assertIsNone(idx.position(self.root))                                  # nothing unpacked here
        self.write("pkg/a.bin", ENTRIES[0][1])
        self.assertEqual(idx.position(self.root), idx.offsets[0] + idx.csizes[0])
        self.write("pkg/sub/b.txt", ENTRIES[2][1][:10000])                           # half written
        self.assertEqual(idx.position(self.root), idx.offsets[1] + idx.csizes[1] // 2)
        self.write("pkg/sub/b.txt", ENTRIES[2][1])
        self.write("pkg/c.bin", ENTRIES[3][1])
        self.write("pkg-1.0.dist-info/RECORD", ENTRIES[4][1])
        self.assertEqual(idx.position(self.root), idx.offsets[-1] + idx.csizes[-1])
        self.assertLessEqual(idx.position(self.root), idx.size)


class RateTests(unittest.TestCase):
    def test_speed_needs_a_span_and_uses_the_window(self):
        meter = dp.RateMeter(window_s=10, min_span_s=2)
        meter.add(0, 0)
        self.assertIsNone(meter.speed())
        meter.add(1, 100)
        self.assertIsNone(meter.speed())                                            # 1 s is too short to trust
        meter.add(2, 200)
        self.assertEqual(meter.speed(), 100)
        for t in range(3, 30):                                                      # steady 100 B/s, then a burst
            meter.add(t, 100 * t)
        meter.add(30, 100 * 30 + 1000)
        self.assertAlmostEqual(meter.speed(), (4000 - 2000) / 10, delta=1)          # burst spread over the window

    def test_speed_never_negative(self):
        meter = dp.RateMeter(min_span_s=0)
        meter.add(0, 100)
        meter.add(5, 50)
        self.assertEqual(meter.speed(), 0)

    def test_eta(self):
        self.assertEqual(dp.eta_seconds(10_000_000, 1_000_000.0), 10)
        self.assertEqual(dp.eta_seconds(0, None), 0)
        self.assertIsNone(dp.eta_seconds(1000, None))
        self.assertIsNone(dp.eta_seconds(1000, 0.0))
        self.assertIsNone(dp.eta_seconds(1000, 10.0))                               # below 1 KiB/s: not worth a number
        self.assertEqual(dp.eta_seconds(150 * 2**20, 1.0 * 2**20), 150)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TrackerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name)
        self.clock = Clock()

    def tracker(self, items, **kw):
        return dp.DownloadTracker(items, self.cache, clock=self.clock, big_bytes=10_000, **kw)

    def test_small_packages_count_when_done(self):
        tr = self.tracker({"a": (3000, "u/a.whl"), "b": (1000, "u/b.whl")})
        self.assertEqual(tr.snapshot()["total_bytes"], 4000)
        self.assertEqual(tr.snapshot()["downloaded_bytes"], 0)
        tr.on_line("Downloading a (2.9KiB)")
        tr.on_line(" Downloaded a")
        self.assertEqual(tr.snapshot()["downloaded_bytes"], 3000)
        tr.on_line("Prepared 2 packages in 1.2s")                                   # b was too small to be announced
        snap = tr.snapshot()
        self.assertEqual(snap["downloaded_bytes"], 4000)
        self.assertEqual(snap["eta_s"], 0)
        self.assertIsNone(snap["speed_bps"])

    def test_small_packages_only_give_no_speed_or_eta(self):
        """Whole-package jumps make speed read 0 and the ETA swing (base phase): show the amount alone."""
        tr = self.tracker({"a": (3000, "u/a.whl"), "b": (1000, "u/b.whl")})
        tr.snapshot()
        self.clock.t = 5
        tr.on_line(" Downloaded a")
        snap = tr.snapshot()
        self.assertEqual(snap["downloaded_bytes"], 3000)
        self.assertIsNone(snap["speed_bps"])
        self.assertIsNone(snap["eta_s"])

    def test_warm_unsized_wheel_never_counts(self):
        """The dry run calls pytorch wheels uncached even when uv's HTTP cache has them: no fake bar."""
        tr = self.tracker({"torch": (None, "u/torch.whl")})
        self.assertIsNone(tr.snapshot())
        tr.on_line("Prepared 3 packages in 1ms")
        self.assertIsNone(tr.snapshot())

    def test_unsized_wheel_counts_once_announced(self):
        tr = self.tracker({"torchvision": (None, "u/tv.whl"), "numpy": (500, "u/np.whl")})
        self.assertEqual(tr.snapshot()["total_bytes"], 500)
        tr.on_line("Downloading torchvision (7.2MiB)")
        self.assertEqual(tr.snapshot()["total_bytes"], 500 + int(7.2 * 2**20))

    def big_tracker(self):
        blob = make_wheel(ENTRIES + [("pkg/z.bin", os.urandom(30000))])
        with mock.patch.object(dp, "_range", ranged(blob)):
            index = dp.fetch_zip_index("u")
        tr = self.tracker({"big": (None, "u/big-1-py3-none-any.whl")})
        tr.on_line("Downloading big (40KiB)")
        tr.fetch_pending(fetch_index=lambda url: index)
        return tr, index

    def test_big_wheel_measured_while_unpacking(self):
        tr, index = self.big_tracker()
        snap = tr.snapshot()
        self.assertEqual(snap["total_bytes"], index.size)                           # exact size replaces "40KiB"
        self.assertEqual(snap["downloaded_bytes"], 0)
        tmp = self.cache / ".tmpAbC123"
        (tmp / "pkg").mkdir(parents=True)
        (tmp / "pkg" / "a.bin").write_bytes(ENTRIES[0][1])
        self.clock.t = 1
        first = tr.snapshot()["downloaded_bytes"]
        self.assertEqual(first, index.offsets[0] + index.csizes[0])
        (tmp / "pkg" / "sub").mkdir()
        (tmp / "pkg" / "sub" / "b.txt").write_bytes(ENTRIES[2][1])
        self.clock.t = 3
        snap = tr.snapshot()
        self.assertGreater(snap["downloaded_bytes"], first)
        self.assertIsNotNone(snap["speed_bps"])
        self.assertIsNotNone(snap["eta_s"])
        # uv renames the dir into archive-v0 when done: the figure must not fall back
        tmp.rename(self.cache / "archive")
        self.clock.t = 4
        self.assertEqual(tr.snapshot()["downloaded_bytes"], snap["downloaded_bytes"])
        tr.on_line(" Downloaded big")
        self.assertEqual(tr.snapshot()["downloaded_bytes"], index.size)

    def test_big_wheel_we_cannot_see_turns_unknown(self):
        tr = self.tracker({"big": (50_000, "u/big-1-py3-none-any.whl")}, untracked_grace_s=5)
        tr.fetch_pending(fetch_index=mock.Mock(side_effect=OSError("offline")))
        tr.on_line("Downloading big (48.8KiB)")
        self.assertEqual(tr.snapshot()["downloaded_bytes"], 0)                      # still inside the grace period
        self.clock.t = 6
        self.assertIsNone(tr.snapshot())
        tr.on_line(" Downloaded big")
        self.assertEqual(tr.snapshot()["downloaded_bytes"], 50_000)

    def test_index_fetched_once(self):
        fetch = mock.Mock(side_effect=OSError("offline"))
        tr = self.tracker({"big": (50_000, "u/big.whl"), "small": (10, "u/s.whl"), "sdist": (60_000, "u/s.tar.gz")})
        tr.fetch_pending(fetch_index=fetch)
        tr.fetch_pending(fetch_index=fetch)
        fetch.assert_called_once_with("u/big.whl")

    def test_unrelated_lines_ignored(self):
        tr = self.tracker({"a": (10, "u/a.whl")})
        for line in ("Downloading other (1.0MiB)", " Downloaded other", " + a==1", "Downloading a (weird)", ""):
            tr.on_line(line)
        self.assertEqual(tr.snapshot()["downloaded_bytes"], 0)


class SamplerTests(unittest.TestCase):
    def test_reports_until_stopped_and_never_after(self):
        tr = dp.DownloadTracker({"a": (100, "u/a.whl")}, ".")
        got = []
        sampler = dp.ProgressSampler(tr, got.append, interval_s=0.02)
        sampler.start()
        deadline = time.monotonic() + 5
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()
        count = len(got)
        time.sleep(0.1)
        self.assertGreater(count, 0)
        self.assertEqual(len(got), count)
        self.assertEqual(got[-1]["total_bytes"], 100)
        sampler.stop()                                                              # second stop is harmless

    def test_stop_before_start_and_broken_callback(self):
        tr = dp.DownloadTracker({"a": (100, "u/a.whl")}, ".")
        dp.ProgressSampler(tr, lambda _: None).stop()
        sampler = dp.ProgressSampler(tr, mock.Mock(side_effect=RuntimeError("ui")), interval_s=0.01)
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        self.assertFalse(sampler._thread.is_alive())


STUB_UV = r'''
import os, sys, time
def emit(t):
    sys.stdout.write(t + "\n"); sys.stdout.flush()
if "--dry-run" in sys.argv:
    emit(os.environ.get("STUB_DRY", "").replace("|", "\n"))
    sys.exit(0)
for line in os.environ.get("STUB_OUT", "").split("|"):
    if line.startswith("sleep "):
        time.sleep(float(line[6:]))
    elif line:
        emit(line)
time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
'''

STUB_LOCK = '''version = 1
requires-python = "==3.13.*"

[[package]]
name = "numpy"
version = "2.2.6"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://x/numpy-2.2.6-cp313-cp313-win_amd64.whl", size = 1000 }]

[[package]]
name = "six"
version = "1.17.0"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://x/six-1.17.0-py2.py3-none-any.whl", size = 24 }]
'''


class InstallerHookTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.root = self.tmp / "install"
        self.app = layout.app_dir(self.root)
        self.app.mkdir(parents=True)
        (self.app / "uv.lock").write_text(STUB_LOCK, encoding="utf-8")
        stub = self.tmp / "stub_uv.py"
        stub.write_text(STUB_UV, encoding="utf-8")
        self.uv_cmd = [sys.executable, str(stub)]

    def run_sync(self, dry, out, **kw):
        env = {"STUB_DRY": dry, "STUB_OUT": out}
        with mock.patch.dict(os.environ, env), mock.patch.object(dp, "SAMPLE_S", 0.05):
            return installer.run_uv_sync(self.uv_cmd, self.app, self.tmp / "env", self.root, ["sync"], phase="base", **kw)

    COLD = ("DEBUG Identified uncached distribution: numpy==2.2.6|DEBUG Identified uncached distribution: six==1.17.0|"
            "Would download 2 packages")

    def test_cold_sync_reports_progress_and_stops(self):
        got = []
        result = self.run_sync(self.COLD, "Downloading numpy (1000B)|sleep 0.3| Downloaded numpy|sleep 0.2|"
                                          "Prepared 2 packages in 0.5s|Installed 2 packages", on_progress=got.append)
        self.assertTrue(result.ok)
        reported = [g for g in got if g]
        self.assertTrue(reported)
        self.assertEqual(reported[-1]["downloaded_bytes"], 1024)
        self.assertEqual(reported[-1]["total_bytes"], 1024)
        self.assertTrue(all(g["downloaded_bytes"] <= g["total_bytes"] for g in reported))
        count = len(got)
        time.sleep(0.2)
        self.assertEqual(len(got), count)                                           # nothing after return
        log = result.log_path.read_text(encoding="utf-8")
        self.assertIn("下載進度", log)
        self.assertNotIn("Identified uncached", log)                                # the dry run is not in the uv log

    def test_warm_cache_reports_nothing(self):
        got = []
        result = self.run_sync("Checked 2 packages in 4ms|Would make no changes", "Checked 2 packages in 4ms",
                               on_progress=got.append)
        self.assertTrue(result.ok)
        self.assertEqual(got, [])

    def test_unreadable_plan_reports_nothing(self):
        got = []
        result = self.run_sync("Would download 7 packages", "Prepared 7 packages", on_progress=got.append)
        self.assertTrue(result.ok)
        self.assertEqual(got, [])

    def test_without_callback_no_dry_run(self):
        result = self.run_sync("BOOM", "Checked 2 packages")
        self.assertTrue(result.ok)
        self.assertNotIn("下載進度", result.log_path.read_text(encoding="utf-8"))

    def test_cancel_stops_sampler(self):
        got = []
        event = threading.Event()
        timer = threading.Timer(0.4, event.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with mock.patch.dict(os.environ, {"STUB_SLEEP": "30"}):
            t0 = time.monotonic()
            result = self.run_sync(self.COLD, "Downloading numpy (1000B)", on_progress=got.append, cancel_event=event)
        self.assertTrue(result.cancelled)
        self.assertLess(time.monotonic() - t0, 15)
        count = len(got)
        time.sleep(0.2)
        self.assertEqual(len(got), count)
        self.assertFalse(any(t.name == "uv-download-progress" and t.is_alive() for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
