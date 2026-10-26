"""Offline tests for queue_thumbs.py (film-strip thumbnails): ffmpeg / ffprobe are mocked, no image is ever decoded.
Run from streamer/: python -m unittest tests.test_queue_thumbs."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from oh_streamer.queue import queue_store
from oh_streamer.queue import queue_thumbs as QT


def tiny_jpeg(w: int = 320, h: int = 180) -> bytes:
    """Header-only JPEG (SOI + SOF0 + EOI): enough for jpeg_size(); never decodable, never viewed."""
    return (b"\xff\xd8\xff\xc0\x00\x11\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big")
            + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01" + b"\xff\xd9")


class FakeRunner:
    """Stands in for ffmpeg: records the command and writes a header-only JPEG to its last argument."""

    def __init__(self, fail_when=None):
        self.cmds: list[list[str]] = []
        self.fail_when = fail_when

    def __call__(self, cmd, timeout, register):
        self.cmds.append(list(cmd))
        if self.fail_when and self.fail_when(cmd):
            return 1, "boom"
        Path(cmd[-1]).write_bytes(tiny_jpeg())
        return 0, ""


def manifest_doc(done=(0, 1, 2), n=120, fps="30000/1001", frames=120, partial=(), key=None, total=None, last=None):
    return {"version": 2, "key": key or {"source": "x", "fps": fps, "n": n, "enc": "hevc"}, "n": n,
            "segments": [{"k": k, "file": f"seg_{k:06d}.mkv", "frames": frames, **({"partial": True} if k in partial else {})}
                         for k in done],
            "last_k": last, "complete": False, "total_frames": total,
            "play": {"fps": fps, "n": n, "video_start": 0.0, "rate_mult": 2, "layers": ["processed"], "count": 10}}


class PureHelpersTest(unittest.TestCase):
    def test_frame_time_even_spacing(self):
        self.assertAlmostEqual(QT.frame_time(0, 120.0), 5.0)
        self.assertAlmostEqual(QT.frame_time(11, 120.0), 115.0)
        times = [QT.frame_time(i, 600.0) for i in range(QT.FRAME_COUNT)]
        self.assertEqual(times, sorted(times))
        self.assertTrue(0 < times[0] and times[-1] < 600.0)

    def test_segment_for_time_ntsc(self):
        fps = Fraction(30000, 1001)
        self.assertEqual(QT.segment_for_time(0.0, fps, 120)[0], 0)
        k, off = QT.segment_for_time(4.0, fps, 120)              # source frame 119 -> still segment 0
        self.assertEqual(k, 0)
        self.assertAlmostEqual(off, 4.0)
        k, off = QT.segment_for_time(4.2, fps, 120)              # frame 125 -> segment 1, starts at 120/fps = 4.004 s
        self.assertEqual(k, 1)
        self.assertAlmostEqual(off, 4.2 - 120 / float(fps), places=6)
        k, off = QT.segment_for_time(100.0, fps, 120)
        self.assertEqual(k, int(100.0 * 29.97002997) // 120)
        self.assertTrue(0 <= off < 120 / float(fps))

    def test_segment_for_time_integer_fps_boundary(self):
        self.assertEqual(QT.segment_for_time(8.0, 25, 100), (2, 0.0))
        k, off = QT.segment_for_time(7.999, 25, 100)
        self.assertEqual(k, 1)
        self.assertAlmostEqual(off, 3.999, places=6)

    def test_processed_track_is_first_video_stream(self):
        cmd = QT.grab_cmd("seg.mkv", 1.25, "out.jpg", processed_track=True)
        self.assertEqual(QT.processed_track_args(), ["-map", "0:v:0"])
        i = cmd.index("-map")
        self.assertEqual(cmd[i:i + 2], ["-map", "0:v:0"])
        self.assertGreater(i, cmd.index("-i"))                   # an OUTPUT option: after the input
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))        # input seeking
        self.assertEqual(cmd[cmd.index("-ss") + 1], "1.250")
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "1")
        self.assertIn("scale=320:-2", cmd)

    def test_source_grab_has_no_track_map_and_remote_options(self):
        cmd = QT.grab_cmd("https://h/x.mp4", 3.0, "o.jpg", http_args=("-headers", "A: b\r\n"), remote=True)
        self.assertNotIn("-map", cmd)
        self.assertLess(cmd.index("-headers"), cmd.index("-i"))
        self.assertIn("-rw_timeout", cmd)
        self.assertIn("-allowed_extensions", QT.grab_cmd("a.m3u8", 0, "o.jpg", playlist=True))
        self.assertNotIn("-allowed_extensions", cmd)

    def test_merge_ranges(self):
        self.assertEqual(QT.merge_ranges([(0.5, 0.6), (0.0, 0.2), (0.2, 0.3)]), [[0.0, 0.3], [0.5, 0.6]])
        self.assertEqual(QT.merge_ranges([(0.0, 0.5), (0.1, 0.2)]), [[0.0, 0.5]])
        self.assertEqual(QT.merge_ranges([(0.3, 0.3), (0.9, 0.1), (-1, 2)]), [[0.0, 1.0]])
        self.assertEqual(QT.merge_ranges([]), [])

    def test_segment_ranges_from_manifest_with_gaps_and_partial(self):
        info = QT.parse_manifest(manifest_doc(done=(0, 1, 2, 5), partial=(1,)))
        self.assertEqual(info.n, 120)
        self.assertEqual(info.fps, Fraction(30000, 1001))
        self.assertTrue(info.segments[1][2])                      # partial flag kept ...
        r = QT.segment_ranges({k: v[1] for k, v in info.segments.items()}, info.n, 1200)   # ... and it still counts
        self.assertEqual(r, [[0.0, 0.3], [0.5, 0.6]])
        self.assertEqual(QT.segment_ranges({0: 120}, 120, 0), [])
        # the last segment is shorter than n
        self.assertEqual(QT.segment_ranges({9: 60}, 120, 1140), [[0.94737, 1.0]])

    def test_parse_manifest_rejects_garbage(self):
        self.assertIsNone(QT.parse_manifest([]))
        self.assertIsNone(QT.parse_manifest({"segments": []}))
        v1 = {"version": 1, "key": {"fps": "25", "n": 100}, "n": 100, "segments": [{"k": 0, "file": "seg_000000.mkv", "frames": 100}]}
        self.assertEqual(QT.parse_manifest(v1).fps, Fraction(25))

    def test_download_fraction_rules(self):
        def item(**kw):
            base = dict(state="downloading", kind="url", target="https://example.com/v", source_path="", source_status="",
                        download_progress={})
            base.update(kw)
            return base
        self.assertEqual(QT.download_fraction(item(kind="file", target="C:/v.mp4", state="waiting")), 1.0)
        self.assertEqual(QT.download_fraction(item(state="done")), 1.0)
        self.assertEqual(QT.download_fraction(item(state="queued", source_path="C:/s.mp4")), 1.0)
        self.assertEqual(QT.download_fraction(item(state="queued", source_path="p.m3u8", source_status="complete")), 1.0)
        self.assertEqual(QT.download_fraction(item(download_progress={"fragment_index": 3, "fragment_count": 12,
                                                                      "downloaded_bytes": 1, "total_bytes": 2})), 0.25)
        self.assertEqual(QT.download_fraction(item(download_progress={"downloaded_bytes": 25, "total_bytes": 100,
                                                                      "fragment_index": None, "fragment_count": None})), 0.25)
        self.assertEqual(QT.download_fraction(item(download_progress={"downloaded_bytes": 300, "total_bytes": 100})), 1.0)
        self.assertEqual(QT.download_fraction(item(state="queued", source_path="p.m3u8", source_status="downloading",
                                                   download_progress={"ready_segments": 5, "segment_count": 20})), 0.25)
        self.assertIsNone(QT.download_fraction(item(state="waiting")))
        self.assertIsNone(QT.download_fraction(item(download_progress={"downloaded_bytes": 10, "total_bytes": None})))

    def test_scan_dir_and_payload(self):
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td) / "thumbs"
            tdir.mkdir()
            (tdir / "meta.json").write_text(json.dumps({"duration": 120.0, "rkey": "abc"}), encoding="utf-8")
            (tdir / "f_00_s.jpg").write_bytes(tiny_jpeg(320, 180))
            (tdir / "f_00_r.jpg").write_bytes(tiny_jpeg(320, 180))
            (tdir / "f_03_s.jpg").write_bytes(tiny_jpeg(320, 240))
            (tdir / "f_05_s.jpg").write_bytes(b"")                 # empty -> ignored
            (tdir / "f_06_s.jpg").write_bytes(b"not a jpeg")       # unreadable -> ignored
            (tdir / "f_12_s.jpg").write_bytes(tiny_jpeg())         # slot out of range -> ignored
            (tdir / "f_01_s.tmp.jpg").write_bytes(tiny_jpeg())     # temp file -> ignored
            meta, frames = QT.scan_thumbs_dir(tdir)
            self.assertEqual(meta, {"duration": 120.0, "rkey": "abc"})
            self.assertEqual(sorted(frames), [(0, "r"), (0, "s"), (3, "s")])
            self.assertEqual(frames[(3, "s")][1:3], (320, 240))
            payload = QT.build_thumbs_payload(meta["duration"], frames)
            self.assertEqual(payload["duration"], 120.0)
            self.assertEqual([f["at"] for f in payload["frames"]], [5.0, 35.0])
            f0, f3 = payload["frames"]
            self.assertTrue(f0["restored"] and f0["url"].startswith("file:///") and "f_00_r.jpg?v=" in f0["url"])
            self.assertIn("f_00_s.jpg?v=", f0["src_url"])
            self.assertFalse(f3["restored"])
            self.assertEqual((f3["w"], f3["h"]), (320, 240))
            self.assertNotIn("src_url", f3)
            int(f0["url"].split("?v=")[1])
            self.assertIsNone(QT.build_thumbs_payload(None, frames))
            self.assertIsNone(QT.build_thumbs_payload(120.0, {}))
            meta, frames = QT.scan_thumbs_dir(Path(td) / "missing")
            self.assertEqual((meta, frames), ({"duration": None, "rkey": None}, {}))


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.store = queue_store.QueueStore(self.root / "q")
        self.runner = FakeRunner()
        self.probed: list = []
        self.enabled = [True]
        self.changes = [0]
        self.mgr = QT.ThumbManager(
            self.store, log=lambda m: None, on_change=lambda: self.changes.__setitem__(0, self.changes[0] + 1),
            enabled=lambda: self.enabled[0], runner=self.runner,
            probe=lambda src, http=(), playlist=False: (self.probed.append(src) or 120.0))
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"x")

    def add_file_item(self, state="queued"):
        it = self.store.add(str(self.video), "file", {}, None, {})
        return self.store.update(it.id, state=state, source_path=str(self.video))

    def run_all(self, limit=80):
        n = 0
        while n < limit:
            job = self.mgr._next_job()
            if job is None:
                break
            self.mgr._execute(job)
            n += 1
        return n

    def test_source_frames_grabbed_for_local_item(self):
        it = self.add_file_item()
        self.run_all()
        self.assertEqual(self.probed, [str(self.video)])
        grabs = [c for c in self.runner.cmds if "-ss" in c]
        self.assertEqual(len(grabs), QT.FRAME_COUNT)
        seeks = [float(c[c.index("-ss") + 1]) for c in grabs]
        for i, s in enumerate(seeks):
            self.assertAlmostEqual(s, (i + 0.5) / 12 * 120.0, places=3)
        self.assertTrue(all("-map" not in c for c in grabs))
        thumbs = self.mgr.thumbs_for(it)
        self.assertEqual(len(thumbs["frames"]), QT.FRAME_COUNT)
        self.assertEqual(thumbs["duration"], 120.0)
        self.assertFalse(any(f["restored"] for f in thumbs["frames"]))
        self.assertTrue((self.store.item_dir(it.id) / "thumbs" / "f_00_s.jpg").is_file())
        self.assertTrue(self.changes[0] > 0)
        # a fresh manager derives the same state from the files alone
        mgr2 = QT.ThumbManager(self.store, log=lambda m: None, runner=self.runner)
        self.assertEqual(mgr2.thumbs_for(it)["frames"][3]["at"], thumbs["frames"][3]["at"])
        self.assertIsNone(mgr2._next_job())                       # nothing left to do

    def test_disabled_setting_hides_and_stops(self):
        it = self.add_file_item()
        self.run_all()
        self.enabled[0] = False
        self.assertIsNone(self.mgr.thumbs_for(it))

    def test_failed_source_grab_backs_off_and_retries_from_local_file(self):
        it = self.store.add("https://cdn.example.com/v.mp4", "url", {"Referer": "https://x/"}, None, {})
        self.store.update(it.id, state="downloading")
        self.runner.fail_when = lambda c: True
        self.mgr._execute(self.mgr._next_job())                    # probe of the direct URL
        job = self.mgr._next_job()
        self.assertEqual(job.srckind, "remote")
        self.mgr._execute(job)
        self.assertIn("-headers", self.runner.cmds[-1])
        self.assertEqual(self.mgr._fails[job.fail_key][0], 1)
        nxt = self.mgr._next_job()
        self.assertTrue(nxt is None or nxt.i != job.i or nxt.fail_key != job.fail_key)   # the failed frame backs off
        # the download finished: the local file takes over (its fail keys are separate)
        self.store.update(it.id, state="queued", source_path=str(self.video), source_status="")
        self.runner.fail_when = None
        job = self.mgr._next_job()
        self.assertEqual((job.srckind, job.i, job.out), ("local", 0, "s"))
        self.assertEqual(job.src, str(self.video))

    def test_restored_frames_from_finished_segments(self):
        it = self.add_file_item(state="processing")
        work = self.store.item_dir(it.id) / "work"
        (work / "segments").mkdir(parents=True)
        # fps 25, n = 100 (4 s); video 120 s -> 30 segments. frame 0 at 5 s -> segment 1, offset 1 s
        doc = manifest_doc(done=(1, 2), n=100, fps="25", frames=100, partial=(2,), total=3000)
        (work / "segments.json").write_text(json.dumps(doc), encoding="utf-8")
        for k in (1, 2):
            (work / "segments" / f"seg_{k:06d}.mkv").write_bytes(b"x")
        self.run_all()
        restored = [c for c in self.runner.cmds if "-map" in c]
        for c in restored:
            self.assertEqual(c[c.index("-map") + 1], "0:v:0")
            self.assertTrue(c[c.index("-i") + 1].endswith(("seg_000001.mkv", "seg_000002.mkv")))
        # slot 0 (t = 5 s) is in segment 1 (4..8 s) at offset 1.0; slot 1 (15 s) is segment 3: not finished
        c0 = restored[0]
        self.assertEqual(c0[c0.index("-ss") + 1], "1.000")
        thumbs = self.mgr.thumbs_for(it)
        flags = [f["restored"] for f in thumbs["frames"]]
        self.assertTrue(flags[0])
        self.assertEqual(sum(flags), len(restored))
        self.assertEqual(len(thumbs["frames"]), QT.FRAME_COUNT)
        # source frames come after the restored ones for the processing item, restored frames carry src_url
        self.assertTrue(self.runner.cmds[0][-1].endswith("f_00_r.tmp.jpg"))
        self.assertIn("src_url", thumbs["frames"][0])
        self.assertEqual(self.mgr.processed_ranges(self.store.get(it.id)), [[0.03333, 0.1]])
        # worker position: segment 7 of 30 (200-frame total check by k * n / total)
        self.store.update(it.id, progress={"stage": "processing", "frame": 900, "frames": 3000, "k": 7, "phase": "restore"})
        self.assertAlmostEqual(self.mgr.worker_fraction(self.store.get(it.id)), 700 / 3000, places=4)
        self.store.update(it.id, progress={"stage": "processing", "frame": 900, "frames": 3000, "k": None, "phase": "restore"})
        self.assertAlmostEqual(self.mgr.worker_fraction(self.store.get(it.id)), 0.3, places=4)
        self.store.update(it.id, progress={"stage": "processing", "frame": 3000, "frames": 3000, "phase": "finalize"})
        self.assertIsNone(self.mgr.worker_fraction(self.store.get(it.id)))
        self.store.update(it.id, state="queued")
        self.assertIsNone(self.mgr.worker_fraction(self.store.get(it.id)))

    def test_new_manifest_key_or_drop_restored_falls_back_to_source_frames(self):
        it = self.add_file_item(state="processing")
        work = self.store.item_dir(it.id) / "work"
        (work / "segments").mkdir(parents=True)
        (work / "segments.json").write_text(json.dumps(manifest_doc(done=(1,), n=100, fps="25", frames=100, total=3000)),
                                            encoding="utf-8")
        (work / "segments" / "seg_000001.mkv").write_bytes(b"x")
        self.run_all()
        self.assertTrue(self.mgr.thumbs_for(it)["frames"][0]["restored"])
        meta = json.loads((self.store.item_dir(it.id) / "thumbs" / "meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["rkey"])
        # the worker restarted under another job key (empty manifest, new key): restored frames are dropped
        newdoc = manifest_doc(done=(), n=100, fps="25", key={"source": "x", "fps": "25", "n": 100, "enc": "av1"})
        (work / "segments.json").write_text(json.dumps(newdoc), encoding="utf-8")
        self.assertIsNone(self.mgr._next_job())
        thumbs = self.mgr.thumbs_for(it)
        self.assertFalse(any(f["restored"] for f in thumbs["frames"]))
        self.assertFalse(list((self.store.item_dir(it.id) / "thumbs").glob("f_*_r.jpg")))
        # explicit drop (apply-model path)
        (work / "segments.json").write_text(json.dumps(manifest_doc(done=(1,), n=100, fps="25", frames=100, total=3000,
                                                                    key={"source": "x", "fps": "25", "n": 100, "enc": "av1"})),
                                            encoding="utf-8")
        self.run_all()
        self.assertTrue(self.mgr.thumbs_for(it)["frames"][0]["restored"])
        self.mgr.drop_restored(it.id)
        self.assertFalse(self.mgr.thumbs_for(it)["frames"][0]["restored"])

    def test_done_item_backfill_from_output_uses_processed_track(self):
        it = self.store.add("https://example.com/page", "url", {}, None, {})
        out = self.store.item_dir(it.id) / "output.mkv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        it = self.store.update(it.id, state="done", output_path=str(out))
        self.run_all()
        self.assertEqual(self.probed, [str(out)])                  # duration from the output when no source is left
        grabs = [c for c in self.runner.cmds if "-ss" in c]
        self.assertEqual(len(grabs), QT.FRAME_COUNT)
        for i, c in enumerate(grabs):
            self.assertEqual(c[c.index("-map") + 1], "0:v:0")
            self.assertAlmostEqual(float(c[c.index("-ss") + 1]), (i + 0.5) / 12 * 120.0, places=3)
        self.assertTrue(all(f["restored"] for f in self.mgr.thumbs_for(it)["frames"]))
        self.assertEqual(self.mgr.processed_ranges(self.store.get(it.id)), [[0.0, 1.0]])

    def test_forget_blocks_new_grabs(self):
        it = self.add_file_item()
        self.mgr.forget(it.id)
        self.assertIsNone(self.mgr._next_job())
        self.assertEqual(self.runner.cmds, [])

    def test_hold_pauses_grabs_for_that_item_only_and_resumes(self):
        a = self.add_file_item()
        video2 = self.root / "clip2.mp4"
        video2.write_bytes(b"y")
        b = self.store.update(self.store.add(str(video2), "file", {}, None, {}).id, state="queued", source_path=str(video2))
        self.assertNotEqual(a.id, b.id)
        with self.mgr.hold(a.id):
            seen = set()
            for _ in range(40):
                job = self.mgr._next_job()
                if job is None:
                    break
                seen.add(job.item_id)
                self.mgr._execute(job)
            self.assertEqual(seen, {b.id})
            with self.mgr.hold(a.id):                              # re-entrant
                pass
            self.assertIn(a.id, self.mgr._held)
            self.assertFalse(self.mgr._begin(QT._Job("probe", a.id, src="x", srckind="local", fail_key=(a.id, -1, "p"))))
        self.assertNotIn(a.id, self.mgr._held)
        self.assertEqual(self.mgr._next_job().item_id, a.id)

    def test_hold_kills_the_grab_in_flight_without_counting_a_failure(self):
        import threading
        it = self.add_file_item()
        self.mgr._execute(self.mgr._next_job())                    # probe
        job = self.mgr._next_job()
        started, killed = threading.Event(), threading.Event()

        class Proc:
            def kill(self_inner):
                killed.set()

        def blocking_runner(cmd, timeout, register):
            register(Proc())
            started.set()
            killed.wait(5)
            register(None)
            return -9, "killed"

        self.mgr._runner = blocking_runner
        t = threading.Thread(target=self.mgr._execute, args=(job,))
        t.start()
        self.assertTrue(started.wait(5))
        with self.mgr.hold(it.id):
            self.assertTrue(killed.is_set())
            t.join(5)
            self.assertFalse(t.is_alive())
            self.assertIsNone(self.mgr._inflight)
            self.assertNotIn(job.fail_key, self.mgr._fails)
            self.assertIsNone(self.mgr._next_job())
        self.assertEqual(self.mgr._next_job().fail_key, job.fail_key)   # same frame is retried after the hold

    def test_removed_item_folder_is_not_recreated(self):
        it = self.add_file_item()
        job = self.mgr._next_job()                                 # probe
        self.mgr._execute(job)
        job = self.mgr._next_job()                                 # first grab
        self.store.remove(it.id)
        self.mgr._execute(job)
        self.assertFalse(self.store.item_dir(it.id).exists())

    def test_settings_default_and_not_part_of_cache_key(self):
        self.assertIs(queue_store.DEFAULT_SETTINGS["show_thumbnails"], True)
        self.assertNotIn("show_thumbnails", queue_store.PIXEL_SETTINGS)
        a = queue_store.cache_key("t", None, {"encoder": "hevc"})
        b = queue_store.cache_key("t", None, {"encoder": "hevc", "show_thumbnails": False})
        self.assertEqual(a, b)


class ServiceIntegrationTest(unittest.TestCase):
    def test_get_state_carries_the_film_strip_fields_and_setting_is_accepted(self):
        from oh_streamer.queue import queue_service as QS
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "clip.mp4"
            video.write_bytes(b"x")
            runner = FakeRunner()
            holder = []
            svc = QS.QueueService(root / "q", player_name="thumbs-test", tick=0.02,
                                  thumbs=None)
            svc.thumbs = QT.ThumbManager(svc.store, log=lambda m: None, runner=runner,
                                         probe=lambda *a, **k: 100.0)
            it = svc.store.add(str(video), "file", {}, None, {})
            svc.store.update(it.id, state="queued", source_path=str(video))
            state = svc.get_state()
            d = state["items"][0]
            self.assertEqual((d["thumbs"], d["processed_ranges"], d["worker_fraction"], d["download_fraction"]),
                             (None, [], None, 1.0))
            while True:
                job = svc.thumbs._next_job()
                if job is None:
                    break
                svc.thumbs._execute(job)
            d = svc.get_state()["items"][0]
            self.assertEqual(len(d["thumbs"]["frames"]), 12)
            json.dumps(d)                                           # the whole item dict stays JSON serialisable
            self.assertTrue(svc.get_state()["settings"]["show_thumbnails"])
            self.assertTrue(svc.set_settings({"show_thumbnails": 0})["ok"])
            self.assertIs(svc.get_state()["settings"]["show_thumbnails"], False)
            self.assertEqual(svc.store.get(it.id).cache_key, queue_store.cache_key(str(video), None, {}))
            svc.store.update(it.id, state="done", output_path=str(video))
            self.assertEqual(svc.get_state()["items"][0]["processed_ranges"], [[0.0, 1.0]])


class FrameDataUriTest(unittest.TestCase):
    def test_only_frame_files_inside_item_thumbs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thumbs = root / "items" / "abc" / "thumbs"
            thumbs.mkdir(parents=True)
            frame = thumbs / "f_03_r.jpg"
            frame.write_bytes(b"\xff\xd8jpeg")
            (root / "queue.json").write_text("{}", encoding="utf-8")
            (thumbs / "notes.txt").write_text("x", encoding="utf-8")
            self.assertEqual(QT.frame_data_uri(root, frame.as_uri() + "?v=5"), "data:image/jpeg;base64,/9hqcGVn")
            self.assertIsNone(QT.frame_data_uri(root, (root / "queue.json").as_uri()))
            self.assertIsNone(QT.frame_data_uri(root, (thumbs / "notes.txt").as_uri()))
            self.assertIsNone(QT.frame_data_uri(root, (thumbs / "f_04_r.jpg").as_uri()))   # missing
            self.assertIsNone(QT.frame_data_uri(root, "http://example.test/items/abc/thumbs/f_03_r.jpg"))
            other = root / "elsewhere" / "thumbs"
            other.mkdir(parents=True)
            (other / "f_00_s.jpg").write_bytes(b"x")
            self.assertIsNone(QT.frame_data_uri(root, (other / "f_00_s.jpg").as_uri()))


if __name__ == "__main__":
    unittest.main()
