"""Characterization ("contract") tests of the lada restoration path, through the entry points that stay importable
from oh_streamer.player.oh_player (RestorerFactory, Source, STOP_MARKER, ErrorMarker, the hooks) plus
compare_stack.install_original_capture, buffer_progress.PipelineProbe and detect_gate.DetectGate. They lock the
current behavior before the engine is moved behind an interface and are re-run after the move.

Needs a CUDA GPU, the lada model weights and ../testdata/720p_bench_00.mp4, and is skipped unless OH_STREAMER_GPU_TESTS=1
(from streamer\\, cmd):

    set OH_STREAMER_GPU_TESTS=1
    ..\\lada\\.venv\\Scripts\\python.exe -m unittest tests.engine_contract.test_lada_contract_gpu -v

Each test pulls a few seconds of video at most. Timings are printed to stderr as "[contract-timing] ...".
"""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[2]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import contextlib
import gc
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENABLED = os.environ.get("OH_STREAMER_GPU_TESTS") == "1"
TESTDATA = Path(os.environ.get("OH_STREAMER_TESTDATA") or _STREAMER_ROOT.parent / "testdata")
CLIP = TESTDATA / "720p_bench_00.mp4"     # 1280x720 24 fps, 300 s, real mosaics from ~120 s on (none in the first seconds)
MOSAIC_S = 120.0                          # a start inside a mosaic scene
PLAIN_S = 10.0                            # a start without mosaics: the first frame is out in about a second
DET_NAME, RES_NAME = "v4-fast", "basicvsrpp-v1.2"
# Generous limits (measured on the RTX 5060 laptop): they catch a hang or a lost fast-stop, not speed.
STOP_LIMIT_S = 3.0                        # stop() while a clip restoration runs (cancelled: well under 1 s; not cancelled: ~3.7 s)
FIRST_FRAME_LIMIT_S = 60.0


def timing(name, seconds):
    print(f"[contract-timing] {name}: {seconds:.3f}s", file=_sys.stderr, flush=True)


def setUpModule():
    if not ENABLED:
        raise unittest.SkipTest("set OH_STREAMER_GPU_TESTS=1 to run the lada GPU contract tests")
    try:
        import torch
    except ImportError as e:
        raise unittest.SkipTest(f"torch is not importable: {e}")
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    if not CLIP.exists():
        raise unittest.SkipTest(f"test clip missing: {CLIP}")
    try:
        from oh_streamer.player import oh_player  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"oh_player is not importable: {e!r}")
    from lada import ModelFiles
    for getter, name in ((ModelFiles.get_detection_model_by_name, DET_NAME),
                         (ModelFiles.get_restoration_model_by_name, RES_NAME)):
        m = getter(name)
        if not m or not m.path or not os.path.exists(m.path):
            raise unittest.SkipTest(f"model weights missing: {name}")


# ---------------------------------------------------------------------------------------------------------------
# helpers (drive a FrameRestorer the way player/pipeline.py and queue/queue_worker.py do)
# ---------------------------------------------------------------------------------------------------------------

def _ls():
    from tests.engine_contract.lada_names import namespace
    return namespace()


def stop_like_pipeline(restorer):
    """Session.stop(): abandon the clip restoration in progress, stop, wake a blocked consumer."""
    LS = _ls()
    cancel = getattr(restorer.mosaic_restoration_model, "cancel_event", None)
    if cancel is not None:
        cancel.set()
    try:
        restorer.stop()
    finally:
        if cancel is not None:
            cancel.clear()
    q = restorer.get_frame_restoration_queue()
    LS.threading_utils.put_queue_stop_marker(q)
    return q


@contextlib.contextmanager
def running(restorer, start_s):
    """restorer.start(start_ns) ... and always a clean stop afterwards."""
    q = restorer.get_frame_restoration_queue()
    _ls().threading_utils.empty_out_queue(q)         # start() asserts empty queues
    restorer.eof = False                             # Lada never resets it
    restorer.start(start_ns=int(start_s * 1e9))
    try:
        yield restorer
    finally:
        q = stop_like_pipeline(restorer)
        _ls().threading_utils.empty_out_queue(q)


def pull(restorer, n, timeout=FIRST_FRAME_LIMIT_S):
    """Up to n (frame, pts) elements from the output queue; stops early at a marker / end of video.
    Returns (items, last_marker)."""
    from lada.utils.threading_utils import EOF_MARKER
    LS = _ls()
    q = restorer.get_frame_restoration_queue()
    items, deadline, marker = [], time.monotonic() + timeout, None
    while len(items) < n:
        left = deadline - time.monotonic()
        if left <= 0:
            raise AssertionError(f"timed out after {len(items)}/{n} frames")
        try:
            elem = q.get(timeout=min(left, 1.0))
        except Exception:  # noqa: BLE001 - queue.Empty
            continue
        if elem is LS.STOP_MARKER or isinstance(elem, LS.ErrorMarker) or elem is EOF_MARKER:
            marker = elem
            break
        items.append(elem)
    return items, marker


def times(items, time_base):
    return [float(pts * time_base) for _, pts in items]


class _RangeHandler(BaseHTTPRequestHandler):
    """Static file server with Range support (PyAV / ffmpeg seek with Range requests) that records every request."""
    protocol_version = "HTTP/1.1"
    path_on_disk = None
    seen = None            # list of header dicts

    def log_message(self, *a):
        pass

    def _serve(self, head):
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        size = os.path.getsize(self.path_on_disk)
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        try:
            with open(self.path_on_disk, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionError, OSError):
            pass

    def do_GET(self):
        self._serve(False)

    def do_HEAD(self):
        self._serve(True)


# ---------------------------------------------------------------------------------------------------------------

class LadaRestorerContract(unittest.TestCase):
    """One loaded RestorerFactory for all tests of the class."""

    @classmethod
    def setUpClass(cls):
        LS = _ls()
        from oh_streamer.processing import compare_stack
        cls.LS = LS
        cls.device = LS.device_backend.select_device("auto")
        t = time.perf_counter()
        cls.factory = LS.RestorerFactory(cls.device, 240)
        cls.factory.load(DET_NAME, RES_NAME, True)
        timing("model load", time.perf_counter() - t)
        compare_stack.install_original_capture()
        cls.probe = LS.ffprobe(str(CLIP), LS.FFmpegHttpOptions({}))

    @classmethod
    def tearDownClass(cls):
        cls.factory.release()
        cls.factory = None
        gc.collect()

    def source(self):
        return self.LS.Source(str(CLIP), {}, probe=self.probe)

    def make(self, source=None, clip_len=None, restoration_fps="original"):
        self.factory.max_clip_length = clip_len or 240
        self.factory.restoration_fps = restoration_fps
        try:
            return self.factory.make(source or self.source())
        finally:
            self.factory.max_clip_length = 240
            self.factory.restoration_fps = "original"

    # -- play / seek -------------------------------------------------------------------------------------------

    def test_play_from_start(self):
        src = self.source()
        r = self.make(src)
        tb, fps = float(src.meta.time_base), float(src.meta.video_fps_exact)
        t = time.perf_counter()
        with running(r, 0.0):
            items, marker = pull(r, 48)
        timing("play from start, 48 frames", time.perf_counter() - t)
        self.assertIsNone(marker)
        self.assertEqual(len(items), 48)
        for frame, pts in items:
            self.assertEqual(tuple(frame.shape), (720, 1280, 3))
            self.assertEqual(str(frame.dtype), "torch.uint8")
        ts = times(items, tb)
        self.assertLess(abs(ts[0] - int(src.meta.start_pts) * tb), 0.5 / fps)      # the first frame of the video stream
        for a, b in zip(ts, ts[1:]):
            self.assertAlmostEqual(b - a, 1.0 / fps, delta=0.2 / fps)     # monotonic, one frame apart

    def test_seek_start_lands_on_target(self):
        src = self.source()
        tb, fps = float(src.meta.time_base), float(src.meta.video_fps_exact)
        for target in (PLAIN_S, 37.3):
            r = self.make(src)
            t = time.perf_counter()
            with running(r, target):
                items, marker = pull(r, 6)
            timing(f"seek to {target}s, 6 frames", time.perf_counter() - t)
            self.assertIsNone(marker)
            ts = times(items, tb)
            self.assertLess(abs(ts[0] - target), 1.0 / fps, f"first frame at {ts[0]:.3f}s for target {target}")
            self.assertEqual(sorted(ts), ts)

    def test_stride_halves_frame_rate(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg not found")
        with tempfile.TemporaryDirectory(prefix="contract_stride_") as tmp:
            clip60 = str(Path(tmp) / "clip60.mp4")
            p = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "10", "-t", "6", "-i", str(CLIP), "-an",
                                "-vf", "fps=60", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", clip60],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            LS = self.LS
            src = LS.Source(clip60, {})
            self.assertAlmostEqual(src.meta.video_fps, 60.0, places=3)
            native_frames = src.meta.frames_count
            r = self.make(src, restoration_fps="30")        # "30" on a 60 fps source: every second frame
            self.assertEqual(src.meta.restoration_stride, 2)
            self.assertAlmostEqual(float(src.meta.video_fps_exact), 30.0, places=3)
            self.assertEqual(src.meta.frames_count, (native_frames + 1) // 2)
            tb = float(src.meta.time_base)
            with running(r, 0.0):
                items, marker = pull(r, 400, timeout=60)     # to the end of the 6 s clip
            self.assertGreaterEqual(len(items), 170)
            self.assertLessEqual(len(items), 190)            # 360 native frames -> 180
            ts = times(items, tb)
            for a, b in zip(ts, ts[1:]):
                self.assertAlmostEqual(b - a, 1 / 30, delta=0.1 / 30)
            del r
            gc.collect()
            LS.forget_sources()

    # -- stopping ----------------------------------------------------------------------------------------------

    def test_fast_stop_during_clip_restoration(self):
        LS = self.LS
        r = self.make(clip_len=240)
        probe = LS.buffer_progress.PipelineProbe.attach(r)
        probe.reset(MOSAIC_S)
        with running(r, MOSAIC_S):
            deadline = time.monotonic() + FIRST_FRAME_LIMIT_S
            while time.monotonic() < deadline:       # wait until a clip is really being restored
                snap = probe.snapshot()
                if any(c[2] is not None and c[3] is None for c in snap["clips"]):
                    break
                time.sleep(0.05)
            else:
                self.fail("no clip restoration started")
            time.sleep(0.5)
            t = time.perf_counter()
            done = threading.Event()

            def stopper():
                stop_like_pipeline(r)
                done.set()
            threading.Thread(target=stopper, daemon=True).start()
            self.assertTrue(done.wait(STOP_LIMIT_S * 4), "stop() hung")
            took = time.perf_counter() - t
        timing("stop during clip restoration (cancel event set)", took)
        self.assertLess(took, STOP_LIMIT_S)

    def test_stop_marker_wakes_blocked_consumer(self):
        LS = self.LS
        r = self.make(clip_len=240)
        seen, finished = [], threading.Event()

        def consumer():
            for elem in r:             # blocks on the output queue: the first mosaic clip takes seconds
                seen.append(elem)
                if elem is LS.STOP_MARKER or isinstance(elem, LS.ErrorMarker):
                    break
            finished.set()
        with running(r, MOSAIC_S):
            threading.Thread(target=consumer, daemon=True).start()
            time.sleep(1.5)
            self.assertFalse(finished.is_set(), "a frame came out before the stop: pick a later start")
            t = time.perf_counter()
            stop_like_pipeline(r)      # FrameRestorer.stop() + put_queue_stop_marker on the output queue
            self.assertTrue(finished.wait(STOP_LIMIT_S), "the consumer stayed blocked after stop + stop marker")
            timing("consumer woken by the stop marker", time.perf_counter() - t)
        self.assertIs(seen[-1], LS.STOP_MARKER)

    def test_repeated_start_stop_on_one_restorer(self):
        src = self.source()
        r = self.make(src)
        tb = float(src.meta.time_base)
        gc.collect()
        threads_before = threading.active_count()
        alloc_ref = None
        t_all = time.perf_counter()
        for i in range(10):
            target = PLAIN_S + 1.0 * i
            t = time.perf_counter()
            with running(r, target):
                items, marker = pull(r, 2, timeout=30)
            self.assertIsNone(marker, f"cycle {i}")
            self.assertEqual(len(items), 2, f"cycle {i}")
            self.assertLess(abs(float(items[0][1] * tb) - target), 1 / 24, f"cycle {i}")
            self.assertLess(time.perf_counter() - t, 30, f"cycle {i}")
            if i == 1:
                gc.collect()
                alloc_ref = self._allocated()
        timing("10 start/stop cycles", time.perf_counter() - t_all)
        gc.collect()
        self.assertLessEqual(threading.active_count(), threads_before + 1)
        self.assertLess(self._allocated() - alloc_ref, 256 * 1024 * 1024, "VRAM grew over repeated start/stop")

    def _allocated(self):
        import torch
        return torch.cuda.memory_allocated(self.device)

    # -- original capture / probes -----------------------------------------------------------------------------

    def test_original_capture_patches_on_mosaic_frames(self):
        r = self.make(clip_len=60)
        r.capture_original = True
        with running(r, MOSAIC_S):
            items, marker = pull(r, 12)
        self.assertEqual(len(items), 12)
        with_patches = [(f, f._lada_orig) for f, _ in items if getattr(f, "_lada_orig", None)]
        self.assertGreaterEqual(len(with_patches), 10, "mosaic frames must carry _lada_orig")
        changed = 0
        for frame, patches in with_patches:
            for (t, l, b, rr), patch in patches:
                self.assertTrue(0 <= t <= b < frame.shape[0] and 0 <= l <= rr < frame.shape[1])
                self.assertEqual(tuple(patch.shape), (b - t + 1, rr - l + 1, 3))
                if not (frame[t:b + 1, l:rr + 1] == patch).all():
                    changed += 1
        self.assertGreater(changed, 0, "the restored region must differ from the captured mosaic patch")

    def test_no_capture_when_flag_is_off(self):
        r = self.make(clip_len=60)
        r.capture_original = False
        with running(r, MOSAIC_S):
            items, _ = pull(r, 4)
        self.assertTrue(items)
        self.assertFalse(any(hasattr(f, "_lada_orig") for f, _ in items))

    def test_pipeline_probe_and_detect_gate(self):
        from oh_streamer.player import buffer_progress
        from oh_streamer.processing import detect_gate
        r = self.make(clip_len=60)
        probe = buffer_progress.PipelineProbe.attach(r)
        self.assertIsNotNone(probe)
        self.assertIs(buffer_progress.PipelineProbe.attach(r), probe)
        gate = detect_gate.DetectGate.attach(r)
        self.assertIsNotNone(gate)
        self.assertIs(detect_gate.DetectGate.attach(r), gate)
        probe.reset(MOSAIC_S)
        gate.on_held = probe.note_detector_held
        gate.reset(r.max_clip_length + detect_gate.MARGIN)
        self.assertFalse(gate.is_open)
        with running(r, MOSAIC_S):
            items, marker = pull(r, 8)
            self.assertEqual(len(items), 8)
            snap = probe.snapshot()
            self.assertIsNotNone(snap["base"])
            self.assertGreater(snap["det"], 0)
            self.assertTrue(snap["mosaic"], "the detector must have seen mosaics")
            self.assertTrue(snap["clips"], "a clip must have been cut")
            self.assertTrue(any(c[3] is not None for c in snap["clips"]), "a clip must have been restored")
            self.assertIn("frames detected", probe.summary())
            self.assertGreaterEqual(gate.out_n, 8)
            self.assertGreater(gate.det_n, 0)
            self.assertLessEqual(gate.max_ahead, gate.limit + 1)
            gate.open("target reached")
            self.assertTrue(gate.is_open)
            self.assertIn("target reached", gate.summary())
            probe.freeze()

    # -- HTTP source -------------------------------------------------------------------------------------------

    def test_http_source_with_header(self):
        LS = self.LS
        handler = type("H", (_RangeHandler,), {"path_on_disk": str(CLIP), "seen": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/clip.mp4"
            src = LS.Source(url, {"X-Contract-Test": "lada-contract", "Referer": "https://example.invalid/"})
            self.assertEqual((src.meta.video_width, src.meta.video_height), (1280, 720))
            r = self.make(src)
            with running(r, PLAIN_S):
                items, marker = pull(r, 6)
            self.assertEqual(len(items), 6)
            ts = times(items, float(src.meta.time_base))
            self.assertLess(abs(ts[0] - PLAIN_S), 1 / 24)
        finally:
            server.shutdown()
            server.server_close()
            LS.forget_sources()
        self.assertTrue(handler.seen)
        self.assertTrue(all(h.get("x-contract-test") == "lada-contract" for h in handler.seen),
                        f"every request must carry the header: {handler.seen[:3]}")
        self.assertTrue(all(h.get("referer") == "https://example.invalid/" for h in handler.seen))


class LadaLoadReleaseContract(unittest.TestCase):

    def test_load_open_release_frees_vram(self):
        import torch
        LS = _ls()
        device = LS.device_backend.select_device("auto")
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        f = LS.RestorerFactory(device, 240)
        self.assertIsNone(f.det_model)
        f.load(DET_NAME, RES_NAME, True)
        loaded = torch.cuda.memory_allocated(device)
        self.assertEqual((f.det_name, f.res_name, f.fp16), (DET_NAME, RES_NAME, True))
        self.assertGreater(loaded - base, 50 * 1024 * 1024, "loading must put weights on the GPU")
        # "open": a restorer can be built around the loaded models
        r = f.make(LS.Source(str(CLIP), {}))
        self.assertTrue(hasattr(r.mosaic_restoration_model, "cancel_event"))
        del r
        LS.forget_sources()
        t = time.perf_counter()
        f.release()
        timing("release", time.perf_counter() - t)
        self.assertIsNone(f.det_model)
        self.assertIsNone(f.res_model)
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(device)
        # current behavior: about 32 MiB of the 75 MiB stay allocated after release (not freed by the factory itself)
        self.assertGreater(loaded - after, 20 * 1024 * 1024, f"VRAM must drop: base={base} loaded={loaded} after={after}")
        self.assertLess(after - base, (loaded - base) * 0.6, f"VRAM must drop: base={base} loaded={loaded} after={after}")

    def test_unknown_model_name_is_an_error(self):
        LS = _ls()
        f = LS.RestorerFactory(LS.device_backend.select_device("auto"), 240)
        with self.assertRaises(RuntimeError):
            f.load("no-such-detector", RES_NAME, True)


if __name__ == "__main__":
    unittest.main()
