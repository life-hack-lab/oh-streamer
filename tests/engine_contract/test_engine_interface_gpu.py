"""The engine contract driven through the engine
interface only: get_engine(name) -> load -> open(SourceSpec) -> session.start / frames / stop / wake / metrics.
Same gating and test data as test_lada_contract_gpu.py (which locks the pre-interface behavior).

ENGINE_NAMES lists the engines the contract runs against; a test class is generated per name. The lada classes need the
GPU gate above; "passthrough" needs no GPU and no weights (only ffprobe and the test clip), so it runs in the normal
suite. Tests that need a capability skip when the engine's EngineInfo.capabilities lacks it.

    set OH_STREAMER_GPU_TESTS=1
    ..\\lada\\.venv\\Scripts\\python.exe -m unittest tests.engine_contract.test_engine_interface_gpu -v
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

import gc
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from http.server import ThreadingHTTPServer
from pathlib import Path

from oh_streamer.engines import base, get_engine
from oh_streamer.sources.source_resolve import FFmpegHttpOptions
from tests.engine_contract.test_lada_contract_gpu import (   # same gate, clip, limits and HTTP test server
    CLIP, DET_NAME, ENABLED, FIRST_FRAME_LIMIT_S, MOSAIC_S, PLAIN_S, RES_NAME, STOP_LIMIT_S, _RangeHandler, timing)

ENGINE_NAMES = ["lada", "passthrough"]
GPU_ENGINES = {"lada"}                                         # the others run on any machine
MODELS = {"lada": (DET_NAME, RES_NAME), "passthrough": ("passthrough", "passthrough")}


def require_engine(name):
    """SkipTest unless engine `name` can run here: the test clip, and for the GPU engines OH_STREAMER_GPU_TESTS=1, CUDA and weights."""
    try:
        import torch
    except ImportError as e:
        raise unittest.SkipTest(f"torch is not importable: {e}")
    if not CLIP.exists():
        raise unittest.SkipTest(f"test clip missing: {CLIP}")
    if name in GPU_ENGINES:
        if not ENABLED:
            raise unittest.SkipTest("set OH_STREAMER_GPU_TESTS=1 to run the engine contract GPU tests")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("no CUDA device")
    try:
        engine = get_engine(name, new=True)
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"the {name} engine is not importable: {e!r}")
    det, res = MODELS[name]
    cat = engine.list_models()
    if det not in cat.detection or res not in cat.restoration:
        raise unittest.SkipTest(f"model weights missing: {det} / {res}")


def _fraction(s):
    num, _, den = str(s).partition("/")
    return Fraction(int(num), int(den or 1))


def spec_for(path, headers=None) -> base.SourceSpec:
    """A SourceSpec from ffprobe (what the player's probing step produces)."""
    http = FFmpegHttpOptions(headers or {})
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format"]
    if str(path).startswith("http"):
        cmd += http.cli_args()
    p = subprocess.run(cmd + [str(path)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    probe = json.loads(p.stdout)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = _fraction(vs["r_frame_rate"])
    duration = float(vs.get("duration") or probe["format"].get("duration") or 0)
    return base.SourceSpec(
        url=str(path), http=http, width=int(vs["width"]), height=int(vs["height"]), fps_exact=fps,
        time_base=_fraction(vs["time_base"]), start_pts=vs.get("start_pts"), duration=duration,
        frames_count=int(vs.get("nb_frames") or 0) or int(duration * float(fps)), codec_name=vs["codec_name"])


def pull(session, n, timeout=FIRST_FRAME_LIMIT_S):
    """Up to n RestoredFrames from session.frames() (read on a helper thread so a hang fails instead of blocking).
    Returns (frames, last_marker); marker is an EndOfStream / EngineError or None."""
    items, state = [], {"marker": None}

    def run():
        for x in session.frames():
            if isinstance(x, base.RestoredFrame):
                items.append(x)
                if len(items) >= n:
                    return
            else:
                state["marker"] = x
                return
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise AssertionError(f"timed out after {len(items)}/{n} frames")
    return items, state["marker"]


def times(frames, time_base):
    return [float(f.pts * time_base) for f in frames]


class EngineContract:
    """The contract; one TestCase subclass per engine name is generated below (class attribute engine_name)."""
    engine_name = None

    @classmethod
    def setUpClass(cls):
        require_engine(cls.engine_name)
        from oh_streamer.processing import device_backend
        cls.engine = get_engine(cls.engine_name, new=True)
        cls.caps = cls.engine.info().capabilities
        cls.device = device_backend.select_device("auto") if cls.engine_name in GPU_ENGINES else None
        cls.det, cls.res = MODELS[cls.engine_name]
        cls.models = base.ModelChoice(cls.det, cls.res, True)
        t = time.perf_counter()
        cls.engine.load(cls.device, cls.models, 240)
        timing(f"{cls.engine_name}: model load", time.perf_counter() - t)
        cls.spec = spec_for(CLIP)

    @classmethod
    def tearDownClass(cls):
        cls.engine.release()
        gc.collect()

    def setUp(self):
        self.engine.restoration_fps = "original"
        self.engine.max_clip_length = 240
        self._sessions = []

    def tearDown(self):
        for s in self._sessions:
            s.close()
        self.engine.max_clip_length = 240

    def need(self, cap):
        if cap not in self.caps:
            self.skipTest(f"{self.engine_name} lacks the {cap} capability")

    def open(self, spec=None, clip_len=None, restoration_fps="original"):
        self.engine.restoration_fps = restoration_fps
        if clip_len:
            self.engine.max_clip_length = clip_len
        s = self.engine.open(spec or self.spec)
        self._sessions.append(s)
        return s

    @staticmethod
    def stop(s, **kw):
        s.stop(**kw)
        s.wake()
        s.drain()

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def test_info_and_models(self):
        info = self.engine.info()
        self.assertEqual(info.name, self.engine_name)
        self.assertTrue(info.version)
        self.assertEqual(self.engine.models, self.models)
        cat = self.engine.list_models()
        self.assertIn(self.det, cat.detection)
        self.assertIn(self.res, cat.restoration)

    def test_load_same_models_is_a_no_op_and_applies_clip_length(self):
        t = time.perf_counter()
        self.engine.load(self.device, self.models, 120)
        self.assertLess(time.perf_counter() - t, 2.0, "an unchanged model choice must not reload")
        self.assertEqual(self.engine.max_clip_length, 120)
        self.assertEqual(self.open().max_clip_length, 120)
        self.engine.load(self.device, self.models, 240)

    def test_open_without_models_fails(self):
        with self.assertRaises(RuntimeError):
            get_engine(self.engine_name, new=True).open(self.spec)

    # -- play / seek -------------------------------------------------------------------------------------------

    def test_play_from_start(self):
        s = self.open()
        tb, fps = float(self.spec.time_base), float(self.spec.fps_exact)
        t = time.perf_counter()
        s.start(0.0)
        try:
            frames, marker = pull(s, 48)
        finally:
            self.stop(s)
        timing(f"{self.engine_name}: play from start, 48 frames", time.perf_counter() - t)
        self.assertIsNone(marker)
        self.assertEqual(len(frames), 48)
        for f in frames:
            self.assertEqual(tuple(f.image.shape), (720, 1280, 3))
            self.assertEqual(str(f.image.dtype), "torch.uint8")
        ts = times(frames, tb)
        self.assertLess(abs(ts[0] - int(self.spec.start_pts) * tb), 0.5 / fps)
        for a, b in zip(ts, ts[1:]):
            self.assertAlmostEqual(b - a, 1.0 / fps, delta=0.2 / fps)

    def test_seek_start_lands_on_target(self):
        s = self.open()
        tb, fps = float(self.spec.time_base), float(self.spec.fps_exact)
        for target in (PLAIN_S, 37.3):       # the same session: seeking is stop + start
            s.start(target)
            try:
                frames, marker = pull(s, 6)
            finally:
                self.stop(s)
            self.assertIsNone(marker)
            ts = times(frames, tb)
            self.assertLess(abs(ts[0] - target), 1.0 / fps, f"first frame at {ts[0]:.3f}s for target {target}")
            self.assertEqual(sorted(ts), ts)

    def test_stride_halves_frame_rate(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg not found")
        with tempfile.TemporaryDirectory(prefix="engine_stride_") as tmp:
            clip60 = str(Path(tmp) / "clip60.mp4")
            p = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "10", "-t", "6", "-i", str(CLIP), "-an",
                                "-vf", "fps=60", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", clip60],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            spec = spec_for(clip60)
            self.assertAlmostEqual(float(spec.fps_exact), 60.0, places=3)
            s = self.open(spec, restoration_fps="30")        # "30" on a 60 fps source: every second frame
            s.start(0.0)
            try:
                frames, marker = pull(s, 400, timeout=60)    # to the end of the 6 s clip
            finally:
                self.stop(s)
            self.assertGreaterEqual(len(frames), 170)
            self.assertLessEqual(len(frames), 190)           # 360 native frames -> 180
            for a, b in zip(*(lambda ts: (ts, ts[1:]))(times(frames, float(spec.time_base)))):
                self.assertAlmostEqual(b - a, 1 / 30, delta=0.1 / 30)
            s.close()
            gc.collect()
            self.engine.forget_sources()

    # -- stopping ----------------------------------------------------------------------------------------------

    def test_fast_stop_during_clip_restoration(self):
        self.need(base.FAST_CANCEL)
        self.need(base.CLIP_METRICS)
        s = self.open(clip_len=240)
        probe = s.attach_progress_probe()
        probe.reset(MOSAIC_S)
        s.start(MOSAIC_S)
        try:
            deadline = time.monotonic() + FIRST_FRAME_LIMIT_S
            while time.monotonic() < deadline:       # wait until a clip is really being restored
                if s.metrics().clips_restoring:
                    break
                time.sleep(0.05)
            else:
                self.fail("no clip restoration started")
            time.sleep(0.5)
            t = time.perf_counter()
            done = threading.Event()

            def stopper():
                s.stop(abandon=True)
                done.set()
            threading.Thread(target=stopper, daemon=True).start()
            self.assertTrue(done.wait(STOP_LIMIT_S * 4), "stop() hung")
            took = time.perf_counter() - t
        finally:
            self.stop(s)
        timing(f"{self.engine_name}: stop during clip restoration", took)
        self.assertLess(took, STOP_LIMIT_S)

    def test_stop_returns_quickly(self):
        s = self.open()
        s.start(PLAIN_S)
        try:
            frames, marker = pull(s, 2)
            self.assertEqual(len(frames), 2)
            t = time.perf_counter()
            s.stop(abandon=True)
            took = time.perf_counter() - t
        finally:
            self.stop(s)
        timing(f"{self.engine_name}: stop", took)
        self.assertLess(took, STOP_LIMIT_S)

    def test_wake_unblocks_a_consumer_waiting_for_the_first_frame(self):
        s = self.open()
        seen, finished = [], threading.Event()

        def consumer():
            for x in s.frames():      # nothing was started: blocks on the output
                seen.append(x)
                if not isinstance(x, base.RestoredFrame):
                    break
            finished.set()
        threading.Thread(target=consumer, daemon=True).start()
        time.sleep(0.3)
        self.assertFalse(finished.is_set())
        s.wake()
        self.assertTrue(finished.wait(STOP_LIMIT_S), "the consumer stayed blocked after wake")
        s.drain()
        self.assertIsInstance(seen[-1], base.EndOfStream)

    def test_wake_unblocks_a_blocked_consumer(self):
        self.need(base.CLIP_METRICS)     # needs an engine whose first frame takes seconds (clip restoration)
        s = self.open(clip_len=240)
        seen, finished = [], threading.Event()

        def consumer():
            for x in s.frames():      # blocks: the first mosaic clip takes seconds
                seen.append(x)
                if not isinstance(x, base.RestoredFrame):
                    break
            finished.set()
        s.start(MOSAIC_S)
        try:
            threading.Thread(target=consumer, daemon=True).start()
            time.sleep(1.5)
            self.assertFalse(finished.is_set(), "a frame came out before the stop: pick a later start")
            t = time.perf_counter()
            s.stop()
            s.wake()
            self.assertTrue(finished.wait(STOP_LIMIT_S), "the consumer stayed blocked after stop + wake")
            timing(f"{self.engine_name}: consumer woken", time.perf_counter() - t)
        finally:
            s.drain()
        self.assertIsInstance(seen[-1], base.EndOfStream)

    def test_repeated_start_stop_on_one_session(self):
        import torch
        s = self.open()
        tb = float(self.spec.time_base)
        gc.collect()
        threads_before = threading.active_count()
        gpu = self.engine_name in GPU_ENGINES
        alloc_ref = None
        t_all = time.perf_counter()
        for i in range(10):
            target = PLAIN_S + 1.0 * i
            t = time.perf_counter()
            s.start(target)
            try:
                frames, marker = pull(s, 2, timeout=30)
            finally:
                self.stop(s)
            self.assertIsNone(marker, f"cycle {i}")
            self.assertEqual(len(frames), 2, f"cycle {i}")
            self.assertLess(abs(float(frames[0].pts * tb) - target), 1 / 24, f"cycle {i}")
            self.assertLess(time.perf_counter() - t, 30, f"cycle {i}")
            if i == 1 and gpu:
                gc.collect()
                alloc_ref = torch.cuda.memory_allocated(self.device)
        timing(f"{self.engine_name}: 10 start/stop cycles", time.perf_counter() - t_all)
        gc.collect()
        self.assertLessEqual(threading.active_count(), threads_before + 1)
        if gpu:
            self.assertLess(torch.cuda.memory_allocated(self.device) - alloc_ref, 256 * 1024 * 1024,
                            "VRAM grew over repeated start/stop")

    def test_restart_after_the_end_of_the_source_yields_again(self):
        s = self.open()
        tail = max(0.0, self.spec.duration - 1.0)
        for _ in range(2):          # the second start must not see the first run's eof
            s.start(tail)
            try:
                frames, marker = pull(s, 400, timeout=60)
            finally:
                self.stop(s)
            self.assertGreater(len(frames), 5)
            self.assertIsNone(marker)

    # -- original capture / metrics ----------------------------------------------------------------------------

    def test_original_patches_on_mosaic_frames(self):
        self.need(base.ORIGINAL_CAPTURE)
        s = self.open(clip_len=60)
        s.start(MOSAIC_S, capture_original=True)
        try:
            frames, marker = pull(s, 12)
        finally:
            self.stop(s)
        self.assertEqual(len(frames), 12)
        with_patches = [f for f in frames if f.original_patches]
        self.assertGreaterEqual(len(with_patches), 10, "mosaic frames must carry original_patches")
        changed = 0
        for f in with_patches:
            h, w = f.image.shape[:2]
            for (t, l, b, r), patch in f.original_patches:
                self.assertTrue(0 <= t <= b < h and 0 <= l <= r < w)
                self.assertEqual(tuple(patch.shape), (b - t + 1, r - l + 1, 3))
                if not (f.image[t:b + 1, l:r + 1] == patch).all():
                    changed += 1
        self.assertGreater(changed, 0, "the restored region must differ from the captured mosaic patch")

    def test_no_patches_without_capture(self):
        s = self.open(clip_len=60)
        s.start(MOSAIC_S)
        try:
            frames, _ = pull(s, 4)
        finally:
            self.stop(s)
        self.assertTrue(frames)
        self.assertTrue(all(f.original_patches is None for f in frames))

    def test_metrics_and_probe_and_gate(self):
        self.need(base.CLIP_METRICS)
        self.need(base.LOOKAHEAD_LIMIT)
        s = self.open(clip_len=60)
        probe = s.attach_progress_probe()
        self.assertIsNotNone(probe)
        self.assertIs(s.attach_progress_probe(), probe)
        gate = s.attach_detect_gate()
        self.assertIsNotNone(gate)
        self.assertIs(s.attach_detect_gate(), gate)
        probe.reset(MOSAIC_S)
        gate.on_held = probe.note_detector_held
        s.set_lookahead_limit(s.max_clip_length + 30)
        self.assertFalse(gate.is_open)
        idle = s.metrics()
        self.assertIsInstance(idle, base.PipelineMetrics)
        self.assertEqual((idle.frames_detected, idle.output_queue), (0, 0))
        s.start(MOSAIC_S)
        try:
            frames, marker = pull(s, 8)
            self.assertEqual(len(frames), 8)
            m = s.metrics()
            self.assertGreater(m.frames_detected, 0)
            self.assertGreater(m.output_capacity, 0)
            self.assertLessEqual(m.output_queue, m.output_capacity)
            self.assertTrue({"feeder", "clips", "out"} <= set(m.queues))
            self.assertFalse(m.eof)
            snap = probe.snapshot()
            self.assertTrue(snap["clips"], "a clip must have been cut")
            self.assertTrue(any(c[3] is not None for c in snap["clips"]), "a clip must have been restored")
            self.assertLessEqual(gate.max_ahead, gate.limit + 1)
            s.set_lookahead_limit(None)
            self.assertTrue(gate.is_open)
        finally:
            self.stop(s)

    # -- HTTP source -------------------------------------------------------------------------------------------

    def test_http_source_with_header(self):
        handler = type("H", (_RangeHandler,), {"path_on_disk": str(CLIP), "seen": []})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/clip.mp4"
            spec = spec_for(url, {"X-Contract-Test": "engine-contract", "Referer": "https://example.invalid/"})
            self.assertEqual((spec.width, spec.height), (1280, 720))
            s = self.open(spec)
            s.start(PLAIN_S)
            try:
                frames, marker = pull(s, 6)
            finally:
                self.stop(s)
            self.assertEqual(len(frames), 6)
            self.assertLess(abs(times(frames, float(spec.time_base))[0] - PLAIN_S), 1 / 24)
        finally:
            server.shutdown()
            server.server_close()
            self.engine.forget_sources()
        self.assertTrue(handler.seen)
        self.assertTrue(all(h.get("x-contract-test") == "engine-contract" for h in handler.seen),
                        f"every request must carry the header: {handler.seen[:3]}")

    def test_closed_session_refuses_to_start(self):
        s = self.open()
        s.close()
        with self.assertRaises(RuntimeError):
            s.start(0.0)


class EngineLifecycleContract:
    """VRAM and weight loading: only the GPU engines."""
    engine_name = None

    @classmethod
    def setUpClass(cls):
        if cls.engine_name not in GPU_ENGINES:
            raise unittest.SkipTest(f"{cls.engine_name} loads no weights")
        require_engine(cls.engine_name)

    def test_load_open_release_frees_vram_and_reload_switches_fp16(self):
        import torch
        from oh_streamer.processing import device_backend
        device = device_backend.select_device("auto")
        engine = get_engine(self.engine_name, new=True)
        gc.collect()
        torch.cuda.empty_cache()
        base_alloc = torch.cuda.memory_allocated(device)
        engine.load(device, base.ModelChoice(DET_NAME, RES_NAME, True), 240)
        loaded = torch.cuda.memory_allocated(device)
        self.assertGreater(loaded - base_alloc, 20 * 1024 * 1024, "loading must put weights on the GPU")
        s = engine.open(spec_for(CLIP))
        s.close()
        del s
        # a changed FP16 setting reloads (old weights released first)
        engine.load(device, base.ModelChoice(DET_NAME, RES_NAME, False), 240)
        self.assertEqual(engine.models, base.ModelChoice(DET_NAME, RES_NAME, False))
        engine.forget_sources()
        t = time.perf_counter()
        engine.release()
        timing(f"{self.engine_name}: release", time.perf_counter() - t)
        self.assertIsNone(engine.models)
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(device)
        self.assertLess(after - base_alloc, (loaded - base_alloc) * 0.6,
                        f"VRAM must drop: base={base_alloc} loaded={loaded} after={after}")

    def test_unknown_model_name_is_an_error(self):
        from oh_streamer.processing import device_backend
        engine = get_engine(self.engine_name, new=True)
        with self.assertRaises(RuntimeError):
            engine.load(device_backend.select_device("auto"), base.ModelChoice("no-such-detector", RES_NAME, True), 240)
        self.assertIsNone(engine.models)


for _name in ENGINE_NAMES:
    for _mixin in (EngineContract, EngineLifecycleContract):
        _cls = f"{_name.title()}{_mixin.__name__}"
        globals()[_cls] = type(_cls, (_mixin, unittest.TestCase), {"engine_name": _name, "__module__": __name__})
del _name, _mixin, _cls, EngineContract, EngineLifecycleContract


if __name__ == "__main__":
    unittest.main()
