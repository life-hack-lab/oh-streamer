"""Tests for queue/subtitle_host_gpu.py (M7 / T2): QueueGpuHost (lease, stop / resume restoration with the turn
hold, restore info, the model-process registry) and the local half of the scheduler mixin, against a fake
service. No GPU: fake models; the only real process is a sleeping python child this test starts and kills."""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import subtitle_host_gpu as gh
from oh_streamer.queue import subtitle_local as sl
from oh_streamer.queue.queue_service import _Run, kill_tree
from oh_streamer.subtitles import slice_plan as sp

from test_queue_subtitle_host import (CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, SAMPLE_A_PLAY_POS, FakeService,
                                      frozen, make_item, sample_a_plan)

ASR_ID = {"provider": "local", "asr_mode": "dual", "models": {"a": "x"}}
MT_ID = {"provider": "local", "backend": "sakura", "model": "m"}
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def every_30s_silence(_source, start, end):
    return [(30.0 * i - 0.5, 30.0 * i + 0.5) for i in range(int(start // 30), int(end // 30) + 2)
            if start <= 30.0 * i <= end]


class GpuService(gh.GpuSubtitleSchedulerMixin, FakeService):
    """FakeService (T1 tests) + the members the local half reads: `_wk`, player busy, perf, worker start."""

    def __init__(self, root: Path, **settings):
        super().__init__(root, **settings)
        self._wk = None
        self._subtitle_local = None
        self._subtitle_cloud = None
        self._subtitle_turn_hold = None
        self._subtitle_windows = []
        self.busy = False
        self.perf = SimpleNamespace(holding=False)
        self._subtitle_gpu_info_cache = sl.GpuInfo()      # never run nvidia-smi in tests
        self.deps_built = 0
        self.deps_factory = None
        self.worker_starts = []
        self.memory_path = root / "modes.json"

    def is_player_busy(self):
        return self.busy

    def _request_stop(self, run, reason):          # the real one: cancel + kill the registered process tree
        self.stops.append((run.item_id, reason))
        with self._lock:
            if run.reason is None:
                run.reason = reason
            run.cancel.set()
            proc = run.proc
        if proc is not None and proc.poll() is None:
            kill_tree(proc.pid)

    def _subtitle_gpu_memory(self):
        return gt.GpuModeMemory(self.memory_path)

    def _subtitle_new_local_deps(self, item, host, config):
        self.deps_built += 1
        return self.deps_factory(host)

    # what T3 adds to QueueService._maybe_start_worker: no restoration while a subtitle turn holds the GPU
    def maybe_start_worker(self, item_id):
        with self._lock:
            if self._wk is not None or self._subtitle_turn_held():
                return None
            run = FakeWorker.start(self, item_id)
            self.worker_starts.append(item_id)
            return run


class FakeWorker:
    """A restoration worker thread in `_wk` that runs until cancelled, then (after `exit_delay`) clears the slot
    like `_run_worker`'s finally."""

    @staticmethod
    def start(svc, item_id, exit_delay=0.1):
        run = _Run(item_id)
        run.stage = "restore"

        def body():
            run.cancel.wait(10)
            time.sleep(exit_delay)
            with svc._lock:
                if svc._wk is run:
                    svc._wk = None
        run.thread = threading.Thread(target=body, daemon=True)
        svc._wk = run
        run.thread.start()
        return run


def local_item(item_id="it1", source=None, **kw):
    return make_item(item_id, plan=frozen(asr=LOCAL_ASR, mt=LOCAL_MT), source=source, **kw)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.source = self.root / "movie.mp4"
        self.source.write_bytes(b"x" * 5000)
        self.svc = GpuService(self.root)
        self.children = []
        self.addCleanup(self._kill_children)

    def _kill_children(self):
        for proc in self.children:              # only the processes this test started
            if proc.poll() is None:
                kill_tree(proc.pid)
            proc.wait(5)

    def item(self, item_id="it1", **kw):
        return self.svc.store.add(local_item(item_id, source=str(self.source), **kw))

    def host(self, item_id="it1"):
        run = _Run(item_id)
        run.stage = gh.RUN_STAGE_LOCAL
        return gh.QueueGpuHost(self.svc, item_id, run)

    def deps(self, asr_batch):
        def mt_session():
            import contextlib

            @contextlib.contextmanager
            def session():
                yield lambda ja_doc, workdir: [dict(r, text="譯") for r in ja_doc["segments"]]
            return session()
        return sl.LocalDeps(asr_batch=asr_batch, mt_session=mt_session, asr_identity=ASR_ID, mt_identity=MT_ID,
                            probe_duration=lambda s: 600.0, detect_silences=every_30s_silence,
                            asr_params={"pipeline": "test"}, mt_params={"batch_size": 30})

    def sleeping_asr(self, started: threading.Event):
        """asr_batch that runs a real (sleeping) model process registered through the host's on_process hook,
        like run_asr_child: a killed process surfaces as an ordinary RuntimeError."""
        def factory(host):
            def asr_batch(windows, workdir):
                proc = subprocess.Popen(SLEEPER, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.children.append(proc)
                host.register_process(proc)
                started.set()
                if proc.wait() != 0:
                    raise RuntimeError(f"ASR model process exited with status {proc.returncode}")
                return [{"segments": [], "speech_intervals": [], "model": "A1+A2", "warnings": []}
                        for _ in windows]
            return self.deps(asr_batch)
        return factory


# --------------------------------------------------------------------------------------------------

class TurnHoldTests(Base):
    def test_hold_keeps_tick_from_restarting_restoration_until_resume(self):
        self.item(state="processing")
        worker = FakeWorker.start(self.svc, "it1", exit_delay=0.3)   # slow exit: the gap a tick could use
        host = self.host()
        ticking = threading.Event()
        stop_ticks = threading.Event()

        def tick_loop():                         # the queue tick, as fast as it can
            while not stop_ticks.is_set():
                self.svc.maybe_start_worker("it1")
                ticking.set()
                time.sleep(0.005)
        t = threading.Thread(target=tick_loop, daemon=True)
        t.start()
        ticking.wait(1)
        self.assertTrue(host.stop_restore())
        self.assertIn(("it1", "preempt"), self.svc.stops)
        self.assertTrue(worker.cancel.is_set())
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.svc._subtitle_turn_hold, "it1")
        time.sleep(0.05)
        self.assertEqual(self.svc.worker_starts, [])            # gap: nothing restarted

        lease = host.acquire_lease(True)
        self.assertIsNotNone(lease)
        self.assertIs(self.svc._wk, lease)
        self.assertEqual(lease.stage, gh.RUN_STAGE_LEASE)
        self.assertEqual(len(self.svc._subtitle_windows), 1)
        host.release_lease(lease)
        self.assertIsNone(self.svc._wk)
        self.assertIsNotNone(self.svc._subtitle_windows[0][1])
        time.sleep(0.05)
        self.assertEqual(self.svc.worker_starts, [])            # released, but restoration not resumed yet
        host.resume_restore()
        deadline = time.monotonic() + 1
        while not self.svc.worker_starts and time.monotonic() < deadline:
            time.sleep(0.01)
        stop_ticks.set()
        t.join(1)
        self.assertEqual(self.svc.worker_starts, ["it1"])
        self.assertFalse(self.svc._subtitle_turn_held())
        self.svc._wk.cancel.set()

    def test_failed_lease_after_stop_is_undone_by_resume(self):
        """The executor's _begin_gpu_turn: stop_restore, acquire fails (player busy) -> resume_restore."""
        self.item(state="processing")
        FakeWorker.start(self.svc, "it1", exit_delay=0.0)
        host = self.host()
        ex = SimpleNamespace(host=host, _lease=None, _stopped_restore=False, _turn_started=None, switches=0,
                             clock=time.time, events=[], _event=lambda kind, **kw: ex.events.append(kind))
        self.svc.busy = True
        self.assertFalse(sl.LocalSliceExecutor._begin_gpu_turn(ex, True))
        self.assertEqual(ex.events, ["gpu_busy"])
        self.assertFalse(self.svc._subtitle_turn_held())
        self.assertEqual(host.yield_reason, "player_busy")
        self.svc.busy = False
        self.assertIsNotNone(self.svc.maybe_start_worker("it1"))   # restoration may start again
        self.svc._wk.cancel.set()

    def test_no_worker_to_stop_drops_hold_on_failure_and_on_release(self):
        self.item(state="done")
        host = self.host()
        self.assertFalse(host.stop_restore())
        self.assertTrue(self.svc._subtitle_turn_held())
        other = _Run("other")
        self.svc._wk = other                                  # GPU taken in between
        self.assertIsNone(host.acquire_lease(True))
        self.assertFalse(self.svc._subtitle_turn_held())
        self.svc._wk = None
        self.assertFalse(host.stop_restore())
        lease = host.acquire_lease(True)
        host.release_lease(lease)                             # executor calls no resume_restore (stopped=False)
        self.assertFalse(self.svc._subtitle_turn_held())

    def test_another_items_worker_is_not_stopped(self):
        self.item("it1", state="processing")
        other = FakeWorker.start(self.svc, "it2", exit_delay=0.0)
        host = self.host("it1")
        self.assertFalse(host.stop_restore())
        self.assertFalse(other.cancel.is_set())
        self.assertIsNone(host.acquire_lease(True))
        self.assertFalse(self.svc._subtitle_turn_held())
        other.cancel.set()

    def test_shared_lease_leaves_restoration_in_wk(self):
        self.item(state="processing")
        worker = FakeWorker.start(self.svc, "it1")
        host = self.host()
        lease = host.acquire_lease(False)
        self.assertIsInstance(lease, gh._SharedLease)
        self.assertIs(self.svc._wk, worker)
        host.release_lease(lease)
        self.assertIs(self.svc._wk, worker)
        worker.cancel.set()

    def test_lease_refused_while_paused_perf_or_model_test(self):
        self.item()
        host = self.host()
        for attr, value, reason in (("_paused_all", True, "queue_paused_or_stopping"),
                                    ("_subtitle_model_test_busy", True, "model_test")):
            setattr(self.svc, attr, value)
            self.assertIsNone(host.acquire_lease(True))
            self.assertEqual(host.yield_reason, reason)
            setattr(self.svc, attr, False)
        self.svc.perf.holding = True
        self.assertIsNone(host.acquire_lease(False))
        self.svc.perf.holding = False
        self.assertIsNotNone(host.acquire_lease(False))

    def test_close_releases_a_leftover_lease_and_hold(self):
        self.item(state="processing")
        FakeWorker.start(self.svc, "it1", exit_delay=0.0)
        host = self.host()
        self.assertTrue(host.stop_restore())
        lease = host.acquire_lease(True)
        self.assertIs(self.svc._wk, lease)
        host.close()
        self.assertIsNone(self.svc._wk)
        self.assertFalse(self.svc._subtitle_turn_held())


class RestoreInfoTests(Base):
    def stream_item(self, item_id="stream01"):
        item = self.item(item_id)
        item.kind = "stream"
        return item

    def set_live_report(self, **values):
        self.svc._clock = lambda: 1005.0
        self.svc.busy_expire = 30.0
        self.svc._live = {"stream_item": "stream01", "ahead_s": 45.0, "speed": 0.7,
                          "restore_done": False, **values}
        self.svc._live_ts = 1004.0
        self.svc._busy = True
        self.svc._busy_ts = 1004.0

    def test_running_has_work_lead_and_oom(self):
        it = self.item(state="processing")
        host = self.host()
        info = host.restore_info()
        self.assertFalse(info.running)
        self.assertTrue(info.has_work)
        self.assertEqual(info.lead_s, 0.0)                     # no segments.json yet
        worker = FakeWorker.start(self.svc, "it1")
        it.progress = {"frame": 100, "fps": 15.0, "source_fps": 30.0}
        info = host.restore_info()
        self.assertTrue(info.running)
        self.assertAlmostEqual(info.rate, 0.5)                 # worker fps / source fps before 5 s of samples
        self.assertFalse(info.oom)
        (self.svc.store.item_dir("it1") / "worker.log").write_text("x\ntorch.OutOfMemoryError: CUDA out of memory",
                                                                   "utf-8")
        self.assertTrue(host.restore_info().oom)
        worker.cancel.set()
        it.state = "done"
        info = host.restore_info()
        self.assertEqual(info.lead_s, float("inf"))
        self.assertFalse(info.has_work)

    def test_rate_from_frame_samples(self):
        it = self.item(state="processing")
        host = self.host()
        worker = FakeWorker.start(self.svc, "it1")
        now = [100.0]
        with mock.patch.object(gh.time, "monotonic", lambda: now[0]):
            it.progress = {"frame": 0, "fps": 99.0, "source_fps": 30.0}
            host.restore_info()
            now[0] += 10.0
            it.progress = {"frame": 150, "fps": 99.0, "source_fps": 30.0}
            self.assertAlmostEqual(host.restore_info().rate, 0.5)     # 150 frames / 10 s / 30 fps
        worker.cancel.set()

    def test_stream_host_reads_fresh_matching_player_restore_telemetry(self):
        self.stream_item()
        self.set_live_report()
        host = self.svc._subtitle_new_gpu_host(self.svc.store.get("stream01"), None)
        self.assertIsInstance(host, gh.QueueStreamGpuHost)
        info = host.restore_info()
        self.assertEqual((info.lead_s, info.rate, info.running, info.has_work), (45.0, 0.7, True, True))

        # Existing callers of the common adapter also get the stream contract.
        common = gh.QueueGpuHost(self.svc, "stream01")
        self.assertEqual((common.restore_info().lead_s, common.restore_info().rate), (45.0, 0.7))

    def test_stream_restore_info_ignores_stale_and_other_stream_reports(self):
        self.stream_item()
        self.set_live_report()
        host = gh.QueueStreamGpuHost(self.svc, "stream01")
        self.svc._busy_ts = 974.0
        stale_busy = host.restore_info()
        self.assertEqual((stale_busy.lead_s, stale_busy.rate, stale_busy.running, stale_busy.has_work),
                         (45.0, 0.7, False, True))

        self.svc._busy_ts = 1004.0
        self.svc._live_ts = 974.0
        stale = host.restore_info()
        self.assertEqual((stale.lead_s, stale.rate, stale.running, stale.has_work), (0.0, None, False, True))

        self.svc._live_ts = 1004.0
        self.svc._live["stream_item"] = "different-stream"
        other = host.restore_info()
        self.assertEqual((other.lead_s, other.rate, other.running, other.has_work), (0.0, None, False, True))

    def test_stream_restore_info_handles_unknown_metrics_and_completion(self):
        self.stream_item()
        self.set_live_report(ahead_s=float("nan"), speed=True)
        host = gh.QueueStreamGpuHost(self.svc, "stream01")
        unknown = host.restore_info()
        self.assertEqual((unknown.lead_s, unknown.rate, unknown.running, unknown.has_work), (0.0, None, True, True))

        self.set_live_report(ahead_s=12.5, speed=0.8, restore_done=True)
        finished = host.restore_info()
        self.assertEqual((finished.lead_s, finished.rate, finished.running, finished.has_work), (12.5, None, False, False))

    def test_start_local_selects_stream_gpu_host(self):
        item = self.stream_item()
        finished = threading.Event()
        with mock.patch.object(self.svc, "_subtitle_new_local_executor", return_value=object()), \
                mock.patch.object(self.svc, "_run_subtitle_local", new=lambda *_args: finished.set()):
            self.svc._subtitle_start_local(item)
            run = self.svc._subtitle_local
            self.assertIsInstance(run.host, gh.QueueStreamGpuHost)
            self.assertTrue(finished.wait(2.0))
            run.thread.join(2.0)
            run.host.close()
            self.svc._subtitle_local = None

    def test_gpu_info_cached_and_missing_nvidia_smi(self):
        svc = GpuService(self.root)
        svc._subtitle_gpu_info_cache = None
        with mock.patch.object(gh.subprocess, "run", side_effect=FileNotFoundError) as run:
            self.assertEqual(svc._subtitle_gpu_info(), sl.GpuInfo())
            svc._subtitle_gpu_info()
        self.assertEqual(run.call_count, 1)
        with mock.patch.object(gh.subprocess, "run",
                               return_value=SimpleNamespace(stdout="NVIDIA GeForce RTX 5060 Laptop GPU, 8151\n")):
            self.assertEqual(gh.gpu_identity(), sl.GpuInfo("NVIDIA GeForce RTX 5060 Laptop GPU", 8151.0))


class ProcessRegistryTests(Base):
    def test_registered_after_cancel_is_killed_at_once(self):
        host = self.host()
        host.run.cancel.set()
        proc = subprocess.Popen(SLEEPER, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.children.append(proc)
        host.register_process(proc)
        self.assertIsNotNone(proc.wait(5))
        self.assertIs(host.run.proc, proc)

    def test_request_stop_kills_the_run_and_the_lease_process(self):
        self.item(state="done")
        for owner in ("run", "lease"):
            with self.subTest(owner=owner):
                host = self.host()
                lease = host.acquire_lease(True)
                proc = subprocess.Popen(SLEEPER, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.children.append(proc)
                host.register_process(proc)
                self.assertIs(lease.proc, proc)
                self.svc._request_stop(host.run if owner == "run" else lease, "auto")
                self.assertIsNotNone(proc.wait(5))
                host.release_lease(lease)

    def test_pipeline_child_and_llama_server_hooks(self):
        from oh_streamer.subtitles import llm, pipeline
        seen = []

        class Proc:
            stdout = iter(())
            returncode = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def wait(self, *a, **k):
                return 0

            def poll(self):
                return 1

        with mock.patch.object(pipeline.subprocess, "Popen", return_value=Proc()):
            pipeline.run_asr_child({"folder": str(self.root / "asr")}, on_process=seen.append, python="py")
        self.assertEqual(len(seen), 1)
        model = self.root / "m.gguf"
        model.write_bytes(b"x")
        client = llm.LlamaServerClient("sakura", model, sys.executable, on_process=seen.append, startup_timeout=1)
        with mock.patch.object(llm.subprocess, "Popen", return_value=Proc()):
            with self.assertRaises(llm.LlamaServerError):          # the fake exits during startup
                client.start()
        self.assertEqual(len(seen), 2)
        # windows runner forwards the hook to its default child runner
        with mock.patch.object(pipeline, "run_asr_child") as child:
            pipeline.run_production_asr_windows([], models_dir=self.root, on_process=seen.append,
                                                prepare_plan=lambda *a, **k: {})
        child.assert_not_called()                                  # no windows: no child; signature accepted


class LocalCancelTests(Base):
    """R1: a local ASR in progress is killed by the item's stop, the thread ends within 1 s, nothing failed."""

    def start_running_asr(self):
        self.item(state="done")
        started = threading.Event()
        self.svc.deps_factory = self.sleeping_asr(started)
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_local
        self.assertIsNotNone(run)
        self.assertTrue(started.wait(20), "ASR child never started")
        self.assertEqual(run.host._lease.stage, gh.RUN_STAGE_LEASE)
        return run

    def assert_clean_stop(self, run, t0):
        run.thread.join(5)
        elapsed = time.monotonic() - t0
        self.assertFalse(run.thread.is_alive())
        self.assertLess(elapsed, 1.0)
        self.assertTrue(all(p.poll() is not None for p in self.children))
        self.assertIsNone(self.svc._subtitle_local)
        self.assertIsNone(self.svc._wk)
        self.assertFalse(self.svc._subtitle_turn_held())
        self.assertEqual(self.svc._subtitle_results["it1"][gh.LOCAL].reason, "cancelled")
        plan = sp.load_slice_plan(self.svc.store.item_dir("it1") / "subtitles")
        asr = [s.get(sp.STAGE_ASR, plan.primary) for s in plan.slices]
        self.assertTrue(all(st.status == sp.PENDING for st in asr), [st.status for st in asr])
        self.assertTrue(all(not st.attempt for st in asr))

    def test_stop_executors_kills_asr_and_returns_fast(self):
        run = self.start_running_asr()
        t0 = time.monotonic()
        self.assertTrue(self.svc._subtitle_stop_executors("it1", "remove", wait=1.0))
        self.assert_clean_stop(run, t0)

    def test_request_stop_alone_is_enough_for_remove(self):
        """QueueService.remove() calls _request_stop on every run of the item, then joins (15 s limit)."""
        run = self.start_running_asr()
        t0 = time.monotonic()
        for r in (self.svc._wk, run):
            self.svc._request_stop(r, "remove")
        self.assert_clean_stop(run, t0)

    def test_step_stopping_the_lease_for_the_player_yields(self):
        """_step: `busy and wk.stage != 'restore'` -> _request_stop(wk, 'auto') on the lease."""
        run = self.start_running_asr()
        t0 = time.monotonic()
        self.svc._request_stop(self.svc._wk, "auto")
        self.assert_clean_stop(run, t0)
        self.assertEqual(run.host.yield_reason, "lease_stopped")


class StreamLocalTests(Base):
    """T4.2 B1: the live stream item's own local subtitles (stream_items in the tick, player-busy exemption,
    stream TurnConfig, no exclusive lease before the player confirmed its restoration pause)."""

    def stream_item(self, item_id="stream01", audio="ready"):
        item = self.item(item_id)
        item.kind = "stream"
        item.stream = {"audio_status": audio}
        marker = self.root / "audio.rstream"                  # what queue/stream_audio.py attaches
        marker.write_text("{}", "utf-8")
        item.source_path = str(marker)
        store = self.svc.store                                # like QueueStore: stream items are not in items()
        store.items = lambda: [i for i in store.items_by_id.values() if getattr(i, "kind", "file") != "stream"]
        store.stream_items = lambda: [i for i in store.items_by_id.values() if getattr(i, "kind", "file") == "stream"]
        return item

    def live(self, stream_item="stream01", busy=True):
        self.svc._clock = lambda: 1005.0
        self.svc.busy_expire = 30.0
        self.svc._live = {"stream_item": stream_item, "ahead_s": 45.0, "speed": 0.7, "restore_done": False}
        self.svc._live_ts = 1004.0
        self.svc.busy = busy

    def test_stream_item_local_wanted_while_player_busy_streaming_it(self):
        stream = self.stream_item()
        self.live()
        self.assertTrue(self.svc.is_player_busy())
        self.assertEqual(self.svc._subtitle_local_gate_reason(stream), "")
        self.assertTrue(self.svc._subtitle_local_wanted(stream, self.svc.now))

    def test_other_items_stay_blocked_while_the_player_is_busy(self):
        self.stream_item()
        other = self.item("it1", state="done")
        self.live()
        self.assertEqual(self.svc._subtitle_local_gate_reason(other), "player_busy")
        self.assertFalse(self.svc._subtitle_local_wanted(other, self.svc.now))
        self.assertEqual(self.host("it1")._lease_blocked(other), "player_busy")

    def test_exemption_needs_a_fresh_report_for_that_very_stream(self):
        stream = self.stream_item()
        self.live(stream_item="different-stream")
        self.assertEqual(self.svc._subtitle_local_gate_reason(stream), "player_busy")
        self.live()
        self.svc._live_ts = 960.0                                      # older than busy_expire
        self.assertEqual(self.svc._subtitle_local_gate_reason(stream), "player_busy")
        self.live()
        self.svc._live = None
        self.assertEqual(self.svc._subtitle_local_gate_reason(stream), "player_busy")

    def test_stream_local_waits_for_the_subtitle_audio(self):
        stream = self.stream_item(audio="probing")
        self.live()
        self.assertFalse(self.svc._subtitle_local_wanted(stream, self.svc.now))
        stream.stream["audio_status"] = "ready"
        self.assertTrue(self.svc._subtitle_local_wanted(stream, self.svc.now))

    def test_tick_starts_the_watched_stream_item_before_queue_items(self):
        stream = self.stream_item()
        self.item("it1", state="done")
        self.live()
        self.svc._watched = {"stream01"}
        with mock.patch.object(self.svc, "_subtitle_start_local") as start:
            self.svc._subtitle_local_tick()
        start.assert_called_once()
        self.assertEqual(start.call_args.args[0].id, stream.id)

    def test_tick_keeps_the_stream_runs_thread_while_busy_but_stops_another_items(self):
        self.stream_item()
        self.item("it1")
        self.live()
        for item_id, expected in (("stream01", None), ("it1", "player_busy")):
            run = _Run(item_id)
            run.stage = gh.RUN_STAGE_LOCAL
            run.host = gh.QueueGpuHost(self.svc, item_id, run)
            run.executor = SimpleNamespace(stop=lambda: None)
            self.svc._subtitle_local = run
            self.svc._subtitle_local_tick()
            self.assertEqual(run.cancel.is_set(), expected is not None, item_id)
            self.assertEqual(run.host.yield_reason, expected or "")
            self.svc._subtitle_local = None

    def test_stream_exclusive_lease_refused_until_the_pause_is_confirmed(self):
        self.stream_item()
        self.live()
        host = gh.QueueStreamGpuHost(self.svc, "stream01", _Run("stream01"))
        self.assertFalse(host._player_pause_confirmed())
        self.assertIsNone(host.acquire_lease(True))
        self.assertEqual(host.yield_reason, "player_restore_running")
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.svc._subtitle_windows, [])
        host._player_paused = True                                       # B2 sets this from the IPC
        lease = host.acquire_lease(True)
        self.assertIsNotNone(lease)
        self.assertIs(self.svc._wk, lease)
        self.assertEqual(lease.stage, gh.RUN_STAGE_LEASE)
        host.release_lease(lease)
        self.assertIsNone(self.svc._wk)

    def test_stream_exclusive_lease_still_needs_a_free_gpu_slot(self):
        self.stream_item()
        self.live()
        host = gh.QueueStreamGpuHost(self.svc, "stream01", _Run("stream01"))
        host._player_paused = True
        worker = FakeWorker.start(self.svc, "it9")
        self.assertIsNone(host.acquire_lease(True))
        self.assertEqual(host.yield_reason, "gpu_in_use")
        worker.cancel.set()

    def test_stream_shared_lease_needs_no_pause(self):
        self.stream_item()
        self.live()
        host = gh.QueueStreamGpuHost(self.svc, "stream01", _Run("stream01"))
        self.assertIsInstance(host.acquire_lease(False), gh._SharedLease)

    def test_stream_turn_config_low_watermark_600_and_user_override(self):
        base = gt.TurnConfig()
        cfg = gh.stream_turn_config({})
        self.assertEqual(cfg.low_watermark_s, 600.0)
        self.assertEqual(dataclasses.replace(cfg, low_watermark_s=base.low_watermark_s, media_turns=False,
                                             subtitle_load_s=base.subtitle_load_s,
                                             restore_restart_s=base.restore_restart_s), base)   # nothing else moved
        self.assertEqual((cfg.subtitle_load_s, cfg.restore_restart_s), (20.0, 10.0))      # stream defaults (second round)
        cfg = gh.stream_turn_config({"subtitle_local_subtitle_load_s": 5, "subtitle_local_restore_restart_s": 3})
        self.assertEqual((cfg.subtitle_load_s, cfg.restore_restart_s), (5.0, 3.0))        # the user's setting wins
        self.assertTrue(cfg.media_turns)                                # stream turns are sized in media seconds
        self.assertEqual((cfg.turn_first_s, cfg.turn_growth, cfg.turn_max_s), (60.0, 2.0, 300.0))
        self.assertEqual(gh.stream_turn_config({"subtitle_local_turn_first_s": 90}).turn_first_s, 90.0)
        self.assertEqual(gh.stream_turn_config(None).low_watermark_s, 600.0)
        cfg = gh.stream_turn_config({"subtitle_local_urgent_s": 200, "subtitle_local_low_watermark_s": None})
        self.assertEqual((cfg.urgent_s, cfg.low_watermark_s), (200.0, 600.0))
        self.assertEqual(gh.stream_turn_config({"subtitle_local_low_watermark_s": "450"}).low_watermark_s, 450.0)

    def test_executor_gets_the_stream_turns_only_for_stream_items(self):
        stream = self.stream_item()
        other = self.item("it1")
        self.svc.deps_factory = lambda host: object()
        with mock.patch.object(gh.sl, "LocalSliceExecutor") as executor:
            self.svc._subtitle_new_local_executor(stream, gh.QueueStreamGpuHost(self.svc, "stream01"))
            self.assertEqual(executor.call_args.kwargs["turns"].low_watermark_s, 600.0)
            self.svc._subtitle_new_local_executor(other, gh.QueueGpuHost(self.svc, "it1"))
            self.assertIsNone(executor.call_args.kwargs["turns"])        # LocalSliceExecutor's own default (900)


class LocalTickTests(Base):
    def instant_deps(self):
        def factory(host):
            def asr_batch(windows, workdir):
                return [{"segments": [{"start": a + 1.0, "end": a + 2.0, "text": "テスト"}],
                         "speech_intervals": [[a, b]], "model": "A1+A2", "warnings": []} for a, b in windows]
            return self.deps(asr_batch)
        return factory

    def test_only_local_work_starts_the_local_thread(self):
        self.svc.store.settings["subtitle_cloud_enabled"] = False     # keep the cloud half quiet
        self.svc.store.add(make_item("cloudy", plan=frozen(asr=CLOUD_ASR, mt=CLOUD_MT), source=str(self.source)))
        self.svc.deps_factory = self.instant_deps()
        self.svc._subtitle_executor_tick()
        self.assertIsNone(self.svc._subtitle_local)
        self.assertEqual(self.svc.deps_built, 0)                       # heavy deps never built for cloud work
        self.item("it1", state="done")
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_local
        self.assertIsNotNone(run)
        self.assertEqual(run.item_id, "it1")
        self.assertEqual(run.stage, gh.RUN_STAGE_LOCAL)
        run.thread.join(20)
        self.assertEqual(self.svc.deps_built, 1)
        self.assertEqual(self.svc._subtitle_results["it1"][gh.LOCAL].reason, "complete")
        self.assertIsNone(self.svc._subtitle_local)
        self.svc._subtitle_executor_tick()                             # complete: gate closed until reset
        self.assertIsNone(self.svc._subtitle_local)
        self.assertEqual(self.svc.deps_built, 1)
        kinds = [e["kind"] for e in run.executor.events]
        self.assertIn("gpu_acquired", kinds)
        self.assertIn("gpu_released", kinds)

    def test_player_busy_blocks_start_and_stops_a_running_thread(self):
        self.item(state="done")
        self.svc.busy = True
        self.svc.deps_factory = self.instant_deps()
        self.svc._subtitle_executor_tick()
        self.assertIsNone(self.svc._subtitle_local)
        self.svc.busy = False
        started = threading.Event()
        self.svc.deps_factory = self.sleeping_asr(started)
        self.svc._subtitle_executor_tick()
        run = self.svc._subtitle_local
        self.assertTrue(started.wait(20))
        self.svc.busy = True
        t0 = time.monotonic()
        self.svc._subtitle_executor_tick()                             # the local side gives way
        run.thread.join(5)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(run.host.yield_reason, "player_busy")
        self.assertIn(("it1", "auto"), self.svc.stops)
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.svc._subtitle_results["it1"][gh.LOCAL].reason, "cancelled")
        self.svc.now += 2                                               # cancelled: 1 s gate, then player gate
        self.svc._subtitle_executor_tick()
        self.assertIsNone(self.svc._subtitle_local)

    def test_other_items_restoration_keeps_interleave_thread_off(self):
        self.item("it1", state="done")
        worker = FakeWorker.start(self.svc, "it2")
        self.svc.deps_factory = self.instant_deps()
        self.svc._subtitle_executor_tick()
        self.assertIsNone(self.svc._subtitle_local)
        self.svc.store.settings[gh.RESOURCE_MODE_KEY] = "parallel"     # parallel shares the GPU: may start
        self.assertTrue(self.svc._subtitle_local_wanted(self.svc.store.get("it1"), self.svc.now))
        worker.cancel.set()

    def test_local_work_held_by_the_horizon_does_not_start_the_local_thread(self):
        """sample video A on the local side: the same dispatchable rule as the cloud host."""
        self.item(audio_offset=0.0)
        folder = self.svc.store.item_dir("it1") / "subtitles"
        folder.mkdir(parents=True)
        sp.save_slice_plan(sample_a_plan(), folder)
        self.svc._watched = {"it1"}
        self.svc._subtitle_clocks["it1"] = (SAMPLE_A_PLAY_POS, True)
        self.assertFalse(self.svc._subtitle_local_wanted(self.svc.store.get("it1"), self.svc.now))
        self.svc._subtitle_clocks["it1"] = (2000.0, True)              # buffer below the low watermark
        self.assertTrue(self.svc._subtitle_local_wanted(self.svc.store.get("it1"), self.svc.now))

    def test_resource_mode_setting(self):
        it = self.item()
        self.assertEqual(self.svc._subtitle_resource_mode(it), "auto")
        self.svc.store.settings[gh.RESOURCE_MODE_KEY] = "interleave"
        self.assertEqual(self.svc._subtitle_resource_mode(it), "interleave")
        it.settings[gh.RESOURCE_MODE_KEY] = "Parallel"
        self.assertEqual(self.svc._subtitle_resource_mode(it), "parallel")
        it.settings[gh.RESOURCE_MODE_KEY] = "nonsense"
        self.assertEqual(self.svc._subtitle_resource_mode(it), "interleave")

    def test_runs_include_local_slot(self):
        run = _Run("it1")
        self.svc._subtitle_local = run
        self.assertEqual(self.svc._subtitle_runs(), [run])


if __name__ == "__main__":
    unittest.main()
