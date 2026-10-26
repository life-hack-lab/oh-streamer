"""Tests for queue/subtitle_local.py + queue/gpu_turns.py (unified scheduler, M3: local GPU executor) and the
shared ASR pre-roll of the cloud executor. No GPU: fake ASR / translation / GpuHost, fake clock."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import stream_subtitle_gpu as ssg
from oh_streamer.queue import subtitle_local as sl
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr

ASR_ID = {"provider": "local", "asr_mode": "dual", "models": {"a": "x"}}
MT_ID = {"provider": "local", "backend": "sakura", "model": "m"}


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def every_30s_silence(_source, start, end):
    return [(30.0 * i - 0.5, 30.0 * i + 0.5) for i in range(int(start // 30), int(end // 30) + 2)
            if start <= 30.0 * i <= end]


class FakeModels:
    """asr_batch: a 3 s cue every 4 s over each window (a cue running past the window end is cut there, like
    real audio that stops); translation: one row per cue. Records the order of calls."""

    def __init__(self, clock=None):
        self.calls = []
        self.asr_errors = []          # consumed one per asr_batch call (None = succeed)
        self.mt_errors = []           # consumed one per translate call
        self.after_asr = None         # hook(fake) run after a successful asr_batch
        self.fail_head = 0            # this many asr_batch calls whose windows include slice 0 fail
        self.clock = clock
        self.asr_seconds = 0.0

    def asr_batch(self, windows, workdir):
        self.calls.append(("asr", [tuple(w) for w in windows]))
        if self.asr_errors:
            err = self.asr_errors.pop(0)
            if err:
                raise err
        if self.fail_head and any(a == 0.0 for a, _b in windows):
            self.fail_head -= 1
            raise RuntimeError("model crashed")
        if self.clock is not None:
            self.clock.sleep(self.asr_seconds)
        out = []
        for a, b in windows:
            segs, k = [], 0
            while a + 0.5 + 4 * k < b:
                s = a + 0.5 + 4 * k
                e = min(s + 3.0, b)
                segs.append({"start": round(s, 3), "end": round(e, 3), "text": f"これは{s:.1f}の文です"})
                k += 1
            out.append({"segments": segs, "speech_intervals": [[a, b]], "model": "A1+A2", "warnings": []})
        if self.after_asr:
            self.after_asr(self)
        return out

    def mt_session(self):
        outer = self

        @contextlib.contextmanager
        def session():
            outer.calls.append(("mt_open",))

            def translate(ja_doc, workdir):
                rows = ja_doc["segments"]
                outer.calls.append(("mt", rows[0]["start"] if rows else None))
                if outer.mt_errors:
                    err = outer.mt_errors.pop(0)
                    if err:
                        raise err
                return [{"start": r["start"], "end": r["end"], "text": "這是" + r["text"][3:-3] + "的句子",
                         "text_ja": r["text"]} for r in rows]
            try:
                yield translate
            finally:
                outer.calls.append(("mt_close",))
        return session()

    def kinds(self):
        return [c[0] for c in self.calls]


class FakeGpuHost(sl.GpuHost):
    def __init__(self, total_mib=8151.0, name="RTX Test"):
        self.restore = sl.RestoreInfo(lead_s=math.inf, has_work=False)
        self.context = sp.SchedulerContext(item_state="done")
        self.gpu = sl.GpuInfo(name, total_mib)
        self.busy = False
        self.restore_running = True
        self.stops = self.resumes = 0
        self.leases = []              # (exclusive, lease)
        self.released = []
        self.published = []
        self.cancel_after = None
        self._checks = 0
        self.limit = None

    def source_state(self):
        return True, self.limit, 1000.0

    def media_id(self):
        return "hls:test"

    def scheduler_context(self):
        return self.context

    def check_cancel(self):
        self._checks += 1
        if self.cancel_after is not None and self._checks > self.cancel_after:
            raise RuntimeError("cancel")

    def publish(self, result):
        self.published.append(result)

    def acquire_lease(self, exclusive):
        if self.busy:
            return None
        lease = object()
        self.leases.append((exclusive, lease))
        return lease

    def release_lease(self, lease):
        self.released.append(lease)

    def stop_restore(self):
        self.stops += 1
        was, self.restore_running = self.restore_running, False
        return was

    def resume_restore(self):
        self.resumes += 1
        self.restore_running = True

    def restore_info(self):
        return self.restore

    def gpu_info(self):
        return self.gpu


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.folder = self.root / "subtitles"
        self.source = self.root / "movie.mp4"
        self.source.write_bytes(b"x" * 5000)
        self.clock = FakeClock()
        self.fake = FakeModels(self.clock)
        self.host = FakeGpuHost()
        self.memory = gt.GpuModeMemory(self.root / "modes.json")

    def deps(self, **over):
        values = dict(asr_batch=self.fake.asr_batch, mt_session=self.fake.mt_session, asr_identity=ASR_ID,
                      mt_identity=MT_ID, probe_duration=lambda s: 1000.0, detect_silences=every_30s_silence,
                      asr_params={"pipeline": "test"}, mt_params={"batch_size": 30})
        values.update(over)
        return sl.LocalDeps(**values)

    def executor(self, **kw):
        kw.setdefault("asr_window", sp.AsrWindow())  # explicit unaligned fixture windows; production defaults to 60 s
        kw.setdefault("deps", self.deps())
        kw.setdefault("host", self.host)
        kw.setdefault("clock", self.clock)
        kw.setdefault("sleep", self.clock.sleep)
        kw.setdefault("memory", self.memory)
        return sl.LocalSliceExecutor(self.folder, self.source, **kw)

    def run_ex(self, wait=False, **kw):
        ex = self.executor(**kw)
        return ex, ex.run(wait=wait)

    def statuses(self, plan):
        return {(s.index, st): s.get(st).status for s in plan.slices for st in sp.STAGES}


# --------------------------------------------------------------------------------------------------
# Pure policy
# --------------------------------------------------------------------------------------------------

class DecideTurnTests(unittest.TestCase):
    cfg = gt.TurnConfig()

    def turn(self, sub, restore, watched, **kw):
        return gt.decide_gpu_turn(gt.TurnState(sub, restore, watched, **kw), self.cfg)

    def test_defaults_are_the_design_values(self):
        self.assertEqual((self.cfg.urgent_s, self.cfg.low_watermark_s, self.cfg.batch_slices,
                          self.cfg.urgent_batch_slices, self.cfg.unwatched_restore_ahead_s,
                          self.cfg.urgent_min_restore_s), (300.0, 900.0, 2, 1, 300.0, 150.0))
        cfg = gt.TurnConfig.from_settings({"subtitle_local_urgent_min_restore_s": 0})
        self.assertEqual(gt.decide_gpu_turn(gt.TurnState(0, 0, True), cfg).reason, "urgent")

    def test_table(self):
        R, S = gt.RESTORE, gt.SUBTITLES
        cases = [
            # sub, restore, watched, extra -> kind, slices, reason
            ((0, 1000, True, {"subtitle_work": False}), (R, 0, "no_subtitle_work")),
            ((0, math.inf, True, {}), (S, 2, "restore_idle")),
            ((5000, 10, False, {"restore_work": False}), (S, 2, "restore_idle")),
            ((299, 0, True, {}), (R, 0, "urgent_restore_low")),
            ((299, 149, True, {}), (R, 0, "urgent_restore_low")),
            ((299, 150, True, {}), (S, 1, "urgent")),
            ((299, 5000, True, {}), (S, 1, "urgent")),
            ((300, 300, True, {}), (S, 2, "low_watermark")),
            ((899, 1000, True, {}), (S, 2, "low_watermark")),
            ((600, 299, True, {}), (R, 0, "watched_restore")),
            ((900, 5000, True, {}), (R, 0, "watched_restore")),
            ((0, 299, False, {}), (R, 0, "unwatched_restore")),
            ((0, 300, False, {}), (S, 2, "restore_ahead")),
            ((1200, 1499, False, {}), (R, 0, "unwatched_restore")),
            ((1200, 1500, False, {}), (S, 2, "restore_ahead")),
        ]
        for (sub, restore, watched, extra), (kind, n, reason) in cases:
            with self.subTest(sub=sub, restore=restore, watched=watched, **extra):
                got = self.turn(sub, restore, watched, **extra)
                self.assertEqual((got.kind, got.slices, got.reason), (kind, n, reason))
                self.assertEqual(got.urgent, reason == "urgent")

    def test_everything_configurable(self):
        cfg = gt.TurnConfig.from_settings({"subtitle_local_urgent_s": 60, "subtitle_local_batch_slices": 3,
                                           "subtitle_local_unwatched_restore_ahead_s": 100})
        self.assertEqual((cfg.urgent_s, cfg.batch_slices, cfg.unwatched_restore_ahead_s), (60.0, 3, 100.0))
        self.assertEqual(gt.decide_gpu_turn(gt.TurnState(100, 1000, True), cfg).reason, "low_watermark")
        self.assertEqual(gt.decide_gpu_turn(gt.TurnState(0, 150, False), cfg).slices, 3)


class ResourceModeTests(unittest.TestCase):
    def test_vram_estimate(self):
        self.assertAlmostEqual(gt.vram_needed_mib("dual"), 2910 + 4801 + 1536)
        self.assertEqual(gt.choose_resource_mode("auto", 8151)[0], gt.MODE_INTERLEAVE)      # 8 GB: interleave
        self.assertEqual(gt.choose_resource_mode("auto", 12282)[0], gt.MODE_PARALLEL)
        self.assertEqual(gt.choose_resource_mode("auto", None), (gt.MODE_INTERLEAVE, "vram_unknown"))

    def test_table_overridable_and_explicit_settings_win(self):
        small = {"restore": 1000, "asr_dual": 2000, "mt_sakura_7b": 2000, "margin": 500}
        self.assertEqual(gt.choose_resource_mode("auto", 8151, requirements=small)[0], gt.MODE_PARALLEL)
        self.assertEqual(gt.choose_resource_mode("parallel", 4000)[0], gt.MODE_PARALLEL)
        self.assertEqual(gt.choose_resource_mode("interleave", 48000)[0], gt.MODE_INTERLEAVE)

    def test_remembered_demotion(self):
        mode, reason = gt.choose_resource_mode("auto", 24000, remembered={"mode": "interleave", "reason": "oom"})
        self.assertEqual((mode, reason), (gt.MODE_INTERLEAVE, "remembered:oom"))
        self.assertEqual(gt.choose_resource_mode("parallel", 24000, remembered={"mode": "interleave"})[0],
                         gt.MODE_PARALLEL)

    def test_guard(self):
        self.assertEqual(gt.parallel_guard(gt.GuardInput(oom=True)), "oom")
        slow = gt.GuardInput(watched=True, restore_rate=0.8, restore_lead_s=200)
        self.assertEqual(gt.parallel_guard(slow), "restore_too_slow")
        for ok in (gt.GuardInput(watched=False, restore_rate=0.1, restore_lead_s=0),       # not watched
                   gt.GuardInput(watched=True, restore_rate=0.8, restore_lead_s=400),      # still safe lead
                   gt.GuardInput(watched=True, restore_rate=1.2, restore_lead_s=10),       # keeps up
                   gt.GuardInput(watched=True, restore_rate=None, restore_lead_s=10)):     # unknown
            self.assertIsNone(gt.parallel_guard(ok))

    def test_oom_text(self):
        self.assertTrue(gt.is_cuda_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"))
        self.assertTrue(gt.is_cuda_oom("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"))
        self.assertFalse(gt.is_cuda_oom("connection refused"))

    def test_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = gt.GpuModeMemory(Path(tmp) / "m.json")
            key = gt.gpu_key("NVIDIA GeForce RTX 5060 Laptop GPU", 8151)
            self.assertEqual(key, "NVIDIA GeForce RTX 5060 Laptop GPU|8128")
            self.assertIsNone(mem.get(key))
            mem.remember(key, "interleave", "oom", "detail", at=5.0)
            self.assertEqual(gt.GpuModeMemory(Path(tmp) / "m.json").get(key)["reason"], "oom")
            mem.forget(key)
            self.assertIsNone(mem.get(key))


# --------------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------------

class BatchTests(Base):
    def test_whole_file_in_batches_of_two_asr_then_mt(self):
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "complete")
        self.assertEqual([(s.start, s.end) for s in ex.plan.slices],
                         [(0.0, 30.0), (30.0, 330.0), (330.0, 630.0), (630.0, 930.0), (930.0, 1000.0)])     # T3.6: 30 s first
        self.assertEqual(self.fake.calls, [
            ("asr", [(0.0, 30.0)]), ("mt_open",), ("mt", 0.5), ("mt_close",),
            ("asr", [(15.0, 330.0), (315.0, 630.0)]), ("mt_open",), ("mt", 27.5), ("mt", 327.5), ("mt_close",),
            ("asr", [(615.0, 930.0), (915.0, 1000.0)]), ("mt_open",), ("mt", 627.5), ("mt", 927.5), ("mt_close",)])
        self.assertTrue(res.publication["complete"])
        self.assertTrue((self.folder / "sakura.zh-TW.srt").is_file())
        self.assertTrue(all(s.all_done() for s in ex.plan.slices))

    def test_each_cue_published_once_and_crossing_cue_owned_by_later_slice(self):
        ex, res = self.run_ex()
        rows = []
        for s in ex.plan.slices:
            doc = sr.read_asr(self.folder, ex._asr_key(s))
            rows += [dict(r, start=r["start"] + s.start, end=r["end"] + s.start) for r in doc["segments"]]
        texts = [r["text"] for r in rows]
        self.assertEqual(len(texts), len(set(texts)))
        # The same fixed per-slice pre-roll is used even when two slices share a GPU turn.
        doc1 = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[1]))
        self.assertEqual(doc1["segments"][0]["start"], -2.5)
        # batch boundary at 390: the earlier run's copy (388.5-390 cut off by the audio end) is dropped, the later
        # run's pre-roll copy is kept
        self.assertFalse(any(abs(r["end"] - 390.0) < 1e-6 for r in rows))
        self.assertTrue(any(r["start"] < 390.0 < r["end"] for r in rows))
        self.assertEqual(doc1["window"], sp.AsrWindow().cache_params())

    def test_separate_runs_keep_raw_preroll_copy_but_publication_drops_it(self):
        orig = self.fake.asr_batch

        def asr_batch(windows, workdir):
            out = orig(windows, workdir)
            for (a, _b), res in zip(windows, out):
                if a == 0.0:
                    res["segments"].append({"start": 85.0, "end": 89.6, "text": "前片完整句"})
                if a == 75.0:      # slice 1's pre-roll hears slice 0's last sentence split differently
                    res["segments"].append({"start": 85.0, "end": 89.5, "text": "別の切り方"})
                    res["segments"].append({"start": 85.0, "end": 91.0, "text": "跨切點的新尾句"})
            return out
        # the injected cues sit around a 90 s first cut: keep that first slice (the default is 30 s since T3.6)
        ex, res = self.run_ex(deps=self.deps(asr_batch=asr_batch), turns=gt.TurnConfig(batch_slices=1),
                              first_slice_seconds=90.0)
        self.assertEqual(res.reason, "complete")
        doc1 = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[1]))
        self.assertIn("別の切り方", [r["text"] for r in doc1["segments"]])
        per_slice = [(s.start, [dict(r, start=r["start"] + s.start, end=r["end"] + s.start)
                               for r in sr.read_asr(self.folder, ex._asr_key(s))["segments"]])
                     for s in ex.plan.slices]
        self.assertNotIn("別の切り方", [r["text"] for r in sr.resolve_preroll_overlaps(per_slice)])
        self.assertIn("跨切點的新尾句", [r["text"] for r in sr.resolve_preroll_overlaps(per_slice)])
        self.assertEqual(doc1["segments"][0]["start"], 87.5 - 90.0)         # cut-off sentence still kept here

    def test_later_slice_first_does_not_change_earlier_raw_asr(self):
        ex = self.executor()
        ex._prepare()
        sp.extend_lazy_cuts(ex.plan, self.source, None, detect=every_30s_silence)
        ex._media_id = "hls:test"
        nxt = ex.plan.slices[1]
        sr.write_asr(self.folder, ex._asr_key(nxt), {"segments": [{"start": -5.5, "end": -1.0, "text": "x"}]})
        ex._run_asr_batch([sp.Task(0, sp.STAGE_ASR, "primary")])
        before = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[0]))["segments"]
        (self.folder / sp.asr_cache_path(ex._asr_key(ex.plan.slices[0]))).unlink()
        (self.folder / sp.asr_cache_path(ex._asr_key(nxt))).unlink()
        ex._run_asr_batch([sp.Task(0, sp.STAGE_ASR, "primary")])
        after = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[0]))["segments"]
        self.assertEqual(before, after)

    def test_raw_asr_and_keys_are_independent_of_batch_size_order_and_restart(self):
        def execute(label, order, batch_size, restart=False):
            folder = self.root / label

            def prepare():
                ex = sl.LocalSliceExecutor(folder, self.source, deps=self.deps(), host=self.host,
                                           clock=self.clock, sleep=self.clock.sleep, memory=self.memory)
                ex._prepare()
                sp.extend_lazy_cuts(ex.plan, self.source, None, detect=every_30s_silence)
                return ex

            ex = prepare()
            for offset in range(0, len(order), batch_size):
                if restart and offset:
                    ex = prepare()
                ex._run_asr_batch([sp.Task(i, sp.STAGE_ASR, "primary")
                                   for i in order[offset:offset + batch_size]])
            docs = [sr.read_asr(folder, ex._asr_key(s)) for s in ex.plan.slices]
            return [ex._asr_key(s) for s in ex.plan.slices], [d["segments"] for d in docs]

        reference = execute("batch1", list(range(5)), 1)
        self.assertEqual(reference, execute("batch2", list(range(5)), 2))
        self.assertEqual(reference, execute("seek_order", [3, 1, 4, 0, 2], 2))
        self.assertEqual(reference, execute("resumed", list(range(5)), 1, restart=True))

    def test_real_adapter_keeps_vad_inputs_isolated_without_loading_gpu(self):
        inputs = []
        batch_sizes = []

        def extract(_source, path, start, end, **kwargs):
            path.write_text(json.dumps([start, end]), "utf-8")

        def production(jobs, **kwargs):
            batch_sizes.append(len(jobs))
            docs = []
            for job in jobs:
                window = json.loads(Path(job["source"]).read_text("utf-8"))
                inputs.append(window)
                # Model residency is shared, but VAD and outputs keep each entire input isolated.
                doc = {"segments": [{"start": 0.5, "end": 2.0, "text": f"input:{window}"}],
                       "speech_intervals": [[0.5, 2.0]], "model": "test", "asr_warnings": []}
                folder = Path(job["folder"])
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "ja.json").write_text(json.dumps(doc), "utf-8")
                docs.append(doc)
            return docs

        with patch.object(sl, "local_identities", return_value={"asr_identity": ASR_ID, "mt_identity": MT_ID}), \
                patch("oh_streamer.subtitles.runtime.resample_window_to_wav", side_effect=extract), \
                patch("oh_streamer.subtitles.pipeline.run_production_asr_windows_child", side_effect=production):
            deps = sl.default_local_deps(self.folder, self.source, {"translation": {"model_file": "m.gguf"}},
                                         runtime_paths={"models": self.root, "server": "unused"})
            windows = [(0.0, 300.0), (285.0, 600.0)]
            batched = deps.asr_batch(windows, self.root / "paired")
            isolated = [deps.asr_batch([w], self.root / f"single_{i}")[0] for i, w in enumerate(windows)]
        self.assertEqual(inputs, [[0.0, 300.0], [285.0, 600.0], [0.0, 300.0], [285.0, 600.0]])
        self.assertEqual(batch_sizes, [2, 1, 1])
        self.assertEqual(batched, isolated)
        self.assertEqual(batched[1]["segments"][0]["start"], 285.5)

    def test_seek_batch_uses_separate_windows_in_one_asr_call(self):
        self.host.context = sp.SchedulerContext(play_pos=700.0, watched=True)
        ex = self.executor(resource_mode="interleave")
        ex._prepare()
        ex._resolve_mode()
        sp.extend_lazy_cuts(ex.plan, self.source, None, detect=every_30s_silence)
        ex._run_batch(2, None)
        # the seek slice [630, 930] (audio from 615) + the next one; the head slice no longer jumps the queue once
        # the viewer is past it (T3.6: rule 2 skipped when watching with play_pos >= head.end)
        self.assertEqual(self.fake.calls[0], ("asr", [(615.0, 930.0), (915.0, 1000.0)]))
        self.assertEqual(self.fake.kinds(), ["asr", "mt_open", "mt", "mt", "mt_close"])

    def test_restart_reuses_cache_and_releases_running_work(self):
        ex, res = self.run_ex()
        plan = sp.load_slice_plan(self.folder)
        plan.mark_running(2, "mt", "local")
        plan.slices[3].set("asr", "primary", sp.TaskState())
        plan.slices[3].set("mt", "primary", sp.TaskState())
        sp.save_slice_plan(plan, self.folder)
        self.fake.calls.clear()
        ex2, res2 = self.run_ex()
        self.assertEqual(res2.reason, "complete")
        self.assertEqual(self.fake.calls, [("mt_open",), ("mt_close",)])    # all results came from the cache
        self.assertEqual(ex2.plan.slices[2].get("mt").status, sp.DONE)

    def test_cancel_releases_running_work_and_lease(self):
        self.host.cancel_after = 6
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "cancelled")
        self.assertFalse(any(st.status == sp.RUNNING for s in ex.plan.slices for st in s.tasks.values()))
        self.assertEqual(len(self.host.leases), len(self.host.released))
        self.assertFalse(any(st.status == sp.RUNNING for s in sp.load_slice_plan(self.folder).slices
                             for st in s.tasks.values()))

    def test_local_error_retried_then_failed_without_cloud_spill(self):
        self.fake.fail_head = 3
        ex, res = self.run_ex(wait=True)
        self.assertEqual(res.reason, "stuck")
        st = ex.plan.slices[0].get("asr")
        # M4b: the only (local) candidate gave up = failed(exhausted), its own reason kept in `tried`
        self.assertEqual((st.status, st.fail_kind, st.attempt), (sp.FAILED, "exhausted", 3))
        self.assertEqual(st.tried["local-asr"]["kind"], "error")
        self.assertIsNone(st.wait_executor)
        self.assertTrue(ex.plan.slices[2].all_done())        # other slices continue

    def test_local_errors_hand_off_to_the_cloud_backup(self):
        # M4b (replaces the M3 `handoff` hook): after 3 local errors the chain moves to the cloud backup
        frozen = {"models": {"asr": {"id": "local-asr", "kind": "asr", "provider": "local"},
                             "translation": {"id": "local-sakura", "kind": "translation", "provider": "local"}},
                  "fallback": {"asr": [{"id": "groq", "kind": "asr", "provider": "openai-compatible",
                                        "remote_model": "whisper"}], "translation": []}, "policy": {}}
        self.fake.fail_head = 3
        ex, res = self.run_ex(wait=True, plan_provider=lambda: frozen)
        st = ex.plan.slices[0].get("asr")
        self.assertEqual((st.status, st.wait_executor, st.candidate), (sp.PENDING, "cloud", "groq"))
        self.assertEqual(st.tried["local-asr"]["attempt"], 3)
        self.assertEqual((res.reason, res.waiting_executor), ("idle", "cloud"))
        self.assertEqual(ss.candidate_executor({"provider": "local"}), sl.LOCAL_EXECUTOR.name)

    def test_tasks_reserved_for_cloud_are_not_taken(self):
        ex = self.executor()
        ex._prepare()
        sp.extend_lazy_cuts(ex.plan, self.source, None, detect=every_30s_silence)
        for s in ex.plan.slices:
            ex.plan.wait_for(s.index, "asr", "cloud")
        ex._save()
        res = ex.run()
        self.assertEqual(res.reason, "idle")
        self.assertEqual(self.fake.calls, [])

    def test_publication_keeps_blocks_display_rule(self):
        # translate_file rows are already Traditional (OpenCC inside translate_segments); Taiwan wording on top
        row = sl.display_row({"start": 1, "end": 2, "text": "這是測試。", "text_ja": "テスト", "low_confidence": True})
        self.assertEqual(row["zh"], "這是測試")
        self.assertTrue(row["full_only"])
        self.assertEqual(row["full_prefix"], "（低信心）")
        hidden = sl.display_row({"start": 1, "end": 2, "text": "テスト", "text_ja": "テスト"})
        self.assertTrue(hidden["is_filtered"])


class InterleaveTests(Base):
    def setUp(self):
        super().setUp()
        self.host.restore = sl.RestoreInfo(lead_s=1000.0, rate=1.5, running=True, has_work=True)

    def test_no_urgent_switch_while_restore_lead_is_short(self):
        self.host.context = sp.SchedulerContext(play_pos=0.0, watched=True)
        self.host.restore = sl.RestoreInfo(lead_s=100.0, rate=1.5, running=True, has_work=True)
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual((res.reason, self.fake.calls, self.host.stops), ("restore_turn", [], 0))
        self.assertEqual(ex.last_turn.reason, "urgent_restore_low")

    def test_watched_urgent_one_slice_then_batches_until_low_watermark(self):
        self.host.context = sp.SchedulerContext(play_pos=0.0, watched=True)
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual(res.reason, "restore_turn")
        self.assertEqual([c[1] for c in self.fake.calls if c[0] == "asr"],
                         [[(0.0, 30.0)], [(15.0, 330.0)], [(315.0, 630.0), (615.0, 930.0)]])
        turns = [e["reason"] for e in ex.events if e["kind"] == "turn"]
        self.assertEqual(turns, ["urgent", "urgent", "low_watermark"])
        self.assertEqual((self.host.stops, self.host.resumes), (1, 1))      # one switch for three batches
        self.assertEqual(len(self.host.leases), 1)
        self.assertTrue(self.host.leases[0][0])                            # exclusive
        self.assertEqual(self.host.released, [self.host.leases[0][1]])
        self.assertEqual(ex.switches, 1)

    def test_unwatched_follows_restore_lead(self):
        self.host.restore = sl.RestoreInfo(lead_s=200.0, has_work=True)
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual(res.reason, "restore_turn")
        self.assertEqual(self.fake.calls[0], ("asr", [(0.0, 30.0)]))  # publish the short head first
        self.assertEqual(self.host.stops, 1)
        self.host.restore = sl.RestoreInfo(lead_s=700.0, has_work=True)
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual([c[1] for c in self.fake.calls if c[0] == "asr"],
                         [[(0.0, 30.0)], [(15.0, 330.0), (315.0, 630.0)]])
        self.assertEqual(res.reason, "restore_turn")                         # 700 - 930 < 300

    def test_gpu_busy_is_not_a_failure(self):
        self.host.restore = sl.RestoreInfo(lead_s=math.inf, has_work=False)
        self.host.busy = True
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual(res.reason, "gpu_busy")
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(all(st == sp.PENDING for st in self.statuses(ex.plan).values()))
        self.assertEqual(self.host.stops, self.host.resumes)                 # restore given back
        self.assertFalse(sp.derive_status(ex.plan).stuck)

    def test_oom_alone_on_the_gpu_is_a_counted_retry(self):
        self.host.restore = sl.RestoreInfo(lead_s=math.inf, has_work=False)
        self.fake.asr_errors = [sl.GpuOutOfMemory("CUDA out of memory")]
        ex, res = self.run_ex(resource_mode="interleave")
        self.assertEqual(ex.plan.slices[0].get("asr").attempt, 1)
        self.assertEqual(res.reason, "quota")                                # waits for the retry time
        ex, res = self.run_ex(wait=True, resource_mode="interleave")
        self.assertEqual(res.reason, "complete")
        self.assertIsNone(self.memory.get(gt.gpu_key("RTX Test", 8151)))


class ParallelTests(Base):
    def setUp(self):
        super().setUp()
        self.host.gpu = sl.GpuInfo("RTX Big", 16376.0)
        self.key = gt.gpu_key("RTX Big", 16376.0)
        self.host.restore = sl.RestoreInfo(lead_s=2000.0, rate=1.5, running=True, has_work=True)
        self.host.context = sp.SchedulerContext(play_pos=0.0, watched=True)

    def test_auto_picks_parallel_on_a_big_card_and_never_stops_restore(self):
        ex, res = self.run_ex()
        self.assertEqual(ex.mode, gt.MODE_PARALLEL)
        self.assertTrue(ex.mode_reason.startswith("vram_fits"))
        self.assertEqual(self.host.stops, 0)
        self.assertEqual([e for e, _l in self.host.leases], [False])         # one shared lease
        self.assertEqual(res.reason, "complete")                             # 1000 s fit in the 30-min window
        self.assertEqual(self.fake.kinds()[:4], ["asr", "mt_open", "mt", "mt_close"])

    def test_oom_releases_running_work_demotes_and_remembers(self):
        self.host.context = sp.SchedulerContext(item_state="done")
        self.fake.asr_errors = [sl.GpuOutOfMemory("CUDA out of memory")]
        ex, res = self.run_ex()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(ex.mode, gt.MODE_INTERLEAVE)
        demote = [e for e in ex.events if e["kind"] == "demote"]
        self.assertEqual(demote[0]["reason"], "oom")
        self.assertEqual(self.memory.get(self.key)["reason"], "oom")
        self.assertEqual(ex.plan.slices[0].get("asr").attempt, 0)            # redone, not counted as failure
        self.assertEqual(self.fake.calls[1], ("asr", [(0.0, 30.0)]))        # head first after demotion
        self.assertGreaterEqual(self.host.stops, 1)                          # now interleaving
        ex2 = self.executor()
        ex2._prepare()
        ex2._resolve_mode()
        self.assertEqual((ex2.mode, ex2.mode_reason), (gt.MODE_INTERLEAVE, "remembered:oom"))

    def test_restore_too_slow_while_watched_demotes_mid_batch(self):
        def slow(_fake):
            self.host.restore = sl.RestoreInfo(lead_s=200.0, rate=0.6, running=True, has_work=True)
        self.fake.after_asr = slow
        ex = self.executor()
        res = ex.run()
        demote = [e for e in ex.events if e["kind"] == "demote"]
        self.assertEqual(demote[0]["reason"], "restore_too_slow")
        self.assertEqual(self.memory.get(self.key)["reason"], "restore_too_slow")
        self.assertEqual(ex.plan.slices[0].get("asr").status, sp.DONE)       # finished ASR kept
        batches = [e for e in ex.events if e["kind"] == "batch_start"]
        self.assertEqual((batches[0]["mode"], batches[0]["stages"]), ("parallel", ["asr"]))
        self.assertEqual((batches[1]["mode"], batches[1]["stages"], batches[1]["slices"]), ("interleave", ["mt"], [0]))
        self.assertEqual(self.fake.kinds()[:2], ["asr", "mt_open"])         # MT only after the switch
        self.assertEqual(ex.mode, gt.MODE_INTERLEAVE)
        self.assertEqual(res.reason, "restore_turn")                         # restore lead 200 < safe 300
        self.assertFalse(any(st.status == sp.RUNNING for s in ex.plan.slices for st in s.tasks.values()))

    def test_forced_parallel_demotes_for_the_run_but_is_not_remembered(self):
        self.host.gpu = sl.GpuInfo("RTX Small", 8151.0)
        self.host.context = sp.SchedulerContext(item_state="done")
        self.fake.asr_errors = [sl.GpuOutOfMemory("out of memory")]
        ex, res = self.run_ex(resource_mode="parallel")
        self.assertEqual(ex.mode, gt.MODE_INTERLEAVE)
        self.assertIsNone(self.memory.get(gt.gpu_key("RTX Small", 8151.0)))
        self.assertEqual(res.reason, "complete")

    def test_lease_released_when_done(self):
        self.host.context = sp.SchedulerContext(item_state="done")
        ex, res = self.run_ex(resource_mode="parallel")
        self.assertEqual(res.reason, "complete")
        self.assertEqual(len(self.host.leases), 1)
        self.assertEqual(self.host.released, [self.host.leases[0][1]])


class PublicationOverlapTests(unittest.TestCase):
    def test_overlapping_sentence_with_new_tail_is_preserved(self):
        previous = [{"start": 295.0, "end": 299.0, "text": "前半"}]
        later = [{"start": 297.0, "end": 305.0, "text": "前半加新尾句"}]
        self.assertEqual(sr.resolve_preroll_overlaps([(0.0, previous), (300.0, later)]), previous + later)

    def test_preroll_copy_of_a_complete_sentence_is_dropped(self):
        # T3 batch-of-one: slice 0 kept 290.3-301.2 whole, slice 1's pre-roll split it differently
        s0 = [{"start": 288.3, "end": 289.8, "text": "なるほどね"}, {"start": 290.3, "end": 301.2, "text": "全文"}]
        s1 = [{"start": 290.4, "end": 296.2, "text": "前半"}, {"start": 296.2, "end": 301.2, "text": "後半"},
              {"start": 301.4, "end": 306.9, "text": "次"}]
        out = sr.resolve_preroll_overlaps([(0.0, s0), (301.555, s1)])
        self.assertEqual([r["text"] for r in out], ["なるほどね", "全文", "次"])

    def test_cut_off_sentence_comes_from_the_later_slice(self):
        s0 = [{"start": 280.0, "end": 284.0, "text": "a"}]                  # 298-300 cut off: dropped by tail guard
        s1 = [{"start": 298.0, "end": 301.4, "text": "完全"}, {"start": 302.0, "end": 305.0, "text": "b"}]
        out = sr.resolve_preroll_overlaps([(0.0, s0), (300.0, s1)])
        self.assertEqual([r["text"] for r in out], ["a", "完全", "b"])

    def test_far_preroll_cue_is_kept_when_previous_slice_has_nothing_overlapping(self):
        # slice 0 dropped "罪を認めて" (21.9-25.53, tail guard); slice 1's pre-roll heard it as 21.45-24.03 (< cut - 1 s)
        s0 = [{"start": 18.94, "end": 21.90, "text": "えっ"}]
        s1 = [{"start": 21.45, "end": 24.03, "text": "罪を認めていきなさい"}, {"start": 24.5, "end": 28.0, "text": "次"}]
        out = sr.resolve_preroll_overlaps([(0.0, s0), (25.532, s1)])
        self.assertEqual([r["text"] for r in out], ["えっ", "罪を認めていきなさい", "次"])

    def test_far_preroll_cue_is_dropped_when_previous_slice_has_an_overlapping_cue(self):
        s0 = [{"start": 18.94, "end": 21.90, "text": "えっ"}, {"start": 22.0, "end": 24.0, "text": "別の文字"}]
        s1 = [{"start": 18.49, "end": 21.45, "text": "え 判断は"}, {"start": 21.45, "end": 24.03, "text": "罪を認めて"},
              {"start": 24.5, "end": 28.0, "text": "次"}]
        out = sr.resolve_preroll_overlaps([(0.0, s0), (25.532, s1)])
        self.assertEqual([r["text"] for r in out], ["えっ", "別の文字", "次"])

    def test_composed_srt_has_no_duplicates_for_far_preroll(self):
        plan = sp.build_slice_plan(60.0, media_id="m", slice_seconds=30.0, first_slice_seconds=30.0)
        s1 = plan.slices[1]
        row = lambda a, b, t: {"start": a, "end": b, "text": t, "text_ja": t, "zh": "譯" + t}
        done = {0: [row(1.0, 4.0, "甲"), row(20.0, 23.0, "乙")],
                1: [row(20.5 - s1.start, 23.5 - s1.start, "乙'"), row(24.0 - s1.start, 26.0 - s1.start, "丙"),
                    row(31.0 - s1.start, 34.0 - s1.start, "丁")]}
        composed = sr._compose(plan, done, {})
        self.assertEqual([c["text_ja"] for c in composed["deduped"]], ["甲", "乙", "丙", "丁"])
        self.assertEqual(composed["ja_srt"].count("-->"), 4)

    def test_preroll_repeat_squeezed_past_prior_end_is_dropped(self):
        # 03 at 60 s slices: zero-length ASR segment at the 210.7 cut repeated the prior cue right after its end
        s0 = [{"start": 204.64, "end": 210.37, "text": "あのよかったら事件の話聞かせてもらえませんか"}]
        s1 = [{"start": 210.37, "end": 210.45, "text": "あの、よかったら事件の話聞かせてもらえませんか?"},
              {"start": 211.32, "end": 217.79, "text": "あ、いえ、それはあまりにも"}]
        out = sr.resolve_preroll_overlaps([(150.7, s0), (210.7, s1)])
        self.assertEqual([r["start"] for r in out], [204.64, 211.32])

    def test_short_repeat_across_cut_is_kept(self):
        s0 = [{"start": 58.0, "end": 59.0, "text": "だめ"}]
        s1 = [{"start": 59.2, "end": 59.9, "text": "だめ!"}]
        self.assertEqual(len(sr.resolve_preroll_overlaps([(0.0, s0), (60.0, s1)])), 2)

    def test_unpublished_previous_slice_does_not_block(self):
        s1 = [{"start": 298.0, "end": 301.4, "text": "完全"}]
        self.assertEqual(len(sr.resolve_preroll_overlaps([(0.0, []), (300.0, s1)])), 1)


class CloudPreRollTests(Base):
    """The cloud executor uses the same pre-roll and ownership rule; the window is part of its cache key."""

    def cloud(self, **kw):
        starts = {}

        def pieces(source, out_path, start, duration):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"a")
            starts[str(out_path)] = (start, duration)
            return [(out_path, 0.0)]

        def asr(piece, s):
            a, dur = starts[str(piece)]
            # a cue every 4 s over the audio (relative to the audio start)
            return [{"start": 0.5 + 4 * k, "end": min(3.5 + 4 * k, dur), "text": f"c{a + 0.5 + 4 * k:.1f}"}
                    for k in range(int(dur // 4) + 1) if 0.5 + 4 * k < dur]

        class Client:
            def translate(self, numbered):
                return "\n".join(f"{ln.split('.', 1)[0]}. 譯" for ln in numbered.splitlines())

            def close(self):
                pass

        deps = ss.CloudDeps(asr=asr, translator=Client, asr_identity={"id": "g"}, mt_identity={"id": "q"},
                            pieces=pieces, probe_duration=lambda s: 1000.0, detect_silences=every_30s_silence)
        kw.setdefault("slice_seconds", 300.0)
        ex = ss.CloudSliceExecutor(self.folder, self.source, deps=deps, clock=self.clock, sleep=self.clock.sleep,
                                   **kw)
        return ex, starts

    def test_audio_starts_before_the_slice_and_far_preroll_cues_are_kept_for_publication(self):
        ex, starts = self.cloud(first_slice_seconds=90.0)      # cue layout below is around a 90 s cut (default 30 s)
        res = ex.run()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(sorted(v[0] for v in starts.values()), [0.0, 75.0, 375.0, 675.0, 975.0])
        doc = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[1]))
        self.assertEqual(doc["audio_start"], 75.0)
        # pipeline slice-asr-v3: the whole pre-roll stays in the doc; publication drops what slice 0 covers
        self.assertEqual(doc["segments"][0]["start"], 75.5 - 90.0)
        self.assertTrue(any(r["end"] <= -sp.DEFAULT_LEAD_KEEP_SECONDS for r in doc["segments"]))
        self.assertTrue(any(r["start"] < 0.0 < r["end"] for r in doc["segments"]))      # 87.5-90.5 crosses the cut
        texts = [r["text"] for r in sr.resolve_preroll_overlaps(
            [(s.start, [dict(r, start=r["start"] + s.start, end=r["end"] + s.start)
                        for r in sr.read_asr(self.folder, ex._asr_key(s))["segments"]])
             for s in sp.slices_by_time(ex.plan)])]
        self.assertEqual(len(texts), len(set(texts)))
        doc0 = sr.read_asr(self.folder, ex._asr_key(ex.plan.slices[0]))
        self.assertTrue(all(r["end"] <= 90.0 - sp.DEFAULT_TAIL_GUARD_SECONDS for r in doc0["segments"]))

    def test_window_is_part_of_the_cache_key(self):
        ex, _ = self.cloud()
        ex.run()
        ex2, starts = self.cloud(asr_window=sp.AsrWindow(pre_roll_s=5.0))
        ex2.run()
        self.assertEqual(len(starts), 0)               # done slices keep their own result (reconcile) despite the new key
        self.assertNotEqual(ex._asr_key(ex.plan.slices[1]), ex2._asr_key(ex2.plan.slices[1]))


class PlanSliceLengthTests(Base):
    """New plans: cloud ASR slices are 60 s (Groq skips lines in 300 s uploads), local GPU slices stay 300 s."""

    def new_plan(self, ex):
        ex._media_id = "m"
        return ex._load_or_create_plan(None, None)

    def test_cloud_plan_uses_60s_and_local_plan_300s(self):
        cloud = ss.CloudSliceExecutor(self.folder, self.source, deps=ss.CloudDeps(
            asr=None, translator=None, asr_identity={"id": "g"}, mt_identity={"id": "q"},
            probe_duration=lambda s: 1000.0))
        self.assertEqual(self.new_plan(cloud).slice_seconds, sp.CLOUD_SLICE_SECONDS)
        self.assertEqual(sp.CLOUD_SLICE_SECONDS, 60.0)
        local = self.executor()
        self.assertEqual(self.new_plan(local).slice_seconds, sp.DEFAULT_SLICE_SECONDS)
        self.assertEqual(sp.DEFAULT_SLICE_SECONDS, 300.0)

    def test_persisted_plan_keeps_its_slice_length(self):
        old = sp.new_slice_plan(1000.0, media_id="m", slice_seconds=300.0)
        sp.save_slice_plan(old, self.folder)
        cloud = ss.CloudSliceExecutor(self.folder, self.source, deps=ss.CloudDeps(
            asr=None, translator=None, asr_identity={"id": "g"}, mt_identity={"id": "q"},
            probe_duration=lambda s: 1000.0))
        self.assertEqual(self.new_plan(cloud).slice_seconds, 300.0)

    def test_join_allowance_scales_with_short_plans(self):
        for secs, gap, kind in ((60.0, 80.0, "join"), (60.0, 100.0, "detect"), (300.0, 400.0, "join")):
            with self.subTest(secs=secs, gap=gap):
                plan = sp.new_slice_plan(5000.0, media_id="m", slice_seconds=secs, first_slice_seconds=None)
                chain = {"frontier": 0.0}
                orig = sp._frontier, sp._right_start
                sp._frontier = lambda c: 0.0
                sp._right_start = lambda p, c: gap
                try:
                    self.assertEqual(sp._next_step(plan, chain, None)[0], kind)
                finally:
                    sp._frontier, sp._right_start = orig


class MediaTurnTests(Base):
    """T4.2 fixes 1-3 (stream items, TurnConfig.media_turns): turns sized in media seconds, CPU preparation before
    the restoration pause, early end by the restore lead. Slices are 30 s, restoration is watched and running."""

    def setUp(self):
        super().setUp()
        self.host.context = sp.SchedulerContext(play_pos=0.0, watched=True)
        self.host.restore = sl.RestoreInfo(lead_s=5000.0, rate=1.5, running=True, has_work=True)
        self.log = []
        plan = sp.new_slice_plan(1000.0, media_id="hls:test", slice_seconds=30.0, first_slice_seconds=30.0)
        sp.save_slice_plan(plan, self.folder)
        stop = self.host.stop_restore
        self.host.stop_restore = lambda: (self.log.append("stop"), stop())[1]

    def cfg(self, **kw):
        kw.setdefault("media_turns", True)
        return gt.TurnConfig(**kw)

    def primed(self, **kw):
        """Executor ready for single step() calls: source prepared, mode resolved, every lazy cut decided."""
        ex = self.executor(resource_mode="interleave", turns=kw.pop("turns", None) or self.cfg(), **kw)
        self.assertTrue(ex._prepare())
        ex._resolve_mode()
        while ex._advance_cuts(None):
            pass
        return ex

    def test_turns_double_from_60_to_the_cap_and_merge_slices_into_one_batch(self):
        ex, res = self.run_ex(resource_mode="interleave", turns=self.cfg())
        self.assertEqual(res.reason, "complete")
        sizes = [e["seconds"] for e in ex.events if e["kind"] == "turn"]
        self.assertEqual(sizes[:4], [60.0, 120.0, 240.0, 300.0])
        asr = [c[1] for c in self.fake.calls if c[0] == "asr"]
        self.assertEqual([len(w) for w in asr[:3]], [2, 4, 8])            # consecutive 30 s slices, one ASR call
        self.assertEqual(self.fake.kinds()[:7], ["asr", "mt_open", "mt", "mt", "mt_close", "asr", "mt_open"])
        self.assertEqual(self.fake.kinds().count("mt_open"), len(asr))     # each model loaded once per turn
        self.assertEqual(self.host.stops, len(asr))                        # one pause per turn
        self.assertEqual(self.host.resumes, len(asr))

    # -- fix 4: free VRAM is measured after the pause, before a model loads -------------------------------
    def probed(self, answers, **kw):
        """Executor whose vram_probe answers from `answers` (dicts, or an Exception to raise); records the calls
        and the order against the pause (self.log) and the models (self.fake.calls)."""
        self.probes, noted = [], []
        queue = list(answers)

        def probe(need, wait_s):
            self.probes.append((need, wait_s, list(self.log), list(self.fake.calls)))
            answer = queue.pop(0) if queue else {"ok": True}
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.noted = noted
        self.host.note_vram_short = lambda need, free: noted.append((need, free))
        return self.primed(deps=self.deps(vram_probe=probe), **kw)

    def test_probe_runs_after_the_pause_and_before_any_model_with_asr_plus_margin(self):
        ex = self.probed([{"ok": True, "free_mib": 7000.0}])
        self.assertEqual(ex.step(None), "batch")
        need, wait_s, log_then, calls_then = self.probes[0]
        self.assertEqual((need, wait_s), (4801.0 + 512.0, 10.0))
        self.assertEqual((log_then, calls_then), (["stop"], []))           # paused, nothing loaded yet
        self.assertEqual(len(self.probes), 1)                              # one measurement per turn
        self.assertTrue(any(e["kind"] == "vram_ok" for e in ex.events))

    def test_a_translation_only_turn_needs_the_sakura_figure(self):
        ex = self.probed([{"ok": True}])
        mt_only = [sp.Task(0, sp.STAGE_MT, ex.plan.primary)]
        asr_mt = mt_only + [sp.Task(1, sp.STAGE_ASR, ex.plan.primary)]
        self.assertEqual(ex._vram_need_mib(mt_only), 4489.0 + 512.0)
        self.assertEqual(ex._vram_need_mib(asr_mt), max(4801.0, 4489.0) + 512.0)
        ex.vram_requirements = {"mt_sakura_7b": 6000.0}
        self.assertEqual(ex._vram_need_mib(mt_only), 6000.0 + 512.0)
        ex.turns = dataclasses.replace(ex.turns, vram_margin_mib=100.0)
        self.assertEqual(ex._vram_need_mib(mt_only), 6100.0)

    def test_short_vram_gives_the_turn_up_resumes_restore_and_keeps_the_work_pending(self):
        ex = self.probed([{"ok": False, "free_mib": 3000.0, "waited_s": 10.0}])
        self.assertEqual(ex.step(None), "restore_turn")
        self.assertEqual(ex.last_turn.reason, "vram_short")
        self.assertEqual((self.host.stops, self.host.resumes), (1, 1))      # paused then resumed
        self.assertEqual(self.fake.calls, [])                               # no model loaded
        self.assertIsNone(ex._lease)
        self.assertEqual(self.noted, [(4801.0 + 512.0, 3000.0)])            # counted as a GPU-short occurrence
        self.assertTrue(any(e["kind"] == "vram_short" for e in ex.events))
        statuses = self.statuses(ex.plan)
        self.assertFalse(any(v in (sp.RUNNING, sp.FAILED) for v in statuses.values()))
        self.assertEqual(ex._turn_s, 60.0)                                  # the size does not grow

    def test_after_a_short_turn_no_new_pause_until_the_retry_time(self):
        ex = self.probed([{"ok": False, "free_mib": 100.0}, {"ok": True}])
        self.assertEqual(ex.step(None), "restore_turn")
        self.clock.t += 30.0
        self.assertEqual(ex.step(None), "restore_turn")                    # inside vram_retry_s (60 s): never paused
        self.assertEqual((self.host.stops, len(self.probes)), (1, 1))
        self.clock.t += 31.0
        self.assertEqual(ex.step(None), "batch")
        self.assertEqual((self.host.stops, len(self.probes)), (2, 2))
        self.assertIn("asr", self.fake.kinds())

    def test_a_seek_clears_the_retry_hold(self):
        ex = self.probed([{"ok": False}])
        ex.step(None)
        self.assertGreater(ex._vram_retry_at, self.clock())
        with patch.object(self.host, "seek_pending", create=True, return_value=True):
            ex.step(None)
        self.assertEqual(ex._vram_retry_at, 0.0)

    def test_a_probe_that_cannot_run_does_not_block_the_turn(self):
        ex = self.probed([RuntimeError("python missing")])
        self.assertEqual(ex.step(None), "batch")
        self.assertTrue(any(e["kind"] == "vram_probe_failed" for e in ex.events))
        self.assertIn("asr", self.fake.kinds())

    def test_unmeasured_answer_counts_as_ok(self):
        ex = self.probed([{"ok": True, "measured": False}])
        self.assertEqual(ex.step(None), "batch")

    def test_without_a_probe_nothing_changes(self):
        ex = self.primed()
        self.assertIsNone(ex.deps.vram_probe)
        self.assertEqual(ex.step(None), "batch")

    def test_settings_override_margin_wait_and_retry(self):
        cfg = gt.TurnConfig.from_settings({"subtitle_local_vram_wait_s": 3, "subtitle_local_vram_margin_mib": 256,
                                           "subtitle_local_vram_retry_s": 5})
        self.assertEqual((cfg.vram_wait_s, cfg.vram_margin_mib, cfg.vram_retry_s), (3.0, 256.0, 5.0))

    def test_seek_restarts_at_the_first_size(self):
        ex = self.primed()
        ex._turn_s = 240.0
        with patch.object(self.host, "seek_pending", create=True, return_value=True):
            self.assertEqual(ex.step(None), "idle")
        self.assertEqual(ex._turn_s, 60.0)

    def test_sizes_are_settings(self):
        cfg = gt.TurnConfig.from_settings({"subtitle_local_turn_first_s": 30, "subtitle_local_turn_growth": 3,
                                           "subtitle_local_turn_max_s": 100})
        self.assertEqual((cfg.turn_first_s, cfg.turn_growth, cfg.turn_max_s, cfg.turn_min_s),
                         (30.0, 3.0, 100.0, 60.0))
        cfg = dataclasses.replace(cfg, media_turns=True, turn_min_s=30.0)
        ex, _res = self.run_ex(resource_mode="interleave", turns=cfg)
        self.assertEqual([e["seconds"] for e in ex.events if e["kind"] == "turn"][:3], [30.0, 90.0, 100.0])

    def test_turn_shrinks_to_fit_the_restore_lead(self):
        self.host.restore = sl.RestoreInfo(lead_s=400.0, has_work=True)
        ex = self.primed(turns=self.cfg(turn_first_s=240.0))
        self.assertEqual(ex.step(None), "batch")
        # ASR windows carry 15 s of pre-roll: 240 s = 8 x 45 s of work = 234 s of GPU; 120 s = 4 x 45 s = 117 s
        # 240 s: 400 - 234 - 90 < 150; 120 s: 400 - 117 - 90 >= 150
        self.assertEqual(next(e for e in ex.events if e["kind"] == "turn")["seconds"], 120.0)

    def test_no_switch_when_even_the_minimum_turn_does_not_fit(self):
        self.host.restore = sl.RestoreInfo(lead_s=240.0, has_work=True)      # 60 s: 240 - 58.5 - 90 < 150
        ex = self.primed()
        self.assertEqual(ex.step(None), "restore_turn")
        self.assertEqual(ex.last_turn.reason, "subtitle_budget")
        self.assertEqual((self.log, self.fake.calls), ([], []))              # never paused

    def drop_after_asr(self, lead_s, play_pos=500.0):
        """After ASR the restore lead is `lead_s` and the play position has moved on to `play_pos`."""
        def drop(_fake):
            self.host.restore = sl.RestoreInfo(lead_s=lead_s, has_work=True)
            self.host.context = sp.SchedulerContext(play_pos=play_pos, watched=True)
        self.fake.after_asr = drop

    def test_translation_waits_when_asr_leaves_too_little_restore_lead(self):
        def drop(_fake):
            self.host.restore = sl.RestoreInfo(lead_s=100.0, has_work=True)
            self.host.context = sp.SchedulerContext(play_pos=500.0, watched=True)   # not in the batch
        self.host.restore = sl.RestoreInfo(lead_s=400.0, has_work=True)
        self.fake.after_asr = drop
        ex = self.primed()
        self.assertEqual(ex.step(None), "batch")
        self.assertEqual(self.fake.kinds(), ["asr"])        # no MT: the ASR results stay saved
        self.assertIn("turn_early_end", [e["kind"] for e in ex.events])
        self.assertEqual((self.host.stops, self.host.resumes), (1, 1))
        for i in (0, 1):
            self.assertEqual(ex.plan.slices[i].get("asr").status, sp.DONE)
            self.assertEqual(ex.plan.slices[i].get("mt").status, sp.PENDING)
        self.assertEqual(ex._turn_s, 60.0)                  # an early end does not grow the turn
        self.fake.after_asr = None
        self.host.context = sp.SchedulerContext(play_pos=0.0, watched=True)
        self.host.restore = sl.RestoreInfo(lead_s=5000.0, has_work=True)
        self.fake.calls.clear()
        self.assertEqual(ex.step(None), "batch")             # the next turn translates first (no second ASR)
        self.assertEqual(self.fake.kinds()[:2], ["mt_open", "mt"])
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    # -- second round, fix 3: ASR is followed by MT unless the lead after MT drops below mt_min_restore_s ----
    def early_kinds(self, lead_s, play_pos=500.0, turns=None):
        self.host.restore = sl.RestoreInfo(lead_s=5000.0, has_work=True)
        self.drop_after_asr(lead_s, play_pos)
        ex = self.primed(turns=turns)
        self.assertEqual(ex.step(None), "batch")
        return ex, [e["kind"] for e in ex.events]

    def test_translation_is_finished_by_default(self):
        # 2 slices = 60 s of MT: 10 s load + 60 x 0.2 + 30 s restart = 52 s; 150 - 52 = 98 >= 60 (the old rule ended at 150)
        ex, kinds = self.early_kinds(150.0)
        self.assertNotIn("turn_early_end", kinds)
        self.assertIn("mt", self.fake.kinds())
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_early_end_estimate_counts_the_mt_load_only(self):
        # needs 60 (floor) + 22 (MT) + 30 (restart) = 112; the 50 s ASR load (subtitle_load_s 60 - mt 10) is paid
        ex, kinds = self.early_kinds(112.0)
        self.assertNotIn("turn_early_end", kinds)

    def test_early_end_just_below_the_floor(self):
        ex, kinds = self.early_kinds(111.0)
        self.assertIn("turn_early_end", kinds)
        self.assertEqual(self.fake.kinds(), ["asr"])

    def test_the_slice_with_the_play_position_is_never_deferred(self):
        ex, kinds = self.early_kinds(10.0, play_pos=5.0)         # lead far too low, but slice 0 holds the play position
        self.assertNotIn("turn_early_end", kinds)
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_mt_min_restore_and_mt_costs_are_settings(self):
        cfg = gt.TurnConfig.from_settings({"subtitle_local_mt_min_restore_s": 20, "subtitle_local_mt_load_s": 3,
                                           "subtitle_local_mt_seconds_per_media_second": 0.1})
        self.assertEqual((cfg.mt_min_restore_s, cfg.mt_load_s, cfg.mt_seconds_per_media_second), (20.0, 3.0, 0.1))
        self.assertEqual((gt.TurnConfig().mt_min_restore_s, gt.TurnConfig().mt_load_s), (60.0, 10.0))
        ex, kinds = self.early_kinds(111.0, turns=self.cfg(mt_min_restore_s=20.0))
        self.assertNotIn("turn_early_end", kinds)

    # -- second round, fix 1: the plan is cut ahead of the play position while preparing -------------------
    def paced(self, **kw):
        """Executor on a paced (stream-like) host: one lazy cut per round, none decided yet but the first."""
        self.host.pace_cuts = lambda: True
        ex = self.executor(resource_mode="interleave", turns=kw.pop("turns", None) or self.cfg(), **kw)
        self.assertTrue(ex._prepare())
        ex._resolve_mode()
        ex._advance_cuts(None)
        return ex

    def test_a_batch_takes_several_slices_because_the_prepare_cut_ahead(self):
        ex = self.paced()
        before = len(ex.plan.slices)
        self.assertEqual(ex.step(None), "batch")
        start = next(e for e in ex.events if e["kind"] == "batch_start")
        self.assertGreater(len(ex.plan.slices), before)
        self.assertGreaterEqual(len(set(start["slices"])), 2)
        self.assertGreaterEqual(ex._covered_s([sp.Task(i, sp.STAGE_ASR, ex.plan.primary) for i in start["slices"]]), 60.0)

    def test_without_cutting_ahead_the_batch_stays_small(self):
        ex = self.paced()
        before = len(ex.plan.slices)
        with patch.object(ex, "_cut_ahead", lambda *a: None):
            self.assertEqual(ex.step(None), "batch")
        self.assertEqual(len(ex.plan.slices), before)
        self.assertEqual(len(set(next(e for e in ex.events if e["kind"] == "batch_start")["slices"])), before)

    def test_cut_ahead_is_bounded_and_stops_at_the_readable_limit(self):
        self.host.limit = 45.0                                   # only slice 0 (and maybe one more) is readable
        ex = self.paced()
        ex._cut_ahead(self.host.limit, 300.0)
        self.assertLessEqual(len(ex.plan.slices), sl.MAX_CUT_AHEAD + 2)
        ex._cut_ahead(self.host.limit, 300.0)                    # nothing more to cut: returns, no loop

    def test_cut_ahead_cuts_short_slices_even_when_the_plan_slice_is_300_s(self):
        # r2-main (fix I): cutting ahead of the short slice at the play position grew its chain with the plan's
        # normal 300 s slice, so one slice satisfied the target and the batch work stayed far above it
        plan = sp.new_slice_plan(2000.0, media_id="hls:test", slice_seconds=300.0, first_slice_seconds=30.0)
        sp.save_slice_plan(plan, self.folder)
        ex = self.paced()
        ex._cut_ahead(None, 60.0)
        lengths = [s.end - s.start for s in ex.plan.slices]
        self.assertGreaterEqual(len(lengths), 2)
        self.assertLessEqual(max(lengths), 45.0)
        batch = ex._preview_batch(0, None, 60.0)
        self.assertGreaterEqual(ex._covered_s(batch), 60.0)
        self.assertLessEqual(ex._batch_work_s(batch), 60.0 + 45.0 + 30.0)

    def test_a_turn_leaves_out_an_oversized_slice_that_would_blow_the_target(self):
        # an existing 300 s slice behind the playhead slice is not picked into a 60 s turn
        plan = sp.new_slice_plan(2000.0, media_id="hls:test", slice_seconds=300.0, first_slice_seconds=30.0)
        sp.save_slice_plan(plan, self.folder)
        ex = self.paced()
        self.assertTrue(sp.extend_lazy_cuts(ex.plan, ex.source, None, detect=ex.deps.detect_silences, max_cuts=1))
        lengths = [s.end - s.start for s in ex.plan.slices]
        self.assertGreater(max(lengths), 200.0)
        batch = ex._preview_batch(0, None, 60.0)
        self.assertTrue(ex._preview_oversize)
        self.assertLessEqual(ex._covered_s(batch), 60.0 + 30.0)
        before = len(ex.plan.slices)
        ex._cut_ahead(None, 60.0)                                  # nothing to cut beyond the oversized slice
        self.assertEqual(len(ex.plan.slices), before)

    def test_non_stream_turns_never_cut_ahead(self):
        self.host.pace_cuts = lambda: True
        ex = self.executor(resource_mode="interleave", turns=gt.TurnConfig())
        with patch.object(ex, "_extend_one_cut") as cut:
            ex._prepare(); ex._resolve_mode()
            ex.step(None)
        cut.assert_not_called()

    # -- second round, fix 4: a refused turn is logged ----------------------------------------------------
    def test_budget_refusal_logs_an_event_with_the_numbers(self):
        self.host.restore = sl.RestoreInfo(lead_s=240.0, has_work=True)
        ex = self.primed()
        self.assertEqual(ex.step(None), "restore_turn")
        ev = [e for e in ex.events if e["kind"] == "subtitle_budget"]
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((e["lead_s"], e["target_s"], e["min_s"]), (240.0, 60.0, 60.0))
        self.assertEqual((e["load_s"], e["restart_s"], e["floor_s"]), (60.0, 30.0, 150.0))
        self.assertAlmostEqual(e["work_s"], 75.0)
        self.assertEqual(e["s"], 0.65)
        json.dumps(e)

    def test_budget_event_is_throttled_but_not_lost(self):
        self.host.restore = sl.RestoreInfo(lead_s=240.0, has_work=True)
        ex = self.primed()
        for _ in range(3):
            ex.step(None)
        self.assertEqual(len([e for e in ex.events if e["kind"] == "subtitle_budget"]), 1)
        self.clock.t += sl.BUDGET_EVENT_EVERY_S + 1
        ex.step(None)
        self.assertEqual(len([e for e in ex.events if e["kind"] == "subtitle_budget"]), 2)

    def test_budget_event_reaches_the_progress_stream(self):
        seen = []
        self.host.on_progress = lambda info: seen.append(dict(info))
        self.host.restore = sl.RestoreInfo(lead_s=240.0, has_work=True)
        ex = self.primed()
        ex.step(None)
        self.assertTrue(any(i.get("state") == "subtitle_budget" and i.get("lead_s") == 240.0 for i in seen))

    # -- second round, fix 2: learned model loads and restart feed the budget -------------------------------
    def costs(self, **values):
        mem = ssg.SwitchCostMemory(self.root / "costs.json")
        key = gt.gpu_key("RTX Test", 8151.0)
        for name, v in values.items():
            mem.record(key, name, v)
        return mem

    def test_learned_costs_replace_the_config_in_the_budget(self):
        self.host.restore = sl.RestoreInfo(lead_s=240.0, has_work=True)      # refused with 60 + 30 (see above)
        mem = self.costs(asr_load_s=2.0, mt_load_s=1.0, restore_restart_s=1.0)
        ex = self.primed(cost_memory=mem)
        self.assertEqual((ex._asr_load_s, ex._mt_load_s, ex._restart_s), (2.0, 1.0, 1.0))
        self.assertEqual(ex.step(None), "batch")                              # 240 - 58.5 * 0.65 - 3 - 1 >= 150

    def test_without_learned_costs_the_config_split_is_used(self):
        ex = self.primed(cost_memory=self.costs())
        self.assertEqual((ex._asr_load_s, ex._mt_load_s, ex._restart_s), (50.0, 10.0, 30.0))
        ex = self.primed(turns=self.cfg(subtitle_load_s=20.0, restore_restart_s=10.0))
        self.assertEqual((ex._asr_load_s, ex._mt_load_s, ex._restart_s), (10.0, 10.0, 10.0))

    def test_file_items_keep_the_config_costs_even_with_a_memory(self):
        ex = self.executor(resource_mode="interleave", turns=gt.TurnConfig(),
                           cost_memory=self.costs(asr_load_s=1.0, mt_load_s=1.0, restore_restart_s=1.0))
        ex._prepare(); ex._resolve_mode()
        self.assertEqual(ex._load_total() + ex._restart_s, 90.0)

    def probed_deps(self, asr_load=8.0, mt_load=4.0):
        """Deps whose fake models announce their load like the real children (LoadProbe events)."""
        probe = sl.LoadProbe(clock=self.clock)
        fake = self.fake
        outer_asr = fake.asr_batch

        def asr_batch(windows, workdir):
            probe.event("prepare_asr", None, "啟動字幕程序")
            self.clock.t += asr_load
            probe.event("prepare_asr", None, "whisper：本機 ASR 模型與處理器已就緒")
            return outer_asr(windows, workdir)

        def mt_session():
            inner = fake.mt_session()

            @contextlib.contextmanager
            def session():
                with inner as translate:
                    first = []

                    def tr(ja_doc, workdir):
                        if not first:
                            first.append(1)
                            probe.event("prepare_translate", None, "啟動字幕程序")
                            self.clock.t += mt_load
                            probe.event("prepare_translate", 100, "本機翻譯伺服器已就緒")
                        return translate(ja_doc, workdir)
                    yield tr
            return session()
        return self.deps(asr_batch=asr_batch, mt_session=mt_session, load_probe=probe)

    def test_measured_loads_are_remembered_per_card_and_leave_the_speed(self):
        mem = self.costs()
        speed = ssg.SubtitleSpeedMemory(self.root / "speed.json")
        self.fake.asr_seconds = 20.0
        ex = self.primed(deps=self.probed_deps(), cost_memory=mem, speed_memory=speed)
        self.assertEqual(ex.step(None), "batch")
        key = gt.gpu_key("RTX Test", 8151.0)
        self.assertEqual(mem.get(key, ssg.COST_ASR_LOAD), 8.0)
        self.assertEqual(mem.get(key, ssg.COST_MT_LOAD), 4.0)
        self.assertEqual([e["stage"] for e in ex.events if e["kind"] == "load_measured"], ["asr", "mt"])
        # 60 s of media; wall = 8 + 20 (ASR) + 4 (MT load) = 32 s; speed = (32 - 12) / 60
        self.assertAlmostEqual(speed.get(key), 20.0 / 60.0, places=3)

    def test_no_probe_no_measurement(self):
        mem = self.costs()
        ex = self.primed(cost_memory=mem)
        ex.step(None)
        self.assertIsNone(mem.get(gt.gpu_key("RTX Test", 8151.0), ssg.COST_ASR_LOAD))
        self.assertFalse(any(e["kind"] == "load_measured" for e in ex.events))

    def test_a_broken_cost_memory_never_breaks_the_turn(self):
        class Broken:
            def get(self, *a):
                raise RuntimeError("disk")

            def record(self, *a):
                raise RuntimeError("disk")
        ex = self.primed(deps=self.probed_deps(), cost_memory=Broken())
        self.assertEqual(ex.step(None), "batch")
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_load_probe_pairs_start_and_ready_events(self):
        t = [0.0]
        probe = sl.LoadProbe(clock=lambda: t[0])
        probe.event("prepare_asr", None, "啟動字幕程序")                 # not armed: ignored
        probe.begin()
        t[0] = 1.0
        probe.event("prepare_asr", None, "whisper-a：啟動字幕程序")
        t[0] = 6.0
        probe.event("prepare_asr", None, "whisper-a：本機 ASR 模型與處理器已就緒")
        t[0] = 30.0
        probe.event("asr", 50, "辨識中")                                  # not a ready event
        t[0] = 31.0
        probe.event("prepare_asr", None, "whisper-b：啟動字幕程序")
        t[0] = 40.0
        probe.event("prepare_asr", None, "whisper-b：本機 ASR 模型與處理器已就緒")
        self.assertEqual(probe.take(), 5.0 + 9.0)
        self.assertIsNone(probe.take())                                   # nothing since: disarmed
        probe.begin()
        self.assertIsNone(probe.take())                                   # all cached: no model loaded

    def test_cut_ahead_helper_counts_unique_slices(self):
        ex = self.primed()
        tasks = [sp.Task(0, sp.STAGE_ASR, ex.plan.primary), sp.Task(0, sp.STAGE_MT, ex.plan.primary),
                 sp.Task(1, sp.STAGE_ASR, ex.plan.primary)]
        self.assertEqual(ex._covered_s(tasks), 60.0)

    def test_cpu_preparation_happens_before_the_pause_and_is_reused(self):
        prepared, seen = [], []

        def prepare(windows, workdir):
            self.log.append("prepare")
            prepared.append((list(windows), workdir))
            workdir.mkdir(parents=True, exist_ok=True)

        def asr_batch(windows, workdir):
            self.log.append("asr")
            seen.append((list(windows), workdir))
            return self.fake.asr_batch(windows, workdir)
        ex = self.primed(deps=self.deps(asr_batch=asr_batch, asr_prepare=prepare))
        self.assertEqual(ex.step(None), "batch")
        self.assertEqual(self.log[:3], ["prepare", "stop", "asr"])
        self.assertEqual(seen[0], prepared[0])               # same windows, same workdir
        self.assertEqual(ex._prepared, {})

    def test_failed_preparation_falls_back_to_the_batch(self):
        def prepare(windows, workdir):
            raise RuntimeError("ffmpeg died")
        ex = self.primed(deps=self.deps(asr_prepare=prepare))
        self.assertEqual(ex.step(None), "batch")
        self.assertIn("prepare_failed", [e["kind"] for e in ex.events])
        self.assertEqual(ex.plan.slices[0].get("mt").status, sp.DONE)

    def test_nothing_is_prepared_when_the_budget_refuses_the_turn(self):
        self.host.restore = sl.RestoreInfo(lead_s=200.0, has_work=True)
        prepared = []
        ex = self.primed(deps=self.deps(asr_prepare=lambda w, d: prepared.append(w)))
        self.assertEqual(ex.step(None), "restore_turn")
        self.assertEqual((prepared, self.log), ([], []))

    def test_slice_count_turns_are_unchanged_without_media_turns(self):
        ex, res = self.run_ex(resource_mode="interleave", turns=gt.TurnConfig())
        self.assertTrue(all("seconds" not in e for e in ex.events if e["kind"] == "turn"))


class SpeedLearningTests(Base):
    """T4.3: finished batches teach the real subtitle speed of the card (queue/stream_subtitle_gpu.py)."""

    def setUp(self):
        super().setUp()
        self.speed = ssg.SubtitleSpeedMemory(self.root / "speed.json")
        self.key = gt.gpu_key("RTX Test", 8151.0)

    def test_records_after_completed_batches(self):
        self.fake.asr_seconds = 60.0
        ex, res = self.run_ex(speed_memory=self.speed)
        self.assertEqual(res.reason, "complete")
        # batches of 30 s, 600 s and 370 s of media; the 30 s one is long enough to count
        self.assertEqual(self.speed.count(self.key), 3)
        self.assertIsNotNone(self.speed.get(self.key))

    def test_nothing_recorded_after_cancel(self):
        self.fake.asr_seconds = 60.0
        self.host.cancel_after = 2
        ex, res = self.run_ex(speed_memory=self.speed)
        self.assertEqual(res.reason, "cancelled")
        self.assertEqual(self.speed.count(self.key), 0)

    def test_learned_value_feeds_the_budget(self):
        self.speed.record(self.key, 1500.0, 1000.0)          # 1.5 s per media second
        ex = self.executor(speed_memory=self.speed)
        ex._resolve_mode()
        self.assertAlmostEqual(ex._seconds_per_media_second, 1.5)
        self.speed.record(self.key, 10.0, 1000.0)
        for _ in range(30):
            self.speed.record(self.key, 10.0, 1000.0)
        ex2 = self.executor(speed_memory=self.speed)
        ex2._resolve_mode()
        self.assertEqual(ex2._seconds_per_media_second, 0.2)   # floor

    def test_no_memory_keeps_default(self):
        ex = self.executor()
        ex._resolve_mode()
        self.assertEqual(ex._seconds_per_media_second, ex.turns.subtitle_seconds_per_media_second)

    def test_broken_memory_never_raises(self):
        class Broken:
            def get(self, key):
                raise RuntimeError("disk")

            def record(self, *a):
                raise RuntimeError("disk")
        ex, res = self.run_ex(speed_memory=Broken())
        self.assertEqual(res.reason, "complete")


if __name__ == "__main__":
    unittest.main()
