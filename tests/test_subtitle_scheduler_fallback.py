"""M4b: cross-executor fallback with the cloud and the local executor on the SAME item folder (design 4.7,
M4 acceptance). No network, no GPU: candidate-aware fake cloud provider, fake local models / GPU host, a real
fallback.Ledger in a temp folder, one fake clock for everything."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import copy
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from oh_streamer.queue import subtitle_local as sl
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr
from oh_streamer.subtitles.providers.http import ProviderError
from tests.test_subtitle_scheduler_cloud import FakeHost
from tests.test_subtitle_scheduler_local import FakeClock, FakeGpuHost, FakeModels, every_30s_silence

MEDIA = "hls:m4b-media"


def cloud_row(cid, kind="asr", model=None):
    return {"id": cid, "kind": kind, "provider": "openai-compatible", "connection_id": "conn-" + cid,
            "remote_model": model or cid, "base_url": "https://example.org/v1",
            "recipe": "whisper" if kind == "asr" else "general-llm"}


def local_row(cid, kind="asr", root="."):
    return {"id": cid, "kind": kind, "provider": "local", "recipe": "__dual__" if kind == "asr" else "sakura-v1",
            "model_path": str(Path(root) / ("missing-" + cid))}


def frozen(asr, mt, asr_fallback=(), mt_fallback=()):
    return {"models": {"asr": asr, "translation": mt},
            "fallback": {"asr": list(asr_fallback), "translation": list(mt_fallback)},
            "policy": {"max_attempts": 4, "retries": 1, "cooldown_seconds": 60},
            "whisper_policy": "fixed", "whisper_models": []}


def err(kind, retry_after=None):
    return ProviderError(kind, kind + " happened", retryable=True, retry_after=retry_after)


class FakeCloud:
    """Candidate-aware cloud provider. `errors[candidate id]` is consumed one per request (None = succeed)."""

    def __init__(self, cues=4):
        self.cues = cues
        self.calls = []                 # ("asr" | "mt", candidate id, slice index | batch size)
        self.errors = {}
        self.piece_starts = {}
        self.on_request = None          # hook(kind, candidate) before every request
        self._lock = threading.Lock()

    def _next_error(self, cid):
        with self._lock:
            queue = self.errors.get(cid) or []
            return queue.pop(0) if queue else None

    def pieces(self, source, out_path, start, duration):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake-audio")
        self.piece_starts[str(out_path)] = start
        return [(out_path, 0.0)]

    def asr_run(self, candidate, piece, s, run):
        def op(row, key):
            assert key == "k-" + row["id"]
            with self._lock:
                self.calls.append(("asr", candidate.id, s.index))
            if self.on_request:
                self.on_request("asr", candidate)
            e = self._next_error(candidate.id)
            if e:
                raise e
            lead = s.start - self.piece_starts.get(str(piece), s.start)
            return [{"start": lead + 1.2 * i, "end": lead + 1.2 * i + 1.1, "text": f"スライス{s.index}の第{i}文です。"}
                    for i in range(self.cues)]
        return run(op, audio=1.0)

    def translator_for(self, candidate, run):
        outer = self

        class Client:
            last_source = fb.identity(candidate.row)

            def translate(self, numbered):
                lines = numbered.splitlines()

                def op(row, key):
                    with outer._lock:
                        outer.calls.append(("mt", candidate.id, len(lines)))
                    if outer.on_request:
                        outer.on_request("mt", candidate)
                    e = outer._next_error(candidate.id)
                    if e:
                        raise e
                    return "\n".join(re.sub(r"^(\d+)\.\s*(.*)$", r"\1. 譯\2", ln) for ln in lines)
                return run(op, tokens=100)

            def close(self):
                pass
        return Client()

    def count(self, kind, cid=None):
        return sum(1 for k, c, _ in self.calls if k == kind and (cid is None or c == cid))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.folder = self.root / "subtitles"
        self.source = self.root / "movie.mp4"
        self.source.write_bytes(b"x" * 5000)
        self.clock = FakeClock()
        self.ledger = fb.Ledger(self.root / "ledger.sqlite3", clock=self.clock)
        self.cloud = FakeCloud()
        self.models = FakeModels()
        self.chost = FakeHost()
        self.chost.media_id = lambda: MEDIA
        self.lhost = FakeGpuHost()
        self.lhost.media_id = lambda: MEDIA
        self.memory_path = self.root / "modes.json"
        self.GROQ = cloud_row("groq-asr")
        self.LLM = cloud_row("llm-mt", "translation")
        self.GPU = local_row("gpu-asr", root=self.root)
        self.SAKURA = local_row("sakura-mt", "translation", root=self.root)
        self.plan = None                                    # current frozen plan (plan_provider)

    def credentials(self, row):
        return "k-" + row["id"]

    def cloud_ex(self, **kw):
        deps = ss.CloudDeps(asr=None, translator=None, asr_run=self.cloud.asr_run,
                            translator_for=self.cloud.translator_for, asr_identity={"id": "unused"},
                            mt_identity={"id": "unused"}, pieces=self.cloud.pieces, probe_duration=lambda s: 1000.0,
                            detect_silences=every_30s_silence, asr_params={"p": 1}, mt_params={"batch_size": 30},
                            ledger=self.ledger, credentials=self.credentials)
        kw.setdefault("host", self.chost)
        kw.setdefault("slice_seconds", 300.0)       # fixtures assume 300 s slices (cloud default is 60 s)
        return ss.CloudSliceExecutor(self.folder, self.source, deps=deps, clock=self.clock, sleep=self.clock.sleep,
                                     plan_provider=lambda: self.plan, poll_s=0.01, **kw)

    def local_ex(self, **kw):
        deps = sl.LocalDeps(asr_batch=self.models.asr_batch, mt_session=self.models.mt_session,
                            asr_identity={"provider": "local", "asr_mode": "dual", "models": {"a": "x"}},
                            mt_identity={"provider": "local", "backend": "sakura", "model": "m"},
                            probe_duration=lambda s: 1000.0, detect_silences=every_30s_silence,
                            asr_params={"pipeline": "test"}, mt_params={"batch_size": 30},
                            ledger=self.ledger, credentials=self.credentials)
        kw.setdefault("host", self.lhost)
        kw.setdefault("asr_window", sp.AsrWindow())
        from oh_streamer.queue import gpu_turns as gt
        return sl.LocalSliceExecutor(self.folder, self.source, deps=deps, clock=self.clock, sleep=self.clock.sleep,
                                     plan_provider=lambda: self.plan, memory=gt.GpuModeMemory(self.memory_path), **kw)

    def disk(self):
        return sp.load_slice_plan(self.folder)

    def state(self, i, stage="asr"):
        return self.disk().slices[i].get(stage)

    def doc(self, i, stage="asr"):
        st = self.state(i, stage)
        return (sr.read_asr if stage == "asr" else sr.read_mt)(self.folder, st.result_key)

    def owner(self, i, stage="asr"):
        plan = self.disk()
        owner = fb.owner_function(self.plan, cooldown=fb.ledger_cooldown(self.ledger, self.credentials),
                                  now=self.clock())
        return owner(stage, "primary", plan.slices[i].get(stage))

    def drive(self, rounds=20, advance=0.0):
        """Alternate the two executors (each run() reloads the shared plan from disk) until complete."""
        for _ in range(rounds):
            c = self.cloud_ex().run()
            l = self.local_ex().run()
            if c.reason == "complete" and l.reason == "complete":
                return c, l
            self.clock.t += advance
        self.fail(f"not complete: cloud {c.reason}, local {l.reason}")


class CloudPrimaryLocalBackupTests(Base):
    """Scenario A: cloud primary + local backup."""

    def setUp(self):
        super().setUp()
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU], mt_fallback=[self.SAKURA])

    def test_long_429_hands_off_to_local_which_finishes_and_the_cloud_recovers(self):
        self.cloud.errors["groq-asr"] = [err("rate_limit", 60.0)]
        res = self.cloud_ex().run()
        st = self.state(0)
        self.assertEqual((st.status, st.candidate, st.wait_executor), (sp.PENDING, "gpu-asr", "local"))
        self.assertEqual((st.reclaim_executor, st.reclaim_at), ("cloud", 1060.0))
        self.assertEqual(st.tried["groq-asr"]["kind"], "rate_limit")
        self.assertEqual(st.tried["groq-asr"]["retry_at"], 1060.0)
        self.assertEqual((res.reason, res.waiting_executor), ("idle", "local"))
        self.assertEqual(self.cloud.count("asr"), 1)          # the other units skip the cooling model at once
        self.assertEqual([self.owner(i) for i in range(5)], ["local"] * 5)

        self.lhost.limit = 95.0                               # the local GPU only gets to the head slice
        lres = self.local_ex().run()
        self.assertEqual(lres.reason, "download")
        st = self.state(0)
        self.assertEqual((st.status, st.candidate), (sp.DONE, "gpu-asr"))
        self.assertEqual(self.doc(0)["candidate"], "gpu-asr")
        self.assertEqual(self.owner(0, "mt"), "cloud")        # translation chain head is healthy: cloud
        self.cloud_ex().run()
        self.assertEqual(self.state(0, "mt").candidate, "llm-mt")
        self.assertEqual(self.models.kinds().count("mt"), 0)

        self.clock.t += 61                                    # the Groq cooldown is over
        self.assertEqual(self.owner(1), "cloud")              # new work goes back to the cloud
        res = self.cloud_ex().run()
        self.assertEqual(res.reason, "complete")
        self.assertEqual([self.state(i).candidate for i in range(5)], ["gpu-asr"] + ["groq-asr"] * 4)
        self.assertTrue(res.publication["complete"])
        self.assertEqual(res.publication["cues"], 4 * 4 + len(self.doc(0)["segments"]))

    def test_cloud_takes_back_handed_off_work_the_local_gpu_has_not_started(self):
        self.cloud.errors["groq-asr"] = [err("rate_limit", 60.0)]
        self.cloud_ex().run()
        self.assertEqual(self.state(0).wait_executor, "local")
        self.lhost.busy = True                                # local never got a GPU turn
        self.assertEqual(self.local_ex().run().reason, "gpu_busy")
        self.clock.t += 61
        self.assertEqual(self.owner(0), "cloud")              # reclaim
        res = self.cloud_ex().run()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.state(0).candidate, "groq-asr")
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.cloud.count("asr", "groq-asr"), 6)   # 1 429 + 5 slices

    def test_cloud_never_idles_without_a_wake_time_while_cooling_work_will_be_its_own(self):
        """Regression: slices that merely sit behind the cloud candidate's Ledger cooldown belong to the local
        executor for now, but come back to the cloud when it ends. Neither executor may go idle with
        wait_until=None, or nobody picks them up."""
        self.cloud.errors["groq-asr"] = [err("rate_limit", 40.0)]
        self.cloud_ex().run()                                 # slice 0 handed to local; cooldown ends at 1040
        self.lhost.limit = 95.0
        self.local_ex().run()                                 # local does slice 0 only
        self.assertEqual(self.state(0).status, sp.DONE)
        self.assertEqual(self.state(1).status, sp.PENDING)
        self.assertEqual(self.owner(1), "local")              # behind the cooldown
        res = self.cloud_ex().run()
        self.assertEqual((res.reason, res.waiting_executor), ("idle", "local"))
        self.assertEqual(res.wait_until, 1040.0)              # was None: nobody would ever come back
        self.clock.t = 1041.0                                 # local finished its batch after the cooldown
        res = self.cloud_ex().run(wait=True)                  # standalone: sleeps to the wake time, then works
        self.assertEqual(res.reason, "complete")
        self.assertEqual({self.state(i).candidate for i in range(1, 5)}, {"groq-asr"})

    def test_cloud_run_wait_sleeps_until_the_cooldown_ends_then_takes_the_work(self):
        self.cloud.errors["groq-asr"] = [err("rate_limit", 40.0)]
        self.cloud_ex().run()
        self.lhost.limit = 95.0
        self.local_ex().run()
        res = self.cloud_ex().run(wait=True)
        self.assertEqual(res.reason, "complete")
        self.assertGreaterEqual(self.clock(), 1040.0)
        self.assertEqual({self.state(i).candidate for i in range(1, 5)}, {"groq-asr"})

    def test_short_429_waits_on_the_cloud(self):
        self.cloud.errors["groq-asr"] = [err("rate_limit", 10.0)]
        res = self.cloud_ex().run()
        self.assertEqual(res.reason, "quota")
        st = self.state(0)
        self.assertEqual((st.status, st.candidate, st.wait_executor, st.retry_at),
                         (sp.PENDING, "groq-asr", None, 1010.0))
        self.assertEqual([self.owner(i) for i in range(5)], ["cloud"] * 5)
        self.assertEqual(self.local_ex().run().reason, "idle")
        self.assertEqual(self.models.calls, [])               # the GPU does not grab short-wait cloud work
        self.clock.t += 11
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        self.assertEqual({self.state(i).candidate for i in range(5)}, {"groq-asr"})

    def test_cloud_mt_switch_keeps_finished_batches_and_records_the_real_model(self):
        llm2 = cloud_row("llm2-mt", "translation")
        self.plan = frozen(self.GROQ, self.LLM, mt_fallback=[llm2])
        self.cloud.cues = 70                                  # 30 + 30 + 10 per slice
        self.cloud.errors["llm-mt"] = [None, err("auth")]     # second batch: key revoked
        res = self.cloud_ex(first_slice_seconds=90.0).run()      # a 30 s first slice (the default) holds 24 cues
        self.assertEqual(res.reason, "complete")
        mt0 = [c for c in self.cloud.calls if c[0] == "mt"][:4]
        self.assertEqual(mt0, [("mt", "llm-mt", 30), ("mt", "llm-mt", 30), ("mt", "llm2-mt", 30),
                               ("mt", "llm2-mt", 10)])        # batch 1 not translated again
        doc = self.doc(0, "mt")
        self.assertEqual(doc["model_id"], "mixed")
        self.assertEqual(doc["models_used"], ["llm-mt", "llm2-mt"])
        self.assertEqual(doc["candidate"], "llm2-mt")
        sources = [r["translation_source"]["id"] for r in doc["segments"]]
        self.assertEqual(sources[:30], ["llm-mt"] * 30)
        self.assertEqual(set(sources[30:]), {"llm2-mt"})
        self.assertFalse(list((self.folder / "cache" / "mt").glob("*.partial.json")))


class LocalPrimaryCloudBackupTests(Base):
    """Scenario B: local primary + cloud backup (D1: a busy GPU never spills over)."""

    def setUp(self):
        super().setUp()
        self.plan = frozen(self.GPU, self.SAKURA, asr_fallback=[self.GROQ], mt_fallback=[self.LLM])

    def test_three_local_errors_hand_off_to_the_cloud(self):
        self.models.fail_head = 3
        lres = self.local_ex().run(wait=True)
        st = self.state(0)
        self.assertEqual((st.status, st.candidate, st.wait_executor), (sp.PENDING, "groq-asr", "cloud"))
        self.assertEqual((st.tried["gpu-asr"]["kind"], st.tried["gpu-asr"]["attempt"]), ("error", 3))
        self.assertIsNone(st.reclaim_executor)                 # an error is not taken back
        self.assertEqual((lres.reason, lres.waiting_executor), ("idle", "cloud"))
        self.assertEqual(self.owner(0), "cloud")
        # the abandoned local model cools down once in the Ledger for the other units (policy 60 s)
        self.assertIsNotNone(self.ledger.peek(fb.candidate_key(self.ledger.account(self.GPU, "k-gpu-asr"),
                                                               self.GPU)))
        self.clock.t += 61
        self.drive()
        self.assertEqual(self.state(0).candidate, "groq-asr")
        self.assertEqual(self.cloud.count("asr"), 1)
        self.assertEqual(self.state(0, "mt").candidate, "sakura-mt")     # translation stays local
        self.assertEqual(self.cloud.count("mt"), 0)

    def test_busy_gpu_is_never_taken_over_by_the_cloud(self):
        self.lhost.busy = True
        self.assertEqual(self.local_ex().run().reason, "gpu_busy")
        res = self.cloud_ex().run()
        self.assertEqual((res.reason, res.waiting_executor), ("idle", "local"))
        self.assertEqual(self.cloud.calls, [])
        self.assertTrue(all(st.status == sp.PENDING and not st.tried
                            for s in self.disk().slices for st in s.tasks.values()))
        self.lhost.busy = False
        self.assertEqual(self.local_ex().run().reason, "complete")
        self.assertEqual(self.cloud.calls, [])


class ExhaustionAndChangesTests(Base):
    def test_every_candidate_used_up_is_failed_exhausted_with_each_reason(self):
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU])
        self.cloud.errors["groq-asr"] = [err("auth")]
        self.models.fail_head = 3
        self.cloud_ex().run()
        self.assertEqual(self.state(0).wait_executor, "local")
        lres = self.local_ex().run(wait=True)
        st = self.state(0)
        self.assertEqual((st.status, st.fail_kind), (sp.FAILED, "exhausted"))
        failure = [f for f in lres.status.failures if f.slice_index == 0][0]
        self.assertTrue(failure.exhausted)
        self.assertEqual([(c.id, c.kind, c.executor) for c in failure.candidates],
                         [("groq-asr", "auth", "cloud"), ("gpu-asr", "error", "local")])
        self.assertIn("groq-asr: auth", failure.detail)

    def test_a_backup_added_while_the_request_runs_is_used(self):
        self.plan = frozen(self.GROQ, self.LLM)               # no backup yet

        def add_backup(kind, candidate):
            if kind == "asr" and self.cloud.count("asr") == 1:
                self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU])     # user edits settings
        self.cloud.on_request = add_backup
        self.cloud.errors["groq-asr"] = [err("auth")]
        self.cloud_ex().run()
        st = self.state(0)
        self.assertEqual((st.status, st.candidate, st.wait_executor), (sp.PENDING, "gpu-asr", "local"))
        self.drive()
        self.assertEqual(self.state(0).candidate, "gpu-asr")

    def test_restart_reuses_results_of_the_other_executor(self):
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU], mt_fallback=[self.SAKURA])
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        asr_calls, mt_calls = self.cloud.count("asr"), self.cloud.count("mt")
        # restart with the order swapped (local primary now): only the local executor runs in this process
        self.plan = frozen(self.GPU, self.SAKURA, asr_fallback=[self.GROQ], mt_fallback=[self.LLM])
        res = self.local_ex().run()
        self.assertEqual(res.reason, "complete")
        self.assertEqual(self.models.calls, [])                # nothing recognised / translated again
        self.assertEqual({self.state(i).candidate for i in range(5)}, {"groq-asr"})
        self.assertTrue(res.publication["complete"])
        # ... and the other way round: local-made results, cloud alone after a restart
        self.plan = frozen(self.GPU, self.SAKURA)
        for f in (self.folder / "cache").rglob("*.json"):
            f.unlink()
        (self.folder / sp.SLICE_PLAN_FILE).unlink()
        self.assertEqual(self.local_ex().run().reason, "complete")
        calls = len(self.models.calls)
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU], mt_fallback=[self.SAKURA])
        res = self.cloud_ex().run()
        self.assertEqual(res.reason, "complete")
        self.assertEqual((self.cloud.count("asr"), self.cloud.count("mt")), (asr_calls, mt_calls))
        self.assertEqual(len(self.models.calls), calls)

    def test_removed_model_results_stay_done_while_their_file_exists(self):
        # a settings change must not silently redo finished work: the unit's own recorded result still counts
        self.plan = frozen(self.GPU, self.SAKURA)
        self.assertEqual(self.local_ex().run().reason, "complete")
        self.plan = frozen(self.GROQ, self.LLM)                # the local model is no longer configured
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        self.assertEqual(self.cloud.count("asr"), 0)
        self.assertEqual(self.cloud.count("mt"), 0)

    def test_done_unit_is_redone_when_its_own_result_file_is_missing(self):
        self.plan = frozen(self.GPU, self.SAKURA)
        self.assertEqual(self.local_ex().run().reason, "complete")
        for f in (self.folder / "cache" / "asr").glob("*.json"):
            f.unlink()                                          # the recorded recognition results are gone
        self.plan = frozen(self.GROQ, self.LLM)
        self.assertEqual(self.cloud_ex().run().reason, "complete")
        self.assertEqual(self.cloud.count("asr"), 5)


class SharedPlanTests(Base):
    def test_two_executors_at_once_share_one_plan_and_publication(self):
        # cloud primary + local backup; odd slices hit a long 429 on the cloud -> handed to the GPU while the
        # cloud keeps working on the rest, both threads in the same folder at the same time
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU], mt_fallback=[self.SAKURA])
        lock = threading.Lock()

        def slow(kind, candidate):
            import time as _t
            _t.sleep(0.002)
        self.cloud.on_request = slow
        self.cloud.errors["groq-asr"] = [None, err("rate_limit", 600.0)]
        for _ in range(12):
            c, l = self.cloud_ex(), self.local_ex()
            results = {}
            threads = [threading.Thread(target=lambda ex=ex, n=n: results.__setitem__(n, ex.run()))
                       for ex, n in ((c, "cloud"), (l, "local"))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            if results["cloud"].reason == "complete" and results["local"].reason == "complete":
                break
        else:
            self.fail(f"not complete: {results['cloud'].reason} / {results['local'].reason}")
        plan = self.disk()
        self.assertTrue(all(s.all_done() for s in plan.slices))
        by = {i: plan.slices[i].get("asr").candidate for i in range(5)}
        self.assertIn("gpu-asr", by.values())                  # some work went local ...
        self.assertIn("groq-asr", by.values())                 # ... the rest stayed on the cloud
        for i in range(5):                                      # every unit's result exists under its key
            self.assertIsNotNone(self.doc(i))
            self.assertIsNotNone(self.doc(i, "mt"))
            self.assertEqual(self.doc(i)["candidate"], by[i])
        self.assertIsNone(ss.PlanStore.peek(self.folder))     # released after both runs
        pub = results["cloud"].publication or results["local"].publication
        self.assertTrue(pub["complete"])
        self.assertEqual(pub["slices_done"], 5)

    def test_one_store_per_folder_while_both_run(self):
        c, l = self.cloud_ex(), self.local_ex()
        self.plan = frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU])
        self.assertTrue(c._prepare())
        self.assertTrue(l._prepare())
        self.assertIs(c.plan, l.plan)
        self.assertIs(c._lock, l._lock)
        self.assertIs(c._mt_rows, l._mt_rows)
        self.assertEqual(set(ss.PlanStore.peek(self.folder).profiles), {"cloud", "local"})
        c._release_store()
        l._release_store()
        self.assertIsNone(ss.PlanStore.peek(self.folder))


if __name__ == "__main__":
    unittest.main()
