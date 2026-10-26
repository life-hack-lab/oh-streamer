"""Tests for queue/subtitle_view.py (M5a). Pure logic, CPU only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oh_streamer.queue.subtitle_scheduler import PlanStore
from oh_streamer.queue.subtitle_view import candidate_names, retry_failed, subtitle_view
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp

NOW = 1000.0


def uniform(duration=900.0):
    return sp.build_slice_plan(duration, first_slice_seconds=None)   # 3 slices of 300 s


def finish(plan, indices):
    for i in indices:
        plan.mark_done(i, "asr", "a")
        plan.mark_done(i, "mt", "m")


def view(plan, **kw):
    kw.setdefault("now", NOW)
    return subtitle_view(plan, **kw)


class StateTests(unittest.TestCase):
    def test_gap_before_island_waits_for_cutting(self):
        plan = sp.build_slice_plan(3600.0, first_slice_seconds=None)
        plan.slices = [sp.Slice(0, 2400.0, 2430.0)]
        plan.lazy["closed"] = False
        v = view(plan, play_pos=1200.0)
        self.assertEqual(v["state"], "waiting_cut")
        self.assertEqual(v["wait"]["kind"], "cut")

    def test_unreadable_gap_still_waits_for_download(self):
        plan = sp.build_slice_plan(3600.0, first_slice_seconds=None)
        plan.slices = []
        plan.lazy["closed"] = False
        v = view(plan, play_pos=1200.0, max_available=100.0)
        self.assertEqual(v["state"], "waiting_download")

    def test_complete(self):
        plan = uniform()
        finish(plan, [0, 1, 2])
        plan.lazy["closed"] = True
        v = view(plan)
        self.assertEqual(v["state"], "complete")
        self.assertEqual(v["progress"], 1.0)
        self.assertFalse(v["retryable"])
        json.dumps(v)

    def test_stuck_lists_candidates_with_names(self):
        plan = uniform()
        plan.slices[0].set("asr", "primary", sp.TaskState(
            status=sp.FAILED, fail_kind="exhausted", fail_at=900.0, candidate="local",
            tried={"groq": {"kind": "rate_limit", "at": 800.0, "detail": "429", "executor": "cloud"},
                   "local": {"kind": "server", "at": 900.0, "detail": "oom", "executor": "local"}}))
        v = view(plan, candidate_names={"groq": "Groq"})
        self.assertEqual(v["state"], "stuck")
        self.assertTrue(v["retryable"])
        cands = v["stuck"]["candidates"]
        self.assertEqual([c["name"] for c in cands], ["Groq", "local"])
        self.assertEqual(cands[0]["kind"], "rate_limit")
        self.assertEqual(v["stuck"]["stage"], "asr")
        self.assertEqual(v["failures_elsewhere"], 0)
        json.dumps(v)

    def test_stuck_without_tried_gets_synthetic_row(self):
        plan = uniform()
        plan.mark_failed(0, "asr", "auth", at=5.0, model_id="groq", detail="bad key")
        v = view(plan)
        self.assertEqual(v["stuck"]["candidates"][0]["kind"], "auth")
        self.assertEqual(v["stuck"]["candidates"][0]["name"], "groq")

    def test_running(self):
        plan = uniform()
        plan.mark_running(0, "asr", "cloud", candidate="groq")
        v = view(plan, candidate_names={"groq": "Groq"})
        self.assertEqual(v["state"], "running")
        self.assertEqual(v["running"]["executor"], "cloud")
        self.assertEqual(v["running"]["candidate_name"], "Groq")
        self.assertEqual((v["running"]["slice"], v["running"]["stage"]), (0, "asr"))

    def test_waiting_quota_and_expiry(self):
        plan = uniform()
        plan.defer(0, "asr", NOW + 30)
        v = view(plan)
        self.assertEqual(v["state"], "waiting_quota")
        self.assertEqual(v["wait"]["remaining_s"], 30.0)
        self.assertEqual(v["wait"]["retry_at"], NOW + 30)
        # past retry_at: no longer a quota wait, no stale event
        v = view(plan, now=NOW + 31)
        self.assertEqual(v["state"], "queued")
        self.assertIsNone(v["wait"])

    def test_waiting_quota_names_the_waiting_candidate(self):
        plan = uniform()
        plan.slices[0].set("asr", "primary", sp.TaskState(
            status=sp.PENDING, retry_at=NOW + 10, candidate="local",
            tried={"groq": {"kind": "rate_limit", "at": 990.0, "executor": "cloud", "retry_at": NOW + 10}}))
        v = view(plan, candidate_names={"groq": "Groq"})
        self.assertEqual(v["wait"]["candidate_name"], "Groq")

    def _pending(self, kind, *, at, retry_at, cid="local", executor="local", attempt=1):
        return sp.TaskState(status=sp.PENDING, retry_at=retry_at, attempt=attempt, candidate=cid,
                            tried={cid: {"kind": kind, "at": at, "executor": executor, "detail": "x"}})

    def test_local_error_backoff_is_retrying_not_waiting_quota(self):
        for kind in ("error", "oom", "network", "timeout", "server"):
            plan = uniform()
            plan.slices[0].set("mt", "primary", self._pending(kind, at=NOW, retry_at=NOW + 30))
            plan.mark_done(0, "asr", "a")
            v = view(plan)
            self.assertEqual(v["state"], "retrying", kind)
            self.assertEqual(v["wait"]["kind"], "retry")
            self.assertEqual(v["wait"]["reason"], kind)
            self.assertEqual(v["wait"]["remaining_s"], 30.0)
            self.assertEqual(v["wait"]["stage"], "mt")
            json.dumps(v)
        # backoff over: plain queued, no stale event
        self.assertEqual(view(plan, now=NOW + 31)["state"], "queued")

    def test_rate_limit_and_daily_quota_are_waiting_quota(self):
        for kind in ("rate_limit", "quota_daily"):
            plan = uniform()
            plan.slices[0].set("asr", "primary", self._pending(kind, at=NOW - 20, retry_at=NOW + 30,
                                                               cid="groq", executor="cloud", attempt=0))
            v = view(plan)
            self.assertEqual(v["state"], "waiting_quota", kind)
            self.assertEqual(v["wait"]["kind"], "quota")

    def test_short_wait_while_running_stays_running(self):
        # 429 pacing: the blocking unit bounces between running and a 2 s wait; it must not flip the state
        plan = uniform()
        plan.slices[0].set("asr", "primary", self._pending("rate_limit", at=NOW - 1, retry_at=NOW + 1,
                                                           cid="qwen", executor="cloud", attempt=0))
        for now in (NOW, NOW + 1.5, NOW + 3.0):    # waiting, just past retry_at, before the worker picks it up
            v = view(plan, now=now, candidate_names={"qwen": "Qwen"})
            self.assertEqual(v["state"], "running", now)
            self.assertEqual(v["running"]["executor"], "cloud")
            self.assertEqual(v["running"]["candidate_name"], "Qwen")
            self.assertEqual(v["running"]["quota_wait"], now < NOW + 1)
        self.assertNotEqual(view(plan, now=NOW + 10)["state"], "running")   # long idle: no longer smoothed

    def test_other_work_running_while_blocking_unit_waits_quota(self):
        plan = uniform()
        plan.slices[0].set("asr", "primary", self._pending("rate_limit", at=NOW - 5, retry_at=NOW + 30,
                                                           cid="qwen", executor="cloud", attempt=0))
        plan.mark_running(1, "asr", "cloud", candidate="qwen")
        v = view(plan)
        self.assertEqual(v["state"], "running")
        self.assertEqual((v["running"]["slice"], v["running"]["stage"]), (1, "asr"))
        self.assertTrue(v["running"]["quota_wait"])
        self.assertEqual(v["running"]["quota_retry_at"], NOW + 30)

    def test_waiting_gpu(self):
        plan = uniform()
        plan.wait_for(0, "asr", "local")
        self.assertEqual(view(plan, gpu_turn=False)["state"], "waiting_gpu")
        self.assertEqual(view(plan, gpu_turn=True)["state"], "waiting_executor")
        self.assertEqual(view(plan, gpu_turn=None)["state"], "waiting_executor")

    def test_waiting_executor_around_reclaim_at(self):
        plan = uniform()
        plan.slices[0].set("asr", "primary", sp.TaskState(
            status=sp.PENDING, wait_executor="local", reclaim_executor="cloud", reclaim_at=NOW + 60))
        v = view(plan)
        self.assertEqual(v["state"], "waiting_executor")
        self.assertEqual(v["wait"]["executor"], "local")
        self.assertEqual(v["wait"]["reclaim_in_s"], 60.0)
        v = view(plan, now=NOW + 61)       # cloud may take it back
        self.assertEqual(v["state"], "waiting_executor")
        self.assertEqual(v["wait"]["executor"], "cloud")
        self.assertIsNone(v["wait"]["reclaim_at"])

    def test_waiting_download(self):
        plan = uniform()
        v = view(plan, max_available=100.0)
        self.assertEqual(v["state"], "waiting_download")
        self.assertEqual(view(plan, max_available=None)["state"], "queued")

    def test_waiting_download_when_everything_cut_is_done(self):
        plan = uniform()
        plan.lazy["closed"] = False
        finish(plan, [0, 1, 2])
        self.assertEqual(view(plan)["state"], "waiting_download")

    def test_queued(self):
        self.assertEqual(view(uniform())["state"], "queued")

    def test_priority_quota_over_download_and_executor(self):
        plan = uniform()
        plan.wait_for(0, "asr", "local")
        plan.defer(0, "asr", NOW + 20)
        v = view(plan, max_available=10.0, gpu_turn=False)
        self.assertEqual(v["state"], "waiting_quota")
        self.assertTrue(v["wait"]["also_waiting_download"])

    def test_priority_gpu_over_download(self):
        plan = uniform()
        plan.wait_for(0, "asr", "local")
        self.assertEqual(view(plan, max_available=10.0, gpu_turn=False)["state"], "waiting_gpu")

    def test_running_over_stuck_elsewhere(self):
        plan = uniform()
        finish(plan, [0])
        plan.mark_failed(2, "asr", "exhausted")
        plan.mark_running(1, "asr", "cloud")
        v = view(plan)
        self.assertEqual(v["state"], "running")
        self.assertEqual(v["failures_elsewhere"], 1)
        self.assertTrue(v["partial_stuck"])

    def test_stuck_over_everything(self):
        plan = uniform()
        plan.mark_failed(0, "asr", "exhausted")
        plan.mark_running(1, "asr", "cloud")
        self.assertEqual(view(plan)["state"], "stuck")


def frozen_plan(local_primary):
    cloud = {"id": "groq-asr", "kind": "asr", "provider": "openai-compatible", "connection_id": "c",
             "remote_model": "whisper-large-v3", "base_url": "https://x.invalid/v1"}
    llm = {"id": "llm-mt", "kind": "translation", "provider": "openai-compatible", "connection_id": "c2",
           "remote_model": "sim-llm", "base_url": "https://x.invalid/v1"}
    gpu = {"id": "gpu-asr", "kind": "asr", "provider": "local", "recipe": "__dual__", "model_path": "C:/m/whisper"}
    sakura = {"id": "sakura-mt", "kind": "translation", "provider": "local", "recipe": "sakura-v1",
              "model_path": "C:/m/m.gguf"}
    if local_primary:
        return {"models": {"asr": gpu, "translation": sakura}, "fallback": {"asr": [cloud], "translation": [llm]},
                "policy": {}, "whisper_policy": "fixed", "whisper_models": [], "ready": True}
    return {"models": {"asr": cloud, "translation": llm}, "fallback": {"asr": [gpu], "translation": [sakura]},
            "policy": {}, "whisper_policy": "fixed", "whisper_models": [], "ready": True}


class OwnerTests(unittest.TestCase):
    def test_fresh_local_primary_waits_for_gpu(self):
        plan = uniform()
        owner = fb.owner_function(frozen_plan(True), now=NOW)
        self.assertEqual(view(plan, gpu_turn=False)["state"], "queued")           # without owner: unchanged
        v = view(plan, gpu_turn=False, owner=owner)
        self.assertEqual(v["state"], "waiting_gpu")
        self.assertEqual(v["wait"]["kind"], "gpu")
        self.assertEqual(view(plan, gpu_turn=True, owner=owner)["state"], "queued")
        self.assertEqual(view(plan, gpu_turn=None, owner=owner)["state"], "queued")

    def test_fresh_cloud_primary_stays_queued(self):
        owner = fb.owner_function(frozen_plan(False), now=NOW)
        self.assertEqual(view(uniform(), gpu_turn=False, owner=owner)["state"], "queued")

    def test_handoff_record_keeps_waiting_executor(self):
        plan = uniform()
        plan.wait_for(0, "asr", "local")
        owner = fb.owner_function(frozen_plan(False), now=NOW)       # cloud primary; stored handoff to local
        self.assertEqual(view(plan, gpu_turn=False, owner=owner)["state"], "waiting_gpu")
        self.assertEqual(view(plan, gpu_turn=True, owner=owner)["state"], "waiting_executor")

    def test_other_executor_owner_without_record_is_queued(self):
        owner = lambda stage, track, state: "cloud"
        self.assertEqual(view(uniform(), gpu_turn=False, owner=owner)["state"], "queued")

    def test_candidate_names(self):
        names = candidate_names(frozen_plan(True))
        self.assertEqual(names["groq-asr"], "whisper-large-v3")
        self.assertEqual(names["llm-mt"], "sim-llm")
        self.assertEqual(names["sakura-mt"], "sakura-v1")
        self.assertEqual(names["gpu-asr"], "whisper")


class FailuresElsewhereTests(unittest.TestCase):
    def test_failure_outside_playback_position(self):
        plan = uniform()
        plan.mark_failed(0, "asr", "exhausted")
        v = view(plan, play_pos=700.0)       # playing in slice 2; slice 0 failed behind us
        self.assertNotEqual(v["state"], "stuck")
        self.assertEqual(v["failures_elsewhere"], 1)
        self.assertTrue(v["partial_stuck"])
        self.assertTrue(v["retryable"])

    def test_done_work_with_tried_is_not_a_failure(self):
        plan = uniform()
        plan.slices[0].set("asr", sp.PRIMARY_TRACK, sp.TaskState(
            status=sp.RUNNING, candidate="c2", tried={"c1": {"kind": "rate_limit", "at": 1.0}}))
        plan.mark_done(0, "asr", "a")
        plan.mark_done(0, "mt", "m")
        v = view(plan, play_pos=700.0)
        self.assertEqual((v["failures_elsewhere"], v["partial_stuck"], v["retryable"]), (0, False, False))
        self.assertNotEqual(v["state"], "stuck")

    def test_no_failures_means_not_retryable(self):
        v = view(uniform())
        self.assertEqual((v["failures_elsewhere"], v["partial_stuck"], v["retryable"]), (0, False, False))


class RetryTests(unittest.TestCase):
    def test_reset_leaves_no_stuck_and_clears_tried(self):
        plan = uniform()
        plan.slices[0].set("asr", "primary", sp.TaskState(
            status=sp.FAILED, fail_kind="exhausted", tried={"groq": {"kind": "rate_limit", "at": 1.0}}))
        self.assertEqual(view(plan)["state"], "stuck")
        self.assertEqual(retry_failed(plan), 1)
        v = view(plan)
        self.assertEqual(v["state"], "queued")
        self.assertIsNone(v["stuck"])
        self.assertFalse(v["retryable"])
        self.assertEqual(plan.slices[0].get("asr").tried, {})

    def test_retry_with_store_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PlanStore(Path(tmp))
            store.plan = uniform()
            store.plan.mark_failed(1, "mt", "exhausted")
            self.assertEqual(retry_failed(store, index=0), 0)
            self.assertEqual(retry_failed(store, index=1, stage="mt"), 1)
            loaded = sp.load_slice_plan(tmp)
            self.assertFalse(any(t.status == sp.FAILED for s in loaded.slices for t in s.tasks.values()))

    def test_view_is_json_serializable_with_mixed_state(self):
        plan = uniform()
        plan.defer(0, "asr", NOW + 5)
        plan.mark_failed(2, "asr", "exhausted")
        json.dumps(view(plan))


if __name__ == "__main__":
    unittest.main()
