"""M7 / T3: the unified subtitle scheduler wired into a real QueueService (fake executor class, no model, no cloud).

Covers the lifecycle: item queued -> tick starts the cloud executor thread -> RunResult idle with wait_until ->
restarted only when the time is reached -> complete (not restarted); pause / remove / pause_all / consent
withdrawal stop the thread; `_step` no longer starts the old subtitle pipeline; the R2 turn hold keeps the
restoration worker and the mux from starting."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import threading
import time
import unittest
from unittest.mock import patch

from oh_streamer.queue import queue_service, queue_store
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import slice_plan as sp
from tests import test_queue_service as qts
from tests.test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, frozen


class FakeExecutor:
    """Stands in for CloudSliceExecutor: records starts, blocks until told how to end."""

    instances: list = []

    def __init__(self, outcomes):
        self.outcomes = outcomes            # shared list of RunResult to return, one per run
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.release = threading.Event()
        FakeExecutor.instances.append(self)

    def run(self, wait=True):
        self.started.set()
        outcome = self.outcomes.pop(0) if self.outcomes else ss.RunResult("complete")
        if outcome == "block":
            while not self.stopped.is_set():
                self.stopped.wait(0.02)
            return ss.RunResult("cancelled")
        return outcome

    def stop(self):
        self.stopped.set()


class WiringTests(qts.ServiceTestBase):
    def setUp(self):
        super().setUp()
        FakeExecutor.instances = []
        self.outcomes = []
        self.t = [5000.0]
        self.svc._stopping.set()                       # drive the tick by hand
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._subtitle_clock_fn = lambda: self.t[0]
        self.svc.store.set_settings(subtitle_cloud_enabled=True)
        patcher = patch.object(self.svc, "_subtitle_new_cloud_executor",
                               side_effect=lambda item, host: FakeExecutor(self.outcomes))
        patcher.start()
        self.addCleanup(patcher.stop)
        # the local half needs real models: keep it quiet
        local = patch.object(self.svc, "_subtitle_local_tick", lambda: None)
        local.start()
        self.addCleanup(local.stop)
        self.item_id = self.ready(state="done")
        item = self.svc.store.get(self.item_id)
        settings = {**item.settings, "subtitles": True, "subtitle_runtime_contract": "cloud-first-1",
                    "subtitle_resolved": frozen(CLOUD_ASR, CLOUD_MT), "subtitle_configs": [{}]}
        self.svc.store.update(self.item_id, settings=settings, subtitle={"status": "queued", "percent": 0, "only": True})
        self.svc._paused_all = False                   # ready() leaves pause_all on

    def wait_for(self, fn, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if fn():
                return True
            time.sleep(0.01)
        return False

    def slot_empty(self):
        return self.svc._subtitle_cloud is None

    def tick(self):
        self.svc._subtitle_executor_tick()

    def test_step_runs_the_executor_tick_not_the_old_pipeline(self):
        with patch.object(self.svc, "_subtitle_executor_tick") as tick:
            self.svc._step()
        tick.assert_called_once()
        self.assertFalse(hasattr(self.svc, "_subtitle_pipeline"))

    def test_idle_wait_until_then_restart_then_complete(self):
        self.outcomes[:] = [ss.RunResult("idle", wait_until=self.t[0] + 100), ss.RunResult("complete")]
        self.tick()
        self.assertTrue(self.wait_for(lambda: FakeExecutor.instances and self.slot_empty()))
        self.assertEqual(len(FakeExecutor.instances), 1)
        self.tick()                                            # wait_until not reached: no new thread
        self.assertTrue(self.slot_empty())
        self.assertEqual(len(FakeExecutor.instances), 1)
        self.t[0] += 101
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(FakeExecutor.instances) == 2 and self.slot_empty()))
        self.t[0] += 1000
        self.tick()                                            # complete: never restarted
        self.assertTrue(self.slot_empty())
        self.assertEqual(len(FakeExecutor.instances), 2)

    def start_blocking(self):
        self.outcomes[:] = ["block"]
        self.tick()
        self.assertTrue(self.wait_for(lambda: FakeExecutor.instances and FakeExecutor.instances[0].started.is_set()))
        self.assertEqual(self.svc._subtitle_cloud.stage, "subtitle_cloud")
        return FakeExecutor.instances[0]

    def test_pause_stops_the_executor_thread(self):
        ex = self.start_blocking()
        thread = self.svc._subtitle_cloud.thread
        self.svc.pause(self.item_id)
        self.assertTrue(ex.stopped.is_set())
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_remove_stops_and_waits_for_the_executor_thread(self):
        ex = self.start_blocking()
        thread = self.svc._subtitle_cloud.thread
        result = self.svc.remove(self.item_id)
        self.assertTrue(result["ok"], result)
        self.assertTrue(ex.stopped.is_set())
        self.assertFalse(thread.is_alive())

    def test_pause_all_stops_the_executor_thread(self):
        ex = self.start_blocking()
        thread = self.svc._subtitle_cloud.thread
        self.svc.pause_all()
        self.assertTrue(ex.stopped.is_set())
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_service_stop_stops_the_executor_thread(self):
        ex = self.start_blocking()
        thread = self.svc._subtitle_cloud.thread
        self.svc.stop()
        self.assertTrue(ex.stopped.is_set())
        self.assertFalse(thread.is_alive())

    def test_withdrawing_cloud_consent_stops_the_cloud_executor(self):
        ex = self.start_blocking()
        thread = self.svc._subtitle_cloud.thread
        self.svc.set_settings({"subtitle_cloud_enabled": False})
        self.assertTrue(ex.stopped.is_set())
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_turn_hold_keeps_restoration_worker_and_mux_from_starting(self):
        queued = self.ready(state="queued")
        self.svc._paused_all = False
        self.svc._subtitle_turn_hold = queued
        self.svc._maybe_start_worker()
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.svc.store.get(queued).state, "queued")
        self.svc.store.update(self.item_id, subtitle={"status": "mux_pending", "only": True, "percent": 100})
        with patch.object(self.svc, "_maybe_start_subtitle_only") as mux:
            self.svc._step()
        mux.assert_not_called()
        self.svc._subtitle_turn_hold = None
        with patch.object(self.svc, "_maybe_start_subtitle_only") as mux:
            self.svc._step()
        mux.assert_called()


    # -- M7 T5: view, mux, play, legacy items -------------------------------------------------------
    def folder(self):
        folder = self.svc.store.item_dir(self.item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_plan(self, finished=(), *, closed=True, duration=900.0):
        plan = sp.build_slice_plan(duration, first_slice_seconds=None)          # 3 slices of 300 s
        for i in finished:
            plan.mark_done(i, "asr", "a")
            plan.mark_done(i, "mt", "m")
        plan.lazy["closed"] = closed
        sp.save_slice_plan(plan, self.folder())
        return plan

    def sub(self):
        return self.svc.store.get(self.item_id).subtitle

    def test_tick_writes_the_view_and_a_changed_table_updates_it(self):
        self.save_plan([0])
        self.svc._subtitle_view_tick()
        view = self.sub()["view"]
        self.assertEqual(view["v"], 2)
        self.assertEqual((view["slices_done"], view["slices_total"]), (1, 3))
        self.save_plan([0, 1])
        self.svc._subtitle_view_tick()
        self.assertEqual(self.sub()["view"]["slices_done"], 2)
        json.dumps(self.sub()["view"])

    def test_view_countdown_does_not_rewrite_queue_json(self):
        """R6: remaining_s changes every second; only a real change may write and emit."""
        plan = sp.build_slice_plan(900.0, first_slice_seconds=None)
        plan.slices[0].set("asr", "primary", sp.TaskState(retry_at=self.t[0] + 100))
        sp.save_slice_plan(plan, self.folder())
        self.svc._subtitle_view_tick()
        first = self.sub()["view"]
        self.assertEqual(first["state"], "waiting_quota")
        self.assertEqual(first["wait"]["remaining_s"], 100.0)
        with patch.object(self.svc, "_subtitle_update", wraps=self.svc._subtitle_update) as write:
            for _ in range(5):
                self.t[0] += 1
                self.svc._subtitle_view_tick()
            write.assert_not_called()
        self.assertEqual(self.sub()["view"], first)
        self.t[0] += 200                                         # retry_at passed: a real change
        self.svc._subtitle_view_tick()
        self.assertEqual(self.sub()["view"]["state"], "queued")

    def test_view_play_pos_follows_the_clock_minus_audio_offset(self):
        self.save_plan([0])
        self.svc._subtitle_update(self.item_id, audio_offset=10.0)
        self.svc._subtitle_clocks[self.item_id] = (320.0, True)            # source time; 310 s of subtitle time
        with patch("oh_streamer.queue.subtitle_view.subtitle_view", wraps=__import__(
                "oh_streamer.queue.subtitle_view", fromlist=["x"]).subtitle_view) as spy:
            self.svc._subtitle_view_tick()
        self.assertAlmostEqual(spy.call_args.kwargs["play_pos"], 310.0)
        self.assertTrue(spy.call_args.kwargs["gpu_turn"])
        self.assertTrue(callable(spy.call_args.kwargs["owner"]))
        self.assertIsNotNone(spy.call_args.kwargs["candidate_names"])

    def test_gpu_turn_is_false_only_while_restoration_holds_the_gpu(self):
        item = self.svc.store.get(self.item_id)
        self.assertTrue(self.svc._subtitle_gpu_turn(item))
        run = queue_service._Run(self.item_id, 0)
        run.stage = "restore"
        self.svc._wk = run
        try:
            self.assertFalse(self.svc._subtitle_gpu_turn(item))
            run.stage = "subtitle_lease"
            self.assertTrue(self.svc._subtitle_gpu_turn(item))
        finally:
            self.svc._wk = None

    def test_item_without_subtitles_or_table_gets_no_view(self):
        self.svc._subtitle_view_tick()
        self.assertNotIn("view", self.sub())                      # no slice table yet: the UI shows queued
        self.svc._subtitle_update(self.item_id, view={"v": 2, "state": "running"})
        self.svc._subtitle_view_tick()
        self.assertNotIn("view", self.sub())                      # table deleted (restart): stale view removed

    def test_finished_table_and_srt_make_a_done_item_mux_pending(self):
        self.save_plan([0, 1, 2])
        self.svc._subtitle_update(self.item_id, status="partial", only=False)
        self.svc._subtitle_view_tick()                           # no srt yet: nothing to mux
        self.assertEqual(self.sub()["status"], "partial")
        (self.folder() / "sakura.zh-TW.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        self.svc._subtitle_view_tick()
        self.assertEqual(self.sub()["status"], "mux_pending")
        self.assertTrue(self.sub()["only"])
        self.assertEqual(self.sub()["view"]["state"], "complete")

    def test_unfinished_table_never_becomes_mux_pending(self):
        self.save_plan([0, 1])
        (self.folder() / "sakura.zh-TW.srt").write_text("partial", encoding="utf-8")
        self.svc._subtitle_update(self.item_id, status="partial")
        self.svc._subtitle_view_tick()
        self.assertEqual(self.sub()["status"], "partial")

    def test_app_reopened_in_mux_pending_completes(self):
        """R11: mux_pending + finished slice table + final srt after a restart -> the mux thread starts."""
        self.save_plan([0, 1, 2])
        (self.folder() / "sakura.zh-TW.srt").write_text("complete", encoding="utf-8")
        self.svc._subtitle_update(self.item_id, status="mux_pending", only=True)
        restarted = queue_store.QueueStore(self.svc.store.root)
        recovered = restarted.get(self.item_id)
        self.assertEqual((recovered.state, recovered.subtitle["status"]), ("done", "mux_pending"))
        self.assertTrue(self.svc._subtitle_mux_ready(recovered))
        with patch.object(queue_service.threading, "Thread") as thread:
            self.svc._maybe_start_subtitle_only()
        thread.return_value.start.assert_called_once()
        self.svc._wk = None

    def test_executor_complete_on_a_done_item_publishes_mux_pending(self):
        from oh_streamer.queue.subtitle_host import QueueSubtitleHost
        host = QueueSubtitleHost(self.svc, self.item_id)
        with patch.object(self.svc, "_send_to_player", return_value={"ok": True}):
            host.publish({"complete": True, "covered_s": 900.0, "srt": ["a.srt"], "cues": [1], "done_indices": [0, 1, 2]})
        self.assertEqual((self.sub()["status"], self.sub()["only"]), ("mux_pending", True))
        self.svc._subtitle_after_restore(self.item_id)           # the video finishing later keeps it
        self.assertEqual(self.sub()["status"], "mux_pending")

    def test_source_is_kept_until_the_slice_table_is_complete(self):
        pending = queue_service.QueueService._subtitle_source_pending
        item = self.svc.store.get(self.item_id)
        self.assertTrue(pending(item))                           # queued, no view yet
        item.subtitle = {"status": "partial", "view": {"state": "running"}}
        self.assertTrue(pending(item))
        item.subtitle = {"status": "partial", "view": {"state": "complete"}}
        self.assertFalse(pending(item))
        item.subtitle = {"status": "paused"}                    # a paused item may resume: keep the source
        self.assertTrue(pending(item))
        item.subtitle = {"status": "mux_pending"}
        self.assertFalse(pending(item))
        item.subtitle = {}                                       # no subtitle job at all
        self.assertFalse(pending(item))

    def test_ipc_clock_ok_follows_the_slice_table(self):
        self.assertFalse(self.svc._ipc_clock({"id": self.item_id, "pos": 5, "playing": True})["ok"])
        self.save_plan([0])
        self.assertTrue(self.svc._ipc_clock({"id": self.item_id, "pos": 6, "playing": True})["ok"])
        self.assertEqual(self.svc._subtitle_clocks[self.item_id], (6.0, True))

    def test_old_finished_item_loads_done_and_keeps_its_subtitle(self):
        """R10: a completed pre-M7 item (legacy fields, no slice table) stays done, stays playable, and shows as
        complete; the legacy fields are dropped on load."""
        folder = self.folder()
        srt = folder / "sakura.zh-TW.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕\n", encoding="utf-8")
        legacy = {"status": "done", "percent": 100, "embedded": True, "audio_offset": 0.5, "covered_s": 900.0,
                  "block_srt": [str(srt)], "block_mode": True, "blocks_done": 5, "blocks_total": 5,
                  "cloud_mode": True, "cloud_done_slices": 3, "backup_status": "x", "retry_note": "x",
                  "exhausted": True, "ready_identity": {"a": 1}, "combos": {"items": []},
                  "fallback": {"state": "exhausted"}, "view": {"v": 1}}
        self.svc.store.update(self.item_id, subtitle=legacy)
        recovered = queue_store.QueueStore(self.svc.store.root).get(self.item_id)
        self.assertEqual((recovered.state, recovered.subtitle["status"]), ("done", "done"))
        for key in ("block_mode", "blocks_done", "blocks_total", "cloud_mode", "cloud_done_slices", "backup_status",
                    "retry_note", "exhausted", "ready_identity", "combos", "fallback", "view"):
            self.assertNotIn(key, recovered.subtitle)
        self.assertEqual(recovered.subtitle["block_srt"], [str(srt)])
        self.assertEqual(recovered.subtitle["covered_s"], 900.0)
        self.svc.store.update(self.item_id, subtitle=recovered.subtitle)
        self.svc._subtitle_view_tick()
        self.assertEqual(self.sub()["view"]["state"], "complete")
        self.assertEqual(self.sub()["status"], "done")
        payload = {"processed": True, "progressive": True}
        self.svc._add_external_subtitles(payload, self.svc.store.get(self.item_id))
        self.assertEqual([t["path"] for t in payload["subtitles"]], [str(srt)])

    def test_old_blocked_item_is_requeued_not_lost(self):
        self.svc.store.update(self.item_id, subtitle={"status": "blocked", "covered_s": 120.0, "retry_at": 5.0})
        recovered = queue_store.QueueStore(self.svc.store.root).get(self.item_id)
        self.assertEqual(recovered.subtitle["status"], "partial")
        self.assertNotIn("retry_at", recovered.subtitle)


if __name__ == "__main__":
    unittest.main()
