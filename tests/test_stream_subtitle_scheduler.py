"""T3.1-A: a stream item runs the cloud subtitle executor in its own slot (`_subtitle_stream_cloud`) next to the regular
cloud slot. Real QueueService, fake executors (the `_subtitle_new_cloud_executor` seam), no model, no network, no GPU.
Run from streamer/: PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_subtitle_scheduler."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading
import time
import unittest
from unittest.mock import patch

from oh_streamer.queue import queue_service as QS
from oh_streamer.queue import subtitle_host as sh
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import fallback as fb

try:
    from tests import test_queue_service as qts
    from tests.test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen
except ImportError:                                    # run from tests/ directly
    import test_queue_service as qts
    from test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen


class FakeExecutor:
    """Stands in for CloudSliceExecutor: blocks until stop(), or returns a scripted RunResult at once."""

    def __init__(self, host, outcome="block"):
        self.host = host
        self.outcome = outcome
        self.started = threading.Event()
        self.stopped = threading.Event()

    def run(self, wait=True):
        self.started.set()
        if self.outcome == "block":
            while not self.stopped.is_set():
                self.stopped.wait(0.02)
            return ss.RunResult("cancelled")
        return self.outcome

    def stop(self):
        self.stopped.set()


class Base(qts.ServiceTestBase):
    def setUp(self):
        super().setUp()
        self.svc._stopping.set()                        # the tick is driven by hand
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.t = [5000.0]
        self.svc._subtitle_clock_fn = lambda: self.t[0]
        self.svc._subtitle_ledger_obj = fb.Ledger(path=self.tmp / "ledger.sqlite3")      # never the user's ledger
        self.svc.store.set_settings(subtitle_runtime_contract="legacy", subtitle_cloud_enabled=True)
        self.executors = []
        self.outcome = "block"
        patcher = patch.object(self.svc, "_subtitle_new_cloud_executor", side_effect=self.new_executor)
        patcher.start()
        self.addCleanup(patcher.stop)
        local = patch.object(self.svc, "_subtitle_local_tick", lambda: None)      # no local models here
        local.start()
        self.addCleanup(local.stop)
        self.source = self.tmp / "audio.rstream"
        self.source.write_bytes(b"x")

    def tearDown(self):
        for ex in self.executors:
            ex.stopped.set()
        super().tearDown()

    def new_executor(self, item, host):
        ex = FakeExecutor(host, self.outcome)
        ex.item = item
        self.executors.append(ex)
        return ex

    def make_stream(self, audio="ready", error="", plan=None, subtitles=True, **settings):
        item = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": subtitles}, title="Clip")
        merged = {**item.settings, "subtitles": subtitles, "subtitle_runtime_contract": "cloud-first-1",
                  "subtitle_resolved": plan or frozen(CLOUD_ASR, CLOUD_MT), "subtitle_configs": [{}], **settings}
        self.svc.store.update(item.id, settings=merged, source_path=str(self.source),
                              stream={**item.stream, "audio_status": audio, "audio_error": error})
        return item.id

    def make_queue_item(self):
        item_id = self.ready(state="processing")
        item = self.svc.store.get(item_id)
        settings = {**item.settings, "subtitles": True, "subtitle_runtime_contract": "cloud-first-1",
                    "subtitle_resolved": frozen(CLOUD_ASR, CLOUD_MT), "subtitle_configs": [{}]}
        self.svc.store.update(item_id, settings=settings, subtitle={"status": "queued", "percent": 0})
        self.svc._paused_all = False
        return item_id

    def wait_for(self, fn, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if fn():
                return True
            time.sleep(0.01)
        return False

    def tick(self):
        self.svc._subtitle_executor_tick()

    def subtitle(self, item_id):
        return self.svc.store.get(item_id).subtitle

    def start_stream_executor(self, item_id=None):
        item_id = item_id or self.make_stream()
        self.tick()
        self.assertTrue(self.wait_for(lambda: self.executors and self.executors[-1].started.is_set()))
        return item_id, self.executors[-1]


class StartConditionTests(Base):
    def test_audio_ready_starts_the_executor_in_the_stream_slot(self):
        sid = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: self.executors and self.executors[0].started.is_set()))
        run = self.svc._subtitle_stream_cloud
        self.assertEqual((run.item_id, run.stage), (sid, "subtitle_cloud"))
        self.assertIsNone(self.svc._subtitle_cloud)
        self.assertEqual(self.executors[0].host.item_id, sid)
        self.assertEqual(self.executors[0].host.executor, "cloud")

    def test_probing_does_not_start(self):
        sid = self.make_stream("probing")
        self.tick()
        self.assertIsNone(self.svc._subtitle_stream_cloud)
        self.assertEqual(self.executors, [])
        self.assertEqual(self.subtitle(sid)["status"], "queued")              # nothing failed, it just waits

    def test_not_attached_yet_does_not_start(self):
        item = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": True}, title="Clip")
        self.svc.store.update(item.id, settings={**item.settings, "subtitle_runtime_contract": "cloud-first-1",
                                                 "subtitle_resolved": frozen(CLOUD_ASR, CLOUD_MT),
                                                 "subtitle_configs": [{}]}, source_path=str(self.source))
        self.tick()
        self.assertEqual(self.executors, [])

    def test_probing_then_ready_starts_without_a_restart_of_anything_else(self):
        sid = self.make_stream("probing")
        self.tick()
        self.assertEqual(self.executors, [])
        item = self.svc.store.get(sid)
        self.svc.store.update(sid, stream={**item.stream, "audio_status": "ready"})
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))

    def test_failed_fails_the_subtitles_with_the_audio_error_once(self):
        sid = self.make_stream("failed", error="字幕音訊網址已失效")
        self.tick()
        self.assertEqual(self.executors, [])
        self.assertEqual(self.subtitle(sid)["status"], "failed")
        self.assertEqual(self.subtitle(sid)["error"], "字幕音訊網址已失效")
        with patch.object(self.svc, "_subtitle_update", wraps=self.svc._subtitle_update) as update:
            self.tick()
            self.tick()
        update.assert_not_called()                                             # written once, not on every tick
        self.assertEqual(self.executors, [])

    def test_failed_without_an_error_text_still_gets_a_reason(self):
        sid = self.make_stream("failed", error="")
        self.tick()
        self.assertEqual(self.subtitle(sid)["status"], "failed")
        self.assertTrue(self.subtitle(sid)["error"])

    def test_a_local_main_model_does_not_start_the_cloud_stream_executor(self):
        """P4 brings the local half to stream items; until then the item waits (the view shows waiting_gpu)."""
        sid = self.make_stream("ready", plan=frozen(LOCAL_ASR, LOCAL_MT))
        self.tick()
        self.assertEqual(self.executors, [])
        self.assertIsNone(self.svc._subtitle_stream_cloud)
        self.assertEqual(self.subtitle(sid)["status"], "queued")

    def test_a_stream_without_subtitles_never_starts(self):
        self.make_stream("ready", subtitles=False)
        self.tick()
        self.assertEqual(self.executors, [])

    def test_failed_audio_of_a_stream_without_subtitles_is_not_reported(self):
        sid = self.make_stream("failed", error="x", subtitles=False)
        self.tick()
        self.assertNotEqual(self.subtitle(sid).get("status"), "failed")

    def test_player_busy_does_not_stop_or_prevent_it(self):
        sid = self.make_stream("ready")
        with self.svc._lock:
            self.svc._busy = True
            self.svc._busy_ts = self.svc._clock()
        self.assertTrue(self.svc.is_player_busy())
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1 and self.executors[0].started.is_set()))
        self.tick()                                                            # running: busy never closes the gate
        self.assertFalse(self.executors[0].stopped.is_set())
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, sid)

    def test_pause_all_does_not_block_or_stop_the_stream_subtitles(self):
        self.make_stream("ready")
        self.svc.pause_all()
        self.assertTrue(self.svc._paused_all)
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1 and self.executors[0].started.is_set()))
        self.svc.pause_all()                                                   # pausing again while it runs
        self.tick()
        self.assertFalse(self.executors[0].stopped.is_set())
        self.assertIsNotNone(self.svc._subtitle_stream_cloud)
        self.assertFalse(self.svc._subtitle_stream_cloud.cancel.is_set())

    def test_pause_all_still_blocks_a_queue_item(self):
        self.make_queue_item()
        self.svc.pause_all()
        self.tick()
        self.assertEqual(self.executors, [])

    def test_pause_all_stops_a_running_queue_executor_but_not_the_stream_one(self):
        queue_id = self.make_queue_item()
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1 and self.executors[0].started.is_set()))
        sid = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2 and self.executors[1].started.is_set()))
        queue_ex, stream_ex = self.executors
        self.svc.pause_all()
        self.assertTrue(self.wait_for(queue_ex.stopped.is_set))
        self.assertFalse(stream_ex.stopped.is_set())
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, sid)
        self.assertEqual(queue_ex.host.item_id, queue_id)

    def test_stopping_the_service_still_stops_the_stream_executor(self):
        self.start_stream_executor()
        self.svc._stopping.set()
        self.tick()
        self.assertTrue(self.wait_for(self.executors[0].stopped.is_set))

    def test_a_tombstoned_stream_item_does_not_start(self):
        sid = self.make_stream("ready")
        self.svc._tombstones.add(sid)
        self.tick()
        self.assertEqual(self.executors, [])


class SlotTests(Base):
    def test_a_queue_item_and_a_stream_item_each_hold_a_slot_and_run_together(self):
        queue_id = self.make_queue_item()
        sid = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2 and all(e.started.is_set() for e in self.executors)))
        self.assertEqual(self.svc._subtitle_cloud.item_id, queue_id)
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, sid)
        self.assertEqual({r.item_id for r in self.svc._subtitle_runs()}, {queue_id, sid})
        self.assertFalse(any(e.stopped.is_set() for e in self.executors))

    def test_the_regular_tick_never_takes_a_stream_item_and_the_stream_slot_never_a_queue_item(self):
        queue_id = self.make_queue_item()
        self.svc._subtitle_stream_tick()
        self.assertEqual(self.executors, [])                                   # no stream item: nothing in that slot
        sid = self.make_stream("ready")
        self.svc._subtitle_start_cloud(self.svc.store.get(queue_id))
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))
        self.svc._subtitle_executor_tick()                                     # regular slot taken by the queue item
        self.assertEqual(self.svc._subtitle_cloud.item_id, queue_id)
        self.assertEqual(len(self.executors), 2)                               # the stream got its own via the tick
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, sid)

    def test_a_busy_regular_slot_does_not_hold_the_stream_back(self):
        self.make_queue_item()
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))
        sid = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2))
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, sid)

    def test_a_running_stream_executor_does_not_hold_the_queue_back(self):
        self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))
        queue_id = self.make_queue_item()
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2))
        self.assertEqual(self.svc._subtitle_cloud.item_id, queue_id)

    def test_the_stream_slot_is_listed_by_subtitle_runs(self):
        _sid, _ex = self.start_stream_executor()
        self.assertIn(self.svc._subtitle_stream_cloud, self.svc._subtitle_runs())

    def test_the_slot_is_freed_when_the_executor_ends_and_the_result_is_recorded(self):
        self.outcome = ss.RunResult("complete")
        sid = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: self.svc._subtitle_stream_cloud is None and self.executors))
        finished = self.svc._subtitle_results_for(sid)["cloud"]
        self.assertEqual(finished.reason, "complete")
        self.tick()                                                            # complete: never restarted
        self.assertEqual(len(self.executors), 1)

    def test_only_one_stream_executor_at_a_time(self):
        first = self.make_stream("ready")
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))
        self.make_stream("ready")
        self.tick()
        self.tick()
        self.assertEqual(len(self.executors), 1)
        self.assertEqual(self.svc._subtitle_stream_cloud.item_id, first)

    def test_a_failed_start_frees_the_stream_slot_and_fails_the_item(self):
        sid = self.make_stream("ready")
        with patch.object(self.svc, "_subtitle_new_cloud_executor", side_effect=RuntimeError("boom")):
            self.tick()
        self.assertIsNone(self.svc._subtitle_stream_cloud)
        self.assertEqual(self.subtitle(sid)["status"], "failed")

    def test_the_executor_is_stopped_when_the_gate_closes(self):
        _sid, ex = self.start_stream_executor()
        self.svc.store.set_settings(subtitle_cloud_enabled=False)              # direct store write: no set_settings hook
        self.tick()
        self.assertTrue(self.wait_for(ex.stopped.is_set))
        self.assertTrue(self.wait_for(lambda: self.svc._subtitle_stream_cloud is None))


class StopTests(Base):
    def test_withdrawing_cloud_consent_stops_the_stream_slot(self):
        sid, ex = self.start_stream_executor()
        queue_ex_count = len(self.executors)
        self.svc.set_settings({"subtitle_cloud_enabled": False})
        self.assertTrue(ex.stopped.is_set())
        self.assertTrue(self.svc._subtitle_stream_cloud is None or self.svc._subtitle_stream_cloud.cancel.is_set())
        self.assertEqual(queue_ex_count, 1)

    def test_withdrawing_consent_stops_both_slots(self):
        self.make_queue_item()
        self.tick()
        sid, stream_ex = self.start_stream_executor()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2))
        self.svc.set_settings({"subtitle_cloud_enabled": False})
        self.assertTrue(all(e.stopped.is_set() for e in self.executors))

    def test_removing_the_stream_item_stops_the_stream_slot(self):
        sid, ex = self.start_stream_executor()
        run = self.svc._subtitle_stream_cloud
        self.assertTrue(self.svc._drop_stream_item(sid))
        self.assertTrue(ex.stopped.is_set())
        self.assertTrue(run.cancel.is_set())
        self.assertEqual(run.reason, "remove")
        self.assertTrue(self.wait_for(lambda: not run.thread.is_alive()))
        self.assertIsNone(self.svc.store.get(sid))

    def test_stream_end_message_stops_the_stream_slot(self):
        sid, ex = self.start_stream_executor()
        self.assertEqual(self.svc._on_message({"cmd": "stream_end", "id": sid}), {"ok": True})
        self.assertTrue(ex.stopped.is_set())
        self.assertIsNone(self.svc.store.get(sid))

    def test_removing_a_stream_item_leaves_the_queue_executor_alone(self):
        self.make_queue_item()
        self.tick()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 1))
        queue_ex = self.executors[0]
        sid, _ = self.start_stream_executor()
        self.assertTrue(self.wait_for(lambda: len(self.executors) == 2))
        self.svc._drop_stream_item(sid)
        self.assertFalse(queue_ex.stopped.is_set())

    def test_service_stop_stops_the_stream_executor(self):
        sid, ex = self.start_stream_executor()
        self.svc.stop()
        self.assertTrue(ex.stopped.is_set())


class ExecutorBuildTests(Base):
    """The REAL `_subtitle_new_cloud_executor` (the seam patched out in the other classes)."""

    def build(self, item_id):
        item = self.svc.store.get(item_id)
        host = sh.QueueSubtitleHost(self.svc, item_id, executor="cloud")
        return QS.QueueService._subtitle_new_cloud_executor(self.svc, item, host)

    def test_the_stream_executor_leads_by_10_minutes_with_the_smart_horizon_strategy(self):
        sid = self.make_stream("ready", subtitle_cloud_strategy="full_preload")        # D4: the setting is overridden
        ex = self.build(sid)
        self.assertEqual(ex.horizon_minutes, 10)
        self.assertEqual(ex.strategy, "smart_horizon")
        self.assertEqual(ex.horizon_minutes * 60.0, sh.STREAM_HORIZON_S)             # one source for both numbers
        self.assertEqual(sh.STREAM_HORIZON_MINUTES, 10)

    def test_the_stream_executor_runs_the_main_track_only(self):
        rows = [{"id": "second", "asr": {"mode": "single", "model": "whisper-large-v3"},
                 "translation": {"profile": "sakura-v1"}}]
        sid = self.make_stream("ready", subtitle_configs=[{}, *rows], subtitle_comparison_enabled=True)
        ex = self.build(sid)
        self.assertEqual(ex._track_plans(), {})
        self.assertEqual(ex._ready_tracks(), ["primary"])

    def test_a_queue_item_keeps_its_comparison_tracks_and_default_horizon(self):
        queue_id = self.make_queue_item()
        item = self.svc.store.get(queue_id)
        with patch.object(ss, "CloudSliceExecutor") as cls:
            QS.QueueService._subtitle_new_cloud_executor(
                self.svc, item, sh.QueueSubtitleHost(self.svc, queue_id, executor="cloud"))
        kwargs = cls.call_args.kwargs
        self.assertNotIn("horizon_minutes", kwargs)
        self.assertNotIn("strategy", kwargs)

    def test_the_stream_context_forces_the_horizon_strategy(self):
        sid = self.make_stream("ready", subtitle_cloud_strategy="full_preload")
        item = self.svc.store.get(sid)
        ctx = sh.QueueSubtitleHost(self.svc, sid).context_for(item, None)
        self.assertEqual(ctx.strategy, "smart_horizon")
        queue_id = self.make_queue_item()
        self.svc.store.update(queue_id, settings={**self.svc.store.get(queue_id).settings,
                                                  "subtitle_cloud_strategy": "full_preload"})
        qctx = sh.QueueSubtitleHost(self.svc, queue_id).context_for(self.svc.store.get(queue_id), None)
        self.assertEqual(qctx.strategy, "full_preload")


class PublishTests(Base):
    RESULT = {"complete": True, "covered_s": 120.0, "srt": ["a.srt"], "cues": [1], "done_indices": [0]}

    def test_a_finished_stream_item_is_ready_not_mux_pending(self):
        sid = self.make_stream("ready")
        host = sh.QueueSubtitleHost(self.svc, sid, executor="cloud")
        host.publish(dict(self.RESULT))
        sub = self.subtitle(sid)
        self.assertEqual(sub["status"], "ready")
        self.assertNotIn("only", sub)
        self.assertEqual(sub["percent"], 100)
        updates = [m for m in self.players if m.get("cmd") == "subtitle_update"]
        self.assertEqual([m["item_id"] for m in updates], [sid])

    def test_a_partial_publish_is_partial(self):
        sid = self.make_stream("ready")
        host = sh.QueueSubtitleHost(self.svc, sid, executor="cloud")
        host.publish({**self.RESULT, "complete": False})
        self.assertEqual(self.subtitle(sid)["status"], "partial")

    def test_maybe_start_subtitle_only_does_not_touch_a_ready_stream_item(self):
        sid = self.make_stream("ready")
        sh.QueueSubtitleHost(self.svc, sid, executor="cloud").publish(dict(self.RESULT))
        self.svc._paused_all = False
        with patch("threading.Thread.start") as start:
            self.svc._maybe_start_subtitle_only()
        start.assert_not_called()
        self.assertIsNone(self.svc._wk)
        self.assertEqual(self.subtitle(sid)["status"], "ready")

    def test_a_queue_item_that_is_done_still_goes_to_mux_pending(self):
        queue_id = self.make_queue_item()
        self.svc.store.update(queue_id, state="done")
        sh.QueueSubtitleHost(self.svc, queue_id, executor="cloud").publish(dict(self.RESULT))
        self.assertEqual(self.subtitle(queue_id)["status"], "mux_pending")


if __name__ == "__main__":
    unittest.main()
