"""Operations on a settings_pending prompt: rest / keep / redo_mt (+ the *_all variants)."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oh_streamer.queue import subtitle_recovery as recovery
from oh_streamer.queue.subtitle_jobs import content_signature
from oh_streamer.subtitles import slice_plan as sp
from test_subtitle_settings_lock import LockBase, variant


class ApplyTests(LockBase):
    def pending_item(self, *, asr=False, **subtitle):
        item_id = self.started_item(**subtitle)
        old = copy.deepcopy(self.settings(item_id))
        self.fresh = variant(old, **({"asr": "other-asr"} if asr else {"mt": "other-mt"}))
        self.lock(item_id, self.fresh)
        self.assertEqual(self.pending(item_id), {"asr_changed": asr})
        return item_id

    def apply(self, item_id, choice):
        with patch.object(recovery, "refresh_snapshot_settings", return_value=self.fresh):
            return self.svc.apply_subtitle_settings(item_id, choice)

    def plan(self, item_id):
        return sp.load_slice_plan(self.svc.store.item_dir(item_id) / "subtitles")

    def test_rest_adopts_fresh_keeps_done_units_and_clears_hold_and_pending(self):
        item_id = self.pending_item()
        self.svc.store.update(item_id, settings={**self.settings(item_id), "subtitle_hold": True,
                                                 "subtitle_hold_sig": "x"})
        self.assertEqual(self.apply(item_id, "rest"), {"ok": True})
        settings = self.settings(item_id)
        self.assertEqual(settings["subtitle_resolved"]["models"]["translation"]["id"], "other-mt")
        self.assertNotIn("subtitle_hold", settings)
        self.assertNotIn("subtitle_hold_sig", settings)
        self.assertIsNone(self.pending(item_id))
        plan = self.plan(item_id)
        self.assertTrue(plan.slices[0].is_done(sp.STAGE_ASR, plan.primary))
        self.assertTrue(plan.slices[0].is_done(sp.STAGE_MT, plan.primary))

    def test_keep_sets_hold_with_signature_and_clears_pending(self):
        item_id = self.pending_item()
        old = copy.deepcopy(self.settings(item_id))
        self.assertEqual(self.apply(item_id, "keep"), {"ok": True})
        settings = self.settings(item_id)
        self.assertTrue(settings["subtitle_hold"])
        self.assertEqual(settings["subtitle_hold_sig"], content_signature(self.fresh))
        self.assertEqual(settings["subtitle_resolved"], old["subtitle_resolved"])
        self.assertIsNone(self.pending(item_id))

    def test_redo_mt_resets_only_translation_and_adopts_fresh(self):
        item_id = self.pending_item(covered_s=100.0, block_srt="x.srt")
        self.assertEqual(self.apply(item_id, "redo_mt"), {"ok": True})
        plan = self.plan(item_id)
        for s in plan.slices:
            for track in plan.tracks:
                self.assertFalse(s.is_done(sp.STAGE_MT, track))
        self.assertTrue(plan.slices[0].is_done(sp.STAGE_ASR, plan.primary))
        sub = self.sub(item_id)
        self.assertNotIn("covered_s", sub)
        self.assertNotIn("block_srt", sub)
        self.assertIsNone(sub.get("settings_pending"))
        self.assertEqual(self.settings(item_id)["subtitle_resolved"]["models"]["translation"]["id"], "other-mt")

    def test_redo_mt_refuses_when_asr_changed(self):
        item_id = self.pending_item(asr=True)
        res = self.apply(item_id, "redo_mt")
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"])
        self.assertEqual(self.pending(item_id), {"asr_changed": True})
        self.assertTrue(self.plan(item_id).slices[0].is_done(sp.STAGE_MT, "primary"))

    def test_errors_without_pending_or_with_bad_choice(self):
        item_id = self.started_item()
        self.assertEqual(self.svc.apply_subtitle_settings(item_id, "rest"), {"ok": False, "error": "此項目沒有待套用的設定變更"})
        self.assertFalse(self.svc.apply_subtitle_settings(item_id, "bogus")["ok"])

    def test_apply_all_counts_pending_items_only(self):
        a, b = self.pending_item(), self.pending_item()
        c = self.started_item()
        with patch.object(recovery, "refresh_snapshot_settings", return_value=self.fresh):
            self.assertEqual(self.svc.apply_subtitle_settings_all("keep"), {"ok": True, "count": 2})
            self.assertEqual(self.svc.apply_subtitle_settings_all("rest"), {"ok": True, "count": 0})
        self.assertTrue(self.settings(a)["subtitle_hold"] and self.settings(b)["subtitle_hold"])
        self.assertNotIn("subtitle_hold", self.settings(c))
        self.assertFalse(self.svc.apply_subtitle_settings_all("redo_mt")["ok"])


class DrainTests(ApplyTests):
    """A running executor drains softly on a settings change; redo_mt halts it first."""

    def running(self, item_id):
        from oh_streamer.queue.queue_service import _Run
        from unittest import mock
        run = _Run(item_id)
        run.executor = mock.Mock()
        run.thread = None
        self.svc._subtitle_cloud = run
        self.addCleanup(setattr, self.svc, "_subtitle_cloud", None)
        return run

    def test_rest_asks_the_running_executor_to_drain_without_halting(self):
        item_id = self.pending_item()
        run = self.running(item_id)
        calls = []
        self.svc._subtitle_stop_executors = lambda *a, **k: calls.append(a) or True
        self.assertEqual(self.apply(item_id, "rest"), {"ok": True})
        run.executor.request_refresh.assert_called_once()
        run.executor.stop.assert_not_called()
        self.assertFalse(run.cancel.is_set())
        self.assertEqual(calls, [])

    def test_keep_does_not_drain(self):
        item_id = self.pending_item()
        run = self.running(item_id)
        self.apply(item_id, "keep")
        run.executor.request_refresh.assert_not_called()

    def test_adopted_content_change_of_unstarted_running_item_drains(self):
        item_id = self.item()
        run = self.running(item_id)
        self.lock(item_id, variant(self.settings(item_id), mt="other-mt"))
        run.executor.request_refresh.assert_called_once()

    def test_how_change_adopted_by_gate_drains_but_strategy_only_does_not(self):
        item_id = self.started_item()
        run = self.running(item_id)
        self.lock(item_id, variant(self.settings(item_id), strategy="other_strategy"))
        run.executor.request_refresh.assert_not_called()          # the strategy is read live
        fresh = variant(self.settings(item_id), strategy="other_strategy")
        fresh["subtitle_runtime_contract"] = "cloud-first-1"
        fresh["subtitle_profile"] = dict(fresh.get("subtitle_profile") or {}, marker="how")
        self.lock(item_id, fresh)
        run.executor.request_refresh.assert_called_once()

    def test_redo_mt_halts_before_resetting(self):
        item_id = self.pending_item()
        run = self.running(item_id)
        order = []
        alive = {"v": True}
        self.svc._subtitle_executor_running = lambda i: alive["v"]

        def stop(item, reason="pause", wait=0.0):
            order.append(("halt", reason, self.svc._lock._is_owned()))
            alive["v"] = False
            return True
        self.svc._subtitle_stop_executors = stop
        real = self.svc._subtitle_reset_mt
        self.svc._subtitle_reset_mt = lambda i: order.append(("reset",)) or real(i)
        self.assertEqual(self.apply(item_id, "redo_mt"), {"ok": True})
        self.assertEqual(order, [("halt", "cancel", False), ("reset",)])
        self.assertNotEqual(self.sub(item_id).get("status"), "failed")

    def test_redo_mt_halt_failure_returns_error_and_changes_nothing(self):
        item_id = self.pending_item()
        self.running(item_id)
        self.svc._subtitle_executor_running = lambda i: True
        self.svc._subtitle_stop_executors = lambda *a, **k: False
        old = copy.deepcopy(self.settings(item_id))
        res = self.apply(item_id, "redo_mt")
        self.assertEqual(res, {"ok": False, "error": "字幕工作尚未停止，請稍後再試"})
        self.assertEqual(self.settings(item_id), old)
        self.assertTrue(self.plan(item_id).slices[0].is_done(sp.STAGE_MT, "primary"))
        self.assertEqual(self.pending(item_id), {"asr_changed": False})

    def test_refresh_exit_is_not_a_finished_run_nor_a_status_change(self):
        from oh_streamer.queue import subtitle_scheduler as ss
        from oh_streamer.queue.subtitle_host import CLOUD, FinishedRun
        from unittest import mock
        item_id = self.started_item(status="asr")
        run = self.running(item_id)
        run.executor.run.return_value = ss.RunResult(reason="refresh")
        self.svc._subtitle_results_for(item_id)[CLOUD] = FinishedRun(None, None, 0.0)
        self.svc._run_subtitle_executor(item_id, run, CLOUD)
        self.assertNotIn(CLOUD, self.svc._subtitle_results_for(item_id))
        self.assertTrue(self.svc._subtitle_gate_open(item_id, CLOUD, self.svc._subtitle_time()))
        self.assertEqual(self.sub(item_id)["status"], "asr")
        self.assertTrue(self.svc._wake.is_set())
        self.assertIsNone(self.svc._subtitle_cloud)


if __name__ == "__main__":
    unittest.main()
