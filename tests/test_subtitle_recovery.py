"""The subtitle user operations (retry / restart / thaw) on the unified scheduler.

Also: speech-recognition results survive a changed translation/backup setting.
No network, credentials or real audio: providers are fakes.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oh_streamer.queue import subtitle_recovery as recovery
from oh_streamer.subtitles.providers.http import ProviderError
from tests import test_queue_service as queue_test_support
from tests.test_subtitle_cloud_first_runtime import settings as cloud_settings


def recovery_combo_presets():
    from oh_streamer.subtitles import combos
    return combos.load_presets()

class ServiceBase(queue_test_support.ServiceTestBase):
    def setUp(self):
        class FakeInstance:
            def serve(self, _handler, log=None):
                pass

            def close(self):
                pass

        with patch.object(queue_test_support.QS.single_instance, "acquire", return_value=FakeInstance()):
            super().setUp()
        self.svc.pause_all()
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.svc._stopping.clear()
        self.svc._paused_all = False

    def item(self, **subtitle):
        self.count = getattr(self, "count", 0) + 1
        item = self.svc.store.add(f"http://h/rec{self.count}", "url", None, None, dict(cloud_settings(), fake_mode="ok"), title="rec")
        self.svc.store.update(item.id, state="queued", subtitle={"status": "asr", "percent": 10, **subtitle})
        return item.id

    def sub(self, item_id):
        return self.svc.store.get(item_id).subtitle


class ServiceTests(ServiceBase):
    # -- M7: no item-level "blocked"; the slice table keeps the waits, these tests cover the user operations --

    def plan_with(self, item_id, *, failed=(), done=(), closed=True):
        """A slice table on disk: 3 slices; `failed` / `done` are slice indexes of the primary ASR (done: both stages)."""
        from oh_streamer.subtitles import slice_plan as sp
        plan = sp.build_slice_plan(900, first_slice_seconds=None)
        plan.lazy["closed"] = closed
        for index in failed:
            plan.mark_failed(index, "asr", "exhausted")
        for index in done:
            plan.mark_done(index, "asr")
            plan.mark_done(index, "mt")
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        sp.save_slice_plan(plan, folder)
        return folder

    def spy_executors(self):
        calls = []
        self.svc._subtitle_stop_executors = lambda item_id, reason="pause", wait=0.0: calls.append(("stop", item_id, reason)) or True
        self.svc._subtitle_executor_reset = lambda item_id, executor=None: calls.append(("reset", item_id))
        return calls

    def test_a_rate_limit_is_recorded_as_a_plain_failure(self):
        item_id = self.item()
        self.svc._subtitle_fail(item_id, ProviderError("rate_limit", "Rate limit exceeded", retryable=True, http_status=429))
        sub = self.sub(item_id)
        self.assertEqual(sub["status"], "failed")
        self.assertIn("429", sub["error"])
        self.assertFalse({"blocked_reason", "blocked_kind", "retry_at", "retry_count"} & set(sub))

    def test_final_error_fails(self):
        item_id = self.item()
        self.svc._subtitle_fail(item_id, ProviderError("content_blocked", "no"))
        self.assertEqual(self.sub(item_id)["status"], "failed")
        item_id = self.item()
        self.svc._subtitle_fail(item_id, RuntimeError("boom"), percent=0)
        self.assertEqual(self.sub(item_id)["status"], "failed")

    def test_retry_clears_failed_units_adopts_settings_and_wakes_the_executor(self):
        from oh_streamer.subtitles import slice_plan as sp
        item_id = self.item(covered_s=300)
        folder = self.plan_with(item_id, failed=(1,), done=(0,))
        self.svc._subtitle_update(item_id, status="failed", error="exhausted")
        calls = self.spy_executors()
        same = dict(self.svc.store.get(item_id).settings, marker=1)          # a "how"-only change: content unchanged
        with patch.object(recovery, "refresh_snapshot_settings", return_value=same) as refresh:
            self.assertTrue(self.svc.retry_subtitles(item_id)["ok"])
        refresh.assert_called_once()
        self.assertEqual(self.svc.store.get(item_id).settings.get("marker"), 1)
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["error"]), ("partial", ""))
        plan = sp.load_slice_plan(folder)
        self.assertFalse(any(t.status == sp.FAILED for s in plan.slices for t in s.tasks.values()))
        self.assertTrue(plan.slices[0].all_done())
        self.assertEqual(calls, [("reset", item_id)])
        self.assertFalse(self.svc.retry_subtitles(item_id)["ok"], "nothing left to retry")

    def test_retry_also_works_on_stuck_work_while_the_item_looks_healthy(self):
        from oh_streamer.subtitles import slice_plan as sp
        item_id = self.item()
        folder = self.plan_with(item_id, failed=(2,))
        self.svc._subtitle_update(item_id, status="partial")
        calls = self.spy_executors()
        with patch.object(recovery, "refresh_snapshot_settings", return_value={"subtitles": True}):
            self.assertTrue(self.svc.retry_subtitles(item_id)["ok"])
        self.assertEqual(self.sub(item_id)["status"], "partial")
        self.assertEqual(sp.load_slice_plan(folder).slices[2].get("asr").status, sp.PENDING)
        self.assertEqual(calls, [("reset", item_id)])

    def test_retry_of_a_finished_table_whose_mux_failed_goes_back_to_ready(self):
        item_id = self.item()
        self.plan_with(item_id, done=(0, 1, 2))
        self.svc._subtitle_update(item_id, status="failed", error="mux failed")
        self.spy_executors()
        self.assertTrue(self.svc.retry_subtitles(item_id)["ok"])
        self.assertEqual(self.sub(item_id)["status"], "ready")

    def test_retry_drops_legacy_pipeline_fields(self):
        item_id = self.item(block_mode=True, blocks_done=2, cloud_mode=True, fallback={"state": "exhausted"},
                            blocked_reason="x", retry_count=3, combos={"selected": []}, covered_s=120)
        self.svc._subtitle_update(item_id, status="failed")
        self.spy_executors()
        self.assertTrue(self.svc.retry_subtitles(item_id)["ok"])
        sub = self.sub(item_id)
        self.assertFalse(set(recovery.LEGACY_SUBTITLE_FIELDS) & set(sub))
        self.assertEqual(sub["covered_s"], 120)

    def test_retry_with_nothing_wrong_is_refused(self):
        item_id = self.item()
        self.plan_with(item_id, done=(0,))
        self.assertFalse(self.svc.retry_subtitles(item_id)["ok"])

    def test_thaw_requeues_a_failed_item_and_keeps_finished_slices_visible_as_partial(self):
        item_id = self.item(covered_s=300, block_srt=["a.srt"])
        self.svc._subtitle_fail(item_id, ProviderError("quota_daily", "x"))
        calls = self.spy_executors()
        self.assertTrue(self.svc._subtitle_thaw(item_id))
        self.assertEqual(self.sub(item_id)["status"], "partial")
        self.assertEqual(calls, [("reset", item_id)])

    def test_thaw_with_refresh_adopts_the_current_settings_of_a_failed_item(self):
        item_id = self.item()
        self.svc._subtitle_fail(item_id, ProviderError("auth", "x"))
        self.spy_executors()
        with patch.object(recovery, "refresh_snapshot_settings", return_value={"subtitles": True, "marker": 1}) as refresh:
            self.assertTrue(self.svc._subtitle_thaw(item_id, refresh=True, reset=True))
        refresh.assert_called_once()
        self.assertEqual(self.svc.store.get(item_id).settings.get("marker"), 1)
        self.assertEqual(self.sub(item_id)["status"], "queued")

    def test_thaw_does_not_touch_the_settings_of_a_healthy_running_item(self):
        item_id = self.item()
        self.spy_executors()
        with patch.object(recovery, "refresh_snapshot_settings") as refresh:
            self.assertTrue(self.svc._subtitle_thaw(item_id, refresh=True, reset=True))
        refresh.assert_not_called()
        self.assertEqual(self.sub(item_id)["status"], "asr", "an executor-owned status stays as it is")

    def test_a_failed_refresh_keeps_the_old_snapshot(self):
        item_id = self.item()
        self.svc._subtitle_fail(item_id, ProviderError("auth", "x"))
        self.spy_executors()
        with patch.object(recovery, "refresh_snapshot_settings", side_effect=ValueError("bad")):
            self.assertTrue(self.svc._subtitle_thaw(item_id, refresh=True))
        self.assertEqual(self.sub(item_id)["status"], "queued")
        self.assertTrue(self.svc.store.get(item_id).settings["subtitles"])

    def test_thaw_all_wakes_only_failed_or_stuck_items(self):
        failed, stuck, healthy = self.item(), self.item(), self.item()
        self.svc._subtitle_fail(failed, RuntimeError("boom"))
        self.plan_with(stuck, failed=(0,))
        self.spy_executors()
        woken = []
        with patch.object(type(self.svc), "_subtitle_thaw", lambda svc, item_id, **kw: woken.append(item_id) or True):
            self.svc._subtitle_thaw_all(refresh=True)
        self.assertEqual(sorted(woken), sorted([failed, stuck]))
        self.assertNotIn(healthy, woken)

    def test_restart_drops_every_cached_result_and_every_published_field(self):
        item_id = self.item(block_srt=["a.srt"], covered_s=200, cloud_mode=True, retry_count=2, audio_offset=1.5)
        folder = self.svc.store.item_dir(item_id) / "subtitles" / "cache" / "asr"
        folder.mkdir(parents=True)
        (folder / "k.json").write_text("{}", encoding="utf-8")
        self.svc._subtitle_fail(item_id, ProviderError("quota_daily", "x"))
        calls = self.spy_executors()
        self.assertTrue(self.svc.restart_subtitles(item_id)["ok"])
        self.assertFalse((self.svc.store.item_dir(item_id) / "subtitles").exists())
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["percent"], sub["error"]), ("queued", 0, ""))
        self.assertFalse((set(recovery.LEGACY_SUBTITLE_FIELDS) | set(recovery.PUBLISH_FIELDS)) & set(sub))
        self.assertEqual(calls, [("stop", item_id, "cancel"), ("reset", item_id)])

    def test_restart_refuses_while_the_mux_step_runs_or_an_executor_does_not_stop(self):
        item_id = self.item()
        self.svc._subtitle_update(item_id, status="mux")
        self.assertFalse(self.svc.restart_subtitles(item_id)["ok"])
        self.svc._subtitle_update(item_id, status="partial")
        self.svc._subtitle_stop_executors = lambda *args, **kwargs: False
        self.assertFalse(self.svc.restart_subtitles(item_id)["ok"])
        self.assertEqual(self.sub(item_id)["status"], "partial", "nothing was reset")

    def test_restart_keeps_a_done_item_marked_subtitle_only(self):
        item_id = self.item()
        self.svc.store.update(item_id, state="done")
        self.spy_executors()
        self.assertTrue(self.svc.restart_subtitles(item_id)["ok"])
        self.assertTrue(self.sub(item_id)["only"])
        self.assertIn(item_id, self.svc.store.protected_ids)

    def test_cancel_stops_the_executors_and_forgets_how_they_ended(self):
        item_id = self.item()
        self.svc.store.update(item_id, state="done")
        self.svc._subtitle_update(item_id, status="partial", only=True)
        calls = self.spy_executors()
        self.assertTrue(self.svc.cancel_subtitles(item_id)["ok"])
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["only"]), ("cancelled", False))
        self.assertEqual(calls, [("stop", item_id, "cancel"), ("reset", item_id)])

    def test_rerun_resets_the_slice_table_but_keeps_the_cache(self):
        from oh_streamer.subtitles import slice_plan as sp
        item_id = self.item(block_srt=["a.srt"], covered_s=100)
        self.svc.store.update(item_id, state="done")
        folder = self.plan_with(item_id, done=(0, 1, 2))
        (folder / "cache" / "asr").mkdir(parents=True)
        (folder / "cache" / "asr" / "k.json").write_text("{}", encoding="utf-8")
        self.svc._subtitle_update(item_id, status="done", only=False)
        calls = self.spy_executors()
        with patch.object(self.svc, "_subtitle_environment", return_value=dict(available=True, reason="", python="", models="", server="")):
            self.assertTrue(self.svc.rerun_subtitles(item_id)["ok"])
        self.assertFalse((folder / sp.SLICE_PLAN_FILE).exists())
        self.assertTrue((folder / "cache" / "asr" / "k.json").is_file())
        sub = self.sub(item_id)
        self.assertEqual((sub["status"], sub["only"]), ("queued", True))
        self.assertNotIn("block_srt", sub)
        self.assertEqual(calls, [("reset", item_id)])

    def test_changing_a_job_s_rows_stops_the_executor_and_starts_over_with_a_fresh_table(self):
        from oh_streamer.subtitles import slice_plan as sp
        item_id = self.item(block_srt=["a.srt"], covered_s=100, combos={"selected": []})
        folder = self.plan_with(item_id, done=(0,))
        self.svc._subtitle_update(item_id, status="partial")
        calls = self.spy_executors()
        presets, _ = recovery_combo_presets()
        ids = list(presets)[:2]
        with patch.object(self.svc, "_subtitle_environment", return_value=dict(available=True, reason="", python="", models="", server="")):
            self.assertTrue(self.svc.set_subtitle_combos(item_id, ids)["ok"])
        self.assertFalse((folder / sp.SLICE_PLAN_FILE).exists())
        sub = self.sub(item_id)
        self.assertEqual(sub["status"], "queued")
        self.assertFalse((set(recovery.LEGACY_SUBTITLE_FIELDS) | set(recovery.PUBLISH_FIELDS)) & set(sub))
        self.assertIn(("stop", item_id, "pause"), calls)
        self.assertEqual(calls[-1], ("reset", item_id))




if __name__ == "__main__":
    unittest.main()
