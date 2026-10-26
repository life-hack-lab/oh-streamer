"""Subtitle settings lock: a content change of a started item waits for the user (settings_pending), a "how" change
or an unstarted item adopts the fresh settings."""
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
import test_subtitle_recovery as _rec
ServiceBase = _rec.ServiceBase


def variant(settings, *, mt=None, asr=None, strategy=None):
    out = copy.deepcopy(settings)
    models = out["subtitle_resolved"]["models"]
    if mt:
        models["translation"] = dict(models.get("translation") or {}, id=mt)
    if asr:
        models["asr"] = dict(models.get("asr") or {}, id=asr)
    if strategy:
        out["subtitle_cloud_strategy"] = strategy
    return out


class LockBase(ServiceBase):
    plan_with = _rec.ServiceTests.plan_with

    def started_item(self, **subtitle):
        item_id = self.item(**subtitle)
        self.plan_with(item_id, done=(0,))
        return item_id

    def lock(self, item_id, fresh):
        item = self.svc.store.get(item_id)
        with patch.object(recovery, "refresh_snapshot_settings", return_value=fresh):
            return self.svc._subtitle_lock_settings(item)

    def settings(self, item_id):
        return self.svc.store.get(item_id).settings

    def pending(self, item_id):
        return self.sub(item_id).get("settings_pending")


class LockTests(LockBase):
    def test_unstarted_item_adopts_content_change_without_prompt(self):
        item_id = self.item()
        fresh = variant(self.settings(item_id), mt="other-mt")
        self.assertIsNotNone(self.lock(item_id, fresh))
        self.assertEqual(self.settings(item_id)["subtitle_resolved"]["models"]["translation"]["id"], "other-mt")
        self.assertIsNone(self.pending(item_id))

    def test_started_item_adopts_how_only_change(self):
        item_id = self.started_item()
        fresh = variant(self.settings(item_id), strategy="other_strategy")
        self.assertEqual(content_signature(fresh), content_signature(self.settings(item_id)))
        self.lock(item_id, fresh)
        self.assertEqual(self.settings(item_id)["subtitle_cloud_strategy"], "other_strategy")
        self.assertIsNone(self.pending(item_id))

    def test_started_item_content_change_keeps_locked_and_sets_pending(self):
        item_id = self.started_item()
        old = copy.deepcopy(self.settings(item_id))
        result = self.lock(item_id, variant(old, mt="other-mt"))
        self.assertIsNotNone(result)
        self.assertEqual(self.settings(item_id), old)
        self.assertEqual(self.pending(item_id), {"asr_changed": False})
        self.lock(item_id, variant(old, asr="other-asr"))
        self.assertEqual(self.pending(item_id), {"asr_changed": True})

    def test_pending_clears_when_signatures_match_again(self):
        item_id = self.started_item()
        old = copy.deepcopy(self.settings(item_id))
        self.lock(item_id, variant(old, mt="other-mt"))
        self.assertIsNotNone(self.pending(item_id))
        self.lock(item_id, copy.deepcopy(old))
        self.assertIsNone(self.pending(item_id))

    def test_hold_with_same_signature_does_not_prompt_and_applies_strategy(self):
        item_id = self.started_item()
        old = copy.deepcopy(self.settings(item_id))
        fresh = variant(old, mt="other-mt", strategy="other_strategy")
        self.svc.store.update(item_id, settings={**old, "subtitle_hold": True,
                                                 "subtitle_hold_sig": content_signature(fresh)})
        self.lock(item_id, fresh)
        settings = self.settings(item_id)
        self.assertEqual(settings["subtitle_resolved"], old["subtitle_resolved"])
        self.assertEqual(settings["subtitle_cloud_strategy"], "other_strategy")
        self.assertIsNone(self.pending(item_id))

    def test_hold_then_another_change_prompts_again(self):
        item_id = self.started_item()
        old = copy.deepcopy(self.settings(item_id))
        declined = variant(old, mt="other-mt")
        self.svc.store.update(item_id, settings={**old, "subtitle_hold": True,
                                                 "subtitle_hold_sig": content_signature(declined)})
        self.lock(item_id, variant(old, mt="third-mt"))
        self.assertEqual(self.pending(item_id), {"asr_changed": False})

    def test_no_runnable_plan_returns_none(self):
        item_id = self.item()
        self.svc.store.update(item_id, settings={"subtitles": True})
        with patch.object(recovery, "refresh_snapshot_settings", return_value={"subtitles": True}):
            self.assertIsNone(self.svc._subtitle_lock_settings(self.svc.store.get(item_id)))


class GlobalChangeTests(LockBase):
    def setUp(self):
        super().setUp()
        self.svc._subtitle_stop_executors = lambda *a, **k: True
        self.svc._subtitle_executor_reset = lambda *a, **k: None

    def test_thaw_all_sets_pending_for_started_items_only(self):
        started, unstarted = self.started_item(), self.item()
        old = copy.deepcopy(self.settings(started))
        fresh = variant(old, mt="other-mt")
        with patch.object(recovery, "refresh_snapshot_settings", return_value=fresh):
            self.svc._subtitle_thaw_all(refresh=True)
        self.assertEqual(self.pending(started), {"asr_changed": False})
        self.assertEqual(self.settings(started), old)
        self.assertIsNone(self.pending(unstarted))
        self.assertNotIn("subtitle_hold", self.settings(unstarted))

    def test_failed_started_item_keeps_locked_settings_but_is_requeued(self):
        item_id = self.started_item()
        self.svc._subtitle_update(item_id, status="failed", error="x")
        old = copy.deepcopy(self.settings(item_id))
        with patch.object(recovery, "refresh_snapshot_settings", return_value=variant(old, mt="other-mt")):
            self.svc._subtitle_thaw_all(refresh=True)
        self.assertEqual(self.settings(item_id), old)
        self.assertEqual(self.pending(item_id), {"asr_changed": False})
        self.assertNotEqual(self.sub(item_id)["status"], "failed")

    def test_retry_does_not_bypass_the_gate(self):
        item_id = self.item()
        self.plan_with(item_id, failed=(1,), done=(0,))
        self.svc._subtitle_update(item_id, status="failed", error="x")
        old = copy.deepcopy(self.settings(item_id))
        with patch.object(recovery, "refresh_snapshot_settings", return_value=variant(old, mt="other-mt")):
            self.assertTrue(self.svc.retry_subtitles(item_id)["ok"])
        self.assertEqual(self.settings(item_id), old)
        self.assertEqual(self.pending(item_id), {"asr_changed": False})

    def test_restart_adopts_fresh_and_clears_pending_and_hold(self):
        item_id = self.started_item(settings_pending={"asr_changed": False})
        old = copy.deepcopy(self.settings(item_id))
        self.svc.store.update(item_id, settings={**old, "subtitle_hold": True, "subtitle_hold_sig": "x"})
        fresh = variant(old, mt="other-mt")
        with patch.object(recovery, "refresh_snapshot_settings", return_value=dict(fresh, subtitle_hold=True)):
            self.assertTrue(self.svc.restart_subtitles(item_id)["ok"])
        settings = self.settings(item_id)
        self.assertEqual(settings["subtitle_resolved"]["models"]["translation"]["id"], "other-mt")
        self.assertNotIn("subtitle_hold", settings)
        self.assertNotIn("subtitle_hold_sig", settings)
        self.assertIsNone(self.pending(item_id))

    def test_ready_clears_pending(self):
        item_id = self.started_item(settings_pending={"asr_changed": False})
        self.svc._subtitle_update(item_id, status="ready", percent=100)
        self.assertIsNone(self.pending(item_id))


if __name__ == "__main__":
    unittest.main()
