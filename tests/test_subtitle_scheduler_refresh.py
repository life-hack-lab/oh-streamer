"""Soft drain of the cloud executor (settings changed): the in-flight unit finishes with the settings it started
with, nothing new is launched, the run ends with reason "refresh", and the next executor uses the new deps."""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oh_streamer.subtitles import slice_plan as sp
from test_subtitle_scheduler_cloud import Base, MT_ID


class RefreshDrainTests(Base):
    def test_in_flight_unit_finishes_then_exit_refresh_then_new_snapshot(self):
        holder = {}
        real_asr = self.fake.asr

        def asr(piece, s):
            if not holder.get("asked"):
                holder["asked"] = True
                holder["executor"].request_refresh()          # settings changed while this unit is in flight
            return real_asr(piece, s)

        ex = self.executor(deps=self.deps(asr=asr))
        holder["executor"] = ex
        res = ex.run(wait=True)
        self.assertEqual(res.reason, "refresh")
        self.assertEqual(self.fake.asr_calls, [0])             # the in-flight unit completed, no second one launched
        self.assertEqual(self.fake.mt_calls, [])
        plan = sp.load_slice_plan(self.folder)
        self.assertTrue(plan.slices[0].is_done(sp.STAGE_ASR, plan.primary))
        self.assertFalse(plan.slices[0].is_done(sp.STAGE_MT, plan.primary))
        self.assertFalse(any(t.status == sp.FAILED for s in plan.slices for t in s.tasks.values()))

        ex2 = self.executor(deps=self.deps(mt_identity=dict(MT_ID, id="mt-new")))      # the newly locked snapshot
        res2 = ex2.run(wait=False)
        self.assertEqual(res2.reason, "complete")
        self.assertEqual(self.fake.asr_calls.count(0), 1)       # recognition of the finished unit is not redone

    def test_refresh_before_run_launches_nothing(self):
        ex = self.executor()
        ex.request_refresh()
        res = ex.run(wait=True)
        self.assertEqual(res.reason, "refresh")
        self.assertEqual(self.fake.asr_calls, [])


if __name__ == "__main__":
    unittest.main()
