"""sample video A: usage-log diagnostics of the slice schedulers (run reasons, candidate changes, hand-off events) and
the publication dedupe that survives executor instances. No network, no GPU."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import unittest
from pathlib import Path
from unittest import mock

from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import slice_results as sr
from oh_streamer.subtitles import usage_log as ul
from tests import test_subtitle_scheduler_cloud as cloud_tests
from tests import test_subtitle_scheduler_fallback as fb_tests


class UsageMixin:
    def start_usage(self):
        self._usage_tmp = Path(self._tmp.name) / "usage"
        ul._dir_override = self._usage_tmp
        ul.reset()
        self.addCleanup(self._stop_usage)

    @staticmethod
    def _stop_usage():
        ul._dir_override = None
        ul.reset()

    @property
    def item_id(self):
        return self.folder.parent.name

    def lines(self, *evs):
        log = ul.get(self.item_id)
        if log is None:
            return []
        rows = [json.loads(x) for x in log.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [r for r in rows if not evs or r["ev"] in evs]


class RunLinesTests(UsageMixin, fb_tests.Base):
    def setUp(self):
        super().setUp()
        self.start_usage()
        self.plan = fb_tests.frozen(self.GROQ, self.LLM, asr_fallback=[self.GPU], mt_fallback=[self.SAKURA])

    def test_run_start_reason_and_run_end_details_across_restarts(self):
        self.cloud.errors["groq-asr"] = [fb_tests.err("rate_limit", 60.0)]
        res = self.cloud_ex().run()
        self.assertEqual(res.reason, "idle")
        start, end = self.lines("run_start", "run_end")
        self.assertEqual((start["reason"], start.get("gap_s")), ("first_run", None))
        self.assertEqual((end["reason"], end["waiting_executor"], end["executor"]), ("idle", "local", "cloud"))
        self.assertEqual((end["units_launched"], end["units_done"], end["units_failed"]), (1, 0, 1))
        self.assertIn("run_s", end)
        self.clock.t += 7
        self.cloud_ex().run()
        start2 = self.lines("run_start")[1]
        self.assertEqual(start2["reason"], "restart_after_idle")
        self.assertEqual(start2["gap_s"], 7.0)

    def test_run_end_of_a_cancelled_run_and_of_an_exception(self):
        self.chost.cancel_after = 1
        self.assertEqual(self.cloud_ex().run().reason, "cancelled")
        self.assertEqual(self.lines("run_end")[-1]["reason"], "cancelled")
        self.chost.cancel_after = None
        with mock.patch.object(ss.CloudSliceExecutor, "_launch", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                self.cloud_ex().run()
        end = self.lines("run_end")[-1]
        self.assertEqual(end["reason"], "error")
        self.assertIn("ValueError: boom", end["error"])

    def test_handoff_to_local_is_logged_end_to_end(self):
        self.cloud.errors["groq-asr"] = [fb_tests.err("rate_limit", 60.0)]
        self.cloud_ex().run()
        changed, = self.lines("candidate_changed")
        self.assertEqual((changed["slice"], changed["stage"], changed["from"], changed["to"]),
                         (0, "asr", "groq-asr", "gpu-asr"))
        self.assertEqual((changed["reason"], changed["handoff"], changed["wait_executor"]),
                         ("rate_limit", True, "local"))
        self.assertLessEqual(len(changed["detail"]), ul.DETAIL_MAX)

        self.lhost.busy = True                                # the GPU is not available
        self.assertEqual(self.local_ex().run().reason, "gpu_busy")
        self.assertEqual(self.local_ex().run().reason, "gpu_busy")
        waits = self.lines("local_wait_started")
        self.assertEqual(len(waits), 1)                       # once per wait, not per poll or per run
        self.assertEqual((waits[0]["slice"], waits[0]["stage"], waits[0]["reason"]), (0, "asr", "gpu_busy"))
        self.assertEqual(self.lines("local_started"), [])

        self.lhost.busy = False
        self.lhost.limit = 95.0
        self.clock.t += 30
        self.local_ex().run()
        started, = self.lines("local_started")
        done = self.lines("local_done")[0]
        self.assertEqual((started["slice"], started["stage"], started["candidate"]), (0, "asr", "gpu-asr"))
        self.assertEqual(started["waited_s"], 30.0)
        self.assertEqual((done["slice"], done["stage"]), (0, "asr"))
        self.assertIn("elapsed_s", done)
        ends = [r for r in self.lines("run_end") if r["executor"] == "local"]
        self.assertTrue(ends and all("reason" in r for r in ends))
        self.assertEqual(ends[-1]["units_done"] >= 1, True)

    def test_local_failure_of_a_handed_off_unit_is_logged(self):
        self.cloud.errors["groq-asr"] = [fb_tests.err("rate_limit", 60.0)]
        self.cloud_ex().run()
        self.models.asr_errors = [RuntimeError("model exploded at https://example.org/x?key=1")]
        self.lhost.limit = 95.0
        self.local_ex().run()
        started, failed = self.lines("local_started")[0], self.lines("local_failed")[0]
        self.assertEqual((started["slice"], failed["slice"], failed["stage"]), (0, 0, "asr"))
        self.assertNotIn("https://", failed["detail"])
        self.assertIn("elapsed_s", failed)
        self.assertIn(failed["action"], ("retry", "switch", "failed"))


class PublicationDedupeTests(UsageMixin, cloud_tests.Base):
    def setUp(self):
        super().setUp()
        self.start_usage()

    def test_a_new_executor_does_not_rewrite_files_or_relog_slices(self):
        _ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 380.0))
        self.assertEqual(res.reason, "complete")
        slices = self.lines("slice")
        self.assertTrue(slices)
        srt = self.folder / "partial.zh-TW.srt"
        stamp = srt.stat().st_mtime_ns
        host2 = cloud_tests.FakeHost()
        with mock.patch.object(ss.sr, "publish", wraps=sr.publish) as publish:
            for _ in range(3):
                _ex2, res2 = self.run_ex(host=host2, deps=self.deps(probe_duration=lambda s: 380.0))
                self.assertEqual(res2.reason, "complete")
        self.assertEqual(publish.call_count, 0)
        self.assertEqual(srt.stat().st_mtime_ns, stamp)
        self.assertEqual(self.lines("slice"), slices)
        # a fresh host still gets its first publication (status / ready / mux_pending come from the callback)
        self.assertGreaterEqual(len(host2.published), 3)
        self.assertEqual(host2.published[0]["plan_key"], res.publication["plan_key"])
        self.assertTrue(host2.published[0]["complete"])
        # every run writes its reason
        ends = self.lines("run_end")
        self.assertEqual([r["reason"] for r in ends], ["complete"] * 4)

    def test_missing_files_are_written_again(self):
        self.run_ex(deps=self.deps(probe_duration=lambda s: 380.0))
        (self.folder / "partial.zh-TW.srt").unlink()
        (self.folder / "sakura.zh-TW.srt").unlink()
        with mock.patch.object(ss.sr, "publish", wraps=sr.publish) as publish:
            self.run_ex(host=cloud_tests.FakeHost(), deps=self.deps(probe_duration=lambda s: 380.0))
        self.assertGreaterEqual(publish.call_count, 1)
        self.assertTrue((self.folder / "partial.zh-TW.srt").exists())
        self.assertTrue((self.folder / "sakura.zh-TW.srt").exists())

    def test_changed_rows_are_published(self):
        self.fake.cues_per_slice = 70
        self.host.total = 90.0
        with mock.patch.object(ss.sr, "publish", wraps=sr.publish) as publish:
            _ex, res = self.run_ex(deps=self.deps(probe_duration=lambda s: 90.0))
        self.assertEqual(res.reason, "complete")
        self.assertGreaterEqual(publish.call_count, 3)          # 30 / 60 / 70 cues: each a different signature


if __name__ == "__main__":
    unittest.main()
