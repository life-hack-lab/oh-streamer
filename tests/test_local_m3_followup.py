"""M3 playback budget and incremental publication regressions; no GPU or network."""
import contextlib
import unittest

from tests.test_subtitle_scheduler_local import Base
from oh_streamer.queue import subtitle_local as sl, gpu_turns as gt
from oh_streamer.subtitles import slice_plan as sp, slice_results as sr


class FollowupTests(Base):
    def test_local_default_keeps_vad_rescue_grid_in_source_time(self):
        ex = sl.LocalSliceExecutor(self.folder, self.source, deps=self.deps(), host=self.host)
        self.assertEqual(ex.asr_window.context_grid_s, 60.0)
        a = sp.Slice(1, 68.57, 360.754)
        b = sp.Slice(2, 360.754, 600.0)
        self.assertEqual((ex.asr_window.audio_start(a), ex.asr_window.audio_start(b)), (0.0, 300.0))
        plain = sp.AsrWindow()
        self.assertNotEqual(ex.asr_window.cache_params(), plain.cache_params())
        self.assertLessEqual(a.start - ex.asr_window.audio_start(a), 75.0)

    def test_bad_window_does_not_exhaust_its_successful_neighbor(self):
        real_asr = self.fake.asr_batch
        def asr(windows, workdir):
            rows = real_asr(windows, workdir)
            for index, (start, end) in enumerate(windows):
                if start == 15.0:
                    rows[index] = {"segments": [], "error": "unreadable window"}
            return rows
        ex = self.executor(deps=self.deps(asr_batch=asr))
        result = ex.run(wait=True)
        self.assertEqual(result.reason, "stuck")
        self.assertEqual(ex.plan.slices[1].get("asr").status, sp.FAILED)
        self.assertTrue(ex.plan.slices[2].all_done())
        self.assertEqual(ex.plan.slices[2].get("asr").attempt, 0)

    def test_appended_slices_publish_head_then_batch_in_time_order(self):
        # An island was created before the main line. Indices remain creation order.
        ex = self.executor(resource_mode="interleave", turns=gt.TurnConfig(batch_slices=4))
        self.assertTrue(ex._prepare())
        ex._advance_cuts(None)
        ordered = sp.slices_by_time(ex.plan)
        ex.plan.slices = [ordered[-1], *ordered[:-1]]
        for index, piece in enumerate(ex.plan.slices):
            piece.index = index
        ex._save()
        ex._release_store()
        restarted = self.executor(resource_mode="interleave", turns=gt.TurnConfig(batch_slices=4))
        self.assertEqual(restarted.run().reason, "complete")
        windows = [call[1] for call in self.fake.calls if call[0] == "asr"]
        self.assertEqual(windows, [[(0.0, 30.0)],
                                   [(15.0, 330.0), (315.0, 630.0), (615.0, 930.0), (915.0, 1000.0)]])
        self.assertTrue(all(piece.all_done() for piece in restarted.plan.slices))

    def test_island_without_a_head_uses_a_normal_batch(self):
        ex = self.executor(resource_mode="interleave", turns=gt.TurnConfig(batch_slices=4))
        self.assertTrue(ex._prepare())
        ex._advance_cuts(None)
        ex.plan.slices = sp.slices_by_time(ex.plan)[1:]
        for index, piece in enumerate(ex.plan.slices):
            piece.index = index
        ex._save()
        ex._release_store()
        restarted = self.executor(resource_mode="interleave", turns=gt.TurnConfig(batch_slices=4))
        self.assertEqual(restarted.run().reason, "complete")
        windows = [call[1] for call in self.fake.calls if call[0] == "asr"]
        self.assertEqual(windows, [[(15.0, 330.0), (315.0, 630.0), (615.0, 930.0), (915.0, 1000.0)]])

    def test_contiguous_translation_follows_time_order_after_a_split(self):
        ex = self.executor()
        self.assertTrue(ex._prepare())
        self.addCleanup(ex._release_store)
        ex.plan.slices = [sp.Slice(0, 0.0, 30.0), sp.Slice(1, 630.0, 1000.0),
                          sp.Slice(2, 30.0, 330.0), sp.Slice(3, 330.0, 630.0)]
        ex.plan.mark_done(2, "asr", "a")
        ex.plan.mark_done(2, "mt", "m")
        ex.plan.mark_done(3, "asr", "a")
        task = ex._blocking_task(sp.SchedulerContext(play_pos=100.0, watched=True))
        self.assertEqual((task.slice_index, task.stage), (3, "mt"))

    def test_incomplete_asr_response_is_retried_instead_of_leaving_running_tasks(self):
        ex = self.executor(deps=self.deps(asr_batch=lambda *_: []))
        result = ex.run()
        self.assertEqual(result.reason, "quota")
        self.assertFalse(any(value == sp.RUNNING for value in self.statuses(ex.plan).values()))
        self.assertGreater(ex.plan.slices[0].get("asr").attempt, 0)

    def test_preroll_with_new_tail_is_not_discarded_as_duplicate(self):
        previous = {"start": 295.0, "end": 299.0, "text": "before"}
        crossing = {"start": 297.0, "end": 305.0, "text": "before and new words"}
        self.assertEqual(sr.resolve_preroll_overlaps([(0.0, [previous]), (300.0, [crossing])]),
                         [previous, crossing])
        duplicate = {"start": 296.0, "end": 299.0, "text": "before"}
        self.assertEqual(sr.resolve_preroll_overlaps([(0.0, [previous]), (300.0, [duplicate])]), [previous])

    def test_long_urgent_slice_requires_more_than_fixed_150_second_lead(self):
        self.host.context = sp.SchedulerContext(play_pos=100.0, watched=True)
        self.host.restore = sl.RestoreInfo(lead_s=200.0, has_work=True, running=True)
        ex, result = self.run_ex(resource_mode="interleave")
        self.assertEqual(result.reason, "restore_turn")
        self.assertEqual(ex.last_turn.reason, "subtitle_budget")
        self.assertEqual(self.host.stops, 0)
        self.assertEqual(self.fake.calls, [])
        self.assertFalse(any(state == sp.RUNNING for state in self.statuses(ex.plan).values()))

    def test_short_head_is_published_before_normal_unwatched_batch(self):
        self.host.restore = sl.RestoreInfo(lead_s=0.0, has_work=True, running=True)
        ex, result = self.run_ex(resource_mode="interleave")
        self.assertEqual(result.reason, "restore_turn")
        self.assertEqual(self.fake.calls[0], ("asr", [(0.0, 30.0)]))
        self.assertTrue(ex.plan.slices[0].all_done())
        self.assertFalse(ex.plan.slices[1].all_done())
        self.assertTrue(any(r["cues"] > 0 for r in self.host.published))

    def test_partial_batch_is_visible_before_mt_done_and_cleared_after_completion(self):
        seen = []

        @contextlib.contextmanager
        def session():
            def translate(doc, workdir, *, on_batch):
                rows = [dict(r, text_ja=r["text"], text="這是一句話") for r in doc["segments"]]
                on_batch(rows[:1])
                seen.append((self.host.published[-1]["cues"],
                             ex.plan.slices[0].get("mt").status,
                             bool(ex._partial)))
                on_batch(rows)
                return rows
            yield translate

        ex = self.executor(deps=self.deps(mt_session=session, mt_batch_callback=True))
        result = ex.run()
        self.assertEqual(result.reason, "complete")
        self.assertEqual(seen[0], (1, sp.RUNNING, True))
        self.assertEqual(ex._partial, {})
        self.assertEqual(list((self.folder / "cache" / "mt").glob("*.partial.json")), [])

    def test_partial_does_not_mark_failed_translation_complete(self):
        @contextlib.contextmanager
        def session():
            def translate(doc, workdir, *, on_batch):
                on_batch([dict(doc["segments"][0], text_ja="これは文です", text="這是一句話")])
                raise RuntimeError("translation interrupted")
            yield translate

        ex = self.executor(deps=self.deps(mt_session=session, mt_batch_callback=True))
        ex.run()
        self.assertNotEqual(ex.plan.slices[0].get("mt").status, sp.DONE)
        self.assertTrue(any(r["cues"] > 0 for r in self.host.published))
        self.assertTrue(list((self.folder / "cache" / "mt").glob("*.partial.json")))
        self.assertTrue(all(not sr.read_mt(self.folder, p.stem.removesuffix(".partial"))
                            for p in (self.folder / "cache" / "mt").glob("*.partial.json")))
        # Restart must not replace the visible partial SRT with an empty publication.
        restarted = self.executor(deps=self.deps(mt_session=session, mt_batch_callback=True))
        self.assertTrue(restarted._prepare())
        restarted._publish()
        self.assertTrue(restarted._partial)
        self.assertGreater(restarted.last_publication["cues"], 0)
        self.assertFalse(restarted.last_publication["complete"])


if __name__ == "__main__":
    unittest.main()
