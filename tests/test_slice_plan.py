"""Tests for subtitles/slice_plan.py (unified scheduler design, M1). Pure logic."""

from __future__ import annotations

import tempfile
import unittest

from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles.slice_plan import (
    Executor, SchedulerContext, SlicePlan, build_slice_plan, derive_status, select_next_task,
)

MT_ONLY = Executor("igpu", frozenset({sp.STAGE_MT}))


def uniform(duration, **kw):
    kw.setdefault("first_slice_seconds", None)
    return build_slice_plan(duration, **kw)


def finish(plan, indices, track=sp.PRIMARY_TRACK):
    for i in indices:
        plan.mark_done(i, "asr", "m-asr", track)
        plan.mark_done(i, "mt", "m-mt", track)


def pick(plan, **kw):
    return select_next_task(plan, SchedulerContext(**kw))


class PlanningTests(unittest.TestCase):
    def test_uniform_planning_and_roundtrip(self):
        plan = uniform(7200.0)
        self.assertEqual(len(plan.slices), 24)
        self.assertEqual((plan.slices[0].start, plan.slices[0].end), (0.0, 300.0))
        self.assertEqual((plan.slices[23].start, plan.slices[23].end), (6900.0, 7200.0))
        self.assertFalse(plan.open)
        finish(plan, [0])
        plan.mark_failed(1, "asr", "rate_limit", at=5.0, attempt=3, model_id="groq", detail="x")
        with tempfile.TemporaryDirectory() as tmp:
            sp.save_slice_plan(plan, tmp)
            self.assertTrue((__import__("pathlib").Path(tmp) / "slice-plan.json").is_file())
            loaded = sp.load_slice_plan(tmp)
        self.assertEqual(loaded.to_dict(), plan.to_dict())
        self.assertEqual(loaded.slices[5].start, 1500.0)
        self.assertTrue(loaded.slices[0].all_done())
        self.assertEqual(loaded.slices[1].get("asr").fail_kind, "rate_limit")
        self.assertIsNone(sp.load_slice_plan(tempfile.gettempdir() + "/definitely-missing-dir"))

    def test_silence_snaps_cut(self):
        plan = uniform(900.0, silence_intervals=[(310.0, 312.0)])
        self.assertEqual(plan.lazy["cuts"][1], 311.0)
        self.assertEqual(len(plan.slices), 3)

    def test_short_video_is_single_slice_even_with_short_first(self):
        plan = build_slice_plan(200.0)
        self.assertEqual([(s.start, s.end) for s in plan.slices], [(0.0, 200.0)])
        self.assertFalse(plan.open)

    def test_short_first_slice(self):
        plan = build_slice_plan(1000.0)           # default first slice 30 s (T3.6, D7); the rest is 5 minutes
        self.assertEqual([(s.start, s.end) for s in plan.slices],
                         [(0.0, 30.0), (30.0, 330.0), (330.0, 630.0), (630.0, 930.0), (930.0, 1000.0)])
        plan2 = build_slice_plan(1000.0, first_slice_seconds=60.0)       # a custom longer first slice: old rule
        self.assertEqual(plan2.slices[0].end, 60.0)
        plan3 = build_slice_plan(1000.0, first_slice_seconds=90.0)
        self.assertEqual(plan3.slices[0].end, 90.0)

    def test_short_first_slice_snaps_to_silence(self):
        plan = build_slice_plan(1000.0, silence_intervals=[(24.0, 26.0)])     # inside the short slice's [20, 45]
        self.assertEqual(plan.slices[0].end, 25.0)
        legacy = build_slice_plan(1000.0, silence_intervals=[(100.0, 102.0)], first_slice_seconds=90.0)
        self.assertEqual(legacy.slices[0].end, 101.0)                          # custom 90 s first slice: old rule

    def test_lazy_cuts_follow_readable_prefix_and_never_change(self):
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        calls = []

        def detect(_src, a, b):
            calls.append((a, b))
            return [(299.0, 301.0)]

        self.assertFalse(sp.lazy_cut_ready(plan, 100.0))
        self.assertFalse(sp.extend_lazy_cuts(plan, "src", 100.0, detect=detect))
        self.assertTrue(sp.lazy_cut_ready(plan, 330.0))
        self.assertTrue(sp.extend_lazy_cuts(plan, "src", 330.0, detect=detect))
        self.assertEqual([(s.start, s.end) for s in plan.slices], [(0.0, 300.0)])
        self.assertTrue(plan.open)
        cut = plan.slices[0].end
        sp.extend_lazy_cuts(plan, "src", None, detect=lambda *a: [])
        self.assertEqual(plan.slices[0].end, cut)
        self.assertFalse(plan.open)
        self.assertEqual(plan.slices[-1].end, 1000.0)

    def test_max_cuts_decides_one_cut_per_call(self):
        plan = sp.new_slice_plan(3600.0, first_slice_seconds=None)
        calls = []

        def detect(_src, a, b):
            calls.append((a, b))
            return [(b - 1.0, b)]          # a silence at the end of every window

        for n in range(1, 5):
            self.assertTrue(sp.extend_lazy_cuts(plan, "s", None, detect=detect, max_cuts=1))
            self.assertEqual(len(plan.slices), n)
            self.assertEqual(len(calls), n)             # one silence search per cut
        self.assertEqual([s.index for s in plan.slices], [0, 1, 2, 3])
        self.assertEqual(plan.slices[0].end, plan.slices[1].start)

    def test_max_cuts_default_keeps_deciding_everything(self):
        one = sp.new_slice_plan(3600.0, first_slice_seconds=None)
        many = sp.new_slice_plan(3600.0, first_slice_seconds=None)
        detect = lambda _s, a, b: [(a, b)]
        sp.extend_lazy_cuts(one, "s", None, detect=detect)
        n = 0
        while sp.extend_lazy_cuts(many, "s", None, detect=detect, max_cuts=1):
            n += 1
        self.assertEqual(n, len(many.slices))
        self.assertEqual([(s.start, s.end) for s in one.slices], [(s.start, s.end) for s in many.slices])
        self.assertFalse(one.open)
        self.assertFalse(many.open)

    def test_max_cuts_zero_changes_nothing_and_closing_cut_counts(self):
        plan = sp.new_slice_plan(400.0, first_slice_seconds=None)
        calls = []
        self.assertFalse(sp.extend_lazy_cuts(plan, "s", None, detect=lambda *a: calls.append(a) or [], max_cuts=0))
        self.assertEqual((calls, plan.slices), ([], []))
        self.assertTrue(sp.extend_lazy_cuts(plan, "s", None, detect=lambda _s, a, b: [(a, b)], max_cuts=1))
        self.assertTrue(plan.open)
        self.assertTrue(sp.extend_lazy_cuts(plan, "s", None, detect=lambda _s, a, b: [(a, b)], max_cuts=1))
        self.assertFalse(plan.open)                      # the cut that closes at the duration is one cut
        self.assertEqual(plan.slices[-1].end, 400.0)

    def test_max_cuts_with_limit_still_stops_at_the_limit(self):
        plan = sp.new_slice_plan(3600.0, first_slice_seconds=None)

        def detect(_s, a, b):                            # a 1 s pause at every multiple of 30 s
            return [(30.0 * i - 0.5, 30.0 * i + 0.5) for i in range(int(a // 30), int(b // 30) + 2)
                    if a <= 30.0 * i <= b]

        self.assertTrue(sp.extend_lazy_cuts(plan, "s", 700.0, detect=detect, max_cuts=1))
        self.assertTrue(sp.extend_lazy_cuts(plan, "s", 700.0, detect=detect, max_cuts=1))
        self.assertFalse(sp.extend_lazy_cuts(plan, "s", 700.0, detect=detect, max_cuts=1))   # window past limit
        self.assertEqual(len(plan.slices), 2)

    def test_detect_failure_freezes_nominal_cut_after_three_tries(self):
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)

        def boom(*a):
            raise RuntimeError("ffmpeg")

        for _ in range(2):
            sp.extend_lazy_cuts(plan, "s", None, detect=boom)
            self.assertEqual(plan.slices, [])
        sp.extend_lazy_cuts(plan, "s", None, detect=boom)
        self.assertEqual(plan.slices[0].end, 300.0)

    def test_find_slice_index(self):
        plan = uniform(900.0)
        self.assertEqual(sp.find_slice_index_for_position(plan.slices, 350.0), 1)
        self.assertEqual(sp.find_slice_index_for_position(plan.slices, 5000.0), 2)
        self.assertIsNone(sp.find_slice_index_for_position([], 1.0))


class LegacySelectorCasesTests(unittest.TestCase):
    """Cases moved from select_next_cloud_action tests (stage qwen no longer exists, see design D2)."""

    def test_smart_horizon_and_early_play(self):
        plan = uniform(7200.0)
        a = pick(plan)
        self.assertEqual((a.slice_index, a.stage, a.urgent), (0, "asr", True))
        plan.mark_done(0, "asr")
        a = pick(plan)
        self.assertEqual((a.slice_index, a.stage), (0, "mt"))
        finish(plan, range(6))
        self.assertIsNone(pick(plan, restore_pos=600.0), "stands by at the 30 min horizon")
        a = pick(plan, restore_pos=1200.0)
        self.assertEqual((a.slice_index, a.stage), (6, "asr"))
        a = pick(plan, play_pos=4800.0, restore_pos=600.0, watched=True)
        self.assertEqual((a.slice_index, a.stage, a.urgent, a.reason), (16, "asr", True, "seek_urgent"))

    def test_low_watermark(self):
        plan = uniform(7200.0)
        finish(plan, range(6))
        self.assertIsNone(pick(plan, play_pos=100.0, watched=True), "1700 s ahead >= 20 min")
        a = pick(plan, play_pos=700.0, watched=True)       # 1100 s ahead < 1200
        self.assertEqual((a.slice_index, a.stage), (6, "asr"))

    def test_full_preload(self):
        plan = uniform(7200.0, strategy="full_preload")
        a = pick(plan, strategy="full_preload")
        self.assertEqual((a.slice_index, a.stage), (0, "asr"))
        finish(plan, [0])
        a = pick(plan, strategy="full_preload")
        self.assertEqual((a.slice_index, a.stage, a.reason), (1, "asr", "preload"))
        plan.mark_done(1, "asr")
        # all asr first, then mt
        for i in range(2, 24):
            plan.mark_done(i, "asr")
        a = pick(plan, strategy="full_preload")
        self.assertEqual((a.slice_index, a.stage), (1, "mt"))

    def test_item_done_means_everything_due(self):
        plan = uniform(3000.0)
        finish(plan, [0])
        self.assertEqual(pick(plan, item_state="done").slice_index, 1)
        finish(plan, range(10))
        self.assertIsNone(pick(plan, item_state="done"))

    def test_download_bound_and_seek_priority(self):
        plan = uniform(900.0)
        self.assertIsNone(pick(plan, max_available=100.0), "head slice waits for its own audio")
        a = pick(plan, max_available=300.0)
        self.assertEqual((a.slice_index, a.stage), (0, "asr"))
        finish(plan, [0])
        self.assertIsNone(pick(plan, max_available=450.0), "slice 1 ends at 600")
        self.assertEqual(pick(plan, max_available=600.0).slice_index, 1)
        a = pick(plan, play_pos=650.0, watched=True, max_available=900.0)
        self.assertEqual((a.slice_index, a.urgent), (2, True))
        self.assertIsNone(pick(plan, play_pos=650.0, watched=True, max_available=600.0))
        self.assertEqual(pick(plan, max_available=None).slice_index, 1)

    def test_downloaded_slice_can_still_translate_past_bound(self):
        plan = uniform(900.0)
        finish(plan, [0])
        plan.mark_done(1, "asr")
        a = pick(plan, max_available=100.0)
        self.assertEqual((a.slice_index, a.stage), (1, "mt"))


class StateAndExecutorTests(unittest.TestCase):
    def test_mt_only_executor_gets_only_mt(self):
        plan = uniform(900.0)
        self.assertIsNone(select_next_task(plan, SchedulerContext(), MT_ONLY), "nothing transcribed yet")
        finish(plan, [0])
        plan.mark_done(1, "asr")
        t = select_next_task(plan, SchedulerContext(), MT_ONLY)
        self.assertEqual((t.slice_index, t.stage), (1, "mt"))
        self.assertEqual(select_next_task(plan, SchedulerContext()).stage, "asr", "unfiltered: asr sweep first")
        plan.mark_done(2, "asr")
        plan.mark_running(1, "mt", "igpu")
        t = select_next_task(plan, SchedulerContext(), MT_ONLY)
        self.assertEqual((t.slice_index, t.stage), (2, "mt"))

    def test_asr_only_executor_skips_mt(self):
        plan = uniform(900.0)
        plan.mark_done(0, "asr")
        t = select_next_task(plan, SchedulerContext(), Executor("asr", frozenset({"asr"})))
        self.assertEqual((t.slice_index, t.stage), (1, "asr"))

    def test_running_failed_and_retry_are_skipped(self):
        plan = uniform(900.0)
        plan.mark_running(0, "asr", "cloud")
        self.assertEqual(pick(plan).slice_index, 1, "falls through while the head slice runs elsewhere")
        plan.mark_failed(1, "asr", "auth")
        self.assertEqual(pick(plan).slice_index, 2)
        plan.defer(0, "asr", retry_at=1000.0)
        plan.release(2, "asr")
        self.assertEqual(pick(plan, now=500.0).slice_index, 2, "quota wait of the head slice skipped")
        self.assertEqual(pick(plan, now=1001.0).slice_index, 0)

    def test_deferred_is_pending_not_failed(self):
        plan = uniform(600.0)
        plan.defer(0, "asr", retry_at=1000.0)
        st = derive_status(plan, now=500.0)
        self.assertTrue(st.waiting_quota)
        self.assertFalse(st.stuck)
        self.assertEqual(st.next_retry_at, 1000.0)
        self.assertFalse(derive_status(plan, now=1001.0).waiting_quota)

    def test_wait_for_executor_reserves_task(self):
        plan = uniform(600.0)
        plan.mark_done(0, "asr")
        plan.wait_for(0, "mt", "local")
        self.assertEqual(plan.slices[0].get("mt").status, sp.PENDING)
        cloud = Executor("cloud")
        local = Executor("local")
        self.assertEqual(select_next_task(plan, SchedulerContext(), cloud).slice_index, 1)
        t = select_next_task(plan, SchedulerContext(), local)
        self.assertEqual((t.slice_index, t.stage, t.urgent), (0, "mt", True))
        st = derive_status(plan)
        self.assertEqual(st.waiting_executor, "local")

    def test_reset_failed_and_release_running(self):
        plan = uniform(600.0)
        plan.mark_failed(0, "asr", "quota")
        plan.mark_running(1, "asr", "cloud")
        self.assertEqual(plan.release_running("cloud"), 1)
        self.assertEqual(plan.slices[1].get("asr").status, sp.PENDING)
        self.assertEqual(plan.reset_failed(), 1)
        self.assertEqual(plan.slices[0].get("asr").status, sp.PENDING)
        self.assertFalse(derive_status(plan).stuck)


class TrackTests(unittest.TestCase):
    def two_tracks(self, duration=3000.0):
        return uniform(duration, tracks=["primary", "cmp"])

    def test_secondary_waits_for_primary_lead(self):
        plan = self.two_tracks()
        finish(plan, [0])                                    # lead 300 < 600
        ctx = SchedulerContext(restore_pos=0.0)
        # primary still has horizon work, so it is chosen first
        self.assertEqual(select_next_task(plan, ctx).track, "primary")
        finish(plan, range(1, 6))                            # primary caught up with the horizon, lead 1800
        t = select_next_task(plan, ctx)
        self.assertEqual((t.slice_index, t.stage, t.track), (0, "asr", "cmp"))
        self.assertEqual(t.reason, "secondary_track")

    def test_secondary_blocked_below_threshold_and_param(self):
        plan = self.two_tracks(900.0)
        finish(plan, [0, 1])                                 # lead 600 -> not complete, equals threshold
        plan.mark_running(2, "asr", "x")                     # primary has nothing pickable
        self.assertEqual(select_next_task(plan, SchedulerContext()).track, "cmp")
        self.assertIsNone(select_next_task(plan, SchedulerContext(secondary_min_lead_s=900.0)))

    def test_secondary_runs_when_primary_complete_and_not_ahead(self):
        plan = self.two_tracks(600.0)
        finish(plan, [0, 1])
        finish(plan, [0], "cmp")
        t = select_next_task(plan, SchedulerContext())
        self.assertEqual((t.slice_index, t.track), (1, "cmp"))
        finish(plan, [1], "cmp")
        self.assertIsNone(select_next_task(plan, SchedulerContext()))

    def test_secondary_never_ahead_of_primary(self):
        plan = self.two_tracks(3000.0)
        finish(plan, range(3))
        plan.mark_running(3, "asr", "x")
        finish(plan, range(3), "cmp")
        self.assertNotEqual(select_next_task(plan, SchedulerContext(restore_pos=5000.0)).track, "cmp")

    def test_track_filter_on_executor(self):
        plan = self.two_tracks(600.0)
        finish(plan, [0, 1])
        ex = Executor("only-primary", tracks=frozenset({"primary"}))
        self.assertIsNone(select_next_task(plan, SchedulerContext(), ex))
        self.assertIsNotNone(select_next_task(plan, SchedulerContext()))


class DoneKeepsTriedTests(unittest.TestCase):
    def test_mark_done_keeps_tried_and_candidate(self):
        plan = uniform(600.0)
        plan.slices[0].set("asr", sp.PRIMARY_TRACK, sp.TaskState(
            status=sp.RUNNING, candidate="c2", tried={"c1": {"kind": "rate_limit", "at": 1.0}}))
        plan.mark_done(0, "asr", "m")
        st = plan.slices[0].get("asr")
        self.assertEqual(st.status, sp.DONE)
        self.assertEqual(st.tried, {"c1": {"kind": "rate_limit", "at": 1.0}})
        self.assertEqual(st.candidate, "c2")
        plan.mark_done(0, "asr", "m", candidate="c3")
        self.assertEqual(plan.slices[0].get("asr").candidate, "c3")
        self.assertIn("c1", plan.slices[0].get("asr").tried)


class StatusTests(unittest.TestCase):
    def test_progress_and_coverage(self):
        plan = uniform(900.0)
        finish(plan, [0, 1])
        st = derive_status(plan)
        self.assertEqual((st.slices_total, st.slices_done), (3, 2))
        self.assertAlmostEqual(st.progress, 600 / 900)
        self.assertEqual(st.covered_s, 600.0)
        self.assertFalse(st.complete)
        st = derive_status(plan, play_pos=700.0)
        self.assertEqual((st.covered_s, st.covered_ahead_s), (700.0, 0.0))
        self.assertEqual(st.blocking_slice, 2)
        st = derive_status(plan, play_pos=100.0)
        self.assertEqual(st.covered_ahead_s, 500.0)
        finish(plan, [2])
        st = derive_status(plan)
        self.assertTrue(st.complete)
        self.assertEqual(st.progress, 1.0)

    def test_stuck_reports_model_and_reason(self):
        plan = uniform(900.0)
        finish(plan, [0])
        plan.mark_done(1, "asr")
        plan.mark_failed(1, "mt", "quota", at=7.0, attempt=3, model_id="groq-llama", detail="daily cap")
        st = derive_status(plan)
        self.assertTrue(st.stuck)
        self.assertTrue(st.blocked_playback)
        f = st.failures[0]
        self.assertEqual((f.slice_index, f.stage, f.model_id, f.kind, f.attempt), (1, "mt", "groq-llama", "quota", 3))
        self.assertEqual(f.detail, "daily cap")
        # failure behind the playback position is stuck but does not block playback there
        st = derive_status(plan, play_pos=800.0)
        self.assertTrue(st.stuck)
        self.assertFalse(st.blocked_playback)

    def test_secondary_failure_is_not_primary_stuck(self):
        plan = uniform(600.0, tracks=["primary", "cmp"])
        plan.mark_failed(0, "asr", "auth", track="cmp")
        self.assertFalse(derive_status(plan).stuck)
        self.assertEqual(len(derive_status(plan).failures), 1)

    def test_waiting_download(self):
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        st = derive_status(plan)
        self.assertTrue(st.waiting_download)                 # no slice yet
        sp.extend_lazy_cuts(plan, "s", 330.0, detect=lambda *a: [(299.0, 301.0)])
        st = derive_status(plan, max_available=250.0)
        self.assertTrue(st.waiting_download)                 # slice 0 not fully downloaded
        st = derive_status(plan, max_available=330.0)
        self.assertFalse(st.waiting_download)
        finish(plan, [0])
        st = derive_status(plan, max_available=330.0)
        self.assertTrue(st.waiting_download)                 # all existing slices done, plan still open
        self.assertFalse(st.complete)

    def test_running_executor(self):
        plan = uniform(600.0)
        plan.mark_running(0, "asr", "cloud")
        self.assertEqual(derive_status(plan).running_executor, "cloud")


class CacheKeyTests(unittest.TestCase):
    ident = {"id": "whisper", "provider": "local", "revision": [1, 2]}

    def test_asr_key_stable_and_sensitive(self):
        k = sp.asr_cache_key("hls:abc", 0.0, 300.0, self.ident, {"lang": "ja"})
        self.assertEqual(k, sp.asr_cache_key("hls:abc", 0, 300, dict(reversed(list(self.ident.items()))),
                                             {"lang": "ja"}))
        self.assertEqual(len(k), 32)
        for other in (sp.asr_cache_key("hls:abd", 0.0, 300.0, self.ident, {"lang": "ja"}),
                      sp.asr_cache_key("hls:abc", 0.0, 301.0, self.ident, {"lang": "ja"}),
                      sp.asr_cache_key("hls:abc", 0.0, 300.0, dict(self.ident, revision=[1, 3]), {"lang": "ja"}),
                      sp.asr_cache_key("hls:abc", 0.0, 300.0, self.ident, {"lang": "en"})):
            self.assertNotEqual(k, other)

    def test_keys_do_not_depend_on_item_or_settings(self):
        # the signature has no item id / settings snapshot; identical content -> identical key
        a = sp.asr_cache_key("m", 0, 90, self.ident)
        b = sp.asr_cache_key("m", 0.0004, 90.0, self.ident)       # sub-millisecond noise rounds away
        self.assertEqual(a, b)

    def test_mt_key(self):
        digest = sp.asr_result_digest([{"start": 0, "end": 1, "text": "x"}])
        self.assertEqual(digest, sp.asr_result_digest([{"text": "x", "end": 1, "start": 0}]))
        self.assertNotEqual(digest, sp.asr_result_digest([{"start": 0, "end": 1, "text": "y"}]))
        mt = {"id": "sakura"}
        k = sp.mt_cache_key(digest, mt, "p1", "g1")
        self.assertEqual(k, sp.mt_cache_key(digest, dict(mt), "p1", "g1"))
        self.assertNotEqual(k, sp.mt_cache_key(digest, mt, "p2", "g1"))
        self.assertNotEqual(k, sp.mt_cache_key(digest, mt, "p1", "g2"))
        self.assertNotEqual(k, sp.mt_cache_key(digest, {"id": "other"}, "p1", "g1"))

    def test_paths(self):
        self.assertEqual(sp.asr_cache_path("abc").as_posix(), "cache/asr/abc.json")
        self.assertEqual(sp.mt_cache_path("abc").as_posix(), "cache/mt/abc.json")


class SilenceOnlyCutTests(unittest.TestCase):
    """v2 cut rule (design 7.2): a cut only lands in a silence; the search widens instead of cutting at the
    nominal time."""

    @staticmethod
    def detector(silences, calls=None):
        def detect(_src, a, b):
            if calls is not None:
                calls.append((a, b))
            return [(max(x, a), min(y, b)) for x, y in silences if y > a and x < b]
        return detect

    def test_no_silence_near_nominal_widens_window(self):
        plan = build_slice_plan(1000.0, silence_intervals=[(380.0, 381.0), (700.0, 701.0)], first_slice_seconds=None)
        self.assertEqual(plan.slices[0].end, 380.5)          # not 300.0 in the middle of a sentence
        self.assertEqual(plan.slices[1].end, 700.5)
        self.assertNotIn("hard_cuts", plan.lazy)

    def test_nearest_silence_wins_in_both_directions(self):
        plan = build_slice_plan(1000.0, silence_intervals=[(230.0, 231.0), (380.0, 381.0)], first_slice_seconds=None)
        self.assertEqual(plan.slices[0].end, 230.5)          # 70 s early beats 80 s late

    def test_widening_waits_for_the_download_and_remembers_the_search(self):
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        calls = []
        detect = self.detector([(380.0, 381.0)], calls)
        self.assertTrue(sp.extend_lazy_cuts(plan, "s", 340.0, detect=detect))   # ±30 s searched, no silence
        self.assertEqual(plan.slices, [])
        self.assertEqual(calls, [(270.0, 330.0)])
        self.assertFalse(sp.lazy_cut_ready(plan, 340.0))                       # next window ends at 390
        self.assertFalse(sp.extend_lazy_cuts(plan, "s", 345.0, detect=detect))
        self.assertEqual(len(calls), 1)                                        # not searched again
        sp.extend_lazy_cuts(plan, "s", 400.0, detect=detect)
        self.assertEqual(calls[1], (210.0, 390.0))
        self.assertEqual(plan.slices[0].end, 380.5)
        self.assertNotIn("search", plan.lazy)

    def test_continuous_sound_falls_back_to_relaxed_then_long_silence_then_forced(self):
        long_only = build_slice_plan(1000.0, silence_intervals=[(400.0, 420.0)], first_slice_seconds=None)
        self.assertEqual(long_only.slices[0].end, 410.0)                     # long pause only if nothing else
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        relaxed = self.detector([(450.0, 450.4)])
        sp.extend_lazy_cuts(plan, "s", None, detect=lambda *a: [], relaxed_detect=relaxed)
        self.assertEqual(plan.slices[0].end, 450.2)

    def test_relaxed_pauses_tried_before_widening_longer_pause_preferred(self):
        # T3 clip: nothing at -30 dB near 300 s, short pauses a few dB higher; adjacent pieces merge to 0.7 s
        relaxed = self.detector([(299.3, 299.6), (301.2, 301.6), (301.6, 301.9), (317.0, 317.3)])
        plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(plan, "s", 340.0, detect=lambda *a: [], relaxed_detect=relaxed)
        self.assertAlmostEqual(plan.slices[0].end, 301.55)
        only_short = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(only_short, "s", 340.0, detect=lambda *a: [], relaxed_detect=self.detector([(299.3, 299.6)]))
        self.assertAlmostEqual(only_short.slices[0].end, 299.45)
        strict_first = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(strict_first, "s", 340.0, detect=self.detector([(320.0, 321.0)]), relaxed_detect=relaxed)
        self.assertEqual(strict_first.slices[0].end, 320.5)            # a real silence wins over a nearer pause
        forced = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(forced, "s", None, detect=lambda *a: [])
        self.assertEqual(forced.slices[0].end, 300.0)
        self.assertIn(300.0, forced.lazy["hard_cuts"])

    def test_forced_cut_needs_the_whole_search_range(self):
        plan = sp.new_slice_plan(2000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(plan, "s", 500.0, detect=lambda *a: [])
        self.assertEqual(plan.slices, [])                                    # range goes to 600 s
        sp.extend_lazy_cuts(plan, "s", 610.0, detect=lambda *a: [])
        self.assertEqual(plan.slices[0].end, 300.0)

    def test_first_slice_rule_is_the_same(self):
        # a custom first slice longer than 30 s still follows the normal widening rule (target 90, widened)
        plan = build_slice_plan(1000.0, silence_intervals=[(150.0, 151.0), (450.0, 451.0)], first_slice_seconds=90.0)
        self.assertEqual(plan.slices[0].end, 150.5)
        # the default 30 s first slice does not widen: no silence in [20, 45] is a hard cut at 30 (T3.6)
        short = build_slice_plan(1000.0, silence_intervals=[(150.0, 151.0), (450.0, 451.0)])
        self.assertEqual(short.slices[0].end, 30.0)


class PreRollOwnershipTests(unittest.TestCase):
    def cue(self, a, b, text=None):
        return {"start": a, "end": b, "text": text or f"{a}-{b}"}

    def test_window(self):
        w = sp.AsrWindow()
        self.assertEqual(w.pre_roll_s, 15.0)
        self.assertEqual(w.audio_start(sp.Slice(0, 0.0, 90.0)), 0.0)
        self.assertEqual(w.audio_start(sp.Slice(1, 90.0, 390.0)), 75.0)
        self.assertEqual(w.audio_start(sp.Slice(1, 10.0, 390.0)), 0.0)
        self.assertNotEqual(w.cache_params(), sp.AsrWindow(pre_roll_s=10.0).cache_params())

    def test_preroll_cues_ending_before_the_cut_are_dropped(self):
        segs = [self.cue(76.0, 80.0), self.cue(85.0, 88.9), self.cue(88.0, 89.5), self.cue(92.0, 95.0)]
        own = sp.own_segments(segs, 90.0, 390.0, first=False, last=False)
        self.assertEqual([s["text"] for s in own], ["88.0-89.5", "92.0-95.0"])

    def test_keep_preroll_keeps_cues_ending_before_the_cut(self):
        segs = [self.cue(76.0, 80.0), self.cue(85.0, 88.9), self.cue(88.0, 89.5), self.cue(92.0, 95.0)]
        own = sp.own_segments(segs, 90.0, 390.0, first=False, last=False, keep_preroll=True)
        self.assertEqual(len(own), 4)
        tail = sp.own_segments([self.cue(380.0, 389.9)], 90.0, 390.0, first=False, last=False, keep_preroll=True)
        self.assertEqual(tail, [])                  # the tail guard is unchanged

    def test_cue_crossing_the_cut_belongs_to_the_later_slice(self):
        # earlier slice's audio ends at the cut: its copy of the sentence is cut off (ends at 90.0)
        earlier = sp.own_segments([self.cue(80.0, 84.0), self.cue(88.0, 90.0, "途中")], 0.0, 90.0,
                                  first=True, last=False)
        later = sp.own_segments([self.cue(88.0, 91.4, "完全な文"), self.cue(92.0, 95.0)], 90.0, 390.0,
                                first=False, last=False)
        self.assertEqual([s["text"] for s in earlier], ["80.0-84.0"])
        self.assertEqual([s["text"] for s in later], ["完全な文", "92.0-95.0"])

    def test_one_run_over_contiguous_slices_splits_exactly(self):
        cues = [self.cue(0.5 + 4 * k, 3.5 + 4 * k) for k in range(100)]          # 0.5 .. 399.5
        cuts = [(0.0, 90.0), (90.0, 390.0)]
        owned = []
        for i, (a, b) in enumerate(cuts):
            g = sp.DEFAULT_TAIL_GUARD_SECONDS
            owned += sp.own_segments(cues, a, b, first=i == 0, last=False, lead_keep_s=g, tail_guard_s=g)
        texts = [s["text"] for s in owned]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(set(texts), {c["text"] for c in cues if c["end"] <= 390.0 - sp.DEFAULT_TAIL_GUARD_SECONDS})

    def test_first_and_last_slices_have_no_outer_bound(self):
        segs = [self.cue(-0.5, 0.2), self.cue(995.0, 1000.0)]
        self.assertEqual(len(sp.own_segments(segs, 0.0, 1000.0, first=True, last=True)), 2)
        plan = build_slice_plan(1000.0, first_slice_seconds=None)
        self.assertTrue(sp.is_last_slice(plan, plan.slices[-1]))
        self.assertFalse(sp.is_last_slice(plan, plan.slices[0]))
        open_plan = sp.new_slice_plan(1000.0, first_slice_seconds=None)
        sp.extend_lazy_cuts(open_plan, "s", 340.0, detect=lambda *a: [(299.0, 301.0)])
        self.assertFalse(sp.is_last_slice(open_plan, open_plan.slices[-1]))


if __name__ == "__main__":
    unittest.main()
