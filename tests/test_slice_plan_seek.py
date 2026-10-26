"""T3.6-A: the slice table after a seek.

Pure logic (subtitles/slice_plan.py, slice_results.py): 30 s first slice, island cuts in an uncut region, splits of
a slice nothing has started on, slices that are not in time order. No I/O, no GPU."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr
from oh_streamer.subtitles.slice_plan import SchedulerContext, build_slice_plan, select_next_task


def silent_at(*points, width=2.0):
    """Silences of `width` seconds centred on each point."""
    return [(p - width / 2.0, p + width / 2.0) for p in points]


def detector(silences, calls=None):
    def detect(_src, a, b):
        if calls is not None:
            calls.append((a, b))
        return [(max(x, a), min(y, b)) for x, y in silences if y > a and x < b]
    return detect


def everywhere(_src, a, b):
    """Silent everywhere: every cut lands on its nominal time."""
    return [(a, b)]


def uniform(duration, **kw):
    kw.setdefault("first_slice_seconds", None)
    return build_slice_plan(duration, **kw)


def seek(plan, pos, silences=(), relaxed=None, limit=None):
    """seek_cut_request + apply_seek_cuts the way an executor does; returns (request, applied)."""
    request = sp.seek_cut_request(plan, pos, limit=limit)
    if request is None:
        return None, False
    return request, sp.apply_seek_cuts(plan, request, list(silences), relaxed)


def stream_plan(duration=3600.0, **kw):
    kw.setdefault("media_id", "stream:test")
    return sp.new_slice_plan(duration, **kw)


def ranges(plan):
    return [(s.index, s.start, s.end) for s in plan.slices]


def by_time(plan):
    return [(s.start, s.end) for s in sp.slices_by_time(plan)]


class OpeningTests(unittest.TestCase):
    """Case 1 and 2: the first slice of every new plan is a 30 s slice."""

    def test_silence_at_24_s_cuts_there(self):
        plan = build_slice_plan(3600.0, silence_intervals=silent_at(24.0))
        self.assertEqual((plan.slices[0].start, plan.slices[0].end), (0.0, 24.0))
        self.assertNotIn(24.0, plan.lazy.get("hard_cuts", []))          # a real silence, not a forced cut

    def test_silences_outside_20_to_45_are_a_hard_cut_at_30(self):
        plan = build_slice_plan(3600.0, silence_intervals=silent_at(18.0, 50.0))
        self.assertEqual(plan.slices[0].end, 30.0)
        self.assertIn(30.0, plan.lazy["hard_cuts"])

    def test_silence_at_41_s_cuts_there(self):
        plan = build_slice_plan(3600.0, silence_intervals=silent_at(41.0))
        self.assertEqual(plan.slices[0].end, 41.0)

    def test_the_short_search_range_is_not_widened(self):
        calls = []
        plan = sp.new_slice_plan(3600.0)
        sp.extend_lazy_cuts(plan, "s", None, detect=detector(silent_at(100.0), calls), max_cuts=1)
        self.assertEqual(calls, [(20.0, 45.0)])                      # one search, exactly the short range
        self.assertEqual(plan.slices[0].end, 30.0)

    def test_the_relaxed_tier_is_used_inside_the_short_range_before_a_hard_cut(self):
        plan = sp.new_slice_plan(3600.0)
        sp.extend_lazy_cuts(plan, "s", None, detect=detector([]), relaxed_detect=detector([(36.0, 36.6)]), max_cuts=1)
        self.assertAlmostEqual(plan.slices[0].end, 36.3)
        self.assertNotIn("hard_cuts", plan.lazy)

    def test_a_250_s_file_is_not_cut(self):
        plan = build_slice_plan(250.0, silence_intervals=silent_at(24.0))
        self.assertEqual([(s.start, s.end) for s in plan.slices], [(0.0, 250.0)])
        self.assertFalse(plan.open)

    def test_the_short_slice_needs_only_45_s_of_audio(self):
        plan = sp.new_slice_plan(3600.0)
        self.assertFalse(sp.lazy_cut_ready(plan, 44.0))
        self.assertTrue(sp.lazy_cut_ready(plan, 45.0))
        self.assertFalse(sp.extend_lazy_cuts(plan, "s", 44.0, detect=everywhere))
        self.assertTrue(sp.extend_lazy_cuts(plan, "s", 45.0, detect=everywhere, max_cuts=1))
        self.assertEqual(plan.slices[0].end, 30.0)

    def test_default_first_slice_is_30_and_old_plans_keep_their_value(self):
        self.assertEqual(sp.SHORT_SLICE_SECONDS, 30.0)
        self.assertEqual(sp.DEFAULT_FIRST_SLICE_SECONDS, 30.0)
        self.assertEqual(sp.new_slice_plan(1000.0).first_slice_seconds, 30.0)
        legacy = build_slice_plan(1000.0, first_slice_seconds=90.0)
        self.assertEqual(legacy.slices[0].end, 90.0)

    def test_after_the_short_slice_the_normal_5_minute_rule_resumes(self):
        plan = build_slice_plan(3600.0, silence_intervals=silent_at(25.0, 325.0))
        self.assertEqual(by_time(plan)[:2], [(0.0, 25.0), (25.0, 325.0)])
        everywhere_plan = build_slice_plan(3600.0)
        self.assertEqual(by_time(everywhere_plan)[:3], [(0.0, 30.0), (30.0, 330.0), (330.0, 630.0)])


class IslandTests(unittest.TestCase):
    """Cases 3-6: a seek into the region no slice covers."""

    def test_seek_into_an_empty_plan_makes_an_island_with_a_short_slice(self):
        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)
        self.assertEqual((request.kind, request.position), ("island", 2400.0))
        self.assertEqual(request.window, (2388.0, 2445.0))
        applied = sp.apply_seek_cuts(plan, request, silent_at(2392.0, 2422.0))
        self.assertTrue(applied)
        self.assertEqual(ranges(plan), [(0, 2392.0, 2422.0)])
        self.assertEqual(plan.lazy["islands"], [{"cuts": [2392.0, 2422.0], "closed": False}])
        self.assertEqual(plan.lazy["cuts"], [0.0])                       # nothing before 2392 was cut
        self.assertTrue(plan.open)
        self.assertIs(sp.slice_containing(plan, 2400.0), plan.slices[0])
        self.assertIsNone(sp.slice_containing(plan, 2391.0))
        task = select_next_task(plan, SchedulerContext(play_pos=2400.0, watched=True))
        self.assertEqual((task.slice_index, task.stage, task.urgent, task.reason), (0, "asr", True, "seek_urgent"))
        # the island grows from its last cut by the normal rule, the main chain does not move
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, near=sp.stream_grow_window(2400.0), max_cuts=2)
        self.assertEqual(plan.lazy["cuts"], [0.0])
        self.assertEqual(plan.lazy["islands"][0]["cuts"], [2392.0, 2422.0, 2722.0, 3022.0])

    def test_no_silence_near_the_target_uses_the_relaxed_pauses_then_a_hard_cut(self):
        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)
        strict = silent_at(2422.0)                    # a silence for the short cut only; none in [2388, 2400]
        self.assertTrue(sp.seek_needs_relaxed(plan, request, strict))
        self.assertTrue(sp.apply_seek_cuts(plan, request, strict, [(2395.0, 2396.0)]))
        self.assertEqual(plan.slices[0].start, 2395.5)                    # the relaxed pause
        self.assertNotIn("hard_cuts", plan.lazy)

        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)
        self.assertTrue(sp.apply_seek_cuts(plan, request, [], []))
        self.assertEqual((plan.slices[0].start, plan.slices[0].end), (2400.0, 2430.0))
        self.assertEqual(plan.lazy["hard_cuts"], [2400.0, 2430.0])           # both are forced cuts
        self.assertFalse(sp.seek_needs_relaxed(stream_plan(), request, silent_at(2395.0, 2422.0)))

    def test_the_start_is_the_silence_nearest_to_the_position(self):
        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)
        sp.apply_seek_cuts(plan, request, silent_at(2390.0, 2398.0, 2430.0))
        self.assertEqual(plan.slices[0].start, 2398.0)                       # 2 s before beats 10 s before
        self.assertEqual(plan.slices[0].end, 2430.0)                         # nearest to c + 30 in [2418, 2443]
        self.assertEqual(plan.seek_events[0]["kind"], "island")

    def test_a_start_within_10_s_of_the_decided_cut_on_its_left_reuses_that_cut(self):
        plan = build_slice_plan(3600.0)
        plan.lazy["closed"] = False                       # take only the first two slices of the plan
        plan = stream_plan()
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=2)
        self.assertEqual(plan.lazy["cuts"], [0.0, 30.0, 330.0])
        request = sp.seek_cut_request(plan, 345.0)
        self.assertEqual(request.window[0], 333.0)
        self.assertTrue(sp.apply_seek_cuts(plan, request, silent_at(334.5, 360.0)))
        self.assertEqual(plan.lazy["islands"], [])                           # continued the main chain
        self.assertEqual(plan.lazy["cuts"], [0.0, 30.0, 330.0, 360.0])
        self.assertEqual(ranges(plan)[-1], (2, 330.0, 360.0))
        self.assertTrue(plan.seek_events[0]["snapped"])
        self.assertTrue(all(s.end - s.start >= 10.0 for s in plan.slices))
        # a position just past the front: every possible start is within 10 s, no silence is needed
        request = sp.seek_cut_request(plan, 366.0)
        self.assertEqual(request.window[0], 360.0)

    def test_a_start_further_than_10_s_from_the_left_cut_makes_an_island_and_the_gap_is_joined_later(self):
        plan = stream_plan()
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=2)
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 360.0), silent_at(355.5, 385.5))
        self.assertEqual(plan.lazy["islands"], [{"cuts": [355.5, 385.5], "closed": False}])
        # the main chain's next cut closes the 25.5 s gap onto the island start, without detecting anything
        calls = []
        sp.extend_lazy_cuts(plan, "s", None, detect=detector([], calls), max_cuts=1)
        self.assertEqual(calls, [])
        self.assertEqual(plan.lazy["islands"], [])
        self.assertEqual(plan.lazy["cuts"], [0.0, 30.0, 330.0, 355.5, 385.5])
        self.assertTrue(all(s.end - s.start >= 10.0 for s in plan.slices))

    def test_less_than_75_s_up_to_the_next_chain_is_one_slice(self):
        plan = stream_plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))
        # seek 60 s before the island: 2392 - 2340 = 52 s <= 75: [c, 2392] is one slice, merged with the island
        request = sp.seek_cut_request(plan, 2350.0)
        sp.apply_seek_cuts(plan, request, silent_at(2340.0))
        self.assertEqual(by_time(plan), [(2340.0, 2392.0), (2392.0, 2422.0)])
        self.assertEqual(plan.lazy["islands"], [{"cuts": [2340.0, 2392.0, 2422.0], "closed": False}])

    def test_a_seek_near_the_end_takes_the_rest_as_one_slice(self):
        plan = stream_plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 3550.0), silent_at(3545.0))
        self.assertEqual(by_time(plan), [(3545.0, 3600.0)])
        self.assertEqual(plan.lazy["islands"], [{"cuts": [3545.0, 3600.0], "closed": True}])
        self.assertTrue(sp.is_last_slice(plan, plan.slices[0]))
        self.assertTrue(plan.open)                                           # not joined to the main chain yet

    def test_positions_that_need_no_cut(self):
        plan = stream_plan()
        self.assertIsNone(sp.seek_cut_request(plan, 0.0))
        self.assertIsNone(sp.seek_cut_request(plan, 3600.0))
        self.assertIsNone(sp.seek_cut_request(plan, 4000.0))
        self.assertIsNone(sp.seek_cut_request(plan, 2400.0, limit=2420.0))     # window ends at 2445: not readable
        self.assertIsNotNone(sp.seek_cut_request(plan, 2400.0, limit=2445.0))


class IslandJoinTests(unittest.TestCase):
    """Case 5: closing the gap between the main chain and an island."""

    def setUp(self):
        self.plan = stream_plan()
        sp.apply_seek_cuts(self.plan, sp.seek_cut_request(self.plan, 2400.0), silent_at(2392.0, 2422.0))

    def test_main_chain_closes_onto_the_island_without_detection_and_then_continues(self):
        calls = []
        detect = detector([], calls)
        everyw = lambda src, a, b: (calls.append((a, b)) or [(a, b)])
        sp.extend_lazy_cuts(self.plan, "s", None, detect=everyw)           # grows both chains to the end
        cuts = self.plan.lazy["cuts"]
        self.assertEqual(self.plan.lazy["islands"], [])
        self.assertFalse(self.plan.open)
        self.assertEqual((cuts[0], cuts[-1]), (0.0, 3600.0))
        self.assertIn(2392.0, cuts)
        at = cuts.index(2392.0)
        self.assertEqual(cuts[at + 1], 2422.0)                                # the island's own cuts follow
        self.assertLessEqual(2392.0 - cuts[at - 1], 420.0)                    # joined within 300 + 120 s
        self.assertIn((2130.0, 2392.0), by_time(self.plan))                    # closed onto the island start
        self.assertNotIn(2392.0, self.plan.lazy.get("hard_cuts", []))
        # no detection window of the main chain reaches past island start - 120 s
        before_island = [c for c in calls if c[0] < 2392.0]
        self.assertTrue(all(b <= 2392.0 - 120.0 + 1e-6 for _a, b in before_island))
        self.assertEqual(len(self.plan.slices), len(cuts) - 1)

    def test_open_turns_false_only_when_everything_is_joined_and_reaches_the_end(self):
        everyw = lambda src, a, b: [(a, b)]
        sp.extend_lazy_cuts(self.plan, "s", 2000.0, detect=everyw)           # the main chain grows to the limit
        self.assertTrue(self.plan.open)
        self.assertEqual(len(self.plan.lazy["islands"]), 1)
        sp.extend_lazy_cuts(self.plan, "s", None, detect=everyw, near=(0.0, 2500.0))
        self.assertTrue(self.plan.open)                                       # the island's front is past 'near'
        sp.extend_lazy_cuts(self.plan, "s", None, detect=everyw)
        self.assertFalse(self.plan.open)

    def test_stream_growth_window_leaves_far_fronts_alone_and_the_play_chain_goes_first(self):
        win = sp.stream_grow_window(2400.0)
        self.assertEqual(win, (1800.0, 3600.0))
        calls = []
        everyw = lambda src, a, b: (calls.append((a, b)) or [(a, b)])
        self.assertTrue(sp.extend_lazy_cuts(self.plan, "s", 3600.0, detect=everyw, near=win, focus=2400.0,
                                            max_cuts=3))
        self.assertEqual(self.plan.lazy["cuts"], [0.0])                       # front 0 is outside [1800, 3600]
        self.assertEqual(self.plan.lazy["islands"][0]["cuts"], [2392.0, 2422.0, 2722.0, 3022.0, 3322.0])
        self.assertTrue(all(a >= 2422.0 for a, _b in calls))

    def test_focus_picks_the_chain_at_the_play_position_first(self):
        plan = stream_plan(7200.0)
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=2)       # main front 330
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 5000.0), silent_at(4995.0, 5025.0))
        calls = []
        everyw = lambda src, a, b: (calls.append((a, b)) or [(a, b)])
        sp.extend_lazy_cuts(plan, "s", None, detect=everyw, focus=5000.0, max_cuts=1)
        self.assertGreaterEqual(calls[0][0], 5025.0)                         # the island, not the main chain
        calls.clear()
        sp.extend_lazy_cuts(plan, "s", None, detect=everyw, max_cuts=1)          # no focus: lowest front first
        self.assertLess(calls[0][1], 4000.0)


class SplitTests(unittest.TestCase):
    """Cases 7-11: a slice nothing has started on is split."""

    def plan(self, **kw):
        return uniform(3600.0, **kw)

    def test_a_pending_long_slice_is_split_around_the_position(self):
        plan = self.plan()
        starts = [s.start for s in plan.slices]
        n = len(plan.slices)
        request = sp.seek_cut_request(plan, 480.0)
        self.assertEqual((request.kind, request.slice_index, request.slice_range), ("split", 1, (300.0, 600.0)))
        self.assertEqual(request.window, (468.0, 525.0))
        self.assertTrue(sp.apply_seek_cuts(plan, request, silent_at(470.0, 500.0)))
        self.assertEqual(ranges(plan)[1], (1, 300.0, 470.0))                   # same index and start, shorter end
        self.assertEqual(ranges(plan)[n:], [(n, 470.0, 500.0), (n + 1, 500.0, 600.0)])
        self.assertEqual([s.start for s in plan.slices[:n]], starts)           # starts never change
        self.assertEqual([c for c in plan.lazy["cuts"] if 300 <= c <= 600], [300.0, 470.0, 500.0, 600.0])
        self.assertEqual([c for c in plan.lazy["cuts"]], sorted(plan.lazy["cuts"]))
        task = select_next_task(plan, SchedulerContext(play_pos=480.0, watched=True))
        self.assertEqual((task.slice_index, task.stage, task.urgent), (n, "asr", True))   # the short slice
        event = plan.seek_events[0]
        self.assertEqual((event["kind"], event["old_range"], event["slice_index"]), ("split", [300.0, 600.0], 1))
        self.assertEqual(event["new_slices"], [[1, 300.0, 470.0], [n, 470.0, 500.0], [n + 1, 500.0, 600.0]])

    def test_pop_seek_events_drains_the_log(self):
        plan = self.plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 480.0), silent_at(470.0, 500.0))
        self.assertEqual(len(sp.pop_seek_events(plan)), 1)
        self.assertEqual(sp.pop_seek_events(plan), [])
        self.assertNotIn("seek_events", plan.to_dict())

    def test_too_close_to_the_start_is_not_split_and_a_short_rest_gives_two_pieces(self):
        plan = self.plan()
        self.assertIsNone(sp.seek_cut_request(plan, 320.0))                   # 20 s into the slice
        self.assertIsNone(sp.seek_cut_request(plan, 330.0))                   # exactly 30 s: not more than 30
        self.assertIsNotNone(sp.seek_cut_request(plan, 331.0))
        n = len(plan.slices)
        request = sp.seek_cut_request(plan, 590.0)
        self.assertTrue(sp.apply_seek_cuts(plan, request, silent_at(586.0)))
        self.assertEqual(ranges(plan)[1], (1, 300.0, 586.0))
        self.assertEqual(ranges(plan)[n:], [(n, 586.0, 600.0)])                 # b - c = 14 <= 75: two pieces

    def test_split_cuts_keep_10_s_from_the_slice_ends(self):
        plan = self.plan()
        request = sp.seek_cut_request(plan, 599.0)
        self.assertEqual(request.window[0], 587.0)
        sp.apply_seek_cuts(plan, request, [])                                   # hard cut at the latest allowed
        self.assertEqual(plan.slices[1].end, 590.0)
        self.assertEqual(plan.slices[-1].start, 590.0)

    def test_started_or_finished_slices_are_not_split(self):
        for label, mark in (("running", lambda p: p.mark_running(1, "asr", "cloud")),
                            ("done", lambda p: p.mark_done(1, "asr", "m")),
                            ("failed", lambda p: p.mark_failed(1, "asr", "auth")),
                            ("only mt pending", lambda p: (p.mark_done(1, "asr", "m"))),
                            ("mt running", lambda p: (p.mark_done(1, "asr", "m"), p.mark_running(1, "mt", "x"))),
                            ("all done", lambda p: (p.mark_done(1, "asr", "m"), p.mark_done(1, "mt", "m")))):
            with self.subTest(label):
                plan = self.plan()
                mark(plan)
                before = plan.to_dict()
                self.assertIsNone(sp.seek_cut_request(plan, 480.0))
                self.assertEqual(plan.to_dict(), before)

    def test_waits_and_fallback_history_do_not_count_as_started(self):
        plan = self.plan()
        plan.slices[1].set("asr", sp.PRIMARY_TRACK, sp.TaskState(
            status=sp.PENDING, retry_at=5000.0, wait_executor="local", candidate="groq", attempt=1,
            tried={"groq": {"kind": "rate_limit", "at": 1.0}}, reclaim_at=6000.0, reclaim_executor="cloud"))
        n = len(plan.slices)
        request = sp.seek_cut_request(plan, 480.0)
        self.assertIsNotNone(request)
        self.assertTrue(sp.apply_seek_cuts(plan, request, silent_at(470.0, 500.0)))
        old = plan.slices[1].get("asr")
        for i in (1, n, n + 1):
            st = plan.slices[i].get("asr")
            self.assertEqual((st.status, st.retry_at, st.wait_executor, st.candidate, st.attempt, st.tried,
                              st.reclaim_at, st.reclaim_executor),
                             (sp.PENDING, 5000.0, "local", "groq", 1, {"groq": {"kind": "rate_limit", "at": 1.0}},
                              6000.0, "cloud"))
        self.assertIsNot(plan.slices[n].get("asr"), old)                        # copies, not shared
        self.assertIsNot(plan.slices[n].get("asr").tried, old.tried)
        # the waits still apply to the new slices: the cloud executor skips all three, the local one gets the short one
        t = select_next_task(plan, SchedulerContext(now=100.0, play_pos=480.0, watched=True), sp.Executor("cloud"))
        self.assertNotIn(t.slice_index, (1, n, n + 1))
        t = select_next_task(plan, SchedulerContext(now=5500.0, play_pos=480.0, watched=True), sp.Executor("local"))
        self.assertEqual((t.slice_index, t.reason), (n, "seek_urgent"))           # retry time passed, owner "local"

    def test_comparison_tracks_follow_the_primary(self):
        plan = self.plan(tracks=["primary", "cmp"])
        plan.slices[1].set("asr", "cmp", sp.TaskState(status=sp.PENDING, wait_executor="local"))
        n = len(plan.slices)
        self.assertTrue(sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 480.0), silent_at(470.0, 500.0)))
        for i in (1, n, n + 1):
            self.assertEqual(plan.slices[i].get("asr", "cmp").wait_executor, "local")
            self.assertEqual(plan.slices[i].get("asr", "primary").status, sp.PENDING)

    def test_a_started_track_blocks_the_split(self):
        for track, how in (("primary", "asr"), ("cmp", "asr")):
            plan = self.plan(tracks=["primary", "cmp"])
            plan.mark_running(1, how, "x", track)
            self.assertIsNone(sp.seek_cut_request(plan, 480.0), track)

    def test_a_position_in_a_finished_region_changes_nothing(self):
        plan = self.plan()
        for i in range(4):
            plan.mark_done(i, "asr", "m")
            plan.mark_done(i, "mt", "m")
        before = copy.deepcopy(plan.to_dict())
        self.assertIsNone(sp.seek_cut_request(plan, 700.0))
        self.assertEqual(plan.to_dict(), before)

    def test_island_start_plus_split_pieces_never_move_decided_cuts(self):
        plan = stream_plan()
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=4)
        decided = list(plan.lazy["cuts"])
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 500.0), silent_at(490.0, 520.0))
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 3000.0), silent_at(2995.0, 3025.0))
        now = sorted(set(plan.lazy["cuts"]) | {c for i in plan.lazy["islands"] for c in i["cuts"]})
        self.assertTrue(set(decided) <= set(now))                           # only insertions


class MillisecondBoundaryTests(unittest.TestCase):
    def test_offset_subtraction_does_not_create_another_island_at_its_persisted_start(self):
        # A probed 1.445 s offset with float noise lands just below the persisted millisecond cut.
        position = 2400.0 - 1.4450000000004
        self.assertLess(position, round(position, 3))  # actual audio-offset binary rounding edge
        plan = stream_plan()
        request, applied = seek(plan, position)
        self.assertTrue(applied)
        original = plan.slices[0]
        self.assertEqual(original.start, 2398.555)
        plan = sp.SlicePlan.from_dict(plan.to_dict())
        self.assertIs(sp.slice_containing(plan, position), plan.slices[original.index])
        before = copy.deepcopy(plan.to_dict())
        self.assertIsNone(sp.seek_cut_request(plan, position))
        self.assertEqual(plan.to_dict(), before)
        plan.mark_done(original.index, sp.STAGE_ASR, "a")
        task = select_next_task(plan, SchedulerContext(play_pos=position, watched=True))
        self.assertEqual((task.slice_index, task.stage), (original.index, sp.STAGE_MT))

    def test_offset_boundary_chooses_the_right_slice_in_either_creation_order(self):
        # A probed 1.445 s offset with float noise lands just below the persisted millisecond cut.
        position = 2400.0 - 1.4450000000004
        for right_first in (False, True):
            with self.subTest(right_first=right_first):
                plan = stream_plan()
                left = sp.Slice(0, 2368.555, 2398.555)
                right = sp.Slice(1, 2398.555, 2428.555)
                plan.slices = [right, left] if right_first else [left, right]
                for index, piece in enumerate(plan.slices):
                    piece.index = index
                for query in (position, 2398.555):
                    self.assertIs(sp.slice_containing(plan, query), right)
                    self.assertEqual(sp.find_slice_index_for_position(plan.slices, query), right.index)
                    task = select_next_task(plan, SchedulerContext(play_pos=query, watched=True))
                    self.assertEqual(task.slice_index, right.index)
                self.assertIsNone(sp.slice_containing(plan, right.end))


class ConflictTests(unittest.TestCase):
    """Case 14: the plan changed while the silences were being detected."""

    def test_the_slice_was_claimed_meanwhile(self):
        plan = uniform(3600.0)
        request = sp.seek_cut_request(plan, 480.0)
        plan.mark_running(1, "asr", "cloud")                                    # the executor took it
        before = copy.deepcopy(plan.to_dict())
        self.assertFalse(sp.apply_seek_cuts(plan, request, silent_at(470.0, 500.0)))
        self.assertEqual(plan.to_dict(), before)
        self.assertEqual(plan.seek_events, [])

    def test_the_slice_was_split_by_someone_else(self):
        plan = uniform(3600.0)
        request = sp.seek_cut_request(plan, 480.0)
        self.assertTrue(sp.apply_seek_cuts(plan, request, silent_at(470.0, 500.0)))
        before = copy.deepcopy(plan.to_dict())
        self.assertFalse(sp.apply_seek_cuts(plan, request, silent_at(470.0, 500.0)))
        self.assertEqual(plan.to_dict(), before)

    def test_the_gap_was_cut_meanwhile(self):
        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))
        before = copy.deepcopy(plan.to_dict())
        self.assertFalse(sp.apply_seek_cuts(plan, request, silent_at(2392.0, 2422.0)))
        self.assertEqual(plan.to_dict(), before)

    def test_the_kind_changed(self):
        plan = stream_plan()
        request = sp.seek_cut_request(plan, 2400.0)                         # an island request
        plan.slices.append(sp.Slice(index=0, start=2000.0, end=2600.0))     # someone cut a long slice around 2400
        plan.lazy["cuts"] = [0.0]
        plan.lazy["islands"] = [{"cuts": [2000.0, 2600.0], "closed": False}]
        self.assertEqual(sp.seek_cut_request(plan, 2400.0).kind, "split")
        before = copy.deepcopy(plan.to_dict())
        self.assertFalse(sp.apply_seek_cuts(plan, request, silent_at(2392.0, 2422.0)))
        self.assertEqual(plan.to_dict(), before)

    def test_a_request_for_another_position_or_window_is_not_applied_with_foreign_silences(self):
        plan = stream_plan(7200.0)
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=2)
        request = sp.seek_cut_request(plan, 345.0)
        # the left front moved after the request: the start window no longer starts at 333
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, max_cuts=1)
        self.assertFalse(sp.apply_seek_cuts(plan, request, silent_at(334.5, 360.0)))


class TimeOrderTests(unittest.TestCase):
    """Case 13: the slice list is not in time order."""

    def build(self):
        """main (0,30)(30,330) done; island (2392,2422) done; a third piece (2422,2722) pending (list order !=
        time order: the island slice was created before the main chain's second slice? create it so)."""
        plan = stream_plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))   # index 0
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, near=(2400.0, 2500.0), max_cuts=1)  # (2422, 2722) = 1
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, near=(0.0, 400.0), max_cuts=2)      # main: 2, 3
        return plan

    def test_plan_is_out_of_order(self):
        plan = self.build()
        self.assertEqual(ranges(plan), [(0, 2392.0, 2422.0), (1, 2422.0, 2722.0), (2, 0.0, 30.0), (3, 30.0, 330.0)])
        self.assertEqual([s.index for s in sp.slices_by_time(plan)], [2, 3, 0, 1])

    def test_find_slice_index_and_containing(self):
        plan = self.build()
        f = lambda pos: sp.find_slice_index_for_position(plan.slices, pos)
        self.assertEqual((f(10.0), f(100.0), f(2400.0), f(2500.0)), (2, 3, 0, 1))
        self.assertEqual(f(5000.0), 1)                                      # past the end: the last by time
        self.assertEqual(f(1000.0), 0)                                      # in the gap: the next slice
        self.assertIsNone(sp.slice_containing(plan, 1000.0))
        self.assertIsNone(sp.slice_containing(plan, 5000.0))
        self.assertEqual([s.index for s in sp.slices_from_position(plan, 2400.0)], [0, 1])
        self.assertEqual([s.index for s in sp.slices_from_position(plan, 100.0)], [3, 0, 1])

    def test_covered_end_walks_in_time_order_and_stops_at_a_gap(self):
        plan = self.build()
        for i in (0, 1, 2, 3):
            plan.mark_done(i, "asr", "m")
            plan.mark_done(i, "mt", "m")
        self.assertEqual(sp.covered_end(plan, 0.0), 330.0)                      # then the gap to 2392
        self.assertEqual(sp.covered_end(plan, 2400.0), 2722.0)
        self.assertEqual(sp.covered_end(plan, 1000.0), 1000.0)                  # in the gap: nothing ahead
        plan.slices[2].tasks.clear()
        self.assertEqual(sp.covered_end(plan, 0.0), 0.0)
        self.assertEqual(sp.covered_end(plan, 100.0), 330.0)

    def test_derive_status_uses_time_order_and_reports_a_pending_cut(self):
        plan = self.build()
        for i in (2, 3, 0):
            plan.mark_done(i, "asr", "m")
            plan.mark_done(i, "mt", "m")
        st = sp.derive_status(plan, play_pos=0.0)
        self.assertEqual((st.covered_s, st.covered_ahead_s), (330.0, 330.0))
        self.assertIsNone(st.blocking_slice)
        self.assertTrue(st.waiting_cut)                                         # the finished run ends in a gap
        st = sp.derive_status(plan, play_pos=1000.0)                           # in the gap
        self.assertTrue(st.waiting_cut)
        self.assertIsNone(st.blocking_slice)
        self.assertEqual(st.covered_ahead_s, 0.0)
        st = sp.derive_status(plan, play_pos=2400.0)                           # island done, next piece pending
        self.assertEqual((st.blocking_slice, st.blocking_stage, st.covered_s), (1, "asr", 2422.0))
        self.assertFalse(st.waiting_cut)
        self.assertEqual(st.slices_done, 3)
        self.assertFalse(st.complete)

    def test_derive_status_for_an_empty_plan_before_the_cut(self):
        plan = stream_plan()
        st = sp.derive_status(plan, play_pos=2400.0)
        self.assertTrue(st.waiting_download and st.waiting_cut)
        self.assertFalse(sp.derive_status(plan, play_pos=0.0).waiting_cut)

    def test_is_last_slice_is_about_the_end_not_the_list_position(self):
        plan = uniform(3000.0)
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 1700.0), silent_at(1690.0, 1720.0))
        self.assertEqual(plan.slices[-1].end, 1800.0)                           # the list's last slice is not the last
        self.assertFalse(sp.is_last_slice(plan, plan.slices[-1]))
        self.assertTrue(sp.is_last_slice(plan, plan.slices[9]))
        self.assertFalse(sp.is_last_slice(plan, plan.slices[0]))

    def test_head_slice_rules(self):
        # the head is the slice starting at 0, wherever it is in the list; it may not exist yet
        plan = self.build()
        ctx = SchedulerContext(play_pos=2400.0, watched=True)
        self.assertEqual(select_next_task(plan, ctx).slice_index, 0)             # seek slice
        plan.mark_done(0, "asr", "m")
        plan.mark_done(0, "mt", "m")
        # watching past the head: it does not jump the queue
        t = select_next_task(plan, ctx)
        self.assertEqual((t.slice_index, t.reason), (1, "horizon_buffer"))
        # not watching: the head (index 2) comes first
        t = select_next_task(plan, SchedulerContext(play_pos=0.0, watched=False))
        self.assertEqual((t.slice_index, t.reason, t.urgent), (2, "head_slice", True))
        # watching at its start: the head is also the seek slice
        t = select_next_task(plan, SchedulerContext(play_pos=10.0, watched=True))
        self.assertEqual((t.slice_index, t.reason), (2, "seek_urgent"))
        # only an island exists: no head, no error, the horizon rules apply
        island_only = stream_plan()
        sp.apply_seek_cuts(island_only, sp.seek_cut_request(island_only, 2400.0), silent_at(2392.0, 2422.0))
        t = select_next_task(island_only, SchedulerContext(play_pos=0.0, watched=False))
        self.assertIsNone(t)                                                   # beyond the 30 min horizon cap
        t = select_next_task(island_only, SchedulerContext(play_pos=0.0, watched=False, item_state="done"))
        self.assertEqual(t.slice_index, 0)

    def test_a_play_position_in_a_gap_picks_no_seek_slice(self):
        plan = self.build()
        t = select_next_task(plan, SchedulerContext(play_pos=1000.0, watched=True))
        self.assertEqual(t.reason, "horizon_buffer")                            # not "seek_urgent"
        # past the end of every slice: the old behaviour (the last slice)
        t = select_next_task(plan, SchedulerContext(play_pos=3000.0, watched=True))
        self.assertEqual((t.slice_index, t.reason), (1, "seek_urgent"))

    def test_secondary_track_walks_in_time_order(self):
        plan = stream_plan(tracks=["primary", "cmp"])
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))       # 0
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, near=(0.0, 10.0), max_cuts=1)           # (0,30) = 1
        for i in (0, 1):
            plan.mark_done(i, "asr", "m")
            plan.mark_done(i, "mt", "m")
        ctx = SchedulerContext(play_pos=0.0, watched=True, secondary_min_lead_s=0.0)
        t = select_next_task(plan, ctx)
        self.assertEqual((t.track, t.slice_index), ("cmp", 1))                  # from the play position's slice
        ctx = SchedulerContext(play_pos=2400.0, watched=True, secondary_min_lead_s=0.0)
        t = select_next_task(plan, ctx)
        self.assertEqual((t.track, t.slice_index), ("cmp", 0))

    def test_compose_orders_the_srt_by_time_and_reports_the_covered_ranges(self):
        plan = self.build()                                                    # list order: 0 island, 1, 2 head, 3
        row = lambda text, a, b: {"start": a, "end": b, "text": text, "text_ja": text, "zh": text}
        done = {0: [row("島A", 1.0, 3.0)], 2: [row("頭A", 1.0, 3.0)], 3: [row("頭B", 5.0, 8.0)]}
        partial = {1: [row("島B", 2.0, 5.0)]}
        composed = sr._compose(plan, done, partial)
        texts = [c["zh"] for c in composed["deduped"]]
        self.assertEqual(texts, ["頭A", "頭B", "島A", "島B"])
        self.assertEqual([c["start"] for c in composed["deduped"]], [1.0, 35.0, 2393.0, 2424.0])
        self.assertEqual(composed["covered_ranges"], [[0.0, 330.0], [2392.0, 2427.0]])
        self.assertEqual(composed["covered_s"], 330.0)                          # the continuous run from 0
        self.assertEqual(composed["done_indices"], [2, 3, 0])                    # time order
        srt = composed["concise_srt"]
        self.assertLess(srt.index("頭A"), srt.index("頭B"))
        self.assertLess(srt.index("頭B"), srt.index("島A"))
        self.assertLess(srt.index("島A"), srt.index("島B"))

    def test_compose_covered_s_with_a_partial_head_and_no_continuity(self):
        plan = self.build()
        row = lambda a, b: {"start": a, "end": b, "text": "x", "text_ja": "x", "zh": "x"}
        composed = sr._compose(plan, {0: [row(1.0, 3.0)]}, {2: [row(1.0, 9.0)]})
        self.assertEqual(composed["covered_s"], 9.0)                            # partial head slice, then stop
        self.assertEqual(composed["covered_ranges"], [[0.0, 9.0], [2392.0, 2422.0]])
        island_only = stream_plan()
        sp.apply_seek_cuts(island_only, sp.seek_cut_request(island_only, 2400.0), silent_at(2392.0, 2422.0))
        composed = sr._compose(island_only, {0: [row(1.0, 3.0)]}, {})
        self.assertEqual((composed["covered_s"], composed["covered_ranges"]), (0.0, [[2392.0, 2422.0]]))

    def test_publish_complete_needs_a_closed_plan_and_everything_done(self):
        plan = build_slice_plan(500.0)
        row = lambda text: {"start": 1.0, "end": 3.0, "text": text, "text_ja": text, "zh": text}
        done = {s.index: [row(f"s{s.index}")] for s in plan.slices}
        with tempfile.TemporaryDirectory() as tmp:
            result = sr.publish(tmp, plan, done)
            self.assertTrue(result["complete"])
            self.assertEqual(result["covered_ranges"], [[0.0, 500.0]])
            open_plan = stream_plan(500.0)
            sp.apply_seek_cuts(open_plan, sp.seek_cut_request(open_plan, 300.0), silent_at(295.0, 325.0))
            result = sr.publish(tmp, open_plan, {0: [row("a")]})
            self.assertFalse(result["complete"])
            self.assertEqual(result["covered_ranges"], [[295.0, 325.0]])
            self.assertEqual(result["covered_s"], 0.0)


class PersistenceTests(unittest.TestCase):
    """Case 15."""

    def test_islands_and_unordered_slices_survive_save_and_load(self):
        plan = stream_plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))
        sp.extend_lazy_cuts(plan, "s", None, detect=everywhere, near=(0.0, 400.0), max_cuts=2)
        plan.mark_done(0, "asr", "m")
        plan.mark_running(1, "asr", "cloud")
        with tempfile.TemporaryDirectory() as tmp:
            sp.save_slice_plan(plan, tmp)
            loaded = sp.load_slice_plan(tmp)
        self.assertEqual(loaded.to_dict(), plan.to_dict())
        self.assertEqual(sp.PLAN_VERSION, 2)
        self.assertEqual(loaded.to_dict()["version"], 2)
        self.assertEqual(ranges(loaded), ranges(plan))
        self.assertEqual(loaded.lazy["islands"], plan.lazy["islands"])
        self.assertTrue(loaded.open)
        self.assertEqual([s.index for s in loaded.slices], [0, 1, 2])
        self.assertEqual(ranges(loaded), [(0, 2392.0, 2422.0), (1, 0.0, 30.0), (2, 30.0, 330.0)])
        self.assertEqual(loaded.slices[0].get("asr").status, sp.DONE)
        # the loaded plan keeps working: the main chain still joins the island
        sp.extend_lazy_cuts(loaded, "s", None, detect=everywhere)
        self.assertFalse(loaded.open)
        self.assertEqual(loaded.lazy["islands"], [])

    def test_a_version_1_plan_loads(self):
        plan = build_slice_plan(1000.0, first_slice_seconds=90.0)
        data = plan.to_dict()
        data["version"] = 1
        data["lazy"].pop("islands", None)
        loaded = sp.SlicePlan.from_dict(data)
        self.assertEqual(ranges(loaded), ranges(plan))
        self.assertEqual(loaded.lazy["islands"], [])
        self.assertEqual(loaded.first_slice_seconds, 90.0)                      # the stored value is kept
        self.assertFalse(loaded.open)
        open_v1 = {"version": 1, "duration": 1000.0, "media_id": "m", "tracks": ["primary"], "slice_seconds": 300.0,
                   "first_slice_seconds": 90.0, "strategy": "smart_horizon", "horizon_minutes": 30,
                   "lazy": {"cuts": [0.0, 90.0], "closed": False, "failures": 0},
                   "slices": [{"index": 0, "start": 0.0, "end": 90.0, "tasks": {}}]}
        loaded = sp.SlicePlan.from_dict(open_v1)
        self.assertTrue(loaded.open)
        sp.extend_lazy_cuts(loaded, "s", None, detect=everywhere)
        self.assertFalse(loaded.open)
        self.assertEqual(loaded.slices[-1].end, 1000.0)
        # and a plan without a `lazy` field at all
        bare = sp.SlicePlan.from_dict({"duration": 100.0, "slices": []})
        self.assertEqual(bare.lazy["islands"], [])

    def test_a_loaded_plan_has_its_own_lists(self):
        plan = stream_plan()
        sp.apply_seek_cuts(plan, sp.seek_cut_request(plan, 2400.0), silent_at(2392.0, 2422.0))
        loaded = sp.SlicePlan.from_dict(plan.to_dict())
        loaded.lazy["islands"][0]["cuts"].append(9999.0)
        self.assertEqual(plan.lazy["islands"][0]["cuts"], [2392.0, 2422.0])


if __name__ == "__main__":
    unittest.main()
