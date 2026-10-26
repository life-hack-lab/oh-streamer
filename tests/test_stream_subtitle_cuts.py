"""T3.1-B: cut limit and one cut per round for stream items (remote source: every silence search downloads
audio). Executor tests use fake providers and a counting fake silence detector; no network, no GPU."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading
import unittest

from oh_streamer.queue import subtitle_host as sh
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import slice_plan as sp
from tests import test_queue_subtitle_host as host_tests
from tests import test_subtitle_scheduler_cloud as cloud_tests


class StreamHost(cloud_tests.FakeHost):
    """Remote source (limit None) whose cut limit follows a play position, like QueueSubtitleHost for a stream."""

    def __init__(self, play_pos=0.0, horizon=600.0, max_chunk=600.0):
        super().__init__(total=3600.0, limit=None)
        self.play_pos = play_pos
        self.horizon, self.max_chunk = horizon, max_chunk
        self.limits_asked = []

    def cut_limit(self, limit):
        self.limits_asked.append(limit)
        return self.play_pos + self.horizon + self.max_chunk

    def scheduler_context(self):
        return sp.SchedulerContext(item_state="streaming", play_pos=self.play_pos, restore_pos=self.play_pos,
                                   watched=True)


class CuttingBase(cloud_tests.Base):
    def setUp(self):
        super().setUp()
        self.detect_calls = []
        self.events = []
        self.asr_started = threading.Event()
        self.second_detect_saw_asr = None
        self.gate_second_detect = False
        outer = self

        def asr(piece, s):
            outer.events.append(("asr", s.index))
            outer.asr_started.set()
            return cloud_tests.FakeProviders.asr(outer.fake, piece, s)

        self.fake.asr = asr

    def detect(self, source, start, end):
        self.detect_calls.append((start, end))
        self.events.append(("detect", start))
        if self.gate_second_detect and len(self.detect_calls) == 2:
            # the first slice must already be on its way to ASR before the second silence search starts
            self.second_detect_saw_asr = self.asr_started.wait(5.0)
        return cloud_tests.every_30s_silence(source, start, end)

    def deps(self, **over):
        over.setdefault("probe_duration", lambda s: 3600.0)
        over.setdefault("detect_silences", self.detect)
        return super().deps(**over)


class CutLimitExecutorTests(CuttingBase):
    def test_default_host_cut_limit_is_the_limit(self):
        host = ss.CloudHost()
        self.assertIsNone(host.cut_limit(None))
        self.assertEqual(host.cut_limit(123.0), 123.0)

    def test_normal_host_still_decides_every_cut_in_one_call(self):
        self.host = cloud_tests.FakeHost(total=3600.0)
        self.host.folder = self.folder
        ex = self.executor(host=self.host)
        ex._prepare()
        try:
            self.assertTrue(ex._advance_cuts(None))
            self.assertFalse(ex._cuts_paced)
            self.assertFalse(ex.plan.open)
            self.assertGreater(len(ex.plan.slices), 10)
            self.assertEqual(len(self.detect_calls), len(ex.plan.slices) - 1)   # the closing cut needs no search
        finally:
            ex._release_store()

    def test_first_round_searches_one_cut_only(self):
        self.host = StreamHost(play_pos=0.0)
        self.host.folder = self.folder
        ex = self.executor(host=self.host)
        ex._prepare()
        try:
            self.assertTrue(ex._advance_cuts(None))
            self.assertTrue(ex._cuts_paced)
            self.assertEqual(len(self.detect_calls), 1)
            self.assertEqual(len(ex.plan.slices), 1)
            self.assertTrue(ex.plan.open)
        finally:
            ex._release_store()

    def test_cut_limit_stops_cuts_and_first_asr_precedes_the_second_search(self):
        self.gate_second_detect = True
        self.host = StreamHost(play_pos=0.0)
        self.host.folder = self.folder
        ex, res = self.run_ex()
        self.assertTrue(self.second_detect_saw_asr)
        self.assertEqual(self.events[0], ("detect", self.detect_calls[0][0]))
        self.assertIn(("asr", 0), self.events[:3])
        # cut limit = 0 + 600 + 600 = 1200 s: cuts 30, 330, 630, 930 (the next window ends at 1260 > 1200)
        # (T3.6: the first slice is 30 s, not 90 s)
        self.assertEqual([s.end for s in ex.plan.slices], [30.0, 330.0, 630.0, 930.0])
        self.assertEqual(len(self.detect_calls), 4)
        self.assertTrue(ex.plan.open)
        self.assertEqual(res.reason, "download")        # plan.open for the whole show: waits like a download
        self.assertTrue(all(limit is None for limit in self.host.limits_asked))   # limit itself stays None

    def test_cuts_follow_the_play_position(self):
        self.host = StreamHost(play_pos=0.0)
        self.host.folder = self.folder
        ex, _ = self.run_ex()
        before = [s.end for s in ex.plan.slices]
        self.host.play_pos = 600.0                      # cut limit 1800 s
        ex2, _ = self.run_ex()
        after = [s.end for s in ex2.plan.slices]
        self.assertEqual(after[:len(before)], before)   # decided cuts never change
        self.assertEqual(after[len(before):], [1230.0, 1530.0])      # T3.6: the cuts sit 60 s earlier (30 s first slice)
        self.assertLessEqual(max(s.start for s in ex2.plan.slices), 600.0 + 600.0 + 600.0)
        self.host.play_pos = 3600.0                     # past the end: the rest is cut and the plan closes
        ex3, _ = self.run_ex()
        # T3.6 grows only the nearby chain; jumping past its frontier leaves the old gap alone.
        self.assertTrue(ex3.plan.open)
        self.assertEqual([s.end for s in ex3.plan.slices], after)

    def test_cut_limit_never_exceeds_the_readable_limit(self):
        self.host = StreamHost(play_pos=3000.0)         # asks for a lot, but only 700 s are readable
        self.host.folder = self.folder
        ex = self.executor(host=self.host)
        ex._prepare()
        try:
            for _ in range(10):
                ex._advance_cuts(700.0)
            self.assertEqual(ex.plan.slices, [])
            self.assertEqual(self.detect_calls, [])
        finally:
            ex._release_store()

    def test_max_available_stays_the_real_limit(self):
        self.host = StreamHost(play_pos=0.0)
        self.host.folder = self.folder
        seen = []
        orig = sp.select_next_task

        def spy(plan, ctx, *a, **kw):
            seen.append(ctx.max_available)
            return orig(plan, ctx, *a, **kw)

        sp.select_next_task = spy
        try:
            ex, _ = self.run_ex()
        finally:
            sp.select_next_task = orig
        self.assertTrue(seen)
        self.assertTrue(all(value is None for value in seen))
        # slices past cut_limit are simply not decided yet; no decided slice is "unreadable"
        self.assertTrue(sp.derive_status(ex.plan, 0.0, 0.0, None).open)


class QueueHostCutLimitTests(host_tests.Base):
    def stream_item(self, item_id="st1", **kwargs):
        item = self.add(item_id, state="streaming", **kwargs)
        item.kind = "stream"
        item.source_status = ""
        return item

    def test_stream_item_cut_limit_is_play_pos_plus_horizon_plus_chunk(self):
        self.stream_item(audio_offset=0.0)
        self.svc._subtitle_clocks["st1"] = (0.0, True)
        host = self.host("st1")
        self.assertEqual(host.cut_limit(None), 1200.0)
        self.svc._subtitle_clocks["st1"] = (2400.0, True)
        self.assertEqual(host.cut_limit(None), 2400.0 + sh.STREAM_HORIZON_S + sh.STREAM_MAX_CHUNK_S)

    def test_stream_item_uses_the_audio_offset(self):
        self.stream_item(audio_offset=7.5)
        self.svc._subtitle_clocks["st1"] = (100.0, True)
        self.assertEqual(self.host("st1").cut_limit(None), 92.5 + 1200.0)
        self.svc._subtitle_clocks["st1"] = (3.0, True)               # before the audio starts: play_pos 0
        self.assertEqual(self.host("st1").cut_limit(None), 1200.0)

    def test_regular_items_keep_their_limit(self):
        self.add("it1")
        self.svc._subtitle_clocks["it1"] = (500.0, True)
        host = self.host("it1")
        self.assertIsNone(host.cut_limit(None))
        self.assertEqual(host.cut_limit(321.0), 321.0)

    def test_missing_item_keeps_the_limit(self):
        self.assertEqual(self.host("nope").cut_limit(55.0), 55.0)

    def test_stream_max_available_is_not_cut_limit(self):
        self.stream_item(audio_offset=0.0)
        self.svc._subtitle_clocks["st1"] = (100.0, True)
        host = self.host("st1")
        self.assertEqual(host.source_state(), (True, None, None))
        ctx = host.scheduler_context()
        self.assertIsNone(ctx.max_available)
        self.assertEqual(ctx.play_pos, 100.0)


if __name__ == "__main__":
    unittest.main()
