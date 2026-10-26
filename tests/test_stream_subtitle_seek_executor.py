"""P3 seek integration: remote IO locality, debounce and stale-clock races."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from oh_streamer.queue.queue_service import QueueService
from oh_streamer.queue.subtitle_host import QueueSubtitleHost
from oh_streamer.subtitles import slice_plan as sp
from tests.test_stream_subtitle_cuts import CuttingBase, StreamHost
from tests.test_subtitle_scheduler_cloud import FakeHost
from tests import test_queue_subtitle_host as host_tests


class SeekExecutorTests(CuttingBase):
    def make_seek_executor(self):
        self.host = StreamHost(play_pos=2400)
        self.host.seek_position = lambda: self.host.play_pos
        self.host.folder = self.folder
        ex = self.executor(host=self.host)
        ex._prepare()
        self.addCleanup(ex._release_store)
        return ex

    def assert_drag_dispatches_only_the_final_short_slice(self, ex, positions):
        item = SimpleNamespace(id="s", subtitle={})
        service = SimpleNamespace(
            _clock=self.clock, _subtitle_wants={}, _subtitle_clocks={},
            _watched_ids=lambda: {"s"}, store=SimpleNamespace(get=lambda _: item))
        host = QueueSubtitleHost(service, "s", probe_offset=False)
        self.host.seek_position, self.host.seek_pending = host.seek_position, host.seek_pending
        ex.concurrency = 1
        pool, inflight = mock.Mock(), {}
        for pos in positions:
            self.host.play_pos = pos
            service._subtitle_wants["s"] = (pos, self.clock())
            service._subtitle_clocks["s"] = (pos, False)
            self.assertFalse(ex._advance_cuts(None))
            ex._launch(pool, inflight, None)
            self.assertFalse(any(st.status == sp.RUNNING for s in ex.plan.slices for st in s.tasks.values()))
            self.clock.sleep(0.2)
        pool.submit.assert_not_called()
        self.assertEqual(self.detect_calls, [])
        self.clock.sleep(1.31)
        self.assertTrue(ex._advance_cuts(None))
        short = sp.slice_containing(ex.plan, positions[-1])
        self.assertLessEqual(short.end - short.start, 57)
        ex._launch(pool, inflight, None)
        pool.submit.assert_called_once()
        claimed = pool.submit.call_args.args[1]
        self.assertEqual((claimed.slice_index, claimed.stage), (short.index, sp.STAGE_ASR))
        self.assertEqual(short.get(sp.STAGE_ASR).status, sp.RUNNING)

    def test_paused_drag_into_empty_table_waits_before_cutting_or_dispatching(self):
        ex = self.make_seek_executor()
        self.assert_drag_dispatches_only_the_final_short_slice(ex, (300, 600, 900, 1200, 2400))

    def test_paused_drag_does_not_claim_an_unsplit_long_slice(self):
        ex = self.make_seek_executor()
        closed = sp.build_slice_plan(3600, first_slice_seconds=None)
        ex.plan.slices, ex.plan.lazy = closed.slices, closed.lazy
        self.assert_drag_dispatches_only_the_final_short_slice(ex, (350, 650, 950, 1250, 2450))

    def test_seek_fetches_only_target_window_then_dispatches_short_slice(self):
        ex = self.make_seek_executor()
        with mock.patch('oh_streamer.subtitles.usage_log.event') as event:
            self.assertTrue(ex._advance_cuts(None))
        self.assertEqual(len(self.detect_calls), 1)
        self.assertGreaterEqual(self.detect_calls[0][0], 2388)
        self.assertLessEqual(self.detect_calls[0][1], 2445)
        short = sp.slice_containing(ex.plan, 2400)
        self.assertIsNotNone(short)
        self.assertLessEqual(short.end - short.start, 57)
        task = sp.select_next_task(ex.plan, self.host.scheduler_context(), sp.Executor("cloud"))
        self.assertEqual(task.slice_index, short.index)
        self.assertEqual(task.stage, sp.STAGE_ASR)
        event.assert_called_once()
        self.assertEqual(event.call_args.args[1], 'slice_split')
        self.assertFalse(ex._advance_cuts(None))
        self.assertEqual(len(self.detect_calls), 1)
        self.assertTrue(all(start >= 2388 for start, _ in self.detect_calls))

    def test_queue_seek_translation_finishes_before_unpaced_silence_growth(self):
        # Ordinary downloaded queue sources have no cut cap. Deciding the rest of a long
        # HLS source under the plan lock must not hold up the urgent short slice's MT.
        self.host = FakeHost(total=3600, limit=None)
        self.host.context = sp.SchedulerContext(play_pos=2400, watched=True, item_state="queued")
        self.host.seek_position = lambda: 2400
        self.host.folder = self.folder
        self.assertIsNone(self.host.cut_limit(None))
        ex = self.executor(host=self.host)
        self.assertTrue(ex._prepare())
        self.addCleanup(ex._release_store)
        self.assertTrue(ex._advance_cuts(None))
        short = sp.slice_containing(ex.plan, 2400)
        self.assertLessEqual(short.end - short.start, 75)
        self.assertEqual(len(self.detect_calls), 1)
        ex.plan.mark_done(short.index, sp.STAGE_ASR, "asr-test")
        self.assertFalse(ex._advance_cuts(None))
        self.assertEqual(len(self.detect_calls), 1)
        task = sp.select_next_task(ex.plan, self.host.scheduler_context(), sp.Executor("cloud"))
        self.assertEqual((task.slice_index, task.stage), (short.index, sp.STAGE_MT))
        ex.plan.mark_done(short.index, sp.STAGE_MT, "mt-test")
        self.assertTrue(ex._advance_cuts(None))
        self.assertGreater(len(self.detect_calls), 1)
        self.assertFalse(ex._cuts_paced)
        self.assertFalse(ex.plan.open)

    def test_audio_offset_float_target_keeps_the_original_short_slice_for_translation(self):
        # A probed 1.445 s offset with float noise lands just below the persisted millisecond cut.
        position = 2400.0 - 1.4450000000004
        self.host = FakeHost(total=3600, limit=None)
        self.host.context = sp.SchedulerContext(play_pos=position, watched=True, item_state="queued")
        self.host.seek_position = lambda: position
        self.host.folder = self.folder
        calls = []
        detect = lambda source, start, end: (calls.append((start, end)) or [])
        ex = self.executor(host=self.host, deps=self.deps(detect_silences=detect))
        self.assertTrue(ex._prepare())
        self.addCleanup(ex._release_store)
        self.assertTrue(ex._advance_cuts(None))
        original = ex.plan.slices[0]
        self.assertEqual(original.start, 2398.555)
        self.assertIsNone(sp.seek_cut_request(ex.plan, position))
        ex.plan.mark_done(original.index, sp.STAGE_ASR, "asr-test")
        self.assertFalse(ex._advance_cuts(None))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(ex.plan.lazy["islands"]), 1)
        self.assertEqual(len(ex.plan.slices), 1)
        task = sp.select_next_task(ex.plan, self.host.scheduler_context(), sp.Executor("cloud"))
        self.assertEqual((task.slice_index, task.stage), (original.index, sp.STAGE_MT))

    def test_queue_background_cut_yields_to_a_watch_that_arrives_during_detection(self):
        service = host_tests.FakeService(self.folder.parent / "queue-items")
        service._clock = self.clock
        item = service.store.add(host_tests.make_item(source=str(self.source)))
        host = QueueSubtitleHost(service, item.id, probe_offset=False)
        self.assertIsNone(host.cut_limit(None))
        self.assertTrue(host.pace_cuts())
        self.assertIsNone(host.seek_position())

        def detect(source, start, end):
            result = self.detect(source, start, end)
            if len(self.detect_calls) == 1:
                # Public play / the player's first clock arrives during the background silence IO.
                service._watched.add(item.id)
                service._subtitle_clocks[item.id] = (2400.0, True)
            return result

        ex = self.executor(host=host, deps=self.deps(detect_silences=detect))
        self.assertTrue(ex._prepare())
        self.addCleanup(ex._release_store)
        self.assertTrue(ex._advance_cuts(None))
        self.assertTrue(ex._cuts_paced)
        self.assertEqual(len(self.detect_calls), 1)
        self.assertEqual([(s.start, s.end) for s in ex.plan.slices], [(0, 30)])
        self.assertTrue(ex.plan.open)
        self.assertTrue(ex._advance_cuts(None))
        self.assertEqual(len(self.detect_calls), 2)
        self.assertGreaterEqual(self.detect_calls[-1][0], 2388)
        self.assertLessEqual(self.detect_calls[-1][1], 2445)
        short = sp.slice_containing(ex.plan, 2400)
        self.assertIsNotNone(short)
        self.assertLessEqual(short.end - short.start, 75)
        task = sp.select_next_task(ex.plan, host.scheduler_context(), sp.Executor("cloud"))
        self.assertEqual((task.slice_index, task.stage), (short.index, sp.STAGE_ASR))

    def test_detection_runs_without_plan_lock_and_rechecks_latest_target(self):
        ex = self.make_seek_executor()
        def detect(source, start, end):
            acquired = []
            def other_thread():
                with ex._lock:
                    acquired.append(True)
            t = threading.Thread(target=other_thread)
            t.start()
            t.join(1)
            self.assertEqual(acquired, [True])
            self.host.play_pos = 3000
            return []
        ex.deps.detect_silences = detect
        ex._advance_cuts(None)
        self.assertIsNone(sp.slice_containing(ex.plan, 2400))

    def test_new_paused_seek_during_detection_does_not_fall_back_to_normal_cutting(self):
        ex = self.make_seek_executor()
        pending = False
        calls = []
        self.host.seek_pending = lambda: pending
        self.host.seek_position = lambda: None if pending else self.host.play_pos

        def detect(source, start, end):
            nonlocal pending
            calls.append((start, end))
            if len(calls) == 1:
                # The new target's window includes the main frontier at zero. Falling through
                # to ordinary growth would fetch a second window and cut the head while dragging.
                self.host.play_pos = 300
                pending = True
            return []

        ex.deps.detect_silences = detect
        self.assertFalse(ex._advance_cuts(None))
        self.assertEqual(len(calls), 1)
        self.assertEqual(ex.plan.slices, [])
        pending = False
        self.assertTrue(ex._advance_cuts(None))
        self.assertEqual(len(calls), 2)
        self.assertEqual((ex.plan.slices[0].start, ex.plan.slices[0].end), (300, 330))
        self.assertGreaterEqual(calls[-1][0], 288)

    def test_unreadable_seek_does_not_fetch_target_audio(self):
        ex = self.make_seek_executor()
        ex._advance_cuts(700)
        self.assertEqual(self.detect_calls, [])
        self.assertEqual(ex.plan.slices, [])


class DebounceTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.item = SimpleNamespace(id='s', subtitle={'audio_offset': 2})
        self.service = SimpleNamespace(
            _clock=lambda: self.now, _subtitle_wants={}, _subtitle_clocks={}, _subtitle_seek_guard={},
            _watched_ids=lambda: {'s'}, store=SimpleNamespace(get=lambda _: self.item))
        self.host = QueueSubtitleHost(self.service, 's', probe_offset=False)

    def test_five_drag_targets_only_last_is_eligible_after_1_5_seconds(self):
        for pos in (300, 600, 900, 1200, 2400):
            self.service._subtitle_wants['s'] = (pos, self.now)
            self.service._subtitle_clocks['s'] = (pos, False)
            self.assertIsNone(self.host.seek_position())
            self.now += 0.2
        self.now += 1.29
        self.assertIsNone(self.host.seek_position())
        self.now += 0.02
        self.assertEqual(self.host.seek_position(), 2398)

    def test_playing_accepts_target_immediately_and_stale_clock_is_ignored(self):
        svc = self.service
        svc._subtitle_wants['s'] = (2400, self.now)
        svc._subtitle_seek_guard['s'] = (2400, self.now)
        svc._subtitle_clocks['s'] = (2400, False)
        self.assertFalse(QueueService._subtitle_accept_clock(svc, 's', 30))
        self.assertTrue(QueueService._subtitle_accept_clock(svc, 's', 2401))
        svc._subtitle_clocks['s'] = (2401, True)
        self.assertEqual(self.host.seek_position(), 2399)
        self.now += 2.01
        self.assertTrue(QueueService._subtitle_accept_clock(svc, 's', 30))

    def test_repeated_clock_only_native_seeks_keep_the_latest_target(self):
        svc = self.service
        svc._subtitle_clocks['s'] = (0, False)
        for pos in (300, 600, 900, 1200, 2400):
            self.assertTrue(QueueService._subtitle_accept_clock(svc, 's', pos))
            svc._subtitle_clocks['s'] = (pos, False)
            self.assertIsNone(self.host.seek_position())
            self.now += 0.2
        self.assertEqual(svc._subtitle_wants['s'][0], 2400)
        self.now += 1.29
        self.assertIsNone(self.host.seek_position())
        self.now += 0.02
        self.assertEqual(self.host.seek_position(), 2398)
        self.assertEqual(svc._subtitle_seek_guard, {})

    def test_clock_only_jump_is_debounced(self):
        svc = self.service
        svc._subtitle_clocks['s'] = (0, False)
        self.assertTrue(QueueService._subtitle_accept_clock(svc, 's', 2400))
        svc._subtitle_clocks['s'] = (2400, False)
        self.assertIsNone(self.host.seek_position())
        self.now += 1.5
        self.assertEqual(self.host.seek_position(), 2398)



class ProgressiveWantTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.item = SimpleNamespace(id="q", kind="file", source_path="movie.mp4", state="queued", error="")
        self.svc = SimpleNamespace(
            _lock=threading.RLock(), _watch={}, _subtitle_wants={}, _subtitle_clocks={}, _subtitle_seek_guard={},
            _clock=lambda: self.now, _paused_all=False, _wk=None,
            store=SimpleNamespace(get=lambda _: self.item, mark_seen=mock.Mock()),
            _skip_subtitles_for_play=mock.Mock(), _source_ready=lambda _: True,
            _write_play_clock=mock.Mock(), _watched_ids=lambda: {"q"},
            goto=mock.Mock(), _changed=mock.Mock(), _wake=threading.Event(),
            _subtitle_has_table=lambda _: True)
        self.svc._subtitle_accept_clock = lambda item_id, position: QueueService._subtitle_accept_clock(
            self.svc, item_id, position)

    def want(self, **fields):
        return QueueService._ipc_want(self.svc, {"id": "q", "k": 120, **fields})

    def test_foreground_want_debounces_the_actual_target(self):
        self.assertTrue(self.want(pos=2400, foreground=True)["ok"])
        self.assertEqual(self.svc._subtitle_wants["q"], (2400, self.now))
        self.assertEqual(self.svc._subtitle_clocks["q"], (2400, False))
        QueueService._ipc_clock(self.svc, {"id": "q", "pos": 30, "playing": True})
        self.assertEqual(self.svc._subtitle_clocks["q"], (2400, False))
        QueueService._ipc_clock(self.svc, {"id": "q", "pos": 2401, "playing": True})
        self.assertEqual(self.svc._subtitle_clocks["q"], (2401, True))

    def test_lookahead_does_not_restart_the_paused_seek_debounce(self):
        self.want(pos=2400, foreground=True)
        self.now += 0.8
        self.want(pos=2400)  # feeder waiting_k repeat, foreground=False
        self.assertEqual(self.svc._subtitle_wants["q"], (2400, 100.0))
        self.now += 0.8
        host = QueueSubtitleHost(self.svc, "q", probe_offset=False)
        self.item.subtitle = {}
        self.assertEqual(host.seek_position(), 2400)

    def test_stale_lookahead_cannot_replace_a_new_foreground_seek(self):
        self.want(pos=2400, foreground=True)
        self.now += 0.2
        self.want(pos=30)  # a queued feeder message carries its old last_position
        self.assertEqual(self.svc._subtitle_clocks["q"], (2400, False))
        self.assertEqual(self.svc._subtitle_wants["q"], (2400, 100.0))
        self.svc.goto.assert_called_with("q", 120)  # repair lookahead still works

    def test_foreground_without_position_uses_the_requested_segment(self):
        from oh_streamer.queue.queue_service import SEGMENT_SECONDS
        self.want(foreground=True)
        self.assertEqual(self.svc._subtitle_wants["q"], (120 * SEGMENT_SECONDS, self.now))
        self.assertEqual(self.svc._subtitle_clocks["q"], (120 * SEGMENT_SECONDS, False))


if __name__ == '__main__':
    unittest.main()
