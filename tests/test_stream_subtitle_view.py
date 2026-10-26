"""T3.1-C: stream items get item.subtitle["view"] like queue items, and a stream item whose main model is local shows
"waiting_gpu" while the player is busy (the local executor does not run for stream items). Run from streamer/:
CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_subtitle_view."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import unittest

try:
    from tests.test_queue_service import ServiceTestBase
    from tests.test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen
except ImportError:
    from test_queue_service import ServiceTestBase
    from test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen

from oh_streamer.subtitles import slice_plan as sp


class StreamSubtitleViewTests(ServiceTestBase):
    def setUp(self):
        super().setUp()
        # These fixtures drive view/status ticks explicitly. pause_all only pauses background jobs;
        # it does not stop the scheduler from concurrently refreshing/clearing the same views.
        self.svc._stopping.set()
        self.svc._wake.set()
        self.svc._thread.join(5)
        self.assertFalse(self.svc._thread.is_alive())
        self.svc._stopping.clear()
        self.svc.pause_all()
        self.svc.store.set_settings(subtitle_runtime_contract="legacy")
        self.svc._subtitle_clock_fn = lambda: self.now[0]
        self.stream_id = self.make_stream(CLOUD_ASR, CLOUD_MT)

    def make_stream(self, asr, mt):
        item = self.svc.store.add_stream("https://x.test/v", "url", {}, {"subtitles": True}, title="Clip")
        settings = {**item.settings, "subtitles": True, "subtitle_runtime_contract": "cloud-first-1",
                    "subtitle_resolved": frozen(asr, mt), "subtitle_configs": [{}]}
        self.svc.store.update(item.id, settings=settings)
        return item.id

    def save_plan(self, item_id, finished=(), duration=900.0):
        plan = sp.build_slice_plan(duration, first_slice_seconds=None)          # 3 slices of 300 s
        for i in finished:
            plan.mark_done(i, "asr", "a")
            plan.mark_done(i, "mt", "m")
        plan.lazy["closed"] = True
        folder = self.svc.store.item_dir(item_id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        sp.save_slice_plan(plan, folder)
        return plan

    def subtitle(self, item_id=None):
        return self.svc.store.get(item_id or self.stream_id).subtitle

    def make_busy(self, busy=True):
        with self.svc._lock:
            self.svc._busy = busy
            self.svc._busy_ts = self.svc._clock()

    # --- the view exists ------------------------------------------------------------------------------------------
    def test_a_stream_item_gets_a_view_and_no_table_means_none(self):
        self.svc._subtitle_view_tick()
        self.assertNotIn("view", self.subtitle())                     # no slice table yet
        self.save_plan(self.stream_id, [0])
        self.svc._subtitle_view_tick()
        view = self.subtitle()["view"]
        self.assertEqual((view["v"], view["slices_done"], view["slices_total"]), (2, 1, 3))
        self.assertNotIn(view["state"], ("complete", "waiting_gpu"))

    def test_the_view_follows_the_table_and_the_play_clock(self):
        self.save_plan(self.stream_id, [0, 1, 2])
        self.svc._subtitle_view_tick()
        self.assertEqual(self.subtitle()["view"]["state"], "complete")
        self.save_plan(self.stream_id, [0])
        self.svc._subtitle_clocks[self.stream_id] = (10.0, True)
        self.svc._subtitle_view_tick()
        self.assertEqual(self.subtitle()["view"]["slices_done"], 1)
        self.assertEqual(self.subtitle()["view"]["covered_s"], 300.0)

    def test_a_stream_item_without_subtitles_gets_no_view(self):
        plain = self.svc.store.add_stream("https://x.test/w", "url", {}, {}, title="Plain")
        self.save_plan(plain.id, [0])
        self.svc._subtitle_view_tick()
        self.assertNotIn("view", self.subtitle(plain.id))

    def test_a_tombstoned_stream_item_gets_no_view(self):
        self.save_plan(self.stream_id, [0])
        self.svc._tombstones.add(self.stream_id)
        self.svc._subtitle_view_tick()
        self.assertNotIn("view", self.subtitle())

    def test_the_view_tick_still_covers_queue_items(self):
        queue_id = self.ready(state="queued")
        self.svc.store.update(queue_id, settings={**self.svc.store.get(queue_id).settings, "subtitles": True,
                                                  "subtitle_resolved": frozen(CLOUD_ASR, CLOUD_MT)})
        self.save_plan(queue_id, [0])
        self.save_plan(self.stream_id, [0, 1])
        self.svc._subtitle_view_tick()
        self.assertEqual(self.subtitle(queue_id)["view"]["slices_done"], 1)
        self.assertEqual(self.subtitle(self.stream_id)["view"]["slices_done"], 2)

    # --- waiting for the GPU --------------------------------------------------------------------------------------
    def test_gpu_turn_of_a_stream_item_follows_the_player_busy_flag(self):
        item = self.svc.store.get(self.stream_id)
        self.assertTrue(self.svc._subtitle_gpu_turn(item))
        self.make_busy()
        self.assertTrue(self.svc.is_player_busy())
        self.assertFalse(self.svc._subtitle_gpu_turn(item))
        self.make_busy(False)
        self.assertTrue(self.svc._subtitle_gpu_turn(item))

    def test_a_local_main_model_with_a_busy_player_waits_for_the_gpu(self):
        local = self.make_stream(LOCAL_ASR, LOCAL_MT)
        self.save_plan(local, [])
        self.make_busy()
        self.svc._subtitle_view_tick()
        view = self.subtitle(local)["view"]
        self.assertEqual(view["state"], "waiting_gpu")
        self.assertEqual(view["wait"]["kind"], "gpu")

    def test_a_local_main_model_with_an_idle_player_is_not_waiting_for_the_gpu(self):
        local = self.make_stream(LOCAL_ASR, LOCAL_MT)
        self.save_plan(local, [])
        self.svc._subtitle_view_tick()
        self.assertNotEqual(self.subtitle(local)["view"]["state"], "waiting_gpu")

    def test_a_cloud_main_model_never_waits_for_the_gpu_even_with_a_busy_player(self):
        self.save_plan(self.stream_id, [])
        self.make_busy()
        self.svc._subtitle_view_tick()
        self.assertNotEqual(self.subtitle()["view"]["state"], "waiting_gpu")

    def test_the_busy_rule_does_not_change_a_queue_item(self):
        queue_id = self.ready(state="queued")
        item = self.svc.store.get(queue_id)
        self.make_busy()
        self.assertTrue(self.svc._subtitle_gpu_turn(item))            # no restoration worker running: GPU turn


if __name__ == "__main__":
    unittest.main()
