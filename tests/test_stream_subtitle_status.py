"""T3.3: the subtitle job status reaches the player (queue `subtitle_status` message), the player validates it and shows
it in the 進階資訊 panel (字幕 section) and, for a failure, as a one-time OSD. Run from streamer/:
CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_stream_subtitle_status."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import dataclasses
import json
import unittest
from unittest import mock
from types import SimpleNamespace

try:                                    # (modules, not their test classes: discovery must not run those again here)
    from tests import test_stream_subtitle_link as link_tests
    from tests import test_stream_subtitle_view as view_tests
    from tests.test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen
except ImportError:
    import test_stream_subtitle_link as link_tests
    import test_stream_subtitle_view as view_tests
    from test_queue_subtitle_host import CLOUD_ASR, CLOUD_MT, LOCAL_ASR, LOCAL_MT, frozen

from oh_streamer.player import info_panel
from oh_streamer.player import subtitle_status as ss
from oh_streamer.queue import subtitle_player_status as qps
from oh_streamer.subtitles import slice_plan as sp


def message(**over):
    base = {"cmd": "subtitle_status", "item_id": "abc", "state": "running", "stage": "asr", "executor": "cloud",
            "done": 3, "total": 8, "covered_s": 600.0, "offset": 0.0, "complete": False, "wait": None, "stuck": None,
            "elsewhere": 0, "error": ""}
    base.update(over)
    return base


def status(**over):
    return ss.status_from_message(message(**over))


# ---------------------------------------------------------------------------------------------------------------
# queue message builder (pure)
# ---------------------------------------------------------------------------------------------------------------

def view(state="running", **over):
    base = {"v": 2, "state": state, "progress": 0.3, "covered_s": 900.0, "slices_done": 3, "slices_total": 8,
            "open": True, "running": None, "wait": None, "stuck": None, "failures_elsewhere": 0,
            "partial_stuck": False, "retryable": False}
    base.update(over)
    return base


class PlayerStatusBuilderTests(unittest.TestCase):
    def test_waiting_cut_reaches_player_panel_with_preparing_text(self):
        from oh_streamer.queue.subtitle_view import subtitle_view
        plan = sp.new_slice_plan(3600.0)
        req = sp.seek_cut_request(plan, 2400.0)
        sp.apply_seek_cuts(plan, req, [], [])
        derived = subtitle_view(plan, play_pos=1200.0)
        self.assertEqual(derived["state"], "waiting_cut")
        msg = qps.player_status("abc", {"status": "queued", "view": derived}, offset=2.0)
        received = ss.status_from_message(msg)
        self.assertEqual(ss.state_text(received), "準備切片")
        self.assertEqual(ss.level(received), "warn")
        rows = {r["k"]: r for r in info_panel.subtitle_rows(received, 2402.0, 0.0)}
        self.assertEqual(rows["字幕"]["v"], "準備切片")

    def test_a_running_view(self):
        sub = {"status": "running", "view": view(running={"executor": "cloud", "slice": 3, "stage": "asr"})}
        msg = qps.player_status("abc", sub, offset=2.5)
        self.assertEqual((msg["cmd"], msg["item_id"], msg["state"]), ("subtitle_status", "abc", "running"))
        self.assertEqual((msg["stage"], msg["executor"], msg["done"], msg["total"]), ("asr", "cloud", 3, 8))
        self.assertEqual((msg["covered_s"], msg["offset"], msg["complete"]), (900.0, 2.5, False))
        json.dumps(msg)                                                    # JSON-serialisable

    def test_waiting_and_stuck_views(self):
        wait = qps.player_status("a", {"status": "running", "view": view(
            "retrying", wait={"kind": "retry", "slice": 2, "stage": "mt", "reason": "network", "remaining_s": 12.5})})
        self.assertEqual(wait["wait"], {"kind": "retry", "reason": "network", "remaining_s": 12.5, "executor": None})
        self.assertEqual(wait["stage"], "mt")
        gpu = qps.player_status("a", {"status": "queued", "view": view("waiting_gpu", wait={"kind": "gpu", "executor": "local"})})
        self.assertEqual((gpu["state"], gpu["wait"]["executor"]), ("waiting_gpu", "local"))
        reclaim = qps.player_status("a", {"status": "queued", "view": view(
            "waiting_executor", wait={"kind": "executor", "executor": "local", "reclaim_in_s": 40.0})})
        self.assertEqual(reclaim["wait"]["remaining_s"], 40.0)
        stuck = qps.player_status("a", {"status": "running", "view": view(
            "stuck", stuck={"slice": 1, "stage": "asr", "track": "primary", "kind": "rate_limit", "candidates": []})})
        self.assertEqual((stuck["state"], stuck["stuck"], stuck["stage"]), ("stuck", {"kind": "rate_limit", "stage": "asr"}, "asr"))

    def test_complete_and_partial_failures(self):
        done = qps.player_status("a", {"status": "ready", "view": view("complete", covered_s=1800.0)})
        self.assertTrue(done["complete"])
        part = qps.player_status("a", {"status": "running", "view": view(failures_elsewhere=2)})
        self.assertEqual(part["elsewhere"], 2)

    def test_job_level_states(self):
        failed = qps.player_status("a", {"status": "failed", "error": "字幕音訊無法取得", "view": view()})
        self.assertEqual((failed["state"], failed["error"]), ("failed", "字幕音訊無法取得"))
        self.assertEqual(qps.player_status("a", {"status": "paused"})["state"], "paused")
        self.assertEqual(qps.player_status("a", {"status": "queued"})["state"], "preparing")   # no slice table yet
        for hidden in ("", "cancelled", "skipped"):
            self.assertIsNone(qps.player_status("a", {"status": hidden}))
        self.assertIsNone(qps.player_status("a", {}))

    def test_garbage_in_the_view_does_not_raise(self):
        msg = qps.player_status("a", {"status": "running", "view": view(
            covered_s=float("nan"), slices_done="x", slices_total=None, running="no", wait=[1])})
        self.assertEqual((msg["covered_s"], msg["done"], msg["total"], msg["wait"]), (None, 0, 0, None))

    def test_the_signature_ignores_the_countdown_only(self):
        a = qps.player_status("a", {"status": "running", "view": view(
            "waiting_quota", wait={"kind": "quota", "remaining_s": 30.0})})
        b = qps.player_status("a", {"status": "running", "view": view(
            "waiting_quota", wait={"kind": "quota", "remaining_s": 29.0})})
        c = qps.player_status("a", {"status": "running", "view": view(
            "waiting_quota", wait={"kind": "quota", "remaining_s": 29.0}, slices_done=4)})
        self.assertEqual(qps.signature(a), qps.signature(b))
        self.assertNotEqual(qps.signature(a), qps.signature(c))
        self.assertIsNone(qps.signature(None))


# ---------------------------------------------------------------------------------------------------------------
# queue sends it
# ---------------------------------------------------------------------------------------------------------------

class QueueSendsStatusTests(view_tests.StreamSubtitleViewTests):
    """Reuses the stream item fixtures of test_stream_subtitle_view (a real QueueService, manual ticks)."""

    def setUp(self):
        super().setUp()
        self.player_reply = {"ok": True}
        self.svc._watch[self.stream_id] = self.svc._clock()       # the player reported watching
        self.players.clear()

    def statuses(self):
        return [m for m in self.players if m.get("cmd") == "subtitle_status"]

    def tick(self, advance=0.0):
        self.now[0] += advance
        self.svc._watch[self.stream_id] = self.svc._clock()           # (the player's watching heartbeat keeps the lease)
        self.svc._subtitle_view_tick()

    def test_a_status_is_sent_when_the_view_appears_and_not_again_while_unchanged(self):
        self.tick()
        self.assertEqual([m["state"] for m in self.statuses()], ["preparing"])      # no slice table yet
        self.save_plan(self.stream_id, [0])
        self.tick(1)
        msgs = self.statuses()
        self.assertEqual(len(msgs), 2)
        self.assertEqual((msgs[-1]["item_id"], msgs[-1]["done"], msgs[-1]["total"]), (self.stream_id, 1, 3))
        for _ in range(5):                                                        # idle ticks: nothing new
            self.tick(1)
        self.assertEqual(len(self.statuses()), 2)

    def test_a_change_of_the_table_sends_again(self):
        self.save_plan(self.stream_id, [0])
        self.tick()
        self.save_plan(self.stream_id, [0, 1])
        self.tick(1)
        self.assertEqual([m["done"] for m in self.statuses()][-2:], [1, 2])

    def test_the_state_follows_the_play_clock(self):
        self.save_plan(self.stream_id, [0])
        self.svc._subtitle_clocks[self.stream_id] = (10.0, True)
        self.tick()
        first = self.statuses()[-1]
        self.assertEqual(first["covered_s"], 300.0)
        self.assertEqual(first["offset"], 0.0)

    def test_a_keepalive_is_sent_after_the_interval(self):
        self.save_plan(self.stream_id, [0])
        self.tick()
        count = len(self.statuses())
        self.tick(self.svc.STATUS_KEEPALIVE_S - 1)
        self.assertEqual(len(self.statuses()), count)
        self.tick(2)
        self.assertEqual(len(self.statuses()), count + 1)

    def test_a_refused_status_is_retried_soon_and_not_every_loop(self):
        self.player_reply = {"ok": False, "error": "invalid subtitle status"}
        self.save_plan(self.stream_id, [0])
        self.tick()
        count = len(self.statuses())
        self.tick(1)
        self.tick(1)
        self.assertEqual(len(self.statuses()), count)                              # throttled
        self.player_reply = {"ok": True}
        self.tick(self.svc.STATUS_RETRY_S)
        self.assertEqual(len(self.statuses()), count + 1)
        self.tick(1)
        self.assertEqual(len(self.statuses()), count + 1)                          # accepted now

    def test_a_failure_of_the_send_is_swallowed(self):
        self.svc._send_to_player = lambda payload: (_ for _ in ()).throw(OSError("no player"))
        self.save_plan(self.stream_id, [0])
        self.tick()                                                                # does not raise
        self.assertEqual(self.players, [])

    def test_a_failed_job_is_sent_with_its_reason(self):
        self.svc._subtitle_update(self.stream_id, status="failed", error="字幕音訊無法取得")
        self.tick()
        msg = self.statuses()[-1]
        self.assertEqual((msg["state"], msg["error"]), ("failed", "字幕音訊無法取得"))

    def test_only_items_the_player_is_watching(self):
        other = self.make_stream(CLOUD_ASR, CLOUD_MT)
        self.save_plan(other, [0])
        self.tick()
        self.assertEqual({m["item_id"] for m in self.statuses()}, {self.stream_id})

    def test_items_without_subtitles_or_removed_get_nothing(self):
        plain = self.svc.store.add_stream("https://x.test/w", "url", {}, {}, title="Plain")
        self.svc._watch[plain.id] = self.svc._clock()
        self.svc._tombstones.add(self.stream_id)
        self.tick()
        self.assertEqual(self.statuses(), [])

    def test_a_queue_item_the_player_watches_gets_it_too(self):
        queue_id = self.ready(state="queued")
        self.svc.store.update(queue_id, settings={**self.svc.store.get(queue_id).settings, "subtitles": True,
                                                  "subtitle_resolved": frozen(CLOUD_ASR, CLOUD_MT)})
        self.save_plan(queue_id, [0])
        self.svc._subtitle_update(queue_id, status="running")
        self.svc._watch[queue_id] = self.svc._clock()
        self.tick()
        self.assertIn(queue_id, {m["item_id"] for m in self.statuses()})

    def test_the_state_cleanup_forgets_items_that_are_no_longer_watched(self):
        self.save_plan(self.stream_id, [0])
        self.tick()
        self.assertIn(self.stream_id, self.svc._subtitle_status_sent)
        self.svc._watch.pop(self.stream_id)
        self.now[0] += 1
        self.svc._subtitle_view_tick()
        self.assertNotIn(self.stream_id, self.svc._subtitle_status_sent)

    def test_waiting_gpu_reaches_the_player(self):
        local = self.make_stream(LOCAL_ASR, LOCAL_MT)
        self.svc._watch[local] = self.svc._clock()
        self.save_plan(local, [])
        self.make_busy()
        self.tick()
        got = [m for m in self.statuses() if m["item_id"] == local]
        self.assertEqual(got[-1]["state"], "waiting_gpu")
        self.assertEqual(got[-1]["wait"]["executor"], "local")

    # --- T4.3 gpu_short ---------------------------------------------------------------------------------------
    def live(self, item, ahead, **over):
        report = {"title": "t", "pos": 100.0, "duration": 3600.0, "playing": True, "buffering": False,
                  "ahead_s": ahead, "speed": 1.0, "restore_done": False, "stream_item": item}
        report.update(over)
        self.svc._on_message({"cmd": "player_busy", "busy": True, "live": report})

    def short_run(self, item, aheads, step=15.0):
        """Feed one live report per `step` seconds, ticking the status each time; the last status message."""
        for ahead in aheads:
            self.live(item, ahead)
            self.tick()
            self.now[0] += step
        self.now[0] -= step                                            # (the last status is judged right at its sample)
        return [m for m in self.statuses() if m["item_id"] == item][-1]

    def local_stream(self):
        local = self.make_stream(LOCAL_ASR, LOCAL_MT)
        self.svc._watch[local] = self.svc._clock()
        self.save_plan(local, [])
        self.svc._subtitle_clocks[local] = (100.0, True)
        return local

    def test_gpu_short_when_the_restore_lead_falls_and_the_subtitles_are_behind(self):
        local = self.local_stream()
        msg = self.short_run(local, [120, 110, 100, 90, 80])          # 75 s, -40 s
        self.assertEqual(msg["state"], "waiting_gpu")
        self.assertIs(msg.get("gpu_short"), True)

    def test_gpu_short_needs_a_falling_lead(self):
        local = self.local_stream()
        self.assertFalse(self.short_run(local, [80, 90, 100, 110, 120]).get("gpu_short"))
        self.assertFalse(self.short_run(local, [120, 120, 119, 121, 120]).get("gpu_short"))      # flat (|d| <= 5)

    def test_gpu_short_when_two_turns_gave_up_for_vram_even_with_a_flat_lead(self):
        local = self.local_stream()
        self.svc._stream_vram_short_note(local)
        self.assertFalse(self.short_run(local, [120, 120, 120]).get("gpu_short"))          # one occurrence is not enough
        self.svc._stream_vram_short_note(local)
        msg = self.short_run(local, [120, 120, 120])
        self.assertIs(msg.get("gpu_short"), True)
        self.assertEqual(msg["state"], "waiting_gpu")

    def test_vram_short_still_needs_the_subtitles_to_be_behind(self):
        self.assertFalse(qps.gpu_short({"executor": "local", "covered_s": 900.0, "offset": 0.0}, [], 100.0, 100.0,
                                       300.0, vram_short=True))
        self.assertTrue(qps.gpu_short({"executor": "local", "covered_s": 150.0, "offset": 0.0}, [], 100.0, 100.0,
                                      300.0, vram_short=True))
        self.assertFalse(qps.gpu_short({"executor": "cloud", "covered_s": 0.0}, [], 100.0, 100.0, 300.0,
                                       vram_short=True))
        self.assertFalse(qps.gpu_short({"executor": "local", "covered_s": 0.0}, [], 100.0, 100.0, 300.0))

    def test_gpu_short_needs_a_long_enough_span(self):
        local = self.local_stream()
        msg = self.short_run(local, [120, 100, 80], step=15.0)          # only 30 s of samples
        self.assertFalse(msg.get("gpu_short"))

    def test_gpu_short_is_forgotten_when_the_restore_pauses_or_the_video_pauses(self):
        local = self.local_stream()
        self.assertTrue(self.short_run(local, [120, 110, 100, 90, 80]).get("gpu_short"))
        self.now[0] += 15
        self.live(local, 70, playing=False)
        self.tick()
        self.assertFalse([m for m in self.statuses() if m["item_id"] == local][-1].get("gpu_short"))
        self.assertEqual(self.svc._lead_samples, {})

    def test_gpu_short_needs_the_subtitles_to_be_behind(self):
        local = self.local_stream()
        self.svc._subtitle_clocks[local] = (100.0, True)
        sub = self.svc.store.get(local).subtitle
        with mock.patch.object(qps, "gpu_short", wraps=qps.gpu_short) as spy:
            self.short_run(local, [120, 110, 100, 90, 80])
        self.assertTrue(spy.called)
        msg = {"state": "running", "executor": "local", "wait": None, "complete": False, "covered_s": 600.0, "offset": 0.0}
        falling = [(0.0, 120.0), (60.0, 100.0)]
        self.assertTrue(qps.gpu_short(dict(msg, covered_s=350.0), falling, 60.0, 100.0, 300.0))      # lead 250
        self.assertFalse(qps.gpu_short(msg, falling, 60.0, 100.0, 300.0))                          # lead 500

    def test_cloud_subtitles_never_set_it(self):
        self.save_plan(self.stream_id, [0])
        self.svc._subtitle_clocks[self.stream_id] = (100.0, True)
        msg = self.short_run(self.stream_id, [120, 110, 100, 90, 80])
        self.assertNotIn("gpu_short", msg)

    def test_gpu_short_makes_a_new_message_at_once(self):
        local = self.local_stream()
        self.short_run(local, [120, 110, 100, 90])
        self.assertFalse([m for m in self.statuses() if m["item_id"] == local][-1].get("gpu_short"))
        self.now[0] += 15
        self.live(local, 80)
        self.tick()                                                    # span reaches 60 s: sent at once (no keep-alive due)
        self.assertEqual(self.svc._subtitle_status_sent[local][1], self.now[0])
        self.assertIs([m for m in self.statuses() if m["item_id"] == local][-1].get("gpu_short"), True)


# ---------------------------------------------------------------------------------------------------------------
# player parses it
# ---------------------------------------------------------------------------------------------------------------

class ParseTests(unittest.TestCase):
    def test_a_good_message(self):
        st = ss.status_from_message(message(wait={"kind": "retry", "reason": "timeout", "remaining_s": 9}))
        self.assertEqual((st.item_id, st.state, st.stage, st.executor), ("abc", "running", "asr", "cloud"))
        self.assertEqual((st.done, st.total, st.covered_s, st.offset, st.complete), (3, 8, 600.0, 0.0, False))
        self.assertEqual(st.wait, {"kind": "retry", "reason": "timeout", "executor": None, "remaining_s": 9.0})

    def test_bad_messages_are_refused(self):
        for bad in (None, [], {}, {"cmd": "subtitle_update"}, message(item_id=""), message(item_id=None),
                    message(item_id=True), message(state=""), message(state=3), message(state=None)):
            self.assertIsNone(ss.status_from_message(bad), bad)

    def test_garbage_fields_are_neutral(self):
        st = ss.status_from_message(message(covered_s=float("inf"), offset="x", done=-4, total=True, wait="no",
                                            stuck=3, elsewhere=None, error=5, complete="yes", stage=9, executor=""))
        self.assertEqual((st.covered_s, st.offset, st.done, st.total, st.wait, st.stuck, st.elsewhere, st.error),
                         (None, 0.0, 0, 0, None, None, 0, ""))
        self.assertEqual((st.complete, st.stage, st.executor), (False, None, None))
        self.assertIsNone(ss.status_from_message(message(covered_s=-1)).covered_s)

    def test_an_integer_item_id_is_accepted_as_text(self):
        self.assertEqual(ss.status_from_message(message(item_id=7)).item_id, "7")

    def test_a_long_error_is_cut(self):
        self.assertEqual(len(ss.status_from_message(message(state="failed", error="x" * 500)).error), 200)


# ---------------------------------------------------------------------------------------------------------------
# wording
# ---------------------------------------------------------------------------------------------------------------

class TextTests(unittest.TestCase):
    def test_running_names_the_stage_the_place_and_the_progress(self):
        self.assertEqual(ss.state_text(status()), "語音辨識中（雲端） 3/8 段")
        self.assertEqual(ss.state_text(status(stage="mt", executor="local")), "翻譯中（本地） 3/8 段")
        self.assertEqual(ss.state_text(status(total=0, stage=None, executor=None)), "字幕中")

    def test_progress_never_exceeds_the_total(self):
        self.assertEqual(ss.state_text(status(done=9, total=8)), "語音辨識中（雲端） 8/8 段")

    def test_waiting_states(self):
        self.assertEqual(ss.state_text(status(state="waiting_gpu")), "等待 GPU")
        self.assertEqual(ss.state_text(status(state="waiting_download")), "等待下載")
        self.assertEqual(ss.state_text(status(state="queued")), "排隊中")
        self.assertEqual(ss.state_text(status(state="preparing")), "準備中")
        self.assertEqual(ss.state_text(status(state="paused")), "已暫停")
        self.assertEqual(ss.state_text(status(state="complete", complete=True)), "已完成")
        self.assertEqual(ss.state_text(status(state="waiting_executor", wait={"executor": "local", "remaining_s": 5})),
                         "已交給本地備援，等待接手")
        self.assertEqual(ss.state_text(status(state="waiting_executor", wait=None)), "已交給備援，等待接手")
        self.assertEqual(ss.state_text(status(state="something_new")), "處理中")

    def test_the_countdown_keeps_running_between_snapshots(self):
        st = status(state="waiting_quota", wait={"kind": "quota", "remaining_s": 30})
        st.received = 100.0
        self.assertEqual(ss.state_text(st, 100.0), "等待額度中（30 秒後重試）")
        self.assertEqual(ss.state_text(st, 112.2), "等待額度中（18 秒後重試）")
        self.assertEqual(ss.state_text(st, 140.0), "等待額度中（即將重試）")
        self.assertEqual(ss.state_text(status(state="waiting_quota", wait=None)), "等待額度中")

    def test_retrying_names_the_reason(self):
        st = status(state="retrying", wait={"kind": "retry", "reason": "network", "remaining_s": 12})
        self.assertEqual(ss.state_text(st, 0.0), "重試中（12 秒後重試，網路問題）")
        st = status(state="retrying", wait={"kind": "retry", "reason": "odd_kind", "remaining_s": None})
        self.assertEqual(ss.state_text(st), "重試中（odd_kind）")

    def test_failures(self):
        self.assertEqual(ss.state_text(status(state="failed", error="字幕音訊無法取得")), "失敗（字幕音訊無法取得）")
        self.assertEqual(ss.state_text(status(state="failed", error="")), "失敗")
        self.assertEqual(ss.state_text(status(state="stuck", stuck={"kind": "rate_limit", "stage": "asr"})),
                         "語音辨識卡住（被限流）")
        self.assertEqual(ss.state_text(status(state="stuck", stuck={"kind": None, "stage": "mt"})), "翻譯卡住")
        self.assertEqual(ss.state_text(status(state="stuck", stuck=None, stage=None)), "字幕卡住")

    def test_other_failed_slices_are_mentioned_unless_the_state_already_says_so(self):
        self.assertEqual(ss.state_text(status(elsewhere=2)), "語音辨識中（雲端） 3/8 段；另有 2 段失敗")
        self.assertNotIn("另有", ss.state_text(status(state="stuck", elsewhere=2, stuck={"kind": "timeout", "stage": "asr"})))
        self.assertNotIn("另有", ss.state_text(status(state="complete", elsewhere=1)))

    def test_lead(self):
        st = status(covered_s=900.0, offset=0.0)
        self.assertEqual(ss.lead_text(st, 180.0), "12 分鐘")
        self.assertEqual(ss.lead_text(st, 860.0), "40 秒")
        self.assertEqual(ss.lead_text(st, 900.0), "尚未追上播放位置")
        self.assertEqual(ss.lead_text(st, 1200.0), "尚未追上播放位置")
        self.assertEqual(ss.lead_text(st, None), "—")
        self.assertEqual(ss.lead_text(status(covered_s=None), 10.0), "—")
        self.assertEqual(ss.lead_s(status(covered_s=900.0, offset=5.0), 100.0), 805.0)       # offset maps the audio clock
        self.assertEqual(ss.lead_text(status(state="complete", complete=True), 10.0), "全部完成")

    def test_the_headline(self):
        pos = 180.0
        self.assertEqual(ss.headline(status(covered_s=900.0), pos), "字幕：已領先 12 分鐘")
        self.assertEqual(ss.headline(status(covered_s=900.0, state="waiting_gpu"), pos), "字幕：已領先 12 分鐘")
        self.assertEqual(ss.headline(status(covered_s=190.0), pos), "字幕：語音辨識中（雲端） 3/8 段")
        self.assertEqual(ss.headline(status(state="waiting_gpu", covered_s=None), pos), "字幕：等待 GPU")
        self.assertEqual(ss.headline(status(state="failed", error="額度用完", covered_s=900.0), pos), "字幕：失敗（額度用完）")
        self.assertEqual(ss.headline(status(state="complete", complete=True), pos), "字幕：已完成")
        self.assertEqual(ss.headline(status(), None), "字幕：語音辨識中（雲端） 3/8 段")

    def test_levels(self):
        self.assertEqual(ss.level(status(state="failed")), "bad")
        self.assertEqual(ss.level(status(state="stuck")), "bad")
        self.assertEqual(ss.level(status(state="complete")), "good")
        self.assertEqual(ss.level(status(state="waiting_gpu")), "warn")
        self.assertEqual(ss.level(status(state="queued")), "idle")
        self.assertEqual(ss.level(status(covered_s=900.0), 100.0), "good")
        self.assertEqual(ss.level(status(covered_s=105.0), 100.0), "warn")
        self.assertEqual(ss.lead_level(status(covered_s=900.0), 100.0), "good")
        self.assertEqual(ss.lead_level(status(covered_s=130.0), 100.0), "warn")
        self.assertEqual(ss.lead_level(status(covered_s=100.0), 100.0), "bad")
        self.assertIsNone(ss.lead_level(status(state="failed"), 100.0))

    def test_problem_keys_identify_a_failure_once(self):
        self.assertIsNone(ss.problem_key(status()))
        self.assertEqual(ss.problem_key(status(state="failed", error="a")), ("failed", "a"))
        self.assertNotEqual(ss.problem_key(status(state="failed", error="a")), ss.problem_key(status(state="failed", error="b")))
        self.assertEqual(ss.problem_key(status(state="stuck", stuck={"kind": "timeout", "stage": "asr"})),
                         ("stuck", "asr", "timeout"))


# ---------------------------------------------------------------------------------------------------------------
# panel rows
# ---------------------------------------------------------------------------------------------------------------

def section(info, title):
    return next((s for s in info["sections"] if s["title"] == title), None)


class PanelTests(unittest.TestCase):
    def rows(self, st, pos=180.0, now=0.0):
        return {r["k"]: r for r in info_panel.subtitle_rows(st, pos, now)}

    def test_no_status_no_section(self):
        self.assertIsNone(section(info_panel.build_info({"pos": 5.0}), "字幕"))
        self.assertIsNone(section(info_panel.build_info({"pos": 5.0, "subtitle_status": None}), "字幕"))

    def test_the_section_follows_the_processing_section(self):
        info = info_panel.build_info({"pos": 180.0, "subtitle_status": status(), "subtitle_now": 0.0})
        self.assertEqual([s["title"] for s in info["sections"]], ["影片", "處理狀態", "字幕", "設定", "GPU"])
        with_cast = info_panel.build_info({"pos": 180.0, "subtitle_status": status(),
                                           "cast": {"phase": "playing", "tv_pos": 1.0, "cached_ranges": []}})
        self.assertEqual([s["title"] for s in with_cast["sections"]][:4], ["影片", "投放", "處理狀態", "字幕"])

    def test_running_rows(self):
        info = info_panel.build_info({"pos": 180.0, "subtitle_status": status(covered_s=900.0), "subtitle_now": 0.0})
        rows = {r["k"]: r for r in section(info, "字幕")["rows"]}
        self.assertEqual(rows["字幕"]["v"], "語音辨識中（雲端） 3/8 段")
        self.assertEqual(rows["字幕領先"], {"k": "字幕領先", "v": "12 分鐘", "c": "good"})

    def test_waiting_gpu_and_failed_rows(self):
        gpu = self.rows(status(state="waiting_gpu", covered_s=None))
        self.assertEqual((gpu["字幕"]["v"], gpu["字幕"]["c"], gpu["字幕領先"]["v"]), ("等待 GPU", "warn", "—"))
        failed = self.rows(status(state="failed", error="額度用完"))
        self.assertEqual((failed["字幕"]["v"], failed["字幕"]["c"]), ("失敗（額度用完）", "bad"))
        self.assertNotIn("字幕領先", failed)                                          # nothing to lead with

    def test_preparing_and_paused_have_only_the_state_row(self):
        self.assertEqual(list(self.rows(status(state="preparing", covered_s=None))), ["字幕"])
        self.assertEqual(list(self.rows(status(state="paused"))), ["字幕"])

    def test_complete_row(self):
        rows = self.rows(status(state="complete", complete=True))
        self.assertEqual((rows["字幕"]["v"], rows["字幕"]["c"], rows["字幕領先"]["v"]), ("已完成", "good", "全部完成"))

    def test_a_long_reason_is_truncated_to_the_panel_width(self):
        rows = self.rows(status(state="failed", error="很長的原因" * 20))
        self.assertLessEqual(info_panel.display_width(rows["字幕"]["v"]), 34)

    def test_the_countdown_uses_the_given_clock(self):
        st = status(state="waiting_quota", wait={"kind": "quota", "remaining_s": 30})
        st.received = 10.0
        self.assertEqual(self.rows(st, now=20.0)["字幕"]["v"], "等待額度中（20 秒後重試）")

    def test_the_section_is_json_serialisable(self):
        json.dumps(info_panel.build_info({"pos": 1.0, "subtitle_status": status()}), ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------
# controller: receives, keeps, shows
# ---------------------------------------------------------------------------------------------------------------

class ControllerTests(link_tests.PlayerSide):
    def make_player(self, **kw):
        player = super().make_player(**kw)
        player.subtitle_status_box = __import__("queue").Queue()          # (created by run() in the real player)
        player.pending_subtitle_status, player.pending_subtitle_updates = {}, {}
        player.subtitle_status = player.subtitle_problem_shown = None
        return player

    def ready_live(self, item="abc"):
        player = self.make_player()
        self.begin(player, stream_item=item)
        player.ipc = link_tests.LiveMpv(player.ipc.events)
        player.ipc.props.update({"path": "http://127.0.0.1/seg", "time-pos": 42.5})
        player.session.url = "http://127.0.0.1/seg"
        return player

    def deliver(self, player, **over):
        reply = player.on_message(message(**over))
        status = player.subtitle_status_box.get_nowait() if not player.subtitle_status_box.empty() else None
        if status is not None:
            player.apply_subtitle_status(status)
        return reply

    def osd_texts(self, player):
        return [c[1] for c in player.ipc.commands if c[0] == "show-text"]

    def test_on_message_validates_and_queues_it(self):
        player = self.ready_live()
        self.assertEqual(player.on_message(message()), {"ok": True})
        got = player.subtitle_status_box.get_nowait()
        self.assertEqual((got.item_id, got.state, got.received > 0), ("abc", "running", True))
        for bad in (message(state=""), message(item_id=""), {"cmd": "subtitle_status"}):
            self.assertEqual(player.on_message(bad), {"ok": False, "error": "invalid subtitle status"})
        self.assertTrue(player.subtitle_status_box.empty())

    def test_the_status_of_the_playing_stream_item_is_kept(self):
        player = self.ready_live()
        self.assertIsNone(player.current_subtitle_status())
        self.deliver(player, done=1)
        self.deliver(player, done=2)
        self.assertEqual(player.current_subtitle_status().done, 2)

    def test_the_status_of_another_item_is_not_shown_and_waits(self):
        player = self.ready_live()
        self.deliver(player, item_id="zzz")
        self.assertIsNone(player.current_subtitle_status())
        self.assertIn("zzz", player.pending_subtitle_status)

    def test_a_status_before_the_open_is_applied_when_the_stream_item_opens(self):
        player = self.make_player()
        self.assertIsNone(player.subtitle_owner())
        player.apply_subtitle_status(ss.status_from_message(message()))
        self.assertIsNone(player.current_subtitle_status())
        self.begin(player, stream_item="abc")
        self.assertEqual(player.current_subtitle_status().state, "running")
        self.assertEqual(player.pending_subtitle_status, {})

    def test_the_pending_table_is_bounded(self):
        player = self.make_player()
        for i in range(30):
            player.apply_subtitle_status(ss.status_from_message(message(item_id=f"i{i}")))
        self.assertEqual(len(player.pending_subtitle_status), 16)

    def test_a_new_source_without_a_stream_item_shows_nothing(self):
        player = self.ready_live()
        self.deliver(player)
        player.stream_subs = None
        self.assertIsNone(player.current_subtitle_status())

    def test_a_failure_is_shown_once_as_an_osd(self):
        player = self.ready_live()
        self.deliver(player, state="failed", error="字幕音訊無法取得")
        self.deliver(player, state="failed", error="字幕音訊無法取得")
        self.assertEqual(self.osd_texts(player), ["字幕：失敗（字幕音訊無法取得）"])
        self.deliver(player, state="running")                               # recovered ...
        self.deliver(player, state="failed", error="字幕音訊無法取得")      # ... and failed again: shown again
        self.assertEqual(len(self.osd_texts(player)), 2)

    def test_gpu_short_osd_once_per_source(self):
        player = self.ready_live()
        self.deliver(player, state="waiting_gpu", executor="local")
        self.deliver(player, state="waiting_gpu", executor="local", gpu_short=False)
        self.assertEqual(self.osd_texts(player), [])
        self.deliver(player, state="waiting_gpu", executor="local", gpu_short=True)
        self.deliver(player, state="waiting_gpu", executor="local", gpu_short=True)
        self.deliver(player, state="running", executor="local")
        self.deliver(player, state="waiting_gpu", executor="local", gpu_short=True)
        self.assertEqual(self.osd_texts(player), [ss.GPU_SHORT_OSD])
        player.epoch += 1                                                   # a new source
        self.deliver(player, state="waiting_gpu", executor="local", gpu_short=True)
        self.assertEqual(self.osd_texts(player), [ss.GPU_SHORT_OSD] * 2)

    def test_gpu_short_parse_and_panel_row(self):
        self.assertFalse(status().gpu_short)
        self.assertFalse(status(gpu_short="yes").gpu_short)
        st = status(gpu_short=True)
        self.assertTrue(st.gpu_short)
        self.assertIn("算力", [r["k"] for r in info_panel.subtitle_rows(st, 10.0, 0.0)])
        self.assertNotIn("算力", [r["k"] for r in info_panel.subtitle_rows(status(), 10.0, 0.0)])

    def test_a_different_failure_is_shown_too_and_progress_is_silent(self):
        player = self.ready_live()
        self.deliver(player, state="stuck", stuck={"kind": "rate_limit", "stage": "asr"})
        self.deliver(player, state="stuck", stuck={"kind": "timeout", "stage": "asr"})
        self.deliver(player, state="running")
        self.deliver(player, state="waiting_gpu")
        self.deliver(player, state="complete", complete=True)
        self.assertEqual(self.osd_texts(player), ["字幕：語音辨識卡住（被限流）", "字幕：語音辨識卡住（逾時）"])

    def test_no_osd_without_mpv(self):
        player = self.ready_live()
        player.ipc = None
        self.deliver(player, state="failed", error="x")                     # does not raise
        self.assertEqual(player.current_subtitle_status().state, "failed")

    def test_publish_info_hands_the_status_to_the_panel(self):
        player = self.ready_live()
        self.deliver(player, covered_s=900.0)
        snaps = []
        player._deps.info_panel.build_info = lambda snap: snaps.append(snap) or {"state": "", "sections": []}
        player.session = None
        player.processed = SimpleNamespace(queue_id="abc", tags=None, subtitles=[])
        player.stream_subs = None
        player.pending_open = None
        player.title = player.source_text = "clip"
        player.published_duration = 100.0
        player.cast.info_snapshot = lambda: None
        player.engine.lada = None                                           # (no restoration engine loaded)
        player.gpu, player.hist = SimpleNamespace(snapshot=lambda: None), SimpleNamespace(snapshot=lambda: {})
        player.sc = None
        player._deps = dataclasses.replace(player._deps, mpv_playback_snapshot=lambda *a: {"pos": 180.0, "paused": None})
        player.publish_info()
        self.assertEqual(snaps[-1]["subtitle_status"].state, "running")
        self.assertIn("subtitle_now", snaps[-1])
        player.processed = SimpleNamespace(queue_id="other", tags=None, subtitles=[])
        player.publish_info()
        self.assertIsNone(snaps[-1]["subtitle_status"])


if __name__ == "__main__":
    unittest.main()
