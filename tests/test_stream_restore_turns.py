"""T4.2 B2: the stream GpuHost's restore_pause / restore_resume over the player IPC, and the queue's restore_needed
handler (kill the local turn, resume the player). Fakes only: the "player" is the service's send_to_player stub that
answers restore_pause by writing the confirmed state into the live report; the only real process is a sleeping
python child this test starts and kills."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import subtitle_host_gpu as gh
from oh_streamer.queue import subtitle_local as sl
from oh_streamer.queue.queue_service import _Run, QueueService
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.queue.subtitle_scheduler import Cancelled

import test_queue_subtitle_host_gpu as tg
from test_queue_subtitle_host_gpu import SLEEPER


class RestoreTurnBase(tg.Base):
    stream_item = tg.StreamLocalTests.stream_item
    live = tg.StreamLocalTests.live

    def setUp(self):
        super().setUp()
        self.now = 1005.0
        self.sent = []                       # every payload sent to the player, in order
        self.events = []                     # (kind, extra) in the same order, for kill / resume ordering
        self.player_mode = "confirm"         # confirm | never | fail
        self.svc._send_to_player = self.fake_player
        self.svc._clock = lambda: self.now
        self.svc.busy_expire = 30.0
        for name, value in (("STOP_RESTORE_TIMEOUT_S", 0.4), ("RESTORE_POLL_S", 0.01)):
            patcher = mock.patch.object(gh, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stream_item()
        self.live()
        self.svc._clock = lambda: self.now              # live() pins its own constant clock
        self.svc._live["restore_pause"] = {"state": "running", "turn": None}
        import tempfile
        from pathlib import Path
        from oh_streamer.queue import stream_subtitle_gpu as _ssg
        _tmp = tempfile.TemporaryDirectory()
        self.addCleanup(_tmp.cleanup)
        _costs = _ssg.SwitchCostMemory(Path(_tmp.name) / "costs.json")
        self.svc._subtitle_cost_memory = lambda: _costs          # never the user's real file
        patcher = mock.patch.object(gh, "RESTART_WATCH_S", 0.3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_player(self, payload):
        self.sent.append(dict(payload))
        cmd = payload.get("cmd")
        if self.player_mode == "fail":
            return None
        with self.svc._lock:
            if cmd == "restore_pause" and self.player_mode == "confirm":
                self.svc._live["restore_pause"] = {"state": "paused", "turn": payload["turn"]}
                if getattr(self, "reserved_mib", None) is not None:
                    self.svc._live["restore_pause"]["reserved_mib"] = self.reserved_mib
            elif cmd == "restore_pause":
                self.svc._live["restore_pause"] = {"state": "pausing", "turn": payload["turn"]}
            elif cmd == "restore_resume":
                self.svc._live["restore_pause"] = {"state": "running", "turn": None}
        return {"ok": True}

    def dispatch(self, msg):
        return QueueService._on_message(self.svc, msg)      # the real handler on the fake service

    def cmds(self):
        return [p["cmd"] for p in self.sent]

    def stream_host(self):
        run = _Run("stream01")
        run.stage = gh.RUN_STAGE_LOCAL
        run.executor = SimpleNamespace(stop=lambda: None)
        run.host = gh.QueueStreamGpuHost(self.svc, "stream01", run)
        return run.host


class RestartCostTests(RestoreTurnBase):
    """Second round fix 2: the host learns the restoration restart = restore_resume -> the next live report that shows
    restoration output again (restore_pause running and speed > 0)."""

    def setUp(self):
        super().setUp()
        import tempfile
        from pathlib import Path
        from oh_streamer.queue import stream_subtitle_gpu as ssg
        self.ssg = ssg
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.costs = ssg.SwitchCostMemory(Path(tmp.name) / "costs.json")
        self.svc._subtitle_cost_memory = lambda: self.costs
        self.svc._subtitle_gpu_info = lambda: sl.GpuInfo("Test GPU", 8192.0)
        self.key = gt.gpu_key("Test GPU", 8192.0)
        self.speed_now = 0.0
        for name, value in (("RESTART_POLL_S", 0.01), ("RESTART_WATCH_S", 1.0)):
            patcher = mock.patch.object(gh, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def report(self, speed):
        """The player's next live report (received after the resume)."""
        self.now += 1.0
        with self.svc._lock:
            self.svc._live["speed"] = speed
            self.svc._live["restore_pause"] = {"state": "running", "turn": None}
            self.svc._live_ts = self.now - 0.2

    def wait_cost(self, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = self.costs.get(self.key, self.ssg.COST_RESTART)
            if value is not None:
                return value
            time.sleep(0.01)
        return None

    def test_restart_is_measured_until_output_flows_again(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.svc._live["speed"] = 0.0
        host.resume_restore()
        time.sleep(0.15)
        self.assertIsNone(self.costs.get(self.key, self.ssg.COST_RESTART))      # still no output
        self.report(0.0)
        time.sleep(0.05)
        self.assertIsNone(self.costs.get(self.key, self.ssg.COST_RESTART))      # report after the resume, speed 0
        self.report(1.4)
        value = self.wait_cost()
        self.assertIsNotNone(value)
        self.assertTrue(0.1 <= value < 1.0)

    def test_a_report_from_before_the_resume_does_not_count(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.svc._live["speed"] = 1.4                                            # stale speed in an old report
        self.svc._live_ts = self.now - 1.0
        host.resume_restore()
        time.sleep(0.2)
        self.assertIsNone(self.costs.get(self.key, self.ssg.COST_RESTART))

    def test_no_measurement_when_the_turn_was_cancelled_or_never_paused(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        host._resume_player("restore_needed")                                    # a cancelled turn: not a normal restart
        self.report(1.4)
        time.sleep(0.2)
        self.assertIsNone(self.costs.get(self.key, self.ssg.COST_RESTART))

    def test_a_new_pause_ends_the_watch(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        host.resume_restore()
        self.assertTrue(host.stop_restore())                                     # next turn paused again at once
        self.report(1.4)
        time.sleep(0.2)
        self.assertIsNone(self.costs.get(self.key, self.ssg.COST_RESTART))

    def test_a_failing_memory_never_breaks_the_resume(self):
        self.svc._subtitle_cost_memory = lambda: (_ for _ in ()).throw(OSError("disk"))
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        host.resume_restore()
        self.report(1.4)
        time.sleep(0.2)
        self.assertFalse(host.holds_player_pause())


class PausedVramTests(RestoreTurnBase):
    """T4.3: the confirmed pause teaches the real paused restore VRAM; a short turn counts toward gpu_short."""

    def setUp(self):
        super().setUp()
        import tempfile
        from pathlib import Path
        from oh_streamer.queue import stream_subtitle_gpu as ssg
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.memory = ssg.RestoreVramMemory(Path(tmp.name) / "restore-vram.json")
        self.svc._subtitle_restore_vram_memory = lambda: self.memory
        self.svc._subtitle_gpu_info = lambda: sl.GpuInfo("Test GPU", 8192.0)
        self.key = gt.gpu_key("Test GPU", 8192.0)
        self.svc._vram_short_times = {}
        for name in ("_vram_short_count", "_stream_vram_short_note"):          # the real methods on the fake service
            setattr(self.svc, name, getattr(QueueService, name).__get__(self.svc))

    def test_reserved_mib_of_the_confirmed_pause_is_remembered_per_card(self):
        self.reserved_mib = 1505.0
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.assertEqual(self.memory.get(self.key), 1505.0)
        host.resume_restore()
        self.reserved_mib = 1300.0
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.assertEqual(self.memory.get(self.key), 1505.0)                 # the recent larger value stays

    def test_no_figure_in_the_report_remembers_nothing(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.assertIsNone(self.memory.get(self.key))

    def test_a_failing_memory_never_breaks_the_pause(self):
        self.reserved_mib = 1505.0
        self.svc._subtitle_restore_vram_memory = lambda: (_ for _ in ()).throw(OSError("disk"))
        self.assertTrue(self.stream_host().stop_restore())

    def test_note_vram_short_is_counted_per_item_within_the_window(self):
        host = self.stream_host()
        self.assertEqual(self.svc._vram_short_count("stream01", 1000.0), 0)
        host.note_vram_short(5313.0, 3000.0)
        host.note_vram_short(5313.0, None)
        now = self.svc._subtitle_time()
        self.assertEqual(self.svc._vram_short_count("stream01", now), 2)
        self.assertEqual(self.svc._vram_short_count("other", now), 0)
        self.assertEqual(self.svc._vram_short_count("stream01", now + 601.0), 0)          # forgotten
        self.assertNotIn("stream01", self.svc._vram_short_times)


class PauseResumeTests(RestoreTurnBase):
    def test_pause_confirmed_then_lease_granted(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        self.assertEqual(self.cmds(), ["restore_pause"])
        pause = self.sent[0]
        self.assertEqual(pause["stream_item"], "stream01")
        self.assertEqual(len(pause["turn"]), 8)
        self.assertTrue(host._player_pause_confirmed())
        lease = host.acquire_lease(True)
        self.assertIsNotNone(lease)
        self.assertIs(self.svc._wk, lease)
        host.release_lease(lease)

    def test_pause_not_confirmed_resumes_and_refuses_the_lease(self):
        self.player_mode = "never"
        host = self.stream_host()
        self.assertFalse(host.stop_restore())
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])
        self.assertEqual(self.sent[0]["turn"], self.sent[1]["turn"])
        self.assertFalse(host._player_pause_confirmed())
        self.assertIsNone(host.acquire_lease(True))
        self.assertEqual(host.yield_reason, "player_restore_running")
        self.assertIsNone(self.svc._wk)

    def test_a_report_for_another_turn_is_not_a_confirmation(self):
        self.player_mode = "never"
        host = self.stream_host()
        self.svc._live["restore_pause"] = {"state": "paused", "turn": "someoneelse"}
        self.assertFalse(host.stop_restore())

    def test_unreachable_player_fails_at_once(self):
        self.player_mode = "fail"
        host = self.stream_host()
        t0 = time.monotonic()
        self.assertFalse(host.stop_restore())
        self.assertLess(time.monotonic() - t0, 0.3)
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])

    def test_cancel_while_waiting_resumes_and_raises(self):
        self.player_mode = "never"
        host = self.stream_host()
        threading.Timer(0.05, host.run.cancel.set).start()
        with self.assertRaises(Cancelled):
            host.stop_restore()
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])
        self.assertFalse(host.holds_player_pause())

    def test_resume_restore_sends_resume_once(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        turn = self.sent[0]["turn"]
        host.resume_restore()
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])
        self.assertEqual(self.sent[1], {"cmd": "restore_resume", "stream_item": "stream01", "turn": turn})
        self.assertFalse(host._player_pause_confirmed())
        host.resume_restore()
        host.close()
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])       # the token is spent

    def test_close_resumes_a_pause_still_held(self):
        host = self.stream_host()
        self.assertTrue(host.stop_restore())
        lease = host.acquire_lease(True)
        host.close()                                    # thread end without end_gpu_turn
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])
        self.assertIsNone(self.svc._wk)
        self.assertIsNotNone(lease)

    def test_close_without_a_pause_sends_nothing(self):
        self.stream_host().close()
        self.assertEqual(self.sent, [])

    def test_a_late_resume_of_an_old_host_cannot_release_a_newer_pause(self):
        old = self.stream_host()
        self.assertTrue(old.stop_restore())
        old._resume_player("test")
        new = self.stream_host()
        self.assertTrue(new.stop_restore())
        old.close()
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume", "restore_pause"])

    def test_executor_sees_gpu_busy_when_the_pause_fails(self):
        """_begin_gpu_turn: stop_restore False -> lease refused -> "gpu_busy" (the work stays pending, retried later);
        no resume_restore follows (stopped is False), the host already told the player."""
        self.player_mode = "never"
        host = self.stream_host()
        executor = sl.LocalSliceExecutor.__new__(sl.LocalSliceExecutor)
        executor.host, executor._lease, executor.clock = host, None, time.monotonic
        executor._event = lambda *a, **k: None
        executor._stopped_restore = False
        executor._turn_started = None
        self.assertFalse(executor._begin_gpu_turn(exclusive=True))
        self.assertEqual(self.cmds(), ["restore_pause", "restore_resume"])


class RestoreNeededTests(RestoreTurnBase):
    def paused_run(self, with_process=False):
        """A stream local run holding a confirmed pause and the exclusive lease (like mid-turn)."""
        host = self.stream_host()
        run = host.run
        self.svc._subtitle_local = run
        self.assertTrue(host.stop_restore())
        host.acquire_lease(True)
        proc = None
        if with_process:
            proc = subprocess.Popen(SLEEPER, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.children.append(proc)
            host.register_process(proc)
        return run, host, proc

    def wait_for(self, cond, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return cond()

    def test_restore_needed_kills_the_turn_then_resumes(self):
        run, host, proc = self.paused_run(with_process=True)
        order = []
        real_send = self.svc._send_to_player

        def send(payload):
            if payload["cmd"] == "restore_resume":
                order.append(("resume", proc.poll() is not None))      # was the model process dead already?
            return real_send(payload)
        self.svc._send_to_player = send
        reply = self.dispatch({"cmd": "restore_needed", "id": "stream01", "reason": "seek"})
        self.assertEqual(reply, {"ok": True})
        self.assertTrue(run.cancel.is_set())                           # set synchronously, before the reply
        self.assertEqual(host.yield_reason, "player_needs_restore")
        self.assertTrue(self.wait_for(lambda: order))
        self.assertEqual(order, [("resume", True)])
        self.assertIsNotNone(proc.poll())
        self.assertFalse(host.holds_player_pause())
        self.assertEqual(self.svc.stops[0], ("stream01", "auto"))
        host.release_lease(host._lease)
        host.resume_restore()                                          # the executor's own end of turn: no 2nd resume
        host.close()
        self.assertEqual(self.cmds().count("restore_resume"), 1)

    def test_restore_needed_without_a_paused_turn_still_resumes(self):
        reply = self.dispatch({"cmd": "restore_needed", "id": "stream01", "reason": "buffer_low"})
        self.assertEqual(reply, {"ok": True})
        self.assertTrue(self.wait_for(lambda: "restore_resume" in self.cmds()))
        self.assertEqual(self.cmds(), ["restore_resume"])

    def test_restore_needed_for_another_run_does_not_cancel_it(self):
        other = _Run("it9")
        other.host = gh.QueueGpuHost(self.svc, "it9", other)
        self.svc._subtitle_local = other
        self.assertEqual(self.dispatch({"cmd": "restore_needed", "id": "stream01", "reason": "seek"}),
                         {"ok": True})
        self.assertFalse(other.cancel.is_set())

    def test_unknown_id_and_non_stream_item(self):
        self.assertEqual(self.dispatch({"cmd": "restore_needed", "id": "nope", "reason": "seek"}),
                         {"ok": False, "error": "unknown id"})
        self.item("it1")
        self.assertEqual(self.dispatch({"cmd": "restore_needed", "id": "it1", "reason": "seek"}),
                         {"ok": False, "error": "not a stream item"})
        self.assertEqual(self.sent, [])

    def test_live_report_keeps_restore_pause(self):
        clean = QueueService._clean_live({"stream_item": "s", "restore_pause": {"state": "paused", "turn": "ab12"}})
        self.assertEqual(clean["restore_pause"], {"state": "paused", "turn": "ab12"})
        self.assertNotIn("restore_pause", QueueService._clean_live({"stream_item": "s"}))
        self.assertNotIn("restore_pause", QueueService._clean_live({"restore_pause": {"state": "weird"}}))
        self.assertEqual(QueueService._clean_live({"restore_pause": {"state": "running", "turn": 5}})["restore_pause"],
                         {"state": "running", "turn": None})

    def test_live_report_keeps_a_valid_reserved_mib_only_while_paused(self):
        def pause(**over):
            return QueueService._clean_live({"restore_pause": {"turn": "t", **over}})["restore_pause"]
        self.assertEqual(pause(state="paused", reserved_mib=1505.5),
                         {"state": "paused", "turn": "t", "reserved_mib": 1505.5})
        self.assertEqual(pause(state="pausing", reserved_mib=1505.5), {"state": "pausing", "turn": "t"})
        for bad in (-1, float("nan"), "x", True, None):
            self.assertEqual(pause(state="paused", reserved_mib=bad), {"state": "paused", "turn": "t"})


class AfterSeekTests(RestoreTurnBase):
    def test_no_subtitle_turn_when_the_lead_is_about_zero(self):
        """After a seek the restore lead is ~0: decide_gpu_turn says restore (the player is not paused again)."""
        self.svc._live["ahead_s"] = 0.0
        host = self.stream_host()
        info = host.restore_info()
        self.assertEqual(info.lead_s, 0.0)
        state = gt.TurnState(sub_lead_s=0.0, restore_lead_s=info.lead_s, watched=True, subtitle_work=True,
                             restore_work=info.has_work)
        turn = gt.decide_gpu_turn(state, gh.stream_turn_config({}))
        self.assertEqual(turn.kind, gt.RESTORE)

    def test_cooldown_blocks_a_pause_request_right_after_restore_needed(self):
        self.dispatch({"cmd": "restore_needed", "id": "stream01", "reason": "seek"})
        self.assertTrue(self.wait_sent("restore_resume"))
        self.sent.clear()
        self.svc._live["ahead_s"] = 5000.0              # a stale pre-seek report would invite a new turn
        host = self.stream_host()
        self.assertFalse(host.stop_restore())
        self.assertEqual(self.sent, [])                 # no restore_pause at all
        self.assertEqual(host.yield_reason, "restore_needed_cooldown")
        self.assertIsNone(host.acquire_lease(True))
        self.now += gh.RESTORE_NEEDED_COOLDOWN_S + 1
        self.svc._live_ts = self.now - 1.0              # keep the report fresh
        self.assertTrue(host.stop_restore())
        self.assertEqual(self.cmds(), ["restore_pause"])

    def test_timeout_reason_waits_longer(self):
        self.dispatch({"cmd": "restore_needed", "id": "stream01", "reason": "timeout"})
        self.assertGreater(self.svc._stream_restore_cooldown_until("stream01") - self.now,
                           gh.RESTORE_NEEDED_COOLDOWN_S)

    def wait_sent(self, cmd, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cmd in self.cmds():
                return True
            time.sleep(0.01)
        return False


if __name__ == "__main__":
    unittest.main()
