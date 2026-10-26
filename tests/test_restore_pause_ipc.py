"""Offline tests for the player side of the restore pause IPC (T4.2): restore_pause / restore_resume routing,
restore_needed, the live report field and its immediate player_busy (fakes only)."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.player import controller as C  # noqa: E402
from oh_streamer.player.restore_pause import RestorePause  # noqa: E402
from oh_streamer.player.queue_bridge import BusyReporter  # noqa: E402


class FakeSession:
    def __init__(self):
        self.state, self.calls = "running", []

    def pause_state(self):
        return self.state

    def pause_restore(self):
        self.calls.append("pause")
        self.state = "pausing"
        return self.state

    def resume_restore(self):
        self.calls.append("resume")
        self.state = "running"
        return self.state


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(**kw):
    sent, logs, clock = [], [], Clock()
    rp = RestorePause(sent.append, logs.append, clock, **kw)
    return rp, sent, logs, clock


def msg(cmd, item="s1", turn="t1"):
    return {"cmd": cmd, "stream_item": item, "turn": turn}


PLAYING = lambda: (100.0, True)  # noqa: E731


class RoutingTests(unittest.TestCase):
    def test_pause_and_resume_reach_the_session(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.assertEqual(rp.submit(msg("restore_pause"), "s1", True), {"ok": True})
        self.assertEqual(s.calls, [])                 # the listener thread only queues
        rp.tick(s, "s1", PLAYING)
        self.assertEqual((s.calls, s.state, rp.turn), (["pause"], "pausing", "t1"))
        self.assertEqual(rp.report(s), {"state": "pausing", "turn": "t1"})
        self.assertEqual(rp.submit(msg("restore_resume"), "s1", True), {"ok": True})
        rp.tick(s, "s1", PLAYING)
        self.assertEqual((s.calls, rp.turn), (["pause", "resume"], None))
        self.assertEqual(rp.report(s), {"state": "running", "turn": None})
        self.assertEqual(sent, [])

    def test_wrong_stream_or_no_session_rejected(self):
        rp, *_ = make()
        bad = {"ok": False, "error": "not this stream"}
        self.assertEqual(rp.submit(msg("restore_pause", item="other"), "s1", True), bad)
        self.assertEqual(rp.submit(msg("restore_pause"), "s1", False), bad)
        self.assertEqual(rp.submit(msg("restore_pause"), None, True), bad)
        self.assertTrue(rp.box.empty())

    def test_stale_turn_resume_still_resumes_and_is_logged(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        rp.submit(msg("restore_pause", turn="t1"), "s1", True)
        rp.tick(s, "s1", PLAYING)
        rp.submit(msg("restore_resume", turn="old"), "s1", True)
        rp.tick(s, "s1", PLAYING)
        self.assertEqual(s.state, "running")
        self.assertTrue(any("stale turn" in m for m in logs))

    def test_reset_drops_token_and_queued_commands(self):
        rp, *_ = make()
        s = FakeSession()
        rp.submit(msg("restore_pause"), "s1", True)
        rp.tick(s, "s1", PLAYING)
        rp.submit(msg("restore_resume"), "s1", True)
        rp.reset()
        self.assertEqual((rp.turn, rp.box.empty()), (None, True))

    def test_controller_on_message_routes_with_current_stream_item(self):
        pl = object.__new__(C.PlayerController)
        pl.restore_pause, pl.stream_item, pl.session = RestorePause(lambda p: None), "s1", object()
        self.assertEqual(pl.on_message(msg("restore_pause")), {"ok": True})
        self.assertEqual(pl.on_message(msg("restore_resume", item="x")), {"ok": False, "error": "not this stream"})
        pl.session = None
        self.assertEqual(pl.on_message(msg("restore_pause")), {"ok": False, "error": "not this stream"})


class ReservedMibTests(unittest.TestCase):
    def test_report_carries_reserved_mib_only_while_paused(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        s.pause_reserved_mib = lambda: 1505.5
        rp.submit(msg("restore_pause"), "s1", True)
        rp.tick(s, "s1", PLAYING)
        self.assertEqual(rp.report(s), {"state": "pausing", "turn": "t1"})
        s.state = "paused"
        self.assertEqual(rp.report(s), {"state": "paused", "turn": "t1", "reserved_mib": 1505.5})

    def test_unknown_or_junk_reserved_mib_is_left_out(self):
        rp, *_ = make()
        s = FakeSession()
        s.state = "paused"
        for value in (None, -1, float("nan"), "x", True):
            s.pause_reserved_mib = lambda v=value: v
            self.assertEqual(rp.report(s), {"state": "paused", "turn": None})
        self.assertEqual(rp.report(FakeSession()), {"state": "running", "turn": None})


class StateChangeTests(unittest.TestCase):
    def test_state_change_makes_tick_report_and_busy_reporter_sends_now(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        out = []
        busy = BusyReporter(lambda b, **extra: out.append(extra), clock=clock, live_interval=2.0)
        busy.live = lambda: {"restore_pause": rp.report(s)}
        busy.update(False)
        self.assertEqual(len(out), 1)
        clock.t += 0.1                                # inside the 2 s live interval: nothing yet
        busy.update(False)
        self.assertEqual(len(out), 1)
        rp.submit(msg("restore_pause"), "s1", True)
        self.assertTrue(rp.tick(s, "s1", PLAYING))    # running -> pausing
        busy.next_live = 0.0                          # what the controller does on a change
        busy.update(False)
        self.assertEqual(out[-1]["live"]["restore_pause"], {"state": "pausing", "turn": "t1"})
        s.state = "paused"                            # the pause thread finished
        self.assertTrue(rp.tick(s, "s1", PLAYING))
        self.assertFalse(rp.tick(s, "s1", lambda: (100.0, False)))


class NeededTests(unittest.TestCase):
    def pause(self, rp, s):
        rp.submit(msg("restore_pause"), "s1", True)
        rp.tick(s, "s1", lambda: (999.0, True))

    def test_seek_while_paused_resumes_on_queue_resume(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        sleeps = []

        def sleep(d):
            sleeps.append(d)
            clock.sleep(d)
            if len(sleeps) == 3:
                rp.submit(msg("restore_resume"), "s1", True)
        self.assertEqual(rp.wait_for_restart(s, "s1", "seek", sleep=sleep), "queue")
        self.assertEqual(sent, [{"cmd": "restore_needed", "id": "s1", "reason": "seek"}])
        self.assertEqual((s.state, len(sleeps)), ("running", 3))

    def test_seek_while_paused_resumes_after_3s_without_answer(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        t0 = clock.t
        self.assertEqual(rp.wait_for_restart(s, "s1", "seek", sleep=clock.sleep), "timeout")
        self.assertGreaterEqual(clock.t - t0, 3.0)
        self.assertLess(clock.t - t0, 3.2)
        self.assertEqual((s.state, rp.turn), ("running", None))

    def test_seek_without_pause_does_nothing(self):
        rp, sent, logs, clock = make()
        self.assertEqual(rp.wait_for_restart(FakeSession(), "s1", "seek", sleep=clock.sleep), "none")
        self.assertEqual((sent, clock.t), ([], 100.0))

    def test_buffer_low_only_while_playing(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        clock.t += 2
        rp.tick(s, "s1", lambda: (10.0, False))       # lead low but the video is paused by the user
        self.assertEqual(sent, [])
        clock.t += 2
        rp.tick(s, "s1", lambda: (45.0, True))        # playing, lead fine
        self.assertEqual(sent, [])
        clock.t += 2
        rp.tick(s, "s1", lambda: (10.0, True))
        self.assertEqual(sent, [{"cmd": "restore_needed", "id": "s1", "reason": "buffer_low"}])
        clock.t += 2
        rp.tick(s, "s1", lambda: (10.0, True))        # waiting for the answer: not sent twice
        self.assertEqual(len(sent), 1)
        self.assertEqual(s.state, "pausing")

    def test_buffer_low_resumes_after_3s_without_answer(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        clock.t += 2
        rp.tick(s, "s1", lambda: (10.0, True))
        clock.t += 3.1
        self.assertTrue(rp.tick(s, "s1", lambda: (10.0, True)))
        self.assertEqual((s.state, rp.turn), ("running", None))
        self.assertTrue(any("resuming anyway" in m for m in logs))

    def test_buffer_low_stops_waiting_on_queue_resume(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        clock.t += 2
        rp.tick(s, "s1", lambda: (10.0, True))
        rp.submit(msg("restore_resume"), "s1", True)
        clock.t += 1
        rp.tick(s, "s1", lambda: (10.0, True))
        self.assertEqual((s.state, rp.need_deadline), ("running", None))

    def test_timeout_after_900s(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        clock.t += 899
        rp.tick(s, "s1", lambda: (999.0, False))
        self.assertEqual(sent, [])
        clock.t += 2
        rp.tick(s, "s1", lambda: (999.0, False))
        self.assertEqual(sent, [{"cmd": "restore_needed", "id": "s1", "reason": "timeout"}])

    def test_session_clearing_the_pause_drops_the_token(self):
        rp, sent, logs, clock = make()
        s = FakeSession()
        self.pause(rp, s)
        s.state = "running"                           # Session.start / stop cleared it
        self.assertTrue(rp.tick(s, "s1", PLAYING))
        self.assertIsNone(rp.turn)


if __name__ == "__main__":
    unittest.main()
