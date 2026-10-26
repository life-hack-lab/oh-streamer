"""
Restore pause of the live-mode player (T4.2, queue_ipc.py "restore_pause" / "restore_needed"): the queue asks the player
to hold its detector so a local subtitle (ASR) turn can use the GPU, and the player asks for the GPU back when it
needs it. Stdlib only; the Session does the real work (Session.pause_restore / resume_restore / pause_state).

Threads: submit() is the listener thread (never blocks, only queues); everything else runs on the controller loop.
`send(payload)` must not block (QueueLink.post). The clock is injectable for tests.
"""
import queue
import time

NEED_WAIT_S = 3.0          # the player waits this long for the queue's restore_resume after restore_needed
BUFFER_LOW_S = 30.0        # restored lead (ahead_s) below this while paused and playing -> restore_needed "buffer_low"
PAUSE_TIMEOUT_S = 900.0    # paused this long -> restore_needed "timeout"
CHECK_INTERVAL = 1.0       # mpv is read at most this often while a pause lasts


class RestorePause:
    def __init__(self, send, log=lambda m: None, clock=time.monotonic, wait_s: float = NEED_WAIT_S,
                 buffer_low_s: float = BUFFER_LOW_S, timeout_s: float = PAUSE_TIMEOUT_S):
        self._send, self._log, self._clock = send, log, clock
        self.wait_s, self.buffer_low_s, self.timeout_s = wait_s, buffer_low_s, timeout_s
        self.box = queue.Queue()      # (cmd, turn) from the listener thread
        self.turn = None              # the queue's token of the pause being held (None = not paused by the queue)
        self.paused_at = None
        self.need_deadline = None     # restore_needed sent, waiting for restore_resume until then
        self.last_state = "running"
        self._next_check = 0.0

    # -- listener thread -----------------------------------------------------------------------------------------

    def submit(self, msg: dict, stream_item, has_session: bool) -> dict:
        """Reply to a queue -> player restore_pause / restore_resume at once; the loop applies it."""
        if not has_session or not stream_item or str(msg.get("stream_item")) != str(stream_item):
            return {"ok": False, "error": "not this stream"}
        self.box.put((msg.get("cmd"), msg.get("turn")))
        return {"ok": True}

    # -- controller loop -----------------------------------------------------------------------------------------

    def _drain(self, session) -> bool:
        """Apply queued commands; True if a resume was handled."""
        resumed = False
        while True:
            try:
                cmd, turn = self.box.get_nowait()
            except queue.Empty:
                return resumed
            if session is None:
                continue
            if cmd == "restore_pause":
                self.turn, self.paused_at, self.need_deadline = turn, self._clock(), None
                self._log(f"restore pause (queue turn {turn}): {session.pause_restore()}")
            elif cmd == "restore_resume":
                if self.turn is None or turn != self.turn:
                    self._log(f"restore resume with {'no pause held' if self.turn is None else 'a stale turn'}"
                              f" (got {turn}, held {self.turn}): resuming anyway")
                session.resume_restore()
                self._clear()
                resumed = True

    def _clear(self):
        self.turn, self.paused_at, self.need_deadline = None, None, None

    def reset(self):
        """The source ended / was replaced / closed (Session.stop already clears the pause): drop the token."""
        while True:
            try:
                self.box.get_nowait()
            except queue.Empty:
                break
        self._clear()
        self.last_state = "running"

    def report(self, session) -> dict | None:
        """The live report's "restore_pause" field."""
        fn = getattr(session, "pause_state", None)
        if fn is None:
            return None
        state = fn()
        out = {"state": state, "turn": self.turn if state != "running" else None}
        reserved = getattr(session, "pause_reserved_mib", None)
        mib = reserved() if callable(reserved) and state == "paused" else None
        if isinstance(mib, (int, float)) and not isinstance(mib, bool) and mib >= 0:
            out["reserved_mib"] = float(mib)      # what the paused restoration still holds (the queue learns it)
        return out

    def tick(self, session, stream_item, probe) -> bool:
        """Once per loop iteration: apply commands, send restore_needed when the player wants the GPU back, resume
        after the wait. probe() -> (ahead_s | None, playing) is called only while a pause lasts. Returns True when
        the pause state changed (the caller sends the live report at once)."""
        if session is None or not callable(getattr(session, "pause_state", None)):
            return False       # no pausable session (passthrough / test fakes): nothing to pause
        self._drain(session)
        state = session.pause_state()
        now = self._clock()
        if state == "running":
            if self.turn is not None:
                self._clear()          # the session cleared it (new segment / stop)
        elif self.need_deadline is not None:
            if now >= self.need_deadline:
                self._log(f"restore_needed: no restore_resume within {self.wait_s:.1f}s, resuming anyway")
                session.resume_restore()
                self._clear()
        elif now >= self._next_check:
            self._next_check = now + CHECK_INTERVAL
            reason = None
            if self.paused_at is not None and now - self.paused_at >= self.timeout_s:
                reason = "timeout"
            else:
                try:
                    ahead, playing = probe()
                except Exception as e:  # noqa: BLE001 - never affects playback
                    self._log(f"restore pause probe skipped: {e!r}")
                    ahead, playing = None, False
                if playing and ahead is not None and ahead < self.buffer_low_s:
                    reason = "buffer_low"
            if reason is not None:
                self._send_need(stream_item, reason)
                self.need_deadline = self._clock() + self.wait_s
        state = session.pause_state()
        changed = state != self.last_state
        self.last_state = state
        return changed

    def _send_need(self, stream_item, reason):
        self._log(f"restore_needed ({reason}): asking the queue for the GPU back")
        try:
            self._send({"cmd": "restore_needed", "id": stream_item, "reason": reason})
        except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
            self._log(f"restore_needed not sent: {e!r}")

    def wait_for_restart(self, session, stream_item, reason: str = "seek", sleep=time.sleep) -> str:
        """A segment restart (seek / quality / settings) is about to start restoring: while the pause lasts, send
        restore_needed and wait up to wait_s for restore_resume, then resume anyway. Blocks (handle_requests already
        blocks on stop / probe, and mpv is frozen for the restart). Returns "none" | "queue" | "timeout"."""
        if session is None:
            return "none"
        self._drain(session)
        if session.pause_state() == "running":
            return "none"
        self._send_need(stream_item, reason)
        deadline = self._clock() + self.wait_s
        while True:
            if self._drain(session) or session.pause_state() == "running":
                self._log("restore_needed (restart): resumed by the queue")
                self._clear()
                return "queue"
            if self._clock() >= deadline:
                break
            sleep(0.05)
        self._log(f"restore_needed (restart): no restore_resume within {self.wait_s:.1f}s, resuming anyway")
        session.resume_restore()
        self._clear()
        return "timeout"
