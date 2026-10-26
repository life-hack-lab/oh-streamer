"""
Startup detector gate for oh_player: holds Lada's detector back while a start / seek is waiting for its first clip.

After a start / seek nothing is playable until the first mosaic clip covering the target is detected AND restored,
but the detector keeps racing 1000-2800 frames ahead and shares the GPU with that restoration, which then runs at
about half speed (bench_startup.py: 27 -> 58 clip frames/s, seek wait for a 300-frame first clip ~15 s -> ~8 s).
So, until the segment is playable, the detector may not run further than `limit` frames ahead of the restored output:

- DetectGate wraps put of a FrameRestorer's frame_detection_queue (instance attribute; lada itself is unchanged): a
  per-frame result waits while the gate is closed and detected - output > limit. Markers (EOF / STOP) never wait.
- It also wraps put of the frame_restoration_queue to count the output frames; the gate opens by itself once
  release_out_n of them came out, or when the caller says so (open(): the target plus the prebuffer is output, the
  segment is stopped, ...). Then the detector runs freely again.

limit is never below max_clip_length + MARGIN: a clip only closes max_clip_length frames after its first frame (one
more to see the scene is over), and the restorer waits for it, so a smaller limit would deadlock. As a second
safety valve the gate opens after WATCHDOG_S seconds in which the detector waited and no frame came out.

Independent of that limit there is a hard hold (hold() / release(), used to give the GPU to another job): while held,
no per-frame result passes at all, whatever the limit. Frames already detected keep flowing through the restorer. The
watchdog never opens a hold; reset() and open("stopped" / "closed") release it so a stop cannot deadlock.

The wrappers chain with other hooks on the same queues (buffer_progress.PipelineProbe), in either attach order.
"""

import threading
import time

from oh_streamer.engines.lada import probes as lada_probes

MARGIN = 30           # frames beyond max_clip_length the detector may run ahead (a clip closes one frame after it ends)
WATCHDOG_S = 30.0     # the detector waited this long and the output did not advance: open the gate (a 300-frame
                      # first clip took up to ~20 s end to end; a real deadlock would never recover, so be generous)


HOLD_RELEASING = ("stopped", "closed")    # open() reasons that also lift a hard hold (segment / stream ended)


class DetectGate:
    """Per-restorer detector gate (see the module docstring)."""

    enabled = True    # class default (--detect-gate / --no-detect-gate); False: reset() leaves the gate open

    def __init__(self, restorer):
        self.restorer = restorer
        self.cond = threading.Condition()
        self.watchdog_s = WATCHDOG_S
        self.on_held = None           # optional callable(t_start, t_end): the detector waited from t_start to t_end
        self.limit, self.release_out_n = restorer.max_clip_length + MARGIN, None
        self.det_n = self.out_n = self.max_ahead = 0
        self.held_s = 0.0
        self.t_out = time.perf_counter()
        self.is_open, self.reason, self.open_out_n = True, "not started", 0
        self.hold_reason, self.hold_t = None, 0.0    # hard hold (see hold()); hold_t: perf_counter when it was set

    @classmethod
    def attach(cls, restorer):
        """The gate of `restorer` (created and hooked on first use), or None if it has no such queues."""
        g = getattr(restorer, "_lada_detect_gate", None)
        if g is not None:
            return g
        try:
            det_q, out_q = lada_probes.gate_queues(restorer)
        except AttributeError:
            return None
        g = cls(restorer)
        g._wrap_det_put(det_q)
        g._wrap_out_put(out_q)
        restorer._lada_detect_gate = g
        return g

    def reset(self, limit: int, release_out_n: int | None = None):
        """A segment starts (call right before FrameRestorer.start): close the gate (open if disabled).

        limit: frames the detector may be ahead of the output, raised to max_clip_length + MARGIN;
        release_out_n: open by itself once this many frames came out (None: only open() does)."""
        with self.cond:
            self.limit = max(int(limit), self.restorer.max_clip_length + MARGIN)
            self.release_out_n = release_out_n
            self.det_n = self.out_n = self.max_ahead = 0
            self.held_s = 0.0
            self.t_out = time.perf_counter()
            self.is_open, self.reason, self.open_out_n = False, "", 0
            self.hold_reason = None      # a new segment starts unheld
            if not self.enabled:
                self._open("off")
            self.cond.notify_all()

    def open(self, reason: str):
        """Let the detector run freely (idempotent: the first reason is kept)."""
        with self.cond:
            if reason in HOLD_RELEASING:
                self.hold_reason = None
            self._open(reason)
            self.cond.notify_all()

    def hold(self, reason: str = "hold") -> bool:
        """Hard hold: from now on every per-frame put into the detection queue waits (markers never do) until
        release(), open("stopped" / "closed") or reset(). Returns False if it was already held."""
        with self.cond:
            if self.hold_reason is not None:
                return False
            self.hold_reason, self.hold_t = reason, time.perf_counter()
            return True

    def release(self) -> bool:
        """Lift the hard hold; returns False if there was none."""
        with self.cond:
            was = self.hold_reason is not None
            self.hold_reason = None
            self.cond.notify_all()
            return was

    @property
    def held(self) -> bool:
        return self.hold_reason is not None

    def _open(self, reason: str):    # with self.cond held
        if not self.is_open:
            self.is_open, self.reason, self.open_out_n = True, reason, self.out_n
            self.cond.notify_all()

    def summary(self) -> str:
        """For the timing log."""
        with self.cond:
            if not self.enabled:
                return "detect gate: off"
            state = f"opened at out {self.open_out_n} ({self.reason})" if self.is_open else "still closed"
            return f"detect gate: held {self.held_s:.1f}s, max ahead {self.max_ahead}, {state}"

    # -- hooks (Lada's worker threads) -----------------------------------------------------------------------------

    def _wrap_det_put(self, q):
        orig = q.put

        def put(item, *a, **kw):
            if isinstance(item, tuple):
                try:
                    self._admit()
                except Exception:  # noqa: BLE001 - instrumentation must never break the pipeline
                    pass
            return orig(item, *a, **kw)    # without the lock: the real put may block on a full queue
        q.put = put

    def _wrap_out_put(self, q):
        orig = q.put

        def put(item, *a, **kw):
            if isinstance(item, tuple):
                try:
                    self._on_output()
                except Exception:  # noqa: BLE001
                    pass
            return orig(item, *a, **kw)
        q.put = put

    def _admit(self):
        """One more detected frame: wait while the detector is too far ahead of the output."""
        t_block = now = None
        with self.cond:
            while self.hold_reason is not None and not getattr(self.restorer, "stop_requested", False):
                self.cond.wait(0.05)     # hard hold: not counted as held time, no watchdog
            while not self.is_open and not getattr(self.restorer, "stop_requested", False) \
                    and self.det_n - self.out_n > self.limit:
                now = time.perf_counter()
                t_block = t_block or now
                if now - max(t_block, self.t_out) > self.watchdog_s:
                    self._open("watchdog")
                    break
                self.cond.wait(0.05)
            self.det_n += 1
            self.max_ahead = max(self.max_ahead, self.det_n - self.out_n)
            if t_block:
                now = time.perf_counter()
                self.held_s += now - t_block
        if t_block and self.on_held is not None:
            self.on_held(t_block, now)

    def _on_output(self):
        with self.cond:
            self.out_n += 1
            self.t_out = time.perf_counter()
            if self.release_out_n is not None and self.out_n >= self.release_out_n:
                self._open("output reached")
            self.cond.notify_all()
