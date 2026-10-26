"""
Seek tail for oh_player: the old timeline keeps playing while a seek outside mpv's cache is being prepared.

A far seek used to freeze the picture until Lada had restored enough at the target (seconds; longer in mosaic-heavy
stretches, where Lada runs at 0.6-1.1x realtime). Now, while the user is not paused:

1. The segment mpv is playing hands its ffmpeg muxer over (Session.detach_output) and its Lada pipeline is stopped at
   once (fast stop), so the GPU goes to the new position. The muxer and its TCP stream stay open, and mpv just keeps
   playing: first the restored video it has cached already (with audio; often only 0-8 s in mosaic-heavy stretches),
2. then the passthrough: this module's thread decodes the source from the frame after the last restored one and
   writes those frames, unrestored, into the same muxer (same size / pixel format / frame rate as Lada's output, audio
   keeps coming from ffmpeg's own input), so mpv sees one continuous stream. The thread stays only `lead` seconds (of
   frames written) ahead of mpv's playback position, which the controller thread samples (it owns the mpv IPC) and
   hands over with observe(); so the tail can end quickly. The source is only opened shortly before the restored video
   runs out (an idle HTTP connection could be dropped meanwhile).
3. Meanwhile Lada starts at the target (a new Session segment, detect gate as usual). Once its output is far enough past
   the target (counted on Lada's side, Session.output_reached: the new muxer takes only a few frames before mpv
   connects, the rest waits in Lada's output queue), the controller appends the new segment to mpv's playlist and calls finish(): the passthrough stops and
   the muxer's stdin is closed (no kill: nothing written is lost), so mpv reaches the end of the old stream within
   about the lead and continues with the next playlist entry (--prefetch-playlist: ~0.1 s gap, time-pos continuous).
   If much restored video is still ahead, the controller replaces instead (loadfile replace: 0.5-0.9 s, the last
   frame stays on screen) rather than playing all of it first.
4. A newer seek before the switch keeps the tail playing and only restarts the segment being prepared. close() discards
   a tail (quality / settings change, source switch, casting, exit): passthrough stopped, muxer killed and reaped.

If decoding fails, the source ends or MAX_PASSTHROUGH_S of passthrough were written, the tail closes the muxer's stdin
itself: mpv plays to the end of what it got and keep-open pauses there (as when a segment ends); the switch is then a
replace once the new segment is ready.

Pacing never deadlocks against mpv's cache-pause: while mpv waits for data (paused-for-cache) the passthrough writes
regardless of the lead. Writes into the muxer take a lock that finish() / close() also take, so the stdin is never
closed under a write; close() kills the muxer first, which unblocks a write that cannot finish.

The frame source is injected (oh_player passes one built on Lada's patched VideoReader, so HTTP sources with headers
work too) and the pacing is a pure function: this module needs neither PyAV nor torch (test_seek_tail.py).
"""

import threading
import time

LEAD_S = 2.0               # passthrough written ahead of mpv's playback position (seconds of video)
PREPARE_S = 4.0            # open the source this long before the passthrough is needed (open + seek + GOP decode)
MAX_PASSTHROUGH_S = 120.0  # end the tail after this much unrestored video (the new segment should long be ready)
WRITE_LOCK_TIMEOUT = 2.0   # finish(): a write still blocked after this long is cut off by killing the muxer
APPEND_MAX_REMAINING_S = 4.0   # switch by appending only if mpv has at most this much of the tail left to play;
                               # with more restored video ahead, replace (reaching the target matters more)


def switch_mode(remaining_s: float) -> str:
    """How to hand over from a tail mpv is playing to the new segment: "append" (seamless, after the rest of the tail)
    or "replace" (at once, the last frame stays on screen for 0.5-0.9 s)."""
    return "append" if remaining_s <= APPEND_MAX_REMAINING_S else "replace"


def wait_before_write(t_next: float, pos: float | None, buffering: bool, lead: float) -> float:
    """Seconds to wait before writing the frame at source time t_next (0 = write it now).

    pos: mpv's playback position (extrapolated; None = not sampled yet); buffering: mpv is waiting for data
    (paused-for-cache), so it gets frames whatever the lead (it would otherwise wait for a cache that never comes)."""
    if buffering:
        return 0.0
    if pos is None:
        return 0.1
    return max(0.0, t_next - (pos + lead))


class PlayClock:
    """mpv's playback position as last observed by the controller, extrapolated while it plays (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._s = None        # (position, advancing, buffering, perf_counter time)

    def update(self, pos: float, advancing: bool, buffering: bool, now: float | None = None):
        with self._lock:
            self._s = (float(pos), bool(advancing), bool(buffering), time.perf_counter() if now is None else now)

    def sample(self, now: float) -> tuple[float | None, bool]:
        """(position at `now`, buffering)."""
        with self._lock:
            s = self._s
        if s is None:
            return None, False
        pos, advancing, buffering, t = s
        return (pos + max(0.0, now - t) if advancing else pos), buffering

    def last(self) -> float | None:
        with self._lock:
            return self._s[0] if self._s is not None else None


class SeekTail:
    """An old segment's muxer, taken over from its Session, plus the passthrough thread (see the module docstring).

    muxer: the running ffmpeg (subprocess.Popen with a binary stdin; it may already be closed when the segment had
    reached the end of the video: then there is no passthrough). next_t: source time of the first frame the
    passthrough writes (one frame after the last restored one). open_frames(t) -> iterator of (frame ndarray,
    source time or None) from source time t on (the caller's reader; closed with .close() if it has one).
    rate_mult 2: the muxer runs at twice the frame rate (interpolation): every frame goes in as (previous, frame),
    and the muxer drops the copy again, as with the pump's copies. layers > 1 (compare): the muxer takes stacked
    frames, and as a passthrough frame is unrestored (original == processed == restored) it is written `layers` times."""

    def __init__(self, muxer, url: str, start_pos: float, next_t: float, fps: float, open_frames, log,
                 rate_mult: int = 1, lead: float = LEAD_S, prepare_s: float = PREPARE_S,
                 max_s: float = MAX_PASSTHROUGH_S, clock: PlayClock | None = None, layers: int = 1):
        self.muxer, self.url, self.fps, self.rate_mult = muxer, url, float(fps), rate_mult
        self.layers = max(1, int(layers))   # compare: the pipe frame is `layers` stacked copies (passthrough: all equal)
        self.start_pos, self.next_t = float(start_pos), float(next_t)
        self.open_frames, self.log = open_frames, log
        self.lead, self.prepare_s, self.max_frames = lead, prepare_s, int(max_s * float(fps))
        self.clock = clock or PlayClock()
        self.clock.update(start_pos, True, False)
        self.frames = 0                 # passthrough frames written (source frames; 2 muxer frames each at 2x)
        self.t_passthrough = None       # perf_counter time of the first passthrough frame
        self.end_reason = None          # why the stream was ended ("finished", "source ended", ...)
        self._stop = threading.Event()
        self._wlock = threading.Lock()  # one write at a time; finish() / close() never close stdin under a write
        self._stdin_closed = getattr(muxer.stdin, "closed", False)
        if self._stdin_closed:
            self.end_reason = "segment had ended"
        self._thread = None
        self._killed = False

    # -- controller thread -----------------------------------------------------------------------------------------

    def start(self):
        if not self._stdin_closed:
            self._thread = threading.Thread(target=self._run, name="seek tail", daemon=True)
            self._thread.start()
        return self

    def observe(self, pos: float, advancing: bool, buffering: bool, now: float | None = None):
        """mpv's state while it plays this tail (controller thread, a few times a second)."""
        self.clock.update(pos, advancing, buffering, now)

    @property
    def written_end(self) -> float:
        """Source time right after the last frame in the stream (restored or passthrough)."""
        return self.next_t + self.frames / self.fps

    def remaining(self, pos: float) -> float:
        """Seconds of video mpv still has to play from pos to the end of what was written."""
        return self.written_end - pos

    @property
    def ended(self) -> bool:
        """No more frames go into the stream (its stdin is closed)."""
        return self._stdin_closed

    def finish(self, reason: str = "switched"):
        """End the stream gracefully: stop the passthrough, close the muxer's stdin (mpv plays it to its end)."""
        self._stop.set()
        if self._wlock.acquire(timeout=WRITE_LOCK_TIMEOUT):
            try:
                self._close_stdin(reason)
            finally:
                self._wlock.release()
        else:   # a write that does not finish (nobody reads the stream): cut it off
            self.log(f"tail: a write did not finish within {WRITE_LOCK_TIMEOUT:g}s; killing the old muxer")
            self._kill()
            self._close_stdin(reason)

    def close(self, reason: str = "discarded"):
        """Discard the tail: stop the passthrough, kill the muxer (also unblocks a write) and reap it."""
        self._stop.set()
        self._kill()
        with self._wlock:
            self._close_stdin(reason)
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # a decode blocked on the network ends on its own (daemon thread)

    def summary(self, final_pos: float | None = None) -> str:
        """For the timing log: restored video played, passthrough written / played."""
        last = self.clock.last() if final_pos is None else final_pos
        last = self.start_pos if last is None else last
        restored = max(0.0, min(last, self.next_t) - self.start_pos)
        pt_played = max(0.0, min(last, self.written_end) - self.next_t)
        return (f"tail: {restored:.1f}s restored played, passthrough {self.frames / self.fps:.1f}s written / "
                f"{pt_played:.1f}s played ({self.end_reason or 'running'})")

    # -- internals ---------------------------------------------------------------------------------------------------

    def _kill(self):
        p = self.muxer
        if p.poll() is None:
            self._killed = True
            try:
                p.kill()
            except OSError:
                pass
        try:
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001 - a fake / a process that does not go away: nothing more to do
            pass

    def _close_stdin(self, reason: str):   # with self._wlock held
        if self._stdin_closed:
            return
        self._stdin_closed = True
        self.end_reason = self.end_reason or reason
        try:
            self.muxer.stdin.close()
        except (OSError, ValueError):
            pass

    def _wait_until(self, t: float, lead: float) -> bool:
        """Wait until the frame at source time t may be written with `lead`; False = stop requested."""
        while not self._stop.is_set():
            pos, buffering = self.clock.sample(time.perf_counter())
            w = wait_before_write(t, pos, buffering, lead)
            if w <= 0:
                return True
            self._stop.wait(min(w, 0.1))
        return False

    def _run(self):
        it, reason = None, None
        try:
            # open the source only shortly before the restored video runs out
            if not self._wait_until(self.next_t, self.lead + self.prepare_s):
                return
            it = self.open_frames(self.next_t)
            prev, expect = None, self.next_t
            for arr, t in it:
                t = expect if t is None else t
                expect = t + 1.0 / self.fps
                if not self._wait_until(t, self.lead):
                    return
                if self.frames >= self.max_frames:
                    reason = f"passthrough limit {self.max_frames / self.fps:.0f}s"
                    break
                with self._wlock:
                    if self._stdin_closed:
                        return
                    outs = (arr,) if self.rate_mult == 1 else (arr if prev is None else prev, arr)
                    for out in outs:
                        for _ in range(self.layers):
                            self.muxer.stdin.write(memoryview(out.reshape(-1)))
                if self.frames == 0:
                    self.t_passthrough = time.perf_counter()
                    self.log(f"tail: passthrough started at {t:.3f}s (restored video ran to {self.next_t:.3f}s)")
                self.frames += 1
                prev = arr
            else:
                reason = "source ended"
        except (OSError, ValueError) as e:     # the muxer went away (killed by close(), or mpv disconnected)
            reason = None if self._stop.is_set() else f"output closed ({e!r})"
        except Exception as e:  # noqa: BLE001 - decoding failed: mpv plays what it has and pauses at its end
            reason = f"passthrough failed: {e!r}"
        finally:
            if it is not None and hasattr(it, "close"):
                try:
                    it.close()      # a generator: runs its cleanup (closes the reader) in this thread
                except Exception:  # noqa: BLE001
                    pass
            if reason is not None:
                self.log(f"tail: {reason}; ending the old stream at {self.written_end:.3f}s")
                with self._wlock:
                    self._close_stdin(reason)
