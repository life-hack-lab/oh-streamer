"""
Buffering progress for oh_player: how far a start / seek / switch is from being playable, as a percentage plus
the current stage (shown by oh_seek.lua instead of a bare "處理中…").

Nothing reports real progress here: the first frame at the target only comes out once every mosaic clip covering
it is detected (a clip closes when its scene ends or reaches max_clip_length frames) and restored, then mpv still
wants cache-pause-wait seconds cached. So we watch Lada's pipeline from outside and estimate the remaining time:

- PipelineProbe wraps put/get of a FrameRestorer's queues (instance attributes; lada itself is unchanged) and
  records, per segment, the frames detected (and which had mosaics), the clips cut and when each one's restoration
  started / finished.
- estimate() turns that, the frames the segment has output and mpv's cache into the remaining time, using speeds
  measured in this segment or remembered from earlier ones (RATES), and reports elapsed / (elapsed + remaining).

The estimate assumes the worst case for a scene that is still open (it runs to max_clip_length), so it tends to
under-promise and then jump ahead when the scene ends early; the controller keeps the shown value monotonic.

The probe also keeps the clips that are cut but not restored yet (and a scene still being collected) for the whole
segment: PipelineProbe.restore_margin() tells the adaptive interpolation (rife_interp.py) how much of its lead the
restoration ahead will need, since Lada outputs nothing past a mosaic clip until all of it is restored.
"""

import threading
import time

from oh_streamer.engines.lada import probes as lada_probes

DEFAULT_DET_FPS = 150.0     # frames/s through the detector before anything was measured
DEFAULT_RES_FPS = 50.0      # clip frames/s through the restoration model (RTX 5060 laptop; ~55 for the first clip of
                            # a segment with the detector held back by detect_gate.py, ~27 while it runs freely)
DEFAULT_P_MOSAIC = 0.6      # share of starts whose first frames need a clip restored
DEFAULT_FIRST_DET = 1.0     # seconds from segment start to the first detection result (open + seek + first batch)
DEFAULT_PREROLL_S = 1.0     # seconds between the keyframe Lada starts at and the target
MAX_TRACK = 20000           # mosaic frame indices kept per segment (only the first few hundred matter)


class _Rates:
    """Pipeline speeds remembered across segments (exponential averages)."""

    def __init__(self):
        self.det_fps = None
        self.res_fps = None
        self.first_det = None
        self.preroll_s = None
        self.p_mosaic = None    # share of starts that had to wait for a clip covering the target

    @staticmethod
    def blend(old, new, k=0.4):
        return new if old is None else old + k * (new - old)


RATES = _Rates()


def _wrap_put(q, hook):
    orig = q.put

    def put(item, *a, **kw):
        try:
            hook(item)
        except Exception:  # noqa: BLE001 - instrumentation must never break the pipeline
            pass
        return orig(item, *a, **kw)
    q.put = put


def _wrap_get(q, hook):
    orig = q.get

    def get(*a, **kw):
        item = orig(*a, **kw)
        try:
            hook(item)
        except Exception:  # noqa: BLE001
            pass
        return item
    q.get = get


class PipelineProbe:
    """Per-restorer recorder of the current segment's pipeline events (see the module docstring)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frozen = True
        self.reset(0.0)
        self.frozen = True

    @classmethod
    def attach(cls, restorer):
        """The probe of `restorer` (created and hooked on first use), or None if it has no such queues."""
        p = getattr(restorer, "_lada_progress_probe", None)
        if p is not None:
            return p
        try:
            det_q, clip_q, done_q = lada_probes.progress_queues(restorer)
        except AttributeError:
            return None
        p = cls()
        _wrap_put(det_q, p._on_detection)
        _wrap_put(clip_q, p._on_clip)
        _wrap_get(clip_q, p._on_clip_taken)
        _wrap_put(done_q, p._on_clip_done)
        restorer._lada_progress_probe = p
        return p

    def reset(self, target_s: float, t0: float | None = None):
        """A segment starts at target_s (call right before FrameRestorer.start)."""
        with self.lock:
            self.t0 = t0 or time.perf_counter()
            self.target = target_s
            self.base = None              # Lada frame number of the first detected frame (index 0)
            self.det = 0                  # frames detected (indices 0..det-1)
            self.t_first_det = self.t_last_det = None
            self.last_mosaic = False      # the latest detected frame had mosaics
            self.held = []                # (t start, t end) the detector was held back (detect_gate.py), not slow
            self.mosaic = []              # indices of frames with mosaics (ascending)
            self.clips = {}               # id(clip) -> [start idx, end idx, t restoration started, t done]
            self.pending = {}             # id(clip) -> [start idx, end idx, t restoration started]: not restored yet
            self.cut_end = -1             # highest end index of the clips cut so far
            self.open_start = None        # first mosaic frame after cut_end: a scene not cut into a clip yet
            self.last_mosaic_idx = -1
            self.frozen = False

    def freeze(self):
        """The segment is being stopped: ignore what the draining pipeline does (cancelled restorations etc.)."""
        with self.lock:
            self.frozen = True

    def started(self, out_n: int):
        """The first frame at the target was output (the out_n-th frame of the segment): remember this segment's
        detection speed and whether it had to wait for a clip."""
        with self.lock:
            first = min((c for c in self.clips.values() if c[0] < out_n), default=None, key=lambda c: c[0])
            RATES.p_mosaic = RATES.blend(RATES.p_mosaic if RATES.p_mosaic is not None else DEFAULT_P_MOSAIC,
                                         1.0 if first else 0.0, k=0.3)
            span = self._active_span()
            if self.det >= 24 and span > 0.2:
                RATES.det_fps = RATES.blend(RATES.det_fps, (self.det - 1) / span)

    def note_detector_held(self, t_start: float, t_end: float):
        """The detector was held back from t_start to t_end (the gate calls this; it says nothing about speed)."""
        with self.lock:
            if not self.frozen and len(self.held) < MAX_TRACK:
                self.held.append((t_start, t_end))

    def _active_span(self) -> float:
        """Seconds between the first and the latest detection, minus the time the detector was held back (lock held;
        a hold may be recorded before or after the frame it delayed, so only its part inside the span counts)."""
        if self.t_first_det is None:
            return 0.0
        a, b = self.t_first_det, self.t_last_det
        return b - a - sum(max(0.0, min(e, b) - max(s, a)) for s, e in self.held)

    def summary(self) -> str:
        """For the timing log: what the pipeline did until now (relative to the segment start)."""
        with self.lock:
            if self.base is None:
                return "nothing detected"
            parts = [f"first detection +{self.t_first_det - self.t0:.2f}s",
                     f"{self.det} frames detected ({len(self.mosaic)} with mosaics) by +{self.t_last_det - self.t0:.2f}s"]
            for s, e, ts, td in sorted(self.clips.values())[:4]:
                r = f"clip {s}-{e}"
                if ts is not None:
                    r += f" restore +{ts - self.t0:.2f}s" + (f"..+{td - self.t0:.2f}s" if td is not None else "…")
                parts.append(r)
            if len(self.clips) > 4:
                parts.append(f"+{len(self.clips) - 4} clips")
            return " | ".join(parts)

    def note_first_output(self, t: float):
        """The very first frame Lada output (possibly before the target: it starts at a keyframe) is at t."""
        RATES.preroll_s = RATES.blend(RATES.preroll_s, max(0.0, self.target - t))

    # -- hooks (Lada's worker threads) -----------------------------------------------------------------------------

    def _on_detection(self, item):
        if not (isinstance(item, tuple) and len(item) == 2):
            return
        num, n = item
        now = time.perf_counter()
        with self.lock:
            if self.frozen:
                return
            if self.base is None:
                self.base, self.t_first_det = num, now
                RATES.first_det = RATES.blend(RATES.first_det, now - self.t0)
            idx = num - self.base
            self.det, self.t_last_det = idx + 1, now
            self.last_mosaic = n > 0
            if n > 0:
                self.last_mosaic_idx = idx
                if self.open_start is None or self.open_start <= self.cut_end:
                    self.open_start = idx
                if len(self.mosaic) < MAX_TRACK:
                    self.mosaic.append(idx)

    def _on_clip(self, clip):
        fs, fe = getattr(clip, "frame_start", None), getattr(clip, "frame_end", None)
        if fs is None or fe is None:
            return
        with self.lock:
            if not self.frozen and self.base is not None:
                self.clips[id(clip)] = [fs - self.base, fe - self.base, None, None]
                self.pending[id(clip)] = [fs - self.base, fe - self.base, None]
                self.cut_end = max(self.cut_end, fe - self.base)

    def _on_clip_taken(self, clip):
        with self.lock:
            pc = self.pending.get(id(clip))
            if pc is not None and pc[2] is None:
                pc[2] = time.perf_counter()
            c = self.clips.get(id(clip))
            if c is not None and c[2] is None and not self.frozen:
                c[2] = time.perf_counter()

    def _on_clip_done(self, clip):
        now = time.perf_counter()
        with self.lock:
            self.pending.pop(id(clip), None)
            c = self.clips.get(id(clip))
            if c is None or c[2] is None or c[3] is not None or self.frozen:
                return
            c[3] = now
            n, dt = c[1] - c[0] + 1, now - c[2]
            if n >= 8 and dt > 0.05:
                RATES.res_fps = RATES.blend(RATES.res_fps, n / dt)

    # -- reading (controller thread) ----------------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return {"t0": self.t0, "base": self.base, "det": self.det, "t_first_det": self.t_first_det,
                    "t_last_det": self.t_last_det, "det_span": self._active_span(), "last_mosaic": self.last_mosaic, "mosaic": list(self.mosaic),
                    "clips": [list(c) for c in self.clips.values()]}


    def restore_margin(self, cur: int, fps: float, now: float, max_clip: int) -> float | None:
        """How the restoration ahead limits playback (adaptive interpolation, rife_interp.py), for the output being at
        frame index `cur`: for every clip that is cut but not restored (in start order, the order it is restored in)
        and a scene still being collected (worst case: it runs to max_clip), the video time from `cur` to its first
        frame minus the restoration work queued up to and including it; the smallest of these. None = nothing pending
        (or the segment is stopping). Negative: playback would catch up with the restoration by that much unless the
        output is already that far ahead of it."""
        res_fps = RATES.res_fps or DEFAULT_RES_FPS
        det_fps = RATES.det_fps or DEFAULT_DET_FPS
        with self.lock:
            if self.frozen or self.base is None:
                return None
            # by range only: clips of one scene (one per mosaic region) share it, and ts may be None or a float
            items = sorted(((s, e, ts, None) for s, e, ts in self.pending.values() if e >= cur),
                           key=lambda it: (it[0], it[1]))
            if self.open_start is not None and self.open_start > self.cut_end:
                s = self.open_start
                e = s + max_clip - 1 if self.last_mosaic else min(self.last_mosaic_idx, s + max_clip - 1)
                items.append((s, e, None, self.det))
        margin, work = None, 0.0
        for s, e, ts, det in items:
            n = e - s + 1
            if ts is not None:   # restoring right now
                n = max(0.1 * n, n - (now - ts) * res_fps)
            work += n / res_fps
            if det is not None:  # the open scene must also be detected to its end before it is cut
                work += max(0, e + 1 - det) / det_fps
            m = (max(s, cur) - cur) / fps - work
            margin = m if margin is None else min(margin, m)
        return margin


def estimate(snap: dict | None, now: float, *, elapsed: float, target_s: float, fps: float, wait_s: float,
             max_clip: int, out_n: int, first_out: float | None, loaded: bool, buffered_s: float | None):
    """Progress of the request in flight.

    snap: PipelineProbe.snapshot() (None: no probe, only output / mpv are known); elapsed: seconds since the
    request; out_n: frames the segment has output so far (before + after the target); first_out: time of the first
    one; loaded: mpv has the segment; buffered_s: seconds mpv has cached past the target.
    Returns (fraction 0..0.99, stage, remaining seconds).
    """
    fps = fps if fps and fps > 0 else 30.0
    wait_s = max(0.1, float(wait_s or 0))
    max_clip = max(1, int(max_clip or 1))
    det_fps = RATES.det_fps or DEFAULT_DET_FPS
    res_fps = RATES.res_fps or DEFAULT_RES_FPS
    preroll = max(0.0, target_s - first_out) if first_out is not None else (RATES.preroll_s or DEFAULT_PREROLL_S)
    pre_n = preroll * fps
    # frames (from Lada's first one) that must have come out before mpv starts playing
    if loaded:
        buf = max(0.0, buffered_s or 0.0)
        need = max(out_n, pre_n) + max(0.0, wait_s - buf) * fps
    else:
        need = pre_n + wait_s * fps
    emit_fps = max(4 * fps, 120.0)

    stage = "讀取影片"
    remaining = 0.0
    if snap is None:                    # nothing observable inside Lada: its output stands in for the detector
        det, mosaic, clips, last_mosaic = out_n, [], [], False
    elif snap["base"] is None:          # still opening / seeking the source, nothing detected yet
        first = RATES.first_det or DEFAULT_FIRST_DET
        remaining += max(first - (now - snap["t0"]), 0.3 * first)
        det, mosaic, clips, last_mosaic = 0, [], [], False
    else:
        det = snap["det"]
        span = snap.get("det_span", snap["t_last_det"] - snap["t_first_det"])   # without the time the gate held it
        if det >= 24 and span > 0.2:
            det_fps = (det - 1) / span
        mosaic, clips, last_mosaic = snap["mosaic"], snap["clips"], snap["last_mosaic"]

    # a mosaic frame among the needed ones that no clip covers yet: its scene is still being collected
    m0 = None
    for m in mosaic:
        if m >= need:
            break
        if not any(c[0] <= m <= c[1] for c in clips):
            m0 = m
            break
    det_needed, open_n = need, 0.0
    if m0 is not None:
        # still open: worst case it runs to the clip length cap; scene already over (the latest frame is clean):
        # its clip is about to be cut at the last mosaic frame
        end = m0 + max_clip - 1 if last_mosaic else min(mosaic[-1], m0 + max_clip - 1)
        det_needed = max(need, end + 1)
        open_n = end - m0 + 1
    det_rem = max(0.0, det_needed - det) / det_fps
    prior_n = 0.0
    if m0 is None and det < need and not any(c[0] < need and c[3] is None for c in clips):
        # the frames still to detect may need a clip restored too: expect a full-length one as often as earlier
        # starts needed one, scaled by the part not detected yet (so the first mosaic does not make the estimate
        # jump back; clean frames make it shrink, and a scene that ends early makes the bar jump ahead)
        p_mosaic = RATES.p_mosaic if RATES.p_mosaic is not None else DEFAULT_P_MOSAIC
        prior_n = p_mosaic * max_clip * (need - det) / max(need, 1.0)
    closed_rem, restoring = 0.0, False
    for s, e, t_start, t_done in clips:
        if s >= need or t_done is not None:
            continue
        n = (e - s + 1) / res_fps
        if t_start is not None:
            restoring = True
            n = max(0.1 * n, n - (now - t_start))
        closed_rem += n
    remaining += max(det_rem, closed_rem) + (open_n + prior_n) / res_fps + max(0.0, need - out_n) / emit_fps
    if snap is not None and snap["base"] is not None:
        if restoring or open_n or closed_rem:
            stage = "修復馬賽克"
        elif det < need:
            stage = "偵測馬賽克"
        else:
            stage = "緩衝" if loaded else "輸出"
    elif loaded:
        stage = "緩衝"
    if loaded:
        remaining = max(remaining, 0.05)
    frac = elapsed / (elapsed + remaining) if elapsed + remaining > 0 else 0.0
    return min(0.99, max(0.0, frac)), stage, remaining
