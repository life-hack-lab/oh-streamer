"""
Adaptive RIFE frame interpolation for oh_player (setting `interpolation` = "auto").

The restored video is output on a 2x time grid: between two source frames goes either a RIFE middle frame or a copy
of the first one, decided per frame pair (FrameDoubler + InterpPolicy). The muxer drops the exact copies (variable
frame rate output, oh_player.start_muxer), so stretches without interpolation cost the encoder and mpv (decoding,
RTX VSR) no more than 1x output; positions / timestamps stay source time either way.

- lead: seconds of video written to the muxer ahead of mpv's playback position. Only the controller thread talks to
  mpv (MpvIpc is strictly single-threaded), so it samples time-pos / core-idle a few times a second into a
  PlaybackClock and the pump thread extrapolates that with the wall clock. No sample yet (a start / seek that is not
  playing yet) = no lead = copies only: interpolation never delays a start or a seek. The controller also samples
  mpv's demuxer-cache-time, and the lead never counts more than that + IN_FLIGHT_S: frames written but held up on the
  way (encoder / muxer queues) do not protect playback yet. Without a player (--output <file> / null) a
  SimulatedPlayback stands in for mpv.
- restoration ahead: Lada outputs nothing past a mosaic clip until the whole clip is restored (up to max_clip_length
  frames, seconds of GPU work), so a lead that looks comfortable can be eaten by the next clip. The lead the policy
  sees is lead + min(0, buffer_progress.PipelineProbe.restore_margin()), i.e. what is left once the clips already
  detected ahead (and a scene still being collected, worst case) are restored at the measured speed.
- hysteresis: interpolate once the lead reaches START_LEAD_S, go back to copies below STOP_LEAD_S. Restoration alone
  runs ahead of realtime on most content; interpolating spends that lead (RIFE and Lada share the GPU, the costs add
  up), copying earns it back.
- budget: no interpolation while one RIFE frame costs more than BUDGET_FRAC of a source frame interval (median of the
  last MS_WINDOW measurements, so one frame stuck behind a burst of Lada's kernels does not count; e.g. a 4K source or
  another app on the GPU); tried again after RETRY_S.
- scene cuts: Practical-RIFE's check (SSIM of 32x32 thumbnails < SCENE_SSIM) -> a copy instead of a blend of two
  different shots.
- static overlays: burned-in subtitles / logos over a moving background are taken from the source frames instead of
  RIFE's (warped) middle frame, so they do not shimmer (protect_static; every RIFE path, player and queue).
- sources above MAX_SOURCE_FPS (50 / 60 fps) or without a usable frame rate stay at 1x (eligible()).

The player's model (RIFE v4.6 from third_party/rife, fp16, channels_last, CUDA graph on NVIDIA; measured by
bench_rife.py)
is built per frame size on a background thread (Interpolator) the first time a segment of that size is playing, so
neither the first frame nor a seek waits for it; on NVIDIA it runs on its own CUDA stream, so waiting for an
interpolated frame does not also wait for Lada's kernels. ROCm uses its default stream until the private-stream path
is validated. Only the pump thread of the playing segment uses it. Casting (hls_cast.py) has
its own frame path and stays at 1x.

FullDoubler (offline queue jobs, setting "full") skips the whole policy: every pair is interpolated, scene cuts repeat.
WatchedDoubler is a FullDoubler for a job somebody is watching while it runs: the viewer's position arrives as a
RemoteClock (fed by the player through the queue service), and while the job is less than WATCH_STOP_LEAD_S ahead of
it the pairs get a copy instead of RIFE (back to every frame at WATCH_START_LEAD_S); the segments with such copies are
redone with full RIFE later (queue_worker.py). No viewer (stale clock) or a viewer already past the frame = FullDoubler.

The policy / clock classes are pure Python (test_rife_interp.py); torch is imported only by Rife / Interpolator.
Standalone CPU inference uses FP32 eagerly. Experimental MPS inference requires an explicit opt-in; the Mac player
and queue remain gated by device_backend.rife_available until validation on a Mac.
"""

import importlib
import math
import statistics
import sys
import os
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

from oh_streamer.paths import checkout_root, upstream_root
from oh_streamer.processing import device_backend

PROJECT_ROOT = upstream_root()


def _resolve_rife_dir() -> Path:
    # RIFE code lives in the checkout's own third_party/rife (next to pyproject.toml; in a frozen build that is the
    # bundle root, where the installer stages it). Fall back to the legacy outer-project location.
    own = checkout_root() / "third_party" / "rife"
    return own if own.is_dir() else PROJECT_ROOT / "third_party" / "rife"


def _resolve_rife_weights_dir(rife_dir: Path) -> Path:
    # Weights (flownet_v<model>.pkl) are NOT in the repo: put them in <RIFE_DIR>/weights (the installer / inputs
    # manifest downloads them there); a dev machine may still keep them in the outer <project>/third_party/rife/weights.
    own = rife_dir / "weights"
    return own if own.is_dir() else PROJECT_ROOT / "third_party" / "rife" / "weights"


RIFE_DIR = _resolve_rife_dir()
RIFE_WEIGHTS_DIR = _resolve_rife_weights_dir(RIFE_DIR)

# model name -> (module in third_party/rife/vsrife_models, has Head/encoder, padding modulo) -- same table as vs-rife
MODELS = {
    "4.6": ("IFNet_HDv3_v4_6", False, 32),
    "4.22.lite": ("IFNet_HDv3_v4_22_lite", True, 32),
    "4.25.lite": ("IFNet_HDv3_v4_25_lite", True, 128),
    "4.26": ("IFNet_HDv3_v4_26", True, 64),
}

MODEL = "4.6"                 # fastest with acceptable quality (testdata/bench/rife_result.txt)
SCALE_HIGH = 0.5              # flow estimated at half resolution for frames taller than SCALE_SWITCH_H ...
SCALE_LOW = 1.0               # ... full resolution up to it
SCALE_SWITCH_H = 720
MIN_SOURCE_FPS = 10.0         # below / above this the source stays at 1x
MAX_SOURCE_FPS = 40.0
SCENE_SSIM = 0.2              # 32x32 SSIM below this = scene cut (Practical-RIFE's threshold)
IN_FLIGHT_S = 1.5             # written output may be this far ahead of what mpv has cached (encoder + muxer, ~1 s)
START_LEAD_S = 6.0            # start interpolating at this many seconds of output ahead of playback ...
STOP_LEAD_S = 3.0             # ... and stop below this (mpv rebuffers only at 0, and then waits for --prebuffer)
BUDGET_FRAC = 0.9             # max RIFE ms per frame pair as a share of the source frame interval
RETRY_S = 15.0                # after the budget stopped it, try again this long later (re-measured)
MIN_SAMPLES = 5               # RIFE timings needed before the budget is judged
MS_WINDOW = 15                # the budget looks at the median of this many recent RIFE timings
RATIO_WINDOW_S = 10.0         # the interpolated share shown in the info panel covers this much video
WATCH_START_LEAD_S = 20.0     # queue job being watched (WatchedDoubler): every frame again at this much lead ...
WATCH_STOP_LEAD_S = 10.0      # ... copies to keep up below this
WATCH_BEHIND_S = 8.0          # a viewer further than this past the frame being processed is not waiting for it
WATCH_STALE_S = 5.0           # no clock update for this long = nobody is watching
# static overlays (burned-in subtitles, logos): RIFE's flow drags them along with a moving background, so the middle
# frames show them warped / doubled and they shimmer at 2x. protect_static() takes them from the source frames instead.
PROTECT_STATIC = True
STATIC_DIFF = 10 / 255        # max per-channel change between the two frames for a pixel to count as static
STATIC_EDGE = 0.12            # min local contrast (3x3 max - min luma) of a static pixel to seed the overlay mask
STATIC_DENSITY = 0.15         # min share of seeds around a seed (text = clusters of strokes; drops lone spots)
STATIC_GROW_PX = 6            # the mask grows this far around the strokes (720p pixels) ...
STATIC_FEATHER_PX = 3         # ... and is feathered over this many pixels


def model_scale(height: int) -> float:
    return SCALE_HIGH if height > SCALE_SWITCH_H else SCALE_LOW


def protect_static(out, img0, img1, F):
    """out with static overlays taken from the source frames. out / img0 / img1: (1, 3, H, W) float tensors in [0, 1]
    (the middle frame and its two source frames), F = torch.nn.functional. Seeds are pixels that are static between
    the frames (with a static 3x3 neighbourhood) and lie on a strong edge, in clusters (text strokes and outlines,
    not a lone static spot of a moving shot); the mask grows around them so the doubled copies RIFE drew next to the
    strokes are covered too, and inside it the middle frame is the average of the two source frames. The mask is
    worked out on a grid of about 720 rows (1080p: half resolution; pooling at full size costs ~2 ms) and scaled up."""
    h, w = out.shape[-2:]
    k = max(1, round(h / 720))                                   # grid cell = k x k pixels
    s = h / k / 720.0
    grow = max(1, round(STATIC_GROW_PX * s))
    feather = max(1, round(STATIC_FEATHER_PX * s))
    dens_r = max(1, round(4 * s))
    moving = ((img0 - img1).abs().amax(1, keepdim=True) >= STATIC_DIFF).to(out.dtype)
    avg = (img0 + img1) * 0.5
    lum = avg.mean(1, keepdim=True)
    if k > 1:
        moving = F.max_pool2d(moving, k, k, ceil_mode=True)      # a cell is static only if all its pixels are
        lum = F.avg_pool2d(lum, k, k, ceil_mode=True)
    static = 1 - F.max_pool2d(moving, 3, 1, 1)                  # the neighbourhood is static too
    contrast = F.max_pool2d(lum, 3, 1, 1) + F.max_pool2d(-lum, 3, 1, 1)
    seed = static * (contrast > STATIC_EDGE).to(out.dtype)
    seed = seed * (_pool(F.avg_pool2d, seed, dens_r) > STATIC_DENSITY).to(out.dtype)
    m = _pool(F.max_pool2d, seed, grow)
    m = _pool(F.avg_pool2d, m, feather)
    if k > 1:
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    return out + (avg - out) * m


def _pool(pool, x, r):
    """pool over a (2r+1)^2 window as a row pass then a column pass (same result, far fewer reads for big r)."""
    return pool(pool(x, (1, 2 * r + 1), 1, (0, r)), (2 * r + 1, 1), 1, (r, 0))


def eligible(fps) -> tuple[bool, str]:
    """(interpolate this source?, reason shown when not). VFR sources are already treated as constant rate at their
    r_frame_rate by the pipeline; a weird one (1000/1, 90000/1 ...) ends up above MAX_SOURCE_FPS and stays at 1x."""
    try:
        fps = float(fps)
    except (TypeError, ValueError, ZeroDivisionError):
        return False, "影格率不明"
    if not fps > 0 or math.isinf(fps):
        return False, "影格率不明"
    if fps < MIN_SOURCE_FPS:
        return False, f"來源僅 {fps:.3g} fps"
    if fps > MAX_SOURCE_FPS:
        return False, f"來源已是 {fps:.3g} fps"
    return True, ""


# ---------------------------------------------------------------------------------------------------------------
# playback position (lead signal)
# ---------------------------------------------------------------------------------------------------------------

class PlaybackClock:
    """mpv's playback position as last sampled by the controller thread, extrapolated with the wall clock while it
    plays (thread-safe). reset() at every segment start: no position = no lead until the controller samples again,
    which it only does once the new segment is playing."""

    def __init__(self):
        self._lock = threading.Lock()
        self._anchor = None   # (position, advancing, perf_counter time, mpv's cache end or None)

    def reset(self):
        with self._lock:
            self._anchor = None

    def update(self, pos: float, advancing: bool, now: float | None = None, cache_end: float | None = None):
        with self._lock:
            self._anchor = (float(pos), bool(advancing), time.perf_counter() if now is None else now,
                            None if cache_end is None else float(cache_end))

    def position(self, now: float) -> float | None:
        with self._lock:
            a = self._anchor
        if a is None:
            return None
        pos, advancing, t, _ = a
        return pos + max(0.0, now - t) if advancing else pos

    def cache_end(self) -> float | None:
        """mpv's demuxer-cache-time at the last sample (the newest video it has received)."""
        with self._lock:
            return self._anchor[3] if self._anchor is not None else None

    def on_output(self, t_frame: float, frame_s: float, now: float):
        pass   # mpv sees the output itself


class RemoteClock(PlaybackClock):
    """A PlaybackClock fed from outside (the position of a player in another process, relayed about once a second):
    position() is None once the last update is older than stale_s, i.e. nobody is watching (player closed, queue app
    not forwarding)."""

    def __init__(self, stale_s: float = WATCH_STALE_S):
        super().__init__()
        self.stale_s = stale_s

    def position(self, now: float) -> float | None:
        with self._lock:
            a = self._anchor
        if a is None or now - a[2] > self.stale_s:
            return None
        return super().position(now)

    def fresh(self, now: float) -> bool:
        with self._lock:
            a = self._anchor
        return a is not None and now - a[2] <= self.stale_s


class SimulatedPlayback:
    """Stands in for mpv where there is no player (--output <file> / null): a viewer that starts once `initial_s`
    seconds of video are ready, plays in real time and, on running dry, waits for `rebuffer_s` like mpv's
    cache-pause. The output then gets the interpolation decisions the player would make, and the stalls are counted."""

    def __init__(self, initial_s: float, rebuffer_s: float):
        self.initial_s, self.rebuffer_s = max(0.0, initial_s), max(0.0, rebuffer_s)
        self._lock = threading.Lock()
        self.avail = None          # video is ready up to this source time
        self.pos = None            # playback position
        self.playing = self.started = False
        self.pos0 = self.t0 = None  # while playing: at wall time t0 the position was pos0
        self.stalls, self.stall_s = 0, 0.0
        self._stall_t = None

    def _advance(self, now: float):
        if self.pos is None:
            return
        if self.playing:
            p = self.pos0 + (now - self.t0)
            if p >= self.avail:   # ran dry
                self.playing = False
                self.pos = self.avail
                self._stall_t = self.t0 + (self.avail - self.pos0)
                self.stalls += 1
            else:
                self.pos = p
        else:
            need = self.rebuffer_s if self.started else self.initial_s
            if self.avail - self.pos >= need - 1e-9:
                self.playing = True
                self.pos0, self.t0 = self.pos, now
                if self.started and self._stall_t is not None:
                    self.stall_s += max(0.0, now - self._stall_t)
                self.started = True

    def on_output(self, t_frame: float, frame_s: float, now: float):
        with self._lock:
            if self.pos is None:
                self.pos = t_frame
            self.avail = t_frame + frame_s
            self._advance(now)

    def position(self, now: float) -> float | None:
        with self._lock:
            self._advance(now)
            return self.pos if self.started else None

    def cache_end(self) -> float | None:
        return None   # nothing between the writes and this viewer

    def summary(self, now: float) -> str:
        with self._lock:
            self._advance(now)
            stall = self.stall_s + (now - self._stall_t if self.started and not self.playing and self._stall_t else 0.0)
            lead = (self.avail - self.pos) if self.pos is not None else 0.0
            return (f"simulated playback: {'playing' if self.playing else 'buffering'}, lead {lead:.1f}s, "
                    f"{self.stalls} stall(s) {stall:.1f}s")


# ---------------------------------------------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------------------------------------------

class InterpPolicy:
    """Per segment: interpolate the next frame pair, or copy? (see the module docstring). Called by the pump thread;
    snapshot() may be read from the controller thread."""

    def __init__(self, fps: float, start_lead: float = START_LEAD_S, stop_lead: float = STOP_LEAD_S,
                 budget_frac: float = BUDGET_FRAC, retry_s: float = RETRY_S, window_s: float = RATIO_WINDOW_S):
        self.start_lead, self.stop_lead, self.retry_s = start_lead, stop_lead, retry_s
        self.budget_ms = budget_frac * 1000.0 / fps
        self.active = False
        self.state = "waiting"      # waiting | lead | loading | on | budget | error
        self.lead = None
        self.rife_ms = None         # median of the recent RIFE ms per frame pair
        self.ms = deque(maxlen=MS_WINDOW)
        self.blocked_at = None      # when the budget last stopped / refused it
        self.error = None
        self.window = deque(maxlen=max(1, round(window_s * fps)))
        self.win_interp = 0
        self.counts = {"interp": 0, "copy": 0, "scene": 0}

    def budget_ok(self) -> bool:
        return len(self.ms) < MIN_SAMPLES or self.rife_ms <= self.budget_ms

    def decide(self, lead: float | None, now: float, ready: bool = True) -> bool:
        """Interpolate the pair ending at the frame whose source time is `lead` seconds ahead of playback (None =
        playback not established)? ready: the model for this frame size is loaded."""
        self.lead = lead
        if self.error is not None:
            self.active, self.state = False, "error"
            return False
        if lead is None:
            self.active, self.state = False, "waiting"
            return False
        if self.active:
            if lead < self.stop_lead:
                self.active, self.state = False, "lead"
            elif not self.budget_ok():
                self.active, self.state, self.blocked_at = False, "budget", now
        elif lead >= self.start_lead:
            if self.budget_ok():
                self.active, self.blocked_at = True, None
            elif self.blocked_at is None:
                self.blocked_at, self.state = now, "budget"
            elif now - self.blocked_at >= self.retry_s:
                self.ms.clear()                       # measure again under today's load
                self.rife_ms, self.blocked_at = None, None
                self.active = True
            else:
                self.state = "budget"
        else:
            self.state = "budget" if self.blocked_at is not None else "lead"
        if not self.active:
            return False
        self.state = "on" if ready else "loading"
        return ready

    def note_ms(self, ms: float):
        self.ms.append(ms)
        self.rife_ms = statistics.median(self.ms)

    def record(self, kind: str):
        """What the pair became: "interp", "copy" or "scene" (a copy because of a scene cut)."""
        v = 1 if kind == "interp" else 0
        if len(self.window) == self.window.maxlen:
            self.win_interp -= self.window[0]
        self.window.append(v)
        self.win_interp += v
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def fail(self, msg: str):
        self.error, self.active, self.state = msg, False, "error"

    def ratio(self) -> float | None:
        n = len(self.window)
        return self.win_interp / n if n else None

    def snapshot(self) -> dict:
        return {"ratio": self.ratio(), "rife_ms": self.rife_ms, "state": self.state, "lead": self.lead,
                "active": self.active, "start_lead": self.start_lead, "stop_lead": self.stop_lead,
                "budget_ms": self.budget_ms, "counts": dict(self.counts), "error": self.error}


class FrameDoubler:
    """One segment's 2x output. push() each restored frame and write what it returns, in order (the pair's middle
    frame or a copy of the previous frame, then the frame itself); finish() at the end of the video (not on a stop)
    returns the copy of the last frame that completes the duration (N source frames = 2N output frames)."""

    def __init__(self, interpolator, fps: float, width: int, height: int, clock, log=print, margin=None):
        """margin: (now) -> seconds or None, the restoration ahead (PipelineProbe.restore_margin at the current
        output frame); None = only the lead counts."""
        self.interp, self.clock, self.log, self.margin = interpolator, clock, log, margin
        self.w, self.h = width, height
        self.frame_s = 1.0 / fps
        self.policy = InterpPolicy(fps)
        self.prev = self.prev_arr = None
        self.t_last = None
        self.requested = False
        self.raw_lead = self.restore_m = None   # last lead before the restoration margin, and that margin

    def push(self, frame, arr, t: float | None, now: float | None = None) -> list:
        now = time.perf_counter() if now is None else now
        if t is None:
            t = self.t_last + self.frame_s if self.t_last is not None else 0.0
        out = []
        if self.prev is not None:
            mid = self._middle(frame, t, now)
            out.append(mid if mid is not None else self.prev_arr)
        out.append(arr)
        self.prev, self.prev_arr, self.t_last = frame, arr, t
        self.clock.on_output(t, self.frame_s, now)
        return out

    def finish(self) -> list:
        return [] if self.prev_arr is None else [self.prev_arr]

    def _middle(self, frame, t: float, now: float):
        pol = self.policy
        pos = self.clock.position(now)
        lead = self.raw_lead = None if pos is None else t - pos
        cache_end = self.clock.cache_end() if lead is not None else None
        if cache_end is not None:
            lead = self.raw_lead = min(lead, cache_end - pos + IN_FLIGHT_S)
        if lead is not None and self.margin is not None:
            self.restore_m = m = self.margin(now)
            if m is not None and m < 0:
                lead += m      # what the clips ahead will leave of it
        if pos is not None and not self.requested:
            self.requested = True     # playback established: build the model for this size now (background)
            self.interp.prepare(self.w, self.h)
        rife = self.interp.get(self.w, self.h) if self.requested else None
        if self.requested and rife is None and self.interp.error_for(self.w, self.h) and pol.error is None:
            pol.fail(self.interp.error_for(self.w, self.h))
        if not pol.decide(lead, now, ready=rife is not None):
            pol.record("copy")
            return None
        t0 = time.perf_counter()
        try:
            mid = rife.middle(self.prev, frame)
        except Exception as e:  # noqa: BLE001 - never take the segment down: copy frames from now on
            pol.fail(repr(e))
            self.log(f"interpolation: RIFE failed, copying frames for the rest of the segment: {e!r}")
            pol.record("copy")
            return None
        pol.note_ms((time.perf_counter() - t0) * 1000.0)
        pol.record("interp" if mid is not None else "scene")
        return mid

    def summary(self) -> str:
        s = self.policy.snapshot()
        r, ms, lead, raw, m = s["ratio"], s["rife_ms"], s["lead"], self.raw_lead, self.restore_m
        c = s["counts"]
        lead_s = "-" if lead is None else f"{lead:.1f}s" + (f" (out {raw:.1f}s, restore {m:+.1f}s)"
                                                         if raw is not None and m is not None else "")
        return (f"interp {'-' if r is None else f'{r * 100:.0f}%'} (10s) {s['state']}"
                f" | RIFE {'-' if ms is None else f'{ms:.1f}ms'} | lead {lead_s}"
                f" | pairs {c['interp']} interp / {c['copy']} copy / {c['scene']} scene")


class FullDoubler(FrameDoubler):
    """Always-interpolating 2x output for offline jobs (queue_worker.py, interpolation "full"): no player, so no lead /
    budget / hysteresis policy -- every frame pair gets a RIFE middle frame, except at a scene cut (Practical-RIFE's
    SSIM check, see Rife.middle), where the previous frame is repeated. Deterministic: the output depends on the frames
    only, never on timing. Same push()/finish() contract as FrameDoubler (N frames -> 2N: the first frame alone, then
    [middle, frame] for every further one, finish() = the closing copy of the last frame), plus the stage attributes
    rate_mult / layers that queue_worker's encoder command takes.

    Output frame 2j is source frame j and 2j+1 the middle frame between j and j+1, so a run that starts at source frame
    s emits exactly what a run from frame 0 emits for those positions, except that its first slot-before (the middle
    frame between s-1 and s) belongs to the previous run's last slot (see queue_worker's docstring).

    Failures are not hidden: a RIFE model that cannot be built or a frame pair that raises fails the job (the player's
    copy fallback would silently give a half-interpolated file). prepare() starts the model build on a background
    thread (e.g. while Lada's models load); the first push waits for it."""

    rate_mult = 2
    layers = 1

    def __init__(self, interpolator, fps: float, width: int, height: int, log=print, wait_s: float = 300.0):
        super().__init__(interpolator, fps, width, height, PlaybackClock(), log)
        self.wait_s = wait_s
        self.rife = None

    def prepare(self):
        self.interp.prepare(self.w, self.h)

    def _model(self):
        if self.rife is not None:
            return self.rife
        self.prepare()
        deadline = time.monotonic() + self.wait_s
        while True:
            rife = self.interp.get(self.w, self.h)
            if rife is not None:
                self.rife = rife
                return rife
            err = self.interp.error_for(self.w, self.h)
            if err:
                raise RuntimeError(f"RIFE could not be built: {err}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"RIFE model not ready after {self.wait_s:.0f}s")
            time.sleep(0.02)

    def _middle(self, frame, t: float, now: float):
        mid = self._model().middle(self.prev, frame)
        self.policy.record("interp" if mid is not None else "scene")
        return mid

    def summary(self) -> str:
        c = self.policy.counts
        return f"pairs {c['interp']} interp / {c['scene']} scene cut (repeated)"


class WatchedDoubler(FullDoubler):
    """FullDoubler for a queue job that is being watched while it runs: lead = frame time - the viewer's position
    (RemoteClock). Hysteresis like the player's policy: below stop_lead the pair gets a copy of the previous frame
    (what FrameDoubler writes for a copy; the encoder's mpdecimate drops it) so the job keeps up with the viewer, from
    start_lead on every pair is interpolated again. No viewer (clock without a fresh position) or a viewer more than
    behind_s past this frame = interpolate. `copied` counts the policy copies only (not the scene-cut repeats), so the
    caller can mark the segments that need a redo. With a clock that never updates it is a FullDoubler."""

    def __init__(self, interpolator, fps: float, width: int, height: int, clock, log=print, wait_s: float = 300.0,
                 start_lead: float = WATCH_START_LEAD_S, stop_lead: float = WATCH_STOP_LEAD_S,
                 behind_s: float = WATCH_BEHIND_S):
        super().__init__(interpolator, fps, width, height, log, wait_s)
        self.clock = clock
        self.start_lead, self.stop_lead, self.behind_s = start_lead, stop_lead, behind_s
        self.copying = False
        self.copied = 0

    def _set_copying(self, copying: bool, lead: float | None):
        if copying == self.copying:
            return
        self.copying = copying
        if copying:
            self.log(f"interpolation: viewer close (lead {lead:.1f}s) -> copies to keep up")
        elif lead is None:
            self.log("interpolation: nobody watching -> every frame again")
        elif lead >= self.start_lead:
            self.log(f"interpolation: lead {lead:.1f}s -> every frame again")
        else:
            self.log(f"interpolation: viewer far behind (lead {lead:.1f}s) -> every frame again")

    def _middle(self, frame, t: float, now: float):
        pos = self.clock.position(now)
        lead = None if pos is None else t - pos
        if lead is None or lead < -self.behind_s:
            self._set_copying(False, lead)
        elif lead < self.stop_lead:
            self._set_copying(True, lead)
        elif lead >= self.start_lead:
            self._set_copying(False, lead)
        if self.copying:
            self.copied += 1
            self.policy.record("copy")
            return None
        return super()._middle(frame, t, now)

    def summary(self) -> str:
        s = super().summary()
        return f"{s} | {self.copied} copied (watched)" if self.copied else s


# ---------------------------------------------------------------------------------------------------------------
# RIFE model (torch)
# ---------------------------------------------------------------------------------------------------------------

def load_state_dict(model: str):
    import torch
    sd = torch.load(RIFE_WEIGHTS_DIR / f"flownet_v{model}.pkl", map_location="cpu", weights_only=True)
    return {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}


def _backend_options(device, *, fp16, channels_last, cuda_graph, own_stream, experimental_mps, rocm=False):
    """Choose safe defaults without importing torch (Interpolator's setup stays lightweight).

    ROCm uses torch.cuda APIs but its RIFE CUDA-graph and private-stream paths are not validated yet. They stay off
    by default on HIP; callers can still opt in explicitly with cuda_graph=True or own_stream=True.
    """
    backend = getattr(device, "type", str(device).split(":", 1)[0])
    if backend not in ("cpu", "cuda", "mps"):
        raise ValueError(f"RIFE does not support device {device!r}; use cpu, cuda:N or mps")
    if backend == "mps" and not experimental_mps:
        raise ValueError("RIFE MPS is unvalidated; use experimental_mps=True only for an explicit standalone probe")
    is_cuda = backend == "cuda"
    use_fp16 = is_cuda if fp16 is None else bool(fp16)
    optimized_default = is_cuda and not rocm
    if backend == "cpu" and use_fp16:
        raise ValueError("RIFE CPU inference requires FP32; use fp16=False or the automatic default")
    return {
        "fp16": use_fp16,
        "channels_last": is_cuda if channels_last is None else bool(channels_last),
        "cuda_graph": is_cuda and (optimized_default if cuda_graph is None else bool(cuda_graph)),
        "own_stream": is_cuda and (optimized_default if own_stream is None else bool(own_stream)),
    }


class Rife:
    """Minimal port of vs-rife's inference (no VapourSynth): pad to the model's modulo, cached per-frame encoder
    features (v4.7+ models), timestep 0.5 = the middle frame. step() is bench_rife.py's streaming interface, middle()
    the player's (frame pairs of CPU uint8 HWC BGR frames in, numpy frame out, scene-cut check, static overlays
    such as burned-in subtitles taken from the source frames: protect_static).

    own_stream: run on a private CUDA stream (the player: synchronizing on an interpolated frame then waits for RIFE's
    own work only, not for Lada's kernels queued on the default stream). CUDA stream/graph requests are ignored on
    CPU/MPS. FP16 defaults to CUDA only; CPU requires FP32. MPS requires experimental_mps=True and remains unvalidated.
    ROCm shares torch.cuda APIs, but graph capture and a private stream default off until validated. The optional graph
    is captured in thread-local mode; if capture fails it runs eagerly."""

    def __init__(self, model: str, w: int, h: int, scale: float, device, fp16: bool | None = None,
                 channels_last: bool = False, cuda_graph: bool = False, state_dict=None, own_stream: bool = False,
                 *, experimental_mps: bool = False, protect: bool = PROTECT_STATIC):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        device = torch.device(device)
        options = _backend_options(device, fp16=fp16, channels_last=channels_last, cuda_graph=cuda_graph,
                                   own_stream=own_stream, experimental_mps=experimental_mps,
                                   rocm=device_backend.is_rocm(torch))
        fp16, channels_last = options["fp16"], options["channels_last"]
        cuda_graph, own_stream = options["cuda_graph"], options["own_stream"]
        if model not in MODELS:
            raise ValueError(f"unknown RIFE model {model!r}; choose from {', '.join(MODELS)}")
        if w <= 0 or h <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("RIFE needs positive frame dimensions and a finite positive scale")
        if str(RIFE_DIR) not in sys.path:
            sys.path.insert(0, str(RIFE_DIR))
        mod_name, has_head, modulo = MODELS[model]
        mod = importlib.import_module(f"vsrife_models.{mod_name}")
        self.device, self.w, self.h = device, w, h
        self.dtype = torch.half if fp16 else torch.float
        tmp = max(modulo, int(modulo / scale))
        self.pw, self.ph = math.ceil(w / tmp) * tmp, math.ceil(h / tmp) * tmp
        self.padding = (0, self.pw - w, 0, self.ph - h)
        self.need_pad = any(self.padding)
        self.mf = torch.channels_last if channels_last else torch.contiguous_format
        self.stream = torch.cuda.Stream(device) if own_stream else None
        self.capture_error = None
        self.protect = protect     # middle(): static overlays from the source frames (protect_static)

        sd = state_dict if state_dict is not None else load_state_dict(model)
        net = mod.IFNet(scale, False)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # vs-rife loads with strict=False too; make sure nothing that matters is missing
        if missing:
            raise RuntimeError(f"RIFE {model}: missing weights {missing[:5]}")
        self.unexpected = list(unexpected)
        with self._ctx():
            self.flownet = net.eval().to(device, self.dtype).to(memory_format=self.mf)
            self.encode = self.flownet.encode if has_head else None

            self.tenFlow_div = torch.tensor([(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0], dtype=torch.float,
                                            device=device)
            hor = torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float, device=device).view(1, 1, 1, self.pw).expand(-1, -1, self.ph, -1)
            ver = torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float, device=device).view(1, 1, self.ph, 1).expand(-1, -1, -1, self.pw)
            self.grid = torch.cat([hor, ver], 1)
            self.timestep = torch.full([1, 1, self.ph, self.pw], 0.5, dtype=self.dtype, device=device)
            # scene-cut check: 11x11 gaussian window (sigma 1.5) per channel, as in SSIM
            g = torch.exp(-((torch.arange(11, dtype=torch.float, device=device) - 5) ** 2) / (2 * 1.5 ** 2))
            g = g / g.sum()
            self.ssim_win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
        self.prev = None   # step(): (img, feat) of the previous frame
        self._last = None  # middle(): the source frame object whose prepared image is in "slot 1"
        self._p1 = None    # middle() eager path: (img, feat, thumb) of that frame
        self._thumb1 = None
        self.graph = None
        if cuda_graph:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001 - eager is only a few % slower
                self.graph = None
                self.capture_error = repr(e)

    def _ctx(self):
        return self.torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

    # --- pre / post processing (uint8 HWC BGR <-> padded FP16/FP32 NCHW RGB in [0,1])
    def prep(self, frame_u8_hwc):
        """frame_u8_hwc: uint8 HWC BGR torch tensor on CPU or GPU."""
        t = frame_u8_hwc.to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype).div_(255.0)
        if self.need_pad:
            t = self.F.pad(t, self.padding)
        return t.contiguous(memory_format=self.mf)

    def post(self, out):
        out = out[0, :, :self.h, :self.w].flip(0).mul(255.0).round_().clamp_(0, 255)
        return out.to(self.torch.uint8).permute(1, 2, 0).contiguous()

    def feat(self, img):
        return self.encode(img) if self.encode is not None else None

    def _net(self, img0, img1, f0, f1):
        if self.encode is not None:
            return self.flownet(img0, img1, self.timestep, self.tenFlow_div, self.grid, f0, f1)
        return self.flownet(img0, img1, self.timestep, self.tenFlow_div, self.grid)

    def _protected(self, out, img0, img1):
        """middle()'s frame: RIFE's output with the static overlays from the source frames (cropped to w x h)."""
        if not self.protect:
            return out
        h, w = self.h, self.w
        return protect_static(out[:, :, :h, :w], img0[:, :3, :h, :w], img1[:, :3, :h, :w], self.F)

    # --- optional CUDA graph of "encode(new) + flownet(prev, new) [+ protect_static]" (fixed shapes, static buffers)
    def _capture(self):
        torch = self.torch
        if self.device.type != "cuda":
            raise RuntimeError("RIFE CUDA graph capture requires a CUDA device")
        shape = (1, 3, self.ph, self.pw)
        with self._ctx():
            self.g_img0 = torch.zeros(shape, dtype=self.dtype, device=self.device).contiguous(memory_format=self.mf)
            self.g_img1 = torch.zeros_like(self.g_img0)
            self.g_f0 = self.feat(self.g_img0) if self.encode is not None else None
        s = torch.cuda.Stream(self.device)
        s.wait_stream(self.stream if self.stream is not None else torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                f1 = self.feat(self.g_img1)
                self._protected(self._net(self.g_img0, self.g_img1, self.g_f0, f1), self.g_img0, self.g_img1)
        (self.stream if self.stream is not None else torch.cuda.current_stream(self.device)).wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        # thread_local: other threads (Lada's workers) may keep allocating / launching while this one captures
        with torch.cuda.device(self.device), self._ctx(), \
                torch.cuda.graph(graph, capture_error_mode="thread_local"), torch.inference_mode():
            self.g_f1 = self.feat(self.g_img1)
            self.g_raw = self._net(self.g_img0, self.g_img1, self.g_f0, self.g_f1)
            self.g_out = self._protected(self.g_raw, self.g_img0, self.g_img1)
        self.graph = graph

    def step(self, img1):
        """img1: preprocessed new frame. Returns the middle frame between the previous and this one (None for the
        first frame)."""
        if self.graph is not None:
            # rotate static buffers: previous "new" becomes img0 / f0
            self.g_img0.copy_(self.g_img1)
            if self.g_f0 is not None:
                self.g_f0.copy_(self.g_f1)
            self.g_img1.copy_(img1)
            first = self.prev is None
            self.prev = True
            if first:
                if self.g_f0 is not None:
                    self.g_f1.copy_(self.feat(img1))
                return None
            self.graph.replay()
            return self.g_raw
        f1 = self.feat(img1)
        if self.prev is None:
            self.prev = (img1, f1)
            return None
        img0, f0 = self.prev
        out = self._net(img0, img1, f0, f1)
        self.prev = (img1, f1)
        return out

    # --- player interface
    def thumb(self, img):
        return self.F.interpolate(img[:, :3, :self.h, :self.w].float(), size=(32, 32), mode="bilinear",
                                  align_corners=False)

    def ssim(self, a, b):
        """SSIM of two (1, 3, 32, 32) float thumbnails in [0, 1], as a scalar on the selected device."""
        F, w = self.F, self.ssim_win
        mu_a, mu_b = F.conv2d(a, w, groups=3), F.conv2d(b, w, groups=3)
        aa = F.conv2d(a * a, w, groups=3) - mu_a * mu_a
        bb = F.conv2d(b * b, w, groups=3) - mu_b * mu_b
        ab = F.conv2d(a * b, w, groups=3) - mu_a * mu_b
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mu_a * mu_b + c1) * (2 * ab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (aa + bb + c2))
        return m.mean()

    def _tensor(self, f):
        return f if isinstance(f, self.torch.Tensor) else self.torch.from_numpy(f)

    def middle(self, f0, f1, scene_ssim: float = SCENE_SSIM):
        """Middle frame between the source frames f0 and f1 (uint8 HWC BGR, CPU torch tensors or numpy arrays) as a
        uint8 HWC BGR numpy array, or None at a scene cut. Consecutive pairs (f0 = the previous call's f1, the same
        object) reuse its upload and features."""
        torch = self.torch
        with self._ctx(), torch.inference_mode():
            if self.graph is not None:
                if f0 is not self._last:
                    self.g_img1.copy_(self.prep(self._tensor(f0)))   # rotated into slot 0 below
                    if self.g_f0 is not None:
                        self.g_f1.copy_(self.feat(self.g_img1))
                    self._thumb1 = self.thumb(self.g_img1)
                self.g_img0.copy_(self.g_img1)
                if self.g_f0 is not None:
                    self.g_f0.copy_(self.g_f1)
                th0 = self._thumb1
                self.g_img1.copy_(self.prep(self._tensor(f1)))
                self._thumb1 = th1 = self.thumb(self.g_img1)
                sim = self.ssim(th0, th1)
                self.graph.replay()
                out = self.g_out   # protect_static is part of the graph
            else:
                if f0 is not self._last or self._p1 is None:
                    img = self.prep(self._tensor(f0))
                    self._p1 = (img, self.feat(img), self.thumb(img))
                img0, feat0, th0 = self._p1
                img1 = self.prep(self._tensor(f1))
                feat1, th1 = self.feat(img1), self.thumb(img1)
                sim = self.ssim(th0, th1)
                out = self._protected(self._net(img0, img1, feat0, feat1), img0, img1)
                self._p1 = (img1, feat1, th1)
            self._last = f1
            res = self.post(out).cpu()   # waits for this stream's work only (own_stream)
            if float(sim) < scene_ssim:
                return None
            return res.numpy()


class Interpolator:
    """The player's RIFE model, shared by every segment and source. prepare(w, h) builds the model for a frame size
    on a background thread (weights read once, model per size; a new size replaces the old model);
    get(w, h) returns it once ready. Only one pump thread uses the model at a time (the playing segment)."""

    def __init__(self, device, log=print, model: str = MODEL, *, fp16: bool | None = None,
                 channels_last: bool | None = None, cuda_graph: bool | None = None,
                 own_stream: bool | None = None, experimental_mps: bool = False):
        self.device, self.log, self.model = device, log, model
        self._requested_options = dict(fp16=fp16, channels_last=channels_last, cuda_graph=cuda_graph,
                                       own_stream=own_stream)
        self._options = _backend_options(device, fp16=fp16, channels_last=channels_last, cuda_graph=cuda_graph,
                                         own_stream=own_stream, experimental_mps=experimental_mps)
        self._experimental_mps = experimental_mps
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        self._sd = None
        self._key = None          # (w, h) of the model that is ready or being built
        self._rife = None
        self._error = None        # (key, message) of a failed build
        self._thread = None
        self.info = None          # description of the ready model (log / info panel)

    def preload(self):
        """Read the weights on a background thread (e.g. while Lada's models load)."""
        threading.Thread(target=self._preload, name="rife weights", daemon=True).start()

    def _preload(self):
        try:
            self._weights()
        except Exception as e:  # noqa: BLE001 - the build reports it again
            self.log(f"interpolation: could not read the RIFE weights: {e!r}")

    def _weights(self):
        with self._wlock:
            if self._sd is None:
                self._sd = load_state_dict(self.model)
            return self._sd

    def prepare(self, w: int, h: int):
        with self._lock:
            if self._key == (w, h):
                return
            self._key, self._rife, self._error, self.info = (w, h), None, None, None
            th = self._thread = threading.Thread(target=self._build, args=(w, h), name="rife build", daemon=True)
        th.start()

    def _build(self, w: int, h: int):
        t = time.perf_counter()
        try:
            import torch
            # Keep Interpolator.__init__ free of torch imports. Resolve HIP defaults when model construction already
            # needs torch, so the runtime options still match the actual PyTorch backend.
            self._options = _backend_options(
                self.device, **self._requested_options, experimental_mps=self._experimental_mps,
                rocm=device_backend.is_rocm(torch))
            scale = model_scale(h)
            rife = Rife(self.model, w, h, scale, self.device, state_dict=self._weights(),
                        experimental_mps=self._experimental_mps, **self._options)
            z = torch.zeros((h, w, 3), dtype=torch.uint8)
            rife.middle(z, torch.full_like(z, 16))   # warm up the eager parts (upload, thumbnails, post)
            rife._last = rife._p1 = None
            backend = getattr(self.device, "type", str(self.device).split(":", 1)[0])
            precision = "FP16" if self._options["fp16"] else "FP32"
            backend_label = device_backend.backend_name(self.device, torch_module=torch)
            mode = "CUDA graph" if rife.graph is not None else "eager"
            detail = f"; graph capture failed: {rife.capture_error}" if rife.capture_error else ""
            experimental = "; experimental MPS, Mac validation pending" if backend == "mps" else ""
            backend_detail = f" ({backend_label})" if backend == "cuda" and backend_label == "ROCm/HIP" else ""
            info = (f"RIFE v{self.model} {w}x{h} scale {scale:g} (pad {rife.pw}x{rife.ph}), "
                    f"{self.device}{backend_detail} {precision} {mode}{detail}{experimental}")
            with self._lock:
                if self._key != (w, h):
                    return            # superseded by another size meanwhile
                self._rife, self.info = rife, info
            self.log(f"interpolation: {info} ready in {time.perf_counter() - t:.2f}s")
        except Exception as e:  # noqa: BLE001
            with self._lock:
                if self._key == (w, h):
                    self._error = ((w, h), repr(e))
            self.log(f"interpolation: building RIFE for {w}x{h} failed: {e!r}")

    def get(self, w: int, h: int):
        with self._lock:
            return self._rife if self._key == (w, h) else None

    def error_for(self, w: int, h: int) -> str | None:
        with self._lock:
            return self._error[1] if self._error and self._error[0] == (w, h) else None

    def building(self) -> bool:
        th = self._thread
        return th is not None and th.is_alive()

    def wait_idle(self, timeout: float = 30.0):
        """Wait for a build in progress (before empty_cache() / freeing models: a capture must not overlap it)."""
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout)

    def release(self):
        """Drop the model (VRAM); the caller runs gc / empty_cache. The weights stay in RAM for a later prepare()."""
        self.wait_idle()
        with self._lock:
            self._key = self._rife = self._error = self.info = None
