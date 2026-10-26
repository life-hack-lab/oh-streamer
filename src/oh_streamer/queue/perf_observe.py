"""
Observation accumulator for the queue worker. Pure stdlib, no torch.

The worker runs at full speed, so its source-frames/second says how much the machine can really do for this video.
PerfObserver turns the frame loop into one observation: fps over the steady part of a restore run (the first
`warmup_s` of every run are skipped: model/ffmpeg start-up, caches) and the mosaic density (detected regions per
frame, counted on the consumer side of Lada's frame_detection_queue the way benchmarks/bench_settings.py does).

  * `begin_run(mode, height, src_fps)` at the start of every restorer run, `frame()` once per source frame pushed.
  * `frame()` returns an observation dict every `interval_s` of steady processing; `finish()` returns the rest at the
    end of a run. Both return None unless >= `min_frames` frames were measured, and the window is cleared after it
    was returned.
  * a gap between two frames longer than `max_gap_s` (stall, held worker, system sleep) is left out of the elapsed time.
  * `discard()` drops the current window (source still downloading, ...); `taint()` drops it and ignores the rest of
    the run (the mode is ambiguous: WatchedDoubler made policy copies).
  * the observation is the worker's `perf_obs` message: {"type", "mode", "height", "src_fps", "fps", "density",
    "frames"}; `mode` is "full" or "off" (anything else, e.g. "auto", is never observed).
"""

import math
import threading
import time
from typing import Optional

WARMUP_S = 15.0
MIN_FRAMES = 1500
INTERVAL_S = 300.0
MAX_GAP_S = 10.0
OBSERVED_MODES = ("full", "off")


class PerfObserver:
    def __init__(self, *, warmup_s: float = WARMUP_S, min_frames: int = MIN_FRAMES, interval_s: float = INTERVAL_S,
                 max_gap_s: float = MAX_GAP_S, clock=time.monotonic):
        self.warmup_s, self.min_frames = warmup_s, min_frames
        self.interval_s, self.max_gap_s, self.clock = interval_s, max_gap_s, clock
        self._lock = threading.Lock()           # the detection counter runs on Lada's worker thread
        self._det_frames = 0                    # detection entries consumed / mosaics in them (monotonic)
        self._det_mosaics = 0
        self._active = False
        self._mode: Optional[str] = None
        self._height = 0
        self._src_fps = 0.0
        self._tainted = False
        self._t_run: Optional[float] = None     # first frame of the run
        self._measuring = False
        self._t_last = 0.0
        self._elapsed = 0.0
        self._frames = 0
        self._det0 = (0, 0)

    # ------------------------------------------------------------------------------------------------- detection
    def note_detection(self, count) -> None:
        """One consumed frame_detection_queue entry carrying `count` mosaic regions."""
        try:
            n = int(count)
        except (TypeError, ValueError):
            return
        if n < 0:
            return
        with self._lock:
            self._det_frames += 1
            self._det_mosaics += n

    def wrap_queue_get(self, q) -> bool:
        """Count the (frame_num, n_mosaics) entries `q.get` hands out. False when `q` cannot be wrapped."""
        try:
            original = q.get
        except AttributeError:
            return False

        def counting_get(*args, **kwargs):
            elem = original(*args, **kwargs)
            if isinstance(elem, tuple) and len(elem) == 2:
                self.note_detection(elem[1])
            return elem
        try:
            q.get = counting_get
        except (AttributeError, TypeError):
            return False
        return True

    # ------------------------------------------------------------------------------------------------- runs
    def begin_run(self, mode, height, src_fps) -> None:
        """A new restorer run: warm-up starts again. Modes other than full / off are not observed."""
        with self._lock:
            self._mode, self._height, self._src_fps = mode, int(height or 0), float(src_fps or 0)
            self._active = (mode in OBSERVED_MODES and self._height > 0 and math.isfinite(self._src_fps)
                            and self._src_fps > 0)
            self._tainted = False
            self._t_run = None
            self._reset_window()

    def _reset_window(self) -> None:
        self._measuring = False
        self._elapsed = 0.0
        self._frames = 0

    def discard(self) -> None:
        """Throw the current window away; measuring starts over after a new warm-up."""
        with self._lock:
            self._t_run = None
            self._reset_window()

    def taint(self) -> None:
        """The rest of this run is not representative of `mode` (policy copies): nothing is observed until begin_run."""
        with self._lock:
            self._tainted = True
            self._reset_window()

    # ------------------------------------------------------------------------------------------------- frames
    def frame(self) -> Optional[dict]:
        with self._lock:
            if not self._active or self._tainted:
                return None
            now = self.clock()
            if self._t_run is None:
                self._t_run = now
                return None
            if not self._measuring:
                if now - self._t_run < self.warmup_s:
                    return None
                self._measuring = True
                self._t_last = now
                self._det0 = (self._det_frames, self._det_mosaics)
                return None
            gap = now - self._t_last
            self._t_last = now
            if gap <= self.max_gap_s:
                self._elapsed += gap
                self._frames += 1
            if self._elapsed >= self.interval_s:
                return self._take()
            return None

    def finish(self) -> Optional[dict]:
        """End of a run: the observation of what is left, or None."""
        with self._lock:
            if not self._active or self._tainted:
                return None
            return self._take()

    def _take(self) -> Optional[dict]:
        frames, elapsed = self._frames, self._elapsed
        det_frames = self._det_frames - self._det0[0]
        mosaics = self._det_mosaics - self._det0[1]
        if not (frames >= self.min_frames and elapsed > 0 and det_frames > 0):
            return None                         # too little yet: the window keeps growing
        obs = {"type": "perf_obs", "mode": self._mode, "height": self._height,
               "src_fps": round(self._src_fps, 3), "fps": round(frames / elapsed, 3),
               "density": round(mosaics / det_frames, 4), "frames": frames}
        self._elapsed = 0.0
        self._frames = 0
        self._det0 = (self._det_frames, self._det_mosaics)
        return obs
