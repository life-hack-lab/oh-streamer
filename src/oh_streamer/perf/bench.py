"""Computer performance check: measurements.

The parent (run_benchmark) never imports torch: it generates synthetic sources with ffmpeg, runs every measurement
in a child process (clean VRAM peak, timeout, cancellable) and assembles the profile dict. The children
(`python -m oh_streamer.perf child <json>`) import torch / lada and print one `PERF_RESULT {json}` line.

Restore measurement = the real FrameRestorer on a synthetic file; only the detections are replaced (the YOLO inference
still runs, MosaicDetector._create_or_append_scenes_based_on_prediction_result is swapped on the instance for one that
adds `density` synthetic, slowly moving, non-overlapping regions per frame).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform as _platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from oh_streamer.engines import base as engine_base  # pure python (the engine itself is imported lazily)
from oh_streamer.engines.lada import bench_hooks  # pure python at import; torch / lada only inside its functions

SCHEMA = 1
RESULT_PREFIX = "PERF_RESULT "
HEIGHTS_ALL = (480, 720, 1080)
WIDTH_FOR = {480: 854, 720: 1280, 1080: 1920}
FPS = 30
SOURCE_MARGIN_FRAMES = 600   # synthetic source: frames beyond the window end + detector lead (see source_seconds)
CLIP_QUEUE_BYTES = 512 * 1024 * 1024   # FrameRestorer's mosaic_clip_queue budget (clip_length x 256 x 256 x 4 per clip)
DETECTOR = "v4-fast"
RESTORATION = "basicvsrpp-v1.2"
CLIP_LENGTH = 180
MEASURE_CLIP_MAX = 240       # longer clip settings are measured with 240-frame clips: the same speed within noise
                             # (2026-10-13, RTX 5060: 240 vs 300 in bench and on a real video), ~35 s shorter
DEFAULT_TIMEOUT = 300.0
BOX_HEIGHT_FRAC = 0.30
WINDOW_MIN_S = 3.0           # steady-state window: at least this long ...
WINDOW_CAP_S = 35.0          # ... and ended after this long at the latest (slow machines); must fit
                             # 1 + WINDOW_MIN_PERIODS clips of the combined run (~27 fps x 240 frames on an RTX 5060)
WINDOW_MIN_FRAMES = 90
RIFE_CHUNK = 10             # RIFE pairs per timed chunk (median of chunks)
WINDOW_MIN_PERIODS = 3       # aligned windows span at least this many clips: one clip's burst timing is too noisy
WARMUP_CLIPS = 2             # aligned windows start after this many clips: the first clip after one is still 30-75%
                             # slower than steady state, at 720p/480p the second ~10% (RTX 5060, 2026-10-13)
BUSY_GPU_PCT = 20.0
DENSE_HEIGHT = 720           # density >= 1 is measured at this height only (see plan_steps)
MEASURE_DETECTORS = ("v4-fast", "v4-accurate")   # the one not in use is measured too (density 0, see for_detector)
QUICK_D0_HEIGHT = 1080       # quick mode: d0 here too, so the per-frame cost has a fixed and a per-pixel part (see
                             # profile._interp); one height alone cannot tell them apart (1080p was off by -18%)

# Progress estimate: wall seconds per step on the reference machine (RTX 5060 laptop, v4-accurate, standard check at
# clip 240, 2026-10-13: restore 84-85 s (d0 at 3 heights + d1 at 720), the other detector 23 s, RIFE 5 + 2 s and
# encode 3 + 2 s per height, combined 72-73 s). Only the
# proportions matter: once a step has finished, the rest is rescaled by how long the finished steps really took.
EST_CHILD_S = 12.0           # child process start, imports, model load
EST_SOURCE_S = 7.0           # one synthetic source (ffmpeg); steps for another detector reuse the sources
EST_D0_S = 4.0               # one density-0 measurement
EST_DENSE_FPS = 28.0         # density >= 1: warm-up + WINDOW_MIN_PERIODS clips at about this rate (incl. fill)
EST_COMBINED_BASE_S = 29.0
EST_COMBINED_FPS = 28.0
EST_RIFE_S = (5.0, 2.0)      # base, per height
EST_ENCODE_S = (3.0, 2.0)
PROGRESS_TICK_S = 1.0
PROGRESS_HOLD = 0.95         # a running step stops at this share of its estimate until it really ends
PROGRESS_SCALE_LIMITS = (0.33, 3.0)

STREAMER_SRC = Path(__file__).resolve().parents[2]

Progress = Callable[..., None]   # progress(message, fraction, remaining_s=None)


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# pure helpers (unit tested; no torch)
# ---------------------------------------------------------------------------------------------------------------

def width_for(height: int) -> int:
    w = WIDTH_FOR.get(height)
    if w is None:
        w = int(round(height * 16 / 9 / 2)) * 2
    return w


def parse_heights(value: str | list | tuple) -> list[int]:
    parts = value.split(",") if isinstance(value, str) else list(value)
    try:
        heights = [int(str(p).strip().lower().rstrip("p")) for p in parts if str(p).strip()]
    except ValueError:
        raise ValueError(f"heights must be comma-separated integers like 480,720,1080: {value!r}") from None
    if not heights or any(h < 144 or h > 4320 or h % 2 for h in heights):
        raise ValueError(f"unsupported heights {value!r}")
    return list(dict.fromkeys(heights))


def region_boxes(frame_idx: int, density: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Synthetic mosaic regions (t, l, b, r) of one frame: `density` squares of BOX_HEIGHT_FRAC of the frame height,
    one per equal-width column (so they never overlap each other), each bouncing slowly (a few px per frame) inside
    its column so that consecutive boxes of a region overlap (Scene.belongs) and a scene reaches max_clip_length."""
    if density <= 0:
        return []
    side = max(16, int(round(height * BOX_HEIGHT_FRAC)))
    col_w = width // density
    margin = max(2, int(width * 0.02))
    span_x = col_w - 2 * margin - side
    if span_x < 0:
        raise ValueError(f"density {density} does not fit a {width}x{height} frame")
    speed = max(2, int(round(height / 240)))
    span_y = max(0, int(height * 0.12))
    top0 = (height - side) // 2 - span_y // 2
    out = []
    for k in range(density):
        dx = _triangle(frame_idx * speed + k * 41, span_x)
        dy = _triangle(frame_idx + k * 17, span_y)
        l = k * col_w + margin + dx
        t = top0 + dy
        out.append((t, l, t + side, l + side))
    return out


def _triangle(pos: int, span: int) -> int:
    """Back-and-forth position in [0, span] for a monotonically growing pos."""
    if span <= 0:
        return 0
    s = pos % (2 * span)
    return s if s <= span else 2 * span - s


def boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class SteadyWindow:
    """Throughput of a pipeline that delivers frames in per-clip bursts: timing starts when frame `start_idx` (the
    first frame after WARMUP_CLIPS clips, so the cold start is excluded) arrives and ends when a frame at
    a whole number (>= min_periods) of clip periods later arrives (and at least min_s passed) or after cap_s.
    fps = frames / seconds between those two arrivals, exact for burst-aligned ends. A single period is not enough:
    on an RTX 5060 one-clip windows of the same run varied by up to 2.5x between runs. An aligned window that ends
    on a period boundary reports period / median(per-period seconds) instead, so one clip slowed by a transient
    (another app briefly on the GPU/CPU, a clock dip) does not drag the result down; fps_mean keeps the plain rate.
    A capped aligned window ends at its last period boundary: ending mid-burst counted part of a clip's frames
    against the time of whole ones (combined runs at clip 240/300 read 17-31 fps on one machine)."""

    def __init__(self, start_idx: int, period: int, min_s: float = WINDOW_MIN_S, cap_s: float = WINDOW_CAP_S,
                 aligned: bool = True, min_frames: int = WINDOW_MIN_FRAMES, min_periods: int = WINDOW_MIN_PERIODS):
        self.start_idx, self.period, self.min_s, self.cap_s = start_idx, max(1, period), min_s, cap_s
        self.aligned, self.min_frames, self.min_periods = aligned, min_frames, max(1, min_periods)
        self.t0: float | None = None
        self.end: tuple[int, float] | None = None
        self.capped = False
        self.boundaries: list[float] = []      # arrival times of the burst-aligned frames, from start_idx on

    def update(self, idx: int, now: float) -> bool:
        """Call for every delivered frame (0-based index). True when the window is complete."""
        if self.end is not None:
            return True
        if idx < self.start_idx:
            return False
        if idx == self.start_idx:
            self.t0 = now
            self.boundaries = [now]
            return False
        elapsed, n = now - self.t0, idx - self.start_idx
        if n % self.period == 0:
            self.boundaries.append(now)
        if elapsed >= self.cap_s:
            self.capped = True
            k = len(self.boundaries) - 1
            if self.aligned and k >= 1:                  # back to the last whole clip, never mid-burst
                self.end = (self.start_idx + k * self.period, self.boundaries[k])
            else:
                self.end = (idx, now)
        elif elapsed >= self.min_s and (not self.aligned
                                        or (n % self.period == 0 and n >= self.min_periods * self.period)):
            self.end = (idx, now)
        return self.end is not None

    def finish_eof(self, last_idx: int, now: float):
        """The video ended first: use everything after the start if that is enough frames."""
        if self.end is None and self.t0 is not None and last_idx - self.start_idx >= self.min_frames:
            self.end = (last_idx, now)

    def result(self) -> dict | None:
        if self.end is None or self.t0 is None:
            return None
        idx, now = self.end
        frames, secs = idx - self.start_idx, now - self.t0
        if frames <= 0 or secs <= 0:
            return None
        out = {"fps": frames / secs, "frames": frames, "window_s": secs, "capped": self.capped}
        spans = [b - a for a, b in zip(self.boundaries, self.boundaries[1:])]
        if self.aligned and frames % self.period == 0 and spans:
            out["period_s"] = [round(x, 2) for x in spans]
            if len(spans) >= 3 and (mid := statistics.median(spans)) > 0:
                out["fps_mean"], out["fps"] = out["fps"], self.period / mid
        return out


def detector_lead_frames(clip_length: int) -> int:
    """How far (in frames, at one region per frame) Lada's detector can run ahead of the restored output: the
    mosaic_clip_queue holds max(1, 512 MiB // (clip_length * 256 * 256 * 4)) clips, plus one clip being restored and
    one being built."""
    L = max(1, int(clip_length))
    return (max(1, CLIP_QUEUE_BYTES // (L * 256 * 256 * 4)) + 2) * L


def source_seconds(clip_length: int, fps: int = FPS) -> int:
    """Synthetic source length such that the detector is still running when the d>=1 window ends.

    Once the detector reaches the end of the file the restorer has the GPU to itself and output speeds up ~20%
    (RTX 5060, 1080p d=1: 44 fps while detecting, 54 after): a source shorter than window end + detector lead
    measures that tail instead of the steady state a real video runs in."""
    L = max(1, int(clip_length))
    frames = (WARMUP_CLIPS + WINDOW_MIN_PERIODS) * L + detector_lead_frames(L) + SOURCE_MARGIN_FRAMES
    return math.ceil(frames / fps)


def parse_result(text: str) -> dict | None:
    """Last well-formed `PERF_RESULT {json}` object in a child's output, or None."""
    found = None
    for line in text.splitlines():
        i = line.find(RESULT_PREFIX)
        if i < 0:
            continue
        try:
            obj = json.loads(line[i + len(RESULT_PREFIX):])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found = obj
    return found


def source_cmd(width: int, height: int, path: str, seconds: int, fps: int = FPS) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={seconds}",
            "-vf", "noise=alls=7:allf=t", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-an", path]


def child_command(kind: str, payload: dict) -> list[str]:
    arg = json.dumps({**payload, "kind": kind})
    if getattr(sys, "frozen", False):          # frozen_entry routes --perf to oh_streamer.perf.__main__
        return [sys.executable, "--perf", "child", arg]
    return [sys.executable, "-m", "oh_streamer.perf", "child", arg]


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(STREAMER_SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


MODEL_KEYS = ("detector", "restoration", "clip_length")


def default_options() -> dict:
    return {"mode": "default", "heights": None, "device": "auto", "out": None, "timeout": DEFAULT_TIMEOUT,
            "keep_temp": False, "combined_height": 720, "workdir": None,
            "detector": DETECTOR, "restoration": RESTORATION, "clip_length": CLIP_LENGTH}


def normalize_options(options: dict | None) -> dict:
    o = {**default_options(), **(options or {})}
    if o["mode"] not in ("default", "quick", "full"):
        raise ValueError(f"unknown mode {o['mode']!r}")
    if o["heights"] is None:
        o["heights"] = [720] if o["mode"] == "quick" else list(HEIGHTS_ALL)
    else:
        o["heights"] = parse_heights(o["heights"])
    o["timeout"] = float(o["timeout"])
    if not (o["timeout"] > 0 and math.isfinite(o["timeout"])):
        raise ValueError("timeout must be positive")
    # model settings: None means "use the default"
    defaults = default_options()
    for key in ("detector", "restoration"):
        o[key] = defaults[key] if o[key] is None else str(o[key]).strip()
        if not o[key]:
            raise ValueError(f"{key} must not be empty")
    o["clip_length"] = defaults["clip_length"] if o["clip_length"] is None else o["clip_length"]
    if isinstance(o["clip_length"], float) and o["clip_length"].is_integer():
        o["clip_length"] = int(o["clip_length"])
    if isinstance(o["clip_length"], bool) or not isinstance(o["clip_length"], int) or o["clip_length"] < 1:
        raise ValueError(f"clip_length must be an integer >= 1, got {o['clip_length']!r}")
    return o


def _model_settings(args: dict) -> dict:
    """detector / restoration / clip_length from a child payload (module defaults when absent)."""
    o = normalize_options({k: args[k] for k in MODEL_KEYS if args.get(k) is not None})
    return {k: o[k] for k in MODEL_KEYS}


def plan_steps(options: dict) -> list[dict]:
    """Ordered measurement steps; the encoders are resolved later (they need an ffmpeg probe).

    Density >= 1 is measured at one height only (DENSE_HEIGHT, else the middle one): a restored clip frame is a
    256x256 crop whatever the video size, and on an RTX 5060 d1 was 40-42 fps at 480/720/1080p alike once warm-up
    was excluded. The per-frame part (density 0: decode, detection, frame handling) grows with pixels and is
    measured at every height, in the same child. Other detection models (MEASURE_DETECTORS) get density 0 only:
    detection is a per-frame cost. --full keeps every density at every height (diagnostics)."""
    o = normalize_options(options)
    heights, mode = o["heights"], o["mode"]
    if mode == "full":
        steps = [{"kind": "restore", "height": h, "densities": [0, 1, 2]} for h in heights]
        d0_plan = [(h, []) for h in heights]
    else:
        dh = DENSE_HEIGHT if DENSE_HEIGHT in heights else heights[len(heights) // 2]
        rest = [h for h in heights if h != dh]
        if mode == "quick" and QUICK_D0_HEIGHT not in heights:
            rest.append(QUICK_D0_HEIGHT)
        step = {"kind": "restore", "height": dh, "densities": [0, 1]}
        if rest:
            step["d0_heights"] = rest
        steps = [step]
        d0_plan = [(dh, rest)]
    if mode != "quick":
        for det in MEASURE_DETECTORS:
            if det == o["detector"]:
                continue
            for h, rest in d0_plan:
                step = {"kind": "restore", "height": h, "densities": [0], "detector": det}
                if rest:
                    step["d0_heights"] = rest
                steps.append(step)
    steps.append({"kind": "rife", "heights": heights})
    steps.append({"kind": "encode", "family": "hevc", "heights": heights})
    if mode == "full":
        steps.append({"kind": "encode", "family": "h264", "heights": heights})
        steps.append({"kind": "encode", "family": "libx264", "heights": heights})
    if mode != "quick":
        ch = o["combined_height"] if o["combined_height"] in heights else heights[min(len(heights) - 1, 1)]
        steps.append({"kind": "combined", "height": ch})
    for s in steps:
        s["est_s"] = step_seconds(s, measure_clip_length(o["clip_length"]))
    return steps


def measure_clip_length(clip_length: int) -> int:
    """Clip length the measurement runs with (the profile still records the user's setting)."""
    return min(int(clip_length), MEASURE_CLIP_MAX)


def _dense_window_s(clip_length: int, fps: float) -> float:
    """A density >= 1 measurement: WARMUP_CLIPS warm-up clips, then WINDOW_MIN_PERIODS clips (at most WINDOW_CAP_S)."""
    L = max(1, int(clip_length))
    return WARMUP_CLIPS * L / fps + min(WINDOW_MIN_PERIODS * L / fps, WINDOW_CAP_S)


def step_seconds(step: dict, clip_length: int) -> float:
    """Expected wall time of one planned step on the reference machine (progress bar only)."""
    kind = step["kind"]
    if kind == "restore":
        n0 = 1 + len(step.get("d0_heights") or [])
        dense = sum(1 for d in step["densities"] if d > 0)
        source = 0.0 if step.get("detector") else EST_SOURCE_S
        return EST_CHILD_S + n0 * (source + EST_D0_S) + dense * _dense_window_s(clip_length, EST_DENSE_FPS)
    if kind == "rife":
        return EST_RIFE_S[0] + EST_RIFE_S[1] * len(step["heights"])
    if kind == "encode":
        return EST_ENCODE_S[0] + EST_ENCODE_S[1] * len(step["heights"])
    return EST_COMBINED_BASE_S + _dense_window_s(clip_length, EST_COMBINED_FPS)


class ProgressClock:
    """Time-based progress over planned steps.

    A running step advances with its elapsed time against its estimate and holds at PROGRESS_HOLD of it until the
    step really ends. Finished steps count with their real duration; the steps still to come are rescaled by
    (real / estimated) of the finished ones, which is also what makes the remaining time meaningful."""

    def __init__(self, estimates, clock: Callable[[], float] = time.monotonic):
        self._est = [max(0.0, float(e)) for e in estimates]
        self._clock = clock
        self._lock = threading.Lock()
        self._actual: dict[int, float] = {}       # finished step -> real seconds (skipped steps: 0, not measured)
        self._measured_est = 0.0
        self._measured_actual = 0.0
        self._current: int | None = None
        self._t0 = 0.0

    def start(self, i: int) -> None:
        with self._lock:
            self._current, self._t0 = i, self._clock()

    def finish(self, i: int, skipped: bool = False) -> float:
        """Step i ended (returns its seconds); skipped = it did no real work, its time says nothing about the machine."""
        with self._lock:
            took = self._clock() - self._t0 if self._current == i else 0.0
            if skipped:
                self._actual[i] = 0.0
            else:
                self._actual[i] = took
                self._measured_est += self._est[i]
                self._measured_actual += took
            if self._current == i:
                self._current = None
            return took

    def scale(self) -> float:
        with self._lock:
            return self._scale()

    def _scale(self) -> float:
        if self._measured_est <= 0 or self._measured_actual <= 0:
            return 1.0
        lo, hi = PROGRESS_SCALE_LIMITS
        return min(hi, max(lo, self._measured_actual / self._measured_est))

    def status(self) -> tuple[float, float | None]:
        """(fraction 0..1, remaining seconds or None before any step has been timed)."""
        with self._lock:
            s = self._scale()
            done = sum(self._actual.values())
            total, shown, left = done, done, 0.0
            for i, est in enumerate(self._est):
                if i in self._actual:
                    continue
                est *= s
                total += est
                if i == self._current:
                    elapsed = self._clock() - self._t0
                    shown += min(elapsed, PROGRESS_HOLD * est)
                    left += max(0.0, est - elapsed)
                else:
                    left += est
            frac = shown / total if total > 0 else 0.0
            return min(1.0, max(0.0, frac)), (left if self._measured_actual > 0 else None)


# --- profile assembly ----------------------------------------------------------------------------------------

def new_profile(options: dict | None = None) -> dict:
    model = normalize_options(options)
    return {"schema": SCHEMA, "created": None, "duration_s": None,
            "fingerprint": {"gpu": None, "driver": None, "backend": None, "device": None, "torch": None,
                            "lada": None, "streamer": None, "os": None, "vram_total_gib": None,
                            "engine": None, "engine_version": None},
            "conditions": {"on_ac_power": None, "battery_percent": None, "gpu_util_before_pct": None,
                           "power_limit_w": None},
            "settings": {"detector": model["detector"], "restoration": model["restoration"],
                         "clip_length": model["clip_length"], "fp16": None},
            "restore": {}, "restore_alt": {}, "rife": {}, "encode": {}, "combined": None, "calibration": {},
            "warnings": [],
            "detail": {}}


def merge_env(profile: dict, env: dict | None):
    """Take the first non-null torch-side facts a child reported (device, backend, torch, fp16, gpu ...)."""
    if not env:
        return
    fp = profile["fingerprint"]
    for key in ("gpu", "backend", "device", "torch", "lada", "vram_total_gib", "engine", "engine_version"):
        if fp.get(key) is None and env.get(key) is not None:
            fp[key] = env[key]
    if profile["settings"].get("fp16") is None and env.get("fp16") is not None:
        profile["settings"]["fp16"] = bool(env["fp16"])
    if env.get("engine") == "lada" and env.get("engine_version"):
        from oh_streamer.engines.lada import compat
        result = compat.evaluate(env["engine_version"])
        if not result.ok and result.message not in profile["warnings"]:
            profile["warnings"].append(result.message)


def _r(value, nd=2):
    return None if value is None else round(float(value), nd)


def _alt_view(profile: dict, detector: str) -> dict:
    """merge_restore target for another detection model: its restore section is profile.restore_alt[detector],
    its detail profile.detail.restore_alt[detector]; fingerprint / settings / warnings are shared."""
    return {"fingerprint": profile["fingerprint"], "settings": profile["settings"], "warnings": profile["warnings"],
            "restore": profile.setdefault("restore_alt", {}).setdefault(detector, {}),
            "detail": profile["detail"].setdefault("restore_alt", {}).setdefault(detector, {})}


def merge_restore(profile: dict, height: int, result: dict | None, error: str | None = None, densities=(0, 1)):
    """result = the restore child's payload (None when the child failed -> `error`)."""
    entry = {"d0_fps": None, "d1_fps": None, "d2_fps": None, "vram_peak_gib": None, "error": None}
    detail = {}
    errors = []
    if result is None:
        errors.append(error or "no result")
    else:
        merge_env(profile, result.get("env"))
        if result.get("error"):
            errors.append(str(result["error"]))
        peaks = []
        for key, d in (result.get("densities") or {}).items():
            try:
                di = int(key)
            except ValueError:
                continue
            if d.get("error"):
                errors.append(f"d{di}: {d['error']}")
            if d.get("fps") is not None and f"d{di}_fps" in entry:
                entry[f"d{di}_fps"] = _r(d["fps"], 1)
            if d.get("vram_peak_gib") is not None:
                peaks.append(d["vram_peak_gib"])
            detail[f"d{di}"] = {k: d[k] for k in ("frames", "window_s", "capped", "period_s", "clips", "clip_frames",
                                                 "detector_done_early") if k in d}
            if d.get("detector_done_early"):
                _warn_detector_done(profile, f"restore {height}p d={di}")
        entry["vram_peak_gib"] = _r(max(peaks), 2) if peaks else None
        for d in densities:                      # asked for but missing from the payload
            if f"d{d}_fps" in entry and entry[f"d{d}_fps"] is None and not any(e.startswith(f"d{d}:") for e in errors):
                errors.append(f"d{d}: no measurement")
    entry["error"] = "; ".join(errors) or None
    profile["restore"][str(height)] = entry
    if detail:
        profile["detail"].setdefault("restore", {})[str(height)] = detail


def merge_rife(profile: dict, heights, result: dict | None, error: str | None = None):
    got = (result or {}).get("heights") or {}
    if result is not None:
        merge_env(profile, result.get("env"))
    for h in heights:
        d = got.get(str(h))
        if d is None:
            profile["rife"][str(h)] = {"pair_ms": None, "error": (result or {}).get("error") or error or "no result"}
        else:
            profile["rife"][str(h)] = {"pair_ms": _r(d.get("pair_ms"), 2), "error": d.get("error")}
            if d.get("mode"):
                profile["detail"].setdefault("rife", {})[str(h)] = {"mode": d["mode"], "pairs": d.get("pairs")}


def merge_encode(profile: dict, encoder: str, heights, result: dict | None, error: str | None = None):
    per = profile["encode"].setdefault(encoder, {})
    got = (result or {}).get("heights") or {}
    for h in heights:
        d = got.get(str(h))
        if d is None:
            per[str(h)] = {"fps": None, "error": (result or {}).get("error") or error or "no result"}
        else:
            per[str(h)] = {"fps": _r(d.get("fps"), 1), "error": d.get("error")}


def merge_combined(profile: dict, height: int, encoder: str, result: dict | None, error: str | None = None):
    entry = {"height": height, "density": 1.0, "encoder": encoder, "rife": "full", "fps": None, "error": None}
    if result is None:
        entry["error"] = error or "no result"
    else:
        merge_env(profile, result.get("env"))
        entry["fps"] = _r(result.get("fps"), 1)
        entry["error"] = result.get("error")
        if result.get("frames") is not None:
            profile["detail"]["combined"] = {k: result[k] for k in ("frames", "window_s", "capped", "period_s",
                                                                    "pairs", "detector_done_early") if k in result}
        if result.get("detector_done_early"):
            _warn_detector_done(profile, f"combined {height}p")
    profile["combined"] = entry


def _warn_detector_done(profile: dict, what: str):
    w = f"{what}: the test video ended inside the measurement window: result may be too high"
    if w not in profile["warnings"]:
        profile["warnings"].append(w)


def build_warnings(conditions: dict) -> list[str]:
    out = []
    if conditions.get("on_ac_power") is False:
        out.append("on battery power: results may be lower than when plugged in")
    util = conditions.get("gpu_util_before_pct")
    if util is not None and util > BUSY_GPU_PCT:
        out.append(f"GPU was busy before the test ({util:.0f}% utilization): results may be lower; "
                   f"close other GPU programs and run again")
    return out


# ---------------------------------------------------------------------------------------------------------------
# parent: system conditions, source generation, child processes
# ---------------------------------------------------------------------------------------------------------------

def _gpu_index(device: str) -> int:
    m = re.fullmatch(r"cuda:(\d+)", device or "")
    return int(m.group(1)) if m else 0


def _meta_version(*names: str) -> str | None:
    from importlib import metadata
    for name in names:
        try:
            return metadata.version(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def read_system_conditions(device: str = "auto") -> dict:
    """Battery / power / GPU facts without importing torch (psutil, nvidia-ml-py; every probe may fail -> None)."""
    cond = {"on_ac_power": None, "battery_percent": None, "gpu_util_before_pct": None, "power_limit_w": None}
    fp = {"gpu": None, "driver": None, "vram_total_gib": None}
    try:
        import psutil
        b = psutil.sensors_battery()
        if b is not None:
            cond["on_ac_power"] = None if b.power_plugged is None else bool(b.power_plugged)
            cond["battery_percent"] = float(b.percent)
        else:
            cond["on_ac_power"] = True        # no battery (desktop) = mains
    except Exception:  # noqa: BLE001
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(_gpu_index(device))
            name = pynvml.nvmlDeviceGetName(h)
            fp["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
            drv = pynvml.nvmlSystemGetDriverVersion()
            fp["driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
            try:
                fp["vram_total_gib"] = round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 2)
            except Exception:  # noqa: BLE001
                pass
            try:
                cond["power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1)
            except Exception:  # noqa: BLE001
                pass
            samples = []
            for _ in range(4):
                try:
                    samples.append(float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
                except Exception:  # noqa: BLE001
                    break
                time.sleep(0.15)
            if samples:
                cond["gpu_util_before_pct"] = round(sum(samples) / len(samples), 1)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    return {"conditions": cond, "fingerprint": fp}


def terminate_process_tree(process: subprocess.Popen):
    """Kill a child we started and its descendants (the venv launcher's real interpreter, ffmpeg). Only this tree."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


STARTED_PIDS: list[int] = []        # every process the benchmark started (diagnostics / manual cleanup)


def run_child(kind: str, payload: dict, *, timeout: float, log_dir: Path, workdir: Path,
              cancel: threading.Event | None = None, log: Callable[[str], None] = lambda m: None,
              tag: str | None = None) -> tuple[dict | None, str | None]:
    """Run one measurement in a child process. Returns (payload, None) or (None, error)."""
    tag = tag or kind
    out_path, err_path = log_dir / f"{tag}.out.txt", log_dir / f"{tag}.err.txt"
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(child_command(kind, payload), stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                   cwd=str(workdir), env=child_env(), creationflags=flags,
                                   start_new_session=sys.platform != "win32")
        STARTED_PIDS.append(process.pid)
        log(f"child {tag} pid={process.pid}")
        deadline = time.monotonic() + timeout
        error = None
        try:
            while True:
                try:
                    process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel is not None and cancel.is_set():
                    terminate_process_tree(process)
                    raise Cancelled()
                if time.monotonic() > deadline:
                    terminate_process_tree(process)
                    error = f"timed out after {timeout:g}s"
                    break
        except BaseException:
            terminate_process_tree(process)
            raise
    result = parse_result(out_path.read_text(encoding="utf-8", errors="replace"))
    if result is None:
        if error is None:
            tail = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] or [""]
            error = f"child exited with status {process.returncode} without a result" + (f" ({tail[0][:200]})" if tail[0] else "")
        return None, error
    if error:
        result.setdefault("error", error)
    return result, None


def ensure_source(height: int, tmp: Path, log=lambda m: None, cancel=None, clip_length: int = CLIP_LENGTH) -> Path:
    seconds = source_seconds(clip_length)
    path = tmp / f"synthetic_{height}p_{seconds}s.mp4"
    if path.is_file() and path.stat().st_size > 0:
        return path
    cmd = source_cmd(width_for(height), height, str(path), seconds)
    log(f"generating synthetic {width_for(height)}x{height} source")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)
    STARTED_PIDS.append(p.pid)
    try:
        while True:
            try:
                _, err = p.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    terminate_process_tree(p)
                    raise Cancelled()
    except BaseException:
        terminate_process_tree(p)
        raise
    if p.returncode != 0 or not path.is_file():
        raise RuntimeError(f"ffmpeg could not generate the {height}p test source: "
                           f"{(err or b'').decode('utf-8', 'replace').strip()[-300:]}")
    return path


def _resolve_family(family: str) -> tuple[str, str | None]:
    """(encoder, warning) for 'hevc' / 'h264' / a concrete software encoder, using the platform default policy."""
    from oh_streamer.processing import encoding_backend as eb
    if family in ("libx264", "libx265"):
        return family, None
    enc = eb.resolve_encoder(family)
    warn = None
    if not eb.is_hardware_encoder(enc):
        warn = f"no usable hardware {family} encoder: measured the software fallback {enc}"
    return enc, warn


class _KeepDisplayOn:
    """Keep the system and the display awake while the check runs (Windows: per-thread execution state, cleared on
    exit; macOS: caffeinate -di). The check takes minutes with nobody touching the computer, and on a laptop the
    idle display dimming / power-saving that follows can come with power-management changes that slow a
    measurement (suspected in one run where RIFE and 1080p restore dropped ~30%)."""

    def __enter__(self):
        self._proc = None
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000003))  # CONTINUOUS|SYSTEM|DISPLAY
            elif sys.platform == "darwin":
                self._proc = subprocess.Popen(["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())],
                                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001 - a nicety: never fail the check over it
            print(f"[perf] could not keep the display on: {e!r}", file=sys.stderr, flush=True)
        return self

    def __exit__(self, *exc):
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000))
            elif self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        return False


def run_benchmark(options: dict | None = None, progress: Progress | None = None,
                  cancel: threading.Event | None = None) -> dict:
    """Run the whole check and return the profile dict (schema 1). Raises Cancelled when `cancel` is set.
    options: mode ('default' | 'quick' | 'full'), heights, device, timeout (per child, s), keep_temp, out (path to
    write the JSON to), combined_height, workdir."""
    with _KeepDisplayOn():
        return _run_benchmark(options, progress, cancel)


def _run_benchmark(options: dict | None, progress: Progress | None, cancel: threading.Event | None) -> dict:
    o = normalize_options(options)
    steps = plan_steps(o)
    clip = measure_clip_length(o["clip_length"])
    clock = ProgressClock(s["est_s"] for s in steps)
    t_start = time.perf_counter()

    def report(msg: str, frac: float | None = None):
        if progress is None:
            return
        f, remaining = clock.status()
        try:
            progress(msg, f if frac is None else frac, None if frac is not None else remaining)
        except Exception:  # noqa: BLE001 - a UI callback must not break the measurement
            pass

    def say(msg: str, frac: float | None = None):
        print(f"[perf] {msg}", file=sys.stderr, flush=True)
        report(msg, frac)

    ticking = threading.Event()

    def tick():
        while not ticking.wait(PROGRESS_TICK_S):
            report("")

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    profile = new_profile(o)
    profile["detail"]["measure_clip_length"] = clip
    profile["created"] = datetime.now().astimezone().isoformat(timespec="seconds")
    sysinfo = read_system_conditions(o["device"])
    profile["conditions"].update(sysinfo["conditions"])
    for k, v in sysinfo["fingerprint"].items():
        if v is not None:
            profile["fingerprint"][k] = v
    profile["fingerprint"].update(
        torch=_meta_version("torch"), lada=_meta_version("lada"),
        streamer=_meta_version("oh-streamer", "oh_streamer") or "0.1.0", os=_platform.platform())
    profile["warnings"].extend(build_warnings(profile["conditions"]))

    own_tmp = o["workdir"] is None
    tmp = Path(o["workdir"]) if o["workdir"] else Path(tempfile.mkdtemp(prefix="oh_perf_"))
    tmp.mkdir(parents=True, exist_ok=True)
    log_dir = tmp / "logs"
    log_dir.mkdir(exist_ok=True)
    ticker = threading.Thread(target=tick, name="perf-progress", daemon=True) if progress is not None else None
    if ticker is not None:
        ticker.start()
    try:
        for i, step in enumerate(steps):
            check_cancel()
            clock.start(i)
            kind = step["kind"]
            common = {"device": o["device"], **{k: o[k] for k in MODEL_KEYS}, "clip_length": clip}
            if kind == "restore":
                h, det = step["height"], step.get("detector")
                hs = ",".join(map(str, [h, *(step.get("d0_heights") or [])]))
                say(f"detector {det} {hs}p d=0 ..." if det else f"restore {hs}p d={','.join(map(str, step['densities']))} ...")
                try:
                    src = ensure_source(h, tmp, lambda m: say(m), cancel, clip)
                    d0_sources = [{"height": x, "source": str(ensure_source(x, tmp, lambda m: say(m), cancel,
                                                                            clip))}
                                  for x in step.get("d0_heights") or []]
                    res, err = run_child("restore", {**common, **({"detector": det} if det else {}), "source": str(src),
                                                     "height": h, "densities": step["densities"],
                                                     "d0_sources": d0_sources},
                                         timeout=o["timeout"], log_dir=log_dir, workdir=tmp, cancel=cancel,
                                         log=lambda m: say(m), tag=f"restore_{h}" + (f"_{det}" if det else ""))
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    res, err = None, f"{type(e).__name__}: {e}"
                target = _alt_view(profile, det) if det else profile
                merge_restore(target, h, res, err, step["densities"])
                for x in step.get("d0_heights") or []:
                    got = ((res or {}).get("d0_extra") or {}).get(str(x))
                    merge_restore(target, x, {"densities": {"0": got}} if got else None,
                                  err or "no d0 measurement", (0,))
            elif kind == "rife":
                say(f"RIFE {','.join(map(str, step['heights']))}p ...")
                res, err = run_child("rife", {**common, "heights": step["heights"]}, timeout=o["timeout"],
                                     log_dir=log_dir, workdir=tmp, cancel=cancel, log=lambda m: say(m), tag="rife")
                merge_rife(profile, step["heights"], res, err)
            elif kind == "encode":
                try:
                    enc, warn = _resolve_family(step["family"])
                except Exception as e:  # noqa: BLE001
                    enc, warn = step["family"], None
                    merge_encode(profile, enc, step["heights"], None, f"{type(e).__name__}: {e}")
                    clock.finish(i, skipped=True)
                    continue
                if enc in profile["encode"] and any(v.get("fps") is not None for v in profile["encode"][enc].values()):
                    clock.finish(i, skipped=True)
                    continue                   # same encoder as an earlier family (e.g. libx264 fallback)
                if warn and warn not in profile["warnings"]:
                    profile["warnings"].append(warn)
                say(f"encode {enc} {','.join(map(str, step['heights']))}p ...")
                res, err = run_child("encode", {"encoder": enc, "heights": step["heights"], **common},
                                     timeout=o["timeout"],
                                     log_dir=log_dir, workdir=tmp, cancel=cancel, log=lambda m: say(m),
                                     tag=f"encode_{enc}")
                merge_encode(profile, enc, step["heights"], res, err)
            elif kind == "combined":
                h = step["height"]
                try:
                    enc, warn = _resolve_family("hevc")
                except Exception:  # noqa: BLE001
                    enc, warn = "libx265", None
                say(f"combined {h}p restore d=1 + RIFE full + {enc} ...")
                try:
                    src = ensure_source(h, tmp, lambda m: say(m), cancel, clip)
                    res, err = run_child("combined", {**common, "source": str(src), "height": h, "encoder": enc},
                                         timeout=o["timeout"], log_dir=log_dir, workdir=tmp, cancel=cancel,
                                         log=lambda m: say(m), tag=f"combined_{h}")
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001
                    res, err = None, f"{type(e).__name__}: {e}"
                merge_combined(profile, h, enc, res, err)
            took = clock.finish(i)
            say(f"{kind} done in {took:.0f} s (reference estimate {step['est_s']:.0f} s)")
    finally:
        ticking.set()
        if ticker is not None:
            ticker.join(timeout=PROGRESS_TICK_S + 1)
        if own_tmp and not o["keep_temp"]:
            shutil.rmtree(tmp, ignore_errors=True)
        elif o["keep_temp"]:
            print(f"[perf] kept temp dir {tmp}", file=sys.stderr, flush=True)
    profile["duration_s"] = round(time.perf_counter() - t_start, 1)
    if not profile["fingerprint"].get("driver") and not profile["fingerprint"].get("gpu"):
        profile["warnings"].append("GPU information unavailable (nvidia-ml-py / NVIDIA driver not found)")
    say("finished", 1.0)
    if o.get("out"):
        out = Path(o["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return profile


def format_summary(p: dict) -> str:
    """Short text table of a profile (used by the CLI after `run`; profile.format_table is the full report)."""
    lines = [f"duration {p.get('duration_s')} s, {p['fingerprint'].get('gpu')} ({p['fingerprint'].get('device')}), "
             f"fp16={p['settings'].get('fp16')}"]
    lines.append("restore fps (source frames/s):  height   d0     d1     d2   vramGiB  error")
    for h, e in p["restore"].items():
        f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
        lines.append(f"  {h:>6}p {f(e['d0_fps']):>8} {f(e['d1_fps']):>6} {f(e['d2_fps']):>6} "
                     f"{'-' if e['vram_peak_gib'] is None else e['vram_peak_gib']:>8}  {e['error'] or ''}")
    for h, e in p["rife"].items():
        lines.append(f"  rife {h}p: {'-' if e['pair_ms'] is None else e['pair_ms']} ms/pair {e['error'] or ''}")
    for enc, per in p["encode"].items():
        for h, e in per.items():
            lines.append(f"  encode {enc} {h}p: {'-' if e['fps'] is None else e['fps']} fps {e['error'] or ''}")
    c = p.get("combined")
    if c:
        lines.append(f"  combined {c['height']}p ({c['encoder']}, RIFE full, d=1): "
                     f"{'-' if c['fps'] is None else c['fps']} fps {c['error'] or ''}")
    lines += [f"  warning: {w}" for w in p.get("warnings", [])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# child side (torch / lada / ffmpeg)
# ---------------------------------------------------------------------------------------------------------------

def _eprint(msg: str):
    print(f"[perf-child] {msg}", file=sys.stderr, flush=True)


def synthetic_frames(width: int, height: int, count: int, noise: int = 4):
    """A few distinct, smoothly shifting uint8 HWC BGR frames (consecutive ones similar: no scene cut)."""
    import numpy as np
    yy, xx = np.mgrid[0:height, 0:width]
    base = ((xx * 255 // max(1, width) + yy * 255 // max(1, height))[..., None] + np.array([0, 70, 140])) % 256
    rng = np.random.default_rng(1234)
    frames = []
    for i in range(count):
        f = np.roll(base, 7 * i, axis=1)
        if noise:
            f = f + rng.integers(-noise, noise + 1, size=f.shape)
        frames.append(np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8)))
    return frames


def _device_env(ls_or_none, device, fp16=None) -> dict:
    import torch
    from oh_streamer.processing import device_backend
    env = {"device": str(device), "backend": device_backend.backend_name(device, torch_module=torch),
           "torch": torch.__version__, "lada": _meta_version("lada"), "fp16": fp16}
    try:
        from oh_streamer.engines import get_engine
        info = get_engine().info()
        env["engine"], env["engine_version"] = info.name, info.version
    except Exception:  # noqa: BLE001 - informational only
        pass
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            env["gpu"] = props.name
            env["vram_total_gib"] = round(props.total_memory / 2**30, 2)
        except Exception:  # noqa: BLE001
            pass
    return env


def count_restored_clips(session) -> dict:
    """Wrap the session's clip restoration so the child can prove injected detections became restored clips."""
    return bench_hooks.count_restored_clips(session)


def install_synthetic_detections(session, density: int):
    """Replace the detector's scene builder on this instance: real YOLO inference still runs, but the scenes get
    `density` synthetic regions per frame (masks / boxes in the format convert_yolo_mask_tensor / convert_yolo_box
    produce: full-frame (H, W, 1) uint8 0/255 tensor on the frame's device, integer (t, l, b, r) box). The per-frame
    detection counts the detector worker puts on frame_detection_queue are computed from the scenes afterwards, so
    they stay consistent with what was injected. density 0 adds nothing (no mosaic anywhere)."""
    bench_hooks.install_synthetic_detections(session, density, region_boxes)


def _load_lada(device_arg: str):
    """(oh_player module: torch + Source, the Lada engine, device, fp16). The measurements need the engine's
    benchmark hooks (replaced detections, restored-clip counts)."""
    import oh_streamer.player.oh_player as ls
    from oh_streamer.engines import get_engine
    from oh_streamer.engines.lada.adapter import gpu_has_fp16_acceleration
    from oh_streamer.processing import device_backend
    engine = get_engine()
    if engine_base.BENCHMARK_HOOKS not in engine.info().capabilities:
        raise RuntimeError(f"engine {engine.info().name!r} has no benchmark hooks")
    device = device_backend.select_device(device_arg, torch_module=ls.torch)
    fp16 = device_backend.fp16_default(device, gpu_has_fp16_acceleration, torch_module=ls.torch)
    return ls, engine, device, fp16


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stop_restorer(session):
    session.stop(abandon=True)


def watch_detector_done(session) -> dict:
    """Record when the detector finishes the file (it puts EOF on frame_detection_queue): {"t": perf_counter or None}."""
    return bench_hooks.watch_detector_done(session)


def detector_done_early(window: SteadyWindow, seen: dict) -> bool:
    """True when the detector finished before the window ended (part of the window ran without detection load)."""
    return window.end is not None and seen.get("t") is not None and seen["t"] < window.end[1]


def _consume(session, window: SteadyWindow, on_frame=None, max_frames: int | None = None) -> dict | None:
    """Iterate a started session until the steady window is complete (or the video ends)."""
    idx, last = 0, None
    for elem in session.frames():
        if isinstance(elem, (engine_base.EndOfStream, engine_base.EngineError)):
            raise RuntimeError(f"restorer stopped: {getattr(elem, 'reason', elem)}")
        now = time.perf_counter()
        last = now
        if on_frame is not None:
            on_frame(elem, idx)
            now = last = time.perf_counter()
        if window.update(idx, now):
            return window.result()
        idx += 1
    if last is not None:
        window.finish_eof(idx - 1, last)
    return window.result()


def child_restore(args: dict) -> dict:
    ls, engine, device, fp16 = _load_lada(args["device"])
    torch = ls.torch
    result = {"kind": "restore", "height": args["height"], "env": _device_env(ls, device, fp16), "densities": {}}
    source = ls.Source(str(Path(args["source"]).resolve()), {})
    _eprint(f"source {source.describe()}")
    m = _model_settings(args)
    t = time.perf_counter()
    engine.load(device, engine_base.ModelChoice(m["detector"], m["restoration"], bool(fp16)), m["clip_length"])
    _sync(torch, device)
    _eprint(f"models loaded in {time.perf_counter() - t:.1f}s")
    for d in args["densities"]:
        result["densities"][str(d)] = _measure_restore(ls, engine, device, source, d, m["clip_length"])
    extra = {}
    for item in args.get("d0_sources") or []:
        src = ls.Source(str(Path(item["source"]).resolve()), {})
        _eprint(f"d0-only source {src.describe()}")
        extra[str(item["height"])] = _measure_restore(ls, engine, device, src, 0, m["clip_length"])
    if extra:
        result["d0_extra"] = extra
    return result


def _measure_restore(ls, engine, device, source, d: int, clip_length: int) -> dict:
    """One steady-state restore measurement at density d on `source` (models already loaded in `engine`)."""
    torch = ls.torch
    entry = {"fps": None, "vram_peak_gib": None, "error": None}
    restorer = None
    try:
        restorer = engine.open(source)
        install_synthetic_detections(restorer, d)
        clip_stats = count_restored_clips(restorer)
        det_done = watch_detector_done(restorer)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        window = SteadyWindow((WARMUP_CLIPS if d > 0 else 1) * clip_length, clip_length, aligned=d > 0)
        restorer.start(0.0)
        r = _consume(restorer, window)
        entry.update(clip_stats)
        entry["detector_done_early"] = detector_done_early(window, det_done)
        if device.type == "cuda":
            entry["vram_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if r is None:
            entry["error"] = "video ended before a steady-state window could be measured"
        elif d > 0 and clip_stats["clips"] == 0:
            entry["error"] = "synthetic detections produced no restored clips (injection failed)"
        else:
            entry.update(r)
        _eprint(f"d={d}: {entry}")
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)
    finally:
        if restorer is not None:
            try:
                _stop_restorer(restorer)
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return entry


def _build_rife(device, w: int, h: int, timeout: float = 120.0):
    from oh_streamer.processing import rife_interp
    interp = rife_interp.Interpolator(device, log=_eprint)
    interp.prepare(w, h)
    deadline = time.monotonic() + timeout
    while True:
        rife = interp.get(w, h)
        if rife is not None:
            return interp, rife
        err = interp.error_for(w, h)
        if err:
            raise RuntimeError(f"RIFE could not be built: {err}")
        if time.monotonic() > deadline:
            raise RuntimeError("RIFE model build timed out")
        time.sleep(0.02)


def child_rife(args: dict) -> dict:
    import torch
    from oh_streamer.processing import device_backend
    device = device_backend.select_device(args["device"], torch_module=torch)
    result = {"kind": "rife", "env": _device_env(None, device), "heights": {}}
    for h in args["heights"]:
        entry = {"pair_ms": None, "error": None}
        result["heights"][str(h)] = entry
        if not device_backend.rife_available(device):
            entry["error"] = "unsupported"
            continue
        try:
            w = width_for(h)
            interp, rife = _build_rife(device, w, h)
            frames = synthetic_frames(w, h, 6)
            n_frames = len(frames)
            i = 0
            for _ in range(20):                           # warm up (graph replay, allocator, pinned copies)
                rife.middle(frames[i % n_frames], frames[(i + 1) % n_frames])
                i += 1
            _sync(torch, device)
            # middle() returns a host array, so each call is complete when it returns: time chunks of RIFE_CHUNK
            # pairs and take the median chunk, robust to a transient (clock dip, another app) during the ~2 s
            n, t0, chunks = 0, time.perf_counter(), []
            tc = t0
            while n < 8 * RIFE_CHUNK or time.perf_counter() - t0 < 2.0:
                rife.middle(frames[i % n_frames], frames[(i + 1) % n_frames])
                i += 1
                n += 1
                if n % RIFE_CHUNK == 0:
                    now = time.perf_counter()
                    chunks.append(now - tc)
                    tc = now
                    if n >= 600:
                        break
            elapsed = time.perf_counter() - t0
            entry["pair_ms"] = statistics.median(chunks) / RIFE_CHUNK * 1000.0
            entry["pair_ms_mean"] = elapsed / n * 1000.0
            entry["pairs"] = n
            entry["mode"] = "CUDA graph" if rife.graph is not None else "eager"
            interp.release()
            del rife, interp
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=sys.stderr)
    return result


def encode_cmd(encoder: str, width: int, height: int, fps: float = 60.0, gop: int = 120) -> list[str]:
    from oh_streamer.processing import encoding_backend as eb
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", str(fps),
           "-i", "pipe:0", "-map", "0:v:0", "-an"]
    cmd += eb.encoder_args(encoder, purpose="segmented")
    cmd += eb.quality_args(encoder) + eb.gop_args(encoder, gop)
    return cmd + ["-f", "null", "-"]


def _spawn(cmd, **kw):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    return subprocess.Popen(cmd, creationflags=flags, **kw)


def child_encode(args: dict) -> dict:
    encoder = args["encoder"]
    result = {"kind": "encode", "encoder": encoder, "heights": {}}
    for h in args["heights"]:
        entry = {"fps": None, "error": None}
        result["heights"][str(h)] = entry
        proc = None
        try:
            w = width_for(h)
            frames = synthetic_frames(w, h, 8)
            proc = _spawn(encode_cmd(encoder, w, h), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
            i = 0
            for _ in range(24):                         # encoder start-up / first buffers
                proc.stdin.write(memoryview(frames[i % len(frames)]).cast("B"))
                i += 1
            t0, n = time.perf_counter(), 0
            while (n < 120 or time.perf_counter() - t0 < 2.0) and time.perf_counter() - t0 < 20.0:
                proc.stdin.write(memoryview(frames[i % len(frames)]).cast("B"))
                i += 1
                n += 1
            proc.stdin.close()
            code = proc.wait(timeout=60)
            elapsed = time.perf_counter() - t0
            if code != 0:
                err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
                raise RuntimeError(f"ffmpeg exited with status {code}: {err[-200:]}")
            entry["fps"] = n / elapsed
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            if proc is not None and proc.poll() is None:
                proc.kill()
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
    return result


def child_combined(args: dict) -> dict:
    ls, engine, device, fp16 = _load_lada(args["device"])
    torch = ls.torch
    from oh_streamer.processing import device_backend, rife_interp
    h = args["height"]
    result = {"kind": "combined", "env": _device_env(ls, device, fp16), "fps": None, "error": None}
    if not device_backend.rife_available(device):
        result["error"] = "unsupported"
        return result
    source = ls.Source(str(Path(args["source"]).resolve()), {})
    meta = source.meta
    w, hh = int(meta.video_width), int(meta.video_height)
    interp = rife_interp.Interpolator(device, log=_eprint)
    interp.prepare(w, hh)                                   # builds in the background while the models load
    m = _model_settings(args)
    engine.load(device, engine_base.ModelChoice(m["detector"], m["restoration"], bool(fp16)), m["clip_length"])
    doubler = rife_interp.FullDoubler(interp, float(meta.video_fps_exact), w, hh, _eprint)
    doubler._model()                                        # wait for the build / raise its error
    restorer = proc = None
    try:
        restorer = engine.open(source)
        install_synthetic_detections(restorer, 1)
        det_done = watch_detector_done(restorer)
        proc = _spawn(encode_cmd(args["encoder"], w, hh, fps=2 * FPS, gop=2 * m["clip_length"]), stdin=subprocess.PIPE,
                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        window = SteadyWindow(WARMUP_CLIPS * m["clip_length"], m["clip_length"], aligned=True)
        pairs = [0]

        def on_frame(elem, idx):
            frame = elem.image
            arr = frame.numpy() if isinstance(frame, torch.Tensor) else frame
            for out in doubler.push(frame, arr, None):
                proc.stdin.write(memoryview(out.reshape(-1)))
            pairs[0] += 1

        restorer.start(0.0)
        r = _consume(restorer, window, on_frame)
        if r is None:
            result["error"] = "video ended before a steady-state window could be measured"
        else:
            result.update(r)
            result["pairs"] = pairs[0]
            result["detector_done_early"] = detector_done_early(window, det_done)
        if device.type == "cuda":
            result["vram_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    except BrokenPipeError:
        err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip() if proc else ""
        result["error"] = f"encoder closed its input: {err[-200:]}"
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            proc.kill()
        if restorer is not None:
            _stop_restorer(restorer)
    return result


CHILD_KINDS = {"restore": child_restore, "rife": child_rife, "encode": child_encode, "combined": child_combined}


def child_main(payload_json: str) -> int:
    payload = json.loads(payload_json)
    kind = payload.get("kind")
    try:
        result = CHILD_KINDS[kind](payload)
        code = 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        result = {"kind": kind, "error": f"{type(e).__name__}: {e}"}
        code = 1
    print(RESULT_PREFIX + json.dumps(result, allow_nan=False, default=str), flush=True)
    return code


# ---------------------------------------------------------------------------------------------------------------
# CLI helpers shared with __main__
# ---------------------------------------------------------------------------------------------------------------

def build_run_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description="Measure this computer's restoration speed.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="720p only, d0/d1, RIFE 720, one encoder, no combined run")
    mode.add_argument("--full", action="store_true", help="adds density 2 and all encoders")
    ap.add_argument("--heights", default=None, help="comma-separated, default 480,720,1080 (720 with --quick)")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda:N")
    ap.add_argument("--out", default=None, help="write the profile JSON here")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per measurement child")
    ap.add_argument("--keep-temp", action="store_true", help="keep the synthetic sources and child logs")
    ap.add_argument("--combined-height", type=int, default=720)
    ap.add_argument("--detector", default=None, help=f"detection model (default {DETECTOR})")
    ap.add_argument("--restoration", default=None, help=f"restoration model (default {RESTORATION})")
    ap.add_argument("--clip-length", type=int, default=None, help=f"restoration clip length (default {CLIP_LENGTH})")
    return ap


def options_from_args(ns: argparse.Namespace) -> dict:
    return {"mode": "quick" if ns.quick else "full" if ns.full else "default", "heights": ns.heights,
            "device": ns.device, "out": ns.out, "timeout": ns.timeout, "keep_temp": ns.keep_temp,
            "combined_height": ns.combined_height, "detector": ns.detector, "restoration": ns.restoration,
            "clip_length": ns.clip_length}
