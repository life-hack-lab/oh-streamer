"""Performance profile: schema checks, speed estimation, verdicts, calibration, storage.

Pure standard library on purpose (imported by the queue app, which has no torch).
Rates are in *source* frames per
second; ``ratio`` = rate / src_fps (x real time).
"""

import copy
import json
import math
import os
import re
import tempfile
from pathlib import Path

SCHEMA = 1
DENSITY_PRESETS = {"light": 0.3, "typical": 0.7, "heavy": 1.2}
HEIGHTS = (480, 720, 1080)
MODES = ("off", "auto", "full")
LIVE_RATIO = 1.1
HEAVY_SEGMENT_S = 60
MAX_PREBUFFER_S = 600
MIN_PREBUFFER_S = 10   # heavy ratio in [1, LIVE_RATIO): no deficit on paper, but keep a margin
AUTO_INTERP_MIN = 0.3
MEASURE_CLIP_MAX = 240   # = bench.MEASURE_CLIP_MAX: longer clip settings are measured with 240-frame clips
INTERP_FPS_RANGE = (10.0, 40.0)   # = rife_interp.MIN/MAX_SOURCE_FPS: other sources stay 1x in the player and the queue
K_RANGE = (0.5, 3.0)
OBSERVATION_RANGE = (0.2, 5.0)
STALE_KEYS = ("gpu", "driver", "backend", "torch", "lada")
_SECTIONS = ("fingerprint", "restore", "rife", "encode")


# ---- helpers ---------------------------------------------------------------------

def _num(x) -> float | None:
    """A positive finite number, else None (tolerates nulls and junk)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) and x > 0 else None


def _section(profile, name) -> dict:
    s = profile.get(name) if isinstance(profile, dict) else None
    return s if isinstance(s, dict) else {}


def _by_height(section: dict, field: str, invert=False) -> dict[int, float]:
    """{height: value} from a {"720": {field: v}} section; invert=True returns 1/value."""
    out = {}
    for key, entry in section.items():
        v = _num(entry.get(field)) if isinstance(entry, dict) else None
        if v is not None and str(key).isdigit():
            out[int(key)] = 1.0 / v if invert else v
    return out


def _interp(values: dict[int, float], height, single="pixels") -> float | None:
    """Value at ``height`` from measured heights, linear in pixel count (~height^2).

    Costs are a fixed part plus a part proportional to pixels (RTX 5060, synthetic d0 frame time 2.3 / 3.4 / 7.6 ms
    at 360 / 1080 / 2160p), so outside the measured range the line through the two nearest points is extended,
    kept between "all fixed" (nearest value) and "all per-pixel" (nearest scaled by pixels). With one point,
    single="pixels" scales it by pixels, single="const" keeps it."""
    if not values or not isinstance(height, (int, float)) or height <= 0:
        return None
    if height in values:
        return values[height]
    hs = sorted(values)
    lo = max((h for h in hs if h < height), default=None)
    hi = min((h for h in hs if h > height), default=None)
    if lo is not None and hi is not None:
        w = (height ** 2 - lo ** 2) / (hi ** 2 - lo ** 2)
        return values[lo] + w * (values[hi] - values[lo])
    near = lo if lo is not None else hi
    scaled = values[near] * (height / near) ** 2
    if len(hs) < 2:
        return scaled if single == "pixels" else values[near]
    far = hs[-2] if lo is not None else hs[1]
    line = values[near] + (values[near] - values[far]) * (height ** 2 - near ** 2) / (near ** 2 - far ** 2)
    return _clamp(line, min(values[near], scaled), max(values[near], scaled))


def _fit(entry: dict) -> tuple[float, float] | None:
    """Least-squares t = a + b*d over the measured densities (0, 1, 2) of one height."""
    pts = [(d, 1.0 / v) for d, f in ((0, "d0_fps"), (1, "d1_fps"), (2, "d2_fps"))
           if (v := _num(entry.get(f))) is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    md, mt = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    b = sum((d - md) * (t - mt) for d, t in pts) / sum((d - md) ** 2 for d, _ in pts)
    return mt - b * md, b


def _density(density) -> float:
    if isinstance(density, str):
        if density not in DENSITY_PRESETS:
            raise ValueError(f"unknown density preset: {density!r}")
        return DENSITY_PRESETS[density]
    return max(0.0, float(density))


def _encoder_fps(profile, height, encoder=None) -> tuple[str | None, float | None]:
    """(name, fps at height). ``encoder``: concrete name, "hevc"/"h264" family, or None/"auto".
    Auto prefers hardware, then hevc, then speed; a concrete name that wasn't measured
    falls back to auto."""
    cands = {}
    for name, per_h in _section(profile, "encode").items():
        t = _interp(_by_height(per_h, "fps", invert=True), height) if isinstance(per_h, dict) else None
        if t:
            cands[name] = 1.0 / t
    if encoder in cands:
        return encoder, cands[encoder]
    if encoder in ("hevc", "h264"):
        fam = {n: f for n, f in cands.items() if n.startswith(encoder)}
        cands = fam or cands
    if not cands:
        return None, None
    best = max(cands, key=lambda n: (not n.startswith("lib"), "hevc" in n, cands[n]))
    return best, cands[best]


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---- validation ------------------------------------------------------------------

def validate(profile) -> list[str]:
    """Problems found in a profile (empty list = usable). Null values are tolerated."""
    if not isinstance(profile, dict):
        return ["profile is not an object"]
    probs = []
    if profile.get("schema") != SCHEMA:
        probs.append(f"schema is {profile.get('schema')!r}, expected {SCHEMA}")
    for name in _SECTIONS:
        if name not in profile:
            probs.append(f"missing section: {name}")
        elif profile[name] is not None and not isinstance(profile[name], dict):
            probs.append(f"section {name} is not an object")
    for h, e in _section(profile, "restore").items():
        if not str(h).isdigit():
            probs.append(f"restore: bad height key {h!r}")
        elif e is not None and not isinstance(e, dict):
            probs.append(f"restore.{h} is not an object")
    for enc, per_h in _section(profile, "encode").items():
        if per_h is not None and not isinstance(per_h, dict):
            probs.append(f"encode.{enc} is not an object")
    return probs


# ---- detection models --------------------------------------------------------------

def detectors(profile) -> list[str]:
    """Detection models the profile can estimate: the one it was measured with, then those in restore_alt."""
    main = _section(profile, "settings").get("detector")
    alt = [k for k, v in _section(profile, "restore_alt").items()
           if isinstance(v, dict) and any(isinstance(e, dict) and _num(e.get("d0_fps")) for e in v.values())]
    return ([main] if main else []) + [k for k in alt if k != main]


def measured_clip_length(profile) -> int | None:
    """Clip length the measurement ran with (detail.measure_clip_length; older profiles: settings.clip_length)."""
    v = _num(_section(profile, "detail").get("measure_clip_length"))
    if v is None:
        v = _num(_section(profile, "settings").get("clip_length"))
    return None if v is None else int(v)


def covers_clip_length(profile, clip_length) -> bool:
    """Whether the profile was measured the way ``clip_length`` would be measured now (unknown: True)."""
    measured, cur = measured_clip_length(profile), _num(clip_length)
    return measured is None or cur is None or measured == min(int(cur), MEASURE_CLIP_MAX)


def for_detector(profile, detector):
    """The profile as seen with ``detector``: itself when that is the measured one (or unknown / not covered),
    else a copy whose restore section uses restore_alt[detector].

    The other model is measured at density 0 only: detection is a per-frame cost (a = 1/d0), while the cost of a
    restored clip frame (b) belongs to the restoration model, so b is taken from the main measurement at the same
    height. The combined run used the main model, so its RIFE sharing factor k is computed from the original."""
    main = _section(profile, "settings").get("detector")
    alt = _section(profile, "restore_alt").get(detector) if detector else None
    if not detector or detector == main or detector not in detectors(profile) or not isinstance(alt, dict):
        return profile
    out = dict(profile)
    restore = {}
    for key, entry in alt.items():
        if not (isinstance(entry, dict) and str(key).isdigit()):
            continue
        new = dict(entry)
        d0 = _num(entry.get("d0_fps"))
        fit = _fit(_section(profile, "restore").get(str(key)) or {})
        if d0 is not None and fit and _num(entry.get("d1_fps")) is None:
            for d, field in ((1, "d1_fps"), (2, "d2_fps")):
                new[field] = 1.0 / (1.0 / d0 + fit[1] * d)
        restore[str(key)] = new
    out["restore"] = restore
    if isinstance(profile.get("combined"), dict):
        out["combined"] = {**profile["combined"], "k": rife_k(profile)}
    out["settings"] = {**_section(profile, "settings"), "detector": detector}
    return out


# ---- estimation ------------------------------------------------------------------

def restore_frame_time(profile, height, density="typical") -> float | None:
    """Seconds of restore work per source frame at ``density`` (clip-frames/frame).

    Per measured height t = a + b*d with a = 1/d0, b = 1/d1 - 1/d0 (least squares if d2 is present). a (decode,
    detection, per frame) and b (one restored clip frame: crop, BasicVSR++ at 256x256, blend) scale differently
    with resolution, so each is interpolated by pixel count on its own (see _interp). b is nearly independent of
    resolution (~19 ms at 360-1080p, ~23 ms at 2160p on an RTX 5060), so a single measured height keeps it as is,
    while a lone d0 is scaled by pixels. A height with only d0 still contributes a. None if no usable data.
    """
    d = _density(density)
    a_vals, b_vals = {}, {}
    for key, entry in _section(profile, "restore").items():
        if not (isinstance(entry, dict) and str(key).isdigit()):
            continue
        fit = _fit(entry)
        if fit:
            a_vals[int(key)], b_vals[int(key)] = fit
        elif (d0 := _num(entry.get("d0_fps"))) is not None:
            a_vals[int(key)] = 1.0 / d0
    a, b = _interp(a_vals, height), _interp(b_vals, height, single="const")
    if a is None or b is None:
        return None
    return max(a + b * d, 1e-4)


def rife_pair_ms(profile, height) -> float | None:
    """Milliseconds per interpolated (middle) frame at ``height``, or None."""
    return _interp(_by_height(_section(profile, "rife"), "pair_ms"), height)


def rife_k(profile) -> float:
    """GPU-sharing factor for RIFE full, from the combined run (default 1.0).

    Plain sum model says t = t_r + pair_ms; combined.fps (source frames/s) gives the real
    t, so k = (1/fps - t_r) / pair. Skipped (1.0) if the combined run was encode-limited
    or unusable; clamped to K_RANGE.
    """
    c = profile.get("combined") if isinstance(profile, dict) else None
    if not isinstance(c, dict) or c.get("error"):
        return 1.0
    if (k := _num(c.get("k"))) is not None:          # precomputed (for_detector: measured with another model)
        return _clamp(k, *K_RANGE)
    obs, h = _num(c.get("fps")), _num(c.get("height"))
    if obs is None or h is None:
        return 1.0
    d = c.get("density")
    t_r = restore_frame_time(profile, h, 1.0 if _num(d) is None else float(d))
    pair = rife_pair_ms(profile, h)
    if t_r is None or pair is None:
        return 1.0
    _, enc = _encoder_fps(profile, h, c.get("encoder"))
    if enc and obs >= 0.98 * enc / 2:
        return 1.0
    return _clamp((1.0 / obs - t_r) / (pair / 1000.0), *K_RANGE)


def calibration_factor(profile, mode, height) -> float | None:
    cal = _section(profile, "calibration").get(f"{mode}:{height}")
    return _num(cal.get("factor")) if isinstance(cal, dict) else None


def restored_fps(src_fps, restoration_fps="original"):
    """Frame rate a source is restored at: the player's restoration_fps "30" keeps every n-th frame
    (60 -> 30, 59.94 -> 29.97, 50 -> 25; processing/frame_rate.planned_fps). None for a bad fps."""
    f = _num(src_fps)
    if f is None or f <= 0:
        return None
    if restoration_fps in (None, "original"):
        return f
    from oh_streamer.processing.frame_rate import planned_fps   # stdlib only
    return float(planned_fps(f, restoration_fps))


def estimate(profile, *, height, src_fps, mode="off", density="typical", encoder=None,
             compare=None, calibrate=True, restoration_fps="original") -> dict:
    """Predict throughput for one configuration.

    Returns fps (source frames/s), ratio (fps/src_fps), bottleneck ("gpu"|"encode"),
    interp_fraction (share of frame gaps that get an interpolated frame: 0 off, 1 full,
    auto = clamp((1/src_fps - t_r)/t_rife, 0, 1)), calibrated (a calibration factor was
    applied), plus detail (t_restore, t_gpu, k, factor, encoder, encode_fps, notes).
    fps/ratio/bottleneck are None when the profile cannot answer.

    GPU time per source frame = t_r (+ k*pair_ms for full). Encode caps the rate at
    encode_fps / outputs-per-source-frame (2 for full, else 1; auto is not charged for its
    extra frames). compare="encode" also encodes the original stream: modelled simply as
    halving the encoder's rate; compare="copy"/None is free. calibrate=False skips the
    online calibration factor (used to compute the raw prediction for recording).
    A source outside INTERP_FPS_RANGE is never interpolated: auto / full are estimated as off
    (interp_ok False). restoration_fps "30" (player only) restores a faster source at a lower rate:
    the estimate is then that of a source at restored_fps (fps / ratio relative to it).
    """
    d = _density(density)
    src_fps = restored_fps(src_fps, restoration_fps) or src_fps
    res = {"height": height, "mode": mode, "density": d, "fps": None, "ratio": None,
           "bottleneck": None, "interp_fraction": None, "calibrated": False,
           "factor": None, "k": None, "t_restore": None, "t_gpu": None,
           "encoder": None, "encode_fps": None, "interp_ok": interp_eligible(src_fps), "notes": []}
    if mode in ("auto", "full") and not res["interp_ok"]:
        res["notes"].append(f"source fps {src_fps} outside {INTERP_FPS_RANGE}: stays 1x (estimated as off)")
        mode = "off"
    t_r = restore_frame_time(profile, height, d)
    if t_r is None or not _num(src_fps):
        res["notes"].append("no usable restore measurement" if t_r is None else "bad src_fps")
        return res
    res["t_restore"] = t_r
    k = rife_k(profile)
    pair = rife_pair_ms(profile, height)
    t_rife = k * pair / 1000.0 if pair is not None else None
    res["k"] = k
    if mode == "full":
        if t_rife is None:
            res["notes"].append("no usable RIFE measurement")
            return res
        t_gpu, outputs, frac = t_r + t_rife, 2, 1.0
    else:
        t_gpu, outputs, frac = t_r, 1, 0.0
        if mode == "auto":
            if t_rife is None:
                res["notes"].append("no RIFE measurement: assuming no interpolation")
            else:
                frac = _clamp((1.0 / src_fps - t_r) / t_rife, 0.0, 1.0)
    gpu_fps = 1.0 / t_gpu
    enc_name, enc_fps = _encoder_fps(profile, height, encoder)
    if encoder not in (None, "auto", enc_name, "hevc", "h264") and enc_name:
        res["notes"].append(f"encoder {encoder} not measured, using {enc_name}")
    cap = None
    if enc_fps:
        cap = enc_fps / outputs / (2 if compare == "encode" else 1)
    fps, bottleneck = (cap, "encode") if cap is not None and cap < gpu_fps else (gpu_fps, "gpu")
    f = calibration_factor(profile, mode, height) if calibrate else None
    if f:
        fps *= f
    res.update(fps=fps, ratio=fps / src_fps, bottleneck=bottleneck, interp_fraction=frac,
               calibrated=bool(f), factor=f, t_gpu=t_gpu, encoder=enc_name, encode_fps=enc_fps)
    return res


def interp_eligible(src_fps) -> bool:
    """Whether the player / queue interpolate a source of this frame rate at all."""
    f = _num(src_fps)
    return f is not None and INTERP_FPS_RANGE[0] <= f <= INTERP_FPS_RANGE[1]


def level_for(rt, rh):
    """(level, prebuffer_s) for a typical / heavy ratio (None = unknown); the rule of verdict()."""
    if rh is not None and rh >= LIVE_RATIO:
        return "live", 0.0
    if rt is not None and rt >= 1.0:
        deficit = HEAVY_SEGMENT_S * (1 - rh) / rh if rh else MAX_PREBUFFER_S
        return "prebuffer", _clamp(deficit, MIN_PREBUFFER_S, MAX_PREBUFFER_S)
    return "queue", None


def verdict(profile, *, height, src_fps, mode="off", **kw) -> dict:
    """live (heavy ratio >= LIVE_RATIO) / prebuffer (typical ratio >= 1) / queue.

    prebuffer_s = HEAVY_SEGMENT_S*(1-r)/r with r the heavy ratio, clamped to
    [MIN_PREBUFFER_S, MAX_PREBUFFER_S]; 0 for live, None for queue.
    """
    typ = estimate(profile, height=height, src_fps=src_fps, mode=mode, density="typical", **kw)
    hvy = estimate(profile, height=height, src_fps=src_fps, mode=mode, density="heavy", **kw)
    rt, rh = typ["ratio"], hvy["ratio"]
    level, pre = level_for(rt, rh)
    return {"level": level, "ratio_typical": rt, "ratio_heavy": rh, "prebuffer_s": pre,
            "interp_fraction": typ["interp_fraction"], "bottleneck": typ["bottleneck"],
            "calibrated": typ["calibrated"], "interp_ok": typ["interp_ok"]}


def matrix(profile, src_fps=30, restoration_fps="original") -> list[dict]:
    """One row per HEIGHT x MODE: {height, mode, **verdict} (for the UI table)."""
    return [{"height": h, "mode": m, **verdict(profile, height=h, src_fps=src_fps, mode=m,
                                                 restoration_fps=restoration_fps)}
            for h in HEIGHTS for m in MODES]


def recommend(profile, src_fps=30) -> dict:
    """Suggested settings. Keys: live_height (best live with RIFE off), auto_interp (bool: auto
    interpolates >= AUTO_INTERP_MIN of frames at the best watchable height), auto_fraction, full_height
    (best live with full), full_watch_height (best live-or-prebuffer with full, i.e. a queue job can be
    watched while it is processed), prebuffer_height (best prebuffer-or-better with off), use_queue
    (nothing is live), summary (zh-TW; for 30 fps it also covers 60 fps sources).

    auto only spends spare speed (its speed is off's), so it never makes playback worse: it is advised
    wherever playback works at all, live or after a prebuffer."""
    rows = {(r["height"], r["mode"]): r for r in matrix(profile, src_fps)}
    best = lambda mode, levels: max((h for h in HEIGHTS if rows[h, mode]["level"] in levels),
                                    default=None)
    live_h, full_h = best("off", ("live",)), best("full", ("live",))
    pre_h = best("off", ("live", "prebuffer"))
    full_watch_h = best("full", ("live", "prebuffer"))
    watch_h = live_h or pre_h
    auto_frac = rows[watch_h, "auto"]["interp_fraction"] if watch_h else None
    auto = bool(watch_h and rows[watch_h, "auto"]["level"] in ("live", "prebuffer")
                and auto_frac is not None and auto_frac >= AUTO_INTERP_MIN)
    rec = {"live_height": live_h, "auto_interp": auto, "auto_fraction": auto_frac,
           "full_height": full_h, "full_watch_height": full_watch_h, "prebuffer_height": pre_h,
           "use_queue": live_h is None}
    wait = lambda h, mode: f"先緩衝約 {round(rows[h, mode]['prebuffer_s'] or 0)} 秒"
    if all(r["ratio_typical"] is None for r in rows.values()):
        s = "沒有可用的效能資料，請重新執行效能檢測。"
    elif pre_h is None:
        s = "此電腦無法即時修復，建議使用佇列在背景處理後再觀看。"
    else:
        if live_h:
            s = f"即時播放最高 {live_h}p" + (f"，{pre_h}p 要{wait(pre_h, 'off')}" if pre_h > live_h else "")
        else:
            s = f"{wait(pre_h, 'off')}後可播放，最高 {pre_h}p"
        s += f"；自動補幀可用（約補 {auto_frac:.0%} 的格數）" if auto else "；不建議自動補幀"
        if full_h:
            s += f"；佇列全補幀最高 {full_h}p 可邊處理邊看"
        elif full_watch_h:
            s += f"；佇列全補幀最高 {full_watch_h}p 可邊處理邊看（{wait(full_watch_h, 'full')}）"
        else:
            s += "；全補幀要等佇列處理完再看"
        s += "。"
    rec["restoration_fps"] = "original"
    if src_fps == 30 and pre_h is not None:
        s += _summary_60(profile, rec)
    rec["summary"] = s
    return rec


def _summary_60(profile, rec) -> str:
    """The 60 fps sentence of the 30 fps summary; sets rec["restoration_fps"] ("30" when only the player's
    "最高 30fps" restoration frame rate makes 60 fps sources watchable, at a higher resolution than original)."""
    def best(rows):
        live = max((h for h, r in rows.items() if r["level"] == "live"), default=None)
        watch = max((h for h, r in rows.items() if r["level"] in ("live", "prebuffer")), default=None)
        return live, watch
    orig = {r["height"]: r for r in matrix(profile, 60) if r["mode"] == "off"}
    half = {r["height"]: r for r in matrix(profile, 60, "30") if r["mode"] == "auto"}
    live60, watch60 = best(orig)
    live30, watch30 = best(half)
    if watch30 is not None and (watch60 is None or watch30 > watch60):
        rec["restoration_fps"] = "30"
        r = half[watch30]
        s = "60 fps 影片原始幀率跟不上" if watch60 is None else f"60 fps 影片原始幀率最高只到 {watch60}p"
        s += "；修復幀率設「最高 30fps」後，"
        s += (f"即時最高 {live30}p" if live30 == watch30
              else f"先緩衝約 {round(r['prebuffer_s'] or 0)} 秒即可播放，最高 {watch30}p")
        frac = r.get("interp_fraction")
        if r.get("interp_ok") and frac is not None and frac >= AUTO_INTERP_MIN:
            s += f"，自動補幀約補 {frac:.0%} 回到 60 fps"
        return s + "。"
    if live60:
        return f"60 fps 影片即時最高 {live60}p。"
    if watch60:
        return f"60 fps 影片先緩衝約 {round(orig[watch60]['prebuffer_s'] or 0)} 秒後可播放，最高 {watch60}p。"
    return "60 fps 影片跟不上即時，請用佇列。"


# Detection models from most to least accurate: the check recommends the most accurate one whose verdicts
# (live / prebuffer / queue in every cell of the table) are the same as those of the fastest measured model.
DETECTOR_ACCURACY = ("v4-accurate", "v4-fast")


def recommend_detector(profile) -> dict:
    """{detector, differs: [(src_fps, height, mode, {detector: level})...]}. detector is None when fewer than
    two known models were measured or when a more accurate model would change some verdict (then it is the
    user's trade-off: speed or accuracy)."""
    dets = [d for d in DETECTOR_ACCURACY if d in detectors(profile)]
    if len(dets) < 2:
        return {"detector": None, "differs": []}
    def cells(view):
        out = {(30, r["height"], r["mode"]): r["level"] for r in matrix(view, 30)}
        out.update({(60, r["height"], "off"): r["level"] for r in matrix(view, 60) if r["mode"] == "off"})
        out.update({("60r30", r["height"], "auto"): r["level"] for r in matrix(view, 60, "30") if r["mode"] == "auto"})
        return out      # 60 fps is never interpolated at its own rate: its off row only
    levels = {d: cells(for_detector(profile, d)) for d in dets}
    differs = [(f, h, m, {d: levels[d][f, h, m] for d in dets})
               for (f, h, m) in levels[dets[-1]] if len({levels[d][f, h, m] for d in dets}) > 1]
    return {"detector": dets[0] if not differs else None, "differs": differs}


def job_seconds(profile, *, duration_s, height, src_fps, mode="full", compare=None) -> float | None:
    """Queue ETA prior: wall seconds to process ``duration_s`` of video (typical density)."""
    e = estimate(profile, height=height, src_fps=src_fps, mode=mode, density="typical",
                 compare=compare)
    if not e["fps"] or not _num(duration_s):
        return None
    return duration_s * src_fps / e["fps"]


# ---- calibration & staleness -----------------------------------------------------

def record_observation(profile, *, mode, height, observed_fps, predicted_fps=None,
                       density=None, detector=None, alpha=0.2) -> dict:
    """Fold a real measurement (source frames/s) into calibration["<mode>:<height>"].

    factor = EMA of observed/predicted (first sample taken as-is). predicted_fps must be
    the *uncalibrated* prediction; when omitted it is computed from the profile at
    ``density`` (default typical) without calibration, with the detection model the run used (for_detector).
    Ratios outside [0.2, 5] are ignored.
    Returns a modified deep copy; the argument is not mutated.
    """
    out = copy.deepcopy(profile)
    obs = _num(observed_fps)
    pred = _num(predicted_fps)
    if pred is None:
        pred = estimate(for_detector(profile, detector), height=height, src_fps=30, mode=mode,
                        density="typical" if density is None else density,
                        calibrate=False)["fps"]
    if obs is None or not pred or not OBSERVATION_RANGE[0] <= obs / pred <= OBSERVATION_RANGE[1]:
        return out
    ratio = obs / pred
    cal = out.get("calibration")
    if not isinstance(cal, dict):
        cal = out["calibration"] = {}
    key = f"{mode}:{height}"
    old = cal.get(key) if isinstance(cal.get(key), dict) else None
    if old and _num(old.get("factor")):
        cal[key] = {"factor": (1 - alpha) * old["factor"] + alpha * ratio,
                    "n": int(old.get("n") or 0) + 1}
    else:
        cal[key] = {"factor": ratio, "n": 1}
    return out


def stale_reasons(profile, current_fingerprint: dict) -> list[str]:
    """Differences between the profile's fingerprint and the current machine
    (a key missing/None in ``current_fingerprint`` is not compared)."""
    saved = _section(profile, "fingerprint")
    out = []
    for key in STALE_KEYS:
        cur = (current_fingerprint or {}).get(key)
        old = saved.get(key)
        if cur is not None and str(old).casefold() != str(cur).casefold():
            out.append(f"{key}: {old} -> {cur}")
    return out


# ---- storage ---------------------------------------------------------------------

DEFAULT_ENGINE = "lada"


def profile_engine(profile) -> str:
    """The restoration engine a profile was measured with (fingerprint.engine); profiles from before the engine
    field (or with it unset) are lada's."""
    name = _section(profile, "fingerprint").get("engine")
    return name if isinstance(name, str) and name else DEFAULT_ENGINE


def current_engine() -> str:
    """The restoration engine in use by this process (never raises)."""
    try:
        from oh_streamer import engines
        return engines.default_engine_name()
    except Exception:  # noqa: BLE001
        return DEFAULT_ENGINE


def path_for_engine(base, engine=None) -> Path:
    """Where the profile of `engine` lives next to the lada one (``base``): lada keeps the original file name,
    every other engine gets ``<stem>.<engine><suffix>``, so one engine's numbers never feed another's estimates."""
    base = Path(base)
    name = re.sub(r"[^A-Za-z0-9_-]", "", engine) if isinstance(engine, str) else ""
    return base if name in ("", DEFAULT_ENGINE) else base.with_name(f"{base.stem}.{name}{base.suffix}")


def default_path(engine=None) -> Path:
    """The profile file of `engine` (default: lada's, the original perf_profile.json)."""
    from oh_streamer.platform import platform_support
    return path_for_engine(platform_support.data_dir() / "perf_profile.json", engine)


def load(path=None, engine=None) -> dict | None:
    """The saved profile, or None if missing/unreadable/not an object. engine: also None for a profile measured
    with another engine (no engine field = lada)."""
    try:
        data = json.loads(Path(path or default_path(engine)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data if engine is None or profile_engine(data) == engine else None


def save(profile, path=None) -> Path:
    """Atomically write the profile (temp file + os.replace); returns the path."""
    p = Path(path or default_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


# ---- text table ------------------------------------------------------------------

def _width(s: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, w: int) -> str:
    return s + " " * (w - _width(s))


def _cell(row: dict) -> str:
    """One table cell: speed (x real time) + level. Auto shows the share of frames it can interpolate instead: it
    only spends spare speed, so its speed is off's."""
    r = row["ratio_typical"]
    if r is None:
        return "—"
    mark = {"live": "✅", "prebuffer": "⏳", "queue": "📥"}[row["level"]]
    if row["mode"] == "auto":
        frac = row.get("interp_fraction") or 0.0
        s = (f"補{frac:.0%} " if frac > 0 else "不補 ") + mark
    else:
        s = f"{r:.1f}x " + mark
    if row["level"] == "prebuffer":
        s += f" {row['prebuffer_s']:.0f}s"
    return s


# (source fps, mode, header): 60 fps sources are never interpolated (INTERP_FPS_RANGE), so only their off column
TABLE_COLUMNS = ((30, "off", "30fps 不補幀"), (30, "auto", "自動補幀"), (30, "full", "全補幀"), (60, "off", "60fps"))


def format_table(profile) -> str:
    """Compact zh-TW table: rows heights, columns TABLE_COLUMNS, then warnings + advice."""
    rows = {(f, r["height"], r["mode"]): r for f in {c[0] for c in TABLE_COLUMNS} for r in matrix(profile, f)}
    head = ["解析度"] + [c[2] for c in TABLE_COLUMNS]
    body = [[f"{h}p"] + [_cell(rows[f, h, m]) for f, m, _ in TABLE_COLUMNS] for h in HEIGHTS]
    widths = [max(_width(r[i]) for r in [head] + body) for i in range(len(head))]
    fmt = lambda r: "  ".join(_pad(c, w) for c, w in zip(r, widths)).rstrip()
    fp = _section(profile, "fingerprint")
    lines = [f"效能檢測（{fp.get('gpu') or '未知 GPU'}，一般馬賽克密度）",
             fmt(head), *map(fmt, body),
             "✅ 可即時  ⏳ 需預緩衝（秒數）  📥 建議用佇列  數字＝幾倍即時  補N%＝自動補幀能補的格數",
             f"超過 {INTERP_FPS_RANGE[1]:g} fps 的影片不補幀"]
    for w in profile.get("warnings") or [] if isinstance(profile, dict) else []:
        lines.append(f"⚠ {w}")
    lines.append(recommend(profile, 30)["summary"])
    return "\n".join(lines)
