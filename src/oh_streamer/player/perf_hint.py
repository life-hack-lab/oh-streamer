"""Performance hints for the live player from the saved performance profile (perf/profile.py).

Pure logic, no mpv / torch: the controller asks for a one-off OSD hint when a source starts (hint_for), marks in the
quality menu (mark_for / annotate_public_list) and a line in the info panel (info_for). Everything returns None
(or the input unchanged) when there is no profile or anything goes wrong: a hint must never disturb playback.
"""
from __future__ import annotations

import json
import os

from oh_streamer.perf import profile as _profile
from oh_streamer.processing.frame_rate import planned_fps

DEFAULT_FPS = 30.0
# quality-menu mark = the verdict level; oh_seek.lua draws it as a coloured dot (emoji may be tofu in the OSD font)
MARKS = {"live": "live", "prebuffer": "prebuffer", "queue": "queue"}
LABELS = {"live": "可即時", "prebuffer": "需預緩衝", "queue": "建議佇列"}
OSD_MS = {"queue": 6000, "prebuffer": 5000}

_cache: dict = {}   # path -> (mtime_ns, profile)


def get_profile(path=None):
    """The saved profile, loaded once per process and again when the file's mtime changes; None if absent/unreadable."""
    try:
        engine = None if path else _profile.current_engine()
        p = str(path or _profile.default_path(engine))
        try:
            mtime = os.stat(p).st_mtime_ns
        except OSError:
            _cache.pop(p, None)
            return None
        hit = _cache.get(p)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        prof = _profile.load(p, engine)
        _cache[p] = (mtime, prof)
        return prof
    except Exception:  # noqa: BLE001
        return None


def for_detector(profile, detector):
    """The profile as estimated with the player's detection model (profile.for_detector); unchanged on any error."""
    if not profile:
        return profile
    try:
        return _profile.for_detector(profile, detector)
    except Exception:  # noqa: BLE001
        return profile


def mode_for(interpolation) -> str:
    return "auto" if interpolation == "auto" else "off"


def verdict_for(profile, height, fps, interpolation) -> dict | None:
    """profile.verdict for a source, or None (no profile, unknown height, any error, no usable ratio)."""
    if not profile or not height:
        return None
    try:
        v = _profile.verdict(profile, height=int(height), src_fps=float(fps or DEFAULT_FPS), mode=mode_for(interpolation))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(v, dict) or v.get("level") not in MARKS or v.get("ratio_typical") is None:
        return None
    return v


def hint_for(profile, height, fps, interpolation) -> tuple[str, str] | None:
    """(level, OSD text) for a "queue" / "prebuffer" source; None for "live" or when nothing is known."""
    v = verdict_for(profile, height, fps, interpolation)
    if v is None:
        return None
    if v["level"] == "queue":
        return "queue", (f"這台電腦處理此畫質約 {v['ratio_typical']:.1f}x 即時，可能會卡頓："
                         "按 r 降畫質，或按 a 交給背景佇列")
    if v["level"] == "prebuffer":
        pre = v.get("prebuffer_s") or 0.0
        return "prebuffer", (f"預估約 {v['ratio_typical']:.1f}x 即時，馬賽克多的段落可能追不上；"
                             f"建議先暫停緩衝約 {pre:.0f} 秒")
    return None


def mark_for(profile, height, fps, interpolation) -> str:
    """Quality-menu mark for a rendition ("" without a profile); unknown fps -> 30."""
    v = verdict_for(profile, height, fps, interpolation)
    return MARKS[v["level"]] if v else ""


def info_for(profile, height, fps, interpolation) -> dict | None:
    """{"ratio": x, "level": ..., "label": zh} for the info panel, or None."""
    v = verdict_for(profile, height, fps, interpolation)
    if v is None:
        return None
    return {"ratio": v["ratio_typical"], "level": v["level"], "label": LABELS[v["level"]]}


def restored_fps(fps, restoration_fps="original"):
    """Frame rate the pipeline restores a source of ``fps`` at (the restoration_fps setting); unknown fps -> None."""
    try:
        return float(planned_fps(fps, restoration_fps or "original")) if fps else None
    except Exception:  # noqa: BLE001
        return fps


def annotate_public_list(public_json: str, qualities, profile, interpolation, restoration_fps="original") -> str:
    """Add a "mark" to each entry of qualities.public_list(); unchanged without a profile. A rendition is judged at the
    frame rate it is restored at (60 fps with restoration_fps "30" -> 30)."""
    if not profile:
        return public_json
    try:
        items = json.loads(public_json)
        by_id = {q.get("id"): q for q in qualities}
        for it in items:
            q = by_id.get(it.get("id")) or {}
            m = mark_for(profile, it.get("height") or q.get("height"), restored_fps(q.get("fps"), restoration_fps),
                         interpolation)
            if m:
                it["mark"] = m
        return json.dumps(items, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return public_json


class HintTracker:
    """Decides when the OSD hint is shown: once per opened source (epoch), and again only if a quality
    switch changed the level. "live" shows nothing but is remembered as the level."""

    def __init__(self):
        self.state: tuple | None = None   # (epoch, level)

    def next(self, epoch, profile, height, fps, interpolation) -> tuple[str, str, int] | None:
        v = verdict_for(profile, height, fps, interpolation)
        if v is None:
            return None
        level = v["level"]
        prev, self.state = self.state, (epoch, level)
        if prev is not None and prev[0] == epoch and prev[1] == level:
            return None
        h = hint_for(profile, height, fps, interpolation)
        return (h[0], h[1], OSD_MS[h[0]]) if h else None
