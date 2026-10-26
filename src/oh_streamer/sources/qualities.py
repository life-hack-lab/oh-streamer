"""
Video quality (resolution) handling for oh_player: building the list of selectable renditions from a yt-dlp
info dict, picking the default / a requested one, and the offline --fake-formats source used for testing.

Pure Python (no torch / lada imports) so it can be unit-tested quickly: see test_qualities.py.
"""

import json
import math
import re
from pathlib import Path

DEFAULT_MAX_HEIGHT = 1080


def quality_label(height: int, fps: float | None) -> str:
    """'1080p', or '720p60' for high-frame-rate renditions (29.97 / 30 / 25 fps get no suffix)."""
    label = f"{int(height)}p"
    if fps and round(fps) > 30:
        label += str(int(round(fps)))
    return label


def _has_video_and_audio(f: dict) -> bool:
    """Combined (muxed) formats only: lada reads video and ffmpeg takes audio from the same URL.

    A codec of "none" means the stream is absent. None / missing means "unknown" (some extractors and HLS variants
    don't say), which is accepted: the height requirement already filters audio-only formats out.
    """
    return f.get("vcodec") != "none" and f.get("acodec") != "none"


def _is_video_only(f: dict) -> bool:
    return f.get("acodec") == "none" and f.get("vcodec") not in (None, "none")


def _is_audio_only(f: dict) -> bool:
    return f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")


def _is_hls(f: dict) -> bool:
    return str(f.get("protocol") or "").startswith("m3u8")


def pick_audio_format(info: dict) -> dict | None:
    """Best audio-only format to pair with video-only renditions (YouTube and others serve no muxed formats).

    Plain HTTPS beats HLS (seekable, no 'untested' manifests), DRC-boosted variants are skipped, AAC (m4a) wins over
    Opus at equal quality because every ffmpeg build decodes it, then the highest bitrate."""
    best = None
    for index, f in enumerate(info.get("formats") or []):
        if not f.get("url") or not _is_audio_only(f) or f.get("has_drm"):
            continue
        if "drc" in str(f.get("format_id") or "").lower() or "drc" in str(f.get("format_note") or "").lower():
            continue
        key = (not _is_hls(f), f.get("ext") == "m4a", f.get("abr") or f.get("tbr") or 0, index)
        if best is None or key > best[0]:
            best = (key, f)
    return best[1] if best else None


def build_quality_list(info: dict) -> list[dict]:
    """Playable renditions from a yt-dlp info dict, one per distinct height, best first (high -> low).

    Returns [{id, height, width, fps, tbr, url, audio_url, http_headers, label}]. Muxed video+audio formats are used
    as they are (audio_url None: audio comes from `url`); when a site offers none (YouTube), video-only formats are
    paired with the best audio-only one (audio_url). Per height the entry with the highest bitrate wins (then fps);
    among video-only formats plain HTTPS beats HLS and H.264 beats VP9 / AV1 (Lada decodes in software). Audio-only,
    storyboard and DRM-protected formats are excluded.
    """
    formats = info.get("formats") or []
    # separate streams are only paired up when the site offers no muxed format at all (YouTube)
    pair = not any(f.get("url") and f.get("height") and _has_video_and_audio(f) for f in formats)
    audio = pick_audio_format(info) if pair else None
    best: dict[int, tuple[tuple, dict]] = {}
    for index, f in enumerate(formats):
        url, height = f.get("url"), f.get("height")
        if not url or not height:
            continue
        if pair:
            if audio is None or not _is_video_only(f):
                continue
        elif not _has_video_and_audio(f):
            continue
        if (f.get("ext") or "") == "mhtml" or str(f.get("format_note") or "").lower() == "storyboard":
            continue
        if f.get("has_drm"):
            continue
        if pair:
            h264 = str(f.get("vcodec") or "").startswith(("avc1", "h264"))
            key = (not _is_hls(f), h264, f.get("tbr") or 0, f.get("fps") or 0, f.get("quality") or 0, index)
        else:
            key = (f.get("tbr") or 0, f.get("fps") or 0, f.get("quality") or 0, index)
        h = int(height)
        if h not in best or key > best[h][0]:
            best[h] = (key, f)
    out = []
    for h in sorted(best, reverse=True):
        f = best[h][1]
        fps = f.get("fps")
        out.append({
            "id": str(f.get("format_id") if f.get("format_id") is not None else f"h{h}"),
            "height": h,
            "width": int(f.get("width") or 0),
            "fps": float(fps) if fps else None,
            "tbr": f.get("tbr"),
            "url": f["url"],
            "audio_url": audio["url"] if pair else None,
            "http_headers": dict(f.get("http_headers") or info.get("http_headers") or {}),
            "label": quality_label(h, fps),
        })
    return out


def single_quality_from_info(info: dict) -> list[dict]:
    """Fallback when the info dict has no usable 'formats': the one URL yt-dlp resolved (no selectable qualities)."""
    if not info.get("url"):
        return []
    h = int(info.get("height") or 0)
    return [{"id": str(info.get("format_id") or "0"), "height": h, "width": int(info.get("width") or 0),
             "fps": info.get("fps"), "tbr": info.get("tbr"), "url": info["url"],
             "http_headers": dict(info.get("http_headers") or {}),
             "label": quality_label(h, info.get("fps")) if h else "source"}]


def parse_quality_spec(spec: str | None) -> str | int | None:
    """'best' / 'worst' / a height ('720' or '720p') -> 'best' | 'worst' | int; None stays None."""
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("best", "worst"):
        return s
    m = re.fullmatch(r"(\d{2,5})p?", s)
    if not m:
        raise ValueError(f"invalid quality '{spec}': expected a height like 720, or 'best' / 'worst'")
    return int(m.group(1))


def pick_quality(qualities: list[dict], spec: str | int | None = None, max_height: int | None = None) -> dict:
    """Choose the starting rendition.

    spec None   -> best with height <= max_height (default 1080); if everything is taller, the smallest one
    spec <N>    -> best with height <= N (and <= max_height when given); if none is that small, the smallest one
    spec 'best' -> highest (still capped by an explicit max_height)
    spec 'worst'-> lowest
    qualities is sorted high -> low.
    """
    if not qualities:
        raise ValueError("no playable qualities")
    if spec == "worst":
        return qualities[-1]
    limit = math.inf
    if isinstance(spec, int):
        limit = spec
    elif spec is None:
        limit = DEFAULT_MAX_HEIGHT
    if max_height:
        limit = min(limit, max_height)
    for q in qualities:
        if q["height"] <= limit:
            return q
    return qualities[-1]


def find_quality(qualities: list[dict], fid: str | None = None, height: int | None = None) -> dict | None:
    for q in qualities:
        if fid is not None and q["id"] == fid:
            return q
    if height is not None:
        for q in qualities:
            if q["height"] == height:
                return q
    return None


def public_list(qualities: list[dict]) -> str:
    """JSON published to the player UI as user-data/oh/qualities (no URLs / headers: they may be secrets)."""
    return json.dumps([{"id": q["id"], "label": q["label"], "height": q["height"]} for q in qualities],
                      ensure_ascii=False)


def format_table(qualities: list[dict], chosen: dict | None = None) -> str:
    lines = [f"{'':1} {'id':<18} {'label':<8} {'size':<10} {'fps':>5} {'tbr':>8}"]
    for q in qualities:
        size = f"{q['width']}x{q['height']}" if q["width"] else f"?x{q['height']}"
        fps = f"{q['fps']:g}" if q["fps"] else "-"
        tbr = f"{q['tbr']:.0f}k" if q.get("tbr") else "-"
        mark = "*" if chosen is not None and q["id"] == chosen["id"] else " "
        lines.append(f"{mark:1} {q['id']:<18.18} {q['label']:<8} {size:<10} {fps:>5} {tbr:>8}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --fake-formats: an info-dict-like JSON pointing at local files (offline testing)
# ---------------------------------------------------------------------------

def load_fake_info(path: str, key: str = "formats") -> dict:
    """Read the fake info JSON; relative 'url's resolve against the JSON's directory.

    {"title": "...", "formats": [{"format_id": "1080", "url": "../a.mp4", "height": 1080, ...}, ...],
     "refresh_formats": [...],       # optional: what a yt-dlp re-run returns (simulates fresh signed URLs)
     "duration": 123.4}              # optional: seconds, passed through for the intake panel (source_info)
    """
    p = Path(path).resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    formats = data.get(key)
    if formats is None:
        formats = data.get("formats") or []
    resolved = []
    for f in formats:
        f = dict(f)
        u = f.get("url")
        if u and not re.match(r"^[a-z][a-z0-9+.-]*://", u):
            f["url"] = str((p.parent / u).resolve())
        resolved.append(f)
    return {"title": data.get("title") or p.stem, "formats": resolved, "http_headers": data.get("http_headers") or {},
            "duration": data.get("duration")}
