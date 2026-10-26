"""Choose lightweight cast preview inputs without starting restoration or queue work."""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import json
import math
import os
import subprocess
import threading

from oh_streamer.platform.cast_thumbnails import HlsThumbnails


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PROBE_CACHE_SIZE = 64


def _processed_track(path):
    """Return (absolute stream index, relative video start), accepting only safe track choices."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3,
        creationflags=_NO_WINDOW, check=True,
    )
    probe = json.loads(result.stdout)
    videos = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    stream = next((s for s in videos if s.get("tags", {}).get("title") == "oh:processed"), None)
    if stream is None and len(videos) == 1 and not videos[0].get("tags", {}).get("title"):
        stream = videos[0]
    if stream is None:
        return None
    fmt_start = float((probe.get("format") or {}).get("start_time") or 0)
    video_start = float(stream.get("start_time") or fmt_start)
    offset = video_start - fmt_start
    if not math.isfinite(offset):
        return None
    return int(stream["index"]), offset


def _progressive_ready_source(vod, session):
    """Snapshot-only lookup; failed probes are cached until the file changes."""
    cache = OrderedDict()
    lock = threading.Lock()

    def ready_source(k):
        try:
            if not 0 <= k < vod.plan.count:
                return None
            state = session.watcher.state
            play = state.play
            # Do the subtraction before converting to float so fractional-rate segment
            # boundaries do not drift into the preceding queue segment.
            timestamp = Fraction(str(vod.plan.video_start)) + vod.plan.start(k)
            elapsed = timestamp - Fraction(str(play.video_start))
            if elapsed < 0:
                return None
            position = elapsed * play.fps / play.n
            queue_k = math.floor(position + Fraction(1, 10**9))
            segment = state.segments.get(queue_k)
            if segment is None or segment.frames <= 0:
                return None
            local = elapsed - Fraction(queue_k * play.n, 1) / play.fps
            if local < -Fraction(1, 10**9) or local >= Fraction(segment.frames, 1) / play.fps:
                return None
            name = segment.file
            if not isinstance(name, str) or not name or os.path.basename(name) != name or name in (".", ".."):
                return None
            path = session.watcher.segment_path(segment)
            stat = os.stat(path)
            if not os.path.isfile(path) or stat.st_size <= 0:
                return None
            signature = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            with lock:
                if signature in cache:
                    track = cache.pop(signature)
                    cache[signature] = track
                else:
                    try:
                        track = _processed_track(path)
                    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                        track = None
                    cache[signature] = track
                    while len(cache) > _PROBE_CACHE_SIZE:
                        cache.popitem(last=False)
            if track is None:
                return None
            index, offset = track
            return path, (), max(0.0, float(local) + offset), index
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            return None

    return ready_source


def create_thumbnails(vod, session) -> HlsThumbnails | None:
    """Build cast previews from the current source plus already available restored content."""
    if getattr(session, "is_progressive_cast", False):
        src = session.po.source_url or session.po.path
        if not src:
            return None
        return HlsThumbnails(vod, src, fmt_start=session.fmt_start,
                             ready_source=_progressive_ready_source(vod, session))
    if hasattr(session, "stream_index") and getattr(session, "path", None):
        return HlsThumbnails(vod, session.path, fmt_start=session.fmt_start,
                             stream_index=session.stream_index)
    source = getattr(session, "source", None)
    if source is None or not getattr(source, "src", None):
        return None
    return HlsThumbnails(vod, source.src, source.http.cli_args(), fmt_start=source.fmt_start)
