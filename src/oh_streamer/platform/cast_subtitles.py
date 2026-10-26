"""Snapshot mpv's text subtitles as segmented HLS WebVTT."""

import math
import os
import re
import subprocess

from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE


def _seconds(value):
    parts = value.replace(",", ".").split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def _timestamp(seconds):
    ms = max(0, round(seconds * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"


_ISO639_TO_BCP47 = {
    "chi": "zh", "zho": "zh", "cmn": "zh", "yue": "yue", "jpn": "ja", "eng": "en", "kor": "ko",
    "fra": "fr", "fre": "fr", "deu": "de", "ger": "de", "spa": "es", "rus": "ru", "tha": "th",
    "vie": "vi", "ita": "it", "por": "pt", "ind": "id", "msa": "ms", "may": "ms",
}


def hls_language(lang=None, title=None):
    """BCP-47 tag for the HLS LANGUAGE attribute (Apple TV shows 'Unknown' without it).

    Prefers the track's own language tag, then falls back to hints in the title; None if undetermined."""
    code = str(lang or "").strip().lower().replace("_", "-")
    if code and code not in {"und", "unk", "mis", "mul", "zxx"}:
        head, _, rest = code.partition("-")
        head = _ISO639_TO_BCP47.get(head, head)
        if re.fullmatch(r"[a-z]{2,3}", head):
            if head == "zh" and rest in {"tw", "hk", "mo", "hant"}:
                return "zh-Hant"
            if head == "zh" and rest in {"cn", "sg", "hans"}:
                return "zh-Hans"
            return head
    text = str(title or "")
    for hints, tag in ((("\u4e2d\u6587", "\u7e41", "\u7c21", "\u7b80", "\u570b\u8a9e", "\u7cb5"), "zh"),
                       (("\u65e5\u6587", "\u65e5\u8a9e", "\u65e5\u672c\u8a9e"), "ja"),
                       (("\u82f1\u6587", "\u82f1\u8a9e", "English"), "en"),
                       (("\u97d3\u6587", "\u97d3\u8a9e"), "ko")):
        if any(h in text for h in hints):
            return tag
    return None


class HlsSubtitles:
    def __init__(self, plan, webvtt, offset=0.0, *, title="字幕", default=True, track_index=None, timing=None,
                 language=None):
        self.plan = plan
        self.title, self.default = title, default
        self.language = language or hls_language(None, title)
        self.offset = offset
        self.timing = dict(timing or {})     # the terms of `offset`, for the timing log only
        self.cue_count = 0
        self.first_cue = None                # (start, end) in playlist time of the earliest written cue
        self.last_end = None
        self.playlist_uri = "subtitles.m3u8" if track_index is None else f"subtitles_{track_index}.m3u8"
        self.segment_prefix = "sub_" if track_index is None else f"sub_{track_index}_"
        self.segments = {}
        for block in re.split(r"\n\s*\n", webvtt.replace("\r\n", "\n")):
            lines = block.strip().splitlines()
            for i, line in enumerate(lines):
                match = re.fullmatch(r"([\d:.]+) --> ([\d:.]+)(.*)", line)
                if not match:
                    continue
                start, end = _seconds(match[1]) + offset, _seconds(match[2]) + offset
                if end <= max(0, start) or start >= float(plan.total_duration):
                    break
                cue = f"{_timestamp(start)} --> {_timestamp(end)}{match[3]}\n" + "\n".join(lines[i + 1:])
                self.cue_count += 1
                if self.first_cue is None or start < self.first_cue[0]:
                    self.first_cue = (start, end)
                self.last_end = end if self.last_end is None else max(self.last_end, end)
                first = plan.index_at(max(0, start))
                last = plan.index_at(max(0, end))
                for k in range(first, last + 1):
                    if float(plan.start(k)) < end:
                        self.segments.setdefault(k, []).append(cue)
                break

    def playlist(self):
        return self.plan.media_playlist().replace("seg_", self.segment_prefix).replace(".ts\n", ".vtt\n")

    @staticmethod
    def timestamp_clock():
        """MPEGTS ticks (90 kHz) of WebVTT local time 0: the video's first PTS (every run adds the same offset)."""
        from oh_streamer.platform.hls_cast import TS_BASE, MPEGTS_DELAY
        return round((TS_BASE + MPEGTS_DELAY) * 90000)

    def describe(self, start_k=0):
        """One log line with every term of the cue-time map (see sample video A)."""
        from oh_streamer.platform.hls_cast import TS_BASE, MPEGTS_DELAY
        plan, t = self.plan, self.timing
        clock = self.timestamp_clock()
        first = "none"
        if self.first_cue is not None:
            s, e = self.first_cue
            first = f"{_timestamp(s)} --> {_timestamp(e)} (source {s + plan.video_start:.3f}s)"
        last = "-" if self.last_end is None else f"{self.last_end:.3f}s"
        terms = ""
        if t:
            terms = (f" = sub-delay {t.get('delay', 0.0):+.3f} - video_start {plan.video_start:.3f}"
                     f" + demuxer-start {t.get('origin', 0.0):.3f}"
                     f" ({'external' if t.get('external') else 'embedded'} ff-index {t.get('ff_index')}"
                     f" {os.path.basename(str(t.get('path') or ''))})")
        return (f"cast subtitles '{self.title}': cue offset {self.offset:+.3f}s{terms} | {self.cue_count} cues, "
                f"first {first}, last end {last} playlist | X-TIMESTAMP-MAP LOCAL=00:00:00.000 MPEGTS={clock} "
                f"({clock / 90000:.3f}s = TS_BASE {TS_BASE:.1f} + mux delay {MPEGTS_DELAY:.1f}) | first video segment "
                f"pts {TS_BASE + MPEGTS_DELAY + float(plan.start(0)):.3f}s = source {plan.source_time(0):.3f}s; "
                f"start segment {start_k} pts {TS_BASE + MPEGTS_DELAY + float(plan.start(start_k)):.3f}s = source "
                f"{plan.source_time(start_k):.3f}s")

    def segment(self, k):
        # Video producer adds this same offset to every run, including seeks.
        clock = self.timestamp_clock()
        header = f"WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:{clock}\n\n"
        return (header + "\n\n".join(self.segments.get(k, [])) + "\n\n").encode("utf-8")


def seek_line(plan, v0, pos, relative, subtitles=None):
    """One log line for a cast seek: source time -> playlist time -> segment / MPEGTS, and the cue offset in force."""
    from oh_streamer.platform.hls_cast import TS_BASE, MPEGTS_DELAY
    k = plan.index_at(relative)
    offsets = [f"{sub.offset:+.3f}s" for sub in (getattr(subtitles, "renditions", (subtitles,))
                                                  if subtitles is not None else ())]
    cue = f" | subtitle cue offset {'/'.join(offsets)} (fixed at cast start, same for every run)" if offsets else ""
    return (f"cast seek: source {pos:.3f}s - video_start {v0:.3f}s = playlist {relative:.3f}s -> segment {k} "
            f"(starts {float(plan.start(k)):.3f}s, source {plan.source_time(k):.3f}s, video pts "
            f"{TS_BASE + MPEGTS_DELAY + float(plan.start(k)):.3f}s){cue}")


class HlsSubtitleTracks:
    """Independent renditions sharing the same video plan and timestamp map."""

    def __init__(self, renditions):
        self.renditions = tuple(renditions)


_IMAGE_CODECS = {"dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle", "pgssub", "dvdsub", "dvbsub", "xsub"}


def _ipc_get(ipc, key):
    if ipc is None:
        return None
    if callable(ipc):
        return ipc("get", key)
    if hasattr(ipc, "get"):
        return ipc.get(key)
    return None


def _is_chinese(track):
    lang = str(track.get("lang") or "").lower().replace("_", "-")
    return lang in {"chi", "zho", "zh"} or lang.startswith("zh-")


def _is_switchable_text_track(track):
    if track.get("type") != "sub":
        return False
    if str(track.get("codec") or "").lower() in _IMAGE_CODECS or track.get("image"):
        return False
    return _is_chinese(track) or track.get("title") == "日文（AI）"


def selected_subtitles(ipc, plan):
    """Convert external SRT/ASS/VTT or a selected embedded text track.

    Selection is captured before the local producer stops. With multiple Chinese
    or switchable text tracks, expose them all and retain the primary selection as default.
    Other sources retain the legacy single-selected-track behavior.
    """
    if _ipc_get(ipc, "sub-visibility") is False:
        return None
    tracks = _ipc_get(ipc, "track-list")
    if not isinstance(tracks, list):
        return None
    selected = [t for t in tracks if t.get("type") == "sub" and t.get("selected")]
    track = next((t for t in selected if t.get("main-selection") == 0),
                 next((t for t in selected if t.get("main-selection") != 1), None))
    if track is None:
        return None
    delay = _ipc_get(ipc, "sub-delay")
    delay = float(delay) if isinstance(delay, (int, float)) and math.isfinite(delay) else 0.0
    origin = _ipc_get(ipc, "demuxer-start-time")
    origin = float(origin) if isinstance(origin, (int, float)) and math.isfinite(origin) else 0.0
    switchable = [t for t in tracks if _is_switchable_text_track(t)]
    if any(t is track for t in switchable) and len(switchable) > 1:
        # mpv's subtitle delay is global, so use an identical timeline for each
        # rendition (including external progressive subtitles).
        return HlsSubtitleTracks([
            _convert_track(ipc, plan, t, delay, origin, title=t.get("title") or f"中文字幕 {i + 1}",
                           default=t is track, track_index=i,
                           language=hls_language(t.get("lang"), t.get("title")))
            for i, t in enumerate(switchable)
        ])
    return _convert_track(ipc, plan, track, delay, origin,
                          language=hls_language(track.get("lang"), track.get("title")))


def _convert_file(path, plan, offset, *, delay=0.0, title="字幕", default=True, track_index=None, language=None):
    if not isinstance(path, str) or not path:
        raise ValueError("無法取得字幕來源")
    timing = {"delay": delay, "origin": 0.0, "external": True, "path": path}
    if os.path.isfile(path):
        try:
            if os.path.getsize(path) == 0:
                return HlsSubtitles(plan, "WEBVTT\n\n", offset, title=title, default=default,
                                    track_index=track_index, timing=timing, language=language)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if not f.read().strip():
                    return HlsSubtitles(plan, "WEBVTT\n\n", offset, title=title, default=default,
                                        track_index=track_index, timing=timing, language=language)
        except OSError:
            pass
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:s:0", "-f", "webvtt", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("字幕無法轉為投放格式，請使用 SRT、ASS 或 WebVTT 文字字幕")
    return HlsSubtitles(plan, result.stdout.decode("utf-8-sig"), offset, title=title,
                        default=default, track_index=track_index, timing=timing, language=language)


def _convert_track(ipc, plan, track, delay, origin, **metadata):
    path = track.get("external-filename") or _ipc_get(ipc, "path")
    if not isinstance(path, str) or not path:
        raise ValueError("無法取得字幕來源")
    external = bool(track.get("external-filename"))
    index = track.get("ff-index")
    if not external and (not isinstance(index, int) or index < 0):
        raise ValueError("無法取得內嵌字幕軌道")
    if external and os.path.isfile(path):
        try:
            empty = False
            if os.path.getsize(path) == 0:
                empty = True
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    if not f.read().strip():
                        empty = True
            if empty:
                offset = delay - plan.video_start
                timing = {"delay": delay, "origin": 0.0, "external": True,
                          "ff_index": index, "path": path}
                return HlsSubtitles(plan, "WEBVTT\n\n", offset, timing=timing, **metadata)
        except OSError:
            pass
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path,
           "-map", "0:s:0" if external else f"0:{index}", "-f", "webvtt", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("字幕無法轉為投放格式，請使用 SRT、ASS 或 WebVTT 文字字幕")
    # ffmpeg normalizes embedded timestamps to the container's start time.
    offset = delay - plan.video_start
    if not external:
        offset += origin
    timing = {"delay": delay, "origin": origin if not external else 0.0, "external": external,
              "ff_index": index, "path": path}
    return HlsSubtitles(plan, result.stdout.decode("utf-8-sig"), offset, timing=timing, **metadata)


def owner_subtitles(po, ipc, plan, *, selection=None, visibility=None):
    """Build or update casting subtitles from a subtitle owner (StreamSubtitles / ProcessedOpen / ProgressiveOpen)."""
    if po is None:
        return None
    if visibility is False or selection is False:
        return None
    if _ipc_get(ipc, "sub-visibility") is False:
        return None
    delay = _ipc_get(ipc, "sub-delay")
    delay = float(delay) if isinstance(delay, (int, float)) and math.isfinite(delay) else 0.0
    tracks = list(getattr(po, "subtitles", None) or ())
    if not tracks:
        if getattr(po, "is_stream", False):
            offset = delay - plan.video_start
            timing = {"delay": delay, "origin": 0.0, "external": True}
            return HlsSubtitleTracks([
                HlsSubtitles(plan, "WEBVTT\n\n", offset, title=CONCISE_TITLE, default=True, track_index=0, timing=timing),
                HlsSubtitles(plan, "WEBVTT\n\n", offset, title=FULL_TITLE, default=False, track_index=1, timing=timing),
            ])
        return None

    try:
        offset_val = float(getattr(po, "subtitle_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset_val = 0.0
    offset = delay + offset_val - plan.video_start

    default_idx = None
    if isinstance(selection, tuple) and len(selection) == 2:
        for i, t in enumerate(tracks):
            if (t.get("title"), t.get("lang")) == selection:
                default_idx = i
                break
    if default_idx is None:
        for i, t in enumerate(tracks):
            title = str(t.get("title") or "")
            if "精簡" in title and _is_chinese(t):
                default_idx = i
                break
    if default_idx is None:
        for i, t in enumerate(tracks):
            if t.get("default"):
                default_idx = i
                break
    if default_idx is None:
        default_idx = 0

    if len(tracks) == 1:
        t = tracks[0]
        return _convert_file(t["path"], plan, offset, delay=delay, title=t.get("title") or "字幕",
                             default=True, track_index=None,
                             language=hls_language(t.get("lang"), t.get("title")))

    return HlsSubtitleTracks([
        _convert_file(t["path"], plan, offset, delay=delay, title=t.get("title") or f"字幕 {i + 1}",
                      default=(i == default_idx), track_index=i,
                      language=hls_language(t.get("lang"), t.get("title")))
        for i, t in enumerate(tracks)
    ])
