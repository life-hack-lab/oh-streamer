"""One-line diagnostics for subtitle timing in local mpv playback (sample video A).

A report like "subtitles appeared early at 01:05:23" has to be classified as a global timing-map problem
(``subtitle_offset`` / ``sub-delay`` / stream start vs source clock) or a per-cue ASR problem. These helpers build the
log lines that carry every term of the map: they are emitted when a subtitle track set is (re)loaded and when a
progressive seek restarts the stream, never per second. Everything here is best effort and must not raise.
"""

import math
import os


def _num(value):
    """A finite float, or None (bools and non-numbers are not times)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _fmt(value, spec="+.3f"):
    value = _num(value)
    return "?" if value is None else format(value, spec)


def mapping_terms(po, play=None):
    """The numbers every subtitle time depends on, as a dict (None for unknown).

    srt zero = the extracted audio's first sample; ``subtitle_offset`` (the queue's audio_offset) maps it onto the
    absolute source clock. mpv runs with rebase-start-time=no, so ``time-pos`` is that same absolute clock."""
    info = getattr(po, "info", None)
    info = info if isinstance(info, dict) else {}
    return {
        "subtitle_offset": _num(getattr(po, "subtitle_offset", None)),
        "covered_s": _num(getattr(po, "subtitle_covered_s", None)),
        "stream_start": _num(getattr(play, "video_start", None)),
        "source_video_start": _num(info.get("video_start")),
    }


def load_line(po, *, kind, sub_delay=None, demuxer_start=None, time_pos=None, files=(), selected=None, play=None,
              force_reload=False):
    """The single "subtitle timing: load ..." line. files: [(path_in_mpv, original_srt_or_None), ...]."""
    t = mapping_terms(po, play)
    names = []
    for shown, original in files:
        name = os.path.basename(str(shown))
        if original and os.path.normcase(str(original)) != os.path.normcase(str(shown)):
            name += f"<-{os.path.basename(str(original))}"
        names.append(name)
    parts = [
        f"subtitle timing: {'reload' if force_reload else 'load'} ({kind}) item={getattr(po, 'queue_id', None)}",
        f"files=[{', '.join(names) or '-'}]" + (f" selected={selected}" if selected else ""),
        f"subtitle_offset(audio_offset)={_fmt(t['subtitle_offset'])}s",
        f"sub-delay={_fmt(sub_delay)}s",
        f"stream_start={_fmt(t['stream_start'], '.3f')}s",
        f"source_video_start={_fmt(t['source_video_start'], '.3f')}s",
        f"demuxer-start-time={_fmt(demuxer_start, '.3f')}s",
        f"time-pos={_fmt(time_pos, '.3f')}s",
        f"covered_s={_fmt(t['covered_s'], '.3f')}",
        "(cue shown at mpv time = srt_time + subtitle_offset + sub-delay)",
    ]
    return " | ".join(parts)


def seek_line(po, pos, *, sub_delay=None, play=None, segment=None, segment_start=None):
    """The "subtitle timing: seek ..." line for a progressive seek request to mpv/source time ``pos``."""
    t = mapping_terms(po, play)
    offset = t["subtitle_offset"] or 0.0
    delay = _num(sub_delay) or 0.0
    pos = _num(pos)
    srt = "?" if pos is None else f"{pos - offset - delay:.3f}"
    seg = "" if segment is None else f" -> segment {segment} starting {_fmt(segment_start, '.3f')}s"
    return (f"subtitle timing: seek time-pos {_fmt(pos, '.3f')}s = source time (rebase-start-time=no){seg}"
            f" | srt_time={srt}s (= time-pos - subtitle_offset {_fmt(offset)}s - sub-delay {_fmt(delay)}s)")


def log_load(controller, url, po, force_reload):
    """Called after a subtitle set was added to mpv. ``controller`` supplies ipc, _deps.log and progressive_sub_dir."""
    try:
        ipc = controller.ipc
        tracks = list(getattr(po, "subtitles", ()) or ())
        offset = _num(getattr(po, "subtitle_offset", None)) or 0.0
        sub_dir = getattr(controller, "progressive_sub_dir", None)
        files = []
        for index, track in enumerate(tracks):
            original = track["path"]
            shown = os.path.join(sub_dir, f"track-{index}.srt") if offset and sub_dir else original
            files.append((shown, original))
        sid = ipc.get("sid")
        selected = None
        for track in ipc.get("track-list", []) or []:
            if track.get("type") == "sub" and track.get("id") == sid and sid not in (None, False, "no"):
                selected = f"#{sid}:{track.get('title') or track.get('external-filename') or ''}".rstrip(":")
        progressive = getattr(controller, "progressive", None)
        is_progressive = progressive is not None and getattr(progressive, "po", None) is po
        play = None
        if is_progressive:
            play = getattr(getattr(getattr(progressive, "watcher", None), "state", None), "play", None)
        controller._deps.log(load_line(
            po, kind="progressive" if is_progressive else "stream" if getattr(po, "is_stream", False) else "processed",
            sub_delay=ipc.get("sub-delay"), demuxer_start=ipc.get("demuxer-start-time"),
            time_pos=ipc.get("time-pos"), files=files, selected=selected, play=play, force_reload=force_reload))
    except Exception as e:  # noqa: BLE001 - diagnostics never affect playback
        try:
            controller._deps.log(f"subtitle timing: load line skipped: {e!r}")
        except Exception:  # noqa: BLE001
            pass


def log_seek(controller, pos):
    """Called when a progressive seek request restarts the feeder at ``pos``."""
    try:
        po = controller.progressive.po
        play = getattr(getattr(controller.progressive.watcher, "state", None), "play", None)
        segment = segment_start = None
        if play is not None and _num(pos) is not None:
            segment = play.segment_at(pos)
            segment_start = play.seg_start(segment)
        controller._deps.log(seek_line(po, pos, sub_delay=controller.ipc.get("sub-delay"), play=play,
                                       segment=segment, segment_start=segment_start))
    except Exception as e:  # noqa: BLE001
        try:
            controller._deps.log(f"subtitle timing: seek line skipped: {e!r}")
        except Exception:  # noqa: BLE001
            pass
