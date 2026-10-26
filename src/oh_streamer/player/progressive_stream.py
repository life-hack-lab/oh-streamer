"""
progressive_stream: watch a queue item while the background worker is still processing it.

The worker (queue_worker.py) writes video-only Matroska segments work_dir/segments/seg_<k:06d>.mkv (each starts with an
IDR, timestamps from 0, possibly several video tracks titled oh:processed / oh:original / oh:restored, already
frame-interpolated) and lists the closed ones in work_dir/segments.json (atomically replaced). Segment k covers source
time video_start + k*n/fps .. + frames/fps (frames = SOURCE frames, also with 2x interpolation).

Manifest version 2: {"version": 2, "segments": [{"k", "file", "frames"[, "partial"]}, ...] (sorted by k, may have gaps), "last_k":
int | null (known once a run reached the natural end), "complete": bool, "total_frames", "play": {"fps": "30000/1001",
"n": N, "video_start": s, "rate_mult": 1 | 2, "layers": [...], "count": estimated segment count}}. Version 1 has no
"play" block and contiguous segments from 0 ("n" and key.fps / key.interpolation / key.compare are there); the rest
(video start, frame count) comes from an ffprobe of the source (probe_source / play_from_probe).

The player (oh_player.py, progressive mode) uses no Lada and no GPU: a Feeder thread remuxes the segments' packets
(every video track, no re-encoding, timestamps shifted to absolute source time) plus the source's audio (stream-copied
from the matching position) into ONE Matroska stream with absolute timestamps, served on tcp://127.0.0.1:<port> exactly
like a live segment, so oh_seek.lua's live-mode seeking works unchanged: seeks inside mpv's cache are native, others
come as seek requests, served by a new Feeder that starts at the segment containing the target (at the keyframe at /
before it). A segment that is not there yet is waited for with the stream open (mpv pauses for cache); meanwhile the
queue is asked for it (want, every WANT_INTERVAL s) and the manifest is polled every POLL_INTERVAL s.

ProgressivePlayback is the controller-side bundle (manifest watcher + current feeder + want / watching messages +
the processed ranges for user-data/oh/ranges). Pure parts (manifest parsing, ranges, segment lookup, progress text)
need neither mpv nor the queue: test_progressive_stream.py.
"""

import json
import math
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from oh_streamer.processing import compare_stack  # (pure python + numpy: the oh:* track titles)
from oh_streamer.queue import queue_ipc  # (stdlib only: parse_start)

POLL_INTERVAL = 0.5          # seconds between looks at segments.json
WANT_INTERVAL = 5.0          # a want for the segment the feeder waits for is repeated this often
SPEED_WINDOW = 30.0         # background speed (info panel): processed seconds over this many wall seconds
SPEED_MIN_SPAN = 8.0        # ... once the samples span at least this long
WATCHING_INTERVAL = 10.0     # = queue_ipc.BUSY_HEARTBEAT (not imported: this module stays free of the player's imports)
CLOCK_INTERVAL = 1.0         # the playback position goes to the queue (clock message) this often
OPEN_ATTEMPTS = 20           # a segment file that cannot be opened (a redo swapped it) is retried this many times
OPEN_RETRY_WAIT = 0.25
RESTORE_SOURCE_TIMEOUT = 5.0  # candidate preparation must not sit forever in proxy open/read I/O
SEGMENT_SECONDS = 4.0        # queue_worker's default segment length: the fallback before any manifest exists
AUDIO_LEAD = 0.1             # audio is muxed up to this far past the video packet being written
SOCKET_TICK = 0.25           # accept / send timeouts: how quickly a feeder notices it is being stopped
US = Fraction(1, 1_000_000)  # time base of the packets we hand to the muxer (it rescales to Matroska's 1 ms)
WAIT_STAGE = "背景處理中"    # progress stage while the segment needed is not processed yet
HOLD_STAGES = {"paused": "背景處理已暫停", "failed": "背景處理失敗"}   # ... while the queue will not work on it (want's reply)
LOAD_STAGE = "載入中"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------------------------------------------
# segment geometry + manifest (pure)
# ---------------------------------------------------------------------------------------------------------------

def _fraction(v, default=None):
    try:
        f = Fraction(str(v))
        return f if f > 0 else default
    except (ValueError, ZeroDivisionError, TypeError):
        return default


class PlayInfo:
    """How segments map onto source time: segment k starts at video_start + k*n/fps (seconds)."""

    def __init__(self, fps, n: int, video_start: float = 0.0, rate_mult: int = 1, layers=("processed",),
                 count: int | None = None, compare: bool = False, original: str | None = None):
        self.fps = Fraction(fps)
        if self.fps <= 0:
            raise ValueError(f"bad frame rate {fps!r}")
        self.n = max(1, int(n))
        self.video_start = float(video_start)
        self.rate_mult = 2 if rate_mult == 2 else 1
        self.layers = tuple(layers) or ("processed",)
        self.count = int(count) if count else None
        self.compare = bool(compare)                  # the item was queued with compare on
        self.original = original if original in ("copy", "encode") else None   # where the original track comes from

    def seg_start(self, k: int) -> float:
        return self.video_start + float(Fraction(k * self.n) / self.fps)

    def seg_end(self, k: int, frames: int) -> float:
        return self.video_start + float(Fraction(k * self.n + frames) / self.fps)

    def segment_at(self, t: float) -> int:
        """The segment containing source time t (clamped to 0 .. count-1 when the count is known)."""
        k = max(0, math.floor((float(t) - self.video_start) * float(self.fps) / self.n + 1e-9))
        return min(k, self.count - 1) if self.count else k

    def __eq__(self, other):
        return isinstance(other, PlayInfo) and (self.fps, self.n, self.video_start, self.rate_mult, self.layers,
                                                self.count, self.compare, self.original) == (
            other.fps, other.n, other.video_start, other.rate_mult, other.layers, other.count, other.compare,
            other.original)

    def __repr__(self):
        return (f"PlayInfo(fps={self.fps}, n={self.n}, video_start={self.video_start}, rate_mult={self.rate_mult}, "
                f"layers={self.layers}, count={self.count}, compare={self.compare}, original={self.original})")


class Segment:
    __slots__ = ("k", "file", "frames", "partial")

    def __init__(self, k: int, file: str, frames: int, partial: bool = False):
        self.k, self.file, self.frames, self.partial = k, file, frames, bool(partial)

    def __repr__(self):
        return f"Segment({self.k}, {self.file!r}, {self.frames})"


class ManifestState:
    """One reading of segments.json (or the empty state before the worker wrote one)."""

    def __init__(self, play: PlayInfo, segments: dict | None = None, last_k: int | None = None,
                 complete: bool = False, total_frames: int | None = None, version: int | None = None):
        self.play, self.segments = play, segments or {}
        self.last_k, self.complete, self.total_frames, self.version = last_k, complete, total_frames, version
        self._final_k_cached = False
        self._final_k_value = None

    def final_k(self) -> int | None:
        """Index of the video's last segment when known: last_k, a complete v1 list, or a segment shorter than N."""
        if self._final_k_cached:
            return self._final_k_value
        if self.last_k is not None:
            final = self.last_k
        else:
            short = [k for k, s in self.segments.items() if s.frames < self.play.n]
            if short:
                final = min(short)
            elif self.complete and self.segments:
                final = max(self.segments)
            else:
                final = None
        self._final_k_value = final
        self._final_k_cached = True
        return final

    def ranges(self) -> list[tuple[float, float]]:
        """Processed parts as absolute source seconds, adjacent segments merged."""
        p = self.play
        return merge_ranges([(p.seg_start(k), p.seg_end(k, s.frames)) for k, s in sorted(self.segments.items())])


def merge_ranges(spans, eps: float = 0.002) -> list[tuple[float, float]]:
    """Sorted, merged (a, b) spans; spans closer than eps (rounding) count as touching."""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def ranges_text(ranges) -> str:
    """user-data/oh/ranges: "a-b,c-d" (seconds, 3 decimals); "" = nothing processed."""
    return ",".join(f"{a:.3f}-{b:.3f}" for a, b in ranges)


def parse_manifest(raw, fallback: PlayInfo) -> ManifestState:
    """A ManifestState of a segments.json document (version 1 or 2). fallback: the PlayInfo derived from the source
    (probe), used where the manifest says nothing; a v1 manifest still corrects n / fps / rate / layers from its key."""
    if not isinstance(raw, dict):
        raise ValueError("manifest is not an object")
    version = raw.get("version", 1)
    key = raw.get("key") if isinstance(raw.get("key"), dict) else {}
    pl = raw.get("play") if isinstance(raw.get("play"), dict) else None
    fps = _fraction((pl or {}).get("fps"), None) or _fraction(key.get("fps"), None) or fallback.fps
    try:
        n = int((pl or {}).get("n") or raw.get("n") or key.get("n") or fallback.n)
    except (TypeError, ValueError):
        n = fallback.n
    if pl is not None:
        try:
            vstart = float(pl.get("video_start", fallback.video_start))
        except (TypeError, ValueError):
            vstart = fallback.video_start
        rate = pl.get("rate_mult", fallback.rate_mult)
        layers = tuple(x for x in pl.get("layers") or () if isinstance(x, str)) or fallback.layers
        count = pl.get("count")
        compare = pl["compare"] if isinstance(pl.get("compare"), bool) else len(layers) > 1   # (older v2: no key)
        original = pl.get("original") if pl.get("original") in ("copy", "encode") else             "encode" if "original" in layers else None
    else:
        vstart = fallback.video_start
        rate = 2 if key.get("interpolation") == "full" else 1 if "interpolation" in key else fallback.rate_mult
        layers = compare_stack.layer_names(rate) if key.get("compare") == "on" else \
            ("processed",) if "compare" in key else fallback.layers
        count = None
        compare = key.get("compare") == "on"
        original = "encode" if "original" in layers else None
    total = raw.get("total_frames")
    total = total if isinstance(total, int) and total > 0 else None
    if not isinstance(count, int) or count <= 0:
        frames = total or (fallback.count * fallback.n if fallback.count else None)
        count = -(-frames // n) if frames else None
    play = PlayInfo(fps, n, vstart, rate if rate in (1, 2) else 1, layers, count, compare, original)
    segs = {}
    for s in raw.get("segments") or []:
        try:
            k, name, frames = int(s["k"]), str(s["file"]), int(s["frames"])
        except (KeyError, TypeError, ValueError):
            continue
        if k < 0 or frames <= 0 or not name or os.path.basename(name) != name:
            continue                        # never a path outside segments/
        segs[k] = Segment(k, name, frames, s.get("partial") is True)
    last_k = raw.get("last_k")
    last_k = last_k if isinstance(last_k, int) and last_k >= 0 else None
    if version == 1 and raw.get("complete") and segs and last_k is None:
        last_k = max(segs)
    if last_k is not None and play.count != last_k + 1:
        play.count = last_k + 1
    return ManifestState(play, segs, last_k, bool(raw.get("complete")), total, version)


def read_manifest(path: str, fallback: PlayInfo) -> ManifestState | None:
    """parse_manifest of a file; None when it is missing / mid-replace / unreadable (the caller keeps what it had)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return parse_manifest(raw, fallback)
    except (OSError, ValueError):
        return None


def progress_json(req: str, waiting: bool, buffered_s: float | None = None, wait_s: float = 1.0,
                  floor_pct: int = 0, hold: str | None = None) -> tuple[str, int]:
    """user-data/oh/progress for a progressive start / seek that is not playing yet: 0 % with the 背景處理中 stage
    while the segment is not processed, else the share of mpv's startup buffer (cache-pause-wait) already there.
    Returns (json, pct); pct never goes below floor_pct (it only goes up within one request)."""
    if waiting:
        pct, stage = floor_pct, HOLD_STAGES.get(hold, WAIT_STAGE)
    else:
        frac = 0.0 if buffered_s is None or wait_s <= 0 else max(0.0, min(1.0, buffered_s / wait_s))
        pct, stage = max(floor_pct, min(99, int(frac * 100))), LOAD_STAGE
    return json.dumps({"req": req, "pct": pct, "stage": stage, "eta": None}, ensure_ascii=False), pct


# ---------------------------------------------------------------------------------------------------------------
# source probe + the single-instance message
# ---------------------------------------------------------------------------------------------------------------

def probe_source(path: str) -> dict:
    """{duration, fps (Fraction), video_start, frames, has_audio} of a local video file (ffprobe). Raises on failure."""
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
                       creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()[:200]}")
    return info_from_probe(json.loads(p.stdout))


def info_from_probe(probe: dict, *, full_vod: bool = False) -> dict:
    """The fields of probe_source() from an ffprobe -show_streams -show_format result (queue_worker.parse_probe's
    reading: r_frame_rate, nb_frames or duration * fps, the video stream's start time)."""
    streams = probe.get("streams") or []
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None and isinstance(probe.get("fps"), (str, int, float)):
        fps = _fraction(probe.get("fps"), None)
        if fps is not None and fps > 0:
            duration = float(probe.get("duration") or 0)
            frames = max(1, int(round(duration * float(fps)))) if full_vod else int(probe.get("frames") or 0)
            return {"duration": duration, "fps": fps, "video_start": float(probe.get("video_start") or 0),
                    "frames": frames or max(1, int(round(duration * float(fps)))),
                    "has_audio": bool(probe.get("has_audio"))}
    if vs is None:
        raise RuntimeError("the source has no video stream")
    fps = _fraction(vs.get("r_frame_rate"), None)
    if fps is None:
        raise RuntimeError("cannot determine the frame rate")
    fmt = probe.get("format") or {}
    metadata = probe.get("metadata") if isinstance(probe.get("metadata"), dict) else {}
    duration = float(probe.get("duration") or metadata.get("duration") or fmt.get("duration")
                     or vs.get("duration") or 0)
    frames = (max(1, int(round(duration * float(fps)))) if full_vod else
              int(vs.get("nb_frames") or 0) or max(1, int(round(duration * float(fps)))) )
    vstart = float(vs.get("start_time") or fmt.get("start_time") or 0)
    return {"duration": duration, "fps": fps, "video_start": vstart, "frames": frames,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def play_from_probe(info: dict, seconds: float = SEGMENT_SECONDS) -> PlayInfo:
    """The fallback PlayInfo before / besides the manifest: the worker's default plan for this source."""
    n = max(1, round(seconds * info["fps"]))
    return PlayInfo(info["fps"], n, info["video_start"], 1, ("processed",), -(-int(info["frames"]) // n))


def derive_1x_param(fps, video_start, rate_mult, compare, titles) -> str:
    """user-data/oh/derive-1x ("<video_start>,<src_fps>", e.g. "0.033367,30000/1001") for oh_seek.lua: the "restored"
    compare layer of a rate_mult-2 output without a stored oh:restored track is the even slots of oh:processed
    "" = nothing to derive (1x output, compare off, a real
    restored track, no processed track, or unusable numbers)."""
    titles = list(titles or ())
    f = _fraction(fps, None)
    try:
        start = float(video_start)
    except (TypeError, ValueError):
        return ""
    if rate_mult != 2 or not compare or f is None or "oh:processed" not in titles or "oh:restored" in titles \
            or not math.isfinite(start):
        return ""
    return f"{start:.6f},{f.numerator}/{f.denominator}"


class ProgressiveOpen:
    """A queue item to watch while it is processed: {"input", "progressive": true, "queue_id", "work_dir", "title"}."""

    def __init__(self, path: str, work_dir: str, queue_id: str | None, title: str | None, info: dict,
                 source_url: str | None = None, source_probe: dict | None = None):
        self.path, self.work_dir, self.queue_id, self.info = path, work_dir, queue_id, info
        self.source_url, self.source_probe = source_url, source_probe
        self.title = title or Path(path).name
        self.duration = float(info.get("duration") or 0.0)
        self.has_audio = bool(info.get("has_audio"))
        self.fallback = play_from_probe(info)
        self.start = 0.0
        self.start_requested = False       # the message carried a valid "start" (play_progressive then always reopens there)
        self.pressed_at: float | None = None   # wall clock of the queue's 播放 press (T1.3); logged once, then cleared
        self.subtitles = []
        self.subtitle_offset = 0.0
        self.subtitle_error = False
        self.subtitle_covered_s = None
        self.subtitle_covered_ranges = None
        self.subtitle_complete = True


def subtitle_tracks(value):
    """Ignore malformed optional metadata without making the video unplayable."""
    if not isinstance(value, list):
        return []
    tracks, seen = [], set()
    for track in value:
        if not isinstance(track, dict):
            continue
        path = track.get("path")
        if not isinstance(path, str) or not path or not os.path.isfile(path):
            continue
        if not all(isinstance(track.get(k), str) for k in ("lang", "title")) \
                or not isinstance(track.get("default"), bool):
            continue
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        if key not in seen:
            tracks.append(dict(track, path=path))
            seen.add(key)
    return tracks


def subtitle_covered_ranges(value):
    """Validated audio-clock coverage intervals, or None for an absent/malformed optional field.

    An empty list explicitly means no coverage; None keeps compatibility with the old covered_s prefix.
    Normalize ordering/overlaps so every consumer sees one consistent snapshot.
    """
    if not isinstance(value, list):
        return None
    ranges = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        a, b = pair
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair) \
                or a < 0 or b < a:
            return None
        ranges.append([float(a), float(b)])
    merged = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def shifted_srt(path: str, offset: float, directory: str, index: int) -> str:
    """Map audio-relative SRT onto the feeder's absolute source clock; preserve text/style."""
    if not offset:
        return path
    def stamp(match):
        h, m, s, ms = map(int, match.groups())
        total = max(0, round(((h * 3600 + m * 60 + s) * 1000 + ms) + offset * 1000))
        seconds, millis = divmod(total, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    text = Path(path).read_text(encoding="utf-8-sig")
    # Only timing lines: dialogue containing a timestamp must remain unchanged.
    lines = [re.sub(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", stamp, line)
             if re.match(r"^\s*\d+:\d{2}:\d{2}[,.]\d{3}\s+-->", line) else line
             for line in text.splitlines()]
    target = Path(directory) / f"track-{index}.srt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def progressive_from_message(msg: dict, probe=probe_source) -> ProgressiveOpen | None:
    """Open the stable local playlist identity, using the shared full VOD probe when the queue supplies one."""
    path, work = msg.get("input"), msg.get("work_dir")
    if msg.get("progressive") is not True or not isinstance(path, str) or not path.strip() \
            or not isinstance(work, str) or not work.strip():
        return None
    path = os.path.abspath(path.strip().strip('"'))
    if not os.path.isfile(path):
        return None
    source_url = msg.get("source_url")
    if source_url is not None and not is_local_runtime_url(source_url):
        return None
    source_probe = msg.get("source_probe")
    try:
        if source_url is not None:
            if not isinstance(source_probe, dict):
                return None
            info = info_from_probe(source_probe, full_vod=True)
        else:
            info = probe(path)
    except Exception:  # noqa: BLE001 - unreadable source: the caller answers "not playable"
        return None
    qid, title = msg.get("queue_id"), msg.get("title")
    po = ProgressiveOpen(path, os.path.abspath(work.strip().strip('"')), str(qid) if qid not in (None, "") else None,
                         title if isinstance(title, str) and title.strip() else None, info, source_url, source_probe)
    start = queue_ipc.parse_start(msg.get("start"), po.duration)
    if start is not None:
        po.start, po.start_requested = start, True
    pressed = msg.get("pressed_at")
    if isinstance(pressed, (int, float)) and not isinstance(pressed, bool) and math.isfinite(pressed):
        po.pressed_at = float(pressed)
    offset = msg.get("subtitle_offset", 0.0)
    po.subtitle_error = msg.get("subtitle_error") is True
    if isinstance(offset, (int, float)) and not isinstance(offset, bool) and math.isfinite(offset):
        po.subtitle_offset = float(offset)
        po.subtitles = subtitle_tracks(msg.get("subtitles"))
    po.subtitle_covered_ranges = subtitle_covered_ranges(msg.get("subtitle_covered_ranges"))
    covered = msg.get("subtitle_covered_s")
    if isinstance(covered, (int, float)) and not isinstance(covered, bool) and math.isfinite(covered) and covered >= 0:
        po.subtitle_covered_s = float(covered)
        complete = msg.get("subtitle_complete")
        po.subtitle_complete = complete if isinstance(complete, bool) else True
    return po


def is_local_runtime_url(value) -> bool:
    """The shared source cache is exposed only on plain HTTP IPv4 loopback URLs."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None
                and 0 < parsed.port <= 65535 and parsed.username is None and parsed.password is None
                and not parsed.fragment)
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# manifest watcher
# ---------------------------------------------------------------------------------------------------------------

class ManifestWatcher:
    """Polls work_dir/segments.json every `interval` s on its own thread (re-reading it only when its mtime / size
    changed). state = the latest ManifestState; wait_newer() lets a feeder sleep until it changes."""

    def __init__(self, work_dir: str, fallback: PlayInfo, interval: float = POLL_INTERVAL, log=lambda m: None):
        self.path = os.path.join(work_dir, "segments.json")
        self.seg_dir = os.path.join(work_dir, "segments")
        self.fallback, self.interval, self.log = fallback, interval, log
        self.state = ManifestState(fallback)
        self.seq = 0
        self._sig = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "ManifestWatcher":
        self.poll()
        self._thread = threading.Thread(target=self._run, name="manifest watcher", daemon=True)
        self._thread.start()
        return self

    def poll(self) -> bool:
        """Re-read the manifest if it changed; True when the state did."""
        try:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return False
        if sig == self._sig:
            return False
        state = read_manifest(self.path, self.fallback)
        if state is None:
            return False                    # mid-replace / partial: next poll
        self._sig = sig
        with self._cond:
            self.state = state
            self.seq += 1
            self._cond.notify_all()
        return True

    def wait_newer(self, seq: int, timeout: float) -> tuple[ManifestState, int]:
        """(state, its seq) as soon as the state is newer than seq, or after timeout."""
        with self._cond:
            if self.seq == seq:
                self._cond.wait(timeout)
            return self.state, self.seq

    def segment_path(self, seg: Segment) -> str:
        return os.path.join(self.seg_dir, seg.file)

    def wake(self):
        """Wake every wait_newer() at once (a feeder being stopped)."""
        with self._cond:
            self._cond.notify_all()

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.poll()
            except Exception as e:  # noqa: BLE001 - a broken read must not end watching
                self.log(f"progressive: manifest poll failed: {e!r}")

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------------------------------------------
# feeder: segments + source audio -> one Matroska stream over local TCP
# ---------------------------------------------------------------------------------------------------------------

class _Stopped(Exception):
    pass


class _NullSink:
    """The restore candidate uses Feeder's demux/remux helpers without opening a TCP listener."""

    url = ""

    def close(self, graceful: bool = False):
        pass


class TcpSink:
    """A listening socket on 127.0.0.1 (bound at once: the URL can be handed to mpv before the stream starts); the
    feeder accepts one connection and writes the Matroska stream into it."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.sock.settimeout(SOCKET_TICK)
        self.port = self.sock.getsockname()[1]
        self.url = f"tcp://127.0.0.1:{self.port}"
        self.conn = None

    def accept(self, stop: threading.Event):
        while True:
            if stop.is_set():
                raise _Stopped()
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            conn.settimeout(SOCKET_TICK)
            self.conn = conn
            return conn

    def close(self, graceful: bool = False):
        for s in (self.conn, self.sock):
            if s is None:
                continue
            try:
                if graceful and s is self.conn:
                    s.shutdown(socket.SHUT_WR)   # EOF after everything sent (end of the video)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        self.conn = None


class _SockWriter:
    """File-like output for PyAV: send() with a short timeout so a stop is noticed while mpv does not read (its cache
    is full / it is gone). PyAV sees no seek(): the Matroska muxer writes a live (non-seekable) stream."""

    def __init__(self, conn, stop: threading.Event):
        self.conn, self.stop, self.bytes = conn, stop, 0

    def _send(self, data):
        view = memoryview(data)
        while view.nbytes:
            if self.stop.is_set():
                raise _Stopped()
            try:
                sent = self.conn.send(view)
            except socket.timeout:
                continue
            view = view[sent:]

    def write(self, data) -> int:
        self._send(data)
        self.bytes += len(data)
        return len(data)


class _Side:
    """A source stream copied into the output next to the segments' tracks (the audio, the original video of
    compare=copy): its packets are muxed, in source time, up to the time of the segment packet being written."""

    def __init__(self, kind: str, container, stream, out, start: float):
        self.kind, self.container, self.stream, self.out, self.start = kind, container, stream, out, start
        self.demux = container.demux(stream)
        self.pkt = None                       # read but not muxed yet


def _video_packets(container):
    """Keep initial B-frame keyframes whose Matroska DTS is only known after a few packets."""
    pending = {}
    rates = {s.index: _fraction(s.average_rate, Fraction(30)) for s in container.streams.video}
    for packet in container.demux(container.streams.video):
        if packet.pts is None or packet.time_base is None:
            continue
        index = packet.stream.index
        if packet.dts is None:
            prefix = pending.setdefault(index, [])
            if len(prefix) >= 64:
                raise ValueError("video packet decode timestamps remain unavailable")
            prefix.append(packet)
            continue
        prefix = pending.pop(index, [])
        if prefix:
            step = Fraction(1, 1) / rates[index] / packet.time_base
            anchor = packet.dts
            for i, earlier in enumerate(prefix):
                earlier.dts = anchor - round((len(prefix) - i) * step)
            yield from prefix
        yield packet
    for index, prefix in pending.items():
        # A very short final segment may end before its demuxer inferred any DTS.
        step = Fraction(1, 1) / rates[index] / prefix[0].time_base
        anchor = min(p.pts for p in prefix) - round((len(prefix) - 1) * step)
        for i, packet in enumerate(prefix):
            packet.dts = anchor + round(i * step)
        yield from prefix


class Feeder:
    """One stream for mpv, from the keyframe at / before `target` on (absolute source time).

    Thread: wait for the target's segment (waiting_k meanwhile) -> find the start keyframe -> ready (url / titles /
    first_s valid) -> accept mpv's connection -> remux segment after segment, waiting for each missing one, with the
    source's audio interleaved -> after the video's last segment the stream ends (EOF). close() waits for cleanup;
    retire() only requests it so the controller thread never waits on slow input I/O."""

    def __init__(self, watcher: ManifestWatcher, source: str, target: float, has_audio: bool = True,
                 log=lambda m: None, sink: TcpSink | None = None, *, audio_floor: float | None = None):
        self.watcher, self.source, self.target = watcher, source, float(target)
        self.cache_source = is_local_runtime_url(source)
        self.has_audio, self.log = has_audio, log
        self.sink = sink or TcpSink()
        self.url = self.sink.url
        self.stop_ev = threading.Event()
        self.ready = threading.Event()
        self.prepared = threading.Event()       # a restore candidate clip is closed and seek-verified
        self.waiting_k: int | None = None     # segment the stream is waiting for (not processed yet)
        self.k: int | None = None             # segment being fed / waited for
        self.first_s: float | None = None     # start of the stream (keyframe at / before the target)
        self.titles: list[str] = []           # video track titles (output track order)
        self.play: PlayInfo | None = None     # the manifest's play block when the stream was set up
        self.out_t: float | None = None       # timestamp of the last video packet written
        self.ended = False                    # the whole rest of the video was written (EOF sent)
        self.error: Exception | None = None
        self.bytes = 0
        self._sides: list[_Side] = []         # source audio / original video being copied in
        self._vsrc = None                     # the source opened for the original video track (compare=copy)
        self._audio_floor = audio_floor       # continuation: the startup clip owns packets beginning before here
        self._audio_limit = None              # finite startup clips must not prefetch audio beyond their boundary
        self._thread = threading.Thread(target=self._run, name="progressive feeder", daemon=True)

    def start(self) -> "Feeder":
        self._thread.start()
        return self

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def derive_1x(self) -> str:
        """user-data/oh/derive-1x for this stream ("" = none): see derive_1x_param."""
        p = self.play
        return derive_1x_param(p.fps, p.video_start, p.rate_mult, p.compare, self.titles) if p is not None else ""

    def _open_source(self, av):
        """Open the source; ordinary feeders preserve their existing unlimited streaming read behavior."""
        timeout = getattr(self, "_source_timeout", None)
        return av.open(self.source, timeout=timeout) if timeout is not None else av.open(self.source)

    def compare_titles(self) -> str:
        """Comma list of the oh:* track titles when there are at least two, or one when the "restored" layer is
        derived from it (oh_seek.lua's compare keys)."""
        known = set(compare_stack.LAYER_TITLES.values())
        keep = [t for t in self.titles if t in known]
        return ",".join(keep) if len(keep) >= 2 or (keep and self.derive_1x()) else ""

    def retire(self):
        """Request cancellation without joining; safe on the controller thread even if a source read is blocked."""
        self.stop_ev.set()
        self.sink.close()
        self.watcher.wake()

    def close(self, timeout: float = 5.0):
        """Request cancellation and wait for the thread to release files and sockets when it can."""
        self.retire()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.log("progressive: feeder did not stop in time")

    # -- thread ----------------------------------------------------------------------------------------------------

    def _await(self, k_of, *, require_complete: bool = False) -> tuple[int, Segment, ManifestState] | None:
        """Block until the segment k_of(state) is listed and its file exists; optionally require a non-partial segment.
        None = past the video's last segment."""
        seq = -1
        while True:
            if self.stop_ev.is_set():
                raise _Stopped()
            state, seq = self.watcher.wait_newer(seq, SOCKET_TICK)
            k = k_of(state)
            final = state.final_k()
            if final is not None and k > final:
                return None
            self.k = k
            seg = state.segments.get(k)
            if seg is not None and (not require_complete or not seg.partial) \
                    and os.path.isfile(self.watcher.segment_path(seg)):
                self.waiting_k = None
                return k, seg, state
            if self.waiting_k != k:
                self.log(f"progressive: waiting for segment {k} ({state.play.seg_start(k):.1f}s, not processed yet)")
            self.waiting_k = k

    def _open_retry(self, k: int, seg: Segment, fn):
        """(fn(path of seg's file), seg). A redo of a segment (queue_worker) swaps the file name in the manifest and
        removes the old file: when opening fails, re-read the manifest and retry with the file listed NOW instead of
        failing; a file that stays unreadable under an unchanged name fails after OPEN_ATTEMPTS."""
        for attempt in range(OPEN_ATTEMPTS):
            path = self.watcher.segment_path(seg)
            try:
                return fn(path), seg
            except _Stopped:
                raise
            except Exception as e:  # noqa: BLE001 - FileNotFoundError / PermissionError / PyAV's own errors
                if self.stop_ev.is_set():
                    raise _Stopped()
                if attempt == OPEN_ATTEMPTS - 1:
                    raise
                self.watcher.poll()
                fresh = self.watcher.state.segments.get(k)
                if fresh is not None and fresh.file != seg.file:
                    self.log(f"progressive: segment {k} is now {fresh.file} (was {seg.file}: {e!r})")
                    seg = fresh
                    continue
                if self.stop_ev.wait(OPEN_RETRY_WAIT):
                    raise _Stopped()
        raise RuntimeError(f"segment {k} cannot be opened")     # (not reached)

    def _run(self):
        import av   # PyAV (Lada's venv); imported here so the pure parts above need nothing but the stdlib + numpy
        out = src = None
        try:
            target = self.target
            got = self._await(lambda st: self._first_k(st, target))
            if got is None:
                self.log(f"progressive: nothing to play at {target:.2f}s")
                return
            k, seg, state = got
            rel = target - state.play.seg_start(k)
            (drop, titles, start_rel), seg = self._open_retry(k, seg, lambda path: self._scan_start(av, path, rel))
            if state.play.compare and titles and not titles[0]:
                titles = ["oh:processed"] + titles[1:]   # single-track segments (original=copy) carry no title
            self.titles = titles
            self.play = state.play
            self.first_s = state.play.seg_start(k) + start_rel
            if state.play.compare and state.play.original == "copy" and "oh:original" not in titles:
                self._vsrc = self._open_original(av)        # (adds the title when the source video can be copied)
            self.ready.set()
            conn = self.sink.accept(self.stop_ev)
            writer = _SockWriter(conn, self.stop_ev)
            out = av.open(writer, "w", format="matroska",
                          container_options={"flush_packets": "1", "avoid_negative_ts": "disabled"})
            vmap, seg, src = self._setup_output(out, av, k, seg)
            self.log(f"progressive: stream {self.url} from segment {k} at {self.first_s:.3f}s (target {target:.3f}s, "
                     f"{len(vmap)} video track(s)"
                     f"{', original copied' if any(sd.kind == 'video' for sd in self._sides) else ''}"
                     f"{', audio copied' if any(sd.kind == 'audio' for sd in self._sides) else ''})")
            while True:
                c, seg = self._open_retry(k, seg, av_open)
                with c:
                    self._feed(out, k, seg, state.play, c, vmap, drop)
                drop = None
                end_t = state.play.seg_end(k, seg.frames)
                nk = k + 1
                got = self._await(lambda st: nk)
                if got is None:
                    break
                k, seg, state = got
            self._mux_sides_until(out, end_t)
            out.close()
            out = None
            self.bytes = writer.bytes
            self.ended = True
            self.sink.close(graceful=True)
            self.log(f"progressive: end of the video written ({self.out_t:.2f}s), stream closed")
        except _Stopped:
            pass
        except ConnectionError as e:
            if not self.stop_ev.is_set():
                self.error = e
                self.log(f"progressive: mpv closed the stream {self.url} ({e!r})")
        except Exception as e:  # noqa: BLE001 - reported to the controller (log); mpv just sees the stream end
            if not self.stop_ev.is_set():
                self.error = e
                self.log(f"progressive: feeder failed: {e!r}")
        finally:
            for c in (out, src, self._vsrc):
                if c is not None:
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001 - the socket may be gone: the trailer cannot be written
                        pass
            self._sides = []
            self._vsrc = None
            self.waiting_k = None
            self.sink.close()                       # (idempotent; a normal end closed it gracefully already)

    def _setup_output(self, out, av, k: int, seg: Segment):
        """Add the first segment's video templates and the copied source streams to an output muxer."""
        src = None
        vmap, seg = self._open_retry(k, seg, lambda path: self._add_video(out, av, path))
        if self._vsrc is not None:
            self._add_original(out)
        if self.has_audio:
            src = self._add_audio(out, av)
        return vmap, seg, src

    @staticmethod
    def _first_k(state: ManifestState, target: float) -> int:
        k = state.play.segment_at(target)
        final = state.final_k()
        return min(k, final) if final is not None else k

    def _scan_start(self, av, path: str, rel: float):
        """Where the stream starts inside its first segment: the first track's last keyframe at / before `rel` (seconds
        into the segment), the other tracks at their last keyframe at / before that (normally the same instant: the
        worker forces keyframes on all tracks alike). Returns ({stream index: dts to start at (s)}, titles, start)."""
        keys: dict[int, list[tuple[float, float]]] = {}
        with av.open(path) as c:
            vs = list(c.streams.video)
            titles = [(s.metadata or {}).get("title", "") for s in vs]
            idx = [s.index for s in vs]          # (PyAV stream objects must not be touched once c is closed)
            for p in _video_packets(c):
                if p.dts is None or p.pts is None:
                    continue
                t = float(p.dts * p.time_base)
                if t > rel + 1.0:
                    break
                if p.is_keyframe:
                    keys.setdefault(p.stream.index, []).append((float(p.pts * p.time_base), t))
        idx0 = idx[0]
        k0 = [kf for kf in keys.get(idx0, []) if kf[0] <= rel + 1e-6] or keys.get(idx0, [])[:1] or [(0.0, 0.0)]
        start_pts, start_dts = k0[-1]
        drop = {idx0: start_dts}
        for i in idx[1:]:
            kk = [kf for kf in keys.get(i, []) if kf[0] <= start_pts + 1e-6] or keys.get(i, [])[:1]
            drop[i] = kk[-1][1] if kk else start_dts
        return drop, titles, start_pts

    def _add_video(self, out, av, path: str) -> dict:
        """Output video tracks from the first segment's (codec parameters, title, default flag); no codec is opened for
        encoding (opaque template: the packets are copied)."""
        vmap = {}
        with av.open(path) as c:
            for i, s in enumerate(c.streams.video):
                o = out.add_stream_from_template(s, opaque=True)
                o.codec_context.thread_count = 1     # start_encoding opens the (decoder) context: keep it cheap
                title = (s.metadata or {}).get("title") or (self.titles[i] if i < len(self.titles) else "")
                if title:
                    o.metadata["title"] = title
                try:
                    o.disposition = s.disposition.value
                except (AttributeError, TypeError, ValueError):
                    pass
                vmap[s.index] = o
        return vmap

    def _add_audio(self, out, av):
        """Stream-copy the source's first audio track from first_s on; the open source container is returned (closed
        by the thread). No audio (logged) when the codec cannot go into Matroska or the source cannot be read."""
        src = None
        try:
            src = self._open_source(av)
            if not src.streams.audio:
                if self.cache_source and self.has_audio:
                    raise RuntimeError("shared probe reports audio, but the runtime source has no audio stream")
                src.close()
                return None
            a = src.streams.audio[0]
            start = self.first_s or 0.0
            # HLS may start its audio slightly before video. Seeking to that audio
            # timestamp can fail (EPERM) or discard the first audio packets. At the
            # beginning just demux forward; _Side already drops packets before start.
            source_start = max((float(s.start_time * s.time_base) for s in src.streams
                                if s.start_time is not None and s.time_base is not None), default=0.0)
            if start > max(0.0, source_start) + 0.5:
                src.seek(_seek_ts(a, start - 0.5), stream=a, backward=True)
            # Do not add an empty output track if opening/seeking the source failed.
            o = out.add_stream_from_template(a)
            try:
                o.disposition = a.disposition.value
            except (AttributeError, TypeError, ValueError):
                pass
            if not o.codec_context.extradata:
                asc = aac_config(a.codec_context)
                if asc:
                    o.codec_context.extradata = asc
            self._sides.append(_Side("audio", src, a, o, start))
            return src
        except Exception as e:  # noqa: BLE001 - play without sound rather than not at all
            if self.cache_source:
                raise RuntimeError(f"cached source audio could not be opened: {e!r}") from e
            self.log(f"progressive: source audio not used ({e!r})")
            if src is not None:
                src.close()
            return None

    def _open_original(self, av):
        """compare=copy: open the source for its video stream (stream-copied into the output as oh:original, like the
        audio). Returns the open container, or None (logged) when the source cannot be read: the compare then has the
        processed track only."""
        src = None
        try:
            src = self._open_source(av)
            if not src.streams.video:
                if self.cache_source:
                    raise RuntimeError("the runtime source has no video stream")
                src.close()
                return None
            self.titles = self.titles + ["oh:original"]
            return src
        except Exception as e:  # noqa: BLE001 - play without the original rather than not at all
            if self.cache_source:
                raise RuntimeError(f"cached source video could not be opened: {e!r}") from e
            self.log(f"progressive: source video not used as the original track ({e!r})")
            if src is not None:
                try:
                    src.close()
                except Exception:  # noqa: BLE001
                    pass
            return None

    def _add_original(self, out):
        """Add the source's video stream (packets copied; source timestamps = output timestamps) as oh:original,
        from the keyframe at / before first_s on."""
        src = self._vsrc
        try:
            vs = src.streams.video[0]
            o = out.add_stream_from_template(vs)
            o.metadata["title"] = "oh:original"
            try:
                o.disposition = 0
            except (AttributeError, TypeError, ValueError):
                pass
            start = self.first_s or 0.0
            if start > 0.0:
                src.seek(_seek_ts(vs, start), stream=vs, backward=True)
            self._sides.append(_Side("video", src, vs, o, start))
        except Exception as e:  # noqa: BLE001
            self.log(f"progressive: original track not added ({e!r})")
            self.titles = [t for t in self.titles if t != "oh:original"]

    def _mux_sides_until(self, out, t: float):
        """Mux the copied source packets (audio, original video) that start before t (absolute seconds; the source's
        own timestamps)."""
        for sd in self._sides:
            while sd.demux is not None:
                p = sd.pkt
                if p is None:
                    p = next(sd.demux, None)
                    if p is None:
                        sd.demux = None
                        break
                    if p.pts is None or p.dts is None or (sd.kind == "audio" and p.pts < 0):
                        continue
                    if sd.kind == "audio" and float((p.pts + (p.duration or 0)) * p.time_base) <= sd.start:
                        continue                   # wholly before the stream start (the seek lands earlier)
                    if sd.kind == "audio" and self._audio_floor is not None and \
                            float(p.pts * p.time_base) < self._audio_floor - 1e-6:
                        continue                   # the startup clip already contains this entire cross-boundary packet
                    sd.pkt = p
                limit = min(t, self._audio_limit) if sd.kind == "audio" and self._audio_limit is not None else t
                if float((p.dts if sd.kind == "video" else p.pts) * p.time_base) >= limit:
                    break
                sd.pkt = None
                p.stream = sd.out
                out.mux(p)

    def _feed(self, out, k: int, seg: Segment, play: PlayInfo, c, vmap: dict, drop: dict | None):
        """Remux one segment (c = its open container): every video packet shifted by the segment's start time, audio
        interleaved."""
        off = Fraction(play.seg_start(k))
        for p in _video_packets(c):
            if self.stop_ev.is_set():
                raise _Stopped()
            if p.dts is None or p.pts is None:
                continue
            o = vmap.get(p.stream.index)
            if o is None:
                continue
            tb = p.time_base
            if drop is not None and float(p.dts * tb) < drop.get(p.stream.index, 0.0) - 1e-6:
                continue
            pts = round((p.pts * tb + off) / US)
            dts = round((p.dts * tb + off) / US)
            p.pts, p.dts, p.time_base = pts, dts, US
            self._mux_sides_until(out, dts * 1e-6 + AUDIO_LEAD)
            p.stream = o
            out.mux(p)
            t = pts * 1e-6
            if self.out_t is None or t > self.out_t:
                self.out_t = t


class RestoreCandidate(Feeder):
    """A finite, seekable Matroska clip for the processed segment covering target, plus its next Feeder."""

    def __init__(self, watcher: ManifestWatcher, source: str, target: float, has_audio: bool,
                 work_dir: str, log=lambda m: None):
        super().__init__(watcher, source, target, has_audio, log, sink=_NullSink())
        self._source_timeout = (RESTORE_SOURCE_TIMEOUT, RESTORE_SOURCE_TIMEOUT)
        fd, self.url = tempfile.mkstemp(prefix=".restore-candidate-", suffix=".mkv", dir=work_dir)
        os.close(fd)
        self.path = self.url
        self.end_s: float | None = None
        self.continuation: Feeder | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._thread = threading.Thread(target=self._build_clip, name="progressive restore candidate", daemon=True)

    def start(self) -> "RestoreCandidate":
        self._thread.start()
        return self

    def _build_clip(self):
        import av
        out = src = None
        try:
            target = self.target
            got = self._await(lambda st: self._first_k(st, target), require_complete=True)
            if got is None:
                raise RuntimeError(f"no complete restored segment at {target:.3f}s")
            k, seg, state = got
            self.k = k
            rel = target - state.play.seg_start(k)
            (drop, titles, start_rel), seg = self._open_retry(k, seg, lambda path: self._scan_start(av, path, rel))
            if state.play.compare and titles and not titles[0]:
                titles = ["oh:processed"] + titles[1:]
            self.titles = titles
            self.play = state.play
            self.first_s = state.play.seg_start(k) + start_rel
            self.end_s = state.play.seg_end(k, seg.frames)
            self._audio_limit = self.end_s
            if state.play.compare and state.play.original == "copy" and "oh:original" not in titles:
                self._vsrc = self._open_original(av)

            out = av.open(self.url, "w", format="matroska", container_options={"avoid_negative_ts": "disabled"})
            vmap, seg, src = self._setup_output(out, av, k, seg)
            if self.has_audio and not any(sd.kind == "audio" for sd in self._sides):
                raise RuntimeError("candidate source audio could not be prepared")
            c, seg = self._open_retry(k, seg, av_open)
            with c:
                self._feed(out, k, seg, state.play, c, vmap, drop)
            self._mux_sides_until(out, self.end_s)
            out.close()                          # seekable Matroska writes its cue index on close
            out = None
            if not os.path.getsize(self.url):
                raise RuntimeError("candidate clip is empty")
            self._verify_seek(av)

            self.watcher.poll()
            current = self.watcher.state
            final_k = current.final_k()
            if final_k is None or k < final_k:
                next_start = state.play.seg_start(k + 1)
                self.end_s = next_start
                if self.stop_ev.is_set():
                    raise _Stopped()
                continuation = Feeder(self.watcher, self.source, next_start, self.has_audio, self.log,
                                      audio_floor=next_start).start()
                self.continuation = continuation
                if self.stop_ev.is_set():
                    self.continuation = None
                    continuation.retire()
                    raise _Stopped()
            self.bytes = os.path.getsize(self.url)
            self.ended = True
            self.ready.set()
            self.prepared.set()
            self.log(f"progressive: prepared seekable clip {self.url} from {self.first_s:.3f}s to "
                     f"{self.end_s:.3f}s (target {target:.3f}s)")
        except _Stopped:
            pass
        except Exception as e:  # noqa: BLE001 - candidate failure leaves the original source playing
            if not self.stop_ev.is_set():
                self.error = e
                self.log(f"progressive: restore candidate failed: {e!r}")
        finally:
            for c in (out, src, self._vsrc):
                if c is not None:
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001
                        pass
            self._sides = []
            self._vsrc = None
            self.waiting_k = None
            self.sink.close()

    def _verify_seek(self, av):
        """Reopen the closed clip, seek to target through its cue index and require a nearby packet."""
        with av.open(self.url) as c:
            videos = list(c.streams.video)
            if not videos or len(videos) != len(self.titles):
                raise RuntimeError("candidate clip video tracks failed verification")
            if self.has_audio and not c.streams.audio:
                raise RuntimeError("candidate clip audio failed verification")
            video = videos[0]
            c.seek(_seek_ts(video, self.target), stream=video, backward=True)
            for packet in c.demux(video):
                if packet.pts is None or packet.time_base is None:
                    continue
                t = float(packet.pts * packet.time_base)
                if t >= self.target - 0.1:
                    if t > (self.end_s or t) + 0.05:
                        break
                    return
            raise RuntimeError("candidate clip could not seek near the requested position")

    def retire(self, keep_continuation: bool = False):
        """Cancel preparation and optionally its continuation, then delete the clip off the controller thread."""
        super().retire()
        if self.continuation is not None and not keep_continuation:
            continuation, self.continuation = self.continuation, None
            continuation.retire()
        if self._cleanup_thread is None:
            self._cleanup_thread = threading.Thread(target=self._delete_clip_when_released,
                                                    name="progressive clip cleanup", daemon=True)
            self._cleanup_thread.start()

    def _delete_clip_when_released(self):
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join()
        for _ in range(120):
            try:
                os.remove(self.url)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.5)         # mpv can still have the active clip open on Windows
            except OSError as e:
                self.log(f"progressive: candidate clip cleanup failed: {e!r}")
                return
        self.log(f"progressive: candidate clip remains locked: {self.url}")


def av_open(path: str):
    import av
    return av.open(path)


def _seek_ts(stream, t: float) -> int:
    """t (absolute seconds) in stream's time base, never before the stream's start: FFmpeg's HLS demuxer refuses
    (EPERM) a seek before the first segment's timestamps (MPEG-TS sources start at ~1.4 s, not 0)."""
    ts = max(0, int(t / stream.time_base))
    first = stream.start_time
    return max(ts, first) if first is not None else ts


AAC_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)


def aac_config(cc) -> bytes | None:
    """AudioSpecificConfig for an AAC stream that has none (ADTS in MPEG-TS / HLS): Matroska written to a socket needs
    it in the header (FFmpeg: "Error parsing AAC extradata"); the muxer strips the ADTS headers itself
    (aac_adtstoasc). None for other codecs / unusual parameters."""
    if getattr(cc, "name", None) != "aac":
        return None
    try:
        rate = AAC_RATES.index(int(cc.sample_rate))
        ch = cc.layout.nb_channels if getattr(cc, "layout", None) is not None else int(cc.channels)
    except (AttributeError, TypeError, ValueError):
        return None
    if not 0 < ch <= 7:
        return None
    obj = {"Main": 1, "LC": 2, "SSR": 3, "LTP": 4, "HE-AAC": 2, "HE-AACv2": 2}.get(getattr(cc, "profile", None) or "LC", 2)
    return ((obj << 11) | (rate << 7) | (ch << 3)).to_bytes(2, "big")


# ---------------------------------------------------------------------------------------------------------------
# controller side
# ---------------------------------------------------------------------------------------------------------------

class ProgressivePlayback:
    """Everything the player's controller needs for one progressive source: the manifest watcher, the current
    feeder (start / seek replace it), the queue messages (watching every WATCHING_INTERVAL s, want for the segment
    the feeder waits for: at once and every WANT_INTERVAL s, and for the start segment of every (re)start) and the
    processed ranges. send(payload) must not block (QueueLink.post); None = no queue to talk to."""

    def __init__(self, po: ProgressiveOpen, send=None, log=lambda m: None, clock=time.monotonic,
                 watching_interval: float = WATCHING_INTERVAL, replies: bool = False):
        self.po, self.log, self._now = po, log, clock
        self._send = send if po.queue_id else None
        self.watching_interval = watching_interval
        self.watcher = ManifestWatcher(po.work_dir, po.fallback, log=log).start()
        self.feeder: Feeder | None = None
        self.target = 0.0
        self.last_position: float | None = None
        self._want = (None, 0.0)          # (k, when) of the last want sent
        self._want_generation = 0         # ignore replies for superseded starts / source transitions
        self._candidate_mode = False      # a restore candidate must not issue queue wants on behalf of source playback
        self._replies = replies           # send(payload, callback) delivers the queue's reply (QueueLink.post)
        self.hold: str | None = None      # "paused" / "failed": the queue's last want reply says it will not work on it
        self._next_watching = 0.0
        self._next_clock = 0.0
        self._samples: deque = deque()   # (wall time, processed seconds), see sample()

    def send(self, payload: dict):
        if self._send is not None:
            self._send(payload)

    def want(self, k: int, *, foreground: bool = False):
        """Request a segment; foreground marks a user-directed start/seek, not feeder lookahead."""
        self._want_generation += 1
        generation = self._want_generation
        self._want = (k, self._now())
        self.log(f"progressive: want segment {k} (queue item {self.po.queue_id})")
        payload = {"cmd": "want", "id": self.po.queue_id, "k": int(k), "foreground": bool(foreground)}
        if self.last_position is not None:
            payload["pos"] = round(self.last_position, 3)
        if self._replies and self._send is not None:
            self._send(payload, lambda reply: self._on_want_reply(reply, generation))
        else:
            self.send(payload)

    def _on_want_reply(self, reply, generation: int | None = None):
        """(queue sender thread) The user paused the item / pause_all, or it failed: the waiting status says so; any
        other reply (ok, no queue app) clears it. Old replies cannot change the current request's hold state."""
        if generation is not None and generation != self._want_generation:
            return
        err = reply.get("error") if isinstance(reply, dict) and reply.get("ok") is False else None
        hold = err if err in HOLD_STAGES else None
        if hold != self.hold:
            self.log(f"progressive: queue item {self.po.queue_id} " + (f"is {hold}" if hold else "is being processed"))
        self.hold = hold

    def start(self, pos: float) -> Feeder:
        """(Re)start the stream at source time pos (a new url for mpv)."""
        self._candidate_mode = False
        self._invalidate_want_replies()
        self._retire_feeder()
        self.target = float(pos)
        self.last_position = self.target
        self.feeder = Feeder(self.watcher, self.po.source_url or self.po.path, pos, self.po.has_audio,
                             self.log).start()
        self.watcher.poll()
        self.want(Feeder._first_k(self.watcher.state, pos), foreground=True)
        return self.feeder

    def start_source(self, pos: float) -> str:
        """Retire the candidate and return the original input URL for an immediate source load."""
        source = self.po.source_url or self.po.path
        if self.po.path.lower().endswith(".rcache") and not self.po.source_url:
            raise ValueError("原片無法播放：範圍快取標記（.rcache）缺少本機來源網址。")
        if self.po.source_url and not is_local_runtime_url(self.po.source_url):
            raise ValueError("原片無法播放：來源代理網址無效。")
        self._candidate_mode = False
        self._invalidate_want_replies()
        self._retire_feeder()
        self.target = float(pos)
        self.last_position = self.target
        self.hold = None
        self.watcher.poll()
        self.want(Feeder._first_k(self.watcher.state, pos), foreground=True)
        return source

    def prepare_restore(self, pos: float) -> RestoreCandidate:
        """Build a finite seekable clip for the current segment without sending a want or moving playback position."""
        self._candidate_mode = True
        self._invalidate_want_replies()
        self._retire_feeder()
        candidate = RestoreCandidate(self.watcher, self.po.source_url or self.po.path, pos, self.po.has_audio,
                                     self.po.work_dir, self.log).start()
        self.feeder = candidate
        return candidate

    def _invalidate_want_replies(self):
        self._want_generation += 1

    def _retire_feeder(self):
        if self.feeder is not None:
            feeder, self.feeder = self.feeder, None
            feeder.retire()

    def _current_manifest(self) -> ManifestState:
        self.watcher.poll()
        return self.watcher.state

    def segment_available(self, pos: float) -> bool:
        """True only when the manifest lists a segment covering pos and that segment file currently exists."""
        if not isinstance(pos, (int, float)) or isinstance(pos, bool) or not math.isfinite(pos):
            return False
        state = self._current_manifest()
        k = state.play.segment_at(float(pos))
        seg = state.segments.get(k)
        if seg is None:
            return False
        start, end = state.play.seg_start(k), state.play.seg_end(k, seg.frames)
        return start <= float(pos) < end and os.path.isfile(self.watcher.segment_path(seg))

    def restore_range(self, pos: float, min_ahead: float = 8.0) -> tuple[float, float] | None:
        """Return the contiguous, fully restored range from pos without rescanning the whole manifest."""
        if not isinstance(pos, (int, float)) or isinstance(pos, bool) or not math.isfinite(pos):
            return None
        if not isinstance(min_ahead, (int, float)) or isinstance(min_ahead, bool) or not math.isfinite(min_ahead):
            min_ahead = 8.0
        pos, min_ahead = float(pos), max(0.0, float(min_ahead))
        # ManifestWatcher polls on its own thread. In the playback tick, use its latest snapshot rather than
        # rereading a possibly large JSON document on every position update.
        state = self.watcher.state
        k = state.play.segment_at(pos)
        at_pos = state.segments.get(k)
        if at_pos is None or at_pos.partial \
                or not (state.play.seg_start(k) <= pos < state.play.seg_end(k, at_pos.frames)) \
                or not os.path.isfile(self.watcher.segment_path(at_pos)):
            return None
        a = state.play.seg_start(k)
        b = state.play.seg_end(k, at_pos.frames)
        current_k = k
        required_end = pos + min_ahead
        while b < required_end - 1e-6:
            next_k = current_k + 1
            next_seg = state.segments.get(next_k)
            if next_seg is None or next_seg.partial or not os.path.isfile(self.watcher.segment_path(next_seg)):
                break
            next_start = state.play.seg_start(next_k)
            if next_start > b + 0.002:
                break
            b = state.play.seg_end(next_k, next_seg.frames)
            current_k = next_k
        if b >= required_end - 1e-6:
            return a, b

        final_k = state.final_k()
        final_seg = state.segments.get(final_k) if final_k is not None else None
        final_end = state.play.seg_end(final_k, final_seg.frames) if final_seg is not None else None
        reaches_final_end = final_end is not None and current_k == final_k and abs(b - final_end) <= 0.002 \
            and os.path.isfile(self.watcher.segment_path(final_seg))
        one_frame = 1.0 / float(state.play.fps)
        if reaches_final_end and (state.complete or state.last_k is not None or final_seg.frames < state.play.n) \
                and b - pos >= one_frame - 1e-6:
            return a, b
        return None

    def activate_restore(self, pos: float):
        """Promote the candidate's continuation after the clip is acknowledged, without a foreground want."""
        candidate = self.feeder
        if isinstance(candidate, RestoreCandidate):
            self.feeder = candidate.continuation
        self._candidate_mode = False
        self.target = self.last_position = float(pos)

    def clock_due(self, now: float | None = None) -> bool:
        """True when clock() would send (the player reads mpv's position only then)."""
        return self._send is not None and (self._now() if now is None else now) >= self._next_clock

    def clock(self, pos: float, playing: bool, now: float | None = None, subtitle_pos: float | None = None) -> bool:
        """The viewer's position (absolute source seconds; playing = advancing: not paused / buffering) for the
        worker's interpolation policy, sent at most every CLOCK_INTERVAL s. Fire-and-forget, no reply handling."""
        if isinstance(pos, (int, float)) and not isinstance(pos, bool) and math.isfinite(pos) and pos >= 0:
            self.last_position = float(pos)
        now = self._now() if now is None else now
        if not self.clock_due(now):
            return False
        self._next_clock = now + CLOCK_INTERVAL
        payload = {"cmd": "clock", "id": self.po.queue_id, "pos": round(float(pos), 3), "playing": bool(playing)}
        if isinstance(subtitle_pos, (int, float)) and not isinstance(subtitle_pos, bool) \
                and math.isfinite(subtitle_pos) and subtitle_pos >= 0:
            payload["subtitle_pos"] = round(float(subtitle_pos), 3)
        self.send(payload)
        return True

    def stop_feeder(self):
        self._candidate_mode = False
        self._invalidate_want_replies()
        if self.feeder is not None:
            f, self.feeder = self.feeder, None
            if isinstance(f, RestoreCandidate):
                f.retire()  # candidate source reads and clip cleanup finish off the controller thread
            else:
                f.close()   # preserve the ordinary feeder's synchronous segment-file release contract

    @property
    def waiting_k(self) -> int | None:
        return self.feeder.waiting_k if self.feeder is not None else None

    def ready(self) -> bool:
        return self.feeder is not None and self.feeder.ready.is_set()

    def tick(self):
        """Controller iteration: watching heartbeat, want repeats while the stream waits for a segment."""
        now = self._now()
        if now >= self._next_watching:
            self._next_watching = now + self.watching_interval
            self.send({"cmd": "watching", "id": self.po.queue_id})
        if not self._candidate_mode and self.po.source_url and self.feeder is not None \
                and self.feeder.error is not None and self.hold != "failed":
            self.hold = "failed"
            self.log(f"progressive: cached source failed: {self.feeder.error!r}")
        k = self.waiting_k
        if not self._candidate_mode and k is not None \
                and (k != self._want[0] or now - self._want[1] >= WANT_INTERVAL):
            self.want(k)

    def ranges_text(self) -> str:
        return ranges_text(self.watcher.state.ranges())

    def processed_seconds(self) -> float:
        """Seconds of the source processed so far (all ranges, for the info panel)."""
        return sum(b - a for a, b in self.watcher.state.ranges())

    def sample(self, now: float | None = None):
        """Once a second (the player's history tick): processed seconds over time, for status()'s speed."""
        now = self._now() if now is None else now
        self._samples.append((now, self.processed_seconds()))
        while len(self._samples) > 2 and now - self._samples[0][0] > SPEED_WINDOW:
            self._samples.popleft()

    def speed(self) -> float | None:
        """Background processing speed in source seconds per second (= x realtime) over the last SPEED_WINDOW s;
        None until SPEED_MIN_SPAN s of samples (segments land in steps of a few seconds)."""
        if len(self._samples) < 2:
            return None
        (t0, p0), (t1, p1) = self._samples[0], self._samples[-1]
        return max(0.0, p1 - p0) / (t1 - t0) if t1 - t0 >= SPEED_MIN_SPAN else None

    def status(self, pos) -> dict:
        """The info panel's view of the background processing (info_panel "progressive" key): source / output frame
        rate, processed seconds and total, speed (x realtime, None = not measured yet), ahead = processed seconds
        right after the playback position (0: the position is not processed yet), done = every part processed."""
        st = self.watcher.state
        play, ranges = st.play, st.ranges()
        total = self.po.duration
        done_s = sum(b - a for a, b in ranges)
        ahead = None
        if isinstance(pos, (int, float)):
            ahead = next((b - pos for a, b in ranges if a - 0.05 <= pos < b), 0.0)
        return {"src_fps": float(play.fps), "out_fps": float(play.fps) * play.rate_mult, "processed": done_s,
                "total": total, "speed": self.speed(), "ahead": ahead,
                "done": st.complete or (total > 0 and done_s >= total - 0.05)}

    def close(self):
        self.stop_feeder()
        self.watcher.close()
