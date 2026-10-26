"""Player-side bridge to the background processing queue (queue_ipc.py)."""

import json
import math
import os
import queue
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

from oh_streamer.player import player_settings as PS
from oh_streamer.player import progressive_stream
from oh_streamer.processing import compare_stack
from oh_streamer.queue import queue_ipc
from oh_streamer.sources.source_resolve import is_url


PROCESSED_WATCHED_FRACTION = 0.9       # a processed file counts as watched from this fraction (or at its end)
QUEUE_SETTING_KEYS = tuple(k for k in PS.KEYS if k not in PS.LIVE_KEYS) + ("device",)


class QueueLink:
    """Everything the player says to the queue app, off the controller thread: post() hands a message to a daemon
    sender thread (callbacks run there), request() is a synchronous call for threads that may wait (the Opener).
    send(payload, timeout) -> reply dict | None (None = the queue app is not running: logged once per outage)."""

    def __init__(self, send=None, name=None, log=lambda m: None):
        self._send = send or (lambda payload, timeout: queue_ipc.send_to_queue(payload, name, timeout=timeout))
        self._log, self._warned = log, False
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._thread = None
        self._lock = threading.Lock()

    def request(self, payload: dict, timeout: float = 2.0) -> dict | None:
        try:
            reply = self._send(payload, timeout)
        except Exception as e:  # noqa: BLE001 - the queue must never hurt playback
            reply = None
            self._log(f"queue: {payload.get('cmd')} failed ({e!r})")
        if reply is None:
            if not self._warned:
                self._warned = True
                self._log("queue app not running (queue messages are dropped quietly until it is)")
        else:
            self._warned = False
        return reply

    def post(self, payload: dict, callback=None) -> bool:
        """Queue a message; callback(reply | None) runs on the sender thread. False when the backlog is full."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="queue sender", daemon=True)
                self._thread.start()
        try:
            self._q.put_nowait((payload, callback))
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            payload, callback = item
            reply = self.request(payload)
            if callback is not None:
                try:
                    callback(reply)
                except Exception as e:  # noqa: BLE001
                    self._log(f"queue: callback failed ({e!r})")

    def close(self, timeout: float = 2.0):
        """Send what is pending (the final busy=false), then stop the sender; never waits longer than `timeout`."""
        with self._lock:
            t = self._thread
        if t is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            return
        t.join(timeout)


class BusyReporter:
    """player_busy messages: busy=true when restoring starts and again every `interval` s while it lasts, busy=false
    once when it stops. update() is called every controller iteration with the current state; it only sends on a
    change or when a heartbeat is due (send(busy) must not block).

    `live` (optional, set by the controller): a zero-argument callable returning the live-stream report
    (queue_ipc.py, "Live stream report") or None. It is asked at most every `live_interval` s; a report goes out as
    send(busy, live=<dict>) whether or not the player is busy, and once it disappears a last send(busy, live=None)
    tells the queue to clear it. Without `live` send(busy) is called exactly as before."""

    def __init__(self, send, clock=time.monotonic, interval: float = queue_ipc.BUSY_HEARTBEAT,
                 live_interval: float = queue_ipc.LIVE_INTERVAL):
        self.send, self.clock, self.interval = send, clock, interval
        self.live, self.live_interval = None, live_interval
        self.busy, self.next_beat = False, 0.0
        self.next_live, self.live_sent = 0.0, False

    def update(self, busy: bool):
        now = self.clock()
        beat = busy and (not self.busy or now >= self.next_beat)
        stopped = self.busy and not busy
        report, clear = None, False
        if self.live is not None and now >= self.next_live:
            self.next_live = now + self.live_interval
            report = self.live()
            clear = report is None and self.live_sent
        if beat:
            self.next_beat = now + self.interval
        if beat or stopped or report is not None or clear:
            if report is not None or clear:
                self.send(busy, live=report)
                self.live_sent = report is not None
            else:
                self.send(busy)
        self.busy = busy


class ProcessedOpen:
    """An already processed file (a queue output) to play as is, without Lada."""

    def __init__(self, path: str, title: str | None = None, queue_id: str | None = None, layers: str = "",
                 from_lookup: bool = False, derive: str = "", tags: dict | None = None, combos_dir: str | None = None):
        self.path, self.queue_id, self.layers, self.from_lookup = path, queue_id, layers, from_lookup
        self.combos_dir = combos_dir      # subtitle combos folder of a queue item (None = derived from the path / none)
        self.derive = derive              # user-data/oh/derive-1x: the 1x "restored" layer is derived from v:0
        self.tags = tags                  # lada_tags_from_probe(): the queue output's global tags (None = old file)
        self.engine = (tags or {}).get("engine") or "lada"   # the engine that made the file (no tag = lada)
        self.title = title or Path(path).name
        self.start = 0.0
        self.subtitle_error = False
        self.subtitles = []
        self.subtitle_offset = 0.0
        self.subtitle_covered_s = None
        self.subtitle_covered_ranges = None
        self.subtitle_complete = True


def mpv_playback_snapshot(get, duration=None) -> dict:
    """info_panel.build_info keys read from mpv itself, for the modes without a Session (a processed file, progressive
    playback): what plays (size, frame rate, codec, length) and where (position, cache end, dropped frames).
    get(name) -> value or None (MpvIpc.get); duration: the known source length (None = mpv's own)."""
    fd, vd = get("frame-drop-count"), get("vo-delayed-frame-count")
    paused = "緩衝中" if get("paused-for-cache") else "暫停" if get("pause") else None
    return {
        "width": get("video-params/w") or get("width"), "height": get("video-params/h") or get("height"),
        "fps": get("container-fps") or get("estimated-vf-fps"), "codec": get("video-format"),
        "duration": duration or get("duration"), "pos": get("time-pos"), "cache_time": get("demuxer-cache-time"),
        "drops": (int(fd or 0), int(vd or 0)) if fd is not None or vd is not None else None,
        "paused": paused,
    }


def _probe_titles(probe: dict) -> list[str]:
    """The known oh:* video track titles of an ffprobe result, in track order."""
    known = set(compare_stack.LAYER_TITLES.values())
    titles = [(s.get("tags") or {}).get("title") for s in probe.get("streams") or [] if s.get("codec_type") == "video"]
    return [t for t in titles if t in known]


def compare_titles_from_probe(probe: dict) -> str:
    """Comma list of the compare track titles of an ffprobe result; empty unless there are at least two."""
    keep = _probe_titles(probe)
    return ",".join(keep) if len(keep) >= 2 else ""


def lada_tags_from_probe(probe: dict) -> dict | None:
    """The global Matroska tags of a queue output, or None for absent or unusable metadata."""
    raw = (probe.get("format") or {}).get("tags") or {}
    tags = {str(k).upper(): str(v).strip() for k, v in raw.items() if str(k).upper().startswith("OH_")}
    try:
        fps = Fraction(tags["OH_SOURCE_FPS"])
        mult = int(tags["OH_RATE_MULT"])
        start = float(tags["OH_VIDEO_START"]) if "OH_VIDEO_START" in tags else 0.0
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    if fps <= 0 or mult not in (1, 2) or not math.isfinite(start):
        return None
    original = tags.get("OH_ORIGINAL", "").lower()
    return {"source_fps": fps, "rate_mult": mult, "video_start": start, "compare": tags.get("OH_COMPARE") == "1",
            "original": original if original in ("copy", "encode", "none") else None,
            "engine": tags.get("OH_ENGINE") or "lada",              # files from before the engine tags are lada's
            "engine_version": tags.get("OH_ENGINE_VERSION") or None}


def compare_setup_from_probe(probe: dict) -> tuple[str, str, dict | None]:
    """(compare_layers, derive_1x, tags) of an ffprobe result: the layers user-data/oh/compare_layers gets (the real
    tracks; one is enough when the 1x layer is derived from it), the user-data/oh/derive-1x value ("" = none) and
    the queue output's tags."""
    titles, tags = _probe_titles(probe), lada_tags_from_probe(probe)
    derive = progressive_stream.derive_1x_param(tags["source_fps"], tags["video_start"], tags["rate_mult"],
                                                tags["compare"], titles) if tags else ""
    return (",".join(titles) if len(titles) >= 2 or (titles and derive) else ""), derive, tags


def probe_compare_setup(path: str) -> tuple[str, str, dict | None]:
    """compare_setup_from_probe() of a local file; ("", "", None) on any failure (compare is optional)."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-select_streams", "v",
                            "-show_entries", "stream=codec_type:stream_tags=title:format_tags", path],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        return compare_setup_from_probe(json.loads(p.stdout)) if p.returncode == 0 else ("", "", None)
    except (OSError, ValueError, subprocess.SubprocessError):
        return "", "", None


def probe_compare_titles(path: str) -> str:
    """The compare layers of a local file (probe_compare_setup()'s first part); "" on any failure."""
    return probe_compare_setup(path)[0]


def _combos_dir(value) -> str | None:
    return os.path.abspath(value) if isinstance(value, str) and value.strip() else None


class SubtitleUpdate:
    """A validated queue -> player snapshot of the external subtitle tracks for one progressive item."""

    def __init__(self, item_id: str, subtitles: list[dict], covered_s: float, complete: bool,
                 subtitle_offset: float, covered_ranges=None):
        self.item_id, self.subtitles = item_id, subtitles
        self.covered_s, self.complete = covered_s, complete
        self.subtitle_offset = subtitle_offset
        self.covered_ranges = covered_ranges


def subtitle_update_from_message(msg: dict) -> SubtitleUpdate | None:
    """Validate a dynamic subtitle snapshot without making a malformed optional update fatal to playback.

    ``srt`` is one cumulative SRT path or a list of cumulative per-track SRT paths. Optional ``subtitles`` metadata
    is aligned with those paths by index, as it is on the initial progressive-open message.
    """
    if not isinstance(msg, dict) or msg.get("cmd") != "subtitle_update":
        return None
    raw_id = msg.get("item_id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
        return None
    raw_covered = msg.get("covered_s")
    if isinstance(raw_covered, bool) or not isinstance(raw_covered, (int, float)) \
            or not math.isfinite(raw_covered) or raw_covered < 0:
        return None
    complete = msg.get("complete")
    if not isinstance(complete, bool):
        return None
    raw_srt = msg.get("srt")
    values = [raw_srt] if isinstance(raw_srt, (str, dict)) else raw_srt
    if not isinstance(values, list) or not values or len(values) > 32:
        return None
    metadata = progressive_stream.subtitle_tracks(msg.get("subtitles"))
    tracks, seen = [], set()
    for index, value in enumerate(values):
        descriptor = value if isinstance(value, dict) else {}
        path = descriptor.get("path") if descriptor else value
        if not isinstance(path, str) or not path.strip():
            return None
        path = os.path.abspath(path.strip().strip('"'))
        if not os.path.isfile(path):
            return None
        key = os.path.normcase(path)
        if key in seen:
            continue
        meta = metadata[index] if index < len(metadata) else {}
        lang = descriptor.get("lang") if isinstance(descriptor.get("lang"), str) else meta.get("lang")
        title = descriptor.get("title") if isinstance(descriptor.get("title"), str) else meta.get("title")
        default = descriptor.get("default") if isinstance(descriptor.get("default"), bool) else meta.get("default")
        if not isinstance(lang, str):
            lang = "chi"
        if not isinstance(title, str):
            title = "繁體中文（AI）" if index == 0 else f"AI 字幕 {index + 1}"
        if not isinstance(default, bool):
            default = index == 0
        tracks.append({"path": path, "lang": lang, "title": title, "default": default})
        seen.add(key)
    if not tracks:
        return None
    raw_offset = msg.get("subtitle_offset", 0.0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, (int, float)) or not math.isfinite(raw_offset):
        return None
    ranges = progressive_stream.subtitle_covered_ranges(msg.get("covered_ranges"))
    if msg.get("covered_ranges") is not None and ranges is None:
        return None
    return SubtitleUpdate(str(raw_id).strip(), tracks, float(raw_covered), complete, float(raw_offset), ranges)


def processed_open(path: str, title: str | None, queue_id: str | None, probe=probe_compare_setup,
                   from_lookup: bool = False, combos_dir: str | None = None) -> ProcessedOpen:
    """A ProcessedOpen with what the file carries. probe(path) returns (layers, derive, tags); a plain string (the
    compare layers only: the older hook) is accepted too."""
    r = probe(path)
    layers, derive, tags = (r, "", None) if isinstance(r, str) else r
    return ProcessedOpen(path, title, queue_id, layers, from_lookup=from_lookup, derive=derive, tags=tags,
                         combos_dir=combos_dir)


def processed_from_message(msg: dict, probe_layers=probe_compare_setup) -> ProcessedOpen | None:
    """The ProcessedOpen of a single-instance message {"input": path, "processed": true, "queue_id", "title"};
    None when the file does not exist (or the message is not one)."""
    path = msg.get("input")
    if msg.get("processed") is not True or not isinstance(path, str) or not path.strip():
        return None
    path = os.path.abspath(path.strip().strip('"'))
    if not os.path.isfile(path):
        return None
    qid, title = msg.get("queue_id"), msg.get("title")
    po = processed_open(path, title if isinstance(title, str) and title.strip() else None,
                        str(qid) if qid not in (None, "") else None, probe_layers,
                        combos_dir=_combos_dir(msg.get("subtitle_combos_dir")))
    start = queue_ipc.parse_start(msg.get("start"))
    po.subtitle_error = msg.get("subtitle_error") is True
    offset = msg.get("subtitle_offset", 0.0)
    if isinstance(offset, (int, float)) and not isinstance(offset, bool) and math.isfinite(offset):
        po.subtitle_offset = float(offset)
        po.subtitles = progressive_stream.subtitle_tracks(msg.get("subtitles"))
    po.subtitle_covered_ranges = progressive_stream.subtitle_covered_ranges(msg.get("subtitle_covered_ranges"))
    covered = msg.get("subtitle_covered_s")
    if isinstance(covered, (int, float)) and not isinstance(covered, bool) and math.isfinite(covered) and covered >= 0:
        po.subtitle_covered_s = float(covered)
        complete = msg.get("subtitle_complete")
        po.subtitle_complete = complete if isinstance(complete, bool) else True
    if start is not None:
        po.start = start
    return po


def queue_lookup(link: QueueLink, target: str, quality, timeout: float = 1.0,
                 probe_layers=probe_compare_setup) -> ProcessedOpen | None:
    """Ask the queue whether `target` is processed already: a ProcessedOpen for a `done` item whose output exists."""
    reply = link.request({"cmd": "lookup", "input": target, "quality": quality}, timeout)
    item = reply.get("item") if isinstance(reply, dict) else None
    if not isinstance(item, dict) or item.get("state") != "done":
        return None
    out = item.get("output")
    if not isinstance(out, str) or not os.path.isfile(out):
        return None
    title = item.get("title") if isinstance(item.get("title"), str) else None
    return processed_open(os.path.abspath(out), title or (Path(target).name if not is_url(target) else None),
                          str(item["id"]) if item.get("id") not in (None, "") else None, probe_layers,
                          from_lookup=True, combos_dir=_combos_dir(item.get("subtitle_combos_dir")))


def processed_watched(pos, duration, eof, fraction: float = PROCESSED_WATCHED_FRACTION) -> bool:
    """Played to the end, or to `fraction` of the length."""
    if eof:
        return True
    return isinstance(pos, (int, float)) and isinstance(duration, (int, float)) and duration > 0 \
        and pos >= fraction * duration


def queue_quality(quals, current):
    """The quality a queue item should be processed at: the height of the rendition playing now when the source
    offers a choice (yt-dlp), else None (a file / a single stream)."""
    if quals and len(quals) > 1 and current and current.get("height"):
        return int(current["height"])
    return None


def build_add_payload(target: str, headers: dict, quality, title: str | None, cfg: dict, fp16_on: bool) -> dict:
    """The queue_ipc "add" message: the input as given by the user (page URL / absolute path, the user's own headers,
    not the resolved rendition URL) and the settings that change the output (not the live ones: VSR, smooth, prebuffer)."""
    settings = {k: cfg[k] for k in QUEUE_SETTING_KEYS if k in cfg}
    settings["fp16"] = bool(fp16_on)
    return {"cmd": "add", "input": target if is_url(target) else os.path.abspath(target),
            "headers": {str(k): str(v) for k, v in (headers or {}).items()}, "quality": quality, "title": title,
            "settings": settings}
