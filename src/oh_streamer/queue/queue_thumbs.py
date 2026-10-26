"""
Film-strip thumbnails for the queue cards (runs in the queue GUI process; no torch, no lada).

Every item gets FRAME_COUNT frames evenly spaced over the video (frame i at (i + 0.5) / FRAME_COUNT * duration), stored as
small JPEGs in items/<id>/thumbs/:

    f_XX_s.jpg   the SOURCE frame (mosaic), grabbed from the source file / a direct media URL as soon as one is available
    f_XX_r.jpg   the RESTORED frame, grabbed from a finished worker segment (work/segments/seg_*.mkv) or, for a done item,
                 from output.mkv. Always the PROCESSED track: the first video stream (segments and outputs list
                 oh:processed first; compare tracks oh:original / oh:restored come after it), selected with
                 `-map 0:v:0` -- never ffmpeg's default stream choice. Times, not frame indexes (output may be 2x fps).
    meta.json    {"duration": s, "rkey": sha1 of the worker manifest key the restored frames were grabbed under}

The state is derived from the files (scan_thumbs_dir) and kept in memory afterwards. One daemon thread does all the work
with a tiny priority order: (0) restored frames of the item being processed, (1) source frames of items in queue order
(+ restored frames of other items that have worker segments), (2) done-item backfill from output.mkv. ffmpeg runs below
normal priority without a window, never under the service lock, with a per-frame timeout; a failed grab is retried later
with a back-off (and from another source, e.g. the local file once the download is complete).

The service only calls the thin API: thumbs_for / processed_ranges / worker_fraction (state push), download_fraction (a
pure function), forget (item removal), drop_restored (cache cleared), start / stop / poke.
"""

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional

FRAME_COUNT = 12
THUMB_WIDTH = 320
JPEG_QUALITY = 5
FRAME_TIMEOUT = 20.0            # seconds per ffmpeg grab (remote seeking can be slow)
PROBE_TIMEOUT = 30.0
IDLE_WAIT = 3.0
BACKFILL_GAP = 0.25             # pause between two done-item backfill grabs (keeps the CPU free for the worker)
RETRY_BASE = 10.0               # failed grab: next try after RETRY_BASE * 2**fails seconds (max RETRY_MAX)
RETRY_MAX = 300.0
REMOTE_MAX_FAILS = 2            # direct-URL frames of an item are given up (until a local source exists) after this many
THUMBS_DIR = "thumbs"
META_FILE = "meta.json"

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

_FRAME_RE = re.compile(r"^f_(\d{2})_([sr])\.jpg$")
_URL_RE = re.compile(r"^https?://", re.I)
_MEDIA_EXT = (".mp4", ".mkv", ".webm", ".ts", ".mov", ".flv")


def _noop_log(msg: str) -> None:
    print(f"[queue_thumbs] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------------
# pure helpers
# ------------------------------------------------------------------------------------------------------------

def frame_time(i: int, duration: float, count: int = FRAME_COUNT) -> float:
    """Media time of thumbnail i: the middle of the i-th of `count` equal slices."""
    return (i + 0.5) / count * float(duration)


def segment_for_time(t: float, fps, n: int) -> tuple[int, float]:
    """(segment index k, offset in seconds inside that segment file) of video time t. Segment k holds source frames
    k*n .. k*n+n-1 and its file starts at 0, so the offset is t minus the segment's start time."""
    fps = Fraction(fps)
    frame = int(Fraction(max(0.0, float(t))) * fps)         # floor
    k = frame // max(1, int(n))
    return k, max(0.0, float(t) - float(Fraction(k * int(n)) / fps))


def processed_track_args() -> list[str]:
    """ffmpeg output selection for restored frames: the PROCESSED track = first video stream of a segment / output."""
    return ["-map", "0:v:0"]


def grab_cmd(src: str, seek: float, out: str, *, processed_track: bool = False, http_args=(), playlist: bool = False,
             remote: bool = False) -> list[str]:
    """One-frame JPEG grab by input seeking: ffmpeg -ss t -i src -frames:v 1 -vf scale=320:-2 -q:v 5 out."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    cmd += list(http_args)
    if remote:
        cmd += ["-rw_timeout", str(int(FRAME_TIMEOUT * 1_000_000))]
    if playlist:
        cmd += ["-allowed_extensions", "ALL"]
    cmd += ["-ss", f"{max(0.0, seek):.3f}", "-i", src]
    if processed_track:
        cmd += processed_track_args()
    cmd += ["-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2", "-q:v", str(JPEG_QUALITY), "-update", "1", out]
    return cmd


def merge_ranges(ranges, gap: float = 1e-6) -> list[list[float]]:
    """Sorted [a, b] pairs merged where they touch (within `gap`); empty / reversed pairs dropped; clamped to 0..1."""
    clean = []
    for a, b in ranges:
        a, b = max(0.0, min(1.0, float(a))), max(0.0, min(1.0, float(b)))
        if b > a:
            clean.append([a, b])
    clean.sort()
    merged: list[list[float]] = []
    for a, b in clean:
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[round(a, 5), round(b, 5)] for a, b in merged]


def segment_ranges(segments, n: int, total_frames: float) -> list[list[float]]:
    """Fractions of the video covered by finished segments. `segments`: {k: frames} or [(k, frames)] (partial segments
    count: their frames are restored, only the 2x in-betweens are copies). total_frames: source frames of the video."""
    if not total_frames or total_frames <= 0 or not n:
        return []
    pairs = segments.items() if isinstance(segments, dict) else segments
    return merge_ranges(((k * n) / total_frames, (k * n + frames) / total_frames) for k, frames in pairs)


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def download_fraction(it) -> Optional[float]:
    """0..1 download progress of an item's source, 1.0 for local files / complete sources, None when unknown.
    HLS / DASH: fragment_index / fragment_count; otherwise downloaded_bytes / total_bytes."""
    get = (lambda name, default=None: it.get(name, default)) if isinstance(it, dict) else (
        lambda name, default=None: getattr(it, name, default))
    state, status = get("state"), get("source_status") or ""
    if state == "done" or status == "complete":
        return 1.0
    target = str(get("target") or "")
    if get("kind") == "file" or not _URL_RE.match(target):
        return 1.0
    if get("source_path") and status == "":          # the downloader hands the path over only for a finished download
        return 1.0
    dp = get("download_progress") or {}
    if not isinstance(dp, dict):
        return None
    fc, fi = _num(dp.get("fragment_count")), _num(dp.get("fragment_index"))
    if fc and fc > 0 and fi is not None:
        return _clamp01(fi / fc)
    total, done = _num(dp.get("total_bytes")), _num(dp.get("downloaded_bytes"))
    if total and total > 0 and done is not None:
        return _clamp01(done / total)
    seg_total, seg_ready = _num(dp.get("segment_count")), _num(dp.get("ready_segments"))
    if seg_total and seg_total > 0 and seg_ready is not None:
        return _clamp01(seg_ready / seg_total)
    return None


def jpeg_size(path) -> Optional[tuple[int, int]]:
    """(width, height) from a JPEG's SOF marker (header bytes only), None if it is not a readable JPEG."""
    try:
        with open(path, "rb") as f:
            data = f.read(65536)
    except OSError:
        return None
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return (w, h) if w > 0 and h > 0 else None
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def _frame_url(path: Path, ver: int) -> str:
    return f"{path.as_uri()}?v={ver}"


def frame_data_uri(store_root: Path, url: str) -> str | None:
    """The image behind a frame `url` (as built by _frame_url) as a data: URI, for the page.

    pywebview serves the UI from its local HTTP server, so the page cannot load file:// images itself. Only files
    named like a frame inside <store_root>/items/<id>/thumbs/ are read; anything else gives None.
    """
    from urllib.parse import urlsplit
    from urllib.request import url2pathname
    import base64
    try:
        parts = urlsplit(str(url))
        if parts.scheme != "file":
            return None
        path = Path(url2pathname(parts.path)).resolve()
        items = (Path(store_root) / "items").resolve()
        if (not _FRAME_RE.match(path.name) or path.parent.name != "thumbs"
                or path.parent.parent.parent != items):
            return None
        data = path.read_bytes()
    except (OSError, ValueError, RuntimeError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def scan_thumbs_dir(tdir) -> tuple[dict, dict]:
    """(meta, frames) of an items/<id>/thumbs directory. meta: {"duration", "rkey"} (missing -> None); frames:
    {(i, "s"|"r"): (ver, w, h, url)} for every readable f_XX_[sr].jpg. Missing directory -> ({...None}, {})."""
    tdir = Path(tdir)
    meta = {"duration": None, "rkey": None}
    try:
        raw = json.loads((tdir / META_FILE).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            d = _num(raw.get("duration"))
            meta["duration"] = d if d and d > 0 else None
            meta["rkey"] = raw.get("rkey") if isinstance(raw.get("rkey"), str) else None
    except (OSError, ValueError):
        pass
    frames: dict = {}
    try:
        entries = list(os.scandir(tdir))
    except OSError:
        return meta, frames
    for e in entries:
        m = _FRAME_RE.match(e.name)
        if not m or int(m.group(1)) >= FRAME_COUNT:
            continue
        try:
            st = e.stat()
        except OSError:
            continue
        size = jpeg_size(e.path) if st.st_size > 0 else None
        if size is None:
            continue
        ver = st.st_mtime_ns // 1_000_000
        frames[(int(m.group(1)), m.group(2))] = (ver, size[0], size[1], _frame_url(Path(e.path), ver))
    return meta, frames


def build_thumbs_payload(duration: Optional[float], frames: dict) -> Optional[dict]:
    """The UI's `thumbs` value: None or {"duration", "frames": [{"at", "url", "restored", "w", "h"} ...]} in time order.
    A restored frame wins over the source frame of the same slot; its entry then also carries "src_url" (the source
    frame, when it exists) so the UI can show the mosaic underneath."""
    if not duration or duration <= 0 or not frames:
        return None
    out = []
    for i in range(FRAME_COUNT):
        r, s = frames.get((i, "r")), frames.get((i, "s"))
        use = r or s
        if use is None:
            continue
        entry = {"at": round(frame_time(i, duration), 3), "url": use[3], "restored": r is not None,
                 "w": use[1], "h": use[2]}
        if r is not None and s is not None:
            entry["src_url"] = s[3]
        out.append(entry)
    return {"duration": float(duration), "frames": out} if out else None


# ------------------------------------------------------------------------------------------------------------
# manifest (work/segments.json) as the thumbs need it
# ------------------------------------------------------------------------------------------------------------

@dataclass
class ManifestInfo:
    fps: Fraction
    n: int                               # source frames per segment
    segments: dict                       # k -> (file name, frames, partial)
    total_frames: Optional[int]
    key_hash: str


def parse_manifest(doc) -> Optional[ManifestInfo]:
    """ManifestInfo of a decoded segments.json (version 1 or 2), None when unusable."""
    if not isinstance(doc, dict):
        return None
    key = doc.get("key") if isinstance(doc.get("key"), dict) else {}
    play = doc.get("play") if isinstance(doc.get("play"), dict) else {}
    try:
        fps = Fraction(str(play.get("fps") or key.get("fps")))
        n = int(play.get("n") or doc.get("n") or key.get("n"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if fps <= 0 or n <= 0:
        return None
    segs: dict = {}
    for s in doc.get("segments") or []:
        try:
            segs[int(s["k"])] = (str(s["file"]), int(s["frames"]), s.get("partial") is True)
        except (KeyError, TypeError, ValueError):
            continue
    tf = doc.get("total_frames")
    total = tf if isinstance(tf, int) and not isinstance(tf, bool) and tf > 0 else None
    key_hash = hashlib.sha1(json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
                            .encode("utf-8")).hexdigest()
    return ManifestInfo(fps, n, segs, total, key_hash)


# ------------------------------------------------------------------------------------------------------------
# the manager
# ------------------------------------------------------------------------------------------------------------

class _ItemThumbs:
    __slots__ = ("duration", "frames", "rkey", "loaded", "epoch", "batch")

    def __init__(self):
        self.duration: Optional[float] = None
        self.frames: dict = {}               # (i, kind) -> (ver, w, h, url)
        self.rkey: Optional[str] = None
        self.loaded = False
        self.epoch = 0                       # bumped when restored frames are dropped: older in-flight grabs are void
        self.batch: dict = {}                # kind -> [count, seconds] of the grabs since the last summary log line


@dataclass
class _Job:
    kind: str                # "probe" | "grab"
    item_id: str
    i: int = -1
    out: str = ""            # "s" | "r"
    t: float = 0.0
    src: str = ""
    seek: float = 0.0
    srckind: str = ""        # local | hls | remote | seg | out
    http_args: tuple = ()
    processed: bool = False
    playlist: bool = False
    rkey: Optional[str] = None
    epoch: int = 0
    fail_key: tuple = ()


def _default_probe(src: str, http_args=(), playlist: bool = False) -> Optional[float]:
    """Duration (s) of a media file / URL by ffprobe, None on failure."""
    cmd = ["ffprobe", "-v", "error", *http_args]
    if playlist:
        cmd += ["-allowed_extensions", "ALL"]
    cmd += ["-show_entries", "format=duration", "-of", "json", src]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                           creationflags=NO_WINDOW | BELOW_NORMAL)
        if p.returncode != 0:
            return None
        d = _num(float((json.loads(p.stdout).get("format") or {}).get("duration") or 0))
        return d if d and d > 0 else None
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None


class ThumbManager:
    """Owns the thumbs/ directories of the queue items and the single grabbing thread. `store` is a QueueStore (items(),
    item_dir()); `enabled` is the show_thumbnails setting; `hls_duration(item)` an optional duration hint for a
    progressive HLS source; `runner(cmd, timeout, register)` / `probe(src, http_args, playlist)` are injectable for tests
    (runner returns (returncode, stderr_text) and calls register(proc_or_None) around the process)."""

    def __init__(self, store, *, log: Callable[[str], None] = _noop_log, on_change: Optional[Callable[[], None]] = None,
                 enabled: Callable[[], bool] = lambda: True, hls_duration: Optional[Callable] = None,
                 runner: Optional[Callable] = None, probe: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.store = store
        self._log = log
        self._on_change = on_change or (lambda: None)
        self._enabled = enabled
        self._hls_duration = hls_duration
        self._runner = runner or self._popen_runner
        self._probe = probe or _default_probe
        self._clock = clock
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._mem: dict[str, _ItemThumbs] = {}
        self._manifests: dict[str, tuple] = {}        # item id -> ((mtime_ns, size), ManifestInfo | None)
        self._fails: dict[tuple, tuple] = {}          # fail key -> (count, retry-at)
        self._removed: set = set()
        self._held: dict[str, int] = {}              # item id -> hold depth (hold(): no ffmpeg on the item's files)
        self._inflight: Optional[tuple] = None        # (item id, Popen or None)
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="queue-thumbs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        with self._cv:
            if self._inflight and self._inflight[1] is not None:
                self._kill(self._inflight[1])
        if self._thread:
            self._thread.join(timeout=5)

    def poke(self) -> None:
        self._wake.set()

    # -- state API (state push) -------------------------------------------------------------------------------

    def _th(self, item_id: str) -> _ItemThumbs:
        with self._lock:
            th = self._mem.get(item_id)
            if th is None:
                th = self._mem[item_id] = _ItemThumbs()
            if not th.loaded:
                th.loaded = True
                meta, frames = scan_thumbs_dir(self._tdir(item_id))
                th.duration, th.rkey, th.frames = meta["duration"], meta["rkey"], frames
            return th

    def _tdir(self, item_id: str) -> Path:
        return self.store.item_dir(item_id) / THUMBS_DIR

    def thumbs_for(self, it) -> Optional[dict]:
        if not self._enabled():
            return None
        with self._lock:
            th = self._th(it.id)
            return build_thumbs_payload(th.duration, th.frames)

    def manifest(self, it) -> Optional[ManifestInfo]:
        """Cached parse of the item's work/segments.json (re-read only when its mtime / size change)."""
        path = self.store.item_dir(it.id) / "work" / "segments.json"
        try:
            st = os.stat(path)
        except OSError:
            with self._lock:
                self._manifests.pop(it.id, None)
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._manifests.get(it.id)
        if hit and hit[0] == stamp:
            return hit[1]
        try:
            info = parse_manifest(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return hit[1] if hit else None         # being replaced right now: keep the previous parse, retry next push
        with self._lock:
            self._manifests[it.id] = (stamp, info)
        return info

    def processed_ranges(self, it) -> list:
        """[[a, b], ...] fractions of the video time that have restored segments ([[0, 1]] when done)."""
        if it.state == "done":
            return [[0.0, 1.0]]
        m = self.manifest(it)
        if m is None or not m.segments:
            return []
        total = m.total_frames or _num((it.progress or {}).get("frames"))
        if not total:
            with self._lock:
                duration = self._th(it.id).duration
            total = duration * float(m.fps) if duration else None
        return segment_ranges({k: v[1] for k, v in m.segments.items()}, m.n, total or 0)

    def worker_fraction(self, it) -> Optional[float]:
        """Video-time fraction where the worker is (start of the segment being produced), None unless processing."""
        if it.state != "processing":
            return None
        prog = it.progress or {}
        if prog.get("phase") == "finalize":
            return None
        frames = _num(prog.get("frames"))
        k = prog.get("k")
        if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
            m = self.manifest(it)
            total = frames or (m.total_frames if m else None)
            if m is not None and total:
                return round(_clamp01(k * m.n / total), 5)
        done = _num(prog.get("frame"))
        if frames and done is not None:
            return round(_clamp01(done / frames), 5)
        return None

    # -- cache invalidation / removal -------------------------------------------------------------------------

    def drop_restored(self, item_id: str) -> None:
        """The item's processing cache is gone: restored frames fall back to the source frames."""
        with self._lock:
            th = self._th(item_id)
            th.epoch += 1
            th.rkey = None
            th.frames = {k: v for k, v in th.frames.items() if k[1] != "r"}
            self._fails = {k: v for k, v in self._fails.items() if not (k[0] == item_id and k[2] == "r")}
        tdir = self._tdir(item_id)
        try:
            for e in os.scandir(tdir):
                if re.match(r"^f_\d{2}_r(\.tmp)?\.jpg$", e.name):
                    try:
                        os.remove(e.path)
                    except OSError:
                        pass
        except OSError:
            pass
        self._save_meta(item_id)
        self._on_change()
        self._wake.set()

    def forget(self, item_id: str) -> None:
        """The item is being removed: no new grab for it, kill / wait out the one in flight (a process holding a file
        would block deleting the item folder on Windows)."""
        with self._cv:
            self._removed.add(item_id)
            if self._inflight and self._inflight[0] == item_id and self._inflight[1] is not None:
                self._kill(self._inflight[1])
            deadline = self._clock() + 5.0
            while self._inflight and self._inflight[0] == item_id and self._clock() < deadline:
                self._cv.wait(0.1)
            self._mem.pop(item_id, None)
            self._manifests.pop(item_id, None)

    @contextlib.contextmanager
    def hold(self, item_id: str):
        """No ffmpeg grab on the item's files while the block runs (the subtitle mux must take an exclusive-ish lock on
        output.mp4, which a reading ffmpeg breaks on Windows). Kills / waits out the one in flight like forget(), but
        the item stays usable: grabbing resumes when the (re-entrant) hold is released."""
        with self._cv:
            self._held[item_id] = self._held.get(item_id, 0) + 1
            if self._inflight and self._inflight[0] == item_id and self._inflight[1] is not None:
                self._kill(self._inflight[1])
            deadline = self._clock() + 5.0
            while self._inflight and self._inflight[0] == item_id and self._clock() < deadline:
                self._cv.wait(0.1)
        try:
            yield
        finally:
            with self._cv:
                n = self._held.get(item_id, 0) - 1
                if n > 0:
                    self._held[item_id] = n
                else:
                    self._held.pop(item_id, None)
            self._wake.set()

    @staticmethod
    def _kill(proc) -> None:
        try:
            proc.kill()                      # our own ffmpeg only (a Popen we started)
        except OSError:
            pass

    # -- job selection ----------------------------------------------------------------------------------------

    def _blocked(self, key: tuple, now: float) -> bool:
        f = self._fails.get(key)
        return f is not None and now < f[1]

    def _note_fail(self, key: tuple, now: float) -> int:
        count = (self._fails.get(key, (0, 0.0))[0]) + 1
        self._fails[key] = (count, now + min(RETRY_MAX, RETRY_BASE * 2 ** count))
        return count

    def _source_ref(self, it) -> Optional[tuple]:
        """(src, srckind, http_args, playlist) of the best SOURCE for source frames, None if there is none yet."""
        if (it.source_path and os.path.isfile(it.source_path) and it.source_status in ("", "complete")
                and not it.source_path.lower().endswith(".rcache")):      # a range-cache marker is not media
            return it.source_path, "local", (), it.source_path.lower().endswith(".m3u8")
        if it.kind == "file" and it.target and os.path.isfile(it.target):
            return it.target, "local", (), False
        if it.source_path and it.source_path.lower().endswith(".m3u8") and os.path.isfile(it.source_path) \
                and it.source_status == "downloading":
            return it.source_path, "hls", (), True
        if it.kind == "url" and _URL_RE.match(it.target or "") and it.target.split("?", 1)[0].lower().endswith(_MEDIA_EXT):
            if self._fails.get((it.id, -1, "remote"), (0, 0))[0] < REMOTE_MAX_FAILS:
                from oh_streamer.sources import source_resolve
                return it.target, "remote", tuple(source_resolve.FFmpegHttpOptions(it.headers or {}).cli_args()), False
        return None

    def _next_job(self) -> Optional[_Job]:
        items = self.store.items()
        now = self._clock()
        live = {it.id for it in items}
        with self._lock:
            for gone in [i for i in self._mem if i not in live]:
                self._mem.pop(gone, None)
                self._manifests.pop(gone, None)
        with self._lock:
            held = set(self._held)
        items = [it for it in items if it.id not in held]
        order = [it for it in items if it.state == "processing"]
        for it in order:
            if it.id in self._removed:
                continue
            job = self._probe_job(it, now) or self._restore_job(it, now)
            if job:
                return job
        for it in items:
            if it.id in self._removed:
                continue
            job = (self._probe_job(it, now) or self._source_job(it, now)
                   or (self._restore_job(it, now) if it.state not in ("done", "processing") else None))
            if job:
                return job
        for it in items:
            if it.state == "done" and it.id not in self._removed:
                job = self._done_job(it, now)
                if job:
                    return job
        return None

    def _probe_job(self, it, now: float) -> Optional[_Job]:
        with self._lock:
            th = self._th(it.id)
            if th.duration:
                return None
        hint = self._hls_duration(it) if self._hls_duration else None
        if hint and hint > 0:
            with self._lock:
                th.duration = float(hint)
            self._save_meta(it.id)
            return None
        ref = self._source_ref(it)
        if ref is not None:
            src, srckind, http, playlist = ref
        elif it.state == "done" and it.output_path and os.path.isfile(it.output_path):
            src, srckind, http, playlist = it.output_path, "out", (), False
        else:
            return None
        key = (it.id, -1, "p", srckind)
        with self._lock:
            if self._blocked(key, now):
                return None
        return _Job("probe", it.id, src=src, srckind=srckind, http_args=tuple(http), playlist=playlist, fail_key=key)

    def _source_job(self, it, now: float) -> Optional[_Job]:
        with self._lock:
            th = self._th(it.id)
            duration, have = th.duration, set(th.frames)
        if not duration:
            return None
        ref = self._source_ref(it)
        if ref is None:
            return None
        src, srckind, http, playlist = ref
        for i in range(FRAME_COUNT):
            if (i, "s") in have:
                continue
            key = (it.id, i, "s", srckind)
            with self._lock:
                if self._blocked(key, now):
                    continue
            t = frame_time(i, duration)
            return _Job("grab", it.id, i, "s", t, src, t, srckind, http_args=tuple(http), playlist=playlist,
                        epoch=th.epoch, fail_key=key)
        return None

    def _restore_job(self, it, now: float) -> Optional[_Job]:
        m = self.manifest(it)
        if m is None:
            return None
        with self._lock:
            stale = bool(self._th(it.id).rkey) and self._th(it.id).rkey != m.key_hash
        if stale:                                        # the worker restarted from scratch under a new job key
            self.drop_restored(it.id)
        with self._lock:
            th = self._th(it.id)
            duration, have, epoch = th.duration, set(th.frames), th.epoch
        if not duration or not m.segments:
            return None
        seg_dir = self.store.item_dir(it.id) / "work" / "segments"
        for i in range(FRAME_COUNT):
            if (i, "r") in have:
                continue
            t = frame_time(i, duration)
            k, off = segment_for_time(t, m.fps, m.n)
            seg = m.segments.get(k)
            if seg is None:
                continue
            key = (it.id, i, "r", "seg")
            with self._lock:
                if self._blocked(key, now):
                    continue
            path = seg_dir / seg[0]
            try:
                if os.path.getsize(path) <= 0:
                    continue
            except OSError:
                continue
            off = min(off, max(0.0, seg[1] / float(m.fps) - 0.5 / float(m.fps)))
            return _Job("grab", it.id, i, "r", t, str(path), off, "seg", processed=True, rkey=m.key_hash, epoch=epoch,
                        fail_key=key)
        return None

    def _done_job(self, it, now: float) -> Optional[_Job]:
        if not it.output_path or not os.path.isfile(it.output_path):
            return None
        with self._lock:
            th = self._th(it.id)
            duration, have, epoch = th.duration, set(th.frames), th.epoch
        if not duration:
            return None
        for i in range(FRAME_COUNT):
            if (i, "r") in have:
                continue
            key = (it.id, i, "r", "out")
            with self._lock:
                if self._blocked(key, now):
                    continue
            t = frame_time(i, duration)
            return _Job("grab", it.id, i, "r", t, it.output_path, t, "out", processed=True, epoch=epoch, fail_key=key)
        return None

    # -- execution --------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._next_job() if self._enabled() else None
            except Exception as e:  # noqa: BLE001 - the thumbnail thread must survive anything
                self._log(f"thumbs: planning failed: {e!r}")
                job = None
            if job is None:
                self._wake.wait(IDLE_WAIT)
                self._wake.clear()
                self._stopping.wait(0.2)
                continue
            try:
                self._execute(job)
            except Exception as e:  # noqa: BLE001
                self._log(f"thumbs: job failed unexpectedly: {e!r}")
                with self._lock:
                    self._note_fail(job.fail_key, self._clock())
            if job.srckind == "out" and job.kind == "grab":
                self._stopping.wait(BACKFILL_GAP)

    def _begin(self, job: _Job) -> bool:
        with self._cv:
            if job.item_id in self._removed or job.item_id in self._held or self._stopping.is_set():
                return False
            self._inflight = (job.item_id, None)
            return True

    def _end(self) -> None:
        with self._cv:
            self._inflight = None
            self._cv.notify_all()

    def _register(self, proc) -> None:
        with self._cv:
            if self._inflight:
                self._inflight = (self._inflight[0], proc)
                if proc is not None and (self._inflight[0] in self._removed or self._inflight[0] in self._held):
                    self._kill(proc)

    def _popen_runner(self, cmd, timeout, register):
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                creationflags=NO_WINDOW | BELOW_NORMAL)
        register(proc)
        try:
            _out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            proc.communicate()
            return -9, "timeout"
        finally:
            register(None)
        return proc.returncode, (err or b"").decode("utf-8", "replace")[-300:]

    def _execute(self, job: _Job) -> None:
        if not self._begin(job):
            return
        try:
            if job.kind == "probe":
                self._execute_probe(job)
            else:
                self._execute_grab(job)
        finally:
            self._end()

    def _execute_probe(self, job: _Job) -> None:
        d = self._probe(job.src, job.http_args, job.playlist)
        with self._lock:
            if job.item_id in self._removed:
                return
            if d:
                self._th(job.item_id).duration = float(d)
            else:
                self._note_fail(job.fail_key, self._clock())
                if job.srckind == "remote":
                    self._note_fail((job.item_id, -1, "remote"), self._clock())
        if d:
            self._save_meta(job.item_id)
            self._on_change()

    def _execute_grab(self, job: _Job) -> None:
        if self.store.get(job.item_id) is None:      # removed meanwhile: never recreate its folder
            return
        tdir = self._tdir(job.item_id)
        try:
            tdir.mkdir(parents=True, exist_ok=True)  # forget() runs before a removal, so this cannot race it
        except OSError:
            with self._lock:
                self._note_fail(job.fail_key, self._clock())
            return
        name = f"f_{job.i:02d}_{job.out}"
        tmp, final = tdir / f"{name}.tmp.jpg", tdir / f"{name}.jpg"
        cmd = grab_cmd(job.src, job.seek, str(tmp), processed_track=job.out == "r", http_args=job.http_args,
                       playlist=job.playlist, remote=job.srckind == "remote")
        t0 = self._clock()
        rc, err = self._runner(cmd, FRAME_TIMEOUT, self._register)
        dt = self._clock() - t0
        ok = rc == 0 and tmp.is_file() and tmp.stat().st_size > 0
        if ok:
            ok = self._install(tmp, final)
        with self._lock:
            th = self._mem.get(job.item_id)
            void = job.item_id in self._removed or th is None or (job.out == "r" and th.epoch != job.epoch)
        if void:
            self._discard(tmp, final if ok else None)
            return
        with self._lock:
            held = job.item_id in self._held
        if held and not ok:                          # killed by hold(): not a failure, just retried afterwards
            self._discard(tmp, None)
            return
        if not ok:
            self._discard(tmp, None)
            with self._lock:
                count = self._note_fail(job.fail_key, self._clock())
                if job.srckind == "remote":
                    self._note_fail((job.item_id, -1, "remote"), self._clock())
            self._log(f"thumbs {job.item_id} {name} {job.srckind}: failed after {dt:.2f}s (try {count}): "
                      f"{(err or '').strip()[-120:]}")
            return
        size = jpeg_size(final)
        if size is None:
            self._discard(None, final)
            with self._lock:
                self._note_fail(job.fail_key, self._clock())
            return
        with self._lock:
            prev = th.frames.get((job.i, job.out))
            ver = max(final.stat().st_mtime_ns // 1_000_000, (prev[0] + 1) if prev else 0)
            th.frames[(job.i, job.out)] = (ver, size[0], size[1], _frame_url(final, ver))
            if job.rkey and not th.rkey:
                th.rkey = job.rkey
            count_s = th.batch.setdefault(job.srckind, [0, 0.0])
            count_s[0] += 1
            count_s[1] += dt
            summary = None
            if job.srckind not in ("remote", "hls"):
                if count_s[0] >= FRAME_COUNT:
                    summary = f"thumbs {job.item_id}: {count_s[0]} {job.srckind} frames, {count_s[1]:.2f}s total"
                    th.batch[job.srckind] = [0, 0.0]
            else:
                summary = f"thumbs {job.item_id} {name} {job.srckind}: {dt:.2f}s"
        if summary:
            self._log(summary)
        if job.rkey:
            self._save_meta(job.item_id)
        self._on_change()

    @staticmethod
    def _install(tmp: Path, final: Path) -> bool:
        for attempt in range(6):
            try:
                os.replace(tmp, final)
                return True
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
            except OSError:
                return False
        return False

    @staticmethod
    def _discard(tmp: Optional[Path], final: Optional[Path]) -> None:
        for p in (tmp, final):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _save_meta(self, item_id: str) -> None:
        with self._lock:
            th = self._mem.get(item_id)
            if th is None or item_id in self._removed:
                return
            doc = {"duration": th.duration, "rkey": th.rkey}
        tdir = self._tdir(item_id)
        try:
            if self.store.get(item_id) is None:
                return
            tdir.mkdir(parents=True, exist_ok=True)
            tmp = tdir / (META_FILE + ".tmp")
            tmp.write_text(json.dumps(doc), encoding="utf-8")
            os.replace(tmp, tdir / META_FILE)
        except OSError:
            pass
