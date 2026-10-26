"""Queue downloader: fetches a queue item's source into its folder (no torch; runs inside the queue process).

    download(target, headers, quality, dest_dir, on_progress, cancel) -> DownloadResult(path, title, duration)

- A local file is returned as-is (no copy).
- A URL is resolved like the player does (external resolver from resolvers.json when one matches, otherwise yt-dlp
  directly) and downloaded with the yt-dlp Python API to <dest_dir>/source.<ext> (separate video + audio are merged
  into .mkv). `quality` follows the player's spec (height, 'best', 'worst'; None = best); `continuedl` makes a restarted
  download resume from the .part files.
- An expired URL (HTTP 403 / 404 / 410) is re-resolved once and the download resumed; a second failure raises
  DownloadFailed.
- on_progress(dict) keys: stage ('downloading' | 'finished' | 'postprocessing'), downloaded_bytes, total_bytes
  (exact or an estimate, else None), total_approx (True when total_bytes is an estimate), speed, eta,
  fragment_index / fragment_count (HLS / DASH), filename.
- Setting `cancel` aborts the download: DownloadCancelled. yt-dlp's native downloaders abort from the next progress
  hook. Downloaders without hooks (ffmpeg, used for AES-128 HLS when pycryptodomex is missing, or for some
  protocols) block in proc.wait() and can't be interrupted from a hook, so the watcher thread (below) kills the
  ffmpeg child process(es) that appeared during this download; yt-dlp then raises and download() maps it to
  DownloadCancelled because `cancel` is set.
- A _ProgressWatcher thread covers downloaders that report nothing: while no hook fired for > 2 s it polls the size of
  <dest_dir>/source.* (.part, .part-FragN, merged outputs; .ytdl excluded) every second and reports
  downloaded_bytes + smoothed speed (+ total / eta from a filesize / tbr x duration estimate, total_approx=True).
"""

import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import threading
import time
import urllib.request
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yt_dlp

from oh_streamer.sources import qualities

from oh_streamer.sources import source_resolve
from oh_streamer.sources import hls_cache
from oh_streamer.sources import range_cache
from oh_streamer.sources.http_headers import scope_for_url


EXPIRED_HTTP_CODES = (403, 404, 410)
_EXPIRED_RE = re.compile(r"HTTP Error (%s)\b" % "|".join(str(c) for c in EXPIRED_HTTP_CODES))


def log(msg):
    """Hook: the queue process replaces this with its own logger."""
    print(f"[queue_download] {msg}", file=sys.stderr, flush=True)


class DownloadCancelled(Exception):
    pass


class DownloadFailed(Exception):
    pass


class _CancelSignal(Exception):
    """Raised from yt-dlp's progress hook when the cancel event is set (private: callers see DownloadCancelled)."""


@dataclass
class DownloadResult:
    path: str
    title: str
    duration: float | None = None


class _YdlLogger:
    """Forwards yt-dlp's messages to the `log` hook (looked up at call time so it can be replaced)."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log(f"yt-dlp warning: {msg}")

    def error(self, msg):
        log(f"yt-dlp error: {msg}")


def format_selector(quality) -> str:
    """yt-dlp format string for a player-style quality spec: None / 'best' = best; N = best up to N pixels high (the
    smallest rendition when none is that small, like qualities.pick_quality); 'worst' = smallest."""
    spec = qualities.parse_quality_spec(quality)
    if spec == "worst":
        return "wv*+wa/w"
    if isinstance(spec, int):
        return f"bv*[height<={spec}]+ba/b[height<={spec}]/wv*+wa/w"
    return "bv*+ba/b"


def _is_expired_error(exc: BaseException | None) -> bool:
    """HTTP 403 / 404 / 410 anywhere in the exception chain (DownloadError text or an underlying HTTPError)."""
    seen = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if getattr(e, "status", None) in EXPIRED_HTTP_CODES or getattr(e, "code", None) in EXPIRED_HTTP_CODES:
            return True
        if _EXPIRED_RE.search(str(e)):
            return True
        stack.append(e.__cause__)
        stack.append(e.__context__)
        stack.extend(x for x in (getattr(e, "exc_info", None) or ())[1:2])
    return False


class _TotalSmoother:
    """yt-dlp's HLS estimate (bytes so far / fragments done x fragment count) starts high (concurrent fragments count
    bytes of unfinished fragments) and keeps shrinking. Hide it until 5 % of the fragments are done, then only move
    the shown value when it is off by more than 5 %."""

    def __init__(self):
        self.shown: float | None = None

    def __call__(self, est, frag_index, frag_count):
        if not est:
            return None
        if frag_count and (frag_index or 0) < max(3, 0.05 * frag_count):
            return None
        if self.shown is None or abs(est - self.shown) > 0.05 * self.shown:
            self.shown = float(est)
        return self.shown


def _map_progress(d: dict, smooth: "_TotalSmoother | None" = None) -> dict:
    status = d.get("status")
    exact = d.get("total_bytes")
    total = exact
    if not exact:
        est = d.get("total_bytes_estimate")
        total = smooth(est, d.get("fragment_index"), d.get("fragment_count")) if smooth else est
    return {
        "stage": "finished" if status == "finished" else "downloading",
        "downloaded_bytes": d.get("downloaded_bytes"),
        "total_bytes": total,
        "total_approx": bool(total) and not exact,
        "speed": d.get("speed"),
        "eta": d.get("eta"),
        "fragment_index": d.get("fragment_index"),
        "fragment_count": d.get("fragment_count"),
        "filename": d.get("filename"),
    }


def _resolve(target: str, headers: dict) -> tuple[str, dict, str | None]:
    """(url to hand to yt-dlp, headers, title if the resolver knows it). Same order as the player: an external
    resolver that matches wins, otherwise yt-dlp gets the URL itself."""
    resolver = source_resolve.find_resolver(target)
    if resolver:
        url, merged, title = source_resolve.resolve_external(target, headers, resolver)
        return url, merged, title
    return target, dict(headers), None


def _find_output(dest: Path, info: dict | None) -> Path | None:
    for rd in (info or {}).get("requested_downloads") or []:
        fp = rd.get("filepath") or rd.get("_filename")
        if fp and Path(fp).is_file():
            return Path(fp)
    fp = (info or {}).get("filepath")
    if fp and Path(fp).is_file():
        return Path(fp)
    done = sorted(p for p in dest.glob("source.*") if p.is_file() and p.suffix not in (".part", ".ytdl", ".temp"))
    return done[0] if done else None


def estimate_total_bytes(info: dict | None) -> float | None:
    """Best-effort size of what is about to be downloaded: filesize / filesize_approx, else (tbr or vbr + abr) x
    duration, summed over the requested formats (video + audio). None when any part is unknown."""
    info = info or {}
    parts = info.get("requested_formats") or [info]
    duration = info.get("duration")
    total = 0.0
    for f in parts:
        size = f.get("filesize") or f.get("filesize_approx")
        if not size:
            rate = f.get("tbr") or ((f.get("vbr") or 0) + (f.get("abr") or 0))      # kbit/s
            dur = f.get("duration") or duration
            size = rate * 1000 / 8 * dur if rate and dur else None
        if not size:
            return None
        total += float(size)
    return total or None


class _ProgressWatcher:
    """Reports progress for downloaders that fire no progress hook (ffmpeg & co) by polling the files in dest, and
    makes cancel work for them by killing the ffmpeg child process(es) started during this download."""

    IDLE_AFTER = 2.0        # hooks silent for this long -> start polling
    POLL_EVERY = 1.0
    TICK = 0.25
    EMA = 0.3

    def __init__(self, dest: Path, on_progress, cancel, clock=time.monotonic):
        self.dest, self.on_progress, self.cancel, self._clock = dest, on_progress, cancel, clock
        self.total_estimate: float | None = None
        self._last_hook = clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = self._ffmpeg_pids()
        self._prev_size: int | None = None
        self._prev_t: float | None = None
        self._speed: float | None = None
        self._max_size = 0

    def note_hook(self) -> None:
        self._last_hook = self._clock()
        self._prev_size = self._prev_t = self._speed = None     # restart the speed estimate after a hook period

    def set_estimate(self, total) -> None:
        if total:
            self.total_estimate = float(total)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="queue-dl-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    @staticmethod
    def _ffmpeg_pids() -> set:
        try:
            import psutil
            return {c.pid for c in psutil.Process().children(recursive=True)
                    if c.name().lower().startswith("ffmpeg")}
        except Exception:
            return set()

    def _kill_new_ffmpeg(self) -> None:
        try:
            import psutil
            for c in psutil.Process().children(recursive=True):
                if c.pid not in self._baseline and c.name().lower().startswith("ffmpeg"):
                    c.kill()
        except Exception as e:
            log(f"could not stop ffmpeg: {e}")

    def sizes(self) -> int:
        total = 0
        try:
            for p in self.dest.glob("source.*"):
                if p.suffix == ".ytdl":
                    continue
                try:
                    if p.is_file():
                        total += p.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
        return total

    def _run(self) -> None:
        next_poll = self._clock() + self.POLL_EVERY
        while not self._stop.wait(self.TICK):
            if self.cancel.is_set():
                self._kill_new_ffmpeg()
                continue
            now = self._clock()
            if now < next_poll:
                continue
            next_poll = now + self.POLL_EVERY
            if now - self._last_hook > self.IDLE_AFTER:
                self.poll(now)

    def poll(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        size = max(self.sizes(), self._max_size)
        self._max_size = size
        if size <= 0:
            return
        if self._prev_size is not None and now > self._prev_t:
            inst = max(0.0, (size - self._prev_size) / (now - self._prev_t))
            self._speed = inst if self._speed is None else self._speed + self.EMA * (inst - self._speed)
        self._prev_size, self._prev_t = size, now
        total = self.total_estimate
        if total and size >= total * 0.98:
            total = None            # the estimate was too low: don't show a bogus 100 %
        eta = (total - size) / self._speed if total and self._speed else None
        try:
            self.on_progress({
                "stage": "downloading", "downloaded_bytes": size, "total_bytes": total, "total_approx": bool(total),
                "speed": self._speed, "eta": eta, "fragment_index": None, "fragment_count": None, "filename": None,
            })
        except Exception as e:
            log(f"progress callback failed: {e}")


def _download_once(url: str, headers: dict, fmt: str, dest: Path, on_progress, cancel) -> dict:
    watcher = _ProgressWatcher(dest, on_progress, cancel)

    def match_filter(info, *, incomplete=False):
        # Called by yt-dlp before downloading (with the selected formats): grab a size estimate for the watcher.
        try:
            watcher.set_estimate(estimate_total_bytes(info))
        except Exception:
            pass
        return None

    smooth = _TotalSmoother()
    speed_tracker = [None, None, None]  # [prev_bytes, prev_time, speed_ema]

    def hook(d):
        watcher.note_hook()
        if cancel.is_set():
            raise _CancelSignal()
        now = time.monotonic()
        dl_bytes = d.get("downloaded_bytes")
        speed = d.get("speed")
        if (speed is None or speed <= 0) and dl_bytes is not None:
            if speed_tracker[0] is not None and speed_tracker[1] is not None:
                dt = now - speed_tracker[1]
                if dt >= 0.25:
                    inst = max(0.0, (dl_bytes - speed_tracker[0]) / dt)
                    speed_tracker[2] = inst if speed_tracker[2] is None else speed_tracker[2] * 0.7 + inst * 0.3
                    speed_tracker[0] = dl_bytes
                    speed_tracker[1] = now
                    speed = speed_tracker[2]
                else:
                    speed = speed_tracker[2]
            else:
                speed_tracker[0] = dl_bytes
                speed_tracker[1] = now
        elif speed is not None and speed > 0:
            speed_tracker[0] = dl_bytes
            speed_tracker[1] = now
            speed_tracker[2] = speed

        mapped = _map_progress(d, smooth)
        if speed is not None and speed > 0 and (mapped.get("speed") is None or mapped.get("speed") <= 0):
            mapped["speed"] = speed
            if (mapped.get("eta") is None or mapped.get("eta") <= 0) and mapped.get("total_bytes") and dl_bytes:
                mapped["eta"] = max(0.0, (mapped["total_bytes"] - dl_bytes) / speed)
        try:
            on_progress(mapped)
        except Exception as e:     # a broken UI callback must not kill the download
            log(f"progress callback failed: {e}")

    def pp_hook(d):
        watcher.note_hook()
        if cancel.is_set():
            raise _CancelSignal()
        if d.get("status") == "started":
            try:
                on_progress({"stage": "postprocessing", "downloaded_bytes": None, "total_bytes": None,
                             "total_approx": False, "speed": None, "eta": None, "fragment_index": None, "fragment_count": None, "filename": None})
            except Exception as e:
                log(f"progress callback failed: {e}")

    opts = {
        "format": fmt,
        "outtmpl": str(dest / "source.%(ext)s"),
        "merge_output_format": "mkv",
        "continuedl": True,
        "concurrent_fragment_downloads": 4,
        "match_filter": match_filter,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "logger": _YdlLogger(),
        "http_headers": dict(headers),
        "progress_hooks": [hook],
        "postprocessor_hooks": [pp_hook],
        "retries": 5,
        "fragment_retries": 10,
    }
    if shutil.which("node"):    # YouTube needs a JS runtime, as in source_resolve.resolve_with_ytdlp
        opts["js_runtimes"] = {"node": {}}
    watcher.start()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            from oh_streamer.sources.http_transport import configure
            configure(ydl, url, headers)
            return ydl.extract_info(url, download=True)
    finally:
        watcher.stop()


def _extract_selected(url: str, headers: dict, fmt: str) -> dict:
    """Resolve and select a single yt-dlp format without downloading it."""
    opts = {
        "format": fmt,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "logger": _YdlLogger(),
        "http_headers": dict(headers),
    }
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        from oh_streamer.sources.http_transport import configure
        configure(ydl, url, headers)
        return ydl.extract_info(url, download=False) or {}


def _selected_hls(info: dict) -> tuple[str, dict] | None:
    """Return the selected HLS media URL and format only for one rendition."""
    if not isinstance(info, dict) or info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        return None
    if info.get("has_drm") or info.get("drm_family"):
        return None
    requested = info.get("requested_formats")
    if requested and (not isinstance(requested, (list, tuple)) or len(requested) != 1):
        return None
    selected = requested[0] if requested else info
    if not isinstance(selected, dict) or selected.get("has_drm") or selected.get("drm_family"):
        return None
    url = selected.get("url") or info.get("url")
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return None
    protocol = str(selected.get("protocol") or info.get("protocol") or "").lower()
    ext = str(selected.get("ext") or "").lower()
    if "m3u8" not in protocol and ".m3u8" not in url.lower().split("?", 1)[0] and ext not in ("m3u8",):
        return None
    if selected.get("vcodec") == "none" or selected.get("acodec") == "none":
        return None
    return url, selected


class _ScopedHLSRedirect(urllib.request.HTTPRedirectHandler):
    """Follow HLS redirects without forwarding credentials to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            allowed = scope_for_url(req.full_url, newurl,
                                    {**redirected.headers, **redirected.unredirected_hdrs})
            for collection in (redirected.headers, redirected.unredirected_hdrs):
                for name in list(collection):
                    if name.lower() not in {key.lower() for key in allowed}:
                        del collection[name]
        return redirected


def _open_hls_request(url: str, headers: dict):
    # A per-request opener retains urllib's default proxy and verified TLS setup.
    return urllib.request.build_opener(_ScopedHLSRedirect()).open(
        urllib.request.Request(url, headers=dict(headers)), timeout=30)


def _fetch_bytes(url: str, headers: dict, limit: int = 4 * 1024 * 1024) -> tuple[bytes, str]:
    with _open_hls_request(url, headers) as response:
        chunks = []
        total = 0
        while True:
            part = response.read(min(64 * 1024, limit + 1 - total))
            if not part:
                break
            chunks.append(part)
            total += len(part)
            if total > limit:
                raise hls_cache.HLSCacheError("HLS response exceeds the size limit")
        return b"".join(chunks), response.geturl()


def _probe_ts(path: str | Path) -> dict:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise hls_cache.HLSCacheError(f"ffprobe could not inspect the first HLS segment: {exc}") from exc
    if p.returncode:
        raise hls_cache.HLSCacheError(f"ffprobe failed: {p.stderr.strip()[:200]}")
    try:
        value = json.loads(p.stdout)
    except (TypeError, ValueError) as exc:
        raise hls_cache.HLSCacheError("ffprobe returned invalid metadata") from exc
    if not isinstance(value, dict):
        raise hls_cache.HLSCacheError("ffprobe returned invalid metadata")
    return value


def _validate_mpegts_probe(probe: dict, selected: dict) -> None:
    fmt = probe.get("format") or {}
    names = str(fmt.get("format_name") or "").lower().split(",")
    if "mpegts" not in names:
        raise hls_cache.HLSCacheError("selected HLS segments are not MPEG-TS")
    streams = probe.get("streams") or []
    types = {str(s.get("codec_type")) for s in streams if isinstance(s, dict)}
    if not {"video", "audio"}.issubset(types):
        raise hls_cache.HLSCacheError("selected HLS rendition is not muxed audio/video")
    if selected.get("vcodec") == "none" or selected.get("acodec") == "none":
        raise hls_cache.HLSCacheError("selected format is not muxed audio/video")


def _playlist_metadata(probe: dict, duration: float) -> dict:
    """Use the complete playlist duration while preserving TS time origins."""
    result = deepcopy(probe)
    duration_text = f"{duration:.6f}"
    result.setdefault("format", {})["duration"] = duration_text
    for stream in result.get("streams") or []:
        if stream.get("codec_type") == "video":
            stream["duration"] = duration_text
            stream.pop("nb_frames", None)
    return result


def _request_key(target: str, fmt: str) -> str:
    return hashlib.sha256((target + "\0" + fmt).encode("utf-8", "surrogatepass")).hexdigest()


def _notify_source_ready(callback, session) -> None:
    try:
        callback(session)
    except Exception as exc:
        log(f"source-ready callback failed: {exc}")


def _progressive_candidate(info: dict) -> tuple[str, tuple[str, dict] | None]:
    """(reason, (url, selected format)) for a single muxed progressive http(s) file; candidate None + reason otherwise.
    Separate audio/video (requested_formats > 1), fragmented (HLS / DASH) and non-http protocols are not eligible."""
    if not isinstance(info, dict):
        return "no format information", None
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        return "live stream", None
    if info.get("has_drm") or info.get("drm_family"):
        return "DRM protected", None
    requested = info.get("requested_formats")
    if requested and (not isinstance(requested, (list, tuple)) or len(requested) != 1):
        return ("audio and video are separate files (not supported yet: second phase of progressive sources)",
                None)
    selected = requested[0] if requested else info
    if not isinstance(selected, dict) or selected.get("has_drm") or selected.get("drm_family"):
        return "DRM protected", None
    url = selected.get("url") or info.get("url")
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return "the selected format has no direct http(s) URL", None
    protocol = str(selected.get("protocol") or info.get("protocol") or "").lower()
    if protocol not in ("http", "https"):
        return f"protocol={protocol or None} is not a single progressive http(s) file", None
    ext = str(selected.get("ext") or "").lower()
    path = url.lower().split("?", 1)[0]
    if ext in ("m3u8", "mpd") or path.endswith((".m3u8", ".mpd")) or selected.get("fragments"):
        return "the selected format is segmented (HLS / DASH)", None
    if selected.get("vcodec") == "none" or selected.get("acodec") == "none":
        return "the selected format is not muxed audio/video", None
    return "", (url, selected)


def _run_ffprobe_json(cmd: list, cancel: threading.Event, timeout: float) -> dict:
    """Run an ffprobe command that prints JSON; cancel aborts promptly. Raises range_cache.RangeCacheError."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
    except OSError as exc:
        raise range_cache.RangeCacheError(f"ffprobe could not start: {exc}") from exc
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set() or time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                if cancel.is_set():
                    raise DownloadCancelled()
                raise range_cache.RangeCacheError("ffprobe timed out reading the range cache")
    if proc.returncode:
        raise range_cache.RangeCacheError(f"ffprobe failed: {(err or '').strip()[:200]}")
    try:
        value = json.loads(out)
    except (TypeError, ValueError) as exc:
        raise range_cache.RangeCacheError("ffprobe returned invalid metadata") from exc
    if not isinstance(value, dict):
        raise range_cache.RangeCacheError("ffprobe returned invalid metadata")
    return value


def _probe_range_url(url: str, cancel: threading.Event, timeout: float = 90.0) -> dict:
    """ffprobe a (local range-cache) URL; cancel aborts promptly. Raises range_cache.RangeCacheError on any problem."""
    return _run_ffprobe_json(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", url], cancel, timeout)


def _positive_float(raw) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 and value == value and value != float("inf") else None


def _frame_rate(video: dict) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        text = str(video.get(key) or "")
        try:
            num, _, den = text.partition("/")
            value = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            return value
    return None


def _tail_span(url: str, cancel: threading.Event, video: dict, base_duration: float,
               timeout: float = 30.0) -> float | None:
    """The video's real span (end of its last packet - its start_time), from one ffprobe over the last seconds of the
    file. Container durations of webm / fMP4-with-global-sidx are absolute end times (span + start_time), which makes
    the worker expect one frame too many. Returns None when it cannot be determined."""
    start = _positive_float(video.get("start_time")) or 0.0
    fps = _frame_rate(video)
    window = 10.0
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-select_streams", "v:0",
           "-read_intervals", f"{max(0.0, base_duration - window):.3f}%",
           "-show_entries", "packet=pts_time,duration_time", url]
    packets = _run_ffprobe_json(cmd, cancel, timeout).get("packets") or []
    end = None
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        pts = packet.get("pts_time")
        try:
            pts = float(pts)
        except (TypeError, ValueError):
            continue
        dur = _positive_float(packet.get("duration_time")) or (1.0 / fps if fps else None)
        if dur is None:
            continue
        if end is None or pts + dur > end:
            end = pts + dur
    if end is None:
        return None
    span = end - start
    # Sanity: it must be close to the duration we already know (never longer than it by more than a frame or two
    # plus the start offset, never less than half), otherwise keep the old behaviour.
    if span <= 0 or span > base_duration + 1.0 or span < base_duration * 0.5:
        return None
    return span


def _range_metadata(probe: dict, url: str | None = None, cancel: threading.Event | None = None) -> dict:
    """Probe of a progressive file for the worker/player: the container duration can exceed the video's (longer audio,
    or an absolute end time for webm / fMP4+sidx), but the worker checks the decoded frame count and end against the
    video span, so use the video stream's span. With `url`, the span is measured from the last video packet."""
    result = deepcopy(probe)
    streams = result.get("streams") or []
    video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if video is None:
        raise range_cache.RangeCacheError("the file has no video stream")
    fmt = result.setdefault("format", {})
    duration = None
    for raw in (video.get("duration"), fmt.get("duration")):
        value = _positive_float(raw)
        if value is not None:
            duration = value
            break
    if duration is None:
        raise range_cache.RangeCacheError("the file has no known duration")
    if url:
        try:
            span = _tail_span(url, cancel or threading.Event(), video, duration)
        except DownloadCancelled:
            raise
        except range_cache.RangeCacheError as exc:
            span = None
            log(f"range cache: tail span probe failed, using container duration ({exc})")
        if span is None:
            log("range cache: tail span unavailable, using container duration")
        else:
            if abs(span - duration) > 0.001:
                log(f"range cache: video span {span:.3f}s (container duration {duration:.3f}s)")
            duration = span
    text = f"{duration:.6f}"
    fmt["duration"] = text
    video["duration"] = text
    return result


def _download_range(url: str, headers: dict, selected: dict, probe_info: dict, dest: Path, target: str,
                    orig_headers: dict, fmt: str, on_progress, cancel, on_source_ready,
                    title: str) -> DownloadResult | None:
    """Progressive single-file path: a byte-range cache session owns the download, the worker/player read it through
    session.runtime_url while it grows. Returns None (caller falls back to the full download) when the cache cannot be
    used before the source was announced."""
    cache_dir = dest / "range-cache"
    request_key = _request_key(target, fmt)
    ext = str(selected.get("ext") or "mp4").lower()

    def on_expired(code, old_url):
        if cancel.is_set():
            return None
        log(f"range cache: upstream URL expired (HTTP {code}); re-resolving")
        new_url, new_headers, _title = _resolve(target, orig_headers)
        cand = _progressive_candidate(_extract_selected(new_url, new_headers, fmt))[1]
        if not cand:
            return None
        return cand[0], {**new_headers, **(cand[1].get("http_headers") or {})}

    def prepare():
        return range_cache.prepare_session(
            cache_dir, url=url, total_bytes=probe_info["total_bytes"], request_key=request_key, headers=headers,
            ext=ext, etag=probe_info.get("etag"), on_expired=on_expired)

    try:
        try:
            session, ready = prepare()
        except range_cache.RangeCacheError as exc:
            # A different source / size / etag: the old partial cache can never be mixed in, start over.
            log(f"range cache: discarding the previous cache ({exc})")
            range_cache.close_session(cache_dir / range_cache.MARKER_NAME)
            shutil.rmtree(cache_dir, ignore_errors=True)
            session, ready = prepare()
    except (range_cache.RangeCacheError, OSError) as exc:
        log(f"Range cache not used, full download instead: {exc}")
        return None

    marker = session.playlist_path

    try:
        metadata = session.metadata if (session.metadata.get("streams") and session.metadata.get("format")) else None
        if metadata is None:
            try:
                metadata = _range_metadata(_probe_range_url(session.runtime_url, cancel), session.runtime_url, cancel)
            except range_cache.RangeCacheError as exc:
                if session.status == "failed":
                    raise DownloadFailed(session.snapshot().get("error") or str(exc)) from exc
                range_cache.close_session(marker)
                shutil.rmtree(cache_dir, ignore_errors=True)
                log(f"Range cache not used, full download instead: {exc}")
                return None
            session.set_metadata(metadata)
    except DownloadCancelled:
        range_cache.fail_session(session, "download cancelled", paused=True)
        raise
    except DownloadFailed:
        raise
    except Exception as exc:
        range_cache.fail_session(session, exc)
        raise DownloadFailed(str(exc) or exc.__class__.__name__) from exc

    duration = float((metadata.get("format") or {}).get("duration") or 0) or None
    _notify_source_ready(on_source_ready, session)

    snap = session.snapshot()
    total = snap["total_bytes"]
    speed = [None, time.monotonic(), snap["downloaded_bytes"]]      # EMA, last time, last bytes

    def emit(stage, snap):
        done = snap["downloaded_bytes"]
        now = time.monotonic()
        dt = now - speed[1]
        if dt >= 0.3:
            inst = max(0.0, (done - speed[2]) / dt)
            speed[0] = inst if speed[0] is None else speed[0] * 0.7 + inst * 0.3
            speed[1], speed[2] = now, done
        eta = (total - done) / speed[0] if speed[0] and total > done else None
        try:
            on_progress({"stage": stage, "downloaded_bytes": done, "total_bytes": total, "total_approx": False,
                         "speed": speed[0], "eta": eta, "fragment_index": len(snap["ready_segments"]),
                         "fragment_count": snap["segment_count"], "filename": marker})
        except Exception as exc:
            log(f"progress callback failed: {exc}")

    try:
        while True:
            if cancel.is_set():
                raise DownloadCancelled()
            status = session.wait(0.25)
            snap = session.snapshot()
            if status != "downloading":
                break
            emit("downloading", snap)
        if status != "complete":
            if cancel.is_set():
                raise DownloadCancelled()
            raise DownloadFailed(snap.get("error") or f"range cache ended as {status}")
    except DownloadCancelled:
        range_cache.fail_session(session, "download cancelled", paused=True)
        raise
    except DownloadFailed:
        raise
    except Exception as exc:
        range_cache.fail_session(session, exc)
        if cancel.is_set():
            raise DownloadCancelled() from exc
        raise DownloadFailed(str(exc) or exc.__class__.__name__) from exc
    emit("finished", snap)
    return DownloadResult(marker, str(title), duration)


def _try_range(progressive: tuple[str, dict], selected_info: dict, resolved_headers: dict, dest: Path, target: str,
               orig_headers: dict, fmt: str, on_progress, cancel, on_source_ready) -> DownloadResult | None:
    url, selected = progressive
    hdrs = {**resolved_headers, **(selected.get("http_headers") or {})}
    probe_info = range_cache.probe_range_support(url, hdrs)
    if cancel.is_set():
        raise DownloadCancelled()
    if probe_info is None:
        log("Range cache not used, full download instead: the server does not support byte ranges")
        return None
    title = selected_info.get("title") or selected.get("title") or target
    try:
        return _download_range(url, hdrs, selected, probe_info, dest, target, orig_headers, fmt,
                               on_progress, cancel, on_source_ready, title)
    except (DownloadCancelled, DownloadFailed):
        raise
    except Exception as e:
        if cancel.is_set():
            raise DownloadCancelled() from e
        raise DownloadFailed(str(e) or e.__class__.__name__) from e


def _fetch_segment_plain(segment, headers: dict, path: Path, keys: dict, cancel, on_progress, count: int,
                         *, source_url: str | None = None, **kw) -> int:
    """Download one segment to `path`; AES-128 segments are decrypted so the cache only ever holds plain TS."""
    segment_headers = scope_for_url(source_url or segment.url, segment.url, headers)
    if not segment.key_uri:
        return _download_segment(segment.url, segment_headers, path, cancel, on_progress, segment.index, count, **kw)
    key = keys.get(segment.key_uri)
    if key is None:
        # Keys live in memory only (one fetch per distinct URI per session); same headers as segments.
        key_headers = scope_for_url(source_url or segment.url, segment.key_uri, headers)
        key = keys[segment.key_uri] = _fetch_bytes(segment.key_uri, key_headers, limit=4096)[0]
    raw_path = path.with_name(path.name + ".enc")
    try:
        got = _download_segment(segment.url, segment_headers, raw_path, cancel, on_progress,
                                segment.index, count, **kw)
        plain = hls_cache.decrypt_segment(raw_path.read_bytes(), key, segment.iv)
        path.write_bytes(plain)
    finally:
        raw_path.unlink(missing_ok=True)
    return got


def _download_hls(url: str, headers: dict, selected: dict, dest: Path, target: str, fmt: str,
                  on_progress, cancel, on_source_ready) -> DownloadResult | None:
    """Try the progressive shared-cache path; return None for unsupported HLS."""
    try:
        manifest_raw, manifest_final_url = _fetch_bytes(url, headers)
        manifest = hls_cache.parse_media_playlist(manifest_raw, url, manifest_final_url)
    except hls_cache.HLSCacheError as exc:
        log(f"HLS cache not used, full download instead: {exc}")
        return None
    except Exception as exc:
        raise DownloadFailed(str(exc) or exc.__class__.__name__) from exc

    cache_dir = dest / "hls-cache"
    request_key = _request_key(target, fmt)
    try:
        saved_metadata, verified = hls_cache.inspect_cache(cache_dir, manifest, request_key)
    except hls_cache.HLSCacheError as exc:
        raise DownloadFailed(str(exc)) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    keys: dict[str, bytes] = {}
    stage_path = cache_dir / f".probe-{uuid.uuid4().hex}.tmp"
    probe = None
    if 0 in verified:
        if saved_metadata:
            probe = saved_metadata
        else:
            try:
                probe = _probe_ts(cache_dir / manifest.segments[0].filename)
            except Exception as exc:
                # Do not silently replace a verified segment with a fresh URL response.
                log(f"HLS cache not used, full download instead: cached first segment unreadable ({exc})")
                return None
    if probe is None:
        try:
            _fetch_segment_plain(manifest.segments[0], headers, stage_path, keys, cancel, on_progress,
                                 len(manifest.segments), source_url=url)
            probe = _probe_ts(stage_path)
        except DownloadCancelled:
            stage_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            stage_path.unlink(missing_ok=True)
            if isinstance(exc, hls_cache.HLSCacheError):
                log(f"HLS cache not used, full download instead: {exc}")
                return None
            raise DownloadFailed(str(exc) or exc.__class__.__name__) from exc

    try:
        _validate_mpegts_probe(probe, selected)
    except hls_cache.HLSCacheError as exc:
        stage_path.unlink(missing_ok=True)
        log(f"HLS cache not used, full download instead: {exc}")
        return None
    metadata = _playlist_metadata(probe, manifest.duration)
    try:
        session, ready = hls_cache.prepare_session(
            manifest, cache_dir, metadata=metadata, request_key=request_key,
            seed_path=stage_path if stage_path.exists() else None,
        )
    except hls_cache.HLSCacheError as exc:
        stage_path.unlink(missing_ok=True)
        raise DownloadFailed(str(exc)) from exc
    finally:
        stage_path.unlink(missing_ok=True)

    if 0 not in ready:
        hls_cache.fail_session(session, "first segment was not committed")
        raise DownloadFailed("first HLS segment was not committed")
    _notify_source_ready(on_source_ready, session)

    downloaded_before = sum(
        (cache_dir / f"segment-{i:06d}.ts").stat().st_size
        for i in ready if (cache_dir / f"segment-{i:06d}.ts").is_file()
    )
    started = time.monotonic()
    downloaded_now = 0
    try:
        while True:
            if cancel.is_set():
                raise DownloadCancelled()
            index = session.next_download_segment()
            if index is None:
                if session.status != "downloading":
                    raise DownloadCancelled()
                break
            segment = manifest.segments[index]
            temp_path = cache_dir / f".{segment.filename}.{uuid.uuid4().hex}.tmp"
            try:
                count = _fetch_segment_plain(segment, headers, temp_path, keys, cancel, on_progress,
                                            len(manifest.segments),
                                            source_url=url,
                                            byte_offset=downloaded_before + downloaded_now,
                                            bytes_this_session=downloaded_now,
                                            clock_start=started)
                downloaded_now += count
                hls_cache.commit_segment(session, segment.index, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
        hls_cache.complete_session(session)
    except DownloadCancelled:
        hls_cache.fail_session(session, "download cancelled", paused=True)
        raise
    except Exception as exc:
        hls_cache.fail_session(session, exc)
        if cancel.is_set():
            raise DownloadCancelled() from exc
        raise DownloadFailed(str(exc) or exc.__class__.__name__) from exc

    try:
        on_progress({"stage": "finished", "downloaded_bytes": downloaded_before + downloaded_now,
                     "total_bytes": None, "total_approx": False, "speed": None, "eta": None,
                     "fragment_index": len(manifest.segments), "fragment_count": len(manifest.segments),
                     "filename": session.playlist_path})
    except Exception as exc:
        log(f"progress callback failed: {exc}")
    title = selected.get("title") or target
    return DownloadResult(session.playlist_path, str(title), manifest.duration)


def _download_segment(url: str, headers: dict, path: Path, cancel: threading.Event,
                      on_progress, index: int, count: int, *, byte_offset: int = 0,
                      bytes_this_session: int = 0,
                      clock_start: float | None = None) -> int:
    downloaded = 0
    started = clock_start or time.monotonic()
    last_callback = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_hls_request(url, headers) as response:
        expected = response.headers.get("Content-Length")
        expected_n = int(expected) if expected and expected.isdigit() else None
        with path.open("wb") as out:
            while True:
                if cancel.is_set():
                    raise DownloadCancelled()
                block = response.read(128 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if on_progress and (now - last_callback >= 0.2 or (expected_n and downloaded >= expected_n)):
                    elapsed = max(0.001, now - started)
                    speed = (bytes_this_session + downloaded) / elapsed
                    try:
                        on_progress({"stage": "downloading", "downloaded_bytes": byte_offset + downloaded,
                                     "total_bytes": None, "total_approx": False, "speed": speed, "eta": None,
                                     "fragment_index": index + 1, "fragment_count": count,
                                     "filename": path.name})
                    except Exception as exc:
                        log(f"progress callback failed: {exc}")
                    last_callback = now
            out.flush()
            os.fsync(out.fileno())
        if expected_n is not None and downloaded != expected_n:
            raise DownloadFailed(f"incomplete HLS segment {index}: received {downloaded} of {expected_n} bytes")
    if not downloaded:
        raise DownloadFailed(f"HLS segment {index} is empty")
    return downloaded


def download(target: str, headers: dict | None, quality, dest_dir, on_progress: Callable[[dict], None] | None,
             cancel: threading.Event | None, *, on_source_ready: Callable[[hls_cache.HLSSession], None] | None = None) -> DownloadResult:
    headers = dict(headers or {})
    on_progress = on_progress or (lambda _d: None)
    cancel = cancel or threading.Event()

    if not source_resolve.is_url(target):
        p = Path(target)
        if not p.is_file():
            raise DownloadFailed(f"file not found: {target}")
        return DownloadResult(path=str(p), title=p.stem, duration=None)

    try:
        fmt = format_selector(quality)
    except ValueError as e:
        raise DownloadFailed(str(e)) from e
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if cancel.is_set():
        raise DownloadCancelled()

    if on_source_ready is not None:
        cache_dir = dest / "hls-cache"
        request_key = _request_key(target, fmt)
        cached = hls_cache.open_session(cache_dir / hls_cache.PLAYLIST_NAME)
        if cached:
            if (cached._source_descriptor or {}).get("request_key") == request_key:
                _notify_source_ready(on_source_ready, cached)
                raw_duration = (cached.metadata.get("format") or {}).get("duration")
                try:
                    duration = float(raw_duration) if raw_duration is not None else None
                except (TypeError, ValueError):
                    duration = None
                return DownloadResult(cached.playlist_path, target, duration)
            hls_cache.close_session(cached.playlist_path)

        range_marker = dest / "range-cache" / range_cache.MARKER_NAME
        cached_range = range_cache.open_session(range_marker)
        if cached_range:
            if cached_range._data.get("request_key") == request_key:
                _notify_source_ready(on_source_ready, cached_range)
                raw_duration = (cached_range.metadata.get("format") or {}).get("duration")
                try:
                    duration = float(raw_duration) if raw_duration is not None else None
                except (TypeError, ValueError):
                    duration = None
                return DownloadResult(cached_range.playlist_path, target, duration)
            range_cache.close_session(cached_range.playlist_path)

        # Selection is a read-only yt-dlp pass.  The original complete-download
        # path below remains the fallback for every non-eligible source.
        selected_info = None
        try:
            resolved_url, resolved_headers, resolved_title = _resolve(target, headers)
            selected_info = _extract_selected(resolved_url, resolved_headers, fmt)
            candidate = _selected_hls(selected_info)
        except Exception as e:
            if cancel.is_set():
                raise DownloadCancelled() from e
            log(f"HLS cache not used, full download instead: format selection failed ({e.__class__.__name__}: {e})")
            candidate = None
        else:
            if not candidate:
                log("HLS cache not used, full download instead: the selected format is not a single plain HLS "
                    f"rendition (protocol={selected_info.get('protocol') if isinstance(selected_info, dict) else None})")
        if candidate:
            try:
                result = _download_hls(candidate[0], resolved_headers, candidate[1], dest, target, fmt,
                                       on_progress, cancel, on_source_ready)
            except DownloadCancelled:
                raise
            except DownloadFailed:
                raise
            except Exception as e:
                if cancel.is_set():
                    raise DownloadCancelled() from e
                raise DownloadFailed(str(e) or e.__class__.__name__) from e
            if result is not None:
                result.title = resolved_title or result.title
                return result
        elif selected_info is not None:
            reason, progressive = _progressive_candidate(selected_info)
            if progressive is None:
                log(f"Range cache not used, full download instead: {reason}")
            else:
                result = _try_range(progressive, selected_info, resolved_headers, dest, target, headers, fmt,
                                    on_progress, cancel, on_source_ready)
                if result is not None:
                    result.title = resolved_title or result.title
                    return result

    last_error = None
    for attempt in (1, 2):          # the second attempt re-resolves (fresh signed URLs) and resumes the .part files
        if cancel.is_set():
            raise DownloadCancelled()
        try:
            url, hdrs, title = _resolve(target, headers)
            info = _download_once(url, hdrs, fmt, dest, on_progress, cancel)
        except _CancelSignal as e:
            raise DownloadCancelled() from e
        except DownloadCancelled:
            raise
        except Exception as e:
            if cancel.is_set():
                raise DownloadCancelled() from e
            last_error = e
            if attempt == 1 and _is_expired_error(e):
                log(f"URL expired ({e}); re-resolving once and resuming")
                continue
            raise DownloadFailed(str(e) or e.__class__.__name__) from e
        path = _find_output(dest, info)
        if path is None:
            raise DownloadFailed("download finished but no source file was found")
        info = info or {}
        duration = info.get("duration")
        return DownloadResult(path=str(path), title=title or info.get("title") or target,
                              duration=float(duration) if duration else None)
    raise DownloadFailed(str(last_error))
