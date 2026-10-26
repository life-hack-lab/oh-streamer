"""Resolve one pasted input to the facts the queue's intake panel (解析面板) shows: title, duration, fps, renditions,
codecs and a small thumbnail. No torch / lada imports (the queue window process stays light).

The dispatch mirrors player/opening.resolve_input: a page URL goes through yt-dlp, a URL with an external resolver
through that resolver, a direct media URL or a local file straight to ffprobe. Everything that talks to the outside
(yt-dlp, the resolver, ffprobe, the ffmpeg frame grab) is looked up at call time or injectable, so tests never hit
the network. `TokenCache` is the server-side handle store: the panel only gets a token back, the headers and the
resolved format URLs stay here.
"""

import base64
import json
import math
import os
import subprocess
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from oh_streamer.player import open_input
from oh_streamer.sources import qualities, source_resolve
from oh_streamer.sources.http_headers import merge_for_media, scope_for_url

THUMB_TIMEOUT = 20.0            # allow remote HLS playlist/key/segment requests to finish
PROBE_TIMEOUT = 20.0
THUMB_AT = 0.1                  # the frame comes from 10 % into the video
THUMB_FALLBACK_AT = 2.0         # ... or this many seconds in when the duration is unknown
CACHE_SIZE = 50
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_CODEC_NAMES = (("avc", "h264"), ("h264", "h264"), ("hev", "hevc"), ("hvc", "hevc"), ("h265", "hevc"), ("vp09", "vp9"),
                ("vp9", "vp9"), ("av01", "av1"), ("mp4a", "aac"), ("opus", "opus"), ("ec-3", "eac3"),
                ("ac-3", "ac3"))


class ResolveError(Exception):
    """Resolving failed. `short` is the zh-TW line for the panel, `detail` the raw last line of the underlying error."""

    def __init__(self, short: str, detail: str = ""):
        super().__init__(short)
        self.short, self.detail = short, detail


@dataclass
class Resolved:
    """One resolved input. quals is the yt-dlp rendition list (None for direct URLs / files / external resolvers);
    heights is the public list ({height, fps, label, size_bytes}, high -> low) in every case."""

    target: str
    kind: str
    headers: dict
    title: str
    site: str
    duration: Optional[float] = None
    fps: Optional[float] = None
    default_height: Optional[int] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    hls: bool = False
    heights: list = field(default_factory=list)
    quals: Optional[list] = None
    thumb_src: Optional[tuple] = None      # (url or path, headers) the frame is grabbed from
    fps_by_height: dict = field(default_factory=dict)
    cover: Optional[str] = None           # website cover data URI; preferred over a video frame
    cover_src: Optional[tuple] = None     # yt-dlp website cover (URL, request headers)
    # Subtitle-audio facts (T2.3, see queue/stream_audio.py). media_src: (url or path, headers) actually read by the
    # direct / external-resolver / file path (None on the yt-dlp path). audio_format: the yt-dlp audio-only format
    # {"id", "url", "headers", "protocol"} ffmpeg can read directly (None when the site has none, or only DASH).
    media_src: Optional[tuple] = None
    audio_format: Optional[dict] = None

    def fps_at(self, height) -> Optional[float]:
        return self.fps_by_height.get(height) or self.fps


class TokenCache:
    """Thread-safe LRU of Resolved objects keyed by an opaque token."""

    def __init__(self, size: int = CACHE_SIZE):
        self._size = size
        self._items: "OrderedDict[str, Resolved]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, resolved: Resolved) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = resolved
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return token

    def get(self, token) -> Optional[Resolved]:
        with self._lock:
            found = self._items.get(token)
            if found is not None:
                self._items.move_to_end(token)
            return found

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------

def short_codec(name) -> Optional[str]:
    """yt-dlp / ffprobe codec string -> a short name ('avc1.640028' -> 'h264'); 'none' / empty -> None."""
    s = str(name or "").strip().lower()
    if not s or s == "none":
        return None
    for prefix, short in _CODEC_NAMES:
        if s.startswith(prefix):
            return short
    return s.split(".")[0]


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 else None


def _site_of(target: str, kind: str) -> str:
    if kind == "url":
        try:
            return (urlsplit(target).hostname or "").lower()
        except ValueError:
            return ""
    return str(Path(target).parent)


def _is_hls(url: str, protocol: str = "") -> bool:
    return str(protocol or "").startswith("m3u8") or urlsplit(url).path.lower().endswith(".m3u8")


def _size_bytes(fmt: dict, duration: Optional[float]) -> Optional[int]:
    size = _num(fmt.get("filesize")) or _num(fmt.get("filesize_approx"))
    if size is None and _num(fmt.get("tbr")) and duration:
        size = fmt["tbr"] * 1000 / 8 * duration
    return int(size) if size else None


def probe_media(src: str, headers: Optional[dict] = None) -> Optional[dict]:
    """ffprobe of a file / URL: {height, fps, duration, vcodec, acodec}; None when it cannot be read (any field may be
    None when the stream does not say)."""
    http = source_resolve.FFmpegHttpOptions(headers or {}).cli_args() if source_resolve.is_url(src) else []
    cmd = ["ffprobe", "-v", "error", *http, "-show_entries",
           "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
           "-of", "json", src]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=PROBE_TIMEOUT)
        if p.returncode != 0:
            return None
        data = json.loads(p.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        return None

    def rate(text) -> Optional[float]:
        try:
            num, _, den = str(text or "0/1").partition("/")
            value = float(num) / float(den or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return value if math.isfinite(value) and 0 < value <= 240 else None

    duration = _num(_to_float((data.get("format") or {}).get("duration"))) or _num(_to_float(video.get("duration")))
    height = video.get("height")
    return {"height": int(height) if isinstance(height, int) and height > 0 else None,
            "fps": rate(video.get("r_frame_rate")) or rate(video.get("avg_frame_rate")),
            "duration": duration, "vcodec": short_codec(video.get("codec_name")),
            "acodec": short_codec(audio.get("codec_name")) if audio else None}


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def error_line(e: Exception) -> str:
    lines = [ln.strip() for ln in str(e).strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else repr(e)).removeprefix("ERROR: ")


def _short_error(e: Exception) -> str:
    """The panel's one-line zh-TW reason for a failed resolve."""
    text = error_line(e).lower()
    if "unsupported url" in text:
        return "不支援這個網站或網址"
    if any(w in text for w in ("403", "401", "forbidden", "sign in", "login", "private", "members")):
        return "網站拒絕存取（可能需要登入或 headers）"
    if "404" in text or "not found" in text or "unavailable" in text:
        return "找不到這部影片"
    if any(w in text for w in ("timed out", "timeout", "connection", "resolve host", "getaddrinfo", "network")):
        return "連線失敗，請檢查網路"
    return "無法解析這個來源"


# ------------------------------------------------------------------------------------------------------------
# resolving
# ------------------------------------------------------------------------------------------------------------

_FFMPEG_AUDIO_PROTOCOLS = ("http", "https", "m3u8", "m3u8_native")


def _audio_format_for_ffmpeg(info: dict, headers: dict) -> Optional[dict]:
    """The player's audio-only pick (qualities.pick_audio_format) restricted to formats ffmpeg can `-ss` straight
    from a url (plain http(s) / HLS; DASH segment formats are skipped). headers = format/info headers + the user's."""
    usable = [f for f in info.get("formats") or [] if isinstance(f, dict)
              and (str(f.get("protocol") or "").lower() in _FFMPEG_AUDIO_PROTOCOLS
                   or (not f.get("protocol") and source_resolve.is_url(str(f.get("url") or ""))))]
    audio = qualities.pick_audio_format({**info, "formats": usable})
    if not audio:
        return None
    return {"id": str(audio.get("format_id") if audio.get("format_id") is not None else ""), "url": audio["url"],
            "headers": {**(audio.get("http_headers") or info.get("http_headers") or {}), **headers},
            "protocol": str(audio.get("protocol") or "")}


def _from_ytdlp(oi: open_input.OpenInput, ytdlp: Callable, probe: Callable,
                *, scoped_media: bool = False) -> Resolved:
    quals, title, thumb_pick, info = ytdlp(oi.target, dict(oi.headers), with_info=True)
    chosen = qualities.pick_quality(quals, None, None)
    formats = {str(f.get("format_id")): f for f in info.get("formats") or [] if isinstance(f, dict)}
    duration = _num(info.get("duration"))
    heights = []
    for q in quals:
        f = formats.get(str(q["id"])) or {}
        heights.append({"height": q["height"], "fps": q["fps"], "label": q["label"],
                        "size_bytes": _size_bytes(f, duration)})
    cf = formats.get(str(chosen["id"])) or {}
    media_origin = info.get("browser_media_url") or oi.target
    if scoped_media or info.get("browser_resolved"):
        chosen_headers = merge_for_media(oi.target, chosen["url"],
            scope_for_url(media_origin, chosen["url"], chosen["http_headers"]), oi.headers)
    else:
        chosen_headers = {**chosen["http_headers"], **oi.headers}
    audio_codec = None
    if chosen.get("audio_url"):
        audio = qualities.pick_audio_format(info)
        audio_codec = audio.get("acodec") if audio else None
    r = Resolved(oi.target, "url", dict(oi.headers), title, _site_of(oi.target, "url"), duration=duration,
                 fps=chosen["fps"] or _num(info.get("fps")), default_height=chosen["height"] or None,
                 vcodec=short_codec(cf.get("vcodec") or info.get("vcodec")),
                 acodec=short_codec(audio_codec or cf.get("acodec") or info.get("acodec")),
                 hls=_is_hls(chosen["url"], cf.get("protocol")), heights=heights, quals=quals,
                 fps_by_height={q["height"]: q["fps"] for q in quals if q["fps"]},
                 audio_format=_audio_format_for_ffmpeg(info, dict(oi.headers)))
    if thumb_pick:
        thumb_headers = (scope_for_url(oi.target, thumb_pick[0], oi.headers)
                         if scoped_media or info.get("browser_resolved") else oi.headers)
        format_thumb_headers = (scope_for_url(media_origin, thumb_pick[0], thumb_pick[1])
                                if scoped_media or info.get("browser_resolved") else thumb_pick[1])
        r.thumb_src = (thumb_pick[0], {**format_thumb_headers, **thumb_headers})
    else:
        r.thumb_src = (chosen["url"], chosen_headers)
    cover = info.get("thumbnail")
    if isinstance(cover, str) and source_resolve.is_url(cover):
        cover_headers = {"Referer": oi.target, **(info.get("http_headers") or {}), **oi.headers}
        r.cover_src = (cover, cover_headers)
    browser_cover = info.get("cover") if info.get("browser_resolved") else None
    if isinstance(browser_cover, str) and browser_cover.startswith((
            "data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
        r.cover = browser_cover
    if r.duration is None or r.fps is None or not chosen.get("height") or not chosen.get("fps"):
        # Opaque HLS formats may omit resolution even when yt-dlp knows duration.
        sample = (lambda url, hdrs: _probe_hls_sample(url, hdrs, probe)) if (
            info.get("browser_resolved") and r.hls) else probe
        facts = sample(chosen["url"], chosen_headers)
        if facts:
            r.duration = r.duration or facts.get("duration")
            r.fps = r.fps or facts.get("fps")
            r.vcodec = r.vcodec or facts.get("vcodec")
            r.acodec = r.acodec or facts.get("acodec")
            if not chosen.get("height") and facts.get("height"):
                chosen["height"] = int(facts["height"])
                r.default_height = chosen["height"]
            if not chosen.get("fps") and facts.get("fps"):
                chosen["fps"] = float(facts["fps"])
            if chosen.get("height"):
                chosen["label"] = qualities.quality_label(chosen["height"], chosen.get("fps"))
                if chosen.get("fps"):
                    r.fps_by_height[chosen["height"]] = chosen["fps"]
            for index, quality in enumerate(quals):
                if quality is chosen:
                    heights[index].update(height=chosen["height"], fps=chosen["fps"], label=chosen["label"])
                    break
    return r


def _from_probe(oi: open_input.OpenInput, src: str, headers: dict, title: str, probe: Callable) -> Resolved:
    """Direct media URL, external-resolver stream or local file: one rendition from ffprobe."""
    facts = probe(src, headers)
    if not facts:
        raise ResolveError("無法讀取影片（檔案損毀、格式不支援或無法連線）", f"ffprobe failed: {src}")
    height = facts["height"]
    label = qualities.quality_label(height, facts["fps"]) if height else "source"
    r = Resolved(oi.target, oi.kind, dict(oi.headers), title, _site_of(oi.target, oi.kind), duration=facts["duration"],
                 fps=facts["fps"], default_height=height, vcodec=facts["vcodec"], acodec=facts["acodec"],
                 hls=_is_hls(src), heights=[{"height": height, "fps": facts["fps"], "label": label,
                                             "size_bytes": _file_size(src) if oi.kind == "file" else None}]
                 if height else [], thumb_src=(src, headers), media_src=(src, dict(headers)))
    if height and facts["fps"]:
        r.fps_by_height = {height: facts["fps"]}
    return r


def _file_size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _probe_hls_sample(src: str, headers: dict, probe: Callable) -> Optional[dict]:
    """Inspect a bounded local transport-stream sample and use playlist duration."""
    from oh_streamer.queue.queue_download import _fetch_bytes
    from oh_streamer.sources.hls_cache import decrypt_segment, parse_media_playlist
    raw, final_url = _fetch_bytes(src, headers)
    playlist = parse_media_playlist(raw, src, final_url)
    first = playlist.segments[0]
    sample, _ = _fetch_bytes(first.url, scope_for_url(src, first.url, headers), limit=16 * 1024 * 1024)
    if first.key_uri:
        key, _ = _fetch_bytes(first.key_uri, scope_for_url(src, first.key_uri, headers), limit=4096)
        sample = decrypt_segment(sample, key, first.iv)
    with tempfile.TemporaryDirectory(prefix="oh_probe_") as directory:
        path = Path(directory) / "sample.ts"
        path.write_bytes(sample)
        facts = probe(str(path), {})
    if facts:
        facts = dict(facts)
        facts["duration"] = playlist.duration  # not just the duration of the sampled segment
    return facts


def resolve(oi: open_input.OpenInput, *, ytdlp: Optional[Callable] = None, find_resolver: Optional[Callable] = None,
            resolve_external: Optional[Callable] = None, probe: Optional[Callable] = None) -> Resolved:
    """OpenInput -> Resolved (no thumbnail yet; see grab_thumbnail). Raises ResolveError.

    The callables default to source_resolve's (looked up now, so they can be patched); `ytdlp` is called as
    resolve_with_ytdlp(url, headers, with_info=True), `find_resolver(url)`, `resolve_external(url, headers, resolver)`,
    `probe(src, headers)`."""
    ytdlp = ytdlp or source_resolve.resolve_with_ytdlp
    find_resolver = find_resolver or source_resolve.find_resolver
    external = resolve_external or (lambda url, hdrs, rule: source_resolve.resolve_external(
        url, hdrs, rule, with_info=True))
    probe = probe or probe_media
    target, headers = oi.target, dict(oi.headers)
    try:
        if oi.kind == "file":
            if not os.path.isfile(target):
                raise ResolveError("找不到影片檔案", target)
            return _from_probe(oi, str(Path(target).resolve()), {}, Path(target).stem, probe)
        resolver = find_resolver(target)
        if resolver is None and not source_resolve.looks_like_media_url(target):
            return _from_ytdlp(oi, ytdlp, probe)
        if resolver is not None:
            result = external(target, headers, resolver)
            src, resolved_headers, title = result[:3]
            info = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
            if resolver.get("builtin") == "browser" or info.get("media_type") == "hls":
                stream_headers = merge_for_media(target, src, resolved_headers, headers)
                # Select a media rendition from the master playlist with yt-dlp,
                # then probe downloaded bytes locally.
                stream = open_input.OpenInput(src, stream_headers)
                r = _from_ytdlp(stream, ytdlp, lambda url, hdrs: _probe_hls_sample(url, hdrs, probe),
                                scoped_media=True)
                if r.duration is None or r.fps is None:
                    raise ResolveError("無法讀取影片資訊", "HLS sample probe returned incomplete metadata")
                r.target, r.headers, r.title, r.site = target, headers, title, _site_of(target, "url")
            else:
                r = _from_probe(oi, src, {**resolved_headers, **headers}, title, probe)
            cover = info.get("cover")
            if isinstance(cover, str) and cover.startswith((
                    "data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
                r.cover = cover
            return r
        return _from_probe(oi, target, headers, Path(urlsplit(target).path).stem or target, probe)
    except ResolveError:
        raise
    except SystemExit as e:                       # source_resolve helpers may exit on bad input
        raise ResolveError("無法解析這個來源", str(e)) from e
    except Exception as e:  # noqa: BLE001 - yt-dlp / resolver failures come in many shapes
        raise ResolveError(_short_error(e), error_line(e)) from e


# ------------------------------------------------------------------------------------------------------------
# thumbnail
# ------------------------------------------------------------------------------------------------------------

def thumb_cmd(src: str, headers: dict, duration: Optional[float], out: str) -> list[str]:
    """The one-frame grab (queue_thumbs.grab_cmd: 320 px wide JPEG by input seeking), ~10 % into the video."""
    from oh_streamer.queue import queue_thumbs
    remote = source_resolve.is_url(src)
    http = source_resolve.FFmpegHttpOptions(headers or {}).cli_args() if remote else []
    seek = duration * THUMB_AT if duration else THUMB_FALLBACK_AT
    return queue_thumbs.grab_cmd(src, seek, out, http_args=http, remote=remote, playlist=_is_hls(src))


def grab_thumbnail(src: str, headers: Optional[dict], duration: Optional[float],
                   timeout: float = THUMB_TIMEOUT) -> Optional[str]:
    """A 'data:image/jpeg;base64,...' frame of the source, or None on any failure / after `timeout` seconds."""
    try:
        with tempfile.TemporaryDirectory(prefix="oh_intake_") as d:
            out = os.path.join(d, "t.jpg")
            result = subprocess.run(thumb_cmd(src, headers or {}, duration, out), capture_output=True,
                                    creationflags=NO_WINDOW, timeout=timeout, stdin=subprocess.DEVNULL)
            if result.returncode != 0:
                # Do not log stderr: CDN URLs may contain signed credentials.
                source_resolve.log(f"thumbnail grab failed (ffmpeg exit {result.returncode})")
                return None
            data = Path(out).read_bytes() if os.path.isfile(out) else b""
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        source_resolve.log(f"thumbnail grab failed ({type(e).__name__})")
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii") if data else None
