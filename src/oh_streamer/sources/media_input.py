"""Resolve a queue item's ``source_path`` to the string ffmpeg/ffprobe should read.

A range_cache item's ``source_path`` is ``<item_dir>/range-cache/source.rcache``, a tiny marker file and not
media. The readable source is the session's loopback ``runtime_url`` (Range-capable; unfinished blocks block
until fetched). HLS items keep a real local ``.m3u8`` and need no translation. Every reader of
``source_path`` audio goes through ``ffmpeg_input`` so the suffix test lives in one place.

A streaming item's subtitle audio is read straight from the remote site: its ``source_path`` is an
``audio.rstream`` marker (JSON: signed url, headers, generation, identity). ``ffmpeg_input`` turns it into the url
plus the HTTP options; when the signed url expires the reader calls ``run_remote`` which asks the registered
refresher (same process) to rewrite the marker and retries once. The marker url is a secret: no message or log
line in this module (or the readers that follow its contract) may contain it, use ``<remote>``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

RUNTIME_SUFFIXES = (".rcache",)
REMOTE_SUFFIXES = (".rstream",)
REMOTE_FORMAT = "oh-stream-audio-1"
# Minimum seconds between two re-resolves of the same marker (the resolver is a network round trip, and a
# permanently dead item must not hammer the site).
REFRESH_MIN_INTERVAL_S = 60.0
REMOTE_RW_TIMEOUT_US = "30000000"
# ffmpeg >= 7.1 refuses HLS segments whose name has no media extension (Some media servers serve its MPEG-TS as video0.jpeg).
HLS_ANY_EXTENSION_ARGS = ["-extension_picky", "0"]

# Reconnect on dropped loopback reads; -rw_timeout (microseconds) bounds a read stuck on a block that never arrives.
HTTP_INPUT_ARGS = ("-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                   "-rw_timeout", "120000000")



class SourceUnavailable(RuntimeError):
    """The range-cache session behind a marker is not live (paused, finished and recycled, or closed). Retryable:
    callers release the work and wait for the owner to reopen the session; it is not a failure of the media."""


_lock = threading.Lock()
_registry: dict[str, str] = {}


def _clean(path) -> str:
    return os.fspath(path).split("?", 1)[0].split("#", 1)[0]


def _key(path) -> str:
    return os.path.normcase(os.path.abspath(_clean(path)))


def needs_runtime_url(source) -> bool:
    """True when ``source`` is a range_cache marker that must be read through its runtime_url."""
    if source is None:
        return False
    try:
        return Path(_clean(source)).suffix.lower() in RUNTIME_SUFFIXES
    except (TypeError, ValueError):
        return False


def register_runtime_url(source, url: str | None) -> None:
    """Remember the runtime_url of a marker for in-process readers (pass None to forget it)."""
    with _lock:
        if url:
            _registry[_key(source)] = url
        else:
            _registry.pop(_key(source), None)


def runtime_url_for(source, source_url: str | None = None) -> str:
    """The runtime_url for a marker: explicit value, else registered, else the live in-process session."""
    if source_url:
        return source_url
    with _lock:
        url = _registry.get(_key(source))
    if url:
        return url
    try:
        from oh_streamer.sources import range_cache
        session = range_cache.active_session(_clean(source))
        url = getattr(session, "runtime_url", "") if session is not None else ""
    except Exception:  # noqa: BLE001
        url = ""
    if not url:
        raise SourceUnavailable(f"no runtime_url for range-cache source {source}")
    return url


def ffmpeg_input(source, source_url: str | None = None) -> tuple[str, list[str]]:
    """(input string, ffmpeg options to place before ``-i``). Ordinary paths/URLs pass through unchanged."""
    if needs_runtime_url(source):
        return runtime_url_for(source, source_url), list(HTTP_INPUT_ARGS)
    if is_remote_marker(source):
        from oh_streamer.sources.source_resolve import FFmpegHttpOptions
        marker = read_remote_marker(source)
        opts = [*FFmpegHttpOptions(marker["headers"]).cli_args(), "-rw_timeout", REMOTE_RW_TIMEOUT_US]
        if marker.get("hls"):
            opts += HLS_ANY_EXTENSION_ARGS
        return marker["url"], opts
    return os.fspath(source), []


# --------------------------------------------------------------------------------------------------
# Remote audio marker (``audio.rstream``) and signed-url expiry
# --------------------------------------------------------------------------------------------------

T = TypeVar("T")


class RemoteMarkerError(RuntimeError):
    """The remote audio marker is missing or malformed (message never contains the url)."""


class RemoteExpired(RuntimeError):
    """A remote read failed because the signed url expired (403/404/410). ``generation`` is the marker generation
    the failed read used; ``run_remote`` fills in the one it saw before the attempt when the raiser leaves None."""

    def __init__(self, message: str = "字幕音訊網址已過期", generation: int | None = None):
        super().__init__(message)
        self.generation = generation


def is_remote_marker(source) -> bool:
    """True when ``source`` is a remote audio marker (``*.rstream``)."""
    if source is None:
        return False
    try:
        return Path(_clean(source)).suffix.lower() in REMOTE_SUFFIXES
    except (TypeError, ValueError):
        return False


def _retry_os(fn, attempts: int = 8):
    """Windows briefly denies access to a file being replaced/read by another thread; retry a few times."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.02 * (i + 1))


def _validate_marker(data, name: str) -> dict:
    if not isinstance(data, dict):
        raise RemoteMarkerError(f"remote audio marker {name} is not a JSON object")
    if data.get("format") != REMOTE_FORMAT:
        raise RemoteMarkerError(f"remote audio marker {name} has an unknown format")
    url = data.get("url")
    if not isinstance(url, str) or not url:
        raise RemoteMarkerError(f"remote audio marker {name} has no url")
    headers = data.get("headers")
    if headers is None:
        headers = {}
    if not isinstance(headers, dict):
        raise RemoteMarkerError(f"remote audio marker {name} has invalid headers")
    gen = data.get("generation", 0)
    if isinstance(gen, bool) or not isinstance(gen, int):
        raise RemoteMarkerError(f"remote audio marker {name} has an invalid generation")
    return dict(data, headers={str(k): str(v) for k, v in headers.items()}, generation=gen)


def read_remote_marker(source) -> dict:
    """Parse a ``.rstream`` marker. Raises ``RemoteMarkerError`` (never echoing the url) when missing or bad."""
    path = Path(_clean(source))
    try:
        raw = _retry_os(lambda: path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RemoteMarkerError(f"remote audio marker {path.name} is unreadable") from None
    try:
        data = json.loads(raw)
    except ValueError:
        raise RemoteMarkerError(f"remote audio marker {path.name} is not valid JSON") from None
    return _validate_marker(data, path.name)


def write_remote_marker(path, data: dict) -> None:
    """Atomically write a marker (tmp file + ``os.replace``). ``format``/``headers``/``generation`` default in."""
    dest = Path(_clean(path))
    doc = dict(data)
    doc.setdefault("format", REMOTE_FORMAT)
    doc.setdefault("headers", {})
    doc.setdefault("generation", 0)
    doc = _validate_marker(doc, dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(doc, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _retry_os(lambda: os.replace(tmp, dest))
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


_EXPIRED_RE = re.compile(r"HTTP error (?:403|404|410)\b|Server returned (?:403|404)\b")
_FAILED_RE = re.compile(
    r"Failed to open segment|failed too many times|HTTP error \d{3}\b|Server returned (?:\d{3}|[45]XX)\b")


def remote_failure(stderr: str | None) -> str | None:
    """Classify ffmpeg/ffprobe stderr (run with ``-loglevel warning`` / ``-v warning``): ``"expired"`` for an
    expired signed url (403/404/410), ``"failed"`` for other remote read problems (a skipped HLS segment exits 0
    with shorter audio, so the caller must check this even on success), ``None`` when clean."""
    if not stderr:
        return None
    if _EXPIRED_RE.search(stderr):
        return "expired"
    if _FAILED_RE.search(stderr):
        return "failed"
    return None


_remote_lock = threading.Lock()
_refreshers: dict[str, Callable[[str], None]] = {}
_marker_locks: dict[str, threading.Lock] = {}
_last_refresh: dict[str, float] = {}
_monotonic = time.monotonic


def register_refresher(marker, fn: Callable[[str], None]) -> None:
    """Register ``fn(marker) -> None`` which re-resolves the item, rewrites the marker and bumps its generation."""
    with _remote_lock:
        _refreshers[_key(marker)] = fn


def unregister_refresher(marker) -> None:
    k = _key(marker)
    with _remote_lock:
        _refreshers.pop(k, None)
        _marker_locks.pop(k, None)
        _last_refresh.pop(k, None)


def refresh_remote(marker, seen_generation: int) -> None:
    """Replace an expired url once. One lock per marker: a thread that arrives after another already swapped the
    url (marker generation > ``seen_generation``) returns at once. Without a refresher, or within
    ``REFRESH_MIN_INTERVAL_S`` of the previous attempt, raises ``RuntimeError``."""
    k = _key(marker)
    with _remote_lock:
        lock = _marker_locks.setdefault(k, threading.Lock())
    with lock:
        if read_remote_marker(marker)["generation"] > seen_generation:
            return
        with _remote_lock:
            fn = _refreshers.get(k)
            last = _last_refresh.get(k)
        if fn is None:
            raise RuntimeError("字幕音訊網址已過期，且沒有可用的重新解析方式")
        now = _monotonic()
        if last is not None and now - last < REFRESH_MIN_INTERVAL_S:
            raise RuntimeError("字幕音訊網址剛重新解析過，稍後再試")
        with _remote_lock:
            _last_refresh[k] = now
        fn(os.fspath(marker))


def run_remote(source, attempt: Callable[[], T]) -> T:
    """Run ``attempt()``; for a remote marker, a ``RemoteExpired`` triggers one ``refresh_remote`` and one retry."""
    if not is_remote_marker(source):
        return attempt()
    seen = read_remote_marker(source)["generation"]
    try:
        return attempt()
    except RemoteExpired as exc:
        if exc.generation is not None:
            seen = exc.generation
    refresh_remote(source, seen)
    try:
        return attempt()
    except RemoteExpired:
        raise RuntimeError("字幕音訊網址已失效，重新解析後仍無法讀取") from None
