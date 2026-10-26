"""Local byte-range cache and HTTP server for a single progressive (plain HTTP) media file.

Counterpart of ``hls_cache`` for sources that are one big file (mp4/webm/mkv) served with
HTTP ``Range`` support.  The session *owns* up to two fetcher threads (a sequential/hole-filling connection and a
"priority" connection that only serves blocks a reader is blocked on) that download fixed-size blocks
(default 1 MiB) of the upstream file into one preallocated data file, hashing every block.
A loopback HTTP server serves ``GET/HEAD /source.<ext>`` with single-range support and, when a
reader asks for a block that is not ready yet, moves that block to the front of the fetch queue
and blocks until it is committed (so ffmpeg/mpv can seek anywhere, e.g. to a trailing ``moov``).

The queue-facing surface mirrors ``HLSSession``: ``playlist_path`` (here a small marker file),
``status_path`` (sidecar JSON, ``ready_segments``/``segment_count`` count blocks), ``identity``,
``generation``, ``status``, ``metadata``, ``runtime_url``, ``snapshot()``, ``set_metadata()``,
``set_status()``, ``cancel()``, ``close()``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from .hls_cache import _atomic_json, _load_sidecar, _parse_range, _replace_atomic

# 1 MiB: a block is the unit a blocked reader waits for, so on a slow upstream (0.35 MB/s) 4 MiB cost ~12 s per block.
# Sidecars written with another block_size (e.g. the former 4 MiB default) are resumed with their own size.
DEFAULT_BLOCK_SIZE = 1024 * 1024
DEFAULT_CONNECTIONS = 2                 # 1 = single sequential connection (for upstreams with a concurrency limit)
PRIORITY_STALE_SECONDS = 2.0            # a priority request nobody re-asked for within this long is dropped
MARKER_NAME = "source.rcache"
DATA_NAME = "source.data"
SIDE_CAR_NAME = "range-source.json"
EXPIRED_HTTP_CODES = (403, 404, 410)
_REGISTRY_LOCK = threading.RLock()
_REGISTRY: dict[str, "RangeSession"] = {}

# on_expired(http_code, old_url) -> (new_url, new_headers_or_None) | None   (None => give up)
ExpiredCallback = Callable[[int, str], "tuple[str, dict | None] | None"]
# progress(downloaded_bytes, total_bytes, ready_blocks, total_blocks)
ProgressCallback = Callable[[int, int, int, int], None]


class RangeCacheError(RuntimeError):
    """The source cannot safely be served or resumed from the local range cache."""


class _Cancelled(Exception):
    pass


class _Fatal(Exception):
    pass


def sidecar_path(playlist_path: str | os.PathLike[str]) -> Path:
    return Path(playlist_path).resolve().parent / SIDE_CAR_NAME


def _data_stat(path: Path) -> dict | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _block_count(total: int, block_size: int) -> int:
    return (total + block_size - 1) // block_size


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"\s*bytes\s+(\d+)-(\d+)/(\d+)\s*", value or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def probe_range_support(url: str, headers: dict | None = None, timeout: float = 15.0) -> dict | None:
    """Check ``Range: bytes=0-0`` -> 206 + Content-Range total.  Returns None if unsupported.

    On success returns ``{"total_bytes", "etag", "content_type", "final_url"}``.
    """
    req = urllib.request.Request(url, headers={**(headers or {}), "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 206:
                return None
            parsed = _parse_content_range(resp.headers.get("Content-Range"))
            if not parsed or parsed[0] != 0 or parsed[1] != 0 or parsed[2] <= 0:
                return None
            resp.read(2)
            return {
                "total_bytes": parsed[2],
                "etag": resp.headers.get("ETag") or None,
                "content_type": resp.headers.get("Content-Type") or None,
                "final_url": resp.geturl(),
            }
    except (OSError, ValueError, urllib.error.URLError):
        return None


class _LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _public_snapshot(data: dict) -> dict:
    snap = {key: copy.deepcopy(data.get(key)) for key in (
        "version", "identity", "generation", "status", "error", "playlist", "metadata",
        "ready_segments", "segment_count", "total_bytes", "block_size",
    )}
    snap["downloaded_bytes"] = sum(int(x.get("size") or 0) for x in data.get("ready_segments", []))
    return snap


class RangeSession:
    """A local view (and owner of the fetcher) of one persistent range-cache generation."""

    def __init__(self, marker: Path, data: dict, *, url: str | None = None, headers: dict | None = None,
                 on_expired: ExpiredCallback | None = None, progress: ProgressCallback | None = None,
                 start_server: bool = True, retries: int = 3, retry_delay: float = 0.5,
                 timeout: float = 20.0, connections: int = DEFAULT_CONNECTIONS):
        self.playlist_path = str(marker.resolve())
        self._marker = Path(self.playlist_path)
        self._data_path = self._marker.parent / DATA_NAME
        self.status_path = str(sidecar_path(self._marker))
        self.identity = str(data["identity"])
        self.generation = str(data["generation"])
        self.status = str(data["status"])
        self.total_bytes = int(data["total_bytes"])
        self.block_size = int(data["block_size"])
        self.block_count = int(data["segment_count"])
        self.ext = str(data.get("ext") or "mp4")
        self.metadata = copy.deepcopy(data.get("metadata") or {})
        self._data = data
        self._url = url
        self._headers = dict(headers or {})
        self._on_expired = on_expired
        self._progress = progress
        self._retries = retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._condition = threading.Condition(threading.RLock())
        self._io_lock = threading.Lock()
        self._closed = False
        self._cancel = threading.Event()
        self._ready: set[int] = {int(x["index"]) for x in data.get("ready_segments", [])}
        self._connections = 2 if int(connections) >= 2 else 1
        self._priority: dict[int, float] = {}      # block index -> last time a reader asked (insertion = FIFO)
        self._in_flight: set[int] = set()
        self._cursor = 0
        self._fetcher: threading.Thread | None = None            # sequential / hole-filling connection
        self._prio_fetcher: threading.Thread | None = None      # priority connection (only if connections >= 2)
        self._server: _LocalHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self.runtime_url = ""
        if start_server:
            self._start_server()

    # ------------------------------------------------------------------ upstream
    def set_upstream(self, url: str, headers: dict | None = None) -> None:
        """Replace the upstream URL (e.g. after the downloader re-resolved an expired link)."""
        with self._condition:
            self._url = url
            if headers is not None:
                self._headers = dict(headers)
            self._condition.notify_all()

    # ------------------------------------------------------------------ server
    def _start_server(self) -> None:
        session = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                self._serve(head=False)

            def do_HEAD(self):
                self._serve(head=True)

            def _send_error(self, status: int, message: str = "", extra: dict | None = None):
                body = (message or "source unavailable").encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body) if self.command != "HEAD" else 0))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _serve(self, *, head: bool):
                request_path = unquote(self.path.split("?", 1)[0]).lstrip("/")
                if request_path != f"source.{session.ext}":
                    self._send_error(404)
                    return
                length = session.total_bytes
                range_header = self.headers.get("Range")
                if range_header and "," in range_header:
                    range_header = None            # multi-range: ignore and serve the whole file
                start, end, partial = _parse_range(range_header, length)
                if start is None:
                    self._send_error(416, "invalid range", {"Content-Range": f"bytes */{length}"})
                    return
                bs = session.block_size
                first = start // bs
                if not head:
                    block = session._await_block(first)
                    if block is None:
                        self._send_error(503, "source block is not available")
                        return
                ctype = mimetypes.guess_type(f"x.{session.ext}")[0] or "application/octet-stream"
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                if head:
                    return
                try:
                    pos = start
                    while pos <= end:
                        idx = pos // bs
                        if idx != first:
                            block = session._await_block(idx)
                            if block is None:
                                self.close_connection = True
                                return
                        base = idx * bs
                        chunk = block[pos - base:min(end + 1, base + len(block)) - base]
                        self.wfile.write(chunk)
                        pos += len(chunk)
                        if not chunk:
                            return
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._server = _LocalHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self._server.server_address[:2]
        self.runtime_url = f"http://{host}:{port}/source.{self.ext}"
        self._server_thread = threading.Thread(target=self._server.serve_forever,
                                               name="range-cache-http", daemon=True)
        self._server_thread.start()

    # ------------------------------------------------------------------ block access
    def _block_span(self, index: int) -> tuple[int, int]:
        start = index * self.block_size
        return start, min(self.total_bytes, start + self.block_size)

    def _read_verified(self, index: int) -> bytes | None:
        entry = next((x for x in self._data.get("ready_segments", []) if x.get("index") == index), None)
        if not entry:
            return None
        start, end = self._block_span(index)
        try:
            with self._io_lock, self._data_path.open("rb") as f:
                f.seek(start)
                block = f.read(end - start)
        except OSError:
            block = b""
        if len(block) == end - start and entry.get("size") == len(block) and _digest(block) == entry.get("sha256"):
            return block
        return None

    def _invalidate(self, index: int) -> None:
        """An on-disk mutation is never served; downloading sessions simply fetch the block again."""
        self._ready.discard(index)
        self._data["ready_segments"] = [x for x in self._data.get("ready_segments", []) if x.get("index") != index]
        if self.status == "complete":
            self.status = "failed"
            self._data["status"] = "failed"
            self._data["error"] = f"cached block {index} failed verification"
        self._persist_locked()
        self._condition.notify_all()

    def _prioritize_locked(self, index: int) -> None:
        if index in self._ready or index in self._in_flight:
            return
        new = index not in self._priority
        self._priority[index] = time.monotonic()        # refreshed while a reader still waits
        if new:
            self._condition.notify_all()

    def _await_block(self, index: int) -> bytes | None:
        """Return the verified bytes of a block, fetching it with priority if needed."""
        with self._condition:
            while True:
                if self._closed or not 0 <= index < self.block_count:
                    return None
                if index in self._ready:
                    block = self._read_verified(index)
                    if block is not None:
                        return block
                    self._invalidate(index)
                if self.status in ("failed", "paused", "complete"):
                    return None
                self._prioritize_locked(index)
                self._condition.wait(0.25)

    def _persist_locked(self) -> None:
        _atomic_json(Path(self.status_path), self._data)

    # ------------------------------------------------------------------ commit (fetcher side)
    def _commit(self, index: int, payload: bytes) -> None:
        start, end = self._block_span(index)
        if len(payload) != end - start:
            raise RangeCacheError(f"block {index} has {len(payload)} bytes, expected {end - start}")
        with self._condition:
            if self.status != "downloading":
                raise RangeCacheError(f"cache session is {self.status}")
        with self._io_lock:
            with self._data_path.open("r+b") as f:
                f.seek(start)
                f.write(payload)
                f.flush()
        entry = {"index": index, "size": len(payload), "sha256": _digest(payload)}
        with self._condition:
            ready = [x for x in self._data.get("ready_segments", []) if x.get("index") != index]
            ready.append(entry)
            ready.sort(key=lambda x: x["index"])
            self._data["ready_segments"] = ready
            self._ready.add(index)
            self._persist_locked()
            self._condition.notify_all()
            snap = (sum(x["size"] for x in ready), len(ready))
        if self._progress:
            try:
                self._progress(snap[0], self.total_bytes, snap[1], self.block_count)
            except Exception:
                pass

    # ------------------------------------------------------------------ fetcher
    def start_fetcher(self) -> None:
        with self._condition:
            if any(t and t.is_alive() for t in (self._fetcher, self._prio_fetcher)):
                return
            if self.status != "downloading" or self._closed:
                return
            self._cancel.clear()
            self._fetcher = threading.Thread(target=self._fetch_loop, args=(False,),
                                             name="range-cache-fetch", daemon=True)
            self._fetcher.start()
            if self._connections >= 2:
                self._prio_fetcher = threading.Thread(target=self._fetch_loop, args=(True,),
                                                      name="range-cache-fetch-prio", daemon=True)
                self._prio_fetcher.start()

    def _pop_priority_locked(self) -> int | None:
        """Claim the oldest block a reader is still waiting for (stale / ready / already claimed ones are dropped)."""
        now = time.monotonic()
        while self._priority:
            i = next(iter(self._priority))
            asked = self._priority.pop(i)
            if i in self._ready or i in self._in_flight or now - asked > PRIORITY_STALE_SECONDS:
                continue
            self._cursor = i + 1            # sequential fetching continues after the jump target
            self._in_flight.add(i)
            return i
        return None

    def _next_block(self) -> int | None:
        with self._condition:
            i = self._pop_priority_locked()
            if i is not None:
                return i
            for i in list(range(self._cursor, self.block_count)) + list(range(0, min(self._cursor, self.block_count))):
                if i not in self._ready and i not in self._in_flight:
                    self._cursor = i
                    self._in_flight.add(i)
                    return i
            return None

    def _next_priority(self) -> int | None:
        with self._condition:
            while not self._cancel.is_set() and not self._closed:
                i = self._pop_priority_locked()
                if i is not None:
                    return i
                if len(self._ready) >= self.block_count:
                    return None
                self._condition.wait(0.25)
            return None

    def _fetch_loop(self, priority_only: bool = False) -> None:
        idx: int | None = None
        try:
            while not self._cancel.is_set() and not self._closed:
                idx = self._next_priority() if priority_only else self._next_block()
                if idx is None:
                    if priority_only:
                        return
                    with self._condition:
                        if self.status == "downloading" and len(self._ready) == self.block_count:
                            self.set_status("complete")
                            return
                        self._condition.wait(0.25)      # the other connection still holds the last block(s)
                    continue
                payload = self._download_block(idx)
                self._commit(idx, payload)
                with self._condition:
                    self._in_flight.discard(idx)
                    idx = None
        except _Cancelled:
            return
        except (_Fatal, RangeCacheError, OSError) as exc:
            if self.status == "downloading" and not self._cancel.is_set():
                fail_session(self, exc)
        except Exception as exc:  # pragma: no cover - defensive
            fail_session(self, f"unexpected fetcher error: {exc}")
        finally:
            with self._condition:
                if idx is not None:
                    self._in_flight.discard(idx)
                self._condition.notify_all()

    def _download_block(self, index: int) -> bytes:
        start, end = self._block_span(index)
        want = end - start
        attempts = 0
        expirations = 0
        while True:
            if self._cancel.is_set() or self._closed:
                raise _Cancelled()
            with self._condition:
                url, headers = self._url, dict(self._headers)
            if not url:
                raise _Fatal("no upstream URL")
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    if resp.status != 206:
                        raise _Fatal(f"upstream ignored Range (HTTP {resp.status})")
                    cr = _parse_content_range(resp.headers.get("Content-Range"))
                    if cr != (start, end - 1, self.total_bytes):
                        raise _Fatal(f"upstream Content-Range mismatch: {resp.headers.get('Content-Range')!r}")
                    parts, got = [], 0
                    while got < want:
                        if self._cancel.is_set() or self._closed:
                            raise _Cancelled()
                        chunk = resp.read(min(64 * 1024, want - got))
                        if not chunk:
                            break
                        parts.append(chunk)
                        got += len(chunk)
                    if got != want:
                        raise OSError(f"short read: {got} of {want} bytes")
                    return b"".join(parts)
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code in EXPIRED_HTTP_CODES:
                    expirations += 1
                    with self._condition:
                        renewed_meanwhile = self._url != url        # the other connection already renewed it
                    if renewed_meanwhile and expirations <= 3:
                        continue
                    if self._on_expired is None or expirations > 3:
                        raise _Fatal(f"upstream URL expired (HTTP {code})")
                    try:
                        renewed = self._on_expired(code, url)
                    except Exception as cb_exc:
                        raise _Fatal(f"re-resolving expired URL failed: {cb_exc}")
                    if not renewed:
                        raise _Fatal(f"upstream URL expired (HTTP {code}) and could not be renewed")
                    self.set_upstream(renewed[0], renewed[1])
                    continue
                if code == 416:
                    raise _Fatal("upstream rejected the requested range (HTTP 416)")
                attempts += 1
                if attempts > self._retries:
                    raise _Fatal(f"upstream HTTP {code}")
            except (_Fatal, _Cancelled):
                raise
            except (OSError, urllib.error.URLError, ValueError) as exc:
                attempts += 1
                if attempts > self._retries:
                    raise _Fatal(f"upstream download failed: {exc}")
            if self._cancel.wait(self._retry_delay * attempts):
                raise _Cancelled()

    # ------------------------------------------------------------------ queue-facing API
    def set_metadata(self, metadata: dict) -> None:
        with self._condition:
            self.metadata = copy.deepcopy(metadata)
            self._data["metadata"] = copy.deepcopy(metadata)
            self._persist_locked()
            self._condition.notify_all()

    def set_status(self, status: str, error: str | None = None) -> None:
        if status not in ("downloading", "complete", "failed", "paused"):
            raise ValueError(f"unknown cache status: {status}")
        with self._condition:
            self.status = status
            self._data["status"] = status
            self._data["error"] = error
            if status == "complete":
                # Lets a later open_session trust the data file (verify each block when it is served) instead of
                # re-hashing the whole file, as long as its size and mtime are unchanged.
                stat = _data_stat(self._data_path)
                if stat:
                    self._data["data_stat"] = stat
            else:
                self._data.pop("data_stat", None)
            self._persist_locked()
            self._condition.notify_all()
        if status != "downloading":
            self._cancel.set()

    def snapshot(self) -> dict:
        with self._condition:
            return _public_snapshot(self._data)

    def wait(self, timeout: float | None = None) -> str:
        """Block until the session leaves ``downloading`` (or timeout); returns the status."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self.status == "downloading" and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(0.1 if remaining is None else min(0.1, remaining))
            return self.status

    def cancel(self) -> None:
        self._cancel.set()
        if self.status == "downloading":
            self.set_status("paused", "download cancelled")

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self.status == "downloading":
                self.status = "paused"
                self._data["status"] = "paused"
                self._data["error"] = "cache session closed while downloading"
                self._persist_locked()
            self._closed = True
            self._cancel.set()
            self._condition.notify_all()
        for fetcher in (self._fetcher, self._prio_fetcher):
            if fetcher and fetcher is not threading.current_thread():
                fetcher.join(timeout=3)
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread and self._server_thread is not threading.current_thread():
            self._server_thread.join(timeout=2)


# ---------------------------------------------------------------------- module-level helpers
def _valid_sidecar(data: dict, marker: Path) -> bool:
    return (data.get("version") == 1 and isinstance(data.get("identity"), str) and
            isinstance(data.get("generation"), str) and data.get("playlist") == str(marker.resolve()) and
            data.get("status") in ("downloading", "complete", "failed", "paused") and
            isinstance(data.get("ready_segments"), list) and
            isinstance(data.get("total_bytes"), int) and isinstance(data.get("block_size"), int) and
            data.get("block_size", 0) > 0 and data.get("segment_count") ==
            _block_count(data.get("total_bytes", 0), data.get("block_size", 1)))


def _verified_ready(data: dict, marker: Path) -> list[dict]:
    """Hash every recorded block of the data file; keep only those that still match."""
    data_path = marker.parent / DATA_NAME
    total, bs = data["total_bytes"], data["block_size"]
    verified = []
    try:
        with data_path.open("rb") as f:
            for item in data.get("ready_segments", []):
                idx = item.get("index") if isinstance(item, dict) else None
                if not isinstance(idx, int) or not 0 <= idx < data["segment_count"]:
                    continue
                start = idx * bs
                f.seek(start)
                block = f.read(min(total, start + bs) - start)
                if (len(block) == item.get("size") == min(total, start + bs) - start
                        and _digest(block) == item.get("sha256")):
                    verified.append(item)
    except OSError:
        return []
    return sorted(verified, key=lambda x: x["index"])


def _lazily_trusted(data: dict, marker: Path) -> bool:
    """A complete cache whose data file is unchanged since it was completed (same size and mtime_ns as recorded in
    the sidecar, one entry of the right size for every block) is reopened without hashing it. Large files would
    otherwise stall the scheduler thread for seconds. Nothing weaker is served: the local server verifies each
    block's sha256 when a reader reaches it, and a mismatch fails the session (_invalidate)."""
    stat = data.get("data_stat")
    if not isinstance(stat, dict) or stat != _data_stat(marker.parent / DATA_NAME):
        return False
    total, bs, count = data["total_bytes"], data["block_size"], data["segment_count"]
    if stat.get("size") != total:
        return False
    entries = data.get("ready_segments", [])
    if len(entries) != count:
        return False
    for i, item in enumerate(sorted((x for x in entries if isinstance(x, dict)), key=lambda x: x.get("index", -1))):
        if item.get("index") != i or item.get("size") != min(total, (i + 1) * bs) - i * bs or not item.get("sha256"):
            return False
    return True


def active_session(playlist_path: str | os.PathLike[str]) -> RangeSession | None:
    key = str(Path(playlist_path).resolve())
    with _REGISTRY_LOCK:
        session = _REGISTRY.get(key)
        return session if session and not session._closed else None


def get_session(playlist_path: str | os.PathLike[str]) -> RangeSession | None:
    return active_session(playlist_path)


def open_session(playlist_path: str | os.PathLike[str]) -> RangeSession | None:
    """Rebuild an HTTP server only for a verified complete on-disk cache."""
    marker = Path(playlist_path).resolve()
    key = str(marker)
    with _REGISTRY_LOCK:
        current = _REGISTRY.get(key)
        if current and not current._closed and current.status == "complete":
            return current
        data = _load_sidecar(sidecar_path(marker))
        if not data or not _valid_sidecar(data, marker) or data.get("status") != "complete" or not marker.is_file():
            return None
        if not _lazily_trusted(data, marker):
            verified = _verified_ready(data, marker)
            if len(verified) != data["segment_count"] or len(verified) != len(data["ready_segments"]):
                return None
        try:
            session = RangeSession(marker, data)
        except OSError:
            return None
        _REGISTRY[key] = session
        return session


def close_session(playlist_path: str | os.PathLike[str]) -> None:
    key = str(Path(playlist_path).resolve())
    with _REGISTRY_LOCK:
        session = _REGISTRY.pop(key, None)
    if session:
        session.close()


def prepare_session(cache_dir: str | os.PathLike[str], *, url: str, total_bytes: int, request_key: str,
                    headers: dict | None = None, metadata: dict | None = None, ext: str = "mp4",
                    etag: str | None = None, block_size: int | None = None,
                    on_expired: ExpiredCallback | None = None, progress: ProgressCallback | None = None,
                    auto_start: bool = True, retries: int = 3, retry_delay: float = 0.5,
                    timeout: float = 20.0, connections: int = DEFAULT_CONNECTIONS) -> tuple[RangeSession, list[int]]:
    """Create/resume one generation and (by default) start the fetcher thread.

    ``request_key`` identifies the selected source (page URL + format); with ``total_bytes``,
    ``block_size`` and (when known on both sides) ``etag`` it decides whether persisted blocks may
    be reused.  A mismatch refuses to mix generations (RangeCacheError); the caller wipes the folder.
    Returns the session and the verified ready block indices.
    """
    if total_bytes <= 0 or (block_size is not None and block_size <= 0):
        raise RangeCacheError("invalid total size or block size")
    ext = re.sub(r"[^A-Za-z0-9]", "", ext) or "mp4"
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / MARKER_NAME
    status_file = sidecar_path(marker)

    with _REGISTRY_LOCK:
        current = _REGISTRY.get(str(marker))
        if current and not current._closed and current.status == "downloading":
            raise RangeCacheError("a download for this cache is already active")
        old = _load_sidecar(status_file)
        if block_size is None:      # not forced: keep the block size an existing cache was written with
            block_size = old["block_size"] if old and _valid_sidecar(old, marker) else DEFAULT_BLOCK_SIZE
        count = _block_count(total_bytes, block_size)
        ident = {"request_key": request_key, "total_bytes": total_bytes, "block_size": block_size, "ext": ext}
        ready: list[dict] = []
        identity = str(uuid.uuid4())
        if old and _valid_sidecar(old, marker):
            same = all(old.get(k) == v for k, v in ident.items())
            if same and etag and old.get("etag") and old["etag"] != etag:
                same = False
            if not same:
                old["generation"] = str(uuid.uuid4())
                old["status"] = "failed"
                old["error"] = "selected source identity changed; refusing to mix cache generations"
                _atomic_json(status_file, old)
                raise RangeCacheError(old["error"])
            identity = old["identity"]
            if old.get("status") == "complete":
                existing = open_session(marker)
                if existing:
                    return existing, [x["index"] for x in existing._data["ready_segments"]]
            ready = _verified_ready(old, marker)
        popped = _REGISTRY.pop(str(marker), None)
        if popped:
            popped.close()

        data_path = directory / DATA_NAME
        if not data_path.is_file() or data_path.stat().st_size != total_bytes:
            ready = []
            with data_path.open("wb") as f:
                f.truncate(total_bytes)
        marker_tmp = marker.with_name(marker.name + ".tmp-" + uuid.uuid4().hex)
        try:
            marker_tmp.write_text(json.dumps({"format": "range-cache", "version": 1, "data": DATA_NAME}) + "\n",
                                  encoding="utf-8", newline="\n")
            _replace_atomic(marker_tmp, marker)
        finally:
            marker_tmp.unlink(missing_ok=True)

        data = {
            "version": 1, "identity": identity, "generation": str(uuid.uuid4()),
            "status": "downloading", "error": None, "playlist": str(marker),
            "metadata": copy.deepcopy(metadata) if isinstance(metadata, dict) else {},
            "ready_segments": ready, "segment_count": count,
            "total_bytes": total_bytes, "block_size": block_size, "etag": etag, **ident,
        }
        if not data["metadata"] and old and isinstance(old.get("metadata"), dict):
            data["metadata"] = copy.deepcopy(old["metadata"])
        _atomic_json(status_file, data)
        try:
            session = RangeSession(marker, data, url=url, headers=headers, on_expired=on_expired,
                                   progress=progress, retries=retries, retry_delay=retry_delay,
                                   timeout=timeout, connections=connections)
        except OSError as exc:
            data["status"] = "failed"
            data["error"] = f"could not start local range cache server: {exc}"[:500]
            _atomic_json(status_file, data)
            raise RangeCacheError(data["error"]) from exc
        _REGISTRY[str(marker)] = session
    if len(ready) == count:
        complete_session(session)
    elif auto_start:
        session.start_fetcher()
    return session, [x["index"] for x in session._data.get("ready_segments", [])]


def commit_block(session: RangeSession, index: int, data: bytes) -> None:
    """Publish one fully downloaded block (normally done by the session's own fetcher)."""
    session._commit(index, bytes(data))


def complete_session(session: RangeSession) -> None:
    if len(session._data.get("ready_segments", [])) != session._data.get("segment_count"):
        raise RangeCacheError("cannot complete a cache with missing blocks")
    session.set_status("complete")


def fail_session(session: RangeSession, error: BaseException | str, *, paused: bool = False) -> None:
    session.set_status("paused" if paused else "failed", str(error)[:500])
