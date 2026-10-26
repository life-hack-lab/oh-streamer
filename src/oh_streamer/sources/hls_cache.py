"""Local, shared cache and HTTP server for progressive HLS VOD sources.

The downloader owns every upstream request.  This module only serves the local
playlist and already completed segment files, waiting for a missing segment to
be committed by the downloader before responding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


SIDE_CAR_NAME = "hls-source.json"
PLAYLIST_NAME = "playlist.m3u8"
_REGISTRY_LOCK = threading.RLock()
_REGISTRY: dict[str, "HLSSession"] = {}


class HLSCacheError(RuntimeError):
    """The source cannot safely be served or resumed from the local cache."""


@dataclass(frozen=True)
class Segment:
    index: int
    duration: float
    url: str
    filename: str
    # AES-128 only: key URI (absolute) and the resolved 16-byte IV as 32 hex digits.
    # The cache always stores decrypted segments, so these never reach the local playlist.
    key_uri: str | None = None
    iv: str | None = None


@dataclass(frozen=True)
class MediaPlaylist:
    source_url: str
    final_url: str
    raw: bytes
    segments: tuple[Segment, ...]
    duration: float
    fingerprint: str
    local_text: str

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "final_url": self.final_url,
            "manifest_sha256": self.fingerprint,
            "segments": [
                {"index": s.index, "duration": s.duration, "url": s.url, "filename": s.filename,
                 **({"key_uri": s.key_uri, "iv": s.iv} if s.key_uri else {})}
                for s in self.segments
            ],
        }


def _descriptor_duration(source_descriptor: dict | None) -> float | None:
    segments = source_descriptor.get("segments") if isinstance(source_descriptor, dict) else None
    try:
        duration = sum(float(segment["duration"]) for segment in segments)
    except (TypeError, ValueError, KeyError):
        return None
    return duration if duration > 0 else None


def _descriptors_match(old: dict | None, new: dict) -> bool:
    if not isinstance(old, dict):
        return False
    if old == new:
        return True
    # CDNs rotate signed tokens (query strings, hosts, token path prefixes) between fetches, so
    # manifest hash and URLs differ on resume even though the media is identical.  The stream
    # identity is the request key plus each segment's index, duration, IV and URL basename.
    if old.get("request_key") != new.get("request_key"):
        return False
    return _segment_identity(old.get("segments")) == _segment_identity(new.get("segments"))


def _segment_identity(segments) -> list | None:
    import urllib.parse

    if not isinstance(segments, list):
        return None
    result = []
    for s in segments:
        if not isinstance(s, dict):
            return None
        name = urllib.parse.urlsplit(str(s.get("url", ""))).path.rsplit("/", 1)[-1]
        result.append((s.get("index"), s.get("duration"), s.get("filename"), name, s.get("iv")))
    return result


def normalize_metadata(metadata: dict | None, duration: float | None) -> dict:
    """Normalize probe timing to the full playlist wherever a cache session is created."""
    result = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
    if duration is None or duration <= 0:
        return result
    duration_text = f"{duration:.6f}"
    fmt = result.get("format")
    if not isinstance(fmt, dict):
        fmt = {}
        result["format"] = fmt
    fmt["duration"] = duration_text
    streams = result.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and stream.get("codec_type") == "video":
                stream["duration"] = duration_text
                stream.pop("nb_frames", None)
    return result


def parse_media_playlist(raw: bytes | str, source_url: str, final_url: str | None = None) -> MediaPlaylist:
    """Parse one strict, complete media playlist and reject unsupported HLS features.

    Master playlists, live/event playlists, SAMPLE-AES/DRM keys, maps/fMP4, byte
    ranges, discontinuities, and low-latency parts are deliberately outside this
    cache.  Standard AES-128 (full-segment CBC) is accepted: each segment records
    its key URI and IV so the downloader can decrypt it before caching.
    """
    import urllib.parse

    payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HLSCacheError("playlist is not UTF-8") from exc
    lines = [line.strip() for line in text.replace("\r", "").split("\n") if line.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise HLSCacheError("not an HLS playlist")

    unsupported = (
        "#EXT-X-STREAM-INF", "#EXT-X-I-FRAME-STREAM-INF", "#EXT-X-MEDIA:",
        "#EXT-X-SESSION-KEY:", "#EXT-X-MAP:", "#EXT-X-BYTERANGE:",
        "#EXT-X-DISCONTINUITY", "#EXT-X-PART:", "#EXT-X-PRELOAD-HINT:",
        "#EXT-X-RENDITION-REPORT:", "#EXT-X-DEFINE:",
    )
    found = sorted({tag.rstrip(":") for line in lines for tag in unsupported if line.upper().startswith(tag)})
    if found:
        raise HLSCacheError(f"playlist uses an unsupported HLS feature ({', '.join(found)})")
    if any(line.upper().startswith("#EXT-X-PLAYLIST-TYPE:EVENT") for line in lines):
        raise HLSCacheError("event playlists are not cacheable")
    if "#EXT-X-ENDLIST" not in {line.upper() for line in lines}:
        raise HLSCacheError("playlist is not complete (missing ENDLIST)")

    base_url = final_url or source_url
    pending_duration: float | None = None
    ended = False
    media_sequence = 0
    key_uri: str | None = None
    key_iv: bytes | None = None
    entries: list[tuple[float, str, str | None, bytes | None]] = []
    for line in lines[1:]:
        upper = line.upper()
        if upper.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_sequence = int(line.split(":", 1)[1].strip())
                if media_sequence < 0:
                    raise ValueError
            except ValueError as exc:
                raise HLSCacheError("invalid EXT-X-MEDIA-SEQUENCE") from exc
            continue
        if upper.startswith("#EXT-X-KEY:"):
            key_uri, key_iv = _parse_key_tag(line[len("#EXT-X-KEY:"):], base_url)
            continue
        if upper == "#EXT-X-ENDLIST":
            if pending_duration is not None:
                raise HLSCacheError("EXTINF has no media segment")
            ended = True
            continue
        if upper.startswith("#EXTINF:"):
            if pending_duration is not None or ended:
                raise HLSCacheError("malformed EXTINF sequence")
            value = line[len("#EXTINF:"):].split(",", 1)[0].strip()
            try:
                import math
                pending_duration = float(value)
                if not math.isfinite(pending_duration) or pending_duration <= 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise HLSCacheError("invalid EXTINF duration") from exc
            continue
        if line.startswith("#"):
            continue
        if ended or pending_duration is None:
            raise HLSCacheError("media URI without a matching EXTINF")
        uri = urllib.parse.urljoin(base_url, line)
        parsed = urllib.parse.urlsplit(uri)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HLSCacheError("segment URI is not HTTP(S)")
        entries.append((pending_duration, uri, key_uri, key_iv))
        pending_duration = None
    if not ended or pending_duration is not None or not entries:
        raise HLSCacheError("playlist is incomplete or has no media segments")

    segments = []
    for i, (dur, url, k_uri, k_iv) in enumerate(entries):
        iv_hex = None
        if k_uri:
            # RFC 8216 4.3.2.4: without an IV attribute the IV is the media sequence number, big-endian.
            iv_hex = (k_iv or (media_sequence + i).to_bytes(16, "big")).hex()
        segments.append(Segment(i, dur, url, f"segment-{i:06d}.ts", k_uri, iv_hex))
    segments = tuple(segments)
    duration = sum(s.duration for s in segments)
    output = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:%d" % max(1, int(max(s.duration for s in segments) + 0.999999))]
    for segment in segments:
        output.extend((f"#EXTINF:{segment.duration:.6f},", segment.filename))
    output.append("#EXT-X-ENDLIST")
    local_text = "\n".join(output) + "\n"
    return MediaPlaylist(source_url, base_url, payload, segments, duration,
                         hashlib.sha256(local_text.encode("utf-8")).hexdigest(), local_text)


def _parse_key_tag(value: str, base_url: str) -> tuple[str | None, bytes | None]:
    """Return (absolute key URI, explicit IV) for an EXT-X-KEY, or (None, None) for METHOD=NONE."""
    import urllib.parse

    attrs = {k.upper(): v[1:-1] if v.startswith('"') else v
             for k, v in re.findall(r'([A-Za-z0-9-]+)=("[^"]*"|[^,]*)', value)}
    method = attrs.get("METHOD", "").upper()
    if method == "NONE":
        return None, None
    if method != "AES-128":
        raise HLSCacheError(f"playlist uses an unsupported HLS feature (EXT-X-KEY METHOD={method or '?'})")
    if attrs.get("KEYFORMAT", "identity") != "identity" or attrs.get("KEYFORMATVERSIONS", "1") != "1":
        raise HLSCacheError("playlist uses an unsupported HLS feature (EXT-X-KEY KEYFORMAT)")
    uri = attrs.get("URI")
    if not uri:
        raise HLSCacheError("EXT-X-KEY has no URI")
    uri = urllib.parse.urljoin(base_url, uri)
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HLSCacheError("key URI is not HTTP(S)")
    iv = None
    if "IV" in attrs:
        if not re.fullmatch(r"0[xX][0-9A-Fa-f]{1,32}", attrs["IV"]):
            raise HLSCacheError("invalid EXT-X-KEY IV")
        iv = int(attrs["IV"], 16).to_bytes(16, "big")
    return uri, iv


def _aes128_cbc_decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """AES-128-CBC without unpadding, using whichever library is already installed."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
        return dec.update(data) + dec.finalize()
    except ImportError:
        pass
    try:
        from Cryptodome.Cipher import AES
        return AES.new(key, AES.MODE_CBC, iv).decrypt(data)
    except ImportError:
        pass
    from yt_dlp.aes import aes_cbc_decrypt_bytes   # pure Python, slow but always present with yt-dlp
    return aes_cbc_decrypt_bytes(data, key, iv)


def decrypt_segment(data: bytes, key: bytes, iv_hex: str) -> bytes:
    """Decrypt one AES-128 HLS segment (CBC, PKCS7) and sanity-check that it is MPEG-TS."""
    if len(key) != 16:
        raise HLSCacheError("HLS AES-128 key is not 16 bytes")
    if not data or len(data) % 16:
        raise HLSCacheError("encrypted HLS segment length is not a multiple of 16")
    plain = bytes(_aes128_cbc_decrypt(data, key, bytes.fromhex(iv_hex)))
    pad = plain[-1]
    if not 1 <= pad <= 16 or plain[-pad:] != bytes([pad]) * pad:
        raise HLSCacheError("HLS segment decryption failed (wrong key or IV)")
    plain = plain[:-pad]
    if not plain or plain[0] != 0x47:
        raise HLSCacheError("HLS segment decryption failed (not MPEG-TS after decrypt)")
    return plain


def sidecar_path(playlist_path: str | os.PathLike[str]) -> Path:
    return Path(playlist_path).resolve().parent / SIDE_CAR_NAME


def _replace_atomic(src: Path, dst: Path, max_attempts: int = 10, backoff: float = 0.05) -> None:
    """Atomically replace dst with src, retrying on transient Windows file lock errors (WinError 5 / PermissionError)."""
    for attempt in range(max_attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff * (attempt + 1))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        _replace_atomic(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _load_sidecar(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict) or value.get("version") != 1:
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(256 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _public_snapshot(data: dict) -> dict:
    return {key: copy.deepcopy(data.get(key)) for key in (
        "version", "identity", "generation", "status", "error", "playlist", "metadata",
        "ready_segments", "segment_count",
    )}


class _LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class HLSSession:
    """A local view of one persistent HLS cache generation."""

    def __init__(self, playlist: Path, data: dict, *, source_descriptor: dict | None,
                 start_server: bool = True):
        self.playlist_path = str(playlist.resolve())
        self._playlist = Path(self.playlist_path)
        self.status_path = str(sidecar_path(self._playlist))
        self.identity = str(data["identity"])
        self.generation = str(data["generation"])
        self.status = str(data["status"])
        self._data = data
        self._source_descriptor = copy.deepcopy(source_descriptor)
        duration = _descriptor_duration(self._source_descriptor)
        normalized = normalize_metadata(data.get("metadata"), duration)
        if normalized != data.get("metadata"):
            data["metadata"] = normalized
            _atomic_json(Path(self.status_path), data)
        self.metadata = copy.deepcopy(normalized)
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._wanted: dict[int, None] = {}
        self._download_cursor = 0
        self._server: _LocalHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self.runtime_url = ""
        if start_server:
            self._start_server()

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

            def _send_error(self, status: int, message: str = ""):
                body = (message or "source unavailable").encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body) if self.command != "HEAD" else 0))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _serve(self, *, head: bool):
                request_path = unquote(self.path.split("?", 1)[0]).lstrip("/")
                if request_path in ("", PLAYLIST_NAME):
                    try:
                        payload = session._playlist.read_bytes()
                    except OSError:
                        self._send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.apple.mpegurl")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if not head:
                        self.wfile.write(payload)
                    return
                if "/" in request_path or not request_path.startswith("segment-") or not request_path.endswith(".ts"):
                    self._send_error(404)
                    return
                try:
                    index = int(request_path[len("segment-"):-len(".ts")])
                except ValueError:
                    self._send_error(404)
                    return
                result = session._await_segment(index)
                if result is None:
                    self._send_error(503, "segment is not available")
                    return
                path, length = result
                start, end, partial = _parse_range(self.headers.get("Range"), length)
                if start is None:
                    body = b"invalid range"
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{length}")
                    self.send_header("Content-Length", str(len(body) if not head else 0))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if not head:
                        self.wfile.write(body)
                    return
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", "video/mp2t")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
                self.send_header("Connection", "close")
                self.end_headers()
                if head:
                    return
                try:
                    with path.open("rb") as f:
                        f.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            block = f.read(min(64 * 1024, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._server = _LocalHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self._server.server_address[:2]
        self.runtime_url = f"http://{host}:{port}/{PLAYLIST_NAME}"
        self._server_thread = threading.Thread(target=self._server.serve_forever,
                                               name="hls-cache-http", daemon=True)
        self._server_thread.start()

    def request_segment(self, index: int) -> None:
        """Prioritize a reader's segment; the downloader remains the sole upstream owner."""
        with self._condition:
            if (not self._closed and self.status == "downloading"
                    and 0 <= index < int(self._data.get("segment_count") or 0)
                    and not any(x.get("index") == index for x in self._data["ready_segments"])):
                self._wanted.setdefault(index, None)
                self._condition.notify_all()

    def request_position(self, seconds: float) -> None:
        """Map a playback position to its HLS segment using actual EXTINF durations."""
        import math

        if not math.isfinite(seconds) or seconds < 0:
            return
        end = 0.0
        segments = (self._source_descriptor or {}).get("segments", [])
        for segment in segments:
            end += float(segment["duration"])
            if seconds < end:
                self.request_segment(segment["index"])
                return

    def next_download_segment(self) -> int | None:
        """Take a priority request, continue after it, then wrap to fill earlier holes."""
        with self._condition:
            if self._closed or self.status != "downloading":
                return None
            ready = {x["index"] for x in self._data["ready_segments"]}
            count = int(self._data["segment_count"])
            while self._wanted:
                index = next(iter(self._wanted))
                del self._wanted[index]
                if index not in ready:
                    self._download_cursor = (index + 1) % count
                    return index
            for offset in range(count):
                index = (self._download_cursor + offset) % count
                if index not in ready:
                    self._download_cursor = (index + 1) % count
                    return index
            return None

    def _await_segment(self, index: int) -> tuple[Path, int] | None:
        self.request_segment(index)
        with self._condition:
            if not 0 <= index < int(self._data.get("segment_count") or 0):
                return None
            while True:
                if self._closed:
                    return None
                if 0 <= index < int(self._data.get("segment_count") or 0):
                    entry = next((x for x in self._data.get("ready_segments", [])
                                  if isinstance(x, dict) and x.get("index") == index), None)
                    if entry:
                        path = self._playlist.parent / f"segment-{index:06d}.ts"
                        try:
                            if path.is_file() and path.stat().st_size == entry.get("size") and _digest_file(path) == entry.get("sha256"):
                                return path, path.stat().st_size
                        except OSError:
                            pass
                        # An on-disk mutation is never served as a ready segment.
                        self._data["ready_segments"] = [x for x in self._data.get("ready_segments", [])
                                                        if x.get("index") != index]
                        self._persist_locked()
                if self.status in ("failed", "paused", "complete"):
                    return None
                self._condition.wait(0.25)

    def _persist_locked(self) -> None:
        _atomic_json(Path(self.status_path), self._data)

    def _add_ready(self, index: int, path: Path) -> None:
        entry = {"index": index, "filename": path.name, "size": path.stat().st_size, "sha256": _digest_file(path)}
        with self._condition:
            ready = [x for x in self._data.get("ready_segments", []) if x.get("index") != index]
            ready.append(entry)
            ready.sort(key=lambda x: x["index"])
            self._data["ready_segments"] = ready
            self._persist_locked()
            self._condition.notify_all()

    def set_metadata(self, metadata: dict) -> None:
        with self._condition:
            normalized = normalize_metadata(metadata, _descriptor_duration(self._source_descriptor))
            self.metadata = copy.deepcopy(normalized)
            self._data["metadata"] = copy.deepcopy(normalized)
            self._persist_locked()
            self._condition.notify_all()

    def set_status(self, status: str, error: str | None = None) -> None:
        if status not in ("downloading", "complete", "failed", "paused"):
            raise ValueError(f"unknown cache status: {status}")
        with self._condition:
            self.status = status
            self._data["status"] = status
            self._data["error"] = error
            self._persist_locked()
            self._condition.notify_all()

    def snapshot(self) -> dict:
        with self._condition:
            return _public_snapshot(self._data)

    def cancel(self) -> None:
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
            self._condition.notify_all()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread and self._server_thread is not threading.current_thread():
            self._server_thread.join(timeout=2)


def _parse_range(value: str | None, length: int) -> tuple[int | None, int, bool]:
    if not value:
        return (0, max(0, length - 1), False) if length else (None, 0, False)
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not m or not length:
        return None, 0, True
    left, right = m.groups()
    if not left and not right:
        return None, 0, True
    if not left:
        suffix = int(right)
        if suffix <= 0:
            return None, 0, True
        start, end = max(0, length - suffix), length - 1
    else:
        start = int(left)
        end = min(int(right), length - 1) if right else length - 1
        if start >= length or end < start:
            return None, 0, True
    return start, end, True


def _valid_sidecar(data: dict, playlist: Path) -> bool:
    return (data.get("version") == 1 and isinstance(data.get("identity"), str) and
            isinstance(data.get("generation"), str) and data.get("playlist") == str(playlist.resolve()) and
            data.get("status") in ("downloading", "complete", "failed", "paused") and
            isinstance(data.get("ready_segments"), list))


def _verified_ready(data: dict, playlist: Path, *, delete_invalid: bool) -> list[dict]:
    verified = []
    for item in data.get("ready_segments", []):
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0:
            continue
        path = playlist.parent / f"segment-{idx:06d}.ts"
        try:
            okay = (path.is_file() and path.stat().st_size == item.get("size") and
                    isinstance(item.get("sha256"), str) and _digest_file(path) == item.get("sha256"))
        except OSError:
            okay = False
        if okay:
            verified.append(item)
        elif delete_invalid:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return verified


def active_session(playlist_path: str | os.PathLike[str]) -> HLSSession | None:
    key = str(Path(playlist_path).resolve())
    with _REGISTRY_LOCK:
        session = _REGISTRY.get(key)
        return session if session and not session._closed else None


def get_session(playlist_path: str | os.PathLike[str]) -> HLSSession | None:
    """Return the live registry session, if this process owns one."""
    return active_session(playlist_path)


def open_session(playlist_path: str | os.PathLike[str]) -> HLSSession | None:
    """Rebuild an HTTP server only for a verified complete on-disk cache."""
    playlist = Path(playlist_path).resolve()
    key = str(playlist)
    with _REGISTRY_LOCK:
        current = _REGISTRY.get(key)
        if current and not current._closed and current.status == "complete":
            # A live but paused / downloading session is not a finished cache: handing it out made a resumed
            # download return at once and mark the partial source "complete".
            return current
        data = _load_sidecar(sidecar_path(playlist))
        if not data or not _valid_sidecar(data, playlist) or data.get("status") != "complete":
            return None
        try:
            local_playlist = playlist.read_text(encoding="utf-8")
        except OSError:
            return None
        if "#EXT-X-ENDLIST" not in local_playlist:
            return None
        verified = _verified_ready(data, playlist, delete_invalid=False)
        if len(verified) != data.get("segment_count") or len(verified) != len(data.get("ready_segments", [])):
            return None
        indexes = {item.get("index") for item in verified}
        if indexes != set(range(int(data.get("segment_count") or 0))):
            return None
        try:
            session = HLSSession(playlist, data, source_descriptor=data.get("source_descriptor"))
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


def inspect_cache(cache_dir: str | os.PathLike[str], playlist: MediaPlaylist,
                  request_key: str) -> tuple[dict | None, list[int]]:
    """Check whether persisted bytes belong to this exact selected playlist.

    A changed URL, manifest body, segment URI, duration, or queue request key
    refuses resume.  Corrupt files are omitted and will be downloaded again.
    """
    local_playlist = (Path(cache_dir).resolve() / PLAYLIST_NAME)
    data = _load_sidecar(sidecar_path(local_playlist))
    if not data or not _valid_sidecar(data, local_playlist):
        return None, []
    descriptor = playlist.descriptor
    descriptor["request_key"] = request_key
    old_descriptor = data.get("source_descriptor")
    if old_descriptor and not _descriptors_match(old_descriptor, descriptor):
        raise HLSCacheError("selected playlist identity changed; refusing to mix cache generations")
    if not old_descriptor:
        return None, []
    ready = _verified_ready(data, local_playlist, delete_invalid=False)
    ready_indices = sorted({x["index"] for x in ready})
    metadata = data.get("metadata") if 0 in ready_indices else None
    return (copy.deepcopy(metadata) if isinstance(metadata, dict) else None), ready_indices


def prepare_session(playlist: MediaPlaylist, cache_dir: str | os.PathLike[str], *,
                    metadata: dict, request_key: str, seed_path: Path | None = None) -> tuple[HLSSession, list[int]]:
    """Create/resume one generation after the selected manifest and first TS are verified.

    Returns the session and the set of verified ready segment indices.  Existing
    bytes are trusted only when both the selected source descriptor and each
    persisted segment hash match.
    """
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    local_playlist = directory / PLAYLIST_NAME
    status_file = sidecar_path(local_playlist)
    descriptor = playlist.descriptor
    descriptor["request_key"] = request_key

    with _REGISTRY_LOCK:
        old = _load_sidecar(status_file)
        current = _REGISTRY.get(str(local_playlist))
        if current and not current._closed and current.status == "downloading":
            raise HLSCacheError("a download for this cache is already active")
        if old and _valid_sidecar(old, local_playlist):
            old_descriptor = old.get("source_descriptor")
            if old_descriptor and not _descriptors_match(old_descriptor, descriptor):
                old["generation"] = str(uuid.uuid4())
                old["status"] = "failed"
                old["error"] = "selected playlist identity changed; refusing to mix cache generations"
                _atomic_json(status_file, old)
                raise HLSCacheError(old["error"])
            if old.get("status") == "complete":
                existing = open_session(local_playlist)
                if existing:
                    return existing, [x["index"] for x in existing._data["ready_segments"]]
            identity = old["identity"]
            ready = _verified_ready(old, local_playlist, delete_invalid=True)
            if len(ready) != len(old.get("ready_segments", [])):
                old["ready_segments"] = ready
        else:
            identity = str(uuid.uuid4())
            old = None
            ready = []
        generation = str(uuid.uuid4())
        data = {
            "version": 1, "identity": identity, "generation": generation,
            "status": "downloading", "error": None,
            "playlist": str(local_playlist), "metadata": normalize_metadata(metadata, playlist.duration),
            "ready_segments": ready, "segment_count": len(playlist.segments),
            "source_descriptor": descriptor,
        }
        current = _REGISTRY.pop(str(local_playlist), None)
        if current:
            current.close()
        local_tmp = local_playlist.with_name(local_playlist.name + ".tmp-" + uuid.uuid4().hex)
        try:
            local_tmp.write_text(playlist.local_text, encoding="utf-8", newline="\n")
            _replace_atomic(local_tmp, local_playlist)
        finally:
            if local_tmp.exists():
                try:
                    local_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        _atomic_json(status_file, data)

        try:
            session = HLSSession(local_playlist, data, source_descriptor=descriptor)
        except OSError as exc:
            data["status"] = "failed"
            data["error"] = f"could not start local HLS cache server: {exc}"[:500]
            _atomic_json(status_file, data)
            raise HLSCacheError(data["error"]) from exc
        _REGISTRY[str(local_playlist)] = session
        if seed_path is not None:
            seed = directory / playlist.segments[0].filename
            if not any(x.get("index") == 0 for x in ready):
                _replace_atomic(seed_path, seed)
                session._add_ready(0, seed)
                ready = [0]
        return session, [x["index"] for x in session._data.get("ready_segments", [])]


def commit_segment(session: HLSSession, index: int, temporary_path: str | os.PathLike[str]) -> Path:
    """Atomically publish a fully downloaded upstream TS segment."""
    if session.status != "downloading":
        raise HLSCacheError(f"cache session is {session.status}")
    source = Path(temporary_path)
    target = Path(session.playlist_path).parent / f"segment-{index:06d}.ts"
    if source.stat().st_size <= 0:
        raise HLSCacheError(f"segment {index} is empty")
    _replace_atomic(source, target)
    session._add_ready(index, target)
    return target


def complete_session(session: HLSSession) -> None:
    if len(session._data.get("ready_segments", [])) != session._data.get("segment_count"):
        raise HLSCacheError("cannot complete a cache with missing segments")
    session.set_status("complete")


def fail_session(session: HLSSession, error: BaseException | str, *, paused: bool = False) -> None:
    status = "paused" if paused else "failed"
    session.set_status(status, str(error)[:500])
