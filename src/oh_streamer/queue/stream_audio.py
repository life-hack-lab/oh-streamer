"""Subtitle audio for a streaming item (T2.3 task C): which source to read, the remote marker, the background
start-time probe and the signed-url refresher. Runs in the queue process: no torch / silero / OpenCC imports.

The audio is read straight from the remote site by ffmpeg in the
queue process (``media_input.ffmpeg_input`` on the ``audio.rstream`` marker). The marker url and headers are secrets:
no log line or error text produced here may contain them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from oh_streamer.player.open_input import OpenInput
from oh_streamer.sources import media_input, qualities, source_info
from oh_streamer.sources.source_resolve import FFmpegHttpOptions, is_url

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PROBE_TIMEOUT = 90.0
START_DIFF_LOG_S = 0.05            # |play_start - audio_start| above this is logged (task E checks real sites)
MARKER_NAME = "audio.rstream"

ERR_UNSUPPORTED = "此來源不支援串流字幕（無法讀取音訊）"
ERR_EXPIRED = "字幕音訊網址已失效"


def log(msg: str) -> None:
    print(f"[stream_audio] {msg}", file=sys.stderr, flush=True)


class RefreshError(RuntimeError):
    """The expired audio url could not be replaced (message is zh-TW and never contains a url)."""


# ----------------------------------------------------------------------------------------------------------
# choosing the source
# ----------------------------------------------------------------------------------------------------------

def _is_hls(url: str, protocol: str = "") -> bool:
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        path = ""
    return str(protocol or "").startswith("m3u8") or path.endswith(".m3u8")


def _rendition(q: dict) -> dict:
    return {"id": str(q["id"]), "height": q.get("height") or None, "label": q.get("label") or ""}


def _same_rendition(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    if a.get("id") is not None and str(a.get("id")) == str(b.get("id")):
        return True
    return bool(a.get("height")) and a.get("height") == b.get("height")


def _quality_pick(kind: str, q: dict, r) -> dict:
    return {"kind": kind, "url": q["url"], "headers": {**(q.get("http_headers") or {}), **r.headers},
            "hls": _is_hls(q["url"]), "rendition": _rendition(q)}


def pick_subtitle_audio(r, prefer: Optional[dict] = None) -> dict:
    """Decide where the subtitle audio is read from (pure computation, no network).

    Returns {"kind": "audio_only"|"lowest"|"same"|"file", "url", "headers", "hls", "rendition"}. Order: local file;
    the site's audio-only format; the lowest rendition (2+); the only rendition; the directly read media url.
    ``prefer`` (a rendition dict, used when re-resolving) selects the same rendition id, else the same height, among
    the renditions; when none matches the normal rule applies and the caller compares ``rendition`` itself."""
    if r.kind == "file":
        path = r.media_src[0] if r.media_src else os.path.abspath(r.target)
        return {"kind": "file", "url": str(path), "headers": {}, "hls": False, "rendition": None}
    af = r.audio_format
    if af:
        return {"kind": "audio_only", "url": af["url"], "headers": dict(af.get("headers") or {}),
                "hls": _is_hls(af["url"], af.get("protocol", "")),
                "rendition": {"id": str(af.get("id") or ""), "height": None, "label": "audio"}}
    quals = r.quals or []
    if quals:
        if prefer:
            for q in quals:
                if str(q["id"]) == str(prefer.get("id")):
                    break
            else:
                q = qualities.find_quality(quals, None, prefer.get("height") or None)
            if q is not None:
                return _quality_pick("same" if len(quals) == 1 else "lowest", q, r)
        if len(quals) >= 2:
            return _quality_pick("lowest", quals[-1], r)
        return _quality_pick("same", quals[0], r)
    if r.media_src:
        url, headers = r.media_src
        return {"kind": "same", "url": url, "headers": dict(headers or {}), "hls": _is_hls(url) or bool(r.hls),
                "rendition": None}
    raise ValueError("no readable audio source")


# ----------------------------------------------------------------------------------------------------------
# probing a:0 start_time
# ----------------------------------------------------------------------------------------------------------

def probe_audio_start(src: str, headers: Optional[dict] = None) -> float:
    """ffprobe the first audio stream's start_time (seconds). Raises ``media_input.RemoteExpired`` when the url
    expired (403/404/410) and ``RuntimeError`` for anything else; messages never contain the url."""
    remote = is_url(src)
    http = [*FFmpegHttpOptions(headers or {}).cli_args(), "-rw_timeout", media_input.REMOTE_RW_TIMEOUT_US] \
        if remote else []
    if remote and _is_hls(src):
        http += media_input.HLS_ANY_EXTENSION_ARGS
    cmd = ["ffprobe", "-v", "warning", *http, "-select_streams", "a:0", "-show_entries", "stream=start_time",
           "-of", "json", src]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=NO_WINDOW, timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ffprobe could not run ({type(exc).__name__})") from None
    failure = media_input.remote_failure(p.stderr) if remote else None
    if failure == "expired":
        raise media_input.RemoteExpired()
    if p.returncode != 0 or failure:
        raise RuntimeError(f"ffprobe could not read the audio (exit {p.returncode})")
    try:
        streams = [s for s in json.loads(p.stdout).get("streams") or [] if isinstance(s, dict)]
    except ValueError:
        streams = []
    if not streams:
        raise RuntimeError("the source has no audio stream")
    try:
        value = float(streams[0].get("start_time", 0.0))
    except (TypeError, ValueError):
        value = 0.0                    # "N/A": container without start times
    return value if math.isfinite(value) else 0.0


# ----------------------------------------------------------------------------------------------------------
# attach: marker + stream fields + refresher + background probe
# ----------------------------------------------------------------------------------------------------------

def _identity(r, pick: dict) -> str:
    rend = pick.get("rendition") or {}
    raw = f"{r.target}|{pick['kind']}|{rend.get('id', '')}|{rend.get('height') or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _update_stream(service, item_id: str, **fields) -> bool:
    """Merge ``fields`` into ``item.stream`` (store.update swaps the whole dict, so read-merge-write under the
    service lock). False when the item is gone."""
    with service._lock:
        item = service.store.get(item_id)
        if item is None:
            return False
        service.store.update(item_id, stream={**item.stream, **fields})
        return True


def _play_source(r, pick: dict, played_height) -> Optional[tuple]:
    """(url, headers) of the rendition the player plays, or None when it reads the same url as the subtitle audio."""
    if pick["kind"] in ("file", "same") or not r.quals:
        return None
    q = qualities.pick_quality(r.quals, played_height, None)
    url = q.get("audio_url") or q["url"]          # video-only renditions carry their audio in audio_url
    if url == pick["url"]:
        return None
    return url, {**(q.get("http_headers") or {}), **r.headers}


def _probe_worker(service, item_id: str, r, pick: dict, marker: Optional[str], played_height) -> None:
    try:
        if marker:
            def attempt():
                m = media_input.read_remote_marker(marker)
                return probe_audio_start(m["url"], m["headers"])
            audio_start = media_input.run_remote(marker, attempt)
        else:
            audio_start = probe_audio_start(pick["url"], {})
        play_start = audio_start
        play = _play_source(r, pick, played_height)
        if play is not None:
            try:
                play_start = probe_audio_start(*play)
            except Exception as exc:  # noqa: BLE001 - keep the audio start; offset may then be off by the start gap
                log(f"{item_id}: play rendition start unreadable ({type(exc).__name__}), using the audio start")
        offset = audio_start + (play_start - audio_start)
        if abs(play_start - audio_start) > START_DIFF_LOG_S:
            log(f"{item_id}: audio_start={audio_start:.3f} play_start={play_start:.3f} "
                f"diff={play_start - audio_start:+.3f}s kind={pick['kind']}")
    except Exception as exc:  # noqa: BLE001 - any failure ends in a visible failed state
        text = str(exc)
        error = ERR_EXPIRED if isinstance(exc, (media_input.RemoteExpired, RefreshError)) or "字幕音訊網址" in text \
            else ERR_UNSUPPORTED
        log(f"{item_id}: audio probe failed ({type(exc).__name__}) -> {error}")
        if not _update_stream(service, item_id, audio_status="failed", audio_error=error) and marker:
            media_input.unregister_refresher(marker)
        return
    # the offset first: whoever waits for "ready" must already find it
    service._subtitle_update(item_id, audio_offset=offset)
    if not _update_stream(service, item_id, audio_status="ready", audio_error="", audio_start=audio_start,
                          play_start=play_start, audio_offset=offset):
        if marker:
            media_input.unregister_refresher(marker)


def attach(service, item_id: str, r, pick: dict, played_height=None) -> threading.Thread:
    """Wire the subtitle audio into stream item ``item_id``: write ``streams/<id>/audio.rstream`` (not for local
    files), set ``source_path`` and the ``stream`` audio fields (``audio_status="probing"``), register the url
    refresher and start the daemon thread that measures the start times and then sets ``ready`` / ``failed``.
    Uses only ``service.store``, ``service._lock`` and ``service._subtitle_update``. Returns the probe thread."""
    marker = None
    if service.store.get(item_id) is None:
        raise KeyError(item_id)
    if pick["kind"] != "file":
        marker = str(service.store.item_dir(item_id) / MARKER_NAME)
        media_input.write_remote_marker(marker, {
            "kind": pick["kind"], "url": pick["url"], "headers": pick["headers"], "hls": pick["hls"],
            "generation": 0, "identity": _identity(r, pick), "rendition": pick["rendition"]})
    fields = dict(audio_kind=pick["kind"], audio_rendition=pick["rendition"], audio_marker=marker or "",
                  audio_status="probing", audio_error="", audio_generation=0)
    with service._lock:
        item = service.store.get(item_id)
        if item is None:
            if marker:
                try:
                    os.unlink(marker)
                except OSError:
                    pass
            raise KeyError(item_id)
        service.store.update(item_id, source_path=marker or pick["url"], stream={**item.stream, **fields})
    if marker:
        media_input.register_refresher(marker, make_refresher(service, item_id))
    thread = threading.Thread(target=_probe_worker, args=(service, item_id, r, pick, marker, played_height),
                              name=f"stream-audio-probe-{item_id}", daemon=True)
    thread.start()
    return thread


# ----------------------------------------------------------------------------------------------------------
# refresher
# ----------------------------------------------------------------------------------------------------------

def _acceptable(old: dict, new: dict) -> bool:
    """Is the freshly picked source the same one the marker had (same kind of source, same rendition)?"""
    if old.get("kind") == "audio_only":
        return new["kind"] == "audio_only"
    if old.get("kind") not in ("lowest", "same") or new["kind"] not in ("lowest", "same"):
        return False
    if old.get("rendition") is None:
        return new["rendition"] is None
    return _same_rendition(old["rendition"], new["rendition"])


def make_refresher(service, item_id: str) -> Callable[[str], None]:
    """``fn(marker)`` for ``media_input.register_refresher``: re-resolve the item, keep the kind and rendition,
    rewrite the marker with the new url (generation + 1, identity unchanged) and mirror the generation into
    ``item.stream``. Raises ``RefreshError`` (zh-TW, no url) when the item is gone, resolving fails, the original
    rendition is not offered any more (user decision: fail, never switch source) or no new url came back."""
    def refresh(marker: str) -> None:
        item = service.store.get(item_id)
        if item is None:
            raise RefreshError("字幕音訊網址已失效（項目已移除）")
        old = media_input.read_remote_marker(marker)
        try:
            r = source_info.resolve(OpenInput(item.target, dict(item.headers or {})))
            new = pick_subtitle_audio(r, prefer=old.get("rendition"))
        except Exception as exc:  # noqa: BLE001 - ResolveError text may carry site details; keep only the type
            log(f"{item_id}: re-resolve failed ({type(exc).__name__})")
            raise RefreshError("字幕音訊網址已失效，重新解析失敗") from None
        if not _acceptable(old, new):
            log(f"{item_id}: re-resolve did not offer the original audio source")
            raise RefreshError("字幕音訊網址已失效，重新解析後找不到原本的音訊來源")
        if new["url"] == old["url"]:
            log(f"{item_id}: re-resolve returned the same url")
            raise RefreshError("字幕音訊網址已失效，重新解析後仍是同一個網址")
        generation = old["generation"] + 1
        with service._lock:
            if service.store.get(item_id) is None:      # removed meanwhile: do not recreate streams/<id>/
                raise RefreshError("字幕音訊網址已失效（項目已移除）")
            media_input.write_remote_marker(marker, {**old, "url": new["url"], "headers": new["headers"],
                                                     "hls": new["hls"], "generation": generation})
        _update_stream(service, item_id, audio_generation=generation)
        log(f"{item_id}: audio url refreshed (generation {generation})")
    return refresh
