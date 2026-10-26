"""Subtitle usage log: one JSONL file per queue item (per process lifetime) under <data_dir>/logs/subtitle-usage/.

Purpose: during real use record enough (cloud requests, published slices, playback view samples, run start/end)
to analyse later whether cloud subtitles keep up with playback. Always on, no UI.

Privacy: lines hold only whitelisted scalar fields. Never an API key, header, URL path/query, audio or subtitle
text. Every public function swallows its own failures: logging must never break subtitles.
Lines are appended in text append mode and flushed per line, so another process (the subtitle runtime child)
may append to the same file via `UsageLog(path)`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

MAX_FILES = 50
MAX_TOTAL_BYTES = 20 * 1024 * 1024
VIEW_INTERVAL_S = 30.0
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")

_clock: Callable[[], float] = time.time
_dir_override: Optional[Path] = None
_logs: Dict[str, "UsageLog"] = {}
_registry_lock = threading.Lock()
_local = threading.local()


def log_dir() -> Path:
    if _dir_override is not None:
        return Path(_dir_override)
    from oh_streamer.platform.platform_support import data_dir
    return data_dir() / "logs" / "subtitle-usage"


def prune(directory: Path, max_files: int = MAX_FILES, max_bytes: int = MAX_TOTAL_BYTES, keep: Optional[Path] = None) -> None:
    """Delete the oldest *.jsonl files until at most max_files / max_bytes remain."""
    try:
        files = []
        for p in Path(directory).glob("*.jsonl"):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, p, st.st_size))
        files.sort(key=lambda t: (t[0], t[1].name))
        total = sum(f[2] for f in files)
        while files and (len(files) > max_files or total > max_bytes):
            _mtime, path, size = files.pop(0)
            if keep is not None and path == keep:
                continue
            try:
                path.unlink()
                total -= size
            except OSError:
                pass
    except Exception:                                   # noqa: BLE001
        pass


class UsageLog:
    """Append-only JSONL writer for one file."""

    def __init__(self, path: Path | str, item_id: str = ""):
        self.path = Path(path)
        self.item_id = item_id
        self._lock = threading.Lock()
        self.last_view_state: Optional[str] = None
        self.last_view_at = -1e18

    def write(self, rec: Dict[str, Any]) -> None:
        try:
            rec = {"ts": round(_clock(), 3), **({"item": self.item_id} if self.item_id else {}), **rec}
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
        except Exception:                               # noqa: BLE001
            pass


def _open(item_id: str) -> Optional[UsageLog]:
    """The item's log for this process (created and pruned on first use)."""
    try:
        with _registry_lock:
            log = _logs.get(item_id)
            if log is None:
                directory = log_dir()
                directory.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(_clock()))
                path = directory / f"{stamp}-{_SAFE_ID.sub('_', item_id)[:40] or 'item'}.jsonl"
                n = 1
                while path.exists():
                    n += 1
                    path = directory / f"{stamp}-{_SAFE_ID.sub('_', item_id)[:40] or 'item'}-{n}.jsonl"
                path.touch()
                prune(directory, keep=path)
                log = _logs[item_id] = UsageLog(path, item_id)
            return log
    except Exception:                                   # noqa: BLE001
        return None


def get(item_id: str) -> Optional[UsageLog]:
    return _logs.get(item_id)


def reset() -> None:
    """Forget open logs (tests)."""
    with _registry_lock:
        _logs.clear()


# -- run start / end ----------------------------------------------------------------------------------------
def run_start(item_id: str, *, executor: str = "", duration_s: Optional[float] = None, profile: Any = None,
              reason: str = "", gap_s: Optional[float] = None) -> None:
    """`reason`: why this run started (first_run / restart_after_<how the previous run ended>); `gap_s`: seconds
    since the previous run of the same executor ended."""
    try:
        log = _open(item_id)
        if log:
            rec = {"ev": "run_start", "executor": executor, "video_duration_s": _num(duration_s),
                   "profile": _profile(profile)}
            if reason:
                rec["reason"] = str(reason)[:60]
            if gap_s is not None:
                rec["gap_s"] = _num(gap_s)
            log.write(rec)
    except Exception:                                   # noqa: BLE001
        pass


def run_end(item_id: str, *, executor: str = "", reason: str = "", **details: Any) -> None:
    """`reason`: how the run ended (complete / idle / horizon / quota / download / stuck / source_unavailable /
    cancelled / error ...). `details`: optional scalar fields (wait_until, wait_s, waiting_executor, run_s,
    units_launched, units_done, units_failed, requests, started, error); None values are left out."""
    try:
        log = get(item_id)
        if log:
            rec = {"ev": "run_end", "executor": executor, "reason": str(reason)[:40]}
            rec.update(_fields(details))
            log.write(rec)
    except Exception:                                   # noqa: BLE001
        pass


# -- unit / fallback events ---------------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+", re.I)
_KEY = re.compile(r"\b(?:sk|gsk|key|token)[-_][A-Za-z0-9_-]{8,}", re.I)
DETAIL_MAX = 160


def clean_text(text: Any, limit: int = DETAIL_MAX) -> str:
    """An error text for a usage line: no URL, no key-like token, one line, truncated."""
    t = _KEY.sub("<key>", _URL.sub("<url>", str(text or "")))
    t = " ".join(t.split())
    return t if len(t) <= limit else t[:limit - 1] + "…"


def _fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (bool, int)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = _num(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (bool, int, str)) else _num(x) for x in list(v)[:20]]
        else:
            out[k] = clean_text(v, DETAIL_MAX if k in ("detail", "error") else 60)
    return out


def event(item_id: str, ev: str, **fields: Any) -> None:
    """One diagnostic line (candidate_changed, local_started, local_done, local_failed, local_wait_started ...).
    Scalar fields only; None values are left out; text fields are cleaned and truncated."""
    try:
        log = get(item_id)
        if log:
            log.write({"ev": str(ev)[:40], **_fields(fields)})
    except Exception:                                   # noqa: BLE001
        pass


def _profile(frozen: Any) -> Dict[str, Any]:
    """Candidate ids per role of a frozen runtime plan; ids only."""
    out: Dict[str, Any] = {}
    try:
        from oh_streamer.subtitles import fallback as fb
        for role in ("asr", "translation"):
            try:
                ids = [c.id for c in fb.candidate_rows(frozen, role)]
            except Exception:                           # noqa: BLE001
                ids = []
            if ids:
                out[role] = ids
    except Exception:                                   # noqa: BLE001
        pass
    return out


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else round(float(v), 3)
    except (TypeError, ValueError):
        return None


# -- thread-local request context (set by the executor around one unit of work) ---------------------------
@contextmanager
def context(item_id: str, *, slice_index: Optional[int] = None, stage: str = "", candidate: str = "",
            track: str = "", on_request: Optional[Callable[[], None]] = None):
    """`on_request`: called once per cloud request logged inside the context (the executor's per-run counter)."""
    prev = getattr(_local, "ctx", None)
    _local.ctx = {"item_id": item_id, "slice": slice_index, "stage": stage, "candidate": candidate,
                  "track": track, "audio_s": None, "on_request": on_request}
    try:
        yield
    finally:
        _local.ctx = prev


def request_event(ev: str, **fields: Any) -> None:
    """event() for the item / slice / candidate of the current request context; nothing outside one."""
    try:
        ctx = getattr(_local, "ctx", None)
        if ctx:
            event(ctx["item_id"], ev, slice=ctx.get("slice"), candidate=ctx.get("candidate") or None, **fields)
    except Exception:                                   # noqa: BLE001
        pass


def set_audio_seconds(seconds: Optional[float]) -> None:
    ctx = getattr(_local, "ctx", None)
    if ctx is not None:
        ctx["audio_s"] = _num(seconds)


# -- cloud requests -----------------------------------------------------------------------------------------
def timed_request(model: Dict[str, Any], route: str, call: Callable[[], Any], *,
                  now: Callable[[], float] = time.monotonic) -> Any:
    """Run `call()` (the real http_post) and log it. Returns / raises exactly what `call` does."""
    ctx = getattr(_local, "ctx", None)
    t0 = now()
    try:
        data = call()
    except BaseException as exc:
        _log_request(ctx, model, route, now() - t0, None, exc)
        raise
    _log_request(ctx, model, route, now() - t0, data, None)
    return data


def _log_request(ctx, model, route, latency, data, exc) -> None:
    try:
        if not ctx:
            return
        if ctx.get("on_request") is not None:
            try:
                ctx["on_request"]()
            except Exception:                           # noqa: BLE001
                pass
        log = get(ctx["item_id"])
        if log is None:
            return
        host = ""
        try:
            host = urlsplit(str(model.get("base_url") or "")).hostname or ""
        except Exception:                               # noqa: BLE001
            pass
        kind = "asr" if "transcriptions" in str(route) else "mt"
        status = 200
        retry_after = None
        err = None
        if exc is not None:
            status = getattr(exc, "http_status", None)
            retry_after = _num(getattr(exc, "retry_after", None))
            err = getattr(exc, "kind", None) or type(exc).__name__
        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        rec = {"ev": "request", "kind": kind, "host": host,
               "model": str(model.get("remote_model") or model.get("id") or ""),
               "status": status, "latency_s": round(latency, 3), "retry_after_s": retry_after,
               "audio_s": ctx.get("audio_s") if kind == "asr" else None,
               "prompt_tokens": _int(usage.get("prompt_tokens")),
               "completion_tokens": _int(usage.get("completion_tokens")),
               "total_tokens": _int(usage.get("total_tokens")),
               "slice": ctx.get("slice"), "candidate": ctx.get("candidate") or None}
        if err:
            rec["error"] = str(err)[:40]
        if getattr(exc, "too_large", False):
            rec["too_large"] = True
        log.write(rec)
    except Exception:                                   # noqa: BLE001
        pass


def _int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


# -- slices -------------------------------------------------------------------------------------------------
def slice_published(item_id: str, *, track: str, slice_index: int, start: Optional[float], end: Optional[float],
                    cues: int, candidate: str = "") -> None:
    try:
        log = get(item_id)
        if log:
            log.write({"ev": "slice", "track": track, "slice": slice_index, "start_s": _num(start),
                       "end_s": _num(end), "cues": int(cues), "candidate": candidate or None})
    except Exception:                                   # noqa: BLE001
        pass


# -- view samples -------------------------------------------------------------------------------------------
def view_due(item_id: str, view: Optional[dict], now: Optional[float] = None) -> bool:
    """True when a sample would be written now (lets callers skip computing extras)."""
    log = get(item_id)
    if log is None or not isinstance(view, dict):
        return False
    state = str(view.get("state") or "")
    t = _clock() if now is None else now
    if state != log.last_view_state:
        return True
    return state != "complete" and t - log.last_view_at >= VIEW_INTERVAL_S


def view_sample(item_id: str, view: Optional[dict], *, play_pos: Optional[float], restore_end: Optional[float] = None,
                download_end: Optional[float] = None, now: Optional[float] = None) -> None:
    """On state change, plus at most every VIEW_INTERVAL_S while the state is not final."""
    try:
        if not view_due(item_id, view, now):
            return
        log = get(item_id)
        t = _clock() if now is None else now
        log.last_view_state, log.last_view_at = str(view.get("state") or ""), t
        log.write({"ev": "view", "state": log.last_view_state, "play_pos_s": _num(play_pos),
                   "covered_end_s": _num(view.get("covered_s")), "restore_end_s": _num(restore_end),
                   "download_end_s": _num(download_end)})
    except Exception:                                   # noqa: BLE001
        pass
