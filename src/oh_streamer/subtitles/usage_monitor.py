"""Best-effort Groq usage statistics: last-seen rate-limit headers, ASR audio-second windows,
request/token counters and an hourly series for charting.

Statistics and warnings only; nothing here influences fallback / switching behaviour.
Storage is a small sqlite file (WAL, one short-lived connection per call), so it is safe
across threads and processes and bounded by pruning:
  * per-minute buckets are kept for ``MINUTE_RETENTION_S`` (a bit over 24 h),
  * per-hour buckets are kept for ``HOUR_RETENTION_S`` (7 days).
No API keys or request payloads are ever stored.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# ---- thresholds / limits (module constants; remaining fraction) -------------------------
WARN_LOW_FRACTION = 0.20
WARN_CRITICAL_FRACTION = 0.05

# Groq bills at least this many audio seconds per ASR request.
ASR_MIN_BILLED_S = 10.0
HOUR_S = 3600
DAY_S = 86400
MINUTE_RETENTION_S = DAY_S + 3600
HOUR_RETENTION_S = 7 * DAY_S

# Free-tier whisper audio limits: 7200 s/hour and 28800 s/day (official docs, see OFFICIAL_LIMITS).
# Global override with env OH_STREAMER_GROQ_ASR_HOUR_LIMIT_S / OH_STREAMER_GROQ_ASR_DAY_LIMIT_S or set_audio_limits();
# an override wins over the per-model table, which wins over the DEFAULT_* values below.
AUDIO_LIMITS_VERIFIED = True   # whisper ASH/ASD come from the official table below
DEFAULT_ASR_HOUR_LIMIT_S = 7200.0
DEFAULT_ASR_DAY_LIMIT_S = 28800.0
_audio_limits: Dict[str, Optional[float]] = {}

# ---- built-in official limits (fallback when response headers lack a limit) --------------
# Source: Groq docs, "Rate Limits" -> free tier base limits, retrieved 2026-10-22. The account's
# Limits page is authoritative (limits are org-level and may differ); override at runtime with
# set_official_limits() or edit this table. Header values always win when present.
# Keys: rpm = requests/minute, rpd = requests/day, tpm = tokens/minute, tpd = tokens/day,
#       ash = audio seconds/hour, asd = audio seconds/day. Missing key = no known limit.
# Note: x-ratelimit-*-requests headers mean RPD (daily); x-ratelimit-*-tokens mean TPM (per minute).
_CHAT_FREE = {'rpm': 30, 'rpd': 1000, 'tpm': 8000, 'tpd': 200000}
_WHISPER_FREE = {'rpm': 20, 'rpd': 2000, 'ash': 7200, 'asd': 28800}
OFFICIAL_LIMITS: Dict[str, Dict[str, int]] = {
    'qwen/qwen3.8-27b': dict(_CHAT_FREE),
    'openai/gpt-oss-120b': dict(_CHAT_FREE),
    'openai/gpt-oss-20b': dict(_CHAT_FREE),
    'openai/gpt-oss-safeguard-20b': {'rpm': 3, 'rpd': 1000, 'tpm': 2000, 'tpd': 200000},
    'whisper-large-v3': dict(_WHISPER_FREE),
    'whisper-large-v3-turbo': dict(_WHISPER_FREE),
    'meta-llama/llama-prompt-guard-2-22m': {'rpm': 30, 'rpd': 14400, 'tpm': 15000, 'tpd': 500000},
    'meta-llama/llama-prompt-guard-2-86m': {'rpm': 30, 'rpd': 14400, 'tpm': 15000, 'tpd': 500000},
    'canopylabs/orpheus-v1-english': {'rpm': 10, 'rpd': 100, 'tpm': 1200, 'tpd': 3600},
    'canopylabs/orpheus-arabic-saudi': {'rpm': 10, 'rpd': 100, 'tpm': 1200, 'tpd': 3600},
}
_limit_overrides: Dict[str, Dict[str, Optional[int]]] = {}


def set_official_limits(model: str, limits: Optional[Mapping[str, Optional[int]]]) -> None:
    """Override (merge over) the built-in table for one model; ``None`` removes the override."""
    if limits is None:
        _limit_overrides.pop(model, None)
    else:
        _limit_overrides[model] = dict(limits)


def official_limits(model: str) -> Dict[str, int]:
    """Table limits for ``model`` (overrides applied; None values drop a key)."""
    merged: Dict[str, Any] = dict(OFFICIAL_LIMITS.get(model, {}))
    merged.update(_limit_overrides.get(model, {}))
    return {k: v for k, v in merged.items() if v is not None}

_DB_OVERRIDE: Optional[Path] = None
_init_lock = threading.Lock()
_initialised: set = set()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS last_seen (
  connection_id TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
  req_limit INTEGER, req_remaining INTEGER, req_reset_at REAL,
  tok_limit INTEGER, tok_remaining INTEGER, tok_reset_at REAL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (connection_id, model));
CREATE TABLE IF NOT EXISTS minute_buckets (
  connection_id TEXT NOT NULL, model TEXT NOT NULL, t INTEGER NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
  audio_s REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (connection_id, model, t));
CREATE TABLE IF NOT EXISTS hour_buckets (
  connection_id TEXT NOT NULL, model TEXT NOT NULL, t INTEGER NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
  audio_s REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (connection_id, model, t));
"""


def set_db_path(path: Optional[os.PathLike]) -> None:
    """Override the sqlite location (tests); ``None`` restores the default."""
    global _DB_OVERRIDE
    _DB_OVERRIDE = Path(path) if path else None


def db_path() -> Path:
    if _DB_OVERRIDE is not None:
        return _DB_OVERRIDE
    env = os.environ.get('OH_STREAMER_USAGE_MONITOR_DB')
    if env:
        return Path(env)
    from ..platform.platform_support import data_dir
    return data_dir() / 'usage_monitor.sqlite3'


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5.0)
    key = str(path)
    if key not in _initialised or not path.exists():
        with _init_lock:
            try:
                con.execute('PRAGMA journal_mode=WAL')
            except sqlite3.Error:
                pass
            con.executescript(_SCHEMA)
            con.commit()
            _initialised.add(key)
    return con


def set_audio_limits(hour_s: Optional[float] = None, day_s: Optional[float] = None) -> None:
    _audio_limits['hour'] = hour_s
    _audio_limits['day'] = day_s


def _env_float(name: str) -> Optional[float]:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return None


def audio_limits() -> Dict[str, Optional[float]]:
    hour = _audio_limits['hour'] if 'hour' in _audio_limits else (
        _env_float('OH_STREAMER_GROQ_ASR_HOUR_LIMIT_S') or DEFAULT_ASR_HOUR_LIMIT_S)
    day = _audio_limits['day'] if 'day' in _audio_limits else (
        _env_float('OH_STREAMER_GROQ_ASR_DAY_LIMIT_S') or DEFAULT_ASR_DAY_LIMIT_S)
    return {'hour': hour, 'day': day}


def _audio_limits_for(model: str):
    """(hour_s, day_s, verified): explicit override / env > official table > DEFAULT_*."""
    table = official_limits(model)
    out = []
    verified = True
    for key, env, tkey, default in (('hour', 'OH_STREAMER_GROQ_ASR_HOUR_LIMIT_S', 'ash', DEFAULT_ASR_HOUR_LIMIT_S),
                                    ('day', 'OH_STREAMER_GROQ_ASR_DAY_LIMIT_S', 'asd', DEFAULT_ASR_DAY_LIMIT_S)):
        if key in _audio_limits:
            out.append(_audio_limits[key])
        elif _env_float(env):
            out.append(_env_float(env))
        elif tkey in table:
            out.append(float(table[tkey]))
        else:
            out.append(default)
            verified = False   # neither override nor official table: built-in guess
    return out[0], out[1], bool(AUDIO_LIMITS_VERIFIED and verified)


# ---- header parsing ---------------------------------------------------------------------
def parse_duration(value: Any) -> Optional[float]:
    """Parse '43.2s', '1m26.4s', '134ms', '6m0s', '2h1m' or bare seconds into seconds."""
    from .providers.http import _parse_duration_string
    if value is None:
        return None
    return _parse_duration_string(str(value))


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def parse_rate_limit_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract x-ratelimit-* values (case-insensitive). Resets are returned in seconds."""
    low = {str(k).lower(): v for k, v in dict(headers or {}).items()}
    out: Dict[str, Optional[float]] = {}
    for scope in ('requests', 'tokens'):
        out[scope + '_limit'] = _to_int(low.get('x-ratelimit-limit-' + scope))
        out[scope + '_remaining'] = _to_int(low.get('x-ratelimit-remaining-' + scope))
        out[scope + '_reset_s'] = parse_duration(low.get('x-ratelimit-reset-' + scope))
    return out


# ---- recording --------------------------------------------------------------------------
def _clock(now: Optional[float]) -> float:
    return time.time() if now is None else float(now)


def billed_audio_s(audio_s: Optional[float]) -> float:
    try:
        return max(ASR_MIN_BILLED_S, float(audio_s or 0.0))
    except (TypeError, ValueError):
        return ASR_MIN_BILLED_S


def _cached_tokens(usage: Mapping[str, Any]) -> int:
    """Cached prompt tokens (not counted toward limits), read defensively; unknown shapes -> 0."""
    for key in ('prompt_tokens_details', 'input_tokens_details'):
        details = usage.get(key)
        if isinstance(details, dict):
            n = _to_int(details.get('cached_tokens'))
            if n:
                return max(0, n)
    n = _to_int(usage.get('cached_tokens'))
    return max(0, n) if n else 0


def _body_tokens(body: Any) -> int:
    usage = body.get('usage') if isinstance(body, dict) else None
    if isinstance(usage, dict):
        total = _to_int(usage.get('total_tokens'))
        if total is None:
            total = (_to_int(usage.get('prompt_tokens')) or 0) + (_to_int(usage.get('completion_tokens')) or 0)
        return max(0, total - _cached_tokens(usage))
    return 0


def _is_groq(row: Optional[Mapping[str, Any]]) -> bool:
    row = row or {}
    return row.get('api_family') == 'groq' or 'api.groq.com' in str(row.get('base_url') or '')


def record_response(connection_id: str, model: str, kind: str, status: int,
                    headers: Optional[Mapping[str, Any]] = None, body: Any = None,
                    audio_s: Optional[float] = None, now: Optional[float] = None) -> None:
    """Record one response (2xx or 429). May raise; use :func:`safe_record` in request paths."""
    ts = _clock(now)
    kind = 'asr' if kind == 'asr' else 'chat'
    ok = 200 <= int(status) < 300
    tokens = _body_tokens(body) if ok and kind == 'chat' else 0
    audio = 0.0
    if ok and kind == 'asr':
        if audio_s is None and isinstance(body, dict):
            audio_s = body.get('duration')
        audio = billed_audio_s(audio_s)
    parsed = parse_rate_limit_headers(headers)
    cid, model = str(connection_id or ''), str(model or '')
    minute, hour = int(ts // 60) * 60, int(ts // HOUR_S) * HOUR_S
    con = _connect()
    try:
        with con:
            for table, t in (('minute_buckets', minute), ('hour_buckets', hour)):
                con.execute(
                    f'INSERT INTO {table}(connection_id,model,t,requests,tokens,audio_s) VALUES(?,?,?,?,?,?) '
                    f'ON CONFLICT(connection_id,model,t) DO UPDATE SET requests=requests+1, '
                    f'tokens=tokens+excluded.tokens, audio_s=audio_s+excluded.audio_s',
                    (cid, model, t, 1, tokens, audio))
            if any(v is not None for v in parsed.values()):
                def reset_at(key):
                    return None if parsed[key] is None else ts + float(parsed[key])
                con.execute(
                    'INSERT INTO last_seen VALUES(?,?,?,?,?,?,?,?,?,?) '
                    'ON CONFLICT(connection_id,model) DO UPDATE SET kind=excluded.kind, '
                    'req_limit=COALESCE(excluded.req_limit,req_limit), '
                    'req_remaining=COALESCE(excluded.req_remaining,req_remaining), '
                    'req_reset_at=COALESCE(excluded.req_reset_at,req_reset_at), '
                    'tok_limit=COALESCE(excluded.tok_limit,tok_limit), '
                    'tok_remaining=COALESCE(excluded.tok_remaining,tok_remaining), '
                    'tok_reset_at=COALESCE(excluded.tok_reset_at,tok_reset_at), updated_at=excluded.updated_at',
                    (cid, model, kind, parsed['requests_limit'], parsed['requests_remaining'],
                     reset_at('requests_reset_s'), parsed['tokens_limit'], parsed['tokens_remaining'],
                     reset_at('tokens_reset_s'), ts))
            else:
                con.execute(
                    'INSERT INTO last_seen(connection_id,model,kind,updated_at) VALUES(?,?,?,?) '
                    'ON CONFLICT(connection_id,model) DO UPDATE SET updated_at=excluded.updated_at',
                    (cid, model, kind, ts))
            con.execute('DELETE FROM minute_buckets WHERE t < ?', (int(ts - MINUTE_RETENTION_S),))
            con.execute('DELETE FROM hour_buckets WHERE t < ?', (int(ts - HOUR_RETENTION_S),))
    finally:
        con.close()


def safe_record(*args: Any, **kwargs: Any) -> None:
    """Never raises: usage statistics must not affect the request path."""
    try:
        record_response(*args, **kwargs)
    except Exception:
        pass


def record_for_row(row: Optional[Mapping[str, Any]], route: str, status: int,
                   headers: Optional[Mapping[str, Any]], body: Any,
                   audio_s: Optional[float] = None) -> None:
    """Hook entry for model rows (as used by model_adapters); only Groq rows are tracked."""
    try:
        if not _is_groq(row):
            return
        kind = 'asr' if str(route).rstrip('/').endswith('/audio/transcriptions') else 'chat'
        safe_record(row.get('connection_id') or row.get('id') or '',
                    row.get('remote_model') or row.get('id') or '', kind, status, headers, body, audio_s)
    except Exception:
        pass


# ---- snapshot ---------------------------------------------------------------------------
def _fraction(remaining: Optional[float], limit: Optional[float]) -> Optional[float]:
    if remaining is None or not limit or limit <= 0:
        return None
    return max(0.0, min(1.0, remaining / limit))


def warning_level(fractions: List[Optional[float]]) -> str:
    known = [f for f in fractions if f is not None]
    if not known:
        return 'unknown'
    worst = min(known)
    if worst < WARN_CRITICAL_FRACTION:
        return 'critical'
    if worst < WARN_LOW_FRACTION:
        return 'low'
    return 'ok'


def _window_sum(con: sqlite3.Connection, cid: str, model: str, now: float, span: int, col: str = 'audio_s') -> float:
    cutoff = int((now - span) // 60) * 60 + 60
    row = con.execute(f'SELECT COALESCE(SUM({col}),0) FROM minute_buckets '
                      'WHERE connection_id=? AND model=? AND t>=? AND t<=?', (cid, model, cutoff, now)).fetchone()
    return float(row[0] or 0.0)


def _scope(limit, remaining, reset_at, now) -> Dict[str, Any]:
    return {'limit': limit, 'remaining': remaining,
            'reset_in_s': None if reset_at is None else round(max(0.0, reset_at - now), 3)}


def usage_snapshot(now: Optional[float] = None) -> Dict[str, Any]:
    """Pure read of the usage store; contract version 1 (see module docs / stage B/C)."""
    ts = _clock(now)
    snap: Dict[str, Any] = {'version': 1, 'generated_at': ts, 'providers': []}
    try:
        if not db_path().exists():
            return snap
        con = _connect()
    except Exception:
        return snap
    try:
        conns: Dict[str, Dict[str, Any]] = {}
        rows = con.execute('SELECT connection_id,model,kind,req_limit,req_remaining,req_reset_at,'
                           'tok_limit,tok_remaining,tok_reset_at,updated_at FROM last_seen '
                           'ORDER BY connection_id, model').fetchall()
        for cid, model, kind, rl, rr, rra, tl, tr, tra, upd in rows:
            entry = conns.setdefault(cid, {'provider': 'groq', 'connection_id': cid, 'models': []})
            table = official_limits(model)
            hour_used = _window_sum(con, cid, model, ts, HOUR_S, 'audio_s')
            day_used = _window_sum(con, cid, model, ts, DAY_S, 'audio_s')
            day_tokens = int(_window_sum(con, cid, model, ts, DAY_S, 'tokens'))
            minute_requests = int(_window_sum(con, cid, model, ts, 60, 'requests'))

            asr = kind == 'asr'
            h_lim, d_lim, audio_verified = _audio_limits_for(model)
            if not asr:
                h_lim, d_lim = None, None
                audio_verified = False

            eff_rl = rl if rl is not None else table.get('rpd')
            eff_tl = tl if tl is not None else table.get('tpm')
            rpm_lim = table.get('rpm')
            tpd_lim = table.get('tpd')

            has_header_limit = (rl is not None) or (tl is not None)
            has_table_limit = bool(table)
            if has_header_limit and has_table_limit:
                limits_source = 'mixed'
            elif has_header_limit:
                limits_source = 'header'
            elif has_table_limit:
                limits_source = 'table'
            else:
                limits_source = 'table'

            fracs = [_fraction(rr, eff_rl), _fraction(tr, eff_tl)]
            if asr:
                fracs += [_fraction(h_lim - hour_used, h_lim) if h_lim else None,
                          _fraction(d_lim - day_used, d_lim) if d_lim else None]
            if tpd_lim:
                fracs.append(_fraction(tpd_lim - day_tokens, tpd_lim))
            if rpm_lim:
                fracs.append(_fraction(rpm_lim - minute_requests, rpm_lim))

            entry['models'].append({
                'model': model, 'kind': kind,
                'requests': _scope(eff_rl, rr, rra, ts),
                'tokens': _scope(eff_tl, tr, tra, ts),
                'tokens_day': {'used': day_tokens, 'limit': tpd_lim, 'reset_in_s': None},
                'requests_minute': {'used': minute_requests, 'limit': rpm_lim},
                'audio': {'hour': {'used_s': round(hour_used, 3), 'limit_s': h_lim},
                          'day': {'used_s': round(day_used, 3), 'limit_s': d_lim},
                          'limits_verified': bool(audio_verified)},
                'limits_source': limits_source,
                'warning': warning_level(fracs), 'updated_at': upd})
        for cid, entry in conns.items():
            pts = con.execute('SELECT t, SUM(tokens), SUM(audio_s), SUM(requests) FROM hour_buckets '
                              'WHERE connection_id=? AND t>=? GROUP BY t ORDER BY t',
                              (cid, int(ts - HOUR_RETENTION_S))).fetchall()
            entry['series'] = {'granularity': 'hour', 'points': [
                {'t': int(t), 'tokens': int(tk or 0), 'audio_s': float(a or 0.0), 'requests': int(r or 0)}
                for t, tk, a, r in pts]}
            snap['providers'].append(entry)
    except Exception:
        pass
    finally:
        con.close()
    return snap
