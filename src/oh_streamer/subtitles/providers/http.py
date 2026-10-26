"""HTTP transport, error classification, and rate-limiting gate for subtitle providers."""
from __future__ import annotations

from collections import deque
import datetime
from email.utils import parsedate_to_datetime
import json
import io
import os
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from oh_streamer.platform import platform_support


_SENSITIVE_PATTERNS = [
    # Common API key prefixes (Groq gsk_, Google AIza, OpenAI/OpenRouter sk-, HuggingFace hf_)
    re.compile(r"\b(gsk_[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,})\b"),
    # Bearer tokens and auth headers in strings
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)((?:api[-_]?key|subscription[-_]?key|x-goog-api-key|authorization)\s*[:=]\s*)([^\s,;\"'&]+)"),
    # Query parameters carrying secrets
    re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token|secret)=)([^&#\s\"']+)"),
]


def redact_sensitive_text(text: str) -> str:
    """Mask any recognizable API keys, bearer tokens, or query secrets in text."""
    if not text:
        return ""
    out = str(text)
    out = _SENSITIVE_PATTERNS[0].sub("[REDACTED_KEY]", out)
    out = _SENSITIVE_PATTERNS[1].sub(r"\1[REDACTED]", out)
    out = _SENSITIVE_PATTERNS[2].sub(r"\1[REDACTED]", out)
    out = _SENSITIVE_PATTERNS[3].sub(r"\1[REDACTED]", out)
    return out


def _is_content_blocked(error):
    try:
        data = json.loads(error.read(65536).decode('utf-8'))
        detail = data.get('error') or {}
        code = str(detail.get('code') or detail.get('type') or '').casefold()
        return code in {'content_filter', 'content_policy_violation', 'content_blocked',
                        'responsibleaipolicyviolation', 'safety_violation'}
    except (ValueError, TypeError, AttributeError, OSError):
        return False


def _is_request_too_large(error):
    """A 429 whose body says the single request exceeds the per-minute token limit (Groq: "Request too large
    for model ... on tokens per minute"). Waiting never helps; only a smaller request does."""
    try:
        data = json.loads(error.read(65536).decode('utf-8'))
        detail = data.get('error') or {}
        text = f"{detail.get('message') or ''} {detail.get('code') or ''}".casefold()
        return 'request too large' in text or 'request_too_large' in text
    except (ValueError, TypeError, AttributeError, OSError):
        return False


class ProviderError(RuntimeError):
    """Normalized provider exception without leaking sensitive payloads or keys."""

    VALID_KINDS = {
        "auth",
        "rate_limit",
        "quota_daily",
        "timeout",
        "network",
        "server",
        "bad_response",
        "content_blocked",
        "unsupported",
    }

    def __init__(
        self,
        kind: str,
        message: str = "",
        *,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        http_status: Optional[int] = None,
        scope: str = "none",
    ):
        if kind not in self.VALID_KINDS:
            kind = "server"
        clean_msg = redact_sensitive_text(message) or f"Provider {kind} error"
        if http_status:
            clean_msg = f"{clean_msg} (status={http_status})"
        super().__init__(clean_msg)
        self.kind = kind
        self.retryable = retryable
        self.retry_after = retry_after
        self.http_status = http_status
        self.scope = scope if scope in {"minute", "day", "none"} else "none"


def _parse_duration_string(val: str, now_ts: Optional[float] = None) -> Optional[float]:
    """Parse numeric seconds, duration expressions like '2.5s', '1m30s', '6m0s', '200ms', or HTTP dates."""
    if not val:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        sec = float(s)
        return max(0.0, sec)
    except ValueError:
        pass

    # Match compound duration tokens, e.g. "1h30m20s", "1m30s", "2.5s", "200ms", "500us"
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(hours?|h|min(?:ute)?s?|ms|m|sec(?:ond)?s?|s|us|µs)",
        s,
        re.IGNORECASE,
    )
    if matches:
        unit_map = {
            "h": 3600.0, "hour": 3600.0, "hours": 3600.0,
            "m": 60.0, "min": 60.0, "minute": 60.0, "minutes": 60.0,
            "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
            "ms": 0.001,
            "us": 0.000001, "µs": 0.000001,
        }
        total = 0.0
        for num_str, unit in matches:
            mul = unit_map.get(unit.lower(), 1.0)
            total += float(num_str) * mul
        return max(0.0, total)

    try:
        # HTTP-date
        dt = parsedate_to_datetime(s)
        target_ts = dt.timestamp()
        current_ts = time.time() if now_ts is None else now_ts
        return max(0.0, target_ts - current_ts)
    except Exception:
        return None


def _parse_retry_after(headers: Any, now_ts: Optional[float] = None) -> Optional[float]:
    """Parse Retry-After or provider rate-limit reset headers from headers mapping or string."""
    if headers is None:
        return None
    if isinstance(headers, (str, int, float)):
        return _parse_duration_string(str(headers), now_ts=now_ts)

    # Check mapping / HTTPMessage keys
    target_keys = (
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset-audio",
    )
    if hasattr(headers, "get"):
        lowered = None
        if hasattr(headers, "items"):
            try:
                lowered = {str(hk).lower(): hv for hk, hv in headers.items()}
            except Exception:
                lowered = None
        for k in target_keys:
            val = headers.get(k) or headers.get(k.title())
            if val is None and lowered is not None:
                val = lowered.get(k)  # header names are case-insensitive
            if val is not None:
                parsed = _parse_duration_string(val, now_ts=now_ts)
                if parsed is not None:
                    return parsed
    return None


def parse_retry_after(header_val: Any, now_ts: Optional[float] = None) -> Optional[float]:
    """Parse a Retry-After header or rate limit reset into seconds relative to now."""
    return _parse_retry_after(header_val, now_ts=now_ts)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent redirecting credential-bearing requests."""

    def redirect_request(self, *args, **kwargs):
        return None


_FORBIDDEN_QUERY_KEY_RE = re.compile(r"(?i)(?:^|&)(?:key|api_key|apikey|token|access_token|secret|password)=")


def http_post(
    url: str,
    payload: bytes,
    headers: Optional[Dict[str, str]] = None,
    *,
    timeout: float = 180.0,
    transport: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
    now_fn: Optional[Callable[[], float]] = None,
    on_response: Optional[Callable[[int, Dict[str, str], Any], None]] = None,
) -> Dict[str, Any]:
    """Execute an HTTPS POST request, classifying errors and safely parsing JSON response.

    No authorization keys or query params are included in error messages.
    """
    from urllib.parse import urlsplit
    parsed = urlsplit(str(url))
    if parsed.username or parsed.password or _FORBIDDEN_QUERY_KEY_RE.search(parsed.query or ""):
        raise ProviderError(
            "unsupported",
            "Credentials in URL or query string are forbidden; use request headers instead",
            retryable=False,
            scope="none",
        )
    now = now_fn() if now_fn else time.time()
    req_headers = dict(headers or {})
    req_headers.setdefault("Accept", "application/json")
    req_headers.setdefault("User-Agent", "OhStreamer/1.0")
    req = urllib.request.Request(url, data=payload, headers=req_headers, method="POST")

    def _default_transport(r: urllib.request.Request, to: float) -> Tuple[int, Dict[str, str], bytes]:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(r, timeout=to) as resp:
            status = getattr(resp, "status", 200)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return status, resp_headers, resp.read()

    try:
        if transport is not None:
            raw_res = transport(req, timeout)
            if isinstance(raw_res, tuple) and len(raw_res) == 3:
                status, resp_headers, body = raw_res
            else:
                status, resp_headers, body = 200, {}, raw_res
        else:
            status, resp_headers, body = _default_transport(req, timeout)
        if status >= 300:
            raise urllib.error.HTTPError(url, status, 'Provider request rejected', resp_headers, io.BytesIO(body))
    except urllib.error.HTTPError as exc:
        status = exc.code
        resp_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        retry_after = _parse_retry_after(resp_headers, now_ts=now)
        if status in (400, 403) and _is_content_blocked(exc):
            raise ProviderError('content_blocked', 'Content blocked by provider', http_status=status) from None
        if status == 404:
            raise ProviderError('unsupported', 'Endpoint or model unavailable', http_status=status) from None
        if status in (401, 403):
            raise ProviderError("auth", "Authentication failed or access denied",
                                retryable=False, http_status=status, scope="none") from None
        if status == 429 and on_response is not None:
            try:
                on_response(429, resp_headers, None)
            except Exception:
                pass
        if status == 429 and _is_request_too_large(exc):
            # bad_response, not rate_limit: the same request would be refused forever (S1). The translation
            # executor halves the batch (slice_results.batch_needs_shrink); other callers retry, then move on.
            too_large = ProviderError("bad_response", "Request too large for the provider's per-minute token limit",
                                      retryable=False, http_status=429, scope="none")
            too_large.too_large = True
            raise too_large from None
        if status == 429:
            # If retry-after is > 90s, do not wait; raise with retry_after for upper level fallback
            if retry_after is not None and retry_after > 90:
                scope = "day" if retry_after > 300 else "minute"
                kind = "quota_daily" if retry_after > 300 else "rate_limit"
                raise ProviderError(kind, f"Rate limit / quota exceeded (long window: {retry_after:.1f}s)",
                                    retryable=False, retry_after=retry_after, http_status=429, scope=scope) from None
            raise ProviderError("rate_limit", "Rate limit exceeded",
                                retryable=True, retry_after=retry_after, http_status=429, scope="minute") from None
        if 500 <= status <= 599:
            raise ProviderError("server", "Remote server error",
                                retryable=True, http_status=status, scope="none") from None
        # 4xx client errors
        raise ProviderError("bad_response", "Client request error",
                            retryable=False, http_status=status, scope="none") from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ProviderError("timeout", "Network request timed out",
                                retryable=True, scope="none") from None
        raise ProviderError("network", "Network connection failed",
                            retryable=True, scope="none") from None
    except (socket.timeout, TimeoutError):
        raise ProviderError("timeout", "Network request timed out",
                            retryable=True, scope="none") from None
    except Exception:
        raise ProviderError("network", "Network error occurred",
                            retryable=True, scope="none") from None

    try:
        text = body.decode("utf-8")
        parsed_body = json.loads(text)
    except Exception:
        parsed_body = None
    if on_response is not None:
        try:
            on_response(status, resp_headers, parsed_body)
        except Exception:
            pass
    if parsed_body is None:
        raise ProviderError("bad_response", "Malformed or non-JSON response from server",
                            retryable=False, http_status=status, scope="none") from None
    return parsed_body


def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    *,
    timeout: float = 60.0,
    transport: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
    now_fn: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Execute an HTTPS GET request, classifying errors and safely parsing JSON response."""
    from urllib.parse import urlsplit
    parsed = urlsplit(str(url))
    if parsed.username or parsed.password or _FORBIDDEN_QUERY_KEY_RE.search(parsed.query or ""):
        raise ProviderError(
            "unsupported",
            "Credentials in URL or query string are forbidden; use request headers instead",
            retryable=False,
            scope="none",
        )
    now = now_fn() if now_fn else time.time()
    req_headers = dict(headers or {})
    req_headers.setdefault("Accept", "application/json")
    req_headers.setdefault("User-Agent", "OhStreamer/1.0")
    req = urllib.request.Request(url, headers=req_headers, method="GET")

    def _default_transport(r: urllib.request.Request, to: float) -> Tuple[int, Dict[str, str], bytes]:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(r, timeout=to) as resp:
            status = getattr(resp, "status", 200)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return status, resp_headers, resp.read()

    try:
        if transport is not None:
            raw_res = transport(req, timeout)
            if isinstance(raw_res, tuple) and len(raw_res) == 3:
                status, resp_headers, body = raw_res
            else:
                status, resp_headers, body = 200, {}, raw_res
        else:
            status, resp_headers, body = _default_transport(req, timeout)
        if status >= 300:
            raise urllib.error.HTTPError(url, status, 'Provider request rejected', resp_headers, io.BytesIO(body))
    except urllib.error.HTTPError as exc:
        status = exc.code
        resp_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        retry_after = _parse_retry_after(resp_headers, now_ts=now)
        if status in (400, 403) and _is_content_blocked(exc):
            raise ProviderError('content_blocked', 'Content blocked by provider', http_status=status) from None
        if status == 404:
            raise ProviderError('unsupported', 'Endpoint or model unavailable', http_status=status) from None
        if status in (401, 403):
            raise ProviderError("auth", "Authentication failed or access denied",
                                retryable=False, http_status=status, scope="none") from None
        if status == 429:
            if retry_after is not None and retry_after > 90:
                scope = "day" if retry_after > 300 else "minute"
                kind = "quota_daily" if retry_after > 300 else "rate_limit"
                raise ProviderError(kind, f"Rate limit / quota exceeded (long window: {retry_after:.1f}s)",
                                    retryable=False, retry_after=retry_after, http_status=429, scope=scope) from None
            raise ProviderError("rate_limit", "Rate limit exceeded",
                                retryable=True, retry_after=retry_after, http_status=429, scope="minute") from None
        if 500 <= status <= 599:
            raise ProviderError("server", "Remote server error",
                                retryable=True, http_status=status, scope="none") from None
        raise ProviderError("bad_response", "Client request error",
                            retryable=False, http_status=status, scope="none") from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ProviderError("timeout", "Network request timed out",
                                retryable=True, scope="none") from None
        raise ProviderError("network", "Network connection failed",
                            retryable=True, scope="none") from None
    except (socket.timeout, TimeoutError):
        raise ProviderError("timeout", "Network request timed out",
                            retryable=True, scope="none") from None
    except Exception:
        raise ProviderError("network", "Network error occurred",
                            retryable=True, scope="none") from None

    try:
        text = body.decode("utf-8")
        return json.loads(text)
    except Exception:
        raise ProviderError("bad_response", "Malformed or non-JSON response from server",
                            retryable=False, http_status=status, scope="none") from None


class RequestGate:
    """Token-bucket and sliding window rate limiter with quota persistence."""

    def __init__(
        self,
        *,
        rpm: Optional[int] = None,
        rpd: Optional[int] = None,
        tpm: Optional[int] = None,
        audio_sec_per_hour: Optional[float] = None,
        audio_sec_per_day: Optional[float] = None,
        quota_file: Optional[Path | str] = None,
        identity: str = "default",
        clock: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.rpm = rpm
        self.rpd = rpd
        self.tpm = tpm
        self.audio_sec_per_hour = audio_sec_per_hour
        self.audio_sec_per_day = audio_sec_per_day
        self.quota_file = Path(quota_file) if quota_file else None
        self.identity = identity
        self.clock = clock
        self.sleep_fn = sleep_fn

        # Sliding window history: deque of (timestamp, tokens, audio_seconds)
        self._history: deque[Tuple[float, int, float]] = deque()

    def _default_quota_file(self) -> Path:
        folder = platform_support.data_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "subtitle_quota.json"

    def _read_persistent_daily(self, today_str: str) -> Dict[str, Any]:
        target = self.quota_file or self._default_quota_file()
        if not target.is_file():
            return {"date": today_str, "requests": 0, "tokens": 0, "audio_seconds": 0.0}
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = data.get(self.identity, {})
            if rec.get("date") == today_str:
                return rec
            return {"date": today_str, "requests": 0, "tokens": 0, "audio_seconds": 0.0}
        except Exception:
            return {"date": today_str, "requests": 0, "tokens": 0, "audio_seconds": 0.0}

    def _write_persistent_daily(self, rec: Dict[str, Any]):
        target = self.quota_file or self._default_quota_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {}
            if target.is_file():
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data[self.identity] = rec
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(target)
        except Exception:
            pass

    def acquire(
        self,
        *,
        requests: int = 1,
        tokens: int = 0,
        audio_seconds: float = 0.0,
        max_wait: float = 30.0,
    ):
        """Acquire permission to send a request, waiting if within rate limit, or raising ProviderError."""
        now = self.clock()
        today_str = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

        # 1. Check daily limits first
        daily_rec = self._read_persistent_daily(today_str)
        if self.rpd is not None and (daily_rec["requests"] + requests) > self.rpd:
            raise ProviderError(
                "quota_daily",
                f"Daily request quota exceeded ({daily_rec['requests']}/{self.rpd})",
                retryable=False,
                scope="day",
            )
        if self.audio_sec_per_day is not None and (daily_rec["audio_seconds"] + audio_seconds) > self.audio_sec_per_day:
            raise ProviderError(
                "quota_daily",
                f"Daily audio seconds quota exceeded ({daily_rec['audio_seconds']}/{self.audio_sec_per_day})",
                retryable=False,
                scope="day",
            )

        # 2. Check 1-hour limits (audio_sec_per_hour)
        hour_ago = now - 3600.0
        while self._history and self._history[0][0] < hour_ago:
            self._history.popleft()

        if self.audio_sec_per_hour is not None:
            hourly_audio = sum(entry[2] for entry in self._history)
            if (hourly_audio + audio_seconds) > self.audio_sec_per_hour:
                oldest_in_hour = self._history[0][0]
                wait_sec = max(0.0, (oldest_in_hour + 3600.0) - now)
                raise ProviderError(
                    "rate_limit",
                    f"Hourly audio seconds quota exceeded",
                    retryable=True,
                    retry_after=wait_sec,
                    scope="minute",
                )

        # 3. Check 1-minute limits (RPM, TPM)
        minute_ago = now - 60.0
        min_entries = [entry for entry in self._history if entry[0] >= minute_ago]

        # Calculate potential wait needed
        wait_for_rpm = 0.0
        if self.rpm is not None and len(min_entries) >= self.rpm:
            # Need to wait until oldest entry drops out of the 60s window
            oldest_ts = min_entries[0][0]
            wait_for_rpm = max(0.0, (oldest_ts + 60.0) - now)

        wait_for_tpm = 0.0
        if self.tpm is not None:
            recent_tokens = sum(entry[1] for entry in min_entries)
            if (recent_tokens + tokens) > self.tpm and min_entries:
                oldest_ts = min_entries[0][0]
                wait_for_tpm = max(0.0, (oldest_ts + 60.0) - now)

        required_wait = max(wait_for_rpm, wait_for_tpm)
        if required_wait > 0.0:
            if required_wait > max_wait:
                raise ProviderError(
                    "rate_limit",
                    f"Rate limit exceeded (wait {required_wait:.1f}s exceeds limit {max_wait:.1f}s)",
                    retryable=True,
                    retry_after=required_wait,
                    scope="minute",
                )
            self.sleep_fn(required_wait)
            now = self.clock()

        # Record usage
        self._history.append((now, tokens, audio_seconds))
        daily_rec["requests"] += requests
        daily_rec["tokens"] += tokens
        daily_rec["audio_seconds"] += audio_seconds
        self._write_persistent_daily(daily_rec)


class CloudAsrRatePacer:
    """Thread-safe dual-dimension rate limiter (RPM + Hourly Audio Seconds) with persistent ledger."""

    def __init__(
        self,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
        ledger_path: Optional[Path | str] = None,
        safe_rpm: float = 18.0,
        burst_limit: int = 2,
        min_interval_sec: float = 3.4,
        clock: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        import threading
        self.base_url = str(base_url or "").rstrip("/")
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.safe_rpm = safe_rpm
        self.burst_limit = burst_limit
        self.min_interval_sec = min_interval_sec
        self.clock = clock
        self.sleep_fn = sleep_fn
        self._lock = threading.RLock()

        # In-memory history: key -> deque of (timestamp, audio_seconds)
        self._history: Dict[str, deque[Tuple[float, float]]] = {}
        # Track last request time for RPM spacing: key -> timestamp
        self._last_req: Dict[str, float] = {}

    def _default_ledger_file(self) -> Optional[Path]:
        env_ledger = os.environ.get("OH_STREAMER_ASR_LEDGER_PATH")
        if env_ledger:
            return Path(env_ledger)
        if "PYTEST_CURRENT_TEST" in os.environ or "example.org" in self.base_url or "example.com" in self.base_url:
            return None
        folder = platform_support.data_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "cloud_asr_ledger.json"

    def _key(self, model_name: str) -> str:
        return f"{self.base_url}::{model_name}"

    def _read_persistent_ledger(self) -> Dict[str, List[Tuple[float, float]]]:
        target = self.ledger_path or self._default_ledger_file()
        if not target or not target.is_file():
            return {}
        try:
            with open(target, "r", encoding="utf-8") as f:
                raw = json.load(f)
            now = self.clock()
            hour_ago = now - 3600.0
            cleaned: Dict[str, List[Tuple[float, float]]] = {}
            for k, entries in raw.items():
                if isinstance(entries, list):
                    valid = [(float(e[0]), float(e[1])) for e in entries if isinstance(e, (list, tuple)) and len(e) >= 2 and float(e[0]) >= hour_ago]
                    cleaned[k] = valid
            return cleaned
        except Exception:
            return {}

    def _write_persistent_ledger(self, data: Dict[str, List[Tuple[float, float]]]):
        target = self.ledger_path or self._default_ledger_file()
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            tmp.replace(target)
        except Exception:
            pass

    def _sync_from_disk(self, key: str):
        target = self.ledger_path or self._default_ledger_file()
        if not target or not target.is_file():
            return
        now = self.clock()
        hour_ago = now - 3600.0
        disk_data = self._read_persistent_ledger()
        if not disk_data:
            return
        entries = disk_data.get(key, [])
        existing = list(self._history.get(key, ()))
        merged = sorted(set(existing) | {(float(ts), float(aud)) for ts, aud in entries if float(ts) >= hour_ago})
        q = deque()
        for ts, aud in merged:
            if ts >= hour_ago:
                q.append((ts, aud))
        self._history[key] = q

    def _prune(self, key: str, now: float):
        hour_ago = now - 3600.0
        q = self._history.setdefault(key, deque())
        while q and q[0][0] < hour_ago:
            q.popleft()

    def get_hourly_ash(self, model_name: str) -> float:
        """Return total audio seconds consumed in the past 3600s for this model."""
        with self._lock:
            key = self._key(model_name)
            self._sync_from_disk(key)
            now = self.clock()
            self._prune(key, now)
            return sum(item[1] for item in self._history.get(key, []))

    def can_afford(
        self,
        model_name: str,
        audio_seconds: float,
        ash_limit: float = 7000.0,
    ) -> Tuple[bool, float]:
        """Check if requested audio duration fits in the 1-hour window.

        Returns (can_afford, wait_seconds).
        """
        with self._lock:
            key = self._key(model_name)
            self._sync_from_disk(key)
            now = self.clock()
            self._prune(key, now)

            needed = max(10.0, float(audio_seconds))
            q = self._history.get(key, deque())
            current_ash = sum(item[1] for item in q)

            if (current_ash + needed) <= ash_limit:
                return True, 0.0

            # Calculate wait needed until enough old entries drop out
            freed = 0.0
            for ts, aud in q:
                freed += aud
                if (current_ash - freed + needed) <= ash_limit:
                    wait_sec = max(0.0, (ts + 3600.0) - now)
                    return False, wait_sec

            # Even if all current drop, if needed still > ash_limit
            if needed > ash_limit:
                return False, 3600.0
            if q:
                wait_sec = max(0.0, (q[-1][0] + 3600.0) - now)
                return False, wait_sec
            return False, 0.0

    def acquire(
        self,
        model_name: str,
        audio_seconds: float,
        *,
        ash_limit: float = 7000.0,
        max_wait: float = 30.0,
    ) -> None:
        """Paces RPM and hourly ASH budget, sleeping if wait <= max_wait or raising ProviderError."""
        with self._lock:
            key = self._key(model_name)
            self._sync_from_disk(key)
            now = self.clock()
            self._prune(key, now)

            # 1. Check ASH affordability
            afford, wait_ash = self.can_afford(model_name, audio_seconds, ash_limit=ash_limit)
            if not afford:
                if wait_ash > max_wait:
                    raise ProviderError(
                        "rate_limit",
                        f"Hourly audio seconds quota exceeded for {model_name} (wait {wait_ash:.1f}s)",
                        retryable=(wait_ash <= 90.0),
                        retry_after=wait_ash,
                        scope="minute" if wait_ash <= 300.0 else "day",
                    )
                if wait_ash > 0.0:
                    self.sleep_fn(wait_ash)
                    now = self.clock()
                    self._prune(key, now)

            # 2. Check RPM window (60s)
            minute_ago = now - 60.0
            q = self._history.get(key, deque())
            min_entries = [e for e in q if e[0] >= minute_ago]
            wait_rpm = 0.0
            if len(min_entries) >= self.safe_rpm:
                oldest_in_min = min_entries[0][0]
                wait_rpm = max(0.0, (oldest_in_min + 60.0) - now)

            # 3. Check burst spacing
            last_t = self._last_req.get(key, 0.0)
            elapsed = now - last_t
            # If recently requested within 60s more than burst_limit, enforce min_interval
            if len(min_entries) >= self.burst_limit and elapsed < self.min_interval_sec:
                wait_burst = self.min_interval_sec - elapsed
                wait_rpm = max(wait_rpm, wait_burst)

            if wait_rpm > 0.0:
                if wait_rpm > max_wait:
                    raise ProviderError(
                        "rate_limit",
                        f"RPM rate limit exceeded for {model_name} (wait {wait_rpm:.1f}s)",
                        retryable=True,
                        retry_after=wait_rpm,
                        scope="minute",
                    )
                self.sleep_fn(wait_rpm)
                now = self.clock()

            self._last_req[key] = now

    def record_usage(
        self,
        model_name: str,
        audio_seconds: float,
        ts: Optional[float] = None,
    ) -> None:
        """Record successful audio transcription usage in memory and persistent ledger."""
        with self._lock:
            key = self._key(model_name)
            now = ts if ts is not None else self.clock()
            billed = max(10.0, float(audio_seconds))
            self._sync_from_disk(key)
            self._prune(key, now)

            q = self._history.setdefault(key, deque())
            q.append((now, billed))
            self._last_req[key] = now

            # Persist all models to ledger
            disk_data = self._read_persistent_ledger()
            disk_data[key] = list(q)
            self._write_persistent_ledger(disk_data)

    def reset(self, model_name: Optional[str] = None):
        """Reset history in memory and on disk for testing."""
        with self._lock:
            if model_name:
                key = self._key(model_name)
                self._history.pop(key, None)
                self._last_req.pop(key, None)
                disk_data = self._read_persistent_ledger()
                disk_data.pop(key, None)
                self._write_persistent_ledger(disk_data)
            else:
                self._history.clear()
                self._last_req.clear()
                self._write_persistent_ledger({})


_GLOBAL_PACERS: Dict[str, CloudAsrRatePacer] = {}
_GLOBAL_PACERS_LOCK = threading.RLock()


def get_cloud_asr_pacer(
    base_url: str = "https://api.groq.com/openai/v1",
    *,
    ledger_path: Optional[Path | str] = None,
) -> CloudAsrRatePacer:
    """Get or create singleton CloudAsrRatePacer instance for a given base URL."""
    clean_url = str(base_url or "").rstrip("/")
    with _GLOBAL_PACERS_LOCK:
        if clean_url not in _GLOBAL_PACERS or ledger_path is not None:
            pacer = CloudAsrRatePacer(clean_url, ledger_path=ledger_path)
            if ledger_path is None:
                _GLOBAL_PACERS[clean_url] = pacer
            return pacer
        return _GLOBAL_PACERS[clean_url]
