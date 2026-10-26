"""Bounded inference policy and account-scoped, process-safe failure ledger.

Only frozen, explicitly selected bindings enter this policy. SQLite transactions
serialize workers without keeping a lock (or a model) alive during waits.
"""
from __future__ import annotations

import hashlib
import copy
import json
import random
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from oh_streamer.platform import platform_support
from .providers.http import ProviderError

RUNTIME_VERSION = 'bounded-fallback-1'


def identity(row, settings=None):
    """Public provenance/cache identity; never includes credentials. A local model is identified by its model
    name (relative to the models folder) plus a file fingerprint, so moving the models folder keeps the key;
    ``settings`` only locates the files for the fingerprint."""
    result = {k: row[k] for k in ('id', 'provider', 'connection_id', 'remote_model',
                                 'base_url', 'recipe', 'model_name', 'model_path') if k in row}
    if row.get('provider') == 'local':
        try:
            from .environment import model_file
            located = model_file(row, settings)
            if not located:
                raise KeyError('model_path')
            path = Path(located)
            if row.get('kind') == 'asr':
                if row.get('recipe') == '__dual__':
                    from .runtime import local_model_fingerprint, model_dir_name
                    p1 = path / model_dir_name('whisper-ja-1.5B')
                    p2 = path / model_dir_name('anime-whisper')
                    result['revision'] = [local_model_fingerprint(p1), local_model_fingerprint(p2)]
                else:
                    from .runtime import local_model_fingerprint
                    result['revision'] = local_model_fingerprint(path)
            else:
                stat = path.stat()
                result['revision'] = [stat.st_size, stat.st_mtime_ns]
        except (OSError, KeyError):
            result['revision'] = 'missing'
    return result


def cache_identity(plan, settings=None):
    result = copy.deepcopy(plan)
    result['fallback_runtime'] = RUNTIME_VERSION
    result['local_revisions'] = [identity(row, settings) for row in
        [*plan['models'].values(), *plan.get('fallback', {}).get('asr', []),
         *plan.get('fallback', {}).get('translation', [])]
        if row and row.get('provider') == 'local']
    return result


# Injectable/patchable RNG (returns float in [0, 1)); tests replace it for determinism.
_random = random.random


def backoff_delay(retry: int) -> float:
    """Exponential backoff with equal jitter: base/2 + random(0, base/2), base = min(2**(retry+1), 30)."""
    base = min(2 ** (retry + 1), 30)
    return base * (0.5 + 0.5 * _random())


class Ledger:
    def __init__(self, path=None, clock=time.time):
        self.path = Path(path or (platform_support.data_dir() / 'subtitle-fallback.sqlite3'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, until REAL, kind TEXT, probe REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS usage (key TEXT, at REAL, tokens REAL, audio REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS usage_key ON usage(key, at)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level='IMMEDIATE')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def account(row, key):
        # Same credential under different connection IDs shares one account.
        raw = [row.get('base_url', '').rstrip('/'), key]
        return hashlib.sha256(json.dumps(raw).encode()).hexdigest()

    def claim(self, key):
        """One probe after cooldown; concurrent workers keep using backups."""
        now = self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rec = db.execute('SELECT until, kind, probe FROM state WHERE key=?', (key,)).fetchone()
            if not rec:
                return None
            until, kind, probe = rec
            if max(until, probe) > now:
                return ProviderError(kind, 'Candidate cooling down', retryable=kind in TRANSIENT,
                                     retry_after=max(until, probe) - now)
            db.execute('UPDATE state SET probe=? WHERE key=?', (now + 600, key))
        return 'probe'

    def fail(self, key, error, cooldown):
        until = self.clock() + max(cooldown, float(error.retry_after or 0))
        with self.connect() as db:
            db.execute('INSERT INTO state VALUES (?, ?, ?, 0) ON CONFLICT(key) DO UPDATE SET until=excluded.until, kind=excluded.kind, probe=0',
                       (key, until, error.kind))

    def success(self, key):
        with self.connect() as db:
            db.execute('DELETE FROM state WHERE key=?', (key,))

    def peek(self, key):
        """(until, kind) of a candidate key that is cooling down now, else None. Read-only: no probe claimed."""
        now = self.clock()
        with self.connect() as db:
            rec = db.execute('SELECT until, kind, probe FROM state WHERE key=?', (key,)).fetchone()
        if not rec or max(rec[0], rec[2]) <= now:
            return None
        return max(rec[0], rec[2]), rec[1]

    def release_probe(self, key):
        with self.connect() as db:
            db.execute('UPDATE state SET probe=0 WHERE key=?', (key,))

    def reserve(self, account, *, tokens=0, audio=0):
        """Conservative Groq pacing shared across models, jobs and processes.

        Record attempts before sending; a lost response cannot refund quota.
        These internal safety budgets are not claims about account allowances.
        `tokens` is the request's max_tokens, a loose upper bound (128 per subtitle line). Reserving it in full let
        only one translation batch per minute through, while the real Groq account accepted about four 30-line
        batches back to back before its first 429 (S6 measurement). So it is weighted /16 (~4 full batches per
        minute); the provider's own 429 + Retry-After remains the actual limit.
        """
        now = self.clock()
        weight = min(tokens / 16, 1000) if tokens else 0
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM usage WHERE at < ?', (now - 3600,))
            rows = db.execute('SELECT at,tokens,audio FROM usage WHERE key=? ORDER BY at', (account,)).fetchall()
            waits = [max(0, rows[-1][0] + (3.4 if audio else 2.0) - now)] if rows else []
            recent = [r for r in rows if r[0] > now - 60]
            if len(recent) >= (18 if audio else 30):
                waits.append(recent[0][0] + 60 - now)
            if weight and sum(r[1] for r in recent) + weight > 1000:
                waits.append(recent[0][0] + 60 - now)
            if audio and sum(r[2] for r in rows) + audio > 6900 and rows:
                waits.append(rows[0][0] + 3600 - now)
            wait = max(waits, default=0)
            if wait > 0:
                raise ProviderError('rate_limit', 'Shared account pacing', retryable=True, retry_after=wait)
            db.execute('INSERT INTO usage VALUES (?, ?, ?, ?)', (account, now, weight, audio))


TRANSIENT = {'rate_limit', 'quota_daily', 'network', 'timeout', 'server'}
NO_SWITCH = {'content_blocked'}
ADMISSION_MAX_WAIT = 90     # our own pacing window is 60 s: wait it out instead of failing the subtitle job
RATE_LIMIT_MAX_WAIT = 60    # per-minute provider limits: honour Retry-After up to one window


def client_source(client):
    source = getattr(client, 'last_source', None)
    return source if isinstance(source, dict) else None


class Policy:
    def __init__(self, plan, role, primary, *, ledger=None, check_cancel=None, on_event=None,
                 clock=time.time, sleep=time.sleep):
        self.plan, self.role = plan, role
        self.policy = {**dict(max_attempts=4, retries=1, cooldown_seconds=60), **plan.get('policy', {})}
        self.ledger = ledger or Ledger()
        self.check_cancel = check_cancel or (lambda: None)
        self.on_event = on_event
        self.clock, self.sleep = clock, sleep
        candidates = [primary]
        if role == 'asr' and plan.get('whisper_policy') == 'auto':
            from .model_adapters import GROQ_SIBLING_MODELS
            sibling = GROQ_SIBLING_MODELS.get(primary.get('remote_model'))
            if (primary.get('api_family') == 'groq' or 'api.groq.com' in primary.get('base_url', '')) and sibling in plan.get('whisper_models', []):
                candidates.append(dict(primary, remote_model=sibling))
        candidates.extend(plan.get('fallback', {}).get(role, []))
        self.candidates, seen = [], set()
        for row in candidates:
            if not row:
                continue
            fingerprint = json.dumps({k: row.get(k) for k in ('provider', 'connection_id', 'base_url', 'remote_model', 'recipe', 'model_name', 'model_path')}, sort_keys=True)
            if fingerprint not in seen:
                self.candidates.append(row)
                seen.add(fingerprint)
        self.last_source = None

    def emit(self, state, row=None, error=None, **extra):
        event = dict(type='subtitle_fallback', role=self.role, state=state,
                     model=identity(row) if row else None,
                     reason=error.kind if error else None, **extra)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass  # An observer cannot turn successful inference into a retry.

    def wait(self, seconds, row, error, **extra):
        self.emit('waiting', row, error, retry_after=seconds, **extra)
        end = self.clock() + seconds
        while self.clock() < end:
            self.check_cancel()
            self.sleep(min(.1, end - self.clock()))
        self.check_cancel()

    def run(self, operation, credentials, *, tokens=0, audio=0):
        attempts, last, primary_error = 0, None, None
        seen_accounts = set()
        failures = []
        primary_account = None
        for index, row in enumerate(self.candidates):
            self.check_cancel()
            if attempts >= self.policy['max_attempts']:
                break
            # Credentials are looked up immediately before each candidate; never
            # saved in events or the ledger. Cloud consent errors propagate.
            try:
                key = credentials(row)
            except ProviderError as exc:
                last = exc
                failures.append(exc)
                if index == 0:
                    primary_error = exc
                self.emit('failed', row, exc, attempts=attempts)
                continue
            account = self.ledger.account(row, key)
            if index == 0:
                primary_account = account
            elif self.role == 'asr' and self.plan.get('whisper_policy') == 'fixed' and account == primary_account:
                from .model_adapters import GROQ_SIBLING_MODELS
                if row.get('remote_model') == GROQ_SIBLING_MODELS.get(self.candidates[0].get('remote_model')):
                    self.emit('failed', row, ProviderError('unsupported', 'Fixed Whisper forbids sibling switching'), attempts=attempts)
                    continue
            candidate_key = account + ':' + json.dumps({k: row.get(k) for k in
                ('remote_model', 'recipe', 'model_name', 'model_path', 'kind')}, sort_keys=True)
            if candidate_key in seen_accounts:
                continue
            seen_accounts.add(candidate_key)
            blocked = self.ledger.claim(candidate_key)
            if isinstance(blocked, ProviderError):
                last = blocked
                failures.append(blocked)
                if index == 0:
                    primary_error = blocked
                self.emit('cooldown', row, blocked, retry_after=blocked.retry_after)
                continue
            try:
                retry = 0
                while True:
                    self.check_cancel()
                    if attempts >= self.policy['max_attempts']:
                        break
                    try:
                        if row.get('api_family') == 'groq' or 'api.groq.com' in row.get('base_url', ''):
                            # Admission waits are our own pacing: they do not count as sent API attempts, and a wait
                            # within one window is waited out (cancellable) rather than failing the job.
                            for _ in range(8):
                                try:
                                    self.ledger.reserve(account, tokens=tokens, audio=audio)
                                    break
                                except ProviderError as admission:
                                    if admission.retry_after is None or admission.retry_after > ADMISSION_MAX_WAIT:
                                        raise
                                    self.wait(admission.retry_after, row, admission, pacing=True)
                            else:
                                self.ledger.reserve(account, tokens=tokens, audio=audio)
                        attempts += 1
                        result = operation(row, key)
                        self.ledger.success(candidate_key)
                        self.last_source = identity(row)
                        self.emit('fallback' if index else 'recovered' if blocked == 'probe' else 'primary', row, primary_error,
                                  attempts=attempts, primary_error=primary_error.kind if primary_error else None)
                        return result
                    except ProviderError as exc:
                        if exc.kind in NO_SWITCH:
                            self.ledger.success(candidate_key)
                            raise
                        last = exc
                        failures.append(exc)
                        if index == 0:
                            primary_error = exc
                        delay = float(exc.retry_after) if exc.retry_after is not None else backoff_delay(retry)
                        # A provider 429 that says when to come back is pacing, not a fault. With backups left they are
                        # tried after the normal retries; on the last candidate (e.g. Groq-only translation) the
                        # remaining attempt budget is spent honouring Retry-After (real Groq batches hit 429s of 2-19 s).
                        paced = (exc.kind == 'rate_limit' and exc.retry_after is not None
                                 and delay <= RATE_LIMIT_MAX_WAIT and index == len(self.candidates) - 1)
                        if (exc.kind in TRANSIENT and exc.retryable and attempts < self.policy['max_attempts']
                                and (paced or (retry < self.policy['retries'] and delay <= 30))):
                            if not paced:
                                retry += 1
                            self.wait(delay, row, exc)
                            continue
                        if getattr(exc, 'no_cooldown', False):
                            # A malformed answer to one prompt says nothing about the provider's health.
                            self.ledger.success(candidate_key)
                        else:
                            self.ledger.fail(candidate_key, exc, self.policy['cooldown_seconds'])
                        self.emit('failed', row, exc, attempts=attempts)
                        break
            finally:
                self.ledger.release_probe(candidate_key)
        # Report the most telling failure: a transient one, else the first real service error. A local backup that
        # is merely unavailable in this process ('unsupported') must not hide why the cloud service failed.
        last = (next((error for error in failures if error.kind in TRANSIENT), None)
                or next((error for error in failures if error.kind != 'unsupported'), None) or last)
        last = last or ProviderError('unsupported', 'No usable subtitle candidate')
        self.emit('exhausted', error=last, attempts=attempts,
                  recoverable=last.kind in TRANSIENT,
                  primary_error=primary_error.kind if primary_error else None)
        raise last


# --------------------------------------------------------------------------------------------------
# M4 (design 4.7): one candidate at a time
# --------------------------------------------------------------------------------------------------
#
# `Policy.run` above tries every candidate inside one call and sleeps through quota waits; it stays for the
# legacy paths until M7. The slice scheduler instead asks, per work unit, "which candidate next?"
# (`CandidateChain.next`), runs exactly one candidate once (`run_candidate`, never sleeps) and lets
# `CandidateChain.decide` turn a failure into a plan action (`apply_decision`). The chain is rebuilt from
# the *current* frozen plan on every call (A2): `tried` is keyed by candidate id, never by position, so a
# backup the user adds later shows up at the end of the chain and gets used.

LOCAL_EXECUTOR = 'local'
CLOUD_EXECUTOR = 'cloud'

# A provider 429 with Retry-After cools the model for exactly Retry-After. The old max(60 s, Retry-After)
# held the whole model back for 60 s + 47 s in the M2 run although Groq asked for a few seconds; the
# per-request pacing is the Ledger's job (reserve), so the cooldown only has to cover what the provider said.
# Without Retry-After a shorter guess than the old 60 s is enough: one per-minute window rarely needs more.
RATE_LIMIT_DEFAULT_COOLDOWN = 20.0
# Quota waits up to this long are waited out on the same candidate (defer); longer ones (and quota_daily)
# hand the work to the next candidate if there is one ("cloud rate-limited -> local takes over").
SHORT_QUOTA_WAIT = 30.0
# ... except after this many provider 429s in a row on one unit without progress: waiting out short windows
# again and again can keep a per-minute limit exhausted (S1: one garbled batch resent every ~14 s), so the
# next candidate gets the unit (reclaimable) even though each wait was short. Our own Ledger pacing (not_sent)
# neither counts nor resets the streak; a 429 after the unit made progress (error.progressed) starts a new one.
SHORT_429_SWITCH = 3
DAILY_QUOTA_DEFAULT_WAIT = 3600.0
CANDIDATE_MAX_ATTEMPTS = 3      # runs of one candidate on retryable errors before moving on (= executors' MAX_ATTEMPTS)
DETAIL_MAX = 200

QUOTA_KINDS = frozenset({'rate_limit', 'quota_daily'})
# Retryable on the same candidate. 'error' / 'oom' are the local executor's kinds: a local failure is retried
# like a transient cloud one and only then handed on (D1: being busy is never a failure, see decide()).
RETRY_KINDS = frozenset({'network', 'timeout', 'server', 'bad_response', 'error', 'oom'})

ROLE_KEYS = {'asr': 'asr', 'mt': 'translation', 'translation': 'translation'}

RETRY = 'retry'            # same candidate later (counted attempt)
WAIT_QUOTA = 'wait_quota'  # same candidate after retry_at (not a failure, attempt unchanged)
SWITCH = 'switch'          # next candidate (wait_executor set when it belongs to another executor)
FAILED = 'failed'          # nothing left: failed(exhausted), or a NO_SWITCH error


def executor_of(row) -> str:
    """Executor that can run a binding (same rule as subtitle_scheduler.candidate_executor)."""
    return LOCAL_EXECUTOR if row.get('provider') == 'local' else CLOUD_EXECUTOR


def _fingerprint(row) -> str:
    return json.dumps({k: row.get(k) for k in ('provider', 'connection_id', 'base_url', 'remote_model',
                                               'recipe', 'model_name', 'model_path')}, sort_keys=True)


def _is_groq(row) -> bool:
    return row.get('api_family') == 'groq' or 'api.groq.com' in (row.get('base_url') or '')


def candidate_key(account: str, row) -> str:
    """Ledger key of one candidate under one account (same format as Policy.run)."""
    return account + ':' + json.dumps({k: row.get(k) for k in ('remote_model', 'recipe', 'model_name', 'model_path', 'kind')},
                                      sort_keys=True)


@dataclass(frozen=True)
class Candidate:
    index: int                  # position in the chain built from the current plan (not stored anywhere)
    id: str                     # stable id stored in TaskState.candidate / tried
    row: Mapping = field(compare=False, hash=False)
    executor: str = CLOUD_EXECUTOR
    sibling: bool = False       # same-family sibling model (Groq whisper turbo <-> large)


@dataclass(frozen=True)
class Pick:
    """Answer of CandidateChain.next: the candidate to use, and when it can be used (None = now).
    candidate None = every candidate is used up (exhausted)."""
    candidate: Optional[Candidate]
    retry_at: Optional[float] = None

    @property
    def exhausted(self) -> bool:
        return self.candidate is None

    @property
    def executor(self) -> Optional[str]:
        return self.candidate.executor if self.candidate else None


@dataclass(frozen=True)
class Decision:
    action: str                             # RETRY / WAIT_QUOTA / SWITCH / FAILED
    kind: str                               # error kind that caused it
    at: float
    candidate: Optional[Candidate] = None   # candidate the work is on next (FAILED: the one that failed)
    retry_at: Optional[float] = None
    attempt: int = 0                        # attempt counter for `candidate`
    wait_executor: Optional[str] = None     # reservation to store (None = anyone / chain decides)
    reclaim_at: Optional[float] = None      # quota handoff: original executor may take it back from then
    reclaim_executor: Optional[str] = None
    tried: Mapping = field(default_factory=dict, compare=False)
    fail_kind: Optional[str] = None         # FAILED: 'exhausted' or the NO_SWITCH kind
    detail: str = ''

    @property
    def handoff(self) -> bool:
        """The work moved to a candidate of another executor."""
        return self.action == SWITCH and self.wait_executor is not None


def error_kind(error, kind: Optional[str] = None) -> str:
    if kind:
        return kind
    if isinstance(error, ProviderError):
        return error.kind
    if isinstance(error, PermissionError):
        return 'permission'
    return getattr(error, 'kind', None) or 'error'


def default_retry_delay(candidate: Candidate, attempt: int) -> float:
    """Backoff before the attempt-th retry of one candidate: 5, 10, 20 ... s, at most 60 s."""
    return min(5.0 * 2 ** max(attempt - 1, 0), 60.0)


def candidate_rows(plan, role) -> List[Candidate]:
    """Candidates of `role` ('asr' / 'mt' / 'translation') from a frozen plan, in policy order: primary,
    Groq same-family sibling (ASR, whisper_policy 'auto', sibling listed in whisper_models), fallbacks.
    Duplicates removed as in Policy. Under whisper_policy 'fixed' a fallback that is only the primary's Groq
    sibling on the same connection is dropped (Policy.run compares accounts; without credentials the same
    connection_id + base_url stands in for "same account")."""
    role = ROLE_KEYS.get(role, role)
    primary = (plan.get('models') or {}).get(role)
    rows, sibling_flags = [], []
    if primary:
        rows.append(primary)
        sibling_flags.append(False)
        if role == 'asr' and plan.get('whisper_policy') == 'auto' and _is_groq(primary):
            from .model_adapters import GROQ_SIBLING_MODELS
            sibling = GROQ_SIBLING_MODELS.get(primary.get('remote_model'))
            if sibling in plan.get('whisper_models', []):
                rows.append(dict(primary, remote_model=sibling))
                sibling_flags.append(True)
    fixed_sibling = None
    if primary and role == 'asr' and plan.get('whisper_policy') == 'fixed':
        from .model_adapters import GROQ_SIBLING_MODELS
        fixed_sibling = GROQ_SIBLING_MODELS.get(primary.get('remote_model'))
    for row in (plan.get('fallback') or {}).get(role, []) or []:
        if not row:
            continue
        if (fixed_sibling and row.get('remote_model') == fixed_sibling
                and row.get('connection_id') == primary.get('connection_id')
                and (row.get('base_url') or '').rstrip('/') == (primary.get('base_url') or '').rstrip('/')):
            continue
        rows.append(row)
        sibling_flags.append(False)
    out, seen, ids = [], set(), set()
    for row, sibling in zip(rows, sibling_flags):
        fingerprint = _fingerprint(row)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cid = str(row.get('id') or fingerprint)
        if sibling:
            cid = f"{cid}#{row.get('remote_model')}"
        base, n = cid, 2
        while cid in ids:       # two distinct bindings sharing an id: keep both, ids stay unique
            cid, n = f'{base}#{n}', n + 1
        ids.add(cid)
        out.append(Candidate(index=len(out), id=cid, row=row, executor=executor_of(row), sibling=sibling))
    return out


def ledger_cooldown(ledger: 'Ledger', credentials: Callable) -> Callable[[Candidate], Optional[float]]:
    """`cooldown` function for CandidateChain backed by the real Ledger: candidate -> absolute time its
    cooldown ends (None = usable). Results are memoised for the closure's lifetime, so build one per
    scheduling pass. Candidates whose credentials fail count as usable (run_candidate reports the error)."""
    memo: Dict[str, Optional[float]] = {}

    def cooldown(candidate: Candidate) -> Optional[float]:
        if candidate.id not in memo:
            try:
                key = credentials(candidate.row)
                rec = ledger.peek(candidate_key(ledger.account(candidate.row, key), candidate.row))
                memo[candidate.id] = rec[0] if rec else None
            except Exception:
                memo[candidate.id] = None
        return memo[candidate.id]
    return cooldown


class CandidateChain:
    """The candidate list of one role, built from the current frozen plan, plus the per-unit decisions.

    Build a new chain on every scheduling decision (cheap); never keep it across plan changes.
    `cooldown(candidate) -> until | None` (injectable; ledger_cooldown for the real Ledger) lets fresh work
    skip a primary that is cooling down without failing on it first.
    """

    def __init__(self, plan, role, *, cooldown: Optional[Callable[[Candidate], Optional[float]]] = None,
                 short_wait: float = SHORT_QUOTA_WAIT, max_attempts: int = CANDIDATE_MAX_ATTEMPTS,
                 retry_delay: Callable[[Candidate, int], float] = default_retry_delay):
        self.role = ROLE_KEYS.get(role, role)
        self.candidates = candidate_rows(plan, self.role)
        self.cooldown = cooldown
        self.short_wait = float(short_wait)
        self.max_attempts = int(max_attempts)
        self.retry_delay = retry_delay

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)

    def get(self, candidate_id) -> Optional[Candidate]:
        return next((c for c in self.candidates if c.id == candidate_id), None)

    # -- which candidate --------------------------------------------------------------------------
    def next(self, state=None, now: Optional[float] = None, *, tried: Optional[Mapping] = None,
             current: Any = ...) -> Pick:
        """Candidate the unit should run on now. `state` is a TaskState (or anything with .tried /
        .candidate); `tried` / `current` override it.

        Walks the chain in order and returns the first candidate that is usable:
        - a candidate with a `tried` entry other than `current` was given up and is skipped, unless the entry
          has `retry_at` (left for a quota wait) and that time has passed (the cloud recovered: it takes the
          work back, so a pending unit returns to its earlier candidate);
        - a candidate cooling down in the Ledger (`cooldown`) is waited for when the wait is <= short_wait
          (Pick.retry_at), otherwise skipped;
        - when nothing is usable now, the candidate available soonest (a quota-left or cooling one) with its
          time; Pick(None) only when every candidate was given up for good (exhausted).
        A fresh unit (no entries) therefore starts at the head of the chain.
        """
        now = time.time() if now is None else now
        tried = dict((getattr(state, 'tried', None) or {}) if tried is None else tried)
        if current is ...:
            current = getattr(state, 'candidate', None)
        later = []
        for c in self.candidates:
            rec = tried.get(c.id)
            if rec is not None and c.id != current:
                until = rec.get('retry_at')
                if until is None:
                    continue                      # given up for good (until the user retries)
                if float(until) > now:
                    later.append((float(until), c.index, c))
                    continue
            until = self.cooldown(c) if self.cooldown else None
            if until is not None and until > now:
                if until - now <= self.short_wait:
                    return Pick(c, until)
                later.append((float(until), c.index, c))
                continue
            return Pick(c)
        if later:
            until, _, c = min(later, key=lambda item: (item[0], item[1]))
            return Pick(c, until)
        return Pick(None)

    def owner(self, state, now: Optional[float] = None) -> Optional[str]:
        """Executor that should take the pending unit now (None = exhausted: nobody in particular)."""
        return self.next(state, now).executor

    # -- what to do after a failure ---------------------------------------------------------------
    def decide(self, state, candidate: Candidate, error, now: float, *, kind: Optional[str] = None) -> Decision:
        """Plan action after `candidate` failed on the unit in `state` (pure; apply with apply_decision).

        - NO_SWITCH (content_blocked): FAILED with that kind, no other candidate (same as Policy.run).
        - quota (rate_limit / quota_daily) and Ledger / admission waits (error.not_sent): the wait is
          Retry-After (else RATE_LIMIT_DEFAULT_COOLDOWN, DAILY_QUOTA_DEFAULT_WAIT for quota_daily). Up to
          short_wait: WAIT_QUOTA on the same candidate. Longer, or quota_daily: SWITCH to the next candidate,
          the left one keeps `retry_at` so it can take the work back (reclaim) when it recovers; with no
          other candidate the unit waits on this one (a quota wait is never a failure).
        - retryable (RETRY_KINDS): RETRY on the same candidate until max_attempts runs failed, then SWITCH
          for good (no reclaim: an error is not given back until the user retries).
        - anything else (auth, unsupported, permission ...): SWITCH for good.
        - SWITCH with nothing left: FAILED('exhausted'); every candidate's last reason is in `tried`.
        D1: there is no "busy" outcome. A local GPU that is busy never runs the unit, so it never fails and
        never spills over to the cloud; only real local errors (after their retries) move on.
        """
        kind = error_kind(error, kind)
        detail = str(error)[:DETAIL_MAX] if error is not None else ''
        retry_after = getattr(error, 'retry_after', None)
        old_candidate = getattr(state, 'candidate', None)
        old_wait = getattr(state, 'wait_executor', None)
        base_attempt = int(getattr(state, 'attempt', 0) or 0) if old_candidate in (None, candidate.id) else 0
        tried = {k: dict(v) for k, v in (getattr(state, 'tried', None) or {}).items()}
        record = dict(kind=kind, at=now, detail=detail, executor=candidate.executor)
        # Same-executor reservations stay as they were; once handed to another executor, follow the candidate.
        keep_wait = candidate.executor if old_wait else None

        def same(action, retry_at, attempt):
            tried[candidate.id] = record
            return Decision(action, kind, now, candidate=candidate, retry_at=retry_at, attempt=attempt,
                            wait_executor=keep_wait, tried=tried, detail=detail)

        if kind in NO_SWITCH:
            tried[candidate.id] = dict(record, attempt=base_attempt + 1)
            return Decision(FAILED, kind, now, candidate=candidate, attempt=base_attempt + 1, tried=tried,
                            fail_kind=kind, detail=detail)

        if kind in QUOTA_KINDS or getattr(error, 'not_sent', False):
            default = DAILY_QUOTA_DEFAULT_WAIT if kind == 'quota_daily' else RATE_LIMIT_DEFAULT_COOLDOWN
            wait = float(retry_after) if retry_after is not None else default
            until = now + max(wait, 0.01)
            record['retry_at'] = until
            if wait <= self.short_wait and kind != 'quota_daily':
                streak = short_429_streak(tried.get(candidate.id), error, kind)
                if streak < SHORT_429_SWITCH:
                    if streak:
                        record['short_429'] = streak
                    return same(WAIT_QUOTA, until, base_attempt)
                record['short_429_switch'] = streak     # the streak itself starts again if the work comes back
            switched = self._switch(tried, candidate, record, now, kind, detail, old_wait, reclaim=True)
            if switched is not None:
                return switched
            return same(WAIT_QUOTA, until, base_attempt)

        attempt = base_attempt + 1
        if kind in RETRY_KINDS and attempt < self.max_attempts:
            delay = max(self.retry_delay(candidate, attempt), float(retry_after or 0))
            record['attempt'] = attempt
            return same(RETRY, now + delay, attempt)

        record['attempt'] = attempt
        switched = self._switch(tried, candidate, record, now, kind, detail, old_wait, reclaim=False)
        if switched is not None:
            return switched
        summary = '; '.join(f"{cid}: {rec.get('kind')}" for cid, rec in tried.items())
        return Decision(FAILED, kind, now, candidate=candidate, attempt=attempt, tried=tried,
                        fail_kind='exhausted', detail=summary[:DETAIL_MAX])

    def _switch(self, tried, candidate, record, now, kind, detail, old_wait, *, reclaim) -> Optional[Decision]:
        tried[candidate.id] = record
        pick = self.next(tried=tried, current=None, now=now)
        if pick.candidate is None or pick.candidate.id == candidate.id:
            return None
        target = pick.candidate
        wait_executor = target.executor if (target.executor != candidate.executor or old_wait) else None
        reclaim_at = reclaim_executor = None
        if wait_executor is not None:
            # Quota-left candidates before the target on another executor may take the work back first.
            back = [(float(rec['retry_at']), c) for c in self.candidates[:target.index]
                    if (rec := tried.get(c.id)) and rec.get('retry_at') is not None
                    and float(rec['retry_at']) > now and c.executor != target.executor]
            if back:
                reclaim_at, first = min(back, key=lambda item: item[0])
                reclaim_executor = first.executor
        return Decision(SWITCH, kind, now, candidate=target, retry_at=pick.retry_at, attempt=0,
                        wait_executor=wait_executor, reclaim_at=reclaim_at, reclaim_executor=reclaim_executor,
                        tried=tried, detail=detail)


def next_candidate(plan, role, state=None, now: Optional[float] = None, **kwargs) -> Pick:
    """CandidateChain(plan, role, **kwargs).next(state, now): one-shot form."""
    return CandidateChain(plan, role, **kwargs).next(state, now)


def short_429_streak(previous: Optional[Mapping], error, kind: str) -> int:
    """Provider 429s in a row on one unit and candidate, including `error` (see SHORT_429_SWITCH)."""
    sent_429 = kind == 'rate_limit' and not getattr(error, 'not_sent', False)
    if getattr(error, 'progressed', False):
        return 1 if sent_429 else 0
    if getattr(error, 'not_sent', False):
        return int((previous or {}).get('short_429') or 0)
    return int((previous or {}).get('short_429') or 0) + 1 if sent_429 else 0


def decide_after_failure(plan, role, state, candidate, error, now: float, *, kind: Optional[str] = None,
                         **kwargs) -> Decision:
    """CandidateChain(plan, role, **kwargs).decide(...): one-shot form. `candidate` may be a Candidate or an
    id; an id no longer in the current plan is treated as a removed candidate (recorded, then the chain moves on)."""
    chain = CandidateChain(plan, role, **kwargs)
    if not isinstance(candidate, Candidate):
        candidate = chain.get(candidate) or Candidate(index=len(chain), id=str(candidate), row={},
                                                      executor=CLOUD_EXECUTOR)
    return chain.decide(state, candidate, error, now, kind=kind)


def stage_role(stage: str) -> str:
    """slice_plan stage -> frozen plan role."""
    return 'asr' if stage == 'asr' else 'translation'


def owner_function(plan, *, cooldown=None, now: Optional[float] = None, **kwargs):
    """`SchedulerContext.owner` for select_next_task: the current chain decides who runs each pending unit
    (fresh: the first usable candidate's executor; handed off: the candidate it is on; quota-left: back to
    the original executor once it recovered). `plan` is the current frozen plan (rebuild per pass)."""
    chains = {role: CandidateChain(plan, role, cooldown=cooldown, **kwargs) for role in ('asr', 'translation')}

    def owner(stage, track, state):
        return chains[stage_role(stage)].owner(state, now)
    return owner


def apply_decision(slice_plan, index: int, stage: str, decision: Decision, *, track: str = 'primary',
                   model_id: str = '') -> None:
    """Store a Decision in a slice_plan.SlicePlan unit (the only writer of candidate / tried / reclaim)."""
    from . import slice_plan as sp
    cid = decision.candidate.id if decision.candidate else None
    tried = {k: dict(v) for k, v in decision.tried.items()}
    if decision.action == FAILED:
        state = sp.TaskState(status=sp.FAILED, fail_kind=decision.fail_kind, fail_at=decision.at,
                             attempt=max(decision.attempt, 1), model_id=model_id or cid, detail=decision.detail,
                             candidate=cid, tried=tried)
    else:
        state = sp.TaskState(status=sp.PENDING, retry_at=decision.retry_at, attempt=decision.attempt,
                             wait_executor=decision.wait_executor, candidate=cid, tried=tried,
                             reclaim_at=decision.reclaim_at, reclaim_executor=decision.reclaim_executor)
    slice_plan.slices[index].set(stage, track, state)


def _not_sent(error: ProviderError) -> ProviderError:
    error.not_sent = True       # our own pacing / cooldown: no API request was made
    return error


def failure_cooldown(error: ProviderError, policy: Optional[Mapping] = None) -> Optional[float]:
    """Ledger cooldown run_candidate records for a sent request that failed (None = none):
    rate_limit: Retry-After, else RATE_LIMIT_DEFAULT_COOLDOWN (not max(60 s, ...), see the constant);
    retryable kinds (network / timeout / server / bad_response): none, the unit retries on its own schedule
    and one bad answer says little about the model; everything else (quota_daily, auth, unsupported ...):
    the plan's cooldown_seconds (default 60) or Retry-After if longer, as Policy.run does."""
    if error.kind == 'rate_limit':
        return float(error.retry_after) if error.retry_after is not None else RATE_LIMIT_DEFAULT_COOLDOWN
    if error.kind in RETRY_KINDS:
        return None
    return float((policy or {}).get('cooldown_seconds', 60))


def run_candidate(candidate: Candidate, operation: Callable, credentials: Callable, *, ledger: 'Ledger',
                  plan: Optional[Mapping] = None, tokens: float = 0, audio: float = 0) -> Any:
    """Run `operation(row, key)` once on one candidate. Never sleeps.

    Ledger claim (cooldown), Groq admission (reserve) and the request itself; success clears the
    candidate's cooldown. Raises ProviderError on failure; a wait that sent no request (cooldown claim or
    admission pacing) has `not_sent = True` and `retry_after`, so the caller defers instead of holding a
    request slot. Ledger cooldowns on failure follow failure_cooldown. NO_SWITCH errors and answers flagged
    `no_cooldown` leave the model healthy. Credential errors (consent, missing key) propagate unchanged.
    """
    row = candidate.row
    key = credentials(row)
    account = ledger.account(row, key)
    ck = candidate_key(account, row)
    blocked = ledger.claim(ck)
    if isinstance(blocked, ProviderError):
        raise _not_sent(blocked)
    try:
        if _is_groq(row):
            try:
                ledger.reserve(account, tokens=tokens, audio=audio)
            except ProviderError as admission:
                raise _not_sent(admission)
        try:
            result = operation(row, key)
        except ProviderError as exc:
            if exc.kind in NO_SWITCH or getattr(exc, 'no_cooldown', False):
                ledger.success(ck)
            else:
                cooldown = failure_cooldown(exc, (plan or {}).get('policy'))
                if cooldown is not None:
                    ledger.fail(ck, exc, cooldown)
            raise
        ledger.success(ck)
        return result
    finally:
        ledger.release_probe(ck)
