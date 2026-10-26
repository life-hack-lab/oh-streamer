"""Unified subtitle scheduler, cloud executor.

v2 of the cloud-horizon runner (the old block-based queue mixin and subtitles/cloud_horizon.py, both removed;
the shared audio/cue/translate helpers now live in subtitles/slice_results.py). It depends on no queue
service object: everything the old code reached through ``self`` is behind the small `CloudHost` interface and
the injectable `CloudDeps`, so it runs on a plain item folder and is testable without a network.

- State: ``subtitles/slice-plan.json`` (subtitles.slice_plan), held in memory by one `PlanStore` per folder and
  process, shared with the local GPU executor (subtitle_local) so both see and save the same plan.
- Dispatch (M4, design 4.7): every scheduling pass rebuilds the candidate chain from the *current* frozen plan
  (`runtime_plan(settings)` or the injected `plan_provider`, A2) and gives the selector
  ``ctx.owner = fallback.owner_function(...)``: a unit is taken by the executor of the candidate it is on.
  Exactly one candidate runs at a time (`fallback.run_candidate`: never sleeps; a Ledger / admission wait comes
  back as `not_sent` and the unit is deferred without holding a request slot).
- Failures: `CandidateChain.decide` + `apply_decision` (retry / quota wait / switch candidate / failed
  exhausted). A candidate given up after its retries also gets one Ledger cooldown so other units skip it.
- Results: content-addressed ``subtitles/cache/asr|mt/<key>.json`` keyed by the identity of the candidate that
  produced them; done units store the key (``result_key``) so the other executor / a restart finds them.
- Translation: same batches of 30 numbered lines, general-llm prompt. A garbled numbering (or a request too
  large for the per-minute token limit) halves the batch for that stretch (`_mt_shrink`, one request per batch,
  slice_results._translate_batch), so whatever passed is checkpointed before the next request. Every finished batch is published at once (D2) and checkpointed under a
  candidate-independent key, so a cloud candidate switch continues with the next batch.
- No draft -> refinement stage (D2). No GPU use; runs in its own thread (`run_in_thread`) with
  ``concurrency`` parallel requests (default 1).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from oh_streamer.platform import platform_support
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr
from oh_streamer.subtitles.providers.http import ProviderError

EXECUTOR_NAME = "cloud"
CLOUD_EXECUTOR = sp.Executor(EXECUTOR_NAME)
LOCAL_EXECUTOR_NAME = "local"

MAX_ATTEMPTS = fb.CANDIDATE_MAX_ATTEMPTS   # runs of one candidate on a transient error before moving on
DEFAULT_QUOTA_WAIT_S = fb.RATE_LIMIT_DEFAULT_COOLDOWN   # rate_limit without Retry-After
DEFAULT_DAILY_WAIT_S = fb.DAILY_QUOTA_DEFAULT_WAIT      # quota_daily without Retry-After
MT_BATCH_SIZE = 30
MT_PROMPT_VERSION = "general-llm-v1"
MT_GLOSSARY_VERSION = "none"     # the cloud mt path sends no glossary
QUOTA_KINDS = set(fb.QUOTA_KINDS)
RETRY_KINDS = {"network", "timeout", "server", "bad_response"}
SYNTHETIC = "_synthetic"         # row flag: a one-candidate chain made from the deps (no frozen plan)
DEFAULT_GIVE_UP_COOLDOWN_S = 60.0
PRIMARY = sp.PRIMARY_TRACK
UNSCHEDULED = "unscheduled"      # owner of the units of a comparison track that is not ready (M6a): nobody


class Cancelled(InterruptedError):
    """Raised by CloudHost.check_cancel (or the executor's own stop flag) to stop work."""


# --------------------------------------------------------------------------------------------------
# The dispatch rule (shared by the executors and the queue host's start decision)
# --------------------------------------------------------------------------------------------------

def dispatch_owner(owners: Any, now: float) -> Callable[[str, str, sp.TaskState], Optional[str]]:
    """`SchedulerContext.owner` of one scheduling pass. `owners`: {track: owner function} (fallback.owner_function
    of the track's current frozen plan) or one owner function for every track. A track without one is
    UNSCHEDULED (comparison track not ready / removed: nobody takes it); a unit reserved without candidate
    bookkeeping (pre-M4 plan, manual wait_for) keeps its stored reservation; otherwise the chain decides."""
    def owner(stage: str, track: str, st: sp.TaskState) -> Optional[str]:
        chain_owner = owners if callable(owners) else owners.get(track)
        if chain_owner is None:
            return UNSCHEDULED
        if st.candidate is None and (st.wait_executor or st.reclaim_executor):
            return sp.task_owner(st, now)
        return chain_owner(stage, track, st)
    return owner


def dispatchable_task(plan: sp.SlicePlan, ctx: sp.SchedulerContext, owners: Any, executor: sp.Executor, *,
                      now: Optional[float] = None) -> Optional[sp.Task]:
    """The unit `executor` would take now, or None: select_next_task with the dispatch_owner of `owners`.

    The one "dispatchable now" rule (sample video A): the executor's launch pass, its idle classification
    (_idle_reason: owned work the selector holds back is "horizon", not "idle until now") and the queue host's
    start decision (subtitle_host) all ask this, so a unit beyond the smart horizon / low watermark, beyond the
    downloaded audio or behind the secondary-track lead never starts an executor that cannot take it."""
    now = (ctx.now if ctx.now is not None else time.time()) if now is None else now
    ctx = dataclasses.replace(ctx, now=now, owner=dispatch_owner(owners, now))
    return sp.select_next_task(plan, ctx, executor)


# --------------------------------------------------------------------------------------------------
# Host interface and injectable dependencies
# --------------------------------------------------------------------------------------------------

class CloudHost:
    """What the executor needs from its surroundings. The default is a standalone run on a fully downloaded
    file with nobody watching (everything due). M7 adapts the queue service to this interface."""

    def source_state(self) -> Tuple[bool, Optional[float], Optional[float]]:
        """(usable, limit_s, total_s). Same contract as QueueService._subtitle_cloud_source_state: limit None =
        whole source readable; total = manifest duration while downloading."""
        return True, None, None

    def media_id(self) -> Optional[str]:
        """Stable media id (sr.hls_media_id(sidecar identity)) or None to fingerprint the file."""
        return None

    def cut_limit(self, limit: Optional[float]) -> Optional[float]:
        """How far (seconds) the executor may DECIDE cuts, given the readable limit (None = whole source).
        Default: no extra cap (= `limit`). A host whose source is remote (a stream) returns something near the
        play position so the whole file is not cut up front. Not used for download waits (max_available)."""
        return limit

    def pace_cuts(self) -> bool:
        """Yield between silence windows when a host can receive a new seek during detection."""
        return False

    def seek_pending(self) -> bool:
        """True while an explicit/clock seek is still being debounced; do not claim its long slice."""
        return False

    def seek_position(self) -> Optional[float]:
        """Stable seek target in audio seconds, or None while dragging/unwatched."""
        return None

    def scheduler_context(self) -> sp.SchedulerContext:
        return sp.SchedulerContext(item_state="done")

    def check_cancel(self) -> None:
        """Raise (anything) to stop."""

    def on_progress(self, info: Dict[str, Any]) -> None:
        """UI progress: {"state","slice","stage","detail","done_slices","total_slices", ...}."""

    def on_event(self, event: Dict[str, Any]) -> None:
        """Fallback events: {"type": "subtitle_fallback", "state": retry | wait_quota | switch | failed,
        "handoff": bool, "role", "slice", "stage", "from", "to", "reason", "retry_at", "executor"}."""

    def publish(self, result: Dict[str, Any]) -> None:
        """New SRT published (per translated batch, per slice, final). Typical M7 body: update the item's
        subtitle fields and `_subtitle_publish_update(item, result)`."""


@dataclass
class CloudDeps:
    """Model access. Candidate-aware form (M4b): `asr_run(candidate, piece, slice, run)` and
    `translator_for(candidate, run)` run the given fallback.Candidate; `run(operation, *, tokens, audio)` is the
    executor's one-shot runner (fallback.run_candidate with `ledger` / `credentials`). The legacy `asr(piece,
    slice)` / `translator()` are used when those are missing (one fixed model; tests, old callers)."""
    asr: Optional[Callable[[Path, sp.Slice], Any]]          # audio piece -> segments (or {"segments","sources"})
    translator: Optional[Callable[[], Any]]                 # -> client with .translate(numbered text)
    asr_identity: Dict[str, Any]                            # fallback.identity() of the primary ASR binding
    mt_identity: Dict[str, Any]                             # fallback.identity() of the primary translation binding
    pieces: Optional[Callable[..., List[Tuple[Path, float]]]] = None   # (source, out_path, start, dur)
    probe_duration: Optional[Callable[[Any], Optional[float]]] = None
    detect_silences: Optional[Callable[..., Any]] = None    # extend_lazy_cuts `detect`
    asr_params: Dict[str, Any] = field(default_factory=dict)
    mt_params: Dict[str, Any] = field(default_factory=dict)
    asr_model_id: str = ""
    mt_prompt_version: str = MT_PROMPT_VERSION
    mt_glossary_version: str = MT_GLOSSARY_VERSION
    relaxed_detect: Optional[Callable[..., Any]] = None     # extend_lazy_cuts `relaxed_detect`
    asr_run: Optional[Callable[..., Any]] = None            # (candidate, piece, slice, run) -> segments
    translator_for: Optional[Callable[..., Any]] = None     # (candidate, run) -> client
    ledger: Optional[Any] = None                            # fallback.Ledger (None: no cooldown bookkeeping)
    credentials: Optional[Callable[[Dict[str, Any]], str]] = None   # row -> api key


def asr_cache_params(settings: Optional[dict] = None) -> Dict[str, Any]:
    """ASR parameters that change the output (part of the cache key). Language is fixed to ja; the audio is
    16 kHz mono FLAC (v2: segment times refilled from word times; v3: the doc keeps the whole pre-roll, see
    slice_plan.owns_cue keep_preroll) and split in halves above the upload limit."""
    from oh_streamer.subtitles import slice_results
    return {"pipeline": "slice-asr-v3", "language": "ja", "audio": "flac-16k-mono",
            "split_over_bytes": slice_results.CLOUD_UPLOAD_MAX_BYTES}


def mt_cache_params(config: Optional[dict] = None, settings: Optional[dict] = None) -> Dict[str, Any]:
    """MT parameters that change the output: batch size and the sentence-splitting / short-filter options
    applied to the cues before translation (slice_results._prepare_slice_cues)."""
    cfg = (config or {}).get("sentence") or {}
    return {
        "batch_size": MT_BATCH_SIZE,
        "cue_timing": "words-v4",        # cue times come from word timestamps (word_timing); old rows are stale
        "sentence_mode": bool(cfg.get("sentence_mode", True)),
        "short_threshold": float(cfg.get("short_threshold", 1.0)),
        "short_policy": str(cfg.get("short_policy", "whitelist")),
        "show_interjections": bool(cfg.get("show_interjections",
                                           (settings or {}).get("subtitle_interjections", False))),
    }


def candidate_executor(row: Dict[str, Any]) -> str:
    """Which executor can run a model binding (M4 interface): local models need the GPU worker."""
    return fb.executor_of(row)


def candidates_for(frozen_plan: Dict[str, Any], role: str) -> List[Tuple[str, str]]:
    """[(candidate id, executor name)] in chain order (fallback.candidate_rows: primary, sibling, fallbacks)."""
    return [(c.id, c.executor) for c in fb.candidate_rows(frozen_plan, role)]


def default_deps(folder: Path, settings: dict, config: Optional[dict] = None, *,
                 check_cancel: Optional[Callable[[], None]] = None,
                 on_event: Optional[Callable[[dict], None]] = None) -> CloudDeps:
    """Real providers from a frozen runtime plan (settings['subtitle_resolved']). The v2 executor only
    understands frozen plans; legacy settings have no stable model identity to key the cache with."""
    from oh_streamer.subtitles import model_adapters, runtime, slice_results
    from oh_streamer.subtitles.configuration import runtime_plan

    frozen = runtime_plan(settings)
    if not frozen:
        raise ValueError("cloud executor v2 needs a frozen subtitle runtime plan")
    if not frozen["ready"]:
        raise ValueError("Subtitle configuration not ready")
    asr_row, mt_row = frozen["models"]["asr"], frozen["models"]["translation"]
    cancel = {"check_cancel": check_cancel} if check_cancel else {}

    def asr_run(candidate: fb.Candidate, piece: Path, s: sp.Slice, run):
        tag = hashlib.sha256(candidate.id.encode("utf-8")).hexdigest()[:8]
        out = Path(folder) / "cache" / "tmp" / f"asr_{s.index:04d}_{tag}_{piece.stem}"
        result = model_adapters.run_asr_candidate(dict(candidate.row), str(piece), str(out), run,
                                                  settings=settings, **cancel)
        shutil.rmtree(out, ignore_errors=True)     # kept on failure: the chunk checkpoint resumes a retry
        return result

    def translator_for(candidate: fb.Candidate, run):
        return model_adapters.make_candidate_translation_client(dict(candidate.row), run, settings=settings,
                                                                prompt_family="general-llm")

    def probe(source):
        return runtime._probe_audio_duration(str(source), runtime.resolve_ffmpeg(), 15)

    return CloudDeps(asr=None, translator=None, asr_run=asr_run, translator_for=translator_for,
                     asr_identity=fb.identity(asr_row, settings), mt_identity=fb.identity(mt_row, settings),
                     pieces=slice_results._audio_pieces, probe_duration=probe,
                     asr_params=asr_cache_params(settings), mt_params=mt_cache_params(config, settings),
                     asr_model_id=asr_row["id"], ledger=fb.Ledger(),
                     credentials=lambda row: model_adapters._binding_credentials(row, settings))


# --------------------------------------------------------------------------------------------------
# Shared plan holder
# --------------------------------------------------------------------------------------------------

class PlanStore:
    """One in-memory slice plan per item folder and process, shared by every executor working on it.

    Holds the plan, one lock for every plan read/transition/save, the finished / partial translations that
    publication combines, and each executor's cache-key function (`profiles`, used to recognise results of
    another executor's candidates). Executors `acquire` it in _prepare and `release` it when run() ends; the
    last release drops it, so the next run loads slice-plan.json from disk again (= what a restart sees)."""

    _registry: Dict[str, "PlanStore"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.lock = threading.RLock()
        self.publish_lock = threading.Lock()
        self.plan: Optional[sp.SlicePlan] = None
        self.mt_rows: Dict[int, List[dict]] = {}      # primary: finished slices' translated segments (slice-relative)
        self.partial: Dict[int, List[dict]] = {}      # primary: slices mid-translation, batches already done
        # M6a: the same per track; track_rows[PRIMARY] IS mt_rows (track_partial[PRIMARY] IS partial)
        self.track_rows: Dict[str, Dict[int, List[dict]]] = {PRIMARY: self.mt_rows}
        self.track_partial: Dict[str, Dict[int, List[dict]]] = {PRIMARY: self.partial}
        self.track_errors: Dict[str, List[str]] = {}  # comparison tracks that are not ready: zh-TW reasons
        self.track_labels: Dict[str, str] = {}        # track id -> row label (comparison tracks)
        self.asr_digest: Dict[str, str] = {}          # ASR result key -> asr_result_digest
        self.profiles: Dict[str, Callable[..., Optional[str]]] = {}
        self.last_publication: Optional[Dict[str, Any]] = None
        self.users = 0

    @staticmethod
    def _key(folder: Path | str) -> str:
        return os.path.normcase(str(Path(folder).resolve()))

    @classmethod
    def acquire(cls, folder: Path | str) -> "PlanStore":
        with cls._registry_lock:
            key = cls._key(folder)
            store = cls._registry.get(key)
            if store is None:
                store = cls._registry[key] = PlanStore(Path(folder))
            store.users += 1
            return store

    def release(self) -> None:
        with PlanStore._registry_lock:
            self.users = max(0, self.users - 1)
            key = self._key(self.folder)
            if self.users == 0 and PlanStore._registry.get(key) is self:
                PlanStore._registry.pop(key, None)

    @classmethod
    def peek(cls, folder: Path | str) -> Optional["PlanStore"]:
        with cls._registry_lock:
            return cls._registry.get(cls._key(folder))

    def save(self) -> None:
        with self.lock:
            if self.plan is not None:
                sp.save_slice_plan(self.plan, self.folder)

    def rows(self, track: str) -> Dict[int, List[dict]]:
        """Finished translated rows of a track by slice index (slice-relative times)."""
        return self.track_rows.setdefault(track, {})

    def partials(self, track: str) -> Dict[int, List[dict]]:
        """Rows of a track's slices mid-translation (batches already done) by slice index."""
        return self.track_partial.setdefault(track, {})

    def clear_results(self) -> None:
        for rows in self.track_rows.values():
            rows.clear()
        for rows in self.track_partial.values():
            rows.clear()


class FolderState:
    """What outlives one executor instance AND the PlanStore (which is dropped when the last executor ends) for an
    item folder, in this process: the publication signature + result per track (files are not rewritten while the
    combined result is unchanged), the (track, slice) pairs already written to the usage log, the handed-off units
    the local side is waiting to start (one usage line per wait), and how each executor's last run ended.
    In memory only: a new process writes the files once and logs the slices once."""

    _registry: Dict[str, "FolderState"] = {}
    _registry_lock = threading.Lock()

    def __init__(self) -> None:
        self.published: Dict[str, Tuple[str, Dict[str, Any]]] = {}     # track -> (signature, result)
        self.logged_slices: set = set()                               # (track, slice index, MT result key)
        self.waiting: Dict[Tuple[int, str, str], float] = {}          # (slice, stage, track) -> wait start
        self.last_end: Dict[str, Tuple[str, float]] = {}              # executor name -> (reason, ended at)

    @classmethod
    def of(cls, folder: Path | str) -> "FolderState":
        key = PlanStore._key(folder)
        with cls._registry_lock:
            state = cls._registry.get(key)
            if state is None:
                state = cls._registry[key] = FolderState()
            return state


def _publication_signature(plan: sp.SlicePlan, done: Dict[int, List[dict]], partial: Dict[int, List[dict]],
                           extra: Any = None) -> str:
    """Everything the published files are a function of: the plan's shape, the finished and partial rows."""
    payload = {"media": plan.media_id, "open": plan.open, "duration": plan.duration,
               "tracks": list(plan.tracks),      # the manifest follows the track order and drops removed rows
               "slices": [(s.index, s.start, s.end) for s in plan.slices],
               "done": sorted(done.items()), "partial": sorted(partial.items()), "extra": extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------------

@dataclass
class RunResult:
    reason: str                 # complete | idle | horizon | quota | download | stuck | source_unavailable | cancelled
    plan: Optional[sp.SlicePlan] = None
    status: Optional[sp.PlanStatus] = None
    publication: Optional[Dict[str, Any]] = None
    wait_until: Optional[float] = None      # quota: earliest retry; idle: earliest reclaim of handed-off work
    waiting_executor: Optional[str] = None  # idle because the remaining work belongs to this other executor


class CloudSliceExecutor:
    """Runs asr / mt work of a slice plan with cloud providers. One instance per item and run.

    ASR audio starts `asr_window.pre_roll_s` before the slice; the slice keeps only the cues it owns
    (slice_plan.owns_cue), and the window parameters are part of the ASR cache key (design 7.2)."""

    EXECUTOR_NAME = EXECUTOR_NAME
    EXECUTOR = CLOUD_EXECUTOR
    PLAN_SLICE_SECONDS = sp.CLOUD_SLICE_SECONDS     # slice length of a NEW plan; LocalSliceExecutor overrides

    def __init__(self, folder: Path | str, source: Path | str, *, host: Optional[CloudHost] = None,
                 deps: Optional[CloudDeps] = None, settings: Optional[dict] = None, config: Optional[dict] = None,
                 concurrency: int = 1, japanese: bool = False, strategy: Optional[str] = None,
                 horizon_minutes: Optional[int] = None, first_slice_seconds: Optional[float] = None,
                 slice_seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 poll_s: float = 0.5, download_poll_s: float = 2.0,
                 asr_window: Optional[sp.AsrWindow] = None,
                 plan_provider: Optional[Callable[[], Optional[dict]]] = None,
                 track_plans_provider: Optional[Callable[[], Dict[str, Optional[dict]]]] = None,
                 ledger: Optional[Any] = None, credentials: Optional[Callable[[dict], str]] = None):
        self.folder = Path(folder)
        self.asr_window = asr_window or sp.AsrWindow()
        self.source = Path(source)
        self.host = host or CloudHost()
        self.settings = settings or {}
        self.config = config or {}
        self.deps = deps or default_deps(self.folder, self.settings, self.config,
                                         check_cancel=self._check, on_event=self.host.on_event)
        self.concurrency = max(1, int(concurrency))
        self.japanese = japanese
        self.strategy = strategy
        self.horizon_minutes = horizon_minutes
        self.first_slice_seconds = first_slice_seconds
        self.slice_seconds = float(slice_seconds) if slice_seconds else self.PLAN_SLICE_SECONDS
        self.clock, self.sleep = clock, sleep
        self.poll_s, self.download_poll_s = poll_s, download_poll_s
        self.plan_provider = plan_provider     # -> current frozen plan (default runtime_plan(settings)), A2
        # M6a: -> {track id: frozen plan} of the comparison tracks (a 'primary' entry is ignored: the primary
        # always comes from plan_provider / runtime_plan). Default configuration.track_runtime_plans(settings).
        self.track_plans_provider = track_plans_provider
        self._track_memo: Optional[Tuple[str, Dict[str, Optional[dict]]]] = None
        self.ledger = ledger if ledger is not None else getattr(self.deps, "ledger", None)
        self.credentials = credentials or getattr(self.deps, "credentials", None) or (lambda row: "")
        self.plan: Optional[sp.SlicePlan] = None
        self.last_publication: Optional[Dict[str, Any]] = None
        self._store: Optional[PlanStore] = None
        self._lock = threading.RLock()                 # replaced by the PlanStore's in _prepare
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._refresh = threading.Event()              # soft drain: launch nothing new, finish in-flight, exit "refresh"
        self._mt_rows: Dict[int, List[dict]] = {}
        self._partial: Dict[int, List[dict]] = {}
        self._track_rows: Dict[str, Dict[int, List[dict]]] = {PRIMARY: self._mt_rows}
        self._track_partial: Dict[str, Dict[int, List[dict]]] = {PRIMARY: self._partial}
        self._asr_digest: Dict[str, str] = {}
        # (track, slice) -> (bad_end, size): a batch ending at bad_end came back garbled; translate up to there
        # in batches of `size` (halved on each garble) so the passed part is checkpointed, then back to full size
        self._mt_shrink: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._identity_memo: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._media_id = ""
        self._thread: Optional[threading.Thread] = None
        self._usage_started = False
        self._last_result: Optional[RunResult] = None
        self._run_t0 = 0.0
        self._run_stats = {"launched": 0, "done": 0, "failed": 0, "requests": 0}
        self._stats_lock = threading.Lock()
        self._handed: Dict[Tuple[int, str, str], float] = {}    # handed-off units this executor started -> t0

    # -- control --------------------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def request_refresh(self) -> None:
        """Soft stop (the item's settings changed): launch no new work unit, let the in-flight ones finish and
        record normally, then end the run with reason "refresh"; the host starts a new executor right away, which
        reads the newly locked settings."""
        self._refresh.set()

    def run_in_thread(self, **kwargs: Any) -> threading.Thread:
        """Own thread, no GPU. Result of run() in `self.result`."""
        def target():
            self.result = self.run(**kwargs)
        self.result: Optional[RunResult] = None
        self._thread = threading.Thread(target=target, name="subtitle-scheduler-cloud", daemon=True)
        self._thread.start()
        return self._thread

    def _check(self) -> None:
        if self._stop.is_set():
            raise Cancelled()
        try:
            self.host.check_cancel()
        except Cancelled:
            raise
        except Exception as exc:
            raise Cancelled() from exc

    # -- candidates -----------------------------------------------------------------------------
    def _frozen(self, track: Optional[str] = None) -> Dict[str, Any]:
        """The current frozen plan of a track (re-read on every decision, A2). Primary without one: a
        one-candidate chain made from the deps (legacy / tests). A comparison track that is missing or not
        ready: an empty plan (no candidates; its units are owned by nobody, see _dispatch)."""
        if track is not None and track != PRIMARY:
            plan = self._track_plans().get(track)
            return plan if plan and plan.get("ready", True) else {"models": {}, "fallback": {}, "policy": {}}
        plan = None
        if self.plan_provider is not None:
            plan = self.plan_provider()
        else:
            from oh_streamer.subtitles.configuration import runtime_plan
            plan = runtime_plan(self.settings)
        if plan and plan.get("models"):
            return plan
        return self._synthetic_plan()

    def _track_plans(self) -> Dict[str, Optional[dict]]:
        """{comparison track id: frozen plan} in row order (M6a); never contains the primary."""
        if self.track_plans_provider is not None:
            plans = dict(self.track_plans_provider() or {})
        else:
            from oh_streamer.subtitles import configuration
            rows = configuration.comparison_rows(self.settings)
            if not rows:
                return {}
            memo_key = json.dumps(rows, sort_keys=True, default=str)
            if self._track_memo is None or self._track_memo[0] != memo_key:
                self._track_memo = (memo_key, configuration.track_runtime_plans(self.settings))
            plans = dict(self._track_memo[1])
        plans.pop(PRIMARY, None)
        return plans

    def _ready_tracks(self) -> List[str]:
        """Tracks that get work: the primary and every ready comparison track of the slice plan."""
        plans = self._track_plans()
        tracks = list(self.plan.tracks) if self.plan is not None else [PRIMARY]
        return [t for t in tracks if t == tracks[0] or (plans.get(t) or {}).get("ready", False)]

    def _sync_tracks(self) -> bool:
        """Under the plan lock: slice plan tracks = [primary] + current comparison rows. A removed track's
        units are dropped (its cached results stay on disk). Not-ready reasons / labels go to the PlanStore."""
        if self.plan is None:
            return False
        plans = self._track_plans()
        wanted = [self.plan.primary] + [t for t in plans if t != self.plan.primary]
        if self._store is not None:
            self._store.track_errors = {t: list((p or {}).get("reasons") or ["比較字幕設定無效"])
                                        for t, p in plans.items() if not (p or {}).get("ready", False)}
            self._store.track_labels = {t: str((p or {}).get("label") or t) for t, p in plans.items()}
        if wanted == list(self.plan.tracks):
            return False
        removed = set(self.plan.tracks) - set(wanted)
        self.plan.tracks = wanted
        for s in self.plan.slices:
            for key in [k for k in s.tasks if k.partition("/")[2] in removed]:
                del s.tasks[key]
        for track in removed:
            self._track_rows.pop(track, None)
            self._track_partial.pop(track, None)
        return True

    def _rows(self, track: str) -> Dict[int, List[dict]]:
        return self._track_rows.setdefault(track, {})

    def _partials(self, track: str) -> Dict[int, List[dict]]:
        return self._track_partial.setdefault(track, {})

    def _track_config(self, track: str) -> dict:
        """Subtitle config (combo) a track's translation uses: the row's own for a comparison track."""
        if track == PRIMARY:
            return self.config
        return (self._track_plans().get(track) or {}).get("combo") or self.config

    def _synthetic_plan(self) -> Dict[str, Any]:
        provider = "local" if self.EXECUTOR_NAME == LOCAL_EXECUTOR_NAME else "openai-compatible"
        name = self.EXECUTOR_NAME
        asr_id = self.deps.asr_model_id or self.deps.asr_identity.get("id") or f"{name}-asr"
        mt_id = getattr(self.deps, "mt_model_id", "") or self.deps.mt_identity.get("id") or f"{name}-mt"
        return {"models": {"asr": {"id": asr_id, "kind": "asr", "provider": provider, SYNTHETIC: True},
                           "translation": {"id": mt_id, "kind": "translation", "provider": provider,
                                           SYNTHETIC: True}},
                "fallback": {}, "policy": {}}

    def _chain_kwargs(self) -> Dict[str, Any]:
        return {}

    def _chain(self, frozen: Dict[str, Any], stage: str, cooldown=None) -> fb.CandidateChain:
        return fb.CandidateChain(frozen, fb.stage_role(stage), cooldown=cooldown, **self._chain_kwargs())

    def _cooldown(self) -> Optional[Callable[[fb.Candidate], Optional[float]]]:
        """Ledger cooldown lookup for one scheduling pass (memoised inside)."""
        if self.ledger is None:
            return None
        return fb.ledger_cooldown(self.ledger, self.credentials)

    def _dispatch(self, limit: Optional[float]):
        """(ctx, plans, cooldown) of one scheduling pass (plans = {track: frozen plan} of the tracks that get
        work, M6a): ctx.owner from the current chain of the unit's track (with the same
        Ledger cooldown lookup _claim uses, so selection and claim agree). A unit reserved without candidate
        bookkeeping (pre-M4 plan, manual wait_for) keeps its stored reservation."""
        now = self.clock()
        if self._sync_tracks():
            self._save()
        cooldown = self._cooldown()
        plans = {track: self._frozen(track) for track in self._ready_tracks()}
        owners = {track: fb.owner_function(frozen, cooldown=cooldown, now=now, **self._chain_kwargs())
                  for track, frozen in plans.items()}
        ctx = dataclasses.replace(self.host.scheduler_context(), max_available=limit, now=now,
                                  owner=dispatch_owner(owners, now))
        return ctx, plans, cooldown

    def _claim(self, task: sp.Task, plans: Dict[str, Any], cooldown, now: float) -> Tuple[Optional[fb.Candidate], bool]:
        """Under the plan lock: pick the candidate for a selected unit (from its track's plan; `plans` from
        _dispatch) and mark it running on it. (candidate, changed): candidate None = not now; changed = the
        unit's state was updated (deferred for a short cooldown, or failed because nothing is left), False =
        not ours after all."""
        if self.host.seek_pending():
            return None, False
        frozen = plans.get(task.track)
        if frozen is None:
            return None, False
        chain = self._chain(frozen, task.stage, cooldown)
        st = self.plan.slices[task.slice_index].get(task.stage, task.track)
        pick = chain.next(st, now)
        if pick.exhausted:
            self.plan.mark_failed(task.slice_index, task.stage, "exhausted", at=now, attempt=max(st.attempt, 1),
                                  model_id=st.candidate or "", detail="no usable candidate", track=task.track)
            return None, True
        if pick.executor != self.EXECUTOR_NAME:
            return None, False
        if pick.retry_at is not None and pick.retry_at > now:
            self.plan.defer(task.slice_index, task.stage, pick.retry_at, task.track)
            return None, True
        waiting, attempt = st.wait_executor, st.attempt
        self.plan.mark_running(task.slice_index, task.stage, self.EXECUTOR_NAME, task.track,
                               candidate=pick.candidate.id)
        self._usage_unit_started(task, waiting, attempt, pick.candidate.id)
        return pick.candidate, True

    def _deps_for(self, candidate: Optional[fb.Candidate]):
        return self.deps

    def _row_identity(self, stage: str, candidate: fb.Candidate) -> Dict[str, Any]:
        """fallback.identity of the candidate's binding (deps identity for the synthetic one)."""
        if candidate.row.get(SYNTHETIC):
            deps = self._deps_for(candidate)
            return dict(deps.asr_identity if stage == sp.STAGE_ASR else deps.mt_identity)
        memo_key = (stage, json.dumps(dict(candidate.row), sort_keys=True, default=str))
        if memo_key not in self._identity_memo:
            self._identity_memo[memo_key] = fb.identity(candidate.row, self.settings)
        return self._identity_memo[memo_key]

    def _identity(self, stage: str, candidate: fb.Candidate) -> Dict[str, Any]:
        """Model identity in this executor's cache keys: the candidate's own binding (M4b; a Groq sibling
        therefore gets its own key, see docs §6.1 M2 note)."""
        return self._row_identity(stage, candidate)

    def _runner(self, candidate: fb.Candidate, frozen: Dict[str, Any]):
        """run(operation, *, tokens, audio): one request on one candidate (fallback.run_candidate)."""
        def run(operation, *, tokens=0, audio=0):
            self._check()
            row = dict(candidate.row)
            if self.ledger is None or row.get(SYNTHETIC):
                return operation(row, "" if row.get(SYNTHETIC) else self.credentials(row))
            return fb.run_candidate(candidate, operation, self.credentials, ledger=self.ledger, plan=frozen,
                                    tokens=tokens, audio=audio)
        return run

    # -- keys -----------------------------------------------------------------------------------
    def _asr_params(self, candidate: fb.Candidate) -> Dict[str, Any]:
        decode = {"hls_window_decode": "prefix-atrim-v1"} if self.source.suffix.lower() in (".m3u8", ".m3u") else {}
        return {**self._deps_for(candidate).asr_params, **self.asr_window.cache_params(), **decode}

    def _home_candidate(self, stage: str, track: Optional[str] = None) -> Optional[fb.Candidate]:
        """First candidate of this executor in the track's current chain (else the chain head)."""
        chain = self._chain(self._frozen(track), stage)
        return next((c for c in chain if c.executor == self.EXECUTOR_NAME), None) or next(iter(chain), None)

    def _asr_key(self, s: sp.Slice, candidate: Optional[fb.Candidate] = None, track: Optional[str] = None) -> str:
        """ASR cache key of `candidate`'s result (default: this executor's first candidate of the track). The
        key has no track in it: tracks with the same ASR model share one result (M6a)."""
        candidate = candidate or self._home_candidate(sp.STAGE_ASR, track)
        return sp.asr_cache_key(self._media_id, s.start, s.end, self._identity(sp.STAGE_ASR, candidate),
                                self._asr_params(candidate))

    def _digest_of(self, asr_key: Optional[str], segments: Sequence[Any]) -> str:
        if asr_key and asr_key in self._asr_digest:
            return self._asr_digest[asr_key]
        digest = sp.asr_result_digest(segments)
        if asr_key:
            self._asr_digest[asr_key] = digest
        return digest

    def _mt_params(self, candidate: fb.Candidate, track: str = PRIMARY) -> Dict[str, Any]:
        """MT parameters in the cache key. A comparison track prepares its cues with its own sentence options
        (its combo), so they are part of its key; the primary keeps the deps' params (keys unchanged)."""
        params = self._deps_for(candidate).mt_params
        if track == PRIMARY:
            return params
        return {**params, **mt_cache_params(self._track_config(track), self.settings)}

    def _mt_key(self, digest: str, candidate: fb.Candidate, track: str = PRIMARY) -> str:
        deps = self._deps_for(candidate)
        return sp.mt_cache_key(digest, self._identity(sp.STAGE_MT, candidate), deps.mt_prompt_version,
                               deps.mt_glossary_version, self._mt_params(candidate, track))

    def _partial_key(self, digest: str, track: str = PRIMARY) -> str:
        """Cloud translation checkpoint: candidate independent (same cue preparation and batches for every
        cloud candidate), so switching candidates continues after the batches already translated. Per track
        (two comparison rows on the same ASR result translate with different models)."""
        ident: Dict[str, Any] = {"partial": "cloud-numbered-batches"}
        params = self.deps.mt_params
        if track != PRIMARY:
            ident["track"] = track
            params = {**params, **mt_cache_params(self._track_config(track), self.settings)}
        return sp.mt_cache_key(digest, ident, self.deps.mt_prompt_version, self.deps.mt_glossary_version, params)

    def _key_for(self, stage: str, s: sp.Slice, candidate: fb.Candidate, digest: Optional[str] = None,
                 track: str = PRIMARY) -> Optional[str]:
        """This executor's cache key of `candidate`'s result for a slice (PlanStore profile); None = unknown."""
        if candidate.executor != self.EXECUTOR_NAME:
            return None
        if stage == sp.STAGE_ASR:
            return self._asr_key(s, candidate)
        return self._mt_key(digest, candidate, track) if digest else None

    # -- lazy cuts ------------------------------------------------------------------------------
    _cuts_paced = False        # the last _advance_cuts decided at most one cut (host.cut_limit was in effect)

    def _advance_cuts(self, limit: Optional[float]) -> bool:
        """Create a stable seek's short slice before growing the readable prefix.
        Seek detection runs outside the plan lock; apply rechecks the target and task states.
        Finish a watched short slice before further silence IO. Streams otherwise grow one
        nearby cut per dispatch round; ordinary sources grow their whole readable prefix.
        The caller keeps the original readable limit for task/download eligibility.
        """
        if self.host.seek_pending():
            return False
        cap = self.host.cut_limit(limit)
        stream_paced = cap != limit
        paced = stream_paced or self.host.pace_cuts()
        if stream_paced and cap is not None and limit is not None:
            cap = min(cap, limit)
        self._cuts_paced = paced
        position = self.host.seek_position()
        request = sp.seek_cut_request(self.plan, position, limit=limit) if position is not None else None
        changed = False
        if request is not None:
            detect = self.deps.detect_silences or sp.cloud_chunks.detect_window_silences
            silences = list(detect(self.source, *request.window))
            relaxed = None
            relaxed_detect = self.deps.relaxed_detect
            if self.deps.detect_silences is None:
                relaxed_detect = relaxed_detect or sp.detect_window_silences_relaxed
            if relaxed_detect is not None and sp.seek_needs_relaxed(self.plan, request, silences):
                relaxed = list(relaxed_detect(self.source, *request.window))
            self._check()
            with self._lock:
                # Position may have changed while remote audio was being fetched.
                current = self.host.seek_position()
                if current is not None and abs(current - position) <= 10:
                    changed = sp.apply_seek_cuts(self.plan, request, silences, relaxed)
                    if changed:
                        self._save()
                        for event in sp.pop_seek_events(self.plan):
                            try:
                                from oh_streamer.subtitles import usage_log
                                usage_log.event(self._usage_id(), "slice_split", **event)
                            except Exception:
                                pass
            if changed:
                # Dispatch the new short slice before another (remote) silence search.
                self._cuts_paced = True
                return True
        with self._lock:
            if self.host.seek_pending() or not self.plan.open:
                return changed
            current = self.host.seek_position()
            urgent = sp.slice_containing(self.plan, current) if current is not None else None
            if urgent is not None and urgent.end - urgent.start <= 75 and not urgent.all_done(self.plan.primary):
                # Finish the watched short slice (including MT) before more silence IO. In particular,
                # a complete local HLS source can take minutes to decide all remaining queue cuts.
                return changed
            ctx = self.host.scheduler_context()
            near = sp.stream_grow_window(ctx.play_pos, self.plan.horizon_minutes * 60) if stream_paced else None
            if sp.extend_lazy_cuts(self.plan, self.source, cap, detect=self.deps.detect_silences,
                                   relaxed_detect=self.deps.relaxed_detect, max_cuts=1 if paced else None,
                                   near=near, focus=ctx.play_pos if stream_paced else None):
                self._save()
                changed = True
            return changed

    def _extend_one_cut(self, limit: Optional[float], slice_seconds: Optional[float] = None) -> bool:
        """One more lazy cut near the play position, the way a stream grows (nearby chains first), whatever the
        pacing. True = the plan changed (and was saved). Media turns call it while preparing, to cut ahead of the
        play position far enough for the turn's target seconds (T4.2/T4.3 second round, fix 1). `slice_seconds` =
        the size of the slice cut (default: the plan's normal 300 s; media turns pass the short stream size, so
        cutting ahead of a short slice at the play position never yields a 300 s slice, fix I)."""
        cap = self.host.cut_limit(limit)
        stream_paced = cap != limit
        if stream_paced and cap is not None and limit is not None:
            cap = min(cap, limit)
        with self._lock:
            if self.host.seek_pending() or not self.plan.open:
                return False
            ctx = self.host.scheduler_context()
            near = sp.stream_grow_window(ctx.play_pos, self.plan.horizon_minutes * 60) if stream_paced else None
            if sp.extend_lazy_cuts(self.plan, self.source, cap, detect=self.deps.detect_silences,
                                   relaxed_detect=self.deps.relaxed_detect, max_cuts=1, near=near,
                                   focus=ctx.play_pos if stream_paced else None, slice_seconds=slice_seconds):
                self._save()
                return True
            return False

    # -- main loop ------------------------------------------------------------------------------
    def run(self, *, wait: bool = False) -> RunResult:
        """Work until the plan is complete. wait=False returns as soon as nothing can run now (reason says why:
        quota / download / stuck / idle / horizon); wait=True sleeps through quota and download waits, and through
        the wait for handed-off work to come back (standalone use). "horizon" (owned work the selector holds back)
        always returns: only a new play position / limit / task state makes it dispatchable again."""
        self.folder.mkdir(parents=True, exist_ok=True)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency,
                                                     thread_name_prefix="subtitle-cloud-req")
        inflight: Dict[concurrent.futures.Future, sp.Task] = {}
        try:
            self._check()
            try:
                if not self._prepare():
                    return self._result("source_unavailable")
            except Exception as exc:  # noqa: BLE001
                if self._source_gone(exc):
                    return self._result("source_unavailable")
                raise
            self._usage_run_start()
            self._publish()
            self._publish_tracks()
            spins = 0
            while True:
                self._check()
                if self.host.seek_pending():
                    self.sleep(min(self.poll_s, 0.1))
                    continue
                usable, limit, _total = self.host.source_state()
                if not usable:
                    self._drain(inflight)
                    return self._result("source_unavailable", limit=limit)
                try:
                    advanced = self._advance_cuts(limit)
                except Exception as exc:  # noqa: BLE001
                    if not self._source_gone(exc):
                        raise
                    self._drain(inflight)
                    return self._result("source_unavailable", limit=limit)
                self._launch(pool, inflight, limit)
                if self._refresh.is_set() and not inflight:
                    self._publish()
                    self._publish_tracks()
                    return self._result("refresh", limit=limit)
                if not inflight and advanced and self._cuts_paced:
                    continue          # one cut per round: dispatch found nothing, decide the next cut right away
                if inflight:
                    spins = 0
                    done, _ = concurrent.futures.wait(list(inflight), timeout=self.poll_s,
                                                      return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        inflight.pop(fut)
                        fut.result()          # Cancelled propagates; other errors were handled in _work
                    continue
                reason, until, other = self._idle_reason(limit)
                if reason == "complete":
                    self._publish()
                    self._publish_tracks()
                    return self._result("complete", limit=limit)
                spins = spins + 1 if (reason == "idle" and other is None and until is not None) else 0
                if spins > 5:
                    return self._result(reason, limit=limit, wait_until=until, waiting_executor=other)
                if wait and (reason in ("quota", "download") or (reason == "idle" and until is not None)):
                    self._progress(state="waiting_" + ("handoff" if reason == "idle" else reason), detail=reason)
                    pause = self.download_poll_s
                    if until is not None:
                        pause = max(0.01, min(until - self.clock(), self.download_poll_s))
                    self.sleep(pause)
                    continue
                return self._result(reason, limit=limit, wait_until=until, waiting_executor=other)
        except Cancelled:
            self._stop.set()
            self._drain(inflight)
            return self._result("cancelled")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self._release_store()
            self._usage_run_end()

    # -- usage log (diagnostics; every method swallows its own failures) ----------------------------
    def _usage_id(self) -> str:
        return self.folder.parent.name

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self._run_stats[key] = self._run_stats.get(key, 0) + 1

    def _usage_run_start(self) -> None:
        try:
            from oh_streamer.subtitles import usage_log
            self._usage_started = True
            now = self.clock()
            self._run_t0 = now
            with self._stats_lock:
                self._run_stats = {"launched": 0, "done": 0, "failed": 0, "requests": 0}
            prev = FolderState.of(self.folder).last_end.get(self.EXECUTOR_NAME)
            usage_log.run_start(self._usage_id(), executor=self.EXECUTOR_NAME,
                                duration_s=getattr(self.plan, "duration", None), profile=self._frozen(),
                                reason="first_run" if prev is None else f"restart_after_{prev[0]}",
                                gap_s=None if prev is None else max(0.0, now - prev[1]))
        except Exception:
            pass

    def _usage_run_end(self) -> None:
        """One run_end per run_start (and for a run that ended before it started): how it ended, what it waits
        for, and what this run did. `_result` records the reason; no result = an exception left run()."""
        try:
            from oh_streamer.subtitles import usage_log
            res, self._last_result = self._last_result, None
            exc = sys.exc_info()[1]
            reason = res.reason if res is not None else ("cancelled" if isinstance(exc, Cancelled) else "error")
            now = self.clock()
            FolderState.of(self.folder).last_end[self.EXECUTOR_NAME] = (reason, now)
            with self._stats_lock:
                stats = dict(self._run_stats)
            usage_log.run_end(
                self._usage_id(), executor=self.EXECUTOR_NAME, reason=reason,
                wait_until=getattr(res, "wait_until", None),
                wait_s=None if res is None or res.wait_until is None else max(0.0, res.wait_until - now),
                waiting_executor=getattr(res, "waiting_executor", None),
                run_s=(now - self._run_t0) if self._usage_started else None,
                units_launched=stats["launched"], units_done=stats["done"], units_failed=stats["failed"],
                requests=stats["requests"] if self.EXECUTOR_NAME == EXECUTOR_NAME else None,
                started=None if self._usage_started else False,
                error=(f"{type(exc).__name__}: {exc}" if exc is not None and res is None else None))
        except Exception:
            pass
        finally:
            self._usage_started = False

    def _usage_slices(self, track: str, done: Dict[int, List[dict]]) -> None:
        """Usage log: one line per slice newly finished on `track` (once per folder and process; called under
        the publish lock)."""
        try:
            from oh_streamer.subtitles import usage_log
            seen = FolderState.of(self.folder).logged_slices
            for i, rows in done.items():
                sl = self.plan.slices[i]
                key = (track, i, getattr(sl.get(sp.STAGE_MT, track), "result_key", None))   # a redone slice logs again
                if key in seen:
                    continue
                seen.add(key)
                usage_log.slice_published(self._usage_id(), track=track, slice_index=i, start=sl.start,
                                          end=sl.end, cues=len(rows),
                                          candidate=getattr(sl.get(sp.STAGE_MT, track), "candidate", "") or "")
        except Exception:
            pass

    def _usage_unit_started(self, task: sp.Task, waiting: Optional[str], attempt: int, candidate_id: str) -> None:
        """A unit handed to this executor by another one (wait_executor == us) starts: `<executor>_started`."""
        try:
            if waiting != self.EXECUTOR_NAME:
                return
            from oh_streamer.subtitles import usage_log
            key = (task.slice_index, task.stage, task.track)
            now = self.clock()
            self._handed[key] = now
            waited = FolderState.of(self.folder).waiting.pop(key, None)
            usage_log.event(self._usage_id(), f"{self.EXECUTOR_NAME}_started", slice=task.slice_index,
                            stage=task.stage, track=task.track, candidate=candidate_id, attempt=attempt or None,
                            waited_s=None if waited is None else max(0.0, now - waited))
        except Exception:
            pass

    def _usage_unit_done(self, task: sp.Task) -> None:
        self._bump("done")
        self._usage_unit_end(task, "done")

    def _usage_unit_end(self, task: sp.Task, outcome: str, **fields: Any) -> None:
        """`<executor>_done` / `_failed` of a unit that was handed to this executor (elapsed since its start)."""
        try:
            t0 = self._handed.pop((task.slice_index, task.stage, task.track), None)
            if t0 is None:
                return
            from oh_streamer.subtitles import usage_log
            usage_log.event(self._usage_id(), f"{self.EXECUTOR_NAME}_{outcome}", slice=task.slice_index,
                            stage=task.stage, track=task.track, elapsed_s=max(0.0, self.clock() - t0), **fields)
        except Exception:
            pass

    def _usage_candidate_changed(self, task: sp.Task, current_id: str, decision: fb.Decision) -> None:
        try:
            from oh_streamer.subtitles import usage_log
            usage_log.event(self._usage_id(), "candidate_changed", slice=task.slice_index, stage=task.stage,
                            track=task.track, action=decision.action, **{"from": current_id},
                            to=decision.candidate.id if decision.action == fb.SWITCH and decision.candidate else None,
                            reason=decision.kind,
                            detail=decision.detail, handoff=bool(decision.handoff),
                            wait_executor=decision.wait_executor, fail_kind=decision.fail_kind,
                            executor=self.EXECUTOR_NAME)
        except Exception:
            pass

    def _drain(self, inflight: Dict[concurrent.futures.Future, sp.Task]) -> None:
        for fut in list(inflight):
            try:
                fut.result()
            except BaseException:
                pass
            inflight.pop(fut, None)

    def _launch(self, pool, inflight, limit: Optional[float]) -> None:
        while len(inflight) < self.concurrency:
            if self._refresh.is_set():
                return            # soft drain: the in-flight units finish, nothing new starts
            with self._lock:
                ctx, plans, cooldown = self._dispatch(limit)
                task = sp.select_next_task(self.plan, ctx, self.EXECUTOR)
                if task is None:
                    return
                candidate, changed = self._claim(task, plans, cooldown, ctx.now)
                frozen = plans.get(task.track)
                if changed:
                    self._save()
                if candidate is None:
                    if not changed:
                        return            # not ours after all (cannot happen: same owner rule); next pass
                    continue
            inflight[pool.submit(self._work, task, candidate, frozen)] = task
            self._bump("launched")

    def _idle_reason(self, limit: Optional[float]) -> Tuple[str, Optional[float], Optional[str]]:
        """(reason, until, other executor). Only work this executor owns counts as a quota wait; pending work
        owned by another executor makes it idle (until = when the handed-off work may come back here). Owned,
        ready work is "idle, now" only when select_next_task would take it now (a cooldown just ended: launch
        again); when the selector holds it back (smart horizon / low watermark) it is "horizon" without a time."""
        with self._lock:
            ctx, plans, _cooldown = self._dispatch(limit)
            now = ctx.now
            chains: Dict[Tuple[str, str], fb.CandidateChain] = {}
            plan = self.plan
            status = sp.derive_status(plan, self.host.scheduler_context().play_pos, now, limit)
            if status.complete:
                # M6a: the primary is complete; the run is complete when every ready comparison track is
                # finished too (a failed unit ends its track; it never blocks the primary)
                tracks = [t for t in plans if t != plan.primary]
                if all(self._track_finished(t) for t in tracks):
                    return "complete", None, None
            else:
                tracks = [plan.primary]
            retries, others, reclaim, ours = [], [], [], []
            for track, s in ((t, s) for t in tracks for s in plan.slices):
                _frozen = plans.get(track) or {}
                for stage in sp.STAGES:           # the slice's next unfinished stage (untouched units too)
                    st = s.get(stage, track)
                    if st.status == sp.DONE:
                        continue
                    if st.status != sp.PENDING:
                        break
                    owner = ctx.owner(stage, track, st)
                    if owner == UNSCHEDULED:
                        break
                    if owner and owner != self.EXECUTOR_NAME:
                        others.append(owner)
                        if st.reclaim_executor == self.EXECUTOR_NAME and st.reclaim_at and st.reclaim_at > now:
                            reclaim.append(st.reclaim_at)
                        for rec in st.tried.values():
                            if rec.get("executor") == self.EXECUTOR_NAME and rec.get("retry_at") \
                                    and float(rec["retry_at"]) > now:
                                reclaim.append(float(rec["retry_at"]))
                        role = fb.stage_role(stage)
                        if (track, role) not in chains:
                            chains[(track, role)] = fb.CandidateChain(_frozen, role, cooldown=_cooldown,
                                                                      **self._chain_kwargs())
                        for c in chains[(track, role)].candidates:   # our candidate still cooling down in the Ledger
                            if c.executor == self.EXECUTOR_NAME and c.id not in st.tried and _cooldown:
                                until = _cooldown(c)
                                if until is not None and until > now:
                                    reclaim.append(float(until))
                    elif st.retry_at is not None and st.retry_at > now:
                        retries.append(st.retry_at)
                    else:
                        ours.append(now)      # ours and ready: the owner changed since the last launch pass
                    break
            if retries:
                return "quota", min(retries), None
            waiting_audio = any(s.get(sp.STAGE_ASR, plan.primary).status == sp.PENDING and limit is not None
                                and s.end > limit + 0.5 for s in plan.slices)
            if plan.open or waiting_audio:
                return "download", None, None
            if status.stuck and not others:
                return "stuck", None, None
            if ours:
                if sp.select_next_task(plan, ctx, self.EXECUTOR) is not None:
                    return "idle", now, None      # became ours between launch and this check (a cooldown ended)
                # Ours and ready, but the selector holds it back (watched: buffer ahead of play_pos >= low
                # watermark; unwatched: beyond the horizon cap). Nothing to wait for here: the run ends and the
                # host starts a new one once dispatchable_task picks a unit again (playback advances, seek,
                # source limit or task state change). Never "idle until now" (sample video A restart loop).
                return "horizon", None, None
            if others:
                return "idle", (min(reclaim) if reclaim else None), others[0]
            return ("stuck" if status.stuck else "idle"), None, None

    def _track_finished(self, track: str) -> bool:
        """Every slice of the track is done, or has a failed unit (M6a: ends that comparison track)."""
        for s in self.plan.slices:
            for stage in sp.STAGES:
                st = s.get(stage, track)
                if st.status == sp.FAILED:
                    break
                if st.status != sp.DONE:
                    return False
        return True

    def _result(self, reason: str, *, limit: Optional[float] = None, wait_until: Optional[float] = None,
                waiting_executor: Optional[str] = None) -> RunResult:
        status = None
        if self.plan is not None:
            try:
                status = sp.derive_status(self.plan, 0.0, self.clock(), limit)
            except Exception:
                status = None
        self._last_result = RunResult(reason=reason, plan=self.plan, status=status,
                                      publication=self.last_publication, wait_until=wait_until,
                                      waiting_executor=waiting_executor)
        return self._last_result

    # -- plan setup & recovery ------------------------------------------------------------------
    def _acquire_store(self) -> PlanStore:
        if self._store is None:
            store = PlanStore.acquire(self.folder)
            self._store = store
            self._lock = store.lock
            self._publish_lock = store.publish_lock
            self._mt_rows = store.mt_rows
            self._partial = store.partial
            self._track_rows = store.track_rows
            self._track_partial = store.track_partial
            self._asr_digest = store.asr_digest
        return self._store

    def _release_store(self) -> None:
        store, self._store = self._store, None
        if store is not None:
            store.release()

    def _prepare(self) -> bool:
        usable, limit, total = self.host.source_state()
        if not usable:
            return False
        self._media_id = (self.host.media_id() or sr.media_fingerprint(self.source))
        store = self._acquire_store()
        with store.lock:
            store.profiles[self.EXECUTOR_NAME] = self._key_for
            fresh = store.plan is None or store.plan.media_id != self._media_id
            if fresh:
                store.plan = self._load_or_create_plan(limit, total)
                store.clear_results()
            self.plan = store.plan
            self.plan.release_running(self.EXECUTOR_NAME)    # a crashed / closed app left these as running
            self._sync_tracks()                               # M6a: comparison rows added / removed
            if fresh:
                self._reconcile()
            self._load_partials()
            self._save()
        return True

    def _load_or_create_plan(self, limit, total) -> sp.SlicePlan:
        plan = sp.load_slice_plan(self.folder)
        if plan is not None and plan.media_id != self._media_id:
            plan = None            # another media: new plan; the content-addressed caches stay
        if plan is None:
            duration = total if limit is not None else None
            if not duration:
                probe = self.deps.probe_duration
                if probe is None:
                    from oh_streamer.subtitles import runtime
                    duration = runtime._probe_audio_duration(str(self.source), runtime.resolve_ffmpeg(), 15)
                else:
                    duration = probe(self.source)
            if not duration or duration <= 0:
                raise ValueError("cannot determine the media duration")
            kwargs: Dict[str, Any] = {"media_id": self._media_id, "slice_seconds": self.slice_seconds}
            if self.strategy:
                kwargs["strategy"] = self.strategy
            if self.horizon_minutes:
                kwargs["horizon_minutes"] = self.horizon_minutes
            if self.first_slice_seconds is not None:
                kwargs["first_slice_seconds"] = self.first_slice_seconds
            plan = sp.new_slice_plan(float(duration), **kwargs)
        return plan

    def _ordered_candidates(self, chain: fb.CandidateChain, st: sp.TaskState) -> List[fb.Candidate]:
        """The unit's own candidate first (state.candidate, else a candidate whose binding id is the stored
        model_id: plans written before M4b), then the rest of the current chain."""
        first = [c for c in chain if c.id == st.candidate]
        if not first and st.model_id:
            first = [c for c in chain if c.id == st.model_id or c.row.get("id") == st.model_id][:1]
        return first + [c for c in chain if c not in first]

    def _find_result(self, stage: str, s: sp.Slice, st: sp.TaskState, chain: fb.CandidateChain,
                     digest: Optional[str] = None,
                     track: str = PRIMARY) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
        """(key, candidate id, doc) of a finished result of any candidate of the current chain, else Nones.
        Keys are computed by the executor that owns the candidate (PlanStore profile); for an executor not
        running in this process the stored `result_key` is accepted when the result says it was made by a
        binding with the same identity (and, for mt, from the same ASR result)."""
        read = sr.read_asr if stage == sp.STAGE_ASR else sr.read_mt
        ordered = self._ordered_candidates(chain, st)
        profiles = self._store.profiles if self._store else {self.EXECUTOR_NAME: self._key_for}
        for c in ordered:
            fn = profiles.get(c.executor)
            if fn is None:
                continue
            try:
                key = fn(stage, s, c, digest, track)
            except Exception:
                key = None
            doc = read(self.folder, key) if key else None
            if doc is not None:
                return key, c.id, doc
        if st.result_key:
            doc = read(self.folder, st.result_key)
            if doc is not None and (stage == sp.STAGE_ASR or doc.get("asr_digest") == digest):
                ident, cid = doc.get("candidate_identity"), doc.get("candidate")
                for c in ordered:
                    if c.executor in profiles or ident is None:
                        continue
                    if (cid is None or cid == c.id) and ident == self._row_identity(stage, c):
                        return st.result_key, c.id, doc
        return None, None, None

    def _own_result(self, stage: str, st: sp.TaskState, digest: Optional[str] = None):
        """(key, doc) of the unit's OWN recorded result while its cache file still exists (and, for mt, was made from
        the same ASR result), else (None, None). A settings change (chain / identity) does not invalidate done work."""
        if not st.result_key:
            return None, None
        doc = (sr.read_asr if stage == sp.STAGE_ASR else sr.read_mt)(self.folder, st.result_key)
        if doc is None or (stage == sp.STAGE_MT and doc.get("asr_digest") != digest):
            return None, None
        return st.result_key, doc

    def _reconcile(self) -> None:
        """Done work whose cached result is gone is pending again. A done unit keeps its own recorded result while
        that file exists, even when the current chain / model identity no longer matches it (a settings change
        must not silently redo finished work); otherwise a result of ANY current candidate counts (cloud-done work
        is not redone by the local executor after a restart, and vice versa). Finished results are loaded for
        publication."""
        for track in self._ready_tracks():
            self._reconcile_track(track)

    def _reconcile_track(self, track: str) -> None:
        frozen = self._frozen(track)
        asr_chain, mt_chain = self._chain(frozen, sp.STAGE_ASR), self._chain(frozen, sp.STAGE_MT)
        rows = self._rows(track)
        for s in self.plan.slices:
            asr_doc = asr_key = None
            st = s.get(sp.STAGE_ASR, track)
            if st.status == sp.DONE:
                asr_key, cid, asr_doc = self._find_result(sp.STAGE_ASR, s, st, asr_chain, track=track)
                if asr_doc is None:
                    asr_key, asr_doc = self._own_result(sp.STAGE_ASR, st)
                    cid = st.candidate
                if asr_doc is None:
                    s.set(sp.STAGE_ASR, track, sp.TaskState())
                else:
                    s.set(sp.STAGE_ASR, track, sp.TaskState(status=sp.DONE, model_id=st.model_id, candidate=cid,
                                                            result_key=asr_key))
            st = s.get(sp.STAGE_MT, track)
            if st.status == sp.DONE:
                mt_doc = key = cid = None
                if asr_doc is not None:
                    digest = self._digest_of(asr_key, asr_doc["segments"])
                    key, cid, mt_doc = self._find_result(sp.STAGE_MT, s, st, mt_chain, digest, track)
                    if mt_doc is None:
                        key, mt_doc = self._own_result(sp.STAGE_MT, st, digest)
                        cid = st.candidate
                if mt_doc is None:
                    s.set(sp.STAGE_MT, track, sp.TaskState())
                else:
                    s.set(sp.STAGE_MT, track, sp.TaskState(status=sp.DONE, model_id=st.model_id, candidate=cid,
                                                           result_key=key))
                    rows[s.index] = list(mt_doc["segments"])

    def _asr_doc(self, s: sp.Slice, track: Optional[str] = None) -> Tuple[Optional[str], Optional[dict]]:
        """(key, doc) of the finished ASR result of a slice, whichever candidate made it."""
        st = s.get(sp.STAGE_ASR, track or self.plan.primary)
        if st.status != sp.DONE or not st.result_key:
            return None, None
        return st.result_key, sr.read_asr(self.folder, st.result_key)

    def _load_partials(self) -> None:
        """Translation batches checkpointed before a restart stay on screen (cloud checkpoints)."""
        for track in self._ready_tracks():
            done, partial = self._rows(track), self._partials(track)
            for s in self.plan.slices:
                if s.index in done or s.index in partial or s.is_done(sp.STAGE_MT, track):
                    continue
                key, doc = self._asr_doc(s, track)
                if doc:
                    rows = sr.read_partial(self.folder,
                                           self._partial_key(self._digest_of(key, doc["segments"]), track))
                    if rows:
                        partial[s.index] = rows

    def _save(self) -> None:
        with self._lock:
            sp.save_slice_plan(self.plan, self.folder)

    # -- one task -------------------------------------------------------------------------------
    def _work(self, task: sp.Task, candidate: fb.Candidate, frozen: Dict[str, Any]) -> None:
        from oh_streamer.subtitles import usage_log
        with usage_log.context(self.folder.parent.name, slice_index=task.slice_index, stage=task.stage,
                               candidate=candidate.id, track=task.track,
                               on_request=lambda: self._bump("requests")):
            self._work_inner(task, candidate, frozen)

    def _work_inner(self, task: sp.Task, candidate: fb.Candidate, frozen: Dict[str, Any]) -> None:
        try:
            self._check()
            self._progress(state=task.stage, slice=task.slice_index, stage=task.stage, candidate=candidate.id)
            if task.stage == sp.STAGE_ASR:
                self._run_asr(task, candidate, frozen)
            else:
                self._run_mt(task, candidate, frozen)
            self._usage_unit_done(task)
            self._progress(state="done", slice=task.slice_index, stage=task.stage, candidate=candidate.id)
        except InterruptedError as exc:
            with self._lock:
                if self.plan.slices[task.slice_index].get(task.stage, task.track).status == sp.RUNNING:
                    self.plan.release(task.slice_index, task.stage, task.track)
                    self._save()
            raise exc if isinstance(exc, Cancelled) else Cancelled()
        except Exception as exc:
            self._failed(task, exc, candidate)

    def _slice(self, task: sp.Task) -> sp.Slice:
        return self.plan.slices[task.slice_index]

    def _asr_call(self, candidate: fb.Candidate, frozen: Dict[str, Any], piece: Path, s: sp.Slice, duration: float):
        run = self._runner(candidate, frozen)
        from oh_streamer.subtitles import usage_log
        usage_log.set_audio_seconds(duration)
        if self.deps.asr_run is not None and not candidate.row.get(SYNTHETIC):
            return self.deps.asr_run(candidate, piece, s, run)
        return run(lambda _row, _key: self.deps.asr(piece, s), audio=duration)

    def _result_meta(self, stage: str, candidate: fb.Candidate) -> Dict[str, Any]:
        return {"candidate": candidate.id, "candidate_identity": self._row_identity(stage, candidate)}

    def _run_asr(self, task: sp.Task, candidate: fb.Candidate, frozen: Dict[str, Any]) -> None:
        s = self._slice(task)
        key = self._asr_key(s, candidate)
        doc = sr.read_asr(self.folder, key)
        if doc is None:
            out = self.folder / "cache" / "audio" / f"{key[:16]}.flac"
            pieces_fn = self.deps.pieces
            if pieces_fn is None:
                from oh_streamer.subtitles import slice_results
                pieces_fn = slice_results._audio_pieces
            audio_start = self.asr_window.audio_start(s)          # pre-roll (design 7.2)
            pieces = pieces_fn(self.source, out, audio_start, s.end - audio_start)
            segments: List[Any] = []
            sources: List[Any] = []
            try:
                from oh_streamer.subtitles.slice_results import _shift_segment
                ends = [off for _p, off in pieces[1:]] + [s.end - audio_start]
                for (piece, offset), end in zip(pieces, ends):
                    self._check()
                    part = self._asr_call(candidate, frozen, Path(piece), s, max(0.0, end - offset))
                    if isinstance(part, dict):
                        sources += list(part.get("sources") or [])
                        part = part.get("segments", [])
                    segments += [_shift_segment(seg, offset + audio_start) for seg in part]   # absolute
            finally:
                for piece, _offset in pieces:
                    try:
                        Path(piece).unlink()
                    except OSError:
                        pass
            self._check()
            raw_count = len(segments)
            w = self.asr_window
            owned = sp.own_segments([seg for seg in segments if isinstance(seg, dict)], s.start, s.end,
                                    first=s.start <= 0.0, last=sp.is_last_slice(self.plan, s),
                                    lead_keep_s=w.lead_keep_s, tail_guard_s=w.tail_guard_s,
                                    keep_preroll=True)     # publication drops those the previous slice covers
            segments = [_shift_segment(seg, -s.start) for seg in owned]                    # slice-relative
            model_id = (self.deps.asr_model_id or self.deps.asr_identity.get("id", "")) \
                if candidate.row.get(SYNTHETIC) else str(candidate.row.get("id") or candidate.id)
            if sources and all(src.get("model") == sources[0].get("model") for src in sources) and sources[0].get("model"):
                model_id = sources[0]["model"].get("id") or model_id
            doc = {"media_id": self._media_id, "start": s.start, "end": s.end,
                   "model": self._identity(sp.STAGE_ASR, candidate), **self._result_meta(sp.STAGE_ASR, candidate),
                   "model_id": model_id, "segments": segments, "sources": sources,
                   "audio_start": audio_start, "window": w.cache_params(), "raw_count": raw_count}
            sr.write_asr(self.folder, key, doc)
        with self._lock:
            self.plan.mark_done(s.index, sp.STAGE_ASR, str(doc.get("model_id") or ""), task.track,
                                candidate=candidate.id, result_key=key)
            self._save()

    def _translator(self, candidate: fb.Candidate, frozen: Dict[str, Any]):
        if self.deps.translator_for is not None and not candidate.row.get(SYNTHETIC):
            return self.deps.translator_for(candidate, self._runner(candidate, frozen))
        return self.deps.translator()

    def _run_mt(self, task: sp.Task, candidate: fb.Candidate, frozen: Dict[str, Any]) -> None:
        from oh_streamer.subtitles import usage_log
        from oh_streamer.subtitles.slice_results import _prepare_slice_cues, _translate_batch, batch_needs_shrink
        from oh_streamer.subtitles.core import parse_numbered_translations
        from oh_streamer.subtitles.postprocess import _safe_to_taiwan

        s = self._slice(task)
        asr_key, asr_doc = self._asr_doc(s, task.track)
        if asr_doc is None:
            raise RuntimeError(f"ASR result missing for slice {s.index}")
        digest = self._digest_of(asr_key, asr_doc["segments"])
        track = task.track
        primary = track == self.plan.primary
        done_rows, partial_rows = self._rows(track), self._partials(track)
        key = self._mt_key(digest, candidate, track)
        doc = sr.read_mt(self.folder, key)
        if doc is not None:
            rows, model_id = list(doc["segments"]), str(doc.get("model_id") or "")
        else:
            # sentence splitting works on absolute times (a pre-roll cue starts before the slice: negative
            # slice-relative times are not valid Segments); the cues are stored slice-relative again
            from oh_streamer.subtitles.slice_results import _shift_segment
            absolute = [_shift_segment(dict(seg), s.start) for seg in asr_doc["segments"]]
            cues = [_shift_segment(dict(c), -s.start)
                    for c in _prepare_slice_cues(absolute, config=self._track_config(track), settings=self.settings)]
            pkey = self._partial_key(digest, track)
            rows = [r for r in sr.read_partial(self.folder, pkey)]
            if any(r.get("text_ja") != c.get("text_ja") for r, c in zip(rows, cues)) or len(rows) > len(cues):
                rows = []                      # stale checkpoint
            with self._lock:
                partial_rows[s.index] = rows
            if rows:
                self._publish_for(track)
            client = self._translator(candidate, frozen) if len(rows) < len(cues) else None
            fallback_source = self._row_identity(sp.STAGE_MT, candidate)
            shrink_key = (track, s.index)
            progressed = False
            try:
                while len(rows) < len(cues):
                    self._check()
                    shrink = self._mt_shrink.get(shrink_key)
                    if shrink and len(rows) >= shrink[0]:
                        self._mt_shrink.pop(shrink_key, None)
                        shrink = None
                    batch = cues[len(rows):len(rows) + (shrink[1] if shrink else MT_BATCH_SIZE)]
                    try:
                        parsed = _translate_batch(client, batch, parse_numbered_translations)
                    except (ProviderError, ValueError) as exc:
                        if len(batch) <= 1 or not batch_needs_shrink(exc):
                            raise
                        # one request at a time: the smaller retry may be paced (not_sent) or hit a 429, and the
                        # unit then resumes here with the smaller size instead of resending the garbled batch
                        self._mt_shrink[shrink_key] = (len(rows) + len(batch), len(batch) // 2)
                        usage_log.event(self._usage_id(), "mt_batch_shrink", slice=s.index, track=track,
                                        at=len(rows), size=len(batch), to=len(batch) // 2,
                                        reason=getattr(exc, "kind", None) or type(exc).__name__,
                                        too_large=bool(getattr(exc, "too_large", False)) or None)
                        continue
                    source = fb.client_source(client) or fallback_source
                    new_rows = []
                    for cue, zh_raw in zip(batch, parsed):
                        zh = _safe_to_taiwan(zh_raw)
                        skip = "[skip]" in zh_raw.casefold() or not zh.strip()
                        row = dict(cue, mt_zh="" if skip else zh, zh="" if skip else zh, is_filtered=skip,
                                   status="mt_ready", translation_source=dict(source))
                        if skip:
                            row["filter_reason"] = "onomatopoeia_or_hallucination"
                        new_rows.append(row)
                    rows = rows + new_rows
                    progressed = True
                    sr.write_partial(self.folder, pkey, rows)     # survives a candidate switch / restart
                    with self._lock:
                        partial_rows[s.index] = rows
                    self._publish_for(track)   # D2: each batch is on screen before the slice is complete
            except ProviderError as exc:
                if progressed:
                    exc.progressed = True      # CandidateChain.decide: a 429 after progress starts a new streak
                raise
            finally:
                close = getattr(client, "close", None)
                if close:
                    close()
            self._check()
            used = sorted({str((r.get("translation_source") or {}).get("id")) for r in rows
                           if (r.get("translation_source") or {}).get("id")})
            model_id = used[0] if len(used) == 1 else ("mixed" if used else str(fallback_source.get("id", "")))
            sr.write_mt(self.folder, key, {"media_id": self._media_id, "start": s.start, "end": s.end,
                                           "model": self._identity(sp.STAGE_MT, candidate),
                                           **self._result_meta(sp.STAGE_MT, candidate), "asr_digest": digest,
                                           "model_id": model_id, "models_used": used, "segments": rows})
            sr.clear_partial(self.folder, pkey)
        with self._lock:
            done_rows[s.index] = rows
            partial_rows.pop(s.index, None)
            self.plan.mark_done(s.index, sp.STAGE_MT, model_id, task.track, candidate=candidate.id,
                                result_key=key)
            self._save()
        self._publish_for(track)

    # -- failure policy: fallback.CandidateChain.decide (design 4.7) --------------------------------
    def _error_kind(self, exc: Exception) -> Optional[str]:
        return None             # fallback.error_kind: ProviderError.kind, permission, else error

    def _source_gone(self, exc: BaseException) -> bool:
        """Did `exc` come from a range-cache source whose session is not live (paused / recycled / closed)? That is
        not a failure of the unit or the model: the work is released and the executor ends as source_unavailable
        (the tick restarts it when the session is back), instead of burning retries or failing the item."""
        from oh_streamer.sources import media_input
        if isinstance(exc, media_input.SourceUnavailable):
            return True
        if not media_input.needs_runtime_url(self.source):
            return False
        try:
            return not self.host.source_state()[0]
        except Exception:  # noqa: BLE001
            return False

    def _failed(self, task: sp.Task, exc: Exception, candidate: Optional[fb.Candidate] = None) -> None:
        """Decide what the failed unit does next with the CURRENT chain (a backup added meanwhile counts, A2)
        and store it (apply_decision). A candidate given up after its retries gets one Ledger cooldown."""
        if self._source_gone(exc):
            with self._lock:
                if self.plan.slices[task.slice_index].get(task.stage, task.track).status == sp.RUNNING:
                    self.plan.release(task.slice_index, task.stage, task.track)
                    self._save()
            self._progress(state="deferred", slice=task.slice_index, stage=task.stage,
                           detail="source_unavailable: 來源暫時無法讀取，稍後重試")
            return
        now = self.clock()
        kind = self._error_kind(exc)
        self._bump("failed")
        with self._lock:
            state = self.plan.slices[task.slice_index].get(task.stage, task.track)
            frozen = self._frozen(task.track)
            chain = self._chain(frozen, task.stage)
            cid = candidate.id if candidate is not None else state.candidate
            current = chain.get(cid) or candidate
            if current is None:                        # unknown / removed candidate: record it and move on
                current = fb.Candidate(index=len(chain), id=str(cid or self.EXECUTOR_NAME), row={},
                                       executor=self.EXECUTOR_NAME)
            decision = chain.decide(state, current, exc, now, kind=kind)
            fb.apply_decision(self.plan, task.slice_index, task.stage, decision, track=task.track,
                              model_id=current.id)
            self._save()
        if decision.action in (fb.SWITCH, fb.FAILED):
            self._usage_candidate_changed(task, current.id, decision)
        self._usage_unit_end(task, "failed", action=decision.action, kind=decision.kind, detail=decision.detail)
        if decision.action in (fb.SWITCH, fb.FAILED) and decision.kind in fb.RETRY_KINDS:
            self._give_up_cooldown(current, decision, frozen)
        if decision.action == fb.FAILED and task.track != self.plan.primary:
            self._publish_for(task.track)          # the track's manifest entry says why it stopped
        state_name = {fb.RETRY: "deferred", fb.WAIT_QUOTA: "deferred", fb.FAILED: "failed"}.get(
            decision.action, "handoff" if decision.handoff else "switch")
        retry_at = decision.retry_at
        self._progress(state=state_name, slice=task.slice_index, stage=task.stage,
                       detail=f"{decision.kind}: {decision.detail}", kind=decision.kind, candidate=current.id,
                       next_candidate=decision.candidate.id if decision.candidate else None,
                       wait_s=max(0.0, retry_at - now) if retry_at else None)
        try:
            self.host.on_event({"type": "subtitle_fallback", "state": decision.action, "handoff": decision.handoff,
                                "role": fb.stage_role(task.stage), "slice": task.slice_index, "stage": task.stage,
                                "from": current.id, "to": decision.candidate.id if decision.candidate else None,
                                "reason": decision.kind, "retry_at": retry_at, "executor": self.EXECUTOR_NAME,
                                "wait_executor": decision.wait_executor, "fail_kind": decision.fail_kind})
        except Exception:
            pass

    def _give_up_cooldown(self, candidate: fb.Candidate, decision: fb.Decision, frozen: Dict[str, Any]) -> None:
        """A candidate abandoned after its retries (network / server / local error ...) cools down once in the
        Ledger (policy cooldown_seconds), so other units skip it instead of each trying it 3 times."""
        if self.ledger is None or not candidate.row or candidate.row.get(SYNTHETIC):
            return
        try:
            key = self.credentials(dict(candidate.row))
            ck = fb.candidate_key(self.ledger.account(candidate.row, key), candidate.row)
            seconds = float((frozen.get("policy") or {}).get("cooldown_seconds", DEFAULT_GIVE_UP_COOLDOWN_S))
            self.ledger.fail(ck, ProviderError(decision.kind, "gave up after retries", retryable=True), seconds)
        except Exception:
            pass

    # -- publication / progress -------------------------------------------------------------------
    def _publish(self) -> Optional[Dict[str, Any]]:
        """Both executors publish the same combined result (the PlanStore's finished + partial rows)."""
        with self._publish_lock:
            with self._lock:
                done = {i: list(rows) for i, rows in self._mt_rows.items()
                        if self.plan.slices[i].is_done(sp.STAGE_MT, self.plan.primary)}
                partial = {i: list(rows) for i, rows in self._partial.items() if i not in done}
            self._usage_slices(self.plan.primary, done)
            state = FolderState.of(self.folder)
            sig = _publication_signature(self.plan, done, partial, self.japanese)
            cached = state.published.get(self.plan.primary)
            if cached is not None and cached[0] == sig and self._files_exist(cached[1]):
                result = cached[1]               # same rows, same files: only the host hears of it again
            else:
                result = sr.publish(self.folder, self.plan, done, partial, japanese=self.japanese)
                state.published[self.plan.primary] = (sig, result)
            self.last_publication = result
            if self._store is not None:
                self._store.last_publication = result
            try:
                self.host.publish(result)
            except Exception:
                pass
            return result

    @staticmethod
    def _files_exist(result: Dict[str, Any]) -> bool:
        """The files of a cached publication are still there (deleted / regenerated folder: publish again)."""
        try:
            paths = [Path(p) for p in (result.get("srt") or {}).values()]
            if not paths:
                return False
            if result.get("complete"):
                root = paths[0].parent
                paths += [root / "sakura.zh-TW.srt", root / "sakura.zh-TW.json"]
            return all(p.exists() for p in paths)
        except Exception:
            return False

    def _combo_manifest_has(self, track: str) -> bool:
        try:
            manifest = sr._read_json(self.folder / sr.COMBOS_DIR / "manifest.json") or {}
            return any(isinstance(e, dict) and e.get("id") == track for e in manifest.get("combos") or [])
        except Exception:
            return False

    def _combo_files_exist(self, track: str, entry: Dict[str, Any]) -> bool:
        try:
            root = self.folder / sr.COMBOS_DIR
            names = [n for n in (entry.get("files") or {}).values() if n]
            if not names or not all((root / n).exists() for n in names):
                return False
            return self._combo_manifest_has(track)
        except Exception:
            return False

    def _publish_for(self, track: str) -> None:
        """A new result of `track`: the primary publication, or a comparison track's combos files + manifest."""
        if track == self.plan.primary:
            self._publish()
        else:
            self._publish_track(track)

    def _combo_video(self) -> Dict[str, Any]:
        return {"path": str(self.source), "media_id": self._media_id}

    def _publish_track(self, track: str) -> Optional[Dict[str, Any]]:
        """Comparison track -> <subtitles>/combos/<track>.zh-TW.srt (+ .ja.srt) and manifest.json (M6b), from the
        same combination as the primary (slice_results). Under the PlanStore publish lock, so the cloud and the
        local executor never overwrite each other's manifest entries."""
        if self.plan is None or track == self.plan.primary:
            return None
        plans = self._track_plans()
        info = plans.get(track) or {}
        try:
            with self._publish_lock:
                with self._lock:
                    if track not in self.plan.tracks:
                        return None
                    done = {i: list(rows) for i, rows in self._rows(track).items()
                            if self.plan.slices[i].is_done(sp.STAGE_MT, track)}
                    partial = {i: list(rows) for i, rows in self._partials(track).items() if i not in done}
                    combo = info.get("combo") or {}
                    state = FolderState.of(self.folder)
                    cached = state.published.get(track)
                    if not info.get("ready", False):
                        reasons = (self._store.track_errors.get(track) if self._store else None)                             or info.get("reasons") or ["比較字幕設定無效"]
                        sig = _publication_signature(self.plan, {}, {}, ["unready", list(reasons),
                                                                         str(info.get("label") or track)])
                        if cached is not None and cached[0] == sig and self._combo_manifest_has(track):
                            return cached[1]
                        entry = sr.publish_combo_unready(self.folder, self.plan, track, reasons,
                                                         label=str(info.get("label") or track),
                                                         video=self._combo_video())
                        state.published[track] = (sig, entry)
                        return entry
                    self._usage_slices(track, done)
                    sig = _publication_signature(self.plan, done, partial, [
                        "ready", str(info.get("label") or track), combo, sr.track_stage_summary(self.plan, track),
                        sr.track_failure(self.plan, track), bool(combo.get("japanese_track")), self._combo_video()])
                    if cached is not None and cached[0] == sig and self._combo_files_exist(track, cached[1]):
                        return cached[1]
                    entry = sr.publish_combo(self.folder, self.plan, track, done, partial,
                                             label=str(info.get("label") or track), combo=combo,
                                             japanese=bool(combo.get("japanese_track")),
                                             video=self._combo_video())
                    state.published[track] = (sig, entry)
                    return entry
        except Exception:
            return None         # a combos file problem must never stop the subtitle work

    def _publish_tracks(self) -> None:
        """Every comparison track once (run start / completion): restart state, not-ready reasons, final status."""
        if self.plan is None:
            return
        for track in list(self.plan.tracks)[1:]:
            self._publish_track(track)

    def _progress(self, **info: Any) -> None:
        plan = self.plan
        if plan is not None:
            info.setdefault("done_slices", sum(1 for s in plan.slices if s.all_done(plan.primary)))
            info.setdefault("total_slices", len(plan.slices))
        info.setdefault("executor", self.EXECUTOR_NAME)
        try:
            self.host.on_progress(info)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------------------------------

def run_cloud_subtitles(folder: Path | str, source: Path | str, *, settings: Optional[dict] = None,
                        config: Optional[dict] = None, host: Optional[CloudHost] = None,
                        deps: Optional[CloudDeps] = None, concurrency: int = 1, wait: bool = True,
                        japanese: bool = False, **kwargs: Any) -> RunResult:
    """Run the v2 cloud executor on one item folder (``<item>/subtitles``) until its plan is complete (wait=True)
    or nothing can run right now (wait=False). Blocks; the caller picks the thread. `settings` = the item's
    settings (frozen runtime plan inside); `deps` replaces the real providers (tests)."""
    executor = CloudSliceExecutor(folder, source, host=host, deps=deps, settings=settings, config=config,
                                  concurrency=concurrency, japanese=japanese, **kwargs)
    return executor.run(wait=wait)


def default_queue_root() -> Path:
    return platform_support.data_dir() / "queue"


def load_queue_item(queue_root: Path | str, item_id: str):
    """Read-only lookup of an item in queue.json (does not start a QueueStore, which would normalise states)."""
    from oh_streamer.queue.queue_store import Item, QUEUE_FILE
    data = json.loads((Path(queue_root) / QUEUE_FILE).read_text(encoding="utf-8"))
    for raw in data.get("items", []):
        if raw.get("id") == item_id:
            return Item.from_dict(raw), data.get("settings") or {}
    raise KeyError(item_id)


class SimulatedDownloadHost(CloudHost):
    """Pretends the source is still downloading: the readable prefix grows `rate` media-seconds per real second
    from 0 up to `total` (then limit None = whole file). For M2 acceptance without the queue service."""

    def __init__(self, total: float, rate: float = 8.0, clock: Callable[[], float] = time.monotonic):
        self.total, self.rate, self.clock = float(total), float(rate), clock
        self.t0 = clock()

    def source_state(self):
        readable = (self.clock() - self.t0) * self.rate
        return True, (None if readable >= self.total else readable), self.total


class TimingHost(CloudHost):
    """Wraps another host and records when things happened (seconds since creation) -> timing.json."""

    def __init__(self, inner: Optional[CloudHost] = None, clock: Callable[[], float] = time.monotonic,
                 echo: bool = False):
        self.inner = inner or CloudHost()
        self.clock, self.echo = clock, echo
        self.t0 = clock()
        self.first_asr_done_s: Optional[float] = None
        self.first_subtitle_s: Optional[float] = None
        self.slice_done_s: Dict[int, float] = {}
        self.publishes: List[Dict[str, Any]] = []
        self.defers = 0
        self.defer_wait_s = 0.0
        self.defer_kinds: Dict[str, int] = {}
        self.complete_s: Optional[float] = None
        self._lock = threading.Lock()

    def _now(self) -> float:
        return round(self.clock() - self.t0, 3)

    def source_state(self):
        return self.inner.source_state()

    def media_id(self):
        return self.inner.media_id()

    def scheduler_context(self):
        return self.inner.scheduler_context()

    def check_cancel(self):
        return self.inner.check_cancel()

    def on_event(self, event):
        return self.inner.on_event(event)

    def on_progress(self, info):
        now = self._now()
        with self._lock:
            state, stage = info.get("state"), info.get("stage")
            if state == "done" and stage == sp.STAGE_ASR and self.first_asr_done_s is None:
                self.first_asr_done_s = now
            if state == "done" and stage == sp.STAGE_MT:
                self.slice_done_s.setdefault(int(info["slice"]), now)
            if state == "deferred":
                self.defers += 1
                self.defer_wait_s += float(info.get("wait_s") or 0.0)
                kind = str(info.get("kind"))
                self.defer_kinds[kind] = self.defer_kinds.get(kind, 0) + 1
        if self.echo:
            print(f"[{now:8.2f}s]", {k: v for k, v in info.items() if v not in (None, "")}, flush=True)
        self.inner.on_progress(info)

    def publish(self, result):
        now = self._now()
        with self._lock:
            self.publishes.append({"t": now, "cues": result["cues"], "slices_done": result["slices_done"],
                                   "covered_s": round(result["covered_s"], 1), "complete": result["complete"]})
            if self.first_subtitle_s is None and result["cues"] > 0:
                self.first_subtitle_s = now
            if result["complete"] and self.complete_s is None:
                self.complete_s = now
        if self.echo:
            print(f"[{now:8.2f}s] publish slices {result['slices_done']}/{result['slices_total']} "
                  f"cues {result['cues']} covered {result['covered_s']:.0f}s complete={result['complete']}",
                  flush=True)
        self.inner.publish(result)

    def report(self, plan: Optional[sp.SlicePlan] = None) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {
                "first_asr_done_s": self.first_asr_done_s,
                "first_subtitle_s": self.first_subtitle_s,
                "complete_s": self.complete_s,
                "slice_done_s": {str(k): v for k, v in sorted(self.slice_done_s.items())},
                "publishes": list(self.publishes),
                "defer_count": self.defers,
                "defer_wait_s": round(self.defer_wait_s, 3),
                "defer_kinds": dict(self.defer_kinds),
            }
        if plan is not None:
            out["slices"] = [[s.index, s.start, s.end] for s in plan.slices]
            out["first_slice_seconds"] = plan.first_slice_seconds
        return out


def write_timing(folder: Path | str, timing: TimingHost, plan: Optional[sp.SlicePlan] = None) -> Path:
    from oh_streamer.subtitles import runtime
    path = Path(folder) / "timing.json"
    runtime.atomic_write_json(path, timing.report(plan))
    return path


def _queue_settings(queue_root: Path | str) -> dict:
    data = json.loads((Path(queue_root) / "queue.json").read_text(encoding="utf-8"))
    return dict(data.get("settings") or {})


def _resolved_settings(base: dict, override: dict) -> dict:
    """Queue settings (+ item snapshot) with the frozen runtime plan (`subtitle_resolved`) filled in the way
    snapshot_subtitle_settings does for a newly queued item."""
    merged = {**base, **override}
    if "subtitle_resolved" not in merged:
        from oh_streamer.queue.subtitle_jobs import snapshot_subtitle_settings
        merged = snapshot_subtitle_settings(merged)      # what an item gets when it is queued
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oh_streamer.queue.subtitle_scheduler",
        description="Run the v2 cloud subtitle executor on one media file (no GPU, real cloud providers). "
                    "Settings (with the frozen runtime plan) come from a queue item (--item) or a JSON file "
                    "(--settings-file).")
    ap.add_argument("--item", help="queue item id (its settings snapshot and source file unless --source is given)")
    ap.add_argument("--queue-root", default=str(default_queue_root()), help="queue folder holding queue.json")
    ap.add_argument("--settings-file", help="JSON with the item settings (same shape as item.settings); "
                                            "alternative to --item")
    ap.add_argument("--source", help="media file (default: the item's source/output file)")
    ap.add_argument("--folder", help="subtitles folder (default <queue-root>/items/<id>/subtitles, or ./subtitle-run "
                                     "without --item); use a scratch folder to keep an item untouched")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-wait", action="store_true", help="return at the first quota/download wait")
    ap.add_argument("--japanese", action="store_true", help="also write the Japanese track")
    ap.add_argument("--simulate-download", type=float, nargs="?", const=8.0, default=None, metavar="RATE",
                    help="pretend the source downloads: readable prefix grows RATE media-seconds per real second "
                         "(default 8) up to the full length")
    ap.add_argument("--first-slice-seconds", type=float, default=None,
                    help="length of the first slice (default 90); 0 = no short first slice")
    args = ap.parse_args(argv)

    settings: dict = {}
    source = args.source
    folder = Path(args.folder) if args.folder else None
    if args.item:
        item, queue_settings = load_queue_item(args.queue_root, args.item)
        source = source or item.source_path or item.output_path
        settings = _resolved_settings(queue_settings, item.settings or {})
        folder = folder or Path(args.queue_root) / "items" / item.id / "subtitles"
    elif args.settings_file:
        settings = _resolved_settings(json.loads(Path(args.settings_file).read_text(encoding="utf-8")), {})
    elif source:
        settings = _resolved_settings(_queue_settings(args.queue_root), {})     # the app's current subtitle settings
    else:
        ap.error("give --source (uses the app's current subtitle settings), --item or --settings-file")
    folder = folder or Path("subtitle-run")
    if not source or not Path(source).is_file():
        print("source file not found:", source, file=sys.stderr)
        return 2

    inner: CloudHost = CloudHost()
    if args.simulate_download is not None:
        from oh_streamer.subtitles import runtime
        total = runtime._probe_audio_duration(str(source), runtime.resolve_ffmpeg(), 15)
        if not total:
            print("cannot probe the media duration", file=sys.stderr)
            return 2
        inner = SimulatedDownloadHost(total, args.simulate_download)
        print(f"simulating download of {total:.0f}s media at {args.simulate_download:g}x")
    timing = TimingHost(inner, echo=True)
    primary = (settings.get("subtitle_configs") or [{}])[0]
    extra = {}
    if args.first_slice_seconds is not None:
        extra["first_slice_seconds"] = args.first_slice_seconds
    result = run_cloud_subtitles(folder, source, settings=settings, config=primary, host=timing,
                                 concurrency=args.concurrency, wait=not args.no_wait, japanese=args.japanese,
                                 **extra)
    path = write_timing(folder, timing, result.plan)
    report = timing.report()
    st = result.status
    print(f"result: {result.reason}" + (f"; slices {st.slices_done}/{st.slices_total}; stuck={st.stuck}" if st else ""))
    print(f"first ASR done {report['first_asr_done_s']}s; first subtitle {report['first_subtitle_s']}s; "
          f"complete {report['complete_s']}s; defers {report['defer_count']} ({report['defer_wait_s']}s waited)")
    print("timing written to", path)
    for f in (st.failures if st else []):
        print(f"  failed slice {f.slice_index} {f.stage}: {f.kind} {f.detail}")
    return 0 if result.reason == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
