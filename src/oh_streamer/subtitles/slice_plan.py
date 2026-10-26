"""Unified subtitle slice plan.

Pure logic, no I/O side effects (save/load are thin wrappers around atomic_write_json):

- lazy-cut slice plan (extracted from cloud_horizon.HorizonPlan): silence cuts decided as the download
  grows, never moved or removed afterwards (T3.6: cuts may only be INSERTED: islands after a seek into the
  uncut region, splits of a slice nothing has started on), a 30 s first slice;
- work units ``(slice, stage, track)`` with a status stored in the plan;
- content-addressed cache keys (independent of item id / settings snapshot);
- ``select_next_task``: the scheduler rule set of cloud_horizon.select_next_cloud_action, generalised with
  executors and extra tracks;
- ``derive_status``: item-level progress / stuck / waiting derived from the slice table (no stored flags).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from oh_streamer.subtitles import cloud_chunks

SLICE_PLAN_FILE = "slice-plan.json"
PLAN_VERSION = 2          # 2: slices are not necessarily in time order, lazy["islands"] (T3.6); version 1 loads as is

DEFAULT_SLICE_SECONDS = 300.0            # local (GPU) plans
CLOUD_SLICE_SECONDS = 60.0               # cloud ASR plans: Groq whisper skips real lines in 300 s uploads
SHORT_SLICE_SECONDS = 30.0               # T3.6: first slice of every plan / slice at a seek target; silence search
SHORT_SEARCH_LO_SECONDS = 20.0           # [start + 20, start + 45], never widened, else a hard cut at start + 30
SHORT_SEARCH_HI_SECONDS = 45.0
DEFAULT_FIRST_SLICE_SECONDS = SHORT_SLICE_SECONDS
DEFAULT_HORIZON_MINUTES = 30
DEFAULT_LOW_WATERMARK_SECONDS = 1200.0
UNWATCHED_AHEAD_SECONDS = 900.0
DEFAULT_SECONDARY_MIN_LEAD_SECONDS = 600.0   # a non-primary track starts when primary leads by this much

STAGE_ASR = "asr"
STAGE_MT = "mt"
STAGES: Tuple[str, ...] = (STAGE_ASR, STAGE_MT)
PRIMARY_TRACK = "primary"

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


# --------------------------------------------------------------------------------------------------
# Work state
# --------------------------------------------------------------------------------------------------

@dataclass
class TaskState:
    status: str = PENDING
    executor: Optional[str] = None        # running: who runs it
    model_id: Optional[str] = None        # done: model that produced it; failed: last model tried
    fail_kind: Optional[str] = None       # failed
    fail_at: Optional[float] = None
    attempt: int = 0                      # counted retries on the current candidate
    detail: str = ""                      # failed: last reason text
    retry_at: Optional[float] = None      # pending: quota wait, not a failure
    wait_executor: Optional[str] = None   # pending: only this executor may take it (cross-executor fallback)
    # M4 (design 4.7) cross-executor fallback bookkeeping. `candidate`: id (fallback.Candidate.id) of the
    # candidate the work is on. `tried`: candidate id -> {kind, at, detail, executor, attempt?, retry_at?},
    # the last reason per candidate. An entry of a candidate other than `candidate` means it was given up;
    # with `retry_at` it was only left for a quota wait and may take the work back from then on.
    candidate: Optional[str] = None
    tried: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Quota handoff: from reclaim_at on, reclaim_executor (the original one) may take the pending work back.
    reclaim_at: Optional[float] = None
    reclaim_executor: Optional[str] = None
    # M4b: done work remembers the cache key of its result (written by whichever candidate / executor made
    # it), so a restart can find it without recomputing another executor's key (executors' _reconcile).
    result_key: Optional[str] = None

    _OPTIONAL = ("executor", "model_id", "fail_kind", "fail_at", "detail", "retry_at", "wait_executor",
                 "candidate", "reclaim_at", "reclaim_executor", "result_key")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        for key in self._OPTIONAL:
            value = getattr(self, key)
            if value not in (None, ""):
                out[key] = value
        if self.attempt:
            out["attempt"] = self.attempt
        if self.tried:
            out["tried"] = {str(k): dict(v) for k, v in self.tried.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskState":
        """Tolerates plans written before M4 (no candidate / tried / reclaim fields)."""
        def num(key):
            value = data.get(key)
            return None if value is None else float(value)
        tried = data.get("tried")
        tried = ({str(k): dict(v) for k, v in tried.items() if isinstance(v, Mapping)}
                 if isinstance(tried, Mapping) else {})
        return cls(status=str(data.get("status", PENDING)), executor=data.get("executor"),
                   model_id=data.get("model_id"), fail_kind=data.get("fail_kind"), fail_at=num("fail_at"),
                   attempt=int(data.get("attempt", 0)), detail=str(data.get("detail", "")),
                   retry_at=num("retry_at"), wait_executor=data.get("wait_executor"),
                   candidate=data.get("candidate"), tried=tried,
                   reclaim_at=num("reclaim_at"), reclaim_executor=data.get("reclaim_executor"),
                   result_key=data.get("result_key"))

    def carry(self, **changes: Any) -> "TaskState":
        """A new state keeping the fallback bookkeeping (candidate, tried) and attempt unless changed."""
        base: Dict[str, Any] = dict(candidate=self.candidate, tried=dict(self.tried), attempt=self.attempt)
        base.update(changes)
        return TaskState(**base)


def task_owner(state: TaskState, now: Optional[float] = None) -> Optional[str]:
    """Executor the stored state reserves the work for (None = anyone): `wait_executor`, or from a quota
    handoff's `reclaim_at` on, the original executor (`reclaim_executor`) again (cloud recovered)."""
    if state.reclaim_executor and state.reclaim_at is not None:
        if (time.time() if now is None else now) >= state.reclaim_at:
            return state.reclaim_executor
    return state.wait_executor


def _task_key(stage: str, track: str) -> str:
    return f"{stage}/{track}"


@dataclass
class Slice:
    index: int
    start: float
    end: float
    tasks: Dict[str, TaskState] = field(default_factory=dict)

    def get(self, stage: str, track: str = PRIMARY_TRACK) -> TaskState:
        """State of a work unit; a unit never touched is a fresh pending one (not stored)."""
        return self.tasks.get(_task_key(stage, track)) or TaskState()

    def set(self, stage: str, track: str, state: TaskState) -> None:
        self.tasks[_task_key(stage, track)] = state

    def is_done(self, stage: str, track: str = PRIMARY_TRACK) -> bool:
        return self.get(stage, track).status == DONE

    def all_done(self, track: str = PRIMARY_TRACK) -> bool:
        return all(self.is_done(stage, track) for stage in STAGES)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
                "tasks": {k: v.to_dict() for k, v in sorted(self.tasks.items())}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Slice":
        tasks = {str(k): TaskState.from_dict(v) for k, v in (data.get("tasks") or {}).items()}
        return cls(index=int(data["index"]), start=float(data["start"]), end=float(data["end"]), tasks=tasks)


@dataclass
class SlicePlan:
    slices: List[Slice] = field(default_factory=list)
    duration: float = 0.0
    media_id: str = ""
    tracks: List[str] = field(default_factory=lambda: [PRIMARY_TRACK])   # tracks[0] is the primary
    slice_seconds: float = DEFAULT_SLICE_SECONDS
    first_slice_seconds: Optional[float] = DEFAULT_FIRST_SLICE_SECONDS
    strategy: str = "smart_horizon"        # "smart_horizon" | "full_preload"
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES
    # Decided cuts: {"cuts": [0.0, ...], "closed": bool, "failures": int, "islands": [...]}. Slices exist only
    # between two decided cuts; decided cuts never move or disappear (T3.6: they can only be inserted).
    # `cuts` / `closed` are the main chain (from 0). Each island is another chain {"cuts": [start, ...],
    # "closed": bool} that began at a seek target in the uncut region; when a chain's last cut reaches another
    # chain's start the two merge (the main chain absorbs the island). `slices` are appended as they are
    # created, so the list is NOT in time order: `Slice.index` is the list position, use slices_by_time().
    lazy: Dict[str, Any] = field(default_factory=lambda: {"cuts": [0.0], "closed": False, "failures": 0,
                                                          "islands": []})
    # Transient (never saved): what apply_seek_cuts did, for the usage log (pop_seek_events).
    seek_events: List[Dict[str, Any]] = field(default_factory=list, repr=False, compare=False)

    @property
    def primary(self) -> str:
        return self.tracks[0]

    @property
    def open(self) -> bool:
        """True while later cuts are still undecided (more slices will appear): the main chain has not reached
        the end, or an island is not joined to it yet."""
        return not self.lazy.get("closed") or bool(self.lazy.get("islands"))

    # -- state transitions ------------------------------------------------------------------------
    def mark_running(self, index: int, stage: str, executor: str, track: str = PRIMARY_TRACK,
                     candidate: Optional[str] = None) -> None:
        """`candidate` (M4): id of the fallback candidate being run; moving to another one resets attempt."""
        old = self.slices[index].get(stage, track)
        if candidate is not None and candidate != old.candidate:
            new = old.carry(status=RUNNING, executor=executor, candidate=candidate,
                            attempt=old.attempt if old.candidate is None else 0)
        else:
            new = old.carry(status=RUNNING, executor=executor)
        self.slices[index].set(stage, track, new)

    def mark_done(self, index: int, stage: str, model_id: str = "", track: str = PRIMARY_TRACK, *,
                  candidate: Optional[str] = None, result_key: Optional[str] = None) -> None:
        """`candidate` / `result_key` (M4b): fallback candidate that produced the result and its cache key."""
        old = self.slices[index].get(stage, track)
        # Keep `tried` (and the candidate unless a new one is given) so the failure history stays visible.
        self.slices[index].set(stage, track, TaskState(status=DONE, model_id=model_id or None,
                                                       candidate=candidate or old.candidate,
                                                       tried=dict(old.tried),
                                                       result_key=result_key or None))

    def mark_failed(self, index: int, stage: str, kind: str, *, at: Optional[float] = None, attempt: int = 1,
                    model_id: str = "", detail: str = "", track: str = PRIMARY_TRACK) -> None:
        """Every candidate gave up (exhausted). Quota waits must use defer(), not this. Keeps `tried`."""
        old = self.slices[index].get(stage, track)
        self.slices[index].set(stage, track, TaskState(
            status=FAILED, fail_kind=kind, fail_at=time.time() if at is None else at, attempt=attempt,
            model_id=model_id or None, detail=detail, candidate=old.candidate, tried=dict(old.tried)))

    def defer(self, index: int, stage: str, retry_at: float, track: str = PRIMARY_TRACK,
              attempt: Optional[int] = None) -> None:
        """Quota wait: back to pending with a retry time. Not a failure. `attempt` (optional) records a counted
        retry of a transient error (network/server); a pure quota wait leaves it unchanged."""
        old = self.slices[index].get(stage, track)
        self.slices[index].set(stage, track, old.carry(
            status=PENDING, retry_at=retry_at, wait_executor=old.wait_executor,
            attempt=old.attempt if attempt is None else attempt))

    def wait_for(self, index: int, stage: str, executor: str, track: str = PRIMARY_TRACK) -> None:
        """Cross-executor fallback: the next candidate belongs to `executor`; only it may take the work."""
        old = self.slices[index].get(stage, track)
        self.slices[index].set(stage, track, old.carry(status=PENDING, wait_executor=executor))

    def release(self, index: int, stage: str, track: str = PRIMARY_TRACK) -> None:
        """running -> plain pending (executor stopped / crashed)."""
        self.slices[index].set(stage, track, self.slices[index].get(stage, track).carry(status=PENDING))

    def release_running(self, executor: Optional[str] = None) -> int:
        """Crash recovery on load: running work of `executor` (None = all) goes back to pending."""
        count = 0
        for s in self.slices:
            for key, state in list(s.tasks.items()):
                if state.status == RUNNING and (executor is None or state.executor == executor):
                    s.tasks[key] = state.carry(status=PENDING)
                    count += 1
        return count

    def reset_failed(self, index: Optional[int] = None, stage: Optional[str] = None,
                     track: Optional[str] = None) -> int:
        """Re-queue = clear `failed` (UI retry button). Returns how many units were cleared."""
        count = 0
        for s in self.slices:
            if index is not None and s.index != index:
                continue
            for key, state in list(s.tasks.items()):
                k_stage, _, k_track = key.partition("/")
                if state.status == FAILED and (stage is None or k_stage == stage) \
                        and (track is None or k_track == track):
                    s.tasks[key] = TaskState()
                    count += 1
        return count

    # -- serialization ----------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PLAN_VERSION,
            "duration": round(self.duration, 3),
            "media_id": self.media_id,
            "tracks": list(self.tracks),
            "slice_seconds": self.slice_seconds,
            "first_slice_seconds": self.first_slice_seconds,
            "strategy": self.strategy,
            "horizon_minutes": self.horizon_minutes,
            "lazy": self.lazy,
            "slices": [s.to_dict() for s in self.slices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SlicePlan":
        first = data.get("first_slice_seconds", DEFAULT_FIRST_SLICE_SECONDS)
        lazy = _load_lazy(data.get("lazy"))
        return cls(
            slices=[Slice.from_dict(s) for s in data.get("slices", [])],
            duration=float(data.get("duration", 0.0)),
            media_id=str(data.get("media_id", "")),
            tracks=[str(t) for t in (data.get("tracks") or [PRIMARY_TRACK])],
            slice_seconds=float(data.get("slice_seconds", DEFAULT_SLICE_SECONDS)),
            first_slice_seconds=None if first is None else float(first),
            strategy=str(data.get("strategy", "smart_horizon")),
            horizon_minutes=int(data.get("horizon_minutes", DEFAULT_HORIZON_MINUTES)),
            lazy=lazy,
        )


def _load_lazy(raw: Any) -> Dict[str, Any]:
    """`lazy` of a saved plan (version 1 has no islands). Copies the lists so the loaded plan owns them."""
    if not isinstance(raw, dict):
        return {"cuts": [0.0], "closed": False, "failures": 0, "islands": []}
    lazy = dict(raw)
    lazy["cuts"] = [float(c) for c in (raw.get("cuts") or [0.0])]
    islands = []
    for isl in raw.get("islands") or []:
        if isinstance(isl, dict) and isl.get("cuts"):
            islands.append({"cuts": [float(c) for c in isl["cuts"]], "closed": bool(isl.get("closed"))})
    lazy["islands"] = islands
    return lazy


def save_slice_plan(plan: SlicePlan, folder: Path | str) -> None:
    from oh_streamer.subtitles import runtime
    runtime.atomic_write_json(Path(folder) / SLICE_PLAN_FILE, plan.to_dict())


def load_slice_plan(folder: Path | str) -> Optional[SlicePlan]:
    path = Path(folder) / SLICE_PLAN_FILE
    if not path.is_file():
        return None
    try:
        return SlicePlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


# --------------------------------------------------------------------------------------------------
# Lazy cuts
# --------------------------------------------------------------------------------------------------

def new_slice_plan(duration: float, *, media_id: str = "", tracks: Optional[Sequence[str]] = None,
                   slice_seconds: float = DEFAULT_SLICE_SECONDS,
                   first_slice_seconds: Optional[float] = DEFAULT_FIRST_SLICE_SECONDS,
                   strategy: str = "smart_horizon", horizon_minutes: int = DEFAULT_HORIZON_MINUTES) -> SlicePlan:
    """Plan with no cut decided yet (see extend_lazy_cuts). first_slice_seconds None/0 = no short first slice."""
    return SlicePlan(duration=float(duration), media_id=media_id, tracks=list(tracks or [PRIMARY_TRACK]),
                     slice_seconds=float(slice_seconds),
                     first_slice_seconds=float(first_slice_seconds) if first_slice_seconds else None,
                     strategy=strategy, horizon_minutes=horizon_minutes)


SEARCH_STEP_SECONDS = 60.0          # each widening of the silence search adds this much on both sides
LONG_SILENCE_SECONDS = 10.0         # silences this long are only used when no shorter one exists (could be whispering)
RELAXED_PREFERRED_SECONDS = 0.5     # relaxed tier: pauses at least this long win over shorter (mid-sentence) ones

# T3.6 chains / seek cuts
JOIN_EXTRA_SECONDS = 120.0          # a chain closes onto the next chain's start when it is within slice_seconds + this
GAP_KEEP_SECONDS = 120.0            # otherwise its cut stays at least this far before that start
SEEK_START_BEHIND_SECONDS = 12.0    # a seek slice may start this much before the play position
SEEK_SNAP_SECONDS = 10.0            # a start this close to the decided cut on its left uses that cut
SEEK_JOIN_SECONDS = 75.0            # less than this left up to the next chain / the end: one slice, no short cut
SPLIT_MIN_INTO_SECONDS = 30.0       # a slice is only split when the position is this far into it
SPLIT_EDGE_SECONDS = 10.0           # split cuts keep this distance from the slice's ends
STREAM_GROW_BEHIND_SECONDS = 600.0          # streams: cut fronts starting earlier than play_pos - this do not grow
STREAM_GROW_AHEAD_MARGIN_SECONDS = 600.0    # ... or later than play_pos + horizon + this
_EPS = 1e-6


def stream_grow_window(play_pos: float, horizon_s: float = 600.0) -> Tuple[float, float]:
    """`near` argument of extend_lazy_cuts for a stream: only cut fronts (a chain's last cut) inside
    [play_pos - 600, play_pos + horizon + 600] grow, so a seek to minute 40 does not cut 0..40 minutes first."""
    return (float(play_pos) - STREAM_GROW_BEHIND_SECONDS,
            float(play_pos) + float(horizon_s) + STREAM_GROW_AHEAD_MARGIN_SECONDS)


# -- chains (main chain = plan.lazy itself, islands = plan.lazy["islands"]) ---------------------------

def _chains(plan: SlicePlan) -> List[Dict[str, Any]]:
    return [plan.lazy] + list(plan.lazy.get("islands") or [])


def _frontier(chain: Mapping[str, Any]) -> float:
    return float(chain["cuts"][-1])


def _chain_start(chain: Mapping[str, Any]) -> float:
    return float(chain["cuts"][0])


def _right_start(plan: SlicePlan, chain: Mapping[str, Any]) -> Optional[float]:
    """Start of the nearest other chain after this chain's front (None = none)."""
    front = _frontier(chain)
    starts = [_chain_start(c) for c in _chains(plan) if c is not chain and _chain_start(c) > front + _EPS]
    return min(starts) if starts else None


def _absorb_joined(plan: SlicePlan, chain: Dict[str, Any]) -> None:
    """When `chain`'s last cut is another chain's start, that chain's cuts continue this one (slices exist
    already; only the bookkeeping merges)."""
    islands = plan.lazy.get("islands")
    while islands:
        front = _frontier(chain)
        joined = next((y for y in islands if y is not chain and abs(_chain_start(y) - front) < _EPS), None)
        if joined is None:
            return
        chain["cuts"].extend(joined["cuts"][1:])
        chain["closed"] = bool(joined.get("closed"))
        islands[:] = [y for y in islands if y is not joined]


def _append_cut(plan: SlicePlan, chain: Dict[str, Any], cut: float, *, closed: bool = False,
                hard: bool = False) -> Slice:
    """Decide the next cut of `chain` and append the slice it closes (new index = list position)."""
    prev = _frontier(chain)
    cut = round(float(cut), 3)
    chain["cuts"].append(cut)
    if closed:
        chain["closed"] = True
    if hard:
        plan.lazy.setdefault("hard_cuts", []).append(cut)
    s = Slice(index=len(plan.slices), start=prev, end=cut)
    plan.slices.append(s)
    _absorb_joined(plan, chain)
    return s


def _growable_chains(plan: SlicePlan, near: Optional[Tuple[float, float]] = None,
                     focus: Optional[float] = None) -> List[Dict[str, Any]]:
    """Chains that may decide another cut: not closed, front inside `near` (lo, hi; None = anywhere). Order:
    fronts at or after `focus` first, nearest first (the play position's chain), then the ones behind it; no
    focus = ascending fronts."""
    chains = [c for c in _chains(plan) if not c.get("closed")]
    if near is not None:
        chains = [c for c in chains if near[0] <= _frontier(c) <= near[1]]
    if focus is None:
        return sorted(chains, key=_frontier)
    return sorted(chains, key=lambda c: (_frontier(c) < focus, abs(_frontier(c) - focus)))


def _cut_params(plan: SlicePlan, prev: float, slice_seconds: Optional[float]) -> Tuple[float, float, float, bool]:
    """(target, min_chunk, max_chunk, short) for the slice starting at `prev`. A file no longer than one normal
    slice stays a single slice; otherwise the first slice uses first_slice_seconds. A first slice of at most
    SHORT_SLICE_SECONDS is a "short" slice (T3.6): silence search in [target * 2/3, target * 3/2] (30 s:
    [20, 45]) only, never widened, else a hard cut at the target. Longer custom first slices keep the old
    rule (target, at least 30 s / half the target, widened up to max_chunk)."""
    normal = float(slice_seconds or plan.slice_seconds)
    max_chunk = max(600.0, normal * 2)
    if prev == 0.0 and plan.first_slice_seconds and plan.duration > normal:
        target = min(plan.first_slice_seconds, normal)
        if target <= SHORT_SLICE_SECONDS + _EPS:
            return (target, target * SHORT_SEARCH_LO_SECONDS / SHORT_SLICE_SECONDS,
                    target * SHORT_SEARCH_HI_SECONDS / SHORT_SLICE_SECONDS, True)
        return target, max(30.0, target * 0.5), max_chunk, False
    return normal, min(120.0, normal), max_chunk, False


def _search_range(plan: SlicePlan, prev: float, slice_seconds: Optional[float],
                  bound: Optional[float] = None) -> Tuple[float, float, float]:
    """(ideal, lo, hi): nominal cut and the widest range a silence may be searched in for the slice starting
    at `prev` ([prev + min_chunk, prev + max_chunk], never closer than 10 s to the end, and GAP_KEEP_SECONDS
    before the start of the next chain `bound`)."""
    target, min_chunk, max_chunk, _short = _cut_params(plan, prev, slice_seconds)
    hi = min(prev + max_chunk, plan.duration - 10.0)
    if bound is not None:
        hi = min(hi, bound - GAP_KEEP_SECONDS)
    return prev + target, prev + min_chunk, hi


def _search_window(ideal: float, lo: float, hi: float, radius: float) -> Tuple[float, float]:
    return max(lo, ideal - radius), min(hi, ideal + radius)


def _radius_after(searched: float) -> float:
    return cloud_chunks.SNAP_RADIUS_SECONDS if searched <= 0 else searched + SEARCH_STEP_SECONDS


def _memo_radius(plan: SlicePlan, prev: float) -> float:
    """Radius already searched without finding a silence for the cut after `prev` (0 = none yet)."""
    memo = plan.lazy.get("search") or {}
    return float(memo.get("radius", 0.0)) if memo.get("prev") == prev else 0.0


def _next_step(plan: SlicePlan, chain: Mapping[str, Any], slice_seconds: Optional[float]):
    """What the next cut of a chain needs: ("join", start of the next chain) when that start is within
    slice_seconds + 120 s (the gap closes with one slice, no detection); ("close", duration) when the rest of the
    file is one slice; else ("detect", (ideal, window start, window end)) (the window widens while no silence
    is found, see extend_lazy_cuts)."""
    prev = _frontier(chain)
    normal = float(slice_seconds or plan.slice_seconds)
    right = _right_start(plan, chain)
    # the join allowance scales down with short plans (60 s plan: +24 s, not +120 s -> never a 180 s slice);
    # keyed on the plan's own slice length so 30 s seek overrides keep the old behaviour; 300 s plans unchanged
    join_extra = JOIN_EXTRA_SECONDS * min(1.0, plan.slice_seconds / DEFAULT_SLICE_SECONDS)
    if right is not None and right - prev <= normal + join_extra:
        return "join", right
    if plan.duration - prev <= normal:
        return "close", plan.duration
    ideal, lo, hi = _search_range(plan, prev, slice_seconds, right)
    ws, we = _search_window(ideal, lo, hi, _radius_after(_memo_radius(plan, prev)))
    return "detect", (ideal, ws, we)


def next_lazy_window(plan: SlicePlan, slice_seconds: Optional[float] = None, chain: Optional[Mapping[str, Any]] = None):
    """Silence-search window (ideal, start, end) the next undecided cut of `chain` (default: the main chain) is
    waiting on (it widens while no silence is found, see extend_lazy_cuts), or None when that cut needs no
    detection (the rest of the file is the last slice, or the next chain's start is the cut)."""
    kind, data = _next_step(plan, plan.lazy if chain is None else chain, slice_seconds)
    return data if kind == "detect" else None


def lazy_cut_ready(plan: SlicePlan, limit: Optional[float], slice_seconds: Optional[float] = None,
                   near: Optional[Tuple[float, float]] = None) -> bool:
    """Can the next cut search of some chain make progress now (its current window is readable, or no
    detection is needed)? A window without silence widens, so a ready cut may still wait for more download."""
    if not plan.open:
        return False
    for chain in _growable_chains(plan, near):
        kind, data = _next_step(plan, chain, slice_seconds)
        if kind != "detect" or limit is None or data[2] <= limit + 0.5:
            return True
    return False


def _nearest_silence(silences: Sequence[Tuple[float, float]], ideal: float, ws: float,
                     we: float) -> Optional[float]:
    """Cut inside the silence nearest to `ideal` within [ws, we] (the ideal itself if it is silent; else the
    middle of the silence's part inside the window), or None."""
    best, best_dist = None, float("inf")
    for a, b in silences:
        if a <= ideal <= b and ws <= ideal <= we:
            return ideal
        if b >= ws and a <= we:
            mid = (max(a, ws) + min(b, we)) / 2.0
            dist = abs(mid - ideal)
            if dist < best_dist:
                best, best_dist = mid, dist
    return best


def _merge_close(silences: Sequence[Tuple[float, float]], gap: float = 0.05) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(silences):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def detect_window_silences_relaxed(source, start: float, end: float) -> List[Tuple[float, float]]:
    """Last-resort silence search (shorter, louder pauses: 0.25 s below -25 dB) used before a forced cut."""
    import tempfile
    from oh_streamer.subtitles import runtime
    with tempfile.TemporaryDirectory(prefix="silence-relaxed-") as tmp:
        wav = Path(tmp) / "window.wav"
        runtime.resample_window_to_wav(source, wav, start, end, pad=False)
        found = cloud_chunks.detect_silence_intervals(wav, min_silence_duration=0.25, silence_threshold_db=-25.0,
                                                      close_open_at=runtime.wav_duration(wav))
    return [(start + a, start + b) for a, b in found]


def _decide_cut(plan: SlicePlan, source, limit: Optional[float], prev: float, slice_seconds: Optional[float],
                detect: Callable[..., Any], relaxed: Optional[Callable[..., Any]], bound: Optional[float] = None):
    """One cut by the v2 rule (design 7.2): a cut only lands in a silence.

    1. silence (< 10 s) nearest to the nominal cut within ±30 s; none: the relaxed detector's pauses in the same
       window (adjacent ones merged, pauses >= 0.5 s preferred);
    2. none: widen the window by 60 s per side per step up to [prev + min_chunk, prev + max_chunk] (both tiers
       per step); a step whose window is not downloaded yet waits (returns "wait"); the searched radius is
       remembered;
    3. none in the whole range: a long silence (>= 10 s) in the range;
    4. none at all (continuous sound): the nominal cut, recorded in plan.lazy["hard_cuts"] (the ASR pre-roll
       and cue ownership rules still keep the sentence that crosses it whole).

    A short first slice (T3.6, first_slice_seconds <= 30) searches [prev + 20, prev + 45] once (steps 1) and
    otherwise cuts hard at prev + 30 (no widening, no long silence). `bound` = start of the next chain: the cut
    stays GAP_KEEP_SECONDS before it.

    Returns (cut, hard) or "wait". Exceptions from `detect` propagate (caller counts failures)."""
    ideal, lo, hi = _search_range(plan, prev, slice_seconds, bound)
    short = _cut_params(plan, prev, slice_seconds)[3]
    if hi <= lo:
        return ideal, True
    radius = _memo_radius(plan, prev)
    found_all: List[Tuple[float, float]] = []
    while True:
        radius = _radius_after(radius)
        ws, we = _search_window(ideal, lo, hi, radius)
        if limit is not None and we > limit + 0.5:
            return "wait"
        found = list(detect(source, ws, we))
        found_all += found
        cut = _nearest_silence([(a, b) for a, b in found if (b - a) < LONG_SILENCE_SECONDS], ideal, ws, we)
        if cut is None and relaxed is not None:
            # Continuous background sound (music, room noise) hides every -30 dB pause: the speech pauses are
            # still there a few dB higher. Try them in the same window before widening (T3: none at -30 dB in
            # 120-510 s, but a 0.7 s pause at 301.2 s, where blocks' VAD cut too).
            try:
                pauses = _merge_close([(a, b) for a, b in relaxed(source, ws, we) if (b - a) < LONG_SILENCE_SECONDS])
            except Exception:
                pauses = []
            longer = [(a, b) for a, b in pauses if (b - a) >= RELAXED_PREFERRED_SECONDS]
            cut = _nearest_silence(longer, ideal, ws, we) or _nearest_silence(pauses, ideal, ws, we)
        if cut is not None:
            plan.lazy.pop("search", None)
            return cut, False
        plan.lazy["search"] = {"prev": prev, "radius": radius}
        if ws <= lo and we >= hi:
            break
    plan.lazy.pop("search", None)
    if short:
        return ideal, True
    cut = _nearest_silence(found_all, ideal, lo, hi)            # a long silence (>= 10 s) in the whole range
    return (cut, False) if cut is not None else (ideal, True)


def _grow_chain(plan: SlicePlan, chain: Dict[str, Any], source, limit: Optional[float],
                slice_seconds: Optional[float], detect: Callable[..., Any],
                relaxed_detect: Optional[Callable[..., Any]]) -> Tuple[str, bool]:
    """Try to decide the next cut of one chain. Returns (outcome, plan changed): "cut" (decided and appended),
    "wait" (its window is not readable yet / widened), "fail" (detector error, retried on the next call)."""
    kind, data = _next_step(plan, chain, slice_seconds)
    prev = _frontier(chain)
    hard = False
    if kind == "close":
        cut, closed = data, True
    elif kind == "join":
        cut, closed = data, False
    else:
        if limit is not None and data[2] > limit + 0.5:
            return "wait", False
        memo_before = dict(plan.lazy.get("search") or {})
        try:
            decided = _decide_cut(plan, source, limit, prev, slice_seconds, detect, relaxed_detect,
                                  _right_start(plan, chain))
        except Exception:
            # The window is readable, so this is not "audio missing": retry a few times, then freeze the
            # nominal cut.
            plan.lazy["failures"] = int(plan.lazy.get("failures", 0)) + 1
            if plan.lazy["failures"] < 3:
                return "fail", True
            decided = (data[0], True)            # the nominal cut
        if decided == "wait":
            return "wait", (plan.lazy.get("search") or {}) != memo_before
        plan.lazy["failures"] = 0
        plan.lazy.pop("search", None)
        cut, hard = decided
        closed = False
    _append_cut(plan, chain, cut, closed=closed, hard=hard)
    return "cut", True


def extend_lazy_cuts(plan: SlicePlan, source, limit: Optional[float], *,
                     slice_seconds: Optional[float] = None,
                     detect: Optional[Callable[..., Any]] = None,
                     relaxed_detect: Optional[Callable[..., Any]] = None,
                     max_cuts: Optional[int] = None,
                     near: Optional[Tuple[float, float]] = None,
                     focus: Optional[float] = None) -> bool:
    """Decide every cut whose silence search can finish on the readable prefix (limit = readable seconds,
    None = whole file), freeze it and append the slice it closes (rule: _decide_cut). Returns True if the plan
    changed (caller saves it). `relaxed_detect` defaults to detect_window_silences_relaxed only when `detect` is
    the real detector. `max_cuts` (None = no cap) stops after that many cuts were decided, so a remote source
    (every silence search downloads audio) can dispatch the first slice before the next cut is searched.

    T3.6: a plan can have several chains (the main chain from 0 and islands, see SlicePlan.lazy); each call
    grows them one cut at a time. A chain whose next chain starts within slice_seconds + 120 s closes onto that
    start without detection (one slice of up to 7 minutes); otherwise its cut stays 120 s before that start.
    `near` = (lo, hi): only chains whose last cut lies in it grow (streams: stream_grow_window(play_pos));
    `focus` = play position: its chain (the nearest front at or after it) grows first."""
    if not plan.open or (max_cuts is not None and max_cuts <= 0):
        return False
    if detect is None:
        detect = cloud_chunks.detect_window_silences
        relaxed_detect = relaxed_detect or detect_window_silences_relaxed
    changed = False
    decided_cuts = 0
    while max_cuts is None or decided_cuts < max_cuts:
        outcome = None
        for chain in _growable_chains(plan, near, focus):
            outcome, touched = _grow_chain(plan, chain, source, limit, slice_seconds, detect, relaxed_detect)
            changed = changed or touched
            if outcome in ("cut", "fail"):
                break
        if outcome != "cut":
            break
        decided_cuts += 1
    return changed


def build_slice_plan(duration: float, silence_intervals: Optional[Sequence[Tuple[float, float]]] = None,
                     **kwargs: Any) -> SlicePlan:
    """Whole-file plan (limit=None): the same lazy-cut rule run to the end. silence_intervals are absolute
    (start, end) seconds. None = no silence information (synthetic plans / tests): every cut is the nominal
    time; an explicit list (even empty) applies the v2 silence-only rule."""
    plan = new_slice_plan(duration, **kwargs)
    if duration <= 0.0:
        plan.lazy["closed"] = True
        return plan
    if silence_intervals is None:
        def detect(_source, start, end):
            return [(start, end)]            # "silent everywhere": the nominal cut is always accepted
    else:
        intervals = list(silence_intervals)

        def detect(_source, start, end):
            return [(max(a, start), min(b, end)) for a, b in intervals if b > start and a < end]

    extend_lazy_cuts(plan, None, None, detect=detect)
    return plan


# --------------------------------------------------------------------------------------------------
# Seek cuts (T3.6): short slices at the play position after a jump
# --------------------------------------------------------------------------------------------------
#
# Rule: decided cuts never move and are never removed; a seek may only INSERT cuts, and only
#  (a) in a region no slice covers ("island": a new chain starts at the seek target, SHORT_SLICE_SECONDS long, and
#      then grows by the normal rule), or
#  (b) inside a slice nothing has started on ("split": the slice keeps its index and start, its end moves in to
#      the new cut; the pieces behind it are new slices appended to the list with the same task states).
# Three steps so the audio fetch stays outside the plan lock:
#  1. seek_cut_request(plan, play_pos)        -> CutRequest (which audio window to analyse), no I/O
#  2. the executor detects silences in request.window (strict, and relaxed when seek_needs_relaxed)
#  3. apply_seek_cuts(plan, request, ...)      -> under the lock: re-checks, applies, False = nothing changed

@dataclass(frozen=True)
class CutRequest:
    """Audio the executor must analyse before apply_seek_cuts. `window` is in absolute media seconds."""
    kind: str                                          # "island" | "split"
    position: float
    window: Tuple[float, float]
    slice_index: Optional[int] = None                  # split: the slice to split
    slice_range: Optional[Tuple[float, float]] = None  # split: its (start, end) when requested


@dataclass
class _SeekGeometry:
    kind: str
    position: float
    window: Tuple[float, float]
    ws: float                                          # the new start cut lies in [ws, we]
    we: float
    left: Optional[Dict[str, Any]] = None              # island: chain whose front is the decided cut left of P
    right: Optional[float] = None                      # island: start of the next chain after P
    slice: Optional[Slice] = None                      # split: the slice


@dataclass
class _SeekDecision:
    start: float                                       # island: its first cut; split: the first new cut
    start_hard: bool
    snapped: bool                                      # island start reused the decided cut on its left
    steps: List[Tuple[float, bool, bool]]              # following cuts: (cut, hard, closes the chain)
    needs_relaxed: bool


def _untouched(s: Slice) -> bool:
    """Nothing started on the slice: every stored work unit (any track, any stage) is plain pending. Waits
    (retry_at, wait_executor, reclaim) and the fallback history (tried) do not count as started."""
    return all(st.status == PENDING for st in s.tasks.values())


def _seek_geometry(plan: SlicePlan, position: float, limit: Optional[float] = None) -> Optional[_SeekGeometry]:
    P = float(position)
    if not (0.0 < P < plan.duration - 1e-3):
        return None
    hit = slice_containing(plan, P)
    if hit is not None:
        a, b = hit.start, hit.end
        if P - a <= SPLIT_MIN_INTO_SECONDS or not _untouched(hit):
            return None
        ws, we = max(a + SPLIT_EDGE_SECONDS, P - SEEK_START_BEHIND_SECONDS), min(P, b - SPLIT_EDGE_SECONDS)
        if we < ws:
            return None
        geo = _SeekGeometry("split", P, (ws, min(P + SHORT_SEARCH_HI_SECONDS, b)), ws, we, slice=hit)
    else:
        if not plan.open:
            return None
        chains = _chains(plan)
        lefts = [c for c in chains if _frontier(c) <= P + _EPS and not c.get("closed")]
        if not lefts:
            return None
        left = max(lefts, key=_frontier)
        rights = [_chain_start(c) for c in chains if _chain_start(c) > P]
        right = min(rights) if rights else None
        ws = max(P - SEEK_START_BEHIND_SECONDS, _frontier(left), 0.0)
        we = P if right is None else min(P, right - SEEK_SNAP_SECONDS)
        we = max(we, ws)
        geo = _SeekGeometry("island", P, (ws, min(max(P + SHORT_SEARCH_HI_SECONDS, we), plan.duration)), ws, we,
                            left=left, right=right)
    if limit is not None and geo.window[1] > limit + 0.5:
        return None
    return geo


def seek_cut_request(plan: SlicePlan, play_pos: float, *, limit: Optional[float] = None) -> Optional[CutRequest]:
    """Does the play position need new cuts? None when it is inside a slice that must stay as it is (finished,
    started, or less than 30 s in), past the end, or its audio is not readable yet (`limit` = readable seconds).

    - not inside any slice (an uncut region): an island request, window [P - 12, P + 45] (the start cut in
      [P - 12, P], the short cut up to 45 s after it);
    - inside a slice where every work unit of every track is still plain pending, and P is more than 30 s into
      it: a split request, window [P - 12, P + 45] clipped to the slice.
    No I/O, does not change the plan."""
    geo = _seek_geometry(plan, play_pos, limit)
    if geo is None:
        return None
    if geo.kind == "split":
        return CutRequest("split", geo.position, geo.window, geo.slice.index, (geo.slice.start, geo.slice.end))
    return CutRequest("island", geo.position, geo.window)


def _seek_pick(strict: Sequence[Tuple[float, float]], relaxed: Optional[Sequence[Tuple[float, float]]],
               ideal: float, ws: float, we: float, flags: Dict[str, bool]) -> Optional[float]:
    """Silence (< 10 s) nearest to `ideal` inside [ws, we]; none: the relaxed detector's pauses (adjacent ones
    merged, >= 0.5 s preferred). `relaxed` None = not detected: flags["needs_relaxed"] says it could matter."""
    if we < ws:
        return None
    cut = _nearest_silence([(a, b) for a, b in strict if (b - a) < LONG_SILENCE_SECONDS], ideal, ws, we)
    if cut is None:
        if relaxed is None:
            flags["needs_relaxed"] = True
            return None
        pauses = _merge_close([(a, b) for a, b in relaxed if (b - a) < LONG_SILENCE_SECONDS])
        longer = [(a, b) for a, b in pauses if (b - a) >= RELAXED_PREFERRED_SECONDS]
        cut = _nearest_silence(longer, ideal, ws, we) or _nearest_silence(pauses, ideal, ws, we)
    return cut


def _seek_decide(plan: SlicePlan, geo: _SeekGeometry, silences: Sequence[Tuple[float, float]],
                 relaxed: Optional[Sequence[Tuple[float, float]]]) -> _SeekDecision:
    flags = {"needs_relaxed": False}
    strict = [(float(a), float(b)) for a, b in (silences or [])]
    soft = None if relaxed is None else [(float(a), float(b)) for a, b in relaxed]
    P = geo.position

    def short_cut(start: float, floor: float = 0.0) -> Tuple[float, bool]:
        hi = min(start + SHORT_SEARCH_HI_SECONDS, plan.duration - 10.0)
        lo = max(start + SHORT_SEARCH_LO_SECONDS, floor)
        cut = _seek_pick(strict, soft, start + SHORT_SLICE_SECONDS, lo, hi, flags)
        return (start + SHORT_SLICE_SECONDS, True) if cut is None else (cut, False)

    if geo.kind == "split":
        a, b = geo.slice.start, geo.slice.end
        c = _seek_pick(strict, soft, geo.we, geo.ws, geo.we, flags)
        c_hard = c is None
        c = geo.we if c is None else c
        steps: List[Tuple[float, bool, bool]] = []
        if b - c > SEEK_JOIN_SECONDS:
            d, d_hard = short_cut(c)
            steps.append((d, d_hard, False))
        return _SeekDecision(c, c_hard, False, steps, flags["needs_relaxed"])

    front = _frontier(geo.left)
    if geo.we - front < SEEK_SNAP_SECONDS:
        c, c_hard, snapped = front, False, True
    else:
        c = _seek_pick(strict, soft, geo.we, geo.ws, geo.we, flags)
        c_hard = c is None
        c = geo.we if c is None else c
        snapped = c - front < SEEK_SNAP_SECONDS
        if snapped:
            c, c_hard = front, False
    if geo.right is not None and geo.right - c <= SEEK_JOIN_SECONDS:
        steps = [(geo.right, False, False)]
    elif plan.duration - c <= SEEK_JOIN_SECONDS:
        steps = [(plan.duration, False, True)]
    else:
        d, d_hard = short_cut(c, floor=P + 1.0)          # the short slice must contain the play position
        steps = [(d, d_hard, False)]
    return _SeekDecision(c, c_hard, snapped, steps, flags["needs_relaxed"])


def seek_needs_relaxed(plan: SlicePlan, request: CutRequest, silences: Sequence[Tuple[float, float]]) -> bool:
    """Would apply_seek_cuts use the relaxed detector for some cut of `request` (strict `silences` found none
    there)? The executor fetches the relaxed silences of request.window only then. Pure."""
    geo = _seek_geometry(plan, request.position)
    if geo is None or geo.kind != request.kind:
        return False
    return _seek_decide(plan, geo, silences, None).needs_relaxed


def _same_request(geo: _SeekGeometry, request: CutRequest) -> bool:
    if geo.kind != request.kind:
        return False
    if geo.kind == "split":
        s = geo.slice
        if s.index != request.slice_index or request.slice_range is None:
            return False
        if abs(s.start - request.slice_range[0]) > 1e-3 or abs(s.end - request.slice_range[1]) > 1e-3:
            return False
    # the silences cover request.window: the situation must not need anything outside it
    return geo.window[0] >= request.window[0] - 1e-3 and geo.window[1] <= request.window[1] + 1e-3


def apply_seek_cuts(plan: SlicePlan, request: CutRequest, silences: Sequence[Tuple[float, float]],
                    relaxed: Optional[Sequence[Tuple[float, float]]] = None) -> bool:
    """Apply a seek request with the silences (absolute (start, end) seconds) detected in request.window; the
    caller holds the plan lock and saves the plan on True. The conditions are checked again on the plan as it is
    now (the detection ran outside the lock); if the situation changed (the slice was claimed or split, a cut
    appeared, the window no longer covers what is needed) it returns False and does not touch the plan.
    `relaxed` = the relaxed detector's pauses of the same window, or None (not detected: hard cuts).

    island: a new chain (SlicePlan.lazy["islands"]) starts at c, the silence nearest to P in [P - 12, P] (none:
      the relaxed pauses, none: a hard cut at P; c within 10 s of the decided cut on its left reuses that cut and
      continues the left chain instead); its first slice is a short one by the short rule (silence in
      [c + 20, c + 45], else a hard cut at c + 30) or, when the next chain starts within 75 s / the file ends
      within 75 s, one slice up to there. Hard cuts go to lazy["hard_cuts"].
    split: slice s = [a, b] keeps its index and start and ends at c (same rule, inside [a + 10, b - 10]); when
      more than 75 s remain the short rule cuts d after c; the new slices [c, d] and [d, b] (or [c, b]) are
      appended with copies of s's task states.
    What was done is appended to plan.seek_events (pop_seek_events) for the usage log."""
    geo = _seek_geometry(plan, request.position)
    if geo is None or not _same_request(geo, request):
        return False
    decision = _seek_decide(plan, geo, silences, relaxed)
    hard_before = len(plan.lazy.get("hard_cuts") or [])
    event: Dict[str, Any] = {"kind": geo.kind, "position": round(geo.position, 3)}
    if geo.kind == "split":
        s = geo.slice
        old_range = [round(s.start, 3), round(s.end, 3)]
        appended = _apply_split(plan, s, decision)
        if appended is None:
            return False
        event.update(slice_index=s.index, old_range=old_range,
                     new_slices=[[x.index, x.start, x.end] for x in [s] + appended])
    else:
        chain = geo.left
        if not decision.snapped:
            chain = {"cuts": [round(decision.start, 3)], "closed": False}
            plan.lazy.setdefault("islands", []).append(chain)
            plan.lazy["islands"].sort(key=_chain_start)
            if decision.start_hard:
                plan.lazy.setdefault("hard_cuts", []).append(round(decision.start, 3))
        created = [_append_cut(plan, chain, cut, closed=closed, hard=hard)
                   for cut, hard, closed in decision.steps]
        event.update(start=round(decision.start, 3), snapped=decision.snapped,
                     new_slices=[[x.index, x.start, x.end] for x in created])
    event["hard_cuts"] = list((plan.lazy.get("hard_cuts") or [])[hard_before:])
    plan.seek_events.append(event)
    plan.lazy.pop("search", None)
    return True


def _apply_split(plan: SlicePlan, s: Slice, decision: _SeekDecision) -> Optional[List[Slice]]:
    """Insert the split cuts into the chain holding s, shorten s to the first one and append the new slices
    (copies of s's task states). Returns the new slices, None (nothing changed) if no chain has s's cuts."""
    a, b = s.start, s.end
    chain, at = None, -1
    for ch in _chains(plan):
        cuts = ch["cuts"]
        at = next((i for i in range(len(cuts) - 1)
                   if abs(cuts[i] - a) < _EPS and abs(cuts[i + 1] - b) < _EPS), -1)
        if at >= 0:
            chain = ch
            break
    if chain is None:
        return None
    new_cuts = [round(decision.start, 3)] + [round(cut, 3) for cut, _hard, _closed in decision.steps]
    chain["cuts"][at + 1:at + 1] = new_cuts
    hard = ([new_cuts[0]] if decision.start_hard else []) + [round(cut, 3) for cut, h, _c in decision.steps if h]
    if hard:
        plan.lazy.setdefault("hard_cuts", []).extend(hard)
    tasks = s.tasks
    s.end = new_cuts[0]
    appended = []
    for start, end in zip(new_cuts, new_cuts[1:] + [b]):
        appended.append(Slice(index=len(plan.slices), start=start, end=end, tasks=copy.deepcopy(tasks)))
        plan.slices.append(appended[-1])
    return appended


def pop_seek_events(plan: SlicePlan) -> List[Dict[str, Any]]:
    """What apply_seek_cuts did since the last call (for the usage log `slice_split`): {"kind", "position",
    "new_slices": [[index, start, end], ...], "hard_cuts", and for a split "slice_index", "old_range"}."""
    events, plan.seek_events = list(plan.seek_events), []
    return events


# --------------------------------------------------------------------------------------------------
# ASR pre-roll and cue ownership (design 7.2; shared by every executor)
# --------------------------------------------------------------------------------------------------

DEFAULT_PRE_ROLL_SECONDS = 15.0     # ASR audio starts this much before the slice (context for the first words)
DEFAULT_LEAD_KEEP_SECONDS = 1.0     # a slice keeps pre-roll cues ending within this much before its start
DEFAULT_TAIL_GUARD_SECONDS = 0.3    # a slice drops cues ending within this much before its end (maybe cut off)


@dataclass(frozen=True)
class AsrWindow:
    """How a slice's ASR audio is taken and which cues the slice owns. Part of the ASR cache key."""
    pre_roll_s: float = DEFAULT_PRE_ROLL_SECONDS
    lead_keep_s: float = DEFAULT_LEAD_KEEP_SECONDS
    tail_guard_s: float = DEFAULT_TAIL_GUARD_SECONDS
    context_grid_s: float = 0.0       # local VAD rescue must retain a stable whole-media time origin

    def cache_params(self) -> Dict[str, Any]:
        params = {"asr_window": "preroll-v1", "pre_roll_s": round(float(self.pre_roll_s), 3),
                  "lead_keep_s": round(float(self.lead_keep_s), 3), "tail_guard_s": round(float(self.tail_guard_s), 3)}
        if self.context_grid_s > 0:
            params.update(asr_window="preroll-grid-v2", context_grid_s=float(self.context_grid_s))
        return params

    def audio_start(self, s: Slice) -> float:
        start = max(0.0, s.start - max(0.0, float(self.pre_roll_s)))
        grid = float(self.context_grid_s)
        return math.floor(start / grid) * grid if math.isfinite(grid) and grid > 0 else start


def is_last_slice(plan: SlicePlan, s: Slice) -> bool:
    """No slice follows: this slice ends at the media end (slices are not in time order, T3.6, so the list
    position says nothing)."""
    return s.end >= plan.duration - 1e-3


def owns_cue(end: float, s_start: float, s_end: float, *, first: bool, last: bool,
             lead_keep_s: float = DEFAULT_LEAD_KEEP_SECONDS,
             tail_guard_s: float = DEFAULT_TAIL_GUARD_SECONDS, keep_preroll: bool = False) -> bool:
    """Ownership by the cue END time (absolute seconds), so every cue lands in exactly one place:

    - the slice keeps a cue iff  s_start - lead_keep < end <= s_end - tail_guard;
    - no lower bound for the first slice, no upper bound for the last one;
    - consequence at a cut c: cues ending at or before c - lead_keep (fully heard in the previous slice) are
      dropped from the pre-roll; a cue crossing c (or ending in its last tail_guard s, i.e. possibly cut off
      in the previous slice's audio) belongs to the later slice, whose pre-roll heard it whole;
    - cues ending in (c - lead_keep, c - tail_guard] may be kept by both slices when they were recognised in
      separate runs (publication deduplicates identical text); one ASR run over several contiguous slices
      passes lead_keep == tail_guard for the inner cuts, which makes the split exact.
    - keep_preroll (cloud executor): the lead_keep drop assumes the previous slice kept that cue, but the previous
      slice may have dropped it by its tail guard (cut-off at its audio end) while this slice's pre-roll heard it
      whole and split it so that it ends before c - lead_keep: lost. keep_preroll keeps such cues, and the
      publication (slice_results.resolve_preroll_overlaps) drops those the previous slice actually covers.
    """
    if not first and not keep_preroll and end <= s_start - lead_keep_s:
        return False
    if not last and end > s_end - tail_guard_s:
        return False
    return True


def own_segments(segments: Iterable[Mapping[str, Any]], s_start: float, s_end: float, *, first: bool, last: bool,
                 lead_keep_s: float = DEFAULT_LEAD_KEEP_SECONDS,
                 tail_guard_s: float = DEFAULT_TAIL_GUARD_SECONDS,
                 keep_preroll: bool = False) -> List[Dict[str, Any]]:
    """The segments (absolute times) a slice owns, see owns_cue."""
    return [dict(seg) for seg in segments
            if owns_cue(float(seg["end"]), s_start, s_end, first=first, last=last,
                        lead_keep_s=lead_keep_s, tail_guard_s=tail_guard_s, keep_preroll=keep_preroll)]


def slices_by_time(plan_or_slices: Any) -> List[Slice]:
    """The slices in time order (the list is in creation order since T3.6: islands and splits are appended)."""
    slices = getattr(plan_or_slices, "slices", plan_or_slices)
    return sorted(slices, key=lambda s: (s.start, s.end))


def slice_containing(plan_or_slices: Any, position: float) -> Optional[Slice]:
    """The slice containing a position at the plan's millisecond precision (end exclusive).
    Audio-offset subtraction can put an exact persisted cut a fraction below its start.
    Canonicalise the query as well, so seeking to that cut cannot create another island.
    """
    position = round(float(position), 3)
    for s in getattr(plan_or_slices, "slices", plan_or_slices):
        if s.start <= position < s.end:
            return s
    return None


def find_slice_index_for_position(slices: Sequence[Slice], position: float) -> Optional[int]:
    """Index of the slice containing position. Outside every slice: the last slice (by time) when the position
    is past the end of all of them, else the next slice after the position (a gap, or before the first one)."""
    if not slices:
        return None
    hit = slice_containing(slices, position)
    if hit is not None:
        return hit.index
    ordered = slices_by_time(slices)
    last = max(ordered, key=lambda s: (s.end, s.start))
    if position >= last.end:
        return last.index
    return next((s.index for s in ordered if s.start > position), ordered[-1].index)


def slices_from_position(plan: SlicePlan, position: float) -> List[Slice]:
    """Slices in time order starting at the one find_slice_index_for_position names (for loops that used
    `plan.slices[start:]` when the list was in time order)."""
    ordered = slices_by_time(plan)
    index = find_slice_index_for_position(plan.slices, position)
    if index is None:
        return []
    first = plan.slices[index]
    return ordered[next(i for i, s in enumerate(ordered) if s is first):]


# --------------------------------------------------------------------------------------------------
# Cache keys (4.3): content-addressed, no item id, no settings snapshot
# --------------------------------------------------------------------------------------------------

def _digest(payload: Any, length: int = 32) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def asr_cache_key(media_id: str, start: float, end: float, asr_identity: Mapping[str, Any],
                  params: Optional[Mapping[str, Any]] = None) -> str:
    """media_id + slice range (ms) + ASR model identity (fallback.identity format) + output-affecting params."""
    return _digest(["asr-v1", str(media_id), round(float(start), 3), round(float(end), 3),
                    dict(asr_identity), dict(params or {})])


def asr_result_digest(segments: Sequence[Any]) -> str:
    """Digest of an ASR result; the MT key depends on this, not on how the ASR was obtained."""
    return _digest(["asr-result-v1", list(segments)])


def mt_cache_key(asr_digest: str, mt_identity: Mapping[str, Any], prompt_version: str = "",
                 glossary_version: str = "", params: Optional[Mapping[str, Any]] = None) -> str:
    return _digest(["mt-v1", str(asr_digest), dict(mt_identity), str(prompt_version), str(glossary_version),
                    dict(params or {})])


def asr_cache_path(key: str) -> Path:
    """Relative to the item's subtitles folder."""
    return Path("cache") / "asr" / f"{key}.json"


def mt_cache_path(key: str) -> Path:
    return Path("cache") / "mt" / f"{key}.json"


# --------------------------------------------------------------------------------------------------
# Selector (4.4)
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Executor:
    """Who runs the work. `stages` limits what it can do (an iGPU translator: {"mt"}); `tracks` None = all."""
    name: str
    stages: frozenset = frozenset(STAGES)
    tracks: Optional[frozenset] = None


@dataclass(frozen=True)
class SchedulerContext:
    play_pos: float = 0.0
    restore_pos: float = 0.0
    watched: bool = False
    item_state: str = "processing"          # "done" = whole video is due
    strategy: Optional[str] = None          # overrides plan.strategy
    max_available: Optional[float] = None   # readable source seconds; None = whole file
    now: Optional[float] = None
    secondary_min_lead_s: float = DEFAULT_SECONDARY_MIN_LEAD_SECONDS
    # M4 dispatch (only used when select_next_task gets an executor). `owner(stage, track, state)` -> executor
    # name that should run the pending unit now, or None = anyone; it overrides the stored reservation (e.g.
    # fallback.owner_function: the current candidate chain decides). Without it the stored reservation
    # (task_owner) applies and units without one belong to `default_owner` (None = anyone, the M1 behaviour).
    owner: Optional[Callable[[str, str, "TaskState"], Optional[str]]] = None
    default_owner: Optional[str] = None


@dataclass(frozen=True)
class Task:
    slice_index: int
    stage: str
    track: str = PRIMARY_TRACK
    urgent: bool = False
    reason: str = ""


def _run_end(ordered: Sequence[Slice], position: float, track: str) -> Optional[float]:
    """End of the run of fully finished, contiguous slices (time order) that starts at the slice containing
    `position`; None when that slice is not finished. A position past the end of every slice counts as in the
    last one (the old behaviour); a position in an uncut gap or before the first slice is in no slice: None."""
    hit = slice_containing(ordered, position)
    if hit is None:
        if position < max(s.end for s in ordered):
            return None
        hit = ordered[-1]
    if not hit.all_done(track):
        return None
    end = hit.end
    at = next(i for i, s in enumerate(ordered) if s is hit)
    for s in ordered[at + 1:]:
        if s.start > end + 1e-3 or not s.all_done(track):
            break
        end = s.end
    return end


def covered_end(plan: SlicePlan, position: float, track: Optional[str] = None) -> float:
    """End of the run of fully finished slices that starts at the slice containing `position` (time order, a
    gap ends the run; == position when that slice is not finished or there is none)."""
    track = track or plan.primary
    if not plan.slices:
        return position
    end = _run_end(slices_by_time(plan), position, track)
    return position if end is None else end


def select_next_task(plan: SlicePlan, ctx: Optional[SchedulerContext] = None,
                     executor: Optional[Executor] = None) -> Optional[Task]:
    """Next work unit for `executor` (None = unfiltered), or None.

    Rules (unchanged from select_next_cloud_action): seek slice while watching -> first slice -> watching:
    low watermark 20 min / window 30 min ahead of play_pos; not watching: horizon cap
    max(window, restore_pos + 15 min); full_preload / done: everything. Download bound: an asr unit whose slice
    end lies beyond max_available is not selectable. Units that are running, failed, waiting for a retry time
    or owned by another executor are skipped (the selector falls through to the next rule). Ownership (M4):
    ctx.owner(stage, track, state) when given, else the stored reservation (task_owner: wait_executor /
    reclaim) or ctx.default_owner; None = any executor. With executor=None nothing is filtered by owner.
    Non-primary tracks only work on slices the primary track finished, and only once the primary lead from
    play_pos reaches ctx.secondary_min_lead_s (or the primary is complete).

    T3.6: `plan.slices` is not in time order (islands and splits are appended), so every "first / next / in
    order" below uses time order. The head slice is the one starting at 0 (it may not exist yet); while
    watching past its end it no longer jumps the queue (rule 2). A play position inside an uncut gap picks no
    seek slice (the executor cuts one first, see seek_cut_request).
    """
    if not plan or not plan.slices:
        return None
    ctx = ctx or SchedulerContext()
    now = time.time() if ctx.now is None else ctx.now
    slices = slices_by_time(plan)
    primary = plan.primary
    max_available = ctx.max_available
    head = next((s for s in slices if s.start <= _EPS), None)

    def readable(s: Slice) -> bool:
        return max_available is None or s.end <= max_available + 0.5

    def owner_of(stage: str, track: str, st: TaskState) -> Optional[str]:
        if ctx.owner is not None:
            return ctx.owner(stage, track, st)
        return task_owner(st, now) or ctx.default_owner

    def can(s: Slice, stage: str, track: str) -> bool:
        if executor is not None:
            if stage not in executor.stages or (executor.tracks is not None and track not in executor.tracks):
                return False
        st = s.get(stage, track)
        if st.status != PENDING:
            return False
        if STAGES.index(stage) > 0 and not s.is_done(STAGES[STAGES.index(stage) - 1], track):
            return False
        if st.retry_at is not None and st.retry_at > now:
            return False
        if executor is not None:
            owner = owner_of(stage, track, st)
            if owner and owner != executor.name:
                return False
        if stage == STAGE_ASR and not readable(s):
            return False
        return True

    def next_of(s: Slice, track: str) -> Optional[str]:
        """First unfinished stage of the slice, if it can be taken now."""
        for stage in STAGES:
            if not s.is_done(stage, track):
                return stage if can(s, stage, track) else None
        return None

    def task(s: Slice, stage: str, urgent: bool, reason: str, track: str = primary) -> Task:
        return Task(slice_index=s.index, stage=stage, track=track, urgent=urgent, reason=reason)

    def sweep(candidates: Iterable[Slice], reason: str) -> Optional[Task]:
        """All asr first, then all mt, over the given slices in order."""
        candidates = list(candidates)
        for stage in STAGES:
            for s in candidates:
                if can(s, stage, primary):
                    return task(s, stage, False, reason)
        return None

    def primary_pick() -> Optional[Task]:
        eff_strategy = ctx.strategy or plan.strategy or "smart_horizon"
        horizon_sec = float(plan.horizon_minutes * 60)

        # 1. seek jump
        if ctx.watched and ctx.play_pos > 0.0:
            s = slice_containing(slices, ctx.play_pos)
            if s is None and ctx.play_pos >= max(x.end for x in slices):
                s = max(slices, key=lambda x: (x.end, x.start))      # past the last slice: the last one
            if s is not None:
                stage = next_of(s, primary)
                if stage:
                    return task(s, stage, True, "seek_urgent")

        # 2. head slice (start == 0); not once the viewer is past it
        s0 = head
        if s0 is not None and not (ctx.watched and ctx.play_pos >= s0.end):
            if not s0.is_done(STAGE_ASR, primary):
                stage = next_of(s0, primary)
                if stage:
                    return task(s0, stage, True, "head_slice")
                if s0.get(STAGE_ASR, primary).status == PENDING and not readable(s0):
                    return None      # nothing else may run before the head slice's audio is downloaded
            elif not s0.is_done(STAGE_MT, primary):
                stage = next_of(s0, primary)
                if stage:
                    return task(s0, stage, True, "head_slice")

        # 3. everything due
        if eff_strategy == "full_preload" or ctx.item_state == "done":
            return sweep(slices, "preload")

        # 4. smart rolling horizon
        if ctx.watched:
            covered_ahead = max(0.0, covered_end(plan, ctx.play_pos, primary) - ctx.play_pos)
            if covered_ahead < min(DEFAULT_LOW_WATERMARK_SECONDS, horizon_sec):
                window = [s for s in slices if s.end > ctx.play_pos and s.start < ctx.play_pos + horizon_sec]
                window.sort(key=lambda s: abs(s.start - ctx.play_pos))
                return sweep(window, "horizon_buffer")
            return None
        cap = max(horizon_sec, ctx.restore_pos + UNWATCHED_AHEAD_SECONDS)
        return sweep([s for s in slices if s.start < cap], "horizon_buffer")

    picked = primary_pick()
    if picked is not None or len(plan.tracks) < 2:
        return picked
    if head is not None and not (ctx.watched and ctx.play_pos >= head.end) \
            and head.get(STAGE_ASR, primary).status == PENDING and not readable(head) \
            and not head.is_done(STAGE_ASR, primary):
        return None

    # 5. other tracks trail the primary (4.8)
    complete = (not plan.open) and all(s.all_done(primary) for s in slices)
    lead = covered_end(plan, ctx.play_pos, primary) - ctx.play_pos
    if not complete and lead < ctx.secondary_min_lead_s:
        return None
    start_idx = 0
    seek_idx = find_slice_index_for_position(plan.slices, ctx.play_pos)
    if seek_idx is not None:
        start_idx = next(i for i, s in enumerate(slices) if s is plan.slices[seek_idx])
    for track in plan.tracks[1:]:
        for s in slices[start_idx:]:
            if not s.all_done(primary):
                break      # the secondary never gets ahead of the primary
            stage = next_of(s, track)
            if stage:
                return task(s, stage, False, "secondary_track", track)
    return None


# --------------------------------------------------------------------------------------------------
# Item-level status (4.2): derived, never stored
# --------------------------------------------------------------------------------------------------

@dataclass
class FailureInfo:
    slice_index: int
    stage: str
    track: str
    model_id: Optional[str]
    kind: Optional[str]
    at: Optional[float]
    attempt: int
    detail: str
    # M4: last reason per candidate (design 4.7: the UI lists every candidate's last reason + a retry button)
    candidates: List["CandidateReason"] = field(default_factory=list)
    candidate: Optional[str] = None       # candidate the work was on when it gave up

    @property
    def exhausted(self) -> bool:
        return self.kind == "exhausted"


@dataclass
class CandidateReason:
    """Last recorded outcome of one fallback candidate for one work unit (TaskState.tried entry)."""
    id: str
    kind: Optional[str]
    at: Optional[float]
    detail: str = ""
    executor: Optional[str] = None
    attempt: int = 0
    retry_at: Optional[float] = None      # left for a quota wait: may take the work back from then on


def candidate_reasons(state: TaskState) -> List[CandidateReason]:
    """`state.tried` as CandidateReason rows, in the order they were recorded."""
    out = []
    for cid, rec in state.tried.items():
        at, retry_at = rec.get("at"), rec.get("retry_at")
        out.append(CandidateReason(id=str(cid), kind=rec.get("kind"), at=None if at is None else float(at),
                                   detail=str(rec.get("detail", "")), executor=rec.get("executor"),
                                   attempt=int(rec.get("attempt", 0) or 0),
                                   retry_at=None if retry_at is None else float(retry_at)))
    return out


@dataclass
class PlanStatus:
    slices_total: int = 0
    slices_done: int = 0               # primary track, all stages
    open: bool = True                  # more slices will appear (download still growing)
    complete: bool = False
    progress: float = 0.0              # finished seconds / duration (primary track)
    covered_s: float = 0.0             # playable up to here, continuous from play_pos
    covered_ahead_s: float = 0.0
    stuck: bool = False                # some primary work failed (exhausted)
    blocked_playback: bool = False     # the work playback needs next has failed
    failures: List[FailureInfo] = field(default_factory=list)
    blocking_slice: Optional[int] = None    # first unfinished primary slice from play_pos
    blocking_stage: Optional[str] = None
    running_executor: Optional[str] = None
    waiting_download: bool = False
    waiting_cut: bool = False          # the play position lies in a region no slice covers yet (cut pending, T3.6)
    waiting_quota: bool = False
    next_retry_at: Optional[float] = None
    waiting_executor: Optional[str] = None
    # M4: the blocking unit's fallback bookkeeping (e.g. "Groq 429, waiting for the local GPU")
    blocking_candidate: Optional[str] = None
    blocking_tried: List[CandidateReason] = field(default_factory=list)
    reclaim_at: Optional[float] = None      # the original executor may take the blocking unit back from then


def derive_status(plan: SlicePlan, play_pos: float = 0.0, now: Optional[float] = None,
                  max_available: Optional[float] = None) -> PlanStatus:
    now = time.time() if now is None else now
    status = PlanStatus(slices_total=len(plan.slices), open=plan.open)
    if not plan.slices:
        status.waiting_download = plan.open
        status.waiting_cut = plan.open and play_pos > 0.0
        return status
    primary = plan.primary
    done_seconds = 0.0
    for s in plan.slices:
        if s.all_done(primary):
            status.slices_done += 1
            done_seconds += s.end - s.start
        for key, st in s.tasks.items():
            if st.status == FAILED:
                stage, _, track = key.partition("/")
                status.failures.append(FailureInfo(s.index, stage, track, st.model_id, st.fail_kind,
                                                   st.fail_at, st.attempt, st.detail,
                                                   candidates=candidate_reasons(st), candidate=st.candidate))
    status.progress = min(1.0, done_seconds / plan.duration) if plan.duration > 0 else 0.0
    status.complete = (not plan.open) and status.slices_done == len(plan.slices)
    status.stuck = any(f.track == primary for f in status.failures)
    status.covered_s = covered_end(plan, play_pos, primary)
    status.covered_ahead_s = max(0.0, status.covered_s - play_pos)
    if status.complete:
        return status

    # Time order (T3.6). The run of contiguous slices from the play position; the first unfinished one blocks.
    ordered = slices_by_time(plan)
    here = slice_containing(ordered, play_pos)
    if here is None and play_pos < max(s.end for s in ordered):
        # in an uncut gap (or before the first slice): nothing to wait for but the cut
        status.waiting_cut = True
        return status
    if here is None:
        here = ordered[-1]                       # past the last slice: the old behaviour (the last one)
        status.waiting_cut = plan.open and play_pos < plan.duration - 1e-3
    block, edge = None, here.start
    for s in ordered[next(i for i, x in enumerate(ordered) if x is here):]:
        if s.start > edge + 1e-3:
            status.waiting_cut = True            # the run of slices ends in a gap
            return status
        if not s.all_done(primary):
            block = s
            break
        edge = s.end
    if block is None:
        # everything existing from here is finished; the next slice appears when the download grows
        status.waiting_download = plan.open
        return status
    stage = next(st for st in STAGES if not block.is_done(st, primary))
    state = block.get(stage, primary)
    status.blocking_slice, status.blocking_stage = block.index, stage
    status.blocking_candidate, status.blocking_tried = state.candidate, candidate_reasons(state)
    if state.status == FAILED:
        status.blocked_playback = True
    elif state.status == RUNNING:
        status.running_executor = state.executor
    else:
        if stage == STAGE_ASR and max_available is not None and block.end > max_available + 0.5:
            status.waiting_download = True
        if state.retry_at is not None and state.retry_at > now:
            status.waiting_quota = True
            status.next_retry_at = state.retry_at
        owner = task_owner(state, now)
        if owner:
            status.waiting_executor = owner
        if state.reclaim_at is not None and state.reclaim_at > now:
            status.reclaim_at = state.reclaim_at
    return status
