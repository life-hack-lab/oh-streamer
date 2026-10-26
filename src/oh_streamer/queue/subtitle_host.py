"""Queue side of the unified subtitle scheduler.

Two parts, neither wired into `QueueService` yet (T3 does that):

- `QueueSubtitleHost(CloudHost)`: what one `CloudSliceExecutor` needs from the queue for ONE item (source state,
  media id, scheduler context, cancellation, progress / event / publish -> the item's `subtitle` fields and the
  player). One instance per item and executor run.
- `SubtitleSchedulerMixin`: the queue-service half that decides which executor runs an item and when, starts and
  stops the executor threads, and re-opens the gate after a run ended (RunResult reason / wait_until).
  Cloud half only here; the local half (QueueGpuHost, `_subtitle_local` slot) is T2.

The small helpers `_subtitle_cloud_source_state` / `_key`, `_subtitle_playlist_durations`,
`_subtitle_publish_update`, `_subtitle_position`, `_subtitle_restore_lead` are COPIED from the old
the old block-based queue mixin (deleted in T8a; these are the surviving copies). Put `SubtitleSchedulerMixin` before
`BlockSubtitleMixin` in the QueueService bases so these copies win; they behave the same.

audio_offset convention (R3, pinned by tests): the queue's IPC clock (`_subtitle_clocks[id][0]`, `_subtitle_position`)
is SOURCE AUDIO time = container time = slice-plan time + audio_offset. The slice plan starts at audio 0, so every
position handed to the scheduler is `max(0, clock - audio_offset)`. `restore_lead` is measured against the raw clock
(segments.json is on the video timeline) and added on top of the corrected play position.
"""

from __future__ import annotations

import json
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles import slice_results as sr

CLOUD = ss.EXECUTOR_NAME            # "cloud"
LOCAL = ss.LOCAL_EXECUTOR_NAME      # "local"
STREAM_HORIZON_MINUTES = 10         # stream items: lead over play_pos, always (design D4); the one source of both numbers
STREAM_HORIZON_S = STREAM_HORIZON_MINUTES * 60.0   # the same lead in seconds, also the cut limit (T3.1-B)
STREAM_MAX_CHUNK_S = 600.0          # longest slice (slice_plan _cut_params max_chunk): cuts are decided this far past it
RUN_STAGE_CLOUD = "subtitle_cloud"  # _Run.stage of a cloud executor thread
RUN_STAGE_LOCAL = "subtitle_local"
CLOUD_SLOT = "_subtitle_cloud"                  # QueueService attribute holding the regular cloud executor's _Run
LOCAL_SLOT = "_subtitle_local"
STREAM_SLOT = "_subtitle_stream_cloud"          # T3.1-A: the dedicated cloud slot of the (one) stream item

# item.subtitle["status"] values for which no executor is started / no status is overwritten
NO_START_STATUSES = frozenset(("paused", "cancelled", "failed", "blocked", "skipped", "done", "mux_pending"))
KEEP_STATUSES = frozenset(("paused", "cancelled", "skipped", "failed", "blocked", "done"))
ACTIVE_RUN_STATUSES = frozenset(("queued", "waiting_gpu", "asr", "translate", "partial"))

RESTART_MIN_INTERVAL_S = 1.0        # after cancelled / source_unavailable / exception
IDLE_RESTART_MIN_INTERVAL_S = 5.0   # after idle without a real wait_until (work is the other executor's) or horizon


def is_stream_item(item) -> bool:
    return item is not None and getattr(item, "kind", "file") == "stream"


def _log(message: str) -> None:
    try:
        from .queue_service import log
        log(message)
    except Exception:
        pass


# --------------------------------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------------------------------

class QueueSubtitleHost(ss.CloudHost):
    """`CloudHost` of one queue item. `service` is the QueueService (or a fake with the same members, see
    m7_t1_interface.md); `run` is the `_Run` the executor thread is registered under (its `cancel` event stops
    the executor from outside: pause, remove, shutdown)."""

    def __init__(self, service: Any, item_id: str, run: Any = None, *, executor: str = CLOUD,
                 probe_offset: bool = True):
        self.service = service
        self.item_id = item_id
        self.run = run
        self.executor = executor
        self.probe_offset = probe_offset   # False (the tick's dispatch check): never ffprobe, stored offset or 0
        self._audio_offset: Optional[float] = None
        self._published_once = False

    # -- helpers ----------------------------------------------------------------------------------
    def _item(self):
        return self.service.store.get(self.item_id)

    def audio_offset(self, item=None) -> float:
        """Stored subtitle audio_offset, else probed once per host (ffprobe of the source); 0 without
        probe_offset (every publication stores the offset, so it is only missing before any slice is done,
        when the horizon cannot hold anything back yet)."""
        item = item or self._item()
        value = item.subtitle.get("audio_offset") if item is not None else None
        if value is None and not self.probe_offset:
            return 0.0
        if value is None:
            if self._audio_offset is None:
                try:
                    source = (item.source_path or item.output_path) if item is not None else None
                    self._audio_offset = float(self.service._subtitle_audio_offset(source)) if source else 0.0
                except Exception as exc:  # noqa: BLE001
                    from oh_streamer.sources import media_input
                    if isinstance(exc, media_input.SourceUnavailable):
                        return 0.0                 # session not live yet: do not remember a guess, probe again
                    self._audio_offset = 0.0
            value = self._audio_offset
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # -- CloudHost --------------------------------------------------------------------------------
    def source_state(self) -> Tuple[bool, Optional[float], Optional[float]]:
        item = self._item()
        if item is None:
            return False, 0.0, None
        return self.service._subtitle_cloud_source_state(item)

    def media_id(self) -> Optional[str]:
        item = self._item()
        key = self.service._subtitle_cloud_source_key(item) if item is not None else None
        if not key:
            return None
        from oh_streamer.sources import media_input
        return sr.rcache_media_id(key) if media_input.needs_runtime_url(item.source_path) else sr.hls_media_id(key)

    def cut_limit(self, limit: Optional[float]) -> Optional[float]:
        """T3.1-B: a stream item (remote source, every silence search downloads audio) only decides cuts up to
        play_pos + horizon + the longest slice (STREAM_HORIZON_S = the horizon_minutes=10 the stream executor is
        built with, STREAM_MAX_CHUNK_S = max_chunk). Other items: `limit` (unchanged). The download boundary
        (max_available) never uses this."""
        item = self._item()
        if not is_stream_item(item):
            return limit
        clock = float(self.service._subtitle_position(item))
        play_pos = max(0.0, clock - self.audio_offset(item))
        return play_pos + STREAM_HORIZON_S + STREAM_MAX_CHUNK_S

    def pace_cuts(self) -> bool:
        # A watch/want may arrive while a background queue item is detecting a window.
        # Yield after each cut so that a new seek can interrupt even a fully readable HLS source.
        return True

    def seek_pending(self) -> bool:
        item = self._item()
        if item is None or item.id not in self.service._watched_ids():
            return False
        _position, playing = self.service._subtitle_clocks.get(item.id, (0.0, False))
        want = getattr(self.service, "_subtitle_wants", {}).get(item.id)
        return bool(want is not None and not playing and self.service._clock() - want[1] < 1.5)

    def seek_position(self) -> Optional[float]:
        item = self._item()
        if item is None or item.id not in self.service._watched_ids() or self.seek_pending():
            return None
        position, _playing = self.service._subtitle_clocks.get(item.id, (0.0, False))
        return max(0.0, float(position) - self.audio_offset(item))

    def scheduler_context(self) -> sp.SchedulerContext:
        item = self._item()
        if item is None:
            return sp.SchedulerContext(item_state="done")
        _usable, limit, _total = self.service._subtitle_cloud_source_state(item)
        return self.context_for(item, limit)

    def context_for(self, item, limit: Optional[float]) -> sp.SchedulerContext:
        """scheduler_context of `item` with an already known source limit (the tick has it at hand)."""
        clock = float(self.service._subtitle_position(item))
        play_pos = max(0.0, clock - self.audio_offset(item))
        lead = self.service._subtitle_restore_lead(item, clock)
        return sp.SchedulerContext(
            play_pos=play_pos, restore_pos=play_pos + lead,
            watched=item.id in self.service._watched_ids(), item_state=item.state,
            strategy=("smart_horizon" if is_stream_item(item)            # D4: a stream item always leads by the horizon
                      else (item.settings or {}).get("subtitle_cloud_strategy") or None),
            max_available=limit)

    def check_cancel(self) -> None:
        if self.run is not None and self.run.cancel.is_set():
            raise ss.Cancelled()
        item = self._item()
        if item is None:
            raise ss.Cancelled()
        if self.run is not None and not self.service._run_is_current(self.run):
            raise ss.Cancelled()
        if self.service._subtitle_cloud_gate_reason(item):
            raise ss.Cancelled()

    def on_progress(self, info: Dict[str, Any]) -> None:
        """Unit start -> phase / detail (status stays; the UI state comes from subtitle.view); waiting_* -> phase waiting. Only while the
        item is in an active state, and only when something changed (no queue.json write per call)."""
        state = str(info.get("state") or "")
        item = self._item()
        if item is None or item.subtitle.get("status") not in ACTIVE_RUN_STATUSES:
            return
        fields: Dict[str, Any] = {}
        if state in ("asr", "mt", "qwen"):
            stage = "ASR" if state == "asr" else "MT"
            slice_no = info.get("slice")
            if isinstance(slice_no, (list, tuple)):
                slice_no = slice_no[0] if slice_no else None
            total = info.get("total_slices") or 0
            detail = f"雲端切片 {int(slice_no) + 1}/{total} ({stage})" if isinstance(slice_no, int) and total else ""
            fields["phase"] = "asr" if state == "asr" else "translation"
            if detail:
                fields["detail"] = detail
        elif state.startswith("waiting_"):
            fields["phase"] = "waiting"
            fields["detail"] = str(info.get("detail") or state)
        else:
            return
        if all(item.subtitle.get(k) == v for k, v in fields.items()):
            return
        # status in fields would clear phase/detail in _subtitle_update; re-assert them explicitly
        self.service._subtitle_update(self.item_id, **fields)

    def on_event(self, event: Dict[str, Any]) -> None:
        """Fallback event: only wakes the tick; the view derives its state from the slice table, so nothing is
        remembered and nothing is written into the item (the old `fallback` field is gone, §1.3)."""
        wake = getattr(self.service, "_wake", None)
        if wake is not None:
            wake.set()

    def publish(self, result: Dict[str, Any]) -> None:
        """A new combined SRT exists (per translated batch / slice / final). Writes the item's subtitle state,
        tells the player, and on the first real publication lets `_maybe_play_after_subtitles` start playback."""
        service = self.service
        item = self._item()
        if item is None:
            return
        complete = bool(result.get("complete"))
        current = item.subtitle.get("status")
        fields: Dict[str, Any] = dict(
            covered_s=result.get("covered_s", 0), block_srt=result.get("srt"),
            block_subtitles=result.get("subtitles"), audio_offset=self.audio_offset(item),
            refinement_pending=bool(result.get("refinement_pending", False)))
        if "covered_ranges" in result:
            fields["covered_ranges"] = result["covered_ranges"]
        if current not in KEEP_STATUSES:
            if complete:
                fields["status"] = "mux_pending" if item.state == "done" else "ready"
                fields["percent"] = 100
                if fields["status"] == "mux_pending":
                    fields["only"] = True
            else:
                fields["status"] = "partial"
        with service._lock:
            service._subtitle_update(self.item_id, **fields)
        service._subtitle_publish_update(self._item(), result)
        wake = getattr(service, "_wake", None)
        if wake is not None:
            wake.set()
        if not self._published_once and (result.get("cues") or result.get("done_indices")
                                         or (result.get("covered_s") or 0) > 0):
            self._published_once = True
            service._maybe_play_after_subtitles(self.item_id)


# --------------------------------------------------------------------------------------------------
# What is left to do for an executor
# --------------------------------------------------------------------------------------------------

def dispatchable_for(plan: sp.SlicePlan, ctx: sp.SchedulerContext, owners: Any, executor_name: str, *,
                     now: Optional[float] = None) -> Optional[sp.Task]:
    """The unit `executor_name` would take now under `ctx`, or None (`owners`: one `fallback.owner_function`
    result for all tracks, or {track: owner function}; a track without one is unscheduled).

    The same rule the executor launches by (ss.dispatchable_task: select_next_task with its owner rule), so the
    host never starts an executor for work it would not take: units owned by the other executor, waiting for a
    retry time, failed (the user's to retry), beyond the downloaded audio, and units the smart horizon holds
    back (watched: buffer ahead of play_pos >= low watermark; unwatched: beyond the horizon cap). The sample video A
    restart loop came from a plain count of pending units here that ignored the horizon."""
    return ss.dispatchable_task(plan, ctx, owners, sp.Executor(executor_name), now=now)


class FinishedRun:
    """What the tick remembers about the last ended executor thread of an item (gate for the next start)."""

    def __init__(self, result: Optional[ss.RunResult], error: Optional[BaseException], at: float):
        self.result = result
        self.error = error
        self.at = at

    @property
    def reason(self) -> str:
        if self.error is not None or self.result is None:
            return "error"
        return self.result.reason


# --------------------------------------------------------------------------------------------------
# Mixin
# --------------------------------------------------------------------------------------------------

class SubtitleSchedulerMixin:
    """Subtitle executor management for `QueueService` (cloud half; T2 adds the local half).

    State kept on the service: `_subtitle_cloud` (the `_Run` of the running cloud executor thread, one at a
    time), `_subtitle_stream_cloud` (T3.1-A: the same for the stream item, so a stream never waits for a queue
    item's executor and vice versa), `_subtitle_local` (other slot, read by `_subtitle_runs`), `_subtitle_results`
    ({item_id: {executor: FinishedRun}})."""

    # -- time / small state -----------------------------------------------------------------------
    def _subtitle_time(self) -> float:
        clock = getattr(self, "_subtitle_clock_fn", None)
        return clock() if callable(clock) else time.time()

    def _subtitle_results_for(self, item_id: str) -> Dict[str, FinishedRun]:
        results = getattr(self, "_subtitle_results", None)
        if results is None:
            results = self._subtitle_results = {}
        return results.setdefault(item_id, {})

    def _subtitle_executor_reset(self, item_id: str, executor: Optional[str] = None) -> None:
        """Forget how the last executor run of the item ended, so the tick re-evaluates it from scratch.
        T4 calls this on retry / restart / resume / resume_all / settings change; removal calls it too."""
        results = getattr(self, "_subtitle_results", None)
        if results and item_id in results:
            if executor is None:
                results.pop(item_id, None)
            else:
                results[item_id].pop(executor, None)
        wake = getattr(self, "_wake", None)
        if wake is not None:
            wake.set()

    # -- run slots --------------------------------------------------------------------------------
    def _subtitle_runs(self) -> List[Any]:
        """The subtitle executor runs that exist now: the regular cloud and local slots and the dedicated stream
        slot (T3.1-A). Every stop / shutdown / sleep / removal path goes through this list."""
        return [run for run in (getattr(self, CLOUD_SLOT, None), getattr(self, LOCAL_SLOT, None),
                                getattr(self, STREAM_SLOT, None)) if run is not None]

    def _subtitle_stop_executors(self, item_id: str, reason: str = "pause", *, wait: float = 0.0) -> bool:
        """Stop every subtitle executor thread of the item (pause / remove / cancel / shutdown). Sets the
        run's cancel (the host's check_cancel raises), stops the executor, kills a registered subprocess
        (`_request_stop`). With wait > 0, joins the threads that long. True when none is left running."""
        runs = [run for run in self._subtitle_runs() if run.item_id == item_id]
        for run in runs:
            self._request_stop(run, reason)
            executor = getattr(run, "executor", None)
            if executor is not None:
                try:
                    executor.stop()
                except Exception:
                    pass
        if wait > 0:
            deadline = time.monotonic() + wait
            for run in runs:
                thread = getattr(run, "thread", None)
                if thread is not None and thread is not threading.current_thread():
                    thread.join(max(0.0, deadline - time.monotonic()))
        return not any(getattr(run, "thread", None) is not None and run.thread.is_alive() for run in runs)

    # -- executor routing (frozen contract) -------------------------------------------------------
    def _subtitle_lock_settings(self, item):
        """The item with its subtitle settings locked for this processing run, or None when no runnable runtime
        plan can be derived. Gate (`subtitle_recovery.settings_decision`): not started or same content -> the fresh
        snapshot is stored (pending cleared); the user declined these settings (hold) -> locked settings stay, only
        the cloud strategy follows; else the locked settings stay and `subtitle.settings_pending` asks the user.
        A legacy item (no frozen contract) that already started keeps its stored snapshot as the locked one."""
        from oh_streamer.subtitles.configuration import runtime_plan
        try:
            if not runtime_plan(item.settings) and self._subtitle_started(item.id):
                locked = {**item.settings, "subtitle_runtime_contract": "cloud-first-1"}
                if runtime_plan(locked):
                    self.store.update(item.id, settings=locked)
                    item = self.store.get(item.id)
            gate = self._subtitle_gate(item)
            if gate is None:                                  # fresh settings cannot be derived: keep what is there
                return item if runtime_plan(item.settings) else None
            if gate[0] == "adopt" and not runtime_plan(gate[1]):     # fresh settings give no runnable plan
                return item if runtime_plan(item.settings) else None
            self._subtitle_apply_gate(item, gate)
        except Exception as exc:
            _log(f"subtitle settings lock failed for {item.id}: {exc!r}")
            return item if runtime_plan(item.settings) else None
        return self.store.get(item.id)

    def _subtitle_ledger_for(self, item) -> Tuple[Optional[Any], Optional[Callable[[dict], str]]]:
        """(Ledger, credentials callback) for cooldown lookups and executors; (None, None) without them."""
        ledger = getattr(self, "_subtitle_ledger_obj", None)
        if ledger is None:
            try:
                ledger = self._subtitle_ledger_obj = fb.Ledger()
            except Exception as exc:
                _log(f"subtitle ledger unavailable: {exc!r}")
                return None, None
        settings = item.settings

        def credentials(row: dict) -> str:
            from oh_streamer.subtitles import model_adapters
            return model_adapters._binding_credentials(row, settings)

        return ledger, credentials

    def _subtitle_owners(self, item, frozen: dict, *, now: float) -> Dict[str, Callable]:
        """{track: owner function} of the item now: primary from `frozen`, ready comparison tracks from their
        own frozen plans (design 4.7: the current chain decides who runs a unit)."""
        from oh_streamer.subtitles import configuration
        ledger, credentials = self._subtitle_ledger_for(item)
        cooldown = fb.ledger_cooldown(ledger, credentials) if ledger is not None else None
        owners = {sp.PRIMARY_TRACK: fb.owner_function(frozen, cooldown=cooldown, now=now)}
        if is_stream_item(item):                       # D3: a stream item runs the main subtitle only
            return owners
        try:
            plans = configuration.track_runtime_plans(item.settings)
        except Exception:
            plans = {}
        for track, plan in plans.items():
            if track != sp.PRIMARY_TRACK and plan and plan.get("ready", False):
                owners[track] = fb.owner_function(plan, cooldown=cooldown, now=now)
        return owners

    def _subtitle_plan_for(self, folder: Path) -> Optional[sp.SlicePlan]:
        """The item's slice plan: the live in-memory one while an executor holds it, else the file."""
        store = ss.PlanStore.peek(folder)
        if store is not None and store.plan is not None:
            return store.plan
        return sp.load_slice_plan(folder)

    def _subtitle_dispatchable(self, item, plan: sp.SlicePlan, owners: Dict[str, Callable], executor: str,
                               limit: Optional[float], now: float) -> Optional[sp.Task]:
        """The unit `executor` would take now (dispatchable_for) under the item's current scheduler context
        (play position, watched, restore lead, source limit). Evaluated on every tick while the slot is free,
        so playback advancing, a seek, a growing download or a task state change starts the executor again."""
        ctx = QueueSubtitleHost(self, item.id, probe_offset=False).context_for(item, limit)
        return dispatchable_for(plan, ctx, owners, executor, now=now)

    def _subtitle_cloud_gate_reason(self, item) -> str:
        """Why the cloud executor must not run for the item now ('' = may run). Cloud consent (only when the
        frozen plan uses cloud models), queue paused / stopping, perf check holding, item unavailable."""
        from oh_streamer.subtitles.combos import is_cloud_enabled
        from oh_streamer.subtitles.configuration import runtime_plan, runtime_uses_cloud
        if not item:
            return "item_removed"
        settings = getattr(item, "settings", None)
        needs_consent = runtime_uses_cloud(settings) if runtime_plan(settings) else True
        if needs_consent and not is_cloud_enabled(self.store.get_settings()):
            return "cloud_disabled"
        # D1: "pause all" is for the background queue; a stream item is being watched, so only shutdown stops it
        if (self._paused_all and not is_stream_item(item)) or self._stopping.is_set():
            return "queue_paused_or_stopping"
        perf = getattr(self, "perf", None)
        if perf is not None and getattr(perf, "holding", False):
            return "perf_holding"
        if item.state in ("paused", "failed"):
            return "item_unavailable"
        if item.subtitle.get("status") in ("paused", "cancelled", "failed", "blocked", "skipped"):
            return "item_unavailable"
        if item.id in self._tombstones or item.id in self._reprocessing:
            return "item_unavailable"
        return ""

    # -- the wake gate ----------------------------------------------------------------------------
    def _subtitle_gate_open(self, item_id: str, executor: str, now: float) -> bool:
        """May a new executor thread start, judging only by how the last one ended (design M7 plan 2.3)?

        complete / stuck / error: no (until `_subtitle_executor_reset`: retry, restart, resume, settings change).
        idle / quota with a real wait (wait_until later than the run's end: quota deadline, handoff reclaim):
        when it is reached. idle without one, or with a wait_until already past when the run ended (no real
        wait), and horizon: 5 s apart (the caller's dispatchable check says when work comes back; a stale
        wait_until never bypasses the interval, sample video A). cancelled / source_unavailable / anything else: 1 s."""
        finished = self._subtitle_results_for(item_id).get(executor)
        if finished is None:
            return True
        reason, result = finished.reason, finished.result
        if reason in ("complete", "stuck", "error"):
            return False
        if reason in ("idle", "quota") and result is not None and result.wait_until is not None \
                and result.wait_until > finished.at:
            return now >= result.wait_until
        interval = IDLE_RESTART_MIN_INTERVAL_S if reason in ("idle", "horizon") else RESTART_MIN_INTERVAL_S
        return now - finished.at >= interval

    # -- tick -------------------------------------------------------------------------------------
    def _subtitle_executor_tick(self) -> None:
        """Called from `QueueService._step`. Stops a running cloud executor whose conditions went away; with the
        slot free, starts one for the first item that has cloud work. Never blocks (the work is in threads)."""
        if self._subtitle_slot_check(CLOUD_SLOT):
            return
        now = self._subtitle_time()
        watched = self._watched_ids()
        items = sorted(self.store.items(), key=lambda it: it.id not in watched)    # stable: queue order after
        for item in items:
            if self._subtitle_cloud_wanted(item, now):
                self._subtitle_start_cloud(item)
                return

    def _subtitle_slot_check(self, slot: str) -> bool:
        """True when the slot holds a run (that run is stopped when its gate closed); False when it is free."""
        run = getattr(self, slot, None)
        if run is None:
            return False
        if run.cancel.is_set():
            return True
        item = self.store.get(run.item_id)
        if self._subtitle_cloud_gate_reason(item):
            self._request_stop(run, "auto")
            executor = getattr(run, "executor", None)
            if executor is not None:
                executor.stop()
        return True

    def _subtitle_stream_tick(self) -> None:
        """T3.1-A: the dedicated cloud slot of the stream item (one player, one stream item). Same rules as the
        regular cloud tick, on `store.stream_items()` only: stops a running executor whose gate closed; with the
        slot free, starts one for a stream item that has cloud work (audio ready, see `_subtitle_cloud_wanted`).
        The regular slot is not touched, so queue items keep their cloud subtitles during a stream (D2: the two
        share the one Ledger, nothing else)."""
        if self._subtitle_slot_check(STREAM_SLOT):
            return
        stream_items = getattr(self.store, "stream_items", None)
        if stream_items is None:
            return
        now = self._subtitle_time()
        for item in stream_items():
            if self._subtitle_cloud_wanted(item, now):
                self._subtitle_start_cloud(item, slot=STREAM_SLOT)
                return

    def _subtitle_cloud_wanted(self, item, now: float) -> bool:
        """Should a cloud executor thread start for the item now?"""
        settings = item.settings or {}
        if not settings.get("subtitles") or item.subtitle.get("status") in NO_START_STATUSES:
            return False
        if is_stream_item(item) and not self._subtitle_stream_audio_ready(item):
            return False
        if not self._subtitle_gate_open(item.id, CLOUD, now):
            return False
        if self._subtitle_cloud_gate_reason(item):
            return False
        source = item.source_path or item.output_path
        if not source or not Path(source).is_file():
            return False
        item = self._subtitle_lock_settings(item)
        if item is None:
            return False
        frozen = item.settings.get("subtitle_resolved") or {}
        if not frozen.get("ready", True):
            self._subtitle_unready(item, frozen)
            return False
        usable, limit, _total = self._subtitle_cloud_source_state(item)
        if not usable:
            return False
        folder = self.store.item_dir(item.id) / "subtitles"
        owners = self._subtitle_owners(item, frozen, now=now)
        plan = self._subtitle_plan_for(folder)
        if plan is None:
            # no slice table yet: whoever the first candidate of the first stage belongs to starts it
            return owners[sp.PRIMARY_TRACK](sp.STAGE_ASR, sp.PRIMARY_TRACK, sp.TaskState()) == CLOUD
        position = QueueSubtitleHost(self, item.id, probe_offset=False).seek_position()
        if (position is not None and sp.seek_cut_request(plan, position, limit=limit) is not None
                and owners[sp.PRIMARY_TRACK](sp.STAGE_ASR, plan.primary, sp.TaskState()) == CLOUD):
            return True
        if self._subtitle_dispatchable(item, plan, owners, CLOUD, limit, now) is not None:
            return True
        # all current slices done but the plan still grows with the download: the home executor follows it
        return bool(plan.open and owners[sp.PRIMARY_TRACK](sp.STAGE_ASR, plan.primary, sp.TaskState()) == CLOUD)

    def _subtitle_stream_audio_ready(self, item) -> bool:
        """A stream item's subtitles need its audio source (queue/stream_audio.py): `ready` starts, `probing` (or
        not attached yet) waits, `failed` fails the subtitles with the audio error. The failed status is in
        NO_START_STATUSES, so it is written once."""
        stream = getattr(item, "stream", None) or {}
        audio = stream.get("audio_status")
        if audio == "ready":
            return True
        if audio == "failed":
            self._subtitle_update(item.id, status="failed",
                                  error=str(stream.get("audio_error") or "") or "字幕音訊無法取得")
        return False

    def _subtitle_unready(self, item, frozen: dict) -> None:
        """A frozen plan that is not ready cannot run. Only 'cloud consent missing' waits quietly (the gate
        starts it once allowed); anything else fails the item with the reason."""
        errors = frozen.get("errors") or []
        if errors and all(e.get("code") == "cloud_disabled" for e in errors if isinstance(e, dict)):
            return
        codes = ", ".join(sorted({str(e.get("code")) for e in errors if isinstance(e, dict)})) or "unknown"
        self._subtitle_update(item.id, status="failed", error=f"字幕設定尚未就緒：{codes}")

    # -- start / thread ---------------------------------------------------------------------------
    def _subtitle_new_cloud_executor(self, item, host: QueueSubtitleHost):
        """Build the CloudSliceExecutor of the item (the seam tests replace with a fake)."""
        from oh_streamer.subtitles import configuration
        item_id = item.id
        folder = self.store.item_dir(item_id) / "subtitles"
        primary = (item.settings.get("subtitle_configs") or [{}])[0]
        japanese = bool(primary.get("japanese_track", item.settings.get("subtitle_japanese", False)))
        ledger, credentials = self._subtitle_ledger_for(item)
        memo: Dict[str, Any] = {}

        def current():
            it = self.store.get(item_id)
            return it.settings if it is not None else item.settings

        def plan_provider():
            return configuration.runtime_plan(current())

        stream = is_stream_item(item)

        def track_plans_provider():
            if stream:                                 # D3: main subtitle only, no comparison tracks
                return {}
            settings = current()
            key = json.dumps(configuration.comparison_rows(settings), sort_keys=True, default=str)
            if memo.get("key") != key:
                memo["key"], memo["plans"] = key, configuration.track_runtime_plans(settings)
            return dict(memo["plans"])

        extra: Dict[str, Any] = {}
        if stream:                                     # D4: always the horizon, whatever the setting says
            extra = dict(strategy="smart_horizon", horizon_minutes=STREAM_HORIZON_MINUTES)
        return ss.CloudSliceExecutor(
            folder, item.source_path or item.output_path, host=host, settings=item.settings, config=primary,
            japanese=japanese, plan_provider=plan_provider, track_plans_provider=track_plans_provider,
            ledger=ledger, credentials=credentials, **extra)

    def _subtitle_start_cloud(self, item, slot: str = CLOUD_SLOT) -> None:
        from .queue_service import _Run
        with self._lock:
            if getattr(self, slot, None) is not None:
                return
            run = _Run(item.id, self._generations.get(item.id, 0))
            run.stage = RUN_STAGE_CLOUD
            run.subtitle_only = True
            setattr(self, slot, run)
        try:
            host = QueueSubtitleHost(self, item.id, run, executor=CLOUD)
            run.executor = self._subtitle_new_cloud_executor(self.store.get(item.id) or item, host)
        except Exception as exc:
            _log(f"subtitle cloud executor could not start for {item.id}: {exc!r}")
            with self._lock:
                if getattr(self, slot, None) is run:
                    setattr(self, slot, None)
            self._subtitle_results_for(item.id)[CLOUD] = FinishedRun(None, exc, self._subtitle_time())
            self._subtitle_update(item.id, status="failed", error=str(exc)[-500:])
            return
        run.thread = threading.Thread(target=self._run_subtitle_executor, args=(item.id, run, CLOUD, slot),
                                      name=f"subtitle-{'stream' if slot == STREAM_SLOT else 'cloud'}-{item.id}",
                                      daemon=True)
        run.thread.start()

    def _run_subtitle_executor(self, item_id: str, run: Any, name: str, slot: Optional[str] = None) -> None:
        """Thread body: `executor.run(wait=True)` (quota / download / handoff waits sleep in here), then record
        how it ended, free the slot (`slot`: the service attribute holding `run`; default by executor name),
        wake the tick."""
        result: Optional[ss.RunResult] = None
        error: Optional[BaseException] = None
        try:
            result = run.executor.run(wait=True)
        except BaseException as exc:                      # noqa: BLE001 - record, never kill the thread silently
            error = exc
            _log(f"subtitle {name} executor failed for {item_id}: {exc!r}")
        finally:
            slot = slot or (CLOUD_SLOT if name == CLOUD else LOCAL_SLOT)
            with self._lock:
                if getattr(self, slot, None) is run:
                    setattr(self, slot, None)
                if item_id not in self._tombstones:
                    if error is None and result is not None and result.reason == "refresh":
                        # soft drain for new settings: not a finished run, so the gate lets the tick relaunch it at once
                        self._subtitle_results_for(item_id).pop(name, None)
                    else:
                        self._subtitle_results_for(item_id)[name] = FinishedRun(result, error, self._subtitle_time())
            try:
                self._subtitle_after_run(item_id, run, result, error)
            finally:
                self._wake.set()

    def _subtitle_after_run(self, item_id: str, run: Any, result: Optional[ss.RunResult],
                            error: Optional[BaseException]) -> None:
        """Leave the item in a resting status: an exception fails it; any other end (cancel, idle, stuck, source
        unavailable) turns a transient 'asr' / 'translate' into partial / queued. complete was published by the
        host already."""
        if item_id in self._tombstones or not self._run_is_current(run):
            return
        item = self.store.get(item_id)
        if item is None:
            return
        status = item.subtitle.get("status")
        if error is not None and not isinstance(error, ss.Cancelled):
            if status not in KEEP_STATUSES:
                self._subtitle_update(item_id, status="failed", error=str(error)[-500:])
            return
        if result is not None and result.reason in ("complete", "refresh"):
            return          # refresh: the tick relaunches with the new settings; the status stays as it is
        if status in ACTIVE_RUN_STATUSES and status != "partial":
            self._subtitle_update(item_id, status="partial" if item.subtitle.get("covered_s") else "queued")

    # -- formerly in the old block-based queue mixin (deleted in T8a) --------------------------
    def _subtitle_position(self, item):
        return getattr(self, "_subtitle_clocks", {}).get(item.id, (0.0, False))[0]

    def _subtitle_restore_lead(self, item, pos):
        if item.state == "done":
            return float("inf")
        try:
            raw = json.loads((self.store.item_dir(item.id) / "work" / "segments.json").read_text("utf-8"))
            play = raw.get("play") or {}
            length = float(raw["n"]) / float(Fraction(play.get("fps", raw["key"]["fps"])))
            origin = float(play.get("video_start", 0))
            k = max(0, int((pos - origin) / length))
            ends = {int(s["k"]): origin + int(s["k"]) * length + int(s["frames"]) / float(Fraction(play.get("fps", raw["key"]["fps"]))) for s in raw["segments"]}
            end = pos
            while k in ends:
                end = ends[k]
                k += 1
            return max(0, end - pos)
        except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError):
            return 0.0

    def _subtitle_publish_update(self, item, result):
        if item is None:
            return
        tracks = result.get("subtitles") or []
        srt = [t["path"] for t in tracks] if tracks else result.get("srt") or []
        if isinstance(srt, dict):
            # slice_results publishes a language -> file map; the IPC accepts paths/descriptors.
            srt = list(dict.fromkeys(p for p in (srt.get("zh-TW") or srt.get("full"), srt.get("ja")) if p))
        if not tracks:
            paths = [srt] if isinstance(srt, str) else srt
            tracks = [dict(path=p, lang="chi" if i == 0 else "jpn", title="繁體中文（AI）" if i == 0 else "日文（AI）", default=i == 0) for i, p in enumerate(paths)]
        if not srt:
            return
        offset = getattr(self, "_subtitle_play_offsets", {}).get(item.id, item.subtitle.get("audio_offset", 0))
        signature = json.dumps({key: result.get(key) for key in
                                ("plan_key", "done_indices", "covered_s", "covered_ranges", "complete", "srt")}, sort_keys=True) + str(offset)
        sent = getattr(self, "_subtitle_sent_updates", {})
        if sent.get(item.id) == signature:
            return
        try:
            reply = self._send_to_player(dict(cmd="subtitle_update", item_id=item.id, srt=srt,
                                     subtitles=tracks, covered_s=result["covered_s"], complete=result["complete"],
                                     **({"covered_ranges": result["covered_ranges"]} if "covered_ranges" in result else {}),
                                     subtitle_offset=offset))
            if isinstance(reply, dict) and reply.get("ok"):
                sent[item.id] = signature
                self._subtitle_sent_updates = sent
        except Exception as exc:
            _log(f"subtitle update could not reach player: {exc!r}")

    def _subtitle_cloud_source_state(self, item):
        """What a cloud subtitle runner may read: (usable, limit_s, total_s).

        limit_s is None once the whole source is on disk. While an HLS source is still downloading it is the
        end time of the contiguous downloaded prefix minus one segment of margin (ffmpeg may touch the next
        segment when a cut falls on a segment boundary), so a slice is only audio-extracted from ready data.
        total_s is the manifest duration (the local m3u8 lists every segment up front)."""
        status = item.source_status
        from oh_streamer.sources import media_input
        if media_input.needs_runtime_url(item.source_path) and status in ("downloading", "complete"):
            return self._subtitle_range_source_state(item)
        if status in ("", "complete"):
            return True, None, None
        if status != "downloading" or not item.source_path:
            return False, 0.0, None
        handle = self._session_for(item, reopen_complete=False)
        if handle is None:
            return False, 0.0, None
        try:
            snap = handle.snapshot()
            durations = self._subtitle_playlist_durations(item.source_path)
        except Exception:
            return False, 0.0, None
        if not isinstance(snap, dict) or not durations:
            return False, 0.0, None
        total = float(sum(durations))
        ready = {x.get("index") for x in (snap.get("ready_segments") or []) if isinstance(x, dict)}
        if snap.get("status") == "complete" and len(ready) >= len(durations):
            return True, None, total
        prefix = 0
        while prefix < len(durations) and prefix in ready:
            prefix += 1
        limit = float(sum(durations[:max(0, prefix - 1)]))
        return True, limit, total

    def _subtitle_range_source_state(self, item):
        """`_subtitle_cloud_source_state` of a range-cache (.rcache) item: (usable, limit_s, total_s).

        The session must be live: its loopback server is the only way to read the data. A complete cache whose
        session was closed (queue restart, recycled after playback) is reopened here; when that is impossible the
        source is unavailable (the executor ends as source_unavailable and the tick retries), never an error.

        limit_s semantics. Unlike HLS, an unready block does not fail a read: the proxy moves it to the front of
        the fetch queue and blocks (up to the ffmpeg rw_timeout) until it arrives. The limit therefore keeps cuts
        and audio extraction off data that would stall the worker on the network, not off data that would break.
        It is the duration of the contiguous ready PREFIX of blocks (prefix bytes / total bytes x video duration),
        minus one block of margin (a read may touch the next block). Chosen over ready-blocks / total blocks
        because that ratio is inflated by out-of-order blocks (moov at the tail, a seek jump, holes filled last)
        and would admit audio from blocks that are not there yet; the prefix is a safe lower bound, and a byte
        position maps to a time position only approximately (bitrate varies), which the margin and a
        conservative floor absorb. total_s is the video-stream duration the download stored in the metadata."""
        handle = self._session_for(item, reopen_complete=True)
        if handle is None:
            return False, 0.0, None
        try:
            snap = handle.snapshot()
        except Exception:  # noqa: BLE001
            return False, 0.0, None
        if not isinstance(snap, dict):
            return False, 0.0, None
        status = snap.get("status")
        if status not in ("downloading", "complete"):
            return False, 0.0, None
        try:
            total = float(((snap.get("metadata") or {}).get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            total = 0.0
        total = total if total > 0 else None
        count = int(snap.get("segment_count") or 0)
        ready = {x.get("index") for x in (snap.get("ready_segments") or []) if isinstance(x, dict)}
        if status == "complete" and len(ready) >= count:
            return True, None, total
        if total is None:
            return False, 0.0, None                  # nothing to scale the block count by yet
        block = int(snap.get("block_size") or 0)
        size = int(snap.get("total_bytes") or 0)
        if count <= 0 or block <= 0 or size <= 0:
            return False, 0.0, None
        prefix = 0
        while prefix < count and prefix in ready:
            prefix += 1
        ready_bytes = min(size, max(0, prefix - 1) * block)
        return True, total * ready_bytes / size, total

    @staticmethod
    def _subtitle_cloud_source_key(item):
        """Stable media id of an HLS / range cache source (same while downloading, after pause/resume and when
        complete): the sidecar identity."""
        path = item.source_path
        if item.source_status not in ("downloading", "complete", "paused") or not path:
            return None
        try:
            from oh_streamer.sources import hls_cache, media_input, range_cache
            if media_input.needs_runtime_url(path):
                data = range_cache._load_sidecar(range_cache.sidecar_path(path))
                return str(data["identity"]) if data and data.get("identity") else None
            data = hls_cache._load_sidecar(hls_cache.sidecar_path(path))
            return str(data["identity"]) if data and data.get("identity") else None
        except Exception:
            return None

    @staticmethod
    def _subtitle_playlist_durations(path):
        durations = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#EXTINF:"):
                    durations.append(float(line[8:].split(",", 1)[0]))
        return durations
