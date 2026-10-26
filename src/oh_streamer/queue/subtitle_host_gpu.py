"""Queue side of the LOCAL subtitle executor.

- `QueueGpuHost(QueueSubtitleHost, GpuHost)`: what one `LocalSliceExecutor` needs from the queue for ONE item:
  the GPU lease (the restoration worker's `_wk` slot), stopping / resuming restoration for an interleave turn
  (with `_subtitle_turn_hold`, so the tick cannot restart restoration in the hand-over gap), restoration progress,
  the GPU identity, and the registry of model subprocesses (ASR child, llama-server) that cancel kills (R1).
- `GpuSubtitleSchedulerMixin(SubtitleSchedulerMixin)`: adds the local half to T1's cloud half: the
  `_subtitle_local` slot, `_subtitle_local_tick()` (start only when the item has local work; stop on gates), the
  resource mode setting. QueueService uses THIS mixin in its bases (in place of SubtitleSchedulerMixin).

Threads: one local executor thread per item at a time (global `_subtitle_local` slot, like the cloud slot), running
`LocalSliceExecutor.run(wait=True)`; GPU turns, restore turns, GPU-busy waits sleep in that thread. The tick only
starts / stops threads. Contract for QueueService (T3): see m7_t2_interface.md in the side-session notes.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

from oh_streamer.queue import gpu_turns as gt
from oh_streamer.queue import subtitle_local as sl
from oh_streamer.queue import subtitle_scheduler as ss
from oh_streamer.queue.subtitle_host import (
    LOCAL, NO_START_STATUSES, RUN_STAGE_LOCAL, STREAM_HORIZON_S, FinishedRun, QueueSubtitleHost,
    SubtitleSchedulerMixin, _log, is_stream_item)
from oh_streamer.subtitles import slice_plan as sp

RUN_STAGE_LEASE = "subtitle_lease"        # _Run.stage of the exclusive GPU lease held in `_wk`
RESOURCE_MODE_KEY = "subtitle_local_resource_mode"   # setting: auto (default) / interleave / parallel
STOP_RESTORE_TIMEOUT_S = 30.0             # how long stop_restore waits for the worker to leave `_wk`
RESTORE_POLL_S = 0.1                      # stream host: poll of the player's live report while it pauses
RESTORE_NEEDED_COOLDOWN_S = 10.0          # after restore_needed (seek / buffer_low): no new pause request for this long
RESTORE_TIMEOUT_COOLDOWN_S = 60.0         # ... after the player gave up a too long pause (reason "timeout")
RATE_WINDOW_S = 20.0                      # restoration rate = frames over the last ~20 s / source fps
RATE_MIN_SPAN_S = 5.0
WINDOW_KEEP_S = 86400.0                   # _subtitle_windows kept one day (perf observations)
UNAVAILABLE_SUBTITLE = frozenset(("paused", "cancelled", "failed", "blocked", "skipped"))
STREAM_LOW_WATERMARK_S = STREAM_HORIZON_S  # stream items: subtitles from a 10 min lead down (= the stream horizon)
STREAM_SUBTITLE_LOAD_S = 20.0             # stream default of TurnConfig.subtitle_load_s (measured: ASR ~12-15 s + warm MT ~5 s)
STREAM_RESTORE_RESTART_S = 10.0           # stream default of TurnConfig.restore_restart_s (resume 0.3-0.9 s + ramp-up)
RESTART_WATCH_S = 60.0                    # how long the host watches for restoration output after a resume
RESTART_POLL_S = 0.25


def stream_turn_config(settings: Optional[dict] = None) -> gt.TurnConfig:
    """TurnConfig of a stream item's local executor. Order (later wins): TurnConfig defaults, then the stream
    default low_watermark_s = STREAM_LOW_WATERMARK_S (600; the live buffer is only ~11-14 min), then the user's
    explicit `subtitle_local_<field>` settings (TurnConfig.from_settings). An unset setting never touches the
    stream default."""
    cfg = dataclasses.replace(gt.TurnConfig.from_settings(settings), media_turns=True)   # turns sized in media seconds
    # shorter load / restart than the file-item defaults (measured on stream turns); the executor and the T4.3
    # verdict replace them with the values learned per card
    for name, value in (("low_watermark_s", STREAM_LOW_WATERMARK_S), ("subtitle_load_s", STREAM_SUBTITLE_LOAD_S),
                        ("restore_restart_s", STREAM_RESTORE_RESTART_S)):
        if (settings or {}).get(f"subtitle_local_{name}") is None:
            cfg = dataclasses.replace(cfg, **{name: value})
    return cfg


def _kill(proc: Any) -> None:
    """Kill one registered model process and its children (queue_service.kill_tree; the seam tests patch)."""
    from oh_streamer.queue.queue_service import kill_tree
    try:
        if proc.poll() is None:
            kill_tree(proc.pid)
    except Exception as exc:                          # noqa: BLE001 - best effort, never raise from a cancel
        _log(f"subtitle local process kill failed: {exc!r}")


class _SharedLease:
    """Parallel-mode lease: restoration keeps `_wk`; the subtitle work only shares the GPU."""

    def __init__(self, item_id: str):
        self.item_id = item_id
        self.cancel = threading.Event()


# --------------------------------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------------------------------

class QueueGpuHost(QueueSubtitleHost, sl.GpuHost):
    """`GpuHost` of one queue item. `run` is the `_Run` of the local executor thread (`_subtitle_local` slot)."""

    def __init__(self, service: Any, item_id: str, run: Any = None):
        super().__init__(service, item_id, run, executor=LOCAL)
        self._procs: List[Any] = []
        self._lease: Any = None                    # the _Run in `_wk` (exclusive) or a _SharedLease
        self._window: Optional[list] = None        # this lease's [start, end] in service._subtitle_windows
        self._hold = False                         # this host set service._subtitle_turn_hold
        self._restore_stopped = False              # stop_restore stopped a worker (resume_restore clears hold)
        self._samples: Deque[Tuple[float, float]] = deque()
        self._oom_key: Any = None
        self._oom = False
        self.yield_reason = ""                     # why check_cancel last cancelled ("" = not by a gate)

    # -- helpers ----------------------------------------------------------------------------------
    def _now(self) -> float:
        clock = getattr(self.service, "_clock", None)
        return clock() if callable(clock) else time.time()

    def _restore_run(self):
        wk = self.service._wk
        if wk is not None and wk.item_id == self.item_id and getattr(wk, "stage", "") == "restore":
            return wk
        return None

    # -- subprocess registry (R1) -----------------------------------------------------------------
    def register_process(self, proc: Any) -> None:
        """`on_process` hook of the model subprocesses (default_local_deps -> pipeline / llama client). Stored on
        the run (and the exclusive lease) as `.proc`, so the queue's `_request_stop` kills it like a worker;
        if a stop already happened, the process is killed at once (no window between Popen and registration)."""
        with self.service._lock:
            self._procs = [p for p in self._procs if p.poll() is None]
            self._procs.append(proc)
            cancelled = False
            for owner in (self.run, self._lease if not isinstance(self._lease, _SharedLease) else None):
                if owner is not None:
                    owner.proc = proc
                    cancelled = cancelled or owner.cancel.is_set()
        if cancelled:
            _kill(proc)

    def kill_processes(self) -> int:
        """Kill every registered model process that still runs; returns how many were alive."""
        with self.service._lock:
            procs, self._procs = list(self._procs), []
        alive = [p for p in procs if p.poll() is None]
        for proc in alive:
            _kill(proc)
        return len(alive)

    # -- CloudHost part ---------------------------------------------------------------------------
    def check_cancel(self) -> None:
        run, lease = self.run, self._lease
        if run is not None and run.cancel.is_set():
            raise ss.Cancelled()
        if lease is not None and lease.cancel.is_set():          # _step stopped the lease (player busy, preempt)
            self.yield_reason = self.yield_reason or "lease_stopped"
            raise ss.Cancelled()
        item = self._item()
        if item is None:
            raise ss.Cancelled()
        if run is not None and not self.service._run_is_current(run):
            raise ss.Cancelled()
        reason = self.service._subtitle_local_gate_reason(item)
        if reason:
            self.yield_reason = reason
            raise ss.Cancelled()

    # -- GpuHost part -----------------------------------------------------------------------------
    def stop_restore(self) -> bool:
        """Interleave turn: set the turn hold FIRST (the tick must not start restoration in the gap until the
        lease is in `_wk`), then hard-stop this item's restoration worker (`preempt`, the worker's own cancel path
        puts the item back to queued) and wait until it left `_wk`. True when a worker was stopped."""
        svc = self.service
        with svc._lock:
            svc._subtitle_turn_hold = self.item_id
            self._hold = True
            wk = self._restore_run()
        if wk is None:
            self._restore_stopped = False
            return False
        svc._request_stop(wk, "preempt")
        deadline = time.monotonic() + STOP_RESTORE_TIMEOUT_S
        thread = getattr(wk, "thread", None)
        while svc._wk is wk and time.monotonic() < deadline:
            if thread is not None and thread is not threading.current_thread():
                thread.join(0.05)
                if not thread.is_alive() and svc._wk is wk:
                    with svc._lock:                       # the worker thread is gone: its slot is stale
                        if svc._wk is wk:
                            svc._wk = None
            else:
                time.sleep(0.05)
        self._restore_stopped = True
        return True

    def resume_restore(self) -> None:
        """After the turn: drop the hold; the tick restarts restoration at the first missing segment."""
        self._restore_stopped = False
        self._clear_hold()
        self.service._wake.set()

    def _clear_hold(self) -> None:
        svc = self.service
        with svc._lock:
            if self._hold and getattr(svc, "_subtitle_turn_hold", None) == self.item_id:
                svc._subtitle_turn_hold = None
            self._hold = False

    def _lease_blocked(self, item) -> str:
        svc = self.service
        if self.run is not None and self.run.cancel.is_set():
            return "cancelled"
        if svc._paused_all or svc._stopping.is_set():
            return "queue_paused_or_stopping"
        perf = getattr(svc, "perf", None)
        if perf is not None and getattr(perf, "holding", False):
            return "perf_holding"
        if getattr(svc, "_subtitle_model_test_busy", False):
            return "model_test"
        if item is None or self.item_id in svc._tombstones:
            return "item_removed"
        if svc._player_busy_for(self.item_id):
            return "player_busy"
        return ""

    def acquire_lease(self, exclusive: bool) -> Optional[Any]:
        """exclusive (interleave): a `_Run` with stage `subtitle_lease` in `_wk`, so no worker / mux / old
        pipeline starts while the subtitle models hold the GPU, and `_step` stops it (player busy, pause,
        preempt) like any non-restore `_wk`. Shared (parallel): restoration keeps `_wk`; the token only records
        the turn. None (not an error, D1) when the GPU cannot be used now."""
        from oh_streamer.queue.queue_service import _Run
        svc = self.service
        item = self._item()
        blocked = self._lease_blocked(item)                    # outside the lock: is_player_busy takes it
        with svc._lock:
            if not blocked and exclusive and svc._wk is not None:
                blocked = "gpu_in_use"
            if blocked:
                if not self._restore_stopped:                  # no resume_restore follows: drop the hold here
                    if self._hold and getattr(svc, "_subtitle_turn_hold", None) == self.item_id:
                        svc._subtitle_turn_hold = None
                    self._hold = False
                lease = None
            elif exclusive:
                lease = _Run(self.item_id, svc._generations.get(self.item_id, 0))
                lease.stage = RUN_STAGE_LEASE
                lease.subtitle_only = True
                lease.thread = self.run.thread if self.run is not None else None   # remove() joins the executor
                svc._wk = lease
            else:
                lease = _SharedLease(self.item_id)
            if lease is not None:
                self._lease = lease
                self._window = [self._now(), None]
                windows = getattr(svc, "_subtitle_windows", None)
                if windows is None:
                    windows = svc._subtitle_windows = []
                windows.append(self._window)
        if lease is None:
            self.yield_reason = blocked
        return lease

    def release_lease(self, lease: Any) -> None:
        svc = self.service
        with svc._lock:
            if svc._wk is lease:
                svc._wk = None
            if self._lease is lease:
                self._lease = None
            if self._window is not None:
                self._window[1] = self._now()
                self._window = None
                windows = getattr(svc, "_subtitle_windows", None)
                if windows is not None:
                    cutoff = self._now() - WINDOW_KEEP_S
                    windows[:] = [w for w in windows if w[1] is None or w[1] > cutoff]
            if not self._restore_stopped and self._hold:      # nothing was stopped: no resume_restore follows
                if getattr(svc, "_subtitle_turn_hold", None) == self.item_id:
                    svc._subtitle_turn_hold = None
                self._hold = False
        svc._wake.set()

    def restore_info(self) -> sl.RestoreInfo:
        item = self._item()
        if item is None:
            return sl.RestoreInfo(lead_s=math.inf, has_work=False)
        if is_stream_item(item):
            return self._stream_restore_info()
        clock = float(self.service._subtitle_position(item))
        lead = float(self.service._subtitle_restore_lead(item, clock))
        running = self._restore_run() is not None
        return sl.RestoreInfo(lead_s=lead, rate=self._rate(item, running) if running else None, running=running,
                              has_work=item.state in ("queued", "processing"), oom=self._worker_oom())

    def _live_restore_report(self) -> Optional[Tuple[dict, bool]]:
        """Return a fresh live report for this item and whether the current busy status is fresh and true."""
        svc = self.service
        lock = getattr(svc, "_lock", None)
        if lock is None:
            return None
        try:
            with lock:
                raw = getattr(svc, "_live", None)
                raw = dict(raw) if isinstance(raw, dict) else raw
                stamp = getattr(svc, "_live_ts", None)
                busy_stamp = getattr(svc, "_busy_ts", None)
                clock = getattr(svc, "_clock", None)
                now = clock() if callable(clock) else self._now()
                expiry = getattr(svc, "busy_expire", None)
                busy = getattr(svc, "_busy", False)
        except Exception:                                      # telemetry must not break subtitle scheduling
            return None
        if not isinstance(raw, dict) or raw.get("stream_item") != self.item_id:
            return None
        try:
            age = float(now) - float(stamp)
            ttl = float(expiry)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(age) and math.isfinite(ttl) and 0.0 <= age < ttl):
            return None
        try:
            busy_age = float(now) - float(busy_stamp)
            busy_fresh = math.isfinite(busy_age) and 0.0 <= busy_age < ttl
        except (TypeError, ValueError, OverflowError):
            busy_fresh = False
        return raw, busy is True and busy_fresh

    @staticmethod
    def _finite_nonnegative(value) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    def _stream_restore_info(self) -> sl.RestoreInfo:
        report = self._live_restore_report()
        if report is None:
            # The current live player's cache is unknown. Do not let missing or stale telemetry imply an infinite
            # restoration runway to the local scheduler.
            return sl.RestoreInfo(lead_s=0.0, has_work=True)
        live, busy = report
        done = live.get("restore_done") is True
        lead = self._finite_nonnegative(live.get("ahead_s"))
        rate = None if done else self._finite_nonnegative(live.get("speed"))
        return sl.RestoreInfo(lead_s=lead if lead is not None else 0.0, rate=rate,
                              running=busy and not done, has_work=not done)

    def _rate(self, item, running: bool) -> Optional[float]:
        """Media seconds restored per wall second over the last ~20 s (frame counter / source fps), else the
        worker's own fps / source fps; None when unknown."""
        progress = getattr(item, "progress", None) or {}
        fps = progress.get("source_fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            return None
        now = time.monotonic()
        frame = progress.get("frame")
        if isinstance(frame, (int, float)):
            if self._samples and frame < self._samples[-1][1]:
                self._samples.clear()                          # a restarted worker counts from its own start
            self._samples.append((now, float(frame)))
            while self._samples and now - self._samples[0][0] > RATE_WINDOW_S:
                self._samples.popleft()
            t0, f0 = self._samples[0]
            if now - t0 >= RATE_MIN_SPAN_S:
                return max(0.0, (float(frame) - f0) / (now - t0) / float(fps))
        worker_fps = progress.get("fps")
        if isinstance(worker_fps, (int, float)) and worker_fps >= 0:
            return float(worker_fps) / float(fps)
        return None

    def _worker_oom(self) -> bool:
        """The restoration worker's log ends in a CUDA out-of-memory error (parallel guard). Cached per size."""
        try:
            path = self.service.store.item_dir(self.item_id) / "worker.log"
            stat = path.stat()
        except OSError:
            return False
        key = (stat.st_size, stat.st_mtime)
        if key != self._oom_key:
            self._oom_key = key
            try:
                with open(path, "rb") as fh:
                    fh.seek(max(0, stat.st_size - 4000))
                    tail = fh.read().decode("utf-8", errors="replace")
                self._oom = gt.is_cuda_oom(tail)
            except OSError:
                self._oom = False
        return self._oom

    def gpu_info(self) -> sl.GpuInfo:
        return self.service._subtitle_gpu_info()

    # -- end of the thread ------------------------------------------------------------------------
    def close(self) -> None:
        """Thread end (whatever happened): kill leftover model processes, give back a lease the executor did
        not release, drop a hold this host still owns (restoration may start again)."""
        self.kill_processes()
        lease = self._lease
        if lease is not None:
            self.release_lease(lease)
        self._restore_stopped = False
        self._clear_hold()
        self.service._wake.set()


class QueueStreamGpuHost(QueueGpuHost):
    """Host of the live-stream item's local subtitles. The restoration is the player's, not a queue worker: the
    queue cannot stop it by itself. An exclusive lease (ASR / MT on the GPU while the player restores) is only
    given after the player confirmed `restore_pause` (`_player_pause_confirmed`); without it the turn is refused
    (yield_reason "player_restore_running") so ASR and restoration never run together.

    T4.2 (B2): stop_restore() sends {"cmd": "restore_pause", "stream_item", "turn"} to the player and waits until
    its live report says restore_pause.state == "paused" for this turn (STOP_RESTORE_TIMEOUT_S); resume_restore()
    / close() / the service's restore_needed handler send restore_resume. Every send goes through the service
    (`_stream_restore_cmd`, never under service._lock). One resume per pause: the turn token is cleared when it is
    sent, so a late resume of an old host can never release a newer host's pause."""

    def __init__(self, service: Any, item_id: str, run: Any = None):
        super().__init__(service, item_id, run)
        self._player_paused = False
        self._turn: Optional[str] = None           # token of the pause request in flight / confirmed
        self._turn_lock = threading.Lock()

    def _player_pause_confirmed(self) -> bool:
        return self._player_paused is True

    def _pause_report(self) -> Tuple[str, Optional[str]]:
        """(state, turn) of the player's live restore_pause report ("" = no fresh report / old player)."""
        report = self._live_restore_report()
        pause = report[0].get("restore_pause") if report is not None else None
        if not isinstance(pause, dict):
            return "", None
        state, turn = pause.get("state"), pause.get("turn")
        return (state if isinstance(state, str) else ""), (turn if isinstance(turn, str) else None)

    def stop_restore(self) -> bool:
        """Ask the player to pause its restoration and wait for the confirmation. True = paused (the lease may be
        given); False = not paused (timeout, send failed, cool-down after restore_needed): the player is told to
        resume (safe direction) and acquire_lease refuses, so the executor sees gpu_busy and retries later.
        Raises ss.Cancelled (after a resume) when the run is cancelled / the item is gone while waiting."""
        svc = self.service
        now = self._now()
        until = svc._stream_restore_cooldown_until(self.item_id)
        if now < until:
            _log(f"stream restore pause skipped for {self.item_id}: cool-down {until - now:.1f}s after restore_needed")
            self.yield_reason = "restore_needed_cooldown"
            return False
        turn = uuid.uuid4().hex[:8]
        with self._turn_lock:
            self._turn, self._player_paused = turn, False
        t0 = time.monotonic()
        _log(f"stream restore pause requested item={self.item_id} turn={turn}")
        failure = ""
        if not svc._stream_restore_cmd("restore_pause", self.item_id, turn):
            failure = "send failed"
        else:
            deadline = t0 + STOP_RESTORE_TIMEOUT_S
            while True:
                try:
                    self.check_cancel()
                except ss.Cancelled:
                    _log(f"stream restore pause aborted item={self.item_id} turn={turn} after "
                         f"{time.monotonic() - t0:.2f}s (cancelled)")
                    self._resume_player("cancelled")
                    raise
                state, got = self._pause_report()
                if state == "paused" and got == turn:
                    break
                if time.monotonic() >= deadline:
                    failure = f"timeout (state={state or 'unknown'})"
                    break
                time.sleep(RESTORE_POLL_S)
        if failure:
            _log(f"stream restore pause failed item={self.item_id} turn={turn} after "
                 f"{time.monotonic() - t0:.2f}s: {failure}")
            self._resume_player("pause_failed")
            return False
        with self._turn_lock:
            if self._turn != turn:                     # restore_needed took the turn away while we confirmed
                return False
            self._player_paused = True
        _log(f"stream restore pause confirmed item={self.item_id} turn={turn} after {time.monotonic() - t0:.2f}s")
        self._learn_paused_vram()
        return True

    def _learn_paused_vram(self) -> None:
        """T4.3: remember the VRAM the paused restoration really holds (live report restore_pause.reserved_mib) for
        this card; the VRAM check of the stream verdict prefers the recent larger value. Never raises."""
        try:
            from oh_streamer.queue.stream_subtitle_gpu import RestoreVramMemory
            report = self._live_restore_report()
            pause = report[0].get("restore_pause") if report is not None else None
            mib = pause.get("reserved_mib") if isinstance(pause, dict) and pause.get("turn") == self._turn else None
            if mib is None:
                return
            info = self.service._subtitle_gpu_info()
            key = gt.gpu_key(info.name, info.total_mib)
            value = self.service._subtitle_restore_vram_memory().record(key, mib)
            _log(f"stream paused restore VRAM item={self.item_id} reserved={float(mib):.0f} MiB -> remembered {value} ({key})")
        except Exception as exc:                       # noqa: BLE001 - learning is best effort
            _log(f"stream paused restore VRAM not remembered for {self.item_id}: {exc!r}")

    def note_vram_short(self, need_mib: float, free_mib: Optional[float]) -> None:
        _log(f"stream local subtitle turn given up for {self.item_id}: free VRAM {free_mib} MiB < need {need_mib:.0f} MiB")
        self.service._stream_vram_short_note(self.item_id)
    def _resume_player(self, why: str) -> bool:
        """Send restore_resume for the turn in flight (once); False when there is none."""
        with self._turn_lock:
            turn, self._turn, self._player_paused = self._turn, None, False
        if turn is None:
            return False
        _log(f"stream restore resume item={self.item_id} turn={turn} ({why})")
        t0, sent_at = time.monotonic(), self._now()
        sent = self.service._stream_restore_cmd("restore_resume", self.item_id, turn)
        if sent and why in ("resume_restore", "close"):          # a normal turn end (not a cancel / failed pause)
            self._watch_restart(t0, sent_at)
        return True

    def _watch_restart(self, t0: float, sent_at: float) -> None:
        """T4.2 fix 2: learn the restoration restart of this card = restore_resume sent -> a live report received AFTER
        it shows restoration output again (restore_pause.state "running" and a restoration speed > 0). Cheap: no new
        player message, only the live report the queue already receives (every ~2 s, so the value is that coarse).
        A short daemon thread; it gives up when a new pause starts or after RESTART_WATCH_S."""
        def watch() -> None:
            try:
                svc = self.service
                while time.monotonic() - t0 < RESTART_WATCH_S:
                    if self.holds_player_pause() or svc._stopping.is_set():
                        return
                    report = self._live_restore_report()
                    stamp = getattr(svc, "_live_ts", None)
                    if report is not None and isinstance(stamp, (int, float)) and stamp > sent_at:
                        live = report[0]
                        pause = live.get("restore_pause")
                        speed = live.get("speed")
                        if ((not isinstance(pause, dict) or pause.get("state") == "running")
                                and isinstance(speed, (int, float)) and speed > 0):
                            self._learn_restart(time.monotonic() - t0)
                            return
                    time.sleep(RESTART_POLL_S)
            except Exception as exc:                   # noqa: BLE001 - learning is best effort
                _log(f"stream restart cost not measured for {self.item_id}: {exc!r}")
        threading.Thread(target=watch, name=f"restart-watch-{self.item_id}", daemon=True).start()

    def _learn_restart(self, seconds: float) -> None:
        from oh_streamer.queue.stream_subtitle_gpu import COST_RESTART
        info = self.service._subtitle_gpu_info()
        key = gt.gpu_key(info.name, info.total_mib)
        value = self.service._subtitle_cost_memory().record(key, COST_RESTART, seconds)
        _log(f"stream restoration restart item={self.item_id} {seconds:.1f}s -> remembered {value} ({key})")

    def holds_player_pause(self) -> bool:
        with self._turn_lock:
            return self._turn is not None

    def resume_restore(self) -> None:
        self._resume_player("resume_restore")
        super().resume_restore()

    def close(self) -> None:
        super().close()                                # model processes are gone before the player resumes
        self._resume_player("close")

    def acquire_lease(self, exclusive: bool) -> Optional[Any]:
        if exclusive and not self._player_pause_confirmed():
            self.yield_reason = "player_restore_running"
            return None
        return super().acquire_lease(exclusive)


def gpu_identity() -> sl.GpuInfo:
    """Name and total MiB of GPU 0 from nvidia-smi (no CUDA context); GpuInfo() without an NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        name, total = [x.strip() for x in out.split(",")[:2]]
        return sl.GpuInfo(name, float(total))
    except Exception:                                  # noqa: BLE001 - no nvidia-smi / no NVIDIA GPU
        return sl.GpuInfo()


# --------------------------------------------------------------------------------------------------
# Mixin (local half)
# --------------------------------------------------------------------------------------------------

class GpuSubtitleSchedulerMixin(SubtitleSchedulerMixin):
    """T1's cloud half plus the local executor (`_subtitle_local` slot, one thread at a time)."""

    # -- public for QueueService (T3) -------------------------------------------------------------
    def _subtitle_turn_held(self) -> bool:
        """True while a local subtitle turn stops restoration: `_maybe_start_worker` and
        `_maybe_start_subtitle_only` must not start anything on the GPU (R2)."""
        return bool(getattr(self, "_subtitle_turn_hold", None))

    def _subtitle_executor_tick(self) -> None:
        super()._subtitle_executor_tick()
        self._subtitle_stream_tick()                   # T3.1-A: the stream item's own cloud slot
        self._subtitle_local_tick()

    def _subtitle_stop_executors(self, item_id: str, reason: str = "pause", *, wait: float = 0.0) -> bool:
        local = getattr(self, "_subtitle_local", None)
        host = getattr(local, "host", None) if local is not None and local.item_id == item_id else None
        if host is not None:
            with self._lock:
                if local.reason is None:
                    local.reason = reason
                local.cancel.set()                     # before the kill: the failure is seen as a cancel
            host.kill_processes()
        return super()._subtitle_stop_executors(item_id, reason, wait=wait)

    # -- settings ---------------------------------------------------------------------------------
    def _subtitle_resource_mode(self, item=None) -> str:
        """auto / interleave / parallel: item snapshot first, then the global setting; anything else = auto."""
        for settings in ((getattr(item, "settings", None) or {}), self.store.get_settings()):
            value = str(settings.get(RESOURCE_MODE_KEY) or "").strip().lower()
            if value in gt.RESOURCE_MODES:
                return value
        return gt.MODE_AUTO

    def _subtitle_gpu_info(self) -> sl.GpuInfo:
        info = getattr(self, "_subtitle_gpu_info_cache", None)
        if info is None:
            info = self._subtitle_gpu_info_cache = gpu_identity()
        return info

    # -- gates ------------------------------------------------------------------------------------
    def _subtitle_local_gate_reason(self, item) -> str:
        """Why the local executor must not run for the item now ('' = may run). No cloud consent here; the
        player being busy makes the local side give way (old _subtitle_check)."""
        if not item:
            return "item_removed"
        if self._paused_all or self._stopping.is_set():
            return "queue_paused_or_stopping"
        perf = getattr(self, "perf", None)
        if perf is not None and getattr(perf, "holding", False):
            return "perf_holding"
        if getattr(self, "_subtitle_model_test_busy", False):
            return "model_test"
        if item.state in ("paused", "failed") or item.subtitle.get("status") in UNAVAILABLE_SUBTITLE:
            return "item_unavailable"
        if item.id in self._tombstones or item.id in self._reprocessing:
            return "item_unavailable"
        if self._player_busy_for(item.id):
            return "player_busy"
        return ""

    def _live_stream_is(self, item_id: str) -> bool:
        """The player's fresh live report (`_live`, player_busy heartbeat) names `item_id` as the stream it plays."""
        with self._lock:
            live, stamp = getattr(self, "_live", None), getattr(self, "_live_ts", 0.0)
            clock = getattr(self, "_clock", None)
            now = clock() if callable(clock) else time.time()
            fresh = now - stamp < getattr(self, "busy_expire", 0.0)
        return bool(fresh and isinstance(live, dict) and item_id and live.get("stream_item") == item_id)

    def _player_busy_for(self, item_id: str) -> bool:
        """is_player_busy(), except for the stream item the busy player is playing right now: its own local
        subtitles run beside the live restoration (the GpuHost lease still needs a confirmed pause)."""
        return self.is_player_busy() and not self._live_stream_is(item_id)

    def _subtitle_gpu_other(self, item_id: str) -> bool:
        """`_wk` is held by ANOTHER item (restoration, mux, a lease): an interleave turn for this item would only
        wait for the GPU, so the local thread is not started / is given back for that item's own work."""
        wk = self._wk
        return wk is not None and wk.item_id != item_id

    # -- tick -------------------------------------------------------------------------------------
    def _subtitle_local_tick(self) -> None:
        run = getattr(self, "_subtitle_local", None)
        if run is not None:
            if run.cancel.is_set():
                return
            item = self.store.get(run.item_id)
            reason = self._subtitle_local_gate_reason(item)
            host = getattr(run, "host", None)
            if (not reason and host is not None and host._lease is None
                    and self._subtitle_resource_mode(item) != gt.MODE_PARALLEL
                    and self._subtitle_gpu_other(run.item_id)):
                reason = "gpu_other_item"
            if reason:
                self._subtitle_stop_local(run, reason)
            return
        now = self._subtitle_time()
        watched = self._watched_ids()
        wk = self._wk
        current = wk.item_id if wk is not None else None
        # watched first (a stream item before a queue item), then the item restoring now (its turns interleave
        # with its own worker), queue order after
        stream_items = getattr(self.store, "stream_items", None)
        candidates = list(self.store.items()) + (list(stream_items()) if stream_items is not None else [])
        items = sorted(candidates, key=lambda it: (it.id not in watched, not is_stream_item(it), it.id != current))
        for item in items:
            if self._subtitle_local_wanted(item, now):
                self._subtitle_start_local(item)
                return

    # -- T4.2: live restore pause / restore_needed (stream items) ----------------------------------
    def _stream_restore_cmd(self, cmd: str, item_id: str, turn: Optional[str]) -> bool:
        """Send restore_pause / restore_resume to the player (the service's send_to_player, short timeout). True when
        the player answered ok. Never raises; never call under self._lock."""
        try:
            reply = self._send_to_player({"cmd": cmd, "stream_item": item_id, "turn": turn or ""})
        except Exception as exc:                       # noqa: BLE001 - the picture never depends on this
            _log(f"{cmd} for {item_id} could not reach the player: {exc!r}")
            return False
        ok = isinstance(reply, dict) and reply.get("ok") is True
        if not ok:
            _log(f"{cmd} for {item_id} not accepted by the player: {reply!r}")
        return ok

    def _stream_restore_cooldown_until(self, item_id: str) -> float:
        return getattr(self, "_stream_restore_cooldowns", {}).get(item_id, 0.0)

    def _stream_restore_note_needed(self, item_id: str, reason: str) -> None:
        clock = getattr(self, "_clock", None)
        now = clock() if callable(clock) else time.time()
        wait = RESTORE_TIMEOUT_COOLDOWN_S if reason == "timeout" else RESTORE_NEEDED_COOLDOWN_S
        with self._lock:
            cooldowns = self.__dict__.setdefault("_stream_restore_cooldowns", {})
            for key in [k for k, v in cooldowns.items() if v <= now]:
                del cooldowns[key]
            cooldowns[item_id] = now + wait

    def _ipc_restore_needed(self, msg: dict) -> dict:
        """player -> queue restore_needed: stop the stream item's local subtitle turn (kill the model subprocesses),
        then answer the player with restore_resume. The reply is immediate (the player waits at most 3 s): the
        cancel flags are set here, the kill (kill_tree: suspends / kills / rescans the process tree, can take a
        while) and the resume run in a short thread, in that order, so the GPU is free when the player restarts.
        The in-flight slice goes back to pending: the executor sees the cancel (Cancelled) and releases its
        running tasks (`_release_mine`), same as every other lease stop."""
        item_id = str(msg.get("id", ""))
        item = self.store.get(item_id)
        if item is None:
            return {"ok": False, "error": "unknown id"}
        if not is_stream_item(item):
            return {"ok": False, "error": "not a stream item"}
        reason = str(msg.get("reason") or "")
        self._stream_restore_note_needed(item_id, reason)
        with self._lock:
            run = getattr(self, "_subtitle_local", None)
            run = run if run is not None and run.item_id == item_id else None
            host = getattr(run, "host", None)
            host = host if isinstance(host, QueueStreamGpuHost) else None
            paused = host is not None and host.holds_player_pause()
            if paused:
                if run.reason is None:
                    run.reason = "auto"
                run.cancel.set()                       # before the kill: the failure is seen as a cancel
                host.yield_reason = "player_needs_restore"
        _log(f"stream restore_needed item={item_id} reason={reason or '?'} paused_turn={paused}")

        def finish() -> None:
            try:
                if paused:
                    self._subtitle_stop_local(run, "player_needs_restore")
                    host._resume_player("restore_needed")
                else:
                    self._stream_restore_cmd("restore_resume", item_id, "")    # harmless when nothing is paused
            except Exception as exc:                   # noqa: BLE001
                _log(f"restore_needed handling failed for {item_id}: {exc!r}")
        threading.Thread(target=finish, name=f"restore-needed-{item_id}", daemon=True).start()
        return {"ok": True}

    def _subtitle_stop_local(self, run, reason: str) -> None:
        if host := getattr(run, "host", None):
            host.yield_reason = reason
        self._request_stop(run, "auto")
        with self._lock:
            lease = getattr(host, "_lease", None) if host is not None else None
        if lease is not None and not isinstance(lease, _SharedLease):
            self._request_stop(lease, "auto")
        executor = getattr(run, "executor", None)
        if executor is not None:
            executor.stop()
        if host is not None:
            host.kill_processes()

    def _subtitle_local_wanted(self, item, now: float) -> bool:
        """Should a local executor thread start for the item now? Only when units are on a local candidate
        that dispatchable_for would hand to local now (or, without a slice table, the first candidate is local):
        default_local_deps is never built otherwise."""
        settings = item.settings or {}
        if not settings.get("subtitles") or item.subtitle.get("status") in NO_START_STATUSES:
            return False
        if is_stream_item(item) and not self._subtitle_stream_audio_ready(item):
            return False
        if not self._subtitle_gate_open(item.id, LOCAL, now):
            return False
        if self._subtitle_local_gate_reason(item):
            return False
        source = item.source_path or item.output_path
        if not source or not Path(source).is_file():
            return False
        if self._subtitle_resource_mode(item) != gt.MODE_PARALLEL and self._subtitle_gpu_other(item.id):
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
        home = owners[sp.PRIMARY_TRACK](sp.STAGE_ASR, sp.PRIMARY_TRACK, sp.TaskState())
        if plan is None:
            return home == LOCAL
        position = QueueSubtitleHost(self, item.id, probe_offset=False).seek_position()
        if position is not None and home == LOCAL and sp.seek_cut_request(plan, position, limit=limit) is not None:
            return True
        if self._subtitle_dispatchable(item, plan, owners, LOCAL, limit, now) is not None:
            return True
        return bool(plan.open and home == LOCAL)

    # -- start ------------------------------------------------------------------------------------
    def _subtitle_local_runtime_paths(self) -> dict:
        from oh_streamer.subtitles.environment import runtime_paths
        return runtime_paths(self.store.get_settings())

    def _subtitle_gpu_memory(self):
        return gt.GpuModeMemory()

    def _subtitle_restore_vram_memory(self):
        from oh_streamer.queue.stream_subtitle_gpu import RestoreVramMemory
        return RestoreVramMemory()

    def _subtitle_speed_memory(self):
        from oh_streamer.queue.stream_subtitle_gpu import SubtitleSpeedMemory
        return SubtitleSpeedMemory()

    def _subtitle_cost_memory(self):
        from oh_streamer.queue.stream_subtitle_gpu import SwitchCostMemory
        return SwitchCostMemory()

    def _subtitle_new_gpu_host(self, item, run) -> QueueGpuHost:
        """Build the GpuHost adapter for an item; live streams read the player's restore telemetry."""
        host_type = QueueStreamGpuHost if is_stream_item(item) else QueueGpuHost
        return host_type(self, item.id, run)

    def _subtitle_new_local_deps(self, item, host: QueueGpuHost, config: dict):
        """The real local models (heavy: model fingerprints) - built only when the thread starts. Every model
        subprocess is registered with the host (cancel kills it)."""
        self._subtitle_require_local_models(item)
        folder = self.store.item_dir(item.id) / "subtitles"
        cfg = dict(config)
        translation = dict(cfg.get("translation") or {})
        translation.setdefault("model_file", item.settings.get("subtitle_translation")
                               or "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf")
        cfg["translation"] = translation
        return sl.default_local_deps(folder, item.source_path or item.output_path, cfg, settings=item.settings,
                                     runtime_paths=self._subtitle_local_runtime_paths(),
                                     on_process=host.register_process)

    def _subtitle_new_local_executor(self, item, host: QueueGpuHost):
        """Build the LocalSliceExecutor of the item (the seam tests replace with a fake)."""
        from oh_streamer.subtitles import configuration
        item_id = item.id
        folder = self.store.item_dir(item_id) / "subtitles"
        primary = (item.settings.get("subtitle_configs") or [{}])[0]
        japanese = bool(primary.get("japanese_track", item.settings.get("subtitle_japanese", False)))
        ledger, credentials = self._subtitle_ledger_for(item)

        def current():
            it = self.store.get(item_id)
            return it.settings if it is not None else item.settings

        deps = self._subtitle_new_local_deps(item, host, primary)
        turns = stream_turn_config(item.settings) if is_stream_item(item) else None   # None: executor default
        return sl.LocalSliceExecutor(
            folder, item.source_path or item.output_path, host=host, deps=deps, config=primary,
            settings=item.settings, japanese=japanese, resource_mode=self._subtitle_resource_mode(item),
            memory=self._subtitle_gpu_memory(),
            speed_memory=self._subtitle_speed_memory(),
            cost_memory=self._subtitle_cost_memory() if turns is not None else None,
            plan_provider=lambda: configuration.runtime_plan(current()),
            track_plans_provider=lambda: dict(configuration.track_runtime_plans(current())),
            ledger=ledger, credentials=credentials, turns=turns)

    def _subtitle_require_local_models(self, item) -> None:
        """Check every local row for downloaded models before work starts. A missing primary fails the item
        (RuntimeError, zh-TW). Comparison rows run as independent tracks (a not-ready track plan gets no work and
        publishes its reasons, the primary and other rows are unaffected), so a comparison row with missing models
        is only marked not ready in the item's `subtitle_track_plans`. Cloud rows are not checked here."""
        from oh_streamer.subtitles import combos, configuration
        rows = item.settings.get("subtitle_configs") or []
        if not rows:
            return
        from oh_streamer.subtitles.environment import runtime_paths
        paths = runtime_paths(self.store.get_settings())

        def check(row):
            if combos.is_cloud_combo(row, item.settings):
                return None
            info = combos.combo_availability(row, paths["models"], paths["server"], item.settings)
            return None if info["available"] else info

        info = check(rows[0])
        if info is not None:
            raise RuntimeError(info["reason"])
        if is_stream_item(item):                       # a stream item runs the main subtitle only
            return
        missing, recovered = {}, []
        for row in rows[1:]:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                info = check(row)
                if info is not None:
                    missing[row["id"]] = info
                else:
                    recovered.append(row["id"])
        plans = configuration.track_runtime_plans(item.settings)
        changed = False
        # A row marked missing on an earlier start whose models are now downloaded gets its frozen plan back.
        fresh = None
        for track in recovered:
            plan = plans.get(track)
            if plan and any(e.get("code") == "models_missing" for e in plan.get("errors") or [] if isinstance(e, dict)):
                if fresh is None:
                    fresh = configuration.track_runtime_plans(
                        {k: v for k, v in item.settings.items() if k != "subtitle_track_plans"})
                if track in fresh:
                    plans[track] = fresh[track]
                    changed = True
        for track, info in missing.items():
            plan = plans.get(track)
            if plan is None or any(isinstance(e, dict) and e.get("code") == "models_missing"
                                   for e in plan.get("errors") or []):
                continue
            plan["ready"] = False
            plan["errors"] = list(plan.get("errors") or []) + [dict(code="models_missing", detail=info["reason"])]
            plan["reasons"] = [info["reason"]]
            changed = True
        if changed:
            item.settings["subtitle_track_plans"] = plans
            self.store.update(item.id, settings=dict(item.settings))

    def _subtitle_start_local(self, item) -> None:
        from .queue_service import _Run
        with self._lock:
            if getattr(self, "_subtitle_local", None) is not None:
                return
            run = _Run(item.id, self._generations.get(item.id, 0))
            run.stage = RUN_STAGE_LOCAL
            run.subtitle_only = True
            self._subtitle_local = run
        host = self._subtitle_new_gpu_host(item, run)
        run.host = host
        run.thread = threading.Thread(target=self._run_subtitle_local, args=(item.id, run),
                                      name=f"subtitle-local-{item.id}", daemon=True)
        try:
            run.executor = self._subtitle_new_local_executor(self.store.get(item.id) or item, host)
        except Exception as exc:
            _log(f"subtitle local executor could not start for {item.id}: {exc!r}")
            with self._lock:
                if self._subtitle_local is run:
                    self._subtitle_local = None
            host.close()
            self._subtitle_results_for(item.id)[LOCAL] = FinishedRun(None, exc, self._subtitle_time())
            self._subtitle_update(item.id, status="failed", error=str(exc)[-500:])
            return
        run.thread.start()

    def _run_subtitle_local(self, item_id: str, run) -> None:
        """Thread body: T1's `_run_subtitle_executor` (run(wait=True), record the end, free the slot), then the
        host's cleanup (leftover processes, lease, hold)."""
        try:
            self._run_subtitle_executor(item_id, run, LOCAL)
        finally:
            run.host.close()
