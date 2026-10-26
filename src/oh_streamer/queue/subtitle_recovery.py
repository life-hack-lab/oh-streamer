"""Subtitle user operations on the unified scheduler (M7): retry / restart / thaw.

The item-level ``blocked`` state is gone (D2): waiting for a quota or a network blip lives in the slice table
(``retry_at`` of the work unit) and the executor thread sleeps / is woken by the queue tick.  What is left here:

* the operations the user triggers (retry, restart, resume after a settings change), which stop or wake the
  subtitle executors (``_subtitle_stop_executors`` / ``_subtitle_executor_reset`` of ``subtitle_host``) and clear
  the legacy per-item fields of the old block / cloud-horizon pipelines.
"""
import copy

# Settings keys that make up a subtitle snapshot; a refresh re-derives them from the current global settings.
SNAPSHOT_KEYS = frozenset((
    "subtitle_configs", "subtitle_combos", "subtitle_profile", "subtitle_resolved", "subtitle_runtime_contract",
    "subtitle_primary", "subtitle_comparison_enabled", "subtitle_comparison_rows",
    "subtitle_cloud_strategy", "subtitle_track_plans", "subtitle_combo_fallback",
))

# Item subtitle-dict keys written only by the pre-M7 pipelines (block / cloud-horizon / combos / blocked).
# The unified scheduler never writes them; user operations drop them so old items stop carrying them.
LEGACY_SUBTITLE_FIELDS = (
    "block_mode", "blocks_done", "blocks_total", "block_ranges", "block_plan_key",
    "igpu_pending", "igpu_asr_ready", "igpu_fallback", "igpu_wait_reason",
    "cloud_mode", "cloud_done_slices", "cloud_total_slices",
    "blocked_reason", "blocked_kind", "blocked_error", "retry_at", "retry_count", "blocked_since",
    "retry_note", "backup_status", "fallback", "combos", "ready_identity", "exhausted",
)
# Published-subtitle fields: a slice table that is deleted / rebuilt invalidates them too.
PUBLISH_FIELDS = ("covered_ranges", "covered_s", "block_srt", "block_subtitles", "audio_offset", "refinement_pending")


def without_legacy(subtitle, *, publish=False):
    """A copy of an item's subtitle dict without the legacy pipeline fields (and, with ``publish``, without the
    published-subtitle fields of the slice table)."""
    drop = LEGACY_SUBTITLE_FIELDS + (PUBLISH_FIELDS if publish else ())
    return {key: value for key, value in (subtitle or {}).items() if key not in drop}


def refresh_snapshot_settings(item_settings, global_settings):
    """Item settings re-derived from the current global subtitle settings (new backups, models, keys)."""
    from .subtitle_jobs import PATH_KEYS, derive_subtitle_snapshot
    fresh = {k: copy.deepcopy(v) for k, v in global_settings.items() if k.startswith("subtitle_")}
    kept = {k: v for k, v in item_settings.items() if k not in SNAPSHOT_KEYS and k not in PATH_KEYS}
    own = {k: v for k, v in kept.items() if k.startswith("subtitle_") and k != "subtitle_overrides"}
    derived = derive_subtitle_snapshot({**fresh, **own}, kept)       # item overrides apply; paths are never stored
    return {**copy.deepcopy(kept), **derived, "subtitles": True}


HOLD_KEYS = ("subtitle_hold", "subtitle_hold_sig")


def settings_decision(locked, fresh, *, started, hold=False, hold_sig=None):
    """Pure gate for a settings change on one item: 'adopt' (store the fresh snapshot), 'hold' (the user declined
    exactly these fresh settings: keep the locked ones, no prompt) or 'pending' (keep the locked settings and ask).
    An item that has not started always adopts; so does a change that does not touch the subtitle content."""
    from .subtitle_jobs import content_signature
    if not started:
        return "adopt"
    fresh_sig = content_signature(fresh)
    if fresh_sig == content_signature(locked):
        return "adopt"
    if hold and hold_sig == fresh_sig:
        return "hold"
    return "pending"


class SubtitleRecoveryMixin:
    """User operations and failure bookkeeping shared by the subtitle executors of the queue service."""

    def _subtitle_fail(self, item_id, exc, **fields):
        """Record a subtitle error as ``failed`` (the item-level ``blocked`` state is gone: a quota or network
        wait now lives in the slice table)."""
        with self._lock:
            self._subtitle_update(item_id, status="failed", error=str(exc)[-500:], **fields)

    # ------------------------------------------------------------------ executor helpers (M7)
    def _subtitle_halt(self, item_id, reason="pause", wait=15.0):
        """Stop every subtitle executor thread of the item and wait for them. True when none is left running.
        Never call with ``self._lock`` held (the threads take it while they end)."""
        stop = getattr(self, "_subtitle_stop_executors", None)
        return True if stop is None else bool(stop(item_id, reason, wait=wait))

    def _subtitle_wake(self, item_id):
        """Forget how the last executor run ended (retry / restart / resume / settings change) and wake the tick."""
        reset = getattr(self, "_subtitle_executor_reset", None)
        if reset is not None:
            reset(item_id)
        self._wake.set()

    def _subtitle_request_refresh(self, item_id):
        """Soft drain of the item's running executors (settings changed): the in-flight work unit finishes with the
        settings it started with, then the executor ends ("refresh") and the tick starts one with the new
        snapshot. True when a running executor was asked."""
        asked = False
        runs = getattr(self, "_subtitle_runs", None)
        for run in (runs() if runs else ()):
            executor = getattr(run, "executor", None)
            if run.item_id == item_id and executor is not None and hasattr(executor, "request_refresh"):
                executor.request_refresh()
                asked = True
        return asked

    def _subtitle_executor_running(self, item_id):
        runs = getattr(self, "_subtitle_runs", None)
        return any(run.item_id == item_id for run in (runs() if runs else ()))

    def _subtitle_mux_busy(self, item_id):
        """The mux step (a worker subprocess under the GPU lease, not an executor) is running for the item."""
        run = self._wk
        item = self.store.get(item_id)
        return bool((run is not None and run.item_id == item_id and getattr(run, "subtitle_only", False)
                     and run.stage in ("mux", "waiting_gpu"))
                    or (item is not None and item.subtitle.get("status") == "mux"))

    def _subtitle_refreshed_settings(self, item):
        """The item's settings re-derived from the current global subtitle settings (new backups / models / keys,
        frozen contract; also upgrades a legacy item). A failed refresh keeps the old snapshot."""
        try:
            return refresh_snapshot_settings(
                item.settings, {**self.store.get_settings(), "subtitle_runtime_contract": "cloud-first-1"})
        except Exception:
            return item.settings

    def _subtitle_started(self, item_id):
        """The item has at least one done unit in its slice plan (any stage, any track)."""
        from oh_streamer.subtitles import slice_plan as sp
        try:
            plan = self._subtitle_plan_now(item_id)
        except Exception:
            return False
        return bool(plan and any(t.status == sp.DONE for s in plan.slices for t in s.tasks.values()))

    def _subtitle_settings_decision(self, item, fresh):
        """'adopt' | 'hold' | 'pending' for the item's locked settings against the fresh ones (settings_decision)."""
        return settings_decision(item.settings, fresh, started=self._subtitle_started(item.id),
                                 hold=bool(item.settings.get("subtitle_hold")),
                                 hold_sig=item.settings.get("subtitle_hold_sig"))

    def _subtitle_gate(self, item):
        """(decision, settings to store, pending dict or None) of a settings refresh through the gate, or None when
        the fresh settings cannot be derived."""
        from .subtitle_jobs import asr_signature
        fresh = self._subtitle_refreshed_settings(item)
        if fresh is item.settings:
            return None
        decision = self._subtitle_settings_decision(item, fresh)
        if decision == "adopt":
            return decision, {k: v for k, v in fresh.items() if k not in HOLD_KEYS}, None
        if decision == "hold":
            kept = dict(item.settings)
            if "subtitle_cloud_strategy" in fresh:
                kept["subtitle_cloud_strategy"] = fresh["subtitle_cloud_strategy"]
            return decision, kept, None
        return decision, item.settings, dict(asr_changed=asr_signature(fresh) != asr_signature(item.settings))

    def _subtitle_set_pending(self, item_id, pending):
        """Set (dict) or clear (None) ``item.subtitle['settings_pending']``."""
        with self._lock:
            item = self.store.get(item_id)
            if item is None or item_id in self._tombstones:
                return
            if pending is None:
                if "settings_pending" in item.subtitle:
                    self.store.update(item_id, subtitle={k: v for k, v in item.subtitle.items()
                                                         if k != "settings_pending"})
            elif item.subtitle.get("settings_pending") != pending:
                self._subtitle_update(item_id, settings_pending=pending)

    def _subtitle_apply_gate(self, item, gate):
        """Store the outcome of ``_subtitle_gate`` (settings if changed, pending flag); returns the decision."""
        decision, settings, pending = gate
        with self._lock:
            changed = settings != item.settings
            if changed:
                self.store.update(item.id, settings=settings)
            self._subtitle_set_pending(item.id, pending)
        if changed and decision == "adopt":          # a running executor holds the old snapshot: drain softly
            only_strategy = ({k: v for k, v in settings.items() if k != "subtitle_cloud_strategy"}
                             == {k: v for k, v in item.settings.items() if k != "subtitle_cloud_strategy"})
            if not only_strategy and self._subtitle_request_refresh(item.id):
                self._wake.set()
        return decision

    def _subtitle_plan_now(self, item_id):
        """The item's slice plan: the live in-memory one while an executor holds it, else the file (or None)."""
        from oh_streamer.subtitles import slice_plan as sp
        from . import subtitle_scheduler as ss
        folder = self.store.item_dir(item_id) / "subtitles"
        store = ss.PlanStore.peek(folder)
        if store is not None and store.plan is not None:
            return store.plan
        return sp.load_slice_plan(folder)

    def _subtitle_reset_failed(self, item_id):
        """Clear failed work units of the item's slice table (the plan an executor holds in memory, else the
        file). Returns how many were cleared."""
        from oh_streamer.subtitles import slice_plan as sp
        from . import subtitle_scheduler as ss
        from .subtitle_view import retry_failed
        folder = self.store.item_dir(item_id) / "subtitles"
        store = ss.PlanStore.peek(folder)
        if store is not None and store.plan is not None:
            return retry_failed(store)
        plan = sp.load_slice_plan(folder)
        if plan is None:
            return 0
        count = retry_failed(plan)
        if count:
            sp.save_slice_plan(plan, folder)
        return count

    def _subtitle_plan_complete(self, item_id):
        """Every slice of the primary track is done and no more slices are coming."""
        plan = self._subtitle_plan_now(item_id)
        return bool(plan and plan.slices and not plan.open and all(s.all_done(plan.primary) for s in plan.slices))

    def _subtitle_has_stuck_work(self, item_id):
        from oh_streamer.subtitles import slice_plan as sp
        plan = self._subtitle_plan_now(item_id)
        return bool(plan and any(t.status == sp.FAILED for s in plan.slices for t in s.tasks.values()))

    def _subtitle_requeue_fields(self, item_id):
        """Subtitle-dict fields that put a failed item back on the scheduler's books."""
        sub = self.store.get(item_id).subtitle
        if self._subtitle_plan_complete(item_id):
            return dict(status="ready", percent=100, error="")
        if sub.get("covered_s") or sub.get("block_srt"):
            return dict(status="partial", percent=None, error="")
        return dict(status="queued", percent=0, error="")

    def _subtitle_drop_legacy(self, item_id):
        item = self.store.get(item_id)
        if item is not None:
            clean = without_legacy(item.subtitle)
            if clean != item.subtitle:
                self.store.update(item_id, subtitle=clean)

    def _subtitle_thaw(self, item_id, *, refresh=False, reset=False):
        """Wake the item's subtitle executors after resume / a settings change. refresh=True adopts the current
        global subtitle settings; reset=True also clears failed work units of the slice table (new backups or
        keys deserve a fresh try). A ``failed`` item goes back to the scheduler. False when there is nothing
        subtitle-related to wake."""
        with self._lock:
            item = self.store.get(item_id)
            if item is None or not item.settings.get("subtitles") or item_id in self._tombstones:
                return False
            if refresh and (item.subtitle.get("status") == "failed" or self._subtitle_has_stuck_work(item_id)):
                gate = self._subtitle_gate(item)           # a content change of a started item waits for the user
                if gate is not None:
                    self._subtitle_apply_gate(item, gate)
            if reset:
                self._subtitle_reset_failed(item_id)
            if self.store.get(item_id).subtitle.get("status") == "failed":
                self._subtitle_update(item_id, **self._subtitle_requeue_fields(item_id))
            self._subtitle_drop_legacy(item_id)
        self._subtitle_wake(item_id)
        return True

    def _subtitle_thaw_all(self, *, refresh=False):
        """Settings changed globally: every started item gets its settings_pending prompt recomputed right away
        (an unstarted one derives at its start, nothing is stored); items that failed or got stuck try again, adopting
        the new settings only when the gate lets them."""
        for item in list(self.store.items()):
            if not item.settings.get("subtitles"):
                continue
            failed = item.subtitle.get("status") == "failed" or self._subtitle_has_stuck_work(item.id)
            if refresh and not failed:
                self._subtitle_refresh_pending(item)
            if failed:
                self._subtitle_thaw(item.id, refresh=refresh, reset=True)

    def _subtitle_refresh_pending(self, item):
        """Recompute the settings_pending prompt of a subtitle item (a decision of 'adopt' also stores the fresh
        'how' changes; a pending or held item keeps its locked settings; an unstarted one stores nothing)."""
        if item.subtitle.get("status") in ("ready", "mux_pending", "done") or not self._subtitle_started(item.id):
            self._subtitle_set_pending(item.id, None)
            return
        gate = self._subtitle_gate(item)
        if gate is not None:
            self._subtitle_apply_gate(item, gate)

    def _subtitle_reset_mt(self, item_id):
        """Fresh pending translation units for every slice and track of the item's slice plan (recognition results
        stay); the plan an executor holds in memory, else the file. Returns how many units were reset."""
        from oh_streamer.subtitles import slice_plan as sp
        from . import subtitle_scheduler as ss
        folder = self.store.item_dir(item_id) / "subtitles"

        def reset(plan):
            count = 0
            for s in plan.slices:
                tracks = set(plan.tracks) | {k.partition("/")[2] for k in s.tasks if k.startswith(sp.STAGE_MT + "/")}
                for track in tracks:
                    s.set(sp.STAGE_MT, track, sp.TaskState())
                    count += 1
            return count

        store = ss.PlanStore.peek(folder)
        if store is not None and store.plan is not None:
            with store.lock:
                count = reset(store.plan)
                store.clear_results()
                store.last_publication = None
                store.save()
            return count
        plan = sp.load_slice_plan(folder)
        if plan is None:
            return 0
        count = reset(plan)
        sp.save_slice_plan(plan, folder)
        return count

    def _subtitle_apply_choice(self, item_id, choice):
        """Apply one of 'rest' / 'keep' / 'redo_mt' to an item with a settings_pending prompt. Caller holds no lock.
        Returns dict(ok, error). A running executor is left running: its next launch picks up the new settings."""
        from .subtitle_jobs import content_signature
        if choice not in ("rest", "keep", "redo_mt"):
            return dict(ok=False, error="不明的選項")
        for attempt in range(2):
            error = self._subtitle_check_choice(item_id, choice)
            if error:
                return dict(ok=False, error=error)
            # redo_mt resets translation units: an in-flight one must not overwrite the reset, so stop the executors
            # first (never with the lock held; the threads take it while they end)
            if choice == "redo_mt" and self._subtitle_executor_running(item_id):
                if not self._subtitle_halt(item_id, "cancel"):
                    return dict(ok=False, error="字幕工作尚未停止，請稍後再試")
            with self._lock:
                if choice == "redo_mt" and self._subtitle_executor_running(item_id):
                    continue                    # a new executor started between the halt and the lock: halt again
                return self._subtitle_apply_choice_locked(item_id, choice)
        return dict(ok=False, error="字幕工作尚未停止，請稍後再試")

    def _subtitle_check_choice(self, item_id, choice):
        """zh-TW reason why the choice cannot be applied to the item, or ''."""
        with self._lock:
            item = self.store.get(item_id)
            if item is None or item_id in self._tombstones or not item.settings.get("subtitles"):
                return "此項目沒有字幕"
            pending = item.subtitle.get("settings_pending")
            if not pending:
                return "此項目沒有待套用的設定變更"
            if choice == "redo_mt" and pending.get("asr_changed"):
                return "語音辨識設定已變更，無法只重做翻譯"
        return ""

    def _subtitle_apply_choice_locked(self, item_id, choice):
        """The body of ``_subtitle_apply_choice`` (self._lock held by the caller)."""
        from .subtitle_jobs import content_signature
        if True:
            item = self.store.get(item_id)
            if item is None or not item.subtitle.get("settings_pending"):
                return dict(ok=False, error="此項目沒有待套用的設定變更")
            fresh = self._subtitle_refreshed_settings(item)
            if fresh is item.settings:
                return dict(ok=False, error="無法取得新的字幕設定")
            fresh = {k: v for k, v in fresh.items() if k not in HOLD_KEYS}
            if choice == "keep":
                settings = {**item.settings, "subtitle_hold": True, "subtitle_hold_sig": content_signature(fresh)}
                self.store.update(item_id, settings=settings)
                self._subtitle_set_pending(item_id, None)
            else:
                self.store.update(item_id, settings=fresh)
                self._subtitle_set_pending(item_id, None)
                if choice == "redo_mt":
                    self._subtitle_reset_mt(item_id)
                    item = self.store.get(item_id)
                    only = item.state == "done"
                    subtitle = {**without_legacy(item.subtitle, publish=True), "status": "queued", "percent": 0,
                                "error": "", "only": only or bool(item.subtitle.get("only")), "embedded": False,
                                "phase": "", "detail": ""}
                    self.store.update(item_id, subtitle=subtitle)
                    if only:
                        self.store.protected_ids = set(self.store.protected_ids) | {item_id}
                else:
                    self._subtitle_request_refresh(item_id)     # "rest": the running unit finishes, the next is new
        self._subtitle_wake(item_id)
        return dict(ok=True)

    def apply_subtitle_settings(self, item_id, choice):
        """The settings_pending prompt of one item: 'rest' (剩下的用新設定), 'keep' (維持原設定) or 'redo_mt'
        (已完成的也用新翻譯重做)."""
        return self._subtitle_apply_choice(item_id, choice)

    def apply_subtitle_settings_all(self, choice):
        """'rest' or 'keep' for every item that has a settings_pending prompt."""
        if choice not in ("rest", "keep"):
            return dict(ok=False, error="不明的選項", count=0)
        ids = [i.id for i in list(self.store.items()) if i.subtitle.get("settings_pending")]
        count = sum(1 for i in ids if self._subtitle_apply_choice(i, choice).get("ok"))
        return dict(ok=True, count=count)

    def restart_subtitles(self, item_id):
        """The circular-arrow button: forget every subtitle result of the item (speech recognition included) and
        process it again from scratch with the current settings. It spends recognition quota again."""
        import shutil
        item = self.store.get(item_id)
        if item is None or not item.settings.get("subtitles"):
            return dict(ok=False, error="此項目沒有字幕")
        if self._subtitle_mux_busy(item_id):
            return dict(ok=False, error="請先暫停字幕工作")
        if item_id in self._watched_ids():
            return dict(ok=False, error="請先關閉播放器中的影片")
        if not self._subtitle_halt(item_id, "cancel"):
            return dict(ok=False, error="字幕工作尚未停止，請稍後再試")
        with self._lock:
            item = self.store.get(item_id)
            if item is None or self._subtitle_executor_running(item_id) or self._subtitle_mux_busy(item_id):
                return dict(ok=False, error="請先暫停字幕工作")
            shutil.rmtree(self.store.item_dir(item_id) / "subtitles", ignore_errors=True)
            settings = {k: v for k, v in self._subtitle_refreshed_settings(item).items() if k not in HOLD_KEYS}
            only = item.state == "done"
            subtitle = {**without_legacy(item.subtitle, publish=True), "status": "queued", "percent": 0,
                        "error": "", "only": only or bool(item.subtitle.get("only")), "embedded": False,
                        "phase": "", "detail": ""}
            subtitle.pop("settings_pending", None)
            self.store.update(item_id, settings={**settings, "subtitles": True}, subtitle=subtitle)
            if only:
                self.store.protected_ids = set(self.store.protected_ids) | {item_id}
        self._subtitle_wake(item_id)
        return dict(ok=True)

    def retry_subtitles(self, item_id):
        """'Retry now': adopt current settings (new backups) and run again, regardless of any wait."""
        item = self.store.get(item_id)
        if (item is None or not item.settings.get("subtitles")
                or (item.subtitle.get("status") != "failed" and not self._subtitle_has_stuck_work(item_id))):
            return dict(ok=False, error="字幕沒有等待重試")
        with self._lock:
            item = self.store.get(item_id)
            if item is None or item_id in self._tombstones:
                return dict(ok=False, error="字幕沒有等待重試")
            gate = self._subtitle_gate(item)
            if gate is not None:
                self._subtitle_apply_gate(item, gate)
            self._subtitle_reset_failed(item_id)
            if self.store.get(item_id).subtitle.get("status") == "failed":
                self._subtitle_update(item_id, **self._subtitle_requeue_fields(item_id))
            self._subtitle_drop_legacy(item_id)
        self._subtitle_wake(item_id)
        return dict(ok=True)
