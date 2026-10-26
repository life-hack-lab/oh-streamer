"""Slice plan -> UI subtitle state.

Pure function of the slice table plus the caller's clock: nothing is read from stored events, so after a
retry / requeue the screen can never show an outdated message. All times are judged against ``now``.

Output of :func:`subtitle_view` (JSON-serializable, ``v`` = 2); front end (M5b) maps ``state`` to text.
Suggested zh-TW wording per state (Taiwan usage, following the old ``subtitleDisplay``):

- ``complete``          字幕：已完成
- ``running``           字幕：語音辨識中（雲端）/ 翻譯中（本地）… + 進度 N/M 段；stage asr=語音辨識 mt=翻譯
- ``waiting_quota``     等待額度中（N 秒後重試）；pacing 類原因可用「排隊送出中（N 秒後送出，避免超出服務限流）」
- ``retrying``          重試中（N 秒後重試，原因）：本地錯誤／網路／逾時／服務故障的退避，不是額度問題
- ``waiting_gpu``       等待 GPU（修復中，稍後輪到字幕）
- ``waiting_executor``  已交給本地/雲端備援，等待接手（N 秒後可收回雲端）；executor local=本地 cloud=雲端
- ``waiting_download``  字幕：等待下載
- ``waiting_cut``       字幕：準備切片
- ``queued``            字幕：排隊中
- ``stuck``             語音辨識卡住：Groq（限流）、本地（錯誤）；後接「重試」按鈕。
                        stage asr=語音辨識 mt=翻譯；原因 kind 對照見 app.js FALLBACK_REASON
                        （rate_limit=限流 quota_daily=今日額度用完 network=網路問題 timeout=逾時 server=服務故障
                        auth=金鑰或權限錯誤 bad_response=回應格式錯誤 content_blocked=內容被拒絕 unsupported=不支援）
- ``partial_stuck`` (flag, not a state) 另有 N 段其他位置的字幕失敗，可按「重試」

State priority (mutually exclusive): complete > stuck > running > waiting_quota > retrying > waiting_gpu >
waiting_executor > waiting_download > waiting_cut > queued.

``waiting_quota`` is only for cloud pacing / rate_limit / quota_daily waits (or a wait whose reason is unknown);
any other backoff (error / oom / unsupported / network / timeout / server ...) is ``retrying`` (wait.reason = kind).
Anti-flicker (pure, judged against ``now``): while the item still has work running, or the waiting unit's wait is
short (span <= SHORT_WAIT_S, e.g. 429 pacing bouncing running <-> 1-2 s wait) the main state stays ``running``;
``running.quota_wait`` / ``quota_retry_at`` tell the front end that part of the work is waiting for quota.
"""

from __future__ import annotations

import time
from pathlib import PurePath
from typing import Any, Callable, Dict, List, Mapping, Optional

from oh_streamer.subtitles import slice_plan as sp

VIEW_VERSION = 2
LOCAL_EXECUTOR = "local"
SHORT_WAIT_S = 3.0          # waits no longer than this (and the same time again after) never change the main state
QUOTA_WAIT_KINDS = frozenset({"rate_limit", "quota_daily"})


def _name(names: Optional[Mapping[str, str]], cid: Optional[str]) -> Optional[str]:
    if cid is None:
        return None
    return str((names or {}).get(cid) or cid)


def _row_name(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("label", "name", "remote_model"):
        if row.get(key):
            return str(row[key])
    recipe = row.get("recipe")
    if recipe and not str(recipe).startswith("__"):
        return str(recipe)
    path = row.get("model_name") if row.get("model_name") not in (None, ".") else row.get("model_path")
    if path:
        return PurePath(str(path)).name or None
    return str(recipe) if recipe else None


def candidate_names(frozen_plan: Mapping[str, Any]) -> Dict[str, str]:
    """{candidate id: display name} for both roles of a frozen plan (pass as ``candidate_names``). The name
    is the row's label / name / remote_model, for local rows the recipe or model file name; the raw id when
    the row offers nothing better."""
    from oh_streamer.subtitles import fallback
    names: Dict[str, str] = {}
    for role in ("asr", "translation"):
        for cand in fallback.candidate_rows(frozen_plan, role):
            names.setdefault(cand.id, _row_name(cand.row) or cand.id)
    return names


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)


def _wait_record(st: sp.PlanStatus, now: float) -> Optional[sp.CandidateReason]:
    """The tried-entry that explains the blocking unit's backoff (None when nothing is recorded)."""
    tried = st.blocking_tried
    who = next((c for c in tried if c.retry_at is not None and c.retry_at > now), None)
    if who is None:
        who = next((c for c in tried if c.id == st.blocking_candidate), None)
    if who is None and tried:
        who = max(tried, key=lambda c: c.at or 0.0)
    return who


def _other_running(plan: sp.SlicePlan, block: Mapping[str, Any]):
    for s in plan.slices:
        for key, state in s.tasks.items():
            stage, _, _track = key.partition("/")
            if state.status == sp.RUNNING and not (s.index == block["slice"] and stage == block["stage"]):
                return s.index, stage, state
    return None


def _candidate_row(c: sp.CandidateReason, names: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    return {"id": c.id, "name": _name(names, c.id), "executor": c.executor, "kind": c.kind,
            "at": _round(c.at), "detail": c.detail}


def _stuck_info(fail: sp.FailureInfo, names: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    rows = [_candidate_row(c, names) for c in fail.candidates]
    if not rows:   # failure recorded without per-candidate bookkeeping (older plans): one synthetic row
        cid = fail.candidate or fail.model_id or ""
        rows = [{"id": cid, "name": _name(names, cid) or "", "executor": None, "kind": fail.kind,
                 "at": _round(fail.at), "detail": fail.detail}]
    return {"slice": fail.slice_index, "stage": fail.stage, "track": fail.track, "kind": fail.kind,
            "exhausted": fail.exhausted, "candidates": rows}


def subtitle_view(plan: sp.SlicePlan, *, play_pos: float = 0.0, now: Optional[float] = None,
                  max_available: Optional[float] = None, gpu_turn: Optional[bool] = None,
                  candidate_names: Optional[Mapping[str, str]] = None,
                  owner: Optional[Callable[[str, str, Any], Optional[str]]] = None) -> Dict[str, Any]:
    """Pure: slice table + clock -> display dict. ``gpu_turn``: True/False when the caller knows whether it is
    the local executor's GPU turn, None when unknown (then ``waiting_gpu`` is never reported).
    ``owner`` (same signature as ``SchedulerContext.owner``, e.g. ``fallback.owner_function(frozen_plan)``):
    fresh work has no stored ``wait_executor``, so who handles it comes from the candidate list; when it names
    the local executor and it is not its GPU turn the state is ``waiting_gpu``. Without it only stored
    reservations count."""
    now = time.time() if now is None else float(now)
    st = sp.derive_status(plan, play_pos, now, max_available)

    blocking_fail = None
    if st.blocked_playback:
        blocking_fail = next((f for f in st.failures if f.slice_index == st.blocking_slice
                              and f.stage == st.blocking_stage and f.track == plan.primary), None)
    elsewhere = [f for f in st.failures if f is not blocking_fail]

    view: Dict[str, Any] = {
        "v": VIEW_VERSION, "state": "queued", "progress": round(st.progress, 4),
        "covered_s": round(st.covered_s, 3), "slices_done": st.slices_done, "slices_total": st.slices_total,
        "open": st.open, "running": None, "wait": None, "stuck": None,
        "failures_elsewhere": len(elsewhere), "partial_stuck": False, "retryable": bool(st.failures),
    }
    local_owned = False
    if owner is not None and st.blocking_slice is not None and not (st.blocked_playback or st.running_executor):
        unit = next((x for x in plan.slices if x.index == st.blocking_slice), None)
        if unit is not None:
            local_owned = owner(st.blocking_stage, plan.primary, unit.get(st.blocking_stage, plan.primary)) == LOCAL_EXECUTOR
    block = ({"slice": st.blocking_slice, "stage": st.blocking_stage}
             if st.blocking_slice is not None else {"slice": None, "stage": None})

    if st.complete:
        view["state"] = "complete"
    elif st.blocked_playback:
        view["state"] = "stuck"
        view["stuck"] = (_stuck_info(blocking_fail, candidate_names) if blocking_fail else
                         {"slice": st.blocking_slice, "stage": st.blocking_stage, "track": plan.primary,
                          "kind": None, "exhausted": False, "candidates": []})
    elif st.running_executor:
        view["state"] = "running"
        view["running"] = {"executor": st.running_executor, **block,
                           "candidate": st.blocking_candidate,
                           "candidate_name": _name(candidate_names, st.blocking_candidate)}
    elif (st.waiting_quota or _in_grace(plan, st, now)) and _smoothed_running(view, plan, st, now, block, candidate_names):
        pass
    elif st.waiting_quota:
        rec = _wait_record(st, now)
        who = rec if rec is not None and rec.retry_at is not None and rec.retry_at > now else None
        cid = who.id if who else (rec.id if rec else st.blocking_candidate)
        is_quota = rec is None or rec.kind is None or rec.kind in QUOTA_WAIT_KINDS
        view["state"] = "waiting_quota" if is_quota else "retrying"
        view["wait"] = {"kind": "quota" if is_quota else "retry", **block, "retry_at": _round(st.next_retry_at),
                        **({} if is_quota else {"reason": rec.kind}),
                        "remaining_s": max(0.0, round(st.next_retry_at - now, 3)),
                        "candidate": cid, "candidate_name": _name(candidate_names, cid),
                        "also_waiting_download": bool(st.waiting_download)}
    elif (st.waiting_executor == LOCAL_EXECUTOR or local_owned) and gpu_turn is False:
        view["state"] = "waiting_gpu"
        view["wait"] = {"kind": "gpu", **block, "executor": LOCAL_EXECUTOR}
    elif st.waiting_executor:
        view["state"] = "waiting_executor"
        view["wait"] = {"kind": "executor", **block, "executor": st.waiting_executor,
                        "reclaim_at": _round(st.reclaim_at),
                        "reclaim_in_s": None if st.reclaim_at is None else max(0.0, round(st.reclaim_at - now, 3)),
                        "candidates": [_candidate_row(c, candidate_names) for c in st.blocking_tried]}
    elif st.waiting_download:
        view["state"] = "waiting_download"
        view["wait"] = {"kind": "download", **block}
    elif st.waiting_cut:
        view["state"] = "waiting_cut"
        view["wait"] = {"kind": "cut", **block}
    else:
        view["state"] = "queued"

    view["partial_stuck"] = bool(elsewhere) and view["state"] != "stuck"
    return view


def _unit_retry_at(plan: sp.SlicePlan, st: sp.PlanStatus) -> Optional[float]:
    unit = next((x for x in plan.slices if x.index == st.blocking_slice), None)
    return unit.get(st.blocking_stage, plan.primary).retry_at if unit is not None and st.blocking_stage else None


def _in_grace(plan: sp.SlicePlan, st: sp.PlanStatus, now: float) -> bool:
    """Pending blocking unit whose short wait ended a moment ago (the worker has not picked it up yet)."""
    retry_at = _unit_retry_at(plan, st)
    return (retry_at is not None and retry_at <= now < retry_at + SHORT_WAIT_S
            and not st.running_executor and not st.blocked_playback)


def _smoothed_running(view: Dict[str, Any], plan: sp.SlicePlan, st: sp.PlanStatus, now: float,
                      block: Mapping[str, Any], names: Optional[Mapping[str, str]]) -> bool:
    """A backoff of the blocking unit must not flip a busy item to waiting: other work running, or a short
    (pacing-sized) wait, keeps ``running``. Fills ``view`` and returns True in that case."""
    rec = _wait_record(st, now)
    is_quota = rec is None or rec.kind is None or rec.kind in QUOTA_WAIT_KINDS
    other = _other_running(plan, block)
    if other is not None:
        index, stage, state = other
        view["state"] = "running"
        view["running"] = {"executor": state.executor, "slice": index, "stage": stage,
                           "candidate": state.candidate, "candidate_name": _name(names, state.candidate),
                           "quota_wait": is_quota, "quota_retry_at": _round(st.next_retry_at) if is_quota else None}
        return True
    retry_at = _unit_retry_at(plan, st)
    if rec is not None and rec.at is not None and retry_at is not None and retry_at - rec.at <= SHORT_WAIT_S:
        view["state"] = "running"
        view["running"] = {"executor": rec.executor, **block, "candidate": rec.id, "candidate_name": _name(names, rec.id),
                           "quota_wait": is_quota and st.waiting_quota,
                           "quota_retry_at": _round(retry_at) if is_quota and st.waiting_quota else None}
        return True
    return False


def retry_failed(store_or_plan: Any, index: Optional[int] = None, stage: Optional[str] = None,
                 track: Optional[str] = None) -> int:
    """User retry: clear failed work (incl. its per-candidate ``tried``, so it restarts from the candidate
    list) and save. Given a PlanStore this happens under its lock and is saved to slice-plan.json; given a
    bare SlicePlan only the in-memory plan changes. Returns the number of cleared units."""
    store = store_or_plan if hasattr(store_or_plan, "plan") and hasattr(store_or_plan, "lock") else None
    if store is None:
        return store_or_plan.reset_failed(index, stage, track)
    with store.lock:
        if store.plan is None:
            return 0
        count = store.plan.reset_failed(index, stage, track)
        if count:
            store.save()
        return count
