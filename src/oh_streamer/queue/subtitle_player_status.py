"""Queue item subtitle state -> the compact `subtitle_status` message for the player (T3.3). Pure, stdlib only.

The player shows it in the 進階資訊 panel (and as an OSD when something goes wrong), so it only needs the few fields
below; the full `item.subtitle["view"]` (slice table summary, candidate lists, ...) stays in the queue. Wording is
built on the player side (player/subtitle_status.py), the same Taiwanese-Chinese terms as queue/subtitle_view.py.

Message (see queue_ipc.py): {"cmd": "subtitle_status", "item_id", "state", "stage", "executor", "done", "total",
"covered_s", "offset", "complete", "wait": {"kind", "reason", "remaining_s", "executor"} | None,
"stuck": {"kind", "stage"} | None, "elsewhere", "error"} plus ``"gpu_short": true`` (T4.3, only when true).

``state``: the view states (complete / stuck / running / waiting_quota / retrying / waiting_gpu / waiting_executor /
waiting_download / queued) plus ``preparing`` (no slice table yet), ``failed`` (the whole job failed, ``error`` says
why) and ``paused``. ``covered_s`` is the audio-clock end of the continuous coverage from the playback position and
``offset`` maps the audio clock onto the player clock (as in `subtitle_update`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

HIDDEN_STATUSES = frozenset({"cancelled", "skipped"})
SIGNATURE_DROP = ("remaining_s",)             # per-second countdowns never make a new message
LEAD_KEEP_S = 75.0                            # restore-lead samples older than this are dropped
LEAD_SPAN_S = 60.0                            # the restore lead must have been falling over at least this long ...
LEAD_DROP_S = 5.0                             # ... by more than this (first sample -> last sample)
LEAD_FRESH_S = 10.0                           # ... and the newest sample must be this recent


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def _int(value: Any) -> int:
    number = _num(value, 0.0)
    return max(0, int(number or 0))


def player_status(item_id: str, subtitle: Mapping[str, Any], *, offset: float = 0.0) -> Optional[Dict[str, Any]]:
    """The message for one item, or None when there is nothing to show (no subtitle job / cancelled / skipped)."""
    status = str(subtitle.get("status") or "")
    if not status or status in HIDDEN_STATUSES:
        return None
    view = subtitle.get("view") if isinstance(subtitle.get("view"), dict) else None
    msg: Dict[str, Any] = {
        "cmd": "subtitle_status", "item_id": str(item_id), "state": "preparing", "stage": None, "executor": None,
        "done": 0, "total": 0, "covered_s": None, "offset": _num(offset, 0.0), "complete": False,
        "wait": None, "stuck": None, "elsewhere": 0, "error": "",
    }
    if status == "failed":
        msg["state"] = "failed"
        msg["error"] = str(subtitle.get("error") or "")[:200]
        return msg
    if status == "paused":
        msg["state"] = "paused"
        return msg
    if view is None:
        return msg
    state = str(view.get("state") or "queued")
    msg["state"] = state
    msg["done"], msg["total"] = _int(view.get("slices_done")), _int(view.get("slices_total"))
    msg["covered_s"] = _num(view.get("covered_s"))
    msg["complete"] = state == "complete"
    msg["elsewhere"] = _int(view.get("failures_elsewhere"))
    running = view.get("running") if isinstance(view.get("running"), dict) else None
    if running is not None:
        msg["stage"] = running.get("stage") if isinstance(running.get("stage"), str) else None
        msg["executor"] = running.get("executor") if isinstance(running.get("executor"), str) else None
    wait = view.get("wait") if isinstance(view.get("wait"), dict) else None
    if wait is not None:
        reason = wait.get("reason")
        executor = wait.get("executor")
        remaining = wait.get("remaining_s")
        if remaining is None:
            remaining = wait.get("reclaim_in_s")
        msg["wait"] = {"kind": str(wait.get("kind") or ""), "reason": reason if isinstance(reason, str) else None,
                       "remaining_s": _num(remaining), "executor": executor if isinstance(executor, str) else None}
        if isinstance(wait.get("stage"), str):
            msg["stage"] = wait["stage"]
    stuck = view.get("stuck") if isinstance(view.get("stuck"), dict) else None
    if stuck is not None:
        msg["stuck"] = {"kind": stuck.get("kind") if isinstance(stuck.get("kind"), str) else None,
                        "stage": stuck.get("stage") if isinstance(stuck.get("stage"), str) else None}
        msg["stage"] = msg["stuck"]["stage"] or msg["stage"]
    return msg


def signature(msg: Optional[Mapping[str, Any]]) -> Any:
    """What makes a message worth sending again: the message without its per-second countdown."""
    if msg is None:
        return None
    out = dict(msg)
    wait = out.get("wait")
    if isinstance(wait, dict):
        out["wait"] = {k: v for k, v in wait.items() if k not in SIGNATURE_DROP}
    return out


def lead_falling(samples, now: float) -> bool:
    """T4.3: the player's restore lead (live report ahead_s) fell by more than LEAD_DROP_S between the oldest and the
    newest of ``samples`` [(t, ahead_s), ...] which span at least LEAD_SPAN_S, the newest one still fresh."""
    pts = [(t, a) for t, a in samples if now - t <= LEAD_KEEP_S]
    if len(pts) < 2:
        return False
    first, last = pts[0], pts[-1]
    return (now - last[0] <= LEAD_FRESH_S and last[0] - first[0] >= LEAD_SPAN_S
            and last[1] - first[1] < -LEAD_DROP_S)


def gpu_short(msg: Optional[Mapping[str, Any]], samples, now: float, position: Any, urgent_s: float,
              vram_short: bool = False) -> bool:
    """T4.3: this stream item's subtitles run on the local executor, the restore lead keeps falling and the subtitle
    lead (covered_s + offset - position, none covered = 0) is below ``urgent_s``: this computer cannot do both.
    vram_short: the executor repeatedly found too little free VRAM after the restore pause (a falling lead is then
    not required: the turns were given up before they could cost any lead)."""
    if not msg or msg.get("complete") or msg.get("state") in ("complete", "failed", "paused", "preparing"):
        return False
    wait = msg.get("wait") if isinstance(msg.get("wait"), dict) else {}
    if "local" not in (msg.get("executor"), wait.get("executor")):
        return False
    pos = _num(position)
    if pos is None or not (vram_short or lead_falling(samples, now)):
        return False
    covered = _num(msg.get("covered_s"))
    lead = 0.0 if covered is None else covered + (_num(msg.get("offset"), 0.0) or 0.0) - pos
    return lead < urgent_s
