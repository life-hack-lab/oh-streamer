"""Subtitle job status shown by the player (T3.3): parse the queue's `subtitle_status` message and build the text.

Stdlib only, no mpv. The queue side is queue/subtitle_player_status.py (message layout in queue/queue_ipc.py). The
wording follows queue/subtitle_view.py and queue_ui/app.js (subtitleViewDisplay): 語音辨識 / 翻譯, 本地 / 雲端,
reason kinds (rate_limit = 被限流 ...). Used by player/info_panel.py (panel rows) and the controller (OSD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

STAGE_NAMES = {"asr": "語音辨識", "mt": "翻譯"}
EXECUTOR_NAMES = {"local": "本地", "cloud": "雲端"}
REASONS = {
    "rate_limit": "被限流", "quota_daily": "今日額度用完", "network": "網路問題", "timeout": "逾時",
    "server": "服務暫時故障", "auth": "金鑰或權限錯誤", "bad_response": "回應格式錯誤",
    "content_blocked": "內容被拒絕", "unsupported": "不支援或環境不符", "error": "執行錯誤",
    "oom": "GPU 記憶體不足", "exhausted": "所有候選都無法使用",
}
PROBLEM_STATES = frozenset({"failed", "stuck"})
WAITING_STATES = frozenset({"waiting_quota", "retrying", "waiting_gpu", "waiting_executor", "waiting_download", "waiting_cut"})
LEAD_HEADLINE_S = 30.0          # the headline switches to "已領先" once the subtitles are this far ahead
GOOD, WARN, BAD, IDLE = "good", "warn", "bad", "idle"      # same levels as info_panel


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _opt_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


@dataclass
class SubtitleStatus:
    """A validated `subtitle_status` snapshot. ``received`` (the player's monotonic clock at arrival) lets the
    countdown in ``wait`` keep running between snapshots."""
    item_id: str
    state: str
    stage: Optional[str] = None
    executor: Optional[str] = None
    done: int = 0
    total: int = 0
    covered_s: Optional[float] = None
    offset: float = 0.0
    complete: bool = False
    wait: Optional[dict] = None
    stuck: Optional[dict] = None
    elsewhere: int = 0
    error: str = ""
    gpu_short: bool = False         # T4.3: local subtitles and the live restore cannot both keep up on this GPU
    received: float = field(default=0.0, compare=False)


def status_from_message(msg: Any) -> Optional[SubtitleStatus]:
    """Validate a queue message; None when it is not a usable `subtitle_status` (harmless to playback)."""
    if not isinstance(msg, dict) or msg.get("cmd") != "subtitle_status":
        return None
    raw_id = msg.get("item_id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
        return None
    state = msg.get("state")
    if not isinstance(state, str) or not state.strip():
        return None
    covered = _num(msg.get("covered_s"))
    if covered is not None and covered < 0:
        covered = None
    wait = msg.get("wait")
    if isinstance(wait, dict):
        remaining = _num(wait.get("remaining_s"))
        wait = {"kind": _opt_str(wait.get("kind")), "reason": _opt_str(wait.get("reason")),
                "executor": _opt_str(wait.get("executor")),
                "remaining_s": None if remaining is None else max(0.0, remaining)}
    else:
        wait = None
    stuck = msg.get("stuck")
    stuck = {"kind": _opt_str(stuck.get("kind")), "stage": _opt_str(stuck.get("stage"))} if isinstance(stuck, dict) else None

    def count(key):
        value = _num(msg.get(key))
        return max(0, int(value)) if value is not None else 0
    error = msg.get("error")
    return SubtitleStatus(
        item_id=str(raw_id).strip(), state=state.strip(), stage=_opt_str(msg.get("stage")),
        executor=_opt_str(msg.get("executor")), done=count("done"), total=count("total"), covered_s=covered,
        offset=_num(msg.get("offset")) or 0.0, complete=msg.get("complete") is True, wait=wait, stuck=stuck,
        elsewhere=count("elsewhere"), error=error.strip()[:200] if isinstance(error, str) else "",
        gpu_short=msg.get("gpu_short") is True)


GPU_SHORT_OSD = "這台電腦同時修復與產生本機字幕的算力不足，影片或字幕可能會跟不上：可暫停讓影片緩衝，或改用加入佇列"


def reason_text(kind: Optional[str]) -> str:
    return REASONS.get(kind or "", kind or "")


def _stage(st: SubtitleStatus, stage: Optional[str] = None) -> str:
    return STAGE_NAMES.get(stage or st.stage or "", "字幕")


def remaining_s(st: SubtitleStatus, now: float) -> Optional[int]:
    """Whole seconds left of the wait countdown (None without one), counted on from the snapshot's arrival."""
    left = (st.wait or {}).get("remaining_s")
    if left is None:
        return None
    return max(0, math.ceil(left - max(0.0, now - st.received)))


def _retry_in(st: SubtitleStatus, now: float) -> str:
    left = remaining_s(st, now)
    if left is None:
        return ""
    return "即將重試" if left <= 0 else f"{left} 秒後重試"


def state_text(st: SubtitleStatus, now: float = 0.0) -> str:
    """What the subtitle job is doing, without the 「字幕：」 prefix (the panel row carries that as its label)."""
    state, wait = st.state, st.wait or {}
    if state == "complete":
        text = "已完成"
    elif state == "failed":
        text = f"失敗（{st.error}）" if st.error else "失敗"
    elif state == "stuck":
        reason = reason_text((st.stuck or {}).get("kind"))
        text = f"{_stage(st, (st.stuck or {}).get('stage'))}卡住" + (f"（{reason}）" if reason else "")
    elif state == "running":
        where = EXECUTOR_NAMES.get(st.executor or "", "")
        text = f"{_stage(st)}中" + (f"（{where}）" if where else "")
        if st.total > 0:
            text += f" {min(st.done, st.total)}/{st.total} 段"
    elif state == "waiting_quota":
        text = "等待額度中" + (f"（{_retry_in(st, now)}）" if _retry_in(st, now) else "")
    elif state == "retrying":
        parts = [p for p in (_retry_in(st, now), reason_text(wait.get("reason"))) if p]
        text = "重試中" + (f"（{'，'.join(parts)}）" if parts else "")
    elif state == "waiting_gpu":
        text = "等待 GPU"
    elif state == "waiting_executor":
        where = EXECUTOR_NAMES.get(wait.get("executor") or "", "")
        text = f"已交給{where}備援，等待接手" if where else "已交給備援，等待接手"
    elif state == "waiting_cut":
        text = "準備切片"
    elif state == "waiting_download":
        text = "等待下載"
    elif state == "queued":
        text = "排隊中"
    elif state == "preparing":
        text = "準備中"
    elif state == "paused":
        text = "已暫停"
    else:
        text = "處理中"
    if st.elsewhere and state not in ("stuck", "failed", "complete"):
        text += f"；另有 {st.elsewhere} 段失敗"
    return text


def lead_s(st: SubtitleStatus, position: Any) -> Optional[float]:
    """Seconds of subtitles ready beyond the playback position (negative = the viewer is already past them)."""
    pos = _num(position)
    if pos is None or st.covered_s is None:
        return None
    return st.covered_s + st.offset - pos


def lead_text(st: SubtitleStatus, position: Any) -> str:
    """「12 分鐘」/「45 秒」/「尚未追上播放位置」; 「—」 without a position or coverage."""
    if st.complete or st.state == "complete":
        return "全部完成"
    lead = lead_s(st, position)
    if lead is None:
        return "—"
    if lead < 1:
        return "尚未追上播放位置"
    if lead >= 90:
        return f"{round(lead / 60)} 分鐘"
    return f"{int(lead)} 秒"


def headline(st: SubtitleStatus, position: Any = None, now: float = 0.0) -> str:
    """One line for the OSD / status: 「字幕：已領先 12 分鐘」, 「字幕：等待 GPU」, 「字幕：失敗（原因）」 ..."""
    lead = lead_s(st, position)
    if st.state not in PROBLEM_STATES and st.state != "complete" and lead is not None and lead >= LEAD_HEADLINE_S:
        return f"字幕：已領先 {lead_text(st, position)}"
    return f"字幕：{state_text(st, now)}"


def level(st: SubtitleStatus, position: Any = None) -> str:
    """Colour level of the state row."""
    if st.state in PROBLEM_STATES:
        return BAD
    if st.state == "complete":
        return GOOD
    if st.state in WAITING_STATES:
        return WARN
    if st.state == "running":
        lead = lead_s(st, position)
        return GOOD if lead is not None and lead >= 60 else WARN if lead is not None and lead < 10 else None
    return IDLE


def lead_level(st: SubtitleStatus, position: Any = None) -> Optional[str]:
    if st.state in PROBLEM_STATES:
        return None
    if st.state == "complete":
        return GOOD
    lead = lead_s(st, position)
    if lead is None:
        return None
    return GOOD if lead >= 60 else WARN if lead >= 10 else BAD


def problem_key(st: SubtitleStatus) -> Optional[tuple]:
    """Identity of a failure worth an OSD (None while things are fine); the controller shows each one once."""
    if st.state == "failed":
        return ("failed", st.error)
    if st.state == "stuck":
        return ("stuck", (st.stuck or {}).get("stage"), (st.stuck or {}).get("kind"))
    return None
