"""GPU sharing policy of the local subtitle executor.
Pure functions + one tiny JSON store; no GPU, no queue service.

- `decide_gpu_turn`: interleave mode, who gets the GPU next (restore or N subtitle slices).
- `choose_resource_mode`: setting auto / interleave / parallel -> the mode to run; auto = VRAM estimate
  (VRAM_REQUIREMENTS_MIB, overridable by the performance check) unless this card was demoted before.
- `parallel_guard`: runtime guard of parallel mode (CUDA OOM, or restore falling behind playback while watched).
- `GpuModeMemory`: per-card record of a demotion (key: GPU name + total VRAM) so the next run interleaves.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from oh_streamer.platform import platform_support

RESTORE = "restore"
SUBTITLES = "subtitles"

MODE_AUTO = "auto"
MODE_INTERLEAVE = "interleave"
MODE_PARALLEL = "parallel"
RESOURCE_MODES = (MODE_AUTO, MODE_INTERLEAVE, MODE_PARALLEL)

# T2 (design 7.1) peaks in MiB over the idle baseline; "margin" = player + desktop. The performance check may
# override any entry (choose_resource_mode(requirements=...)).
VRAM_REQUIREMENTS_MIB: Dict[str, float] = {
    "restore": 2910.0,
    "asr_single": 4801.0,
    "asr_dual": 4801.0,          # the two models load one after the other, not together
    "mt_sakura_7b": 4489.0,
    "margin": 1536.0,
}


# --------------------------------------------------------------------------------------------------
# Interleave turns
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnConfig:
    """Thresholds of the interleave rule (design 7.2 values; all seconds of media)."""
    urgent_s: float = 300.0                  # watched: sub_lead below this -> subtitles now, 1 slice
    low_watermark_s: float = 900.0           # watched: sub_lead below this -> subtitles if restore is safe
    restore_safe_s: float = 300.0            # restore_lead needed to leave restore for a whole batch
    unwatched_restore_ahead_s: float = 300.0  # not watched: restore may lead subtitles by this much
    urgent_min_restore_s: float = 150.0      # watched: no urgent switch while restore_lead is below this
                                             # (a stalled video is worse than a missing subtitle)
    batch_slices: int = 2                    # N (T1: switch cost ~25 s vs ~118 s per 5-minute slice)
    urgent_batch_slices: int = 1
    subtitle_seconds_per_media_second: float = 0.65  # conservative T3/M3 dual-ASR + MT throughput
    subtitle_load_s: float = 60.0                   # cold model load / audio preparation
    restore_restart_s: float = 30.0                 # M3 observed up to 26.6 s
    playback_reserve_s: float = 30.0                # restored video left after restart
    # Stream items (media_turns): a turn is sized in media seconds instead of slices (plan 7, T4.2/T4.3 fixes).
    media_turns: bool = False                       # True = size turns by seconds (set by stream_turn_config)
    turn_first_s: float = 60.0                      # first turn after start / interruption
    turn_growth: float = 2.0                        # each finished turn multiplies the size ...
    turn_max_s: float = 300.0                       # ... up to this many media seconds
    turn_min_s: float = 60.0                        # a turn is shrunk to fit the restore lead, but not below this
    # media_turns, T4.2/T4.3 fix 4: free VRAM is measured (in a subtitle child) after the pause, before a model loads.
    vram_margin_mib: float = 512.0                  # free VRAM must be >= the model's need + this
    vram_wait_s: float = 10.0                       # re-measure this long (the player's in-flight clip may finish)
    vram_retry_s: float = 60.0                      # after a short turn, no new pause for this long (not an error)
    # media_turns, second round fixes 2-3: mt_load_s = the translation share of subtitle_load_s (default split of the
    # learned load); mt_seconds_per_media_second = translation speed; after ASR the turn goes on with MT unless the
    # restore lead after it would drop below mt_min_restore_s.
    mt_load_s: float = 10.0
    mt_seconds_per_media_second: float = 0.2        # warm Sakura 7B: ~5-6 s per ~30 s slice (P4 log)
    mt_min_restore_s: float = 60.0

    @classmethod
    def from_settings(cls, settings: Optional[Mapping[str, Any]] = None) -> "TurnConfig":
        """Overrides from settings keys subtitle_local_<field> (e.g. subtitle_local_urgent_s)."""
        settings = settings or {}
        values = {}
        for name, default in asdict(cls()).items():
            raw = settings.get(f"subtitle_local_{name}")
            if raw is not None:
                values[name] = type(default)(raw)
        return cls(**values)


@dataclass(frozen=True)
class TurnState:
    sub_lead_s: float                        # subtitles finished continuously from the play position
    restore_lead_s: float                    # same for restoration (inf = restoration finished)
    watched: bool = False
    subtitle_work: bool = True               # the local executor has a task it can run now
    restore_work: bool = True                # restoration has something to do (False: done / waiting)


@dataclass(frozen=True)
class GpuTurn:
    kind: str                                # RESTORE | SUBTITLES
    slices: int = 0
    reason: str = ""
    urgent: bool = False


def decide_gpu_turn(state: TurnState, config: Optional[TurnConfig] = None) -> GpuTurn:
    """Interleave rule (design 4.5 / 7.2):

    | condition (first match)                                      | turn                 |
    |--------------------------------------------------------------|----------------------|
    | no local subtitle task can run                               | restore              |
    | restoration has nothing to do                                | subtitles(N)         |
    | watched, sub_lead < urgent, restore_lead < urgent_min_restore | restore              |
    | watched, sub_lead < urgent                                   | subtitles(1), urgent |
    | watched, sub_lead < low watermark, restore_lead >= safe      | subtitles(N)         |
    | watched, otherwise                                           | restore              |
    | not watched, restore_lead - sub_lead >= unwatched ahead      | subtitles(N)         |
    | not watched, otherwise                                       | restore              |
    """
    cfg = config or TurnConfig()
    n = max(1, int(cfg.batch_slices))
    if not state.subtitle_work:
        return GpuTurn(RESTORE, reason="no_subtitle_work")
    restore_lead = state.restore_lead_s
    if not state.restore_work or restore_lead == math.inf:
        return GpuTurn(SUBTITLES, n, "restore_idle")
    if state.watched:
        if state.sub_lead_s < cfg.urgent_s:
            if restore_lead < cfg.urgent_min_restore_s:
                return GpuTurn(RESTORE, reason="urgent_restore_low")
            return GpuTurn(SUBTITLES, max(1, int(cfg.urgent_batch_slices)), "urgent", urgent=True)
        if state.sub_lead_s < cfg.low_watermark_s and restore_lead >= cfg.restore_safe_s:
            return GpuTurn(SUBTITLES, n, "low_watermark")
        return GpuTurn(RESTORE, reason="watched_restore")
    if restore_lead - state.sub_lead_s >= cfg.unwatched_restore_ahead_s:
        return GpuTurn(SUBTITLES, n, "restore_ahead")
    return GpuTurn(RESTORE, reason="unwatched_restore")


# --------------------------------------------------------------------------------------------------
# Resource mode (4.5.1, D6)
# --------------------------------------------------------------------------------------------------

def vram_needed_mib(asr_mode: str = "dual", requirements: Optional[Mapping[str, float]] = None) -> float:
    """restore + max(ASR, translation) + margin (subtitle ASR and translation never run at the same time)."""
    req = dict(VRAM_REQUIREMENTS_MIB)
    req.update(requirements or {})
    asr = req.get(f"asr_{asr_mode}", req.get("asr_dual", 0.0))
    return float(req["restore"]) + max(float(asr), float(req["mt_sakura_7b"])) + float(req["margin"])


def gpu_key(name: Optional[str], total_mib: Optional[float]) -> str:
    """Identity of a card for the persistent record: name + total VRAM (rounded to 64 MiB)."""
    total = "?" if not total_mib else str(int(round(float(total_mib) / 64.0) * 64))
    return f"{(name or 'unknown').strip()}|{total}"


def choose_resource_mode(setting: str, total_mib: Optional[float], *, asr_mode: str = "dual",
                         requirements: Optional[Mapping[str, float]] = None,
                         remembered: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    """(mode, reason). Explicit interleave / parallel win. auto: a remembered demotion of this card ->
    interleave; unknown VRAM -> interleave (safe); total >= vram_needed_mib -> parallel; else interleave."""
    setting = (setting or MODE_AUTO).lower()
    if setting == MODE_INTERLEAVE:
        return MODE_INTERLEAVE, "setting"
    if setting == MODE_PARALLEL:
        return MODE_PARALLEL, "setting"
    if remembered and remembered.get("mode") == MODE_INTERLEAVE:
        return MODE_INTERLEAVE, f"remembered:{remembered.get('reason', '')}"
    if not total_mib:
        return MODE_INTERLEAVE, "vram_unknown"
    need = vram_needed_mib(asr_mode, requirements)
    if float(total_mib) >= need:
        return MODE_PARALLEL, f"vram_fits:{int(total_mib)}>={int(need)}"
    return MODE_INTERLEAVE, f"vram_short:{int(total_mib)}<{int(need)}"


@dataclass(frozen=True)
class GuardInput:
    oom: bool = False                     # a CUDA out-of-memory happened (subtitle or restore side)
    watched: bool = False
    restore_rate: Optional[float] = None  # media seconds restored per wall second (None = unknown)
    restore_lead_s: float = math.inf
    playback_rate: float = 1.0            # media seconds playback consumes per wall second


def parallel_guard(sample: GuardInput, config: Optional[TurnConfig] = None) -> Optional[str]:
    """Reason to leave parallel mode (D6), or None: 'oom'; or 'restore_too_slow' = watched, restoration slower
    than playback and restore_lead below the safe amount (playback is about to stall). No fps-% threshold."""
    cfg = config or TurnConfig()
    if sample.oom:
        return "oom"
    if (sample.watched and sample.restore_rate is not None and sample.restore_rate < sample.playback_rate
            and sample.restore_lead_s < cfg.restore_safe_s):
        return "restore_too_slow"
    return None


_OOM_PATTERNS = re.compile(
    r"out of memory|outofmemoryerror|cuda_error_out_of_memory|cudaErrorMemoryAllocation|"
    r"failed to allocate .*buffer|cudaMalloc failed|unable to allocate .*CUDA", re.I)


def is_cuda_oom(text: Any) -> bool:
    """Does an error text (exception, child log tail, llama-server log) say CUDA ran out of memory?"""
    return bool(text) and bool(_OOM_PATTERNS.search(str(text)))


# --------------------------------------------------------------------------------------------------
# Per-card memory
# --------------------------------------------------------------------------------------------------

def default_memory_path() -> Path:
    base = platform_support.data_dir()
    return base / "gpu-resource-modes.json"


@dataclass
class GpuModeMemory:
    """{gpu_key: {"mode": "interleave", "reason", "detail", "at"}} in a small JSON file. A demotion is kept
    until forget() (e.g. the performance check measures the card again)."""
    path: Path = field(default_factory=default_memory_path)

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self._load().get(key)
        return dict(row) if isinstance(row, dict) else None

    def remember(self, key: str, mode: str, reason: str, detail: str = "", at: Optional[float] = None) -> None:
        from oh_streamer.subtitles import runtime
        data = self._load()
        data[key] = {"mode": mode, "reason": reason, "detail": str(detail)[:300],
                     "at": time.time() if at is None else at}
        runtime.atomic_write_json(Path(self.path), data)

    def forget(self, key: str) -> None:
        from oh_streamer.subtitles import runtime
        data = self._load()
        if data.pop(key, None) is not None:
            runtime.atomic_write_json(Path(self.path), data)
