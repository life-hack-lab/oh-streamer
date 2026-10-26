"""Can this computer restore live AND make local subtitles while streaming? (plan 7, T4.3)

Pure functions + one tiny JSON store; stdlib only, no torch, no GPU.

- `effective_ratio` / `switch_cost_per_media_s` / `verdict_with_subtitles`: the live / prebuffer / queue level
  of the performance check, recomputed with the GPU seconds local subtitles take away from restoration.
- `vram_fits`: restoration stays loaded while paused (T4.0), so ASR / translation must fit beside it.
- `RestoreVramMemory`: the VRAM the paused restoration really still holds per card (player live report
  restore_pause.reserved_mib; recent larger value); `held_restore_mib` gives it to `vram_fits`.
- `SubtitleSpeedMemory`: the real subtitle speed per card (wall seconds per media second, EWMA), learned from
  finished local batches; `subtitle_seconds_per_media_second` prefers it over the built-in default.
- `SwitchCostMemory`: the measured model load times (ASR, translation) and the restoration restart per card (EWMA);
  `switch_costs` gives (load_s, restart_s) for the turn budget and for `switch_cost_per_media_s`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from oh_streamer.perf import profile as _profile
from oh_streamer.platform import platform_support
from oh_streamer.queue import gpu_turns as gt

# T4.0: after frames, session kept (restoration VRAM still held while restoration is paused).
PAUSED_RESTORE_MIB = 1232.0
SLICE_MEDIA_S = 300.0            # a subtitle slice is 5 minutes of media
MIN_SAMPLE_MEDIA_S = 30.0        # shorter batches say little about the speed (model load dominates)
EWMA_ALPHA = 0.3
RESTORE_VRAM_KEEP = 8            # RestoreVramMemory keeps this many recent samples per card
MAX_SAMPLE_MIB = 65536.0         # anything larger is not a VRAM figure


def _finite_pos(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def effective_ratio(r: Any, s: Any, c: Any) -> Optional[float]:
    """r' = 1 / (1/r + s + c): restoration speed (x real time) once every media second also costs s GPU seconds
    of subtitles and c seconds of switching. None for an unknown / zero / infinite r; bad s or c count as 0."""
    rr = _finite_pos(r)
    if rr is None:
        return None
    ss = _finite_pos(s) or 0.0
    cc = _finite_pos(c) or 0.0
    return 1.0 / (1.0 / rr + ss + cc)


def switch_cost_per_media_s(cfg: Optional[gt.TurnConfig] = None, load_s: Optional[float] = None,
                            restart_s: Optional[float] = None) -> float:
    """Restore<->subtitle switch cost spread over the media seconds of one batch. load_s / restart_s: learned values
    (switch_costs) that replace the config's."""
    cfg = cfg or gt.TurnConfig()
    # media_turns (stream items): a steady-state turn is turn_max_s of media; otherwise batch_slices slices
    media = cfg.turn_max_s if cfg.media_turns else max(1, int(cfg.batch_slices)) * SLICE_MEDIA_S
    load = cfg.subtitle_load_s if load_s is None else load_s
    restart = cfg.restore_restart_s if restart_s is None else restart_s
    return (load + restart) / max(1.0, float(media))


def verdict_with_subtitles(base_verdict: Mapping[str, Any], s: Any, c: Any) -> Dict[str, Any]:
    """perf.profile.verdict() result with local subtitles counted in: typical / heavy ratios become r' and the
    same level rule (profile.level_for) is applied. Extra keys: level_no_subtitles, ratio_*_no_subtitles."""
    rt0, rh0 = base_verdict.get("ratio_typical"), base_verdict.get("ratio_heavy")
    rt, rh = effective_ratio(rt0, s, c), effective_ratio(rh0, s, c)
    level, pre = _profile.level_for(rt, rh)
    out = dict(base_verdict)
    out.update({"level": level, "ratio_typical": rt, "ratio_heavy": rh, "prebuffer_s": pre,
                "level_no_subtitles": base_verdict.get("level"),
                "ratio_typical_no_subtitles": rt0, "ratio_heavy_no_subtitles": rh0})
    return out


def vram_fits(total_mib: Any, held_restore_mib: float = PAUSED_RESTORE_MIB, asr_mode: str = "dual",
              requirements: Optional[Mapping[str, float]] = None) -> Tuple[bool, float]:
    """(fits, need_mib): need = VRAM kept by paused restoration + max(ASR, translation) + margin.
    Unknown total VRAM does not block: (True, need) - the runtime guard (OOM) still protects."""
    req = dict(gt.VRAM_REQUIREMENTS_MIB)
    req.update(requirements or {})
    asr = req.get(f"asr_{asr_mode}", req.get("asr_dual", 0.0))
    need = float(held_restore_mib) + max(float(asr), float(req["mt_sakura_7b"])) + float(req["margin"])
    total = _finite_pos(total_mib)
    if total is None:
        return True, need
    return total >= need, need


# --------------------------------------------------------------------------------------------------
# Learned subtitle speed per card
# --------------------------------------------------------------------------------------------------

def default_speed_path() -> Path:
    base = platform_support.data_dir()
    return base / "subtitle-local-speed.json"


@dataclass
class SubtitleSpeedMemory:
    """{gpu_key: {"seconds_per_media_second", "count", "at"}}: EWMA (alpha 0.3) of GPU wall seconds per media
    second of finished local subtitle batches (model load included)."""
    path: Path = field(default_factory=default_speed_path)

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str) -> Optional[float]:
        row = self._load().get(key)
        v = _finite_pos(row.get("seconds_per_media_second")) if isinstance(row, dict) else None
        return v

    def count(self, key: str) -> int:
        row = self._load().get(key)
        try:
            return int(row.get("count", 0)) if isinstance(row, dict) else 0
        except (TypeError, ValueError):
            return 0

    def record(self, key: str, wall_s: Any, media_s: Any, at: Optional[float] = None) -> Optional[float]:
        """Fold one sample in; returns the new value, or None when the sample was ignored."""
        wall, media = _finite_pos(wall_s), _finite_pos(media_s)
        if wall is None or media is None or media < MIN_SAMPLE_MEDIA_S or not key:
            return None
        from oh_streamer.subtitles import runtime
        sample = wall / media
        data = self._load()
        old = _finite_pos((data.get(key) or {}).get("seconds_per_media_second")) if isinstance(data.get(key), dict) else None
        value = sample if old is None else EWMA_ALPHA * sample + (1.0 - EWMA_ALPHA) * old
        data[key] = {"seconds_per_media_second": round(value, 5), "count": self.count(key) + 1,
                     "at": time.time() if at is None else at}
        runtime.atomic_write_json(Path(self.path), data)
        return value


def subtitle_seconds_per_media_second(key: str, memory: Optional[SubtitleSpeedMemory] = None,
                                      cfg: Optional[gt.TurnConfig] = None) -> float:
    """The learned speed of this card, else the built-in default of the turn config."""
    cfg = cfg or gt.TurnConfig()
    learned = None
    if memory is not None and key:
        try:
            learned = memory.get(key)
        except Exception:
            learned = None
    return learned if learned is not None else float(cfg.subtitle_seconds_per_media_second)


# --------------------------------------------------------------------------------------------------
# Learned VRAM of the paused restoration per card
# --------------------------------------------------------------------------------------------------

def default_restore_vram_path() -> Path:
    return default_speed_path().with_name("subtitle-restore-vram.json")


@dataclass
class RestoreVramMemory:
    """{gpu_key: {"samples": [mib, ...last RESTORE_VRAM_KEEP], "count", "at"}}: the VRAM (MiB) the player's paused
    restoration still held (torch reserved after empty_cache), reported in the live report. `get` answers the
    recent larger value (max of the kept samples), so one lucky low reading cannot understate the need."""
    path: Path = field(default_factory=default_restore_vram_path)

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _samples(self, key: str) -> list:
        row = self._load().get(key)
        raw = row.get("samples") if isinstance(row, dict) else None
        out = [_finite_pos(x) for x in raw] if isinstance(raw, list) else []
        return [x for x in out if x is not None and x <= MAX_SAMPLE_MIB]

    def get(self, key: str) -> Optional[float]:
        samples = self._samples(key)
        return max(samples) if samples else None

    def count(self, key: str) -> int:
        row = self._load().get(key)
        try:
            return int(row.get("count", 0)) if isinstance(row, dict) else 0
        except (TypeError, ValueError):
            return 0

    def record(self, key: str, mib: Any, at: Optional[float] = None) -> Optional[float]:
        """Add one sample; returns the new recent larger value, or None when the sample was ignored."""
        value = _finite_pos(mib)
        if value is None or value > MAX_SAMPLE_MIB or not key:
            return None
        from oh_streamer.subtitles import runtime
        data = self._load()
        samples = (self._samples(key) + [round(value, 1)])[-RESTORE_VRAM_KEEP:]
        data[key] = {"samples": samples, "count": self.count(key) + 1, "at": time.time() if at is None else at}
        runtime.atomic_write_json(Path(self.path), data)
        return max(samples)


def held_restore_mib(key: str, memory: Optional[RestoreVramMemory] = None) -> Tuple[float, bool]:
    """(MiB the paused restoration holds on this card, learned?): the recent larger reported value, else the fixed
    PAUSED_RESTORE_MIB. Never raises."""
    learned = None
    if memory is not None and key:
        try:
            learned = memory.get(key)
        except Exception:
            learned = None
    return (learned, True) if learned is not None else (PAUSED_RESTORE_MIB, False)


# --------------------------------------------------------------------------------------------------
# Learned switch costs per card (model load, restoration restart)
# --------------------------------------------------------------------------------------------------

COST_ASR_LOAD = "asr_load_s"            # ASR model load of a turn (all its ASR children)
COST_MT_LOAD = "mt_load_s"              # translation model (llama-server) load of a turn
COST_RESTART = "restore_restart_s"      # restore_resume -> restoration output flowing again
MAX_COST_S = 600.0                      # anything longer is a stuck load, not a cost


def default_cost_path() -> Path:
    return default_speed_path().with_name("subtitle-switch-costs.json")


@dataclass
class SwitchCostMemory:
    """{gpu_key: {field: {"value", "count", "at"}}}: EWMA (alpha 0.3) of the measured seconds of the switching costs
    of local stream subtitle turns. Fields: COST_ASR_LOAD, COST_MT_LOAD (the executor, from the model children's
    "ready" events) and COST_RESTART (the stream host, resume -> restoration speed > 0 in the live report)."""
    path: Path = field(default_factory=default_cost_path)

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _row(self, key: str, name: str) -> Dict[str, Any]:
        card = self._load().get(key)
        row = card.get(name) if isinstance(card, dict) else None
        return row if isinstance(row, dict) else {}

    def get(self, key: str, name: str) -> Optional[float]:
        v = _finite_pos(self._row(key, name).get("value"))
        return v if v is not None and v <= MAX_COST_S else None

    def count(self, key: str, name: str) -> int:
        try:
            return int(self._row(key, name).get("count", 0))
        except (TypeError, ValueError):
            return 0

    def record(self, key: str, name: str, seconds: Any, at: Optional[float] = None) -> Optional[float]:
        """Fold one measurement in; returns the new value, or None when it was ignored."""
        sample = _finite_pos(seconds)
        if sample is None or sample > MAX_COST_S or not key or not name:
            return None
        from oh_streamer.subtitles import runtime
        data = self._load()
        card = data.get(key) if isinstance(data.get(key), dict) else {}
        old = self.get(key, name)
        value = sample if old is None else EWMA_ALPHA * sample + (1.0 - EWMA_ALPHA) * old
        card[name] = {"value": round(value, 3), "count": self.count(key, name) + 1,
                      "at": time.time() if at is None else at}
        data[key] = card
        runtime.atomic_write_json(Path(self.path), data)
        return value


def default_load_split(cfg: Optional[gt.TurnConfig] = None) -> Tuple[float, float]:
    """(asr_load_s, mt_load_s) defaults: the config's mt_load_s is the translation share of subtitle_load_s."""
    cfg = cfg or gt.TurnConfig()
    mt = max(0.0, min(float(cfg.mt_load_s), float(cfg.subtitle_load_s)))
    return max(0.0, float(cfg.subtitle_load_s) - mt), mt


def learned_costs(key: str, memory: Optional[SwitchCostMemory] = None,
                  cfg: Optional[gt.TurnConfig] = None) -> Dict[str, Any]:
    """{"asr_load_s", "mt_load_s", "restart_s", "learned"}: the card's learned values, else the config defaults.
    Never raises."""
    cfg = cfg or gt.TurnConfig()
    asr, mt = default_load_split(cfg)
    restart = float(cfg.restore_restart_s)
    learned = False
    if memory is not None and key:
        try:
            for name in (COST_ASR_LOAD, COST_MT_LOAD, COST_RESTART):
                v = memory.get(key, name)
                if v is None:
                    continue
                learned = True
                if name == COST_ASR_LOAD:
                    asr = v
                elif name == COST_MT_LOAD:
                    mt = v
                else:
                    restart = v
        except Exception:
            pass
    return {"asr_load_s": asr, "mt_load_s": mt, "restart_s": restart, "learned": learned}


def switch_costs(key: str, memory: Optional[SwitchCostMemory] = None,
                 cfg: Optional[gt.TurnConfig] = None) -> Tuple[float, float, bool]:
    """(load_s, restart_s, learned?) of one turn: ASR + translation load and the restoration restart."""
    c = learned_costs(key, memory, cfg)
    return c["asr_load_s"] + c["mt_load_s"], c["restart_s"], c["learned"]
