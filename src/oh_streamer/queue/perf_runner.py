"""
Performance check ("電腦效能檢測") for the queue service.

PerfRunner runs perf.bench.run_benchmark on a background thread, saves the profile and describes everything the queue
UI needs as one plain dict (`snapshot()`, merged into QueueService.get_state() as state["perf"]).

  * Never imports torch: perf.bench / perf.profile are stdlib-only in the parent process (the measurements run in
    child processes of their own); bench is still imported lazily, only when a run starts.
  * `holding` is a separate flag from the user's pause_all: while it is True the service stops its running worker (it
    goes back to `queued`) and starts no new one, then carries on by itself when the run ends (success, failure or
    cancel), without ever touching the user's pause_all / paused states.
  * Each state change calls `notify` (the service's `_changed`); the service already coalesces pushes to <= 2/s.
"""

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

BLOCKED_PLAYER = "播放器正在即時處理，請先關閉播放器再檢測"
CANCELLED_BY_PLAYER = "播放器開始即時處理，檢測已取消"
# table columns: source fps -> restoration_fps; "60r30" = a 60 fps source restored at 30 (player "最高 30fps")
MATRIX_KEYS = (("30", 30, "original"), ("60", 60, "original"), ("60r30", 60, "30"))
FINGERPRINT_TTL = 300.0
KINDS = ("quick", "standard")


def log(msg: str) -> None:
    print(f"[perf] {msg}", flush=True)


def _finite(value):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def current_fingerprint() -> dict:
    """Cheap fingerprint of this machine (no torch import): GPU name + driver via NVML, package versions via metadata."""
    from importlib import metadata

    def version(*names):
        for name in names:
            try:
                return metadata.version(name)
            except Exception:  # noqa: BLE001
                continue
        return None

    fp = {"gpu": None, "driver": None, "torch": version("torch"), "lada": version("lada")}
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            fp["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
            drv = pynvml.nvmlSystemGetDriverVersion()
            fp["driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - no NVIDIA driver / pynvml: GPU fields are simply not compared
        pass
    return fp


def _stage_label(msg: str) -> Optional[str]:
    """zh-TW label for bench's progress messages; None = keep the previous stage text."""
    msg = (msg or "").strip()
    if msg == "finished":
        return "完成"
    if re.match(r"\S+ done\b", msg):             # "restore done in 65 s (...)": keep the stage text
        return None
    m = re.match(r"(restore|detector|RIFE|encode|combined)\b\s*(\S*)", msg)
    if not m:
        return None
    kind, arg = m.groups()
    if kind == "detector":
        return f"量測偵測模型 {arg}".strip()
    if kind == "restore":
        return f"量測修復速度 {arg}".strip()
    if kind == "RIFE":
        return "量測補幀速度"
    if kind == "encode":
        return f"量測編碼速度 {arg}".strip()
    return f"綜合量測 {arg}".strip()


class PerfRunner:
    def __init__(self, *, settings_provider: Callable[[], dict], is_player_busy: Callable[[], bool],
                 notify: Callable[[], None] = lambda: None, profile_path=None,
                 run_benchmark: Optional[Callable] = None,
                 fingerprint_provider: Optional[Callable[[], dict]] = None,
                 notify_player: Callable[[], None] = lambda: None):
        self._settings = settings_provider
        self._busy = is_player_busy
        self._notify = notify
        self._notify_player = notify_player             # the player settings file was written: a running player re-reads it
        self._profile_path = Path(profile_path) if profile_path else None
        self._player_settings_path: Optional[Path] = None   # None = player_settings.DEFAULT_FILE (tests override)
        self._run_benchmark = run_benchmark
        self._fingerprint = fingerprint_provider or current_fingerprint
        self._fp_cache: Optional[tuple] = None          # (monotonic stamp, fingerprint) for the default provider

        self._lock = threading.RLock()
        self._obs_lock = threading.Lock()                # one calibration read-modify-write at a time
        self._thread: Optional[threading.Thread] = None
        self._cancel: Optional[threading.Event] = None
        self._hold = False
        self._running = False
        self._last: Optional[str] = None                 # outcome of the latest run: done | failed | cancelled
        self._kind: Optional[str] = None
        self._stage = ""
        self._progress = 0.0
        self._remaining: Optional[float] = None          # seconds left (bench's estimate), None = unknown yet
        self._error: Optional[str] = None
        self._cancel_note: Optional[str] = None
        self._profile_cache: Optional[tuple] = None      # (stat key, profile or None)
        self._profile_gen = 0                            # bumped whenever a different profile object is cached
        self._summary_cache: Optional[tuple] = None      # (key, summary)
        self._dismissed: Optional[bool] = None

    # ----------------------------------------------------------------------------------------------------- paths
    @property
    def _base_path(self) -> Path:
        """Lada's profile file; the other engines' files sit next to it (profile.path_for_engine)."""
        if self._profile_path is None:
            from oh_streamer.perf import profile
            self._profile_path = Path(profile.default_path())
        return self._profile_path

    @property
    def profile_path(self) -> Path:
        """The profile file of the engine in use (a profile of another engine never feeds this one's estimates)."""
        from oh_streamer.perf import profile
        return profile.path_for_engine(self._base_path, profile.current_engine())

    @property
    def _state_path(self) -> Path:
        return self._base_path.with_name("perf_state.json")

    @property
    def holding(self) -> bool:
        return self._hold

    @property
    def running(self) -> bool:
        return self._running

    # ----------------------------------------------------------------------------------------------------- profile
    def _load_profile(self) -> Optional[dict]:
        from oh_streamer.perf import profile
        path = self.profile_path
        try:
            st = path.stat()
            key = (str(path), st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        with self._lock:
            if self._profile_cache and self._profile_cache[0] == key:
                return self._profile_cache[1]
        data = profile.load(path) if key is not None else None
        if data is not None and (profile.validate(data) or profile.profile_engine(data) != profile.current_engine()):
            data = None
        with self._lock:
            self._profile_cache = (key, data)
            self._profile_gen += 1
        return data

    def _invalidate(self) -> None:
        with self._lock:
            self._profile_cache = None
            self._summary_cache = None

    def current_profile(self) -> tuple:
        """(generation, profile or None) for callers that derive things from the profile (ETA prior): the generation
        changes whenever a different profile (or a calibration update) was loaded."""
        p = self._load_profile()
        with self._lock:
            return self._profile_gen, p

    def record_observation(self, *, mode, height, observed_fps, density=None, detector=None, engine=None) -> bool:
        """Fold one worker throughput observation into the profile's calibration and save it. False (nothing
        written) while the performance check holds the machine, without a usable profile, for a stale profile
        (other GPU / driver / packages), invalid numbers, or an observation of another engine than the one in use
        (each engine calibrates its own profile; no engine = lada)."""
        from oh_streamer.perf import profile
        try:
            height, observed_fps = int(height), float(observed_fps)
            density = None if density is None else float(density)
        except (TypeError, ValueError):
            return False
        if (engine if isinstance(engine, str) and engine else profile.DEFAULT_ENGINE) != profile.current_engine():
            return False
        if mode not in ("off", "full") or height <= 0 or not math.isfinite(observed_fps) or observed_fps <= 0:
            return False
        if density is not None and (not math.isfinite(density) or density < 0):
            density = None
        with self._obs_lock:
            if self._hold or self._running:
                return False
            p = self._load_profile()
            if p is None or profile.stale_reasons(p, self._current_fingerprint()):
                return False
            new = profile.record_observation(p, mode=mode, height=height, observed_fps=observed_fps, density=density,
                                             detector=detector if isinstance(detector, str) else None)
            if new == p:                                  # outside the plausible range: nothing to learn
                return False
            try:
                profile.save(new, self.profile_path)
            except OSError as e:
                log(f"could not save calibration: {e!r}")
                return False
            self._invalidate()
        self._notify()
        return True

    def _current_fingerprint(self) -> dict:
        if self._fingerprint is not current_fingerprint:
            try:
                return dict(self._fingerprint() or {})
            except Exception:  # noqa: BLE001
                return {}
        now = time.monotonic()
        with self._lock:
            if self._fp_cache and now - self._fp_cache[0] < FINGERPRINT_TTL:
                return self._fp_cache[1]
        fp = current_fingerprint()
        with self._lock:
            self._fp_cache = (now, fp)
        return fp

    @staticmethod
    def _settings_mismatch(p: dict, cur: dict) -> list:
        from oh_streamer.perf import profile
        saved = p.get("settings") if isinstance(p.get("settings"), dict) else {}
        out = []
        det = cur.get("detection_model")
        if det and saved.get("detector") and str(det) not in profile.detectors(p):
            out.append(f"量測用 {saved.get('detector')}，目前設定 {det}")
        a, b = saved.get("restoration"), cur.get("restoration_model")
        if a and b and str(a) != str(b):
            out.append(f"量測用 {a}，目前設定 {b}")
        if not profile.covers_clip_length(p, cur.get("max_clip_length")):
            out.append(f"量測用 clip 長度 {profile.measured_clip_length(p)}，目前設定 {cur.get('max_clip_length')}")
        return out

    def _summary(self, p: dict, cur: dict) -> dict:
        from oh_streamer.perf import profile
        fp = self._current_fingerprint()
        key = (self._profile_gen, tuple(sorted((k, str(v)) for k, v in cur.items())),
               tuple(sorted((k, str(v)) for k, v in fp.items())))
        with self._lock:
            if self._summary_cache and self._summary_cache[0] == key:
                return self._summary_cache[1]
        fingerprint = p.get("fingerprint") if isinstance(p.get("fingerprint"), dict) else {}
        view = profile.for_detector(p, cur.get("detection_model"))
        conditions = p.get("conditions") if isinstance(p.get("conditions"), dict) else {}
        # every detection model the check measured, so the panel can switch between them and compare
        by_detector = []
        for det in profile.detectors(p):
            v = profile.for_detector(p, det)
            by_detector.append({"detector": det, "matrix": self._matrix(v),
                                "recommend": profile.recommend(v, 30)})
        summary = _finite({
            "created": p.get("created"),
            "gpu": fingerprint.get("gpu") or "",
            "duration_s": p.get("duration_s"),
            "on_ac_power": conditions.get("on_ac_power"),
            "warnings": [str(w) for w in (p.get("warnings") or [])],
            "stale": profile.stale_reasons(p, fp),
            "settings_mismatch": self._settings_mismatch(p, cur),
            "matrix": self._matrix(view),
            "recommend": profile.recommend(view, 30),
            "detector": (view.get("settings") or {}).get("detector"),
            "current_detector": cur.get("detection_model"),
            "by_detector": by_detector,
            "detector_advice": self._detector_advice(p),
        })
        with self._lock:
            self._summary_cache = (key, summary)
        return summary

    @staticmethod
    def _matrix(view: dict) -> dict:
        from oh_streamer.perf import profile
        return {key: profile.matrix(view, fps, rfps) for key, fps, rfps in MATRIX_KEYS}

    @staticmethod
    def _detector_advice(p: dict) -> dict:
        from oh_streamer.perf import profile
        r = profile.recommend_detector(p)
        return {"detector": r["detector"],
                "differs": [{"src_fps": f, "height": h, "mode": m, "levels": lv} for f, h, m, lv in r["differs"]]}

    # ----------------------------------------------------------------------------------------------------- apply plan
    def _player_file(self, settings_path=None) -> Path:
        from oh_streamer.player import player_settings as PS
        return Path(settings_path or self._player_settings_path or PS.DEFAULT_FILE)

    def apply_plan(self, summary: dict, settings_path=None) -> dict:
        """What "套用建議設定" would write to the player settings, item by item:
        {changes: [{key, label, from, to, reason}], live_height, prebuffer_height, error}.
        The detection model changes only when the check recommends one (profile.recommend_detector); interpolation
        follows that model's recommendation (auto where it interpolates enough, else off; off without RIFE)."""
        from oh_streamer.platform import platform_support
        from oh_streamer.player import player_settings as PS
        try:
            saved, _warn = PS.load_file(self._player_file(settings_path))
        except (OSError, ValueError) as e:
            return {"changes": [], "error": f"無法讀取播放器設定：{e}"}
        cur_det = saved.get("detection_model") or PS.DEFAULTS["detection_model"]
        cur_interp = saved.get("interpolation") or PS.DEFAULTS["interpolation"]
        cur_rfps = saved.get("restoration_fps") or PS.DEFAULTS.get("restoration_fps", "original")
        entries = {e["detector"]: e for e in summary.get("by_detector") or []}
        advice = (summary.get("detector_advice") or {}).get("detector")
        det = advice if advice in entries else cur_det
        rec = (entries.get(det) or {}).get("recommend") or summary.get("recommend") or {}
        try:
            rife_ok = bool(platform_support.capabilities().get("rife", True))
        except Exception:  # noqa: BLE001
            rife_ok = True
        interp = "auto" if (rec.get("auto_interp") and rife_ok) else "off"
        changes = []
        if advice in entries:
            changes.append({"key": "detection_model", "label": "播放器偵測模型", "from": cur_det, "to": advice,
                            "reason": "每種影片的判定（可即時／先緩衝／佇列）都跟較快的模型相同，改用偵測較準的"})
        frac = rec.get("auto_fraction")
        why = (f"只用多出來的速度補幀，不會拖慢；這台電腦約可補 {frac:.0%} 的格數" if interp == "auto"
               else "這台電腦不支援補幀（RIFE）" if not rife_ok and rec.get("auto_interp")
               else "多出來的速度不夠補幀" if rec.get("prebuffer_height") else "這台電腦無法即時修復，補幀也跟不上")
        changes.append({"key": "interpolation", "label": "播放器補幀", "from": cur_interp, "to": interp, "reason": why})
        if "restoration_fps" in PS.DEFAULTS:
            rfps = rec.get("restoration_fps") or "original"
            why = ("60 fps 影片原始幀率跟不上，降到 30 fps 後可以播放" + ("，自動補幀再補回 60 fps" if interp == "auto" else "")
                   + "；30 fps 以下的影片不受影響" if rfps == "30"
                   else "60 fps 影片用原始幀率就能播放，不必降幀" if rec.get("prebuffer_height")
                   else "降幀也跟不上，維持原始幀率")
            changes.append({"key": "restoration_fps", "label": "播放器修復幀率", "from": cur_rfps, "to": rfps,
                            "reason": why})
        return {"changes": changes, "player": {"detection_model": cur_det, "interpolation": cur_interp,
                                               "restoration_fps": cur_rfps},
                "live_height": rec.get("live_height"),
                "prebuffer_height": rec.get("prebuffer_height"), "error": None}

    # ----------------------------------------------------------------------------------------------------- prompt
    def _is_dismissed(self) -> bool:
        with self._lock:
            if self._dismissed is not None:
                return self._dismissed
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            value = bool(isinstance(data, dict) and data.get("prompt_dismissed"))
        except (OSError, ValueError):
            value = False
        with self._lock:
            self._dismissed = value
        return value

    def dismiss_prompt(self) -> dict:
        try:
            path = self._state_path
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                data = {}
            data["prompt_dismissed"] = True
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            return {"ok": False, "error": f"無法儲存：{e}"}
        with self._lock:
            self._dismissed = True
        self._notify()
        return {"ok": True}

    # ----------------------------------------------------------------------------------------------------- state
    def snapshot(self) -> dict:
        p = self._load_profile()
        cur = self._current_settings()
        busy = bool(self._busy())
        with self._lock:
            running = self._running
            if running:
                status = "running"
            elif self._last in ("failed", "cancelled"):
                status = self._last
            else:
                status = "done" if p is not None else "none"
            snap = {"status": status, "kind": self._kind if (running or self._last) else None,
                    "stage": self._stage if (running or self._last) else "",
                    "progress": (round(self._progress, 3) if running or status in ("failed", "cancelled")
                                 else (1.0 if status == "done" else 0.0)),
                    "remaining_s": (round(self._remaining) if running and self._remaining is not None else None),
                    "error": self._error,
                    "can_start": not running and not busy,
                    "blocked_reason": BLOCKED_PLAYER if (busy and not running) else None}
        snap["prompt"] = p is None and not self._is_dismissed()
        snap["profile"] = self._summary(p, cur) if p is not None else None
        snap["apply_plan"] = self.apply_plan(snap["profile"]) if p is not None else None
        return snap

    def _current_settings(self) -> dict:
        try:
            return dict(self._settings() or {})
        except Exception:  # noqa: BLE001
            return {}

    # ----------------------------------------------------------------------------------------------------- run
    def start(self, kind: str = "standard") -> dict:
        kind = kind if kind in KINDS else "standard"
        busy = bool(self._busy())
        cur = self._current_settings()
        with self._lock:
            if self._running:
                return {"ok": False, "error": "效能檢測進行中"}
            if busy:
                return {"ok": False, "error": BLOCKED_PLAYER}
            options = {"mode": "quick" if kind == "quick" else "default",
                       "detector": cur.get("detection_model"), "restoration": cur.get("restoration_model"),
                       "clip_length": cur.get("max_clip_length")}
            self._running = True
            self._hold = True
            self._kind, self._stage, self._progress, self._remaining = kind, "準備中", 0.0, None
            self._error = self._cancel_note = None
            self._last = None
            self._cancel = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(options, self._cancel), name="perf-check",
                                            daemon=True)
            self._thread.start()
        self._notify()
        return {"ok": True, "message": "已開始效能檢測，背景處理暫停中"}

    def cancel(self, note: Optional[str] = None) -> dict:
        with self._lock:
            if not self._running or self._cancel is None:
                return {"ok": True}
            if note:
                self._cancel_note = note
            self._stage = "正在取消…"
            self._cancel.set()
        self._notify()
        return {"ok": True, "message": "正在取消效能檢測"}

    def cancel_for_player(self) -> None:
        """The player started live processing: the measurement would be disturbed, stop it."""
        if self._running:
            self.cancel(CANCELLED_BY_PLAYER)

    def join(self, timeout: float = 15.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def _progress_cb(self, stage, fraction, remaining=None) -> None:
        label = _stage_label(str(stage))
        with self._lock:
            if self._cancel is not None and self._cancel.is_set():
                return
            if label:
                self._stage = label
            try:
                self._progress = max(self._progress, min(1.0, max(0.0, float(fraction))))
            except (TypeError, ValueError):
                pass
            try:
                self._remaining = None if remaining is None else max(0.0, float(remaining))
            except (TypeError, ValueError):
                self._remaining = None
        self._notify()

    def _run(self, options: dict, cancel: threading.Event) -> None:
        outcome, error = "failed", None
        try:
            run = self._run_benchmark
            if run is None:
                from oh_streamer.perf import bench          # lazy: keeps queue startup light
                run = bench.run_benchmark
            result = run(options, self._progress_cb, cancel)
            if cancel.is_set():
                outcome = "cancelled"
            else:
                from oh_streamer.perf import profile
                problems = profile.validate(result)
                if problems:
                    raise ValueError("無效的檢測結果：" + "; ".join(problems[:3]))
                profile.save(result, profile.path_for_engine(self._base_path, profile.profile_engine(result)))
                self._invalidate()
                outcome = "done"
        except BaseException as e:  # noqa: BLE001 - the hold must always be released
            if type(e).__name__ == "Cancelled" or cancel.is_set():
                outcome = "cancelled"
            else:
                outcome, error = "failed", f"{type(e).__name__}: {e}"
                log(f"benchmark failed: {error}")
        with self._lock:
            self._last = outcome
            self._error = error if outcome == "failed" else (self._cancel_note if outcome == "cancelled" else None)
            self._progress = 1.0 if outcome == "done" else self._progress
            self._stage = {"done": "完成", "cancelled": "已取消"}.get(outcome, self._stage)
            self._running = False
            self._hold = False
        self._notify()

    # ----------------------------------------------------------------------------------------------------- apply
    def apply(self, settings_path=None) -> dict:
        """Write apply_plan's changes to the player settings. Returns {ok, message|error, changed: [keys]}."""
        from oh_streamer.player import player_settings as PS
        p = self._load_profile()
        if p is None:
            return {"ok": False, "error": "尚無效能檢測結果，請先執行檢測"}
        plan = self.apply_plan(self._summary(p, self._current_settings()), settings_path)
        if plan.get("error"):
            return {"ok": False, "error": plan["error"]}
        path = self._player_file(settings_path)
        todo = {c["key"]: c["to"] for c in plan["changes"] if c["from"] != c["to"]}
        if todo:
            try:
                saved, _warn = PS.load_file(path)
                PS.save_file(path, PS.updated_file(saved, todo))
            except (OSError, ValueError) as e:
                return {"ok": False, "error": f"無法寫入播放器設定：{e}"}
            self._notify_player()
        names = {"off": "關", "auto": "自動", "original": "原始", "30": "最高 30fps"}
        parts = [f"{c['label']}「{names.get(c['to'], c['to'])}」" for c in plan["changes"] if c["key"] in todo]
        msg = ("已套用：" + "、".join(parts) + "。開著的播放器也會立即更新。") if parts else "播放器設定已經符合建議，沒有變更。"
        return {"ok": True, "message": msg, "changed": sorted(todo), "interpolation":
                next(c["to"] for c in plan["changes"] if c["key"] == "interpolation"),
                "live_height": plan.get("live_height")}
