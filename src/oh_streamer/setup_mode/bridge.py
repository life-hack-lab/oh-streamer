r"""設定模式的 pywebview js_api 後端。不 import webview，測試可以直接建立 SetupApi。

前端（resources/setup_ui）透過 window.pywebview.api 呼叫下面六個方法；每個方法都回傳可 JSON 序列化的 dict，
而且不會把例外丟到前端（一律包成 ok=False 與 message）。

    detect()             偵測顯示卡、磁碟空間、WebView2、目前的環境狀態
    start_install(f, o)  開一條背景執行緒跑 runtime_env.provision（新環境資料夾：基礎環境 -> GPU 版 torch -> 自我檢查 ->
                         寫 current.json -> 刪除舊環境）。有程式正在用環境時回 reason="in_use"；空間放不下新舊兩份、
                         但先刪舊環境就夠時回 reason="low_space"，使用者同意後再以 o={"delete_old_first": True} 呼叫
    poll()               目前階段、最後 80 行 uv 輸出、訊息、顯示卡名稱、下載進度（phase_progress，量不到時為 None）
    cancel()             取消進行中的安裝
    export_diagnostics() 匯出診斷包 zip
    launch_main()        環境就緒後啟動主程式，並關掉設定視窗

pywebview 會遞迴掃描 js_api 物件的所有「不以底線開頭」的屬性並暴露給 JavaScript，所以這個類別的狀態全部用底線開頭，
公開屬性只有六個 API 方法與 window（__main__ 設定的視窗；用 __dir__ 從掃描中隱藏，免得整個 Window 物件被暴露）。
"""

from __future__ import annotations

import copy
import shutil
import threading
from collections import deque
from pathlib import Path

from oh_streamer.branding import APP_NAME
from oh_streamer.platform import platform_support
from oh_streamer.runtime_env import (diagnostics, env_cleanup, env_guard, flavors, gpu_detect, launch_env, layout,
                                     preflight, provision)

ESTIMATED_DOWNLOAD_GB = 4.0                                 # torch 約 3.2 GiB + 基礎環境約 0.8 GB
REQUIRED_DISK_GB = 9.0                                      # 安裝中的峰值：環境＋快取約 7.8 GB（刪 torch 靜態庫前），再留餘裕
REUSE_DISK_GB = 1.0                                         # 同一份 uv.lock 的同一個版本重裝：檔案都從快取硬連結，幾乎不佔新空間
LINE_LIMIT = 80                                             # poll() 回傳的 uv 輸出行數上限
_PHASE_MESSAGES = {
    "base": "正在安裝基礎環境…",
    "gpu": "正在下載並安裝 GPU 版（約 3 GB，請保持網路連線）…",
    "selfcheck": "正在檢查顯示卡能否正常運算…",
    "cleanup": "顯示卡測試通過，正在刪除舊環境、釋放磁碟空間…",
}


def _gpu_dict(gpu) -> dict:
    cc = gpu.compute_capability
    return {
        "name": gpu.name,
        "vendor": gpu.vendor,
        "driver_version": gpu.driver_version,
        "compute_capability": f"{cc[0]}.{cc[1]}" if cc else None,
    }


def _disk_free_gb(path) -> float | None:
    """安裝資料夾所在磁碟的可用空間（GiB，一位小數）。資料夾還不存在時改量最近的上層資料夾。"""
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 2**30, 1)
    except OSError:
        return None


def _progress_dict(snapshot) -> dict | None:
    """poll() 的 phase_progress：{downloaded_bytes, total_bytes, speed_bps, eta_s}。
    total_bytes > 0 才有意義；speed_bps／eta_s 還量不出來時是 None。量不到（或格式不對）就整個是 None，前端改顯示不定進度。"""
    if not isinstance(snapshot, dict):
        return None
    try:
        total = int(snapshot["total_bytes"])
        done = min(total, max(0, int(snapshot["downloaded_bytes"])))
    except (KeyError, TypeError, ValueError):
        return None
    if total <= 0:
        return None
    speed, eta = snapshot.get("speed_bps"), snapshot.get("eta_s")
    return {
        "downloaded_bytes": done,
        "total_bytes": total,
        "speed_bps": int(speed) if isinstance(speed, (int, float)) and speed >= 0 else None,
        "eta_s": int(eta) if isinstance(eta, (int, float)) and eta >= 0 else None,
    }


def _base_ready(install_root) -> bool:
    """app/uv.lock 對應的環境（目前用的或基礎環境）已有 python（只看檔案在不在，不檢查套件是否完整）。"""
    try:
        return layout.python_in_env(launch_env.setup_env_dir(install_root)).is_file()
    except OSError:
        return False


def _installed_env(install_root, current) -> Path | None:
    """current.json 指向、而且 python 還在的環境資料夾。"""
    if current is None:
        return None
    env = layout.current_env_dir(install_root, current)
    return env if layout.python_in_env(env).is_file() else None


def disk_plan(install_root, flavor: str, free_gb: float | None, *, current=None, lock_sha: str | None = None,
              freeable_cache: dict | None = None) -> dict:
    """安裝 flavor 的空間估計。新環境裝好、舊環境刪掉之前兩份同時存在：同一份 uv.lock 的同一個版本重裝時，檔案全部從快取
    硬連結（REUSE_DISK_GB）；其他情況要下載並解開新的 torch（REQUIRED_DISK_GB，與首次安裝相同）。
    空間不夠時量一下舊環境的 torch 刪掉能放出多少（含快取裡的同一份），夠的話 can_delete_old_first=True。"""
    old = _installed_env(install_root, current)
    reuse = old is not None and current.lock_sha256 == lock_sha and current.flavor == flavor
    required = REUSE_DISK_GB if reuse else REQUIRED_DISK_GB
    low = free_gb is not None and free_gb < required
    freeable = 0.0
    if low and old is not None and not reuse:                       # 同一版重裝時刪掉舊的反而要重新下載，不提供
        cache = {} if freeable_cache is None else freeable_cache
        if "gb" not in cache:
            cache["gb"] = env_cleanup.freeable_torch_bytes(old) / 2**30
        freeable = cache["gb"]
    return {
        "required_gb": required,
        "reuses_installed": reuse,
        "low_space": low,
        "freeable_gb": round(freeable, 1),
        "can_delete_old_first": bool(low and freeable > 0 and free_gb + freeable >= required),
    }


def in_use_message(count: int) -> str:
    return (f"{APP_NAME} 還在執行（{count} 個程式）。請先關閉 {APP_NAME} 的所有視窗，包括系統匣圖示、背景處理與播放器，"
            "再按「重試」。")


class SetupApi:
    def __init__(self, install_root):
        self._root = Path(install_root)
        self.window = None                                  # __main__ 建立視窗後設定；launch_main 用它關閉設定視窗
        self._lock = threading.Lock()                       # 保護下面的安裝狀態
        self._phase = "idle"                                # idle | base | gpu | selfcheck | done | failed | cancelled
        self._message = ""
        self._device_name: str | None = None
        self._lines: deque[str] = deque(maxlen=LINE_LIMIT)
        self._progress: dict | None = None                  # 目前階段的下載進度，見 _progress_dict
        self._installing = False
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_detection: dict | None = None            # 最近一次 detect() 的結果，診斷包用

    def __dir__(self):
        return [name for name in super().__dir__() if name != "window"]

    # ------------------------------------------------------------------ 偵測

    def detect(self) -> dict:
        try:
            detection = gpu_detect.detect()
            recommendation = gpu_detect.recommend(detection.gpus, supported=flavors.offered())
            current = layout.read_current(self._root)
            free = _disk_free_gb(self._root)
            plans = self._disk_plans(free, current)
            plan = plans.get(recommendation.flavor) or next(iter(plans.values()), None)
            result = {
                "ok": True,
                "gpus": [_gpu_dict(gpu) for gpu in detection.gpus],
                "recommendation": {
                    "flavor": recommendation.flavor,
                    "status": recommendation.status,
                    "warnings": list(recommendation.warnings),
                },
                "options": [{"flavor": f.name, "label": f.label, "supported": f.offered} for f in flavors.listed()],
                "supported_cards": list(flavors.supported_cards()),     # 沒有可安裝版本時列給使用者看
                "disk_free_gb": free,
                "estimated_download_gb": ESTIMATED_DOWNLOAD_GB,
                "required_disk_gb": plan["required_gb"] if plan else REQUIRED_DISK_GB,
                "disk_plans": plans,                                # 每個可安裝版本各自的空間估計，見 disk_plan
                "in_use": self._in_use(),
                "webview2_installed": bool(preflight.webview2_installed()),
                "current": {"flavor": current.flavor, "verified": current.verified} if current else None,
                "base_ready": _base_ready(self._root),
                "nvidia_smi_ok": bool(detection.nvidia_smi_ok),
            }
        except Exception as exc:  # noqa: BLE001 - 不把例外丟給前端
            return {"ok": False, "message": f"偵測失敗：{exc}"}
        with self._lock:
            self._last_detection = copy.deepcopy(result)
        return result

    def _lock_sha(self) -> str | None:
        try:
            return layout.locate(self._root).lock_sha256
        except OSError:
            return None                                     # 沒有 uv.lock：安裝時 provision 會報錯

    def _disk_plans(self, free_gb, current) -> dict:
        sha = self._lock_sha()
        shared: dict = {}                                   # 舊環境的 torch 只量一次
        return {f: disk_plan(self._root, f, free_gb, current=current, lock_sha=sha, freeable_cache=shared)
                for f in flavors.offered()}

    def _in_use(self) -> list[dict]:
        """envs 底下正在跑的程式（不含設定模式自己與它的子程式）。"""
        return [{"pid": p.pid, "name": p.name} for p in env_guard.processes_under(layout.envs_dir(self._root))]

    # ------------------------------------------------------------------ 安裝

    def start_install(self, flavor, options=None) -> dict:
        try:
            if flavor not in flavors.offered():
                return {"ok": False, "message": f"目前不支援安裝「{flavor}」版本。"}
            delete_old_first = isinstance(options, dict) and options.get("delete_old_first") is True
            with self._lock:
                if self._installing:
                    return {"ok": False, "message": "已經有安裝正在進行中。"}
            busy = self._in_use()
            if busy:
                return {"ok": False, "reason": "in_use", "message": in_use_message(len(busy)), "processes": busy}
            plan = disk_plan(self._root, flavor, _disk_free_gb(self._root), current=layout.read_current(self._root),
                             lock_sha=self._lock_sha())
            if delete_old_first and not plan["low_space"]:
                delete_old_first = False                    # 空間已經夠了：照一般流程，舊環境等新的裝好再刪
            elif plan["low_space"] and plan["can_delete_old_first"] and not delete_old_first:
                return {"ok": False, "reason": "low_space", "plan": plan,
                        "message": "磁碟空間不夠同時放新、舊兩份環境。可以選擇先刪除舊環境再安裝。"}
            elif delete_old_first and not plan["can_delete_old_first"]:
                return {"ok": False, "reason": "low_space", "plan": plan,
                        "message": "就算先刪除舊環境，磁碟空間還是不夠，請先清出空間。"}
            with self._lock:
                if self._installing:
                    return {"ok": False, "message": "已經有安裝正在進行中。"}
                self._installing = True
                self._cancel_event = cancel_event = threading.Event()
                self._lines.clear()
                self._phase = "base"
                self._message = _PHASE_MESSAGES["base"]
                self._device_name = None
                self._progress = None
                self._thread = thread = threading.Thread(
                    target=self._run_install, args=(flavor, cancel_event, delete_old_first), name="setup-install",
                    daemon=True)
            try:
                thread.start()
            except Exception as exc:  # noqa: BLE001 - 執行緒開不起來：還原狀態，讓使用者可以重試
                self._finish("failed", f"無法開始安裝：{exc}", None)
                return {"ok": False, "message": f"無法開始安裝：{exc}"}
            return {"ok": True, "message": "開始安裝。"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"無法開始安裝：{exc}"}

    def poll(self) -> dict:
        with self._lock:
            return {
                "phase": self._phase,
                "lines": list(self._lines),
                "message": self._message,
                "device_name": self._device_name,
                "phase_progress": dict(self._progress) if self._progress else None,
            }

    def cancel(self) -> dict:
        with self._lock:
            if not self._installing:
                return {"ok": False, "message": "目前沒有進行中的安裝。"}
            self._cancel_event.set()
            self._message = "正在取消…"
        return {"ok": True, "message": "正在取消安裝。"}

    def _push_line(self, line: str) -> None:                # 由 installer 的讀取執行緒呼叫
        with self._lock:
            self._lines.append(str(line))

    def _set_progress(self, snapshot) -> None:              # 由 installer 的取樣執行緒呼叫
        progress = _progress_dict(snapshot)
        with self._lock:
            self._progress = progress

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._message = _PHASE_MESSAGES.get(phase, "")
            self._progress = None                           # 進度只屬於一個階段

    def _finish(self, phase: str, message: str, device_name: str | None) -> None:
        with self._lock:
            self._phase = phase
            self._message = message
            self._device_name = device_name
            self._progress = None
            self._installing = False

    def _run_install(self, flavor: str, cancel_event: threading.Event, delete_old_first: bool = False) -> None:
        try:
            outcome = provision.run(self._root, flavor, cancel_event, on_phase=self._set_phase, on_line=self._push_line,
                                    on_progress=self._set_progress, delete_old_first=delete_old_first)
            result = (outcome.phase, outcome.message, outcome.device_name)
        except Exception as exc:  # noqa: BLE001 - 背景執行緒不能默默死掉，否則畫面永遠停在進行中
            result = (provision.FAILED, f"安裝時發生未預期的錯誤：{exc}", None)
        self._finish(*result)

    def _shutdown(self, timeout_s: float = 15.0) -> None:
        """視窗關閉後由 __main__ 呼叫：安裝還在跑就取消，並等 installer 結束它啟動的 uv（它在 5 秒內會 terminate／kill）。"""
        with self._lock:
            thread = self._thread
            if self._installing:
                self._cancel_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout_s)

    # ------------------------------------------------------------------ 診斷與啟動

    def export_diagnostics(self) -> dict:
        try:
            with self._lock:
                detection = copy.deepcopy(self._last_detection)
            path = diagnostics.export_diagnostics(
                self._root, platform_support.data_dir() / "diagnostics", detection=detection)
            return {"ok": True, "path": str(path), "message": "診斷包已匯出。"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "path": "", "message": f"匯出診斷包失敗：{exc}"}

    def launch_main(self) -> dict:
        try:
            root = self._root
            if layout.status(root, layout.app_dir(root) / "uv.lock") != layout.READY:     # READY 已包含 current 存在且已驗證
                return {"ok": False, "message": "環境還沒有準備好，請先完成安裝。"}
            launch_env.start_main(root, launch_env.main_env_dir(root))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"啟動失敗：{exc}"}
        window = self.window
        if window is not None:
            threading.Timer(0.5, window.destroy).start()    # 給前端時間收到回傳值再關窗
        return {"ok": True, "message": "已啟動"}
