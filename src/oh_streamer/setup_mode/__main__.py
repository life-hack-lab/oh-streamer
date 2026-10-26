r"""python -m oh_streamer.setup_mode：開啟首次設定視窗（pywebview）。這裡是設定模式唯一 import webview 的地方。

需要環境變數 OH_STREAMER_INSTALL_ROOT（安裝根目錄；launch.pyw 會設定）。視窗載入 resources/setup_ui/index.html，
前端透過 window.pywebview.api 呼叫 bridge.SetupApi。webview.start 的參數與 queue_app 相同；視窗網址改用 file:// URI（見 setup_url）。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV_INSTALL_ROOT = "OH_STREAMER_INSTALL_ROOT"
WINDOW_SIZE = (820, 640)
WINDOW_MIN_SIZE = (640, 520)
BACKGROUND_COLOR = "#0c0e16"                                # 與 setup.css 的 --bg 相同，避免載入時閃一下白色


def _install_root_from_env() -> Path:
    raw = os.environ.get(ENV_INSTALL_ROOT, "").strip()
    if raw:
        return Path(raw)
    message = (f"找不到安裝根目錄：請設定環境變數 {ENV_INSTALL_ROOT}（通常由 launch.pyw 設定），"
               f"例如 {ENV_INSTALL_ROOT}=C:\\Users\\<你>\\AppData\\Local\\Programs\\oh_streamer。")
    if sys.stderr is not None:                              # pythonw 沒有 stderr
        print(message, file=sys.stderr)
    raise RuntimeError(message)


def setup_url() -> str:
    """設定畫面的 file:// URI。index.html 用 ../queue_ui/branding.js 取得品牌名稱，所以必須從檔案系統載入：
    傳純路徑時 pywebview 會自動開本地 http 伺服器，根目錄就是 setup_ui/，../queue_ui/ 會 404；file:// URI 則不會。
    js_api 在 EdgeChromium 走 chrome.webview.postMessage，不依賴 http 伺服器。"""
    from oh_streamer.paths import resource_root

    return (resource_root() / "setup_ui" / "index.html").resolve().as_uri()


def _clear_web_cache(storage: Path) -> None:
    """設定畫面的檔案會隨程式更新而改變；WebView2 的設定檔會留著，所以每次啟動丟掉 HTTP 快取（只動快取）。"""
    for sub in ("Cache", "Code Cache"):
        shutil.rmtree(storage / "EBWebView" / "Default" / sub, ignore_errors=True)


def _tell(title: str, text: str) -> None:
    """Small native message box (no webview yet); stderr outside Windows. Never raises."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x40 | 0x40000)    # MB_ICONINFORMATION | MB_TOPMOST
        elif sys.stderr is not None:
            print(text, file=sys.stderr)
    except Exception:
        pass


def _hand_off_cleanup(install_root: Path) -> None:
    """Retired envs left (the one this process runs from cannot go while we run): launch.pyw deletes them once we exit
    when it started us; otherwise a detached helper outside envs/ waits for this process (and the venv launcher
    parent) to exit and does it."""
    from oh_streamer.runtime_env import env_cleanup, env_guard, layout

    current = layout.read_current(install_root)
    if current is None or not current.retired or os.environ.get(env_cleanup.ENV_LAUNCHER_CLEANUP) == "1":
        return
    pids = [os.getpid()]
    envs = os.path.normcase(os.path.abspath(layout.envs_dir(install_root))) + os.sep
    for proc in env_guard.list_processes():
        if proc.pid == os.getppid() and proc.exe and os.path.normcase(proc.exe).startswith(envs):
            pids.append(proc.pid)
    env_cleanup.spawn_cleanup_helper(install_root, pids)


def main() -> int:
    from oh_streamer.runtime_env.app_running import hold_app_mutex
    hold_app_mutex()
    install_root = _install_root_from_env()

    from oh_streamer.branding import APP_NAME
    from oh_streamer.platform import platform_support
    from oh_streamer.platform.windows_identity import set_process_app_id
    from oh_streamer.runtime_env.env_guard import SetupLock
    from oh_streamer.setup_mode.bridge import SetupApi

    lock = SetupLock(install_root)                          # held until this process ends; see env_guard
    if not lock.acquire():
        _tell(APP_NAME, "設定畫面已經開著了，請使用已經開啟的那個視窗。")
        return 0

    import webview

    set_process_app_id()                                    # 要在 pywebview 建立任何視窗之前（與佇列視窗共用工作列身分）
    storage = platform_support.data_dir() / "webview_setup"  # 和佇列視窗的 webview 設定檔分開，兩者可以同時開著
    _clear_web_cache(storage)

    api = SetupApi(install_root)
    window = webview.create_window(
        APP_NAME, url=setup_url(), js_api=api, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE, background_color=BACKGROUND_COLOR, text_select=True)
    api.window = window
    from oh_streamer.platform.webview_watchdog import WebviewWatchdog
    from oh_streamer.runtime_env import layout
    watchdog = WebviewWatchdog(APP_NAME, layout.logs_dir(install_root))   # WebView2 壞掉時 pywebview 只印 stderr，視窗會一直空白
    watchdog.attach_logging()
    watchdog.watch(window)
    watchdog.arm(window)
    try:
        webview.start(gui=platform_support.gui_engine(), debug=False, private_mode=False, storage_path=str(storage))
    finally:
        api._shutdown()                                     # 安裝中直接關視窗：取消並結束 uv，別讓它在背景繼續跑
        try:
            _hand_off_cleanup(install_root)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
