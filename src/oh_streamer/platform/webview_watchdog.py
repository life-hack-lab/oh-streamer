"""WebView2 啟動失敗的看門狗（只用標準函式庫）。

WebView2 執行階段不能用時，pywebview 只把錯誤印到 stderr，視窗永遠停在背景色。這裡在建立視窗後開一個計時器，
`window.events.loaded` 觸發就取消；逾時還沒載入就寫記錄檔、跳原生錯誤對話框，再 window.destroy() 讓 webview.start 返回。
非 Windows 平台不做任何事。
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

LOAD_TIMEOUT_SECONDS = 45.0
FAILURE_LOG_PREFIX = "WebView2 initialization failed"
WEBVIEW2_DOWNLOAD_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
_MB_YESNO, _MB_ICONERROR, _MB_TOPMOST, _MB_SETFOREGROUND = 0x4, 0x10, 0x40000, 0x10000
_IDYES = 6


def dialog_title(app_name: str) -> str:
    return f"{app_name} 無法顯示畫面"


def dialog_text(app_name: str, log_path) -> str:
    return (f"這台電腦的 Microsoft Edge WebView2 執行階段無法使用，所以 {app_name} 的畫面顯示不出來。\n\n"
            f"請到微軟網站下載並安裝「WebView2 Runtime」，再重新開啟 {app_name}：\n{WEBVIEW2_DOWNLOAD_URL}\n\n"
            f"如果已經安裝仍然出現這個訊息，請重新開機後再試。\n\n詳細記錄：{log_path}\n\n要現在開啟 WebView2 下載頁面嗎？")


def show_error_box(title: str, text: str) -> bool:
    """Native Yes/No error box (blocks until dismissed). True when the user chose Yes. Never raises."""
    try:
        import ctypes
        flags = _MB_YESNO | _MB_ICONERROR | _MB_TOPMOST | _MB_SETFOREGROUND
        return ctypes.windll.user32.MessageBoxW(None, text, title, flags) == _IDYES
    except Exception:
        return False


def open_download_page() -> None:
    try:
        import webbrowser
        webbrowser.open(WEBVIEW2_DOWNLOAD_URL)
    except Exception:
        pass


class _MemoryHandler(logging.Handler):
    def __init__(self, on_failure: Optional[Callable[[], None]] = None) -> None:
        super().__init__(logging.DEBUG)
        self.on_failure = on_failure
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass
        try:
            if self.on_failure is not None and record.getMessage().startswith(FAILURE_LOG_PREFIX):
                self.on_failure()
        except Exception:
            pass


class WebviewWatchdog:
    """arm(window) right after create_window (or when a hidden window is first shown); disarm() on loaded."""

    def __init__(self, app_name: str, logs_dir, *, timeout: float = LOAD_TIMEOUT_SECONDS,
                 show_dialog: Optional[Callable[[str, str], None]] = None, enabled: Optional[bool] = None,
                 before_destroy: Optional[Callable[[], None]] = None,
                 open_url: Optional[Callable[[str], None]] = None) -> None:
        self.app_name = app_name
        self.open_url = open_url or (lambda _url: open_download_page())
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self.timeout = timeout
        self.show_dialog = show_dialog or show_error_box
        self.before_destroy = before_destroy               # e.g. lift a close-to-tray veto so destroy() really closes
        self.enabled = (sys.platform == "win32") if enabled is None else enabled
        self.log_path: Optional[Path] = None
        self.fired = False
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._window = None
        self._handler = _MemoryHandler(self._on_pywebview_failure)
        self._reason = ""

    def attach_logging(self) -> None:
        """Call before webview.start so pywebview's logger records are kept for the log file."""
        if self.enabled:
            logging.getLogger("pywebview").addHandler(self._handler)

    def watch(self, window) -> None:
        """Hook `loaded` of the window; does not start the timer (see arm)."""
        self._window = window
        if self.enabled:
            window.events.loaded += self.disarm

    def arm(self, window=None) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._timer is not None or self._loaded or self.fired:
                return
            if window is not None:
                self._window = window
            self._timer = threading.Timer(self.timeout, self._expired)
            self._timer.daemon = True
            self._timer.start()

    def disarm(self, *_args) -> None:
        with self._lock:
            self._loaded = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _on_pywebview_failure(self) -> None:
        # Runs inside logging.emit on pywebview's thread: hand off so the dialog never blocks logging/GUI.
        if self.enabled:
            threading.Thread(target=self._fire, args=("pywebview 回報 WebView2 初始化失敗",), daemon=True).start()

    def _write_log(self) -> Optional[Path]:
        if self.logs_dir is None:
            return None
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            path = self.logs_dir / f"webview-{time.strftime('%Y%m%d-%H%M%S')}.log"
            head = [f"{self.app_name}: {self._reason}（疑似 WebView2 執行階段無法啟動）。",
                    f"python {sys.version.split()[0]} / {sys.platform}", "", "pywebview 記錄："]
            body = self._handler.lines or ["（沒有收到 pywebview 記錄）"]
            path.write_text("\n".join(head + body) + "\n", encoding="utf-8")
            return path
        except Exception:
            return None

    def _expired(self) -> None:
        self._fire(f"畫面在 {self.timeout:g} 秒內沒有載入完成")

    def _fire(self, reason: str) -> None:
        with self._lock:
            if self._loaded or self.fired:
                return
            self.fired = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._reason = reason
        self.log_path = self._write_log()
        try:
            if self.show_dialog(dialog_title(self.app_name), dialog_text(self.app_name, self.log_path or "（無法寫入）")):
                self.open_url(WEBVIEW2_DOWNLOAD_URL)
        except Exception:
            pass
        finally:
            try:
                if self.before_destroy is not None:
                    self.before_destroy()
            except Exception:
                pass
            try:
                if self._window is not None:
                    self._window.destroy()
            except Exception:
                pass
