r"""啟動器：捷徑執行 `pythonw.exe launch.pyw`（內嵌 Python）。只用標準函式庫，安裝後位於 <安裝根>\app\launch.pyw。

流程（判斷都在 oh_streamer.runtime_env.launch_env 的純函式裡，這裡只剩流程與介面）：
  - 環境已就緒（current.json 有效、對應目前的 uv.lock、環境的 python 在）且沒有 --setup：直接啟動主程式後結束。
  - 否則進設定模式：基礎環境不存在就先用 uv 建立（內嵌 Python 沒有 pywebview，這段用 tkinter 小視窗顯示進度），
    再用基礎環境的 pythonw 執行 `-m oh_streamer.setup_mode` 並等它結束；「開始使用」由設定模式自己啟動主程式。
    重新安裝時（current.json 對應目前的 uv.lock）設定模式直接從目前的環境執行，不用建基礎環境。
  - 設定模式結束後刪除被換掉的舊環境（current.json 的 retired；設定模式從那個環境執行，自己刪不掉），見 env_cleanup。
  - 設定畫面開著時（env_guard.SetupLock）不啟動主程式，也不開第二個設定畫面。
  - 任何失敗都寫進 <安裝根>\logs\launch-<時間>.log，並盡量用 tkinter 對話框告訴使用者；寫 log 或顯示對話框失敗時不再拋例外。
本檔不連網；current.json 只會經由 env_cleanup 改掉 retired 清單（刪掉的舊環境），其他欄位不動。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

_FALLBACK_NAME = "Streamer"                     # only when oh_streamer.branding cannot be imported
_PREPARING = "正在準備基礎環境"
_HERE = Path(__file__).resolve().parent
_MESSAGE_LIMIT = 1200                           # characters of detail shown in a dialog; the full text goes to the log


# ---- 安裝根目錄與記錄 ----

def _resolve_install_root() -> Path:
    """OH_STREAMER_INSTALL_ROOT, else the folder above this file's folder (<root>\\app\\launch.pyw). Needed before any import."""
    override = os.environ.get("OH_STREAMER_INSTALL_ROOT", "").strip()
    return Path(os.path.abspath(override)) if override else _HERE.parent


def _app_name() -> str:
    try:
        from oh_streamer.branding import APP_NAME
        return APP_NAME
    except Exception:
        return _FALLBACK_NAME


def _logs_dir(install_root: Path) -> Path:
    try:
        from oh_streamer.runtime_env import layout
        return layout.logs_dir(install_root)
    except Exception:
        return install_root / "logs"


def _write_launch_log(install_root: Path, text: str) -> Path | None:
    """Append text to logs\\launch-<YYYYmmdd-HHMMSS>.log; None when it cannot be written. Never raises."""
    try:
        logs = _logs_dir(install_root)
        logs.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        path = logs / f"launch-{now:%Y%m%d-%H%M%S}.log"
        header = (f"# {now.isoformat(timespec='seconds')} launch.pyw\n# install_root: {install_root}\n"
                  f"# python: {sys.executable}\n# argv: {sys.argv[1:]}\n")
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(header + text.rstrip("\n") + "\n\n")
        return path
    except Exception:
        return None


# ---- tkinter（標準函式庫；匯入失敗或沒有顯示器就略過）----

def _enable_dpi_awareness() -> None:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _message_box(title: str, text: str, *, info: bool = False) -> bool:
    """Show an error (or, with info, an information) dialog; False when tkinter is unavailable. Never raises."""
    try:
        import tkinter
        from tkinter import messagebox
        _enable_dpi_awareness()
        root = tkinter.Tk()
    except Exception:
        return False
    try:
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        (messagebox.showinfo if info else messagebox.showerror)(title, text, parent=root)
        return True
    except Exception:
        return False
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _clip(text: str, limit: int = _MESSAGE_LIMIT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _report_failure(install_root: Path, summary: str, *, detail: str = "", log_hint: Path | None = None) -> None:
    """Record the failure in a launch log and tell the user; log_hint is the more useful log to point at (e.g. the uv log)."""
    try:
        log_path = _write_launch_log(install_root, summary + (("\n" + detail) if detail else ""))
        where = log_hint or log_path
        lines = [_clip(summary)]
        if where is not None:
            lines += ["", f"詳細記錄：{where}"]
        _message_box(f"{_app_name()} 啟動失敗", "\n".join(lines))
    except Exception:
        pass


# Colours of the setup window (setup_ui/setup.css), so the two windows read as one product.
_BG, _PANEL, _LINE, _SURFACE = "#0c0e16", "#131725", "#252c42", "#1c2238"
_TEXT, _TEXT_2, _TEXT_3, _HOT = "#eef1f8", "#8e97ae", "#7b839d", "#ffcf3a"
_HOT_DIM = "#5a5233"                                                    # the wave's resting block: --hot 25% over --line
_FONT, _MONO = "Microsoft JhengHei UI", "Cascadia Mono"
_BLOCKS, _WAVE_LEN = 40, 20                                             # segmented bar like setup.css .blocks; wave period


def _dark_title_bar(root) -> None:
    """Windows 10 20H1+/11: dark caption to match the dark window (DWMWA_USE_IMMERSIVE_DARK_MODE). Best effort."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        on = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
    except Exception:
        pass


class _StatusWindow:
    """Small progress window shown while the base env is built; closing it cancels the install.
    Dark like the setup window: a 40-block bar that fills while the download is measurable and pulses otherwise."""

    def __init__(self, tk, root, headline: str, cancel_event: threading.Event) -> None:
        from oh_streamer.runtime_env import progress_text
        self._text = progress_text
        self._root = root
        self._cancel_event = cancel_event
        self._line = ""
        self._shown = ""
        self._snapshot: dict | None = None
        self._frame = 0
        scale = root.winfo_fpixels("1i") / 96                           # tk pixels are not DPI-scaled; fonts (points) are
        px = lambda n: max(1, round(n * scale))                         # noqa: E731
        width = px(520)
        root.configure(bg=_BG)
        try:                                                            # the app icon instead of tk's feather
            import oh_streamer
            root.iconbitmap(default=str(Path(oh_streamer.__file__).parent / "resources" / "icons" / "app.ico"))
        except Exception:
            pass
        box = tk.Frame(root, bg=_PANEL, highlightthickness=px(2), highlightbackground=_LINE)
        box.pack(padx=px(16), pady=(px(16), px(10)), fill="x")
        head = tk.Frame(box, bg=_PANEL)
        head.pack(padx=px(18), pady=(px(16), px(2)), fill="x")
        self._head = tk.Label(head, text=headline, bg=_PANEL, fg=_TEXT, font=(_FONT, 13, "bold"), anchor="w")
        self._head.pack(side="left")
        self._pct = tk.Label(head, text="", bg=_PANEL, fg=_TEXT, font=(_MONO, 12), anchor="e")
        self._pct.pack(side="right")
        tk.Label(box, text="第一次開啟需要下載必要元件，完成後會自動開啟設定畫面。", bg=_PANEL, fg=_TEXT_2,
                 font=(_FONT, 10), anchor="w").pack(padx=px(18), pady=(0, px(12)), fill="x")
        self._gap = px(2)
        self._cell = (width - self._gap * (_BLOCKS - 1)) / _BLOCKS
        self._bar = tk.Canvas(box, width=width, height=px(14), bg=_PANEL, highlightthickness=0, bd=0)
        self._bar.pack(padx=px(18), anchor="w")
        self._cells = [self._bar.create_rectangle(*self._cell_box(k, px(14)), fill=_LINE, width=0) for k in range(_BLOCKS)]
        self._dl = tk.Label(box, text="正在計算要下載的檔案…", bg=_PANEL, fg=_TEXT_2, font=(_FONT, 10), anchor="w")
        self._dl.pack(padx=px(18), pady=(px(8), 0), fill="x")
        self._tail = tk.Label(box, text="", bg=_PANEL, fg=_TEXT_3, font=(_MONO, 9), anchor="w",
                              width=1, wraplength=0)                    # width=1 + fill: long uv lines are cut, never widen
        self._tail.pack(padx=px(18), pady=(px(4), px(16)), fill="x")
        foot = tk.Frame(root, bg=_BG)
        foot.pack(padx=px(16), pady=(0, px(14)), fill="x")
        tk.Label(foot, text="安裝期間請不要關閉這個視窗", bg=_BG, fg=_TEXT_3, font=(_FONT, 9)).pack(side="left")
        self._button = tk.Button(foot, text="取消", width=10, command=self._cancel, font=(_FONT, 10),
                                 bg=_SURFACE, fg=_TEXT, activebackground=_LINE, activeforeground=_TEXT,
                                 disabledforeground=_TEXT_3, relief="flat", bd=0, highlightthickness=px(1),
                                 highlightbackground=_LINE, cursor="hand2", padx=px(8), pady=px(4))
        self._button.pack(side="right")
        root.protocol("WM_DELETE_WINDOW", self._cancel)
        root.update_idletasks()
        _dark_title_bar(root)
        x = max((root.winfo_screenwidth() - root.winfo_reqwidth()) // 2, 0)
        y = max((root.winfo_screenheight() - root.winfo_reqheight()) // 3, 0)
        root.geometry(f"+{x}+{y}")

    def _cell_box(self, k: int, height: int) -> tuple[float, float, float, float]:
        x0 = k * (self._cell + self._gap)
        return x0, 0, x0 + self._cell, height

    def set_line(self, text: str) -> None:
        """Called from the uv reader thread: only stores the text; the tk thread displays it."""
        self._line = text.strip().replace("\t", " ")[:100]

    def set_progress(self, snapshot: dict | None) -> None:
        """Called from the progress sampler thread (download_progress.DownloadTracker.snapshot): only stores it."""
        self._snapshot = dict(snapshot) if snapshot else None

    def _cancel(self) -> None:
        if self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self._head.config(text="正在取消…")
        self._button.config(state="disabled")

    def _draw(self) -> None:
        """Fill the bar to the download fraction; without one (planning, unmeasurable, installing) run a wave."""
        self._frame += 1
        if self._line or self._snapshot is not None:                    # before uv says anything, keep 正在計算…
            fraction, text = self._text.describe(self._snapshot)
            self._dl.config(text=text)
        else:
            fraction = None
        if fraction is not None:
            filled = fraction * _BLOCKS
            self._pct.config(text=f"{int(fraction * 100)}%")
            for k, cell in enumerate(self._cells):
                self._bar.itemconfig(cell, fill=_HOT if k < int(filled) else _LINE)
            return
        self._pct.config(text="")
        lead = self._frame % _WAVE_LEN
        for k, cell in enumerate(self._cells):
            self._bar.itemconfig(cell, fill=_HOT if (k - lead) % _WAVE_LEN < 3 else _HOT_DIM)

    def run_until(self, worker: threading.Thread) -> None:
        """Pump the window until the worker thread ends, then close it."""
        def tick() -> None:
            if not worker.is_alive():
                self._root.destroy()
                return
            if self._line != self._shown:
                self._shown = self._line
                self._tail.config(text=self._shown)
            self._draw()
            self._root.after(90, tick)

        self._root.after(90, tick)
        try:
            self._root.mainloop()
        except Exception:
            pass                                                        # the UI died: the install itself must still finish
        worker.join()


def _open_status_window(title: str, headline: str, cancel_event: threading.Event) -> _StatusWindow | None:
    try:
        import tkinter as tk
        _enable_dpi_awareness()
        root = tk.Tk()
    except Exception:
        return None
    try:
        root.title(title)
        root.resizable(False, False)
        return _StatusWindow(tk, root, headline, cancel_event)
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        return None


# ---- 流程 ----

def _install_base(installer, uv: Path, install_root: Path, python_exe: Path | None):
    """installer.install_base, with a progress window when tkinter works (the install runs on a worker thread)."""
    cancel_event = threading.Event()
    window = _open_status_window(f"{_app_name()} - 準備基礎環境", _PREPARING, cancel_event)
    if window is None:
        return installer.install_base([str(uv)], install_root, python_exe, cancel_event=cancel_event)

    box: dict = {}

    def work() -> None:
        try:
            box["result"] = installer.install_base([str(uv)], install_root, python_exe,
                                                   on_line=window.set_line, on_progress=window.set_progress,
                                                   cancel_event=cancel_event)
        except BaseException as exc:                                    # re-raised on the main thread below
            box["error"] = exc

    worker = threading.Thread(target=work, name="base-install", daemon=True)
    worker.start()
    window.run_until(worker)
    if "error" in box:
        raise box["error"]
    return box["result"]


def _run_setup(install_root: Path, env_dir: Path, launch_env, layout) -> int:
    base_python_exe = layout.python_in_env(env_dir)
    if not base_python_exe.is_file():
        from oh_streamer.runtime_env import installer
        uv = launch_env.uv_command(install_root)
        if uv is None or not uv.is_file():
            expected = uv if uv is not None else install_root / "uv.exe"
            _report_failure(
                install_root,
                f"找不到 uv：{expected}\n請重新安裝，或用環境變數 {launch_env.ENV_UV} 指定 uv.exe 的位置。")
            return 1
        result = _install_base(installer, uv, install_root, launch_env.install_python(install_root, sys.executable))
        if result.cancelled:
            return 0
        if not result.ok:
            _report_failure(install_root, f"基礎環境安裝失敗。\n{result.message}",
                            detail=f"uv log: {result.log_path}", log_hint=result.log_path)
            return 1
        if not base_python_exe.is_file():
            _report_failure(install_root, f"基礎環境安裝完成，但找不到 {base_python_exe}。",
                            detail=f"uv log: {result.log_path}", log_hint=result.log_path)
            return 1

    from oh_streamer.runtime_env import env_cleanup
    child_env = launch_env.main_env(install_root, env_dir)
    child_env[env_cleanup.ENV_LAUNCHER_CLEANUP] = "1"                   # we delete the replaced env once setup has exited
    with tempfile.TemporaryFile() as output:                            # keeps a crash's traceback without leaving files behind
        completed = subprocess.run(
            launch_env.setup_command(env_dir), env=child_env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
        )
        _cleanup_retired(install_root)                                  # the env setup ran from is free now
        if completed.returncode != 0:
            try:
                output.seek(0)
                tail = output.read()[-4000:].decode("utf-8", errors="replace")
            except Exception:
                tail = ""
            _report_failure(install_root, f"設定畫面異常結束（結束碼 {completed.returncode}）。", detail=tail)
            return 1
    return 0


def _cleanup_retired(install_root: Path) -> None:
    """Delete the envs a finished setup replaced (env_cleanup.run_after_setup); problems only go to logs\\env-cleanup.log."""
    try:
        from oh_streamer.runtime_env import env_cleanup
        env_cleanup.run_after_setup(install_root)
    except Exception:
        _write_launch_log(install_root, "刪除舊環境時發生錯誤：\n" + traceback.format_exc())


def _launch(install_root: Path, argv: list[str]) -> int:
    from oh_streamer.runtime_env import env_guard, launch_env, layout

    if env_guard.setup_running(install_root):                           # never start the app (or a 2nd setup) mid-install
        _message_box(_app_name(), f"設定畫面正在執行中。請先完成或關閉設定畫面，再開啟 {_app_name()}。", info=True)
        return 0
    force_setup = launch_env.wants_setup(argv)
    try:
        setup_env = launch_env.setup_env_dir(install_root)
    except OSError as exc:
        _report_failure(install_root, f"讀不到程式的 uv.lock：{exc}")
        return 1
    if launch_env.plan_launch(install_root, force_setup=force_setup) == launch_env.MODE_MAIN:
        launch_env.start_main(install_root, launch_env.main_env_dir(install_root), launch_env.app_args(argv))
        if layout.read_current(install_root).retired:                   # a setup that ended without us: finish its cleanup
            _cleanup_retired(install_root)
        return 0
    return _run_setup(install_root, setup_env, launch_env, layout)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    install_root = _resolve_install_root()
    try:
        sys.path.insert(0, str(install_root / "app" / "src"))          # == layout.app_dir(install_root) / "src"; layout is not importable yet
        try:
            from oh_streamer.runtime_env.app_running import hold_app_mutex
            hold_app_mutex()                                            # the embedded python.exe lives in the install folder too
        except Exception:
            pass
        return _launch(install_root, argv)
    except Exception as exc:
        _report_failure(install_root, f"發生未預期的錯誤：{type(exc).__name__}: {exc}", detail=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
