"""
Desktop shell of the background processing queue: a pywebview window (queue_ui/) + a tray icon (pystray) around
queue_service.QueueService. Never imports torch / oh_player (the GPU work runs in the queue_worker.py child).

    pythonw queue_app.py [--root DIR] [--instance NAME] [--hidden]

  * one queue app per player instance name (the service owns the "queue-<name>" single_instance); starting it again
    sends {"cmd": "show"} to the running one and exits;
  * closing the window only hides it to the tray; the tray menu's 結束 stops the service (kills the worker) and exits;
  * log: user data logs/queue_app.log (previous run: queue_app.prev.log).
"""

import argparse
import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from oh_streamer.platform import platform_support


from oh_streamer.branding import APP_NAME
from oh_streamer.paths import resource_root
from oh_streamer.platform.webview_watchdog import WebviewWatchdog

HERE = resource_root()
LOG_DIR = platform_support.data_dir() / "logs"
LOG_FILE = LOG_DIR / "queue_app.log"
UI_INDEX = HERE / "queue_ui" / "index.html"
WINDOW_TITLE = APP_NAME
NO_CONSOLE = sys.stdout is None or sys.executable.lower().endswith("pythonw.exe")
VIDEO_TYPES = ("影片檔 (*.mp4;*.mkv;*.mov;*.avi;*.webm;*.ts;*.m4v;*.wmv;*.flv;*.mpg;*.mpeg)", "所有檔案 (*.*)")


def default_root() -> Path:
    return platform_support.data_dir() / "queue"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{APP_NAME} processing queue window + tray")
    ap.add_argument("--root", default=None, help=r"Queue folder (default %%LOCALAPPDATA%%\oh_streamer\queue)")
    ap.add_argument("--instance", default=None, metavar="NAME",
                    help="Player instance name (default 'default', or env OH_STREAMER_PLAYER_INSTANCE)")
    ap.add_argument("--log-file", default=None, help="Log to this file (default with pythonw: user data logs/queue_app.log)")
    ap.add_argument("--hidden", action="store_true", help="Start minimised in the tray")
    return ap


def open_log_file(path: Path, keep_previous: bool = True):
    """Send stdout / stderr (our log, the service's log, tracebacks) to `path`, truncated per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if keep_previous and path.exists():
        try:
            os.replace(path, path.with_suffix(".prev" + path.suffix))
        except OSError:
            pass
    f = open(path, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = sys.stderr = f
    return f


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), f"[queue_app] {msg}", file=sys.stderr, flush=True)


def clear_web_cache(storage: Path) -> None:
    """The UI is served from queue_ui/ over a local http server and WebView2's profile persists, so after editing
    index.html / app.js / app.css the old copies kept coming back from its HTTP cache. Drop the caches (only those:
    localStorage and the rest of the profile stay) on every start."""
    for sub in ("Cache", "Code Cache"):
        shutil.rmtree(storage / "EBWebView" / "Default" / sub, ignore_errors=True)


def hide_child_consoles() -> None:
    """Without a console every console program we start would pop up its own window."""
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    orig_init = subprocess.Popen.__init__

    def init(self, *a, **kw):
        if "creationflags" not in kw and len(a) < 14:
            kw["creationflags"] = flag
        orig_init(self, *a, **kw)

    subprocess.Popen.__init__ = init


# ---------------------------------------------------------------------------------------------------------------
# Windows helpers (window handle lookup, foreground, dark title bar)
# ---------------------------------------------------------------------------------------------------------------

def find_hwnd() -> int:
    """Top-level window of THIS process titled WINDOW_TITLE (two queue apps may be running in tests)."""
    if sys.platform != "win32":
        return 0
    user32 = ctypes.windll.user32
    found = []
    pid = os.getpid()
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _lparam):
        p = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value == WINDOW_TITLE:
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(enum_proc(cb), 0)
    return found[0] if found else 0


def bring_to_front() -> None:
    hwnd = find_hwnd()
    if not hwnd:
        return
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)                      # SW_RESTORE
    user32.keybd_event(0x12, 0, 0, 0)               # a synthetic Alt press lets SetForegroundWindow through
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, 2, 0)


def system_uses_dark_theme() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return True


def apply_dark_title_bar() -> None:
    hwnd = find_hwnd()
    if not hwnd:
        return
    try:
        value = ctypes.c_int(1 if system_uses_dark_theme() else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), 20, ctypes.byref(value), 4)
    except Exception as e:  # noqa: BLE001 - cosmetic
        log(f"dark title bar failed: {e!r}")


# ---------------------------------------------------------------------------------------------------------------
# tray icon
# ---------------------------------------------------------------------------------------------------------------

def make_icon_image(size: int | None = None):
    """The app icon (resources/icons, made by tools/make_app_icon.py) at the tray's size.

    Windows: pick the hand-drawn frame for the small-icon metric (16/20/24/32 px depending on DPI). pystray loads
    its ICO at the large-icon size, which is exactly 2x at every DPI, so the shell's scale back down stays sharp."""
    from PIL import Image
    icons = resource_root() / "icons"
    if sys.platform == "win32":
        if size is None:
            try:
                size = ctypes.windll.user32.GetSystemMetrics(49) or 16       # SM_CXSMICON
            except Exception:  # noqa: BLE001
                size = 16
        with Image.open(icons / "app.ico") as ico:
            sizes = sorted(s[0] for s in ico.info["sizes"])
            pick = next((s for s in sizes if s >= size), sizes[-1])
            ico.size = (pick, pick)
            im = ico.convert("RGBA")
    else:
        size = size or 64
        with Image.open(icons / "app.png") as png:
            im = png.convert("RGBA")
    return im if im.size == (size, size) else im.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------------------------------------------
# js_api (every public attribute/method is exposed to the page: keep helpers underscore-private)
# ---------------------------------------------------------------------------------------------------------------

class Api:
    def __init__(self, app: "QueueApp"):
        self._app = app

    @property
    def _svc(self):
        return self._app.service

    def get_state(self):
        return self._svc.get_state()

    def add(self, text):
        return self._svc.add(text)

    def check_source(self, text):
        return self._svc.check_source(text)

    def stream(self, text):
        return self._svc.stream(text)

    def add_files(self):
        """File dialog only: returns the chosen paths; the page opens the intake panel with them (nothing is added)."""
        import webview
        paths = self._app.window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                                    file_types=VIDEO_TYPES)
        return {"ok": True, "paths": [str(p) for p in paths or []]}

    # --- intake panel (解析面板) ----------------------------------------------------
    def parse_inputs(self, text):
        return self._svc.parse_inputs(text)

    def resolve(self, text):
        """Blocking for seconds (yt-dlp / ffprobe / one ffmpeg frame): pywebview runs it on its own call thread."""
        r = self._svc.resolve(text)
        if not r.get("ok"):
            log(f"resolve failed: {r.get('error')} | {r.get('detail')}")
        return r

    def defaults(self):
        return self._svc.defaults()

    def estimate(self, tokens, overrides=None):
        return self._svc.estimate(tokens, overrides)

    def stream_resolved(self, token, overrides=None):
        return self._svc.stream_resolved(token, overrides)

    def add_resolved(self, tokens, overrides=None):
        return self._svc.add_resolved(tokens, overrides)

    def save_defaults(self, overrides=None):
        return self._svc.save_defaults(overrides)

    def remove(self, id):
        return self._svc.remove(id)

    def move(self, id, index):
        return self._svc.move(id, index)

    def pause(self, id):
        return self._svc.pause(id)

    def resume(self, id):
        return self._svc.resume(id)

    def retry(self, id):
        return self._svc.retry(id)

    def set_subtitles(self, id, enabled):
        return self._svc.set_subtitles(id, bool(enabled))

    def rerun_subtitles(self, id):
        return self._svc.rerun_subtitles(id)

    def retry_subtitles(self, id):
        return self._svc.retry_subtitles(id)

    def apply_subtitle_settings(self, id, choice):
        return self._svc.apply_subtitle_settings(id, choice)

    def apply_subtitle_settings_all(self, choice):
        return self._svc.apply_subtitle_settings_all(choice)

    def restart_subtitles(self, id):
        return self._svc.restart_subtitles(id)

    def cancel_subtitles(self, id):
        return self._svc.cancel_subtitles(id)

    def list_subtitle_models(self):
        return self._svc.list_subtitle_models()

    def list_subtitle_connections(self):
        return self._svc.list_subtitle_connections()

    def subtitle_model_downloads(self):
        return self._svc.subtitle_model_downloads()

    def plan_subtitle_model_download(self, item_ids):
        return self._svc.plan_subtitle_model_download(list(item_ids or []))

    def start_subtitle_model_download(self, item_ids):
        return self._svc.start_subtitle_model_download(list(item_ids or []))

    def open_model_card(self, item_id):
        """Open a catalog model's card in the system browser; only catalog https URLs are accepted."""
        import webbrowser
        from oh_streamer.subtitles import model_download as md
        item = next((i for i in md.catalog() if i.id == item_id), None)
        if item is None or not item.card_url.startswith("https://"):
            return {"ok": False, "error": "找不到模型卡"}
        webbrowser.open(item.card_url)
        return {"ok": True}

    def cancel_subtitle_model_download(self):
        return self._svc.cancel_subtitle_model_download()

    def subtitle_model_download_progress(self):
        return self._svc.subtitle_model_download_progress()

    def subtitle_usage(self):
        try:
            return self._svc.subtitle_usage()
        except Exception as exc:
            return {"version": 1, "providers": [], "error": f"{type(exc).__name__}: {exc}"}

    def test_save_subtitle_connection(self, spec):
        return self._svc.test_save_subtitle_connection(spec)

    def remove_subtitle_connection(self, connection_id):
        return self._svc.remove_subtitle_connection(connection_id)

    def remove_subtitle_pool_connection(self, connection_id):
        return self._svc.remove_subtitle_pool_connection(connection_id)

    def set_subtitle_profile(self, profile):
        return self._svc.set_subtitle_profile(profile)

    def test_add_subtitle_model(self, spec):
        return self._svc.test_add_subtitle_model(spec)

    def remove_subtitle_model(self, model_id):
        return self._svc.remove_subtitle_model(model_id)

    def update_subtitle_model_key(self, model_id, api_key):
        return self._svc.update_subtitle_model_key(model_id, api_key)

    def set_subtitle_configuration(self, primary, comparison_enabled, rows, item_id=None):
        return self._svc.set_subtitle_configuration(primary, comparison_enabled, rows, item_id)

    def list_subtitle_combos(self):
        return self._svc.list_subtitle_combos()

    def subtitle_combo_fallback(self):
        return self._svc.subtitle_combo_fallback()

    def set_subtitle_combos(self, id, ids):
        return self._svc.set_subtitle_combos(id, ids)

    def set_default_subtitle_combos(self, ids):
        return self._svc.set_default_subtitle_combos(ids)

    def reprocess_options(self, id):
        return self._svc.reprocess_options(id)

    def reprocess(self, id, changes, options=None):
        return self._svc.reprocess(id, changes, options)

    def pause_all(self):
        return self._svc.pause_all()

    def resume_all(self):
        return self._svc.resume_all()

    def set_keep(self, id, keep):
        return self._svc.set_keep(id, keep)

    def clear_headers(self, id):
        return self._svc.clear_headers(id)

    def set_settings(self, changes):
        return self._svc.set_settings(changes)

    def apply_model_settings(self, id):
        return self._svc.apply_model_settings(id)

    def open_folder(self, id=None):
        return self._svc.open_folder(id or None)

    # --- re-run setup (重新偵測／切換版本): installed copies only, see runtime_env/resetup.py -----------------------
    def runtime_info(self):
        from oh_streamer.runtime_env import resetup
        return resetup.info()

    def resetup(self):
        """Start `launch.pyw --setup` detached, then quit the app so setup mode is not blocked by a running copy."""
        from oh_streamer.runtime_env import resetup
        r = resetup.start_setup()
        log(f"resetup: {r}")
        if r.get("ok"):
            self._app.quit()
        return r

    # --- manual update check (only when the user presses the button; never at startup) ---------------------------
    def app_version(self):
        from oh_streamer.version import __version__
        return __version__

    def check_update(self):
        from oh_streamer import update_check
        return update_check.check()

    def open_release_page(self, url):
        """Open a release page in the browser; only https://github.com/ addresses are accepted."""
        import webbrowser
        from oh_streamer import update_check
        if not update_check.safe_release_url(url):
            return {"ok": False, "error": "不是有效的下載網址"}
        webbrowser.open(url)
        return {"ok": True}

    def gpu(self, active=True):
        """Latest GPU sample ({} when no NVIDIA driver). `active` False = panel closed: stop the nvidia-smi fallback."""
        return self._app.gpu_sample(bool(active))

    def thumb(self, url):
        """A film-strip frame (`url` from an item's thumbs) as a data: URI, or None (the page can't load file://)."""
        from oh_streamer.queue import queue_thumbs
        return queue_thumbs.frame_data_uri(self._svc.store.root, url)

    def save_preview_image(self, uri):
        """Save an enlarged preview using the desktop save dialog."""
        import base64
        import webview
        types = {"data:image/jpeg;base64": "jpg", "data:image/png;base64": "png",
                 "data:image/webp;base64": "webp"}
        try:
            prefix, encoded = str(uri).split(",", 1)
            extension = types.get(prefix)
            if not extension or len(encoded) > 32 * 1024 * 1024:
                return {"ok": False, "error": "不支援的圖片格式"}
            data = base64.b64decode(encoded, validate=True)
            paths = self._app.window.create_file_dialog(webview.FileDialog.SAVE,
                                                       save_filename="封面." + extension)
            if not paths:
                return {"ok": False, "cancelled": True}
            path = paths if isinstance(paths, str) else paths[0]
            Path(path).write_bytes(data)
            return {"ok": True}
        except (ValueError, OSError) as error:
            return {"ok": False, "error": str(error)}

    def player_window_ready(self):
        from oh_streamer.queue import queue_ipc
        return queue_ipc.send_to_player({"cmd": "window_ready"}, self._svc.player_name, timeout=0.5) or {"ready": False}

    def cancel_play_after_subtitles(self, id):
        return self._svc.cancel_play_after_subtitles(id)

    def play(self, id, start=None, wait_for_subtitles=False):
        """`start` = optional start position in seconds (invalid values are ignored); omitted = as before."""
        if wait_for_subtitles:
            return self._svc.play(id, start, wait_for_subtitles=True)
        return self._svc.play(id) if start is None else self._svc.play(id, start)

    def mark_all_seen(self):
        """Mark every item as seen (stops the unwatched-finished logo sparkle); returns {"ok": True, "count": n}."""
        return self._svc.mark_all_seen()

    # --- performance check (電腦效能檢測): state["perf"] carries the status, see perf_runner.py -------------------
    def perf_start(self, kind="standard"):
        r = self._svc.perf.start("quick" if kind == "quick" else "standard")
        log(f"perf_start {kind}: {r}")
        return r

    def perf_cancel(self):
        return self._svc.perf.cancel()

    def perf_apply(self):
        """Write the recommendation to the player settings (interpolation auto / off)."""
        r = self._svc.perf.apply()
        log(f"perf_apply: {r}")
        return r

    def perf_dismiss_prompt(self):
        return self._svc.perf.dismiss_prompt()

    def remove_compare(self, id):
        """Strip the compare tracks from a finished output (slow remux; runs on pywebview's own call thread)."""
        r = self._svc.remove_compare(id)
        log(f"remove_compare {id}: {r}")
        if r.get("ok"):
            gb = r.get("freed_bytes", 0) / 1024 ** 3
            freed = f"{gb:.1f} GB" if gb >= 0.1 else f"{r.get('freed_bytes', 0) / 1024 ** 2:.1f} MB"
            return {"ok": True, "message": f"已移除比對軌，釋放 {freed}"}
        return {"ok": False, "error": str(r.get("error", ""))}

    def export(self, id, include_compare=False):
        """Save dialog, then the (slow) remux; pywebview runs this call on its own thread, so the UI stays live.
        include_compare=True also keeps the compare video tracks (default: processed video + audio only)."""
        import webview
        it = self._svc.store.get(id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        name = "".join(c for c in (it.title or "output") if c not in '<>:"/\\|?*').strip(" .") or "output"
        dest = self._app.window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name + ".mp4",
                                                   file_types=("MP4 影片 (*.mp4)",))
        if not dest:
            return {"ok": True}
        dest = dest if isinstance(dest, str) else dest[0]
        r = self._svc.export(id, dest, bool(include_compare))
        log(f"export {id} -> {dest}: {r}")
        if r.get("ok"):
            return {"ok": True, "message": f"已匯出：{r.get('path', dest)}"}
        return {"ok": False, "error": "匯出失敗：" + str(r.get("error", ""))}


# ---------------------------------------------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------------------------------------------

class QueueApp:
    def __init__(self, service, hidden: bool):
        self.service = service
        self.hidden = hidden
        self.window = None
        self.watchdog = None
        self.icon = None
        self.paused_all = False
        self._loaded = threading.Event()
        self._quitting = threading.Event()
        self._notified = False
        self._last_state = None
        self._push_lock = threading.Lock()
        self._gpu = None

    # --- GPU panel ---------------------------------------------------------------------------------------
    def gpu_sample(self, active: bool) -> dict:
        if self._gpu is None:
            if not active:
                return {}
            from oh_streamer.player.info_panel import GpuMonitor   # stdlib-only module; NVML via pynvml, nvidia-smi fallback
            self._gpu = GpuMonitor(interval=1.0)
        self._gpu.set_active(active)
        s = self._gpu.snapshot() or {}
        return dict(s, backend=self._gpu.backend)

    # --- service callbacks -------------------------------------------------------------------------------
    def on_state(self, state: dict) -> None:
        self._last_state = state
        paused = bool(state.get("paused_all"))
        if paused != self.paused_all:
            self.paused_all = paused
            if self.icon:
                try:
                    self.icon.update_menu()
                except Exception:  # noqa: BLE001
                    pass
        self._push(state)

    def _push(self, state: dict) -> None:
        if self.window is None or not self._loaded.is_set() or self._quitting.is_set():
            return
        js = "window.ohStreamer && window.ohStreamer.onState(" + json.dumps(state, ensure_ascii=True) + ")"
        try:
            with self._push_lock:
                self.window.evaluate_js(js)
        except Exception as e:  # noqa: BLE001 - window closing / page reloading
            log(f"push state failed: {e!r}")

    def on_show(self) -> None:
        log("show requested")
        self.show_window()

    # --- window ------------------------------------------------------------------------------------------
    def show_window(self) -> None:
        if self.window is None or self._quitting.is_set():
            return
        if self.watchdog is not None:
            self.watchdog.arm(self.window)                  # no-op once the page has loaded
        try:
            self.window.show()
            self.window.restore()
        except Exception as e:  # noqa: BLE001
            log(f"show failed: {e!r}")
        bring_to_front()

    def hide_window(self) -> None:
        try:
            self.window.hide()
        except Exception as e:  # noqa: BLE001
            log(f"hide failed: {e!r}")

    def on_closing(self):
        if self._quitting.is_set():
            return True
        if self.icon is None:
            return True
        self.hide_window()
        if not self._notified and self.icon:
            self._notified = True
            try:
                self.icon.notify(f"{APP_NAME} 仍在背景執行", WINDOW_TITLE)
            except Exception as e:  # noqa: BLE001
                log(f"notify failed: {e!r}")
        return False

    def on_loaded(self) -> None:
        first = not self._loaded.is_set()
        self._loaded.set()
        if first:
            log("page loaded")
            apply_dark_title_bar()
            self._register_drop()
        if self._last_state is not None:
            self._push(self._last_state)

    def _register_drop(self) -> None:
        """Dropped files: only pywebview's own DOM drop handler learns the full path (pywebviewFullPath)."""
        try:
            self.window.dom.document.events.drop += self.on_drop
        except Exception as e:  # noqa: BLE001
            log(f"drop handler registration failed: {e!r}")

    def on_drop(self, event) -> None:
        try:
            files = (event.get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            log(f"drop: {paths}")
            if paths:
                self._open_panel(paths)                    # the page opens the intake panel; nothing is added here
        except Exception as e:  # noqa: BLE001
            log(f"drop failed: {e!r}")

    def _open_panel(self, paths: list) -> None:
        """Hand dropped paths to the page's window.ohStreamer.openPaths(paths), which opens the intake panel."""
        js = "window.ohStreamer && window.ohStreamer.openPaths && window.ohStreamer.openPaths(" + json.dumps(paths) + ")"
        try:
            self.window.evaluate_js(js)
        except Exception as e:  # noqa: BLE001 - window closing / page reloading
            log(f"drop hook failed: {e!r}")

    def _toast_error(self, msg: str) -> None:
        try:
            self.window.evaluate_js("window.ohStreamer && window.ohStreamer.notify && window.ohStreamer.notify(" + json.dumps(msg) + ", true)")
        except Exception:  # noqa: BLE001
            pass

    # --- tray --------------------------------------------------------------------------------------------
    def start_tray(self) -> None:
        import pystray
        menu = pystray.Menu(
            pystray.MenuItem(f"開啟 {APP_NAME}", lambda icon, item: self.show_window(), default=True),
            pystray.MenuItem(lambda item: "全部繼續" if self.paused_all else "全部暫停", self.toggle_pause_all),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("結束", lambda icon, item: self.quit()),
        )
        options = {}
        if sys.platform == "darwin":
            # Cocoa pystray shares NSApplication with pywebview's main-thread loop.
            from AppKit import NSApplication
            options["darwin_nsapplication"] = NSApplication.sharedApplication()
        self.icon = pystray.Icon("oh-queue", make_icon_image(), WINDOW_TITLE, menu, **options)
        self.icon.run_detached()

    def toggle_pause_all(self, icon=None, item=None) -> None:
        if self.paused_all:
            self.service.resume_all()
        else:
            self.service.pause_all()

    # --- exit --------------------------------------------------------------------------------------------
    def quit(self) -> None:
        if self._quitting.is_set():
            return
        self._quitting.set()
        log("quit requested")
        threading.Thread(target=self._shutdown, name="queue-quit", daemon=True).start()

    def _shutdown(self) -> None:
        self.hide_window()
        if self._gpu is not None:
            try:
                self._gpu.close()
            except Exception as e:  # noqa: BLE001
                log(f"gpu monitor close failed: {e!r}")
        try:
            self.service.stop()
        except Exception as e:  # noqa: BLE001
            log(f"service.stop failed: {e!r}")
        if self.icon:
            try:
                self.icon.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.window.destroy()
        except Exception as e:  # noqa: BLE001
            log(f"destroy failed: {e!r}")


_JOB = None


def kill_children_on_exit() -> bool:
    """Put this process in a Job Object with KILL_ON_JOB_CLOSE: every child (yt-dlp's ffmpeg downloader, the GPU worker
    and its ffmpeg) inherits the job and is killed when this process ends, even on a crash. Otherwise an orphaned ffmpeg
    keeps writing source.*.part and the next run's download collides with it (WinError 32 on rename)."""
    global _JOB
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED()
    # KILL_ON_JOB_CLOSE | BREAKAWAY_OK (only a child created with CREATE_BREAKAWAY_FROM_JOB escapes: the re-setup launcher)
    info.Basic.LimitFlags = 0x2000 | 0x0800
    ok = (k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))      # ExtendedLimitInformation
          and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    if not ok:
        log(f"job object: could not set up (error {ctypes.get_last_error()}); children may outlive the app")
        k32.CloseHandle(job)
        return False
    _JOB = job                                      # keep the handle open for the life of the process
    return True


def main() -> int:
    args = build_arg_parser().parse_args()
    from oh_streamer.runtime_env.app_running import hold_app_mutex

    hold_app_mutex()  # the installer asks the user to close the program while this exists
    from oh_streamer.platform.windows_identity import set_process_app_id

    set_process_app_id()  # Before pywebview creates any native windows.
    root = Path(args.root) if args.root else default_root()
    job_ok = kill_children_on_exit()                # before svc.start(): the scheduler may spawn children at once

    from oh_streamer.queue import queue_ipc

    from oh_streamer.queue import queue_service

    exe = Path(sys.executable)
    python_exe = exe.with_name("python.exe")        # the GPU worker is a hidden console-subsystem child
    svc = queue_service.QueueService(root, player_name=args.instance,
                                     python_exe=str(python_exe) if python_exe.exists() else sys.executable)
    if not svc.start():
        if NO_CONSOLE:
            open_log_file(LOG_DIR / "queue_app_forward.log", keep_previous=False)
        reply = queue_ipc.send_to_queue({"cmd": "show"}, args.instance, log=log)
        log(f"queue app already running: show -> {reply}")
        return 0 if reply and reply.get("ok") else 1

    if args.log_file or NO_CONSOLE:
        open_log_file(Path(args.log_file) if args.log_file else LOG_FILE)
        if NO_CONSOLE:
            hide_child_consoles()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
    log(f"started: root={root} instance={args.instance or 'default'} hidden={args.hidden} pid={os.getpid()} "
        f"kill_children_on_exit={job_ok}")

    clear_web_cache(root / "webview")
    import webview
    app = QueueApp(svc, args.hidden)
    svc.on_state.append(app.on_state)
    svc.on_show.append(app.on_show)
    api = Api(app)
    app.window = webview.create_window(
        WINDOW_TITLE, url=str(UI_INDEX), js_api=api, width=900, height=640, min_size=(560, 480),
        hidden=args.hidden, background_color="#1a1b1f", text_select=False)
    app.window.events.closing += app.on_closing
    app.window.events.loaded += app.on_loaded
    # WebView2 broken: pywebview only prints to stderr. on_closing vetoes closing (hide to tray) unless _quitting is set.
    app.watchdog = WebviewWatchdog(APP_NAME, LOG_DIR, before_destroy=app._quitting.set)
    app.watchdog.attach_logging()
    app.watchdog.watch(app.window)
    try:
        app.start_tray()
    except Exception as e:  # noqa: BLE001 - without a tray the window is the only UI: closing must really quit
        log(f"tray failed: {e!r}")
        if app.icon:
            try:
                app.icon.stop()
            except Exception:
                pass
        app.icon = None
        # Window.show() waits for the GUI to be running; configure visibility first.
        app.hidden = False
        app.window.hidden = False
    if not app.hidden:
        app.watchdog.arm(app.window)                        # a --hidden start arms it in show_window() instead
    try:
        webview.start(gui=platform_support.gui_engine(), debug=False, private_mode=False,
                      storage_path=str(root / "webview"))
    finally:
        log("window loop ended")
        app._quitting.set()
        try:
            svc.stop()
        except Exception as e:  # noqa: BLE001
            log(f"service.stop failed: {e!r}")
        if app.icon:
            try:
                app.icon.stop()
            except Exception:  # noqa: BLE001
                pass
        log("exit")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout and sys.stdout.flush()
    os._exit(code)
