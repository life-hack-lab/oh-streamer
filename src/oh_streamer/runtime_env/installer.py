r"""uv sync runner for the first-run install. Stdlib only: it runs in the base env, before torch exists.

Two steps share one env dir: install_base (everything except torch) and install_gpu (adds the torch flavor, then
prunes torch's static libs). launch.pyw builds the lock's base env (envs/<lock_id>); setup mode passes env_dir, a new
folder per install (env_cleanup.new_env_name). Both only run `uv sync`, write the full output to <install_root>/logs and
can be cancelled. They never write current.json and never self-check the env; that is the caller's job.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, TextIO

from . import download_progress, flavors, layout, torch_prune

_TAIL_LINES = 15                                # output lines kept for the failure message
_STOP_GRACE_S = 5                               # terminate -> kill grace; also the reader-thread join timeout
_CREATE_NO_WINDOW = 0x08000000                  # the app runs under pythonw: a console child must not pop up a window


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    returncode: int | None                      # None: uv never started
    log_path: Path
    cancelled: bool
    message: str
    prune_bytes_freed: int = 0


def base_args() -> list[str]:
    """uv arguments for the base env: all deps except torch. The lock holds PyPI fallbacks, so every torch package is excluded.
    --inexact keeps an already installed GPU torch: without it, uv removes it (--no-install-package does not protect installed packages)."""
    return [
        "sync", "--frozen", "--inexact", "--no-install-project", "--extra", "subtitles",
        "--no-install-package", "torch", "--no-install-package", "torchvision", "--no-install-package", "torchaudio",
    ]


def gpu_args(flavor: str) -> list[str]:
    """uv arguments that add the torch flavor. --extra subtitles must stay, or uv would remove the subtitle packages."""
    if flavor not in flavors.in_lock():
        raise ValueError(f"不支援的 flavor：{flavor!r}（可用：{', '.join(flavors.in_lock())}）")
    return ["sync", "--frozen", "--no-install-project", "--extra", "subtitles", "--extra", flavor]


def _open_log(install_root, phase: str) -> tuple[Path, TextIO]:
    logs = layout.logs_dir(install_root)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = re.sub(r"[^\w.-]", "_", phase)
    for n in range(100):                                                # same phase twice in one second: add a suffix
        path = logs / f"uv-{stamp}-{name}{f'-{n}' if n else ''}.log"
        try:
            return path, open(path, "x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
    raise FileExistsError(f"log 檔名重複過多：{path}")


def _write(log: TextIO, text: str) -> None:
    """A full disk or a closed log must never break the install itself."""
    try:
        log.write(text)
        log.flush()
    except (OSError, ValueError):
        pass


def _append_log(path: Path, text: str) -> None:
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except OSError:
        pass


def _stop(proc: subprocess.Popen) -> None:
    """Stop only the process we started: terminate, then kill if it is still there after the grace period."""
    proc.terminate()
    try:
        proc.wait(_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _progress_sampler(cmd, app_dir: Path, env: dict, cache_path: Path, log: TextIO,
                      cancel_event: threading.Event | None, on_progress) -> download_progress.ProgressSampler | None:
    """Dry-run to learn what will be downloaded. None (no progress reports) when there is nothing to download or the
    plan cannot be read; any error here only costs the progress display, never the install."""
    if on_progress is None:
        return None
    try:
        output = download_progress.run_dry_run(cmd, app_dir, env, cancel_event)
        lock_text = (Path(app_dir) / "uv.lock").read_text(encoding="utf-8")
        tracker = download_progress.build_tracker(output, lock_text, cache_path)
    except Exception as exc:                                            # noqa: BLE001
        _write(log, f"# 下載進度：無法估計（{exc!r}）\n")
        return None
    if tracker is None:
        _write(log, "# 下載進度：無法估計（dry-run 失敗或輸出看不懂）\n")
        return None
    if tracker.empty:
        _write(log, "# 下載進度：不需要下載\n")
        return None
    _write(log, f"# 下載進度：dry-run 列出 {len(tracker)} 個快取裡沒有的套件\n")
    return download_progress.ProgressSampler(tracker, on_progress)


def run_uv_sync(
    uv_cmd: Sequence[str], app_dir: Path, env_dir: Path, install_root: Path, args: Sequence[str], *,
    python_exe: Path | None = None,
    on_line: Callable[[str], None] | None = None,       # called from a reader thread, one uv output line at a time
    cancel_event: threading.Event | None = None,
    phase: str,
    on_progress: Callable[[dict | None], None] | None = None,  # download progress (download_progress.DownloadTracker.snapshot), from a sampler thread
) -> SyncResult:
    r"""Run `uv sync` in app_dir with stdout+stderr merged; the full output goes to logs\uv-<time>-<phase>.log.
    With on_progress, a `--dry-run` first works out what must be downloaded; on_progress is only called when that
    is something we can measure, and never after this function returns."""
    base = [str(uv_cmd)] if isinstance(uv_cmd, (str, os.PathLike)) else [str(part) for part in uv_cmd]
    cmd = [*base, *args]
    if python_exe is not None:
        cmd += ["--python", str(python_exe)]

    env_path = Path(os.path.abspath(env_dir))
    cache_path = Path(os.path.abspath(layout.cache_dir(install_root)))
    env_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(env_path),
        "UV_CACHE_DIR": str(cache_path),
        "UV_NO_CONFIG": "1",
        "UV_SYSTEM_CERTS": "1",
    }

    log_path, log = _open_log(install_root, phase)
    sampler = None
    try:
        _write(log, f"# {datetime.now().isoformat(timespec='seconds')} phase={phase}\n# cwd: {app_dir}\n"
                    f"# env: UV_PROJECT_ENVIRONMENT={env_path} UV_CACHE_DIR={cache_path}\n# cmd: {subprocess.list2cmdline(cmd)}\n")

        if cancel_event is not None and cancel_event.is_set():
            _write(log, "# 已取消（尚未啟動 uv）\n")
            return SyncResult(False, None, log_path, True, "已取消", 0)

        sampler = _progress_sampler(cmd, app_dir, env, cache_path, log, cancel_event, on_progress)
        if cancel_event is not None and cancel_event.is_set():
            _write(log, "# 已取消（尚未啟動 uv）\n")
            return SyncResult(False, None, log_path, True, "已取消", 0)

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(app_dir), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except OSError as exc:
            _write(log, f"# 無法啟動 uv：{exc}\n")
            return SyncResult(False, None, log_path, False, f"無法啟動 uv：{exc}", 0)

        tail: deque[str] = deque(maxlen=_TAIL_LINES)

        def pump() -> None:                                             # must keep draining, or uv blocks on a full pipe
            try:
                for raw in proc.stdout:
                    line = raw.rstrip("\r\n")
                    tail.append(line)
                    _write(log, line + "\n")
                    if sampler is not None:
                        sampler.on_line(line)
                    if on_line is not None:
                        try:
                            on_line(line)
                        except Exception:                               # a broken UI callback must not stall the install
                            pass
            except (OSError, ValueError):
                pass

        reader = threading.Thread(target=pump, name="uv-sync-reader", daemon=True)
        reader.start()
        if sampler is not None:
            sampler.start()

        cancelled = False
        while True:
            try:
                proc.wait(0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set() and proc.poll() is None:
                    _stop(proc)
                    cancelled = True
                    break
        reader.join(_STOP_GRACE_S)                                      # EOF follows right after exit; the timeout only guards a grandchild holding the pipe
        if not reader.is_alive():
            proc.stdout.close()
        if sampler is not None:
            sampler.stop(final=not cancelled)                           # last figure after uv's "Prepared ..." line

        rc = proc.returncode
        if cancelled:
            message = "已取消"
            _write(log, f"# 已取消（結束碼 {rc}）\n")
        elif rc != 0:
            message = f"uv 結束碼 {rc}" + (f"：\n{chr(10).join(tail)}" if tail else "")
            _write(log, f"# 結束碼 {rc}\n")
        else:
            message = "uv sync 完成"
            _write(log, "# 結束碼 0\n")
        return SyncResult(not cancelled and rc == 0, rc, log_path, cancelled, message, 0)
    finally:
        if sampler is not None:
            sampler.stop(final=False)                                   # every exit path; a second stop() is a no-op
        try:
            log.close()
        except OSError:
            pass


def install_base(
    uv_cmd: Sequence[str], install_root: Path, python_exe: Path | None = None,
    on_line: Callable[[str], None] | None = None, cancel_event: threading.Event | None = None,
    on_progress: Callable[[dict | None], None] | None = None, env_dir: Path | None = None,
) -> SyncResult:
    """Everything except torch, into env_dir (default: the lock's base env). Raises OSError when app/uv.lock is missing."""
    install = layout.locate(install_root)                               # base and GPU steps must land in the same env
    return run_uv_sync(uv_cmd, install.app, env_dir or install.env, install_root, base_args(), python_exe=python_exe,
                       on_line=on_line, cancel_event=cancel_event, phase="base", on_progress=on_progress)


def install_gpu(
    uv_cmd: Sequence[str], install_root: Path, flavor: str, python_exe: Path | None = None,
    on_line: Callable[[str], None] | None = None, cancel_event: threading.Event | None = None,
    on_progress: Callable[[dict | None], None] | None = None, env_dir: Path | None = None,
) -> SyncResult:
    """Add the torch flavor to env_dir (default: the lock's base env), then prune torch's static libs.
    A prune failure does not fail the install."""
    args = gpu_args(flavor)                                             # reject an unsupported flavor before touching the disk
    install = layout.locate(install_root)
    target = env_dir or install.env
    result = run_uv_sync(uv_cmd, install.app, target, install_root, args, python_exe=python_exe,
                         on_line=on_line, cancel_event=cancel_event, phase=f"gpu-{flavor}", on_progress=on_progress)
    if not result.ok:
        return result

    try:
        pruned = torch_prune.prune_torch_static_libs(target, cache_dir=layout.cache_dir(install_root))
    except Exception as exc:
        note = f"清理 torch 靜態庫失敗：{exc!r}"
        freed = 0
        _append_log(result.log_path, f"# {note}\n")
    else:
        freed = pruned.bytes_freed
        note = f"已清理 torch 靜態庫：{len(pruned.files)} 個檔案，釋放 {freed / 2**30:.2f} GiB"
        lines = [f"# {note}\n"] + [f"#   清理錯誤：{path}: {msg}\n" for path, msg in pruned.errors]
        if pruned.errors:
            note += f"（{len(pruned.errors)} 個檔案刪除失敗）"
        _append_log(result.log_path, "".join(lines))
    return replace(result, message=f"GPU 版（{flavor}）安裝完成；{note}", prune_bytes_freed=freed)
