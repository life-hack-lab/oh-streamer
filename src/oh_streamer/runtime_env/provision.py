r"""The whole install: new env folder -> base deps -> torch flavor -> self-check -> current.json -> delete the old env. No UI here.

The setup window (setup_mode.bridge) runs run() on a worker thread and shows the phases and uv lines it reports.
Every install (first run, reinstall, flavor switch, new uv.lock) goes into a NEW folder envs/<lock_id>-<flavor>-<n>;
base and torch are synced fresh into it, which costs little because uv hardlinks every file from its cache. The old
env is never touched until the new one passed the self-check: only then current.json switches to the new folder
and lists the others as retired, and env_cleanup deletes them in the same session (see env_cleanup for who deletes
what). A failure or a cancel deletes the new folder and never touches current.json, so the old env stays selected
and keeps working. The one exception is delete_old_first (the user chose it because the disk is too small for both):
the old env loses torch before the install starts.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import env_cleanup, installer, launch_env, layout, selfcheck

PHASES = ("base", "gpu", "selfcheck", "cleanup")
DONE, FAILED, CANCELLED = "done", "failed", "cancelled"
CANCELLED_MESSAGE = "已取消安裝"


@dataclass(frozen=True)
class Outcome:
    phase: str                                  # DONE | FAILED | CANCELLED
    message: str
    device_name: str | None = None              # the GPU the self-check ran on (DONE only)


def running_env() -> Path | None:
    """The env this process runs from (setup mode runs from an env under envs/); None outside a venv."""
    return Path(sys.prefix) if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else None


def run(install_root, flavor: str, cancel_event: threading.Event, *,
        on_phase: Callable[[str], None] | None = None,          # called with each PHASES entry as it starts
        on_line: Callable[[str], None] | None = None,             # one uv output line, from installer's reader thread
        on_progress: Callable[[dict | None], None] | None = None,
        delete_old_first: bool = False,
        own_env: Path | None = None) -> Outcome:
    """on_progress gets the current phase's download figures (installer / download_progress), only while they are
    measurable; a phase that reports nothing has nothing to download, or nothing we can measure.
    own_env: the env this process runs from (default running_env()); it can only lose its torch packages here."""
    try:
        install = layout.locate(install_root)
    except OSError as exc:
        return Outcome(FAILED, f"找不到程式的鎖定檔（app/uv.lock）：{exc}")
    uv = launch_env.uv_command(install.root)
    if uv is None:
        return Outcome(FAILED, f"找不到 uv：請確認安裝資料夾裡有 uv.exe，或設定環境變數 {launch_env.ENV_UV}。")
    uv_cmd = [str(uv)]
    python_exe = launch_env.install_python(install.root)
    own_env = running_env() if own_env is None else own_env
    cancelled = Outcome(CANCELLED, CANCELLED_MESSAGE)

    def enter(phase: str) -> None:
        if on_phase is not None:
            on_phase(phase)

    if delete_old_first:
        if on_line is not None:
            on_line("正在刪除舊環境以騰出空間…")
        env_cleanup.free_old_envs_first(install.root, running_env=own_env)

    env_cleanup.clear_cache_temp(install.root)                          # leftovers of a sync killed earlier (crash, power off)
    target = layout.envs_dir(install.root) / env_cleanup.new_env_name(install.root, install.lock_sha256, flavor)

    def give_up(outcome: Outcome) -> Outcome:
        errors = env_cleanup.discard_env(install.root, target)
        if errors and on_line is not None:
            on_line(f"無法完全刪除未完成的環境 {target.name}：{errors[0]}")
        env_cleanup.clear_cache_temp(install.root)                      # uv's half-unpacked wheel from the stopped sync
        if delete_old_first and outcome.phase == FAILED:
            return Outcome(FAILED, outcome.message + "\n（舊環境已經先刪除，要重新安裝才能使用。）")
        return outcome

    steps = (
        ("base", "基礎環境安裝失敗",
         lambda: installer.install_base(uv_cmd, install.root, python_exe, on_line, cancel_event, on_progress=on_progress,
                                        env_dir=target)),
        ("gpu", "GPU 版安裝失敗",
         lambda: installer.install_gpu(uv_cmd, install.root, flavor, python_exe, on_line, cancel_event,
                                       on_progress=on_progress, env_dir=target)),
    )
    try:
        for phase, failure, step in steps:
            if cancel_event.is_set():
                return give_up(cancelled)
            enter(phase)
            result = step()
            if result.cancelled:
                return give_up(cancelled)
            if not result.ok:
                return give_up(Outcome(FAILED, f"{failure}：{result.message}"))

        if cancel_event.is_set():
            return give_up(cancelled)
        enter("selfcheck")
        check = selfcheck.run_self_check(layout.python_in_env(target))
        if not check.ok:
            return give_up(Outcome(FAILED, f"自我檢查失敗：{check.message}"))
        if cancel_event.is_set():                                       # the self-check cannot be interrupted; a cancel still wins
            return give_up(cancelled)

        previous = layout.read_current(install.root)
        retired = [name for name in env_cleanup.env_folders(install.root) if name != target.name]
        if previous is not None:
            retired += [name for name in previous.retired if name not in retired and name != target.name]
        try:
            layout.write_current(install.root, layout.Current(
                lock_sha256=install.lock_sha256, flavor=flavor, verified=True,
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                env=target.name, retired=tuple(retired)))
        except OSError as exc:
            return give_up(Outcome(FAILED, f"寫入環境狀態失敗：{exc}"))
    except BaseException:
        if layout.read_current(install.root) is None or layout.read_current(install.root).env != target.name:
            env_cleanup.discard_env(install.root, target)               # an unexpected error: still leave no half env behind
        raise

    # switched: from here on nothing can fail the install; the old envs go now (the one we run from: after we exit)
    enter("cleanup")
    report = env_cleanup.delete_retired(install.root, running_env=own_env)
    if on_line is not None and (report.deleted or report.torch_removed or report.remaining):
        freed = report.cache_bytes / 2**30
        on_line(f"已刪除舊環境 {len(report.deleted)} 個，快取釋放 {freed:.2f} GiB"
                + (f"；關閉這個視窗後再刪除：{', '.join(report.remaining)}" if report.remaining else ""))
    return Outcome(DONE, "安裝完成", check.device_name)
