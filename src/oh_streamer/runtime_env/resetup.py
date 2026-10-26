r"""Main program side of 「重新偵測／切換版本」: what the settings row shows, and starting `launch.pyw --setup` detached.
Stdlib only. Only an installed copy (OH_STREAMER_INSTALL_ROOT set by launch.pyw) can do it; a dev checkout reports unavailable
and never spawns anything. The caller quits the app right after a successful start so setup mode is not blocked."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from . import env_cleanup, flavors, launch_env, layout

_CREATE_BREAKAWAY_FROM_JOB = 0x01000000        # leave the app's kill-on-close job object (see queue_app.kill_children_on_exit)
_CREATE_NO_WINDOW = 0x08000000


def install_root(environ: Mapping[str, str] | None = None) -> Path | None:
    value = ((os.environ if environ is None else environ).get(launch_env.ENV_INSTALL_ROOT) or "").strip()
    return Path(value) if value else None


def flavor_label(name: str | None) -> str:
    if not name:
        return "尚未選擇"
    for flavor in flavors.ALL:
        if flavor.name == name:
            return flavor.label
    return name


def info(environ: Mapping[str, str] | None = None) -> dict:
    """{"available": bool, "flavor": name|None, "label": text, "verified": bool, "note": text}; never raises."""
    root = install_root(environ)
    if root is None:
        return {"available": False, "flavor": None, "label": "", "verified": False,
                "note": "開發環境不提供（只有安裝版可以重新偵測或切換版本）"}
    try:
        current = layout.read_current(root)
    except OSError:
        current = None
    if current is None:
        return {"available": True, "flavor": None, "label": "尚未選擇", "verified": False, "note": ""}
    return {"available": True, "flavor": current.flavor, "label": flavor_label(current.flavor),
            "verified": current.verified, "note": ""}


def setup_launch_command(root) -> list[str] | None:
    """`<embedded or base python> <root>\\app\\launch.pyw --setup`; None when no interpreter outside envs/ or no launch.pyw."""
    python = env_cleanup.helper_python(root)
    script = layout.app_dir(root) / "launch.pyw"
    if python is None or not script.is_file():
        return None
    return [str(python), str(script), launch_env.SETUP_FLAG]


def _clean_env(environ: Mapping[str, str]) -> dict[str, str]:
    env = dict(environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP"):    # launch.pyw runs on the embedded Python, not our env
        env.pop(key, None)
    return env


def start_setup(environ: Mapping[str, str] | None = None) -> dict:
    """Spawn `launch.pyw --setup` detached. {"ok": True} or {"ok": False, "error": zh-TW text}; nothing is spawned without an install root."""
    environ = os.environ if environ is None else environ
    root = install_root(environ)
    if root is None:
        return {"ok": False, "error": "這不是安裝版，無法重新偵測"}
    command = setup_launch_command(root)
    if command is None:
        return {"ok": False, "error": "找不到安裝版的啟動檔，請改用桌面捷徑重新設定"}
    base_flags = launch_env.detached_creationflags() | (_CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    attempts = [base_flags | _CREATE_BREAKAWAY_FROM_JOB, base_flags] if sys.platform == "win32" else [0]
    last: OSError | None = None
    for flags in attempts:
        try:
            subprocess.Popen(command, cwd=str(root), env=_clean_env(environ), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
                             start_new_session=sys.platform != "win32")
            return {"ok": True}
        except OSError as exc:                                  # breakaway refused by the job: retry without it
            last = exc
    return {"ok": False, "error": f"無法開啟設定畫面：{last}"}
