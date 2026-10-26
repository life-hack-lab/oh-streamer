r"""Environment and command for starting the app from the shared env, and where to find uv and the embedded Python.
Stdlib only: launch.pyw uses it before the base env exists. launch.pyw and setup mode both go through here."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import layout


def pythonw_in_env(env_dir) -> Path:
    env_dir = Path(env_dir)
    return env_dir / "Scripts" / "pythonw.exe" if sys.platform == "win32" else layout.python_in_env(env_dir)


def tools_dirs(app: Path) -> list[Path]:
    return [app / "tools", app / "tools" / "mpv"]


def main_env(install_root, env_dir, base=None) -> dict[str, str]:
    """Environment for the app process and its worker children: program root, model weights, bundled tools on PATH."""
    app = layout.app_dir(install_root)
    env = dict(os.environ if base is None else base)
    env["OH_STREAMER_INSTALL_ROOT"] = str(Path(install_root))
    env["OH_STREAMER_PROJECT_ROOT"] = str(app)
    env["LADA_MODEL_WEIGHTS_DIR"] = str(app / "model_weights")
    env["PYTHONPATH"] = str(app / "src")                                # the env has no project install (--no-install-project)
    parts = [str(p) for p in tools_dirs(app)] + [str(Path(env_dir) / "Scripts")]
    env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env


def main_command(env_dir) -> list[str]:
    """The queue window, the default desktop entry. pythonw: no console window."""
    return [str(pythonw_in_env(env_dir)), "-m", "oh_streamer", "queue"]


def start_main(install_root, env_dir, args: Sequence[str] = ()) -> subprocess.Popen:
    """Start the app detached so it outlives the caller (launch.pyw or the setup window)."""
    detach = {"creationflags": detached_creationflags()} if sys.platform == "win32" else {"start_new_session": True}
    return subprocess.Popen(
        main_command(env_dir) + list(args), env=main_env(install_root, env_dir),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach)


# ---- launch.pyw decisions: pure functions, so the launcher script itself is only flow and UI ----

MODE_MAIN = "main"                                                      # start the app directly
MODE_SETUP = "setup"                                                    # first run / lock changed / user asked: setup mode
SETUP_FLAG = "--setup"
ENV_INSTALL_ROOT = "OH_STREAMER_INSTALL_ROOT"
ENV_UV = "OH_STREAMER_UV"
_DETACHED_PROCESS = 0x00000008                                          # numeric: subprocess only defines these on Windows
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def wants_setup(argv: Sequence[str]) -> bool:
    """True when the user asked for setup mode explicitly (shortcut with --setup)."""
    return SETUP_FLAG in argv


def app_args(argv: Sequence[str]) -> list[str]:
    """Command-line arguments to hand on to the app (e.g. --hidden for autostart): everything except the launcher's own flag."""
    return [arg for arg in argv if arg != SETUP_FLAG]


def plan_launch(install_root, *, force_setup: bool = False) -> str:
    """MODE_MAIN only when layout.status says READY and setup was not forced; every other case (including no uv.lock) is MODE_SETUP."""
    if force_setup:
        return MODE_SETUP
    status = layout.status(install_root, layout.app_dir(install_root) / "uv.lock")
    return MODE_MAIN if status == layout.READY else MODE_SETUP


def uv_command(install_root, environ: Mapping[str, str] | None = None) -> Path | None:
    """OH_STREAMER_UV when set (taken as given; a wrong path surfaces when uv fails to start), else the bundled
    <install_root>/uv.exe (or uv), else uv on PATH (development), else None."""
    env = os.environ if environ is None else environ
    override = (env.get(ENV_UV) or "").strip()
    if override:
        return Path(override)
    for name in ("uv.exe", "uv"):
        bundled = Path(install_root) / name
        if bundled.is_file():
            return bundled
    found = shutil.which("uv")
    return Path(found) if found else None


def install_python(install_root, executable: str | os.PathLike | None = None) -> Path | None:
    """The interpreter uv builds the env from: the embedded <install_root>/python/python.exe; without one (development),
    the console python next to `executable` when given, else None and uv picks one itself."""
    embedded = Path(install_root) / "python" / "python.exe"
    if embedded.is_file():
        return embedded
    return base_python(executable) if executable else None


def base_python(executable: str | os.PathLike | None = None) -> Path | None:
    """The console python next to the running one (pythonw.exe -> python.exe in the same folder); None when it does not exist."""
    exe = sys.executable if executable is None else executable
    if not exe:
        return None
    path = Path(exe)
    if path.name.lower() == "pythonw.exe":
        path = path.with_name("python.exe")
    return path if path.is_file() else None


def setup_env_dir(install_root) -> Path:
    """The env setup mode runs from: the one current.json points to when it is for the shipped uv.lock and its python is
    there (reinstall / flavor switch: no base env to build), else the lock's base env envs/<lock_id> (launch.pyw builds
    it when missing). Raises OSError when app/uv.lock is missing."""
    install = layout.locate(install_root)
    current = layout.read_current(install_root)
    if current is not None and current.lock_sha256 == install.lock_sha256:
        env = layout.current_env_dir(install_root, current)
        if layout.python_in_env(env).is_file():
            return env
    return install.env


def main_env_dir(install_root) -> Path | None:
    """The env the app runs from: the one current.json points to; None without a current.json."""
    current = layout.read_current(install_root)
    return layout.current_env_dir(install_root, current) if current is not None else None


def setup_command(env_dir) -> list[str]:
    """Setup mode runs in the base env (it has pywebview, the embedded python does not)."""
    return [str(pythonw_in_env(env_dir)), "-m", "oh_streamer.setup_mode"]


def detached_creationflags() -> int:
    """Popen flags that let the app outlive launch.pyw; 0 outside Windows."""
    return (_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
