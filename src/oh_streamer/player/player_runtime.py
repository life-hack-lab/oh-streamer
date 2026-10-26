"""Locate the external player without depending on the playback pipeline."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def bundled_macos_mpv() -> Path | None:
    """The helper belongs to the outer app, outside PyInstaller's _MEIPASS."""
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return None
    executable = Path(sys.executable).resolve()
    if executable.parent.name != "MacOS" or executable.parent.parent.name != "Contents":
        return None
    return executable.parent.parent / "Helpers" / "oh-player.app" / "Contents" / "MacOS" / "mpv"


def find_mpv(explicit: str | None, project_root: Path) -> str:
    if explicit:
        requested = Path(explicit).expanduser()
        if requested.is_file():
            return str(requested)
        raise SystemExit(f"Requested mpv executable not found: {requested}")

    bundled = bundled_macos_mpv()
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        if bundled is not None and bundled.is_file():
            return str(bundled)
        raise SystemExit("Bundled mpv is missing. Reinstall the application or pass --mpv <executable>.")

    candidates = [shutil.which("mpv"), r"C:\Program Files\MPV Player\mpv.exe",
                  str(project_root / "tools" / "mpv" / "mpv.exe")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("mpv not found. Install it or pass --mpv <executable>.")


def process_kwargs(mpv: str) -> dict:
    """Enable mpv's app-bundle behavior only for our own nested helper."""
    bundled = bundled_macos_mpv()
    if bundled is None or Path(mpv).resolve() != bundled.resolve():
        return {}
    environment = {**os.environ, "MPVBUNDLE": "true"}
    manifest = bundled.parents[1] / "Resources/vulkan/icd.d/MoltenVK_icd.json"
    if manifest.is_file():
        # Both names cover current and older Vulkan loaders. Never consult a
        # build-machine or user-supplied ICD when launching the packaged player.
        environment["VK_DRIVER_FILES"] = str(manifest)
        environment["VK_ICD_FILENAMES"] = str(manifest)
        environment.pop("VK_ADD_DRIVER_FILES", None)
    return {"env": environment}
