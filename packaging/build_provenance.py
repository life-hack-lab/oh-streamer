"""Git identity of a checkout for the installer build manifest."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


def _run_git(path: Path, *args: str) -> str | None:
    process = None
    try:
        process = subprocess.Popen(
            ["git", "-C", str(path.resolve()), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output, _ = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            process.communicate()
        return None
    except OSError:
        return None
    if process.returncode != 0:
        return None
    return output.strip()


def git_identity(path: Path) -> dict[str, Any]:
    """Return commit and dirtiness without exposing the checkout's local path."""
    path = path.resolve()
    root = _run_git(path, "rev-parse", "--show-toplevel")
    if not root or Path(root).resolve() != path:
        return {"revision": None, "dirty": None, "exact_tags": []}
    revision = _run_git(path, "rev-parse", "--verify", "HEAD")
    if not revision:
        return {"revision": None, "dirty": None, "exact_tags": []}
    status = _run_git(path, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        return {"revision": revision.lower(), "dirty": None, "exact_tags": []}
    tags = _run_git(path, "tag", "--points-at", "HEAD") or ""
    return {
        "revision": revision.lower(),
        "dirty": bool(status),
        "exact_tags": sorted(line for line in tags.splitlines() if line),
    }
