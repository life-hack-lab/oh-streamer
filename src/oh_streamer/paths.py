"""Resolve packaged resources, the streamer checkout, and the outer Lada tree."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def bundle_root() -> Path | None:
    """Return PyInstaller's immutable application root when running frozen."""
    if not getattr(sys, "frozen", False):
        return None
    root = getattr(sys, "_MEIPASS", None)
    return Path(root).resolve() if root else None


def checkout_root() -> Path:
    """Return the legacy streamer root used by source migration and root scripts.

    In a frozen build, ``_MEIPASS`` retains the pre-resource-migration root so
    old files there remain discoverable. In a source checkout, this is the
    repository root. A regular wheel has no checkout; its install's
    ``site-packages`` directory is the deterministic fallback.
    """
    bundled = bundle_root()
    if bundled is not None:
        return bundled

    package_file = Path(__file__).resolve()
    source_root = package_file.parents[2]
    checkout = source_root
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return package_file.parents[1]


def resource_root() -> Path:
    """Return the directory containing immutable Lua, UI, and JSON assets."""
    bundled = bundle_root()
    if bundled is not None:
        return bundled / "resources"
    return Path(__file__).resolve().parent / "resources"


def upstream_root() -> Path:
    """Return the outer Lada checkout (or the explicit worktree override)."""
    windows_bundle = getattr(sys, "frozen", False) and sys.platform == "win32"
    configured = os.environ.get("OH_STREAMER_PROJECT_ROOT")
    if configured and not windows_bundle:
        return Path(configured).expanduser().resolve()
    # The Windows portable bundle places Lada and third_party beside the packaged
    # application modules. Source checkouts and existing macOS bundles keep their
    # established parent-project lookup.
    bundled = bundle_root()
    if bundled is not None and windows_bundle:
        return bundled
    return checkout_root().parent
