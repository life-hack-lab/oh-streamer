"""Safe, lazy migration helpers for user files from the old checkout layout."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable


def same_path(left: str | os.PathLike, right: str | os.PathLike) -> bool:
    """Compare paths without requiring either file to exist."""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(left)) == Path(os.path.abspath(right))


def migrate_legacy_file(source: str | os.PathLike, destination: str | os.PathLike,
                        validator: Callable[[bytes], bool] | None = None) -> tuple[bool, str | None]:
    """Copy a valid legacy file once, with an atomic publish that never replaces an existing file.

    Returns ``(copied, warning)``. The legacy file is always left in place. A per-destination temporary file is
    written and flushed before an exclusive atomic publish, so concurrent processes cannot expose partial data or
    overwrite one another's destination.
    """
    source, destination = Path(source), Path(destination)
    if same_path(source, destination) or destination.exists():
        return False, None

    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return False, None
    except OSError as exc:
        return False, _migration_warning(source, destination, f"could not read the legacy file: {exc}")

    if validator is not None:
        try:
            valid = validator(payload)
        except Exception:  # validators are defensive; malformed input should only skip migration
            valid = False
        if not valid:
            return False, (f"Ignoring invalid legacy user data file {source}; it was left in place and the "
                           "application will use its normal fallback.")

    temp_path = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".migrating",
                                         dir=destination.parent)
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            # A hard link is atomic and exclusive on the supported local filesystems. If the platform/filesystem
            # does not support links, Windows rename is also atomic and refuses to replace an existing destination.
            os.link(temp_path, destination)
        except FileExistsError:
            return False, None
        except OSError:
            if os.name != "nt":
                raise
            try:
                os.rename(temp_path, destination)
            except FileExistsError:
                return False, None
        return True, None
    except OSError as exc:
        return False, _migration_warning(source, destination, f"could not publish the migrated file: {exc}")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _migration_warning(source: Path, destination: Path, reason: str) -> str:
    return (f"Could not migrate legacy user data file {source} to {destination}: {reason}. "
            "The legacy file was left in place; the application will use its normal fallback.")
