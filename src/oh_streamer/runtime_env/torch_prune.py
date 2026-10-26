r"""Delete the ~2.7 GB of static/import libraries (torch\lib\*.lib) that the CUDA torch wheel ships.

They are only needed to compile C++/CUDA extensions (torch.compile, torch.utils.cpp_extension), which Oh Streamer never
does; the runtime needs only the .dll files. uv hardlinks every env file from its cache, so a lib deleted from the env
still occupies the disk through the cache copy. Pass cache_dir to remove those cache copies too; only files that are
the very same file (same device+inode) as an env target are touched, never other cache entries.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PruneResult:
    files: list[str] = field(default_factory=list)                  # paths removed (or that would be, with dry_run)
    bytes_freed: int = 0                                            # each inode once, only if every link of it went
    errors: list[tuple[str, str]] = field(default_factory=list)     # (path, message); a bad file never aborts the run
    dry_run: bool = False


def _is_link(path: str) -> bool:
    """Symlink or Windows junction/other reparse point: never walk through these."""
    try:
        st = os.lstat(path)
    except OSError:
        return True
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)   # FILE_ATTRIBUTE_REPARSE_POINT


def _real_dir(path: Path) -> bool:
    return not _is_link(str(path)) and path.is_dir()


def _site_packages(env_dir: Path) -> list[Path]:
    found = [env_dir / "Lib" / "site-packages"]                                     # Windows venv
    lib = env_dir / "lib"
    if lib.is_dir() and not _is_link(str(lib)):                                     # POSIX venv: lib/pythonX.Y
        found += sorted(lib.glob("python*/site-packages"))
    return found


def _lib_files(torch_lib: Path) -> list[tuple[str, os.stat_result]]:
    """Regular *.lib files directly inside torch_lib (symlinks skipped), with their stat."""
    out = []
    if not _real_dir(torch_lib):
        return out
    try:
        with os.scandir(torch_lib) as it:
            for entry in it:
                if not entry.name.lower().endswith(".lib") or _is_link(entry.path):
                    continue
                try:
                    st = os.stat(entry.path, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    out.append((entry.path, st))
    except OSError:
        pass
    return out


def _key(path: str, st: os.stat_result):
    # st_ino 0 means the filesystem gave no file id: never match that file with another path
    return (st.st_dev, st.st_ino) if st.st_ino else ("path", path)


def _remove(path: str) -> str | None:
    """Delete one file, retrying once without the read-only attribute; returns an error message or None."""
    try:
        try:
            os.remove(path)
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            os.remove(path)
        return None
    except FileNotFoundError:
        return None
    except OSError as exc:
        return str(exc)


def prune_torch_static_libs(env_dir, cache_dir=None, dry_run: bool = False) -> PruneResult:
    r"""Remove <env>\Lib\site-packages\torch\lib\*.lib and, when cache_dir is given, their hardlinked uv cache twins."""
    result = PruneResult(dry_run=dry_run)
    env_dir = Path(env_dir)
    targets: dict = {}                                                  # key -> [paths, nlink, size]
    for site in _site_packages(env_dir):
        if _is_link(str(site)) or not site.is_dir() or _is_link(str(site / "torch")):
            continue
        for path, st in _lib_files(site / "torch" / "lib"):
            entry = targets.setdefault(_key(path, st), [[], st.st_nlink, st.st_size])
            entry[0].append(path)

    if cache_dir is not None and targets:
        archives = Path(cache_dir) / "archive-v0"
        if _real_dir(archives):
            try:
                ids = [p for p in archives.iterdir() if _real_dir(p)]
            except OSError:
                ids = []
            for archive in ids:
                if _is_link(str(archive / "torch")):
                    continue
                for path, st in _lib_files(archive / "torch" / "lib"):
                    entry = targets.get(_key(path, st))
                    if entry is not None:                               # same file as an env target; others stay
                        entry[0].append(path)

    for paths, nlink, size in targets.values():
        removed = 0
        for path in paths:
            error = None if dry_run else _remove(path)
            if error is None:
                removed += 1
                result.files.append(path)
            else:
                result.errors.append((path, error))
        if removed >= nlink:                                            # no other link (e.g. a second env) keeps the data
            result.bytes_freed += size
    return result


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: python -m oh_streamer.runtime_env.torch_prune ENV_DIR [CACHE_DIR] [--dry-run]")
    res = prune_torch_static_libs(args[0], args[1] if len(args) > 1 else None, "--dry-run" in sys.argv)
    print(f"{len(res.files)} files, {res.bytes_freed / 2**30:.2f} GiB freed, {len(res.errors)} errors (dry_run={res.dry_run})")
    for path, message in res.errors:
        print(f"  error: {path}: {message}")
