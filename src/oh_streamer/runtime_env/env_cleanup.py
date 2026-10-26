r"""New env folders, deleting replaced envs, and dropping uv-cache entries no env uses any more. Stdlib only: launch.pyw
(embedded Python, outside every env) runs delete_retired() after the setup process has exited.

Who deletes the old env: setup mode runs from an env under envs/ and cannot delete that env while it runs. So the
switch (layout.write_current) lists every other env folder in current.json's `retired`, and then:
  1. setup, right after the switch, deletes the retired envs it is not running from, removes the torch packages from
     the one it runs from (setup never imports torch) and drops the cache entries that freed (delete_retired with
     running_env);
  2. launch.pyw, as soon as the setup process has exited, deletes what is left (delete_retired without running_env);
  3. setup started some other way (no launch.pyw waiting on it) spawns a detached helper on a Python outside envs/
     that waits for the setup process to exit and does step 2 (spawn_cleanup_helper / main).
launch.pyw also runs step 2 before starting the app whenever `retired` is not empty, as a last fallback.
Deleting never touches the folder current.json points to, nor anything that is not a direct child of <root>/envs.

Cache: `uv cache prune` keeps an unpacked wheel as long as uv's own index points at it, which is always the case
after an env is deleted (measured with uv 0.12.23: deleting the env and pruning removes nothing). uv installs by
hardlinking files out of cache/archive-v0/<id>/, so an archive whose files all have a link count of 1 is used by no
env. gc_cache() moves such archives out of archive-v0 (one atomic rename, so uv never sees half an archive) and then
deletes them; uv treats a missing archive as a cache miss and downloads the wheel again (measured, also --offline
fails cleanly instead of linking a broken package).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from . import layout

TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
_DIST_INFO = re.compile(r"(?P<name>[A-Za-z0-9_.]+)-[^-]+\.dist-info", re.IGNORECASE)
_TRASH = ".oh-trash"                            # cache/.oh-trash: archives on their way out
ENV_LAUNCHER_CLEANUP = "OH_STREAMER_SETUP_LAUNCHER_CLEANUP"   # set by launch.pyw: "I delete the retired envs after you exit"
_REPARSE_POINT = 0x400


@dataclass
class CleanupReport:
    deleted: list[str] = field(default_factory=list)        # env folder names fully removed
    torch_removed: list[str] = field(default_factory=list)  # env folder names whose torch packages were removed
    remaining: list[str] = field(default_factory=list)      # still in current.json's retired
    cache_archives: int = 0
    cache_bytes: int = 0
    errors: list[str] = field(default_factory=list)


# ---- names and folders ----

def new_env_name(install_root, lock_sha: str, flavor: str) -> str:
    """<lock_id>-<flavor>-<n>, n = 1, 2, ... skipping folders that exist and names current.json still mentions:
    a reinstall or a flavor switch never reuses the folder that is in use or waiting for deletion."""
    current = layout.read_current(install_root)
    taken = set(current.retired) | {current.env} if current else set()
    envs = layout.envs_dir(install_root)
    prefix = f"{layout.lock_id(lock_sha)}-{flavor}-"
    n = 1
    while (prefix + str(n)) in taken or os.path.lexists(envs / (prefix + str(n))):
        n += 1
    return prefix + str(n)


def env_folders(install_root) -> list[str]:
    """Names of the real folders directly under envs/ (valid names only, links skipped)."""
    envs = layout.envs_dir(install_root)
    try:
        entries = list(os.scandir(envs))
    except OSError:
        return []
    return sorted(e.name for e in entries
                  if layout.valid_env_name(e.name) and not _is_link(e.path) and e.is_dir(follow_symlinks=False))


def _is_link(path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE_POINT)


def is_deletable_env(install_root, path) -> bool:
    """Only a direct child of <root>/envs with a valid name, and never the env current.json points to."""
    path = Path(os.path.abspath(path))
    envs = Path(os.path.abspath(layout.envs_dir(install_root)))
    if os.path.normcase(str(path.parent)) != os.path.normcase(str(envs)) or not layout.valid_env_name(path.name):
        return False
    current = layout.read_current(install_root)
    if current is not None:
        live = Path(os.path.abspath(layout.current_env_dir(install_root, current)))
        if os.path.normcase(str(live)) == os.path.normcase(str(path)):
            return False
    return True


def _rmtree_once(path: Path, errors: list[str]) -> None:
    def on_error(func, target, exc):
        exc = exc[1] if isinstance(exc, tuple) else exc
        if isinstance(exc, FileNotFoundError):
            return
        try:
            os.chmod(target, stat.S_IWRITE)                                 # read-only files
            func(target)
        except FileNotFoundError:
            pass
        except OSError as again:
            errors.append(f"{target}: {again}")

    if _is_link(path):                                                  # a junction/symlink itself: remove the link only
        try:
            os.rmdir(path) if path.is_dir() else os.unlink(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=on_error)
    else:
        shutil.rmtree(path, onerror=on_error)


def remove_tree(path, *, attempts: int = 5, delay_s: float = 1.0) -> list[str]:
    """Delete a folder, retrying while files are still locked (a process that just exited, antivirus).
    Returns the errors of the last attempt; [] means the folder is gone."""
    path = Path(path)
    errors: list[str] = []
    for attempt in range(max(1, attempts)):
        if not os.path.lexists(path):
            return []
        errors = []
        _rmtree_once(path, errors)
        if not os.path.lexists(path):
            return []
        if attempt + 1 < attempts:
            time.sleep(delay_s)
    return errors or [f"{path}: 無法刪除"]


def discard_env(install_root, path, *, attempts: int = 5, delay_s: float = 1.0) -> list[str]:
    """Remove a half-built env after a failed or cancelled install (only a deletable folder; else an error)."""
    if not is_deletable_env(install_root, path):
        return [f"{path}: 不是可以刪除的環境資料夾"]
    return remove_tree(path, attempts=attempts, delay_s=delay_s)


def clear_cache_temp(install_root, *, attempts: int = 3, delay_s: float = 1.0) -> int:
    """Delete uv's top-level `.tmp*` folders in the cache. uv unpacks a wheel there before moving it into archive-v0;
    a cancelled or killed sync leaves it behind (a half-unpacked torch is ~2 GB) and nothing else ever removes it.
    Only call while no uv runs on this cache (setup holds the single-instance lock). Returns bytes freed."""
    cache = layout.cache_dir(install_root)
    try:
        temps = [p for p in cache.iterdir() if p.name.startswith(".tmp") and p.is_dir() and not _is_link(p)]
    except OSError:
        return 0
    freed, lines = 0, []
    for tmp in temps:
        size = 0
        for f in _walk_files(tmp):
            try:
                size += os.stat(f).st_size
            except OSError:
                pass
        errors = remove_tree(tmp, attempts=attempts, delay_s=delay_s)
        if errors:
            lines.append(f"cache temp: {tmp.name} not deleted: {errors[0]}")
        else:
            freed += size
            lines.append(f"cache temp: deleted {tmp.name} ({size / 2**30:.2f} GiB)")
    if lines:
        _log(install_root, lines)
    return freed


# ---- torch packages inside one env ----

def _site_packages(env_dir: Path) -> list[Path]:
    found = [env_dir / "Lib" / "site-packages"]
    lib = env_dir / "lib"
    if lib.is_dir() and not _is_link(lib):
        found += sorted(lib.glob("python*/site-packages"))
    return [p for p in found if p.is_dir() and not _is_link(p)]


def torch_paths(env_dir) -> list[Path]:
    """Top-level site-packages entries that belong to torch/torchvision/torchaudio (from their dist-info RECORD),
    plus the dist-info folders themselves. Entries outside site-packages (Scripts\\torchrun.exe) are left alone."""
    wanted = {name.lower() for name in TORCH_PACKAGES}
    out: dict[str, Path] = {}
    for site in _site_packages(Path(env_dir)):
        try:
            names = os.listdir(site)
        except OSError:
            continue
        for name in names:
            m = _DIST_INFO.fullmatch(name)
            if not m or m.group("name").lower().replace("-", "_") not in wanted:
                continue
            dist = site / name
            tops = {name}
            try:
                record = (dist / "RECORD").read_text(encoding="utf-8", errors="replace")
            except OSError:
                record = ""
                tops.add(m.group("name"))
            for line in record.splitlines():
                rel = line.split(",", 1)[0].strip().replace("\\", "/")
                top = rel.split("/", 1)[0]
                if top and top not in (".", "..") and not rel.startswith("/") and ":" not in top:
                    tops.add(top)
            for top in tops:
                candidate = site / top
                if os.path.lexists(candidate):
                    out[os.path.normcase(str(candidate))] = candidate
    return sorted(out.values())


def remove_torch_packages(env_dir) -> list[str]:
    """Delete the torch packages from an env (it keeps working as a base env). Returns errors."""
    errors: list[str] = []
    for path in torch_paths(env_dir):
        if path.is_dir() and not _is_link(path):
            errors += remove_tree(path, attempts=2, delay_s=0.5)
        else:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{path}: {exc}")
    return errors


def freeable_torch_bytes(env_dir) -> int:
    """Bytes deleting the env's torch packages (and then their cache archives) would give back: each file once,
    only when nothing but this env and the cache links it (link count <= 2)."""
    seen: set = set()
    total = 0
    for top in torch_paths(env_dir):
        for path in _walk_files(top):
            try:
                st = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            key = (st.st_dev, st.st_ino) if st.st_ino else path
            if key in seen or st.st_nlink > 2:
                continue
            seen.add(key)
            total += st.st_size
    return total


def _walk_files(top: Path):
    if not top.is_dir() or _is_link(top):
        if top.is_file():
            yield str(top)
        return
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if not _is_link(os.path.join(dirpath, d))]
        for name in filenames:
            yield os.path.join(dirpath, name)


# ---- uv cache ----

@dataclass
class GcResult:
    archives: list[str] = field(default_factory=list)       # archive ids removed (or that would be, with dry_run)
    bytes_freed: int = 0
    kept: int = 0
    skipped: str = ""                                       # why nothing was done
    errors: list[str] = field(default_factory=list)


def _archive_in_use(archive: Path) -> tuple[bool, int]:
    """(some file has another link, total bytes). Stops at the first linked file. A file we cannot stat counts as in use."""
    size = 0
    any_file = False
    for path in _walk_files(archive):
        any_file = True
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError:
            return True, size
        if st.st_nlink != 1:
            return True, size
        size += st.st_size
    return not any_file, size                                           # an empty archive: leave it alone


def gc_cache(cache_dir, *, dry_run: bool = False) -> GcResult:
    """Remove cache/archive-v0 entries no env links any more. Does nothing when no archive at all is linked: then the
    envs were installed by copying (another drive, or UV_LINK_MODE=copy) and link counts say nothing."""
    result = GcResult()
    cache = Path(cache_dir)
    archives = cache / "archive-v0"
    if not archives.is_dir() or _is_link(archives):
        result.skipped = "沒有 archive-v0"
        return result
    unused: list[tuple[Path, int]] = []
    try:
        entries = [Path(e.path) for e in os.scandir(archives) if e.is_dir(follow_symlinks=False) and not _is_link(e.path)]
    except OSError as exc:
        result.skipped = f"讀不到快取：{exc}"
        return result
    for archive in entries:
        in_use, size = _archive_in_use(archive)
        if in_use:
            result.kept += 1
        else:
            unused.append((archive, size))
    if result.kept == 0:
        result.skipped = "快取裡沒有任何檔案被環境硬連結，不清理"
        return result
    trash = cache / _TRASH
    for archive, size in unused:
        if dry_run:
            result.archives.append(archive.name)
            result.bytes_freed += size
            continue
        try:
            trash.mkdir(exist_ok=True)
            moved = trash / f"{archive.name}-{os.getpid()}-{time.monotonic_ns()}"
            os.rename(archive, moved)                                   # atomic: uv sees the whole archive or none
        except OSError as exc:
            result.errors.append(f"{archive}: {exc}")
            continue
        result.archives.append(archive.name)
        result.bytes_freed += size
        result.errors += remove_tree(moved, attempts=2, delay_s=0.5)
    if not dry_run and trash.is_dir():
        for leftover in trash.iterdir():                                # earlier runs that could not finish
            remove_tree(leftover, attempts=1)
        try:
            trash.rmdir()
        except OSError:
            pass
    return result


# ---- the cleanup after a switch ----

def _log(install_root, lines: list[str]) -> None:
    try:
        logs = layout.logs_dir(install_root)
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(logs / "env-cleanup.log", "a", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(f"{stamp} [{os.getpid()}] {line}\n" for line in lines))
    except Exception:
        pass


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def delete_retired(install_root, *, running_env=None, attempts: int = 5, delay_s: float = 1.0,
                   gc: bool = True) -> CleanupReport:
    """Delete the envs current.json lists as retired, then drop the cache archives that freed. running_env: the env
    this process runs from (setup mode); it only loses its torch packages and stays listed for launch.pyw.
    Never raises; everything is also written to logs/env-cleanup.log."""
    report = CleanupReport()
    try:
        _delete_retired(install_root, running_env, attempts, delay_s, gc, report)
    except Exception as exc:                                            # noqa: BLE001 - cleanup must never break a launch
        report.errors.append(f"未預期的錯誤：{type(exc).__name__}: {exc}")
    lines = ([f"deleted {n}" for n in report.deleted] + [f"torch removed from {n}" for n in report.torch_removed]
             + [f"remaining {report.remaining}"] * bool(report.remaining)
             + [f"cache: {report.cache_archives} archives, {report.cache_bytes / 2**30:.2f} GiB"] * bool(report.cache_archives)
             + [f"error {e}" for e in report.errors[:50]])
    if lines:
        _log(install_root, lines)
    return report


def _delete_retired(install_root, running_env, attempts, delay_s, gc, report: CleanupReport) -> None:
    current = layout.read_current(install_root)
    if current is None or not current.retired:
        return
    envs = layout.envs_dir(install_root)
    live = current.env or layout.lock_id(current.lock_sha256)
    remaining = []
    freed_something = False
    for name in current.retired:
        path = envs / name
        if name == live:                                                # never delete what current.json points to
            continue
        if not os.path.lexists(path):
            continue
        if not is_deletable_env(install_root, path):
            report.errors.append(f"{path}: 不在 envs 底下或名稱不合法，略過")
            continue
        if running_env is not None and _same(path, running_env):
            errors = remove_torch_packages(path)
            report.errors += errors
            if not errors:
                report.torch_removed.append(name)
            freed_something = True
            remaining.append(name)
            continue
        errors = remove_tree(path, attempts=attempts, delay_s=delay_s)
        freed_something = True
        if errors:
            report.errors += errors[:20]
            remaining.append(name)
        else:
            report.deleted.append(name)

    latest = layout.read_current(install_root)                          # write back only the retired list
    if latest is not None and latest.retired != tuple(remaining):
        keep = [n for n in latest.retired if n in remaining or n not in current.retired]
        layout.write_current(install_root, replace(latest, retired=tuple(keep)))
    report.remaining = remaining

    if gc and freed_something:
        result = gc_cache(layout.cache_dir(install_root))
        report.cache_archives = len(result.archives)
        report.cache_bytes = result.bytes_freed
        report.errors += result.errors[:20]
        if result.skipped:
            report.errors.append(f"快取：{result.skipped}")


def free_old_envs_first(install_root, *, running_env=None) -> CleanupReport:
    """「先刪除舊環境再安裝」: mark the current env unverified (it is about to lose torch), then delete every env
    folder except the one this process runs from, which only loses its torch packages, and drop the freed cache
    archives. After this a failed install leaves no working GPU env: the next launch goes to setup again."""
    report = CleanupReport()
    current = layout.read_current(install_root)
    if current is not None:
        layout.write_current(install_root, replace(current, verified=False))
    envs = layout.envs_dir(install_root)
    for name in env_folders(install_root):
        path = envs / name
        if running_env is not None and _same(path, running_env):
            errors = remove_torch_packages(path)
            if not errors:
                report.torch_removed.append(name)
        else:
            errors = remove_tree(path, attempts=3, delay_s=0.5)
            if not errors:
                report.deleted.append(name)
        report.errors += errors[:20]
    result = gc_cache(layout.cache_dir(install_root))
    report.cache_archives, report.cache_bytes = len(result.archives), result.bytes_freed
    report.errors += result.errors[:20]
    current = layout.read_current(install_root)
    if current is not None and report.deleted:
        layout.write_current(install_root, replace(current, retired=tuple(n for n in current.retired if n not in report.deleted)))
    _log(install_root, [f"delete-old-first: deleted {report.deleted}, torch removed {report.torch_removed}, "
                        f"cache {report.cache_archives} archives {report.cache_bytes / 2**30:.2f} GiB"]
         + [f"error {e}" for e in report.errors[:50]])
    return report


# ---- helper process (setup not started by launch.pyw) ----

def helper_python(install_root) -> Path | None:
    """A Python outside envs/: the embedded one, else the base interpreter of the running venv. pythonw when there is one."""
    candidates = [Path(install_root) / "python" / "pythonw.exe", Path(install_root) / "python" / "python.exe"]
    base = getattr(sys, "_base_executable", None) or sys.executable
    if base:
        base = Path(base)
        candidates += [base.with_name("pythonw.exe"), base] if sys.platform == "win32" else [base]
    envs = os.path.normcase(os.path.abspath(layout.envs_dir(install_root))) + os.sep
    for path in candidates:
        if path.is_file() and not os.path.normcase(os.path.abspath(path)).startswith(envs):
            return path
    return None


def spawn_cleanup_helper(install_root, wait_pids: list[int]) -> bool:
    """Start a detached `delete_retired` that first waits for wait_pids to exit. False when no outside Python exists."""
    python = helper_python(install_root)
    if python is None:
        _log(install_root, ["helper: 找不到 envs 以外的 Python，留給下次啟動清理"])
        return False
    src = layout.app_dir(install_root) / "src"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from oh_streamer.runtime_env import env_cleanup; sys.exit(env_cleanup.main(sys.argv[2:]))")
    args = [str(python), "-I", "-c", code, str(src), "--root", str(install_root)]
    for pid in wait_pids:
        args += ["--wait-pid", str(pid)]
    flags = 0x00000008 | 0x00000200 | 0x08000000 if sys.platform == "win32" else 0   # detached, own group, no window
    try:
        subprocess.Popen(args, cwd=str(install_root), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags, start_new_session=sys.platform != "win32")
    except OSError as exc:
        _log(install_root, [f"helper: 無法啟動：{exc}"])
        return False
    _log(install_root, [f"helper: started, waiting for {wait_pids}"])
    return True


def run_after_setup(install_root, *, lock_timeout_s: float = 30.0) -> CleanupReport | None:
    """launch.pyw / the helper: once setup has exited, take the setup lock (so no new setup starts mid-way) and delete
    the retired envs. None when another setup holds the lock (it will clean up itself)."""
    from . import env_guard
    current = layout.read_current(install_root)
    if current is None or not current.retired:
        return CleanupReport()
    lock = env_guard.SetupLock(install_root)
    deadline = time.monotonic() + lock_timeout_s
    while not lock.acquire():
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)
    try:
        return delete_retired(install_root)
    finally:
        lock.release()


def main(argv: list[str]) -> int:
    from . import env_guard
    root, pids = None, []
    it = iter(argv)
    for arg in it:
        if arg == "--root":
            root = next(it, None)
        elif arg == "--wait-pid":
            try:
                pids.append(int(next(it, "")))
            except ValueError:
                pass
    if not root:
        return 2
    for pid in pids:
        env_guard.wait_for_exit(pid, 600)
    run_after_setup(root)
    return 0
