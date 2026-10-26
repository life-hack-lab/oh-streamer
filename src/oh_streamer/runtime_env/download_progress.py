r"""Download progress for `uv sync` (bytes, speed, time left). Stdlib only, like installer.

uv prints no progress bar when its output is a pipe, and no flag or env var forces one. What it does give us
(measured with uv 0.12.23):

* `uv sync ... --dry-run -v` logs `Identified uncached distribution: name==version` for every package it will download
  (nothing on a warm cache) and `Would download N packages`. That is the plan; uv.lock gives each wheel's size.
* While syncing it prints `Downloading torch (3.2GiB)` / ` Downloaded torch` for packages above ~1 MiB.
* It unzips wheels while they stream into `<cache>\.tmpXXXX\`, so the cache grows by *uncompressed* bytes, and the
  uncompressed/compressed ratio inside torch swings from 1.4 to 2.2. Network bytes are not in the process IO counters.

So: finished packages count with their full size, and a large wheel in flight is measured exactly by reading its zip
central directory (two HTTP range requests, ~1 MB for torch): the files uv has already unpacked, in archive order,
map to an offset in the compressed stream. Anything we cannot measure yields None, and the UI keeps its indeterminate
display instead of showing a wrong number.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import tomllib
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

BIG_BYTES = 64 * 2**20                  # wheels at least this big are measured while they download
UNTRACKED_GRACE_S = 10.0                # a big download we cannot see for this long makes progress unknown
SPEED_WINDOW_S = 12.0                   # moving window for the speed (and so the ETA)
MIN_SPEED_SPAN_S = 2.0
SAMPLE_S = 1.0
DRY_RUN_TIMEOUT_S = 60.0
HTTP_TIMEOUT_S = 20.0
_EOCD_TAIL = 65536 + 22                 # max zip comment + end-of-central-directory record
_MAX_CD_BYTES = 32 * 2**20
_CREATE_NO_WINDOW = 0x08000000


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------------------------------------- uv.lock sizes

@dataclass(frozen=True)
class Artifact:
    url: str | None                     # None: a local file (nothing to download)
    size: int | None                    # None: the lock does not say (download.pytorch.org wheels)


def target_python(lock: dict) -> tuple[int, int]:
    """The env's python from requires-python (e.g. "==3.13.*"); falls back to this interpreter."""
    m = re.search(r"==\s*(\d+)\.(\d+)", str(lock.get("requires-python", "")))
    return (int(m.group(1)), int(m.group(2))) if m else sys.version_info[:2]


def platform_tag() -> str:
    return re.sub(r"[-.]", "_", sysconfig.get_platform())


def wheel_rank(filename: str, py: tuple[int, int], plat: str) -> float | None:
    """Lower is preferred (the order pip/uv use: cpXY-cpXY > abi3 > none, platform wheel > any); None = incompatible."""
    parts = filename[:-4].split("-") if filename.endswith(".whl") else []
    if len(parts) < 5:
        return None
    pys, abis, plats = parts[-3].split("."), parts[-2].split("."), parts[-1].split(".")
    if plat in plats:
        plat_penalty = 0
    elif "any" in plats:
        plat_penalty = 3
    else:
        return None
    cp = f"cp{py[0]}{py[1]}"
    best = None
    for abi in abis:
        if abi == cp and cp in pys:
            rank = 0.0
        elif abi == "abi3":
            minors = [int(p[3:]) for p in pys if re.fullmatch(rf"cp{py[0]}\d+", p) and int(p[3:]) <= py[1]]
            if not minors:
                continue
            rank = 1.0 + (py[1] - max(minors)) / 100        # newer abi3 baseline first
        elif abi == "none" and any(p in (cp, f"py{py[0]}{py[1]}", f"py{py[0]}") for p in pys):
            rank = 2.0
        else:
            continue
        best = rank if best is None else min(best, rank)
    return None if best is None else best + plat_penalty


def lock_artifacts(lock_text: str, py: tuple[int, int] | None = None, plat: str | None = None) -> dict:
    """{(normalized name, version): Artifact} for the wheel (or sdist) uv would download on this platform.
    Packages without a usable file map to None: their size is unknown."""
    lock = tomllib.loads(lock_text)
    py = py or target_python(lock)
    plat = plat or platform_tag()
    out: dict = {}
    for pkg in lock.get("package", []):
        key = (normalize(pkg.get("name", "")), str(pkg.get("version", "")))
        source = pkg.get("source", {})
        if not ("registry" in source or "url" in source):
            out[key] = Artifact(None, 0)                    # path / editable / git: not a download we can size
            continue
        ranked = []
        for wheel in pkg.get("wheels", []):
            url = wheel.get("url")
            if not url:
                continue
            rank = wheel_rank(url.rsplit("/", 1)[-1].split("#")[0], py, plat)
            if rank is not None:
                ranked.append((rank, url, wheel.get("size")))
        if ranked:
            _, url, size = min(ranked, key=lambda r: r[0])
            out[key] = Artifact(url, size)
        elif pkg.get("sdist", {}).get("url"):
            sdist = pkg["sdist"]
            out[key] = Artifact(sdist["url"], sdist.get("size"))
        else:
            out[key] = None
    return out


# ---------------------------------------------------------------------------------------------- the plan (dry run)

_UNCACHED = re.compile(r"Identified uncached distribution: (?P<spec>.+?)\s*$")
_WOULD_DOWNLOAD = re.compile(r"Would download (\d+) packages?")


def parse_dry_run(text: str) -> list[tuple[str, str | None]] | None:
    """[(normalized name, version or None for a direct URL / file)] that uv will download; None if the output
    does not look like what we know (then progress stays unknown)."""
    specs: dict[str, tuple[str, str | None]] = {}
    for line in text.splitlines():
        m = _UNCACHED.search(line)
        if not m:
            continue
        spec = m.group("spec")
        if "==" in spec and " @ " not in spec:
            name, version = spec.split("==", 1)
            specs[spec] = (normalize(name.strip()), version.strip())
        elif " @ " in spec:
            name, url = spec.split(" @ ", 1)
            specs[spec] = (normalize(name.strip()), None if url.startswith("file:") else url.strip())
        else:
            return None
    m = _WOULD_DOWNLOAD.search(text)
    if len(specs) != (int(m.group(1)) if m else 0):                     # the two must agree, or we misread uv
        return None
    return list(specs.values())


def run_dry_run(cmd: Sequence[str], cwd, env: dict, cancel_event: threading.Event | None,
                timeout_s: float = DRY_RUN_TIMEOUT_S) -> str | None:
    """`<uv sync ...> --dry-run -v` output, or None (failed, timed out, cancelled). Stops only the process it started."""
    try:
        proc = subprocess.Popen(
            [*cmd, "--dry-run", "-v"], cwd=str(cwd), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except OSError:
        return None
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            out, _ = proc.communicate(timeout=0.2)
            return out if proc.returncode == 0 else None
        except subprocess.TimeoutExpired:
            if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                return None


# ---------------------------------------------------------------------------------------------- zip central directory

@dataclass(frozen=True)
class ZipIndex:
    size: int                                   # compressed file size
    offsets: list                               # local header offset per file entry, ascending
    csizes: list
    usizes: list
    names: list                                 # archive names ("torch/lib/x.dll"), same order

    def position(self, root: Path) -> int | None:
        """Compressed bytes uv has consumed, judged from the files already unpacked under root (uv writes them in
        archive order). None when root holds none of them."""
        n = len(self.names)
        if n == 0 or not _exists(root, self.names[0]):
            return None
        lo, hi = 0, n - 1                       # last index whose file exists
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _exists(root, self.names[mid]):
                lo = mid
            else:
                hi = mid - 1
        try:
            have = os.stat(_join(root, self.names[lo])).st_size
        except OSError:
            have = 0
        usize = self.usizes[lo]
        frac = min(1.0, have / usize) if usize else 1.0
        return min(self.size, int(self.offsets[lo] + self.csizes[lo] * frac))


def _join(root: Path, name: str) -> str:
    return os.path.join(root, *name.split("/"))


def _exists(root: Path, name: str) -> bool:
    return os.path.exists(_join(root, name))


def parse_central_directory(cd: bytes, size: int) -> ZipIndex:
    rows = []
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        csize, usize, nlen, xlen, clen = struct.unpack("<IIHHH", cd[pos + 20:pos + 34])
        offset = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):                                      # zip64 extra field: 64-bit values in this order
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + hlen // 8 * 8]))
                if usize == 0xFFFFFFFF and vals:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF and vals:
                    csize = vals.pop(0)
                if offset == 0xFFFFFFFF and vals:
                    offset = vals.pop(0)
            q += 4 + hlen
        if not name.endswith("/"):
            rows.append((offset, csize, usize, name))
        pos += 46 + nlen + xlen + clen
    if not rows:
        raise ValueError("empty or unreadable central directory")
    rows.sort()
    return ZipIndex(size, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])


def _range(url: str, spec: str, timeout: float) -> tuple[bytes, int]:
    """(body, full file size) of a ranged GET."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={spec}", "User-Agent": "oh-streamer-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(_MAX_CD_BYTES + _EOCD_TAIL + 1)
        m = re.search(r"/(\d+)\s*$", resp.headers.get("Content-Range", ""))
        if resp.status != 206 or not m:
            raise ValueError("server ignored the range request")
        return body, int(m.group(1))


def fetch_zip_index(url: str, timeout: float = HTTP_TIMEOUT_S) -> ZipIndex:
    tail, size = _range(url, f"-{_EOCD_TAIL}", timeout)
    tail_start = size - len(tail)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or i + 22 > len(tail):
        raise ValueError("no end of central directory")
    count, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x07", 0, i)
        if j < 0:
            raise ValueError("zip64 locator missing")
        rec = struct.unpack("<Q", tail[j + 8:j + 16])[0] - tail_start
        if rec < 0 or tail[rec:rec + 4] != b"PK\x06\x06":
            raise ValueError("zip64 record not in the tail")
        cd_size, cd_off = struct.unpack("<QQ", tail[rec + 40:rec + 56])
    if cd_size > _MAX_CD_BYTES:
        raise ValueError("central directory too large")
    if cd_off >= tail_start:
        cd = tail[cd_off - tail_start:cd_off - tail_start + cd_size]
    else:
        cd, _ = _range(url, f"{cd_off}-{cd_off + cd_size - 1}", timeout)
    return parse_central_directory(cd, size)


# ---------------------------------------------------------------------------------------------- speed / ETA

class RateMeter:
    """Bytes per second over a moving window, so one burst does not swing the ETA."""

    def __init__(self, window_s: float = SPEED_WINDOW_S, min_span_s: float = MIN_SPEED_SPAN_S):
        self.window_s = window_s
        self.min_span_s = min_span_s
        self._samples: deque = deque()

    def add(self, t: float, done: int) -> None:
        self._samples.append((t, done))
        while len(self._samples) > 2 and t - self._samples[1][0] >= self.window_s:
            self._samples.popleft()

    def speed(self) -> float | None:
        if len(self._samples) < 2:
            return None
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        if t1 - t0 < self.min_span_s:
            return None
        return max(0.0, (b1 - b0) / (t1 - t0))


def eta_seconds(remaining: int, speed: float | None) -> int | None:
    if remaining <= 0:
        return 0
    if not speed or speed < 1024:
        return None
    return int(round(remaining / speed))


# ---------------------------------------------------------------------------------------------- tracker

_DOWNLOADING = re.compile(r"^\s*Downloading (?P<name>\S+) \((?P<num>[\d.]+)(?P<unit>[KMGT]i?B|B)\)\s*$")
_DOWNLOADED = re.compile(r"^\s*Downloaded (?P<name>\S+)\s*$")
_PREPARED = re.compile(r"^\s*Prepared \d+ packages? in ")
_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


@dataclass
class _Item:
    size: int | None
    url: str | None
    lock_sized: bool                    # the lock gave the size: trust the plan and count it from the start
    announced_at: float | None = None   # uv printed "Downloading <name> (...)"
    done: bool = False
    dropped: bool = False               # planned but never downloaded (uv found it in its HTTP cache)
    index: ZipIndex | None = None
    index_tried: bool = False
    tmp: Path | None = None
    partial: int = 0

    @property
    def counted(self) -> bool:
        return not self.dropped and self.size is not None and (self.lock_sized or self.announced_at is not None)


class DownloadTracker:
    """Turns the plan + uv's output lines + the unpack dir into {downloaded_bytes, total_bytes, speed_bps, eta_s}.

    The dry-run plan over-reports for wheels from download.pytorch.org: it calls them uncached although the real sync
    then takes them from uv's HTTP cache in 1 ms. Those wheels also have no size in the lock, so they only count once
    uv announces their download; one that is never announced was not downloaded.
    """

    def __init__(self, items: dict[str, tuple[int | None, str | None]], cache_dir, *,
                 clock: Callable[[], float] = time.monotonic, big_bytes: int = BIG_BYTES,
                 untracked_grace_s: float = UNTRACKED_GRACE_S):
        self._items = {name: _Item(size, url, size is not None) for name, (size, url) in items.items()}
        self._cache = Path(cache_dir)
        self._clock = clock
        self._big = big_bytes
        self._grace = untracked_grace_s
        self._lock = threading.Lock()
        self._rate = RateMeter()
        self._shown = 0

    @property
    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)

    def fetch_pending(self, fetch_index: Callable[[str], ZipIndex] = fetch_zip_index) -> None:
        """Network part, run on the sampler thread: the zip index of each big wheel that is (or will be) downloading."""
        with self._lock:
            todo = [i for i in self._items.values()
                    if not i.index_tried and not i.done and i.url and i.url.split("#")[0].endswith(".whl")
                    and i.size is not None and i.size >= self._big and (i.lock_sized or i.announced_at is not None)]
            for item in todo:
                item.index_tried = True
        for item in todo:
            try:
                index = fetch_index(item.url)
            except Exception:                                           # noqa: BLE001 - unmeasurable stays unmeasurable
                continue
            with self._lock:
                item.index = index
                if not item.lock_sized:
                    item.size = index.size                              # exact, instead of uv's rounded "3.2GiB"

    def on_line(self, line: str) -> None:
        with self._lock:
            m = _DOWNLOADING.match(line)
            if m:
                item = self._items.get(normalize(m.group("name")))
                if item is not None and not item.done:
                    item.announced_at = self._clock()
                    if item.size is None:                               # uv's rounded figure beats nothing
                        item.size = int(float(m.group("num")) * _UNITS.get(m.group("unit"), 1))
                return
            m = _DOWNLOADED.match(line)
            if m:
                item = self._items.get(normalize(m.group("name")))
                if item is not None:
                    item.done = True
                return
            if _PREPARED.match(line):                                   # every download of this sync is finished
                for item in self._items.values():
                    if not item.done and not item.lock_sized and item.announced_at is None:
                        item.dropped = True
                    item.done = True

    def _measure(self, item: _Item) -> int | None:
        """Compressed bytes of a big wheel in flight, or None when we cannot see it."""
        if item.index is None:
            return None
        if item.tmp is not None:
            pos = item.index.position(item.tmp)
            if pos is not None:
                return pos
            item.tmp = None                                             # dir gone (renamed or retried): look again
        try:
            dirs = [Path(e.path) for e in os.scandir(self._cache) if e.name.startswith(".tmp") and e.is_dir()]
        except OSError:
            return None
        for d in dirs:
            pos = item.index.position(d)
            if pos is not None:
                item.tmp = d
                return pos
        return None

    def snapshot(self) -> dict | None:
        """None while nothing measurable is downloading (warm cache, unreadable plan, a big download we cannot see)."""
        with self._lock:
            items = [i for i in self._items.values() if i.counted]
            if not items:
                return None
            now = self._clock()
            total = sum(i.size for i in items)
            done = 0
            streaming = False                                           # a big wheel measured byte by byte right now
            for item in items:
                if item.done:
                    done += item.size
                    continue
                if item.size < self._big:
                    continue                                            # small: counted when uv says it is done
                pos = self._measure(item)
                if pos is not None:
                    item.partial = max(item.partial, pos)
                    streaming = True
                elif item.partial == 0 and item.announced_at is not None and now - item.announced_at >= self._grace:
                    return None                                         # a big download we cannot see: no fake bar
                done += min(item.partial, item.size)
            self._shown = min(total, max(self._shown, done))
            self._rate.add(now, self._shown)
            speed = self._rate.speed()
            remaining = total - self._shown
            if remaining > 0 and not streaming:
                # only small packages left: bytes move in whole-package jumps, so speed and ETA would read 0 / swing
                # wildly (measured on the base phase); show the amount alone
                speed = None
            return {
                "downloaded_bytes": self._shown,
                "total_bytes": total,
                "speed_bps": None if remaining <= 0 or speed is None else round(speed),
                "eta_s": 0 if remaining <= 0 else eta_seconds(remaining, speed),
            }


def build_tracker(dry_run_output: str | None, lock_text: str, cache_dir, **kw) -> DownloadTracker | None:
    """None when the plan cannot be read or a planned package is not in the lock; an empty tracker on a warm cache."""
    if dry_run_output is None:
        return None
    plan = parse_dry_run(dry_run_output)
    if plan is None:
        return None
    try:
        artifacts = lock_artifacts(lock_text)
    except (tomllib.TOMLDecodeError, ValueError, TypeError, KeyError):
        return None
    items: dict[str, tuple[int | None, str | None]] = {}
    for name, version in plan:
        if version is None:                                             # local file: nothing comes over the network
            continue
        art = artifacts.get((name, version))
        if art is None:
            return None
        if art.url is None:
            continue
        items[name] = (art.size, art.url)
    return DownloadTracker(items, cache_dir, **kw)


class ProgressSampler:
    """Background thread: every SAMPLE_S, fetch what the tracker still needs and call on_progress(snapshot), until stop()."""

    def __init__(self, tracker: DownloadTracker, on_progress: Callable[[dict | None], None], interval_s: float | None = None):
        self._tracker = tracker
        self._on_progress = on_progress
        self._interval = SAMPLE_S if interval_s is None else interval_s
        self._stop = threading.Event()
        self._emit_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="uv-download-progress", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def on_line(self, line: str) -> None:
        try:
            self._tracker.on_line(line)
        except Exception:                                               # noqa: BLE001 - never stall uv's reader thread
            pass

    def _emit(self, value) -> None:
        with self._emit_lock:                                           # after stop() returns, nothing reaches the UI
            if self._stop.is_set():
                return
            try:
                self._on_progress(value)
            except Exception:                                           # noqa: BLE001 - a broken UI callback must not stop us
                pass

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._tracker.fetch_pending()
                self._emit(self._tracker.snapshot())
                self._stop.wait(self._interval)
        except Exception:                                               # noqa: BLE001 - progress is best effort
            self._emit(None)

    def stop(self, final: bool = True, join_s: float = 1.0) -> None:
        if final and self._thread.is_alive():
            try:
                self._emit(self._tracker.snapshot())
            except Exception:                                           # noqa: BLE001
                pass
        with self._emit_lock:
            self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(join_s)                                   # it may still sit in an HTTP call; it is a daemon
