"""Background keyframe thumbnails for the player's seekbar."""

from __future__ import annotations

import bisect
import math
import os
import threading
from typing import Callable

from oh_streamer.sources import source_resolve


class ThumbnailWorker:
    """Keyframe previews of the original source for seekbar hover.

    A single thread owns its PyAV container. It seeks to the keyframe at or before a requested time, decodes one
    frame, scales it to BGRA, and writes a raw bitmap for ``oh_seek.lua`` to display. Files use the decoded
    frame's own time, allowing buckets in the same GOP to share a file. On-demand hover requests take priority over
    background prefetch, which walks the video at a bounded interval at low thread priority.
    """

    BUCKET = 2.0
    MAX_PREFETCH = 600

    def __init__(
        self,
        src: str,
        headers: dict,
        duration: float,
        width: int,
        directory: str,
        *,
        av_module=None,
        http_options_factory: Callable | None = None,
        url_predicate: Callable[[str], bool] | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self.src, self.duration = src, duration
        self.http = (http_options_factory or source_resolve.FFmpegHttpOptions)(headers)
        self.width, self.dir = width, directory
        self._av = av_module
        self._url_predicate = url_predicate
        self._logger = logger
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closing = threading.Event()
        self.frames: dict[int, tuple[str, int, int]] = {}   # frame time (ms) -> (path, w, h)
        self.times: list[int] = []                          # sorted keys of frames
        self.buckets: dict[int, int | None] = {}            # bucket -> frame time (ms); None = decode failed
        self.want: int | None = None                        # latest on-demand bucket
        self.interval = 0.0
        self.next_k = 0
        self.container = None
        self.vstream = None
        self.size = None
        self.errors = 0
        self.warned = False
        self.decoded = 0
        self.thread = threading.Thread(target=self._run, name="thumbnails", daemon=True)

    # -- API used by the controller thread -----------------------------------------------------------------------

    def start(self):
        self.thread.start()

    def start_prefetch(self, interval: float):
        if interval > 0 and self.duration > 0:
            self.interval = max(interval, self.duration / self.MAX_PREFETCH)
            self.wake.set()

    def query(self, t: float):
        """Return ``(path, w, h, exact)`` or None; queue an on-demand decode for a missing bucket."""
        b = int(t // self.BUCKET)
        with self.lock:
            if b in self.buckets and self.buckets[b] is not None:
                return (*self.frames[self.buckets[b]], True)
            if b not in self.buckets:
                self.want = b
                self.wake.set()
            if not self.times:
                return None
            i = bisect.bisect_left(self.times, t * 1000)
            near = [self.times[j] for j in (i - 1, i) if 0 <= j < len(self.times)]
            ms = min(near, key=lambda x: abs(x - t * 1000))
            return (*self.frames[ms], b in self.buckets)   # a failed bucket: the nearest one is final

    def close(self):
        self.closing.set()
        self.wake.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    # -- worker thread -------------------------------------------------------------------------------------------

    @staticmethod
    def _set_priority(level: int):
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), level)
        except Exception:
            pass

    def _warn(self, msg: str):
        if not self.warned:
            self.warned = True
            logger = self._logger or source_resolve.log
            logger(f"thumbnails: {msg} (further errors are not logged)")

    def _open(self):
        is_url = self._url_predicate or source_resolve.is_url
        opts = self.http.av_options() if is_url(self.src) else {}
        av_module = self._av
        if av_module is None:
            import av as av_module
        self.container = av_module.open(self.src, metadata_errors="ignore", options=opts)
        vs = self.vstream = self.container.streams.video[0]
        cc = vs.codec_context
        cc.thread_count = 1                 # keep the CPU footprint to one core
        cc.skip_frame = "NONKEY"            # keyframe-only decoding is all a preview needs
        w, h = cc.width or vs.width, cc.height or vs.height
        self.size = (self.width, max(2, round(self.width * h / w / 2) * 2))

    def _decode(self, bucket: int):
        with self.lock:
            if bucket in self.buckets:
                return
        t = min(bucket * self.BUCKET, max(0.0, self.duration - 0.1))
        try:
            if self.container is None:
                self._open()
            vs = self.vstream
            self.container.seek(int(t / vs.time_base), stream=vs, backward=True, any_frame=False)
            frame = None
            for packet in self.container.demux(vs):
                for frame in packet.decode():
                    break
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError("no frame decoded")
            ft = float(frame.pts * vs.time_base) if frame.pts is not None else t
            ms = int(round(ft * 1000))
            with self.lock:
                known = self.frames.get(ms)
            if known is None:
                w, h = self.size
                data = frame.reformat(width=w, height=h, format="bgra").to_ndarray().tobytes()
                path = os.path.join(self.dir, f"{ms:010d}.bgra")
                with open(path + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(path + ".tmp", path)   # mpv never sees a half-written file
                known = (path, w, h)
            with self.lock:
                if ms not in self.frames:
                    self.frames[ms] = known
                    bisect.insort(self.times, ms)
                # no keyframe lies between ft and t, so every bucket in between has the same preview
                for b in range(math.ceil(ft / self.BUCKET), bucket):
                    self.buckets.setdefault(b, ms)
                self.buckets[bucket] = ms
            self.decoded += 1
            self.errors = 0
        except Exception as e:  # noqa: BLE001 - a preview must never take anything else down
            self.errors += 1
            self._warn(f"decode at {t:.1f}s failed: {e!r}")
            self.container = self.vstream = None   # reopen next time (network hiccup, bad seek)
            if self.errors >= 3:                   # give up on this bucket; the nearest thumbnail stays
                self.errors = 0
                with self.lock:
                    self.buckets[bucket] = None

    def _next_prefetch(self):
        while self.interval > 0:
            t = self.next_k * self.interval
            if t >= self.duration:
                self.interval = 0.0
                return None
            self.next_k += 1
            b = int(t // self.BUCKET)
            with self.lock:
                if b not in self.buckets:
                    return b
        return None

    def _run(self):
        try:
            while not self.closing.is_set():
                with self.lock:
                    b, self.want = self.want, None
                if b is not None:
                    self._set_priority(0)
                    self._decode(b)
                    continue
                b = self._next_prefetch()
                if b is not None:
                    self._set_priority(-2)   # THREAD_PRIORITY_LOWEST: stay out of the restoration pipeline's way
                    self._decode(b)
                    self.closing.wait(0.02)
                    continue
                self.wake.wait(0.5)
                self.wake.clear()
        finally:
            try:
                if self.container is not None:
                    self.container.close()
            except Exception:
                pass
