"""On-demand HLS I-frame previews, independent of the restoration producer.

Reading the producer's cache never requests production or changes its seek
anchor. An original-source preview is replaced when restored bytes arrive.
"""

from collections import OrderedDict
import subprocess
import threading
import time


def _priority(key):
    return {"original": 0, "ready": 1, "restored": 2}[key[2][0]]


class HlsThumbnails:
    codecs = "avc1.42c01e"
    bandwidth = 512_000

    def __init__(self, vod, src, input_args=(), *, fmt_start=0.0,
                 stream_index=None, width=320, height=180,
                 cache_bytes=8 * 1024 * 1024, max_workers=2, timeout=10.0,
                 ready_source=None):
        self.vod, self.plan, self.src = vod, vod.plan, src
        self.input_args = tuple(input_args)
        self.fmt_start, self.stream_index = float(fmt_start), stream_index
        # Optional non-producing lookup of completed progressive queue media:
        # (path, input arguments, seek seconds relative to file, stream index).
        self.ready_source = ready_source
        ratio = min(1.0, max(2, width) / self.plan.width,
                    max(2, height) / self.plan.height)
        self.width = max(2, int(self.plan.width * ratio) // 2 * 2)
        self.height = max(2, int(self.plan.height * ratio) // 2 * 2)
        self.timeout = max(0.01, float(timeout))
        self._cap = max(0, int(cache_bytes))
        self._cache = OrderedDict()
        self._bytes = 0
        self._generation = None
        self._lock = threading.Lock()
        self._workers = threading.BoundedSemaphore(max(1, int(max_workers)))
        self._pending = {}
        self._processes = set()
        self._closed = False

    def playlist(self):
        lines = ["#EXTM3U", "#EXT-X-VERSION:4",
                 f"#EXT-X-TARGETDURATION:{self.plan.target_duration()}",
                 "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD",
                 "#EXT-X-I-FRAMES-ONLY", "#EXT-X-INDEPENDENT-SEGMENTS"]
        for k in range(self.plan.count):
            lines.extend((f"#EXTINF:{float(self.plan.duration(k)):.6f},", f"thumb_{k}.ts"))
        return "\n".join(lines + ["#EXT-X-ENDLIST", ""])

    def get_segment(self, k):
        """Return one independently decodable preview, or None on overload/failure.

        The semaphore bounds both ffmpeg processes and restored input snapshots.
        Concurrent requests for the same generation/frame share one encode.
        """
        if not isinstance(k, int) or not 0 <= k < self.plan.count:
            return None
        deadline = time.monotonic() + self.timeout
        if not self._workers.acquire(timeout=min(0.1, self.timeout)):
            return None
        try:
            generation, restored = self.vod.peek_segment(k)
            ready = self.ready_source(k) if restored is None and self.ready_source else None
            if ready is not None:
                path, args, seek, index = ready
                ready = (str(path), tuple(args), float(seek), index)
            origin = ("restored",) if restored is not None else (("ready",) + ready if ready else ("original",))
            key = (generation, k, origin)
            with self.vod.cond, self._lock:
                if self._closed or self.vod.closed or generation != self.vod.gen:
                    return None
                if generation != self._generation:
                    self._cache.clear()
                    self._bytes = 0
                    self._generation = generation
                cached = self._cache.get(k)
                if cached is not None and (cached[0] == key or _priority(cached[0]) > _priority(key)):
                    self._cache.move_to_end(k)
                    return cached[1]
                event = self._pending.get(key)
                owner = event is None
                if owner:
                    event = self._pending[key] = threading.Event()
            if not owner:
                event.wait(max(0, deadline - time.monotonic()))
                with self.vod.cond, self._lock:
                    cached = self._cache.get(k)
                    return cached[1] if (not self._closed and not self.vod.closed
                        and self.vod.gen == generation and cached is not None
                        and cached[0][0] == generation
                        and (cached[0] == key or _priority(cached[0]) > _priority(key))) else None
            try:
                data = self._encode(k, restored, max(0.01, deadline - time.monotonic()), ready)
                # Match HlsVod's generation under its lock. Quality changes must
                # not publish an encode that was started from old restored data.
                with self.vod.cond:
                    if self.vod.closed or generation != self.vod.gen:
                        return None
                    with self._lock:
                        if self._closed or not data:
                            return None
                        cached = self._cache.get(k)
                        if cached is not None and _priority(cached[0]) > _priority(key):
                            return cached[1]
                        old = self._cache.pop(k, None)
                        if old is not None:
                            self._bytes -= len(old[1])
                        if len(data) <= self._cap:
                            self._cache[k] = (key, data)
                            self._bytes += len(data)
                            while self._bytes > self._cap:
                                _, (_, removed) = self._cache.popitem(last=False)
                                self._bytes -= len(removed)
                        return data
            finally:
                with self._lock:
                    self._pending.pop(key, None)
                    event.set()
        finally:
            self._workers.release()

    def _command(self, k, restored, ready=None):
        from oh_streamer.platform.hls_cast import TS_BASE, MPEGTS_DELAY
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if restored:
            cmd += ["-f", "mpegts", "-i", "pipe:0", "-map", "0:v:0"]
        else:
            src, args, seek, index = ready or (self.src, self.input_args,
                max(0.0, self.plan.source_time(k) - self.fmt_start), self.stream_index)
            cmd += list(args)
            cmd += ["-ss", f"{max(0.0, seek):.9f}", "-i", str(src), "-map",
                    "0:v:0" if index is None else f"0:{index}"]
        # Normalize either input clock before applying the main rendition's
        # output origin. mpegts adds 2 * muxdelay, just as the video encoder does.
        cmd += ["-an", "-sn", "-dn", "-frames:v", "1", "-vf",
                f"scale={self.width}:{self.height},setsar=1,setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast",
                "-tune", "stillimage", "-profile:v", "baseline", "-level:v", "3.0",
                "-pix_fmt", "yuv420p", "-g", "1", "-bf", "0", "-crf", "25",
                "-maxrate", "256k", "-bufsize", "256k",
                "-output_ts_offset", f"{TS_BASE + float(self.plan.start(k)):.9f}",
                "-muxdelay", str(MPEGTS_DELAY / 2), "-mpegts_flags", "+resend_headers",
                "-f", "mpegts", "pipe:1"]
        return cmd

    def _encode(self, k, restored, timeout, ready=None):
        if restored is None and not self.src and ready is None:
            return None
        proc = None
        try:
            with self._lock:
                if self._closed:
                    return None
                proc = subprocess.Popen(
                    self._command(k, restored is not None, ready),
                    stdin=subprocess.PIPE if restored is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._processes.add(proc)
            data, _ = proc.communicate(input=restored, timeout=timeout)
            return data if proc.returncode == 0 and data else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
        finally:
            if proc is not None:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                # Drain pipes after killing: this also waits/reaps on timeout.
                proc.communicate()
                with self._lock:
                    self._processes.discard(proc)

    def close(self):
        with self._lock:
            self._closed = True
            self._cache.clear()
            self._bytes = 0
            for event in self._pending.values():
                event.set()
            for proc in self._processes:
                try:
                    proc.kill()
                except OSError:
                    pass
