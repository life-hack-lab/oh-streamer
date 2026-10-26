"""
hls_cast: on-demand ("virtual VOD") HLS of the restored video for casting (Apple TV via AirPlay, or any HLS player on
the LAN), like Jellyfin / Plex transcoding.

The playlist is complete from the start (EXT-X-PLAYLIST-TYPE:VOD + EXT-X-ENDLIST, the full source duration), so the
receiver shows a full-length seekbar. Segments are produced only when needed:

  request seg k ── cached on disk ─────────────────────────────> served at once
               ├── being produced / just ahead of production ──> wait until it is done
               └── far from production (a seek) ──────────────> restart the producer at k's exact start, then wait

Segment layout (SegmentPlan): every segment is exactly N source frames (N = round(4 s * fps), so 120 frames = 4.004 s
at 29.97 fps, 120 = 4.0 s at 30, 240 at 60), only the last one is shorter. EXTINF values are exact, and segment k
starts at frame k*N, i.e. playlist time k*N/fps.

Producer (_Run): a frame stream from the caller (oh_player.RestorerStream: the FrameRestorer started at k's exact
source time; Lada's reader skips the pre-target frames, so the first frame IS frame k*N) -> one ffmpeg process per
run: H.264 High (NVENC or x264, no B-frames, IDR forced on every N-th frame) + AAC 48 kHz stereo (source audio from
-ss = k's start) -> segment muxer (MPEG-TS, cut at those IDRs, numbered from k). Timestamps: -initial_offset makes
every packet carry TS_BASE + playlist time (+ the mpegts muxer's constant 1.4 s), identical whichever run produced a
segment, so segments of different runs join seamlessly and seeking never drifts. (-output_ts_offset is not used: the
segment muxer's own cut times are absolute, it would then cut at every keyframe; B-frames would make audio split at
the video DTS, 100 ms before the boundary.) MPEG-TS, not fMP4: every segment is self-contained (no init segment that
would have to be byte-identical across encoder restarts). ffmpeg reports each finished segment on stdout
(-segment_list pipe:1); only then it enters the cache.

Heuristics (HlsVod):
  * near = production will reach it soon: run start <= k <= first unfinished segment + NEAR_SEGMENTS. Farther (or
    before the run start) and not cached = a seek: restart there. Restart requests are coalesced (JUMP_DEBOUNCE,
    latest wins); requests that the new run will not produce are answered 503 at once instead of hanging.
  * window: the producer pauses at a segment boundary once it is WINDOW_SECONDS ahead of the latest request (the
    restorer then blocks on its full queues: GPU idle) and continues when playback approaches. When it reaches
    segments that are already cached (>= 2 in a row) it ends; a new run starts at the next gap when needed.
  * end probe: AVPlayer (and other players) sometimes fetch one of the last segments while playing elsewhere. A far
    request for one of the last END_PROBE_SEGMENTS segments while playback is active (another request within
    PROBE_ACTIVE_WINDOW s) does not restart at once: it waits PROBE_GRACE s; if requests near the current
    production keep coming in meanwhile it was a probe and gets 503 (production is not disturbed); otherwise it is
    treated as a seek. A real seek to the very end therefore takes PROBE_GRACE s longer.
  * timeout: a segment that is not ready after REQUEST_TIMEOUT s gets 503 + Retry-After.
  * cache: segment files in a per-cast temp directory, LRU-evicted above the size cap (default 5 GB), never within
    [latest request - 2, latest request + window]; the directory is deleted when the cast ends.
  * invalidate() (quality / model change while casting): drop the cache and restart at the given segment. Simple:
    what the receiver has buffered already plays in the old quality.

Security: the server answers only token-protected video/subtitle playlists and numbered TS/WebVTT segments
(404 otherwise, no listings),
bound to one interface (the LAN address, or 127.0.0.1 for tests).

Pure Python + ffmpeg (no torch / lada imports): see test_hls_cast.py.
"""

import collections
import http.server
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from urllib.parse import urlsplit

from oh_streamer.processing import encoding_backend


SEGMENT_TARGET = 4.0         # seconds; the actual length is a whole number of frames
TS_BASE = 10.0               # added to every timestamp (keeps the AAC priming packet of segment 0 positive)
MPEGTS_DELAY = 1.4           # ffmpeg's mpegts muxer adds 2 * max_delay (0.7 s) to every timestamp
NEAR_SEGMENTS = 2
WINDOW_SECONDS = 300.0
REQUEST_TIMEOUT = 25.0
JUMP_DEBOUNCE = 0.15
PROBE_GRACE = 1.5
PROBE_ACTIVE_WINDOW = 10.0
END_PROBE_SEGMENTS = 2
SKIP_CACHED_MIN = 2          # a run ends when it reaches this many cached segments in a row
ERROR_BACKOFF = 5.0
DEFAULT_CACHE_BYTES = 5 * 2**30
MAX_RATE = "20M"             # video peak bitrate (Wi-Fi friendly); BANDWIDTH in the master playlist follows it
AUDIO_RATE = "192k"
SEG_RE = re.compile(r"seg_(\d{1,7})\.ts")


def _log(msg):
    print(f"[hls] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# segment plan + playlists
# ---------------------------------------------------------------------------------------------------------------

class SegmentPlan:
    """Fixed segment layout of one source: N frames per segment, the last one shorter."""

    def __init__(self, fps, frames_total: int, width: int, height: int, video_start: float = 0.0,
                 has_audio: bool = True, target: float = SEGMENT_TARGET):
        self.fps = Fraction(fps)
        if self.fps <= 0:
            raise ValueError(f"bad frame rate {fps!r}")
        self.frames_per_segment = max(1, round(target * self.fps))
        self.frames_total = max(1, int(frames_total))
        self.count = -(-self.frames_total // self.frames_per_segment)
        self.seg_duration = Fraction(self.frames_per_segment) / self.fps
        self.video_start = float(video_start)
        self.has_audio = has_audio
        # output size: the source size at cast start (even, for yuv420p); later renditions are scaled to it
        self.width, self.height = max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)
        big = self.width * self.height > 1920 * 1088 or self.fps > 60
        self.level, self.level_id = ("5.2", 52) if big else ("4.2", 42)

    def frames_in(self, k: int) -> int:
        return max(0, min(self.frames_per_segment, self.frames_total - k * self.frames_per_segment))

    def start(self, k: int) -> Fraction:
        return Fraction(k * self.frames_per_segment) / self.fps

    def duration(self, k: int) -> Fraction:
        return Fraction(self.frames_in(k)) / self.fps

    @property
    def total_duration(self) -> Fraction:
        return Fraction(self.frames_total) / self.fps

    def source_time(self, k: int) -> float:
        """Absolute source time of segment k's first frame (what the restorer is started at)."""
        return self.video_start + float(self.start(k))

    def index_at(self, t: float) -> int:
        """Segment containing playlist time t (clamped)."""
        k = math.floor(float(t) * float(self.fps) / self.frames_per_segment + 1e-9)
        return max(0, min(self.count - 1, k))

    def target_duration(self) -> int:
        durs = {self.duration(0), self.duration(self.count - 1)}
        return max(1, max(math.floor(d + Fraction(1, 2)) for d in durs))   # EXTINF rounded to the nearest integer

    @property
    def codecs(self) -> str:
        return f"avc1.6400{self.level_id:02x}" + (",mp4a.40.2" if self.has_audio else "")

    @property
    def bandwidth(self) -> int:
        peak = int(float(MAX_RATE.rstrip("M")) * 1_000_000)
        return peak + (int(AUDIO_RATE.rstrip("k")) * 1000 if self.has_audio else 0)

    def media_playlist(self) -> str:
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{self.target_duration()}",
                 "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS"]
        for k in range(self.count):
            lines.append(f"#EXTINF:{float(self.duration(k)):.6f},")
            lines.append(f"seg_{k}.ts")
        lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def master_playlist(self, media: str = "index.m3u8", *, subtitles=False, thumbnails=None) -> str:
        if subtitles is True:
            rendition = ('#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="字幕",DEFAULT=YES,'
                         'AUTOSELECT=YES,FORCED=NO,LANGUAGE="zh",URI="subtitles.m3u8"\n')
        else:
            rendition = ""
            for track in subtitles or ():
                # HLS quoted strings cannot contain quotes or line breaks. Track
                # titles are container metadata, never playlist syntax.
                name = str(track.title).replace('"', "'").replace("\r", " ").replace("\n", " ")
                lang = getattr(track, "language", None)
                lang_attr = f'LANGUAGE="{lang}",' if lang else ""
                rendition += (f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{name}",'
                              f'DEFAULT={"YES" if track.default else "NO"},AUTOSELECT=YES,FORCED=NO,{lang_attr}'
                              f'URI="{track.playlist_uri}"\n')
        group = ',SUBTITLES="subs"' if subtitles else ""
        preview = (f'#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH={thumbnails.bandwidth},'
                   f'RESOLUTION={thumbnails.width}x{thumbnails.height},CODECS="{thumbnails.codecs}",'
                   'URI="thumbnails.m3u8"\n') if thumbnails is not None else ""
        version = 4 if thumbnails is not None else 3
        return (f"#EXTM3U\n#EXT-X-VERSION:{version}\n#EXT-X-INDEPENDENT-SEGMENTS\n"
                + rendition + preview +
                f"#EXT-X-STREAM-INF:BANDWIDTH={self.bandwidth},RESOLUTION={self.width}x{self.height},"
                f"FRAME-RATE={float(self.fps):.3f},CODECS=\"{self.codecs}\"{group}\n{media}\n")


def parse_media_playlist(text: str) -> list[tuple[float, str]]:
    """[(duration, uri)] of a media playlist (tests / sanity checks)."""
    out, dur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",", 1)[0])
        elif line and not line.startswith("#"):
            out.append((dur, line))
            dur = None
    return out


# ---------------------------------------------------------------------------------------------------------------
# encoder command
# ---------------------------------------------------------------------------------------------------------------

def build_ffmpeg_cmd(plan: SegmentPlan, k0: int, in_w: int, in_h: int, encoder: str, out_dir: str,
                     audio: dict | None, *, encoded_video: bool = False) -> list[str]:
    """One producer run: raw BGR frames on stdin from segment k0 on -> MPEG-TS segments seg_<k>.ts in out_dir.

    audio: {"src", "args" (input options like HTTP headers), "ss" (seconds from the file start)} or None."""
    n = plan.frames_per_segment
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{in_w}x{in_h}", "-framerate", str(plan.fps),
           "-i", "pipe:0"]
    if encoded_video:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "mpegts", "-i", "pipe:0"]
    if audio:
        cmd += list(audio.get("args") or [])
        if audio.get("ss", 0) > 0:
            cmd += ["-ss", f"{audio['ss']:.6f}"]
        cmd += ["-vn", "-i", audio["src"], "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0"]
    if not encoded_video and (in_w, in_h) != (plan.width, plan.height):
        cmd += ["-vf", f"scale={plan.width}:{plan.height}:flags=bicubic"]
    # Casting is H.264 High regardless of the queue/export codec; retain the
    # previous software fallback for HEVC selections.
    if encoding_backend.codec_family(encoder) != "h264":
        encoder = "libx264"
    if encoded_video:
        cmd += ["-c:v", "copy"]
    else:
        cmd += encoding_backend.encoder_args(encoder, purpose="hls")
        cmd += ["-maxrate", MAX_RATE, "-bufsize", f"{2 * int(MAX_RATE.rstrip('M'))}M", "-bf", "0",
                "-profile:v", "high", "-level:v", plan.level, "-pix_fmt", "yuv420p"]
        cmd += encoding_backend.gop_args(encoder, n)
    if audio:
        cmd += ["-c:a", "aac", "-b:a", AUDIO_RATE, "-ar", "48000", "-ac", "2", "-shortest"]
    # cut a hair before each boundary: the IDR exactly on it always qualifies, nothing earlier does
    seg_time = (round(float(plan.seg_duration) * 1e6) - 50) / 1e6
    cmd += ["-avoid_negative_ts", "disabled", "-muxdelay", "0.7",
            "-f", "segment", "-segment_format", "mpegts", "-segment_time", f"{seg_time:.6f}",
            "-initial_offset", f"{TS_BASE + float(plan.start(k0)):.6f}", "-segment_start_number", str(k0),
            "-segment_list", "pipe:1", "-segment_list_type", "csv",
            os.path.join(out_dir, "seg_%d.ts")]
    return cmd


# ---------------------------------------------------------------------------------------------------------------
# disk cache
# ---------------------------------------------------------------------------------------------------------------

class SegmentCache:
    """Finished segment files: k -> (path, size, last use). Not thread-safe (HlsVod holds its lock)."""

    def __init__(self, directory: str, cap_bytes: int):
        self.dir, self.cap = directory, int(cap_bytes)
        self.items: dict[int, list] = {}
        self.bytes = 0
        self.evicted = 0

    def has(self, k: int) -> bool:
        return k in self.items

    def path(self, k: int) -> str | None:
        it = self.items.get(k)
        if it is None:
            return None
        it[2] = time.monotonic()
        return it[0]

    def add(self, k: int, path: str) -> bool:
        """Register a finished segment file (already moved into the cache directory by the caller)."""
        if k in self.items:
            _remove(path)
            return False
        size = os.path.getsize(path)
        self.items[k] = [path, size, time.monotonic()]
        self.bytes += size
        return True

    def drop(self, k: int) -> bool:
        it = self.items.get(k)
        if it is None:
            return True
        if not _remove(it[0]):
            return False          # being read right now (Windows): try again later
        del self.items[k]
        self.bytes -= it[1]
        return True

    def evict(self, protect: range) -> int:
        """Drop least recently used segments outside `protect` until under the cap; returns how many."""
        n = 0
        if self.bytes <= self.cap:
            return 0
        for k in sorted((k for k in self.items if k not in protect), key=lambda k: self.items[k][2]):
            if self.bytes <= self.cap:
                break
            if self.drop(k):
                n += 1
        self.evicted += n
        return n

    def clear(self):
        for k in list(self.items):
            self.drop(k)
        for k in list(self.items):         # could not delete (open): forget it anyway, the dir goes at the end
            self.bytes -= self.items.pop(k)[1]

    def ranges(self) -> list[tuple[int, int]]:
        """Contiguous cached segment ranges [(first, last)]."""
        out = []
        for k in sorted(self.items):
            if out and out[-1][1] == k - 1:
                out[-1] = (out[-1][0], k)
            else:
                out.append((k, k))
        return out


def _move(src: str, dst: str, tries: int = 60, delay: float = 0.05):
    """os.replace with retries: on Windows a just-closed file is often still open for a moment (ffmpeg finishing
    up, an antivirus / indexer scanning the new file) and the rename fails with a sharing violation (WinError 32).
    Falls back to copying (works when the other handle shares read access). Raises OSError if all fails."""
    err = None
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except FileNotFoundError:
            raise
        except OSError as e:
            err = e
            time.sleep(delay)
    try:
        shutil.copyfile(src, dst)
    except OSError:
        raise err
    _remove(src)        # (if this fails too the run directory is deleted when the run ends)


def _remove(path) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# producer run
# ---------------------------------------------------------------------------------------------------------------

class RunInput:
    """What a producer run needs from the caller: a frame stream started at the requested source time.

    frames: iterable of (HxWx3 uint8 BGR array, absolute source time or None) with .close() (idempotent, callable from
    another thread; must unblock a pending iteration). audio: {"src", "args", "fmt_start"} or None."""

    def __init__(self, frames, width: int, height: int, audio: dict | None):
        self.frames, self.width, self.height, self.audio = frames, width, height, audio



class FileFrames:
    """Decode a finished file without loading restoration models."""

    def __init__(self, source, start):
        import numpy as np
        self.np, self.source, self.start = np, source, start
        self.closed = False
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(max(0.0, start - source.fmt_start)),
             "-i", source.path, "-map", f"0:{source.stream_index}", "-an",
             "-vf", f"fps={source.meta.video_fps_exact}", "-pix_fmt", "bgr24",
             "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __iter__(self):
        m = self.source.meta
        size = m.video_width * m.video_height * 3
        i = 0
        while not self.closed:
            data = self.proc.stdout.read(size)
            if len(data) != size:
                if not self.closed and self.proc.wait() != 0:
                    raise OSError("無法解碼已處理的影片")
                break
            yield self.np.frombuffer(data, dtype=self.np.uint8).reshape(m.video_height, m.video_width, 3), self.start + i / float(m.video_fps_exact)
            i += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        _kill(self.proc)
        self.proc.stdout.close()


class FileCastSource:
    """The metadata and frame provider for an already processed local file."""

    def __init__(self, path):
        import json
        from types import SimpleNamespace
        self.path = path
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                 "-of", "json", path], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15, check=True)
        probe = json.loads(result.stdout)
        videos = [v for v in probe["streams"] if v["codec_type"] == "video" and not v.get("disposition", {}).get("attached_pic")]
        if not videos:
            raise ValueError("影片沒有可投放的視訊軌道")
        video = next((v for v in videos if v.get("tags", {}).get("title") == "oh:processed"), videos[0])
        try:
            fps = Fraction(video.get("avg_frame_rate") or "0")
        except (ValueError, ZeroDivisionError):
            fps = Fraction(0)
        if fps <= 0:
            fps = Fraction(video["r_frame_rate"])
        self.fmt_start = float(probe["format"].get("start_time") or 0)
        start = float(video.get("start_time") or self.fmt_start)
        duration = float(video.get("duration") or probe["format"]["duration"])
        self.meta = SimpleNamespace(video_fps_exact=fps, frames_count=int(video.get("nb_frames") or round(duration * fps)),
                                    duration=duration, video_width=int(video["width"]), video_height=int(video["height"]),
                                    start_pts=round(start * 1000000), time_base=Fraction(1, 1000000))
        self.stream_index = int(video["index"])
        self.has_audio = any(v["codec_type"] == "audio" for v in probe["streams"])
        self.encoder, self.target = "libx264", start

    def stop(self):
        return 0.0

    def run_input(self, start):
        audio = {"src": self.path, "args": [], "fmt_start": self.fmt_start} if self.has_audio else None
        return RunInput(FileFrames(self, start), self.meta.video_width, self.meta.video_height, audio)


class _Run:
    def __init__(self, vod: "HlsVod", k0: int, run_id: int, gen: int):
        self.vod, self.k0, self.id, self.gen = vod, k0, run_id, gen
        self.next_k = k0              # first segment of this run not finished yet
        self.reported_k = k0 - 1      # highest complete segment ffmpeg reported
        self.frames_written = 0
        self.stopping = False
        self.ended = False
        self.eof = False              # the source ended (not a gate / stop / error)
        self.error: str | None = None
        self.paused = False
        self.pause_logged = False
        self.proc: subprocess.Popen | None = None
        self.inp: RunInput | None = None
        self._close_lock = threading.Lock()
        self._closed = False
        self.reader = None
        self.dir = os.path.join(vod.dir, f"run{run_id}")
        self.t_start = time.perf_counter()
        self.t_first_frame = None
        self.thread = threading.Thread(target=self._pump, name=f"hls run @{k0}", daemon=True)

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        self.thread.start()

    def _close_input(self):
        with self._close_lock:
            if self._closed or self.inp is None:
                return
            self._closed = True
            inp = self.inp
        try:
            inp.frames.close()
        except Exception as e:  # noqa: BLE001
            self.vod.log(f"closing the frame stream failed: {e!r}")

    def stop(self):
        """Abandon the run (from another thread): kill ffmpeg, stop the frame stream, join, delete partial files."""
        self.stopping = True
        with self.vod.cond:
            self.vod.cond.notify_all()      # wakes a paused gate
        _kill(self.proc)
        self._close_input()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            self.vod.log(f"run @{self.k0} did not stop within 60 s")
        if self.reader is not None:
            self.reader.join(timeout=10)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _start_ffmpeg(self, inp: RunInput):
        plan = self.vod.plan
        audio = None
        if inp.audio and plan.has_audio:
            ss = max(0.0, plan.source_time(self.k0) - float(inp.audio.get("fmt_start") or 0.0))
            audio = {"src": inp.audio["src"], "args": inp.audio.get("args"), "ss": ss}
        cmd = build_ffmpeg_cmd(plan, self.k0, inp.width, inp.height, self.vod.encoder, self.dir, audio,
                               encoded_video=bool(self.vod.vsr_exe))
        if self.vod.vsr_exe:
            from oh_streamer.platform.cast_vsr import Pipeline
            self.proc = Pipeline(self.vod.vsr_exe, plan, inp.width, inp.height, cmd, stderr=self.vod.stderr)
        else:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.vod.stderr,
                                         bufsize=0)
        self.vod.child_pids.update(getattr(self.proc, "pids", {self.proc.pid}))
        self.reader = threading.Thread(target=self._read_list, args=(self.proc,), name="hls list", daemon=True)
        self.reader.start()

    def _read_list(self, proc):
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                name = os.path.basename(line.split(",", 1)[0])
                m = SEG_RE.fullmatch(name)
                if m:
                    self.vod._segment_done(self, int(m.group(1)), os.path.join(self.dir, name))
        except (OSError, ValueError):
            pass

    def _pump(self):
        vod, plan = self.vod, self.vod.plan
        n_seg = plan.frames_per_segment
        target = plan.source_time(self.k0)
        half = 0.5 / float(plan.fps)
        skipped = 0
        try:
            inp = vod.provider(target)
            with self._close_lock:
                self.inp = inp
                closed_early = self.stopping
            if closed_early:
                return
            n = 0
            for arr, t in inp.frames:
                if self.stopping:
                    break
                if self.proc is None:
                    if t is not None and t < target - half:
                        skipped += 1
                        continue
                    if t is not None and abs(t - target) > half:
                        vod.log(f"run @{self.k0}: first frame at {t:.3f}s, wanted {target:.3f}s")
                    if t is not None:      # the map the WebVTT X-TIMESTAMP-MAP must agree with (sample video A)
                        pts = TS_BASE + MPEGTS_DELAY + float(plan.start(self.k0))
                        vod.log(f"run @{self.k0}: first source frame {t:.3f}s -> playlist {float(plan.start(self.k0)):.3f}s, "
                                f"video pts {pts:.3f}s (skipped {skipped} early frames)")
                    self.t_first_frame = time.perf_counter()
                    self._start_ffmpeg(inp)
                if self.k0 * n_seg + n >= plan.frames_total:
                    self.eof = True          # the playlist ends here (the source may have a few frames more)
                    break
                if n % n_seg == 0 and not vod._gate(self, self.k0 + n // n_seg):
                    break
                remaining = memoryview(arr.reshape(-1))   # reshape copies only if not contiguous
                while remaining:
                    written = self.proc.stdin.write(remaining)
                    if not written:
                        raise OSError("encoder pipe closed before the complete frame was written")
                    remaining = remaining[written:]
                n += 1
                self.frames_written = n
            else:
                self.eof = not self.stopping
        except (OSError, ValueError) as e:      # ValueError: stdin closed under us by stop()
            if not self.stopping:
                self.error = f"encoder: {e!r}"
        except Exception as e:  # noqa: BLE001 - reported to the requests waiting for this run
            if not self.stopping:
                self.error = str(e).splitlines()[0][:200] if str(e) else repr(e)
        finally:
            self._close_input()
            if self.proc is not None:
                if not self.stopping:
                    # EOF lets ffmpeg finish and report what it has; a trailing partial segment (after an error)
                    # is rejected by _segment_done's frame count check
                    try:
                        self.proc.stdin.close()
                    except OSError:
                        pass
                    try:
                        self.proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        if not self.stopping:
                            self.error = "encoder did not finish within 60 seconds"
                    if self.reader is not None:
                        self.reader.join(timeout=10)
                    if self.proc.returncode not in (0, None) and not self.stopping and not self.error:
                        self.error = f"ffmpeg exited with {self.proc.returncode}"
                _kill(self.proc)
                if self.reader is not None:
                    self.reader.join(timeout=10)
                vod.child_pids.difference_update(getattr(self.proc, "pids", {self.proc.pid}))
            self.ended = True
            vod._run_ended(self, skipped)


def ts_first_pts(path: str, limit: int = 1 << 20) -> tuple[float | None, float | None]:
    """(video, audio) PTS in seconds of the first PES packet carrying one in an MPEG-TS file, None when not found.

    Used only to log whether a produced segment really starts at TS_BASE + MPEGTS_DELAY + start(k), which is what the
    WebVTT X-TIMESTAMP-MAP assumes. Reads at most `limit` bytes."""
    video = audio = None
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except OSError:
        return None, None
    for off in range(0, len(data) - 187, 188):
        pkt = data[off:off + 188]
        if pkt[0] != 0x47 or not pkt[1] & 0x40:             # sync byte, payload_unit_start
            continue
        control = (pkt[3] >> 4) & 3
        if not control & 1:
            continue
        i = 4 + (1 + pkt[4] if control & 2 else 0)
        pes = pkt[i:i + 14]
        if len(pes) < 14 or pes[:3] != b"\x00\x00\x01" or not pes[7] & 0x80:
            continue
        pts = (((pes[9] >> 1) & 7) << 30 | pes[10] << 22 | (pes[11] >> 1) << 15 | pes[12] << 7 | pes[13] >> 1) / 90000
        if 0xE0 <= pes[3] <= 0xEF and video is None:
            video = pts
        elif 0xC0 <= pes[3] <= 0xDF and audio is None:
            audio = pts
        if video is not None and audio is not None:
            break
    return video, audio


def _kill(p: subprocess.Popen | None):
    if p is None:
        return
    if p.poll() is None:
        try:
            p.kill()
        except OSError:
            pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for f in (p.stdin, p.stdout):
        try:
            if f:
                f.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------
# the on-demand VOD
# ---------------------------------------------------------------------------------------------------------------

class HlsVod:
    """Segments on demand for one source (see the module docstring for the heuristics).

    provider(source_time_s) -> RunInput; encoder: H.264 backend (settable, used by the next run)."""

    def __init__(self, plan: SegmentPlan, provider, encoder: str, cache_bytes: int = DEFAULT_CACHE_BYTES,
                 start_k: int = 0, cache_root: str | None = None, log=_log, stderr=None,
                 window_seconds: float = WINDOW_SECONDS, request_timeout: float = REQUEST_TIMEOUT,
                 vsr_exe: str | None = None):
        self.plan, self.provider, self.encoder, self.log, self.stderr = plan, provider, encoder, log, stderr
        self.vsr_exe = vsr_exe
        self.window = max(2, math.ceil(window_seconds / float(plan.seg_duration)))
        self.request_timeout = request_timeout
        root = cache_root or os.path.join(tempfile.gettempdir(), "oh_hls")
        os.makedirs(root, exist_ok=True)
        _clean_stale(root)
        self.dir = tempfile.mkdtemp(prefix=f"p{os.getpid()}_", dir=root)
        self.cache = SegmentCache(self.dir, cache_bytes)
        self.cond = threading.Condition()
        self.run: _Run | None = None
        self.run_ids = 0
        self.gen = 0
        self.anchor = max(0, min(plan.count - 1, start_k))   # latest (non-probe) request, or the cast start
        self.jump: tuple[int, float] | None = None           # (k, time) restart wanted (debounced)
        self.pending_k: int | None = None                    # the run being (re)started at
        self.eof_k: int | None = None                        # segments >= this do not exist (source ended)
        self.suspended = False
        self.stop_req = False
        self.closed = False
        self.idle = threading.Event()
        self.last_error: tuple[int, float, str] | None = None  # (run start, time, message)
        self.history = collections.deque(maxlen=64)          # (time, k) of segment requests
        self.inflight: list[int] = []                        # segments requests are waiting for
        self.child_pids: set[int] = set()
        # stats (info panel / tests)
        self.requests = 0
        self.restarts = 0
        self.last_request: dict | None = None
        self.last_jump_latency: float | None = None
        self.jump_t0: dict[int, float] = {}
        self.runs_started = 0
        self._evict_logged = -1e9
        self.manager = threading.Thread(target=self._manage, name="hls manager", daemon=True)
        self.manager.start()

    # -- public API (any thread) --------------------------------------------------------------------------------

    def prefetch(self, k: int):
        """Move production to an explicit seek target without waiting for its bytes."""
        with self.cond:
            if self.closed or not 0 <= k < self.plan.count:
                return
            self._set_anchor(k)
            if not self.is_cached(k) and not self._run_covers(k):
                self._want_jump(k, time.perf_counter())
            self.cond.notify_all()

    def get_segment(self, k: int) -> tuple[bytes | None, str]:
        """Segment k's bytes, or (None, reason) for an error response. Blocks up to request_timeout."""
        t0 = time.perf_counter()
        plan = self.plan
        with self.cond:
            self.requests += 1
            if self.closed:
                return None, "closed"
            if not 0 <= k < plan.count or (self.eof_k is not None and k >= self.eof_k):
                return None, "missing"
            data = self._read(k)
            if data is not None:
                self._set_anchor(k)
                self.history.append((t0, k))
                self._note(k, t0, "快取", data)
                self.cond.notify_all()          # the window moved: a paused run may continue
                return data, "cached"
            self.inflight.append(k)
            try:
                return self._wait_segment(k, t0)
            finally:
                self.inflight.remove(k)

    def _wait_segment(self, k: int, t0: float) -> tuple[bytes | None, str]:
        """(holds self.cond) Wait for a segment that is not cached."""
        probe = False
        if not self._will_produce(k):
            if self._is_end_probe(k, t0):
                probe = True
                self.log(f"segment {k}: possible end probe, waiting {PROBE_GRACE:g}s before restarting")
            else:
                self._want_jump(k, t0)
        if not probe:
            self._set_anchor(k)
        self.history.append((t0, k))
        self.cond.notify_all()
        deadline = t0 + self.request_timeout
        how = "等待"
        while True:
            data = self._read(k)
            if data is not None:
                break
            now = time.perf_counter()
            if self.closed:
                return None, "closed"
            if self.eof_k is not None and k >= self.eof_k:
                return None, "missing"
            if probe:
                if any(t > t0 and abs(kk - k) > NEAR_SEGMENTS + 1 for t, kk in self.history):
                    self._note(k, t0, "探測")
                    return None, "probe"
                if now - t0 >= PROBE_GRACE:
                    probe = False
                    self._set_anchor(k)
                    if not self._will_produce(k):
                        self._want_jump(k, now)
                    self.cond.notify_all()
            elif not self._will_produce(k) and not self.suspended:
                self._note(k, t0, "取代")
                return None, "superseded"
            err = self.last_error
            if err is not None and err[1] > t0 and err[0] <= k:
                self._note(k, t0, "錯誤")
                return None, "error"
            if now >= deadline:
                self._note(k, t0, "逾時")
                return None, "timeout"
            if k in self.jump_t0:
                how = "跳轉"
            self.cond.wait(min(0.25, deadline - now))
        jt = self.jump_t0.pop(k, None)
        if jt is not None:
            self.last_jump_latency = time.perf_counter() - jt
        self._note(k, t0, how, data)
        return data, how

    def suspend(self, timeout: float = 60.0) -> bool:
        """Stop producing (the frame stream's restorer is free afterwards) until resume()."""
        with self.cond:
            self.suspended = True
            self.idle.clear()
            self.cond.notify_all()
        return self.idle.wait(timeout)

    def resume(self, invalidate: bool = False, at_k: int | None = None):
        with self.cond:
            if invalidate:
                self.gen += 1
                self.cache.clear()
                self.eof_k = None
                self.jump = None
            if at_k is not None:
                self.anchor = max(0, min(self.plan.count - 1, at_k))
            self.last_error = None
            self.suspended = False
            self.cond.notify_all()

    def invalidate(self, at_k: int | None = None):
        """Drop every cached segment and restart production at at_k (default: the latest request)."""
        with self.cond:
            self.gen += 1
            self.stop_req = True
            self.cache.clear()
            self.eof_k = None
            self.jump = None
            if at_k is not None:
                self.anchor = max(0, min(self.plan.count - 1, at_k))
            self.cond.notify_all()

    def restart_point(self) -> int:
        """Where production should continue after a quality / setting change: the segment after the latest request
        (the receiver has that one already), unless it is still waiting for it."""
        with self.cond:
            if self.inflight:
                return min(self.inflight)
            if self.last_request is None:
                return self.anchor
            k = self.last_request["k"]
            return k if self.last_request.get("how") in ("逾時", "錯誤", "取代") else min(self.plan.count - 1, k + 1)

    def is_cached(self, k: int) -> bool:
        with self.cond:
            return self.cache.has(k) or (self.eof_k is not None and k >= self.eof_k)

    def peek_segment(self, k: int) -> tuple[int, bytes | None]:
        """Snapshot a finished segment for previews without changing playback or cache priority."""
        with self.cond:
            item = self.cache.items.get(k) if not self.closed else None
            if item is not None:
                try:
                    with open(item[0], "rb") as source:
                        return self.gen, source.read()
                except OSError:
                    pass
            return self.gen, None

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.manager.join(timeout=90)
        run = self.run
        if run is not None:
            run.stop()
        with self.cond:
            self.cache.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    def status(self) -> dict:
        with self.cond:
            run = self.run
            plan = self.plan
            ranges = self.cache.ranges()
            prod = None
            if run is not None and not run.ended:
                prod = float(plan.start(run.k0)) + run.frames_written / float(plan.fps)
            return {
                "producing": run is not None and not run.ended and not run.paused and not self.suspended,
                "paused": bool(run is not None and run.paused),
                "suspended": self.suspended,
                "prod_pos": prod,
                "run_start": float(plan.start(run.k0)) if run is not None else None,
                "frames_written": run.frames_written if run is not None else 0,
                "processing_done": bool(run is not None and run.ended and run.eof),
                "cached_segments": len(self.cache.items),
                "cached_bytes": self.cache.bytes,
                "cached_ranges": [(float(plan.start(a)), float(plan.start(b) + plan.duration(b))) for a, b in ranges],
                "last_request": dict(self.last_request) if self.last_request else None,
                "last_jump_latency": self.last_jump_latency,
                "requests": self.requests,
                "restarts": self.restarts,
                "evicted": self.cache.evicted,
                "error": self.last_error[2] if self.last_error else None,
                "duration": float(plan.total_duration),
                "segments": plan.count,
            }

    # -- internals (hold self.cond unless noted) ----------------------------------------------------------------

    def _read(self, k: int) -> bytes | None:
        p = self.cache.path(k)
        if p is None:
            return None
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            self.cache.drop(k)
            return None

    def _note(self, k, t0, how, data=None):
        self.last_request = {"k": k, "t": time.perf_counter(), "how": how,
                             "latency": time.perf_counter() - t0, "bytes": len(data) if data else 0,
                             "pos": float(self.plan.start(k))}

    def _set_anchor(self, k: int):
        self.anchor = k

    def _run_covers(self, k: int) -> bool:
        run = self.run
        if self.pending_k is not None and self.pending_k <= k <= self.pending_k + NEAR_SEGMENTS:
            return True
        # (a segment the run has passed already but that is not cached any more was evicted: not covered)
        return (run is not None and not run.ended and not run.stopping and run.gen == self.gen
                and run.next_k <= k <= run.next_k + NEAR_SEGMENTS)

    def _will_produce(self, k: int) -> bool:
        if self._run_covers(k):
            return True
        if self.jump is not None and self.jump[0] <= k <= self.jump[0] + NEAR_SEGMENTS:
            return True
        # idle (no run, nothing pending): the manager starts one at the first gap after the anchor
        if (self.run is None or self.run.ended) and self.jump is None and not self.suspended:
            first = self._first_uncached(self.anchor, self.anchor + self.window)
            if first is not None and first <= k <= first + NEAR_SEGMENTS:
                return True
        return False

    def _want_jump(self, k: int, t: float):
        if self.jump is not None and self.jump[0] != k:
            self.log(f"segment {k} requested: coalescing with the pending restart at {self.jump[0]}")
        self.jump = (k, t)
        self.jump_t0.setdefault(k, t)
        if len(self.jump_t0) > 16:
            self.jump_t0.pop(next(iter(self.jump_t0)))

    def _is_end_probe(self, k: int, now: float) -> bool:
        if k < self.plan.count - END_PROBE_SEGMENTS:
            return False
        return any(now - t <= PROBE_ACTIVE_WINDOW and abs(kk - k) > NEAR_SEGMENTS + 1 for t, kk in self.history)

    def _first_uncached(self, lo: int, hi: int) -> int | None:
        hi = min(hi, self.plan.count - 1, (self.eof_k - 1) if self.eof_k is not None else hi)
        for k in range(max(0, lo), hi + 1):
            if not self.cache.has(k):
                return k
        return None

    def _protect(self) -> range:
        return range(max(0, self.anchor - 2), self.anchor + self.window + 3)

    def _gate(self, run: _Run, j: int) -> bool:
        """Pump thread, at the start of segment j: continue (True) or end the run (False)."""
        with self.cond:
            try:
                while True:
                    if run.stopping or self.closed or run.gen != self.gen or self.suspended or self.stop_req:
                        return False
                    if all(self.cache.has(x) for x in range(j, min(self.plan.count, j + SKIP_CACHED_MIN))):
                        self.log(f"run @{run.k0}: segments from {j} are cached, ending the run")
                        return False
                    # (+1: segment j-1 is only cut once the first frame of j is written)
                    if j <= self.anchor + self.window + 1:
                        return True
                    if not run.paused and not run.pause_logged:
                        run.pause_logged = True     # (it pauses again at every segment while playback continues)
                        self.log(f"run @{run.k0}: {self.window} segments ahead of the playhead ({self.anchor}), "
                                 f"pausing at {j} until playback approaches")
                    run.paused = True
                    self.cond.wait(0.5)
            finally:
                run.paused = False

    def _segment_done(self, run: _Run, k: int, path: str):
        """List reader thread: ffmpeg finished seg_<k>.ts of `run`."""
        with self.cond:
            complete = run.eof or (k - run.k0 + 1) * self.plan.frames_per_segment <= run.frames_written
            if run.gen != self.gen or self.closed or not 0 <= k < self.plan.count or not complete:
                if not complete and not run.stopping:
                    self.log(f"run @{run.k0}: dropping incomplete segment {k}")
                _remove(path)
                return
            run.reported_k = max(run.reported_k, k)   # it exists in the source (even if caching it fails below)
            if self.cache.has(k):
                _remove(path)
                run.next_k = max(run.next_k, k + 1)
                self.cond.notify_all()
                return
        measured = None
        if k == run.k0:
            try:
                measured = ts_first_pts(path)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
        # move into the cache directory outside the lock (the rename may need retries on Windows)
        dst = os.path.join(self.dir, f"seg_{k}.r{run.id}.ts")
        try:
            _move(path, dst)
        except OSError as e:
            self.log(f"could not cache segment {k}: {e}")
            with self.cond:
                self.cond.notify_all()
            return
        with self.cond:
            if run.gen != self.gen or self.closed:
                _remove(dst)
                return
            try:
                self.cache.add(k, dst)
            except OSError as e:
                self.log(f"could not cache segment {k}: {e}")
                _remove(dst)
                return
            run.next_k = max(run.next_k, k + 1)
            if run.t_first_frame is not None and k == run.k0:
                self.log(f"run @{run.k0}: first segment ready {time.perf_counter() - run.t_start:.2f}s after start")
            if measured is not None and measured[0] is not None:
                want = TS_BASE + MPEGTS_DELAY + float(self.plan.start(k))
                audio = "-" if measured[1] is None else f"{measured[1]:.3f}s"
                self.log(f"run @{run.k0}: segment {k} first PTS video {measured[0]:.3f}s (expected {want:.3f}s, "
                         f"{(measured[0] - want) * 1000:+.0f} ms; X-TIMESTAMP-MAP assumes 0) audio {audio}")
            n = self.cache.evict(self._protect())
            if n and time.perf_counter() - self._evict_logged > 30:     # (once the cap is reached: every segment)
                self._evict_logged = time.perf_counter()
                self.log(f"cache at its {self.cache.cap / 2**30:.2f} GiB cap: evicting least recently used segments "
                         f"({self.cache.evicted} so far, {self.cache.bytes / 2**20:.0f} MiB kept)")
            self.cond.notify_all()

    def _run_ended(self, run: _Run, skipped: int):
        with self.cond:
            # Feeding the last frame is not EOF when a seek interrupts the encoder
            # while it is draining. In particular VSR has several pipeline stages.
            if run.eof and not run.stopping and not run.error and run.gen == self.gen:
                # segments ffmpeg reported exist, whether or not they made it into the cache (a failed move must
                # not turn an existing segment into a 404: it is simply produced again when requested)
                end = max(run.next_k, run.reported_k + 1)
                self.eof_k = end if self.eof_k is None else max(self.eof_k, end)
                if self.eof_k < self.plan.count:
                    self.log(f"source ended at segment {self.eof_k} (playlist has {self.plan.count})")
            if run.error and not run.stopping:
                self.last_error = (run.k0, time.perf_counter(), run.error)
                self.log(f"run @{run.k0} failed: {run.error}")
            self.cond.notify_all()

    def _next_action(self, now: float):
        """(action, k) for the manager, or (None, seconds to wait)."""
        if self.closed:
            return "exit", None
        if self.suspended or self.stop_req:
            if self.run is not None:
                return "stop", None
            self.stop_req = False
            if self.suspended:
                self.idle.set()
                return None, 0.5
        if self.jump is not None:
            k, t = self.jump
            if now - t < JUMP_DEBOUNCE:
                return None, JUMP_DEBOUNCE - (now - t)
            self.jump = None
            if not self._run_covers(k) and not self.cache.has(k):
                return "restart", k
        if self.run is None or self.run.ended:
            err = self.last_error
            if err is not None and now - err[1] < ERROR_BACKOFF:
                return None, ERROR_BACKOFF - (now - err[1])
            k = self._first_uncached(self.anchor, self.anchor + self.window)
            if k is not None:
                return "start", k
        return None, 0.5

    def _manage(self):
        while True:
            with self.cond:
                while True:
                    act, arg = self._next_action(time.perf_counter())
                    if act is not None:
                        break
                    self.cond.wait(max(0.01, arg))
                old = self.run
                if act in ("stop", "restart", "exit") and old is not None:
                    old.stopping = True
                if act in ("start", "restart"):
                    self.pending_k = arg            # requests for it keep waiting while the old run stops
            if old is not None and act in ("stop", "restart", "exit", "start"):
                t = time.perf_counter()
                if not old.ended or act != "start":
                    old.stop()
                else:
                    old.thread.join(timeout=10)
                    shutil.rmtree(old.dir, ignore_errors=True)
                if act != "start":
                    self.log(f"run @{old.k0} stopped in {time.perf_counter() - t:.2f}s")
            with self.cond:
                if self.run is old:
                    self.run = None
                if act == "exit":
                    self.pending_k = None
                    return
                if act in ("start", "restart") and not self.closed and not self.suspended and not self.stop_req:
                    self.run_ids += 1
                    run = _Run(self, arg, self.run_ids, self.gen)
                    self.run = run
                    self.runs_started += 1
                    if act == "restart":
                        self.restarts += 1
                    self.log(f"{'restart' if act == 'restart' else 'start'} production at segment {arg} "
                             f"({float(self.plan.start(arg)):.3f}s)")
                    run.start()
                self.pending_k = None
                self.cond.notify_all()


def _clean_stale(root: str):
    """Remove cache directories of player processes that no longer run (p<pid>_*)."""
    try:
        import psutil
    except ImportError:
        return
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        m = re.match(r"p(\d+)_", name)
        if m and int(m.group(1)) != os.getpid() and not psutil.pid_exists(int(m.group(1))):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------------------------------------------

M3U8_TYPE = "application/vnd.apple.mpegurl"
TS_TYPE = "video/mp2t"


class HlsServer:
    """Serves one HlsVod under http://<host>:<port>/<token>/ (everything else: 404)."""

    def __init__(self, vod: HlsVod, host: str, port: int = 0, token: str | None = None, log=_log,
                 *, subtitles=None, thumbnails=None):
        self.vod, self.log = vod, log
        self.subtitles = subtitles
        self.subtitle_tracks = (() if subtitles is None else
                                getattr(subtitles, "renditions", (subtitles,)))
        self.thumbnails = thumbnails
        self.token = token or secrets.token_urlsafe(18)
        self.media = vod.plan.media_playlist().encode()
        self.master = vod.plan.master_playlist(subtitles=self.subtitle_tracks, thumbnails=thumbnails).encode()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "oh-hls"
            sys_version = ""

            def log_message(self, fmt, *args):   # noqa: N802 - our own log lines instead
                pass

            def do_GET(self):   # noqa: N802
                server._handle(self, True)

            def do_HEAD(self):   # noqa: N802
                server._handle(self, False)

        class Httpd(http.server.ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = False

            def handle_error(self, request, client_address):
                exc = sys.exc_info()[1]
                if not isinstance(exc, (ConnectionError, TimeoutError)):
                    server.log(f"http error from {client_address[0]}: {exc!r}")

        self.httpd = Httpd((host, port), Handler)
        self.host, self.port = self.httpd.server_address[:2]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                       name="hls http", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def url(self, name: str = "master.m3u8") -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/{self.token}/{name}"

    def update_subtitles(self, subtitles):
        tracks = () if subtitles is None else getattr(subtitles, "renditions", (subtitles,))
        self.subtitles = subtitles
        self.subtitle_tracks = tracks
        self.master = self.vod.plan.master_playlist(subtitles=tracks, thumbnails=self.thumbnails).encode()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thumbnails is not None:
            self.thumbnails.close()
        self.thread.join(timeout=5)

    def _handle(self, h: http.server.BaseHTTPRequestHandler, body: bool):
        parts = urlsplit(h.path).path.split("/")
        if len(parts) != 3 or not secrets.compare_digest(parts[1].encode(), self.token.encode()):
            return self._send(h, 404, b"not found\n", "text/plain", body)
        name = parts[2]
        if name == "index.m3u8":
            return self._send(h, 200, self.media, M3U8_TYPE, body, cache="no-cache")
        if name == "master.m3u8":
            return self._send(h, 200, self.master, M3U8_TYPE, body, cache="no-cache")
        if self.thumbnails is not None:
            if name == "thumbnails.m3u8":
                return self._send(h, 200, self.thumbnails.playlist().encode(), M3U8_TYPE, body,
                                  cache="no-store")
            thumb = re.fullmatch(r"thumb_(\d{1,7})\.ts", name)
            if thumb and 0 <= int(thumb[1]) < self.vod.plan.count:
                try:
                    data = self.thumbnails.get_segment(int(thumb[1]))
                except Exception as error:  # optional previews cannot interrupt video delivery
                    self.log(f"preview unavailable ({type(error).__name__})")
                    data = None
                if data is None:
                    return self._send(h, 503, b"preview unavailable\n", "text/plain", body,
                                      cache="no-store", extra={"Retry-After": "2"})
                return self._send_ts(h, data, body, cache="no-store")
        if self.subtitle_tracks:
            playlist = re.fullmatch(r"subtitles(?:_\d{1,3})?\.m3u8", name)
            sub = re.fullmatch(r"(sub_(?:\d{1,3}_)?)(\d{1,7})\.vtt", name)
            for track in self.subtitle_tracks:
                if playlist and name == track.playlist_uri:
                    return self._send(h, 200, track.playlist().encode(), M3U8_TYPE, body, cache="no-cache")
                if sub and sub[1] == track.segment_prefix and 0 <= int(sub[2]) < self.vod.plan.count:
                    return self._send(h, 200, track.segment(int(sub[2])), "text/vtt; charset=utf-8", body, cache="no-cache")
        m = SEG_RE.fullmatch(name)
        if not m:
            return self._send(h, 404, b"not found\n", "text/plain", body)
        k = int(m.group(1))
        t0 = time.perf_counter()
        data, how = self.vod.get_segment(k)
        took = time.perf_counter() - t0
        if data is None:
            code = 404 if how == "missing" else 503
            self.log(f"seg {k} -> {code} ({how}) after {took:.2f}s")
            return self._send(h, code, f"{how}\n".encode(), "text/plain", body,
                              extra={"Retry-After": "2"} if code == 503 else None)
        if how != "cached" or took > 0.5:
            self.log(f"seg {k} -> 200 ({how}) {took:.2f}s {len(data) / 1e6:.1f} MB")
        return self._send_ts(h, data, body)

    def _send_ts(self, h, data: bytes, body: bool, cache: str | None = None):
        """Shared byte-range handling for playback and independent preview TS resources."""
        rng = h.headers.get("Range")
        if rng:
            m2 = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", rng)
            size = len(data)
            if m2 and (m2.group(1) or m2.group(2)):
                if m2.group(1):
                    a = int(m2.group(1))
                    b = min(size - 1, int(m2.group(2))) if m2.group(2) else size - 1
                else:
                    a, b = max(0, size - int(m2.group(2))), size - 1
                if a <= b < size:
                    return self._send(h, 206, data[a:b + 1], TS_TYPE, body, cache=cache,
                                      extra={"Content-Range": f"bytes {a}-{b}/{size}"})
            return self._send(h, 416, b"", TS_TYPE, body, cache=cache,
                              extra={"Content-Range": f"bytes */{len(data)}"})
        return self._send(h, 200, data, TS_TYPE, body, cache=cache)

    @staticmethod
    def _send(h, code, data: bytes, ctype: str, body: bool, cache: str | None = None, extra: dict | None = None):
        try:
            h.send_response(code)
            h.send_header("Content-Type", ctype)
            h.send_header("Content-Length", str(len(data)))
            if code in (200, 206) and ctype == TS_TYPE:
                h.send_header("Accept-Ranges", "bytes")
            if cache:
                h.send_header("Cache-Control", cache)
            for k, v in (extra or {}).items():
                h.send_header(k, v)
            h.end_headers()
            if body and data:
                h.wfile.write(data)
        except (ConnectionError, TimeoutError, OSError):
            h.close_connection = True


# ---------------------------------------------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------------------------------------------

def lan_ip() -> str:
    """IPv4 address of the interface that has the default route (UDP connect: no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))          # TEST-NET-1: routed via the default route, never answered
        ip = s.getsockname()[0]
    except OSError:
        ip = None
    finally:
        s.close()
    if not ip or ip.startswith(("127.", "0.")):
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = None
    if not ip or ip.startswith(("127.", "0.")):
        raise OSError("no LAN address found (not connected to a network?)")
    return ip
