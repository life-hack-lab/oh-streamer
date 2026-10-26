"""Frame source for casting the completed segments of an in-progress queue item.

This module deliberately consumes ``segments.json`` and the segment files directly.  It does not
start a restoration worker; missing segments are requested through the existing
``ProgressivePlayback.want`` path and become available when the queue worker finishes them.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

from oh_streamer.player import progressive_stream as progressive


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PROBE_TIMEOUT = 15.0
MANIFEST_INFO_WAIT = 2.0
DECODE_WAIT_TICK = 0.25
OPEN_ATTEMPTS = 20
OPEN_RETRY_WAIT = 0.1


def _as_fraction(value, default=None):
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return default


def _probe_document(po, timeout: float = PROBE_TIMEOUT) -> dict:
    """Use the shared full probe when available, otherwise inspect the original input with ffprobe."""
    raw = po.source_probe
    if isinstance(raw, dict) and isinstance(raw.get("streams"), list):
        return raw
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", po.path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(0.1, timeout),
        creationflags=NO_WINDOW,
        check=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    videos = [
        stream for stream in probe.get("streams", [])
        if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise ValueError("原始影片沒有可用的視訊軌道")
    return videos[0]


class ProgressiveCastSource:
    """HLS cast input backed by the already-written processed queue segments.

    ``playback`` is a ``ProgressivePlayback``.  Its manifest watcher and queue messaging are
    shared, while each HLS run gets an independent, cancellable iterator so the local player's
    TCP feeder remains separate.
    """

    is_progressive_cast = True

    def __init__(self, playback, encoder: str = "libx264", *, log=lambda message: None):
        self.playback = playback
        self.po = playback.po
        self.watcher = playback.watcher
        self.log = log
        self.encoder = encoder

        init_started = time.monotonic()
        self._load_play_info()
        probe = _probe_document(self.po, PROBE_TIMEOUT - (time.monotonic() - init_started))
        video = _video_stream(probe)
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("無法取得投放影片尺寸")

        info = self.po.info
        source_fps = _as_fraction(info.get("fps"), self.play.fps)
        if source_fps is None or source_fps <= 0:
            source_fps = self.play.fps
        source_frames = info.get("frames")
        try:
            source_frames = int(source_frames)
        except (TypeError, ValueError):
            source_frames = 0
        if source_frames <= 0:
            duration = max(0.0, float(info.get("duration") or self.po.duration or 0.0))
            source_frames = max(1, round(duration * float(source_fps)))

        state = self.watcher.state
        manifest_frames = state.total_frames
        if not manifest_frames:
            final = state.final_k()
            final_segment = state.segments.get(final) if final is not None else None
            if final_segment is not None:
                manifest_frames = final * self.play.n + final_segment.frames
        if manifest_frames:
            source_frames = int(manifest_frames)

        out_fps = self.play.fps * self.play.rate_mult
        if out_fps <= 0:
            raise ValueError("無法取得處理後影片的影格率")
        frames_count = max(1, source_frames * self.play.rate_mult)
        self.meta = SimpleNamespace(
            video_fps_exact=out_fps,
            frames_count=frames_count,
            duration=float(Fraction(frames_count, 1) / out_fps),
            video_width=width,
            video_height=height,
            start_pts=round(self.play.video_start * 1_000_000),
            time_base=Fraction(1, 1_000_000),
        )
        self.has_audio = bool(self.po.has_audio)
        fmt = probe.get("format") if isinstance(probe.get("format"), dict) else {}
        raw_fmt_start = fmt.get("start_time")
        try:
            self.fmt_start = float(raw_fmt_start) if raw_fmt_start not in (None, "") else float(
                info.get("video_start") or self.play.video_start
            )
        except (TypeError, ValueError):
            self.fmt_start = float(info.get("video_start") or self.play.video_start)
        self.target = self.play.video_start
        self.audio_source = self.po.source_url or self.po.path
        self.path = self.po.path

        self._active_lock = threading.RLock()
        self._active: set[_ProgressiveCastFrames] = set()
        self._error_lock = threading.Lock()
        self.error: Exception | None = None
        self._want_lock = threading.Lock()
        self._last_want: tuple[int, float] | None = None

    def _load_play_info(self):
        """Wait briefly for the worker's playback block, which exists before its first segment.

        The fallback probe has only the source frame rate.  Using it when an interpolation
        manifest is about to appear would create an HLS plan with the wrong output frame rate.
        """
        self.watcher.poll()
        end = time.monotonic() + MANIFEST_INFO_WAIT
        while self.watcher.state.version is None and time.monotonic() < end:
            state, _ = self.watcher.wait_newer(self.watcher.seq, min(0.1, end - time.monotonic()))
            if state.version is None:
                self.watcher.poll()
        state = self.watcher.state
        if state.version is None:
            raise RuntimeError("佇列尚未提供播放影格率，請等背景工作啟動後再投放")
        self.play = state.play

    @property
    def waiting_k(self) -> int | None:
        with self._active_lock:
            for frames in tuple(self._active):
                if frames.waiting_k is not None:
                    return frames.waiting_k
        return None

    def _set_error(self, error: Exception | None):
        with self._error_lock:
            self.error = error

    def tick(self):
        """Repeat a want while an HLS producer is waiting for an unfinished queue segment."""
        k = self.waiting_k
        if k is None:
            return
        now = self._now()
        with self._want_lock:
            last = self._last_want
            if last is not None and last[0] == k and now - last[1] < progressive.WANT_INTERVAL:
                return
            self._last_want = (k, now)
        try:
            self.playback.want(k)
        except Exception as exc:  # queue transport failures must reach the HLS run
            self._set_error(exc)
            raise
        if getattr(self.playback, "hold", None) == "failed":
            error = RuntimeError("queue item failed while waiting for a processed segment")
            self._set_error(error)

    def _now(self) -> float:
        return self.playback._now() if callable(getattr(self.playback, "_now", None)) else time.monotonic()

    def stop(self) -> float:
        """Stop the local feeder and return elapsed seconds, like Session.stop()."""
        started = time.perf_counter()
        self.playback.stop_feeder()
        return time.perf_counter() - started

    def run_input(self, start):
        from oh_streamer.platform.hls_cast import RunInput

        self._set_error(None)
        frames = _ProgressiveCastFrames(self, float(start))
        with self._active_lock:
            self._active.add(frames)
        audio = {"src": self.audio_source, "args": [], "fmt_start": self.fmt_start} if self.has_audio else None
        return RunInput(frames, self.meta.video_width, self.meta.video_height, audio)

    def _drop_frames(self, frames):
        with self._active_lock:
            self._active.discard(frames)


class _ProgressiveCastFrames:
    """Decode one processed track sequentially, waiting for missing segments as needed."""

    def __init__(self, source: ProgressiveCastSource, start: float):
        self.source = source
        self.watcher = source.watcher
        self.play = source.play
        self.start = max(source.target, float(start))
        self.closed = threading.Event()
        self.waiting_k: int | None = None
        self._iterated = False
        self._next_lock = threading.RLock()
        self._iterator = None
        self._last_ts: Fraction | None = None

    def __iter__(self):
        if not self._iterated:
            self._iterated = True
            self._iterator = self._iterate()
        return self

    def __next__(self):
        with self._next_lock:
            if self.closed.is_set():
                raise StopIteration
            if not self._iterated:
                self.__iter__()
            return next(self._iterator)

    def _iterate(self):
        import av

        try:
            start = _as_fraction(self.start, Fraction(0))
            k = self.play.segment_at(self.start)
            while not self.closed.is_set():
                got = self._await_segment(k)
                if got is None:
                    return
                segment, state = got
                container = self._open_segment(av, k, segment)
                if container is None:
                    continue
                try:
                    video_streams = list(container.streams.video)
                    stream = next((s for s in video_streams
                                   if (s.metadata or {}).get("title") == "oh:processed"), None)
                    # Older progressive manifests can contain a single processed video track
                    # without a title. Keep accepting that legacy shape, but never guess when
                    # a segment has multiple unlabelled/unknown video tracks.
                    if stream is None and len(video_streams) == 1:
                        title = (video_streams[0].metadata or {}).get("title")
                        if not title:
                            stream = video_streams[0]
                    if stream is None:
                        raise ValueError(f"segment {k} has no oh:processed video track")
                    base = _as_fraction(state.play.video_start, Fraction(0)) + Fraction(
                        k * state.play.n, 1
                    ) / state.play.fps
                    frame_index = 0
                    output_fps = state.play.fps * state.play.rate_mult
                    half_frame = Fraction(1, 2) / output_fps
                    emitted = False
                    for decoded in container.decode(stream):
                        if self.closed.is_set():
                            return
                        # Matroska stores timestamps at millisecond precision, which drifts at
                        # rates such as 60000/1001.  Queue segments have a fixed output rate and
                        # known source-frame count, so rebuild exact timestamps from frame order.
                        local = Fraction(frame_index, 1) / output_fps
                        frame_index += 1
                        timestamp = base + local
                        if timestamp < start:
                            if emitted or start - timestamp > half_frame:
                                continue
                            # HLS may request a boundary a fraction of a frame after the
                            # queue's rounded segment grid. Use the nearest frame and label it
                            # at the requested start so the rawvideo encoder stays aligned.
                            timestamp = start
                        if self._last_ts is not None and timestamp <= self._last_ts:
                            continue
                        frame = decoded.to_ndarray(format="bgr24")
                        self._last_ts = timestamp
                        emitted = True
                        yield frame, float(timestamp)
                    expected = segment.frames * state.play.rate_mult
                    if not self.closed.is_set() and frame_index != expected:
                        raise RuntimeError(
                            f"processed segment {k} has {frame_index} frames; expected {expected}"
                        )
                finally:
                    try:
                        container.close()
                    except Exception:  # noqa: BLE001
                        pass
                k += 1
        except Exception as exc:  # noqa: BLE001 - propagate decoder / queue errors to hls_cast
            if not self.closed.is_set():
                self.source._set_error(exc)
                raise
        finally:
            self.close()

    def _await_segment(self, k: int):
        seq = -1
        while not self.closed.is_set():
            state, seq = self.watcher.wait_newer(seq, DECODE_WAIT_TICK)
            final = state.final_k()
            if final is not None and k > final:
                self.waiting_k = None
                return None
            if state.play.count and k >= state.play.count:
                self.waiting_k = None
                return None
            segment = state.segments.get(k)
            if segment is not None:
                path = self.watcher.segment_path(segment)
                if os.path.isfile(path):
                    self.waiting_k = None
                    return segment, state
            self.waiting_k = k
            if self.source.error is not None:
                raise RuntimeError(f"progressive cast failed: {self.source.error}") from self.source.error
            if getattr(self.source.playback, "hold", None) == "failed":
                error = RuntimeError("queue item failed while waiting for a processed segment")
                self.source._set_error(error)
                raise error
            if self.source.po.queue_id is None:
                raise RuntimeError("cannot request a missing progressive segment without a queue item id")
            self.source.tick()
        self.waiting_k = None
        return None

    def _open_segment(self, av, k, segment):
        for attempt in range(OPEN_ATTEMPTS):
            if self.closed.is_set():
                return None
            path = self.watcher.segment_path(segment)
            try:
                container = av.open(path)
                return container
            except Exception as exc:  # noqa: BLE001 - a redo can swap a listed segment file
                if self.closed.is_set():
                    return None
                self.watcher.poll()
                fresh = self.watcher.state.segments.get(k)
                if fresh is not None and fresh.file != segment.file:
                    segment = fresh
                    continue
                if attempt == OPEN_ATTEMPTS - 1:
                    raise RuntimeError(f"cannot open processed segment {k}: {exc}") from exc
                if self.closed.wait(OPEN_RETRY_WAIT):
                    return None
        return None

    def close(self):
        # First signal the decoder/wait loop, then serialize generator.close() with __next__.
        # This avoids PyAV's unsafe cross-thread InputContainer.close(): the generator owns the
        # container and its finally block closes it on the same thread that was decoding it.
        self.closed.set()
        self.waiting_k = None
        self.watcher.wake()
        with self._next_lock:
            iterator = self._iterator
            if iterator is not None:
                try:
                    iterator.close()
                except (ValueError, RuntimeError):
                    # A reentrant close from _iterate's finally runs while the generator is
                    # executing. It will finish naturally as soon as this frame is returned.
                    pass
            self.source._drop_frames(self)


__all__ = ["ProgressiveCastSource"]
