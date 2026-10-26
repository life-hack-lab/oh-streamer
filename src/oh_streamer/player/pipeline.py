"""Restoration segment lifecycle and batch output, with explicit runtime dependencies.

The application facade supplies callbacks and Lada runtime objects after its early
CLI exits. This module neither imports the facade nor changes process globals.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from oh_streamer.engines import base as engine_base
from oh_streamer.engines.base import EndOfStream, EngineError
from oh_streamer.processing import compare_stack, detect_gate, rife_interp
from oh_streamer.sources.source_resolve import FFmpegHttpOptions


@dataclass(frozen=True)
class PipelineDependencies:
    """Operations shared with the facade, resolved through caller-owned callbacks."""

    log: Callable[..., Any]
    start_muxer: Callable[..., Any]
    stop_process: Callable[..., Any]
    free_port: Callable[..., Any]
    is_listening: Callable[..., Any]
    torch: Any
    open_reader: Callable[..., Any]
    stats_factory: Callable[..., Any] | None = None


class Stats:
    def __init__(self, fps: float, start_s: float, t0: float | None = None, first_what="model warmup + open + first clip",
                 *, deps: PipelineDependencies):
        self._deps = deps
        self.fps = fps
        self.start_s = start_s
        self.frames = 0
        self.t0 = t0 or time.perf_counter()
        self.t_first = None
        self.last_print = 0.0
        self.first_what = first_what
        self.extra = None        # () -> str appended to the periodic log line (interpolation state)

    def tick(self):
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
            self.last_print = now
            self._deps.log(f"first frame after {now - self.t0:.2f}s ({self.first_what})")
        self.frames += 1
        if now - self.last_print >= 2.0:
            self.last_print = now
            self._deps.log(self.summary(now) + (f" | {self.extra()}" if self.extra is not None else ""))

    def summary(self, now=None):
        now = now or time.perf_counter()
        el = max(1e-6, now - (self.t_first or self.t0))
        fps = self.frames / el
        pos = self.start_s + self.frames / self.fps
        return (f"pos {pos:7.1f}s | {self.frames} frames | {fps:5.1f} fps | "
                f"{fps / self.fps:4.2f}x realtime | VRAM {self._deps.torch.cuda.max_memory_allocated() / 2**30:.1f}GiB"
                if self._deps.torch.cuda.is_available() else f"pos {pos:7.1f}s | {fps:5.1f} fps")



class Session:
    """One playback segment: Lada from a position -> ffmpeg muxer listening on tcp://127.0.0.1:<port>.

    The engine session (`restorer`: engines/base.py RestorationSession, one FrameRestorer for Lada) and its models are
    shared by all segments; start/stop only restart its worker threads.
    With interpolation ("auto", rife_interp.py) the pump writes two frames per restored frame (a RIFE middle frame or
    a copy, then the frame) into a muxer running at twice the source frame rate, which drops the copies again (VFR);
    everything else keeps counting source frames (stats, frames_total, out_n for buffer_progress, --duration), and
    positions stay source time.
    """

    def __init__(self, restorer: Any, source: Any, encoder: str, max_frames: int | None,
                 *, deps: PipelineDependencies):
        self._deps = deps
        self.encoder, self.max_frames = encoder, max_frames
        self.lock = threading.Lock()  # guards stopping/muxer between the pump thread and stop()
        self.thread = None
        self.muxer = None
        self.stopping = False
        self.no_output = False   # the pump ended before a first frame reached the muxer (reader / decode error)
        self.target = 0.0
        self.url = self.port = self.first_s = None
        self.stats = None
        self.frames_total = 0    # frames sent to the muxer over all segments (info panel)
        self.probe = None        # buffer_progress.PipelineProbe of the restorer (buffering percentage)
        self.gate = None         # detect_gate.DetectGate of the restorer (detector held back until playable)
        self.gate_release_s = 2.0   # open it once the output is this many seconds past the target
        self.out_n = 0          # frames Lada output in this segment, including the skipped pre-target ones
        self.first_out = None    # source time of the first of them (Lada starts at the keyframe before the target)
        self.out_t = None        # source time of the last frame written to the muxer (pump thread)
        self.interp_mode = "off"  # the interpolation setting; with "auto", interp = the shared rife_interp.Interpolator
        self.interp = None
        self.clock = rife_interp.PlaybackClock()   # mpv's position, sampled by the controller: the lead signal
        self.doubler = None      # rife_interp.FrameDoubler of the running segment (2x output), None = 1x
        self.interp_why = ""     # why a segment runs at 1x although interpolation is "auto"
        self.compare_mode = "off"   # the compare setting: "on" = extra tracks oh:original (+ oh:restored with RIFE)
        self.layers = ("processed",)   # tracks of the running segment (compare_stack.layer_names); 1 = plain output
        self.cmp_why = ""        # why a segment runs without compare although the setting is "on"
        self.prev_arr = self.prev_orig = None   # last frame (restored / original) written: middle slot of the next pair
        self._pause_lock = threading.Lock()     # guards the pause state between the caller and the pause thread
        self._pause_state, self._pause_token = "running", 0
        self._pause_reserved_mib = None         # torch-reserved VRAM right after the pause's empty_cache (None = unknown)
        self.set_source(restorer, source)

    def set_source(self, restorer: Any, source: Any):
        """Switch to another rendition (resolution) of the same video, or to a restorer built with other settings;
        only valid while stopped."""
        assert self.thread is None, "set_source() on a running session"
        self.restorer, self.source = restorer, source
        self.src, self.meta, self.has_audio = source.src, source.meta, source.has_audio
        self.audio_src = source.audio_src
        self.fmt_start, self.http = source.fmt_start, source.http
        self.fps = float(source.meta.video_fps_exact)

    def release_restorer(self):
        """Drop the restorer (and with it the models it references) before a model reload; only while stopped."""
        assert self.thread is None, "release_restorer() on a running session"
        self.restorer = None

    def processing_done(self) -> bool:
        """The pump of the current segment has ended (end of video / --duration reached)."""
        return self.thread is not None and not self.thread.is_alive() and not self.no_output

    def start(self, pos_s: float, t_req: float | None = None, *, gate_release_s: float = 2.0):
        """Start the segment at pos_s. gate_release_s: the detector is held back (detect_gate.py) until the output is
        this many seconds past the target, i.e. the prebuffer mpv waits for (+ margin)."""
        self.target = pos_s
        self.gate_release_s = gate_release_s
        self._clear_pause()      # a new segment (seek) starts unpaused; gate.reset() below lifts any hold
        self.stopping = False
        self.no_output = False
        self.muxer = self.url = self.port = self.first_s = None
        self.out_n, self.first_out, self.out_t = 0, None, None
        what = "model warmup + open + first clip" if t_req is None else f"seek to {pos_s:.2f}s"
        self.stats = (self._deps.stats_factory(self.fps, pos_s, t_req, what)
                      if self._deps.stats_factory is not None else
                      Stats(self.fps, pos_s, t_req, what, deps=self._deps))
        self.clock.reset()                 # no lead (no interpolation) until the controller sees this segment play
        self.doubler, self.interp_why = None, ""
        if self.interp_mode == "auto":
            ok, self.interp_why = rife_interp.eligible(self.fps) if self.interp is not None else (False, "需要 CUDA GPU")
            if ok:
                self.doubler = rife_interp.FrameDoubler(self.interp, self.fps, self.meta.video_width,
                                                        self.meta.video_height, self.clock, self._deps.log, self._restore_margin)
                self.stats.extra = self.doubler.summary
        self.restorer.drain()   # start() asserts empty queues (the engine resets eof itself)
        self.layers, self.cmp_why = ("processed",), ""
        self.prev_arr = self.prev_orig = None
        if self.compare_mode == "on":
            names = compare_stack.layer_names(2 if self.doubler is not None else 1)
            ok, self.cmp_why = compare_stack.compare_ok(self.meta, names, self.encoder)
            if ok and engine_base.ORIGINAL_CAPTURE not in getattr(self.restorer, "capabilities", engine_base.ALL_CAPABILITIES):
                ok, self.cmp_why = False, "這個修復引擎無法取得原始畫面"
            if ok:
                try:
                    compare_stack.install_original_capture()
                    self.layers = names
                except Exception as e:  # noqa: BLE001 - compare is optional: never prevent playback
                    self.cmp_why = f"capture hook failed: {e!r}"
            if len(self.layers) == 1:
                self._deps.log(f"compare off for this segment: {self.cmp_why}")
        self.probe = self.restorer.attach_progress_probe()
        if self.probe is not None:
            self.probe.reset(pos_s)
        self.gate = self.restorer.attach_detect_gate()
        if self.gate is not None:
            self.gate.on_held = self.probe.note_detector_held if self.probe is not None else None
            self.gate.reset(self.restorer.max_clip_length + detect_gate.MARGIN)
        self.restorer.start(pos_s, capture_original=len(self.layers) > 1)   # the capture hook keeps the ROIs
        self.thread = threading.Thread(target=self._pump, name="lada pump", daemon=True)
        self.thread.start()

    # -- pause restoration (give the GPU to another job; the models and the engine session stay) -------------------

    pause_idle_s = 1.5      # held and no frame came out for this long: the restorer is idle (a clip still waiting for
                            # more detection frames cannot finish while held; that is accepted)
    pause_poll_s = 0.1

    def pause_state(self) -> str:
        """"running" | "pausing" (detector held, restoring what is in flight) | "paused" (idle, VRAM cache freed)."""
        with self._pause_lock:
            return self._pause_state

    def pause_reserved_mib(self) -> float | None:
        """VRAM (MiB) the paused restoration still holds (torch reserved after empty_cache); None unless "paused"
        and CUDA is readable. The queue learns the real paused footprint from it (live report restore_pause)."""
        with self._pause_lock:
            return self._pause_reserved_mib if self._pause_state == "paused" else None

    def pause_restore(self) -> str:
        """Hold the detector (detect_gate.DetectGate.hold), let the frames already detected come out, then free the
        cached VRAM (torch.cuda.empty_cache) on a background thread; returns the state at once ("pausing", or "paused"
        when the engine has no detect gate: nothing to hold). No-op if already pausing / paused. A new segment
        (start) and stop() clear it."""
        with self._pause_lock:
            if self._pause_state != "running":
                return self._pause_state
            self._pause_token += 1
            token = self._pause_token
            gate = self.gate if self.thread is not None else None
            if gate is None:
                self._pause_state = "paused"
                self._deps.log("restore pause requested: no detect gate, nothing to hold (state paused)")
                return "paused"
            self._pause_state = "pausing"
        self._deps.log("restore pause requested: detector held, waiting for the restorer to go idle")
        gate.hold("restore pause")
        threading.Thread(target=self._pause_wait, args=(token, gate, time.perf_counter()),
                         name="lada pause", daemon=True).start()
        return "pausing"

    def resume_restore(self) -> str:
        """Release the hold: the detector runs again, no restart. Returns "running"."""
        with self._pause_lock:
            was = self._pause_state
            self._pause_token += 1
            self._pause_state = "running"
            self._pause_reserved_mib = None
            gate = self.gate
        if gate is not None:
            gate.release()
        if was != "running":
            self._deps.log(f"restore resumed (was {was})")
        return "running"

    def _clear_pause(self):    # stop() / start(): the gate lifts its own hold (open("stopped") / reset())
        with self._pause_lock:
            self._pause_token += 1
            self._pause_state = "running"
            self._pause_reserved_mib = None

    def _vram_reserved(self) -> float | None:    # GiB, None without CUDA
        try:
            cuda = self._deps.torch.cuda
            return cuda.memory_reserved() / 2**30 if cuda.is_available() else None
        except Exception:  # noqa: BLE001 - telemetry only
            return None

    def _pause_wait(self, token: int, gate, t0: float):
        while True:
            time.sleep(self.pause_poll_s)
            with self._pause_lock:
                if token != self._pause_token:
                    return                      # resumed / stopped / new segment meanwhile
            if not gate.held:
                return
            if time.perf_counter() - max(gate.t_out, gate.hold_t) >= self.pause_idle_s:
                break
        before = self._vram_reserved()
        try:
            if before is not None:
                self._deps.torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"restore pause: empty_cache failed: {e!r}")
        after = self._vram_reserved()
        with self._pause_lock:
            if token != self._pause_token:
                return
            self._pause_state = "paused"
            self._pause_reserved_mib = None if after is None else round(after * 1024.0, 1)
        vram = f" | VRAM reserved {before:.2f} -> {after:.2f} GiB" if before is not None and after is not None else ""
        self._deps.log(f"restore paused after {time.perf_counter() - t0:.2f}s{vram}")

    def stop(self, keep_output: bool = False) -> float:
        """Stop the segment; returns how long that took (seconds). A clip restoration in progress is abandoned (see
        cancellable_restoration_model): the old output is never needed (seeks, switches, model reload, exit).

        keep_output: leave the muxer running for detach_output() (if the pump does not end within a moment, e.g. stuck
        in a write nobody reads, the muxer is killed after all)."""
        if self.thread is None:
            return 0.0
        t = time.perf_counter()
        self._clear_pause()
        if self.probe is not None:
            self.probe.freeze()
        if self.gate is not None:
            self.gate.open("stopped")  # a detector waiting in the gate wakes at once
        with self.lock:
            self.stopping = True
            muxer = self.muxer
        if not keep_output:
            self._deps.stop_process(muxer)  # also unblocks a write stuck on a slow consumer
        self.restorer.stop()    # abandons the clip restoration in progress
        self.restorer.wake()    # stop() doesn't wake a consumer blocked on the output queue
        if keep_output:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self._deps.log("segment output could not be kept (pump stuck in a write): stopping its muxer")
                self._deps.stop_process(muxer)
                with self.lock:
                    self.muxer = None
        self.thread.join()
        self.thread = None
        self.restorer.drain()
        took = time.perf_counter() - t
        gate = f" | {self.gate.summary()}" if self.gate is not None else ""
        interp = f" | {self.doubler.summary()}" if self.doubler is not None else ""
        self._deps.log(f"segment stopped in {took:.2f}s{' (output kept)' if keep_output and self.muxer is not None else ''} | "
            f"{self.stats.summary()}{gate}{interp}")
        return took

    def detach_output(self) -> tuple[float, dict | None]:
        """Stop the segment fast but hand its muxer over instead of killing it (seek_tail.SeekTail: mpv keeps playing
        the stream, the passthrough continues it). Returns (stop time, output) with output = {muxer, url, next_t
        (source time of the frame after the last one written), rate_mult}, or None when there is nothing to continue
        (no frame written yet, muxer gone); the muxer is then stopped like stop() does. A segment that had reached the
        end of the video is handed over too (stdin already closed: mpv plays it to its end, no passthrough)."""
        took = self.stop(keep_output=True)
        with self.lock:
            muxer, self.muxer = self.muxer, None
        if muxer is None:
            return took, None
        if self.out_t is None:
            self._deps.stop_process(muxer)
            return took, None
        return took, {"muxer": muxer, "url": self.url, "next_t": self.out_t + 1.0 / self.fps,
                      "rate_mult": 2 if self.doubler is not None else 1, "layers": len(self.layers)}

    def compare_snapshot(self) -> dict:
        """The compare state for the info panel (info_panel.compare_rows)."""
        return {"mode": self.compare_mode, "layers": list(self.layers), "why": self.cmp_why}

    def _restore_margin(self, now: float) -> float | None:
        """Pump thread (FrameDoubler): the restoration ahead of the frame being output (index out_n - 1)."""
        p, r = self.probe, self.restorer
        if p is None or not r:
            return None
        try:
            return p.restore_margin(self.out_n - 1, self.fps, now, r.max_clip_length)
        except Exception as e:  # noqa: BLE001 - an estimate only: it must never end the segment
            self._deps.log(f"restore margin failed: {e!r}")
            return None

    def interp_snapshot(self) -> dict:
        """The interpolation state for the info panel (info_panel.interp_rows)."""
        d = {"mode": self.interp_mode}
        if self.interp_mode != "auto":
            return d
        if self.doubler is None:
            d.update(eligible=False, why=self.interp_why or "—")
            return d
        d.update(eligible=True, out_fps=2 * self.fps, **self.doubler.policy.snapshot())
        return d

    def _open_muxer(self, first_s: float, skipped: int) -> bool:
        with self.lock:
            if self.stopping:
                return False
            port = self._deps.free_port()
            # Lada seeks to the keyframe before the target; audio and timestamps follow the first frame written
            audio_ss = max(0.0, first_s - self.fmt_start)
            self.muxer = self._deps.start_muxer(self.audio_src, self.meta, self.has_audio, audio_ss, self.http,
                                     f"tcp://127.0.0.1:{port}?listen=1", self.encoder, ts_offset=first_s,
                                     rate_mult=2 if self.doubler is not None else 1, layers=self.layers)
            self.port, self.first_s, self.url = port, first_s, f"tcp://127.0.0.1:{port}"
        self._deps.log(f"segment @ {first_s:.3f}s (requested {self.target:.3f}s, session-skipped {skipped} pre-target frames): "
            f"audio -ss {audio_ss:.3f}, -output_ts_offset {first_s:.3f}, port {port}"
            + (f", 2x output {2 * self.fps:.3f} fps (interpolation auto)" if self.doubler is not None else
               f", 1x output (interpolation: {self.interp_why})" if self.interp_mode == "auto" else "")
            + (f", compare tracks {'/'.join(self.layers)}" if len(self.layers) > 1 else ""))
        return True

    def _write_layers(self, processed, restored1x, original):
        """One pipe frame: the layers back to back (W x n*H) in compare_stack's stack order; plain output = the frame."""
        stdin = self.muxer.stdin
        if len(self.layers) == 1:
            stdin.write(memoryview(processed.reshape(-1)))
            return
        for buf in compare_stack.layers_for_write(processed, restored1x, original, self.layers):
            stdin.write(memoryview(buf.reshape(-1)))

    def _pump(self):
        half_frame = 0.5 / self.fps
        skipped = 0
        ended = False            # end of video / --duration reached (not a stop)
        try:
            for elem in self.restorer.frames():
                if self.stopping:
                    break
                if isinstance(elem, (EndOfStream, EngineError)):
                    self._deps.log(f"restorer stopped: {getattr(elem, 'reason', elem)}")
                    break
                frame, pts = elem.image, elem.pts
                self.out_n += 1
                if self.gate is not None and not self.gate.is_open and pts is not None \
                        and float(pts * self.meta.time_base) >= self.target + self.gate_release_s:
                    self.gate.open("target reached")   # playable by now: the detector may run ahead again
                if self.muxer is None:
                    t = float(pts * self.meta.time_base) if pts is not None else self.target
                    if self.first_out is None:
                        self.first_out = t
                        if self.probe is not None:
                            self.probe.note_first_output(t)
                    if t < self.target - half_frame:
                        skipped += 1  # frames between the keyframe and the target: land exactly on the target
                        continue
                    if not self._open_muxer(t, skipped):
                        break
                    if self.probe is not None:
                        self.probe.started(self.out_n)
                        self._deps.log(f"pipeline until the first frame: {self.probe.summary()}"
                            + (f" | {self.gate.summary()}" if self.gate is not None else ""))
                arr = frame.numpy() if isinstance(frame, self._deps.torch.Tensor) else frame
                orig = compare_stack.apply_patches(arr, elem.original_patches) if len(self.layers) > 1 else None
                if self.doubler is None:
                    self._write_layers(arr, arr, orig)
                else:   # middle frame (or copy of the previous one), then this frame
                    t = float(pts * self.meta.time_base) if pts is not None else None
                    outs = self.doubler.push(frame, arr, t)
                    for i, out in enumerate(outs):
                        # compare: the 1x layers repeat the previous frame in the middle slot (mpdecimate drops it)
                        real = i == len(outs) - 1 or self.prev_arr is None
                        self._write_layers(out, arr if real else self.prev_arr, orig if real else self.prev_orig)
                self.prev_arr, self.prev_orig = arr, orig
                self.out_t = float(pts * self.meta.time_base) if pts is not None else \
                    (self.out_t + 1.0 / self.fps if self.out_t is not None else self.first_s)
                self.frames_total += 1
                self.stats.tick()
                if self.max_frames and self.stats.frames >= self.max_frames:
                    ended = True
                    break
            else:
                ended = True
                self._deps.log("end of video")
            if ended and self.doubler is not None and self.muxer is not None and not self.stopping:
                for out in self.doubler.finish():   # a copy of the last frame: 2N output frames for N source frames
                    self._write_layers(out, self.prev_arr, self.prev_orig)
        except (OSError, ValueError):  # ValueError: stop() closed stdin under us
            if not self.stopping:
                self._deps.log("segment output closed (player disconnected?)")
        finally:
            if self.gate is not None:
                self.gate.open("pump ended")
            with self.lock:
                if not self.stopping and self.muxer is None:
                    self.no_output = True
                if not self.stopping and self.muxer is not None:
                    try:
                        self.muxer.stdin.close()  # EOF: ffmpeg finishes the segment, mpv reaches end of file
                    except OSError:
                        pass

    def ready(self) -> bool:
        return self.url is not None and self._deps.is_listening(self.muxer, self.port)

    def lada_out_n(self) -> int:
        """Frames Lada has output in this segment, taken by the pump or still waiting in its output queue (counted
        by the detect gate's hook), including the pre-target ones."""
        g = self.gate
        return max(self.out_n, g.out_n) if g is not None else self.out_n

    def lada_out_t(self) -> float | None:
        """Source time of the latest frame Lada has output (None before the first one reached the pump)."""
        return None if self.first_out is None else self.first_out + (self.lada_out_n() - 1) / self.fps

    def output_reached(self, ahead_s: float) -> bool:
        """Lada's output is ahead_s seconds of video past the target (the readiness for leaving a seek tail).

        Counted on Lada's side: until mpv connects, the muxer (blocked in its TCP accept after the first frame) takes
        only a few frames and the pump blocks writing; Lada keeps restoring into its output queue (~512 MB, e.g. 2.9 s
        of 1080p30) meanwhile, and mpv gets all of it at once when it connects. Also true when that queue is full (it
        cannot get further ahead before mpv connects: big frames, high frame rates) or the video has ended."""
        t = self.lada_out_t()
        if t is not None and t >= self.target + ahead_s:
            return True
        r = self.restorer
        if r is not None:
            m = r.metrics()
            if m.output_capacity and m.output_queue >= m.output_capacity - 1:
                return True
            if m.eof:
                return True
        return self.processing_done()



def passthrough_frames(source: Any, start_s: float, *, deps: PipelineDependencies):
    """Unrestored frames of `source` from start_s on: the frame source of a seek tail's passthrough (seek_tail.py).

    The engine's reader (Lada's patched VideoReader), so the frames are exactly what the engine itself reads: bgr24 at the source size (what the
    muxer expects), the frames before start_s decoded but skipped, HTTP options / headers of the registered source.
    Yields (ndarray, source time or None); closing the generator closes the reader."""
    tb = source.meta.time_base
    with deps.open_reader(source.src) as reader:
        reader.seek(int(start_s * 1e9))
        for frame, pts in reader.frames():
            arr = frame.numpy() if isinstance(frame, deps.torch.Tensor) else frame
            yield arr, (float(pts * tb) if pts is not None else None)



class RestorerStream:
    """The session's FrameRestorer from an absolute source time, as an hls_cast frame stream.

    Same pattern as Session.start/stop: the reader-level pre-target skipping makes the first frame the one at
    start_s (frame-accurate segment starts), close() abandons a clip restoration in progress (cancel event) and
    unblocks the iteration. Only used while the local Session is stopped (one producer)."""

    def __init__(self, restorer: Any, source: Any, start_s: float,
                 *, deps: PipelineDependencies):
        self._deps = deps
        if restorer is None:
            raise RuntimeError("no restorer (models are being reloaded)")
        self.restorer, self.time_base = restorer, source.meta.time_base
        self.closing = False
        self._lock = threading.Lock()
        self._closed = False
        restorer.drain()   # start() asserts empty queues (it also resets eof and registers the source)
        self.gate = restorer.attach_detect_gate()
        if self.gate is not None:
            self.gate.on_held = None
            # the detector runs freely again after ~4 s of output (the receiver's first segments are short)
            self.gate.reset(restorer.max_clip_length + detect_gate.MARGIN,
                            release_out_n=int(4 * float(source.meta.video_fps_exact)))
        restorer.start(start_s, capture_original=False)   # casting never carries compare tracks

    def __iter__(self):
        for elem in self.restorer.frames():
            if self.closing:
                return
            if isinstance(elem, (EndOfStream, EngineError)):
                raise RuntimeError(f"restorer stopped: {getattr(elem, 'reason', elem)}")
            frame, pts = elem.image, elem.pts
            arr = frame.numpy() if isinstance(frame, self._deps.torch.Tensor) else frame
            yield arr, (float(pts * self.time_base) if pts is not None else None)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.closing = True
        r = self.restorer
        if self.gate is not None:
            self.gate.open("closed")    # a detector waiting in the gate wakes at once
        r.stop()
        r.wake()    # wakes a blocked iteration



def run_batch(args, restorer: Any, src: str, meta: Any, has_audio: bool,
              fmt_start: float, http: FFmpegHttpOptions, encoder: str, max_frames: int | None,
              interp: "rife_interp.Interpolator | None" = None,
              *, deps: PipelineDependencies) -> int:
    """--output null / <file>: one linear pass, no seeking.

    interp (interpolation "auto"): 2x output like the player, deciding per frame pair against a simulated viewer
    (rife_interp.SimulatedPlayback: starts after --initial-prebuffer seconds, rebuffers --prebuffer seconds), so the
    file shows what the player would have shown and the log reports the stalls such a viewer would have had."""
    fps = float(meta.video_fps_exact)
    stats = (deps.stats_factory(fps, args.start) if deps.stats_factory is not None else
             Stats(fps, args.start, deps=deps))
    doubler = sim = None
    if interp is not None:
        usable, why = rife_interp.eligible(fps)
        if usable:
            sim = rife_interp.SimulatedPlayback(args.initial_prebuffer, args.prebuffer)
            probe = restorer.attach_progress_probe()
            if probe is not None:
                probe.reset(args.start)
            out_n = [0]
            margin = (lambda now: probe.restore_margin(out_n[0] - 1, fps, now, restorer.max_clip_length)) \
                if probe is not None else None
            doubler = rife_interp.FrameDoubler(interp, fps, meta.video_width, meta.video_height, sim, deps.log, margin)
            stats.extra = lambda: f"{doubler.summary()} | {sim.summary(time.perf_counter())}"
            deps.log(f"interpolation auto: 2x output ({2 * fps:.3f} fps), decided against a simulated player")
        else:
            deps.log(f"interpolation auto: 1x output ({why})")
    if sim is None and getattr(args, "simulate_player", False):
        sim = rife_interp.SimulatedPlayback(args.initial_prebuffer, args.prebuffer)
        stats.extra = lambda: sim.summary(time.perf_counter())
    restorer.start(args.start)
    muxer = None
    ok = True
    ended = False
    try:
        for elem in restorer.frames():
            if isinstance(elem, (EndOfStream, EngineError)):
                deps.log(f"restorer stopped: {getattr(elem, 'reason', elem)}")
                ok = False
                break
            frame, pts = elem.image, elem.pts
            if doubler is not None:
                out_n[0] += 1

            if muxer is None and args.output != "null":
                # Lada seeks to the keyframe before --start, so align audio to the first frame actually delivered.
                first_s = float(pts * meta.time_base) if pts is not None else args.start
                audio_ss = max(0.0, first_s - fmt_start)
                muxer = deps.start_muxer(src, meta, has_audio, audio_ss, http, args.output, encoder,
                                    rate_mult=2 if doubler is not None else 1)

            arr = frame.numpy() if isinstance(frame, deps.torch.Tensor) else frame
            t = float(pts * meta.time_base) if pts is not None else None
            outs = [arr] if doubler is None else doubler.push(frame, arr, t)
            if doubler is None and sim is not None and t is not None:
                sim.on_output(t, 1.0 / fps, time.perf_counter())
            if muxer is not None:
                try:
                    for out in outs:
                        muxer.stdin.write(memoryview(out.reshape(-1)))
                except (BrokenPipeError, OSError):
                    deps.log("output closed")
                    break

            stats.tick()
            if max_frames and stats.frames >= max_frames:
                ended = True
                break
        else:
            ended = True
        if ended and doubler is not None and muxer is not None:
            for out in doubler.finish():   # a copy of the last frame: 2N output frames for N source frames
                muxer.stdin.write(memoryview(out.reshape(-1)))
    except KeyboardInterrupt:
        deps.log("interrupted")
    except (BrokenPipeError, OSError):
        deps.log("output closed")
    finally:
        deps.log("final: " + stats.summary() + (f" | {stats.extra()}" if stats.extra is not None else ""))
        threading.Thread(target=lambda: restorer.stop(abandon=False), daemon=True).start()
        if muxer is not None:
            try:
                muxer.stdin.close()
            except OSError:
                pass
            muxer.wait()
    return 0 if ok else 1
