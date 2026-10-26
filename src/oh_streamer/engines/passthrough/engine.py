"""The passthrough engine behind the engine interface (engines/base.py): decodes the source with engines/frame_source.py
and yields the frames untouched, in the format and with the pts semantics of the Lada engine (BGR uint8 HxWx3 torch
tensors, first pts = the stream's start, a seek lands on the target frame, restoration_fps strides the source).

It proves the upper layers do not depend on Lada, and gives the contract tests an engine that needs no GPU or weights.
Capabilities: FAST_CANCEL (stop is immediate) and OWN_READER only; no original capture (compare is off), no clip
metrics, no lookahead limit, no benchmark hooks.
"""

import queue
import threading
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

from oh_streamer.engines import base, frame_source
from oh_streamer.processing.frame_rate import planned_fps

VERSION = "1"
MODEL_NAME = "passthrough"
OUT_FRAMES = 24          # frames the decoder may run ahead of the consumer
CAPABILITIES = frozenset({base.FAST_CANCEL, base.OWN_READER})

_EOF, _STOP = object(), object()


def _source_from_spec(spec: base.SourceSpec):
    meta = frame_source.StreamMeta(
        video_file=spec.url, video_height=spec.height, video_width=spec.width, video_fps=float(spec.fps_exact),
        average_fps=float(spec.fps_exact), video_fps_exact=spec.fps_exact, codec_name=spec.codec_name,
        frames_count=spec.frames_count, duration=spec.duration, time_base=spec.time_base, start_pts=spec.start_pts)
    return SimpleNamespace(src=spec.url, http=spec.http, meta=meta, native_meta=meta)


class PassthroughEngine:
    def __init__(self):
        self._models = None
        self._device = None
        self.restoration_fps = "original"
        self.max_clip_length = 180

    def info(self) -> base.EngineInfo:
        return base.EngineInfo("passthrough", VERSION, CAPABILITIES)

    def list_models(self) -> base.ModelCatalog:
        return base.ModelCatalog((MODEL_NAME,), (MODEL_NAME,))

    @property
    def models(self):
        return self._models

    @property
    def device(self):
        return self._device

    def load(self, device, models: base.ModelChoice, max_clip_length: int) -> None:
        """Nothing to load; remembers the choice so callers see what they asked for."""
        self._device, self._models, self.max_clip_length = device, models, max_clip_length

    def release(self) -> None:
        self._models = self._device = None

    def open(self, source) -> "PassthroughSession":
        if self._models is None:
            raise RuntimeError("no models loaded (call load() first)")
        return PassthroughSession(_source_from_spec(source) if isinstance(source, base.SourceSpec) else source,
                                  self.restoration_fps, self.max_clip_length)

    def forget_sources(self, keep=()) -> None:
        frame_source.forget_sources(keep)

    @staticmethod
    def open_reader(src: str):
        """The reader for `src` as the engine itself reads it (registered source: HTTP options, stride)."""
        return frame_source.open_reader(src)


class PassthroughSession:
    capabilities = CAPABILITIES

    def __init__(self, source, restoration_fps: str, max_clip_length: int):
        self.source = source
        self.max_clip_length = max_clip_length
        # like RestorerFactory.make: source.meta becomes the restoration-rate adjusted metadata the callers read
        native = getattr(source, "native_meta", source.meta)
        fps = planned_fps(native.video_fps_exact, restoration_fps)
        stride = int(Fraction(native.video_fps_exact) / fps)
        source.meta = replace(native, video_fps_exact=fps, video_fps=float(fps),
                              average_fps=native.average_fps / stride,
                              frames_count=(native.frames_count + stride - 1) // stride)
        source.meta.restoration_stride = stride
        frame_source.register_source(source)
        self.eof = False
        self._q = queue.Queue(maxsize=OUT_FRAMES)
        self._thread = None
        self._stop_evt = threading.Event()
        self._closed = False

    @property
    def restoration_stride(self) -> int:
        return getattr(self.source.meta, "restoration_stride", 1)

    def start(self, start_s: float, *, capture_original: bool = False) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        self.stop()
        self.drain()
        self.eof = False
        frame_source.register_source(self.source)   # another session of the same URL may have been registered since
        self._stop_evt = stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(int(start_s * 1e9), stop), name="passthrough reader",
                                        daemon=True)
        self._thread.start()

    def _put(self, item, stop) -> bool:
        while not stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _run(self, start_ns: int, stop) -> None:
        try:
            with frame_source.open_reader(self.source.src) as reader:
                if start_ns > 0:
                    reader.seek(start_ns)
                for item in reader.frames():
                    if not self._put(item, stop):
                        return
            self.eof = True
            self._put(_EOF, stop)
        except BaseException as e:  # noqa: BLE001 - reported to the consumer
            self._put(base.EngineError(str(e), e), stop)

    def frames(self):
        """Source frames in order; ends silently at the end of the source, yields EndOfStream when stopped / woken and
        EngineError when reading failed (then ends). Blocks on the output queue."""
        while True:
            item = self._q.get()
            if type(item) is tuple:
                yield base.RestoredFrame(item[0], item[1])
            elif item is _EOF:
                return
            elif isinstance(item, base.EngineError):
                yield item
                return
            else:
                yield base.EndOfStream("stopped")
                return

    def stop(self, *, abandon: bool = True) -> None:
        t, self._thread = self._thread, None
        if t is None:
            return
        self._stop_evt.set()
        t.join(10)

    def wake(self) -> None:
        while True:
            try:
                self._q.put_nowait(_STOP)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.stop()
        self._closed = True

    def metrics(self) -> base.PipelineMetrics:
        return base.PipelineMetrics(output_queue=self._q.qsize(), output_capacity=self._q.maxsize, eof=self.eof)

    def observe_detections(self, on_count) -> bool:
        return False

    def attach_progress_probe(self):
        return None

    def attach_detect_gate(self):
        return None

    def set_lookahead_limit(self, frames) -> None:
        pass
