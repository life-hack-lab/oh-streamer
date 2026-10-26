"""The Lada engine behind the engine interface (engines/base.py): LadaEngine (models) and LadaSession (one source).

Wraps adapter.RestorerFactory / FrameRestorer without changing their behavior; the sequences the callers used to
spell out by hand (empty the output queue, reset eof, cancel event around stop(), stop marker to wake the consumer)
live here once. Importing this module imports torch and lada (LADA_MODEL_WEIGHTS_DIR is pointed at the checkout first).
"""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from oh_streamer.engines.lada import runtime_patches
from oh_streamer.paths import upstream_root

# Lada resolves model weights relative to the CWD unless this is set; must happen before importing lada.
os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", str(upstream_root() / "lada" / "model_weights"))
runtime_patches.apply_env()   # weights-only load + ultralytics offline: before lada / ultralytics are imported

from lada import VERSION as _LADA_VERSION, ModelFiles  # noqa: E402
from lada.utils import video_utils  # noqa: E402

from oh_streamer.engines import base  # noqa: E402
from oh_streamer.engines.lada import compat, io_hooks, probes  # noqa: E402
from oh_streamer.engines.lada.adapter import ErrorMarker, RestorerFactory, threading_utils  # noqa: E402
from oh_streamer.engines.lada.original_capture import install_original_capture  # noqa: E402

runtime_patches.install_nms_default()   # lada (imported above via adapter) calls ultralytics NMS without max_time_img

_commit_cache: list = []


def _lada_commit() -> str:
    """Short commit of the lada in use. Preferred: the "+<commit>" in lada.VERSION (a git checkout computes it, the
    release wheel has it baked in; no subprocess). Fallback: `git rev-parse` of the checkout ("" when it is not a git
    checkout / git is missing). Cached."""
    if not _commit_cache:
        commit = str(_LADA_VERSION).partition("+")[2].strip()
        if not commit:
            try:
                import lada
                root = Path(lada.__file__).resolve().parents[1]
                out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if out.returncode == 0:
                    commit = out.stdout.strip()
            except Exception:  # noqa: BLE001 - informational only
                pass
        _commit_cache.append(commit)
    return _commit_cache[0]


def _source_from_spec(spec: base.SourceSpec):
    """A SourceSpec as the Source-shaped object RestorerFactory.make wants (src, http, meta, native_meta)."""
    meta = video_utils.VideoMetadata(
        video_file=spec.url, video_height=spec.height, video_width=spec.width, video_fps=float(spec.fps_exact),
        average_fps=float(spec.fps_exact), video_fps_exact=spec.fps_exact, codec_name=spec.codec_name,
        frames_count=spec.frames_count, duration=spec.duration, time_base=spec.time_base, start_pts=spec.start_pts)
    return SimpleNamespace(src=spec.url, http=spec.http, meta=meta, native_meta=meta)


class LadaEngine:
    """Models + settings shared by every source; see base.RestorationEngine."""

    def __init__(self):
        self._factory: RestorerFactory | None = None
        self.restoration_fps = "original"
        self.max_clip_length = 180

    # -- identity / catalog --------------------------------------------------------------------------------------

    def info(self) -> base.EngineInfo:
        version = str(_LADA_VERSION)       # a git checkout already reports "<version>+<commit>"
        commit = "" if "+" in version else _lada_commit()
        return base.EngineInfo("lada", f"{version}+{commit}" if commit else version, base.ALL_CAPABILITIES)

    def list_models(self) -> base.ModelCatalog:
        return base.ModelCatalog(tuple(m.name for m in ModelFiles.get_detection_models()),
                                 tuple(m.name for m in ModelFiles.get_restoration_models()))

    # -- lifecycle -----------------------------------------------------------------------------------------------

    @property
    def models(self) -> base.ModelChoice | None:
        f = self._factory
        if f is None or f.det_model is None:
            return None
        return base.ModelChoice(f.det_name, f.res_name, bool(f.fp16))

    @property
    def device(self):
        return None if self._factory is None else self._factory.device

    def load(self, device, models: base.ModelChoice, max_clip_length: int) -> None:
        """Load the models (no-op if these are loaded on this device already; a clip length change alone only applies to
        sessions opened afterwards). A different model / FP16 / device frees the old weights first: sessions opened
        before must be closed by the caller. Raises RuntimeError for an unknown model name."""
        compat.check(self.info().version)   # one warning per process for an unverified lada; never blocks
        f = self._factory
        if f is not None and f.device == device and self.models == models:
            self.max_clip_length = f.max_clip_length = max_clip_length
            return
        if f is not None:
            f.release()
            self._factory = None
        f = RestorerFactory(device, max_clip_length)
        f.load(models.detection, models.restoration, models.fp16)   # raises before anything is kept
        self._factory = f
        self.max_clip_length = max_clip_length

    def release(self) -> None:
        """Drop the models (VRAM). Sessions opened before must be closed (or dropped) by the caller."""
        f, self._factory = self._factory, None
        if f is not None:
            f.release()

    def open(self, source) -> "LadaSession":
        """A session for `source`: a base.SourceSpec, or the player's Source (src, http, meta[, native_meta]); the
        latter's .meta is replaced by the restoration-rate adjusted one (see RestorerFactory.make)."""
        f = self._factory
        if f is None:
            raise RuntimeError("no models loaded (call load() first)")
        f.restoration_fps, f.max_clip_length = self.restoration_fps, self.max_clip_length
        return LadaSession(f, _source_from_spec(source) if isinstance(source, base.SourceSpec) else source)

    def forget_sources(self, keep=()) -> None:
        """Drop the I/O registrations of sources no session uses any more (after a source switch)."""
        io_hooks.forget_sources(keep)

    @staticmethod
    def open_reader(src: str):
        return open_reader(src)


def open_reader(src: str):
    """Lada's (patched, see io_hooks) VideoReader for `src`: frames exactly as the restoration reads them (bgr24 at the
    source size, HTTP options / headers of the registered source). A context manager with seek(ns) and frames()."""
    return video_utils.VideoReader(src)


class LadaSession:
    """One source + one FrameRestorer; see base.RestorationSession."""
    capabilities = base.ALL_CAPABILITIES

    def __init__(self, factory: RestorerFactory, source):
        self.source = source
        self.max_clip_length = factory.max_clip_length
        self._restorer = factory.make(source)
        self._capture = False
        self._running = False

    @property
    def eof(self) -> bool:
        return bool(getattr(self._restorer, "eof", False))

    @property
    def restoration_stride(self) -> int:
        return getattr(self.source.meta, "restoration_stride", 1)

    def start(self, start_s: float, *, capture_original: bool = False) -> None:
        r = self._restorer
        if r is None:
            raise RuntimeError("session is closed")
        if capture_original:
            install_original_capture()   # patches FrameRestorer once; frames carry _lada_orig while r.capture_original
        io_hooks.register_source(self.source)   # another session of the same URL may have been registered since
        threading_utils.empty_out_queue(r.get_frame_restoration_queue())   # start() asserts empty queues
        r.eof = False   # Lada never resets it; a restart after reaching the end would yield nothing
        r.capture_original = self._capture = bool(capture_original)
        r.start(start_ns=int(start_s * 1e9))
        self._running = True

    def frames(self):
        """Restored frames in order; ends silently at the end of the source, yields EndOfStream when stopped / woken and
        EngineError when a worker failed (then ends). Blocks on the output queue."""
        capture = self._capture
        for elem in self._restorer:
            if type(elem) is tuple:       # (frame tensor, pts): the hot path, no copy
                frame = elem[0]
                yield base.RestoredFrame(frame, elem[1], getattr(frame, "_lada_orig", None) if capture else None)
                continue
            if isinstance(elem, ErrorMarker):
                yield base.EngineError(str(elem), elem)
            else:                         # STOP_MARKER (or a leftover EOF marker while stopping)
                yield base.EndOfStream("stopped")
            return

    def stop(self, *, abandon: bool = True) -> None:
        r = self._restorer
        if r is None:
            return
        if abandon:
            probes.request_cancel(r)
        try:
            r.stop()
        finally:
            if abandon:
                probes.clear_cancel(r)
            self._running = False

    def wake(self) -> None:
        """FrameRestorer.stop() doesn't wake a consumer blocked on the output queue; this does (stop marker)."""
        r = self._restorer
        if r is not None:
            threading_utils.put_queue_stop_marker(r.get_frame_restoration_queue())

    def drain(self) -> None:
        r = self._restorer
        if r is not None:
            threading_utils.empty_out_queue(r.get_frame_restoration_queue())

    def close(self) -> None:
        if self._running:
            self.stop()
        self._restorer = None

    # -- observation ---------------------------------------------------------------------------------------------

    def metrics(self) -> base.PipelineMetrics:
        r = self._restorer
        m = base.PipelineMetrics()
        if r is None:
            return m
        out = r.get_frame_restoration_queue()
        m.output_queue, m.output_capacity = out.qsize(), out.maxsize
        try:
            m.queues = probes.pipeline_fill(r)
        except AttributeError:
            pass
        m.eof = bool(getattr(r, "eof", False))
        p = getattr(r, "_lada_progress_probe", None)
        if p is not None:
            with p.lock:
                m.frames_detected = p.det
                m.clips_pending = sum(1 for c in p.clips.values() if c[2] is None)
                m.clips_restoring = sum(1 for c in p.clips.values() if c[2] is not None and c[3] is None)
        return m

    def observe_detections(self, on_count) -> bool:
        """CLIP_METRICS: call on_count(n_mosaics) for every scanned frame as the pipeline consumes the detector's
        per-frame results (queue worker perf observations). False when this session cannot provide them."""
        q = None if self._restorer is None else probes.detection_queue(self._restorer)
        if q is None:
            return False
        try:
            original = q.get
        except AttributeError:
            return False

        def counting_get(*args, **kwargs):
            elem = original(*args, **kwargs)
            if isinstance(elem, tuple) and len(elem) == 2:
                on_count(elem[1])
            return elem
        try:
            q.get = counting_get
        except (AttributeError, TypeError):
            return False
        return True

    def attach_progress_probe(self):
        from oh_streamer.player import buffer_progress   # noqa: PLC0415 - the estimator stays with the player
        return None if self._restorer is None else buffer_progress.PipelineProbe.attach(self._restorer)

    def attach_detect_gate(self):
        from oh_streamer.processing import detect_gate   # noqa: PLC0415
        return None if self._restorer is None else detect_gate.DetectGate.attach(self._restorer)

    def set_lookahead_limit(self, frames) -> None:
        """int: close the detector gate for the segment about to start (limit raised to max_clip_length + margin);
        None: open it (the detector runs freely)."""
        gate = self.attach_detect_gate()
        if gate is None:
            return
        if frames is None:
            gate.open("limit removed")
        else:
            gate.reset(frames)
