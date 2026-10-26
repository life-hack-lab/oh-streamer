"""Engine interface (engine-decoupling plan 4.1 / 4.2). Pure typing + dataclasses: no torch, no lada imports.

A RestorationEngine loads models and opens a RestorationSession per source; the session yields restored frames.
Upper layers check `EngineInfo.capabilities` and degrade instead of assuming every engine looks like Lada.
Implemented by engines/lada/engine.py; the upper layers migrate to it in the rest of phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterator, Optional, Protocol

# Capability flags (EngineInfo.capabilities); what the upper layer does without them is in the plan, section 4.2.
FAST_CANCEL = "fast_cancel"                # stop() can abandon the clip restoration in progress
ORIGINAL_CAPTURE = "original_capture"      # frames can carry the pre-restoration ROIs (compare mode)
CLIP_METRICS = "clip_metrics"              # detection / clip / restoration stage metrics (buffer_progress)
LOOKAHEAD_LIMIT = "lookahead_limit"        # set_lookahead_limit() works (detect_gate)
OWN_READER = "own_reader"                  # the engine reads the input itself
BENCHMARK_HOOKS = "benchmark_hooks"        # perf check may replace detections for measurement

ALL_CAPABILITIES = frozenset({FAST_CANCEL, ORIGINAL_CAPTURE, CLIP_METRICS, LOOKAHEAD_LIMIT, OWN_READER,
                              BENCHMARK_HOOKS})


@dataclass(frozen=True)
class EngineInfo:
    name: str                              # "lada"
    version: str                           # engine version (+ commit)
    capabilities: frozenset = frozenset()


@dataclass(frozen=True)
class ModelChoice:
    detection: str
    restoration: str
    fp16: bool


@dataclass(frozen=True)
class ModelCatalog:
    detection: tuple = ()
    restoration: tuple = ()


@dataclass
class SourceSpec:
    """Our own description of one probed rendition (not the engine's metadata type)."""
    url: str
    http: Any                              # source_resolve.FFmpegHttpOptions
    width: int = 0
    height: int = 0
    fps_exact: Fraction = Fraction(0)
    time_base: Fraction = Fraction(0)
    start_pts: Optional[int] = None
    duration: float = 0.0
    frames_count: int = 0
    restoration_stride: int = 1
    codec_name: str = ""


@dataclass(slots=True)
class RestoredFrame:
    """One output frame. Built once per frame by the session (a slotted object: ~0.3 us, no copy of the image)."""
    image: Any                             # BGR uint8 HxWx3 (torch tensor, as Lada yields it); never copied by the engine
    pts: Optional[int]                     # in SourceSpec.time_base units
    original_patches: Any = None           # ORIGINAL_CAPTURE and capture_original=True only: [((t, l, b, r), patch), ...]


@dataclass
class PipelineMetrics:
    """Cheap snapshot for the info panel / progress estimates (a few reads per second, never per frame)."""
    frames_detected: int = 0               # CLIP_METRICS (needs an attached progress probe, else 0)
    clips_pending: int = 0                 # clips cut, restoration not started
    clips_restoring: int = 0               # restoration started, not finished
    output_queue: int = 0                  # restored frames waiting for the consumer
    output_capacity: int = 0               # 0 = unbounded / unknown
    queues: dict = field(default_factory=dict)   # engine stage name -> (qsize, maxsize); Lada: "feeder", "clips", "out"
    eof: bool = False                      # the source was read to its end (frames() will end after the queue drains)
    events: dict = field(default_factory=dict)


class EndOfStream:
    """Yielded once by RestorationSession.frames() when the session was stopped or woken (wake()); the iteration then
    ends. A source that simply runs out ends the iteration without yielding anything (for/else semantics)."""

    def __init__(self, reason: str = "stopped"):
        self.reason = reason


class EngineError(Exception):
    """Yielded by frames() (then the iteration ends) when the engine failed; replaces the engine's own error marker."""

    def __init__(self, message: str = "", cause: Optional[BaseException] = None):
        super().__init__(message)
        self.cause = cause


class RestorationSession(Protocol):
    """One source + one set of loaded models; reusable across start/stop (a seek is stop + start). One consumer thread
    iterates frames(); the others (stop/wake/metrics/...) may be called from any thread."""
    capabilities: frozenset                # the engine's EngineInfo.capabilities (the upper layers degrade on missing ones)
    max_clip_length: int                   # frames per clip this session was built with
    eof: bool                              # the source ended in the running segment

    def start(self, start_s: float, *, capture_original: bool = False) -> None: ...
    def frames(self) -> Iterator["RestoredFrame | EndOfStream | EngineError"]: ...
    def stop(self, *, abandon: bool = True) -> None: ...      # blocking; does NOT wake a blocked frames() consumer
    def wake(self) -> None: ...                               # wake a consumer blocked in frames() (it gets EndOfStream)
    def drain(self) -> None: ...                              # drop queued output (call after the consumer ended)
    def close(self) -> None: ...                              # stop if running and drop the engine objects
    def metrics(self) -> PipelineMetrics: ...
    def observe_detections(self, on_count: Any) -> bool: ...  # CLIP_METRICS: on_count(n_mosaics) per scanned frame; False: no
    def attach_progress_probe(self) -> Any: ...               # CLIP_METRICS: buffer_progress.PipelineProbe | None
    def attach_detect_gate(self) -> Any: ...                  # LOOKAHEAD_LIMIT: detect_gate.DetectGate | None
    def set_lookahead_limit(self, frames: Optional[int]) -> None: ...   # None/no support: no-op / open the gate


class RestorationEngine(Protocol):
    restoration_fps: str                   # "original" | "30" | ...: applied to sessions opened afterwards
    models: Optional[ModelChoice]          # what is loaded now (None: nothing)
    max_clip_length: int                   # applied to sessions opened afterwards (a loaded engine's value)

    def info(self) -> EngineInfo: ...
    def list_models(self) -> ModelCatalog: ...
    def load(self, device: Any, models: ModelChoice, max_clip_length: int) -> None: ...   # no-op if unchanged
    def release(self) -> None: ...
    def open(self, source: Any) -> RestorationSession: ...    # SourceSpec (or the player's Source object)
    def forget_sources(self, keep: Any = ()) -> None: ...     # drop the registrations of sources no session uses
    def open_reader(self, src: str) -> Any: ...               # the engine's own reader of a registered source (seek tail)
