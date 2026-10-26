"""Engine-neutral frame source: PyAV decoding with our HTTP options, the pre-target skip after a seek and the
restoration-rate stride (engine-decoupling plan 4.3). Used by engines that do not restore anything (passthrough).

The decoding rules mirror engines/lada/io_hooks.py (HttpVideoReader), which stays as it is so the Lada engine's behavior
cannot change: frames come out as (torch uint8 HxWx3 BGR tensor, pts); after seek(ns) the frames before the target are
decoded (they are references) but not converted or yielded; with stride > 1 only every stride-th source frame is kept,
selected from the source timestamps; a corrupted frame repeats the last good one (at most 10 in a row).
Imports torch and av: only engine modules import this one.
"""

import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import av
import torch

from oh_streamer.processing.frame_rate import selected_frame
from oh_streamer.sources.source_resolve import FFmpegHttpOptions, is_url


def log(msg):
    print(f"[oh_player] {msg}", file=sys.stderr, flush=True)


@dataclass
class StreamMeta:
    """The per-source facts the reader needs (same field names as the metadata type the player's Source carries)."""
    video_file: str
    video_height: int
    video_width: int
    video_fps: float
    average_fps: float
    video_fps_exact: Fraction
    codec_name: str
    frames_count: int
    duration: float
    time_base: Fraction
    start_pts: Optional[int]
    restoration_stride: int = 1


_SOURCES: dict[str, tuple[Any, Any]] = {}      # src -> (metadata, http options)


def register_source(source) -> None:
    """Make open_reader(source.src) use this source's metadata / HTTP options (latest registration wins)."""
    _SOURCES[source.src] = (source.meta, source.http)


def forget_sources(keep=()) -> None:
    keep = {s.src for s in keep}
    for k in list(_SOURCES):
        if k not in keep:
            del _SOURCES[k]


class FrameReader:
    """Context manager with seek(ns) and frames(), like Lada's VideoReader: frames are (tensor, pts)."""

    def __init__(self, src: str, meta=None, http: Optional[FFmpegHttpOptions] = None):
        self.file, self.meta, self.http = src, meta, http
        self.container = None
        self.target_s: Optional[float] = None

    def __enter__(self):
        opts = self.http.av_options() if self.http is not None and is_url(self.file) else {}
        self.container = av.open(self.file, metadata_errors="ignore", options=opts)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def seek(self, offset_ns):
        self.target_s = offset_ns / 1e9
        self.container.seek(int((offset_ns / 1_000_000_000) * av.time_base))

    def frames(self):
        meta = self.meta
        stride = getattr(meta, "restoration_stride", 1) if meta is not None else 1
        av.logging.restore_default_callback()
        av.logging.set_libav_level(av.logging.ERROR)
        vstream = self.container.streams.video[0]
        vstream.thread_type = "AUTO"
        time_base = vstream.time_base
        fps = float(meta.video_fps_exact) if meta is not None else float(vstream.average_rate or 30)
        threshold = (self.target_s or 0.0) - 0.5 / fps
        skipping = self.target_s is not None or stride > 1
        decoded_index = 0
        last_good_frame, consecutive_errors, max_consecutive_errors = None, 0, 10
        for packet in self.container.demux(vstream):
            try:
                for frame in packet.decode():
                    if skipping:
                        if frame.pts is not None and float(frame.pts * time_base) < threshold:
                            continue
                        skipping = False
                    if stride > 1:
                        # anchored to source timestamps so a seek and a start from 0 select the same frames
                        index = (round((frame.pts - int(meta.start_pts or 0)) * time_base * meta.video_fps_exact * stride)
                                 if frame.pts is not None else decoded_index)
                        decoded_index += 1
                        if not selected_frame(index, stride):
                            continue
                    torch_frame = torch.from_numpy(frame.to_ndarray(format="bgr24"))
                    last_good_frame = (torch_frame, frame.pts)
                    consecutive_errors = 0
                    yield torch_frame, frame.pts
            except av.error.InvalidDataError as e:
                if last_good_frame is not None and consecutive_errors < max_consecutive_errors:
                    consecutive_errors += 1
                    log(f"corrupted frame, duplicating last good frame ({consecutive_errors}/{max_consecutive_errors})")
                    yield last_good_frame[0], last_good_frame[1]
                elif last_good_frame is None and skipping:
                    continue  # corrupt packet before the first delivered frame: nothing to duplicate or lose
                else:
                    raise Exception(f"Cannot handle corrupted frame: "
                                    f"{'first frame corrupt' if last_good_frame is None else 'too many consecutive corrupted frames'}") from e


def open_reader(src: str) -> FrameReader:
    """A FrameReader for `src` with the registered source's metadata (stride, frame rate) and HTTP options."""
    meta, http = _SOURCES.get(src, (None, None))
    return FrameReader(src, meta, http)
