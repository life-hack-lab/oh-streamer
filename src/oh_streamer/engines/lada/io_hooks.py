# SPDX-License-Identifier: AGPL-3.0-only
# Replaces functions of lada.utils.video_utils at run time (upstream lada, AGPL-3.0). Modified 2026-10-26; see THIRD_PARTY_NOTICES.md, section
# "Modifications to upstream software".
"""Lada I/O hooks: our metadata + HTTP options + pre-target frame skipping for Lada's readers.

Moved verbatim from player/oh_player.py (engine-decoupling phase 2); behavior unchanged. The patched attributes live
on lada.utils.video_utils; Lada's source stays untouched.
"""

import sys
from fractions import Fraction

import av
import torch
from lada.utils import video_utils

from oh_streamer.processing.frame_rate import selected_frame
from oh_streamer.sources.source_resolve import FFmpegHttpOptions, is_url


VideoMetadata = video_utils.VideoMetadata   # the metadata type Lada's readers and the muxer share


def log(msg):
    # oh_player rebinds this to its own (patchable) log, so messages keep going through oh_player.log
    print(f"[oh_player] {msg}", file=sys.stderr, flush=True)


def _fraction(s: str) -> Fraction:
    num, _, den = s.partition("/")
    return Fraction(int(num), int(den or 1)) if int(den or 1) != 0 else Fraction(int(num), 1)


def build_metadata(path: str, probe: dict) -> video_utils.VideoMetadata:
    vs = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    fmt = probe["format"]
    fps_exact = _fraction(vs["r_frame_rate"])
    avg = _fraction(vs.get("avg_frame_rate", "0/0")) if vs.get("avg_frame_rate", "0/0") != "0/0" else fps_exact
    duration = float(vs.get("duration") or fmt.get("duration") or 0)
    frames = int(vs.get("nb_frames") or 0) or int(duration * float(fps_exact))  # streams rarely report nb_frames
    return video_utils.VideoMetadata(
        video_file=path,
        video_height=int(vs["height"]),
        video_width=int(vs["width"]),
        video_fps=float(fps_exact),
        average_fps=float(avg),
        video_fps_exact=fps_exact,
        codec_name=vs["codec_name"],
        frames_count=frames,
        duration=duration,
        time_base=_fraction(vs["time_base"]),
        start_pts=vs.get("start_pts"),
    )


_LADA_VIDEO_READER = video_utils.VideoReader   # pristine class; every (re-)install subclasses this one


# ---------------------------------------------------------------------------
# Lada I/O hooks: our metadata + HTTP options for Lada's readers, per source
# ---------------------------------------------------------------------------

_IO_SOURCES: dict[str, tuple[video_utils.VideoMetadata, FFmpegHttpOptions]] = {}   # src -> (metadata, http options)
_LADA_GET_META = video_utils.get_video_meta_data


def register_source(source: "Source"):
    """Make Lada's lookups for source.src return this source's metadata / HTTP options (latest registration wins)."""
    install_lada_io_hooks()
    _IO_SOURCES[source.src] = (source.meta, source.http)


def forget_sources(keep=()):
    """Drop the registrations of sources no restorer uses any more (after a source switch)."""
    keep = {s.src for s in keep}
    for k in list(_IO_SOURCES):
        if k not in keep:
            del _IO_SOURCES[k]


def install_lada_io_hooks(metadata: video_utils.VideoMetadata | None = None, http: FFmpegHttpOptions | None = None):
    """Make Lada's internal readers use our metadata and send HTTP headers.

    (metadata, http) given: also register them for metadata.video_file (the older single-source call, bench_settings).

    FrameRestorer and MosaicDetector look these up on the video_utils module at call time, so patching the module
    attributes is enough; Lada's source stays untouched. The patched lookups are keyed by the file / URL Lada asks for
    (see register_source), so restorers of different sources (quality switches, source switches) never see each
    other's metadata. Installed once.
    """
    if metadata is not None:
        _IO_SOURCES[metadata.video_file] = (metadata, http or FFmpegHttpOptions({}))
    if getattr(video_utils.VideoReader, "_lada_stream_hook", False):
        return

    def get_meta(path):
        entry = _IO_SOURCES.get(path)
        return entry[0] if entry else _LADA_GET_META(path)

    video_utils.get_video_meta_data = get_meta

    class HttpVideoReader(_LADA_VIDEO_READER):
        """VideoReader with HTTP options and cheap pre-target skipping.

        After seek() the demuxer lands on the keyframe before the target (seconds earlier with long GOPs). Those
        pre-target frames are decoded (they are references) but neither converted nor yielded, so Lada's detection
        and restoration never see them. Lada's two readers (detector feeder, restoration worker) both use this class
        with the same seek target and decoding is deterministic, so they skip exactly the same frames and their
        frame numbering (both start at start_frame) stays in sync.
        """

        target_s = None
        _lada_stream_hook = True

        def __enter__(self):
            entry = _IO_SOURCES.get(self.file)
            opts = entry[1].av_options() if entry and is_url(self.file) else {}
            self.container = av.open(self.file, metadata_errors="ignore", options=opts)
            return self

        def seek(self, offset_ns):
            self.target_s = offset_ns / 1e9
            super().seek(offset_ns)

        def frames(self):
            entry = _IO_SOURCES.get(self.file)
            stride = getattr(entry[0], "restoration_stride", 1) if entry else 1
            if self.target_s is None and stride == 1:
                yield from super().frames()
                return
            # same structure as video_utils.VideoReader.frames(), plus the pre-target skip
            av.logging.restore_default_callback()
            av.logging.set_libav_level(av.logging.ERROR)
            vstream = self.container.streams.video[0]
            vstream.thread_type = "AUTO"
            time_base = vstream.time_base
            entry = _IO_SOURCES.get(self.file)
            fps = float(entry[0].video_fps_exact) if entry else float(vstream.average_rate or 30)
            threshold = (self.target_s or 0.0) - 0.5 / fps
            decoded_index = 0
            skipping = True
            last_good_frame, consecutive_errors, max_consecutive_errors = None, 0, 10
            for packet in self.container.demux(vstream):
                try:
                    for frame in packet.decode():
                        if skipping:
                            if frame.pts is not None and float(frame.pts * time_base) < threshold:
                                continue
                            skipping = False
                        if stride > 1:
                            # Anchor selection to source timestamps so both Lada readers and seeks agree.
                            index = (round((frame.pts - int(entry[0].start_pts or 0)) * time_base *
                                           entry[0].video_fps_exact * stride)
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

    video_utils.VideoReader = HttpVideoReader
