"""Shared FFmpeg encoder selection and arguments (no model or platform UI imports).

VideoToolbox is a hardware backend independent of torch/MPS. Its quality scale
is 1..100, higher is better; CQ/CRF scales are lower-is-better. The Apple defaults
follow Lada's encoding_presets.csv, not an assumed equivalence to NVENC quality.
Actual Mac quality, segment IDR boundaries and timestamps still need validation.

VideoToolbox options are based on FFmpeg's implementation:
https://www.ffmpeg.org/doxygen/trunk/videotoolboxenc_8c_source.html

AMF options are based on FFmpeg's H.264/HEVC encoder implementations and
AMD's FFmpeg encoder guide:
https://www.ffmpeg.org/doxygen/trunk/amfenc__h264_8c_source.html
https://github.com/GPUOpen-LibrariesAndSDKs/AMF/wiki/AMF-Encoder-Settings-and-Tuning-in-FFmpeg
"""

import subprocess
import sys


ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "libx264", "libx265",
            "h264_videotoolbox", "hevc_videotoolbox")
BASE_QUALITY = {"hevc_nvenc": 23, "h264_nvenc": 19, "h264_amf": 22, "hevc_amf": 26,
                "libx265": 24, "libx264": 20,
                "h264_videotoolbox": 55, "hevc_videotoolbox": 55}
_ALIASES = {"h264": "h264", "avc": "h264", "hevc": "hevc", "h265": "hevc",
            "h265_nvenc": "hevc_nvenc", "h265_amf": "hevc_amf",
            "h265_videotoolbox": "hevc_videotoolbox"}


def normalize_encoder_name(name: str) -> str | None:
    """Normalize a user setting; concrete backend names remain concrete."""
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    if n in ENCODERS or n == "auto":
        return n
    return _ALIASES.get(n)


def codec_family(encoder: str) -> str:
    n = normalize_encoder_name(encoder)
    if n is None:
        raise ValueError(f"unknown encoder {encoder!r}")
    return "hevc" if n in ("hevc", "hevc_nvenc", "hevc_amf", "hevc_videotoolbox", "libx265") else "h264"


def is_hardware_encoder(encoder: str) -> bool:
    return encoder in ("h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf",
                       "h264_videotoolbox", "hevc_videotoolbox")


def resolve_encoder(name: str, usable=None, *, platform: str | None = None) -> str:
    """Choose a probed hardware encoder or the matching software fallback.

    auto means H.264. On Windows, automatic codec-family choices try NVENC,
    then AMF, then software. A Mac importing an NVENC selection uses
    VideoToolbox for that codec family. Explicit software choices are never
    changed; explicit hardware backends only try the named backend, except for
    the existing Mac NVENC-to-VideoToolbox mapping.
    usable(name) is injectable for callers which already have probe results.
    """
    n = normalize_encoder_name(name)
    if n is None:
        raise ValueError(f"unknown encoder {name!r}")
    if n in ("libx264", "libx265"):
        return n
    platform = sys.platform if platform is None else platform
    family = codec_family(n)
    fallback = "libx265" if family == "hevc" else "libx264"
    if platform == "darwin":
        if n.endswith("_amf"):
            return fallback
        else:
            candidates = [family + "_videotoolbox"]
    elif n.endswith("_videotoolbox"):
        return fallback
    elif n.endswith("_amf"):
        if platform != "win32":
            return fallback
        candidates = [n]
    elif n.endswith("_nvenc"):
        candidates = [n]
    else:
        candidates = [family + "_nvenc"]
        if platform == "win32":
            candidates.append(family + "_amf")
    probe = encoder_usable if usable is None else usable
    for candidate in candidates:
        if probe(candidate):
            return candidate
    return fallback


def encoder_usable(encoder: str, *, ffmpeg: str = "ffmpeg", timeout: float = 10.0) -> bool:
    """Bounded real encode, including quality, pixel format and keyframe options.

    FFmpeg's encoder listing alone cannot detect missing drivers, unsupported
    hardware options or busy/unavailable hardware. Success means the requested
    hardware encoder itself completed the test encode.
    """
    if encoder not in ENCODERS:
        return False
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "lavfi", "-i", "color=size=256x256:rate=30", "-frames:v", "3", "-an"]
    cmd += encoder_args(encoder, purpose="segmented")
    cmd += quality_args(encoder) + gop_args(encoder, 2) + ["-f", "null", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0


def quality_args(encoder: str, spec: str = "", offset: int = 0, *, purpose: str = "segmented") -> list[str]:
    """Quality for all video tracks or a specifier such as ':v:1'.

    Positive original-track offsets reduce quality on every backend. Apple
    offsets are steps on its own scale, while AMF uses its documented CQP
    values rather than a CQ/CRF conversion.
    """
    if encoder not in BASE_QUALITY:
        raise ValueError(f"unknown encoder {encoder!r}")
    if purpose not in ("live", "compare", "hls", "segmented"):
        raise ValueError(f"unknown encoding purpose {purpose!r}")
    base = BASE_QUALITY[encoder]
    if encoder.endswith("_videotoolbox"):
        return ["-q" + (spec or ":v"), str(max(1, min(100, base - offset)))]
    if encoder.endswith("_amf"):
        if purpose in ("live", "compare") and encoder == "h264_amf":
            base = 20
        # AMF CQP is a lower-is-better QP scale, like NVENC CQ/CRF. Unlike
        # AMF QVBR, CQP does not depend on pre-analysis being enabled.
        qp = max(0, min(51, base + offset))
        return ["-rc" + spec, "cqp", "-qp_i" + spec, str(qp), "-qp_p" + spec, str(qp)]
    if encoder.endswith("_nvenc"):
        if purpose in ("live", "compare") and encoder == "h264_nvenc":
            base = 17
        return ["-cq" + spec, str(base + offset)]
    return ["-crf" + spec, str(base + offset)]


def gop_args(encoder: str, frames: int, *, spec: str = "", force: bool = True) -> list[str]:
    """Per-track GOP and optional segment-boundary keyframe expression.

    NVENC/AMF/x264 forced-IDR and Apple's closed-GOP controls are in
    encoder_args; VideoToolbox uses FFmpeg's generic force_key_frames, not
    forced-idr.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("GOP must be a positive integer")
    args = []
    if encoder == "libx265" and force:
        args += ["-x265-params" + spec, f"keyint={frames}:min-keyint={frames}:scenecut=0:bframes=0"]
    args += ["-g" + spec, str(frames)]
    if force:
        args += ["-force_key_frames" + spec, f"expr:eq(mod(n,{frames}),0)"]
    return args


def encoder_args(encoder: str, *, purpose: str = "live", gop: int | None = None) -> list[str]:
    """Codec base for live/compare, HLS or the segmented queue worker.

    segmented includes B-frame/pixel-format controls but leaves quality and GOP
    to quality_args/gop_args per track. HLS leaves its rate cap/profile/pixel
    format/GOP to the HLS command. Existing Windows argument ordering is kept.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}")
    if purpose not in ("live", "compare", "hls", "segmented"):
        raise ValueError(f"unknown encoding purpose {purpose!r}")
    segment = purpose in ("hls", "segmented")
    if purpose == "hls" and codec_family(encoder) != "h264":
        raise ValueError("HLS casting requires an H.264 encoder")
    args = ["-c:v", encoder]
    if encoder.endswith("_videotoolbox"):
        args += ["-allow_sw", "0", "-realtime", "1" if purpose != "segmented" else "0", "-flags", "+cgop"]
        if purpose != "segmented":
            args += quality_args(encoder, purpose=purpose)
        if purpose != "hls":
            args += ["-bf", "0", "-pix_fmt", "yuv420p"]
    elif encoder.endswith("_amf"):
        # AMF uses CQP's qp_i/qp_p controls for the shared quality scale.
        # Live and HLS paths use AMF's documented live usage and low-latency
        # switch; queued segmented encodes use its transcoding preset.
        if purpose in ("live", "compare", "hls"):
            args += ["-usage", "lowlatency", "-latency", "1", "-quality", "speed"]
        else:
            args += ["-usage", "transcoding", "-quality", "balanced"]
        if purpose == "hls":
            # HLS also supplies maxrate/bufsize. CQP ignores those VBV limits,
            # so use peak-constrained VBR. The HLS command has no target -b:v;
            # AMF uses the low-latency usage default and applies its peak cap.
            args += ["-rc", "vbr_peak"]
        elif purpose != "segmented":
            args += quality_args(encoder, purpose=purpose)
        if segment:
            # AMF spells this option with an underscore; the forced keyframes
            # emitted by gop_args become IDRs at segment boundaries.
            args += ["-forced_idr", "1"]
            if encoder == "hevc_amf":
                args += ["-header_insertion_mode", "idr"]
        if purpose in ("live", "compare"):
            args += ["-bf", "0", "-pix_fmt", "yuv420p"]
    elif encoder.endswith("_nvenc"):
        args += ["-preset", "p5", "-rc", "vbr"]
        if purpose != "segmented":
            args += quality_args(encoder, purpose=purpose)
        args += ["-b:v", "0"]
        if segment:
            args += ["-no-scenecut", "1", "-forced-idr", "1"]
        if purpose in ("live", "compare"):
            args += ["-pix_fmt", "yuv420p"]
    else:
        args += ["-preset", "veryfast"]
        if purpose != "segmented":
            args += quality_args(encoder, purpose=purpose)
        if segment and encoder == "libx264":
            args += ["-sc_threshold", "0", "-forced-idr", "1"]
        if purpose in ("live", "compare"):
            args += ["-pix_fmt", "yuv420p"]
    if purpose == "segmented" and not encoder.endswith("_videotoolbox"):
        args += ["-bf", "0", "-pix_fmt", "yuv420p"]
    if gop is not None:
        args += gop_args(encoder, gop, force=False)
        if not segment:
            if encoder.endswith("_nvenc"):
                args += ["-forced-idr", "1"]
            elif encoder.endswith("_amf"):
                args += ["-forced_idr", "1"]
    return args
