"""
Compare-before/after support for oh_player: pure helpers, no ffmpeg / mpv needed.

The pump writes ONE stacked rawvideo frame per pipe frame into the single ffmpeg stdin: the layers back to back
vertically (W x n*H). ffmpeg splits the stack (split + crop) and encodes one video track per layer into the same
matroska stream, so all tracks share one frame clock (identical pts).

Layers (track order = the order of layer_names): "processed" (what is played today), "original" (the frame before Lada
blended anything), "restored" (Lada output before interpolation, 1x; only when RIFE doubles the rate).
Stack order inside the pipe frame is STACK_ORDER (processed, restored, original), independent of the track order.

Queue outputs (queue_worker, storage layout v2) use a different, cheaper layout built by the
"queue_*" helpers at the end of the ffmpeg section: the worker never stores "restored"; the original track is either the
source's own video stream copied at finalize (no stacking at all) or, with original=encode, a second stacked layer that
is encoded at the SOURCE frame rate (real slots only). Live behaviour (layer_names, GOP, ffmpeg_layer_args) is unchanged.

Originals are not decoded again: install_original_capture() patches FrameRestorer._restore_frame to clone only the
mosaic ROI boxes before Lada blends in place; rebuild_original() pastes them back onto a copy of the restored frame.
"""

import math
from fractions import Fraction

import numpy as np

from oh_streamer.processing import encoding_backend


LAYER_TITLES = {"processed": "oh:processed", "original": "oh:original", "restored": "oh:restored"}
STACK_ORDER = ("processed", "restored", "original")    # vertical order inside the pipe frame
GOP = 15                                               # short GOP: a `vid` switch in mpv decodes at most this much

MAX_PIXEL_RATE = 1920 * 1080 * 60 * 2                  # W*H*fps*layers beyond this is refused (4K30 + RIFE, 4K60)
MAX_PIXEL_RATE_X264 = 1280 * 720 * 30                  # software encoder: W*H*fps per layer
MAX_PIXEL_RATE_VIDEOTOOLBOX = 1920 * 1080 * 30 * 2      # conservative pending Mac multi-track measurements


# -- layers --------------------------------------------------------------------------------------------------------------

def layer_names(rate_mult: int = 1) -> tuple[str, ...]:
    """Tracks of a compare segment, in track order. "restored" (1x, pre-interpolation) only exists at rate_mult 2."""
    return ("processed", "original", "restored") if rate_mult == 2 else ("processed", "original")


def stack_order(layers) -> tuple[str, ...]:
    """The given layers in the order they are stacked in the pipe frame."""
    unknown = [n for n in layers if n not in LAYER_TITLES]
    if unknown:
        raise ValueError(f"unknown layer(s): {unknown}")
    if len(set(layers)) != len(tuple(layers)):
        raise ValueError(f"duplicate layers: {tuple(layers)}")
    return tuple(n for n in STACK_ORDER if n in layers)


def stacked_shape(w: int, h: int, layers) -> tuple[int, int]:
    """(width, height) of one pipe frame, for ffmpeg's -s."""
    return w, h * len(tuple(layers))


def layers_for_write(processed, restored1x, original, layers) -> list:
    """Buffers of one pipe frame in stack order (the caller writes each as memoryview(x.reshape(-1)), one after the
    other: that is the stacked frame). restored1x / original may be None when their layer is not in `layers`."""
    src = {"processed": processed, "restored": restored1x, "original": original}
    out = []
    for n in stack_order(layers):
        if src[n] is None:
            raise ValueError(f"layer {n!r} requested but no frame given")
        out.append(src[n])
    return out


def stack_frames(frames) -> np.ndarray:
    """Stack equally sized (H, W, 3) frames vertically into one (n*H, W, 3) array (tests, tools)."""
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("no frames")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("layers differ in shape")
    return np.concatenate(frames, axis=0)


def build_stacked(processed, restored1x, original, layers, out: np.ndarray | None = None) -> np.ndarray:
    """The W x n*H pipe frame of the given layers (written into `out` when given, to avoid an allocation per frame)."""
    parts = layers_for_write(processed, restored1x, original, layers)
    h = parts[0].shape[0]
    if out is None:
        out = np.empty((h * len(parts),) + parts[0].shape[1:], dtype=parts[0].dtype)
    for i, p in enumerate(parts):
        out[i * h:(i + 1) * h] = p
    return out


def split_stacked(stacked: np.ndarray, layers) -> dict:
    """Inverse of build_stacked: {layer name: view of its rows}."""
    order = stack_order(layers)
    h, rem = divmod(stacked.shape[0], len(order))
    if rem:
        raise ValueError("stacked height is not a multiple of the layer count")
    return {n: stacked[i * h:(i + 1) * h] for i, n in enumerate(order)}


# -- ffmpeg arguments ----------------------------------------------------------------------------------------------------

def encoder_args(encoder: str, gop: int | None = None) -> list[str]:
    """Same encoder arguments as oh_player.start_muxer; gop adds the short-GOP flags for compare segments."""
    return encoding_backend.encoder_args(encoder, purpose="compare", gop=gop)


def ffmpeg_layer_args(w: int, h: int, layers, rate_mult: int, encoder: str):
    """-> (filter_args, map_args, codec_args, meta_args), each a list of ffmpeg argv items.

    The input must be rawvideo bgr24 of size stacked_shape(w, h, layers). `layers` is the track order (layer_names);
    track i of the output is layers[i]. The audio map (-map 1:a:0 ...) is the caller's, after map_args.

    One layer: today's single-track arguments (-map 0:v:0, -vf format=yuv420p,mpdecimate when rate_mult > 1, plain encoder
    arguments, no metadata). Several layers: -filter_complex (split + crop per layer, mpdecimate on every layer when
    rate_mult > 1 since the pump's copies are exact on all of them), -map "[lN]" per track, short-GOP encoder arguments and
    -metadata:s:v:N title=... for each track (first track default, the others not)."""
    layers = tuple(layers)
    if not layers:
        raise ValueError("no layers")
    if layers[0] != "processed":
        raise ValueError("the first track must be 'processed'")
    order = stack_order(layers)
    tail = ["format=yuv420p"] + (["mpdecimate=hi=0:lo=0:frac=0"] if rate_mult > 1 else [])
    vfr = ["-fps_mode", "vfr"] if rate_mult > 1 else []
    if len(layers) == 1:
        filt = ["-vf", ",".join(tail)] + vfr if rate_mult > 1 else []
        return filt, ["-map", "0:v:0"], encoder_args(encoder), []

    n = len(layers)
    split = "[0:v]split=%d%s" % (n, "".join(f"[s{i}]" for i in range(n)))
    chains = [split]
    for i, name in enumerate(order):
        chains.append(f"[s{i}]crop={w}:{h}:0:{i * h}," + ",".join(tail) + f"[v_{name}]")
    filt = ["-filter_complex", ";".join(chains)] + vfr
    maps = []
    meta = []
    for i, name in enumerate(layers):
        maps += ["-map", f"[v_{name}]"]
        meta += [f"-metadata:s:v:{i}", f"title={LAYER_TITLES[name]}"]
    for i in range(n):
        meta += [f"-disposition:v:{i}", "default" if i == 0 else "0"]
    return filt, maps, encoder_args(encoder, GOP), meta


def layer_meta_args(layers) -> list[str]:
    """The track titles / dispositions of ffmpeg_layer_args' meta_args (identical list), for a muxer that copies already
    encoded tracks (queue_worker's concat): -metadata:s:v:N title=... and -disposition:v:N (first default, rest not)."""
    layers = tuple(layers)
    meta = []
    for i, name in enumerate(layers):
        meta += [f"-metadata:s:v:{i}", f"title={LAYER_TITLES[name]}"]
    for i in range(len(layers)):
        meta += [f"-disposition:v:{i}", "default" if i == 0 else "0"]
    return meta


# -- queue outputs (offline worker): storage layout v2 -----------------------------------------------------------------------

ORIGINAL_MODES = ("copy", "encode")                    # job setting compare_original
QUALITY_OFFSETS = {"same": 0, "small": 4, "tiny": 8}   # job setting compare_quality: CQ / CRF offset of an encoded original
COPY_CODECS = ("h264", "hevc")
COPY_PIX_FMTS = ("yuv420p", "yuvj420p")
TAG_FPS, TAG_RATE, TAG_START, TAG_COMPARE, TAG_ORIGINAL = ("OH_SOURCE_FPS", "OH_RATE_MULT", "OH_VIDEO_START",
                                                           "OH_COMPARE", "OH_ORIGINAL")
TAG_ENGINE, TAG_ENGINE_VERSION = "OH_ENGINE", "OH_ENGINE_VERSION"   # which restoration engine made the output (absent = lada, version unknown)


def quality_offset(quality: str) -> int:
    """CQ / CRF offset of the encoded original track for a compare_quality value (ValueError if unknown)."""
    if quality not in QUALITY_OFFSETS:
        raise ValueError(f"unknown compare quality {quality!r}")
    return QUALITY_OFFSETS[quality]


def queue_layers(original: str | None) -> tuple[str, ...]:
    """Video tracks inside the worker's segments (and the pipe): ("processed",) without compare or with original="copy"
    (that track is added at finalize), ("processed", "original") with original="encode". The worker never stores "restored"."""
    return ("processed", "original") if original == "encode" else ("processed",)


def queue_layer_args(w: int, h: int, layers, rate_mult: int, src_fps):
    """-> (filter_args, map_args, meta_args) for the worker's stacked pipe (input W x n*H bgr24, several layers).

    Like ffmpeg_layer_args but for CFR segments: no mpdecimate. Every layer except "processed" is a 1x layer: at rate_mult 2
    it is reduced to the real (even) pipe slots (select 'not(mod(n,2))', the doubler's middle slots are the odd ones) and
    set back on the source frame grid with fps=<src_fps>, so that track is encoded at the SOURCE frame rate and covers the
    same time span as the processed track (one source frame = two processed frames)."""
    layers = tuple(layers)
    if len(layers) < 2:
        raise ValueError("queue_layer_args needs at least two layers")
    if layers[0] != "processed":
        raise ValueError("the first track must be 'processed'")
    order = stack_order(layers)
    n = len(layers)
    chains = ["[0:v]split=%d%s" % (n, "".join(f"[s{i}]" for i in range(n)))]
    for i, name in enumerate(order):
        f = [f"crop={w}:{h}:0:{i * h}"]
        if name != "processed" and rate_mult == 2:
            f += ["select='not(mod(n,2))'", f"fps={Fraction(src_fps)}"]
        f.append("format=yuv420p")
        chains.append(f"[s{i}]" + ",".join(f) + f"[v_{name}]")
    maps = []
    for name in layers:
        maps += ["-map", f"[v_{name}]"]
    return ["-filter_complex", ";".join(chains)], maps, layer_meta_args(layers)


def lada_tags(src_fps, rate_mult: int, video_start: float, compare: bool, original: str | None,
              engine: str | None = None, engine_version: str | None = None) -> dict:
    """Global Matroska tags of a queue output (strings): source fps as a fraction ("30000/1001"), rate multiplier,
    source video start in seconds (= start of v:0 on the file timeline), compare 0/1, original copy / encode / none;
    engine / engine_version (the restoration engine that made it) when given."""
    tags = {TAG_FPS: str(Fraction(src_fps)), TAG_RATE: str(int(rate_mult)), TAG_START: f"{float(video_start):.6f}",
            TAG_COMPARE: "1" if compare else "0", TAG_ORIGINAL: original if compare and original in ORIGINAL_MODES else "none"}
    if engine:
        tags[TAG_ENGINE] = str(engine)
        if engine_version:
            tags[TAG_ENGINE_VERSION] = str(engine_version)
    return tags


def tag_args(tags: dict) -> list[str]:
    """ffmpeg -metadata arguments (global) for a tag dict."""
    out = []
    for k, v in tags.items():
        out += ["-metadata", f"{k}={v}"]
    return out


def _rate(text) -> float:
    num, _, den = str(text or "0/1").partition("/")
    try:
        d = float(den or 1)
        return float(num) / d if d else 0.0
    except ValueError:
        return 0.0


def copy_eligibility(vs: dict) -> str | None:
    """None when the source's video stream (an ffprobe -show_streams entry) can be stream-copied as the original track,
    else the reason: codec h264 / hevc, 8-bit 4:2:0, progressive, no rotation, constant frame rate."""
    codec = vs.get("codec_name")
    if codec not in COPY_CODECS:
        return f"codec {codec} (copy needs h264 / hevc)"
    if vs.get("pix_fmt") not in COPY_PIX_FMTS:
        return f"pixel format {vs.get('pix_fmt')} (copy needs 8-bit 4:2:0)"
    if str(vs.get("field_order") or "progressive") not in ("progressive", "unknown"):
        return f"field order {vs.get('field_order')} (interlaced)"
    rot = (vs.get("tags") or {}).get("rotate")
    try:
        rotated = bool(rot) and int(float(rot)) % 360 != 0
    except ValueError:
        rotated = True
    for sd in vs.get("side_data_list") or []:
        r = sd.get("rotation")
        if r is not None and int(round(float(r))) % 360 != 0:
            rotated = True
    if rotated:
        return "rotation metadata"
    r, a = _rate(vs.get("r_frame_rate")), _rate(vs.get("avg_frame_rate"))
    if r <= 0 or a <= 0:
        return "frame rate not known for sure"
    if not math.isclose(r, a, rel_tol=1e-3):
        return f"variable frame rate (r {r:.4g} / avg {a:.4g})"
    return None


def compare_ok(meta, layers, encoder: str) -> tuple[bool, str]:
    """Guard: (True, "") or (False, reason). meta needs video_width, video_height, video_fps_exact."""
    n = len(tuple(layers))
    if n < 2:
        return False, "compare needs at least two layers"
    w, h = meta.video_width, meta.video_height
    fps = float(Fraction(meta.video_fps_exact))
    rate = w * h * fps
    if rate * n > MAX_PIXEL_RATE:
        return False, f"video too large for {n} tracks ({w}x{h} @ {fps:g} fps)"
    if encoder.endswith("_videotoolbox") and rate * n > MAX_PIXEL_RATE_VIDEOTOOLBOX:
        return False, "video too large for conservative VideoToolbox compare limit (pending Mac validation)"
    if not encoding_backend.is_hardware_encoder(encoder) and rate > MAX_PIXEL_RATE_X264:
        return False, "software encoder too slow for this video"
    return True, ""


# -- original frames -----------------------------------------------------------------------------------------------------

def _clone(x):
    return x.clone() if hasattr(x, "clone") else np.array(x, copy=True)


def capture_boxes(frame, boxes) -> list:
    """Clone the ROI boxes (t, l, b, r inclusive) of `frame` (torch tensor or ndarray, HxWx3) and hang them on it as
    frame._lada_orig = [(box, patch), ...]. Must run before any blend touches the frame."""
    frame._lada_orig = [((t, l, b, r), _clone(frame[t:b + 1, l:r + 1, :])) for (t, l, b, r) in boxes]
    return frame._lada_orig


def apply_patches(arr: np.ndarray, patches) -> np.ndarray:
    """Copy of arr with the saved patches pasted back (the original frame); arr itself when there are none."""
    if not patches:
        return arr
    o = arr.copy()
    for (t, l, b, r), p in patches:
        o[t:b + 1, l:r + 1] = p.numpy() if hasattr(p, "numpy") else p
    return o


def rebuild_original(arr: np.ndarray, frame) -> np.ndarray:
    """Original frame from the restored frame's ndarray `arr` and the patches captured on `frame` (no copy when the
    frame had no mosaic: original == restored)."""
    return apply_patches(arr, getattr(frame, "_lada_orig", None))


def install_original_capture(frame_restorer_cls=None) -> bool:
    """Patch FrameRestorer._restore_frame once so frames get frame._lada_orig when restorer.capture_original is true.
    The patch itself lives in the Lada engine package; imported lazily so this module stays importable without
    torch / lada. Returns True if installed now, False if it already was."""
    from oh_streamer.engines.lada.original_capture import install_original_capture as _install   # noqa: PLC0415
    return _install(frame_restorer_cls)
