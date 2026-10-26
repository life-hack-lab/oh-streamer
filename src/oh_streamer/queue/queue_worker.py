"""
queue_worker: headless background restoration of ONE local video file with Lada.

  lada\\.venv\\Scripts\\python.exe streamer\\queue_worker.py --job <job.json>

job.json:
  {"source": "<local video file>", "work_dir": "<dir for segments/ + segments.json>", "output": "<output.mkv>",
   "segment_seconds": 4,
   "settings": {"detection_model": "v4-fast", "restoration_model": "basicvsrpp-v1.2", "fp16": null,
                "max_clip_length": 240, "encoder": "hevc", "interpolation": "off", "compare": "off"}}
  (missing settings come from player_settings.DEFAULTS, incl. "engine": "lada" (unknown names fall back to lada); encoder: "hevc" / "h265" selects the platform's HEVC backend,
   plus "auto" (H.264), explicit NVENC / VideoToolbox and software backends. Optional "headers": {...} for URL sources.
   interpolation: "off" or "full" (every restored frame pair gets a RIFE in-between frame: constant 2x fps, only for
   sources of 10..40 fps; otherwise the job stays 1x and the log says why, as in the player); "auto" (the player's
   lead-based policy needs a player) is accepted but treated as "off" (a note goes to the log).
   compare: "off" or "on": the output.mkv then carries an extra oh:original track (see "Compare" below);
   compare_original: "copy" (default: the source's own video stream is stream-copied into the output at finalize) or
   "encode" (the original is encoded from the pipe); compare_quality: "same" | "small" (default) | "tiny" (CQ / CRF
   offset 0 / +4 / +8 of the encoded original only). Both are part of the job key only when compare is on.
   Optional "start_k": the segment to start with, default 0.)

stdout, one JSON object per line (flushed; progress at most twice a second; everything else goes to stderr):
  {"type":"progress","frame":<source frames of the finished segments + the current run's progress>,"frames":<total>,
   "fps":<recent frames/s>,"eta":<s or null, from the remaining frames>,"segments":<finished segments>,
   "k":<segment being produced or null>,"phase":"restore"|"finalize"}
  {"type":"perf_obs","mode":"full"|"off","height":<source height>,"src_fps":<source fps>,"fps":<source frames/s>,
   "density":<mosaics per frame>,"frames":<frames measured>,"detector":<detection model>,"engine":<engine name>}   (perf_observe.PerfObserver: steady part of a run, after a
   15 s warm-up, every ~5 min and at the end of a run, >= 1500 frames; never for auto / compare-encode / runs with
   policy copies / a source that is still downloading)
  {"type":"done","output":"<path>"}      exit code 0
  {"type":"error","message":"..."}       exit code 1

Segments: N = round(segment_seconds * fps) source frames each (the last one shorter), video only, Matroska, in
work_dir/segments/seg_<k>.mkv (seg_<k>_f.mkv for a redone one), each starting with an IDR (ffmpeg encodes with a forced keyframe every N frames and
cuts its segment muxer there, as hls_cast.py does). A segment counts only once ffmpeg reports it closed; it is then
added to work_dir/segments.json (atomic replace). The process is safe to kill at any moment; a changed job (source,
models, encoder, ...) invalidates segments.json and starts over.

segments.json (version 2; version 1, a contiguous list from k=0, is still read): {"version", "key", "n", "segments":
[{"k","file","frames"["partial":true]}] sorted by k but NOT necessarily contiguous, "last_k" (null until a run reached the video's natural
end, then the index of the final segment; also inferred from a last listed segment shorter than N), "complete" (last_k
known and every 0..last_k present), "total_frames" (source frames, with last_k), "play": {"fps": "30000/1001", "n": N,
"video_start": s, "rate_mult": 1|2, "layers": [video tracks really inside the segments: ["processed"] or ["processed",
"original"]], "compare": bool, "original": "copy"|"encode"|null (the effective mode), "count": estimated number of
segments}} -- what a player needs: segment k's first frame is at source time video_start + k*N/fps (the files' own
timestamps start at 0). With original "copy" the segments hold only oh:processed; the player adds the source's video
stream itself (it keeps the source timeline, which is the output timeline).
Resume: on start segments.json is read, files it does not list (the partial segment of a killed run) are deleted, and
every listed segment whose file exists and is non-empty is kept (gaps allowed).

Runs: the missing segments ("gaps") are produced by runs. A run starts at a gap k0 (Lada's FrameRestorer at source frame
k0*N: the reader skips the pre-target frames, so the first frame it delivers IS frame k0*N) and ends before the next
already-finished segment k_end (or at the video's natural end): on reaching source frame k_end*N it stops feeding,
finish()es the stage and lets ffmpeg close normally, exactly like at the video end, so every segment of the run is
complete with N source frames. Only a natural end sets last_k / total_frames. After a run the next one starts at the
first gap >= where the run ended (else the first gap from 0); with no gap left (0..last_k all present) the output is
finalised. Job "start_k" (default 0, not part of the job key): the first run starts at the first gap >= start_k. The
models (Lada's, RIFE) are loaded once per process; each run makes a fresh restorer (factory.make; the old one is stopped
first, its clip restoration cancelled) and a fresh stage (FullDoubler / CompareStage are cheap, the RIFE model is shared).

stdin, one JSON object per line: {"cmd":"goto","k":K} and {"cmd":"clock","pos":<source seconds>,"playing":<bool>}
(anything else, EOF, a missing stdin: ignored). goto_action decides: K finished -> the next gap search starts from K when
the current run is over; K within 2 segments ahead of the segment being produced (and inside this run) -> nothing;
otherwise the current run is aborted (its ffmpeg is killed, so the partial segment is untrusted and deleted; segments it
already closed stay) and a run starts at the first gap >= K. During a redo run (below) every goto in the video counts as
"K finished" (the target is playable already): nothing is aborted, the next search starts from K. goto commands are read
on a thread and taken per restored frame. parse_command returns ("goto", K) / ("clock", pos, playing) / None.

Watched jobs (interpolation "full"): the player reports its playback position about once a second (queue app -> queue
service -> "clock" lines here, arrival time = this process's perf_counter) into one rife_interp.RemoteClock per process.
The stage of a normal run is a rife_interp.WatchedDoubler on that clock: while the viewer is close (a few seconds
behind the frame being made) it writes a copy of the previous frame in the middle slot instead of a RIFE frame (the
encoder-side mpdecimate of the player drops it, so such segments play like any other) to keep up. Per segment the policy
copies are counted (SegmentRun.copies, attributed to the segment the middle slot belongs to, i.e. the slot BEFORE the
frame that was pushed); a segment that got any is listed in the manifest with "partial": true (the field is omitted
otherwise; the version stays 2; loading keeps it). Every closed segment logs its speed, e.g. "segment 41: 120 frames in
3.9s (30.8 fps, 1.03x realtime, 37 copies)" (the copies part only when > 0).

Redo: when no real gap is left (every segment up to the natural end exists), partial segments are produced again with
full RIFE (WatchedDoubler never fed = FullDoubler), in redo runs: ascending from the first partial >= the last goto /
run position, covering consecutive partial segments and stopping before a non-partial one (redo_plan). A redone segment
is written under the OTHER file name (seg_%06d.mkv <-> seg_%06d_f.mkv: a different ffmpeg segment pattern for the run), so
a player still reading the old file is never disturbed; when it closes, its manifest entry is replaced atomically (new
file, frames, no "partial") and the old file is removed best-effort (a PermissionError etc. is ignored; the next resume's
clean_work_dir, or the service deleting work_dir, gets it). The segment never leaves the manifest meanwhile. Progress
counts partial segments as unfinished (the card stays below 100 % until the redo is done); the manifest is "complete"
(and the output finalised) only with no gaps AND no partial segments -- last_k / total_frames are still recorded when a
run reaches the video's end, since the redo needs to know where the video stops.

Finalise: the segments 0..last_k are concatenated in order (concat demuxer, -c copy) and the source's audio streams are
stream-copied next to them, the video shifted by the source's video start time so A/V stay in sync -> output.mkv; its
duration must be within 0.5 s of the source's. With compare original=copy the source's first video stream is stream-copied
as the second video track (oh:original) WITHOUT an -itsoffset on that input: it keeps the source's own timeline, which
is where the shifted oh:processed track sits. Every output (compare on or off) carries the global Matroska tags
OH_ENGINE / OH_ENGINE_VERSION (the restoration engine that made it; absent = lada, version unknown),
OH_SOURCE_FPS (fraction), OH_RATE_MULT (1|2), OH_VIDEO_START (s), OH_COMPARE (0|1), OH_ORIGINAL
(copy|encode|none). work_dir is NOT deleted (the queue service does that when nobody watches).

The frame loop pushes every restored frame through a "stage" (PlainStage: one output frame per source frame; with
interpolation "full" rife_interp.FullDoubler, rate_mult 2). Stacked compare layers plug in the same way: a stage reports
rate_mult / layers, which the encoder command takes (build_encode_cmd).

Interpolation and segments: a source segment of N frames is 2N output frames, the keyframe / cut every 2N pipe frames
(so segments stay aligned with source-frame segments, 1x or 2x alike, and the manifest counts SOURCE frames). Output slot
2j is source frame j and slot 2j+1 the RIFE frame between j and j+1 (a repeat of j on a scene cut). The slot after a
segment's last frame needs the NEXT segment's first frame, and is written -- as the middle frame, right before it --
when that frame arrives: it is the last frame of the segment before, and ffmpeg closes that segment only once the next
segment's first frame (the IDR) is in. So a closed segment is always complete and a resumed run (fresh stage, restorer
at frame k*N, first frame written alone) continues exactly where the previous run's output ended, no frame is re-restored
and the boundary pair is interpolated like any other. The last segment of a run (the video's end, or the one before an
already finished segment) ends with finish(): a copy of the last frame, so N source frames are always 2N output frames
(constant 2x fps, same duration, audio untouched; at such a boundary the pair is a repeat, not interpolated). An
interpolated job's manifest key says "full" (and the RIFE model), so changing the mode never reuses segments.

Compare (storage layout v2; compare_stack.py): with compare "on" the output has oh:processed
(default, v:0) and oh:original (v:1); the 1x "restored" track of the old layout is never stored (a rate_mult 2 player
derives it from v:0). original="copy" needs no work at all while restoring: the pipe and the segments are plain single
track and the source's video stream is copied in at finalize. It is only used when the source is eligible
(compare_stack.copy_eligibility: h264 / hevc, 8-bit 4:2:0, progressive, no rotation, CFR); otherwise the reason is logged
and the original is encoded instead (the EFFECTIVE mode is what manifest key, play block and tags say). original="encode":
CompareStage writes ONE stacked rawvideo frame per pipe frame (processed over original; the ROI patches kept by
install_original_capture + restorer.capture_original are pasted back by rebuild_original) and compare_stack.queue_layer_args
splits / crops it into two tracks. At rate_mult 2 the original track keeps only the real (even) pipe slots (select
'not(mod(n,2))', fps=<source fps>): it is encoded at the SOURCE frame rate, N frames per segment next to the processed
track's 2N, over the same time span; the doubler's middle slots are odd (slot 2j = source frame j). Both tracks get a
keyframe at every segment start (-force_key_frames per stream) and GOP = one segment (no short compare GOP any more);
CQ / CRF of the original = main value + the compare_quality offset (-cq:v:1 / -crf:v:1). Segments cut on v:0's forced
IDR; the original's IDR carries the same timestamp and goes into the same new segment. compare_stack.compare_ok may refuse
a video for the encode mode (logged; the job then runs without compare). The 1x timeline never changes the frame counts
the manifest relies on, so segment boundaries and resume work exactly as without compare. A changed compare_original /
compare_quality (encode only) / effective mode changes the manifest key: segments of the old 3-layer layout are never
mixed with the new ones.

The pure logic (plan, manifest, commands, progress throttle) imports no torch; oh_player (torch + Lada) is imported
only when restoring.
"""

import argparse
import collections
import copy
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import types
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from oh_streamer.engines import effective_engine_name, get_engine, set_configured_engine  # noqa: E402  (pure python: the engine is imported lazily)
from oh_streamer.engines.base import EndOfStream, EngineError, ModelChoice  # noqa: E402  (pure python)
from oh_streamer.paths import upstream_root
from oh_streamer.processing import compare_stack  # noqa: E402  (pure python + numpy: layers, stacked frames, ffmpeg split/crop arguments)
from oh_streamer.processing import encoding_backend  # noqa: E402
from oh_streamer.player import player_settings as PS  # noqa: E402  (pure python)
from oh_streamer.processing import rife_interp  # noqa: E402  (pure python at import; torch only inside Rife / Interpolator)
from oh_streamer.queue import perf_observe  # noqa: E402  (pure python)

SEGMENT_SECONDS = 4.0
MANIFEST_VERSION = 2              # version 1 (contiguous segments from k=0) is still read
GOTO_AHEAD = 2                    # a goto this many segments ahead of the one being produced does not abort the run
DURATION_TOLERANCE = 0.5          # seconds: output vs source duration
PROGRESS_INTERVAL = 0.5           # at most two progress lines per second
RATE_WINDOW = 5.0                 # seconds over which the fps / eta of the progress lines are measured
INTERP_VALUES = ("off", "auto", "full")
COMPARE_VALUES = ("off", "on")
ORIGINAL_VALUES = compare_stack.ORIGINAL_MODES           # ("copy", "encode")
QUALITY_VALUES = tuple(compare_stack.QUALITY_OFFSETS)    # ("same", "small", "tiny")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_OUT = sys.stdout                 # the JSON channel; main() points sys.stdout at stderr so library prints cannot mix in


def log(msg):
    print(f"[queue_worker] {msg}", file=sys.stderr, flush=True)


def emit(obj: dict):
    _OUT.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
    _OUT.flush()


class WorkerError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------------------------------------------

def normalize_job(raw: dict) -> tuple[dict, list[str]]:
    """(job with every setting filled in and validated, notes). WorkerError for unusable jobs."""
    notes: list[str] = []
    if not isinstance(raw, dict):
        raise WorkerError("job is not a JSON object")
    for k in ("source", "work_dir", "output"):
        if not isinstance(raw.get(k), str) or not raw[k].strip():
            raise WorkerError(f"job: missing '{k}'")
    st = raw.get("settings") or {}
    if not isinstance(st, dict):
        raise WorkerError("job: 'settings' is not an object")
    from oh_streamer.processing import device_backend

    settings = {k: PS.DEFAULTS[k] for k in ("detection_model", "restoration_model", "fp16", "max_clip_length")}
    settings["encoder"] = "hevc"
    if "device" in st:
        try:
            settings["device"] = device_backend.validate_device(st["device"])
        except ValueError as e:
            raise WorkerError(f"job: {e}") from None
    for k in ("detection_model", "restoration_model", "fp16", "max_clip_length"):
        if k in st:
            try:
                settings[k] = PS.validate(k, st[k])
            except ValueError as e:
                raise WorkerError(f"job: {e}") from None
    # engine: OH_STREAMER_ENGINE (developer override) > the job's setting > lada; an unknown name falls back to lada
    settings["engine"] = effective_engine_name(st.get("engine"), lambda m: notes.append(m))
    if "encoder" in st:
        if not isinstance(st["encoder"], str) or normalize_encoder_name(st["encoder"]) is None:
            raise WorkerError(f"job: bad encoder {st['encoder']!r}")
        settings["encoder"] = st["encoder"]
    interp = st.get("interpolation", "off")
    compare = st.get("compare", "off")
    if interp not in INTERP_VALUES:
        raise WorkerError(f"job: bad interpolation {interp!r}")
    if compare not in COMPARE_VALUES:
        raise WorkerError(f"job: bad compare {compare!r}")
    if interp == "auto":
        notes.append("interpolation='auto' needs a player (lead-based policy): not supported by the queue, treated as 'off'")
    settings["interpolation"], settings["compare"] = ("full" if interp == "full" else "off"), compare
    settings["compare_original"] = st.get("compare_original", "copy")
    settings["compare_quality"] = st.get("compare_quality", "small")
    if settings["compare_original"] not in ORIGINAL_VALUES:
        raise WorkerError(f"job: bad compare_original {settings['compare_original']!r}")
    if settings["compare_quality"] not in QUALITY_VALUES:
        raise WorkerError(f"job: bad compare_quality {settings['compare_quality']!r}")
    seg = raw.get("segment_seconds", SEGMENT_SECONDS)
    if isinstance(seg, bool) or not isinstance(seg, (int, float)) or not 0.2 <= seg <= 60:
        raise WorkerError(f"job: bad segment_seconds {seg!r}")
    headers = raw.get("headers") or {}
    if not isinstance(headers, dict):
        raise WorkerError("job: 'headers' is not an object")
    start_k = raw.get("start_k", 0)
    if isinstance(start_k, bool) or not isinstance(start_k, int) or start_k < 0:
        raise WorkerError(f"job: bad start_k {start_k!r}")
    job = {"source": raw["source"], "work_dir": raw["work_dir"], "output": raw["output"],
           "segment_seconds": float(seg), "settings": settings, "headers": headers, "start_k": start_k}
    cache_fields = ("source_url", "source_identity", "source_generation", "source_status_path", "source_probe")
    present = [k for k in cache_fields if k in raw]
    if present:
        if len(present) != len(cache_fields):
            raise WorkerError("job: cached source needs source_url, source_identity, source_generation, "
                              "source_status_path, and source_probe")
        source_url = raw.get("source_url")
        if not is_local_runtime_url(source_url):
            raise WorkerError("job: source_url must be an HTTP URL on 127.0.0.1")
        identity, generation = raw.get("source_identity"), raw.get("source_generation")
        if identity in (None, "") or isinstance(identity, (bool, int, float)):
            raise WorkerError("job: bad source_identity")
        if generation in (None, "") or isinstance(generation, (bool, dict, list)):
            raise WorkerError("job: bad source_generation")
        status_path, source_probe = raw.get("source_status_path"), raw.get("source_probe")
        if not isinstance(status_path, str) or not status_path.strip():
            raise WorkerError("job: bad source_status_path")
        if not isinstance(source_probe, dict):
            raise WorkerError("job: source_probe is not an object")
        # The first TS can report only its own nb_frames / duration. Cache metadata supplies the VOD span;
        # synthesize a normal ffprobe shape with that span and no misleading one-segment frame count.
        try:
            source_probe = full_vod_probe(source_probe)
            parse_probe(source_probe)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            raise WorkerError(f"job: invalid source_probe ({e})") from None
        job.update({"source_url": source_url, "source_identity": identity, "source_generation": generation,
                    "source_status_path": status_path.strip(), "source_probe": source_probe})
    return job, notes


def is_local_runtime_url(value) -> bool:
    """The cache proxy is deliberately limited to plain HTTP on the IPv4 loopback address."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None
                and parsed.port > 0 and parsed.port <= 65535 and parsed.username is None
                and parsed.password is None and not parsed.fragment)
    except ValueError:
        return False


def full_vod_probe(probe: dict) -> dict:
    """Normalize the shared cache probe to the full VOD duration and discard first-TS frame counts."""
    result = copy.deepcopy(probe)
    streams = result.get("streams")
    if not isinstance(streams, list):
        raise ValueError("missing streams")
    fmt = result.setdefault("format", {})
    if not isinstance(fmt, dict):
        raise ValueError("format is not an object")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    duration = result.get("duration") or metadata.get("duration") or fmt.get("duration")
    if duration is None:
        raise ValueError("missing full VOD duration")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("bad full VOD duration")
    fmt["duration"] = str(duration)
    for stream in streams:
        if stream.get("codec_type") == "video":
            stream.pop("nb_frames", None)
            stream["duration"] = str(duration)
    return result


def normalize_encoder_name(name: str) -> str | None:
    """Canonical encoder family for a job / player encoder value, None if unknown."""
    return encoding_backend.normalize_encoder_name(name)


def resolve_encoder(name: str, usable, *, platform: str | None = None) -> str:
    """The probed platform encoder or matching software fallback."""
    try:
        return encoding_backend.resolve_encoder(name, usable, platform=platform)
    except ValueError as e:
        raise WorkerError(str(e)) from None


def encoder_usable(encoder: str) -> bool:
    """Use the shared bounded real-encode capability probe."""
    return encoding_backend.encoder_usable(encoder)


# ---------------------------------------------------------------------------------------------------------------
# source probing (ffprobe only)
# ---------------------------------------------------------------------------------------------------------------

class SourceInfo:
    def __init__(self, fps: Fraction, frames: int, width: int, height: int, video_start: float, duration: float,
                 has_audio: bool, copy_why: str | None = None):
        self.fps, self.frames, self.width, self.height = fps, frames, width, height
        self.video_start, self.duration, self.has_audio = video_start, duration, has_audio
        self.copy_why = copy_why        # None: the video stream can be stream-copied as the original track, else why not


def parse_probe(probe: dict) -> SourceInfo:
    streams = probe.get("streams") or []
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise WorkerError("the source has no video stream")
    num, _, den = str(vs.get("r_frame_rate", vs.get("avg_frame_rate", "0/1"))).partition("/")
    fps = Fraction(int(num), int(den or 1)) if int(den or 1) else Fraction(0)
    if fps <= 0:
        raise WorkerError("cannot determine the frame rate")
    fmt = probe.get("format") or {}
    duration = float(fmt.get("duration") or vs.get("duration") or 0)
    frames = int(vs.get("nb_frames") or 0) or max(1, int(round(duration * float(fps))))
    vstart = float(vs.get("start_time") or fmt.get("start_time") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return SourceInfo(fps, frames, int(vs["width"]), int(vs["height"]), vstart, duration, has_audio,
                      compare_stack.copy_eligibility(vs))


def source_status_error(status: dict, identity, generation) -> str | None:
    """Pure cache completion gate. A ready playlist from another generation is never trusted."""
    if not isinstance(status, dict):
        return "source status is not an object"
    if status.get("version") != 1:
        return "source status version is invalid"
    if status.get("identity") != identity:
        return "source cache identity changed"
    if status.get("generation") != generation:
        return "source cache generation changed"
    if status.get("error") not in (None, ""):
        return f"source cache has an error: {status.get('error')}"
    if status.get("status") != "complete":
        state = status.get("status") or "unknown"
        return f"source cache is {state}"
    ready, total = status.get("ready_segments"), status.get("segment_count")
    if isinstance(ready, list):
        ready = len(ready)
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0 \
            or isinstance(ready, bool) or not isinstance(ready, int) or ready != total:
        return "source cache is not fully ready"
    return None


def cache_pts_error(pts_times, fps: Fraction | float, video_start: float | None = None,
                    first_frame: int | None = None) -> str | None:
    """Check every decoded cache frame's timestamps, including frame-to-frame continuity.

    With first_frame/video_start, timestamps are checked against the known VOD frame index. Otherwise the sequence
    must be continuous at one source frame per PTS step. A half-frame tolerance absorbs time-base rounding only.
    """
    values = list(pts_times)
    if not values:
        return "source ended before producing a video frame"
    rate = float(fps)
    if not math.isfinite(rate) or rate <= 0:
        return "source frame rate is invalid"
    frame_s = 1.0 / rate
    tol = frame_s * 0.5
    if any(t is None or not math.isfinite(float(t)) for t in values):
        return "source frame has no valid PTS"
    if first_frame is not None and video_start is not None:
        for i, t in enumerate(values):
            want = float(video_start) + (first_frame + i) * frame_s
            if abs(float(t) - want) > tol:
                return (f"source PTS gap at frame {first_frame + i}: got {float(t):.6f}s, "
                        f"expected {want:.6f}s")
    else:
        for i, (a, b) in enumerate(zip(values, values[1:]), 1):
            if abs((float(b) - float(a)) - frame_s) > tol:
                return f"source PTS gap before frame {i}"
    return None


def source_completion_error(status: dict, identity, generation, pts_times, fps: Fraction | float,
                            duration: float, video_start: float, *, natural_end: bool = True,
                            stopped: bool = False, frame_count: int | None = None,
                            expected_frames: int | None = None) -> str | None:
    """Pure no-wait completion gate for a cache-backed VOD worker run."""
    if stopped:
        return "source completion was interrupted"
    if not natural_end:
        return "source did not reach natural EOF"
    status_problem = source_status_error(status, identity, generation)
    if status_problem:
        return status_problem
    pts_times = list(pts_times)
    pts_problem = cache_pts_error(pts_times, fps)
    if pts_problem:
        return pts_problem
    if expected_frames is not None and frame_count != expected_frames:
        return f"source ended after {frame_count} frames, expected {expected_frames}"
    rate = float(fps)
    expected_end = float(video_start) + float(duration)
    observed_end = float(pts_times[-1]) + 1.0 / rate
    # EXTINF is a rounded frame-span measure. Half a frame catches truncated EOF while allowing time-base rounding.
    tolerance = min(0.5, 0.5 / rate)
    if abs(observed_end - expected_end) > tolerance:
        return (f"source ended at {observed_end:.6f}s, expected VOD end {expected_end:.6f}s "
                f"(tolerance {tolerance:.6f}s)")
    return None


def read_source_status(job: dict) -> dict:
    path = job.get("source_status_path")
    try:
        with open(path, "r", encoding="utf-8") as f:
            status = json.load(f)
    except (OSError, ValueError) as e:
        raise WorkerError(f"cannot read source cache status: {e}") from None
    problem = source_status_error(status, job.get("source_identity"), job.get("source_generation"))
    if problem:
        raise WorkerError(problem)
    return status


def ffprobe_json(path: str) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise WorkerError(f"ffprobe failed for {path}: {p.stderr.strip()[:200]}")
    return json.loads(p.stdout)


# ---------------------------------------------------------------------------------------------------------------
# segment plan
# ---------------------------------------------------------------------------------------------------------------

class SegmentPlan:
    """N source frames per segment; segment k starts at source frame k*N."""

    def __init__(self, fps, frames_total: int, video_start: float = 0.0, seconds: float = SEGMENT_SECONDS):
        self.fps = Fraction(fps)
        if self.fps <= 0:
            raise ValueError(f"bad frame rate {fps!r}")
        self.n = max(1, round(seconds * self.fps))
        self.frames_total = max(1, int(frames_total))
        self.count = -(-self.frames_total // self.n)
        self.video_start = float(video_start)

    def first_frame(self, k: int) -> int:
        return k * self.n

    def frames_in(self, k: int) -> int:
        return max(0, min(self.n, self.frames_total - k * self.n))

    def source_time(self, k: int) -> float:
        """Source time of segment k's first frame (video start included)."""
        return self.video_start + float(Fraction(k * self.n) / self.fps)

    def start_ns(self, k: int) -> int:
        """FrameRestorer start offset for segment k. 0 for the first segment (Lada then does not seek at all). Rounded
        up a hair: Lada derives its frame numbers with int(), and 4.004 s must not become frame 119 (the reader's own
        skip threshold is half a frame early, so a few microseconds late cost nothing)."""
        if k == 0:
            return 0
        return math.ceil(Fraction(self.video_start) * 10**9 + Fraction(k * self.n) / self.fps * 10**9) + 1000


REDO_SUFFIX = "_f"                # file name suffix of the alternate (redone) file of a segment


def segment_name(k: int, suffix: str = "") -> str:
    return f"seg_{k:06d}{suffix}.mkv"


def file_suffix(name: str) -> str:
    """"" for a regular segment file (seg_000005.mkv), REDO_SUFFIX for the alternate one (seg_000005_f.mkv)."""
    return REDO_SUFFIX if name.endswith(REDO_SUFFIX + ".mkv") else ""


def alt_suffix(name: str) -> str:
    """The suffix a redone segment currently stored in file `name` gets: always the other one, so the new file never
    overwrites the one a player may be reading."""
    return "" if file_suffix(name) else REDO_SUFFIX


def parse_segment_name(name: str) -> tuple[int, str] | None:
    """(k, suffix) of seg_<k>.mkv / seg_<k>_f.mkv, else None."""
    if not (name.startswith("seg_") and name.endswith(".mkv")):
        return None
    stem = name[len("seg_"):-len(".mkv")]
    suffix = ""
    if stem.endswith(REDO_SUFFIX):
        stem, suffix = stem[:-len(REDO_SUFFIX)], REDO_SUFFIX
    return (int(stem), suffix) if stem.isdigit() else None


# ---------------------------------------------------------------------------------------------------------------
# manifest (work_dir/segments.json) + resume bookkeeping
# ---------------------------------------------------------------------------------------------------------------

def replace_atomic(tmp: str, dst: str, tries: int = 20):
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:       # Windows: a virus scanner / indexer may hold the file for a moment
            if i == tries - 1:
                raise
            time.sleep(0.05)


def write_json_atomic(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    replace_atomic(tmp, path)


def effective_interpolation(requested: str, fps) -> tuple[str, str]:
    """("full" | "off", why not): "full" only for sources the player would interpolate too (rife_interp.eligible)."""
    if requested != "full":
        return "off", ""
    ok, _ = rife_interp.eligible(fps)
    if ok:
        return "full", ""
    f = float(fps) if fps else 0.0
    if not f > 0:
        return "off", "unknown frame rate"
    if f > rife_interp.MAX_SOURCE_FPS:
        return "off", f"source is already {f:.3g} fps (limit {rife_interp.MAX_SOURCE_FPS:g})"
    return "off", f"source is only {f:.3g} fps (minimum {rife_interp.MIN_SOURCE_FPS:g})"


class ComparePlan(collections.namedtuple("ComparePlan", "on original layers notes")):
    """on: the output has a oh:original track; original: "copy" | "encode" | None (the EFFECTIVE mode); layers: the
    video tracks inside the segments / the pipe (compare_stack.queue_layers); notes: why the request was changed."""
    __slots__ = ()

    @property
    def out_layers(self) -> tuple:
        """The video tracks of the finished output, in track order."""
        return self.layers + (("original",) if self.original == "copy" else ())


NO_COMPARE = ComparePlan(False, None, ("processed",), ())


def compare_plan(settings: dict, info: SourceInfo, encoder: str) -> ComparePlan:
    """What compare really does for this job: nothing (compare off, or refused), original=copy (source stream copied at
    finalize; only for eligible sources), or original=encode (stacked pipe, second track from the real slots). A copy that
    is not possible falls back to encode (note says why); compare_stack.compare_ok (the guard the player uses)
    may still refuse the encode mode, then compare is off."""
    if settings.get("compare") != "on":
        return NO_COMPARE
    notes = []
    mode = settings.get("compare_original", "copy")
    if mode == "copy" and info.copy_why:
        notes.append(f"compare: cannot copy the source video as the original track ({info.copy_why}); encoding it instead")
        mode = "encode"
    if mode == "copy":
        return ComparePlan(True, "copy", compare_stack.queue_layers("copy"), tuple(notes))
    names = compare_stack.queue_layers("encode")
    meta = types.SimpleNamespace(video_width=info.width, video_height=info.height, video_fps_exact=info.fps)
    ok, why = compare_stack.compare_ok(meta, names, encoder)
    if not ok:
        return ComparePlan(False, None, ("processed",), tuple(notes) + (f"compare off: {why}",))
    return ComparePlan(True, "encode", names, tuple(notes))


def manifest_engine(doc) -> tuple:
    """(engine name, engine version or None) a manifest document was made with; no fields = ("lada", None)."""
    doc = doc if isinstance(doc, dict) else {}
    name, version = doc.get("engine"), doc.get("engine_version")
    return (name if isinstance(name, str) and name else "lada"), (version if isinstance(version, str) and version else None)


def engine_stamp(name: str) -> tuple:
    """(name, version) of the engine a job runs with; the version is None when the engine cannot tell."""
    try:
        return name, get_engine(name).info().version or None
    except Exception as e:  # noqa: BLE001 - the stamp is a record, never a reason to fail the job
        log(f"engine version unavailable: {e!r}")
        return name, None


def job_key(job: dict, info: SourceInfo, plan: SegmentPlan, encoder: str) -> dict:
    """Everything that makes the finished segments different: if it changes, the segments are not reusable."""
    st = job["settings"]
    interp = effective_interpolation(st["interpolation"], info.fps)[0]
    cp = compare_plan(st, info, encoder)
    if job.get("source_identity") is not None:
        # The stable cache identity defines reusable restored segments. Generation isolates this worker's status gate
        # from concurrent / replaced downloads, but can change when a paused item resumes and is not part of the key.
        key = {"source_identity": job["source_identity"],
               "fps": str(info.fps), "n": plan.n, "size": [info.width, info.height], "encoder": encoder,
               "detection_model": st["detection_model"], "restoration_model": st["restoration_model"],
               "fp16": st["fp16"], "max_clip_length": st["max_clip_length"], "interpolation": interp,
               "compare": "on" if cp.on else "off"}
    else:
        try:
            stat = os.stat(job["source"])
            src_id = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            src_id = None
        key = {"source": os.path.abspath(job["source"]), "source_id": src_id, "fps": str(info.fps), "n": plan.n,
               "size": [info.width, info.height], "encoder": encoder, "detection_model": st["detection_model"],
               "restoration_model": st["restoration_model"], "fp16": st["fp16"],
               "max_clip_length": st["max_clip_length"], "interpolation": interp,
               "compare": "on" if cp.on else "off"}
    if st.get("device", "auto") != "auto":
        key["device"] = st["device"]
    if st.get("engine", "lada") != "lada":     # lada's keys stay as they were: resumes of earlier work dirs keep working
        key["engine"] = st["engine"]
    if cp.on:                       # the quality only matters for an encoded original
        key["original"] = cp.original
        if cp.original == "encode":
            key["compare_quality"] = st.get("compare_quality", "small")
    if interp == "full":
        key["rife"] = rife_interp.MODEL
    return key


class Manifest:
    """The finished segments of a job (sorted by k, gaps allowed). Thread-safe (ffmpeg's segment list is read on its own
    thread)."""

    def __init__(self, path: str, key: dict, play: dict | None = None):
        self.path, self.key, self.play = path, key, play
        self.segments: list[dict] = []          # [{"k", "file", "frames"} + "partial": True when it holds policy copies]
        self.last_k: int | None = None          # index of the final segment, known once a run reached the video's end
        self.total_frames: int | None = None
        self.engine: str | None = None          # the restoration engine / version that makes the segments (stamped on save)
        self.engine_version: str | None = None
        self.lock = threading.RLock()      # save() under the lock reads the segments again

    @classmethod
    def load(cls, path: str, key: dict, seg_dir: str):
        """Manifest of a previous run, or an empty one when there is none / it is unusable / the job changed. Every listed
        segment whose file is present and non-empty is kept (version 1 files are the contiguous case)."""
        m = cls(path, key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return m
        if not isinstance(raw, dict) or raw.get("version") not in (1, MANIFEST_VERSION) or raw.get("key") != key:
            return m
        segs = raw.get("segments")
        if not isinstance(segs, list):
            return m
        by_k = {}
        for s in segs:
            try:
                by_k[int(s["k"])] = {"k": int(s["k"]), "file": str(s["file"]), "frames": int(s["frames"])}
                if s.get("partial") is True:
                    by_k[int(s["k"])]["partial"] = True
            except (KeyError, TypeError, ValueError):
                continue
        for k in sorted(by_k):
            if k < 0:
                continue
            p = os.path.join(seg_dir, by_k[k]["file"])
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                m.segments.append(by_k[k])
        if isinstance(raw.get("play"), dict):
            m.play = raw["play"]
        last_k = raw.get("last_k")
        if raw["version"] == 1 and raw.get("complete") and m.segments and len(m.segments) == len(by_k):
            last_k = m.segments[-1]["k"]            # version 1: complete = the whole contiguous list is the video
        if (isinstance(last_k, int) and not isinstance(last_k, bool) and last_k >= 0
                and all(s["k"] <= last_k for s in m.segments)):
            m.last_k = last_k
            tf = raw.get("total_frames")
            m.total_frames = tf if isinstance(tf, int) and not isinstance(tf, bool) else None
        return m

    def finished(self) -> set[int]:
        with self.lock:
            return {s["k"] for s in self.segments}

    @property
    def next_k(self) -> int:
        """First missing k from 0 (the length of the contiguous prefix)."""
        ks = self.finished()
        k = 0
        while k in ks:
            k += 1
        return k

    def frames_done(self, partial: bool = True) -> int:
        """Source frames of the finished segments; partial=False leaves out the partial ones (progress: they are not
        finished until redone)."""
        return sum(s["frames"] for s in self.segments if partial or not s.get("partial"))

    def partial_ks(self) -> set[int]:
        """The segments that hold policy copies and still wait for their redo."""
        with self.lock:
            return {s["k"] for s in self.segments if s.get("partial")}

    def finished_count(self) -> int:
        """Segments that are final (partial ones do not count)."""
        return sum(1 for s in self.segments if not s.get("partial"))

    def end_k(self, n: int | None = None) -> int | None:
        """Index of the video's final segment if known: last_k, or the last listed segment when it is shorter than N
        (only a run that reached the video's end can have closed a short one)."""
        if self.last_k is not None:
            return self.last_k
        n = n or self.key.get("n")
        if n and self.segments and self.segments[-1]["frames"] < n:
            return self.segments[-1]["k"]
        return None

    def has_all(self, last: int) -> bool:
        ks = self.finished()
        return all(k in ks for k in range(last + 1))

    @property
    def complete(self) -> bool:
        """The end is known, every segment 0..last_k is there and none is partial (still to be redone)."""
        e = self.end_k()
        return e is not None and self.has_all(e) and not self.partial_ks()

    def is_complete(self, n: int) -> bool:
        e = self.end_k(n)
        return e is not None and self.has_all(e) and not self.partial_ks()

    def gap_limit(self, count: int) -> int:
        """Exclusive upper bound of the k values that can still be missing: the segments up to the known end, else the
        estimated segment count (plus one probe beyond the last finished segment: the estimate may be low)."""
        e = self.end_k()
        if e is not None:
            return e + 1
        return max(count, (self.segments[-1]["k"] + 2) if self.segments else 0)

    def doc(self) -> dict:
        e = self.end_k()
        total = self.total_frames
        if total is None and e is not None and self.has_all(e):
            total = self.frames_done()
        doc = {"version": MANIFEST_VERSION, "key": self.key, "n": self.key.get("n"), "segments": self.segments,
               "last_k": e, "complete": self.complete, "total_frames": total, "play": self.play}
        if self.engine:       # manifests from before the engine fields have none = lada, version unknown
            doc["engine"], doc["engine_version"] = self.engine, self.engine_version
        return doc

    def save(self):
        write_json_atomic(self.path, self.doc())

    def add(self, k: int, frames: int, partial: bool = False, file: str | None = None) -> str | None:
        """Any k; a segment listed under the same k is replaced in one step (the list is swapped and saved atomically,
        the segment never disappears). file: its name in seg_dir (default seg_<k>.mkv); partial: it holds policy copies
        (the field is left out otherwise). Returns the file name of the entry that was replaced, or None."""
        with self.lock:
            old = next((s["file"] for s in self.segments if s["k"] == k), None)
            segs = [s for s in self.segments if s["k"] != k]
            entry = {"k": int(k), "file": file or segment_name(k), "frames": int(frames)}
            if partial:
                entry["partial"] = True
            segs.append(entry)
            self.segments = sorted(segs, key=lambda s: s["k"])
            self.save()
            return old

    def mark_complete(self, total_frames: int, last_k: int | None = None):
        """A run reached the video's natural end: total_frames source frames, the final segment is last_k (derived from
        N when not given). The manifest only counts as `complete` once no segment is partial any more."""
        with self.lock:
            if last_k is None:
                last_k = max(0, -(-int(total_frames) // self.key["n"]) - 1)
            self.last_k, self.total_frames = int(last_k), int(total_frames)
            self.save()

    def set_play(self, play: dict):
        with self.lock:
            self.play = play
            self.save()

    def reset(self):
        with self.lock:
            self.segments, self.last_k, self.total_frames = [], None, None


def clean_work_dir(seg_dir: str, manifest: Manifest) -> list[str]:
    """Delete every file in seg_dir that the manifest does not list (the partial segment of a killed run, and the
    segments of an invalidated manifest). Returns the removed names."""
    keep = {s["file"] for s in manifest.segments}
    removed = []
    os.makedirs(seg_dir, exist_ok=True)
    for name in os.listdir(seg_dir):
        if name not in keep:
            try:
                os.remove(os.path.join(seg_dir, name))
                removed.append(name)
            except OSError as e:
                raise WorkerError(f"cannot delete leftover {name}: {e}") from None
    return removed


def closed_segment_frames(k: int, k0: int, n: int, fed: int, natural_end: bool) -> int | None:
    """Frames of segment k just reported closed by ffmpeg, in a run that started at segment k0 and has fed `fed`
    frames so far; None if the segment must not be trusted. A segment closed while the run goes on is full; one closed
    when the input ended cleanly (natural_end: the video really ended, or the run stopped feeding at a segment boundary
    and finish()ed -- then the last segment is full) holds whatever remained. After an abnormal end ffmpeg is killed
    instead of closed, so a short segment can then only be a truncated one."""
    have = min(n, fed - (k - k0) * n)
    if have <= 0:
        return None
    if have == n or natural_end:
        return have
    return None


def closed_source_frames(k: int, k0: int, n: int, rate_mult: int, fed: int, natural_end: bool) -> int | None:
    """closed_segment_frames for a stage that writes rate_mult pipe frames per source frame (fed counts pipe frames):
    the number of SOURCE frames of the closed segment k, None if it must not be trusted (also when the pipe frames are
    not a whole number of source frames)."""
    pipe = closed_segment_frames(k, k0, n * rate_mult, fed, natural_end)
    if pipe is None or pipe % rate_mult:
        return None
    return pipe // rate_mult


# ---------------------------------------------------------------------------------------------------------------
# runs over gaps + goto commands (pure logic)
# ---------------------------------------------------------------------------------------------------------------

def first_gap(finished, start: int, limit: int) -> int | None:
    """First k >= start in range(limit) that is not finished, wrapping to 0; None when every k < limit is finished."""
    start = max(0, start)
    for k in list(range(start, limit)) + list(range(0, min(start, limit))):
        if k not in finished:
            return k
    return None


def run_end_k(finished, k0: int) -> int | None:
    """The next already finished segment after k0: the run stops before it (None: it runs to the video's end)."""
    later = [k for k in finished if k > k0]
    return min(later) if later else None


def run_frame_limit(k0: int, k_end: int | None, n: int) -> int | None:
    """Source frames a run from segment k0 feeds before it stops and finish()es (None: until the video ends)."""
    return None if k_end is None else (k_end - k0) * n


def redo_plan(segments, start: int, last: int | None) -> tuple[int, int | None, str] | None:
    """The next redo run, or None (nothing to redo, the video's end is unknown, or a real gap is left -- gaps come
    first). (k0, k_end, suffix): the first partial segment >= start (else the first one), the run covering the consecutive
    partial segments after it that are stored under the same file variant and stopping before anything else (k_end None:
    the run reaches the video's end), and the file suffix the redone files get (the other variant than k0's current file;
    see alt_suffix)."""
    if last is None:
        return None
    by_k = {s["k"]: s for s in segments if s["k"] <= last}
    if any(k not in by_k for k in range(last + 1)):
        return None
    partial = sorted(k for k, s in by_k.items() if s.get("partial"))
    if not partial:
        return None
    k0 = next((k for k in partial if k >= start), partial[0])
    suffix = alt_suffix(by_k[k0]["file"])
    k = k0 + 1
    while k <= last and by_k[k].get("partial") and alt_suffix(by_k[k]["file"]) == suffix:
        k += 1
    return k0, (None if k > last else k), suffix


def goto_action(k: int, finished, run_k0: int | None, cur_k: int | None, k_end: int | None, limit: int,
                ahead: int = GOTO_AHEAD, redo: bool = False) -> str:
    """What a {"cmd":"goto","k":K} does while a run (from segment run_k0, now producing cur_k, stopping before k_end)
    is going: "next" (K is finished: the next gap search starts at K once this run is over), "keep" (K is the segment
    being produced, one the run already passed, or at most `ahead` segments ahead of it, and the run gets there: nothing
    to do), "ignore" (K outside the video), else "abort" (stop this run, start one at the first gap >= K). During a redo
    run every segment of the video exists and is playable: any K inside the video is "next" (never an abort)."""
    if k < 0 or k >= limit:
        return "ignore"
    if redo:
        return "next"
    if k in finished:
        return "next"
    if run_k0 is not None and cur_k is not None and run_k0 <= k <= cur_k + ahead and (k_end is None or k < k_end):
        return "keep"
    return "abort"


def parse_command(line) -> tuple | None:
    """A stdin line as ("goto", K) for {"cmd":"goto","k":K} or ("clock", pos, playing) for {"cmd":"clock","pos":seconds,
    "playing":bool} (pos a finite number, playing a real boolean); None for anything else (malformed JSON, other
    commands, bad values)."""
    try:
        d = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    if d.get("cmd") == "goto":
        k = d.get("k")
        return ("goto", k) if isinstance(k, int) and not isinstance(k, bool) and k >= 0 else None
    if d.get("cmd") == "clock":
        pos, playing = d.get("pos"), d.get("playing")
        if (isinstance(pos, (int, float)) and not isinstance(pos, bool) and math.isfinite(pos)
                and isinstance(playing, bool)):
            return ("clock", float(pos), playing)
    return None


def pipe_lines(fd: int, interval: float = 0.1):
    """Lines (bytes) arriving on the pipe behind file descriptor fd, until it breaks. Windows: polled with PeekNamedPipe and
    never left blocking in a read -- a synchronous read pending on stdin from a second thread makes other threads'
    queries of that handle (some DLL initialisations, e.g. scipy's) hang for good. Not a pipe (NUL, file, console): no
    lines at all."""
    import ctypes
    import msvcrt
    from ctypes import wintypes
    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                     ctypes.c_void_p]
    handle = msvcrt.get_osfhandle(fd)
    buf = b""
    while True:
        avail = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(avail), None):
            return                      # broken pipe (the parent is gone / closed it) or not a pipe
        if avail.value:
            data = os.read(fd, avail.value)
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line
        else:
            time.sleep(interval)


def start_command_reader(stream=None, clock=None) -> "queue.Queue[int]":
    """Queue of the K values of the goto commands arriving on stdin, read on a daemon thread. clock commands go straight
    into `clock` (a rife_interp.RemoteClock: update(pos, playing), stamped with the arrival time) when there is one, else
    they are dropped. EOF, no stdin, a terminal or an unreadable stream just means no commands. (A given stream is read
    with blocking readline; the real stdin on Windows with pipe_lines.)"""
    q: queue.Queue = queue.Queue()
    real = stream is None
    stream = sys.stdin if real else stream
    try:
        if stream is None or stream.isatty():
            return q
        buf = getattr(stream, "buffer", stream)
        fd = stream.fileno() if real and os.name == "nt" else None
    except (OSError, ValueError, AttributeError):
        return q

    def pump():
        try:
            lines = pipe_lines(fd) if fd is not None else iter(buf.readline, b"")
            for raw in lines:
                cmd = parse_command(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw)
                if cmd is None:
                    continue
                if cmd[0] == "goto":
                    q.put(cmd[1])
                elif clock is not None:
                    clock.update(cmd[1], cmd[2])
        except Exception:  # noqa: BLE001 - a broken command channel must never take the job down
            pass

    threading.Thread(target=pump, name="stdin commands", daemon=True).start()
    return q


# ---------------------------------------------------------------------------------------------------------------
# ffmpeg commands
# ---------------------------------------------------------------------------------------------------------------

BASE_QUALITY = encoding_backend.BASE_QUALITY


def quality_args(encoder: str, spec: str = "", offset: int = 0) -> list[str]:
    """Backend-specific quality; positive offsets lower original-track quality."""
    try:
        return encoding_backend.quality_args(encoder, spec, offset)
    except ValueError as e:
        raise WorkerError(str(e)) from None


def build_encode_cmd(plan: SegmentPlan, k0: int, in_w: int, in_h: int, encoder: str, seg_dir: str,
                     rate_mult: int = 1, layers: tuple = ("processed",), suffix: str = "", cq_offset: int = 0) -> list[str]:
    """One restoring run: raw BGR frames on stdin, from source frame k0*N on -> Matroska segments seg_<k><suffix>.mkv in
    seg_dir (suffix "_f": a redo run's alternate file names), a closed segment reported as 'name,start,end' lines on
    stdout. rate_mult: pipe frames per source frame (a frame doubler writes 2); in_h: the pipe frame height (stacked
    compare layers: n*H). layers (compare_stack.queue_layers, track order): one = a plain single track; several
    (processed + the encoded original) = the pipe frame is the layers stacked and compare_stack.queue_layer_args splits /
    crops it into one titled video track per layer. Processed: CFR at the pipe rate, GOP = one segment (N * rate_mult pipe
    frames). The other layers are 1x tracks: with rate_mult 2 only the real (even) pipe slots are kept, so they are encoded at
    the SOURCE frame rate, GOP = N frames, with lower quality by cq_offset (CQ / CRF increase; Apple quality decreases);
    every track gets a forced keyframe at each
    segment start (per-stream -force_key_frames: the expression's n counts that stream's own frames)."""
    layers = tuple(layers)
    multi = len(layers) > 1
    n = plan.n * rate_mult
    fps = plan.fps * rate_mult
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{in_w}x{in_h}", "-framerate", str(fps), "-i", "pipe:0"]
    meta_args: list[str] = []
    if multi:
        if in_h % len(layers):
            raise WorkerError(f"stacked height {in_h} is not a multiple of {len(layers)} layers")
        filt_args, map_args, meta_args = compare_stack.queue_layer_args(in_w, in_h // len(layers), layers, rate_mult, plan.fps)
        cmd += filt_args + map_args + ["-an"]
    else:
        cmd += ["-map", "0:v:0", "-an"]
    try:
        cmd += encoding_backend.encoder_args(encoder, purpose="segmented")
    except ValueError as e:
        raise WorkerError(str(e)) from None
    for i in range(len(layers)):
        spec = f":v:{i}" if multi else ""
        frames = n if i == 0 else plan.n          # frames of this track per segment: processed at the pipe rate, the rest 1x
        cmd += quality_args(encoder, spec, cq_offset if i else 0)
        cmd += encoding_backend.gop_args(encoder, frames, spec=spec)
    cmd += meta_args
    # cut a hair before each boundary: the forced IDR exactly on it always qualifies, nothing earlier does
    seg_time = (round(float(Fraction(n) / fps) * 1e6) - 50) / 1e6
    cmd += ["-f", "segment", "-segment_format", "matroska", "-segment_time", f"{seg_time:.6f}",
            "-reset_timestamps", "1", "-segment_start_number", str(k0),
            "-segment_list", "pipe:1", "-segment_list_type", "csv",
            os.path.join(seg_dir, f"seg_%06d{suffix}.mkv")]
    return cmd


def concat_list_text(seg_files: list[str]) -> str:
    """Concat demuxer list (absolute paths, forward slashes, quotes escaped) for the finished segments."""
    lines = []
    for p in seg_files:
        q = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{q}'")
    return "\n".join(lines) + "\n"


def build_finalize_cmd(list_path: str, source: str, has_audio: bool, video_start: float, out_path: str,
                       layers: tuple = ("processed",), copy_original: bool = False, tags: dict | None = None,
                       *, preserve_source_clock: bool = False) -> list[str]:
    """Concatenate the segments and stream-copy the source's audio next to them. The segments start at 0 and the
    restored frame 0 is the source's first video frame, so the segments are shifted by the source's video start time
    (-itsoffset on the concat input only): the audio keeps its own source timestamps and both stay in sync. layers = the
    video tracks inside the segments (compare_stack.queue_layers): every one is kept. copy_original: the source's first video
    stream is stream-copied as the last video track (oh:original), from the same source input as the audio -- no
    -itsoffset on it, the source keeps its own timeline, which is where the shifted processed track sits. With more than one
    video track the player's titles / dispositions (compare_stack.layer_meta_args) are set explicitly. tags: global
    Matroska tags (compare_stack.lada_tags)."""
    layers = tuple(layers)
    final = layers + (("original",) if copy_original else ())
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if preserve_source_clock:
        # HLS MPEGTS inputs carry their shared audio/video start clock. Without -copyts FFmpeg rebases input 1 to 0,
        # while the processed concat input is explicitly shifted by video_start, breaking A/V and compare alignment.
        cmd.append("-copyts")
    if video_start:
        cmd += ["-itsoffset", f"{video_start:.6f}"]
    cmd += ["-f", "concat", "-safe", "0", "-i", list_path]
    vmap = ["-map", "0:v" if len(layers) > 1 else "0:v:0"]
    if copy_original:
        vmap += ["-map", "1:v:0"]
    if has_audio or copy_original:
        cmd += ["-i", source] + vmap + (["-map", "1:a"] if has_audio else [])
    else:
        cmd += vmap
    cmd += ["-c", "copy"]
    if len(final) > 1:
        cmd += compare_stack.layer_meta_args(final)
    cmd += compare_stack.tag_args(tags or {})
    cmd += ["-f", "matroska", out_path]
    return cmd


def duration_ok(out_duration: float, src_duration: float, tol: float = DURATION_TOLERANCE) -> bool:
    return abs(out_duration - src_duration) <= tol


def finalized_duration_ok(probe: dict, source_duration: float, job: dict, tol: float = DURATION_TOLERANCE) -> bool:
    """Compare cached-HLS outputs by their timeline span when ffprobe reports absolute nonzero-end timestamps.

    Legacy file sources retain the existing format.duration comparison exactly. For a cached transport stream the
    output may start at the source's MPEGTS clock (for example 1.466s) and format.duration may be the absolute end
    (11.466s) although the VOD span is 10s.
    """
    out_duration = float((probe.get("format") or {}).get("duration") or 0)
    if not job.get("source_identity"):
        return duration_ok(out_duration, source_duration, tol)
    start = probe_start_time(probe)
    return duration_ok(out_duration, source_duration, tol) or duration_ok(out_duration - start, source_duration, tol)


def probe_start_time(probe: dict) -> float:
    """Earliest stream start, falling back to the container start or zero."""
    starts = []
    for stream in probe.get("streams") or []:
        try:
            if stream.get("start_time") not in (None, "N/A", ""):
                starts.append(float(stream["start_time"]))
        except (TypeError, ValueError):
            pass
    if starts:
        return min(starts)
    fmt = probe.get("format") or {}
    try:
        return float(fmt.get("start_time")) if fmt.get("start_time") not in (None, "N/A", "") else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------------------------------------------

class ProgressThrottle:
    """At most one progress line per `interval` seconds (force=True bypasses it: phase changes, the last line)."""

    def __init__(self, interval: float = PROGRESS_INTERVAL, clock=time.monotonic):
        self.interval, self.clock, self.last = interval, clock, None

    def ready(self, force: bool = False) -> bool:
        now = self.clock()
        if force or self.last is None or now - self.last >= self.interval:
            self.last = now
            return True
        return False


class RateMeter:
    """Frames per second over the last `window` seconds, and the ETA that follows."""

    def __init__(self, window: float = RATE_WINDOW, clock=time.monotonic):
        self.window, self.clock = window, clock
        self.samples: collections.deque = collections.deque()

    def tick(self, frames_done: int):
        now = self.clock()
        self.samples.append((now, frames_done))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.window:
            self.samples.popleft()

    def fps(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, f0), (t1, f1) = self.samples[0], self.samples[-1]
        return (f1 - f0) / (t1 - t0) if t1 - t0 >= 0.25 else 0.0   # too short a span says nothing

    def eta(self, frames_done: int, frames_total: int) -> float | None:
        fps = self.fps()
        return max(0.0, (frames_total - frames_done) / fps) if fps > 0 else None


THROTTLE = ProgressThrottle()          # one for the whole process: all progress lines share the rate limit


def progress_line(frame: int, frames: int, fps: float, eta: float | None, segments: int, phase: str = "restore",
                  k: int | None = None, source_fps: float | None = None, duration: float | None = None) -> dict:
    d = {"type": "progress", "frame": int(frame), "frames": int(max(frames, frame)), "fps": round(fps, 2),
         "eta": None if eta is None else round(eta, 1), "segments": int(segments), "k": None if k is None else int(k),
         "phase": phase}
    if source_fps is not None and source_fps > 0:
        d["source_fps"] = round(float(source_fps), 3)
    if duration is not None and duration > 0:
        d["duration"] = round(float(duration), 3)
    return d


# ---------------------------------------------------------------------------------------------------------------
# frame stages (plug-in point for interpolation / compare)
# ---------------------------------------------------------------------------------------------------------------

class PlainStage:
    """One output frame per restored frame. A frame doubler (rate_mult 2: a RIFE middle frame or copy before every
    frame, finish() = the closing copy) or a compare stack (layers > 1: the pipe frame is the layers stacked) replaces
    this and reports its rate_mult / layer count for build_encode_cmd."""

    rate_mult = 1
    layers = 1
    layer_names = ("processed",)

    def push(self, frame, arr, t):
        """frame: Lada's tensor, arr: its numpy view (H, W, 3 bgr), t: source time or None -> list of arrays to write."""
        return [arr]

    def finish(self):
        return []


class CompareStage:
    """Compare layers around another stage (PlainStage or a FullDoubler), as Session._pump does in the player: every
    array the inner stage would write becomes one stacked pipe frame (compare_stack.build_stacked: processed / restored /
    original back to back). The original comes from the ROI patches install_original_capture saved on the frame
    (compare_stack.rebuild_original). The 1x layers ("restored", "original") carry the real frame in the frame's own slot
    and repeat the previous frame in a doubler's middle slot (and in finish()'s closing slot), so every track has the
    same 2N frames per N source frames and the tracks stay in sync; a resumed run starts with an empty history, which is
    fine since the first frame is written alone."""

    def __init__(self, inner, names):
        self.inner, self.layer_names = inner, tuple(names)
        self.rate_mult = inner.rate_mult
        self.layers = len(self.layer_names)
        self.prev_arr = self.prev_orig = None
        self._bufs: dict = {}        # per output slot: the stacked frame is written before the next push, so reuse

    def _stack(self, slot, processed, restored1x, original):
        self._bufs[slot] = compare_stack.build_stacked(processed, restored1x, original, self.layer_names,
                                                       self._bufs.get(slot))
        return self._bufs[slot]

    def push(self, frame, arr, t, original_patches=None):
        orig = compare_stack.apply_patches(arr, original_patches)
        outs = self.inner.push(frame, arr, t)
        res = []
        for i, out in enumerate(outs):
            real = i == len(outs) - 1 or self.prev_arr is None
            res.append(self._stack(i, out, arr if real else self.prev_arr, orig if real else self.prev_orig))
        self.prev_arr, self.prev_orig = arr, orig
        return res

    def finish(self):
        return [self._stack(i, out, self.prev_arr, self.prev_orig) for i, out in enumerate(self.inner.finish())]


# ---------------------------------------------------------------------------------------------------------------
# finalise
# ---------------------------------------------------------------------------------------------------------------

def check_tracks(probe: dict, layers: tuple, tags: dict | None = None) -> str | None:
    """None when the probed output carries exactly the video tracks of `layers` (count; for compare also the titles and
    the default flag on the first one) and the global tags given, else the problem."""
    if tags:
        have = {str(k).upper(): v for k, v in (probe.get("format", {}).get("tags") or {}).items()}
        wrong = {k: have.get(k) for k, v in tags.items() if have.get(k) != v}
        if wrong:
            return f"output tags missing / different: {wrong}"
    vs = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if len(vs) != len(layers):
        return f"output has {len(vs)} video tracks, expected {len(layers)}"
    if len(layers) > 1:
        titles = [(s.get("tags") or {}).get("title") for s in vs]
        want = [compare_stack.LAYER_TITLES[n] for n in layers]
        if titles != want:
            return f"output track titles {titles}, expected {want}"
        flags = [int((s.get("disposition") or {}).get("default", 0)) for s in vs]
        if flags != [1] + [0] * (len(vs) - 1):
            return f"output default flags {flags}"
    return None


def finalize(job: dict, info: SourceInfo, plan: SegmentPlan, manifest: Manifest, work_dir: str, seg_dir: str,
             progress_cb=None, layers: tuple = ("processed",), original: str | None = None, rate_mult: int | None = None):
    """Concat segments 0..last_k in order + audio copy (+ the source's video stream as oh:original when original is
    "copy") + the global OH_* tags -> output, verify the duration and the tracks. layers: the tracks inside the segments;
    original: the effective compare mode ("copy" | "encode" | None = compare off); rate_mult: default the manifest's play
    block (else 1). work_dir is kept (the queue service deletes it). WorkerError on failure."""
    out = job["output"]
    if job.get("source_identity") is not None:
        # Do not let a complete-looking worker manifest finalize against a failed or replaced cache generation.
        read_source_status(job)
    last = manifest.end_k(plan.n)
    if last is None or not manifest.has_all(last):
        raise WorkerError("the segments are not complete")
    if manifest.partial_ks():
        raise WorkerError(f"segments {sorted(manifest.partial_ks())} are still partial (not redone with full interpolation)")
    by_k = {s["k"]: s for s in manifest.segments}
    files = [os.path.join(seg_dir, by_k[k]["file"]) for k in range(last + 1)]
    for p in files:
        if not os.path.isfile(p):
            raise WorkerError(f"segment file missing: {p}")
    if progress_cb:
        progress_cb()
    list_path = os.path.join(work_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write(concat_list_text(files))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".part.mkv"
    if rate_mult is None:
        rate_mult = int((manifest.play or {}).get("rate_mult") or 1)
    copy = original == "copy"
    tags = compare_stack.lada_tags(info.fps, rate_mult, info.video_start, original is not None, original,
                                   manifest.engine, manifest.engine_version)
    out_layers = tuple(layers) + (("original",) if copy else ())
    cmd = build_finalize_cmd(list_path, job.get("source_url") or job["source"], info.has_audio,
                             info.video_start, tmp, layers, copy, tags,
                             preserve_source_clock=job.get("source_identity") is not None)
    log("finalise: " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise WorkerError(f"ffmpeg concat failed ({p.returncode}): {p.stderr.strip()[-300:]}")
    probe = ffprobe_json(tmp)
    out_dur = float(probe["format"].get("duration") or 0)
    if not finalized_duration_ok(probe, info.duration, job):
        start = probe_start_time(probe)
        span = out_dur - start if job.get("source_identity") is not None else out_dur
        raise WorkerError(f"output duration span {span:.2f}s differs from the source's {info.duration:.2f}s")
    if info.has_audio and not any(s.get("codec_type") == "audio" for s in probe["streams"]):
        raise WorkerError("output has no audio stream")
    bad = check_tracks(probe, out_layers, tags)
    if bad:
        raise WorkerError(bad)
    replace_atomic(tmp, out)
    log(f"output {out}: {out_dur:.2f}s (source {info.duration:.2f}s)")


# ---------------------------------------------------------------------------------------------------------------
# restoring (torch + Lada)
# ---------------------------------------------------------------------------------------------------------------

def segment_speed_line(k: int, frames: int, elapsed: float, fps: float, copies: int = 0, redo: bool = False) -> str:
    """e.g. "segment 41: 120 frames in 3.9s (30.8 fps, 1.03x realtime, 37 copies)" (copies only when > 0)."""
    head = f"segment {k}{' (redone)' if redo else ''}: {frames} frames in {elapsed:.1f}s"
    if elapsed <= 0:
        return head + (f" ({copies} copies)" if copies > 0 else "")
    rate = frames / elapsed
    extra = f", {copies} copies" if copies > 0 else ""
    return head + f" ({rate:.1f} fps, {rate / fps:.2f}x realtime{extra})"


class SegmentRun:
    """One ffmpeg encoder process of a restoring run + the thread that turns its 'segment closed' lines into manifest
    entries (frame counts in SOURCE frames; fed counts pipe frames). suffix: the file name variant of this run ("_f" in
    a redo run: the closed segment replaces its manifest entry and the old file is removed best-effort). copies[k]:
    policy copies written into segment k (add_copies BEFORE the write that may close it); a segment with any is
    listed as partial. Every closed segment logs its speed."""

    def __init__(self, plan: SegmentPlan, k0: int, info: SourceInfo, encoder: str, seg_dir: str, manifest: Manifest,
                 work_dir: str, stage, suffix: str = "", clock=time.perf_counter, cq_offset: int = 0):
        self.plan, self.k0, self.manifest, self.seg_dir = plan, k0, manifest, seg_dir
        self.suffix, self.clock = suffix, clock
        self.rate_mult = stage.rate_mult
        self.fed = 0                    # frames written to ffmpeg (counted before the write: see _on_closed)
        self.natural_end = False
        self.error: str | None = None
        self.copies: dict[int, int] = {}
        self.t_mark: float | None = None    # start of the segment being produced (first write, then the last close)
        cmd = build_encode_cmd(plan, k0, info.width, info.height * stage.layers, encoder, seg_dir, stage.rate_mult,
                               getattr(stage, "layer_names", ("processed",)), suffix, cq_offset)
        log("encoder: " + " ".join(cmd))
        self.ffmpeg_log = open(os.path.join(work_dir, "ffmpeg.log"), "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.ffmpeg_log,
                                     bufsize=0, creationflags=NO_WINDOW)
        self.reader = threading.Thread(target=self._read_list, name="segment list", daemon=True)
        self.reader.start()

    def _read_list(self):
        for raw in self.proc.stdout:
            name = raw.decode("utf-8", "replace").strip().split(",")[0]
            if name:
                try:
                    self._on_closed(os.path.basename(name))
                except Exception as e:  # noqa: BLE001
                    self.error = f"segment bookkeeping failed: {e}"

    def add_copies(self, k: int, n: int = 1):
        self.copies[k] = self.copies.get(k, 0) + n

    def _on_closed(self, name: str):
        parsed = parse_segment_name(name)
        if parsed is None:
            return
        k = parsed[0]
        frames = closed_source_frames(k, self.k0, self.plan.n, self.rate_mult, self.fed, self.natural_end)
        if frames is None:
            log(f"segment {k} closed with a short / unexpected frame count: not trusted")
            return
        path = os.path.join(self.seg_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise WorkerError(f"segment {k} reported closed but its file is missing")
        copies = self.copies.get(k, 0)
        old = self.manifest.add(k, frames, partial=copies > 0, file=name)
        if old and old != name:
            try:
                os.remove(os.path.join(self.seg_dir, old))
            except OSError as e:        # a player may still read it: leftover cleanup gets it later
                log(f"segment {k}: old file {old} not removed ({e.__class__.__name__}); left for later cleanup")
        now = self.clock()
        elapsed = now - self.t_mark if self.t_mark is not None else 0.0
        self.t_mark = now
        log(segment_speed_line(k, frames, elapsed, float(self.plan.fps), copies, redo=bool(old and old != name)))
        emit({"type": "segment_done", "k": int(k), "segments": int(self.manifest.finished_count())})    # T1.3

    def write(self, arrays) -> bool:
        if self.t_mark is None:
            self.t_mark = self.clock()
        try:
            for a in arrays:
                self.fed += 1
                self.proc.stdin.write(memoryview(a.reshape(-1)))
        except (BrokenPipeError, OSError):
            return False
        return True

    def finish(self) -> int:
        """End of video: let ffmpeg close the last segment; returns its exit code."""
        self.natural_end = True
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        self.reader.join(timeout=10)
        self.proc.stdout.close()
        self.ffmpeg_log.close()
        return rc

    def kill(self):
        """Abnormal end: no trailer, so no short segment can be mistaken for a complete one."""
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.reader.join(timeout=5)
        self.proc.stdout.close()
        try:
            self.ffmpeg_log.close()
        except OSError:
            pass


def stop_restorer(session, timeout: float = 60.0):
    """Stop an engine session for good before another one starts: a clip restoration in progress is abandoned (the
    engine's cancel, as the player's Session.stop does) and the stop runs on a helper thread so a stuck worker cannot
    hang the process."""
    def stop():
        try:
            session.stop(abandon=True)
        except Exception as e:  # noqa: BLE001
            log(f"restorer stop failed: {e!r}")
    th = threading.Thread(target=stop, name="stop restorer", daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        log(f"warning: the restorer did not stop within {timeout:.0f}s")


def make_doubler(interpolator, fps: float, width: int, height: int, clock, redo: bool = False, log=log):
    """The 2x stage of a run: a redo run is always full RIFE (FullDoubler); a normal run watches the player's clock
    (WatchedDoubler: copies while the viewer is close; with a clock that is never fed it is FullDoubler)."""
    if redo or clock is None:
        return rife_interp.FullDoubler(interpolator, fps, width, height, log)
    return rife_interp.WatchedDoubler(interpolator, fps, width, height, clock, log)


def policy_copies(stage) -> int:
    """Cumulative policy copies of a stage (WatchedDoubler.copied, also behind a CompareStage); 0 for the others."""
    return getattr(getattr(stage, "inner", stage), "copied", 0)


def play_block(plan: SegmentPlan, interp_mode: str, layers: tuple, original: str | None = None) -> dict:
    """The manifest's "play" block: what a player needs to place and read the segments. layers: the video tracks really
    inside the segments; original: the effective compare mode ("copy" | "encode") or None when compare is off."""
    return {"fps": str(plan.fps), "n": plan.n, "video_start": plan.video_start,
            "rate_mult": 2 if interp_mode == "full" else 1, "layers": list(layers), "compare": original is not None,
            "original": original, "count": plan.count}


def restore_segments(job: dict, info: SourceInfo, plan: SegmentPlan, manifest: Manifest, encoder: str, work_dir: str,
                     seg_dir: str, interp_mode: str = "off", layers: tuple = ("processed",), start_k: int = 0,
                     commands=None, cq_offset: int = 0):
    """Produce every missing segment: runs over the gaps (see the module docstring), the first one at the first gap >=
    start_k, goto commands (the K values on `commands`, default: stdin) steering them; then redo runs for the partial
    segments. Returns when segments 0..last_k are all finished and none is partial; WorkerError otherwise. The models
    are loaded once, each run gets a fresh restorer and stage. layers: the tracks inside the segments (several = the
    encoded original); cq_offset: its CQ / CRF offset."""
    from oh_streamer.player import oh_player as LS            # torch + Lada; its import also points LADA_MODEL_WEIGHTS_DIR at lada\model_weights
    from oh_streamer.engines.lada.adapter import gpu_has_fp16_acceleration
    from oh_streamer.processing import device_backend
    import torch

    st = job["settings"]
    clock = rife_interp.RemoteClock() if interp_mode == "full" else None      # fed by the clock lines on stdin
    if commands is None:
        commands = start_command_reader(clock=clock)
    cached_source = job.get("source_identity") is not None
    if cached_source:
        # Decode through the per-run proxy while keeping job["source"] as the stable local identity/playlist path.
        source = LS.Source(job["source_url"], job["headers"], probe=job["source_probe"])
    else:
        # Preserve legacy Source(src, headers) call sites and monkeypatches for ordinary queue jobs.
        source = LS.Source(job["source"], job["headers"])
    device = device_backend.select_device(st.get("device", "auto"))
    fp16 = gpu_has_fp16_acceleration(device) if st["fp16"] is None else st["fp16"]
    t0 = time.perf_counter()
    set_configured_engine(st.get("engine"))     # the rest of the worker (open_reader, ...) asks for the default engine
    engine = get_engine()
    engine.load(device, ModelChoice(st["detection_model"], st["restoration_model"], bool(fp16)), st["max_clip_length"])
    log(f"models loaded in {time.perf_counter() - t0:.1f}s on {device} (det={st['detection_model']}, "
        f"res={st['restoration_model']}, fp16={fp16}, max clip length {st['max_clip_length']})")
    emit({"type": "timing", "models_s": round(time.perf_counter() - t0, 2)})     # T1.3 (queue_service logs it)
    rife = None
    if interp_mode == "full":
        if not device_backend.rife_available(device):
            raise WorkerError("RIFE is unavailable on this device; disable interpolation before processing")
        rife = rife_interp.Interpolator(device, log)
        rife.prepare(info.width, info.height)      # builds the RIFE model in the background while the first restorer starts
        log(f"interpolation full: every frame pair interpolated unless a viewer is close behind, "
            f"{2 * float(plan.fps):.3f} fps output")
    if len(layers) > 1:
        log(f"compare: video tracks {'/'.join(layers)}")
    time_base = float(source.meta.time_base)
    half = 0.5 / float(plan.fps)
    count = [plan.count]                # estimated segment count; lowered when a run finds the video shorter
    # throughput observations for the performance profile (perf_obs lines): full / off only, never with a stacked
    # compare track; a run that made policy copies (WatchedDoubler) or reads a still-downloading source is not measured
    observer = perf_observe.PerfObserver()
    observe_mode = interp_mode if len(layers) == 1 else None
    source_check = [0.0]

    def source_complete() -> bool:
        if not cached_source:
            return True
        now = time.monotonic()
        if now - source_check[0] < 5.0:
            return True
        source_check[0] = now
        try:
            with open(job["source_status_path"], "r", encoding="utf-8") as f:
                return json.load(f).get("status") == "complete"
        except (OSError, ValueError, AttributeError):
            return False

    def make_stage(redo=False):
        # cheap: the doubler only wraps the shared Interpolator (the RIFE model stays loaded), CompareStage is state-free
        stage = (make_doubler(rife, float(plan.fps), info.width, info.height, clock, redo, log) if rife
                 else PlainStage())
        return CompareStage(stage, layers) if len(layers) > 1 else stage

    def do_run(k0: int, k_end: int | None, suffix: str = "", redo: bool = False):
        """One run from segment k0 to before k_end (None: the video's end): over a gap, or (redo, suffix = the file name
        variant to write) over partial segments, always with full RIFE. Returns (kind, where, pending): ("end", the K
        the next search starts from, a goto K to prefer instead or None) or ("abort", the K of the goto that
        interrupted the run, None)."""
        restorer = engine.open(source)         # registers the source's metadata for the engine's readers
        observer.begin_run(observe_mode, info.height, float(plan.fps))
        if not restorer.observe_detections(observer.note_detection):
            observer.begin_run(None, 0, 0)      # no density without the detection counts: nothing to observe
        stage = make_stage(redo)
        run = SegmentRun(plan, k0, info, encoder, seg_dir, manifest, work_dir, stage, suffix, cq_offset=cq_offset)
        throttle, meter = THROTTLE, RateMeter()
        limit = run_frame_limit(k0, k_end, plan.n)
        expected_t = plan.source_time(k0)
        # frames of the final segments; this run adds `pushed` source frames (partial segments count as unfinished, a
        # redo run pushes the frames of the very segments it replaces)
        base_done = manifest.frames_done(partial=False)
        total = manifest.total_frames or plan.frames_total
        pushed = skipped = 0
        checked = ok = video_ended = False
        abort_k = pending = t_first = None
        last_source_t = None

        def cur_k():
            return k0 + max(0, pushed - 1) // plan.n

        src_fps = float(plan.fps) if plan.fps else None
        tot_dur = float(info.duration) if info.duration else (total / src_fps if src_fps and total else None)

        def report(force=False):
            if throttle.ready(force):
                done = base_done + pushed
                emit(progress_line(done, total, meter.fps(), meter.eta(done, total), manifest.finished_count(), "restore",
                                   cur_k(), source_fps=src_fps, duration=tot_dur))

        log(f"{'redo' if redo else 'run'}: segment {k0}" + (f" up to {k_end - 1}" if k_end is not None else " to the end") +
            f" (frame {k0 * plan.n}, {expected_t:.3f}s; {manifest.finished_count()}/{count[0]} segments finished" +
            (f", {len(manifest.partial_ks())} partial)" if manifest.partial_ks() else ")"))
        report(force=True)
        try:
            restorer.start(plan.start_ns(k0) / 1e9, capture_original=len(layers) > 1)   # keeps the ROIs for the original layer
            for elem in restorer.frames():
                while abort_k is None:          # goto commands, taken per frame
                    try:
                        k = commands.get_nowait()
                    except queue.Empty:
                        break
                    act = goto_action(k, manifest.finished(), k0, cur_k(), k_end, manifest.gap_limit(count[0]),
                                      redo=redo)
                    log(f"goto {k}: {act}")
                    if act == "next":
                        pending = k
                    elif act == "abort":
                        abort_k, pending = k, None
                if abort_k is not None:
                    break
                if isinstance(elem, (EndOfStream, EngineError)):
                    raise WorkerError(f"restorer stopped: {getattr(elem, 'reason', elem)}")
                frame, pts = elem.image, elem.pts
                t = float(pts * time_base) if pts is not None else None
                if cached_source and t is None:
                    raise WorkerError("cached source frame has no PTS")
                if not checked and t is not None:
                    if t < expected_t - half:       # the reader should have skipped these already; never write them
                        skipped += 1
                        continue
                    if t > expected_t + half and not cached_source:
                        log(f"warning: first frame at {t:.4f}s, expected {expected_t:.4f}s: segments may be misaligned")
                    checked = True
                if cached_source:
                    frame_index = k0 * plan.n + pushed
                    pts_problem = cache_pts_error((t,), plan.fps, info.video_start, frame_index)
                    if pts_problem:
                        raise WorkerError(pts_problem)
                    last_source_t = t
                arr = frame.numpy() if isinstance(frame, torch.Tensor) else frame
                copied = policy_copies(stage)
                outs = stage.push(frame, arr, t, elem.original_patches) if len(layers) > 1 else stage.push(frame, arr, t)
                if policy_copies(stage) > copied and pushed > 0:
                    # the middle slot belongs to the segment of the PREVIOUS frame; counted before the write that may
                    # close that segment
                    run.add_copies(k0 + (pushed - 1) // plan.n, policy_copies(stage) - copied)
                if policy_copies(stage) > copied:
                    observer.taint()                # interpolation mode is ambiguous from here on
                if not run.write(outs):
                    raise WorkerError("ffmpeg closed its input" + (f" ({run.error})" if run.error else ""))
                pushed += 1
                if t_first is None:
                    t_first = time.perf_counter()
                    meter.tick(base_done)
                meter.tick(base_done + pushed)
                if run.error:
                    raise WorkerError(run.error)
                if not source_complete():
                    observer.discard()
                else:
                    obs = observer.frame()
                    if obs is not None:
                        emit({**obs, "detector": st["detection_model"], "engine": st.get("engine", "lada")})
                report()
                if limit is not None and pushed >= limit:
                    break                           # the next finished segment is next: close normally, as at the video end
            else:
                video_ended = True
            if abort_k is None and video_ended and cached_source:
                try:
                    with open(job["source_status_path"], "r", encoding="utf-8") as f:
                        status = json.load(f)
                except (OSError, ValueError) as e:
                    status = {"identity": None, "generation": None, "status": "unknown", "error": str(e)}
                terminal_t = last_source_t
                if terminal_t is None and k0 > 0:
                    prior = next((s for s in manifest.segments if s["k"] == k0 - 1), None)
                    if prior is not None and prior["frames"] == plan.n:
                        terminal_t = plan.source_time(k0) - float(Fraction(1, 1) / plan.fps)
                completion_problem = source_completion_error(
                    status, job["source_identity"], job["source_generation"],
                    (terminal_t,) if terminal_t is not None else (), plan.fps, info.duration, info.video_start,
                    natural_end=video_ended, frame_count=k0 * plan.n + pushed, expected_frames=plan.frames_total)
                if completion_problem:
                    raise WorkerError(f"cached source completion failed: {completion_problem}")
            if abort_k is None:
                if not run.write(stage.finish()):
                    raise WorkerError("ffmpeg closed its input")
                ok = True
            obs = observer.finish()                 # what is left of the steady part of this run (abort or end)
            if obs is not None:
                emit({**obs, "detector": st["detection_model"], "engine": st.get("engine", "lada")})
        finally:
            if not ok:
                run.kill()                          # no trailer: the partial segment can never pass for a finished one
            stop_restorer(restorer)
        if abort_k is not None:
            removed = clean_work_dir(seg_dir, manifest)
            log(f"run aborted after {pushed} frames by goto {abort_k} ({len(removed)} partial files removed)")
            return "abort", abort_k, None
        if skipped:
            log(f"dropped {skipped} frames before the segment start")
        if run.fed != pushed * stage.rate_mult:
            raise WorkerError(f"{run.fed} frames written for {pushed} source frames and a {stage.rate_mult}x stage")
        if pushed == 0:
            # nothing at all: the video ended before this segment
            run.kill()
            if redo:
                raise WorkerError(f"redo: the video has no frames at segment {k0}")
            by_k = {s["k"]: s for s in manifest.segments}
            if k0 > 0 and by_k.get(k0 - 1, {}).get("frames") == plan.n:
                manifest.mark_complete(k0 * plan.n, k0 - 1)         # it ended exactly on a segment boundary
                return "end", 0, pending
            if k0 == 0:
                raise WorkerError("the video has no frames")
            log(f"the video ends before segment {k0} (estimated {count[0]} segments)")
            count[0] = k0
            return "end", 0, pending
        rc = run.finish()
        if rc != 0:
            raise WorkerError(f"ffmpeg exited with {rc} (see {os.path.join(work_dir, 'ffmpeg.log')})")
        if run.error:
            raise WorkerError(run.error)
        last = k0 + (pushed - 1) // plan.n
        ks = manifest.finished()
        if not all(k in ks for k in range(k0, last + 1)):
            raise WorkerError(f"the run closed segments {sorted(k for k in ks if k0 <= k <= last)}, expected {k0}..{last}")
        if redo:
            by_k = {s["k"]: s for s in manifest.segments}
            stale = [k for k in range(k0, last + 1) if by_k[k].get("partial") or file_suffix(by_k[k]["file"]) != suffix]
            if stale:
                raise WorkerError(f"the redo run did not replace segments {stale}")
        elapsed = time.perf_counter() - (t_first or time.perf_counter())
        log(f"{'redone' if redo else 'restored'} {pushed} frames in {elapsed:.1f}s ({pushed / elapsed:.1f} fps)"
            if elapsed > 0 else "restored")
        if policy_copies(stage):
            log("interpolation: " + getattr(getattr(stage, "inner", stage), "summary", lambda: "")())
        if video_ended:                             # only a natural end tells where the video stops
            if k_end is not None:
                log(f"warning: the video ended in segment {last}, before the finished segment {k_end}")
            manifest.mark_complete(k0 * plan.n + pushed, last)
            return "end", last + 1, pending
        return "end", k_end, pending

    pos = start_k
    while True:
        k0 = first_gap(manifest.finished(), pos, manifest.gap_limit(count[0]))
        if k0 is not None:
            kind, where, pending = do_run(k0, run_end_k(manifest.finished(), k0))
        else:
            again = redo_plan(list(manifest.segments), pos, manifest.end_k(plan.n))
            if again is None:
                break
            r0, r_end, suffix = again
            kind, where, pending = do_run(r0, r_end, suffix, redo=True)
        pos = where if kind == "abort" or pending is None else pending
    log(f"all segments finished ({len(manifest.segments)} segments, {manifest.frames_done()} frames)")


def run_job(job: dict, notes: list[str]) -> str:
    for n in notes:
        log("note: " + n)
    src = job["source"]
    if not os.path.isfile(src):
        raise WorkerError(f"source not found: {src}")
    from oh_streamer.processing import device_backend

    # Lada's ModelFiles registry is initialized at its first import, before LS.
    project_root = upstream_root()
    os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", str(project_root / "lada" / "model_weights"))
    # (importing the adapter imports Lada, so it must stay after the LADA_MODEL_WEIGHTS_DIR line above)
    from oh_streamer.engines.lada.adapter import upstream_gpu_has_fp16_acceleration as gpu_has_fp16_acceleration
    settings = dict(job["settings"])
    device = device_backend.select_device(settings.get("device", "auto"))
    # New explicit choices and Mac jobs record actual precision in their manifest.
    # Legacy Windows auto jobs retain their key so existing resumes remain valid.
    if settings.get("device", "auto") != "auto" or sys.platform == "darwin":
        settings["device"] = str(device)
        if settings["fp16"] is None:
            settings["fp16"] = device_backend.fp16_default(device, gpu_has_fp16_acceleration)
    if not device_backend.rife_available(device) and settings.get("interpolation") != "off":
        settings["interpolation"] = "off"
        log("RIFE is unavailable on this device; using original frame rate")
    job = {**job, "settings": settings}
    work_dir = os.path.abspath(job["work_dir"])
    seg_dir = os.path.join(work_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    info = parse_probe(job["source_probe"] if job.get("source_identity") is not None else ffprobe_json(src))
    plan = SegmentPlan(info.fps, info.frames, info.video_start, job["segment_seconds"])
    encoder = resolve_encoder(job["settings"]["encoder"], encoder_usable)
    interp_mode, why = effective_interpolation(job["settings"]["interpolation"], info.fps)
    if job["settings"]["interpolation"] == "full" and interp_mode == "off":
        log(f"interpolation full: staying at 1x ({why})")
    cp = compare_plan(job["settings"], info, encoder)
    for n in cp.notes:
        log(n)
    if cp.on:
        log(f"compare: oh:original = " + ("the source video stream, copied at finalize" if cp.original == "copy" else
            f"encoded at the source frame rate, quality {job['settings'].get('compare_quality', 'small')}"))
    layers = cp.layers
    key = job_key(job, info, plan, encoder)
    manifest = Manifest.load(os.path.join(work_dir, "segments.json"), key, seg_dir)
    manifest.engine, manifest.engine_version = engine_stamp(job["settings"].get("engine", "lada"))
    removed = clean_work_dir(seg_dir, manifest)
    log(f"{info.width}x{info.height} @ {float(info.fps):.3f} fps, {info.frames} frames, {info.duration:.1f}s, "
        f"audio={'yes' if info.has_audio else 'no'}; {plan.n} frames/segment, ~{plan.count} segments; encoder {encoder}")
    log(f"resume: {len(manifest.segments)} segments kept ({manifest.frames_done()} frames, first gap {manifest.next_k}), "
        f"{len(removed)} leftover files removed"
        + (f", {len(manifest.partial_ks())} partial (to redo)" if manifest.partial_ks() else "")
        + (", complete" if manifest.is_complete(plan.n) else ""))
    manifest.play = play_block(plan, interp_mode, layers, cp.original)
    manifest.save()
    if not manifest.is_complete(plan.n):
        restore_segments(job, info, plan, manifest, encoder, work_dir, seg_dir, interp_mode, layers, job["start_k"],
                         cq_offset=compare_stack.quality_offset(job["settings"].get("compare_quality", "small")))
    while not THROTTLE.ready():     # keep the two-lines-per-second limit across the phase change
        time.sleep(0.05)
    done = manifest.frames_done(partial=False)
    fin_fps = float(plan.fps) if plan.fps else None
    fin_dur = float(info.duration) if info.duration else (done / fin_fps if fin_fps and done else None)
    emit(progress_line(done, done, 0.0, 0.0, len(manifest.segments), "finalize", source_fps=fin_fps, duration=fin_dur))
    finalize(job, info, plan, manifest, work_dir, seg_dir, layers=layers, original=cp.original,
             rate_mult=2 if interp_mode == "full" else 1)
    return job["output"]


def main(argv=None) -> int:
    global _OUT
    from oh_streamer.runtime_env.app_running import hold_app_mutex
    hold_app_mutex()
    ap = argparse.ArgumentParser(description="Restore one local video with Lada into resumable segments.")
    ap.add_argument("--job", required=True, help="job JSON file (see the module docstring)")
    args = ap.parse_args(argv)
    _OUT = sys.stdout
    sys.stdout = sys.stderr                  # library prints must not corrupt the JSON lines
    try:
        with open(args.job, "r", encoding="utf-8") as f:
            job, notes = normalize_job(json.load(f))
        output = run_job(job, notes)
    except KeyboardInterrupt:
        emit({"type": "error", "message": "interrupted"})
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        emit({"type": "error", "message": str(e) if isinstance(e, WorkerError) else f"{type(e).__name__}: {e}"})
        return 1
    emit({"type": "done", "output": output})
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stderr.flush()
    _OUT.flush()
    os._exit(rc)       # Lada's worker threads must not keep a finished (or failed) worker alive
