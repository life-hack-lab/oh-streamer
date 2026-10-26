"""
oh_player: play a local file or a stream URL through Lada's mosaic restoration in (near) real time.

Pipeline:
  input (file / m3u8 / mp4 URL, or page URL resolved by yt-dlp)
    -> Lada FrameRestorer (detection + restoration, reads the input itself via PyAV)
    -> raw BGR frames -> ffmpeg (NVENC + original audio) -> Matroska over local TCP -> mpv

Seeking (--output play): mpv stays open; oh_seek.lua seeks natively inside mpv's cache and otherwise asks us
(via mpv JSON IPC) to restart Lada at the target. Every restart is a new "segment": a fresh ffmpeg muxer listening
on its own port with absolute timestamps, loaded into mpv with loadfile. While the user is not paused, a far seek
keeps the old timeline playing meanwhile (seek_tail.py: the old segment's muxer is taken over, mpv plays the restored
video it has cached and then unrestored passthrough frames; the new segment is appended behind it once its output is
a little past the target, so mpv moves on by itself). Paused, quality / settings changes: playback freezes and the new
segment replaces the old one when ready. Every stop is fast (an in-progress clip restoration is abandoned).

Advanced settings ('s' / gear button; player_settings.py) restart the segment at the current position, reloading the
models first when the detection/restoration model or FP16 changes; choices (and volume / mute) are saved to
oh_settings.json in the platform user data directory, and flags given on the command line override that file. The 進階資訊 panel ('i' / info
button) is fed from here as user-data/oh/info about once a second while it is visible (info_panel.py).

Run with the Lada venv from anywhere, e.g.:
  lada\\.venv\\Scripts\\python.exe streamer\\oh_player.py <input> [options]

Persistent player (--player; streamer\\Oh Player.bat starts it with pythonw, logging to user data logs/): mpv opens
idle with a hint, the models load in the background once, and sources arrive only from the main app (queue) over the
single-instance IPC; the player cannot open videos by itself (no paste, no drag & drop: dropped files / links are
intercepted by oh_seek.lua and discarded with an OSD hint). `--player <input>` is kept for development only.
Opening while playing resolves the new input first and only then replaces the current source (models stay loaded);
on failure the current video keeps playing. Only one player runs per instance name: `--player <input>` while one is
running forwards the input to it over 127.0.0.1 and exits (single_instance.py).

Background queue (queue_ipc.py; the queue app is a separate program): the player tells it when a live restore runs
(player_busy, repeated every BUSY_HEARTBEAT s), asks it on every open whether the input is processed already (lookup;
a finished output plays as is), and the 'a' key adds the playing source (add). Processed mode (a single-instance
message with "processed": true, or --player <path> --processed) is plain mpv playback of a finished file: no Session,
no restorer (ProcessedOpen / switch_to_processed in run_player); at 90 % or the end the queue gets "watched".
Progressive mode (a message with "progressive": true + "work_dir", for an item the queue is still processing;
progressive_stream.py): no Session / Lada / GPU either (the worker has the GPU, so no player_busy): a feeder remuxes the
worker's finished segments (all video tracks, no re-encoding) + the source's audio into one Matroska stream with
absolute timestamps over local TCP, like a live segment, so oh_seek.lua's seek / served / progress protocol is the live
one; a seek outside mpv's cache restarts the feeder at the target's segment. A segment not processed yet is waited for
with the stream open: the queue gets {"cmd": "want", "k", "pos"} (position included when known; at once, every 5 s), the progress / status says 處理中.
user-data/oh/ranges = the processed parts ("a-b,c-d", seconds) for the seekbar; "watching" every BUSY_HEARTBEAT s,
"watched" at 90 % / the end (switch_to_progressive / progressive_requests / progressive_tick in run_player).
All queue traffic runs on a sender thread (QueueLink) and is dropped quietly when the queue app is not running.

Frame interpolation (setting / --interpolation auto; rife_interp.py): the output runs on a 2x frame-rate grid; the
pump writes a RIFE middle frame between two restored frames while the output is comfortably ahead of mpv's playback
position (sampled here by the controller thread) and the restoration ahead allows it, and a copy of the previous
frame otherwise (dropped again by the muxer: variable frame rate). Casting stays at 1x.

Casting (cast button / 'c' in the player, --cast <Apple TV>, --hls-only): hls_cast.py serves the restored video on
the LAN as an on-demand HLS VOD (full-length playlist, segments restored + encoded when requested, restarts at the
exact segment start on seeks) under a random token path; airplay_cast.py pushes its URL to an Apple TV with pyatv
(pair once with --airplay-pair; credentials in the platform user data directory, a local secret). While casting the
local session is stopped (one producer, the HLS runs use its restorer). Windows Firewall asks once whether python
may accept connections on private networks: allow it, or the Apple TV cannot fetch the stream.
"""

import argparse
import collections
import gc
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from oh_streamer.branding import APP_NAME
from oh_streamer.paths import resource_root, upstream_root

PROJECT_ROOT = upstream_root()
LADA_DIR = PROJECT_ROOT / "lada"

# Lada resolves model weights relative to the CWD unless this is set; must happen before importing lada.
os.environ.setdefault("LADA_MODEL_WEIGHTS_DIR", str(LADA_DIR / "model_weights"))

from oh_streamer.engines import ENGINE_NAMES  # noqa: E402  (pure python: names of the restoration engines, for --engine)
from oh_streamer.platform import airplay_cast  # noqa: E402  (streamer/airplay_cast.py: AirPlay pairing / play_url via pyatv, lazy import)
from oh_streamer.player import buffer_progress  # noqa: E402  (streamer/buffer_progress.py: buffering percentage of a start / seek)
from oh_streamer.processing import detect_gate  # noqa: E402  (streamer/detect_gate.py: holds the detector back while a start / seek waits for its first clip)
from oh_streamer.platform import hls_cast  # noqa: E402  (streamer/hls_cast.py: on-demand HLS VOD of the restored video for casting)
from oh_streamer.player import open_input  # noqa: E402  (streamer/open_input.py: clipboard / dropped / forwarded input parsing)
from oh_streamer.player import player_settings as PS  # noqa: E402  (streamer/player_settings.py: settings file, CLI overrides, menu)
from oh_streamer.player import progressive_stream  # noqa: E402  (streamer/progressive_stream.py: watch a queue item while it is processed)
from oh_streamer.queue import queue_ipc  # noqa: E402  (streamer/queue_ipc.py: messages to / from the background processing queue app)
from oh_streamer.sources import qualities  # noqa: E402  (streamer/qualities.py: quality list building, no heavy imports)
from oh_streamer.sources import source_resolve  # noqa: E402  (streamer/source_resolve.py: yt-dlp / external resolver / headers helpers, no torch)
from oh_streamer.sources.source_resolve import (  # noqa: E402,F401  (names stay importable from oh_player)
    DEFAULT_RESOLVERS_FILE, RESOLVERS_FILE, FFmpegHttpOptions, is_url, looks_like_media_url, parse_headers,
    pick_thumbnail_format, resolve_with_ytdlp,
)
from oh_streamer.processing import compare_stack  # noqa: E402  (streamer/compare_stack.py: original-vs-restored tracks)
from oh_streamer.processing.frame_rate import planned_fps, selected_frame
from dataclasses import replace
from oh_streamer.processing import rife_interp  # noqa: E402  (streamer/rife_interp.py: adaptive RIFE 2x interpolation; torch imported lazily)
from oh_streamer.player import seek_tail  # noqa: E402  (streamer/seek_tail.py: the old timeline keeps playing while a far seek is prepared)
from oh_streamer.player import single_instance  # noqa: E402  (streamer/single_instance.py: --player forwarding over 127.0.0.1)
from oh_streamer.platform import platform_support  # noqa: E402
from oh_streamer.platform import player_branding  # noqa: E402
from oh_streamer.player import player_runtime  # noqa: E402
from oh_streamer.processing import device_backend  # noqa: E402
from oh_streamer.processing import encoding_backend  # noqa: E402
from oh_streamer.player.mpv_ipc import MpvEndpoint, MpvError, MpvIpc  # noqa: E402
LOG_DIR = platform_support.data_dir() / "logs"
DEFAULT_PLAYER_LOG = LOG_DIR / "oh_player.log"
NO_CONSOLE = (sys.stderr is None or sys.stdout is None    # pythonw: no console, std streams are None (or DEVNULL
              or os.path.basename(sys.executable).lower() == "pythonw.exe")   # when the queue app starts us)
LOG_TO_FILE = False                                     # stdout / stderr go to a --log-file


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Play a video/stream through Lada mosaic restoration.")
    ap.add_argument("input", nargs="?", default=None,
                    help="Local file, direct media URL (m3u8/mp4), or page URL (resolved via yt-dlp). With --player: "
                         "optional; also a URL + header lines or a 'Copy as cURL' command")
    ap.add_argument("--player", action="store_true",
                    help="Persistent player: stays open, models load once; videos are opened from the main app. "
                         "(Development: with an input, a running player gets it forwarded and this exits)")
    ap.add_argument("--processed", action="store_true",
                    help="With --player <path>: the file is already processed (a queue output): play it as is, without "
                         "Lada (how the queue app launches the player; also forwarded to a running player)")
    ap.add_argument("--queue-id", default=None, metavar="ID", help=argparse.SUPPRESS)   # queue item id of --processed
    ap.add_argument("--instance", default=None, metavar="NAME",
                    help="Single-instance identity for --player (default 'default', or env OH_STREAMER_PLAYER_INSTANCE)")
    ap.add_argument("--log-file", default=None, metavar="PATH",
                    help="Log to this file instead of the console (the previous run's log is kept as *.prev.log). "
                         f"Default without a console (pythonw): {DEFAULT_PLAYER_LOG}")
    ap.add_argument("--header", action="append", default=[], help="Extra HTTP header 'Name: value' (repeatable)")
    ap.add_argument("--ytdl", action="store_true", help="Force yt-dlp resolution even for direct media URLs")
    ap.add_argument("--max-height", type=int, default=None,
                    help="Cap on the resolution picked by yt-dlp (default 1080 unless --quality says otherwise)")
    ap.add_argument("--quality", default=None, metavar="HEIGHT|best|worst",
                    help="Starting video quality for yt-dlp sources: a height (720 = best format up to 720p), 'best' or "
                         "'worst'. Default: best up to --max-height (1080). Switchable in the player (button / 'r' key)")
    ap.add_argument("--list-qualities", action="store_true",
                    help="Print the selectable qualities of the URL (and the one that would start) and exit")
    # offline test source; with --player it stands in for yt-dlp for every http(s) input (no network)
    ap.add_argument("--fake-formats", default=None, metavar="JSON", help=argparse.SUPPRESS)
    # tests: --output <file> / null also logs the stalls of a simulated viewer without interpolation (run_batch)
    ap.add_argument("--simulate-player", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--start", type=float, default=0.0, help="Start position in seconds")
    ap.add_argument("--prebuffer", type=float, default=5.0,
                    help="Seconds of cache mpv wants ahead when it has to rebuffer during normal playback")
    ap.add_argument("--seek-prebuffer", type=float, default=None,
                    help="Smaller startup buffer used right after a seek (restored to --prebuffer once playing; "
                         "default 1, or the saved setting)")
    ap.add_argument("--initial-prebuffer", type=float, default=None,
                    help="Startup buffer for the very first segment (default: same as --prebuffer)")
    ap.add_argument("--output", default="play",
                    help="'play' (mpv, seekable, default), 'null' (benchmark, discard frames), or a file path (.mkv/.mp4)")
    ap.add_argument("--duration", type=float, default=0,
                    help="Stop after this many seconds of video (0 = all; per segment when playing)")
    # The next eight are also in the player's settings menu ('s'). Their defaults are None so an explicitly given flag
    # can be told apart: defaults < user oh_settings.json (saved menu choices) < flags on the command line.
    ap.add_argument("--max-clip-length", type=int, default=None,
                    help=f"Max frames per restoration clip (default {PS.DEFAULTS['max_clip_length']})")
    ap.add_argument("--detection-model", default=None, help=f"default {PS.DEFAULTS['detection_model']}")
    ap.add_argument("--restoration-model", default=None, help=f"default {PS.DEFAULTS['restoration_model']}")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                    help="default: on if the GPU accelerates FP16")
    ap.add_argument("--device", default="auto", type=device_backend.validate_device,
                    help="Inference device: auto, cpu, mps or cuda:N (session only; auto detects CUDA/MPS)")
    ap.add_argument("--encoder", default=None, choices=PS.ENCODERS,
                    help="auto = usable platform hardware encoder (NVENC/AMD AMF/VideoToolbox), else libx264")
    ap.add_argument("--vsr", default=None, choices=PS.ON_OFF,
                    help="NVIDIA RTX Video Super Resolution in mpv: upscale to the display resolution (default on; "
                         "needs an RTX GPU and d3d11va decoding, and RTX Video Enhancement enabled in the NVIDIA App)")
    ap.add_argument("--smooth-playback", default=None, choices=PS.ON_OFF,
                    help="Pace playback to the display refresh rate (mpv video-sync=display-resample + interpolation "
                         "with tscale=oversample; not motion interpolation). Default on")
    ap.add_argument("--restoration-fps", default=None, choices=PS.RESTORATION_FPS,
                    help="Frame rate before detection/restoration: original or capped at 30 fps")
    ap.add_argument("--interpolation", default=None, choices=PS.INTERPOLATION,
                    help="auto = output at 2x the restoration frame rate: RIFE middle frames while the restoration is "
                         "comfortably ahead of playback, copies of the previous frame otherwise (never stalls; "
                         "restoration rates above 40 fps stay 1x). Default off")
    ap.add_argument("--compare", default=None, choices=PS.COMPARE,
                    help="on = also encode the original (unrestored) frames, and with interpolation the 1x restored "
                         "frames, as extra tracks of the stream (oh:original / oh:restored; the player's compare "
                         "key switches between them). Costs one extra NVENC encode per track. Default off")
    ap.add_argument("--detect-gate", action=argparse.BooleanOptionalAction, default=True,
                    help="While a start / seek waits for its first clip, keep the detector at most one clip length "
                         "ahead of the output so the restoration gets the GPU (first frames up to ~2x sooner; "
                         "default on)")
    ap.add_argument("--engine", default=None, choices=ENGINE_NAMES, help=argparse.SUPPRESS)   # developer flag; env OH_STREAMER_ENGINE
    ap.add_argument("--settings-file", default=str(PS.DEFAULT_FILE),
                    help="Where the settings chosen in the player are saved (default in the platform user data directory)")
    ap.add_argument("--mpv", default=None, help="Path to the mpv executable")
    ap.add_argument("--thumbnails", action=argparse.BooleanOptionalAction, default=True,
                    help="Seekbar preview thumbnails (local original source; AirPlay prefers available restored frames; default on)")
    ap.add_argument("--thumb-interval", type=float, default=10.0,
                    help="Background-prefetch a thumbnail every N seconds across the whole video once playing "
                         "(0 = no prefetch, thumbnails are made on hover only; default 10)")
    ap.add_argument("--thumb-width", type=int, default=320, help="Thumbnail bitmap width in pixels (default 320)")
    # casting (hls_cast.py + airplay_cast.py); in the player also from the cast button / 'c' key
    cg = ap.add_argument_group("casting (Apple TV / LAN HLS)")
    cg.add_argument("--airplay-scan", action="store_true", help="List AirPlay devices on the network and exit")
    cg.add_argument("--airplay-pair", default=None, metavar="NAME|ID",
                    help="Pair with an Apple TV (it shows a PIN to type here) and exit; credentials are stored in "
                         "the platform user data directory (a local secret)")
    cg.add_argument("--cast", default=None, metavar="NAME|ID",
                    help="Cast the restored video to this paired Apple TV as soon as it starts")
    cg.add_argument("--hls-only", action="store_true",
                    help="Only serve the restored video as HLS on the LAN and print the URL (VLC / Safari), no AirPlay")
    cg.add_argument("--cast-bind", default=None, metavar="IP",
                    help="Address the HLS server binds to (default: this PC's LAN address, the default-route one)")
    cg.add_argument("--cast-port", type=int, default=0, help="HLS server port (default: any free port)")
    cg.add_argument("--cast-cache-gb", type=float, default=5.0,
                    help="Disk cache cap for HLS segments in GB (temp dir, deleted when casting ends; default 5)")
    cg.add_argument("--cast-playlist", choices=("master", "media"), default="master",
                    help="Which playlist the Apple TV gets: master.m3u8 (default) or the media playlist index.m3u8")
    return ap


def run_airplay_cli(args) -> int:
    """--airplay-scan / --airplay-pair: terminal only, no torch / models."""
    import asyncio
    from oh_streamer.platform import airplay_cast as AC

    store = AC.CredentialStore()
    loop = asyncio.new_event_loop()
    try:
        if args.airplay_scan:
            mod = AC._pyatv()
            print("搜尋 AirPlay 裝置中…", flush=True)
            confs = loop.run_until_complete(AC.scan_configs(loop, mod, AC.SCAN_TIMEOUT))
            devs = AC.merged_devices([AC.device_from_config(c, mod, store) for c in confs], store)
            if not devs:
                print("找不到 AirPlay 裝置（Apple TV 是否開機、與這台電腦在同一個網路？）")
            for d in devs:
                state = ("已配對" if d.paired else "未配對") + ("" if d.found else "，這次沒有找到")
                print(f"  {d.name:<24} {d.id:<20} {d.address or '-':<16} {d.model or '':<22} {state}")
            if devs and not any(d.paired for d in devs):
                print("配對：oh_player.py --airplay-pair \"<名稱或 ID>\"")
        else:
            loop.run_until_complete(
                AC.pair_device(loop, args.airplay_pair, store, input, lambda s: print(s, flush=True)))
        return 0
    except AC.CastError as e:
        print(f"錯誤：{e}")
        return 1
    except KeyboardInterrupt:
        print("已取消")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"失敗：{AC.short_reason(e)}")
        return 1
    finally:
        loop.close()


def _open_log_file(path: Path, keep_previous: bool = True):
    """Send this process's stdout / stderr (our log, Lada's logging, tracebacks) to `path`, truncated per run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if keep_previous and path.exists():
        try:
            os.replace(path, path.with_suffix(".prev" + path.suffix))
        except OSError:
            pass
    f = open(path, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = sys.stderr = f
    global LOG_TO_FILE
    LOG_TO_FILE = True
    return f


def _hide_child_consoles():
    """Without a console every console program we start (ffmpeg, ffprobe, nvidia-smi, ...) would pop up its own
    window: default them to CREATE_NO_WINDOW."""
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    orig_init = subprocess.Popen.__init__

    def init(self, *a, **kw):
        if "creationflags" not in kw and len(a) < 14:   # (creationflags is Popen's 14th positional parameter)
            kw["creationflags"] = flag
        orig_init(self, *a, **kw)

    subprocess.Popen.__init__ = init


def _forward_to_running(args, name: str) -> int:
    """`--player [input]` while a player with this instance name runs: hand the input over and exit."""
    if NO_CONSOLE:
        _open_log_file(LOG_DIR / "oh_player_forward.log", keep_previous=False)
    payload = {"cmd": "show"}
    if args.input:
        inp = args.input
        if not open_input.is_http_url(inp.strip().strip('"')) and os.path.exists(inp.strip().strip('"')):
            inp = os.path.abspath(inp.strip().strip('"'))   # the running player has another working directory
        payload = {"cmd": "open", "input": inp, "headers": parse_headers(args.header)}
        if args.processed:
            payload.update(processed=True, queue_id=args.queue_id)
    reply = single_instance.forward(name, payload, log=log)
    log(f"forwarded to the running player '{name}': {payload['cmd']} -> {reply}")
    return 0 if reply and reply.get("ok") else 1


def _early_main():
    """Runs before torch / lada are imported: forwarding to a running player stays fast, and without a console the
    log file is in place before anything can write to the (missing) stderr."""
    args = build_arg_parser().parse_args()
    args.instance_obj = None
    if args.airplay_scan or args.airplay_pair:
        sys.exit(run_airplay_cli(args))
    if args.player:
        name = single_instance.instance_name(args.instance)
        inst = single_instance.acquire(name)
        if inst is None:
            sys.exit(_forward_to_running(args, name))
        args.instance_obj = inst
    if args.log_file or NO_CONSOLE:
        _open_log_file(Path(args.log_file) if args.log_file else DEFAULT_PLAYER_LOG)
    if NO_CONSOLE:
        _hide_child_consoles()
    return args


_T0 = time.perf_counter()    # module import time: log lines show the wall clock and the seconds since then


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} +{time.perf_counter() - _T0:6.2f}s [oh_player] {msg}", file=sys.stderr, flush=True)


source_resolve.log = lambda msg: log(msg)    # late-bound: source_resolve's messages go through oh_player's (patchable) log


EARLY_ARGS = _early_main() if __name__ == "__main__" else None

log("startup: early init done")
import av  # noqa: E402
import torch  # noqa: E402
log("startup: av + torch imported")

from oh_streamer.player import info_panel  # noqa: E402  (streamer/info_panel.py: 進階資訊 panel content + GPU monitor)
# Everything that touches lada lives in engines/lada/. Importing it initializes Lada's ModelFiles registry, so
# LADA_MODEL_WEIGHTS_DIR (set above) must already be in place.
from oh_streamer.engines import (default_engine_name, effective_engine_name, engine_label, explicit_engine_name,  # noqa: E402
                                   get_engine, set_configured_engine, set_default_engine,
                                   visible_engine_names)
from oh_streamer.engines.base import ModelChoice  # noqa: E402
from oh_streamer.engines.lada import io_hooks as _lada_io_hooks  # noqa: E402
from oh_streamer.engines.lada.engine import LadaEngine, LadaSession, open_reader  # noqa: E402,F401
from oh_streamer.engines.lada.adapter import gpu_has_fp16_acceleration  # noqa: E402
from oh_streamer.engines.lada.io_hooks import (  # noqa: E402
    VideoMetadata, build_metadata, forget_sources, install_lada_io_hooks, register_source,
)
_lada_io_hooks.log = lambda msg: log(msg)    # late-bound: the I/O hooks' messages go through oh_player's (patchable) log
log("startup: lada engine imported")
from oh_streamer.player import controller as _controller  # noqa: E402
from oh_streamer.player import pipeline as _pipeline  # noqa: E402
from oh_streamer.player import opening as _opening  # noqa: E402
from oh_streamer.player.thumbnails import ThumbnailWorker as _ThumbnailWorker  # noqa: E402
from oh_streamer.player.queue_bridge import (  # noqa: E402,F401 - compatibility exports
    PROCESSED_WATCHED_FRACTION, QUEUE_SETTING_KEYS, QueueLink, BusyReporter, ProcessedOpen,
    mpv_playback_snapshot, _probe_titles, compare_titles_from_probe, lada_tags_from_probe,
    compare_setup_from_probe, probe_compare_setup, probe_compare_titles, processed_open,
    processed_from_message, subtitle_update_from_message, queue_lookup, processed_watched, queue_quality,
    build_add_payload,
)

# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def find_resolver(s: str) -> dict | None:
    return source_resolve.find_resolver(s, RESOLVERS_FILE, DEFAULT_RESOLVERS_FILE)


def resolve_external(page_url: str, headers: dict, resolver: dict) -> tuple[str, dict, str]:
    return source_resolve.resolve_external(page_url, headers, resolver, RESOLVERS_FILE)


# ---------------------------------------------------------------------------
# Probing (replacement for lada's get_video_meta_data that works with URLs + headers)
# ---------------------------------------------------------------------------

def ffprobe(path: str, http: FFmpegHttpOptions) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format"]
    if is_url(path):
        cmd += http.cli_args()
    cmd.append(path)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({p.returncode}): {p.stderr.strip()}")
    return json.loads(p.stdout)


class Source:
    """One probed rendition: URL + headers + metadata (what FrameRestorer, the muxer and the hooks need)."""

    def __init__(self, src: str, headers: dict, audio_src: str | None = None, *, probe: dict | None = None):
        # audio_src: separate audio-only URL (sites without muxed formats); None = the audio is in `src`
        self.src, self.headers = src, dict(headers)
        self.audio_src = audio_src or src
        self.http = FFmpegHttpOptions(self.headers)
        probe = ffprobe(src, self.http) if probe is None else probe
        vs = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
        fmt_name = str(probe.get("format", {}).get("format_name") or "")
        # ffmpeg happily "plays" text files (tty demuxer, ansi codec) and still images: not something to restore
        if vs is None or vs.get("codec_name") == "ansi" or fmt_name == "tty" or fmt_name.startswith("image2") \
                or fmt_name.endswith("_pipe"):
            raise open_input.OpenInputError("不是可播放的影片" + (f"（{fmt_name}）" if fmt_name else ""))
        self.meta = build_metadata(src, probe)
        self.native_meta = self.meta
        if audio_src:
            aprobe = ffprobe(audio_src, self.http)
            self.has_audio = any(s.get("codec_type") == "audio" for s in aprobe["streams"])
        else:
            self.has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])
        self.fmt_start = float(probe["format"].get("start_time") or 0)

    def describe(self) -> str:
        m = self.meta
        return (f"{m.video_width}x{m.video_height} @ {float(m.video_fps_exact):.3f}fps, {m.codec_name}, "
                f"{m.duration:.0f}s, audio={'yes' if self.has_audio else 'no'}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def find_mpv(explicit: str | None) -> str:
    return player_runtime.find_mpv(explicit, PROJECT_ROOT)


def nvenc_available() -> bool:
    # FFmpeg 9's h264_nvenc needs NVIDIA driver >= 610 (NVENC API 13.1); a tiny test encode is the reliable check
    return encoding_backend.encoder_usable("h264_nvenc")


def start_muxer(src: str, meta: VideoMetadata, has_audio: bool, audio_ss: float,
                http: FFmpegHttpOptions, out_target: str, encoder: str, ts_offset: float | None = None,
                rate_mult: int = 1, layers: tuple = ("processed",)) -> subprocess.Popen:
    """rate_mult 2: the pipe carries two frames per source frame (interpolation), at twice the source frame rate.
    layers (compare_stack.layer_names): several = the pipe frame is the layers stacked vertically (W x n*H) and the
    output carries one titled video track per layer; one layer = exactly the single-track command below."""
    w, h = meta.video_width, meta.video_height
    layers = tuple(layers)
    multi = len(layers) > 1
    if multi:
        filt_args, map_args, codec_args, meta_args = compare_stack.ffmpeg_layer_args(w, h, layers, rate_mult, encoder)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "x".join(map(str, compare_stack.stacked_shape(w, h, layers))),
           "-framerate", str(Fraction(meta.video_fps_exact) * rate_mult), "-i", "pipe:0"]
    if has_audio:
        if is_url(src):
            cmd += http.cli_args()
        if audio_ss > 0:
            cmd += ["-ss", f"{audio_ss:.6f}"]
        # -shortest: end with the restored video instead of muxing the rest of the source audio. Its sync queue may
        # hold up to -shortest_buf_duration of output back (default 10 s): after Lada's first burst of frames the
        # segment then reached mpv ~10.6 s after it was written (measured), i.e. that much less buffer in mpv and
        # stalls while the frames already existed. 0.5 s keeps it at ~1 s (both streams are dense).
        cmd += ["-vn", "-i", src] + (map_args if multi else ["-map", "0:v:0"]) \
            + ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest", "-shortest_buf_duration", "0.5"]
    else:
        cmd += map_args if multi else ["-map", "0:v:0"]

    if multi:
        cmd += filt_args
    elif rate_mult > 1:
        # where the pump did not interpolate it wrote an exact copy of the previous frame: drop those (variable frame
        # rate output; mpv keeps display sync across 30 <-> 60 fps), so the encoder and mpv's decoding / RTX VSR only
        # work on real frames and copying costs about what 1x output does
        cmd += ["-vf", "format=yuv420p,mpdecimate=hi=0:lo=0:frac=0", "-fps_mode", "vfr"]
    if multi:
        cmd += codec_args + meta_args
    else:
        cmd += encoding_backend.encoder_args(encoder)

    if ts_offset is not None:
        # both inputs start at 0 (audio after -ss), so this makes video and audio absolute source time
        cmd += ["-output_ts_offset", f"{ts_offset:.6f}"]
    if out_target.startswith("tcp://"):
        cmd += ["-f", "matroska"]
    cmd += [out_target]
    # with a log file, ffmpeg's errors go there too (without a console they would be lost)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=0,
                            stderr=sys.stderr if LOG_TO_FILE else None)


def stop_process(p: subprocess.Popen | None):
    if p is None:
        return
    if p.poll() is None:
        p.kill()
    p.wait()
    try:
        p.stdin.close()
    except (OSError, AttributeError):
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_listening(p: subprocess.Popen, port: int) -> bool:
    """ffmpeg only opens (listens on) its TCP output after the first frame; connecting to probe would eat the accept."""
    try:
        import psutil
    except ImportError:
        return p.poll() is None  # best effort; loadfile may then race the listen
    try:
        return any(c.status == psutil.CONN_LISTEN and c.laddr.port == port
                   for c in psutil.Process(p.pid).net_connections("tcp"))
    except psutil.Error:
        return False


def primary_display_height() -> int | None:
    """Physical pixel height of the primary monitor (DESKTOPVERTRES ignores DPI scaling), or None. Only a fallback for
    oh_seek.lua's RTX VSR scale when mpv does not know display-height."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        user32.GetDC.restype = ctypes.c_void_p
        hdc = user32.GetDC(None)
        if not hdc:
            return None
        try:
            gdi32.GetDeviceCaps.argtypes = [ctypes.c_void_p, ctypes.c_int]
            h = gdi32.GetDeviceCaps(hdc, 117)   # DESKTOPVERTRES
        finally:
            user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.ReleaseDC(None, hdc)
        return h if h > 0 else None
    except (OSError, AttributeError):
        return None


def start_mpv(mpv: str, ipc_path: str, title: str, prebuffer: float, idle_hint: bool = False,
              vsr: str = "off", smooth: str = "off", compare: str = "off") -> subprocess.Popen:
    # --osc=no: the stock OSC can't show a seekbar for a live matroska stream; oh_seek.lua draws our own
    # --ytdl=no: mpv must never resolve URLs itself (dropped links are intercepted by oh_seek.lua and come to us)
    # --drag-and-drop=append: a drop while playing is appended to the playlist (oh_seek.lua takes it out again and
    #   sends it here) instead of replacing, and so interrupting, the segment that is playing
    # vsr / smooth: the settings of the same names at startup (later changes arrive over the IPC: run_player)
    # --prefetch-playlist: a segment appended behind a seek tail (seek_tail.py) is opened while the tail's end is
    #   played, so mpv continues with it after ~0.1 s instead of ~0.65 s
    # --network-timeout=0: mpv drops a tcp:// stream after 60 s without data (its default network timeout; measured),
    #   but a segment may stay silent longer: a seek tail with much restored video cached ahead, a very slow clip
    window_title = APP_NAME if idle_hint else f"{title} - {APP_NAME}"
    cmd = [mpv, "--no-terminal", "--osc=no", f"--force-media-title={title}", f"--title={window_title}",
           "--hwdec=auto-safe",
           "--osd-scale-by-window=no", "--osd-font-size=32",  # fixed-size OSD text, like oh_seek.lua's controls
           f"--input-ipc-server={ipc_path}", "--idle=yes", "--force-window=yes", "--keep-open=yes",
           "--prefetch-playlist=yes", "--network-timeout=0",
           "--rebase-start-time=no",  # time-pos = absolute source time (segments carry -output_ts_offset)
           f"--script={resource_root() / 'oh_seek.lua'}", f"--script={resource_root() / 'oh_sub_combos.lua'}",
           f"--script={resource_root() / 'oh_subtitle_preferences.lua'}",
           "--ytdl=no", "--drag-and-drop=append",
           "--slang=chi,zho,zh",
           "--cache=yes", "--cache-secs=36000", "--demuxer-max-bytes=4GiB" if compare == "on" else "--demuxer-max-bytes=2GiB",
           "--demuxer-max-back-bytes=1GiB",   # compare: every track is cached, so twice the bytes
           "--cache-pause=yes", "--cache-pause-initial=yes", f"--cache-pause-wait={prebuffer}"]
    cmd += [f"--{k}={v}" for k, v in PS.smooth_props(smooth).items()]
    # oh_seek.lua: oh-idle = start without a source (hint, no seekbar); oh-vsr = RTX VSR on / off until
    # user-data/oh/vsr says otherwise; oh-screen-h = monitor height if mpv has no display-height
    opts = [f"oh-vsr={vsr}", f"oh-platform={sys.platform}"] + (["oh-idle=yes"] if idle_hint else [])
    screen_h = primary_display_height()
    if screen_h:
        opts.append(f"oh-screen-h={screen_h}")
    cmd.append("--script-opts=" + ",".join(opts))
    process = subprocess.Popen(cmd, **player_runtime.process_kwargs(mpv))
    player_branding.apply_player_branding(process)
    return process


def _pipeline_dependencies():
    """Keep runtime operations and legacy injection points owned by this entry module."""
    return _pipeline.PipelineDependencies(
        log=lambda *a, **k: log(*a, **k),
        start_muxer=lambda *a, **k: start_muxer(*a, **k),
        stop_process=lambda *a, **k: stop_process(*a, **k),
        free_port=lambda: free_port(),
        is_listening=lambda *a, **k: is_listening(*a, **k),
        torch=torch, open_reader=lambda src: get_engine().open_reader(src),
        stats_factory=lambda *a, **k: Stats(*a, **k),
    )


class Stats(_pipeline.Stats):
    def __init__(self, fps: float, start_s: float, t0: float | None = None,
                 first_what="model warmup + open + first clip"):
        super().__init__(fps, start_s, t0, first_what, deps=_pipeline_dependencies())


# ---------------------------------------------------------------------------
# Hover thumbnails
# ---------------------------------------------------------------------------

class ThumbnailWorker(_ThumbnailWorker):
    def __init__(self, src: str, headers: dict, duration: float, width: int, directory: str):
        super().__init__(src, headers, duration, width, directory,
                         av_module=av, http_options_factory=FFmpegHttpOptions,
                         url_predicate=is_url, logger=lambda msg: log(msg))


# ---------------------------------------------------------------------------
# Seekable playback
# ---------------------------------------------------------------------------

class Session(_pipeline.Session):
    def __init__(self, restorer: "LadaSession", source: Source, encoder: str, max_frames: int | None):
        super().__init__(restorer, source, encoder, max_frames, deps=_pipeline_dependencies())


def passthrough_frames(source: Source, start_s: float):
    yield from _pipeline.passthrough_frames(source, start_s, deps=_pipeline_dependencies())


# ---------------------------------------------------------------------------
# Casting (hls_cast.py: on-demand HLS VOD; airplay_cast.py: AirPlay push)
# ---------------------------------------------------------------------------

class RestorerStream(_pipeline.RestorerStream):
    def __init__(self, restorer: "LadaSession", source: Source, start_s: float):
        super().__init__(restorer, source, start_s, deps=_pipeline_dependencies())


class CastManager:
    """Casting from the player: the HLS server + producer (hls_cast) and the AirPlay push (airplay_cast).

    Main thread only (it uses the mpv IPC); the AirPlay thread just queues events. While a cast runs, the local
    Session is stopped and its restorer is used by the HLS producer (one producer, models shared); mpv keeps the
    paused frame with a 「投放中：<裝置>」 status; local seeks control the receiver through MRP. Quality / settings changes while
    casting drop the segment cache and restart production at the receiver's position (handle_requests)."""

    def __init__(self, args, ipc: "MpvIpc"):
        self.args, self.ipc = args, ipc
        # tests only: a stand-in credentials file and a fake pyatv module (a .py file), so no real device is used
        self.store = airplay_cast.CredentialStore(os.environ.get("OH_STREAMER_TEST_AIRPLAY_CREDENTIALS")
                                                  or airplay_cast.CRED_FILE)
        self.fake_pyatv = None
        fake = os.environ.get("OH_STREAMER_TEST_FAKE_PYATV")
        if fake:
            import importlib.util
            spec = importlib.util.spec_from_file_location("fake_pyatv", fake)
            self.fake_pyatv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.fake_pyatv)
            log(f"cast: using the fake pyatv module {fake}")
        self.caster: airplay_cast.AirPlayCaster | None = None
        self.events = queue.Queue()
        self.devices = None                 # [airplay_cast.Device] from the latest scan (None = not scanned yet)
        self.scanning, self.scan_error, self.last_scan = False, None, 0.0
        self.active: dict | None = None
        self.published_menu = None
        self.vsr_enabled = False
        self.vsr_size = "2x"
        self.vsr_commit = None  # controller persists alongside other player settings
        self.vsr_checking = False
        self.subtitle_fallback = None
        try:
            import importlib.util
            self.pyatv_ok = self.fake_pyatv is not None or importlib.util.find_spec("pyatv") is not None
        except (ImportError, ValueError):
            self.pyatv_ok = False

    # -- helpers ---------------------------------------------------------------------------------------------------

    def _osd(self, text: str, ms: int = 4000):
        self._ipc("command", "show-text", text, ms)

    def _ipc(self, method: str, *args):
        """IPC call that never raises (mpv may be gone when the player shuts down); False on failure."""
        try:
            return getattr(self.ipc, method)(*args)
        except (MpvError, OSError, EOFError, ValueError):
            return False

    def _caster(self) -> "airplay_cast.AirPlayCaster":
        if self.caster is None:
            self.caster = airplay_cast.AirPlayCaster(self.store, lambda k, d: self.events.put((k, d)),
                                                     mod=self.fake_pyatv, log=log)
        return self.caster

    def publish_menu(self):
        a = self.active
        casting = {"device": a["device"], "url": a["url"]} if a else None
        rows = airplay_cast.build_cast_menu(casting, self.devices, self.scanning, self.scan_error, self.pyatv_ok)
        enabled = getattr(self, "vsr_enabled", False)
        checking = getattr(self, "vsr_checking", False)
        rows.append({"id": "vsr", "kind": "info" if a or checking else "item",
                     "label": "投放 RTX 超解析：" + ("檢查中…" if checking else "開" if enabled else "關"),
                     "hint": "停止投放後可切換" if a else "需要 NVIDIA RTX 與 NVEncC"})
        rows.append({"id": "vsr-size", "kind": "info" if a or checking else "item",
                     "label": "超解析輸出：" + self._vsr_size_label(),
                     "hint": "停止投放後可切換" if a else "點選切換；升到 4K 可將 720p 放大 3 倍，GPU 負擔較高"})
        pairing = getattr(self, "pairing", None)
        if pairing:
            rows = [{"id": "", "kind": "info", "label": pairing["status"]}]
            if pairing.get("input"):
                rows.append({"id": "", "kind": "info", "label": "PIN：" + (pairing["pin"] or "尚未輸入")})
                rows.extend({"id": f"pin:{n}", "kind": "item", "label": str(n)} for n in range(10))
                rows.extend([{"id": "pin:back", "kind": "item", "label": "刪除一位"},
                             {"id": "pin:submit", "kind": "item", "label": "確認 PIN"}])
            rows.append({"id": "pair:cancel", "kind": "item", "label": "取消配對"})
        s = json.dumps(rows, ensure_ascii=False)
        if s != self.published_menu and self._ipc("set", "user-data/oh/cast-menu", s) is not False:
            self.published_menu = s

    def scan(self):
        if not self.pyatv_ok or self.scanning:
            return
        self.scanning, self.scan_error, self.last_scan = True, None, time.monotonic()
        self._caster().scan()
        self.publish_menu()

    def resolve_device(self, query: str) -> tuple[str, str]:
        """(id, name) of a paired device by name / id (no network: from the credentials file)."""
        stored = self.store.load()
        devs = [airplay_cast.Device(k, str(v.get("name") or k), v.get("address"), paired=True)
                for k, v in stored.items()]
        if not devs:
            raise airplay_cast.CastError("還沒有配對的 Apple TV，請在播放器投放選單選擇配對")
        d = airplay_cast.match_device(devs, query)
        return d.id, d.name

    @staticmethod
    def video_start(session: "Session") -> float:
        m = session.meta
        if m.start_pts is not None:
            try:
                return float(int(m.start_pts) * m.time_base)
            except (TypeError, ValueError):
                pass
        return float(session.fmt_start or 0.0)

    @staticmethod
    def _run_input(session: "Session", t: float) -> "hls_cast.RunInput":
        """HLS run thread: the session's current restorer + source from source time t."""
        from oh_streamer.player.progressive_cast import ProgressiveCastSource
        if isinstance(session, (hls_cast.FileCastSource, ProgressiveCastSource)):
            return session.run_input(t)
        src = session.source
        audio = None
        if src.has_audio:
            audio = {"src": src.audio_src, "args": src.http.cli_args() if is_url(src.audio_src) else [],
                     "fmt_start": src.fmt_start}
        return hls_cast.RunInput(RestorerStream(session.restorer, src, t), src.meta.video_width,
                                 src.meta.video_height, audio)

    # -- start / stop ----------------------------------------------------------------------------------------------

    def start(self, mode: str, session: "Session | None", pos: float | None, dev_id: str | None = None,
              dev_name: str | None = None) -> bool:
        """mode "airplay" (dev_id) or "hls". Stops the local session first."""
        if session is None:
            self._osd("沒有正在播放的影片，無法投放", 4000)
            return False
        if getattr(self, "vsr_checking", False):
            self._osd("正在檢查投放超解析，請稍候", 4000)
            return False
        if self.active is not None:
            self.stop(resume_local=False)
        finished_file = isinstance(session, hls_cast.FileCastSource) or (not hasattr(session, "meta") and hasattr(session, "path"))
        vod = server = thumbnails = None
        try:
            if finished_file and not isinstance(session, hls_cast.FileCastSource):
                session = hls_cast.FileCastSource(session.path)
            pos = session.target if pos is None else pos
            host = self.args.cast_bind or hls_cast.lan_ip()
            m = session.meta
            vsr_exe = None
            out_w, out_h = m.video_width, m.video_height
            if getattr(self, "vsr_enabled", False):
                from oh_streamer.platform import cast_vsr
                out_w, out_h = cast_vsr.output_size(out_w, out_h, getattr(self, "vsr_size", "2x"))
                vsr_exe = cast_vsr.executable()
            v0 = self.video_start(session)
            frames = m.frames_count or int(round(m.duration * float(m.video_fps_exact)))
            plan = hls_cast.SegmentPlan(m.video_fps_exact, frames, out_w, out_h, video_start=v0,
                                        has_audio=session.has_audio)
            start_k = plan.index_at(max(0.0, pos - v0))
            self._ipc("set", "pause", True)
            from oh_streamer.platform.cast_subtitles import selected_subtitles
            subtitle_error = None
            try:
                subtitles = selected_subtitles(self._ipc, plan)
                if subtitles is None and callable(getattr(self, "subtitle_fallback", None)):
                    try:
                        subtitles = self.subtitle_fallback(plan)
                    except Exception as e:
                        log(f"cast subtitles fallback: {e!r}")
            except Exception as e:  # subtitles must not prevent video casting
                log(f"cast subtitles: {e!r}")
                subtitle_error = airplay_cast.short_reason(e)
                subtitles = None
            took = session.stop()             # one producer: the HLS runs use the session's restorer
            hls_log = lambda msg: log(f"hls: {msg}")    # noqa: E731
            vod = hls_cast.HlsVod(plan, lambda t: self._run_input(session, t), session.encoder,
                                  cache_bytes=int(self.args.cast_cache_gb * 2**30), start_k=start_k, log=hls_log,
                                  stderr=sys.stderr if LOG_TO_FILE else None, vsr_exe=vsr_exe)
            if getattr(self.args, "thumbnails", True):
                try:
                    from oh_streamer.player.cast_thumbnail_sources import create_thumbnails
                    thumbnails = create_thumbnails(vod, session)
                except Exception as e:  # previews must not prevent video casting
                    log(f"cast thumbnails: unavailable ({type(e).__name__})")
            subtitle_args = {"subtitles": subtitles} if subtitles is not None else {}
            if thumbnails is not None:
                subtitle_args["thumbnails"] = thumbnails
            server = hls_cast.HlsServer(vod, host, self.args.cast_port, log=hls_log, **subtitle_args).start()
        except Exception as e:  # noqa: BLE001 - never take the player down
            log(f"cast: could not start ({e!r})")
            if server is not None:
                server.stop()
            elif thumbnails is not None:
                thumbnails.close()
            if vod is not None:
                vod.close()
            self._osd(f"無法開始投放：{airplay_cast.short_reason(e)}", 6000)
            if finished_file:
                self._ipc("set", "pause", True)
            else:
                self._resume_local(pos, paused=True)
            return False
        url = server.url("master.m3u8" if subtitles is not None or thumbnails is not None
                         or self.args.cast_playlist == "master" else "index.m3u8")
        self.active = {"mode": mode, "device": dev_name if mode == "airplay" else None, "dev_id": dev_id,
                       "vod": vod, "server": server, "url": url, "local_pos": pos, "v0": v0, "plan": plan,
                       "start_k": start_k, "phase": "prepare" if mode == "airplay" else "hls",
                       "t_start": time.monotonic(), "finished_file": finished_file, "source": session}
        log(f"cast: {mode}{' to ' + dev_name if dev_name else ''} | {url} | from {pos:.2f}s (segment {start_k} of "
            f"{plan.count}, {plan.frames_per_segment} frames = {float(plan.seg_duration):.3f}s each) | local "
            f"stopped in {took:.2f}s")
        self._log_subtitle_timing(plan, subtitles, start_k, pos, v0)
        self._ipc("set", "user-data/oh/cast-position", pos)
        self._ipc("set", "user-data/oh/casting", "1")
        if mode == "hls":
            print(f"HLS: {url}", flush=True)
            self._osd(f"區網串流：{url}", 15000)
        else:
            self._osd(f"準備投放到 {dev_name}…", 3000)
        if subtitle_error:
            self._osd(f"影片投放已開始，但字幕無法投放：{subtitle_error}", 8000)
        self.publish_menu()
        return True

    def update_subtitles(self, subtitles):
        if self.active is not None and self.active.get("server") is not None:
            self.active["server"].update_subtitles(subtitles)
            self._log_subtitle_timing(self.active["plan"], subtitles, self.active["start_k"],
                                      self.active["local_pos"], self.active["v0"])

    @staticmethod
    def _log_subtitle_timing(plan, subtitles, start_k, pos, v0):
        """Cast start: every term of the subtitle/video time map (sample video A); diagnostics only."""
        try:
            if subtitles is None:
                log(f"cast subtitles: none | video_start {v0:.3f}s, start position {pos:.3f}s -> segment {start_k}")
                return
            for sub in getattr(subtitles, "renditions", (subtitles,)):
                log(sub.describe(start_k))
        except Exception as e:  # noqa: BLE001
            log(f"cast subtitles: timing log skipped: {e!r}")

    def stop(self, msg: str | None = None, resume_local: bool = True, paused: bool = True):
        if not resume_local:
            self.pair_retry = None
            if getattr(self, "pairing", None):
                self._caster().cancel_pair()
        a, self.active = self.active, None
        if a is None:
            return
        resume = a["local_pos"]
        if a["mode"] == "airplay" and self.caster is not None:
            tv = self.caster.tv_position()
            if tv is not None:
                resume = a["v0"] + tv              # continue locally where the Apple TV was
            if not self.caster.stop():
                log("cast: AirPlay did not stop within 5 s")
        try:
            a["server"].stop()
        except Exception as e:  # noqa: BLE001
            log(f"cast: stopping the HLS server failed: {e!r}")
        st = a["vod"].status()
        a["vod"].close()
        log(f"cast stopped ({msg or 'by request'}): {st['requests']} requests, {st['restarts']} restarts, "
            f"{st['cached_segments']} segments cached ({st['cached_bytes'] / 2**20:.0f} MiB), evicted {st['evicted']}")
        self._ipc("set", "user-data/oh/casting", "")
        self._ipc("set", "user-data/oh/cast-position", "")
        self._ipc("set", "user-data/oh/cast-paused", "")
        self.publish_menu()
        if msg:
            self._osd(msg, 5000)
        if resume_local:
            if a.get("finished_file"):
                self._ipc("command", "seek", max(0.0, resume), "absolute+exact")
                self._ipc("set", "pause", paused)
            else:
                self._resume_local(resume, paused)

    def _resume_local(self, pos: float, paused: bool):
        """Restart the local session at pos through oh_seek.lua (its normal request path)."""
        self._ipc("command", "script-message", "oh-restart", f"{max(0.0, pos):.3f}", "1" if paused else "0")

    def close(self):
        self.stop(resume_local=False)
        if self.caster is not None:
            self.caster.close()
            self.caster = None

    # -- main loop ---------------------------------------------------------------------------------------------------

    def handle_request(self, req: str, session, pos):
        """user-data/oh/cast-request = "<n>:<menu|scan|hls|stop|dev:<id>>" from oh_seek.lua."""
        what = req.split(":", 1)[1] if ":" in req else ""
        if what == "menu":
            if self.active is None and (self.devices is None or time.monotonic() - self.last_scan > 60):
                self.scan()
            self.publish_menu()
        elif what == "scan":
            self.scan()
        elif what == "vsr-size":
            if self.active or getattr(self, "vsr_checking", False):
                self._osd("請先停止投放並等待檢查完成，再切換超解析輸出", 4000)
                return
            self.vsr_size = "4k" if getattr(self, "vsr_size", "2x") == "2x" else "2x"
            if getattr(self, "vsr_commit", None):
                self.vsr_commit({"cast_vsr_size": self.vsr_size})
            self.publish_menu()
            self._osd("超解析輸出：" + self._vsr_size_label() + "（開啟投放 RTX 超解析後生效）", 4000)
        elif what == "vsr":
            if self.active or getattr(self, "vsr_checking", False):
                self._osd("請先停止投放，再切換投放超解析", 4000)
                return
            enabled = not getattr(self, "vsr_enabled", False)
            if enabled:
                self.vsr_checking = True
                self.publish_menu()
                self._osd("正在檢查投放 RTX 超解析…", 30000)
                def check_vsr():
                    try:
                        from oh_streamer.platform import cast_vsr
                        cast_vsr.probe()
                        self.events.put(("vsr-checked", None))
                    except Exception as e:
                        self.events.put(("vsr-checked", str(e) or type(e).__name__))
                threading.Thread(target=check_vsr, name="cast VSR check", daemon=True).start()
            else:
                self._set_vsr(False)
        elif what == "pair:cancel":
            self._caster().cancel_pair()
        elif what.startswith("pair:"):
            if not getattr(self, "pairing", None) and self.pyatv_ok and self.active is None:
                self.pairing = {"status": "開始配對…", "pin": "", "input": False}
                self._caster().pair(what[5:])
                self.publish_menu()
                self._ipc("command", "script-message", "oh-cast-menu", "open")
        elif what.startswith("pin:"):
            pairing = getattr(self, "pairing", None)
            if pairing and pairing.get("input"):
                key = what[4:]
                if key == "back":
                    pairing["pin"] = pairing["pin"][:-1]
                elif key == "submit" and pairing["pin"]:
                    self._caster().submit_pin(pairing["pin"])
                    pairing.update(input=False, status="驗證 PIN 中…")
                elif len(key) == 1 and key in "0123456789" and len(pairing["pin"]) < 8:
                    pairing["pin"] += key
                self.publish_menu()
        elif what in ("play_pause", "pause", "play"):
            if self.active and self.active["mode"] == "airplay" and self.active["phase"] == "playing":
                self._caster().control(what)
            else:
                self._osd("Apple TV 尚未開始播放，或目前使用區網串流", 3000)
        elif what == "stop":
            self.stop("已停止投放", resume_local=True, paused=True)
        elif what == "hls":
            self.auth_retry_device = None
            self.start("hls", session, pos)
        elif what.startswith("dev:"):
            self.auth_retry_device = None
            dev_id = what[4:]
            entry = self.store.get(dev_id)
            if entry is None:
                self._osd("此裝置尚未配對：請在投放選單選擇配對", 6000)
                return
            self.start("airplay", session, pos, dev_id, str(entry.get("name") or dev_id))

    def _vsr_size_label(self):
        return "升到 4K" if getattr(self, "vsr_size", "2x") == "4k" else "2 倍（最高 4K）"

    def _set_vsr(self, enabled):
        self.vsr_enabled = enabled
        if getattr(self, "vsr_commit", None):
            self.vsr_commit({"cast_vsr": "on" if enabled else "off"})
        self.publish_menu()
        self._osd("投放 RTX 超解析已" + ("開啟：" + self._vsr_size_label() if enabled else "關閉"), 4000)

    def seek(self, pos: float):
        """Seek in source time, keeping the HLS server and processed segments."""
        a = self.active
        if a is None or a["mode"] != "airplay":
            self._osd("區網串流請在接收端調整進度", 3000)
            return
        if not float("-inf") < pos < float("inf"):
            raise ValueError("cast seek position must be finite")
        if a["phase"] != "playing":
            self._osd("Apple TV 尚未開始播放", 3000)
            return
        relative = max(0.0, min(pos - a["v0"], max(0.0, float(a["plan"].total_duration) - 1)))
        try:
            from oh_streamer.platform.cast_subtitles import seek_line
            log(seek_line(a["plan"], a["v0"], pos, relative, getattr(a["server"], "subtitles", None)))
        except Exception as e:  # noqa: BLE001 - diagnostics only
            log(f"cast seek: timing log skipped: {e!r}")
        self._caster().control("seek", relative)

    def poll(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            a = self.active
            if kind == "vsr-checked":
                self.vsr_checking = False
                if data:
                    log(f"cast VSR unavailable: {data}")
                    self._osd("無法開啟投放超解析：" + data.splitlines()[0], 8000)
                else:
                    self._set_vsr(True)
            elif kind == "scan":
                self.scanning, self.devices = False, data
                log(f"airplay scan: {data}")
            elif kind == "pair-status" and getattr(self, "pairing", None):
                self.pairing["status"] = data
            elif kind == "pair-pin" and getattr(self, "pairing", None):
                self.pairing.update(status=data, input=True)
            elif kind == "auth-error" and a and a["mode"] == "airplay":
                retry = (a["source"], a["local_pos"], a["dev_id"], a["device"])
                self.stop("Apple TV 驗證失敗，請重新輸入電視上的 PIN", resume_local=True, paused=True)
                if getattr(self, "auth_retry_device", None) == data["device_id"]:
                    self._osd("重新配對後仍驗證失敗，請確認 Apple TV 的 AirPlay 存取設定", 8000)
                    continue
                self.auth_retry_device = data["device_id"]
                self.pair_retry = retry
                self.handle_request("auth:pair:" + data["device_id"], None, None)
            elif kind in ("paired", "pair-error", "pair-cancelled"):
                self.pairing = None
                if kind == "paired":
                    self.devices = airplay_cast.merged_devices(self.devices or [], self.store)
                    retry = getattr(self, "pair_retry", None)
                    self.pair_retry = None
                    if retry and retry[2] == data.id:
                        self.start("airplay", retry[0], retry[1], retry[2], retry[3])
                    else:
                        self._osd(f"已配對 {data.name}，請選擇投放裝置", 5000)
                else:
                    self.pair_retry = None
                    self._osd(f"配對失敗：{data}" if kind == "pair-error" else "已取消配對", 6000)
            if kind.startswith("pair-") and getattr(self, "pairing", None):
                self.publish_menu()
                self._ipc("command", "script-message", "oh-cast-menu", "open")
            if kind == "scan-error":
                self.scanning, self.scan_error = False, data
            elif kind == "control-done" and a and data["action"] == "seek":
                a["vod"].prefetch(a["plan"].index_at(data["position"]))
            elif kind == "control-error":
                self._osd(data, 6000)
            elif kind == "playing" and a and a["mode"] == "airplay":
                a["phase"] = "playing"
                self._osd(f"已開始投放到 {data}", 3000)
            elif kind == "ended" and a and a["mode"] == "airplay" and a["phase"] in ("playing", "connecting"):
                self.stop("投放已結束（Apple TV 已停止播放）", resume_local=True, paused=True)
            elif kind == "error":
                if a and a["mode"] == "airplay":
                    self.stop(f"投放失敗：{data}", resume_local=True, paused=True)
                else:
                    self._osd(f"AirPlay：{data}", 6000)
            self.publish_menu()
        a = self.active
        if a and getattr(a.get("source"), "is_progressive_cast", False) and a["vod"].last_error:
            self.stop("投放片段讀取失敗（詳見日誌）", resume_local=True)
            return
        if a and getattr(self, "vsr_enabled", False) and a["vod"].last_error:
            error = a["vod"].last_error[2]
            log(f"cast VSR pipeline failed: {error}")
            self.stop("投放超解析處理失敗；請關閉投放超解析後重試（詳見日誌）", resume_local=True)
            return
        if a and a["phase"] == "prepare":
            vod, k = a["vod"], a["start_k"]
            ready = vod.is_cached(k) and (k + 1 >= a["plan"].count or vod.is_cached(k + 1))
            if ready or (not getattr(a.get("source"), "is_progressive_cast", False)
                         and time.monotonic() - a["t_start"] > 40):
                # the Apple TV gets the URL once the first segments exist (pyatv gives up if nothing plays soon)
                a["phase"] = "connecting"
                log(f"cast: first segments ready after {time.monotonic() - a['t_start']:.2f}s, sending to "
                    f"{a['device']}")
                caster = self._caster()
                caster.play(a["dev_id"], a["url"], position=max(0.0, a["local_pos"] - a["v0"]))
        if a and a["mode"] == "airplay":
            tv = self.caster.tv_position() if self.caster and a["phase"] == "playing" else None
            pos = a["v0"] + tv if tv is not None else a["local_pos"]
            a["local_pos"] = pos
            self._ipc("set", "user-data/oh/cast-position", pos)
            state = self.caster.tv_state() if self.caster else None
            self._ipc("set", "user-data/oh/cast-paused", "1" if state and state.get("rate") == 0 else "0")

    def status_text(self) -> str | None:
        a = self.active
        if a is None:
            return None
        if getattr(a.get("source"), "is_progressive_cast", False) and a["source"].waiting_k is not None:
            hold = a["source"].playback.hold
            from oh_streamer.player.progressive_stream import HOLD_STAGES
            return HOLD_STAGES.get(hold, "等待背景處理投放片段…")
        if a["mode"] == "hls":
            return f"區網串流中\n{a['url']}\n按 c 或投放按鈕可停止"
        if a["phase"] == "prepare":
            return f"準備投放到 {a['device']}…"
        if a["phase"] == "connecting":
            return f"連線到 {a['device']}…"
        return f"投放中：{a['device']}\n進度條／方向鍵：跳轉；空白鍵：暫停／播放；c：投放選單"

    def suspend(self):
        """Before a quality / settings change: stop production (the restorer is free afterwards)."""
        if self.active is not None and not self.active["vod"].suspend():
            log("cast: HLS production did not stop within 60 s")

    def resume(self, encoder: str, changed: bool):
        a = self.active
        if a is None:
            return
        vod = a["vod"]
        vod.encoder = encoder
        k = vod.restart_point()
        vod.resume(invalidate=changed, at_k=k)
        if changed:
            log(f"cast: cache dropped, production restarts at segment {k}")

    def info_snapshot(self) -> dict | None:
        a = self.active
        if a is None:
            return None
        st = a["vod"].status()
        lr = st["last_request"]
        tv = self.caster.tv_position() if (self.caster and a["mode"] == "airplay") else None
        return {"mode": a["mode"], "device": a["device"], "phase": a["phase"], "url": a["url"],
                "prod_pos": st["prod_pos"], "producing": st["producing"], "paused": st["paused"],
                "run_start": st["run_start"], "frames_written": st["frames_written"],
                "processing_done": st["processing_done"],
                "cached_segments": st["cached_segments"], "cached_bytes": st["cached_bytes"],
                "cached_ranges": st["cached_ranges"], "requests": st["requests"], "restarts": st["restarts"],
                "last_request": None if lr is None else {**lr, "age": time.perf_counter() - lr["t"]},
                "jump_latency": st["last_jump_latency"], "error": st["error"], "tv_pos": tv,
                "duration": st["duration"]}


class QualityControl:
    """The selectable renditions of the video and which one is playing (single entry = no quality button)."""

    def __init__(self, quals: list[dict], current: dict, lada: "LadaEngine", user_headers: dict,
                 refresh=None):
        self.qualities, self.current, self.lada = quals, current, lada
        self.user_headers = dict(user_headers)   # --header values win over yt-dlp's
        self.refresh = refresh                   # () -> fresh list of qualities (re-runs yt-dlp), or None

    def headers_for(self, q: dict) -> dict:
        return {**q["http_headers"], **self.user_headers}

    def open(self, q: dict) -> Source:
        return Source(q["url"], self.headers_for(q), q.get("audio_url"))


class SettingsControl:
    """The advanced settings in use (cfg), what the settings file holds (saved), and the menu built from them.

    `lada` is None while the models are not loaded (the player loads them on demand, see Engine): the menu, the
    settings file and the mpv-side settings work all the same; model-bound changes are only remembered and take
    effect when the models are loaded the next time."""

    def __init__(self, path, saved: dict, cfg: dict, lada: "LadaEngine | None", det_names: list[str],
                 res_names: list[str], nvenc_ok: bool, videotoolbox_ok: bool = False, amf_ok: bool = False, device=None):
        self.path, self.saved, self.cfg, self.lada = path, dict(saved), dict(cfg), lada
        self.device = device
        self.det_names, self.res_names, self.nvenc_ok = det_names, res_names, nvenc_ok
        self.videotoolbox_ok = videotoolbox_ok
        self.amf_ok = amf_ok
        self.auto_encoder = encoding_backend.resolve_encoder("auto", self._encoder_usable)
        self._menu = None
        self._fp16_last = self._resolve_fp16_unloaded()
        # per-source overrides in effect (an "open" request's settings): key -> the value cfg held before. cfg holds the
        # effective values (what runs, what the menu shows); the settings file only ever gets committed changes.
        self.override_base: dict = {}

    def _resolve_fp16_unloaded(self) -> bool:
        """FP16 as the models would use it without loading them (the check creates no CUDA context)."""
        if self.cfg.get("fp16") is not None:
            return bool(self.cfg["fp16"])
        try:
            return bool(gpu_has_fp16_acceleration(self.device))
        except Exception:  # noqa: BLE001 - the menu still has to open
            return False

    def attach(self, lada: "LadaEngine", device):
        self.lada, self.device = lada, device
        self._menu = None

    def detach(self):
        """The models were released: keep showing the FP16 they used (and the device RIFE needs)."""
        if self.lada is not None and self.lada.models is not None:
            self._fp16_last = bool(self.lada.models.fp16)
        self.lada = None
        self._menu = None

    def rife_ok(self) -> bool:
        device = self.device
        if device is None and self.lada is not None:
            device = self.lada.device
        if device is None:   # not loaded: what the device would be (asking torch creates no CUDA context)
            cuda = not str(self.cfg.get("device", "auto")).startswith("cpu") and torch.cuda.is_available()
            device = torch.device("cuda" if cuda else "cpu")
        return device_backend.rife_available(device)

    def _encoder_usable(self, name):
        return {"h264_nvenc": self.nvenc_ok, "h264_amf": self.amf_ok, "h264_videotoolbox": self.videotoolbox_ok}.get(name, False)

    @property
    def fp16_on(self) -> bool:
        if self.lada is not None and self.lada.models:
            return bool(self.lada.models.fp16)
        return self._fp16_last

    def resolve_encoder(self, name: str) -> str:
        resolved = encoding_backend.resolve_encoder(name, self._encoder_usable)
        if name != "auto" and resolved != name:
            log(f"{name} is not usable here, using {resolved}")
        return resolved

    def menu(self) -> list[dict]:
        if self._menu is None:
            self._menu = PS.build_menu(self.cfg, self.fp16_on, self.det_names, self.res_names, self.nvenc_ok,
                                       self.auto_encoder, videotoolbox_ok=self.videotoolbox_ok, amf_ok=self.amf_ok,
                                       rife_ok=self.rife_ok(),
                                       overridden=frozenset(self.override_base),
                                       engines=[(n, engine_label(n)) for n in visible_engine_names()])
        return self._menu

    def invalidate_menu(self):
        self._menu = None

    def diff(self, wanted: dict) -> dict:
        return PS.diff(self.cfg, self.fp16_on, wanted)

    def commit(self, changes: dict, write: bool = True):
        """Changes that took effect: remember them and persist them to the settings file (write=False: the file
        already holds them, e.g. after a settings_reload: remember only)."""
        if not changes:
            return
        self.cfg.update({k: v for k, v in changes.items() if k in PS.DEFAULTS})
        self._menu = None
        for k in changes:
            self.override_base.pop(k, None)    # a value chosen in the menu is the user's own: no longer an override
        self.saved = PS.updated_file(self.saved, changes)
        if not write:
            return
        try:
            PS.save_file(self.path, self.saved)
            log(f"settings saved to {self.path}: {changes}")
        except (OSError, ValueError) as e:
            log(f"could not save settings to {self.path}: {e}")

    def apply_transient(self, changes: dict):
        """Changes that took effect for the current source only: in cfg (so they are what runs and what the menu shows)
        but not in `saved` and not written to the settings file. Never touches override_base: the caller owns it."""
        self.cfg.update({k: v for k, v in changes.items() if k in PS.DEFAULTS})
        self._menu = None


def req_counter(req) -> int:
    try:
        return int(str(req).split(":", 1)[0])
    except ValueError:
        return 0


def _float_or(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class Engine:
    """Models + advanced settings, shared by every source of a process.

    The CLI loads the models before mpv starts (load()). The persistent player opens its window first and builds the
    settings (init_light_async(): no models, no CUDA); the models load on demand (ensure_loaded(): a source that needs
    the restoration engine) and are freed again when the player has not needed them for a while (unload())."""

    def __init__(self, args, saved: dict, cfg: dict, det_names: list[str], res_names: list[str]):
        cfg, notes = PS.fix_platform(cfg)
        for note in notes:
            log(note)
        self.args, self.saved, self.cfg = args, saved, cfg
        self.device = None
        self.det_names, self.res_names = det_names, res_names
        self.lada: LadaEngine | None = None   # the process's Lada engine while its models are loaded
        self.sc: SettingsControl | None = None
        self.ready = threading.Event()   # set when loading finished (successfully or not: see .error); cleared by unload()
        self.error: Exception | None = None
        self.interp: rife_interp.Interpolator | None = None   # created when interpolation is first "auto"
        self._lock = threading.RLock()
        self._load_started = False
        self._hw: dict = {}
        self._hw_thread: threading.Thread | None = None

    @property
    def loading(self) -> bool:
        return self._load_started and not self.ready.is_set()

    def interpolator(self) -> "rife_interp.Interpolator | None":
        """The shared RIFE interpolator (None without CUDA: interpolation then stays off)."""
        device = self.device if self.device is not None else device_backend.select_device(getattr(self.args, "device", "auto"))
        if self.interp is None and device_backend.rife_available(device):
            self.interp = rife_interp.Interpolator(device, log)
        return self.interp

    def release_interp(self):
        """Interpolation switched off: free the RIFE model's VRAM."""
        if self.interp is not None:
            self.interp.release()
            gc.collect()
            device_backend.empty_cache(self.device)

    def _start_hardware_probe(self):
        """The test encodes (NVENC / AMF / VideoToolbox), on a thread, once."""
        with self._lock:
            if self._hw_thread is not None or self.args.output == "null":
                return
            encoder_name = "h264_videotoolbox" if sys.platform == "darwin" else "h264_nvenc"
            probe = (lambda: encoding_backend.encoder_usable(encoder_name)) if sys.platform == "darwin" else nvenc_available

            def probe_hardware():
                self._hw["ok"] = probe()
                if sys.platform == "win32":
                    self._hw["amf_ok"] = encoding_backend.encoder_usable("h264_amf")
            self._hw_thread = threading.Thread(target=probe_hardware, name="encoder probe", daemon=True)
            self._hw_thread.start()

    def _ensure_sc(self) -> "SettingsControl":
        """The settings control, built once (waits for the encoder probe)."""
        with self._lock:
            if self.sc is None:
                self._start_hardware_probe()
                if self._hw_thread is not None:
                    self._hw_thread.join()
                hw = self._hw
                cfg = self.cfg
                cfg["device"] = getattr(self.args, "device", "auto")
                self.sc = SettingsControl(self.args.settings_file, self.saved, cfg, None, self.det_names,
                                          self.res_names, hw.get("ok", False) if sys.platform != "darwin" else False,
                                          videotoolbox_ok=hw.get("ok", False) if sys.platform == "darwin" else False,
                                          amf_ok=hw.get("amf_ok", False), device=self.device)
            return self.sc

    def init_light_async(self):
        """Settings without models (the persistent player): menu, settings file and mpv-side settings are usable from
        here on; the GPU is not touched."""
        def run():
            try:
                self._ensure_sc()
                log("settings ready (models not loaded)")
            except Exception as e:  # noqa: BLE001
                log(f"preparing the settings failed: {e!r}")
        threading.Thread(target=run, name="settings init", daemon=True).start()

    def load(self, prewarm: tuple[int, int, float] | None = None):
        """prewarm: (width, height, fps) of a source that is already known (CLI): with interpolation "auto" its RIFE
        model is built while Lada's models load (in the player it is built once the first segment plays)."""
        try:
            cfg = self.sc.cfg if self.sc is not None else self.cfg   # the settings in use (they may have changed since)
            device = self.device = device_backend.select_device(getattr(self.args, "device", "auto"))
            cfg["device"] = getattr(self.args, "device", "auto")
            fp16 = gpu_has_fp16_acceleration(device) if cfg["fp16"] is None else cfg["fp16"]
            self._start_hardware_probe()   # the test encode runs while the models load
            if cfg["interpolation"] == "auto" and self.interpolator() is not None:
                if prewarm is not None and rife_interp.eligible(float(planned_fps(prewarm[2], cfg.get("restoration_fps", "original"))))[0]:
                    self.interp.prepare(prewarm[0], prewarm[1])   # background thread
                else:
                    self.interp.preload()                         # weights only (frame size not known yet)
            t = time.perf_counter()
            lada = get_engine()
            lada.restoration_fps = cfg.get("restoration_fps", "original")
            lada.load(device, ModelChoice(cfg["detection_model"], cfg["restoration_model"], bool(fp16)), cfg["max_clip_length"])
            log(f"models loaded in {time.perf_counter() - t:.1f}s on {device} (det={cfg['detection_model']}, "
                f"res={cfg['restoration_model']}, fp16={fp16}, max clip length {cfg['max_clip_length']})")
            sc = self._ensure_sc()
            sc.attach(lada, device)
            self.lada = lada
            if self.args.output != "null":
                log(f"encoder: {self.encoder()}")
        except Exception as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            self.ready.set()

    def load_async(self):
        def run():
            try:
                self.load()
            except Exception as e:  # noqa: BLE001
                log(f"loading the models failed: {e!r}")
        threading.Thread(target=run, name="model loader", daemon=True).start()

    def ensure_loaded(self):
        """Start loading the models (background thread) unless they are loaded or loading; progress: .loading / .ready."""
        with self._lock:
            if self._load_started:
                return
            self._load_started = True
            self.error = None
            self.ready.clear()
        log("loading the models on demand")
        self.load_async()

    def unload(self) -> bool:
        """Free the models and RIFE (VRAM). The caller guarantees that nothing uses them (no session, no restorer).
        False when there is nothing loaded (or loading is still running)."""
        with self._lock:
            if self.lada is None or not self.ready.is_set():
                return False
            lada, self.lada = self.lada, None
            if self.sc is not None:
                self.sc.detach()
        if self.interp is not None:
            self.interp.release()
            self.interp = None
        lada.release()
        gc.collect()
        device_backend.empty_cache(self.device)
        with self._lock:
            self._load_started = False
            self.ready.clear()
        log("models unloaded (VRAM freed)")
        return True

    def encoder(self) -> str:
        if self.sc is not None:
            return self.sc.resolve_encoder(self.sc.cfg["encoder"])
        # Progressive queue playback leaves the AI engine unloaded. FFmpeg
        # backend selection is independent of model loading and device setup.
        requested = self.cfg.get("encoder", "auto")
        cached = getattr(self, "_unloaded_encoder", None)
        if cached is None or cached[0] != requested:
            cached = (requested, encoding_backend.resolve_encoder(requested))
            self._unloaded_encoder = cached
        return cached[1]


def _opening_dependencies():
    return _opening.OpeningDependencies(
        source_factory=lambda *a, **k: Source(*a, **k),
        find_resolver=lambda *a, **k: find_resolver(*a, **k),
        resolve_external=lambda *a, **k: resolve_external(*a, **k),
        resolve_with_ytdlp=lambda *a, **k: resolve_with_ytdlp(*a, **k),
        log=lambda msg: log(msg), is_url=lambda value: is_url(value),
        looks_like_media_url=lambda value: looks_like_media_url(value),
        pick_quality=lambda *a, **k: qualities.pick_quality(*a, **k),
        quality_label=lambda *a, **k: qualities.quality_label(*a, **k),
        source_label=lambda *a, **k: info_panel.source_label(*a, **k),
    )


Opened = _opening.Opened


def resolve_input(inp: str | None, user_headers: dict, args, quality_spec, fake_any: bool = True) -> dict:
    return _opening.resolve_input(inp, user_headers, args, quality_spec, fake_any,
                                  deps=_opening_dependencies())


def probe_input(r: dict) -> Opened:
    return _opening.probe_input(r, deps=_opening_dependencies())


def open_source(inp: str, user_headers: dict, args, quality_spec, fake_any: bool = True) -> Opened:
    """Resolve (yt-dlp) + pick the quality + probe: everything needed before a Session can play `inp`."""
    return probe_input(resolve_input(inp, user_headers, args, quality_spec, fake_any))


short_error = _opening.short_error


class Opener(_opening.Opener):
    def __init__(self, args, quality_spec):
        super().__init__(args, quality_spec,
                         open_source=lambda *a, **k: open_source(*a, **k),
                         log=lambda msg: log(msg))


# ---------------------------------------------------------------------------
# Background processing queue (queue_ipc.py): processed files, busy reporting, add / lookup / watched
# ---------------------------------------------------------------------------

IDLE_HINT = "請從主程式開啟影片"


def _controller_dependencies():
    """Inject the current facade operations without making the controller import this module."""
    return _controller.PlayerDependencies(
        BusyReporter=lambda *a, **kw: BusyReporter(*a, **kw),
        CastManager=lambda *a, **kw: CastManager(*a, **kw),
        Fraction=Fraction,
        IDLE_HINT=IDLE_HINT,
        MpvEndpoint=lambda *a, **kw: MpvEndpoint(*a, **kw),
        MpvError=MpvError,
        MpvIpc=lambda *a, **kw: MpvIpc(*a, **kw),
        Opener=lambda *a, **kw: Opener(*a, **kw),
        PS=PS,
        Path=Path,
        ProcessedOpen=ProcessedOpen,
        QualityControl=lambda *a, **kw: QualityControl(*a, **kw),
        QueueLink=lambda *a, **kw: QueueLink(*a, **kw),
        Session=lambda *a, **kw: Session(*a, **kw),
        ThumbnailWorker=lambda *a, **kw: ThumbnailWorker(*a, **kw),
        _float_or=lambda *a, **kw: _float_or(*a, **kw),
        airplay_cast=airplay_cast,
        buffer_progress=buffer_progress,
        build_add_payload=lambda *a, **kw: build_add_payload(*a, **kw),
        collections=collections,
        compare_stack=compare_stack,
        device_backend=device_backend,
        find_mpv=lambda *a, **kw: find_mpv(*a, **kw),
        forget_sources=lambda *a, **kw: forget_sources(*a, **kw),
        gc=gc,
        info_panel=info_panel,
        json=json,
        log=lambda *a, **kw: log(*a, **kw),
        mpv_playback_snapshot=lambda *a, **kw: mpv_playback_snapshot(*a, **kw),
        open_input=open_input,
        os=os,
        passthrough_frames=lambda *a, **kw: passthrough_frames(*a, **kw),
        platform_support=platform_support,
        processed_from_message=lambda *a, **kw: processed_from_message(*a, **kw),
        processed_watched=lambda *a, **kw: processed_watched(*a, **kw),
        subtitle_update_from_message=lambda *a, **kw: subtitle_update_from_message(*a, **kw),
        progressive_stream=progressive_stream,
        qualities=qualities,
        queue=queue,
        queue_lookup=lambda *a, **kw: queue_lookup(*a, **kw),
        queue_quality=lambda *a, **kw: queue_quality(*a, **kw),
        register_source=lambda *a, **kw: register_source(*a, **kw),
        req_counter=lambda *a, **kw: req_counter(*a, **kw),
        seek_tail=seek_tail,
        short_error=lambda *a, **kw: short_error(*a, **kw),
        shutil=shutil,
        start_mpv=lambda *a, **kw: start_mpv(*a, **kw),
        tempfile=tempfile,
        threading=threading,
        time=time,
    )


def run_player(args, engine: Engine, first: Opened | None = None, quality_spec=None, instance=None,
               initial_inputs=()) -> int:
    """The mpv player loop.

    first: a source opened before mpv starts (CLI usage). Without it mpv opens idle with a hint (--player) and every
    source arrives as an open request: from the main app over `instance` (single-instance IPC), `--player <input>`
    (development) forwarded by a second process, or initial_inputs [(text, headers, start)] from this process's command line.
    """
    return _controller.PlayerController(args, engine, first, quality_spec, instance, initial_inputs,
                                        deps=_controller_dependencies()).run()


def run_batch(args, restorer: "LadaSession", src: str, meta: VideoMetadata, has_audio: bool,
              fmt_start: float, http: FFmpegHttpOptions, encoder: str, max_frames: int | None,
              interp: "rife_interp.Interpolator | None" = None) -> int:
    return _pipeline.run_batch(args, restorer, src, meta, has_audio, fmt_start, http,
                               encoder, max_frames, interp, deps=_pipeline_dependencies())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from oh_streamer.runtime_env.app_running import hold_app_mutex
    hold_app_mutex()
    args = EARLY_ARGS if EARLY_ARGS is not None else build_arg_parser().parse_args()
    if getattr(args, "engine", None):
        set_default_engine(args.engine)
    detect_gate.DetectGate.enabled = args.detect_gate
    ap = build_arg_parser()
    if args.initial_prebuffer is None:
        args.initial_prebuffer = args.prebuffer
    if args.player and args.output != "play":
        ap.error("--player only works with --output play")
    saved, warning = PS.load_file(args.settings_file)
    if warning:
        log(warning)
    try:
        cli = PS.cli_overrides(args)
    except ValueError as e:
        ap.error(str(e))
    # engine: --engine / OH_STREAMER_ENGINE (developer overrides) > the settings file > lada; unknown names fall back to lada
    engine_name = set_configured_engine(effective_engine_name(saved.get("engine"), log), log)
    if explicit_engine_name() is not None and engine_name != "lada":
        log(f"restoration engine: {engine_name} (developer option)")
    log("startup: modules imported, listing models")
    catalog = get_engine().list_models()
    log("startup: model catalog listed")
    det_names, res_names = list(catalog.detection), list(catalog.restoration)
    cfg, notes = PS.fix_models({**PS.effective(saved, cli), "engine": engine_name}, cli, det_names, res_names)
    for n in notes:
        log(n)
    args.seek_prebuffer = cfg["seek_prebuffer"]
    if saved:
        log(f"saved settings ({args.settings_file}): {saved}" + (f" | command line: {cli}" if cli else ""))
    try:
        quality_spec = qualities.parse_quality_spec(args.quality)
    except ValueError as e:
        ap.error(str(e))
    if cfg["detection_model"] not in det_names or cfg["restoration_model"] not in res_names:
        raise SystemExit(f"Unknown detection/restoration model name (installed: detection {', '.join(det_names)}; "
                         f"restoration {', '.join(res_names)})")
    engine = Engine(args, saved, cfg, det_names, res_names)
    install_lada_io_hooks()

    if args.player:
        # persistent player: the window opens right away (idle hint), models load meanwhile, the input (if any, the
        # development `--player <input>`) is opened like an input forwarded from the main app
        find_mpv(args.mpv)
        log(f"player mode (instance '{single_instance.instance_name(args.instance)}', pid {os.getpid()})")
        engine.init_light_async()
        inputs = [(args.input, parse_headers(args.header), args.start)] if args.input else []
        return run_player(args, engine, None, quality_spec, instance=args.instance_obj, initial_inputs=inputs)

    if not args.input and not args.fake_formats:
        ap.error("input is required (or use --player)")

    # 1) resolve input
    user_headers = parse_headers(args.header)
    r = resolve_input(args.input, user_headers, args, quality_spec, fake_any=True)
    if args.list_qualities:
        if r["quals"] is None:
            print("(single quality: the input is not resolved by yt-dlp)")
        else:
            print(qualities.format_table(r["quals"], r["chosen"]))
        return 0

    # 2) probe once, share with Lada
    opened = probe_input(r)
    opened.start = args.start
    source = opened.source
    register_source(source)

    # 3) models (and with interpolation "auto" the RIFE model for this frame size, meanwhile)
    if args.output == "play":
        find_mpv(args.mpv)  # fail before loading models
    engine.load(prewarm=(source.meta.video_width, source.meta.video_height, float(source.meta.video_fps_exact)))

    # 4) consume frames
    if args.output == "play":
        return run_player(args, engine, opened, quality_spec)
    restorer = engine.lada.open(source)
    meta = source.meta
    max_frames = int(args.duration * float(meta.video_fps_exact)) if args.duration > 0 else None
    encoder = engine.encoder() if args.output != "null" else cfg["encoder"]
    interp = engine.interpolator() if cfg["interpolation"] == "auto" else None
    return run_batch(args, restorer, source.audio_src, meta, source.has_audio, source.fmt_start,
                     source.http, encoder, max_frames, interp)


if __name__ == "__main__":
    sys.exit(main())
