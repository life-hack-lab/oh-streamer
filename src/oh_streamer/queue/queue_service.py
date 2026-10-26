"""
Headless controller of the background video-processing queue.

Runs in the queue GUI process, so it must never import torch / oh_player; the pywebview shell (queue_app.py, a
separate file) only calls the public methods below and registers callbacks.

    svc = QueueService(root, player_name=None, launch_player=None)
    svc.on_state.append(lambda state: ...)   # state dict, coalesced to <= 2 per second
    svc.on_show.append(lambda: ...)          # the player asked to bring the window to the front
    if not svc.start():                      # False: another queue app already owns the IPC instance
        queue_ipc.send_to_queue({"cmd": "show"}); sys.exit()
    ...
    svc.stop()

Threading model
  * one scheduler thread (`queue-scheduler`, tick 0.2 s, woken at once by state changes) makes every scheduling decision;
  * at most ONE download thread and ONE worker-supervisor thread exist at a time (a `_Run` each); the supervisor
    reads the worker's stdout JSON lines and applies the outcome (done / failed / paused / back to queued);
  * the single_instance listener thread answers the player's IPC messages;
  * UI methods run on whatever thread the shell uses. All shared service state is guarded by one RLock that is never
    held while joining threads or calling user callbacks. Store writes of progress are throttled to ~1/s.
  A worker is stopped by killing its process tree with psutil (only descendants of the Popen we started).

Pause semantics: pause(id) -> state `paused` (user decision, survives restarts); pause_all() and the player_busy
auto-pause only stop the running work and put it back to `queued` / `waiting`, so it restarts by itself.
Watching (P6b): the player in progressive mode sends `want` / `watching`; an item counts as watched until
busy_expire s without a heartbeat. The most recently wanted watched item that has a downloaded source is the priority
item: it preempts another running worker (that one goes back to `queued`, not user-paused), runs even while
player_busy (not under pause_all or while user-paused: want then answers "paused"; play() resumes a paused item),
is never evicted, and keeps its work/ dir until the watch expires. `goto(id, k)` moves the
running worker (stdin JSON line) or is remembered as the item's next start_k.
Item settings sent to the worker = item.settings overridden by the queue settings encoder / interpolation and
save_compare -> compare (+ compare_original / compare_quality when on; the values used are recorded in the item's
settings when its job is built), except the keys an item pinned in item.settings["pinned"] ({interpolation, encoder,
save_compare}: per-item overrides of the intake panel). The intake panel API (parse_inputs /
resolve / defaults / estimate / stream_resolved / add_resolved / save_defaults) is at the end of the UI API section. Finished outputs may carry compare tracks: get_state()
reports has_compare per done item and remove_compare() strips them. Headers are never written to job.json and never leave get_state().
"""

import json
import inspect
import math
import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

from oh_streamer.player import open_input

from oh_streamer.queue import queue_ipc

from oh_streamer.queue import queue_store

from oh_streamer.queue import perf_runner
from oh_streamer.queue import subtitle_jobs
from oh_streamer.queue import subtitle_player_status
from oh_streamer.queue import stream_audio
from oh_streamer.queue import press_timing
from oh_streamer.queue.subtitle_host_gpu import RUN_STAGE_LEASE, GpuSubtitleSchedulerMixin

from oh_streamer.queue import queue_thumbs

from oh_streamer.player import single_instance

from oh_streamer.player import perf_hint

from oh_streamer.sources import media_input, source_info

from oh_streamer.platform import platform_support


from oh_streamer.paths import checkout_root

HERE = checkout_root()

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

TICK = 0.2
EMIT_INTERVAL = 0.5             # on_state callbacks: at most two per second
PROGRESS_WRITE_INTERVAL = 1.0   # progress is persisted at most once a second
ETA_PROBE_RETRY = 60.0        # seconds before an item whose source facts could not be read is probed again
SPACE_RETRY_INTERVAL = 5.0
USAGE_REFRESH_INTERVAL = 5.0
DEFAULT_DOWNLOAD_ESTIMATE = 1024 ** 3
PROCESS_SIZE_FACTOR = 1.5
SPACE_NOTICE = "空間不足，佇列已暫停"
SHUTDOWN_DELAY = 60
SEGMENT_SECONDS = 4
RANGE_CACHE_SUFFIX = ".rcache"   # source_path marker of a byte-range (progressive http file) cache session
WORK_RETRY_INTERVAL = 5.0       # work/ dir still (partly) locked by the player: try again later
VRAM_SHORT_COUNT = 2            # T4.3: this many VRAM-short turns ...
VRAM_SHORT_WINDOW_S = 600.0     # ... within this many seconds -> the player's gpu_short hint
STREAM_WATCH_EXPIRE = 60.0      # a stream item whose last `watching` is older than this is dropped (player gone without stream_end)
STREAM_FIRST_WATCH_GRACE = 120.0   # a stream item that never got a `watching` this long after its creation is dropped
STREAM_HIGH_FPS = 31            # above this a source is "60 fps class" (restoration_fps applies, no interpolation row)

PENDING_STATES = ("waiting", "downloading", "queued", "processing", "paused")
MP4_AUDIO = {"aac", "mp3", "mp2", "ac3", "eac3", "alac", "opus"}
STATE_PRIORITY = ("done", "processing", "queued", "downloading", "waiting", "paused", "failed")


def log(msg):
    print(f"[queue_service] {msg}", file=sys.stderr, flush=True)


def _still_owns_pid(p: psutil.Process) -> bool:
    """True while p's pid is p itself, or nobody (a dead member's orphans still carry its pid as their ppid)."""
    try:
        return p.is_running() or not psutil.pid_exists(p.pid)
    except psutil.Error:
        return False


def _ppid_map() -> dict:
    """{pid: ppid} of every process from one system snapshot (~10 ms). process_iter(["ppid"]) gives the same but its
    first call in a process took 7-10 s with ~600 processes on Windows (it builds a Process per pid), long enough to
    stall the first worker stop."""
    snapshot = getattr(psutil, "_ppid_map", None)          # what Process.children() uses; private, so a fallback
    if snapshot is not None:
        try:
            return snapshot()
        except Exception:  # noqa: BLE001
            pass
    return {p.pid: p.info["ppid"] for p in psutil.process_iter(["ppid"])}


def _new_children(tree: dict) -> list:
    owners = {p.pid: p for p in tree.values() if _still_owns_pid(p)}
    found = []
    for pid, ppid in _ppid_map().items():
        parent = owners.get(ppid)
        if parent is None or pid in tree:
            continue
        try:
            c = psutil.Process(pid)
            if c.create_time() >= parent.create_time():     # older than its parent: the pid was reused
                found.append(c)
        except psutil.Error:
            pass
    return found


def kill_tree(pid: int, timeout: float = 5.0) -> None:
    """Kill a process we started and everything below it.

    A one-shot children() snapshot misses a child being created right then (the venv python.exe launcher creates the
    real interpreter suspended and resumes it a moment later; the worker starts ffmpeg). Killing its parent orphans it
    out of reach - on Windows frozen forever, holding the inherited worker.log handle so the item dir cannot be
    deleted. So members are suspended before anything is killed (a frozen process spawns nothing new), the tree is
    rescanned until stable, and rescanned again after the kill: a member caught inside CreateProcess finishes that
    call before its suspension or death takes effect, and a dead member's pid still names its orphans' parent."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    tree: dict = {}
    killed: list = []
    batch = [root]
    for _ in range(100):            # bounded: a member we may not kill could keep spawning
        if not batch:
            break
        for p in batch:
            tree[p.pid] = p
            try:
                p.suspend()
            except psutil.Error:
                pass
        batch = _new_children(tree)
        if batch:
            continue
        doomed = [p for p in tree.values() if p not in killed]
        for p in doomed:
            try:
                p.kill()
            except psutil.Error:
                try:
                    p.resume()          # never leave a process we could not kill frozen
                except psutil.Error:
                    pass
        killed += doomed
        psutil.wait_procs(doomed, timeout=timeout)
        batch = _new_children(tree)


def _win_set_execution_state(flags: int) -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def _default_run_shutdown(args: list[str]) -> None:
    if not platform_support.supports_shutdown():
        return
    subprocess.run(args, creationflags=NO_WINDOW, check=False)


def _default_startfile(path: str) -> None:
    platform_support.open_folder(path)


def _default_downloader(*args, **kwargs):
    from oh_streamer.queue import queue_download          # lazy: pulls in yt_dlp
    return queue_download.download(*args, **kwargs)


def _default_launch_player(path: str, payload: dict, player_name: Optional[str] = None) -> None:
    """Start `pythonw oh_player.py --player <path> --processed` detached. A payload that is not a finished file
    (progressive / live restore, which the command line cannot express) starts the player idle and then forwards the
    payload as soon as its listener answers."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    executable = str(pythonw if pythonw.exists() else exe)
    argv = ([executable, str(HERE / "oh_player.py")] if getattr(sys, "frozen", False) or
            (HERE / "oh_player.py").is_file() else
            [executable, "-m", "oh_streamer", "player"])
    argv += ["--player"]
    # Queue metadata (watch lease, external subtitles and error OSD) cannot be
    # represented by the plain processed-file command line.
    forward = not payload.get("processed") or any(payload.get(key) for key in (
        "queue_id", "subtitles", "subtitle_error", "subtitle_combos_dir"))
    argv += [] if forward else [path, "--processed"]
    start = queue_ipc.parse_start(payload.get("start"))
    if not forward and start:
        argv += ["--start", f"{start:.3f}"]
    if player_name:
        argv += ["--instance", player_name]
    # with DEVNULL std streams the player does not see "no console" and would not open its log file by itself
    log_dir = platform_support.data_dir() / "logs"
    argv += ["--log-file", str(log_dir / "oh_player.log")]
    subprocess.Popen(argv, **platform_support.detached_process_kwargs(), close_fds=True, cwd=str(HERE),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if forward:
        def deliver():
            for _ in range(120):
                time.sleep(1.0)
                if queue_ipc.send_to_player(payload, player_name, timeout=1.5) is not None:
                    return
            log("the launched player never answered; gave up forwarding the open request")
        threading.Thread(target=deliver, name="queue-launch-forward", daemon=True).start()


def _norm_quality(q) -> Any:
    if q is None:
        return None
    if isinstance(q, str):
        s = q.strip().lower()
        if s in ("", "best"):
            return None
        return int(s) if s.isdigit() else s
    return q


def _finite_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _round_or_none(v, digits: int):
    return round(float(v), digits) if _finite_num(v) else None


def video_facts(meta) -> Optional[dict]:
    """{"height", "fps", "duration"} from an ffprobe-style dict (HLS handle metadata / ffprobe -of json), else None."""
    if not isinstance(meta, dict):
        return None
    vs = next((s for s in meta.get("streams") or [] if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    if vs is None:
        return None
    try:
        height = int(vs.get("height") or 0)
        num, _, den = str(vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1)
        fmt = meta.get("format") if isinstance(meta.get("format"), dict) else {}
        duration = float(fmt.get("duration") or vs.get("duration") or 0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if height <= 0 or not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        return None
    return {"height": height, "fps": fps, "duration": duration}


def probe_video_facts(path: str) -> Optional[dict]:
    """video_facts() of a local file via ffprobe; None when it cannot be determined."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
                            "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=NO_WINDOW, timeout=60)
        return video_facts(json.loads(p.stdout)) if p.returncode == 0 and p.stdout else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class _Run:
    """One running download or worker: cancel event, why it was asked to stop, the process (workers)."""

    def __init__(self, item_id: str, generation: int = 0):
        self.item_id = item_id
        self.generation = generation
        self.cancel = threading.Event()
        self.reason: Optional[str] = None     # pause | paused_all | auto | remove | stop
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.source_ready = False
        self.stage = "restore"
        self.subtitle_only = False


# ------------------------------------------------------------------------------------------------------------
# export helpers (pure, tested)
# ------------------------------------------------------------------------------------------------------------

def probe_streams(path: str) -> Optional[list[tuple[str, str]]]:
    """[(codec_type, codec_name)] via ffprobe, None when it cannot be determined."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
                            "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=NO_WINDOW, timeout=60)
        if p.returncode != 0 or not p.stdout:
            return None
        return [(s.get("codec_type", ""), s.get("codec_name", "")) for s in json.loads(p.stdout).get("streams", [])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


COMPARE_TITLES = ("oh:original", "oh:restored")
PROCESSED_TITLE = "oh:processed"
DURATION_TOLERANCE = 0.5          # seconds: output vs input duration after a remux
PLAYING_ERROR = "播放中，請先關閉播放器"


def probe_tracks(path: str) -> Optional[dict]:
    """ffprobe of a finished output: {"streams": [{index, codec_type, codec_name, title}], "tags": {UPPER: value},
    "duration": seconds}. None when it cannot be determined."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=index,codec_type,codec_name:stream_tags=title:format=duration:format_tags",
                            "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=NO_WINDOW, timeout=60)
        if p.returncode != 0 or not p.stdout:
            return None
        data = json.loads(p.stdout)
        streams = []
        for s in data.get("streams", []):
            tags = {str(k).lower(): v for k, v in (s.get("tags") or {}).items()}
            streams.append({"index": s.get("index"), "codec_type": s.get("codec_type", ""),
                            "codec_name": s.get("codec_name", ""), "title": tags.get("title") or ""})
        fmt = data.get("format") or {}
        tags = {str(k).upper(): str(v) for k, v in (fmt.get("tags") or {}).items()}
        try:
            duration = float(fmt.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        return {"streams": streams, "tags": tags, "duration": duration}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def has_compare_tracks(info: Optional[dict]) -> bool:
    return bool(info) and any(s.get("codec_type") == "video" and s.get("title") in COMPARE_TITLES
                              for s in info.get("streams", []))


def build_export_cmd(src: str, dest: str, streams: Optional[list[tuple[str, str]]], force_aac: bool = False,
                     include_compare: bool = False, titles: Optional[list] = None) -> list[str]:
    """ffmpeg command remuxing the finished mkv to mp4: video + audio streams are copied (`-c copy`), hvc1 tag for
    HEVC, faststart; audio that mp4 cannot hold (or force_aac) is re-encoded to AAC. By default only the first
    (processed) video track is exported; include_compare keeps every video track, with hvc1 per HEVC track and the
    mkv track titles as the mp4 handler_name (`titles` lines up with the video streams)."""
    streams = streams or []
    vcodecs = [c for t, c in streams if t == "video"]
    acodecs = [c for t, c in streams if t == "audio"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-map", "0:v" if include_compare else "0:v:0", "-map", "0:a?", "-c", "copy"]
    if include_compare:
        for i, codec in enumerate(vcodecs):
            if codec in ("hevc", "h265"):
                cmd += [f"-tag:v:{i}", "hvc1"]
        for i, title in enumerate(titles or []):
            if title:
                cmd += [f"-metadata:s:v:{i}", f"handler_name={title}"]
    elif vcodecs and vcodecs[0] in ("hevc", "h265"):
        cmd += ["-tag:v", "hvc1"]
    if force_aac or any(c not in MP4_AUDIO for c in acodecs):
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", dest]
    return cmd


# ------------------------------------------------------------------------------------------------------------
# service
# ------------------------------------------------------------------------------------------------------------

class QueueService(GpuSubtitleSchedulerMixin, subtitle_jobs.SubtitleQueueMixin):
    def __init__(self, root, player_name: Optional[str] = None, launch_player: Optional[Callable] = None, *,
                 downloader: Optional[Callable] = None,
                 worker_command: Optional[Callable[[str], list]] = None,
                 python_exe: Optional[str] = None,
                 set_execution_state: Optional[Callable[[int], None]] = None,
                 run_shutdown: Optional[Callable[[list], None]] = None,
                 send_to_player: Optional[Callable] = None,
                 run_cmd: Optional[Callable] = None,
                 probe: Optional[Callable] = None,
                 probe_tracks: Optional[Callable] = None,
                 startfile: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 tick: float = TICK,
                 perf_profile_path=None,
                 perf_benchmark: Optional[Callable] = None,
                 perf_fingerprint: Optional[Callable[[], dict]] = None,
                 eta_probe: Optional[Callable[[str], Optional[dict]]] = None,
                 thumbs: Optional["queue_thumbs.ThumbManager"] = None,
                 source_resolver: Optional[Callable] = None,
                 thumb_grabber: Optional[Callable] = None):
        self.player_name = player_name
        self.store = queue_store.QueueStore(Path(root), on_change=self._changed)
        self.on_state: list[Callable[[dict], None]] = []
        self.on_show: list[Callable[[], None]] = []
        self._launch_player = launch_player or (lambda path, payload: _default_launch_player(path, payload, player_name))
        self._downloader = downloader or _default_downloader
        self._downloader_accepts_source_ready = self._accepts_source_ready(self._downloader)
        self._worker_command = worker_command or self._default_worker_command
        self._python = python_exe or sys.executable
        self._sleep_guard = platform_support.PreventSleep()
        self._set_exec = set_execution_state or (lambda flags: self._sleep_guard.set_active(bool(flags & ES_SYSTEM_REQUIRED)))
        self._run_shutdown = run_shutdown or _default_run_shutdown
        self._send_to_player = send_to_player or (lambda payload: queue_ipc.send_to_player(
            payload, player_name, timeout=1.5))
        self._run_cmd = run_cmd or (lambda cmd: subprocess.run(
            cmd, capture_output=True, text=True, creationflags=NO_WINDOW))
        self._probe = probe or probe_streams
        self._probe_tracks = probe_tracks or globals()["probe_tracks"]
        self._track_cache: dict[str, tuple] = {}      # output path -> ((mtime_ns, size), probe_tracks() result)
        self._compare_busy: set = set()               # item ids with a remove_compare in progress
        self._reprocessing: set = set()              # sources pinned while a new version is prepared
        self._eta_probe = eta_probe or probe_video_facts
        self._eta_facts: dict[str, tuple] = {}        # item id -> ((source_path, source_status), facts or None, stamp)
        self._eta_probing: dict = {}                  # item id -> background ffprobe thread in flight
        self._eta_cache: dict[str, tuple] = {}        # item id -> (inputs key, eta_estimate_s)
        self._source_resolver = source_resolver or source_info.resolve      # OpenInput -> source_info.Resolved
        self._thumb_grabber = thumb_grabber or self._grab_thumb             # Resolved -> data: URI | None
        self._tokens = source_info.TokenCache()       # resolve() handles of the intake panel (LRU, process lifetime)
        self._startfile = startfile or _default_startfile
        self._clock = clock                # only used for the player_busy expiry (tests inject a fake)
        self._tick = tick
        self.busy_expire = queue_ipc.BUSY_EXPIRE
        self.stream_watch_expire = STREAM_WATCH_EXPIRE
        self.stream_first_watch_grace = STREAM_FIRST_WATCH_GRACE
        self._stream_created: dict[str, float] = {}   # stream item id -> self._clock() when the queue first knew it

        self._lock = threading.RLock()
        # film-strip thumbnails (queue_thumbs.py): its own thread, started / stopped with the service
        self.thumbs = thumbs or queue_thumbs.ThumbManager(
            self.store, log=log, on_change=self._changed, hls_duration=self._hls_duration,
            enabled=lambda: bool(self.store.get_settings().get("show_thumbnails", True)))
        self._dl: Optional[_Run] = None
        self._wk: Optional[_Run] = None
        self._subtitle_cloud: Optional[_Run] = None
        self._subtitle_local: Optional[_Run] = None
        self._subtitle_stream_cloud: Optional[_Run] = None   # T3.1-A: dedicated cloud slot of the stream item
        self._subtitle_turn_hold: Optional[str] = None
        self._subtitle_plan_files: dict = {}       # item id -> ((mtime_ns, size), SlicePlan): view tick cache
        self._subtitle_status_sent: dict = {}      # item id -> (signature, sent_at, ok): last `subtitle_status` to the player
        self._lead_samples: dict = {}              # stream item id -> [(t, ahead_s)]: live restore lead (T4.3 gpu_short)
        self._vram_short_times: dict = {}          # stream item id -> [t]: turns given up for too little free VRAM
        self._subtitle_windows = []
        self._subtitle_clocks = {}
        self._subtitle_wants = {}
        self._subtitle_seek_guard = {}
        self._generations: dict[str, int] = {}
        self._tombstones: set[str] = set()
        self._sources: dict[str, Any] = {}              # live HLS handles; never persisted
        self._pending_play: Optional[str] = None
        self._pending_start: Optional[float] = None     # play(id, start) waiting for the source: where to begin
        self._press = press_timing.PressTiming()        # T1.3: 按播放 → 第一段可播 log (press stamp, worker reports)
        self._player_item: Optional[str] = None
        self._opening_until = 0.0  # cover detached player startup before its first GPU heartbeat
        self._worker_waiting_for_source: set[str] = set()
        self._paused_all = False
        self._busy = False
        self._busy_ts = 0.0
        self._live: Optional[dict] = None           # player_busy "live": the live stream report of the 「串流中」 card
        self._live_ts = 0.0
        self._playing = False                       # player_playing: mpv plays anything (no effect on scheduling)
        self._playing_ts = 0.0
        self._player_gpu_last: Optional[float] = None   # last time the player was known busy or playing
        self._busy_published = False
        # performance check (perf_runner.py): its `holding` flag stops the worker like pause_all, but is not pause_all
        self.perf = perf_runner.PerfRunner(
            settings_provider=lambda: self._snapshot_settings(), is_player_busy=self.is_player_busy,
            notify=self._changed, profile_path=perf_profile_path, run_benchmark=perf_benchmark,
            fingerprint_provider=perf_fingerprint, notify_player=self.reload_player_settings)
        self._watch: dict[str, float] = {}            # item id -> clock() of the last want / watching
        self._start_k: dict[str, int] = {}            # item id -> start_k for its next worker run
        self._work_cleanup: set[str] = set()          # done items whose work/ dir still has to go
        self._work_retry: dict[str, float] = {}
        self._space_blocked = {"download": False, "process": False}
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        self._shutdown_notice = ""
        self._shutdown_started = False
        self._active_seen = False
        self._sleep_on = False
        self._dirty = threading.Event()
        self._last_emit = 0.0
        self._usage_at = 0.0
        self._usage = 0
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._instance = None
        if not platform_support.supports_shutdown() and self.store.get_settings().get("shutdown_when_done"):
            self.store.set_settings(shutdown_when_done=False)
        # Probe the development interpreter before opening the short-timeout IPC
        # listener, never on its first add/get_state request.
        self._subtitle_environment()

    # --------------------------------------------------------------------------------------------------------
    # lifecycle
    # --------------------------------------------------------------------------------------------------------

    @staticmethod
    def _accepts_source_ready(fn: Callable) -> bool:
        """Detect the new downloader callback without breaking injected six-argument test/downloader functions."""
        try:
            params = inspect.signature(fn).parameters.values()
            return any(p.name == "on_source_ready" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except (TypeError, ValueError):
            return False

    def _next_generation(self, item_id: str) -> int:
        with self._lock:
            generation = self._generations.get(item_id, 0) + 1
            self._generations[item_id] = generation
            return generation

    def _run_is_current(self, run: _Run) -> bool:
        with self._lock:
            return (run.item_id not in self._tombstones
                    and self._generations.get(run.item_id, run.generation) == run.generation)

    def start(self) -> bool:
        """Acquire the IPC instance and start scheduling. False when another queue app already owns the instance."""
        inst = single_instance.acquire(queue_ipc.queue_instance_name(self.player_name))
        if inst is None:
            return False
        self._instance = inst
        for it in self.store.items():                 # work/ dirs left behind by an earlier run
            # A localhost HLS proxy is owned by the old process. A partial cache must be downloaded again so a
            # fresh proxy can serve it; completed disk caches can instead be reopened on demand.
            if it.source_status == "downloading" and it.state != "paused":
                self.store.update(it.id, state="waiting", source_status="paused")
            elif it.source_status == "paused" and it.state in ("queued", "processing", "downloading"):
                self.store.update(it.id, state="waiting")
            if it.state == "done" and os.path.isfile(it.output_path) and (self.store.item_dir(it.id) / "work").exists():
                self._work_cleanup.add(it.id)
        try:
            inst.serve(self._on_message, log=log)
            self._stopping.clear()
            self._thread = threading.Thread(target=self._loop, name="queue-scheduler", daemon=True)
            self._thread.start()
            self.thumbs.start()
        except BaseException:
            inst.close()
            self._instance = None
            self._thread = None
            raise
        return True

    def stop(self) -> None:
        """Stop scheduling, stop running work (items go back to waiting / queued), release the IPC instance."""
        self._stopping.set()
        self._wake.set()
        self.thumbs.stop()
        self.perf.cancel()
        with self._lock:
            runs = [r for r in (self._dl, self._wk, *self._subtitle_runs()) if r]
        for r in runs:
            self._request_stop(r, "stop")
        for item_id in {r.item_id for r in self._subtitle_runs()}:
            self._subtitle_stop_executors(item_id, "stop")
        for r in runs:
            if r.thread:
                r.thread.join(timeout=15)
        self.perf.join(timeout=15)
        if self._thread:
            self._thread.join(timeout=5)
        self._cleanup_source_sessions(self._watched_ids(), force=True)
        try:
            for marker in self._stream_markers():     # the url refreshers live in this process: drop them with the items
                media_input.unregister_refresher(marker)
        except Exception as e:  # noqa: BLE001
            log(f"stream items: unregistering refreshers at stop failed: {e!r}")
        try:
            dropped = self.store.clear_streams()      # stream items live for one play: nothing of them survives the queue
        except Exception as e:  # noqa: BLE001 - a leftover is cleared at the next start anyway
            dropped = 0
            log(f"stream items: clearing at stop failed: {e!r}")
        if dropped:
            log(f"stream items: {dropped} cleared at stop")
        with self._lock:
            self._stream_created.clear()
        if self._instance:
            self._instance.close()
            self._instance = None

    # --------------------------------------------------------------------------------------------------------
    # scheduler
    # --------------------------------------------------------------------------------------------------------

    def _changed(self) -> None:
        self._dirty.set()
        self._wake.set()
        self.thumbs.poke()

    def _hls_duration(self, it: queue_store.Item) -> Optional[float]:
        """Duration hint for the thumbnails of a progressive HLS source (its stored probe), else None."""
        with self._lock:
            handle = self._sources.get(it.id)
        facts = video_facts(self._source_field(handle, "metadata")) if handle is not None else None
        return facts["duration"] if facts else None

    def _loop(self) -> None:
        try:
            while not self._stopping.is_set():
                self._wake.wait(self._tick)
                self._wake.clear()
                try:
                    self._step()
                except Exception as e:  # noqa: BLE001 - the scheduler must survive anything
                    log(f"scheduler error: {e!r}")
        finally:
            if self._sleep_on:
                try:
                    self._set_exec(ES_CONTINUOUS)
                finally:
                    self._sleep_on = False
                    self._sleep_guard.close()

    def is_player_busy(self) -> bool:
        with self._lock:
            return (self._clock() < self._opening_until or
                    (self._busy and (self._clock() - self._busy_ts) < self.busy_expire))

    def is_player_playing(self) -> bool:
        with self._lock:
            return self._playing and (self._clock() - self._playing_ts) < self.busy_expire

    def _note_player_gpu(self) -> None:
        """Remember now as a moment the player used the GPU, if its last report (busy or playing) says it did.
        Called before and after each report, so the moment playback stops is recorded too. Caller holds the lock."""
        now = self._clock()
        if (self._busy and now - self._busy_ts < self.busy_expire) or \
                (self._playing and now - self._playing_ts < self.busy_expire):
            self._player_gpu_last = now

    def player_used_gpu_since(self, t: float) -> bool:
        """Whether the player was restoring or playing anything at some point from clock time `t` on (one heartbeat of
        margin: a player that died sends no final report)."""
        if self.is_player_busy() or self.is_player_playing():
            return True
        with self._lock:
            last = self._player_gpu_last
        return last is not None and last + queue_ipc.BUSY_HEARTBEAT >= t

    @staticmethod
    def _hls_cache_module():
        """Late import keeps the queue usable with older installs and custom non-HLS downloaders."""
        try:
            from oh_streamer.sources import hls_cache
            return hls_cache
        except ImportError:
            return None

    @staticmethod
    def _range_cache_module():
        try:
            from oh_streamer.sources import range_cache
            return range_cache
        except ImportError:
            return None

    def _registry_for(self, path):
        """The cache module owning a source path: byte-range sessions (.rcache marker) or HLS playlists."""
        if str(path or "").lower().endswith(RANGE_CACHE_SUFFIX):
            return self._range_cache_module()
        return self._hls_cache_module()

    def _session_for(self, item: queue_store.Item, *, reopen_complete: bool = True):
        if not item.source_path or item.source_status not in ("downloading", "complete"):
            return None
        with self._lock:
            handle = self._sources.get(item.id)
        if handle is not None:
            return handle
        registry = self._registry_for(item.source_path)
        if registry is None:
            return None
        try:
            handle = registry.get_session(item.source_path)
            if handle is None and reopen_complete and item.source_status == "complete":
                handle = registry.open_session(item.source_path)
        except Exception as e:  # noqa: BLE001 - a cache may have been removed while the queue was closed
            log(f"could not open HLS source for {item.id}: {e!r}")
            return None
        if handle is not None:
            with self._lock:
                if item.id not in self._tombstones:
                    self._sources[item.id] = handle
        return handle

    def _source_ready(self, item: queue_store.Item) -> bool:
        if not item.source_path or not os.path.isfile(item.source_path):
            return False
        if item.source_status in ("downloading", "complete"):
            handle = self._session_for(item, reopen_complete=True)
            if handle is None:
                return False
            # An item wrongly marked complete while its cache is only partial must download the rest again.
            return not (item.source_status == "complete"
                        and self._source_field(handle, "status") in ("downloading", "paused", "failed"))
        return item.source_status == ""

    @staticmethod
    def _source_field(handle, name, default=None):
        if isinstance(handle, dict):
            return handle.get(name, default)
        return getattr(handle, name, default)

    @staticmethod
    def _subtitle_source_pending(item):
        """The slice executors still read the original source (ASR of unfinished slices), also while restoration
        is done: the source file must stay until the slice table is complete."""
        if not item:
            return False
        sub = item.subtitle
        if not sub.get("status") or sub.get("status") in (
                "ready", "mux_pending", "done", "failed", "cancelled", "skipped"):
            return False
        view = sub.get("view")
        return not (isinstance(view, dict) and view.get("state") == "complete")

    # -- subtitle view (design 6.1 M5: item.subtitle.view is a pure function of the slice table) ------------------
    VIEW_COUNTDOWN_KEYS = ("remaining_s", "reclaim_in_s")

    @classmethod
    def _subtitle_view_signature(cls, view):
        """The view without its per-second countdown fields: only a real change is worth a queue.json write."""
        if not isinstance(view, dict):
            return None
        out = dict(view)
        wait = out.get("wait")
        if isinstance(wait, dict):
            out["wait"] = {k: v for k, v in wait.items() if k not in cls.VIEW_COUNTDOWN_KEYS}
        return out

    @staticmethod
    def _subtitle_synthetic_complete_view(item) -> dict:
        """Items finished by the pre-M7 pipelines have no slice table; their subtitles are complete anyway."""
        return {"v": 2, "state": "complete", "progress": 1.0, "covered_s": item.subtitle.get("covered_s") or 0,
                "slices_done": 0, "slices_total": 0, "open": False, "running": None, "wait": None, "stuck": None,
                "failures_elsewhere": 0, "partial_stuck": False, "retryable": False}

    def _subtitle_plan_cached(self, item_id: str):
        """The live in-memory slice table while an executor holds it, else the file (re-read only when it
        changed on disk)."""
        from oh_streamer.subtitles import slice_plan as sp
        from . import subtitle_scheduler as ss
        folder = self.store.item_dir(item_id) / "subtitles"
        live = ss.PlanStore.peek(folder)
        if live is not None and live.plan is not None:
            return live.plan
        path = folder / sp.SLICE_PLAN_FILE
        try:
            stat = path.stat()
        except OSError:
            self._subtitle_plan_files.pop(item_id, None)
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._subtitle_plan_files.get(item_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        plan = sp.load_slice_plan(folder)
        self._subtitle_plan_files[item_id] = (key, plan)
        return plan

    def _subtitle_gpu_turn(self, item) -> Optional[bool]:
        """False while the restoration worker holds the GPU and the local executor has to wait for its turn
        (interleave mode); True otherwise (lease held, GPU free, or parallel mode)."""
        if getattr(item, "kind", "") == queue_store.STREAM_KIND:
            # A stream item has no restoration worker of its own: the player is the GPU user (live restoration). While
            # it is busy the local executor never starts for the item (T3.1-2; P4 wires local subtitles for streams).
            return not self.is_player_busy()
        with self._lock:
            wk = self._wk
        if wk is None or wk.stage != "restore" or wk.cancel.is_set():
            return True
        try:
            from oh_streamer.queue import gpu_turns as gt
            if self._subtitle_resource_mode(item) == gt.MODE_PARALLEL:
                return True
        except Exception:                                   # noqa: BLE001 - the mode only refines the answer
            pass
        return False

    def _subtitle_publish_update(self, item, result):
        """No `subtitle_update` for an item that is being removed: a stream item's executor thread may still publish
        in the moment between the tombstone and the end of the join."""
        if item is None or item.id in self._tombstones:
            return
        super()._subtitle_publish_update(item, result)

    def _subtitle_view_for(self, item, now: float):
        """The item's view dict, or None when it has no slice table (and no legacy completion to report)."""
        from oh_streamer.subtitles import fallback as fb
        from . import subtitle_view as svw
        plan = self._subtitle_plan_cached(item.id)
        if plan is None:
            if item.subtitle.get("status") in ("ready", "mux_pending", "done"):
                return self._subtitle_synthetic_complete_view(item)
            return None
        clock = float(self._subtitle_position(item))
        offset = item.subtitle.get("audio_offset") or 0.0
        try:
            play_pos = max(0.0, clock - float(offset))
        except (TypeError, ValueError):
            play_pos = max(0.0, clock)
        max_available = None
        if item.source_status == "downloading":
            usable, limit, _total = self._subtitle_cloud_source_state(item)
            max_available = limit if usable else 0.0
        frozen = item.settings.get("subtitle_resolved")
        frozen = frozen if isinstance(frozen, dict) else None
        owner = None
        names = None
        if frozen is not None:
            try:
                owner = fb.owner_function(frozen)
                names = svw.candidate_names(frozen)
            except Exception:                                # noqa: BLE001 - a view without names beats no view
                owner = names = None
        return svw.subtitle_view(plan, play_pos=play_pos, now=now, max_available=max_available,
                                 gpu_turn=self._subtitle_gpu_turn(item), candidate_names=names, owner=owner)

    def _subtitle_view_tick(self) -> None:
        """Refresh item.subtitle.view for every subtitle item; writes (and so emits) only on a real change."""
        now = self._subtitle_time()
        for item in [*self.store.items(), *self.store.stream_items()]:     # stream items too (T3.1-C)
            if not item.settings.get("subtitles") or item.id in self._tombstones:
                continue
            try:
                self._subtitle_reconcile_complete(item)
                view = self._subtitle_view_for(item, now)
            except Exception as exc:                         # noqa: BLE001 - never break the scheduler loop
                log(f"subtitle view for {item.id} failed: {exc!r}")
                continue
            current = item.subtitle.get("view")
            if view is not None:
                self._subtitle_usage_sample(item, view)
            if view is None:
                if current is not None:                      # table deleted (restart): no stale view
                    with self._lock:
                        latest = self.store.get(item.id)
                        if latest is not None:
                            self.store.update(item.id, subtitle={k: v for k, v in latest.subtitle.items() if k != "view"})
                continue
            if self._subtitle_view_signature(view) == self._subtitle_view_signature(current):
                continue
            self._subtitle_update(item.id, view=view)
        self._subtitle_status_tick(now)

    STATUS_KEEPALIVE_S = 30.0      # subtitle_status is sent again this often (a player that opened later has none yet)
    STATUS_RETRY_S = 5.0           # ... and this soon after a send the player did not accept (no player running)

    def _subtitle_status_tick(self, now: float) -> None:
        """T3.3: tell the player how the subtitle job of the video it plays is doing (`subtitle_status`, queue_ipc.py).
        Only items the player reported (watching: stream items and progressive items) and only on a change of the
        state, a keep-alive every STATUS_KEEPALIVE_S, a retry after STATUS_RETRY_S when the player did not take it."""
        with self._lock:
            watched = set(self._watch)
        sent = self._subtitle_status_sent
        for gone in [i for i in sent if i not in watched or i in self._tombstones]:
            sent.pop(gone, None)
        for gone in [i for i in self._lead_samples if i not in watched or i in self._tombstones]:
            self._lead_samples.pop(gone, None)
        for item_id in watched:
            item = self.store.get(item_id)
            if item is None or item_id in self._tombstones or not item.settings.get("subtitles"):
                continue
            try:
                offset = getattr(self, "_subtitle_play_offsets", {}).get(item_id, item.subtitle.get("audio_offset", 0))
                msg = subtitle_player_status.player_status(item_id, item.subtitle, offset=offset or 0.0)
                if msg is None:
                    continue
                if self._gpu_short(item, msg, now):
                    msg["gpu_short"] = True
                signature = subtitle_player_status.signature(msg)
                last = sent.get(item_id)
                if last is not None:
                    old_signature, at, ok = last
                    wait = self.STATUS_KEEPALIVE_S if ok else self.STATUS_RETRY_S
                    if signature == old_signature and now - at < wait:
                        continue
                reply = self._send_to_player(msg)
                sent[item_id] = (signature, now, bool(isinstance(reply, dict) and reply.get("ok")))
            except Exception as exc:                         # noqa: BLE001 - a status hint never breaks the scheduler loop
                sent[item_id] = (None, now, False)
                log(f"subtitle status for {item_id} could not reach the player: {exc!r}")

    def _note_lead_sample(self, live: Optional[dict]) -> None:
        """T4.3: keep (t, ahead_s) of the playing stream's restore lead (caller holds the lock). Only while it plays and
        restores normally: a pause, a finished restore or no report forgets the samples (a falling lead then is no sign)."""
        now = self._subtitle_time()
        item_id = live.get("stream_item") if live else None
        pause = live.get("restore_pause") if live else None
        ok = (item_id and live.get("playing") and not live.get("restore_done") and live.get("ahead_s") is not None
              and not (isinstance(pause, dict) and pause.get("state") not in (None, "running")))
        for other in [i for i in self._lead_samples if i != item_id or not ok]:
            self._lead_samples.pop(other, None)
        if not ok:
            return
        samples = self._lead_samples.setdefault(item_id, [])
        samples.append((now, live["ahead_s"]))
        keep = subtitle_player_status.LEAD_KEEP_S
        while len(samples) > 2 and now - samples[1][0] > keep:
            samples.pop(0)

    def _gpu_short(self, item, msg: dict, now: float) -> bool:
        """T4.3: the stream item's local subtitles and the live restore fight for this GPU (see subtitle_player_status)."""
        if item.kind != queue_store.STREAM_KIND:
            return False
        from .subtitle_host_gpu import stream_turn_config
        with self._lock:
            samples = list(self._lead_samples.get(item.id, ()))
        return subtitle_player_status.gpu_short(msg, samples, now, self._subtitle_position(item),
                                                stream_turn_config(item.settings).urgent_s,
                                                vram_short=self._vram_short_count(item.id, now) >= VRAM_SHORT_COUNT)

    def _vram_short_count(self, item_id: str, now: float) -> int:
        """Turns of the stream item given up for too little free VRAM within the last VRAM_SHORT_WINDOW_S."""
        with self._lock:
            times = [t for t in self._vram_short_times.get(item_id, ()) if now - t <= VRAM_SHORT_WINDOW_S]
            if times:
                self._vram_short_times[item_id] = times
            else:
                self._vram_short_times.pop(item_id, None)
        return len(times)

    def _stream_vram_short_note(self, item_id: str) -> None:
        """The stream item's local executor gave a turn up because free VRAM was short (counts toward gpu_short)."""
        with self._lock:
            self._vram_short_times.setdefault(item_id, []).append(self._subtitle_time())

    def _subtitle_usage_sample(self, item, view) -> None:
        """Usage log: playback / coverage sample (state change, else every 30 s). Never raises."""
        try:
            from oh_streamer.subtitles import usage_log
            if not usage_log.view_due(item.id, view):
                return
            watching = item.id in self._subtitle_clocks
            pos = self._subtitle_position(item) if watching else None
            restore = (pos + self._subtitle_restore_lead(item, pos)) if watching else None
            if restore is not None and restore == float("inf"):
                restore = None
            avail = None
            if item.source_status == "downloading":
                usable, limit, _total = self._subtitle_cloud_source_state(item)
                avail = limit if usable else 0.0
            usage_log.view_sample(item.id, view, play_pos=pos, restore_end=restore, download_end=avail)
        except Exception:                                    # noqa: BLE001
            pass

    def _subtitle_has_table(self, item) -> bool:
        """The item's subtitles are driven by a slice table (the player's clock is then worth sending)."""
        if not item or not item.settings.get("subtitles"):
            return False
        if isinstance(item.subtitle.get("view"), dict):
            return True
        return (self.store.item_dir(item.id) / "subtitles" / "slice-plan.json").is_file()

    def _subtitle_srt_exists(self, item_id: str) -> bool:
        return (self.store.item_dir(item_id) / "subtitles" / "sakura.zh-TW.srt").is_file()

    def _subtitle_mux_ready(self, item) -> bool:
        """The finished subtitles can be muxed: the slice table is complete (items of the old pipelines have no
        table: their final srt alone counts) and the primary srt exists (R11)."""
        if not self._subtitle_srt_exists(item.id):
            return False
        plan = self._subtitle_plan_cached(item.id)
        return plan is None or self._subtitle_plan_complete(item.id)

    def _subtitle_reconcile_complete(self, item) -> None:
        """The slice table is complete and the final srt exists, but the status never caught up (the app was
        closed between the last slice and the publish): finish the status so the mux can run (R11)."""
        sub = item.subtitle
        if sub.get("status") not in ("queued", "waiting_gpu", "asr", "translate", "partial"):
            return
        if any(r.item_id == item.id for r in self._subtitle_runs() if r):
            return
        if not self._subtitle_plan_complete(item.id) or not self._subtitle_srt_exists(item.id):
            return
        if item.state == "done":
            self._subtitle_update(item.id, status="mux_pending", only=True, percent=None, error="")
        else:
            self._subtitle_update(item.id, status="ready", percent=100, error="")
        self._wake.set()

    def _step(self) -> None:
        self._expire_streams()
        self._maybe_play_after_subtitles()
        busy = self.is_player_busy()
        if busy != self._busy_published:
            self._busy_published = busy
            self._dirty.set()
        watched = self._watched_ids()
        pri = self._priority_item(watched)
        if not busy:
            pri = next((it for it in self.store.items() if it.play_after_subtitles
                        and it.state in ("queued", "processing") and self._source_ready(it)
                        ), pri)
        if busy:
            self.perf.cancel_for_player()            # live processing would disturb the measurement
        hold = self.perf.holding                     # performance check running: no worker, but not the user's pause
        with self._lock:
            dl, wk, paused_all = self._dl, self._wk, self._paused_all
        if wk:
            # the live stream item's OWN subtitle lease runs beside the player's restoration (T4.2): busy never stops it
            own_lease = wk.stage == RUN_STAGE_LEASE and busy and not self._player_busy_for(wk.item_id)
            if paused_all or hold:                   # the user's pause wins, also over a watched item
                self._request_stop(wk, "paused_all")
            elif busy and wk.stage != "restore" and not own_lease:
                self._request_stop(wk, "auto")
            elif pri is not None and wk.item_id != pri.id:
                self._request_stop(wk, "preempt")
            elif pri is None and busy and not own_lease:
                self._request_stop(wk, "auto")
        if dl and paused_all:
            self._request_stop(dl, "paused_all")
        if not self._stopping.is_set():
            if not paused_all and not dl:
                self._maybe_start_download()
            if not wk and not paused_all and not hold:
                if pri is not None:                  # being watched: runs first, even while player_busy
                    if pri.state != "done":
                        self._maybe_start_worker(pri)
                    if not busy and not self._subtitle_turn_held():
                        self._maybe_start_subtitle_only()
                elif not busy:
                    self._maybe_start_worker()
                    if not self._subtitle_turn_held():
                        self._maybe_start_subtitle_only()
        self._subtitle_executor_tick()                # cloud / local subtitle executors 
        self._subtitle_view_tick()
        self._clean_work_dirs(watched)
        self._cleanup_source_sessions(watched)
        self._update_sleep()
        self._check_shutdown()
        self._emit_if_due()

    def _watched_ids(self) -> set:
        """Items the player reported (want / watching) less than busy_expire s ago; also tells the store to keep them."""
        with self._lock:
            watched = self._live_watched_locked()
        with self._lock:
            self.store.protected_ids = (
                watched | self._reprocessing | self._subtitle_reserved_ids()
            )
        return watched

    def _live_watched_locked(self) -> set:
        """Return current heartbeat leases; caller holds ``_lock``."""
        now = self._clock()
        if self._pending_play and self.store.get(self._pending_play) is not None:
            self._watch[self._pending_play] = now
        for item_id in [item_id for item_id, stamp in self._watch.items()
                        if now - stamp >= self.busy_expire]:
            del self._watch[item_id]
        return set(self._watch)

    def _priority_item(self, watched: set) -> Optional[queue_store.Item]:
        """The pending-play item first, otherwise the most recently watched source-ready item."""
        best, best_ts = None, -1.0
        with self._lock:
            stamps = dict(self._watch)
            pending = self._pending_play
        if pending in watched:
            it = self.store.get(pending)
            if it and it.state in ("queued", "processing") and self._source_ready(it):
                return it
        for it in self.store.items():
            if (it.id in watched and it.state in ("queued", "processing") and self._source_ready(it)
                    and stamps.get(it.id, 0.0) > best_ts):
                best, best_ts = it, stamps.get(it.id, 0.0)
        return best

    def _cleanup_source_sessions(self, watched: set, *, force: bool = False) -> None:
        """Release safe idle proxies and reclaim completed source caches."""
        with self._lock:
            active = {r.item_id for r in (self._dl, self._wk, *self._subtitle_runs()) if r}
            current_watched = self._live_watched_locked()
            idle = []
            for item_id in self._sources:
                if (force and item_id not in active and item_id not in watched and item_id not in current_watched):
                    idle.append(item_id)
            handles = [(item_id, self._sources.pop(item_id)) for item_id in idle]
        for item_id, handle in handles:
            path = self._source_field(handle, "playlist_path")
            if not path:
                item = self.store.get(item_id)
                path = item.source_path if item else ""
            registry = self._registry_for(path)
            try:
                if registry and path:
                    registry.close_session(path)
                else:
                    close = getattr(handle, "close", None)
                    if close:
                        close()
            except Exception as e:  # noqa: BLE001
                log(f"could not close HLS source for {item_id}: {e!r}")

        # A completed queue item plays output.mkv on a fresh play, while an existing progressive player may still
        # read the original audio. Do not close proxies for queued/processing items here: resume/play can race a
        # close and a partial cache cannot be reopened as a live proxy.
        candidates = [it for it in self.store.items()
                     if it.state == "done" and it.source_status == "complete" and it.source_path]
        retired_dirs: list[Path] = []
        with self._lock:
            active = {r.item_id for r in (self._dl, self._wk, *self._subtitle_runs()) if r}
            current_watched = self._live_watched_locked()
            for candidate in candidates:
                if (candidate.id in active or candidate.id in watched or candidate.id in current_watched
                        or candidate.id in self._tombstones or candidate.id in self._reprocessing):
                    continue
                item = self.store.get(candidate.id)
                if not item or item.state != "done" or item.source_status != "complete" or not item.source_path:
                    continue
                if self._subtitle_source_pending(item):
                    continue
                item_dir = self.store.item_dir(item.id)
                is_range = item.source_path.lower().endswith(RANGE_CACHE_SUFFIX)
                cache_name = "range-cache" if is_range else "hls-cache"
                registry = self._registry_for(item.source_path)
                cache_dir = item_dir / cache_name
                retired: Optional[Path] = None
                try:
                    item_root = item_dir.resolve()
                    expected_cache = item_root / cache_name
                    playlist = expected_cache / ("source.rcache" if is_range else "playlist.m3u8")
                    if cache_dir.is_symlink() or cache_dir.resolve() != expected_cache:
                        continue
                    if Path(item.source_path).resolve() != playlist or not _inside(str(playlist), item_dir):
                        continue
                    # Hold the queue lock while the done item's last source lease is closed. A concurrent player
                    # heartbeat is serialized with this check; queued/processing items never enter this path.
                    if registry:
                        registry.close_session(item.source_path)
                    self._sources.pop(item.id, None)
                    retired = item_dir / f".{cache_name}-delete-{time.time_ns()}"
                    if cache_dir.exists():
                        os.replace(cache_dir, retired)
                        retired_dirs.append(retired)
                    self.store.update(item.id, source_path="", source_status="", download_progress={})
                except Exception as e:  # noqa: BLE001 - preserve the source if proxy shutdown or disk cleanup fails
                    if retired is not None and retired.exists() and not cache_dir.exists():
                        try:
                            os.replace(retired, cache_dir)
                            if retired in retired_dirs:
                                retired_dirs.remove(retired)
                        except OSError:
                            pass
                    log(f"could not retire completed HLS cache for {item.id}: {e!r}")

        # Retry any previous cleanup interrupted by a locked file or process shutdown.
        for item in self.store.items():
            item_dir = self.store.item_dir(item.id)
            try:
                retired_dirs.extend(item_dir.glob(".hls-cache-delete-*"))
                retired_dirs.extend(item_dir.glob(".range-cache-delete-*"))
            except OSError:
                pass
        for retired in set(retired_dirs):
            shutil.rmtree(retired, ignore_errors=True)

    def _cancel_close_source(self, item_id: str) -> None:
        with self._lock:
            handle = self._sources.pop(item_id, None)
        item = self.store.get(item_id)
        if handle is None and item is not None:
            handle = self._session_for(item, reopen_complete=False)
            with self._lock:
                self._sources.pop(item_id, None)
        if handle is None:
            return
        try:
            cancel = getattr(handle, "cancel", None)
            if cancel:
                cancel()
        except Exception as e:  # noqa: BLE001
            log(f"could not cancel HLS source for {item_id}: {e!r}")
        path = self._source_field(handle, "playlist_path") or (item.source_path if item else "")
        registry = self._registry_for(path)
        try:
            if registry and path:
                registry.close_session(str(path))
            else:
                close = getattr(handle, "close", None)
                if close:
                    close()
        except Exception as e:  # noqa: BLE001
            log(f"could not close HLS source for {item_id}: {e!r}")

    def _clean_work_dirs(self, watched: set) -> None:
        """Delete the work/ dir of finished items nobody watches; a dir the player still holds open is retried."""
        with self._lock:
            pending = list(self._work_cleanup)
        now = time.monotonic()
        for item_id in pending:
            if item_id in watched or self._running_runs(item_id) or now < self._work_retry.get(item_id, 0.0):
                continue
            it = self.store.get(item_id)
            if self._subtitle_source_pending(it):
                continue
            if it is not None and it.state != "done":
                finished = True
            else:
                work = self.store.item_dir(item_id) / "work"
                # Progressive playback/casting still reads source audio after AI
                # completion. Delete the owned source only after its viewers leave.
                src = it.source_path if it is not None else None
                if (src and it.source_status == "" and _inside(src, self.store.item_dir(item_id))
                        and item_id not in self._reprocessing):
                    try:
                        os.remove(src)
                        self.store.update(item_id, source_path="")
                    except FileNotFoundError:
                        self.store.update(item_id, source_path="")
                    except OSError:
                        self._work_retry[item_id] = now + WORK_RETRY_INTERVAL
                        continue
                shutil.rmtree(work, ignore_errors=True)
                finished = not work.exists()
            if finished:
                with self._lock:
                    self._work_cleanup.discard(item_id)
                self._work_retry.pop(item_id, None)
            else:
                self._work_retry[item_id] = now + WORK_RETRY_INTERVAL

    def _space_ok(self, stage: str, needed: int) -> bool:
        now = time.monotonic()
        if self._space_blocked[stage] and now < self._space_retry_at[stage]:
            return False
        ok = self.store.ensure_space(int(needed))
        was = self._space_blocked[stage]
        self._space_blocked[stage] = not ok
        self._space_retry_at[stage] = now + SPACE_RETRY_INTERVAL
        if (not ok) != was:
            self._dirty.set()
        return ok

    def _maybe_start_download(self) -> None:
        item = next((it for it in self.store.items() if it.play_after_subtitles and it.state == "waiting"),
                    self.store.next_runnable("waiting"))
        if item is None:
            self._space_blocked["download"] = False
            return
        if item.id in self._reprocessing:
            return
        if item.source_status == "complete":
            if self._source_ready(item):
                self.store.update(item.id, state="queued")
                self._wake.set()
                return
            # A completed HLS sidecar which can no longer be reopened needs a fresh proxy/cache run.
            self.store.update(item.id, source_status="paused")
            item = self.store.get(item.id)
        if item.kind != "file":
            est = item.download_progress.get("total_bytes") or item.progress.get("total_bytes") or DEFAULT_DOWNLOAD_ESTIMATE
            if not self._space_ok("download", est):
                return
        self._space_blocked["download"] = False
        generation = self._next_generation(item.id)
        run = _Run(item.id, generation)
        with self._lock:
            if self._dl is not None or item.id in self._tombstones or item.id in self._reprocessing:
                return
            item = self.store.get(item.id)
            if item is None or item.state != "waiting":
                return
            run.thread = threading.Thread(target=self._run_download, args=(item, run), name=f"queue-dl-{item.id}",
                                          daemon=True)
            self.store.update(item.id, state="downloading", error="",
                              source_status="downloading" if self._downloader_accepts_source_ready else "")
            self._dl = run
            run.thread.start()

    def _process_estimate(self, item: queue_store.Item) -> int:
        try:
            path = item.source_path
            if path.lower().endswith(RANGE_CACHE_SUFFIX):       # the marker is tiny: the data file has the real size
                path = str(Path(path).with_name("source.data"))
            return int(os.path.getsize(path) * PROCESS_SIZE_FACTOR)
        except OSError:
            return DEFAULT_DOWNLOAD_ESTIMATE

    def _maybe_start_worker(self, item: Optional[queue_store.Item] = None) -> None:
        if item is None:
            # Subtitle-enabled items start like any other, also while the source downloads: restoration
            # overlaps the download and subtitles follow the downloaded range (or wait for 等待下載).
            for candidate in self.store.items():
                if candidate.state != "queued":
                    continue
                item = candidate
                break
        if item is None:
            self._space_blocked["process"] = False
            return
        if item.id in self._worker_waiting_for_source:
            return
        if not self._source_ready(item):
            if item.source_status == "complete":
                self.store.update(item.id, state="waiting", source_status="paused")
                self._wake.set()
            elif item.source_status in ("downloading", "paused", "failed"):
                with self._lock:
                    downloading_active = self._dl is not None and self._dl.item_id == item.id
                if not downloading_active:
                    self.store.update(item.id, state="waiting", source_status="paused")
                    self._wake.set()
            elif not item.source_path or not os.path.isfile(item.source_path):
                if item.kind == "file":
                    self.store.update(item.id, state="failed", error="找不到來源檔案")
                elif item.source_status == "":
                    self.store.update(item.id, state="waiting", source_path="")
            return
        if not item.source_path or not os.path.isfile(item.source_path):
            if item.kind == "file":
                self.store.update(item.id, state="failed", error="找不到來源檔案")
            else:
                self.store.update(item.id, state="waiting", source_path="")
            return
        if not self._space_ok("process", self._process_estimate(item)):
            return
        self._space_blocked["process"] = False
        with self._lock:
            if (self._wk is not None or getattr(self, "_subtitle_model_test_busy", False) or self._subtitle_turn_held()
                    or item.id in self._tombstones or item.id in self._reprocessing):
                return                                # held: a local subtitle turn is switching the GPU (R2)
            item = self.store.get(item.id)
            if item is None or item.state != "queued":
                return
            generation = self._generations.get(item.id, 0)
            run = _Run(item.id, generation)
            if item.settings.get("subtitles") and item.subtitle.get("status") != "skipped":
                run.stage = "waiting_gpu"
            run.thread = threading.Thread(target=self._run_worker, args=(item, run), name=f"queue-wk-{item.id}",
                                          daemon=True)
            # Serialize this state transition with download completion. Otherwise a completion based on a stale
            # queued snapshot can clear the worker's first progress update.
            self.store.update(item.id, state="processing", error="")
            self._wk = run
            run.thread.start()

    def _request_stop(self, run: _Run, reason: str) -> None:
        with self._lock:
            if run.reason is None:
                run.reason = reason
            run.cancel.set()
            proc = run.proc
        if proc is not None and proc.poll() is None:
            kill_tree(proc.pid)

    def _stop_download_peer(self, run: _Run, reason: str) -> None:
        """Stop the paired worker only while this download generation still owns the item."""
        with self._lock:
            if (run.item_id in self._tombstones
                    or self._generations.get(run.item_id, run.generation) != run.generation):
                return
            peers = [peer for peer in (self._wk, *self._subtitle_runs())
                     if peer and peer.item_id == run.item_id]
            for peer in peers:
                if peer.reason is None:
                    peer.reason = reason
                peer.cancel.set()
            processes = [peer.proc for peer in peers if peer.proc is not None]
        for proc in processes:
            if proc.poll() is None:
                kill_tree(proc.pid)
        self._subtitle_stop_executors(run.item_id, reason)

    def _update_sleep(self) -> None:
        with self._lock:
            active = self._dl is not None or self._wk is not None or bool(self._subtitle_runs())
        want = active and bool(self.store.get_settings().get("prevent_sleep", True))
        if want != self._sleep_on:
            try:
                self._set_exec(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if want else ES_CONTINUOUS)
                self._sleep_on = want
            except Exception as e:  # noqa: BLE001
                log(f"sleep assertion failed: {e!r}")

    def _check_shutdown(self) -> None:
        if not platform_support.supports_shutdown():
            return
        if self.is_player_busy() or self.is_player_playing():
            return
        with self._lock:
            active = self._dl is not None or self._wk is not None or bool(self._subtitle_runs())
        if active:
            self._active_seen = True
            return
        if self._shutdown_started or not self._active_seen or self._paused_all:
            return
        if not self.store.get_settings().get("shutdown_when_done"):
            return
        items = self.store.items()
        if any(it.state in PENDING_STATES for it in items) or not any(it.state == "done" for it in items):
            return
        if any(
            it.subtitle.get("only")
            and it.subtitle.get("status") in subtitle_jobs.ACTIVE | {"ready", "mux_pending"}
            for it in items
        ):
            return
        self._shutdown_started = True
        self._active_seen = False
        self._shutdown_notice = "電腦將於 60 秒後關機"
        args = ["shutdown", "/s", "/t", str(SHUTDOWN_DELAY)]
        log(f"everything is done: running {' '.join(args)}")
        try:
            self._run_shutdown(args)
        except Exception as e:  # noqa: BLE001
            log(f"shutdown failed: {e!r}")
            self._shutdown_notice = ""
        self._dirty.set()

    def cancel_shutdown(self) -> dict:
        """Abort a pending shutdown (`shutdown /a`) and switch shutdown_when_done off."""
        self.store.set_settings(shutdown_when_done=False)
        self._shutdown_notice = ""
        self._shutdown_started = False
        if not platform_support.supports_shutdown():
            self._dirty.set()
            return {"ok": True}
        log("running shutdown /a")
        try:
            self._run_shutdown(["shutdown", "/a"])
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        self._dirty.set()
        return {"ok": True}

    def _emit_if_due(self) -> None:
        now = time.monotonic()
        if not self._dirty.is_set():
            return
        if now - self._last_emit < EMIT_INTERVAL:
            return
        self._dirty.clear()
        self._last_emit = now
        if not self.on_state:
            return
        state = self.get_state()
        for cb in list(self.on_state):
            try:
                cb(state)
            except Exception as e:  # noqa: BLE001
                log(f"on_state callback failed: {e!r}")

    # --------------------------------------------------------------------------------------------------------
    # download stage
    # --------------------------------------------------------------------------------------------------------

    def _safe_update(self, item_id: str, **fields) -> bool:
        try:
            self.store.update(item_id, **fields)
            return True
        except KeyError:
            return False

    def _state_of(self, item_id: str) -> Optional[str]:
        it = self.store.get(item_id)
        return it.state if it else None

    def _safe_run_update(self, run: _Run, **fields) -> bool:
        with self._lock:
            if (run.item_id in self._tombstones
                    or self._generations.get(run.item_id, run.generation) != run.generation
                    or (run.cancel.is_set() and self._wk is run)):
                return False
            try:
                self.store.update(run.item_id, **fields)
                return True
            except KeyError:
                return False

    def _safe_worker_progress(self, run: _Run, progress: dict) -> bool:
        """A cancelled reader may outlive its worker and must never paint progress onto a replacement run."""
        with self._lock:
            if (run.cancel.is_set() or self._wk is not run or run.item_id in self._tombstones
                    or self._generations.get(run.item_id, run.generation) != run.generation):
                return False
            try:
                self.store.update(run.item_id, progress=progress)
                return True
            except KeyError:
                return False

    def _source_ready_callback(self, item: queue_store.Item, run: _Run, *args, **kwargs) -> None:
        """Publish the first cache-ready playlist. Accept either the handle or DownloadResult callback shape."""
        if run.cancel.is_set() or not self._run_is_current(run):
            return
        candidates = list(args) + list(kwargs.values())
        handle = next((v for v in candidates if self._source_field(v, "runtime_url") is not None
                       and self._source_field(v, "playlist_path") is not None), None)
        result = next((v for v in candidates if getattr(v, "path", None) or
                       (isinstance(v, dict) and v.get("path"))), None)
        path = self._source_field(handle, "playlist_path") if handle is not None else None
        path = path or (result.get("path") if isinstance(result, dict) else getattr(result, "path", None))
        path = path or kwargs.get("playlist_path")
        if handle is None and path:
            registry = self._registry_for(path)
            if registry is not None:
                try:
                    handle = registry.get_session(str(path))
                except Exception as e:  # noqa: BLE001
                    log(f"could not find ready HLS session for {item.id}: {e!r}")
        if handle is None or not path:
            log(f"ignored source-ready callback for {item.id}: missing HLS session")
            return
        path = str(path)
        title = (result.get("title") if isinstance(result, dict) else getattr(result, "title", None))
        title = title or kwargs.get("title")
        if not self._run_is_current(run) or run.cancel.is_set():
            return
        try:
            snapshot = handle.snapshot()
        except Exception:
            snapshot = {}
        current = self.store.get(item.id)
        current_prog = current.download_progress if current and isinstance(current.download_progress, dict) else {}
        download_progress = {
            **current_prog,
            "stage": "downloading", "source_ready": True,
            "status": snapshot.get("status", "downloading"),
            "ready_segments": len(snapshot.get("ready_segments") or []),
            "segment_count": snapshot.get("segment_count"),
        }
        if snapshot.get("total_bytes"):          # byte-range sources know their exact size
            download_progress["downloaded_bytes"] = snapshot.get("downloaded_bytes")
            download_progress["total_bytes"] = snapshot.get("total_bytes")
        with self._lock:
            if (item.id in self._tombstones or run.cancel.is_set()
                    or self._generations.get(item.id, run.generation) != run.generation
                    or self._dl is not run):
                return
            current = self.store.get(item.id)
            if current is None:
                return
            fields: dict[str, Any] = {"source_path": path, "source_status": "downloading",
                                      "download_progress": download_progress}
            if current.state == "downloading":
                fields["state"] = "queued"
            if (not current.title or current.title == current.target) and title:
                fields["title"] = title
            try:
                self.store.update(item.id, **fields)
            except KeyError:
                return
            self._sources[item.id] = handle
            run.source_ready = True
            self._worker_waiting_for_source.discard(item.id)
        self._wake.set()
        if not self._pending_play_ready(item.id):
            self._refresh_watched_source(item.id)

    def _pending_play_ready(self, item_id: str) -> bool:
        with self._lock:
            if self._pending_play != item_id:
                return False
            self._pending_play = None
            self._player_item = item_id
            start, self._pending_start = self._pending_start, None
        it = self.store.get(item_id)
        if it is None or not self._source_ready(it):
            return True
        payload = self._source_payload(it)
        if start is not None:
            payload["start"] = start
        self.store.mark_used(item_id)
        if payload.get("progressive"):
            with self._lock:
                self._watch[item_id] = self._clock()
        try:
            reply = self._send_to_player(payload)
            if reply is None:
                self._launch_player(payload["input"], payload)
        except Exception as e:  # noqa: BLE001 - surface an asynchronous pending-play failure on the item
            self._safe_update(item_id, state="failed", source_status="failed", error=str(e))
            with self._lock:
                self._watch.pop(item_id, None)
            self._changed()
            return True
        self._changed()
        return True

    def _refresh_watched_source(self, item_id: str) -> None:
        """Tell an already-open queue player about a replacement proxy after a paused source resumes."""
        with self._lock:
            stamp = self._watch.get(item_id)
            if (self._player_item != item_id or stamp is None
                    or self._clock() - stamp >= self.busy_expire):
                return
        item = self.store.get(item_id)
        if item is None or item.state not in ("queued", "processing") or not self._source_ready(item):
            return
        payload = self._source_payload(item)
        if not payload.get("source_url"):
            return
        try:
            # A heartbeat means an existing player owns this item. If it is gone, do not spawn a second player.
            self._send_to_player(payload)
        except Exception as e:  # noqa: BLE001
            log(f"could not refresh progressive source for {item_id}: {e!r}")

    def _source_payload(self, item: queue_store.Item) -> dict:
        """Build one progressive-player message, including the live proxy and probe for cached HLS sources."""
        payload = {"input": item.source_path, "progressive": True, "queue_id": item.id,
                   "work_dir": str(self.store.item_dir(item.id) / "work"), "title": item.title}
        pressed_at = self._press.pressed_at(item.id)
        if pressed_at is not None:
            payload["pressed_at"] = pressed_at          # T1.3: the player logs press → first playable too
        handle = self._session_for(item, reopen_complete=True)
        if handle is not None:
            runtime_url = self._source_field(handle, "runtime_url")
            probe = self._source_field(handle, "metadata")
            if runtime_url and isinstance(probe, dict):
                payload.update(source_url=runtime_url, source_probe=probe)
        self._add_external_subtitles(payload, item)
        return payload

    def _run_download(self, item: queue_store.Item, run: _Run) -> None:
        last_write = [0.0]
        prev_dl = [None, None]  # [bytes, timestamp]
        speed_est = [None]      # smoothed speed

        def on_progress(d: dict) -> None:
            if run.cancel.is_set() or not self._run_is_current(run):
                return
            now = time.monotonic()
            total = d.get("total_bytes")
            done = d.get("downloaded_bytes") or 0
            speed = d.get("speed")
            if speed is None or speed <= 0:
                if prev_dl[0] is not None and prev_dl[1] is not None:
                    dt = now - prev_dl[1]
                    if dt >= 0.3:
                        inst = max(0.0, (done - prev_dl[0]) / dt)
                        if speed_est[0] is None:
                            speed_est[0] = inst
                        else:
                            speed_est[0] = speed_est[0] * 0.7 + inst * 0.3
                        prev_dl[0] = done
                        prev_dl[1] = now
                        speed = speed_est[0]
                    else:
                        speed = speed_est[0]
                else:
                    prev_dl[0] = done
                    prev_dl[1] = now
            else:
                speed_est[0] = speed
                prev_dl[0] = done
                prev_dl[1] = now

            eta = d.get("eta")
            if (eta is None or eta <= 0) and total and total > done and speed and speed > 0:
                eta = (total - done) / speed

            prog = {"stage": "downloading", "downloaded_bytes": done, "total_bytes": total,
                    "total_approx": bool(d.get("total_approx")),
                    "speed": speed, "eta": eta,
                    "fragment_index": d.get("fragment_index"), "fragment_count": d.get("fragment_count"),
                    "percent": (round(done * 100.0 / total, 1) if total else None)}
            if now - last_write[0] >= PROGRESS_WRITE_INTERVAL or d.get("stage") == "finished":
                last_write[0] = now
                self._safe_run_update(run, download_progress=prog)

        try:
            args = (item.target, item.headers, item.quality, str(self.store.item_dir(item.id)), on_progress,
                    run.cancel)
            if self._downloader_accepts_source_ready:
                result = self._downloader(*args, on_source_ready=lambda *a, **kw:
                                          self._source_ready_callback(item, run, *a, **kw))
            else:
                result = self._downloader(*args)
        except BaseException as e:  # noqa: BLE001
            self._stop_download_peer(
                run, "source_failed" if not run.cancel.is_set() else run.reason or "stop")
            if run.cancel.is_set():
                if run.reason != "remove" and self._run_is_current(run):
                    current = self.store.get(item.id)
                    if current and current.state != "paused":
                        self._safe_run_update(run, state="waiting", source_status="paused")
            else:
                log(f"download of {item.id} failed: {e!r}")
                msg = str(e) or e.__class__.__name__
                changed = False
                with self._lock:
                    if (item.id not in self._tombstones
                            and self._generations.get(item.id, run.generation) == run.generation):
                        current = self.store.get(item.id)
                        if current is not None:
                            self.store.update(item.id, state="failed", source_status="failed", error=msg,
                                              download_progress={"stage": "failed", "error": msg})
                        if self._pending_play == item.id:
                            self._pending_play = None
                        self._watch.pop(item.id, None)
                        self._worker_waiting_for_source.discard(item.id)
                        changed = True
                if changed:
                    self._changed()
        else:
            if run.cancel.is_set():
                self._stop_download_peer(run, run.reason or "stop")
                if run.reason != "remove" and self._run_is_current(run):
                    current = self.store.get(item.id)
                    if current and current.state != "paused":
                        self._safe_run_update(run, state="waiting", source_status="paused")
            elif run.reason != "remove" and self._run_is_current(run):
                path = str(getattr(result, "path", "") or (result.get("path") if isinstance(result, dict) else ""))
                title = getattr(result, "title", None) or (result.get("title") if isinstance(result, dict) else None)
                updated = False
                with self._lock:
                    if (item.id not in self._tombstones
                            and self._generations.get(item.id, run.generation) == run.generation):
                        current = self.store.get(item.id)
                        if current is not None:
                            fields: dict[str, Any] = {"source_path": path,
                                                      "source_status": "complete" if run.source_ready else "",
                                                      "download_progress": {"stage": "complete", "percent": 100.0}
                                                      if run.source_ready else {}}
                            if (not current.title or current.title == current.target) and title:
                                fields["title"] = title
                            if current.state in ("downloading", "waiting"):
                                fields["state"] = "queued"
                            if current.state not in ("processing", "done"):
                                fields["progress"] = {}
                            self.store.update(item.id, **fields)
                            self._worker_waiting_for_source.discard(item.id)
                            updated = True
                if updated:
                    self._wake.set()
                    self._pending_play_ready(item.id)
        finally:
            with self._lock:
                if self._dl is run:
                    self._dl = None
            self._wake.set()

    def _after_cancel(self, item_id: str, run: _Run, expect: str, mapping: dict) -> None:
        """A cancelled run: only touch the item if nobody (pause / retry / remove) already moved it on."""
        new = mapping.get(run.reason or "stop")
        if not new:
            return
        with self._lock:
            if (item_id in self._tombstones
                    or self._generations.get(item_id, run.generation) != run.generation):
                return
            current = self.store.get(item_id)
            if current and current.state == expect:
                self.store.update(item_id, state=new)

    # --------------------------------------------------------------------------------------------------------
    # process stage
    # --------------------------------------------------------------------------------------------------------

    def _default_worker_command(self, job_path: str) -> list:
        entry = ([self._python, str(HERE / "queue_worker.py")] if getattr(sys, "frozen", False) or
                 (HERE / "queue_worker.py").is_file() else
                 [self._python, "-m", "oh_streamer", "worker"])
        return [*entry, "--job", job_path]

    @staticmethod
    def _pinned_of(item: queue_store.Item) -> dict:
        """The per-item overrides (interpolation / encoder / save_compare) of an item added from the intake panel."""
        pinned = (item.settings or {}).get("pinned")
        return pinned if isinstance(pinned, dict) else {}

    def _build_job(self, item: queue_store.Item) -> dict:
        qs = self.store.get_settings()
        settings = dict(item.settings)
        settings["engine"] = self._queue_engine() if not settings.get("engine") else settings["engine"]   # old items: no key
        pinned = self._pinned_of(item)
        settings["encoder"] = pinned.get("encoder", qs.get("encoder", "hevc"))
        settings["interpolation"] = pinned.get("interpolation", qs.get("interpolation", "full"))
        if not platform_support.capabilities()["rife"] or settings.get("device") in ("cpu", "mps"):
            settings["interpolation"] = "off"
        compare = bool(pinned["save_compare"]) if "save_compare" in pinned else bool(qs.get("save_compare"))
        settings["compare"] = "on" if compare else "off"
        settings.pop("compare_original", None)
        settings.pop("compare_quality", None)
        if compare:
            settings["compare_original"] = pinned.get("compare_original", qs.get("compare_original", "copy"))
            settings["compare_quality"] = pinned.get("compare_quality", qs.get("compare_quality", "small"))
        if not item.settings.get("engine"):
            try:    # an item from before the engine setting: remember which engine made it (cache_key stays as it was)
                self.store.update(item.id, settings={**item.settings, "engine": settings["engine"]})
            except Exception as e:  # noqa: BLE001
                log(f"could not record the engine of {item.id}: {e!r}")
        if any(item.settings.get(k) != settings.get(k) for k in ("compare", "compare_original", "compare_quality")):
            try:    # remember what this output is made with (the item view's compare_original reads it)
                self.store.update(item.id, settings=dict(settings))
            except KeyError:
                pass
        d = self.store.item_dir(item.id)
        job = {"source": item.source_path, "work_dir": str(d / "work"), "output": str(d / "output.mkv"),
               "segment_seconds": SEGMENT_SECONDS, "settings": settings}
        handle = self._session_for(item, reopen_complete=True)
        if handle is not None:
            job.update(source_url=self._source_field(handle, "runtime_url"),
                       source_identity=self._source_field(handle, "identity"),
                       source_generation=self._source_field(handle, "generation"),
                       source_probe=self._source_field(handle, "metadata", {}),
                       source_status_path=str(self._source_field(handle, "status_path") or ""))
        with self._lock:
            k = self._start_k.pop(item.id, None)       # used once
        if k is not None:
            job["start_k"] = k
        return job

    def _read_worker_output(self, proc, item_id: str, d: Path, out: dict, run: Optional[_Run] = None) -> None:
        last_write = 0.0
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "progress":
                frames = msg.get("frames") or 0
                frame = msg.get("frame") or 0
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in (frames, frame)):
                    log("ignored malformed worker progress counters")
                    continue
                k = msg.get("k")
                source_fps = msg.get("source_fps")
                duration = msg.get("duration")
                prog = {"stage": "processing", "frame": frame, "frames": frames, "fps": msg.get("fps"),
                        "eta": msg.get("eta"), "segments": msg.get("segments"), "phase": msg.get("phase"),
                        "k": k if isinstance(k, int) and not isinstance(k, bool) and k >= 0 else None,
                        "percent": (round(min(frame, frames) * 100.0 / frames, 1) if frames else None)}
                if source_fps is not None and isinstance(source_fps, (int, float)) and source_fps > 0:
                    prog["source_fps"] = round(float(source_fps), 3)
                if duration is not None and isinstance(duration, (int, float)) and duration > 0:
                    prog["duration"] = round(float(duration), 3)
                now = time.monotonic()
                if now - last_write >= PROGRESS_WRITE_INTERVAL or msg.get("phase") == "finalize":
                    last_write = now
                    if run is not None:
                        self._safe_worker_progress(run, prog)
                    else:
                        self._safe_update(item_id, progress=prog)
            elif kind == "perf_obs":
                self._record_perf_obs(msg)
            elif kind == "timing":
                self._press.models_loaded(item_id, msg.get("models_s"))
            elif kind == "segment_done":
                line = self._press.segment_done(item_id)
                if line:
                    log(line)
            elif kind == "done":
                out["done"] = msg.get("output") or str(d / "output.mkv")
            elif kind == "error":
                out["error"] = str(msg.get("message", ""))

    def _record_perf_obs(self, msg: dict) -> None:
        """A worker throughput observation -> the profile's calibration. Skipped while the performance check holds the
        machine, when the player restored or played anything during the observed window (playing a processed file
        with VSR slowed a worker ~12% on an RTX 5060, and mpv does not show up as a CUDA process), without a (fresh)
        profile (PerfRunner decides the latter)."""
        if self.perf.holding:
            return
        try:
            window_s = float(msg.get("frames")) / float(msg.get("fps"))
        except (TypeError, ValueError, ZeroDivisionError):
            return
        if not math.isfinite(window_s) or window_s <= 0 or self.player_used_gpu_since(self._clock() - window_s):
            return
        now = self._clock()
        with self._lock:
            if any(start <= now and (end is None or end >= now - window_s)
                   for start, end in self._subtitle_windows):
                return
        try:
            self.perf.record_observation(mode=msg.get("mode"), height=msg.get("height"), observed_fps=msg.get("fps"),
                                         density=msg.get("density"), detector=msg.get("detector"),
                                         engine=msg.get("engine"))
        except Exception as e:  # noqa: BLE001 - calibration is a nicety, the reader must go on
            log(f"perf observation failed: {e!r}")

    def _run_worker(self, item: queue_store.Item, run: _Run) -> None:
        d = self.store.item_dir(item.id)
        err_tail = ""
        got_done: Optional[str] = None
        got_error = ""
        rc: Optional[int] = None
        job: dict = {}
        spawned = False
        late = None
        try:
            d.mkdir(parents=True, exist_ok=True)
            if run.cancel.is_set():
                raise _Cancelled()
            run.stage = "restore"
            self._maybe_play_after_subtitles(item.id)
            job = self._build_job(item)
            job_path = d / "job.json"
            job_path.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            errlog = d / "worker.log"
            with open(errlog, "ab") as errf:
                with self._lock:
                    if run.cancel.is_set():
                        raise _Cancelled()
                    run.proc = subprocess.Popen(self._worker_command(str(job_path)), stdout=subprocess.PIPE,
                                                stderr=errf, stdin=subprocess.PIPE, creationflags=NO_WINDOW,
                                                cwd=str(HERE), **({"start_new_session": True} if sys.platform != "win32" else {}))
                    spawned = True
                    self._press.spawned(item.id)
                    late = self._start_k.pop(item.id, None)     # a goto that came in while the job was written
                proc = run.proc
                if late is not None:
                    self._send_goto(proc, late)
                out: dict = {}
                reader = threading.Thread(target=self._read_worker_output, args=(proc, item.id, d, out, run),
                                          name=f"queue-rd-{item.id}", daemon=True)
                reader.start()
                rc = proc.wait()
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
                reader.join(timeout=3)      # a leftover grandchild holding the pipe must not hang the queue
                if not reader.is_alive():
                    proc.stdout.close()
                got_done, got_error = out.get("done"), out.get("error", "")
            try:
                data = errlog.read_bytes()[-2000:]
                err_tail = data.decode("utf-8", "replace").strip()
            except OSError:
                pass
        except _Cancelled:
            self._subtitle_interrupted(item, run)
            if not spawned and "start_k" in job:
                with self._lock:
                    self._start_k.setdefault(item.id, job["start_k"])
        except Exception as e:  # noqa: BLE001
            log(f"worker for {item.id} crashed: {e!r}")
            got_error = got_error or f"{type(e).__name__}: {e}"
        try:
            self._finish_worker(item, run, rc, got_done, got_error, err_tail)
            if rc == 0 and got_done:
                self._subtitle_after_restore(item.id)
        finally:
            with self._lock:
                if self._wk is run:
                    self._wk = None
            self._wake.set()

    def _subtitle_after_restore(self, item_id: str) -> None:
        """The video is complete. Its subtitles are the executors' business; only mark the item so that
        `_maybe_start_subtitle_only` muxes once they are ready (or `host.publish` says mux_pending)."""
        current = self.store.get(item_id)
        if (current is None or current.state != "done" or not current.settings.get("subtitles")
                or current.subtitle.get("status") in (None, "", "skipped", "failed", "cancelled", "done")):
            return
        self._subtitle_update(item_id, only=True)
        # the slice table may have been finished before the video was (status still ready/partial): mux right away
        self._subtitle_reconcile_complete(self.store.get(item_id))

    def _finish_worker(self, item, run: _Run, rc, got_done, got_error, err_tail) -> None:
        if run.reason in ("remove", "source_failed") or not self._run_is_current(run):
            return
        if run.cancel.is_set():
            if run.reason in ("paused_all", "auto", "preempt", "stop"):
                current = self.store.get(item.id)
                state = self._worker_cancel_state(current)
                self._after_cancel(item.id, run, "processing", {run.reason: state})
            return
        if rc == 0 and got_done:
            current = self.store.get(item.id)
            if current and current.source_status == "downloading":
                handle = self._session_for(current, reopen_complete=False)
                try:
                    source_status = handle.snapshot().get("status") if handle else None
                except Exception:
                    source_status = None
                if source_status == "complete":
                    self._safe_run_update(run, source_status="complete")
                else:
                    # A live worker must not interpret the current playlist tail as the source's natural EOF.
                    self._worker_waiting_for_source.add(item.id)
                    self._safe_run_update(run, state="queued")
                    return
            out = got_done
            try:
                size = os.path.getsize(out)
            except OSError:
                self._safe_run_update(run, state="failed", error="成品檔案不存在", progress={})
                return
            fields: dict = {"state": "done", "output_path": out, "size_bytes": size, "error": "",
                            "progress": {"stage": "done", "percent": 100.0}}
            src = item.source_path
            if (src and item.source_status == "" and _inside(src, self.store.item_dir(item.id))
                    and not self._subtitle_source_pending(current)
                    and item.id not in self._reprocessing and item.id not in self._watched_ids()):
                try:
                    os.remove(src)
                    fields["source_path"] = ""
                except OSError as e:
                    log(f"could not delete source {src}: {e!r}")
            self._safe_run_update(run, **fields)
            with self._lock:
                self._work_cleanup.add(item.id)        # the scheduler deletes work/ once nobody watches the item
            self._wake.set()
            return
        msg = got_error or err_tail or f"worker exited with code {rc}"
        if self._state_of(item.id) == "processing":
            self._safe_run_update(run, state="failed", error=msg[-500:], progress={})

    def _worker_cancel_state(self, current: Optional[queue_store.Item]) -> str:
        if current is None or current.source_status != "downloading":
            return "queued"
        with self._lock:
            download = self._dl if self._dl and self._dl.item_id == current.id else None
        if download is not None and not download.cancel.is_set() and self._source_ready(current):
            return "queued"
        return "waiting"

    # --------------------------------------------------------------------------------------------------------
    # IPC (player -> queue)
    # --------------------------------------------------------------------------------------------------------

    def _on_message(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == "add":
                return self._ipc_add(msg)
            if cmd == "player_busy":
                with self._lock:
                    self._note_player_gpu()
                    was_busy = self._busy
                    self._busy = bool(msg.get("busy"))
                    if self._busy:
                        self._opening_until = 0.0
                    self._busy_ts = self._clock()
                    self._note_player_gpu()
                    if "live" in msg:                    # a message without the key leaves the report as it is
                        self._live = self._clean_live(msg.get("live"))
                        self._live_ts = self._clock()
                        self._note_lead_sample(self._live)
                if "live" in msg and was_busy == self._busy:
                    self._dirty.set()                    # the card only: nothing for the scheduler to react to
                else:
                    self._changed()
                return {"ok": True}
            if cmd == "player_playing":
                with self._lock:
                    self._note_player_gpu()
                    self._playing = bool(msg.get("playing"))
                    if self._playing:
                        self._opening_until = 0.0
                    self._playing_ts = self._clock()
                    self._note_player_gpu()
                return {"ok": True}
            if cmd == "lookup":
                found = self._lookup(msg.get("input"), msg.get("quality"))
                if found and found.get("state") == "done":
                    self.store.mark_seen(found["id"])  # the player opens a finished item by itself
                return {"ok": True, "item": found}
            if cmd == "watched":
                item_id = str(msg.get("id", ""))
                if self.store.get(item_id) is None:
                    return {"ok": False, "error": "unknown id"}
                self.store.update(item_id, watched=True, seen=True, last_used=time.time())
                self._space_retry_at = {"download": 0.0, "process": 0.0}
                return {"ok": True}
            if cmd == "want":
                return self._ipc_want(msg)
            if cmd == "watching":
                item_id = str(msg.get("id", ""))
                found = self.store.get(item_id)
                if found is None:
                    return {"ok": False, "error": "unknown id"}
                if found.kind == queue_store.STREAM_KIND:
                    # live-mode stream item: a heartbeat for the subtitle scheduling only; it is not a queue item the
                    # player shows (no _player_item, nothing seen / protected / redrawn)
                    with self._lock:
                        self._watch[item_id] = self._clock()
                    return {"ok": True}
                with self._lock:
                    self._watch[item_id] = self._clock()
                    self._player_item = item_id
                self.store.mark_seen(item_id)
                self._watched_ids()
                self._changed()
                return {"ok": True}
            if cmd == "clock":
                return self._ipc_clock(msg)
            if cmd == "stream_end":
                return self._ipc_stream_end(msg)
            if cmd == "restore_needed":
                return self._ipc_restore_needed(msg)
            if cmd == "show":
                for cb in list(self.on_show):
                    try:
                        cb()
                    except Exception as e:  # noqa: BLE001
                        log(f"on_show callback failed: {e!r}")
                return {"ok": True}
        except Exception as e:  # noqa: BLE001
            log(f"ipc {cmd!r} failed: {e!r}")
            return {"ok": False, "error": str(e)}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    @staticmethod
    def _clean_restore_pause(raw) -> dict:
        """{"restore_pause": {"state", "turn"[, "reserved_mib"]}} of a live report (T4.2); {} for an old player / junk.
        reserved_mib (only while paused, optional): VRAM the paused restoration still holds."""
        if not isinstance(raw, dict) or raw.get("state") not in ("running", "pausing", "paused"):
            return {}
        turn = raw.get("turn")
        out = {"state": raw["state"], "turn": turn if isinstance(turn, str) and turn else None}
        mib = raw.get("reserved_mib")
        if raw["state"] == "paused" and not isinstance(mib, bool) and isinstance(mib, (int, float))                 and math.isfinite(mib) and mib >= 0:
            out["reserved_mib"] = float(mib)
        return {"restore_pause": out}

    @staticmethod
    def _clean_live(raw) -> Optional[dict]:
        """The player's live stream report (queue_ipc.py) with every field checked; None = no report (cleared / junk)."""
        if not isinstance(raw, dict):
            return None

        def number(value, low=0.0):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
                return None
            return float(value)
        stream_item = raw.get("stream_item")
        return {
            "title": str(raw.get("title") or ""), "pos": number(raw.get("pos")), "duration": number(raw.get("duration"), 0.001),
            "playing": raw.get("playing") is True, "buffering": raw.get("buffering") is True,
            "ahead_s": number(raw.get("ahead_s")), "speed": number(raw.get("speed")),
            "restore_done": raw.get("restore_done") is True,
            "stream_item": stream_item if isinstance(stream_item, str) and stream_item else None,
            **QueueService._clean_restore_pause(raw.get("restore_pause")),
        }

    def _streaming_state(self) -> Optional[dict]:
        """get_state()["streaming"]: the 「串流中」 card, None when no live-mode stream plays. Comes from the player's live
        report (player_busy heartbeat); a report older than busy_expire s is void (player gone without saying so). The
        subtitle part exists only for a stream with subtitles (its stream item): the item's subtitle view plus how far
        the subtitles are ahead of the viewer (view.covered_s is on the audio clock, so audio_offset is taken off the
        position)."""
        with self._lock:
            live = dict(self._live) if self._live is not None and self._clock() - self._live_ts < self.busy_expire else None
        if live is None:
            return None
        item = self.store.get(live["stream_item"]) if live["stream_item"] else None
        subtitle = None
        if item is not None and item.kind == queue_store.STREAM_KIND and item.settings.get("subtitles"):
            sub = item.subtitle if isinstance(item.subtitle, dict) else {}
            view = sub.get("view") if isinstance(sub.get("view"), dict) else None
            lead = None
            try:
                if view is not None and live["pos"] is not None:
                    audio_pos = max(0.0, live["pos"] - float(sub.get("audio_offset") or 0.0))
                    lead = round(float(view.get("covered_s") or 0.0) - audio_pos, 1)
            except (TypeError, ValueError):
                lead = None
            subtitle = {"status": sub.get("status"), "view": view, "detail": sub.get("detail"), "error": sub.get("error"),
                        "lead_s": lead}
        return {**live, "subtitle": subtitle}

    def _write_play_clock(self, item_id: str, position: float, playing: bool) -> None:
        try:
            sub_dir = self.store.item_dir(item_id) / "subtitles"
            if sub_dir.is_dir():
                from oh_streamer.subtitles import runtime
                runtime.atomic_write_json(
                    sub_dir / "play_clock.json",
                    {"position": round(float(position), 3), "playing": bool(playing), "updated_at": time.time()},
                )
        except Exception:
            pass

    def _ipc_want(self, msg: dict) -> dict:
        """The player (progressive mode) needs segment k of this item now: make it the running job."""
        item_id = str(msg.get("id", ""))
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        if it.kind == queue_store.STREAM_KIND:
            return self._ipc_stream_want(item_id, msg)
        self._skip_subtitles_for_play(item_id)     # subtitles are never skipped; this only ends the play-wait / a mux
        if not self._source_ready(it) or it.state not in (
                "queued", "processing", "paused", "failed"):
            log(f"want for {item_id} ignored: state {it.state}, source {'ready' if it.source_path else 'missing'}")
            return {"ok": False, "error": "source not downloaded"}
        try:
            k = max(0, int(msg.get("k") or 0))
        except (TypeError, ValueError):
            k = 0
        with self._lock:
            self._watch[item_id] = self._clock()
            self._player_item = item_id
            want_position = queue_ipc.parse_start(msg.get("pos"))
            foreground = bool(msg.get("foreground"))
            if foreground:
                # start()/start_source() set pos to the target before sending a foreground want.
                # Feeder lookahead carries an old playback clock, not a new seek target.
                want_position = want_position if want_position is not None else k * SEGMENT_SECONDS
                self._subtitle_wants[item_id] = (want_position, self._clock())
                self._subtitle_seek_guard[item_id] = self._subtitle_wants[item_id]
            if want_position is not None and (foreground or self._subtitle_accept_clock(item_id, want_position)):
                self._subtitle_clocks[item_id] = (want_position, False)
                if hasattr(self, "_write_play_clock"):
                    self._write_play_clock(item_id, want_position, False)
        self.store.mark_seen(item_id)
        self._watched_ids()
        if str(it.source_path).endswith(".m3u8"):
            source = self._session_for(it)
            if source is not None:
                # k is the requested repair segment; pos may still be the old playback clock.
                source.request_position(k * SEGMENT_SECONDS)
        if it.state == "failed":
            # no automatic retry: the player re-sends want every few seconds while it waits, and a job that fails
            # deterministically would restart forever (重試 in the window is the user's call)
            self._changed()
            return {"ok": False, "error": "failed", "message": it.error or ""}
        self.goto(item_id, k)
        self._changed()
        # a pause by the user wins over the player (it re-sends want every few seconds while it waits): the player
        # shows 背景處理已暫停 instead; play() on a paused item resumes it, 繼續 in the window does too
        if it.state == "paused":
            return {"ok": False, "error": "paused"}
        with self._lock:
            paused_all = self._paused_all
        return {"ok": False, "error": "paused"} if paused_all else {"ok": True}

    def _ipc_stream_want(self, item_id: str, msg: dict) -> dict:
        """A live-mode seek of a stream item: only the subtitle scheduling's data is recorded (T3.1 acts on it). There
        is no restore segment, source download or worker for a stream item, so "k" is ignored and none of the
        regular item logic (goto, download session, mark_seen, _player_item) runs."""
        position = queue_ipc.parse_start(msg.get("pos"))
        if position is None:
            return {"ok": False, "error": "bad pos"}
        with self._lock:
            self._watch[item_id] = self._clock()
            self._subtitle_clocks[item_id] = (position, False)
            self._subtitle_wants[item_id] = (position, self._clock())
            self._subtitle_seek_guard[item_id] = self._subtitle_wants[item_id]
            self._write_play_clock(item_id, position, False)
        self._wake.set()
        return {"ok": True}

    def _ipc_stream_end(self, msg: dict) -> dict:
        """The player is done with a stream item (new source, closed, or it never used it): the item and its
        streams/<id>/ folder go. Only stream items: any other id (a regular queue item, an unknown one) is an error
        and nothing is touched."""
        item_id = str(msg.get("id", ""))
        found = self.store.get(item_id)
        if found is None or found.kind != queue_store.STREAM_KIND:
            return {"ok": False, "error": "unknown id" if found is None else "not a stream item"}
        self._drop_stream_item(item_id)
        return {"ok": True}

    @staticmethod
    def _stream_marker_of(item) -> str:
        """The remote-audio marker path of a stream item ("" for a local file or an item without subtitle audio).
        Only item.stream["audio_marker"] is looked at: source_path of a local-file stream is the user's own video."""
        marker = item.stream.get("audio_marker") if item is not None and isinstance(item.stream, dict) else ""
        return marker if isinstance(marker, str) else ""

    def _stream_markers(self) -> list:
        return [m for m in (self._stream_marker_of(it) for it in self.store.stream_items()) if m]

    STREAM_REMOVE_JOIN = 5.0     # seconds the subtitle executor of a removed stream item gets to stop before its folder goes

    def _remove_stream(self, item_id: str) -> bool:
        """The one way a stream item goes (stream_end, replaced by a new stream, refused open, heartbeat expiry; stop()
        clears the rest): the item and its streams/<id>/ folder, what the player reported about it, and the url
        refresher registered for its audio marker (read before the item is gone).

        Like remove(): tombstone + generation first (a late publish writes nothing and sends no subtitle_update), then
        every subtitle executor of the item is stopped (all slots in _subtitle_runs(), so a dedicated stream slot is
        covered too) and joined for up to STREAM_REMOVE_JOIN seconds, and only then is the folder deleted. A folder
        that cannot be deleted (a thread still holds a file) is left for the next start, which wipes streams/."""
        item = self.store.get(item_id)
        marker = self._stream_marker_of(item)
        if item is not None and item.kind == queue_store.STREAM_KIND:
            with self._lock:
                self._tombstones.add(item_id)
                self._generations[item_id] = self._generations.get(item_id, 0) + 1
            try:
                alive = not self._subtitle_stop_executors(item_id, "remove", wait=self.STREAM_REMOVE_JOIN)
            except Exception as e:  # noqa: BLE001 - the item must go whatever the executor does
                log(f"stream item {item_id}: stopping the subtitle executor failed: {e!r}")
                alive = False
            if alive:
                log(f"stream item {item_id}: subtitle executor still running after {self.STREAM_REMOVE_JOIN:.0f}s; "
                    f"deleting the folder anyway (leftovers are wiped at the next start)")
            self._subtitle_executor_reset(item_id)
        removed = self.store.remove_stream(item_id)
        self._forget_stream_playback(item_id)
        self._subtitle_plan_files.pop(item_id, None)
        for table in ("_subtitle_sent_updates", "_subtitle_play_offsets"):
            found = getattr(self, table, None)
            if isinstance(found, dict):
                found.pop(item_id, None)
        if marker:
            media_input.unregister_refresher(marker)
        return removed

    def _drop_stream_item(self, item_id: str) -> bool:
        """Remove one stream item (folder included) and what the player reported about it."""
        return self._remove_stream(item_id)

    def _expire_streams(self) -> None:
        """Safety net for a player that died without sending stream_end: a stream item is dropped when its last
        `watching` is older than stream_watch_expire, or when none ever came within stream_first_watch_grace of the
        queue first knowing the item. Runs on every scheduler step."""
        streams = self.store.stream_items()
        now = self._clock()
        with self._lock:
            alive = {it.id for it in streams}
            for gone in [i for i in self._stream_created if i not in alive]:
                del self._stream_created[gone]
            stale = []
            for it in streams:
                created = self._stream_created.setdefault(it.id, now)
                last = self._watch.get(it.id)
                if last is not None:
                    if now - last > self.stream_watch_expire:
                        stale.append((it.id, f"no watching for {now - last:.0f}s"))
                elif now - created > self.stream_first_watch_grace:
                    stale.append((it.id, f"no watching within {now - created:.0f}s of its creation"))
        for item_id, why in stale:
            log(f"stream item {item_id} dropped ({why}; the player did not send stream_end)")
            self._drop_stream_item(item_id)

    def _forget_stream_playback(self, item_id: str) -> None:
        """Drop what the player reported about a stream item that is gone."""
        with self._lock:
            self._watch.pop(item_id, None)
            self._subtitle_clocks.pop(item_id, None)
            self._subtitle_wants.pop(item_id, None)
            self._subtitle_seek_guard.pop(item_id, None)
            self._stream_created.pop(item_id, None)

    def _subtitle_accept_clock(self, item_id, position):
        """Ignore pre-seek clocks briefly; remember large clock-only seeks for debounce."""
        now = self._clock()
        want = self._subtitle_seek_guard.get(item_id)
        if want is not None and now - want[1] < 2.0 and abs(position - want[0]) > 10:
            return False
        previous = self._subtitle_clocks.get(item_id)
        if previous is None or abs(position - previous[0]) > 10:
            self._subtitle_wants[item_id] = (position, now)
        return True

    def _ipc_clock(self, msg: dict) -> dict:
        """The viewer's playback position (progressive mode, about once a second): forwarded to the item's running
        worker, which paces its interpolation by it. Nothing else happens (no watch refresh: that is `watching`).
        A stream item (live mode) only records the position for the subtitle scheduling."""
        item_id = str(msg.get("id", ""))
        try:
            pos = float(msg.get("pos"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad pos"}
        if not math.isfinite(pos):
            return {"ok": False, "error": "bad pos"}
        stream = self.store.get(item_id)
        if stream is not None and stream.kind == queue_store.STREAM_KIND:
            if pos < 0:
                return {"ok": False, "error": "bad pos"}
            playing = bool(msg.get("playing"))
            with self._lock:
                if not self._subtitle_accept_clock(item_id, pos):
                    return {"ok": True}
                self._subtitle_clocks[item_id] = (pos, playing)       # source container time; no subtitle_pos for streams
                self._write_play_clock(item_id, pos, playing)
            self._wake.set()
            return {"ok": True}
        subtitle_position = pos
        if "subtitle_pos" in msg:
            current = self.store.get(item_id)
            try:
                relative = float(msg["subtitle_pos"])
                if current is not None and math.isfinite(relative) and relative >= 0:
                    subtitle_position = relative + float(current.subtitle.get("audio_offset", 0))
            except (TypeError, ValueError):
                pass
        playing = bool(msg.get("playing"))
        with self._lock:
            if not self._subtitle_accept_clock(item_id, subtitle_position):
                return {"ok": True}
            self._subtitle_clocks[item_id] = (subtitle_position, playing)
            if hasattr(self, "_write_play_clock"):
                self._write_play_clock(item_id, subtitle_position, playing)
            run = self._wk
            proc = (
                run.proc
                if run
                and run.item_id == item_id
                and run.stage == "restore"
                and not run.cancel.is_set()
                else None
            )
        if proc is None:
            self._wake.set()
            current = self.store.get(item_id)
            return {"ok": bool(current and self._subtitle_has_table(current))}
        self._send_clock(proc, pos, playing)
        return {"ok": True}

    @staticmethod
    def _target_of(text) -> tuple[str, str, dict]:
        text = str(text or "").strip()
        try:
            oi = open_input.parse(text)
            return oi.target, oi.kind, dict(oi.headers)
        except open_input.OpenInputError:
            return text.strip("\"'"), ("url" if text.lower().startswith(("http://", "https://")) else "file"), {}

    def _ipc_add(self, msg: dict) -> dict:
        try:
            oi = open_input.parse(str(msg.get("input") or ""))
        except open_input.OpenInputError as e:
            return {"ok": False, "error": str(e)}
        headers = dict(oi.headers)
        if isinstance(msg.get("headers"), dict):
            headers.update(msg["headers"])
        settings = {**self._snapshot_settings(), **(msg.get("settings") if isinstance(msg.get("settings"), dict) else {})}
        # Explicit queue model choices also apply to sources added from the player.
        for key in ("detection_model", "restoration_model"):
            value = self.store.get_settings().get(key)
            if value:
                settings[key] = value
        item = self.store.add(oi.target, oi.kind, headers, msg.get("quality"), settings,
                              title=msg.get("title") or self._default_title(oi.target, oi.kind))
        return {"ok": True, "id": item.id}

    @staticmethod
    def _default_title(target: str, kind: str) -> str:
        return Path(target).stem if kind == "file" else target

    def _lookup(self, text, quality) -> Optional[dict]:
        target, _kind, _hdr = self._target_of(text)
        q = _norm_quality(quality)
        # the player asks before resolving (quality usually None = "any"), while add sends the playing height
        found = [it for it in self.store.items()
                 if it.target == target and (q is None or _norm_quality(it.quality) == q)]
        if not found:
            return None
        found.sort(key=lambda it: STATE_PRIORITY.index(it.state) if it.state in STATE_PRIORITY else 99)
        it = found[0]
        if it.state == "done" and not os.path.isfile(it.output_path):
            rest = [x for x in found if x.state != "done"]
            if not rest:
                return None
            it = rest[0]
        found_item = {"id": it.id, "state": it.state, "output": it.output_path if it.state == "done" else "",
                      "progress": it.progress,
                      # a done item plays whatever engine made it; the player can tell it differs (items from before: lada)
                      "engine": (it.settings or {}).get("engine") or "lada"}
        self._add_combos_dir(found_item, it)
        return found_item

    def _add_combos_dir(self, payload: dict, it) -> dict:
        """Tell the player where this item keeps its subtitle combos (queue_ipc: subtitle_combos_dir), only when a
        manifest exists; absent otherwise (the player falls back to the sibling folder)."""
        folder = self.store.item_dir(it.id) / "subtitles" / "combos"
        if (folder / "manifest.json").is_file():
            payload["subtitle_combos_dir"] = str(folder)
        return payload

    # --------------------------------------------------------------------------------------------------------
    # UI API (queue_app.py window <-> js_api)
    # --------------------------------------------------------------------------------------------------------

    def get_state(self) -> dict:
        now = time.monotonic()
        if now - self._usage_at > USAGE_REFRESH_INTERVAL:
            self._usage_at = now
            self._usage = self.store.usage_bytes(refresh=True)
        settings = self._effective_queue_settings()
        items = []
        stored = self.store.items()
        eta = self._eta_estimates(stored, settings)
        for it in stored:
            d = it.to_dict()
            d["eta_estimate_s"] = eta.get(it.id)
            d["has_headers"] = bool(d.pop("headers", None))
            info = self._output_info(it) if it.state == "done" else None
            d["has_compare"] = has_compare_tracks(info)
            d["compare_original"] = self._compare_original_of(it, info) if d["has_compare"] else None
            self._thumb_fields(it, d)
            facts = self._source_facts(it)
            if facts:
                d["duration"] = facts.get("duration")
                d["source_fps"] = facts.get("fps")
            elif d.get("thumbs") and d["thumbs"].get("duration"):
                d["duration"] = d["thumbs"]["duration"]
            items.append(d)
        notice = SPACE_NOTICE if any(self._space_blocked.values()) else self._shutdown_notice
        return {"items": items, "settings": settings, "model_options": self._model_options(),
                "engine_options": self._engine_options(), "usage_bytes": self._usage,
                "cap_bytes": int(float(settings.get("cap_gb", 50)) * 1024 ** 3),
                "paused_all": self._paused_all, "player_busy": self.is_player_busy(), "notice": notice,
                "shutdown_pending": bool(self._shutdown_notice), "capabilities": platform_support.capabilities(),
                "perf": self._perf_state(),
                "streaming": self._streaming_state(),
                "subtitle_environment": self._subtitle_environment()}

    def _effective_queue_settings(self) -> dict:
        """The queue settings as the UI / ETA see them: models inherited from the player where the queue has none,
        interpolation off where RIFE is unavailable."""
        settings = self.store.get_settings()
        defaults = self._snapshot_settings()
        for key in ("detection_model", "restoration_model", "engine", "max_clip_length"):
            settings[key] = defaults.get(key)
        if not platform_support.capabilities()["rife"]:
            settings["interpolation"] = "off"
        return settings

    def _thumb_fields(self, it: queue_store.Item, d: dict) -> None:
        """Film-strip fields of one item dict: thumbs, processed_ranges, download_fraction, worker_fraction."""
        try:
            d["cover"] = (self.store.item_dir(it.id) / "cover.txt").read_text(encoding="utf-8")
        except OSError:
            d["cover"] = None
        try:
            d["thumbs"] = self.thumbs.thumbs_for(it)
            d["processed_ranges"] = self.thumbs.processed_ranges(it)
            d["worker_fraction"] = self.thumbs.worker_fraction(it)
        except Exception as e:  # noqa: BLE001 - a thumbnail problem must never take the whole queue state down
            log(f"thumbnail state failed for {it.id}: {e!r}")
            d["thumbs"], d["processed_ranges"], d["worker_fraction"] = None, [], None
        d["download_fraction"] = queue_thumbs.download_fraction(it)

    def _eta_estimates(self, items, settings: dict) -> dict:
        """item id -> eta_estimate_s (queue ETA prior from the performance profile) for queued / processing items that
        have no worker-reported ETA yet. Missing profile / source facts -> no entry (None). Cached per item + inputs."""
        try:
            gen, prof = self.perf.current_profile()
        except Exception as e:  # noqa: BLE001 - never take the state down
            log(f"eta prior failed: {e!r}")
            return {}
        live = {it.id for it in items}
        for cache in (self._eta_cache, self._eta_facts):
            for stale in [k for k in cache if k not in live]:
                cache.pop(stale, None)
        if prof is None:
            return {}
        from oh_streamer.perf import profile as perf_profile
        out: dict = {}
        for it in items:
            if it.state not in ("queued", "processing"):
                continue
            prog = it.progress if isinstance(it.progress, dict) else {}
            if it.state == "processing" and _finite_num(prog.get("eta")) and prog["eta"] > 0:
                continue                                  # the worker knows better
            try:
                value = self._eta_for(it, prog, settings, gen, prof, perf_profile)
            except Exception as e:  # noqa: BLE001
                log(f"eta prior failed for {it.id}: {e!r}")
                value = None
            if value is not None:
                out[it.id] = value
        return out

    def _eta_for(self, it: queue_store.Item, prog: dict, settings: dict, gen, prof, perf_profile) -> Optional[float]:
        facts = self._source_facts(it)
        if facts is None:
            return None
        fps, duration = facts["fps"], facts["duration"]
        frames, frame = prog.get("frames"), prog.get("frame")
        if _finite_num(frames) and frames > 0 and _finite_num(frame) and frame >= 0:
            duration = max(0.0, frames - min(frame, frames)) / fps      # processed fraction known: remaining part
        effective = it.settings if (it.state == "processing" and "compare" in it.settings) else None
        pinned = self._pinned_of(it)
        if effective is not None:
            encode = effective.get("compare") == "on" and effective.get("compare_original") == "encode"
            mode_setting = effective.get("interpolation", settings.get("interpolation"))
        else:
            save_compare = pinned["save_compare"] if "save_compare" in pinned else settings.get("save_compare")
            encode = bool(save_compare) and settings.get("compare_original") == "encode"
            mode_setting = pinned.get("interpolation", settings.get("interpolation"))
        mode = "full" if mode_setting == "full" else "off"
        if it.settings.get("device") in ("cpu", "mps") or not platform_support.capabilities().get("rife", True):
            mode = "off"
        compare = "encode" if encode else None
        detector = (it.settings or {}).get("detection_model") or settings.get("detection_model")
        key = (gen, detector, facts["height"], round(fps, 3), round(duration / 10.0), mode, compare)
        with self._lock:
            hit = self._eta_cache.get(it.id)
        if hit and hit[0] == key:
            return hit[1]
        value = perf_profile.job_seconds(perf_profile.for_detector(prof, detector), duration_s=duration,
                                         height=facts["height"], src_fps=fps, mode=mode, compare=compare)
        value = float(value) if _finite_num(value) and value > 0 else None
        with self._lock:
            self._eta_cache[it.id] = (key, value)
        return value

    def _source_facts(self, it: queue_store.Item) -> Optional[dict]:
        """{"height", "fps", "duration"} of the item's source: the stored HLS probe, else a one-off background ffprobe
        of the downloaded file (None until it answers; failures are retried at most every 60 s)."""
        key = (it.source_path, it.source_status)
        now = time.monotonic()
        with self._lock:
            hit = self._eta_facts.get(it.id)
        if hit and hit[0] == key and (hit[1] is not None or now - hit[2] < ETA_PROBE_RETRY):
            return hit[1]
        with self._lock:
            if it.id in self._tombstones:   # being removed: an ffprobe holding a file would block deleting its folder
                return None
        facts = None
        with self._lock:   # only an already-open session: state reads must never (re)open one (races remove())
            handle = self._sources.get(it.id)
        if handle is not None:
            facts = video_facts(self._source_field(handle, "metadata"))
        elif (it.source_path and not it.source_path.lower().endswith((".m3u8", RANGE_CACHE_SUFFIX))
              and it.source_status in ("", "complete")
              and os.path.isfile(it.source_path)):
            self._probe_facts_async(it.id, key, it.source_path)
            return None
        with self._lock:
            self._eta_facts[it.id] = (key, facts, now)
        return facts

    def _probe_facts_async(self, item_id: str, key: tuple, path: str) -> None:
        def work() -> None:
            try:
                facts = self._eta_probe(path)
            except Exception:  # noqa: BLE001
                facts = None
            with self._lock:
                self._eta_probing.pop(item_id, None)
                self._eta_facts[item_id] = (key, facts, time.monotonic())
            if facts is not None:
                self._changed()
        t = threading.Thread(target=work, name=f"queue-eta-{item_id}", daemon=True)
        with self._lock:   # check + register atomically so remove() always sees (and joins) an in-flight probe
            if item_id in self._eta_probing or item_id in self._tombstones:
                return
            self._eta_probing[item_id] = t
        t.start()

    def _perf_state(self) -> dict:
        try:
            return self.perf.snapshot()
        except Exception as e:  # noqa: BLE001 - a broken profile must never take the whole queue state down
            log(f"perf state failed: {e!r}")
            return {"status": "none", "kind": None, "stage": "", "progress": 0.0, "error": None,
                    "can_start": False, "blocked_reason": "效能檢測狀態無法讀取", "prompt": False, "profile": None}

    def _output_info(self, it: queue_store.Item) -> Optional[dict]:
        """probe_tracks() of the item's finished output, cached per path + mtime + size (no ffprobe per state push)."""
        path = it.output_path
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._track_cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
        info = self._probe_tracks(path)
        if info is not None:                  # a failed probe is retried on the next state push
            with self._lock:
                self._track_cache[path] = (key, info)
        return info

    @staticmethod
    def _compare_original_of(it: queue_store.Item, info: Optional[dict]) -> Optional[str]:
        value = it.settings.get("compare_original")
        if value not in queue_store.COMPARE_ORIGINAL_CHOICES:
            value = ((info or {}).get("tags") or {}).get("OH_ORIGINAL", "").lower()
        return value if value in queue_store.COMPARE_ORIGINAL_CHOICES else None

    def _queue_engine(self) -> str:
        """The restoration engine the queue uses: --engine / OH_STREAMER_ENGINE > the queue's own `engine` setting > the player's
        saved one > lada. An unknown name falls back to lada. Also makes it the process default (model lists, perf)."""
        from oh_streamer import engines
        from oh_streamer.player import player_settings as PS
        name = None
        try:
            name = self.store.get_settings().get("engine")
            if not name:
                name = PS.load_file(self.perf._player_file())[0].get("engine")
        except Exception:  # noqa: BLE001
            pass
        return engines.set_configured_engine(engines.effective_engine_name(name, log), log)

    def _engine_options(self) -> list:
        from oh_streamer import engines
        return [{"value": n, "label": engines.engine_label(n)} for n in engines.visible_engine_names()]

    def _model_options(self) -> dict:
        """The models the queue's engine offers, in the shape the UI wants (Lada's list is read without importing it)."""
        from oh_streamer import engines
        from oh_streamer.player import player_settings as PS
        try:
            catalog = engines.model_catalog(self._queue_engine())
        except Exception as e:  # noqa: BLE001
            log(f"model list of the restoration engine unavailable: {e}")
            return {"detection_model": [], "restoration_model": []}
        return {key: [{"value": name, "label": PS.model_label(name) + ("（建議）" if name in PS.RECOMMENDED else "")}
                      for name in names]
                for key, names in (("detection_model", catalog.detection), ("restoration_model", catalog.restoration))}

    def _snapshot_settings(self) -> dict:
        try:
            from oh_streamer.player import player_settings as PS

            saved, _err = PS.load_file(self.perf._player_file())
        except Exception:  # noqa: BLE001
            saved = {}
        keys = ("detection_model", "restoration_model", "fp16", "max_clip_length")
        out = {}
        queue_defaults = self.store.get_settings()
        for k in keys:
            try:
                out[k] = queue_defaults.get(k) or saved.get(k, PS.DEFAULTS[k])
            except Exception:  # noqa: BLE001
                pass
        out["engine"] = self._queue_engine()
        return out

    def add(self, text: str) -> dict:
        try:
            oi = open_input.parse(text)
        except open_input.OpenInputError as e:
            return {"ok": False, "error": str(e)}
        item = self.store.add(oi.target, oi.kind, oi.headers, None, self._snapshot_settings(),
                              title=self._default_title(oi.target, oi.kind))
        return {"ok": True, "id": item.id}

    def check_source(self, text: str) -> dict:
        """Fast, offline input hint for the queue entry field."""
        try:
            oi = open_input.parse(text)
        except open_input.OpenInputError as e:
            return {"status": "invalid", "message": str(e)}
        if oi.kind != "url":
            return {"status": "file", "message": "本機檔案路徑；加入後再檢查檔案"}
        from oh_streamer.sources.source_resolve import check_url_support
        return check_url_support(oi.target)

    def stream(self, text: str) -> dict:
        """Open a live source with the existing player; never create a queue job."""
        try:
            oi = open_input.parse(text)
        except open_input.OpenInputError as e:
            return {"ok": False, "error": str(e)}
        if oi.kind == "file" and not os.path.isfile(oi.target):
            return {"ok": False, "error": "找不到影片檔案"}
        return self._send_open(oi.target, {"cmd": "open", "input": oi.target, "headers": dict(oi.headers)})

    def _send_open(self, target: str, payload: dict) -> dict:
        """Hand an open payload to the player (launching it when it is not running), with the opening guard that keeps
        the queue worker off the GPU until the player reports busy; shared by stream() and stream_resolved()."""
        with self._lock:
            previous_guard = self._opening_until
            self._opening_until = self._clock() + 120.0
            self._pending_play = None
        self._changed()
        try:
            if self._shutdown_started:
                result = self.cancel_shutdown()
                if not result.get("ok"):
                    with self._lock:
                        self._opening_until = previous_guard
                    self._changed()
                    return result
            reply = self._send_to_player(payload)
            if reply is not None and not reply.get("ok"):
                with self._lock:
                    self._opening_until = previous_guard
                self._changed()
                return {"ok": False, "error": str(reply.get("error") or "播放器無法開啟來源")}
            if reply is None:
                self._launch_player(target, payload)
            return {"ok": True, "launched": reply is None, "message": "已送至播放器"}
        except OSError as e:
            with self._lock:
                self._opening_until = previous_guard
            self._changed()
            return {"ok": False, "error": str(e)}

    def add_paths(self, paths: list) -> dict:
        ids, errors = [], []
        for p in paths or []:
            r = self.add(str(p))
            (ids if r.get("ok") else errors).append(r.get("id") or r.get("error"))
        return {"ok": not errors, "ids": ids, "errors": errors}

    # --------------------------------------------------------------------------------------------------------
    # intake panel (解析面板): parse -> resolve -> estimate -> stream / add
    # --------------------------------------------------------------------------------------------------------

    EXPIRED_ERROR = "解析結果已過期，請重新解析"

    def parse_inputs(self, text) -> dict:
        """Split pasted text into entries: {"ok", "entries": [{"text", "kind", "target", "headers_count"}], "errors"}."""
        errors: list = []
        try:
            found = open_input.parse_many(None if text is None else str(text), errors)
        except open_input.OpenInputError as e:
            return {"ok": False, "error": str(e)}
        if not found:
            return {"ok": False, "error": errors[0] if errors else "沒有可開啟的網址或檔案"}
        return {"ok": True, "errors": errors,
                "entries": [{"text": oi.text, "kind": oi.kind, "target": oi.target,
                             "headers_count": len(oi.headers)} for oi in found]}

    @staticmethod
    def _grab_thumb(r: "source_info.Resolved") -> Optional[str]:
        if r.cover:
            return r.cover
        if r.cover_src:
            from oh_streamer.sources.resolvers.covers import fetch_http_cover
            cover = fetch_http_cover(*r.cover_src)
            if cover:
                r.cover = cover
                return cover
        if not r.thumb_src:
            return None
        return source_info.grab_thumbnail(r.thumb_src[0], r.thumb_src[1], r.duration)

    def _dupe_of(self, target: str) -> Optional[dict]:
        """The existing queue item with this target (the best state first, like _lookup), else None."""
        found = [it for it in self.store.items() if it.target == target]
        found.sort(key=lambda it: STATE_PRIORITY.index(it.state) if it.state in STATE_PRIORITY else 99)
        for it in found:
            if it.state == "done" and not os.path.isfile(it.output_path):
                continue
            percent = 100.0 if it.state == "done" else (it.progress or {}).get("percent")
            return {"id": it.id, "state": it.state,
                    "progress": round(min(1.0, max(0.0, percent / 100.0)), 4) if _finite_num(percent) else None}
        return None

    def resolve(self, text) -> dict:
        """Blocking (seconds): one entry's text -> {"ok": True, "info": Info} or {"ok": False, "error", "detail"}."""
        try:
            oi = open_input.parse(str(text or ""))
        except open_input.OpenInputError as e:
            return {"ok": False, "error": str(e), "detail": str(e)}
        try:
            r = self._source_resolver(oi)
        except source_info.ResolveError as e:
            return {"ok": False, "error": e.short, "detail": e.detail}
        except Exception as e:  # noqa: BLE001 - whatever the resolver raises, the panel gets an error line
            log(f"resolve failed for {oi.target[:120]}: {e!r}")
            return {"ok": False, "error": "無法解析這個來源", "detail": source_info.error_line(e)}
        try:
            thumb = self._thumb_grabber(r)
        except Exception as e:  # noqa: BLE001 - a missing thumbnail is not an error
            log(f"thumbnail failed for {oi.target[:120]}: {e!r}")
            thumb = None
        info = {"token": self._tokens.put(r), "kind": r.kind, "target": r.target, "title": r.title, "site": r.site,
                "duration": r.duration, "fps": r.fps, "heights": [dict(h) for h in r.heights],
                "default_height": r.default_height, "vcodec": r.vcodec, "acodec": r.acodec, "hls": r.hls,
                "headers_count": len(r.headers), "thumb": thumb, "dupe": self._dupe_of(r.target),
                "subtitle_audio": self._subtitle_audio_kind(r)}
        return {"ok": True, "info": info}

    @staticmethod
    def _subtitle_audio_kind(r) -> Optional[str]:
        """Where streaming subtitles would read their audio from ("audio_only" | "lowest" | "same" | "file"); None when
        the source offers nothing readable (the panel then falls back to counting the qualities)."""
        try:
            return stream_audio.pick_subtitle_audio(r)["kind"]
        except Exception as e:  # noqa: BLE001 - the panel must open whatever the source looks like
            log(f"subtitle audio pick failed for {r.target[:120]}: {type(e).__name__}")
            return None

    # --- settings seen by the panel -------------------------------------------------------------------------

    def _player_cfg(self) -> dict:
        """The effective saved player settings (what the player starts with)."""
        from oh_streamer.player import player_settings as PS
        try:
            saved, _warn = PS.load_file(self.perf._player_file())
        except (OSError, ValueError):
            saved = {}
        return PS.fix_platform(PS.effective(saved, {}))[0]

    def _queue_cfg(self) -> dict:
        """The effective queue settings in the panel's vocabulary (interpolation "off" | "auto")."""
        s = self._effective_queue_settings()
        return {"detection_model": s.get("detection_model"), "restoration_model": s.get("restoration_model"),
                "interpolation": "auto" if s.get("interpolation") == "full" else "off",
                "encoder": s.get("encoder", "hevc"), "save_compare": bool(s.get("save_compare")),
                "compare_original": s.get("compare_original")}

    def defaults(self) -> dict:
        """The settings the panel starts from: the saved player settings and the effective queue settings."""
        p, q = self._player_cfg(), self._queue_cfg()
        try:
            _gen, prof = self.perf.current_profile()
        except Exception:  # noqa: BLE001
            prof = None
        p = {**p, "detection_model": q["detection_model"], "restoration_model": q["restoration_model"]}   # one setting for both
        return {"stream": {k: p.get(k) for k in ("detection_model", "restoration_model", "interpolation",
                                                 "restoration_fps")},
                "queue": {k: q[k] for k in ("detection_model", "restoration_model", "interpolation", "encoder",
                                            "save_compare") } | {"subtitles": False},
                "choices": self._model_options(), "has_profile": prof is not None,
                "subtitle_environment": self._subtitle_environment()}

    @staticmethod
    def _heights_of(r: "source_info.Resolved") -> list:
        return [h["height"] for h in r.heights if h.get("height")]

    @staticmethod
    def _offered_height(r: "source_info.Resolved", overrides: dict) -> Optional[int]:
        """overrides["height"] when this source offers that rendition, else None."""
        h = overrides.get("height")
        return int(h) if isinstance(h, (int, float)) and not isinstance(h, bool) and int(h) in {
            x["height"] for x in r.heights} else None

    @staticmethod
    def _stream_mode(fps, overrides: dict, pcfg: dict) -> tuple[str, str]:
        """(interpolation, restoration_fps) a stream of a rendition at `fps` runs with: the panel's overrides over the
        saved player settings; restoration_fps "30" on a > 31 fps source forces interpolation "auto" (the player
        interpolates back up when it is fast enough)."""
        rfps = overrides.get("restoration_fps")
        rfps = rfps if rfps in ("original", "30") else pcfg.get("restoration_fps", "original")
        interp = overrides.get("interpolation")
        interp = interp if interp in ("off", "auto") else pcfg.get("interpolation", "off")
        if fps and fps > STREAM_HIGH_FPS and rfps == "30":
            interp = "auto"
        if not platform_support.capabilities().get("rife", True):
            interp = "off"
        return interp, rfps

    def _stream_overrides(self, r: "source_info.Resolved", overrides: dict, height) -> dict:
        """The stream keys of the panel's overrides, validated (+ the > 31 fps rule) for the player's open message."""
        from oh_streamer.player import player_settings as PS
        out = {}
        qcfg, pcfg = self._queue_cfg(), self._player_cfg()
        for key in ("detection_model", "restoration_model", "interpolation", "restoration_fps"):
            value = overrides.get(key)
            if key not in overrides and key in ("detection_model", "restoration_model") and qcfg.get(key) != pcfg.get(key):
                value = qcfg.get(key)       # an older divergence between the dialog value and the player file: dialog wins
            if value is not None:
                try:
                    out[key] = PS.validate(key, value)
                except ValueError:
                    pass
        fps = r.fps_at(height if height is not None else r.default_height)
        if fps and fps > STREAM_HIGH_FPS and out.get("restoration_fps") == "30":
            out["interpolation"] = "auto"
        return out

    def _queue_ahead(self, has_profile: bool) -> tuple[int, Optional[float]]:
        """(position a new item would get, summed remaining ETA of the items ahead of it). Paused / failed / done items
        are not ahead. The sum covers the items whose ETA is known and is None without a performance profile."""
        stored = self.store.items()
        ahead = [it for it in stored if it.state in ("waiting", "downloading", "queued", "processing")]
        if not has_profile:
            return len(ahead) + 1, None
        eta = self._eta_estimates(stored, self._effective_queue_settings())
        total = 0.0
        for it in ahead:
            prog = it.progress if isinstance(it.progress, dict) else {}
            value = prog.get("eta") if it.state == "processing" and _finite_num(prog.get("eta")) and prog["eta"] > 0 \
                else eta.get(it.id)
            if _finite_num(value):
                total += value
        return len(ahead) + 1, round(total, 1)

    def estimate(self, tokens, overrides=None) -> dict:
        """Speed estimates of resolved sources for the panel's current settings. The stream
        verdict uses the player settings + the panel's overrides, the queue numbers the queue settings."""
        ov = overrides if isinstance(overrides, dict) else {}
        tokens = [tokens] if isinstance(tokens, str) else list(tokens or [])
        try:
            _gen, prof = self.perf.current_profile()
        except Exception as e:  # noqa: BLE001
            log(f"estimate: profile unavailable: {e!r}")
            prof = None
        try:
            position, ahead_s = self._queue_ahead(prof is not None)
        except Exception as e:  # noqa: BLE001
            log(f"estimate: queue position failed: {e!r}")
            position, ahead_s = len(self.store.items()) + 1, None
        out = {"has_profile": prof is not None, "queue_position": position, "queue_ahead_s": ahead_s, "items": []}
        if prof is not None:
            pcfg, qcfg = self._player_cfg(), self._queue_cfg()
        for token in tokens:
            entry = {"token": token, "stream": None, "stream_by_height": {}, "queue": None}
            r = self._tokens.get(token)
            if r is None:
                entry["error"] = self.EXPIRED_ERROR
            elif prof is not None:
                try:
                    self._estimate_one(entry, r, ov, prof, pcfg, qcfg, single=len(tokens) == 1)
                except Exception as e:  # noqa: BLE001 - never fail the whole panel for one item
                    log(f"estimate failed for {r.target[:120]}: {e!r}")
            out["items"].append(entry)
        return out

    def _estimate_one(self, entry: dict, r: "source_info.Resolved", ov: dict, prof: dict, pcfg: dict, qcfg: dict,
                      single: bool) -> None:
        from oh_streamer.perf import profile as P
        heights = self._heights_of(r)
        chosen = self._offered_height(r, ov)
        stream_h = chosen or r.default_height
        # the queue downloads "best" unless a quality is sent (add_resolved sends the chosen height only)
        queue_h = chosen or (heights[0] if heights and r.quals is not None else r.default_height)
        sview = perf_hint.for_detector(prof, ov.get("detection_model") or pcfg.get("detection_model"))

        def stream_verdict(height, forced_rfps=None):
            fps = r.fps_at(height) or perf_hint.DEFAULT_FPS
            interp, rfps = self._stream_mode(fps, dict(ov, restoration_fps=forced_rfps) if forced_rfps else ov, pcfg)
            need = perf_hint.restored_fps(fps, rfps)
            v = perf_hint.verdict_for(sview, height, need, interp)
            return v, need, interp, rfps, fps

        for h in heights:
            v = stream_verdict(h)[0]
            if v is not None:
                entry["stream_by_height"][str(h)] = v["level"]
        if single and stream_h:
            v, need, interp, rfps, fps = stream_verdict(stream_h)
            if v is not None:
                ratio = v["ratio_typical"]
                suggest = None
                if v["level"] != "live":
                    if fps > STREAM_HIGH_FPS and rfps != "30":
                        v30 = stream_verdict(stream_h, "30")[0]
                        if v30 is not None and v30["level"] == "live":
                            suggest = {"restoration_fps": "30"}
                    if suggest is None:
                        for h in heights:
                            if h < stream_h and entry["stream_by_height"].get(str(h)) == "live":
                                suggest = {"height": h}
                                break
                entry["stream"] = {
                    "verdict": v["level"], "restore_fps": round(ratio * need, 2), "need_fps": round(float(need), 3),
                    "ratio": round(ratio, 3),
                    "prebuffer_s": None if v["prebuffer_s"] is None else round(v["prebuffer_s"], 1),
                    "interp_share": round(v["interp_fraction"], 3) if interp == "auto" and v["interp_ok"] else None,
                    "detector": ov.get("detection_model") or pcfg.get("detection_model"), "suggest": suggest}
                if ov.get("subtitles"):
                    gpu = self._stream_subtitle_gpu(v, ov)
                    if gpu is not None:
                        entry["stream"]["subtitle_gpu"] = gpu
        if queue_h and r.duration:
            fps = r.fps_at(queue_h) or perf_hint.DEFAULT_FPS
            interp = ov.get("interpolation") if ov.get("interpolation") in ("off", "auto") else qcfg["interpolation"]
            mode = "full" if interp == "auto" else "off"
            if not platform_support.capabilities().get("rife", True):
                mode = "off"
            save_compare = bool(ov["save_compare"]) if "save_compare" in ov else qcfg["save_compare"]
            compare = "encode" if save_compare and qcfg.get("compare_original") == "encode" else None
            qview = P.for_detector(prof, ov.get("detection_model") or qcfg.get("detection_model"))
            est = P.estimate(qview, height=queue_h, src_fps=fps, mode=mode, density="typical", compare=compare)
            eta = P.job_seconds(qview, duration_s=r.duration, height=queue_h, src_fps=fps, mode=mode, compare=compare)
            if _finite_num(est.get("fps")) and _finite_num(eta) and eta > 0:
                entry["queue"] = {"fps": round(float(est["fps"]), 2), "eta_s": float(eta)}

    def _stream_subtitle_gpu(self, base: dict, ov: dict) -> Optional[dict]:
        """T4.3: the stream verdict with local subtitles counted in (queue/stream_subtitle_gpu.py), or None for cloud
        subtitles / unknown configuration / any failure (the plain verdict then stands). Local = the primary subtitle
        row is not cloud by combos.is_cloud_combo (the rule the scheduler uses to pick the executor; a frozen cloud-first
        plan with any cloud primary model, even with a local fallback, is cloud)."""
        try:
            from oh_streamer.queue import stream_subtitle_gpu as ssg
            from oh_streamer.queue import gpu_turns as gt
            from oh_streamer.subtitles import combos
            snapshot = self._subtitle_snapshot(ov)
            rows = snapshot.get("subtitle_configs") or []
            if not rows or combos.is_cloud_combo(rows[0], snapshot):
                return None
            info = self._subtitle_gpu_info()
            from .subtitle_host_gpu import stream_turn_config
            cfg = stream_turn_config(None)        # the turn size the stream item's executor really uses
            key, memory = gt.gpu_key(info.name, info.total_mib), ssg.SubtitleSpeedMemory()
            learned = memory.get(key)
            s = ssg.subtitle_seconds_per_media_second(key, memory, cfg)
            load_s, restart_s, costs_learned = ssg.switch_costs(key, self._subtitle_cost_memory(), cfg)
            with_sub = ssg.verdict_with_subtitles(base, s, ssg.switch_cost_per_media_s(cfg, load_s, restart_s))
            held, held_learned = ssg.held_restore_mib(key, self._subtitle_restore_vram_memory())
            vram_ok, need = ssg.vram_fits(info.total_mib, held)
            level = with_sub["level"]
            suggest = []
            if level != "live" or not vram_ok:
                suggest.append({"kind": "queue"})
                if level == "prebuffer" and vram_ok:
                    suggest.append({"kind": "prebuffer", "seconds": round(float(with_sub["prebuffer_s"] or 0.0), 1)})
                if self._cloud_subtitles_usable(snapshot):
                    suggest.append({"kind": "cloud"})
            return {"verdict": level, "ratio": _round_or_none(with_sub["ratio_typical"], 3),
                    "ratio_heavy": _round_or_none(with_sub["ratio_heavy"], 3),
                    "prebuffer_s": _round_or_none(with_sub["prebuffer_s"], 1), "verdict_without": base.get("level"),
                    "s": round(float(s), 4), "learned": learned is not None,
                    "load_s": round(float(load_s), 2), "restart_s": round(float(restart_s), 2),
                    "costs_learned": bool(costs_learned), "vram_ok": bool(vram_ok),
                    "vram_need_mib": int(round(need)), "restore_mib": int(round(held)),
                    "restore_mib_learned": held_learned, "suggest": suggest}
        except Exception as e:  # noqa: BLE001 - the plain stream verdict still stands
            log(f"estimate: subtitle gpu verdict failed: {e!r}")
            return None

    @staticmethod
    def _cloud_subtitles_usable(snapshot: dict) -> bool:
        """Could the cloud profile run right now? Its frozen resolution is ready: models configured and verified,
        and the cloud consent given (configuration.resolve reports `cloud_disabled` / `not_configured`)."""
        try:
            import copy
            from oh_streamer.subtitles.configuration import resolve
            profile = copy.deepcopy(snapshot.get("subtitle_profile") or {})
            if not profile.get("profiles", {}).get("cloud"):
                return False
            profile["active_mode"] = "cloud"
            return bool(resolve(profile, snapshot).get("ready"))
        except Exception as e:  # noqa: BLE001
            log(f"estimate: cloud readiness failed: {e!r}")
            return False

    # --- acting on resolved sources ---------------------------------------------------------------------------

    def stream_resolved(self, token, overrides=None) -> dict:
        """stream() for a resolved source: the player opens it with the chosen quality and per-source overrides."""
        r = self._tokens.get(token)
        if r is None:
            return {"ok": False, "error": self.EXPIRED_ERROR}
        if r.kind == "file" and not os.path.isfile(r.target):
            return {"ok": False, "error": "找不到影片檔案"}
        ov = overrides if isinstance(overrides, dict) else {}
        height = self._offered_height(r, ov)
        payload = {"cmd": "open", "input": r.target, "headers": dict(r.headers), "quality": height,
                   "overrides": self._stream_overrides(r, ov, height)}
        item = None
        if ov.get("subtitles"):
            try:
                snapshot = self._subtitle_snapshot(ov)
            except (ValueError, TypeError) as exc:
                text = getattr(exc, "message", None) or str(exc)      # ConfigurationError: zh-TW text, not the code
                return {"ok": False, "error": text, "code": getattr(exc, "code", None)}
            settings = {"subtitles": True}                  # nothing frozen: the item follows the global settings
            overrides = self._subtitle_overrides(ov, snapshot)
            if overrides:
                settings["subtitle_overrides"] = overrides
            # The player still reads the source itself (unchanged); the item only carries the subtitle scheduling data.
            item = self.store.add_stream(r.target, r.kind, dict(r.headers), settings,
                                         title=r.title or self._default_title(r.target, r.kind))
            payload["stream_item"] = item.id
            with self._lock:
                self._stream_created[item.id] = self._clock()
            self._attach_stream_audio(item.id, r, height or r.default_height)
        previous =[it.id for it in self.store.stream_items() if not item or it.id != item.id]
        result = self._send_open(r.target, payload)
        # one player, one stream: the new video replaces the old stream item, or the new item goes when the player refused
        for stale in (previous if result.get("ok") else [item.id] if item else []):
            self._drop_stream_item(stale)
        if item and result.get("ok"):
            result["stream_item"] = item.id
        return result

    def _attach_stream_audio(self, item_id: str, r, played_height) -> None:
        """Give the stream item its subtitle audio (marker, refresher, background start-time probe: queue/stream_audio.py).
        The probe never delays the player; a source with nothing readable marks the item failed instead of refusing."""
        try:
            pick = stream_audio.pick_subtitle_audio(r)
            stream_audio.attach(self, item_id, r, pick, played_height=played_height)
        except Exception as e:  # noqa: BLE001 - the video still plays; the subtitles report the failure
            log(f"stream item {item_id}: subtitle audio not attached ({type(e).__name__})")
            with self._lock:
                item = self.store.get(item_id)
                if item is not None:
                    self.store.update(item_id, stream={**item.stream, "audio_status": "failed",
                                                       "audio_error": stream_audio.ERR_UNSUPPORTED})

    def _subtitle_overrides(self, ov: dict, snapshot: Optional[dict] = None) -> dict:
        """The per-video subtitle choices of an add (``subtitle_comparison_enabled`` only), kept only when they
        differ from the global value; everything else follows the global settings until the item starts."""
        value = ov.get("subtitle_comparison_enabled")
        if not isinstance(value, bool):
            return {}
        try:
            default = subtitle_jobs.snapshot_subtitle_settings(self.store.get_settings()).get("subtitle_comparison_enabled")
        except (ValueError, TypeError):
            default = (snapshot or {}).get("subtitle_comparison_enabled")
        return {} if value == default else {"subtitle_comparison_enabled": value}

    def _subtitle_snapshot(self, ov: dict) -> dict:
        """The subtitle settings a new item (queued or stream) would run with, used to VALIDATE the add (nothing is
        stored from it); ValueError/ConfigurationError when the subtitle configuration is not ready."""
        base = self.store.get_settings()
        if "subtitle_comparison_enabled" in ov:
            base["subtitle_comparison_enabled"] = ov["subtitle_comparison_enabled"]
        return subtitle_jobs.snapshot_subtitle_settings(base)

    def add_resolved(self, tokens, overrides=None) -> dict:
        """Queue resolved sources. quality = overrides["height"] when the source offers it (else None); model overrides
        go into item.settings (cache key); interpolation / encoder / save_compare are pinned per item."""
        ov = overrides if isinstance(overrides, dict) else {}
        tokens = [tokens] if isinstance(tokens, str) else list(tokens or [])
        options = self._model_options()
        models, errors = {}, []
        for key in ("detection_model", "restoration_model"):
            value = ov.get(key)
            if isinstance(value, str) and value.strip():
                if options.get(key) and value not in {o["value"] for o in options[key]}:
                    return {"ok": False, "ids": [], "errors": ["模型未安裝，請選擇可用的模型"], "message": ""}
                models[key] = value.strip()
        pinned = {}
        if ov.get("interpolation") in ("off", "auto"):
            pinned["interpolation"] = "full" if ov["interpolation"] == "auto" else "off"
        if ov.get("encoder") in ("hevc", "h264"):
            pinned["encoder"] = ov["encoder"]
        if "save_compare" in ov:
            pinned["save_compare"] = bool(ov["save_compare"])
        subtitle_overrides = {}
        if bool(ov.get("subtitles", False)):
            try:
                subtitle_overrides = self._subtitle_overrides(ov, self._subtitle_snapshot(ov))
            except (ValueError, TypeError) as exc:
                text = getattr(exc, "message", None) or str(exc)      # ConfigurationError: zh-TW text, not the code
                return {"ok": False, "ids": [], "errors": [text], "message": "", "code": getattr(exc, "code", None)}
        ids = []
        for token in tokens:
            r = self._tokens.get(token)
            if r is None:
                errors.append(self.EXPIRED_ERROR)
                continue
            settings = {**self._snapshot_settings(), **models}
            settings["subtitles"] = bool(ov.get("subtitles", False))
            if subtitle_overrides:
                settings["subtitle_overrides"] = dict(subtitle_overrides)
            if pinned:
                settings["pinned"] = dict(pinned)
                for key in ("encoder", "interpolation"):      # what the item card shows until the job is built
                    if key in pinned:
                        settings[key] = pinned[key]
                if "save_compare" in pinned:
                    settings["compare"] = "on" if pinned["save_compare"] else "off"
            height = self._offered_height(r, ov) if r.quals is not None else None
            item = self.store.add(r.target, r.kind, dict(r.headers), height, settings,
                                  title=r.title or self._default_title(r.target, r.kind))
            if r.cover and r.cover.startswith("data:image/"):
                try:
                    folder = self.store.item_dir(item.id)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "cover.txt").write_text(r.cover, encoding="utf-8")
                except OSError as e:
                    log(f"cover save failed for {item.id}: {e!r}")
            ids.append(item.id)
        message = f"已加入 {len(ids)} 個項目" if ids else ""
        return {"ok": bool(ids) and not errors, "ids": ids, "errors": errors, "message": message}

    def save_defaults(self, overrides=None) -> dict:
        """「設為預設」: write the panel's changed keys to the player settings file (the way the performance check's
        apply does) and to the queue settings. height is ignored."""
        from oh_streamer.player import player_settings as PS
        ov = overrides if isinstance(overrides, dict) else {}
        player, queue = {}, {}
        for key in ("detection_model", "restoration_model"):
            value = ov.get(key)
            if isinstance(value, str) and value.strip():
                player[key] = queue[key] = value.strip()
        if ov.get("interpolation") in ("off", "auto"):
            player["interpolation"] = ov["interpolation"]
            queue["interpolation"] = "full" if ov["interpolation"] == "auto" else "off"
        if ov.get("restoration_fps") in ("original", "30"):
            player["restoration_fps"] = ov["restoration_fps"]
        if ov.get("encoder") in ("hevc", "h264"):
            queue["encoder"] = ov["encoder"]
        if "save_compare" in ov:
            queue["save_compare"] = bool(ov["save_compare"])
        if not player and not queue:
            return {"ok": True, "changed": [], "message": "沒有需要儲存的變更"}
        if queue:                                # first: it validates the models, so a refusal leaves the player file alone
            r = self.set_settings(queue)
            if not r.get("ok"):
                return {"ok": False, "error": r.get("error") or "無法儲存佇列設定"}
        if player:
            err = self._write_player_settings(player)
            if err:
                return {"ok": False, "error": f"無法寫入播放器設定：{err}"}
        return {"ok": True, "changed": sorted(set(player) | set(queue)), "message": "已設為預設"}

    def reload_player_settings(self) -> None:
        """The player settings file was rewritten: a running player keeps its own copy in memory (and would write
        it back on its next menu change), so tell it to re-read the file. No player running is not an error."""
        try:
            self._send_to_player({"cmd": "settings_reload"})
        except Exception as e:  # noqa: BLE001 - the file is written; the player picks it up at its next start
            log(f"could not tell the player to reload its settings: {e!r}")

    def _running_run(self, item_id: str) -> Optional[_Run]:
        with self._lock:
            for r in (self._dl, self._wk, *self._subtitle_runs()):
                if r and r.item_id == item_id:
                    return r
        return None

    def _running_runs(self, item_id: str) -> list[_Run]:
        with self._lock:
            return [r for r in (self._dl, self._wk, *self._subtitle_runs()) if r and r.item_id == item_id]

    @staticmethod
    def _send_goto(proc, k: int) -> None:
        try:
            proc.stdin.write((json.dumps({"cmd": "goto", "k": int(k)}) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (OSError, ValueError):       # broken pipe / closed: the worker is going away anyway
            pass

    @staticmethod
    def _send_clock(proc, pos: float, playing: bool) -> None:
        try:
            proc.stdin.write((json.dumps({"cmd": "clock", "pos": pos, "playing": playing}) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (OSError, ValueError):       # broken pipe / closed: the worker is going away anyway
            pass

    def goto(self, item_id: str, k: int) -> dict:
        """Tell the running worker of this item to work from segment k on; otherwise k becomes the start_k of its
        next worker run."""
        try:
            k = max(0, int(k))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad k"}
        with self._lock:
            run = self._wk
            if (
                run
                and run.item_id == item_id
                and run.stage == "restore"
                and run.proc is not None
                and not run.cancel.is_set()
            ):
                proc = run.proc
            else:
                self._start_k[item_id] = k
                return {"ok": True, "running": False}
        self._send_goto(proc, k)
        return {"ok": True, "running": True}

    def remove(self, item_id: str) -> dict:
        item = self.store.get(item_id)
        if item is None:
            return {"ok": False}
        with self._lock:
            if item_id in self._reprocessing:
                return {"ok": False, "error": "正在準備重新處理的來源，請稍後再刪除"}
            self._tombstones.add(item_id)
            self._generations[item_id] = self._generations.get(item_id, 0) + 1
            self._watch.pop(item_id, None)
            self._start_k.pop(item_id, None)
            self._worker_waiting_for_source.discard(item_id)
            if self._pending_play == item_id:
                self._pending_play = None
            if self._player_item == item_id:
                self._player_item = None
            probe = self._eta_probing.get(item_id)
        if probe is not None:
            probe.join(timeout=10)   # its ffprobe may hold the source open (Windows cannot delete open files)
        runs = self._running_runs(item_id)
        for run in runs:
            self._request_stop(run, "remove")
        self._subtitle_stop_executors(item_id, "remove")
        # Stop the localhost proxy before joining so blocked source reads wake promptly.
        self._cancel_close_source(item_id)
        for run in runs:
            if run.thread and run.thread is not threading.current_thread():
                run.thread.join(timeout=15)
        still_running = [r for r in runs if r.thread and r.thread.is_alive()]
        if still_running:
            return {"ok": False, "error": "工作尚未停止，來源檔案保留以避免寫入競態"}
        self.thumbs.forget(item_id)     # kills / waits out a grab that holds a file of the item folder
        ok = self.store.remove(item_id)
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        return {"ok": ok}

    def move(self, item_id: str, index: int) -> dict:
        try:
            self.store.move(item_id, int(index))
        except KeyError:
            return {"ok": False, "error": "unknown id"}
        return {"ok": True}

    def pause(self, item_id: str) -> dict:
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        if it.state == "done" and it.subtitle.get("only"):
            with self._lock:
                # Mark paused first so the stopping run keeps percent/phase and cannot requeue.
                self._subtitle_update(item_id, status="paused", phase=it.subtitle.get("phase", ""),
                                      detail=it.subtitle.get("detail", ""))
                for run in self._running_runs(item_id):
                    self._request_stop(run, "pause")
                self._subtitle_stop_executors(item_id, "pause")
            return {"ok": True}
        if it.state not in ("waiting", "downloading", "queued", "processing"):
            return {"ok": False, "error": f"cannot pause an item that is {it.state}"}
        fields = {"state": "paused"}
        if it.source_status in ("downloading", "complete"):
            fields["source_status"] = "paused" if it.source_status == "downloading" else "complete"
        with self._lock:
            self._generations[item_id] = self._generations.get(item_id, 0) + 1
            if self._pending_play == item_id:
                self._pending_play = None
        self.store.update(item_id, **fields)       # first: finishing runs must see the user pause
        for run in self._running_runs(item_id):
            self._request_stop(run, "pause")
        self._subtitle_stop_executors(item_id, "pause")
        return {"ok": True}

    def resume(self, item_id: str) -> dict:
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        self._subtitle_thaw(item_id, refresh=True, reset=True)      # blocked subtitles retry with the current settings
        it = self.store.get(item_id) or it
        if it.state == "done" and it.subtitle.get("only") and it.subtitle.get("status") == "paused":
            self._subtitle_update(item_id, status="mux_pending" if self._subtitle_mux_ready(it) else "queued", percent=0)
            self._wake.set()
            return {"ok": True}
        if it.state != "paused":
            return {"ok": False, "error": "not paused"}
        ready = self._source_ready(it)
        self.store.update(item_id, state="queued" if ready else "waiting",
                          source_status=it.source_status if ready else ("paused" if it.source_status else ""))
        return {"ok": True}

    @staticmethod
    def _range_data_file(marker: str):
        """(data file, extension) of a COMPLETE byte-range cache next to its .rcache marker, else None."""
        try:
            folder = Path(marker).resolve().parent
            side = json.loads((folder / "range-source.json").read_text(encoding="utf-8"))
            data = folder / "source.data"
            if side.get("status") != "complete" or not data.is_file() or data.stat().st_size != side.get("total_bytes"):
                return None
            return data, re.sub(r"[^A-Za-z0-9]", "", str(side.get("ext") or "")) or "mp4"
        except (OSError, ValueError, AttributeError):
            return None

    def _reprocess_source(self, it: queue_store.Item) -> dict:
        if it.kind == "file" and os.path.isfile(it.target):
            return {"mode": "file", "path": it.target, "lossy": False,
                    "message": "將使用本機原始檔，無需下載。"}
        if (it.source_path and os.path.isfile(it.source_path)
                and it.source_status in ("", "complete")):
            lossy = it.source_origin == "compare_encode"
            if it.source_path.lower().endswith(RANGE_CACHE_SUFFIX):
                data = self._range_data_file(it.source_path)
                if data is None:
                    return {"mode": "download", "lossy": False, "message": "需要重新解析原始連結並下載來源。"}
                return {"mode": "cache", "path": str(data[0]), "suffix": "." + data[1], "lossy": lossy,
                        "message": "將使用完整來源快取，無需重新下載。"}
            return {"mode": "cache", "path": it.source_path,
                    "lossy": lossy,
                    "message": ("來源快取來自已壓縮的對比影像，可使用此版本或重新下載來源。" if lossy else
                                "將使用完整來源快取，無需重新下載。")}
        if it.state == "done" and os.path.isfile(it.output_path):
            info = self._probe_tracks(it.output_path)
            original = next((s for s in (info or {}).get("streams", [])
                             if s.get("codec_type") == "video" and s.get("title") == "oh:original"), None)
            if original is not None:
                # Use the output's effective mode: requested copy can have fallen back to encode.
                mode = (info.get("tags") or {}).get("OH_ORIGINAL", "").lower()
                lossy = mode != "copy" or it.source_origin == "compare_encode"
                return {"mode": "compare", "path": it.output_path, "index": original["index"],
                        "lossy": lossy, "duration": info.get("duration", 0),
                        "message": ("原始對比影像已壓縮，可使用此版本或重新下載來源。" if lossy else
                                    "將取出成品內的原始對比影像，無需重新下載。")}
        if it.kind == "file":
            return {"mode": "missing", "lossy": False, "message": "本機原始檔已不存在，請重新指定原始檔。"}
        return {"mode": "download", "lossy": False, "message": "需要重新解析原始連結並下載來源。"}

    def reprocess_options(self, item_id: str) -> dict:
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "找不到佇列項目"}
        settings = {**self._snapshot_settings(), **it.settings}
        qs, pinned = self.store.get_settings(), self._pinned_of(it)
        for key in ("encoder", "interpolation", "compare_original", "compare_quality"):
            settings[key] = pinned.get(key, settings.get(key, qs[key]) if it.state == "done" else qs[key])
        settings["save_compare"] = pinned.get("save_compare", settings.get("compare") == "on"
                                              if it.state == "done" else qs["save_compare"])
        source = self._reprocess_source(it)
        edit_in_place = (it.state == "waiting" and not it.source_path and not self._running_runs(item_id)
                         and not (self.store.item_dir(item_id) / "work").exists())
        # Do not expose internal paths or stream indices to the UI.
        return {"ok": True, "settings": settings, "quality": it.quality, "state": it.state,
                "target": it.target, "kind": it.kind, "edit_in_place": edit_in_place,
                "source": {key: source[key] for key in ("mode", "message", "lossy")}}

    def reprocess(self, item_id: str, changes: dict, options=None) -> dict:
        """Prepare an independent source and settings snapshot before scheduling a new version.

        The prior output is never overwritten; paused staging items cannot be picked by the scheduler.
        """
        if not isinstance(changes, dict) or (options is not None and not isinstance(options, dict)):
            return {"ok": False, "error": "設定格式不正確"}
        options = options or {}
        if any(key in options and not isinstance(options[key], bool) for key in ("redownload", "stop_current")):
            return {"ok": False, "error": "重新處理選項格式不正確"}
        new = None
        with self._lock:
            it = self.store.get(item_id)
            if it is None or item_id in self._tombstones:
                return {"ok": False, "error": "找不到佇列項目"}
            if item_id in self._reprocessing or item_id in self._compare_busy:
                return {"ok": False, "error": "此項目正在準備來源，請稍後再試"}
            self._reprocessing.add(item_id)
            self.store.protected_ids = set(self.store.protected_ids) | {item_id}
        try:
            view = self.reprocess_options(item_id)
            settings = dict(view["settings"])
            choices = {"encoder": ("hevc", "h264"), "interpolation": ("full", "off"),
                       "compare_original": queue_store.COMPARE_ORIGINAL_CHOICES,
                       "compare_quality": queue_store.COMPARE_QUALITY_CHOICES}
            for key, values in choices.items():
                if key in changes:
                    if changes[key] not in values:
                        return {"ok": False, "error": f"無效設定：{key}"}
                    settings[key] = changes[key]
            if "save_compare" in changes:
                if not isinstance(changes["save_compare"], bool):
                    return {"ok": False, "error": "比對設定格式不正確"}
                settings["save_compare"] = changes["save_compare"]
            model_options = self._model_options()
            for key in ("detection_model", "restoration_model"):
                if key in changes:
                    value = changes[key]
                    available = {o["value"] for o in model_options.get(key, [])}
                    if not isinstance(value, str) or value not in available:
                        return {"ok": False, "error": "模型未安裝，請選擇可用的模型"}
                    settings[key] = value
            quality = changes.get("quality", it.quality)
            if quality is not None and quality not in ("best", "worst"):
                if isinstance(quality, bool) or not isinstance(quality, int) or quality <= 0:
                    return {"ok": False, "error": "下載解析度格式不正確"}
            settings["pinned"] = {key: settings[key] for key in (*choices, "save_compare")}
            settings["compare"] = "on" if settings["save_compare"] else "off"
            settings["engine"] = self._queue_engine()          # a new version is made by the engine in use now
            # Remove stale pixel/cache metadata; every version gets fresh segments.
            target = it.target
            replacement = options.get("source_path", "")
            if replacement:
                if it.kind != "file" or not isinstance(replacement, str) or not os.path.isfile(replacement):
                    return {"ok": False, "error": "指定的原始檔不存在"}
                target = str(Path(replacement).resolve())
            source = self._reprocess_source(it) if target == it.target else {
                "mode": "file", "path": target, "lossy": False}
            if it.kind == "url" and (options.get("redownload") or quality != it.quality):
                source = {"mode": "download", "lossy": False}
            if source["mode"] == "missing":
                return {"ok": False, "error": "請重新指定原始檔"}
            with self._lock:
                current = self.store.get(item_id)
                if (current and current.state == "waiting" and not current.source_path
                        and not self._running_runs(item_id)
                        and not (self.store.item_dir(item_id) / "work").exists()):
                    self.store.update(item_id, target=target, quality=quality, settings=settings,
                                      cache_key=queue_store.cache_key(target, quality, settings),
                                      error="", progress={})
                    self._wake.set()
                    return {"ok": True, "id": item_id, "message": "已更新此項目的處理設定"}
            needed = (os.path.getsize(source["path"]) if source["mode"] in ("cache", "compare")
                      and not source["path"].lower().endswith(".m3u8") else DEFAULT_DOWNLOAD_ESTIMATE)
            if source["mode"] in ("cache", "compare") and not self.store.ensure_space(needed):
                return {"ok": False, "error": SPACE_NOTICE}
            new = self.store.add(target, it.kind, it.headers, quality, settings, title=it.title,
                                 force=True, state="paused")
            with self._lock:
                self._reprocessing.add(new.id)
            self.store.update(new.id, parent_id=it.id)
            path = ""
            origin = "original"
            if source["mode"] == "file":
                path = source["path"]
            elif source["mode"] in ("cache", "compare"):
                folder = self.store.item_dir(new.id)
                folder.mkdir(parents=True, exist_ok=True)
                src = source["path"]
                remux = source["mode"] == "compare" or src.lower().endswith(".m3u8")
                dest = folder / ("source.mkv" if remux else "source" + source.get("suffix", Path(src).suffix))
                if remux:
                    video = f"0:{source['index']}" if source["mode"] == "compare" else "0:v:0"
                    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                           "-map", video, "-map", "0:a?", "-map_metadata", "-1", "-c", "copy",
                           "-f", "matroska", str(dest)]
                    result = self._run_cmd(cmd)
                    if result.returncode != 0 or not dest.is_file():
                        raise RuntimeError("無法取出原始影像：" + (result.stderr or "")[-300:])
                    info = self._probe_tracks(str(dest))
                    if not info or not any(s.get("codec_type") == "video" for s in info["streams"]):
                        raise RuntimeError("取出的原始影像無法讀取")
                    if source.get("duration") and abs(info.get("duration", 0) - source["duration"]) > DURATION_TOLERANCE:
                        raise RuntimeError("取出的原始影像長度不符")
                else:
                    shutil.copy2(src, dest)
                path = str(dest)
                origin = ("compare_encode" if source["lossy"] else
                          "compare_copy" if source["mode"] == "compare" else "original")
            if options.get("stop_current") and it.state in ("waiting", "downloading", "queued", "processing"):
                self.pause(item_id)
            self.store.update(new.id, source_path=path, source_origin=origin,
                              state="queued" if path else "waiting")
            if target == it.target and quality == it.quality and not options.get("redownload"):
                # D6: only the content-addressed stage cache and the slice table travel (no combos/, no old ASR files)
                old_subs = self.store.item_dir(it.id) / "subtitles"
                new_subs = self.store.item_dir(new.id) / "subtitles"
                try:
                    if (old_subs / "cache").is_dir() and not (new_subs / "cache").exists():
                        new_subs.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(old_subs / "cache", new_subs / "cache")
                    if (old_subs / "slice-plan.json").is_file() and not (new_subs / "slice-plan.json").exists():
                        new_subs.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(old_subs / "slice-plan.json", new_subs / "slice-plan.json")
                except OSError as exc:
                    shutil.rmtree(new_subs / "cache", ignore_errors=True)
                    (new_subs / "slice-plan.json").unlink(missing_ok=True)
                    log(f"Subtitle cache could not be inherited: {exc!r}")
            self._wake.set()
            return {"ok": True, "id": new.id, "message": "已加入重新處理版本，舊版本已保留"}
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if new is not None:
                self.store.remove(new.id)
            return {"ok": False, "error": str(exc)}
        finally:
            with self._lock:
                self._reprocessing.discard(item_id)
                if new is not None:
                    self._reprocessing.discard(new.id)
                self.store.protected_ids = (
                    self._live_watched_locked()
                    | self._reprocessing
                    | self._subtitle_reserved_ids()
                )

    def retry(self, item_id: str) -> dict:
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        if it.state != "failed":
            return {"ok": False, "error": "not failed"}
        self._subtitle_thaw(item_id, refresh=True, reset=True)
        ready = self._source_ready(it)
        with self._lock:
            download = self._dl if self._dl and self._dl.item_id == item_id else None
            worker = self._wk if self._wk and self._wk.item_id == item_id else None
            keep_download = bool(download and not download.cancel.is_set()
                                 and it.source_status in ("downloading", "complete"))
            if not keep_download:
                self._generations[item_id] = self._generations.get(item_id, 0) + 1
            self._worker_waiting_for_source.discard(item_id)
            if self._pending_play == item_id:
                self._pending_play = None
        if worker:
            self._request_stop(worker, "retry")
        if download and not keep_download:
            self._request_stop(download, "retry")
        source_status = (it.source_status if ready or keep_download
                         else ("paused" if it.source_status else ""))
        self.store.update(item_id, state="queued" if ready else "waiting", error="", progress={},
                          source_status=source_status,
                          download_progress={} if not ready and not keep_download else it.download_progress)
        with self._lock:
            if self._player_item == item_id and not ready:
                self._player_item = None
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        return {"ok": True}

    def pause_all(self) -> dict:
        with self._lock:
            self._paused_all = True
            # a stream item's subtitles are not the background queue's (design D1): its slot keeps running
            runs = [r for r in (self._dl, self._wk, *self._subtitle_runs())
                    if r and r is not getattr(self, "_subtitle_stream_cloud", None)]
        for run in runs:
            self._request_stop(run, "paused_all")
        for item_id in {r.item_id for r in runs if r.stage in ("subtitle_cloud", "subtitle_local")}:
            self._subtitle_stop_executors(item_id, "paused_all")
        self._changed()
        return {"ok": True}

    def resume_all(self) -> dict:
        with self._lock:
            self._paused_all = False
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        self._subtitle_thaw_all(refresh=True)
        self._changed()
        return {"ok": True}

    def set_keep(self, item_id: str, keep: bool) -> dict:
        try:
            self.store.update(item_id, keep=bool(keep))
        except KeyError:
            return {"ok": False, "error": "unknown id"}
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        return {"ok": True}

    def clear_headers(self, item_id: str) -> dict:
        try:
            self.store.clear_headers(item_id)
        except KeyError:
            return {"ok": False, "error": "unknown id"}
        return {"ok": True}

    def set_settings(self, changes: dict) -> dict:
        allowed = {k: v for k, v in (changes or {}).items() if k in queue_store.DEFAULT_SETTINGS}
        if 'subtitle_profile' in allowed:
            from oh_streamer.subtitles.configuration import normalize, resolve
            try:
                allowed['subtitle_profile'] = normalize(allowed['subtitle_profile'])
                resolved = resolve(allowed['subtitle_profile'], {**self.store.get_settings(), **allowed})
                if any(e['code'] == 'local_mode_cloud_candidate' for e in resolved['errors']):
                    return dict(ok=False, code='local_mode_cloud_candidate', error='混合雲端模型請使用自訂模式')
            except (ValueError, TypeError):
                return dict(ok=False, code='invalid_configuration', error='字幕配置格式無效')
        if "subtitle_asr_mode" in allowed and allowed["subtitle_asr_mode"] not in ("dual", "single"):
            return {"ok": False, "error": "ASR 模式必須是 dual 或 single"}
        for key in (
            "subtitle_asr",
            "subtitle_translation",
            "subtitle_python",
            "subtitle_models",
            "subtitle_server",
            "custom_models_directory",
        ):
            if key in allowed and not isinstance(allowed[key], str):
                return {"ok": False, "error": "字幕設定格式不正確"}
        for key in ("subtitle_interjections", "subtitle_japanese", "subtitle_cloud_enabled"):
            if key in allowed and not isinstance(allowed[key], bool):
                return {"ok": False, "error": "字幕設定格式不正確"}
        if "subtitle_cloud_strategy" in allowed and allowed["subtitle_cloud_strategy"] not in ("smart_horizon", "full_preload"):
            return {"ok": False, "error": "雲端字幕處理策略必須是 smart_horizon 或 full_preload"}
        if "subtitle_combos_default" in allowed:
            current_settings = {**self.store.get_settings(), **allowed}
            checked, error = subtitle_jobs.validate_combo_ids(allowed["subtitle_combos_default"], settings=current_settings)
            if checked is None:
                return {"ok": False, "error": error}
            allowed["subtitle_combos_default"] = checked
        if any(key in allowed for key in ("subtitle_primary", "subtitle_comparison_enabled", "subtitle_comparison_rows")):
            try:
                allowed.update(subtitle_jobs.subtitle_configuration({**self.store.get_settings(), **allowed}))
            except (ValueError, TypeError) as exc:
                return {"ok": False, "error": str(exc)}
        if "engine" in allowed:
            from oh_streamer import engines
            if allowed["engine"] in ("", None):
                allowed["engine"] = None                      # inherit the player's setting
            elif allowed["engine"] not in engines.visible_engine_names():
                return {"ok": False, "error": "不支援的修復引擎"}
        options = self._model_options()
        for key in ("detection_model", "restoration_model"):
            if key in allowed and allowed[key] not in {o["value"] for o in options[key]}:
                return {"ok": False, "error": "模型未安裝，請選擇可用的模型"}
        if "max_clip_length" in allowed:
            from oh_streamer.player import player_settings as PS
            try:
                allowed["max_clip_length"] = PS.validate("max_clip_length", allowed["max_clip_length"])
            except ValueError:
                return {"ok": False, "error": "處理幀數必須是 1 到 10000 的整數"}
        if not platform_support.supports_shutdown():
            allowed["shutdown_when_done"] = False
        if not platform_support.capabilities()["rife"]:
            allowed["interpolation"] = "off"
        if "cap_gb" in allowed:
            try:
                allowed["cap_gb"] = max(1, float(allowed["cap_gb"]))
            except (TypeError, ValueError):
                del allowed["cap_gb"]
        for k in ("prevent_sleep", "shutdown_when_done", "save_compare", "show_thumbnails", "show_thumbnail_cover"):
            if k in allowed:
                allowed[k] = bool(allowed[k])
        for key, choices in (("compare_original", queue_store.COMPARE_ORIGINAL_CHOICES),
                             ("compare_quality", queue_store.COMPARE_QUALITY_CHOICES)):
            if key in allowed and allowed[key] not in choices:
                del allowed[key]                       # unknown value: ignored, like an unknown key
        if allowed.get("shutdown_when_done") is False:
            self._shutdown_notice = ""
        settings = self.store.set_settings(**allowed)
        if allowed.get('subtitle_cloud_enabled') is False:
            from oh_streamer.subtitles.configuration import runtime_uses_cloud
            from oh_streamer.subtitles.combos import is_cloud_enabled
            if not is_cloud_enabled(settings):
                with self._lock:
                    for run in (getattr(self, '_subtitle_cloud', None), getattr(self, '_subtitle_local', None),
                                getattr(self, '_subtitle_stream_cloud', None)):
                        if run is None or run.stage not in ('subtitle_cloud', 'subtitle_local'):
                            continue
                        item = self.store.get(run.item_id)
                        if item and runtime_uses_cloud(item.settings):
                            self._request_stop(run, 'cancel')
                            if run.stage in ('subtitle_cloud', 'subtitle_local'):
                                self._subtitle_stop_executors(run.item_id, 'cancel')
        self._space_retry_at = {"download": 0.0, "process": 0.0}
        shared = {k: allowed[k] for k in ("detection_model", "restoration_model", "max_clip_length") if k in allowed}
        if shared:                      # the settings dialog is the one place for these: the player follows
            err = self._write_player_settings(shared)
            if err:
                log(f"could not mirror {sorted(shared)} to the player settings: {err}")
        return {"ok": True, "settings": settings}

    def _write_player_settings(self, player: dict) -> str:
        """Merge `player` into the saved player settings and tell a running player; "" or the error text."""
        from oh_streamer.player import player_settings as PS
        path = self.perf._player_file()
        try:
            saved, _warn = PS.load_file(path)
            PS.save_file(path, PS.updated_file(saved, player))
        except (OSError, ValueError) as e:
            return str(e)
        self.reload_player_settings()
        return ""

    def apply_model_settings(self, item_id: str) -> dict:
        models = self._snapshot_settings()
        options = self._model_options()
        if any(models.get(k) not in {o["value"] for o in options[k]}
               for k in ("detection_model", "restoration_model")):
            return {"ok": False, "error": "目前模型未安裝，請先選擇可用的模型"}
        with self._lock:
            item = self.store.get(item_id)
            if item is None:
                return {"ok": False, "error": "unknown id"}
            if item.state in ("processing", "done") or (self._wk and self._wk.item_id == item_id):
                return {"ok": False, "error": "處理中或已完成的項目無法變更模型"}
            settings = dict(item.settings)
            settings.update({k: models[k] for k in ("detection_model", "restoration_model")})
            new_key = queue_store.cache_key(item.target, item.quality, settings)
            self.store.update(item_id, settings=settings, cache_key=new_key, progress={})
            if new_key != item.cache_key:     # the old processing cache is void: no processed ranges / restored frames
                try:
                    (self.store.item_dir(item_id) / "work" / "segments.json").unlink()
                except OSError:
                    pass
                self.thumbs.drop_restored(item_id)
        self._wake.set()
        return {"ok": True}

    def open_folder(self, item_id: Optional[str] = None) -> dict:
        path = self.store.item_dir(item_id) if item_id else self.store.root
        if item_id and self.store.get(item_id) is None:
            return {"ok": False, "error": "unknown id"}
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._startfile(str(path))
        except OSError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def mark_all_seen(self) -> dict:
        """UI: clear the unwatched-finished sparkle for every item."""
        return {"ok": True, "count": self.store.mark_all_seen()}

    def cancel_play_after_subtitles(self, item_id: str) -> dict:
        with self._lock:
            if self.store.get(item_id) is None:
                return {"ok": False, "error": "unknown id"}
            self.store.update(item_id, play_after_subtitles=False, play_after_subtitles_start=None)
        self._wake.set()
        return {"ok": True}

    def _maybe_play_after_subtitles(self, item_id=None):
        if self._paused_all or self._stopping.is_set():
            return
        for item in self.store.items():
            if item_id is not None and item.id != item_id:
                continue
            view = item.subtitle.get("view") if isinstance(item.subtitle.get("view"), dict) else {}
            has_done = bool((view.get("slices_done") or 0) > 0 or (view.get("covered_s") or 0) > 0)
            stuck = view.get("state") == "stuck"
            if item.state == "paused" or not item.play_after_subtitles or (item.subtitle.get("status") not in ("partial", "ready", "done", "failed", "mux_pending") and not has_done and not stuck):
                continue
            with self._lock:
                current = self.store.get(item.id)
                if not current or not current.play_after_subtitles:
                    continue
                self.store.update(item.id, play_after_subtitles=False, play_after_subtitles_start=None)
            result = self._play(item.id, item.play_after_subtitles_start,
                                subtitle_error=item.subtitle.get("status") == "failed" or stuck)
            if result.get("ok"):
                self.store.mark_seen(item.id)

    def play(self, item_id: str, start=None, wait_for_subtitles=False) -> dict:
        self._press.press(item_id)
        if wait_for_subtitles:
            with self._lock:
                item = self.store.get(item_id)
                if item is None:
                    return {"ok": False, "error": "unknown id"}
                view = item.subtitle.get("view") if isinstance(item.subtitle.get("view"), dict) else {}
                has_done = bool((view.get("slices_done") or 0) > 0 or (view.get("covered_s") or 0) > 0
                                or view.get("state") == "stuck")
                if item.settings.get("subtitles") and item.subtitle.get("status") in subtitle_jobs.ACTIVE - {"mux"} and not has_done:
                    fields = dict(play_after_subtitles=True, play_after_subtitles_start=queue_ipc.parse_start(start))
                    if item.state == "paused":
                        fields["state"] = "queued" if self._source_ready(item) else "waiting"
                    self.store.update(item_id, **fields)
                    self._wake.set()
                    return {"ok": True, "waiting_subtitles": True}
        self._skip_subtitles_for_play(item_id)
        r = self._play(item_id, start)
        if r.get("ok"):
            self.store.mark_seen(item_id)
        return r

    def _play(self, item_id: str, start=None, subtitle_error=False) -> dict:
        """Play an item; `start` = seconds to begin at (invalid -> ignored, clamped to the known duration), None = as
        before. A progressive start in an unprocessed part acts like a user seek there (the player asks for it)."""
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        start = queue_ipc.parse_start(start)
        if start is not None:
            try:
                facts = self._source_facts(it)
            except Exception:  # noqa: BLE001 - the duration is only a clamp
                facts = None
            start = queue_ipc.parse_start(start, (facts or {}).get("duration"))
        with self._lock:
            previous_pending = self._pending_play
            self._pending_play = None
            self._pending_start = None
            if previous_pending and previous_pending != item_id:
                self._watch.pop(previous_pending, None)
        if it.state == "done":
            if not os.path.isfile(it.output_path):
                return {"ok": False, "error": "尚未處理完成"}
            payload = {"input": it.output_path, "headers": {}, "processed": True, "queue_id": it.id, "title": it.title}
            self._add_combos_dir(payload, it)
            self._add_external_subtitles(payload, it)
            if start is not None:
                payload["start"] = start
        else:
            work = self.store.item_dir(item_id) / "work"
            has_src = self._source_ready(it)
            if has_src and (it.state in ("queued", "processing", "paused")
                            or (it.state == "failed" and (work / "segments.json").is_file())):
                # source downloaded: the player shows the processed segments as they appear; pressing 播放 on a
                # paused item means "work on it" (pause_all stays the user's call)
                if it.state == "paused":
                    self.store.update(item_id, state="queued", error="")
                payload = self._source_payload(it)
                if start is not None:
                    payload["start"] = start
            elif it.state in ("waiting", "downloading", "queued", "processing", "paused"):
                # Wait for the queue-owned cache/proxy; never open the original target as a second live session.
                with self._lock:
                    self._pending_play = item_id
                    self._pending_start = start
                    self._player_item = item_id
                    self._watch[item_id] = self._clock()
                    if start is not None:
                        self._subtitle_clocks[item_id] = (start, False)
                        self._subtitle_wants[item_id] = (start, self._clock())
                        self._subtitle_seek_guard[item_id] = self._subtitle_wants[item_id]
                if it.state == "paused":
                    self.store.update(item_id, state="waiting", error="",
                                      source_status="paused" if it.source_status else "")
                self.store.mark_used(item_id)
                self._changed()
                return {"ok": True, "pending": True}
            else:
                return {"ok": False, "error": "尚未處理完成"}
        if subtitle_error:
            payload["subtitle_error"] = True
        self.store.mark_used(item_id)
        with self._lock:
            self._player_item = item_id
            self._watch[item_id] = self._clock()
            if start is not None:
                self._subtitle_clocks[item_id] = (start, False)
                self._subtitle_wants[item_id] = (start, self._clock())
                self._subtitle_seek_guard[item_id] = self._subtitle_wants[item_id]
                self._write_play_clock(item_id, start, False)
        reply = self._send_to_player(payload)
        if reply is None:
            try:
                self._launch_player(payload["input"], payload)
            except OSError as e:
                return {"ok": False, "error": str(e)}
            return {"ok": True, "launched": True}
        if reply.get("ok") is False or reply.get("error"):
            return {"ok": False, "error": reply.get("error") or "播放器開啟失敗"}
        return {"ok": True, "launched": False}

    def export(self, item_id: str, dest_path: str, include_compare: bool = False) -> dict:
        """Blocking: remux the finished mkv to an mp4 at dest_path (call it off the UI thread). By default only the
        processed video + audio; include_compare also keeps the compare video tracks."""
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        if it.state != "done" or not os.path.isfile(it.output_path):
            return {"ok": False, "error": "尚未處理完成"}
        dest = Path(dest_path)
        if dest.suffix.lower() != ".mp4":
            dest = dest.with_name(dest.name + ".mp4")
        tmp = dest.with_name(dest.stem + ".part.mp4")
        streams = self._probe(it.output_path)
        titles = None
        if include_compare:
            info = self._probe_tracks(it.output_path)
            titles = [s.get("title") or "" for s in (info or {}).get("streams", []) if s.get("codec_type") == "video"]
        attempts = [False] if streams is not None else [False, True]
        err = ""
        for force_aac in attempts:
            cmd = build_export_cmd(it.output_path, str(tmp), streams, force_aac, include_compare, titles)
            try:
                p = self._run_cmd(cmd)
            except OSError as e:
                return {"ok": False, "error": str(e)}
            if p.returncode == 0:
                os.replace(tmp, dest)
                return {"ok": True, "path": str(dest)}
            err = (p.stderr or "").strip()[-500:]
        try:
            tmp.unlink()
        except OSError:
            pass
        return {"ok": False, "error": err or "ffmpeg failed"}

    def remove_compare(self, item_id: str) -> dict:
        """Blocking (call it off the UI thread): remux the finished output keeping only the processed video track,
        all audio / subtitles / chapters and the global tags (OH_COMPARE=0, OH_ORIGINAL=none), verify the duration,
        swap the file in. Returns {"ok": True, "freed_bytes": n} or {"ok": False, "error": ...}."""
        it = self.store.get(item_id)
        if it is None:
            return {"ok": False, "error": "unknown id"}
        if it.state != "done" or not os.path.isfile(it.output_path):
            return {"ok": False, "error": "尚未處理完成"}
        with self._lock:
            if self._running_runs(item_id) or (
                it.subtitle.get("only")
                and it.subtitle.get("status") in subtitle_jobs.ACTIVE | {"ready"}
            ):
                return {"ok": False, "error": "字幕處理中，請完成或取消後再移除比對軌"}
            if item_id in self._compare_busy or item_id in self._reprocessing:
                return {"ok": False, "error": "正在移除比對軌"}
            self._compare_busy.add(item_id)
        tmp = Path(it.output_path).with_name("output.nocompare.tmp.mkv")
        hold = contextlib.ExitStack()                 # film-strip grabs read the file: pause them while it is replaced
        if hasattr(self.thumbs, "hold"):
            hold.enter_context(self.thumbs.hold(item_id))
        try:
            if item_id in self._watched_ids():
                return {"ok": False, "error": PLAYING_ERROR}
            info = self._probe_tracks(it.output_path)
            if not has_compare_tracks(info):
                return {"ok": False, "error": "此影片沒有比對軌"}
            videos = [s for s in info["streams"] if s.get("codec_type") == "video"]
            pos = next((i for i, s in enumerate(videos) if s.get("title") == PROCESSED_TITLE), 0)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", it.output_path,
                   "-map", f"0:v:{pos}", "-map", "0:a?", "-map", "0:s?", "-map_chapters", "0", "-map_metadata", "0",
                   "-c", "copy", "-copyts", "-disposition:v:0", "default",
                   "-metadata", "OH_COMPARE=0", "-metadata", "OH_ORIGINAL=none", "-f", "matroska", str(tmp)]
            try:
                p = self._run_cmd(cmd)
            except OSError as e:
                return {"ok": False, "error": str(e)}
            if p.returncode != 0 or not tmp.is_file():
                return {"ok": False, "error": "移除失敗：" + ((p.stderr or "").strip()[-300:] or "ffmpeg failed")}
            new_info = self._probe_tracks(str(tmp))
            if new_info is None or abs(new_info.get("duration", 0.0) - info.get("duration", 0.0)) > DURATION_TOLERANCE:
                return {"ok": False, "error": "移除失敗：新檔案長度與原檔不符，已保留原檔"}
            old_size = os.path.getsize(it.output_path)
            try:
                os.replace(tmp, it.output_path)
            except OSError:
                return {"ok": False, "error": PLAYING_ERROR}
            new_size = os.path.getsize(it.output_path)
            settings = dict(it.settings)
            settings["compare"] = "off"
            settings.pop("compare_original", None)
            settings.pop("compare_quality", None)
            with self._lock:
                self._track_cache.pop(it.output_path, None)
            self._usage_at = 0.0                      # usage_bytes is re-measured on the next state push
            self.store.update(item_id, size_bytes=new_size, settings=settings)
            return {"ok": True, "freed_bytes": max(0, old_size - new_size)}
        finally:
            hold.close()
            with self._lock:
                self._compare_busy.discard(item_id)
            try:
                tmp.unlink()
            except OSError:
                pass


class _Cancelled(Exception):
    pass


def _inside(path: str, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except (ValueError, OSError):
        return False
