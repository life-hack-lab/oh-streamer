"""Player state, event routing, source lifecycle, and ordered shutdown.

All operations supplied by the application facade are injected explicitly. Importing this module does not load
Lada or torch, start a player, or change process globals. Listener and queue callbacks only enqueue work; run()
owns mpv IPC and consumes those queues in the same order as the original player loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from oh_streamer.branding import APP_NAME
from oh_streamer.engines.base import ModelChoice
from oh_streamer.player import perf_hint, subtitle_compare, subtitle_status
from oh_streamer.player.restore_pause import RestorePause
from oh_streamer.player.stream_subtitle_link import StreamSubtitleLink, StreamSubtitles

@dataclass(frozen=True)
class PlayerDependencies:
    """Facade operations and shared modules used by a player instance.

    Callable fields are caller-owned adapters: the compatibility facade resolves its current functions when the
    operation executes, preserving monkeypatches without importing or looking up the facade in this module.
    """

    BusyReporter: Callable[..., Any]
    CastManager: Callable[..., Any]
    Fraction: Any
    IDLE_HINT: Any
    MpvEndpoint: Callable[..., Any]
    MpvError: Any
    MpvIpc: Callable[..., Any]
    Opener: Callable[..., Any]
    PS: Any
    Path: Any
    ProcessedOpen: Any
    QualityControl: Callable[..., Any]
    QueueLink: Callable[..., Any]
    Session: Callable[..., Any]
    ThumbnailWorker: Callable[..., Any]
    _float_or: Callable[..., Any]
    airplay_cast: Any
    buffer_progress: Any
    build_add_payload: Callable[..., Any]
    collections: Any
    compare_stack: Any
    device_backend: Any
    find_mpv: Callable[..., Any]
    forget_sources: Callable[..., Any]
    gc: Any
    info_panel: Any
    json: Any
    log: Callable[..., Any]
    mpv_playback_snapshot: Callable[..., Any]
    open_input: Any
    os: Any
    passthrough_frames: Callable[..., Any]
    platform_support: Any
    processed_from_message: Callable[..., Any]
    processed_watched: Callable[..., Any]
    subtitle_update_from_message: Callable[..., Any]
    progressive_stream: Any
    qualities: Any
    queue: Any
    queue_lookup: Callable[..., Any]
    queue_quality: Callable[..., Any]
    register_source: Callable[..., Any]
    req_counter: Callable[..., Any]
    seek_tail: Any
    short_error: Callable[..., Any]
    shutil: Any
    start_mpv: Callable[..., Any]
    tempfile: Any
    threading: Any
    time: Any

class PlayerController:
    """Own one player's source, request, UI, and resource state.

    Construction is side-effect free. run() starts mpv and performs the existing startup and teardown sequence;
    helpers share this instance's state instead of retaining a large closure of nonlocal variables.
    """

    def __init__(self, args, engine, first=None, quality_spec=None, instance=None, initial_inputs=(), *,
                 deps: PlayerDependencies):
        self._deps = deps
        self.args = args
        self.engine = engine
        self.first = first
        self.quality_spec = quality_spec
        self.instance = instance
        self.initial_inputs = initial_inputs
        self.progressive_active_url = None
        self.progressive_source = False
        self.progressive_generation = 0
        self.progressive_handoff = None
        self.progressive_retry_at = 0.0
        self.progressive_candidate_at = None
        self.progressive_clip = None
        self.stream_item = None      # id of the queue's subtitle-only stream item of the live source (open message)
        self.stream_link = None      # StreamSubtitleLink reporting to it (live mode with a queue only)
        self.stream_subs = None      # StreamSubtitles: the item's growing external subtitle tracks (T3.2), live mode

    def run(self) -> int:
        """Start mpv, consume player events, and release resources in lifecycle order."""
        self.endpoint = self._deps.MpvEndpoint()
        self.ipc_path = self.endpoint.path
        self.title = self.first.title if self.first is not None else APP_NAME
        try:
            self.player = self._deps.start_mpv(
                self._deps.find_mpv(self.args.mpv), self.ipc_path, self.title, self.args.initial_prebuffer,
                idle_hint=self.first is None, vsr=self.engine.cfg["vsr"], smooth=self.engine.cfg["smooth_playback"],
                compare=self.engine.cfg.get("compare", "off"))
        except Exception:
            self.endpoint.close()
            raise
        self._deps.log(f"mpv started (pid {self.player.pid})")
        self.t_req = self._deps.time.perf_counter()
        self.ipc = None
        self.session: Any = None
        self.qc: Any = None
        self.sc: Any = self.engine.sc
        self.engine_idle_since: float | None = None   # when the loaded models were last in use (release_idle_engine)
        self.thumbs, self.thumb_dir = None, None
        self.combo_dir, self.combos_published = None, False   # subtitle combos (subtitle_compare.py): temp dir of the composite ASS
        self.progressive_sub_dir = None
        self.progressive_sub_loaded = None
        self.progressive_sub_selection = None
        self.progressive_sub_visibility = None
        self.source_text = ""
        self.gpu = self._deps.info_panel.GpuMonitor()
        self.gpu.start()                            # NVML polls in the background: the panel's charts have history at once
        self.hist = self._deps.info_panel.History()            # per-second samples for the panel's trend charts
        self.opener = self._deps.Opener(self.args, self.quality_spec)
        self.inbox = self._deps.queue.Queue()                  # (text, headers, label, start, quality, overrides, stream_item): forwarded inputs (listener thread)
        self.pinbox = self._deps.queue.Queue()                 # ProcessedOpen: processed files to play as is (listener thread / --processed)
        self.subtitlebox = self._deps.queue.Queue()           # validated subtitle_update snapshots (listener thread)
        self.pending_subtitle_updates = {}                     # updates may arrive just before their queue open is consumed
        self.subtitle_status_box = self._deps.queue.Queue()    # validated subtitle_status snapshots (listener thread, T3.3)
        self.pending_subtitle_status = {}                      # ... that arrived before their open was consumed
        self.subtitle_status = None                            # subtitle_status.SubtitleStatus of the playing queue item
        self.subtitle_problem_shown = None                     # problem_key() of the failure already shown as an OSD
        self.gpu_short_shown = None                            # source epoch the gpu_short OSD was shown for (T4.3)
        self.subtitle_gap_item_id = None
        self.subtitle_gap_warned = False
        self.notices = self._deps.queue.Queue()                # OSD texts for replies of the queue app (sender thread -> controller)
        self.reload_box = self._deps.queue.Queue()             # settings_reload messages (listener thread): re-read the settings file
        self.reload_requested = False                          # a settings_reload waits for the models (self.sc) to exist
        self.reload_pending: dict = {}                         # file changes not applied yet: key -> value (see apply_reloaded_settings)
        self.reload_restart_at = -10.0                         # when oh-restart was last sent for them
        # the background queue app (queue_ipc.py). Only a real player talks to it (not a test driving run_player directly)
        self.link = self._deps.QueueLink(name=getattr(self.args, "instance", None), log=self._deps.log) \
            if self.instance is not None or self.first is not None else None
        self.opener.on_unused_stream_item = self.release_stream_item   # (a stream item no source will use)
        self.restore_pause = RestorePause(             # queue_ipc.py: restore_pause / restore_resume / restore_needed (T4.2)
            lambda payload: self.link.post(payload) if self.link is not None else None, self._deps.log)
        self.busy_rep = self._deps.BusyReporter(
            lambda b, **extra: self.link.post({"cmd": "player_busy", "busy": b, "pid": self._deps.os.getpid(), **extra})) \
            if self.link is not None else None
        if self.busy_rep is not None:
            self.busy_rep.live = self.live_report      # the queue's 「串流中」 card (queue_ipc.py, "Live stream report")
        self.play_rep = self._deps.BusyReporter(       # mpv plays anything: keeps the queue's calibration clean
            lambda p: self.link.post({"cmd": "player_playing", "playing": p, "pid": self._deps.os.getpid()})) \
            if self.link is not None else None
        self.processed: Any = None   # the processed file being played (no Session then)
        self.progressive: Any = None   # progressive mode (no Session either)
        self.ranges_shown, self.progressive_sent, self.progressive_next = None, False, 0.0   # ranges published, "watched" sent, next check
        self.processed_sent = False                 # "watched" was sent for it
        self.processed_next = 0.0
        self.processed_clock_next = 0.0
        self.cur_open = None                        # (input, user headers) of the live source: what "add to queue" sends
        for text, headers, start in self.initial_inputs:
            if getattr(self.args, "processed", False):
                po = self._deps.processed_from_message({"input": text, "processed": True, "queue_id": getattr(self.args, "queue_id", None),
                                                        "start": start if start else None})
                if po is not None:
                    self.pinbox.put(po)
                else:
                    self._deps.log(f"--processed: file not found: {text}")
            else:
                self.inbox.put((text, headers, "命令列", start, None, {}, None))
        if self.link is not None:
            self.opener.lookup = self.lookup_queue
        if self.instance is not None:
            self.instance.serve(self.on_message, self._deps.log)
        self.thumb_sent, thumb_seq, self.prefetch_started = None, 0, False
        self.hint_tracker = perf_hint.HintTracker()   # one OSD performance hint per opened source (perf_hint.py)
        self.epoch = 0                              # sources opened so far (also tags "served" of each initial segment)
        self.cast: Any = None        # set once the IPC is up
        self._cast_stream_restarts = 0
        self._cast_stream_want_k = None
        self.tail: Any = None  # the old timeline mpv keeps playing while a far seek is prepared
        auto_cast = ("airplay", self.args.cast) if getattr(self.args, "cast", None) else ("hls", None) \
            if getattr(self.args, "hls_only", False) else None   # --cast / --hls-only: start with the first source

        try:
            if self.first is not None:
                self.begin_source(self.first)
            self.ipc = self._deps.MpvIpc(self.ipc_path, self.player)
            self._deps.log(f"mpv IPC: {self.ipc_path}")
            self.ipc.command("disable_event", "all")
            self.cast = self._deps.CastManager(self.args, self.ipc)
            self.cast.subtitle_fallback = self.cast_subtitle_fallback
            self.cast.vsr_enabled = self.engine.saved.get("cast_vsr", "off") == "on"
            self.cast.vsr_size = self.engine.saved.get("cast_vsr_size", "2x")
            def save_cast_settings(changes):
                self.engine.saved.update(changes)
                if self.sc:
                    self.sc.commit(changes)
                else:
                    try:
                        self._deps.PS.save_file(self.args.settings_file, self.engine.saved)
                    except (OSError, ValueError) as e:
                        self._deps.log(f"could not save cast settings: {e}")
            self.cast.vsr_commit = save_cast_settings
            self.ipc.set("user-data/oh/casting", "")
            last_castreq = None
            self.published_duration = 0.0
            self.published_list, self.published_settings, self.status_shown = None, None, None
            self.published_opts = {"smooth_playback": self.engine.cfg["smooth_playback"]}   # start_mpv() set these properties
            self.published_derive = ""                  # user-data/oh/derive-1x as last published (publish_compare_layers)

            self.publish_player_opts()
            if self.first is not None:
                self.publish_source()
            else:
                self.ipc.set("user-data/oh/active", "0")
            ui = self._deps.PS.ui_state(self.engine.saved)          # last volume / mute (saved whenever they change, debounced)
            self.ipc.set("volume", ui["volume"])
            self.ipc.set("mute", ui["mute"])
            vol_saved, vol_seen, vol_changed_at, vol_next_check = (ui["volume"], ui["mute"]), None, None, 0.0
            # Pause handling lives in oh_seek.lua: it freezes playback when it publishes a request, restores the
            # user's pause state when the new segment is loaded, and publishes user-data/oh/resumed when it plays.
            # Requests (Lua numbers all three from one counter): seek, quality, settings (restart / model reload).
            self.last_sreq, self.last_qreq, self.last_creq, last_lset, last_qadd = None, None, None, None, None
            self.served, self.loaded, self.resume_wait = f"init{self.epoch}", False, False
            self.mpv_wait = self.args.initial_prebuffer      # cache-pause-wait mpv currently has
            self.restore_deadline = None                # when to put cache-pause-wait back to --prebuffer
            self.n_req, self.stop_s, self.t_stopped, self.t_load = 0, 0.0, self.t_req, None
            self.sw_note = ""                           # extra timing info of a quality switch / settings change, for the log
            self.switch_ctx = None                      # set while a freshly switched rendition has not produced output yet
            self.req_kind = "啟動"                      # what the request in flight was (info panel: last latency)
            self.last_latency, self.last_kind = None, None
            self.loading_models = False
            self.info_visible, info_next_check, info_next_publish = False, 0.0, 0.0
            self.proc_samples = self._deps.collections.deque(maxlen=8)   # (time, frames_total) for the processing-fps estimate
            self.proc_fps_now, hist_next = None, 0.0    # latest estimate from proc_samples (sample_history, once a second)
            self.prog_shown, self.prog_key, self.prog_pct, prog_next = "", None, None, 0.0   # buffering progress (publish_progress)
            clock_next = 0.0                       # next playback-position sample for interpolation (session.clock)
            play_next = 0.0                        # next player_playing check (play_rep)
            self.pending_open: Any = None     # resolved, waiting for the models
            self.reset_seq = 0
            # seek tail (seek_tail.py): how the segment was handed to mpv ("append" behind the tail / "replace"; None =
            # not yet), the tail's summary for the timing log, the next mpv sample for its pacing (and the append watchdog)
            self.tail_switch, self.tail_note, tail_next = None, "", 0.0
            self.switch_ahead = lambda: self.args.seek_prebuffer + 0.5   # noqa: E731 - output past the target before leaving a tail

            combos_next = 0.0                      # next manifest re-check of a growing subtitle combos set (M7)
            while self.player.poll() is None:
                now = self._deps.time.perf_counter()
                if now >= combos_next:
                    combos_next = self.poll_sub_combos(now)
                if self.sc is None and self.engine.sc is not None:
                    self.sc = self.engine.sc
                    if self.session is not None:
                        self.publish_settings()
                self.release_idle_engine(now)
                if self.session is not None and not self.loaded and self.session.ready() and (via := self.switch_via()) is not None:
                    had_tail = self.tail is not None
                    want = self.args.initial_prebuffer if self.n_req == 0 else self.args.seek_prebuffer
                    self.set_cache_wait(want)                    # small startup buffer: Lada is faster than realtime
                    self.ipc.set("user-data/oh/served", self.served)  # lets oh_seek.lua drop its pending target
                    self.ipc.set("user-data/oh/expect-load", self.session.url)   # oh_seek.lua: our load, not a drop
                    self.publish_compare_layers()
                    self.stream_subtitles_before_load()          # keep the chosen track; mpv drops external tracks with the file
                    if via == "append":
                        # seamless: mpv plays the rest of the tail (about its lead), then continues with this entry
                        self.ipc.command("loadfile", self.session.url, "append")
                        self.tail.finish("switched")
                        self.tail_note = ""
                    else:
                        self.ipc.command("loadfile", self.session.url, "replace")
                        self.tail_note = self.drop_tail("replaced by the new segment")
                    self.tail_switch = via if had_tail else None
                    self.loaded, self.resume_wait, self.t_load = True, True, now
                    out_t = self.session.lada_out_t()
                    ahead = "" if out_t is None else f", Lada output {out_t - self.session.target:.1f}s past the target"
                    self._deps.log(f"{'append' if via == 'append' else 'loadfile'} {self.session.url} ({now - self.t_req:.2f}s after request, "
                        f"cache-pause-wait {self.mpv_wait:g}s{ahead})")
                elif self.session is not None and not self.loaded and self.switch_ctx is not None and self.session.no_output:
                    self.revert_switch(self.switch_ctx)
                    continue
                elif self.resume_wait and self.session is not None and self.ipc.get("user-data/oh/resumed") == self.served:
                    self.resume_wait = False
                    self.switch_ctx = None
                    self.show_perf_hint()
                    pos = self.ipc.get("time-pos")
                    t_first = self.session.stats.t_first or self.t_load
                    self.last_latency, self.last_kind = now - self.t_req, self.req_kind
                    if self.tail is not None:          # appended behind it: mpv has moved on, the old stream is done
                        self.tail_note = self.tail.summary()
                        t_old, self.tail = self.tail, None
                        t_old.close()             # (normally just reaps the muxer, which has exited by now)
                        self.ipc.set("user-data/oh/tail", "")
                    self._deps.log(f"playback resumed at {pos if pos is None else round(pos, 3)}s (requested {self.session.target:.3f}s) "
                        f"{now - self.t_req:.2f}s after request")
                    sw = f"switch ({self.tail_switch})" if self.tail_switch else "loadfile"
                    self._deps.log(f"timing {'request #' + self.served.split(':', 1)[0] if self.n_req else 'initial'}: "
                        f"{self.sw_note}stop old {self.stop_s:.2f}s | "
                        f"lada first frame +{t_first - self.t_stopped:.2f}s | to {sw} +{max(0.0, self.t_load - t_first):.2f}s "
                        f"({self.t_load - self.t_req:.2f}s after request) | {sw}->playing {now - self.t_load:.2f}s | total {now - self.t_req:.2f}s"
                        + (f" | {self.tail_note}" if self.tail_note else ""))
                    self.tail_switch, self.tail_note = None, ""
                    if self.mpv_wait != self.args.prebuffer:
                        self.restore_deadline = now + 10.0
                    if self.thumbs is not None and not self.prefetch_started:
                        self.prefetch_started = True
                        self.thumbs.start_prefetch(self.args.thumb_interval)   # first segment is playing: use the idle CPU

                # interpolation's lead signal: mpv's position for the pump thread (MpvIpc is used from this thread only).
                # Only while the segment plays: before that the clock is empty and the pump copies frames.
                if self.session is not None and self.session.doubler is not None and self.loaded and not self.resume_wait \
                        and self.cast.active is None and now >= clock_next:
                    clock_next = now + 0.25
                    pos = self.ipc.get("time-pos")
                    if isinstance(pos, (int, float)):
                        ct = self.ipc.get("demuxer-cache-time")
                        self.session.clock.update(pos, not self.ipc.get("core-idle", True), self._deps.time.perf_counter(),
                                             ct if isinstance(ct, (int, float)) else None)

                if self.restore_deadline is not None and not self.resume_wait:
                    ahead = (self.ipc.get("demuxer-cache-time") or 0) - (self.ipc.get("time-pos") or 0)
                    if ahead >= self.args.prebuffer or now >= self.restore_deadline:
                        self.ipc.set("cache-pause-wait", self.args.prebuffer)
                        self.mpv_wait, self.restore_deadline = self.args.prebuffer, None
                        self._deps.log(f"cache-pause-wait restored to {self.args.prebuffer:g}s ({ahead:.1f}s cached ahead)")

                # hover preview: oh_seek.lua publishes the hovered time; answer with the best bitmap we have right now
                # and again once the exact one is decoded
                hovering = False
                self.processed_thumbnails_tick()
                if self.thumbs is not None:
                    treq = self.ipc.get("user-data/oh/thumb-request")
                    if treq:
                        hovering = True
                        try:
                            found = self.thumbs.query(float(treq))
                        except ValueError:
                            found = None
                        if found is not None and (treq, found) != self.thumb_sent:
                            path, tw, th, exact = found
                            thumb_seq += 1
                            self.ipc.set("user-data/oh/thumb-ready", f"{treq}|{path}|{tw}|{th}|{int(exact)}|{thumb_seq}")
                            self.thumb_sent = (treq, found)
                    else:
                        self.thumb_sent = None

                # info panel (進階資訊): Lua says whether it is visible; publish about once per second while it is
                if now >= info_next_check:
                    info_next_check = now + 0.25
                    vis = bool(self.ipc.get("user-data/oh/info-visible"))
                    if vis != self.info_visible:
                        self.info_visible = vis
                        self.gpu.set_active(vis)
                        info_next_publish = now
                    if self.info_visible and now >= info_next_publish:
                        info_next_publish = now + 1.0
                        self.publish_info()

                if now >= prog_next:
                    prog_next = now + 0.2
                    self.publish_progress(now)
                if now >= hist_next:
                    hist_next = max(hist_next + 1.0, now - 0.5)
                    self.sample_history(now)

                if now >= vol_next_check:
                    vol_next_check = now + 0.5
                    cur = (self._deps._float_or(self.ipc.get("volume"), vol_saved[0]), bool(self.ipc.get("mute")))
                    if cur != vol_seen:
                        vol_seen, vol_changed_at = cur, now
                    elif vol_changed_at is not None and now - vol_changed_at >= 1.0 and cur != vol_saved and self.sc:
                        vol_saved, vol_changed_at = cur, None
                        self.sc.commit({"volume": max(0.0, min(100.0, cur[0])), "mute": cur[1]})

                # opening sources: inputs from the main app (single-instance IPC) / --player, finished resolutions
                while True:
                    try:
                        text, hdrs, label, start, quality, overrides, stream_item = self.inbox.get_nowait()
                    except self._deps.queue.Empty:
                        break
                    self.submit_text(text, hdrs, label, start, quality, overrides, stream_item)
                while True:                        # processed files from the queue app (listener thread) / --processed
                    try:
                        po = self.pinbox.get_nowait()
                    except self._deps.queue.Empty:
                        break
                    if isinstance(po, self._deps.progressive_stream.ProgressiveOpen):
                        self._deps.log(f"open (佇列): progressive {po.path} (work {po.work_dir})")
                        self.play_progressive(po)
                        continue
                    self._deps.log(f"open (佇列): processed file {po.path}")
                    self.play_processed(po, "佇列")
                while True:                        # replies of the queue app (sender thread)
                    try:
                        self.osd(self.notices.get_nowait())
                    except self._deps.queue.Empty:
                        break
                while True:                        # dynamic subtitle snapshots from the queue app
                    try:
                        update = self.subtitlebox.get_nowait()
                    except self._deps.queue.Empty:
                        break
                    self.apply_subtitle_update(update)
                while True:                        # subtitle job status of the playing item (panel + OSD, T3.3)
                    try:
                        status = self.subtitle_status_box.get_nowait()
                    except self._deps.queue.Empty:
                        break
                    self.apply_subtitle_status(status)
                while True:                        # settings_reload (listener thread): the file was rewritten by the queue app
                    try:
                        self.reload_box.get_nowait()
                    except self._deps.queue.Empty:
                        break
                    self.reload_requested = True
                if self.reload_requested and self.sc is not None:
                    self.reload_requested = False
                    self.settings_reload()
                if self.reload_pending:
                    self.apply_reloaded_settings(now)
                qadd = self.ipc.get("user-data/oh/queue-add")
                if qadd and qadd != last_qadd:
                    last_qadd = qadd
                    self.add_to_queue()
                if self.restore_pause.tick(self.session, self.stream_item, self.restore_probe) and self.busy_rep is not None:
                    self.busy_rep.next_live = 0.0       # pause state changed: the live report goes out now, not at the next tick
                if self.busy_rep is not None:           # live restore running -> the queue pauses its own processing
                    th =getattr(self.session, "thread", None)
                    self.busy_rep.update((th is not None and th.is_alive()) or self.cast.active is not None)
                if self.play_rep is not None and now >= play_next:
                    play_next = now + 1.0
                    self.play_rep.update(self.ipc.get("pause") is False and not self.ipc.get("eof-reached", False)
                                         and not self.ipc.get("idle-active", False))
                if self.processed is not None and not self.processed_sent and now >= self.processed_next:
                    self.processed_next = now + 0.5
                    if self.ipc.get("path") == self.processed.path and self._deps.processed_watched(
                            self.ipc.get("time-pos"), self.ipc.get("duration"), self.ipc.get("eof-reached", False)):
                        self.processed_sent = True
                        self._deps.log(f"processed file watched (queue item {self.processed.queue_id})")
                        if self.processed.queue_id and self.link is not None:
                            self.link.post({"cmd": "watched", "id": self.processed.queue_id})
                if self.processed is not None and getattr(self.processed, "subtitles", None) \
                        and self.progressive_sub_loaded != self.processed.path \
                        and self.ipc.get("path") == self.processed.path and self.ipc.get("track-list", []):
                    self.load_progressive_subtitles(self.processed.path, self.processed)
                if self.processed is not None and self.processed.queue_id and self.link is not None \
                        and now >= getattr(self, "processed_watch_next", 0.0):
                    # Protect deferred replacement even when a finished file is paused.
                    self.processed_watch_next = now + 10.0
                    self.link.post({"cmd": "watching", "id": self.processed.queue_id})
                self.send_processed_clock(now)
                self.stream_link_tick(now)
                self.stream_subtitle_tick()
                if self.progressive is not None:
                    self.progressive_tick(now)
                self.subtitle_gap_tick()
                done = self.opener.poll()
                if done is not None:
                    opened, err, took, label = done
                    if err is not None:
                        self._deps.log(f"open ({label}) failed after {took:.2f}s: {err!r}")
                        self.show_error(self._deps.short_error(err))
                    elif isinstance(opened, self._deps.ProcessedOpen):    # the queue has it finished: no Lada, no models needed
                        self._deps.log(f"open ({label}): processed version found in the queue after {took:.2f}s")
                        self.play_processed(opened, label)
                        self.update_status()
                        continue
                    else:
                        self._deps.log(f"open ({label}): resolved in {took:.2f}s")
                        self.drop_pending_open()           # (a source still waiting for the models is replaced)
                        self.pending_open = opened
                        self.engine.ensure_loaded()        # the models load on demand: only a live source needs them
                if self.pending_open is not None and self.engine.ready.is_set():
                    opened, self.pending_open = self.pending_open, None
                    if self.engine.error is not None:
                        self.show_error("模型載入失敗，無法播放（詳見記錄檔）")
                        self.release_stream_item(getattr(opened, "stream_item", None))
                    else:
                        # the loader may have finished after the sync at the top of this iteration: Engine.load sets
                        # engine.sc before engine.ready, so it is there now
                        self.sc = self.engine.sc
                        try:
                            self.switch_to(opened)
                        except Exception as e:  # noqa: BLE001 - e.g. Lada cannot set up this source
                            self._deps.log(f"could not start {opened.title!r}: {e!r}")
                            self.show_error(self._deps.short_error(e))
                            if getattr(opened, "stream_item", None) != self.stream_item:   # (begin_source never took it over)
                                self.release_stream_item(getattr(opened, "stream_item", None))
                            self.end_source()
                            self.ipc.set("user-data/oh/active", "0")
                        self.update_status()
                        continue
                self.update_status()

                if (self.session is not None or self.processed is not None or self.progressive is not None) and self.sc is not None:
                    lset = self.ipc.get("user-data/oh/settings-set")
                    if lset and lset != last_lset:
                        last_lset = lset
                        self.apply_live(lset)

                # casting: cast button / menu ('c'), --cast / --hls-only, AirPlay events, first-segment wait
                creq_cast = self.ipc.get("user-data/oh/cast-request")
                if creq_cast and creq_cast != last_castreq:
                    last_castreq = creq_cast
                    what = creq_cast.split(":", 1)[1] if ":" in creq_cast else ""
                    if what == "hls" or what.startswith("dev:"):
                        self.drop_tail("casting")   # casting stops local playback (one producer)
                    try:
                        local_pos = self._deps._float_or(self.ipc.get("time-pos"), self.session.target) if self.session is not None and self.loaded \
                            else (self.session.target if self.session is not None else None)
                        if self.processed is not None:
                            local_pos = self._deps._float_or(self.ipc.get("time-pos"), 0.0)
                        if self.progressive is not None:
                            fallback_pos = (self.progressive_handoff["position"] if self.progressive_handoff is not None
                                            else self.progressive.target)
                            local_pos = self.cast.active["local_pos"] if self.cast.active is not None else (
                                self._deps._float_or(self.ipc.get("time-pos"), fallback_pos)
                                if self.loaded and not self.resume_wait else fallback_pos)
                        self.cast.handle_request(creq_cast, self.cast_source(what), local_pos)
                    except Exception as e:  # noqa: BLE001 - casting must never take the player down
                        self._deps.log(f"cast request {creq_cast!r} failed: {e!r}")
                        self.show_error(f"投放失敗：{self._deps.airplay_cast.short_reason(e)}")
                    self.update_status()
                if auto_cast is not None and (self.progressive is not None or self.processed is not None or (self.session is not None and self.sc is not None)):
                    mode, query = auto_cast
                    auto_cast = None
                    self.drop_tail("casting")
                    try:
                        dev_id, dev_name = self.cast.resolve_device(query) if mode == "airplay" else (None, None)
                        self.cast.start(mode, self.cast_source("hls"),
                                        self.session.target if self.session is not None else
                                        self.progressive.target if self.progressive is not None else
                                        self._deps._float_or(self.ipc.get("time-pos"), 0.0), dev_id, dev_name)
                    except Exception as e:  # source probing must not terminate the player
                        self._deps.log(f"--cast: {e}")
                        self.show_error(str(e))
                    self.update_status()
                try:
                    self.cast.poll()
                except Exception as e:  # noqa: BLE001
                    self._deps.log(f"cast: {e!r}")
                    self.cast.stop(f"投放失敗：{self._deps.airplay_cast.short_reason(e)}")

                sreq = self.ipc.get("user-data/oh/seek-request")
                qreq = self.ipc.get("user-data/oh/quality-request")
                creq = self.ipc.get("user-data/oh/settings-request")
                if self.cast.active is not None and sreq and sreq != self.last_sreq:
                    self.handle_cast_seek(sreq, now)
                if (sreq and sreq != self.last_sreq) or (qreq and qreq != self.last_qreq) or (creq and creq != self.last_creq):
                    if self.processed is not None:      # plain mpv playback: nothing to restart (see processed_requests)
                        self.processed_requests(sreq, qreq, creq)
                    elif self.progressive is not None:         # the feeder restarts at the target (see progressive_requests)
                        self.progressive_requests(sreq, qreq, creq)
                    elif self.session is None:          # nothing to seek in (oh_seek.lua does not ask while idle)
                        self.last_sreq, self.last_qreq, self.last_creq = sreq, qreq, creq
                    else:
                        self.handle_requests(sreq, qreq, creq)
                        continue  # pick up newer requests (rapid seeks / switches) before waiting

                # seek tail: mpv's position paces its passthrough (the thread never touches the IPC); once the new segment
                # is appended behind it, a tail mpv does not leave by itself (paused by the user meanwhile; keep-open at
                # its end) is left with playlist-next, so the switch is not held up
                if self.tail is not None and now >= tail_next:
                    tail_next = now + 0.2
                    path, tp = self.ipc.get("path"), self.ipc.get("time-pos")
                    if path == self.tail.url and isinstance(tp, (int, float)):
                        self.tail.observe(tp, not self.ipc.get("core-idle", True), bool(self.ipc.get("paused-for-cache")))
                        if self.tail_switch == "append" and self.loaded and self.resume_wait and \
                                (self.ipc.get("pause") or self.ipc.get("eof-reached", False)):
                            self.tail_switch = "append+next"
                            self._deps.log("the tail is paused / at its end with the new segment appended: playlist-next")
                            self.ipc.command("playlist-next")

                self._deps.time.sleep(0.02 if hovering else 0.03 if (not self.loaded or self.resume_wait or self.restore_deadline is not None
                                                          or self.opener.busy) else 0.05)
            self._deps.log("player exited")
        except (EOFError, OSError):
            self._deps.log("player exited (IPC closed)")
        except KeyboardInterrupt:
            self._deps.log("interrupted")
        finally:
            if self.cast is not None:
                try:
                    self.cast.close()                   # HLS server + producer + cache dir, AirPlay connection
                except Exception as e:  # noqa: BLE001
                    self._deps.log(f"cast: closing failed: {e!r}")
            self.end_source()
            if self.busy_rep is not None:
                self.busy_rep.next_live = 0.0           # (the report is gone by now: a last live=null clears the card)
                self.busy_rep.update(False)             # (only sends if busy=true went out)
            if self.play_rep is not None:
                self.play_rep.update(False)
            if self.link is not None:
                self.link.close()                       # lets that last message go (bounded wait)
            self.gpu.close()
            if self.thumb_dir:
                self._deps.shutil.rmtree(self.thumb_dir, ignore_errors=True)
            if self.combo_dir:
                self._deps.shutil.rmtree(self.combo_dir, ignore_errors=True)
            if self.instance is not None:
                self.instance.close()
            if self.player.poll() is None:
                try:
                    self.ipc.command("quit")
                    self.player.wait(timeout=5)
                except Exception:
                    self.player.kill()
                    self.player.wait(timeout=5)
            if self.ipc is not None:
                self.ipc.close()
            self.endpoint.close()
        return 0

    def lookup_queue(self, oi, quality_spec=None):   # Opener thread: a finished queue item for this input plays as is
        return self._deps.queue_lookup(self.link, oi.target if oi.kind == "url" else self._deps.os.path.abspath(oi.target),
                                       self.quality_spec if quality_spec is None else quality_spec)

    def on_message(self, msg):               # listener thread: never touches the mpv IPC, only queues
        if msg.get("cmd") == "window_ready":
            return {"ok": True, "ready": self.ipc is not None}
        if msg.get("cmd") == "settings_reload":      # the queue app rewrote the settings file: the loop re-reads it
            self.reload_box.put(True)
            return {"ok": True}
        if msg.get("cmd") == "subtitle_status":
            try:
                status = subtitle_status.status_from_message(msg)
            except Exception as e:  # noqa: BLE001 - a malformed optional status cannot affect playback
                self._deps.log(f"subtitle status validation skipped: {e!r}")
                return {"ok": False, "error": "invalid subtitle status"}
            if status is None:
                return {"ok": False, "error": "invalid subtitle status"}
            status.received = self._deps.time.perf_counter()
            self.subtitle_status_box.put(status)
            return {"ok": True}
        if msg.get("cmd") in ("restore_pause", "restore_resume"):   # the loop applies it (RestorePause.tick)
            if getattr(self, "restore_pause", None) is None:
                return {"ok": False, "error": "not this stream"}
            return self.restore_pause.submit(msg, self.stream_item, self.session is not None)
        if msg.get("cmd") == "subtitle_update":
            try:
                update =self._deps.subtitle_update_from_message(msg)
            except Exception as e:  # noqa: BLE001 - malformed optional updates cannot affect playback
                self._deps.log(f"subtitle update validation skipped: {e!r}")
                return {"ok": False, "error": "invalid subtitle update"}
            if update is None:
                return {"ok": False, "error": "invalid subtitle update"}
            try:
                self.subtitlebox.put_nowait(update)
            except self._deps.queue.Full:
                self._deps.log(f"subtitle update dropped: controller queue is full ({update.item_id})")
                return {"ok": False, "error": "player busy"}
            return {"ok": True}
        if msg.get("progressive") is True and isinstance(msg.get("input"), str):   # queue item being processed
            po = self._deps.progressive_stream.progressive_from_message(msg)   # (probes the source: here, off the loop)
            if po is None:
                return {"ok": False, "error": "file not found"}
            self.pinbox.put(po)
            return {"ok": True}
        if msg.get("processed") is True and isinstance(msg.get("input"), str):   # from the queue app
            po = self._deps.processed_from_message(msg)
            if po is None:
                return {"ok": False, "error": "file not found"}
            self.pinbox.put(po)
            return {"ok": True}
        if msg.get("cmd") == "open" and isinstance(msg.get("input"), str):
            hdrs = msg.get("headers") if isinstance(msg.get("headers"), dict) else {}
            quality, overrides = self.open_extras(msg)
            stream_item = msg.get("stream_item")
            stream_item = stream_item if isinstance(stream_item, str) and stream_item else None   # (anything else: ignored)
            self.inbox.put((msg["input"], {str(k): str(v) for k, v in hdrs.items()}, "轉送", 0.0, quality, overrides,
                            stream_item))
        return {"ok": True}

    def open_extras(self, msg) -> tuple[Any, dict]:   # listener thread (pure: no mpv IPC, no settings object)
        """The optional "quality" (a height; nearest available, like --quality) and "overrides" (settings for that
        source only) of an open request. Anything invalid is ignored with a log line."""
        quality, raw = msg.get("quality"), msg.get("overrides")
        if quality is not None:
            if isinstance(quality, float) and quality == int(quality):
                quality = int(quality)
            if isinstance(quality, bool) or not isinstance(quality, (int, str)):
                self._deps.log(f"open: quality {quality!r} ignored (expected a height)")
                quality = None
            else:
                try:
                    quality = self._deps.qualities.parse_quality_spec(quality)
                except ValueError as e:
                    self._deps.log(f"open: quality ignored ({e})")
                    quality = None
        overrides, notes = self._deps.PS.clean_overrides(raw) if raw else ({}, [])
        for note in notes:
            self._deps.log(f"open: {note}")
        return quality, overrides

    def max_frames_for(self, source: Any):
        return int(self.args.duration * float(source.meta.video_fps_exact)) if self.args.duration > 0 else None

    def configure_interp(self, sess: Any, mode: str):
        """The interpolation setting for the session's next segments (it is stopped or not started yet)."""
        sess.interp_mode = mode
        sess.interp = self.engine.interpolator() if mode == "auto" else None
        if mode != "auto":
            self.engine.release_interp()

    def configure_compare(self, sess: Any, mode: str):
        """The compare setting for the session's next segments; never while casting (one producer, no extra tracks)."""
        sess.compare_mode = "on" if mode == "on" and (self.cast is None or self.cast.active is None) else "off"

    def begin_source(self, opened: Any):
        """New Session (+ thumbnails) for an opened input and start Lada at its start position (no IPC here)."""
        self.epoch += 1
        self.cur_open = (opened.input, opened.user_headers)
        self.session = self._deps.Session(self.engine.lada.open(opened.source), opened.source, self.engine.encoder(),
                          self.max_frames_for(opened.source))   # no other reference to the restorer: a model reload
        self.qc = self._deps.QualityControl(opened.quals, opened.chosen, self.engine.lada, opened.user_headers,   # can free it
                            opened.refresh)
        self.configure_interp(self.session, self.engine.sc.cfg["interpolation"])
        self.configure_compare(self.session, self.engine.sc.cfg.get("compare", "off"))
        self.title, self.source_text = opened.title, opened.source_text
        self.session.start(opened.start, gate_release_s=self.args.initial_prebuffer + 0.5)   # Lada warms up while we publish / mpv starts
        self._cast_stream_restarts = 0
        self._cast_stream_want_k = None
        self.stream_item = getattr(opened, "stream_item", None) or None
        self.stream_link = None
        self.stream_subs = StreamSubtitles(self.stream_item) if self.stream_item else None
        if self.stream_subs is not None:                  # a snapshot may have arrived before this open was consumed
            self.apply_pending_subtitle_update(self.stream_item)
        if self.stream_item and getattr(self, "link", None) is not None:
            self.stream_link = StreamSubtitleLink(self.stream_item, self.link.post, self._deps.log)
            if opened.start and opened.start > 0:      # opened mid-video: the subtitles start there
                self.stream_link.want(opened.start, self._deps.time.perf_counter())
        self.thumbs, self.thumb_dir, self.prefetch_started, self.thumb_sent = None, None, False, None
        if self.args.thumbnails and opened.thumb_src and self.session.meta.duration > 0:
            self.thumb_dir = self._deps.tempfile.mkdtemp(prefix="oh_thumbs_")
            self.thumbs = self._deps.ThumbnailWorker(opened.thumb_src[0], opened.thumb_src[1], self.session.meta.duration,
                                     self.args.thumb_width, self.thumb_dir)
            self.thumbs.start()

    def drop_tail(self, why: str) -> str:
        """Discard the seek tail (seek_tail.py), if any: passthrough stopped, old muxer killed. Returns its summary."""
        if self.tail is None:
            return ""
        t, self.tail = self.tail, None
        t.close(why)
        note = t.summary()
        self._deps.log(f"{note} | old muxer closed")
        try:
            self.ipc.set("user-data/oh/tail", "")
        except (self._deps.MpvError, OSError, EOFError, ValueError, AttributeError):   # mpv gone / IPC not up (exit)
            pass
        return note

    def end_source(self) -> float:
        """Stop the current Session (abandoning work in progress) and its thumbnails, or the progressive feeder (its
        segment / source files are closed afterwards: the queue may delete work/); returns the stop time."""
        took = 0.0
        self.drop_tail("source closed")
        self._cast_stream_restarts = 0
        self._cast_stream_want_k = None
        ended, self.stream_item, self.stream_link = self.stream_item, None, None   # the item belongs to the closing source
        self.stream_subs = None
        if getattr(self, "restore_pause", None) is not None:
            self.restore_pause.reset()                 # (Session.stop below clears the pause itself)
        self.release_stream_item(ended)                # (not for a segment restart: those never come through here)
        if self.progressive is not None:
            p, self.progressive = self.progressive, None
            p.close()
        self.progressive_active_url = None
        self.progressive_source = False
        self.progressive_generation += 1
        self.progressive_handoff = None
        self.progressive_candidate_at = None
        if self.progressive_clip is not None:
            self.progressive_clip.retire()
            self.progressive_clip = None
        if getattr(self, "progressive_sub_dir", None):
            self._deps.shutil.rmtree(self.progressive_sub_dir, ignore_errors=True)
            self.progressive_sub_dir = None
        self.progressive_sub_loaded = None
        self.progressive_sub_selection = None
        self.progressive_sub_visibility = None
        if self.session is not None:
            took = self.session.stop()
            self.session.release_restorer()
            self.session = None
        if self.thumbs is not None:
            self.thumbs.close()
            self.thumbs = None
        return took

    def perf_inputs(self):
        """(profile, interpolation setting) for the performance hints; profile None = no hints."""
        cfg = self.sc.cfg if self.sc is not None else self.engine.cfg
        return perf_hint.for_detector(perf_hint.get_profile(), cfg.get("detection_model")), cfg.get("interpolation")

    def show_perf_hint(self):
        """Live restore only: a short OSD warning if the saved performance profile says this machine will not keep
        up with the source (once per opened source, again only when a quality switch changed the verdict level)."""
        try:
            prof, interp = self.perf_inputs()
            if prof is None or self.session is None:
                return
            hint = self.hint_tracker.next(self.epoch, prof, self.session.meta.video_height, self.session.fps, interp)
            if hint is not None:
                self.osd(hint[1], hint[2])
        except Exception as e:  # noqa: BLE001  (a hint must never disturb playback)
            self._deps.log(f"perf hint skipped: {e!r}")

    def publish_qualities(self):
        lst = self._deps.qualities.public_list(self.qc.qualities) if len(self.qc.qualities) > 1 else "[]"
        if lst != "[]":
            try:
                prof, interp = self.perf_inputs()
                cfg = self.sc.cfg if self.sc is not None else self.engine.cfg
                lst = perf_hint.annotate_public_list(lst, self.qc.qualities, prof, interp,    # ✅ / ⏳ / 📥 marks
                                                     cfg.get("restoration_fps", "original"))
            except Exception:  # noqa: BLE001
                pass
        if lst != self.published_list:
            self.ipc.set("user-data/oh/qualities", lst)
            self.published_list = lst
        self.ipc.set("user-data/oh/quality-current", self.qc.current["id"])

    ENGINE_IDLE_UNLOAD_S = 180.0   # no live source for this long: the models (and RIFE) leave the GPU

    def live_source_ended(self) -> bool:
        """A live Session whose video has played to its end (mpv holds the last frame): nothing needs the models now."""
        return (self.session is not None and self.loaded and self.tail is None
                and (self.cast is None or self.cast.active is None) and self.ipc.get("eof-reached", False) is True)

    def release_idle_engine(self, now: float):
        """The models load on demand (Engine.ensure_loaded); once nothing has needed them for ENGINE_IDLE_UNLOAD_S
        (a processed / progressive file plays, nothing plays, or a live video ended), free their VRAM so the queue's
        worker has it. A live source that ended is closed first (the player goes idle)."""
        engine = self.engine
        ended = self.live_source_ended() if engine.lada is not None and self.session is not None else False
        busy = ((self.session is not None and not ended) or self.pending_open is not None or self.opener.busy
                or self.loading_models or engine.loading)
        if engine.lada is None or busy:
            self.engine_idle_since = None
            return
        if self.engine_idle_since is None:
            self.engine_idle_since = now
        elif now - self.engine_idle_since >= self.ENGINE_IDLE_UNLOAD_S:
            self.engine_idle_since = None
            if self.session is not None:
                self._deps.log("the video ended and nothing happened since: closing it to free the models")
                self.leave_current()
                self.ipc.set("user-data/oh/active", "0")
            self.qc = None                    # (a finished live source's quality control still points at the engine)
            engine.unload()
            self.update_status()

    def publish_settings(self):
        if self.sc is not None:   # None until the models have loaded: the loop publishes the menu once they have
            s = self._deps.json.dumps(self.sc.menu(), ensure_ascii=False)
            if s != self.published_settings:
                self.ipc.set("user-data/oh/settings", s)
                self.published_settings = s
        self.publish_player_opts()

    def publish_player_opts(self):
        """Settings that act in mpv itself: RTX VSR (oh_seek.lua manages the filter from user-data/oh/vsr)
        and smooth playback (mpv properties). Only what changed is sent; start_mpv() passed the startup values."""
        cfg = self.sc.cfg if self.sc is not None else self.engine.cfg
        if cfg["vsr"] != self.published_opts.get("vsr"):
            self.ipc.set("user-data/oh/vsr", cfg["vsr"])
            self.published_opts["vsr"] = cfg["vsr"]
        if cfg["smooth_playback"] != self.published_opts.get("smooth_playback"):
            for k, v in self._deps.PS.smooth_props(cfg["smooth_playback"]).items():
                try:
                    self.ipc.set(k, v)
                except self._deps.MpvError as e:
                    self._deps.log(f"could not set {k}={v}: {e}")
            self.published_opts["smooth_playback"] = cfg["smooth_playback"]

    def publish_compare_layers(self, titles: str | None = None, derive: str = ""):
        """user-data/oh/compare_layers = comma list of the video track titles the segment about to be loaded
        carries ("oh:processed,oh:original[,oh:restored]"); "" = no compare tracks (oh_seek.lua).
        titles: given for a processed file (probed from the file itself) and progressive playback.
        derive: user-data/oh/derive-1x = "<video_start>,<src_fps>" when the 1x "restored" layer has no track of
        its own and oh_seek.lua derives it from oh:processed (rate_mult 2 queue outputs); "" = a real track /
        none. Both are set before the load (oh_seek.lua's on_preloaded hook reads them)."""
        if titles is None:
            layers = getattr(self.session, "layers", ())
            titles = ",".join(self._deps.compare_stack.LAYER_TITLES[n] for n in layers) if len(layers) > 1 else ""
        try:
            if derive != self.published_derive:
                self.ipc.set("user-data/oh/derive-1x", derive)
                self.published_derive = derive
            self.ipc.set("user-data/oh/compare_layers", titles)
        except (self._deps.MpvError, OSError, EOFError, ValueError):
            pass

    def publish_source(self):
        """Tell oh_seek.lua about the source begin_source() started (length, start, qualities, title)."""
        self.published_duration = self.session.meta.duration
        self.ipc.set("user-data/oh/duration", self.published_duration)
        self.ipc.set("user-data/oh/start", self.session.target)
        self.published_list = None
        self.publish_qualities()
        self.publish_settings()
        if self.epoch > 1 or self.first is None:
            self.ipc.set("force-media-title", self.title)
            self.ipc.set("title", f"{self.title} - {APP_NAME}")
        self.publish_compare_layers()
        self.publish_sub_combos()
        self.ipc.set("user-data/oh/active", "1")

    def publish_sub_combos(self, refresh: bool = False):
        """Subtitle combos next to a local video (<video>.subtitle-combos/manifest.json): write the composite ASS and
        publish the combo list for oh_sub_combos.lua. Videos without a manifest: nothing is published.
        refresh=True (poll_sub_combos): the scheduler may still be writing; rebuild only when the manifest fingerprint
        changed, and when the set of combos is unchanged just bump user-data/oh/sub-combos-rev (the Lua script then
        reloads the subtitle tracks and keeps the user's mode) instead of re-publishing the list."""
        src = self.cur_open[0] if self.cur_open else None
        folder = None
        processed = getattr(self, "processed", None)
        if src is None and processed is not None:        # a queue output: its combos live in the item folder
            src, folder = processed.path, getattr(processed, "combos_dir", None)
        try:
            if not isinstance(src, str) or not self._deps.os.path.isfile(src):
                return
            if subtitle_compare.find_manifest(src, folder) is None:
                return                                   # the usual case: no temp dir, nothing published
            fp = subtitle_compare.manifest_fingerprint(src, folder)
            if refresh and fp == getattr(self, "combos_fp", None):
                return
            self.combos_fp, self.combos_watch = fp, True
            if self.combo_dir is None:
                self.combo_dir = self._deps.tempfile.mkdtemp(prefix="oh_subcombos_")
            prepared = subtitle_compare.prepare(src, self.combo_dir, folder)
            if prepared is None:
                return
            ids = [c["id"] for c in prepared.combos]
            if refresh and self.combos_published and ids == getattr(self, "combos_ids", None):
                self.combos_rev = getattr(self, "combos_rev", 0) + 1
                self.ipc.set("user-data/oh/sub-combos-rev", self.combos_rev)
                self._deps.log(f"subtitle combos updated (rev {self.combos_rev}): " + ", ".join(ids))
                return
            self.combos_ids = ids
            self.ipc.set("user-data/oh/sub-combos", subtitle_compare.publish_json(prepared))
            self.combos_published = True
            self._deps.log(f"subtitle combos: {len(prepared.combos)} for {src}"
                           + (f" (skipped: {'; '.join(prepared.skipped)})" if prepared.skipped else ""))
        except Exception as e:  # noqa: BLE001  (subtitles must never disturb playback)
            self._deps.log(f"subtitle combos skipped: {e!r}")

    def poll_sub_combos(self, now: float) -> float:
        """Main-loop hook: re-check the manifest every POLL_INTERVAL_S while the video has one (the slice scheduler
        grows it). Returns the next check time."""
        if getattr(self, "combos_watch", False):
            self.publish_sub_combos(refresh=True)
        return now + subtitle_compare.POLL_INTERVAL_S

    def clear_sub_combos(self):
        self.combos_watch, self.combos_fp, self.combos_ids = False, None, None
        if not self.combos_published:
            return
        self.combos_published = False
        try:
            self.ipc.set("user-data/oh/sub-combos", "")
        except (self._deps.MpvError, OSError, EOFError, ValueError, AttributeError):
            pass

    def show_error(self, msg):
        try:
            self.ipc.command("show-text", msg, 6000)
        except self._deps.MpvError:
            pass

    def update_status(self):
        """Centred status text drawn by oh_seek.lua: the idle hint, 解析中…, 載入模型中…"""
        if self.cast.active is not None:
            s = self.cast.status_text()
        elif self.opener.busy:
            s = "解析中…"
        elif self.pending_open is not None:
            s = "載入模型中…"
        elif self.progressive is not None and self.progressive_source and self.loaded and not self.resume_wait:
            s = {"paused": "原片播放中，背景修復已暫停", "failed": "原片播放中，背景修復失敗"}.get(
                self.progressive.hold, "原片播放中，背景修復中")
        elif self.progressive is not None and self.loaded and not self.resume_wait and self.progressive.waiting_k is not None \
                and self.ipc.get("paused-for-cache"):
            s = {"paused": "背景處理已暫停\n在佇列視窗按「繼續」", "failed": "背景處理失敗\n在佇列視窗按「重試」"}.get(
                self.progressive.hold, "處理中…\n等待背景處理到這裡")   # ran into a part the worker has not processed yet
        elif self.session is None and self.processed is None and self.progressive is None:
            if self.engine.error is not None:
                s = self._deps.IDLE_HINT + "\n模型載入失敗（詳見記錄檔）"
            else:
                s = self._deps.IDLE_HINT + ("\n模型載入中…" if self.engine.loading else "")
        else:
            s = ""
        if s != self.status_shown:
            self.ipc.set("user-data/oh/status", s)
            self.status_shown = s

    def publish_info(self, force_state=None):
        """Gather the snapshot (cheap: a few IPC gets + cached GPU stats) and publish the panel JSON."""
        now = self._deps.time.perf_counter()
        torch_alloc = self._deps.device_backend.allocated_gib(getattr(self.engine, "device", None))
        if self.session is None:
            state = force_state or ("播放已處理的影片" if self.processed is not None
                                    else "邊處理邊看" if self.progressive is not None else "解析中" if self.opener.busy
                                    else "載入模型中" if self.engine.loading else "閒置")
            f = self.engine.lada
            mc = f and f.models
            snap = {"state": state, "detection_model": mc and mc.detection, "restoration_model": mc and mc.restoration,
                    "max_clip_length": f and f.max_clip_length, "fp16": mc and mc.fp16,
                    "prebuffer": self.args.prebuffer, "torch_alloc_gib": torch_alloc, "gpu": self.gpu.snapshot(),
                    "history": self.hist.snapshot(), "vsr": self.ipc.get("user-data/oh/vsr-status"),
                    "smooth_playback": (self.sc.cfg if self.sc is not None else self.engine.cfg)["smooth_playback"],
                    "interp": {"mode": (self.sc.cfg if self.sc is not None else self.engine.cfg)["interpolation"]},
                    "compare": {"mode": (self.sc.cfg if self.sc is not None else self.engine.cfg).get("compare", "off")}}
            if self.processed is not None or self.progressive is not None:
                # the video info comes from mpv (no Session / meta here)
                pb = self._deps.mpv_playback_snapshot(self.ipc.get, self.published_duration or None)
                paused = pb.pop("paused")
                snap.update(pb, title=self.title, source=self.source_text, cast=self.cast.info_snapshot())
                if self.progressive is not None:
                    # the worker's real progress, from the segment list (progressive_stream.status)
                    ps = self.progressive.status(pb["pos"])
                    snap.update(progressive=ps, fps=ps["src_fps"], out_fps=ps["out_fps"])
                else:
                    snap["proc_done"] = True
                    t = self.processed.tags
                    if t:        # a queue output: the source rate and the (interpolated) output rate from its tags
                        snap.update(fps=float(t["source_fps"]), out_fps=float(t["source_fps"]) * t["rate_mult"])
                if not force_state and paused and self.loaded and not self.resume_wait:
                    snap["state"] = f"{state} · {paused}"
            snap.update(subtitle_status=self.current_subtitle_status(), subtitle_now=now)
            self.ipc.set("user-data/oh/info", self._deps.json.dumps(self._deps.info_panel.build_info(snap), ensure_ascii=False))
            return
        proc_fps = self.proc_fps_now
        pos = self.ipc.get("time-pos")
        cache_time = self.ipc.get("demuxer-cache-time")
        if force_state:
            state = force_state
        elif self.cast.active is not None:
            state = "投放中" if self.cast.active["mode"] == "airplay" else "區網串流中"
        elif not self.loaded or self.resume_wait:
            state = f"緩衝中 {self.prog_pct}%" if self.prog_pct is not None else "處理中"
        elif self.ipc.get("paused-for-cache"):
            state = "緩衝中"
        elif self.ipc.get("pause"):
            state = "暫停"
        else:
            state = "播放中"
        m = self.session.meta
        pipe = None
        r = self.session.restorer
        if r is not None and self.session.thread is not None:
            pipe = r.metrics().queues or None
        fd, vd = self.ipc.get("frame-drop-count"), self.ipc.get("vo-delayed-frame-count")
        snap = {
            "pipe": pipe,
            "drops": (int(fd or 0), int(vd or 0)) if fd is not None or vd is not None else None,
            "title": self.title, "source": self.source_text, "width": m.video_width, "height": m.video_height,
            "native_fps": float(getattr(self.session.source, "native_meta", m).video_fps_exact),
            "fps": self.session.fps, "codec": m.codec_name, "duration": self.published_duration,
            "quality": self.qc.current.get("label"),
            "pos": pos, "cache_time": cache_time, "proc_fps": proc_fps,
            "proc_done": self.loaded and self.session.processing_done(), "frames_total": self.session.frames_total,
            "segment_start": self.session.first_s if self.session.first_s is not None else self.session.target,
            "last_latency": self.last_latency, "last_kind": self.last_kind, "state": state,
            "detection_model": self.sc.lada.models.detection, "restoration_model": self.sc.lada.models.restoration,
            "max_clip_length": self.sc.lada.max_clip_length, "fp16": self.sc.lada.models.fp16, "encoder": self.session.encoder,
            "prebuffer": self.args.prebuffer, "torch_alloc_gib": torch_alloc, "gpu": self.gpu.snapshot(),
            "cast": self.cast.info_snapshot(), "history": self.hist.snapshot(),
            "vsr": self.ipc.get("user-data/oh/vsr-status"), "smooth_playback": self.sc.cfg["smooth_playback"],
            "interp": self.session.interp_snapshot(), "compare": self.session.compare_snapshot(),
        }
        try:
            prof, interp = self.perf_inputs()
            snap["perf"] = perf_hint.info_for(prof, m.video_height, self.session.fps, interp) if prof else None
        except Exception:  # noqa: BLE001
            snap["perf"] = None
        if snap["cast"] is not None:
            cs = snap["cast"]
            pos = cs["tv_pos"]
            if pos is not None:
                pos += self.cast.active.get("v0", 0.0)
            else:
                pos = self.cast.active.get("local_pos")
            offset = self.cast.active.get("v0", 0.0)
            ranges = [(start + offset, end + offset) for start, end in cs["cached_ranges"]]
            cache_time = next((end for start, end in ranges
                               if pos is not None and start <= pos <= end), None)
            snap.update(pos=pos, cache_time=cache_time, frames_total=cs["frames_written"],
                        segment_start=cs["run_start"] + offset if cs["run_start"] is not None else None,
                        proc_done=cs["processing_done"],
                        pipe=None, drops=None)
        snap.update(subtitle_status=self.current_subtitle_status(), subtitle_now=now)
        self.ipc.set("user-data/oh/info", self._deps.json.dumps(self._deps.info_panel.build_info(snap), ensure_ascii=False))

    def sample_history(self, now):
        """Once a second (panel shown or not): processing speed, buffer ahead and GPU stats for the charts."""
        ratio = buffer = None
        if self.cast.active is not None:
            cs = self.cast.info_snapshot()
            key = (id(self.cast.active["vod"]), cs["restarts"], cs["run_start"])
            if key != getattr(self, "cast_sample_key", None):
                self.proc_samples.clear()
                self.cast_sample_key = key
            self.proc_samples.append((now, cs["frames_written"]))
            t0, f0 = self.proc_samples[0]
            self.proc_fps_now = (cs["frames_written"] - f0) / (now - t0) if now - t0 >= 0.5 else None
            fps = float(self.cast.active["plan"].fps)
            if self.proc_fps_now is not None and fps:
                ratio = self.proc_fps_now / fps
            pos = cs["tv_pos"]
            pos = pos + self.cast.active.get("v0", 0.0) if pos is not None else self.cast.active.get("local_pos")
            offset = self.cast.active.get("v0", 0.0)
            ranges = [(start + offset, end + offset) for start, end in cs["cached_ranges"]]
            buffer = next((max(0.0, end - pos) for start, end in ranges
                           if pos is not None and start <= pos <= end), None)
        elif self.session is not None:
            if getattr(self, "cast_sample_key", None) is not None:
                self.proc_samples.clear()
                self.cast_sample_key = None
            self.proc_samples.append((now, self.session.frames_total))
            t0, f0 = self.proc_samples[0]
            self.proc_fps_now = (self.session.frames_total - f0) / (now - t0) if now - t0 >= 0.5 else None
            if self.proc_fps_now is not None and self.session.fps:
                ratio = self.proc_fps_now / self.session.fps
            if self.loaded and not self.resume_wait and self.cast.active is None:
                pos, ct = self.ipc.get("time-pos"), self.ipc.get("demuxer-cache-time")
                if isinstance(pos, (int, float)) and isinstance(ct, (int, float)):
                    buffer = max(0.0, ct - pos)
        else:
            self.proc_fps_now = None
            if self.progressive is not None:
                self.progressive.sample(now)
                ratio = self.progressive.speed()
                if self.loaded and not self.resume_wait:
                    pos, ct = self.ipc.get("time-pos"), self.ipc.get("demuxer-cache-time")
                    if isinstance(pos, (int, float)) and isinstance(ct, (int, float)):
                        buffer = max(0.0, ct - pos)
        g = self.gpu.snapshot() or {}
        self.hist.add(ratio=ratio, buffer=buffer, util=g.get("util"), vram=g.get("mem_used_gib"), power=g.get("power_w"))

    def publish_progress(self, now):
        """user-data/oh/progress = {"req": <request being served>, "pct", "stage", "eta"} while a start / seek /
        switch is not playing yet ("" otherwise); oh_seek.lua shows it instead of a bare 處理中…. The
        percentage only goes up within one request."""
        if self.progressive is not None and self.cast.active is None and (not self.loaded or self.resume_wait):
            # progressive start / seek: 背景處理中 while the segment is not processed, then mpv's startup buffer
            buffered = None
            if self.loaded:
                ct = self.ipc.get("demuxer-cache-time")
                buffered = max(0.0, ct - self.progressive.target) if isinstance(ct, (int, float)) else 0.0
            if self.prog_key != self.served:
                self.prog_key, self.prog_pct = self.served, None
            msg, self.prog_pct = self._deps.progressive_stream.progress_json(self.served, self.progressive.waiting_k is not None, buffered,
                                                             self.mpv_wait, self.prog_pct or 0, self.progressive.hold)
            if msg != self.prog_shown:
                self.ipc.set("user-data/oh/progress", msg)
                self.prog_shown = msg
            return
        if self.session is None or self.cast.active is not None or (self.loaded and not self.resume_wait) or self.session.stats is None:
            if self.prog_shown:
                self.ipc.set("user-data/oh/progress", "")
                self.prog_shown = ""
            self.prog_key, self.prog_pct = None, None
            return
        # while a seek tail plays, mpv's cache is the old timeline's: progress runs toward the switch (the output
        # switch_ahead() past the target) and is measured on Lada's output until mpv is on the new segment
        in_mpv = self.loaded and (self.tail is None or self.ipc.get("path") == self.session.url)
        buffered = None
        if in_mpv:
            ct = self.ipc.get("demuxer-cache-time")
            buffered = max(0.0, ct - self.session.target) if isinstance(ct, (int, float)) else 0.0
        wait = self.args.initial_prebuffer if self.n_req == 0 else self.switch_ahead() if self.tail is not None else self.args.seek_prebuffer
        frac, stage, remaining = self._deps.buffer_progress.estimate(
            self.session.probe.snapshot() if self.session.probe is not None else None, now, elapsed=now - self.t_req,
            target_s=self.session.target, fps=self.session.fps, wait_s=wait,
            max_clip=getattr(self.session.restorer, "max_clip_length", 180), out_n=self.session.lada_out_n(),
            first_out=self.session.first_out, loaded=in_mpv, buffered_s=buffered)
        if self.prog_key != self.served:
            self.prog_key, self.prog_pct = self.served, None
        self.prog_pct = max(self.prog_pct or 0, int(frac * 100))
        msg = self._deps.json.dumps({"req": self.served, "pct": self.prog_pct, "stage": stage,
                          "eta": round(remaining) if remaining >= 1.5 else None}, ensure_ascii=False)
        if msg != self.prog_shown:
            self.ipc.set("user-data/oh/progress", msg)
            self.prog_shown = msg

    def prepare(self, fid: str, state: dict):
        """Probe the rendition `fid`. A failure (e.g. an expired signed URL) re-runs yt-dlp once and retries."""
        q = self._deps.qualities.find_quality(self.qc.qualities, fid)
        if q is None:
            raise RuntimeError(f"unknown quality id {fid!r}")
        try:
            return q, self.qc.open(q)
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"opening {q['label']} failed: {str(e).splitlines()[0] if str(e) else e!r}")
            if self.qc.refresh is None or state["refreshed"]:
                raise
            state["refreshed"] = True
            self._deps.log("re-running yt-dlp to refresh the format URLs, then retrying once")
            fresh = self.qc.refresh()
            q2 = self._deps.qualities.find_quality(fresh, fid, q["height"])
            if q2 is None:
                raise RuntimeError(f"{q['label']} is no longer offered after refresh")
            self.qc.qualities = fresh
            cur = self._deps.qualities.find_quality(fresh, self.qc.current["id"], self.qc.current["height"])
            if cur is not None:
                self.qc.current = cur
            return q2, self.qc.open(q2)

    def apply_source(self, q: dict, source: Any, restorer: Any):
        self.session.set_source(restorer, source)
        self.qc.current = q
        if abs(source.meta.duration - self.published_duration) > 0.01:
            self.published_duration = source.meta.duration
            self.ipc.set("user-data/oh/duration", self.published_duration)
        self.publish_qualities()

    def reload_models(self, changes: dict) -> bool:
        """Model / FP16 change: free the old models first (8 GB GPU), load the new ones; on failure reload the
        previous ones. The session is stopped and its restorer must be rebuilt afterwards either way."""
        lada = self.sc.lada
        if lada is None:
            return True                          # not loaded: the next load uses the new values (Engine.load reads sc.cfg)
        device, max_clip = lada.device, lada.max_clip_length   # release() forgets the device
        old = lada.models
        new = ModelChoice(changes.get("detection_model", old.detection),
                                     changes.get("restoration_model", old.restoration), changes.get("fp16", old.fp16))
        self.loading_models = True
        try:
            if self.info_visible:
                self.publish_info("載入模型中")      # the panel says it; keep its corner free of OSD text
            else:
                self.ipc.command("show-text", "載入模型中…", 120000)
        except self._deps.MpvError:
            pass
        if self.session is not None:             # (none when overrides are applied between two sources)
            self.session.release_restorer()
        if self.engine.interp is not None:
            self.engine.interp.wait_idle()       # a CUDA graph capture must not overlap empty_cache()
        lada.release()
        free_gib = self._deps.device_backend.allocated_gib(device)
        t = self._deps.time.perf_counter()
        ok = True
        try:
            lada.load(device, new, max_clip)
        except Exception as e:  # noqa: BLE001
            ok = False
            self._deps.log(f"loading models {new} failed ({str(e).splitlines()[0][:120] if str(e) else repr(e)}); "
                f"restoring {old}")
            lada.release()
            lada.load(device, old, max_clip)   # if the previous models cannot be loaded either, there is nothing left to play
        self.loading_models = False
        alloc = self._deps.device_backend.allocated_gib(device)
        self._deps.log(f"models {'reloaded' if ok else 'restored'} in {self._deps.time.perf_counter() - t:.1f}s: det={lada.models.detection} "
            f"res={lada.models.restoration} fp16={lada.models.fp16} | torch allocated {free_gib} GiB after release, "
            f"{alloc} GiB now")
        try:
            self.ipc.command("show-text", "已套用設定" if ok else "模型載入失敗，已還原設定", 2000 if ok else 6000)
        except self._deps.MpvError:
            pass
        return ok

    def apply_settings(self, changes: dict, *, transient: bool = False, persist: bool = True) -> tuple[dict, bool]:
        """Apply (non-live and live) setting changes while the session is stopped.

        transient: per-source overrides (apply_overrides) take effect but are not committed to the settings file, and
        nothing is published (the caller does, as it opens the source).
        persist=False: committed (the file's values are in use now) but not written: the file already holds them.
        Returns (changes that took effect, whether the restorer must be rebuilt)."""
        rebuild = False
        if changes.keys() & self._deps.PS.RELOAD_KEYS:
            rebuild = True
            if not self.reload_models(changes):
                changes = {k: v for k, v in changes.items() if k not in self._deps.PS.RELOAD_KEYS}
        if "restoration_fps" in changes and self.cast is not None and self.cast.active is not None:
            changes = {k: v for k, v in changes.items() if k != "restoration_fps"}
            self.ipc.command("show-text", "請先停止投放，再變更修復幀率", 4000)
        if "restoration_fps" in changes:
            if self.sc.lada is not None:
                self.sc.lada.restoration_fps = changes["restoration_fps"]
            self.proc_samples.clear()
            self.proc_fps_now = None
            self.hist.clear("ratio")
            rebuild = True
        if "max_clip_length" in changes:
            if self.sc.lada is not None:
                self.sc.lada.max_clip_length = changes["max_clip_length"]
            rebuild = True
        if "encoder" in changes and self.session is not None:
            self.session.encoder = self.sc.resolve_encoder(changes["encoder"])
        if "seek_prebuffer" in changes:
            self.args.seek_prebuffer = changes["seek_prebuffer"]
        if "interpolation" in changes and self.session is not None:
            self.configure_interp(self.session, changes["interpolation"])   # 2x / 1x output from the restart on
        if "compare" in changes and self.session is not None:
            self.configure_compare(self.session, changes["compare"])
        if transient:
            self.sc.apply_transient(changes)
            return changes, rebuild
        self.sc.commit(changes, write=persist)
        self.publish_settings()
        if "interpolation" in changes and self.qc is not None:
            self.publish_qualities()                 # the menu marks depend on it
        return changes, rebuild

    def apply_overrides(self, raw: dict):
        """Settings for the source about to open only (an "open" request's overrides): applied now, not committed;
        the values a previous source's overrides replaced come back unless these override them again. No session
        exists here (leave_current ran), so a model reload only has the shared models to swap."""
        sc = self.sc
        wanted = {}
        for k, v in (raw or {}).items():
            try:
                wanted[k] = self._deps.PS.parse_value(k, v, sc.menu())     # offered here: model installed, RIFE usable ...
            except ValueError as e:
                self._deps.log(f"override ignored: {e}")
        if not wanted and not sc.override_base:
            return                                   # the usual open: nothing to apply, nothing to revert
        target = {k: v for k, v in sc.override_base.items() if k not in wanted}   # revert what the last source overrode
        target.update(wanted)
        changes = sc.diff(target)
        if not changes:
            return
        before = {k: sc.cfg.get(k) for k in changes}
        self._deps.log(f"overrides for this source: {wanted or 'none'} (applying {changes})")
        took, _rebuild = self.apply_settings(changes, transient=True)
        for k in changes:
            if k not in took:
                continue                             # e.g. the models could not be loaded: nothing changed
            if k in wanted:
                sc.override_base.setdefault(k, before[k])
            else:
                sc.override_base.pop(k, None)
        for k in [k for k, v in sc.override_base.items() if sc.cfg.get(k) == v]:
            del sc.override_base[k]                  # back to the saved value: not an override any more
        sc.invalidate_menu()

    def settings_reload(self):
        """The settings file was rewritten by someone else (the queue app's 設為預設 / 套用建議設定): adopt the values
        that differ from what this player saved, as a menu change would, without writing the file back.

        A key held as a per-video override keeps its effective value: only the value it reverts to changes
        (override_base). The other changes wait in reload_pending for apply_reloaded_settings."""
        PS, sc = self._deps.PS, self.sc
        new, warning = PS.load_file(sc.path)
        if warning:
            self._deps.log(f"settings reload: {warning}")
        if not new:
            self._deps.log(f"settings reload: nothing usable in {sc.path}; keeping the settings in use")
            return
        wanted, base, adopted = {}, [], {}
        for k in PS.KEYS:
            before, after = sc.saved.get(k, PS.DEFAULTS[k]), new.get(k, PS.DEFAULTS[k])
            if after == before:
                continue
            if k in new:
                sc.saved[k] = new[k]
            else:
                sc.saved.pop(k, None)
            adopted[k] = after
            if k in sc.override_base:
                sc.override_base[k] = after          # what the key reverts to; the effective value stays
                if sc.cfg.get(k) == after:
                    del sc.override_base[k]          # the override happens to equal the new default: not one any more
                base.append(k)
            elif k == "fp16" and after is None:
                self._deps.log("settings reload: fp16 back to auto applies at the next start")
            else:
                try:
                    wanted[k] = PS.parse_value(k, PS.value_str(k, after), sc.menu())   # offered here: model installed ...
                except ValueError as e:
                    self._deps.log(f"settings reload: {k} not applied ({e})")
        self._deps.log(f"settings reload: {adopted or 'no changes'}"
                       + (f" (kept for this video, new default after it: {base})" if base else ""))
        if base:
            sc.invalidate_menu()
            self.publish_settings()
        self.reload_pending.update(wanted)

    def apply_reloaded_settings(self, now: float):
        """Put reload_pending to use. No Session runs (idle / a processed or progressive file): applied here, like the
        overrides between two sources. Otherwise it goes through a request like a menu change: while casting
        handle_requests is called directly (the receiver keeps playing), else oh_seek.lua is asked to restart the
        segment (oh-restart: it freezes playback and restores the user's pause state) and handle_requests applies
        the changes with that request."""
        if self.session is None:
            wanted, self.reload_pending = self.reload_pending, {}
            changes = self.sc.diff(wanted)
            if changes:
                self._deps.log(f"settings reload: applying {changes}")
                self.apply_settings(changes, persist=False)
                self.sc.invalidate_menu()
            return
        sreq, qreq, creq = (self.ipc.get(f"user-data/oh/{k}-request") for k in ("seek", "quality", "settings"))
        if (sreq and sreq != self.last_sreq) or (qreq and qreq != self.last_qreq) or (creq and creq != self.last_creq):
            return                                   # a request is on its way: handle_requests takes the changes along
        if self.cast.active is not None:
            self.handle_requests(self.last_sreq, self.last_qreq, self.last_creq)
            return
        if not self.loaded or self.resume_wait or self.restore_deadline is not None or self.tail is not None \
                or now - self.reload_restart_at < 5.0:
            return                                   # a segment is still starting (or the restart was just asked for)
        self.reload_restart_at = now
        pos = self._deps._float_or(self.ipc.get("time-pos"), self.session.target)
        paused = self.ipc.get("pause") is True and not self.ipc.get("eof-reached", False)
        self._deps.log(f"settings reload: restarting the segment at {pos:.2f}s for {self.reload_pending}")
        self.ipc.command("script-message", "oh-restart", f"{max(0.0, pos):.3f}", "1" if paused else "0")

    def set_cache_wait(self, v: float):
        if v != self.mpv_wait:
            self.ipc.set("cache-pause-wait", v)
            self.mpv_wait = v

    def switch_via(self) -> str | None:
        """How to hand the ready segment to mpv: "replace", or "append" behind the playing seek tail; None = not
        yet: while the tail plays, the switch waits until the new output is switch_ahead() past the target, so
        the new segment never starts starved (the tail costs nothing meanwhile). A paused tail, one at its end or
        one mpv no longer plays is replaced at once."""
        if self.tail is None:
            return "replace"
        if self.ipc.get("path") != self.tail.url or self.ipc.get("pause") is not False or self.ipc.get("eof-reached", False):
            return "replace"
        if not self.session.output_reached(self.switch_ahead()):
            return None
        return self._deps.seek_tail.switch_mode(self.tail.remaining(self._deps._float_or(self.ipc.get("time-pos"), self.tail.start_pos)))

    def withdraw_appended(self, url: str) -> bool:
        """Take a segment that was appended behind the tail but has not started out of mpv's playlist again."""
        for i, e in enumerate(self.ipc.get("playlist") or []):
            if e.get("filename") == url and not e.get("current") and not e.get("playing"):
                self.ipc.command("playlist-remove", i)
                self._deps.log(f"withdrew the appended segment {url} (superseded before it played)")
                return True
        return False

    def stop_for_request(self, sreq, qreq, creq) -> float:
        """Stop the segment for new request(s); returns the stop time. Every stop is fast (the old output is not
        needed: the new segment starts elsewhere or with other settings).

        A plain seek while mpv plays (not paused by the user) keeps the old timeline going (seek_tail.py): the
        segment mpv plays becomes the tail, or the tail that is playing already keeps playing and only the segment
        being prepared is dropped (rapid seeks). Anything else (paused, quality / settings change, nothing to
        continue) freezes playback (oh_seek.lua leaves plain seeks playing) and replaces it once the new
        segment is ready, as before."""
        seek_only = not (qreq and qreq != self.last_qreq) and not (creq and creq != self.last_creq)
        path = self.ipc.get("path")
        playing = self.ipc.get("pause") is False and not self.ipc.get("eof-reached", False) \
            and not self.ipc.get("idle-active", False)
        took = 0.0
        if seek_only and self.loaded and self.session.url and path == self.session.url and playing:
            pos = self._deps._float_or(self.ipc.get("time-pos"), self.session.target)
            took, out = self.session.detach_output()
            if out is not None:
                self.drop_tail("superseded: mpv plays a newer segment")
                src = self.session.source
                self.tail = self._deps.seek_tail.SeekTail(out["muxer"], out["url"], pos, out["next_t"], self.session.fps,
                                          lambda t: self._deps.passthrough_frames(src, t), self._deps.log,
                                          rate_mult=out["rate_mult"], layers=out["layers"]).start()
                self.tail_switch, self.tail_note = None, ""
                self.ipc.set("user-data/oh/tail", "1")
                self.set_cache_wait(self.args.seek_prebuffer)   # if the tail ever runs dry, mpv waits for little
                self.restore_deadline = None
                self._deps.log(f"tail: mpv keeps playing {out['url']} (at {pos:.2f}s, restored video to "
                    f"{out['next_t']:.3f}s{', segment had ended: no passthrough' if self.tail.ended else ''})")
                return took
        elif seek_only and self.tail is not None and path == self.tail.url:
            if self.loaded:
                self.withdraw_appended(self.session.url)
            self.tail_switch = None
            return self.session.stop()
        self.drop_tail("request needs a fresh start")
        if playing:
            self.ipc.set("pause", True)   # freeze; oh_seek.lua restores the user's state with the new segment
        return took + self.session.stop()

    def handle_cast_seek(self, sreq, now):
        self.last_sreq = sreq
        try:
            pos = float(sreq.split(":", 1)[1])
            a = self.cast.active
            airplay_playing = a is not None and a.get("mode") == "airplay" and a.get("phase") == "playing"
            self.cast.seek(pos)
            if self.stream_link is not None and airplay_playing:
                plan = a.get("plan")
                if plan is not None:
                    v0 = float(a.get("v0", 0.0))
                    self._cast_stream_want_k = plan.index_at(max(0.0, pos - v0))
                self.stream_link.want(pos, now)
        except (ValueError, IndexError):
            self._deps.log(f"invalid cast seek request: {sreq!r}")
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"cast seek failed: {e!r}")
        if self.ipc is not None:
            self.ipc.set("user-data/oh/request-done", sreq)

    def handle_requests(self, sreq, qreq, creq):
        """Seek / quality / settings request(s) from the UI: stop, apply, (re)open the source, restart at the
        position."""
        self.t_req = self._deps.time.perf_counter()
        self.switch_ctx = None
        casting = self.cast.active is not None
        if casting:
            # the local session is stopped already; the HLS producer is using the restorer: stop it
            self.cast.suspend()
            self.stop_s = 0.0
        else:
            # a paused restore must be back before the new segment restores (Session.start/stop clear the pause anyway):
            # restore_needed + up to 3 s for the queue. Blocking is fine here: this path already blocks on stop/probe.
            if getattr(self, "restore_pause", None) is not None:
                self.restore_pause.wait_for_restart(self.session, self.stream_item, "seek")
            self.stop_s = self.stop_for_request(sreq, qreq, creq)
        self.t_stopped = self._deps.time.perf_counter()
        state = {"refreshed": False}
        prepared = {}                      # fid -> (quality, Source) or None: a rapid sequence probes each id once
        probe_s = 0.0
        fid, pos = None, None              # wanted rendition (None = unchanged) and start position
        seeked = False                     # a seek request decided the position (stream item: tell the queue)
        wanted = {}                        # settings wanted by the UI (option values, typed)
        while True:
            news = [(self._deps.req_counter(r), kind, r) for kind, r, last in
                    (("q", qreq, self.last_qreq), ("s", sreq, self.last_sreq), ("c", creq, self.last_creq)) if r and r != last]
            self.last_sreq, self.last_qreq, self.last_creq = sreq, qreq, creq
            for _, kind, r in sorted(news):   # in request order: the newest one decides the position
                if kind == "s" and casting:
                    continue                  # seeking happens on the receiver while casting
                if kind == "q":
                    fid = r.split(":", 1)[1]
                    pos = self._deps._float_or(self.ipc.get("user-data/oh/quality-pos") or self.ipc.get("time-pos"))
                    self._deps.log(f"quality request #{self._deps.req_counter(r)} -> {fid} at {pos:.2f}s")
                elif kind == "s":
                    pos = self._deps._float_or(r.split(":", 1)[1])
                    seeked = True
                    self._deps.log(f"seek request #{self._deps.req_counter(r)} -> {pos:.2f}s")
                else:
                    w = self._deps.PS.parse_request(r, self.sc.menu())
                    wanted.update(w)
                    pos = self._deps._float_or(self.ipc.get("user-data/oh/settings-pos") or self.ipc.get("time-pos"))
                    self._deps.log(f"settings request #{self._deps.req_counter(r)} -> {w} at {pos:.2f}s")
                self.served = r
            if pos is None:
                pos = self.session.target
            if fid is not None and fid != self.qc.current["id"] and fid not in prepared:
                t = self._deps.time.perf_counter()
                try:
                    prepared[fid] = self.prepare(fid, state)
                except Exception as e:  # noqa: BLE001
                    prepared[fid] = None
                    first_line = str(e).splitlines()[0][:80] if str(e) else repr(e)
                    self._deps.log(f"quality switch failed ({first_line}); staying on {self.qc.current['label']}")
                    self.show_error(f"畫質切換失敗，維持 {self.qc.current['label']}")   # details are in the log
                probe_s += self._deps.time.perf_counter() - t
            # coalescing: anything newer than what this round used replaces it ("latest wins")
            s2, q2 = self.ipc.get("user-data/oh/seek-request"), self.ipc.get("user-data/oh/quality-request")
            c2 = self.ipc.get("user-data/oh/settings-request")
            if (s2, q2, c2) == (sreq, qreq, creq):
                break
            self._deps.log("newer request arrived while preparing: coalescing")
            sreq, qreq, creq = s2, q2, c2
        self.sw_note = ""
        self.req_kind = "seek"
        reloaded, self.reload_pending = self.reload_pending, {}   # file changes (settings_reload) go along; a menu choice wins
        changes = self.sc.diff({**reloaded, **wanted})
        rebuild = False
        if changes:
            t = self._deps.time.perf_counter()
            reload = bool(changes.keys() & self._deps.PS.RELOAD_KEYS)
            changes, rebuild = self.apply_settings(changes, persist=bool(changes.keys() - reloaded.keys()))
            self.req_kind = "模型" if reload else "設定"
            self.sw_note = (f"settings {changes or 'unchanged'} | {'model reload' if reload else 'new restorer'} "
                       f"{self._deps.time.perf_counter() - t:.2f}s | ")
            self._deps.log(f"settings applied: {self.sw_note.rstrip(' |')}")
        switched = False
        if casting and fid is not None and prepared.get(fid) and \
                self._deps.Fraction(prepared[fid][1].meta.video_fps_exact) != self.cast.active["plan"].fps:
            # the cast's segment layout is fixed by the frame rate (other resolutions are scaled to it)
            self._deps.log(f"quality switch to {prepared[fid][0]['label']} refused while casting: other frame rate")
            self.show_error("投放中無法切換到不同影格率的畫質")
            prepared[fid] = None
        if fid is not None and fid != self.qc.current["id"] and prepared.get(fid):
            q, source = prepared[fid]
            t = self._deps.time.perf_counter()
            prev = (self.session.source, self.qc.current)
            try:
                restorer = self.qc.lada.open(source)
            except Exception as e:  # noqa: BLE001
                self._deps.register_source(prev[0])
                self._deps.log(f"could not set up {q['label']}: {e!r}")
                self.show_error(f"畫質切換失敗，維持 {self.qc.current['label']}")
            else:
                old_label = self.qc.current["label"]
                self.apply_source(q, source, restorer)
                rebuild, switched = False, True
                self.switch_ctx = None if casting else {"prev": prev, "pos": pos, "served": self.served, "label": q["label"]}
                self.sw_note += (f"quality {old_label} -> {q['label']} ({source.describe()}) | probe {probe_s:.2f}s | "
                            f"new restorer {self._deps.time.perf_counter() - t:.2f}s (models reused) | ")
                self.req_kind = "畫質" if self.req_kind == "seek" else self.req_kind
                self._deps.log(f"switched source: {self.sw_note.rstrip(' |')}")
        if rebuild or self.session.restorer is None:
            self.session.set_source(self.qc.lada.open(self.session.source), self.session.source)
            self.session.max_frames = self.max_frames_for(self.session.source)
        self.n_req += 1
        if casting:
            # the receiver keeps playing what it has; new segments come with the new quality / settings
            self.cast.resume(self.session.encoder, changed=bool(changes) or switched)
            self.ipc.set("user-data/oh/request-done", self.served)   # oh_seek.lua: no local load follows
            self._deps.log(f"applied while casting: {self.sw_note.rstrip(' |') or 'nothing changed'}")
            return
        if seeked and self.stream_link is not None:
            self.stream_link.want(pos, self._deps.time.perf_counter())
        self.session.start(pos, self.t_req, gate_release_s=self.args.seek_prebuffer + 0.5)   # n_req >= 1: the seek prebuffer
        self.loaded, self.resume_wait, self.restore_deadline = False, False, None

    def revert_switch(self, ctx):
        """The new rendition opened (probe ok) but Lada could not read/decode it: go back to the old one."""
        prev_source, prev_q = ctx["prev"]
        self._deps.log(f"{ctx['label']} could not be played; falling back to {prev_q['label']}")
        self.session.stop()
        self.apply_source(prev_q, prev_source, self.qc.lada.open(prev_source))   # make() re-registers its I/O hooks
        self.show_error(f"畫質切換失敗，已回到 {prev_q['label']}")
        self.switch_ctx, self.sw_note = None, "fallback to previous quality | "
        self.t_req = self._deps.time.perf_counter()
        self.session.start(ctx["pos"], self.t_req, gate_release_s=self.args.seek_prebuffer + 0.5)
        self.loaded, self.resume_wait, self.restore_deadline = False, False, None

    def apply_live(self, lset):
        """Settings that need no restart: seek prebuffer (takes effect at the next seek), RTX VSR and smooth
        playback (sent to mpv at once by publish_settings)."""
        changes = self.sc.diff(self._deps.PS.parse_request(lset, self.sc.menu()))
        changes = {k: v for k, v in changes.items() if k in self._deps.PS.LIVE_KEYS}
        if changes:
            self._deps.log(f"settings (live) -> {changes}")
            self.apply_settings(changes)
        else:
            self.publish_settings()

    def lua_reset(self):
        """Clear oh_seek.lua's per-source state (requests, pending state, thumbnails, qualities, menus, length)
        and wait until it has done so: afterwards every request property holds "" or a request about the new
        source."""
        self.reset_seq += 1
        tag = str(self.reset_seq)
        self.clear_sub_combos()
        self.ipc.command("script-message", "oh-reset", tag)
        deadline = self._deps.time.perf_counter() + 3.0
        while self.ipc.get("user-data/oh/reset-ack") != tag:
            if self._deps.time.perf_counter() > deadline:
                self._deps.log("oh_seek.lua did not acknowledge the reset (continuing)")
                # a request of the old source left in place would be replayed on the new one (seek to the old position)
                for k in ("seek-request", "quality-request", "settings-request"):
                    self.ipc.set(f"user-data/oh/{k}", "")
                break
            self._deps.time.sleep(0.01)
        self.last_sreq = self.last_qreq = self.last_creq = None
        self.thumb_sent = None

    def leave_current(self) -> float:
        """Close whatever plays (a live Session or a processed file) before another source is opened: cast
        stopped, Session ended, mpv emptied, oh_seek.lua's per-source state reset, GPU memory handed back.
        Returns how long stopping the Session took."""
        if self.cast.active is not None:        # the cast belongs to the old source (its playlist, its length)
            self.cast.stop("已停止投放（開啟了新影片）", resume_local=False)
        old_dir = self.thumb_dir
        had = self.session is not None or self.processed is not None or self.progressive is not None
        had_prog = self.progressive is not None
        took = self.end_source()               # (also stops a progressive feeder: segment files released)
        if self.processed is not None:
            self.processed = None
            self.ipc.set("user-data/oh/processed", "")
        if had_prog:
            self.ipc.set("user-data/oh/ranges", "")
        if had:
            self.ipc.command("stop")            # drop the old segment (and its cache ranges) from mpv
        self.lua_reset()                        # also removes the thumbnail overlay before its files go away
        if old_dir:
            self._deps.shutil.rmtree(old_dir, ignore_errors=True)
            self.thumb_dir = None
        self._deps.forget_sources()
        if self.engine.interp is not None:
            self.engine.interp.wait_idle()       # a CUDA graph capture must not overlap empty_cache()
        self._deps.gc.collect()
        if self.engine.lada is not None:
            self._deps.device_backend.empty_cache(getattr(self.engine, "device", None))
        return took

    def osd(self, msg: str, ms: int = 2500):
        try:
            self.ipc.command("show-text", msg, ms)
        except self._deps.MpvError:
            pass

    def load_processed(self, po: Any, pos: float, tag: str):
        """loadfile of a processed file (plain mpv playback: its own seeking, cache and tracks), at pos."""
        self.progressive_sub_loaded = None
        self.served, self.loaded, self.resume_wait, self.restore_deadline = tag, True, False, None
        self.set_cache_wait(self.args.prebuffer)
        self.ipc.set("user-data/oh/served", tag)
        self.ipc.set("user-data/oh/expect-load", po.path)   # oh_seek.lua: our load, not a drop
        self.publish_compare_layers(po.layers, po.derive)     # compare tracks the file carries (before the load)
        if pos > 0:
            self.ipc.command("loadfile", po.path, "replace", -1, f"start={pos:.3f}")
        else:
            self.ipc.command("loadfile", po.path, "replace")

    def switch_to_processed(self, po: Any):
        """Play a processed file (queue output) as is: no Session, no restorer, models untouched."""
        t0 = self._deps.time.perf_counter()
        if hasattr(self.opener, "cancel"):
            self.opener.cancel()                # whatever was being resolved is superseded
        self.drop_pending_open()
        self.leave_current()
        self.epoch += 1
        self.t_req = self._deps.time.perf_counter()
        self.title, self.source_text, self.cur_open = po.title, "已處理（佇列成品）", None
        self.published_duration, self.published_list = 0.0, None
        self.processed, self.processed_sent, self.processed_next = po, False, 0.0
        self.processed_watch_next = 0.0
        self.processed_clock_next = 0.0
        self.ipc.set("user-data/oh/processed", "1")
        self.ipc.set("user-data/oh/qualities", "[]")        # no renditions, no quality button
        self.publish_settings()                               # VSR / smooth stay live; restart-kind rows say so
        self.ipc.set("force-media-title", self.title)
        self.ipc.set("title", f"{self.title} - {APP_NAME}")
        self.publish_sub_combos()
        self.ipc.set("user-data/oh/active", "1")
        self.load_processed(po, getattr(po, "start", 0.0) or 0.0, f"init{self.epoch}")
        self._deps.log(f"opened processed {po.title!r}: {po.path} (queue item {po.queue_id}, compare tracks "
            f"{po.layers or 'none'}) | setup {self._deps.time.perf_counter() - t0:.2f}s")

    def play_processed(self, po: Any, label: str):
        if self.processed is not None and po.from_lookup and self.processed.path == po.path:
            self.osd("已在播放已處理的版本")
            return
        try:
            self.switch_to_processed(po)
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"could not play processed {po.path}: {e!r}")
            self.show_error(self._deps.short_error(e))
            self.ipc.set("user-data/oh/active", "0")
            return
        if po.from_lookup:
            self.osd("播放已處理的版本")
        if getattr(po, "subtitle_error", False):
            self.osd("字幕產生失敗", 6000)
        self.apply_pending_subtitle_update(getattr(po, "queue_id", None))

    @staticmethod
    def subtitle_clock_position(position, po):
        """The audio-relative playback position represented by one external subtitle clock."""
        if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position):
            return None
        offset = getattr(po, "subtitle_offset", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
            offset = 0.0
        relative = float(position) - float(offset)
        return relative if math.isfinite(relative) and relative >= 0 else None

    def send_processed_clock(self, now):
        """Send absolute and audio-relative clocks while a processed queue output is open."""
        po = getattr(self, "processed", None)
        if po is None or not getattr(po, "queue_id", None) or getattr(self, "link", None) is None \
                or now < getattr(self, "processed_clock_next", 0.0):
            return
        self.processed_clock_next = now + 1.0
        try:
            if self.ipc.get("path") != po.path:
                return
            raw_pos = self.ipc.get("time-pos")
            pos = self._deps._float_or(raw_pos, None)
            if isinstance(raw_pos, bool) or not isinstance(pos, (int, float)) or not math.isfinite(pos):
                return
            payload = {
                "cmd": "clock", "id": po.queue_id, "pos": round(float(pos), 3),
                "playing": self.ipc.get("pause") is False and not self.ipc.get("eof-reached", False),
            }
            subtitle_pos = self.subtitle_clock_position(pos, po)
            if subtitle_pos is not None:
                payload["subtitle_pos"] = round(subtitle_pos, 3)
            if not self.link.post(payload):
                self._deps.log(f"processed subtitle clock dropped for queue item {po.queue_id}")
        except Exception as e:  # noqa: BLE001 - a clock hint never affects processed playback
            self._deps.log(f"processed subtitle clock skipped: {e!r}")

    def processed_requests(self, sreq, qreq, creq):
        """Requests from oh_seek.lua while a processed file plays. Seeks are native in mpv; one arrives only
        when mpv has nothing loaded (a dropped link had replaced the file): reload it there."""
        if (qreq and qreq != self.last_qreq) or (creq and creq != self.last_creq):
            self.osd("已處理的影片")
            self.ipc.set("user-data/oh/request-done", qreq if qreq != self.last_qreq else creq)
        if sreq and sreq != self.last_sreq and self.ipc.get("idle-active", False):
            pos = self._deps._float_or(sreq.split(":", 1)[1])
            self.served = sreq
            self._deps.log(f"processed file reloaded at {pos:.2f}s")
            self.load_processed(self.processed, pos, sreq)
        self.last_sreq, self.last_qreq, self.last_creq = sreq, qreq, creq

    def processed_thumbnails_tick(self):
        """Start previews once mpv has loaded the finished file and knows its duration."""
        if self.processed is None or not self.args.thumbnails or self.thumbs is not None:
            return
        if self.ipc.get("path") != self.processed.path:
            return
        duration = self.ipc.get("duration")
        if not isinstance(duration, (int, float)) or not 0 < duration < float("inf"):
            return
        self.thumb_dir = self._deps.tempfile.mkdtemp(prefix="oh_thumbs_")
        self.thumbs = self._deps.ThumbnailWorker(self.processed.path, {}, duration,
                                                 self.args.thumb_width, self.thumb_dir)
        self.thumb_sent, self.prefetch_started = None, True
        self.thumbs.start()
        self.thumbs.start_prefetch(self.args.thumb_interval)

    def switch_to_progressive(self, po: Any):
        """Watch a queue item while the worker processes it: no Session / restorer / GPU, the feeder streams the
        finished segments (progressive_stream.py); loaded by progressive_tick once its first segment is there."""
        t0 = self._deps.time.perf_counter()
        if hasattr(self.opener, "cancel"):
            self.opener.cancel()                # whatever was being resolved is superseded
        self.drop_pending_open()
        self.leave_current()
        self.epoch += 1
        self.t_req = self._deps.time.perf_counter()
        self.title, self.source_text, self.cur_open = po.title, "邊處理邊看（佇列處理中）", None
        self.published_duration, self.published_list = po.duration, None
        self.progressive = self._deps.progressive_stream.ProgressivePlayback(po, send=self.link.post if self.link is not None else None,
                                                             log=self._deps.log, replies=True)
        self.ranges_shown, self.progressive_sent, self.progressive_next = None, False, 0.0
        self.served, self.loaded, self.resume_wait, self.restore_deadline = f"init{self.epoch}", False, False, None
        self.req_kind, self.last_latency, self.last_kind = "開啟", None, None
        self.ipc.set("user-data/oh/duration", po.duration)
        self.ipc.set("user-data/oh/start", po.start)
        self.ipc.set("user-data/oh/qualities", "[]")        # no renditions, no quality button
        self.publish_settings()
        self.ipc.set("force-media-title", self.title)
        self.ipc.set("title", f"{self.title} - {APP_NAME}")
        self.ipc.set("user-data/oh/active", "1")
        self.progressive.start(po.start)
        if self.args.thumbnails and po.duration > 0:          # hover previews of the source, as in live mode
            self.thumb_dir = self._deps.tempfile.mkdtemp(prefix="oh_thumbs_")
            self.thumbs = self._deps.ThumbnailWorker(getattr(po, "source_url", None) or po.path, {},
                                                     po.duration, self.args.thumb_width, self.thumb_dir)
            self.thumbs.start()
            self.prefetch_started, self.thumb_sent = False, None
        self._deps.log(f"opened progressive {po.title!r}: {po.path} (queue item {po.queue_id}, work {po.work_dir}, "
            f"{po.duration:.1f}s) | setup {self._deps.time.perf_counter() - t0:.2f}s")

    def play_progressive(self, po: Any):
        selection = None
        visibility = None
        if self.progressive is not None and self.progressive.po.queue_id and self.progressive.po.queue_id == po.queue_id \
                and self.progressive.po.work_dir == po.work_dir:
            if getattr(self.progressive.po, "subtitles", None):
                self.remember_progressive_subtitle()
                selection = getattr(self, "progressive_sub_selection", None)
                visibility = getattr(self, "progressive_sub_visibility", None)
            if getattr(po, "start_requested", False):
                pass                            # the queue asked for a start time: reopen the feeder there (like a seek)
            elif getattr(self.progressive.po, "source_url", None) == getattr(po, "source_url", None):
                self.osd("已在邊處理邊看")
                return
            else:
                # Resuming a source download replaces its local server. Reopen the
                # feeder at the current position so audio/seek stop using the old port.
                pos = self._deps._float_or(self.ipc.get("time-pos"), 0.0)
                po.start = pos if 0 <= pos <= po.duration else 0.0
        try:
            self.switch_to_progressive(po)
            self.progressive_sub_selection = selection
            self.progressive_sub_visibility = visibility
            self.apply_pending_subtitle_update(getattr(po, "queue_id", None))
            if getattr(po, "subtitle_error", False):
                self.osd("字幕產生失敗", 6000)
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"could not play progressive {po.path}: {e!r}")
            self.show_error(self._deps.short_error(e))
            self.end_source()
            self.ipc.set("user-data/oh/active", "0")

    def progressive_internal_urls(self):
        """Keep owned source URLs out of Lua's dropped-file intake."""
        p = self.progressive
        urls = [p.po.source_url or p.po.path, self.progressive_active_url]
        if p.feeder is not None:
            urls.append(p.feeder.url)
        if self.progressive_clip is not None:
            urls.append(self.progressive_clip.url)
        self.ipc.set("user-data/oh/internal-urls", self._deps.json.dumps(
            list(dict.fromkeys(u for u in urls if u))))

    def progressive_open_source(self, pos, *, internal=False):
        """Load the original at the requested source time without waiting for restoration."""
        p = self.progressive
        url = (p.po.source_url or p.po.path) if internal else p.start_source(pos)
        self.progressive_source = True
        self.progressive_active_url = url
        self.progressive_sub_loaded = None
        self.loaded, self.resume_wait, self.restore_deadline = True, True, None
        self.set_cache_wait(self.args.seek_prebuffer)
        self.ipc.set("user-data/oh/passthrough", "1")
        self.publish_compare_layers("", "")
        self.progressive_internal_urls()
        self.ipc.set("user-data/oh/expect-load", url)
        now = self._deps.time.perf_counter()
        self.t_load = now
        if internal:
            self.progressive_begin_handoff(url, pos, now, "source")
        else:
            self.ipc.set("user-data/oh/served", self.served)
        self.ipc.command("loadfile", url, "replace", -1, f"start={pos:.6f}")
        self._deps.log(f"progressive: 原片過渡 at {pos:.3f}s ({url})")

    def progressive_begin_handoff(self, url, pos, now, kind="restored", *, paused=None):
        token = f"prog-{self.epoch}-{self.progressive_generation}-{now:.6f}"
        handoff = dict(token=token, url=url, position=float(pos),
                       paused=self.ipc.get("pause", False) is True if paused is None else bool(paused))
        self.progressive_handoff = dict(handoff, kind=kind, generation=self.progressive_generation,
                                       deadline=now + 15.0)
        self.ipc.set("user-data/oh/handoff-resumed", "")
        self.ipc.set("user-data/oh/handoff", self._deps.json.dumps(handoff))

    def progressive_check_handoff(self, now):
        """Acknowledge either a seekable startup clip or its continuation independently of user seeks."""
        p = self.progressive
        handoff = self.progressive_handoff
        if handoff is not None:
            if handoff["generation"] != self.progressive_generation:
                return True
            published = self.ipc.get("user-data/oh/handoff", "")
            try:
                published_token = self._deps.json.loads(published).get("token") if published else None
            except (ValueError, TypeError, AttributeError):
                published_token = None
            cancelled = published_token != handoff["token"]
            path = self.ipc.get("path")
            if cancelled:
                seek = self.ipc.get("user-data/oh/seek-request")
                if seek and seek != self.last_sreq:
                    self.progressive_handoff = None
                    return True  # the newer user request is consumed later in this controller iteration
                if handoff["kind"] == "continuation" and self.progressive_clip is not None and \
                        path == self.progressive_clip.url:
                    # A native seek inside the startup clip does not invalidate its appended continuation.
                    self.progressive_begin_handoff(handoff["url"], handoff["position"], now, "continuation")
                    return True
                if path != handoff["url"]:
                    self.progressive_handoff = None
                    if p.feeder is not None:
                        p.feeder.retire()
                        p.feeder = None
                    position = self._deps._float_or(self.ipc.get("time-pos"), handoff["position"])
                    self.progressive_open_source(position, internal=True)
                    return True
            if path == handoff["url"] and (cancelled or
                    self.ipc.get("user-data/oh/handoff-resumed") == handoff["token"]):
                self.progressive_handoff = None
                self.ipc.set("user-data/oh/handoff", "")
                self.resume_wait = False
                if handoff["kind"] == "restored":
                    self.progressive_clip = p.feeder
                    p.activate_restore(handoff["position"])
                    self.progressive_source = False
                    self.ipc.set("user-data/oh/passthrough", "")
                    self._deps.log(f"progressive: 修復交接完成 at {self.ipc.get('time-pos')}s")
                elif handoff["kind"] == "continuation":
                    self.progressive_active_url = handoff["url"]
                    if self.progressive_clip is not None:
                        # The continuation belongs to playback now; only the startup file is retired here.
                        self.progressive_clip.retire(keep_continuation=True)
                        self.progressive_clip = None
                    self.progressive_internal_urls()
                return True
            if handoff["kind"] == "continuation" and self.progressive_clip is not None and \
                    path == self.progressive_clip.url and self.ipc.get("eof-reached", False) and \
                    not handoff.get("advanced"):
                # keep-open may have paused the short clip before an appended stream became playable.
                self.ipc.command("playlist-next")
                handoff["advanced"] = True
                handoff["deadline"] = now + 15.0
                return True
            if now >= handoff["deadline"]:
                if handoff["kind"] == "continuation" and self.ipc.get("pause", False):
                    handoff["deadline"] = now + 15.0
                    return True
                self.progressive_handoff = None
                self.progressive_retry_at = now + 5.0
                self.ipc.set("user-data/oh/handoff", "")
                if p.feeder is not None:
                    p.feeder.retire()
                    p.feeder = None
                if handoff["kind"] in ("restored", "continuation"):
                    self._deps.log("progressive: 修復交接逾時，回到原片")
                    position = self._deps._float_or(self.ipc.get("time-pos"), handoff["position"])
                    self.progressive_open_source(max(position, handoff["position"]), internal=True)
                    if self.progressive_clip is not None:
                        self.progressive_clip.retire()
                        self.progressive_clip = None
                else:
                    self.resume_wait = False
                    self.osd("原片載入失敗，請重新跳轉或檢查下載狀態", 6000)
                return True
            return True
        return False

    def progressive_continuation_tick(self, now):
        """Append live packets at the exact end of the seekable startup clip."""
        clip = self.progressive_clip
        if clip is None or self.progressive_handoff is not None:
            return
        f = self.progressive.feeder
        if f is None:  # the startup clip already covers the natural end
            return
        if f.error is not None:
            position = self._deps._float_or(self.ipc.get("time-pos"), clip.end_s)
            f.retire()
            self.progressive.feeder = None
            self.progressive_open_source(position, internal=True)
            clip.retire()
            self.progressive_clip = None
            return
        if not f.ready.is_set():
            return
        self.progressive_internal_urls()
        self.ipc.set("user-data/oh/expect-load", f.url)
        at_eof = self.ipc.get("eof-reached", False)
        self.progressive_begin_handoff(f.url, clip.end_s, now, "continuation",
                                       paused=self.ipc.get("pause", False) and not at_eof)
        self.ipc.command("loadfile", f.url, "append")
        # Pause/keep-open at the end of the clip must not strand an appended continuation.
        if at_eof:
            self.ipc.command("playlist-next")
            self.progressive_handoff["advanced"] = True

    def progressive_source_tick(self, now):
        """Prepare repaired video around the current cursor while the original keeps advancing."""
        p = self.progressive
        if self.progressive_check_handoff(now):
            return
        if not self.loaded or self.resume_wait or self.ipc.get("path") != self.progressive_active_url:
            if self.loaded and self.resume_wait and now - self.t_load >= 15.0:
                self.resume_wait = False
                self.progressive_retry_at = now + 5.0
                self.osd("原片載入逾時，請重新跳轉或檢查下載狀態", 6000)
            return
        pos = self.ipc.get("time-pos")
        if not isinstance(pos, (int, float)) or isinstance(pos, bool) or not math.isfinite(pos):
            return
        f = p.feeder
        # A native seek can move to another cached region without a Python seek request.
        span = p.restore_range(pos)
        if f is not None and (f.error is not None or span is None or pos < f.target or pos >= span[1] or (
                f.prepared.is_set() and pos >= f.end_s) or (
                self.progressive_candidate_at is not None and now - self.progressive_candidate_at > 10.0)):
            f.retire()
            p.feeder = None
            self.progressive_candidate_at = None
            self.progressive_retry_at = now + 2.0
            return
        if self.ipc.get("pause", False) or self.ipc.get("eof-reached", False) or now < self.progressive_retry_at:
            return
        if f is None:
            if span is not None:
                try:
                    p.prepare_restore(pos)
                    self.progressive_candidate_at = now
                    self.progressive_internal_urls()
                except Exception as e:  # optional prefetch must not close a player that is already playing the source
                    self.progressive_retry_at = now + 5.0
                    self._deps.log(f"progressive: candidate creation failed; source continues ({e!r})")
            return
        if not f.prepared.is_set():
            return
        # Probe completion can race a seek or a manifest redo. Re-read the current cursor before replacing.
        pos = self.ipc.get("time-pos")
        if not isinstance(pos, (int, float)) or p.restore_range(pos) is None or pos < f.first_s or pos >= f.end_s:
            return
        self.remember_progressive_subtitle()
        self.progressive_active_url = f.url
        self.progressive_sub_loaded = None
        self.ipc.set("user-data/oh/expect-load", f.url)
        self.publish_compare_layers(f.compare_titles(), f.derive_1x())
        self.progressive_begin_handoff(f.url, pos, now)
        self.set_cache_wait(self.args.seek_prebuffer)
        self.ipc.command("loadfile", f.url, "replace", -1, f"start={pos:.6f}")
        self.resume_wait, self.t_load = True, now

    def progressive_requests(self, sreq, qreq, creq):
        """Requests from oh_seek.lua in progressive mode. A seek outside mpv's cache restarts the feeder at the
        target (frozen like a live restart; oh_seek.lua restores the user's pause state with the load); there is
        no quality / restart-kind setting to apply."""
        if (qreq and qreq != self.last_qreq) or (creq and creq != self.last_creq):
            self.osd("邊處理邊看：無法切換")
            self.ipc.set("user-data/oh/request-done", qreq if qreq != self.last_qreq else creq)
        if sreq and sreq != self.last_sreq:
            self.remember_progressive_subtitle()
            pos = self._deps._float_or(sreq.split(":", 1)[1])
            self.progressive_generation += 1
            self.progressive_handoff = None
            self.progressive_retry_at = 0.0
            self.progressive_candidate_at = None
            if self.progressive_clip is not None:
                self.progressive_clip.retire()
                self.progressive_clip = None
            self.ipc.set("user-data/oh/handoff", "")
            self.t_req, self.served, self.req_kind = self._deps.time.perf_counter(), sreq, "seek"
            self._deps.log(f"seek request #{self._deps.req_counter(sreq)} -> {pos:.2f}s (progressive)")
            from oh_streamer.player import subtitle_timing
            subtitle_timing.log_seek(self, pos)
            if not self.progressive.segment_available(pos):
                try:
                    self.progressive_open_source(pos)
                except ValueError as e:
                    self.show_error(str(e))
                    self.ipc.set("user-data/oh/request-done", sreq)
            else:
                if self.ipc.get("pause") is False and not self.ipc.get("eof-reached", False):
                    self.ipc.set("pause", True)
                self.progressive_source = False
                self.ipc.set("user-data/oh/passthrough", "")
                self.progressive.start(pos)
                self.loaded, self.resume_wait, self.restore_deadline = False, False, None
        self.last_sreq, self.last_qreq, self.last_creq = sreq, qreq, creq

    def remember_progressive_subtitle(self):
        try:
            sid = self.ipc.get("sid")
            visible = self.ipc.get("sub-visibility")
            if isinstance(visible, bool):
                self.progressive_sub_visibility = visible
            if sid in (False, "no"):
                # Empty/invalid growing SRTs can make sub-add fail. With no selectable subtitle track,
                # sid=no means nothing has loaded yet, not that the user disabled subtitles. Preserve
                # the default/previous choice so the first later publication with cues is selected.
                if any(t.get("type") == "sub" for t in (self.ipc.get("track-list", []) or [])):
                    self.progressive_sub_selection = False
            else:
                selected = next((t for t in self.ipc.get("track-list", [])
                                 if t.get("type") == "sub" and t.get("id") == sid), None)
                if selected:
                    selection = (selected.get("title"), selected.get("lang"))
                    # mpv may expose an external track's title as its filename even when sub-add was
                    # given a queue title. Map the selected file back to queue metadata so a later
                    # feeder reload or changed SRT path can still restore the same language/track.
                    ext = selected.get("external-filename")
                    po = self.subtitle_owner()
                    source = (getattr(po, "source_url", None) or getattr(po, "path", None)) if po is not None else None
                    if getattr(self, "progressive", None) is not None and selected.get("ff-index") is not None and (
                            not selected.get("external") or (isinstance(ext, str) and isinstance(source, str) and
                            self._deps.os.path.normcase(self._deps.os.path.abspath(ext)) ==
                            self._deps.os.path.normcase(self._deps.os.path.abspath(source)))):
                        # Source tracks have no queue SRT identity; retain their demux stream index separately.
                        selection = ("source", selected["ff-index"], selected.get("title"), selected.get("lang"))
                    if isinstance(ext, str) and po is not None:
                        ext_key = self._deps.os.path.normcase(self._deps.os.path.abspath(ext))
                        offset = getattr(po, "subtitle_offset", 0.0)
                        for index, track in enumerate(getattr(po, "subtitles", ()) or ()):
                            try:
                                path = track["path"]
                                if offset:
                                    if self.progressive_sub_dir is None:
                                        self.progressive_sub_dir = self._deps.tempfile.mkdtemp(
                                            prefix="oh_progressive_subs_")
                                    path = self._deps.progressive_stream.shifted_srt(
                                        path, offset, self.progressive_sub_dir, index)
                                path_key = self._deps.os.path.normcase(self._deps.os.path.abspath(path))
                                if path_key == ext_key:
                                    selection = (track.get("title"), track.get("lang"))
                                    break
                            except Exception as e:  # noqa: BLE001 - queue subtitles are optional
                                self._deps.log(f"progressive subtitle metadata lookup skipped: {e!r}")
                    self.progressive_sub_selection = selection
        except Exception as e:  # noqa: BLE001
            self._deps.log(f"progressive subtitle selection skipped: {e!r}")

    def cast_subtitle_fallback(self, plan):
        from oh_streamer.platform import cast_subtitles
        return cast_subtitles.owner_subtitles(
            self.subtitle_owner(),
            self.ipc,
            plan,
            selection=getattr(self, "progressive_sub_selection", None),
            visibility=getattr(self, "progressive_sub_visibility", None),
        )

    def subtitle_owner(self):
        """The object whose `subtitles` (and offset / coverage) the player shows: the progressive item, the processed
        file, or the live source's stream item (StreamSubtitles); None when nothing has queue subtitles."""
        progressive = getattr(self, "progressive", None)
        if progressive is not None:
            return progressive.po
        processed = getattr(self, "processed", None)
        return processed if processed is not None else getattr(self, "stream_subs", None)

    def stream_subtitles_before_load(self):
        """Live mode, right before the loop hands a (re)started segment to mpv: mpv drops external tracks with the old
        file, so note which track the user chose and whether subtitles are visible (restored by load_progressive_subtitles
        once the new file plays), and forget that the old file had its tracks. Only when this file really got our tracks:
        otherwise sid=no (no track was there to select) would be mistaken for a user's choice."""
        subs = self.stream_subs
        if subs is None:
            return
        try:
            loaded = self.progressive_sub_loaded
            if subs.subtitles and loaded and self.ipc.get("path") == loaded:
                self.remember_progressive_subtitle()
        except Exception as e:  # noqa: BLE001 - optional subtitle state must not disturb playback
            self._deps.log(f"stream subtitle selection read skipped: {e!r}")
        self.progressive_sub_loaded = None

    def stream_subtitle_tick(self):
        """Live mode with a stream item: add the item's subtitle tracks once per mpv file load (the segment
        loadfile of every start, seek, quality or settings restart replaces the file and drops them)."""
        subs = self.stream_subs
        if subs is None or not subs.subtitles or self.session is None or not self.loaded or not self.session.url \
                or self.progressive_sub_loaded == self.session.url or self.cast.active is not None:
            return
        try:
            if self.ipc.get("path") != self.session.url or not self.ipc.get("track-list", []):
                return
            self.load_progressive_subtitles(self.session.url, subs)
        except Exception as e:  # noqa: BLE001 - subtitles are optional, never fatal to video
            self._deps.log(f"stream subtitle load skipped: {e!r}")
            self.progressive_sub_loaded = self.session.url         # (no retry every loop iteration)

    def current_subtitle_status(self):
        """The queue's subtitle job status for the video that plays now (None for other items / none received)."""
        status, po = getattr(self, "subtitle_status", None), self.subtitle_owner()
        if status is None or po is None or str(getattr(po, "queue_id", "")) != status.item_id:
            return None
        return status

    def apply_subtitle_status(self, status):
        """Keep the newest status of the playing item for the 進階資訊 panel; a failure is also shown once as an OSD.
        A status for an item that is not open yet (it can arrive just before the open is consumed) waits."""
        po = self.subtitle_owner()
        if po is None or str(getattr(po, "queue_id", "")) != status.item_id:
            pending = getattr(self, "pending_subtitle_status", None)
            if pending is not None:
                pending[status.item_id] = status
                while len(pending) > 16:
                    pending.pop(next(iter(pending)))
            return
        self.subtitle_status = status
        self.note_gpu_short(status)
        key = subtitle_status.problem_key(status)
        if key is None:
            self.subtitle_problem_shown = None
        elif (status.item_id, *key) != getattr(self, "subtitle_problem_shown", None):
            self.subtitle_problem_shown = (status.item_id, *key)
            try:
                if self.ipc is not None:                   # (begin_source applies a waiting status before any IPC)
                    self.osd(subtitle_status.headline(status), 5000)
            except Exception as e:  # noqa: BLE001 - an OSD hint never affects playback
                self._deps.log(f"subtitle status OSD skipped: {e!r}")

    def note_gpu_short(self, status):
        """T4.3: show the 算力不足 hint once per source (self.epoch changes with every begin_source)."""
        if not status.gpu_short or self.ipc is None or getattr(self, "gpu_short_shown", None) == self.epoch:
            return
        self.gpu_short_shown = self.epoch
        try:
            self.osd(subtitle_status.GPU_SHORT_OSD, 6000)
        except Exception as e:  # noqa: BLE001 - an OSD hint never affects playback
            self._deps.log(f"gpu_short OSD skipped: {e!r}")

    def apply_pending_subtitle_update(self, item_id):
        if not item_id:
            return
        status = getattr(self, "pending_subtitle_status", {}).pop(str(item_id), None)
        if status is not None:
            self.apply_subtitle_status(status)
        update = getattr(self, "pending_subtitle_updates", {}).pop(str(item_id), None)
        if update is not None:
            self.apply_subtitle_update(update)

    def apply_subtitle_update(self, update):
        """Apply one queue snapshot to its active processed/progressive item without touching other playback."""
        progressive = getattr(self, "progressive", None)
        processed = getattr(self, "processed", None)
        stream = getattr(self, "stream_subs", None)
        po = self.subtitle_owner()
        if po is None or str(getattr(po, "queue_id", "")) != update.item_id:
            pending = getattr(self, "pending_subtitle_updates", None)
            if pending is not None:
                pending[update.item_id] = update
                while len(pending) > 16:
                    pending.pop(next(iter(pending)))
            return

        old_tracks = list(getattr(po, "subtitles", ()) or ())
        old_offset = getattr(po, "subtitle_offset", 0.0)
        path = None
        try:
            path = self.ipc.get("path") if self.ipc is not None else None
            if path and getattr(self, "progressive_sub_loaded", None) == path:
                self.remember_progressive_subtitle()
        except Exception as e:  # noqa: BLE001 - optional subtitle state must not disturb playback
            self._deps.log(f"subtitle update selection read skipped: {e!r}")

        po.subtitles = list(update.subtitles)
        po.subtitle_offset = update.subtitle_offset
        po.subtitle_covered_s = update.covered_s
        po.subtitle_covered_ranges = getattr(update, "covered_ranges", None)
        po.subtitle_complete = update.complete
        if getattr(self, "subtitle_gap_item_id", None) != update.item_id:
            self.subtitle_gap_item_id, self.subtitle_gap_warned = update.item_id, False
        current_path = None
        if progressive is not None and po is progressive.po:
            feeder = getattr(progressive, "feeder", None)
            current_path = getattr(self, "progressive_active_url", None) or (
                getattr(feeder, "url", None) if feeder is not None else None)
        elif processed is po:
            current_path = getattr(po, "path", None)
        elif stream is po and self.session is not None and self.loaded:
            current_path = getattr(self.session, "url", None)   # live: the segment mpv plays now (a restart re-adds the tracks)
        try:
            if current_path and path == current_path and self.ipc.get("track-list", []):
                self.load_progressive_subtitles(current_path, po, force_reload=True,
                                                previous_tracks=old_tracks, previous_offset=old_offset)
        except Exception as e:  # noqa: BLE001 - subtitle failures are logged, never fatal to video
            self._deps.log(f"progressive subtitle update skipped: {e!r}")
        cast = getattr(self, "cast", None)
        if cast is not None and getattr(cast, "active", None) is not None:
            try:
                plan = cast.active.get("plan")
                if plan is not None:
                    subs = self.cast_subtitle_fallback(plan)
                    if subs is not None:
                        cast.update_subtitles(subs)
            except Exception as e:  # noqa: BLE001 - cast subtitle update is optional
                self._deps.log(f"cast subtitle update skipped: {e!r}")
        self._deps.log(f"subtitle update applied to queue item {update.item_id} through {update.covered_s:.3f}s"
                       f" ({'complete' if update.complete else 'partial'})")

    def subtitle_gap_tick(self):
        """Show one OSD for each uncovered playback interval; reset once subtitles catch the viewer."""
        po = self.subtitle_owner()
        item_id = getattr(po, "queue_id", None) if po is not None else None
        if not item_id:
            self.subtitle_gap_item_id, self.subtitle_gap_warned = None, False
            return
        item_id = str(item_id)
        if getattr(self, "subtitle_gap_item_id", None) != item_id:
            self.subtitle_gap_item_id, self.subtitle_gap_warned = item_id, False
        covered = getattr(po, "subtitle_covered_s", None)
        ranges = getattr(po, "subtitle_covered_ranges", None)
        if getattr(po, "subtitle_complete", True) or (ranges is None and (
                not isinstance(covered, (int, float)) or isinstance(covered, bool) or not math.isfinite(covered))):
            self.subtitle_gap_warned = False
            return
        offset = getattr(po, "subtitle_offset", 0.0)
        if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
            offset = 0.0
        if self.ipc is None:
            return
        try:
            position = self.ipc.get("time-pos")
        except Exception as e:  # noqa: BLE001 - subtitle monitoring must not affect playback
            self._deps.log(f"subtitle coverage check skipped: {e!r}")
            return
        if not isinstance(position, (int, float)) or isinstance(position, bool) or not math.isfinite(position):
            return
        covered_here = (any(a <= position - offset <= b for a, b in ranges) if ranges is not None
                        else position <= covered + offset)
        if covered_here:
            self.subtitle_gap_warned = False
        elif not self.subtitle_gap_warned:
            self.subtitle_gap_warned = True
            try:
                self.osd("此處字幕尚未產生", 4000)
            except Exception as e:  # noqa: BLE001 - OSD is an optional subtitle hint
                self._deps.log(f"subtitle coverage OSD skipped: {e!r}")

    def load_progressive_subtitles(self, url, po=None, *, force_reload=False, previous_tracks=None,
                                   previous_offset=None):
        """Once per mpv file load (seeks replace the file and drop its external tracks)."""
        if getattr(self, "progressive_sub_loaded", None) == url and not force_reload:
            return
        self.progressive_sub_loaded = url
        po = po or self.progressive.po
        tracks = getattr(po, "subtitles", ())
        concise = next((t for t in tracks if "精簡" in t.get("title", "")
                        and t.get("lang", "").lower() in ("chi", "zho", "zh")), None)
        target_paths = set()
        for index, track in enumerate(tracks):
            try:
                target = track["path"]
                if po.subtitle_offset:
                    if self.progressive_sub_dir is None:
                        self.progressive_sub_dir = self._deps.tempfile.mkdtemp(prefix="oh_progressive_subs_")
                    target = self._deps.progressive_stream.shifted_srt(
                        target, po.subtitle_offset, self.progressive_sub_dir, index)
                target_paths.add(self._deps.os.path.normcase(self._deps.os.path.abspath(target)))
            except Exception as e:  # noqa: BLE001 - a failed optional subtitle never stops video playback
                self._deps.log(f"progressive subtitle path check skipped: {e!r}")
        if force_reload:
            old_offset = previous_offset if previous_offset is not None else po.subtitle_offset
            disabled_selection = getattr(self, "progressive_sub_selection", None) is False
            seen_old = set()
            for index, track in enumerate(previous_tracks or ()):
                try:
                    old_path = track["path"]
                    if old_offset:
                        if self.progressive_sub_dir is None:
                            self.progressive_sub_dir = self._deps.tempfile.mkdtemp(prefix="oh_progressive_subs_")
                        old_path = self._deps.progressive_stream.shifted_srt(
                            old_path, old_offset, self.progressive_sub_dir, index)
                    old_key = self._deps.os.path.normcase(self._deps.os.path.abspath(old_path))
                    # sub-reload can make mpv select the track again even after sid=no. When subtitles are
                    # disabled, remove/re-add with auto flags instead; this keeps the tracks fresh but unselected.
                    if (old_key in target_paths and not disabled_selection) or old_key in seen_old:
                        continue
                    for existing in self.ipc.get("track-list", []) or []:
                        ext = existing.get("external-filename")
                        if existing.get("type") == "sub" and isinstance(ext, str) \
                                and self._deps.os.path.normcase(self._deps.os.path.abspath(ext)) == old_key:
                            self.ipc.command("sub-remove", existing["id"])
                    seen_old.add(old_key)
                except Exception as e:  # noqa: BLE001 - an old external track is optional
                    self._deps.log(f"progressive subtitle replacement skipped: {e!r}")
        for index, track in enumerate(tracks):
            try:
                path = track["path"]
                if po.subtitle_offset:
                    if self.progressive_sub_dir is None:
                        self.progressive_sub_dir = self._deps.tempfile.mkdtemp(prefix="oh_progressive_subs_")
                    path = self._deps.progressive_stream.shifted_srt(
                        path, po.subtitle_offset, self.progressive_sub_dir, index)
                path_key = self._deps.os.path.normcase(self._deps.os.path.abspath(path))
                existing = self.ipc.get("track-list", []) or []
                loaded = next((t for t in existing if t.get("type") == "sub"
                               and isinstance(t.get("external-filename"), str)
                               and self._deps.os.path.normcase(self._deps.os.path.abspath(
                                   t["external-filename"])) == path_key), None)
                if loaded is not None:
                    if force_reload:
                        self.ipc.command("sub-reload", loaded.get("id"))
                    continue
                if any(t.get("type") == "sub" and not t.get("external")
                       and t.get("title") == track["title"] for t in existing):
                    continue
                selection = getattr(self, "progressive_sub_selection", None)
                select = ((track == concise if concise else track["default"]) if selection is None
                          else selection == (track["title"], track["lang"]))
                self.ipc.command("sub-add", path, "select" if select else "auto", track["title"], track["lang"])
            except Exception as e:  # noqa: BLE001 - a missing/unreadable subtitle never stops the stream
                self._deps.log(f"progressive subtitle skipped: {e!r}")
        selection = getattr(self, "progressive_sub_selection", None)
        if isinstance(selection, tuple) and len(selection) == 4 and selection[0] == "source":
            try:
                source = getattr(po, "source_url", None) or po.path
                source_key = self._deps.os.path.normcase(self._deps.os.path.abspath(source))

                def source_track(track):
                    external = track.get("external-filename")
                    return isinstance(external, str) and self._deps.os.path.normcase(
                        self._deps.os.path.abspath(external)) == source_key

                current = self.ipc.get("track-list", []) or []
                if url != source and not any(t.get("type") == "sub" and source_track(t) for t in current):
                    self.ipc.command("sub-add", source, "auto")
                    current = self.ipc.get("track-list", []) or []
                    # mpv can expose the container's video alongside its external subtitles.
                    for track in current:
                        if track.get("type") == "video" and track.get("external") and source_track(track):
                            self.ipc.command("video-remove", track["id"])
                    current = self.ipc.get("track-list", []) or []
                selected = next((t for t in current if t.get("type") == "sub" and
                                 t.get("ff-index") == selection[1] and (url == source or source_track(t))), None)
                if selected is not None:
                    self.ipc.set("sid", selected["id"])
            except Exception as e:  # source subtitle formats and demuxers are optional
                self._deps.log(f"progressive source subtitle restore skipped: {e!r}")
        if force_reload and isinstance(selection, tuple):
            try:
                selected = next((track for track in tracks
                                 if (track.get("title"), track.get("lang")) == selection), None)
                if selected is not None:
                    selected_path = selected["path"]
                    selected_index = next(index for index, track in enumerate(tracks) if track is selected)
                    if po.subtitle_offset:
                        if self.progressive_sub_dir is None:
                            self.progressive_sub_dir = self._deps.tempfile.mkdtemp(prefix="oh_progressive_subs_")
                        selected_path = self._deps.progressive_stream.shifted_srt(
                            selected_path, po.subtitle_offset, self.progressive_sub_dir, selected_index)
                    selected_key = self._deps.os.path.normcase(self._deps.os.path.abspath(selected_path))
                    current_tracks = self.ipc.get("track-list", []) or []
                    loaded = next((track for track in current_tracks if track.get("type") == "sub"
                                   and isinstance(track.get("external-filename"), str)
                                   and self._deps.os.path.normcase(self._deps.os.path.abspath(
                                       track["external-filename"])) == selected_key), None)
                    if loaded is not None and self.ipc.get("sid") != loaded.get("id"):
                        # mpv's sub-reload can replace tracks with new ids and switch to another track.
                        self.ipc.set("sid", loaded.get("id"))
            except Exception as e:  # noqa: BLE001 - restoring an optional track must not affect video playback
                self._deps.log(f"progressive subtitle selection restore skipped: {e!r}")
        if getattr(self, "progressive_sub_selection", None) is False:
            try:
                self.ipc.set("sid", "no")
            except Exception as e:  # noqa: BLE001
                self._deps.log(f"progressive subtitle selection skipped: {e!r}")
        visibility = getattr(self, "progressive_sub_visibility", None)
        # sid=no is sufficient to disable subtitles. Do not force sub-visibility off here: the
        # user can later choose a track from mpv's track menu without toggling visibility, and a
        # forced false value would keep that newly selected track invisible. Preserve an explicit
        # visibility choice only when we have observed one from the user.
        if isinstance(visibility, bool):
            try:
                self.ipc.set("sub-visibility", visibility)
            except Exception as e:  # noqa: BLE001
                self._deps.log(f"progressive subtitle visibility skipped: {e!r}")
        from oh_streamer.player import subtitle_timing
        subtitle_timing.log_load(self, url, po, force_reload)

    def cast_source(self, what):
        """Reuse the queue's restoration output; menu actions need no source probe."""
        if self.progressive is not None:
            if self.cast.active is not None:
                return self.cast.active["source"]
            if what == "hls" or what.startswith("dev:"):
                from oh_streamer.player.progressive_cast import ProgressiveCastSource
                source = ProgressiveCastSource(self.progressive, encoder=self.engine.encoder())
                self.progressive_generation += 1
                self.progressive_handoff = None
                self.ipc.set("user-data/oh/handoff", "")
                if self.progressive_source:
                    if self.progressive.feeder is not None:
                        self.progressive.feeder.retire()
                        self.progressive.feeder = None
                if self.progressive_clip is not None:
                    self.progressive_clip.retire()
                    self.progressive_clip = None
                return source
            return None
        return self.session if self.session is not None else self.processed

    def progressive_tick(self, now):
        """Controller iteration in progressive mode: queue messages (watching / want), loading the stream once the
        feeder is ready, its resume, the processed ranges and "watched"."""
        self.progressive.tick()
        if self.cast.active is not None:
            # Keep the queue's files protected after the local feeder has stopped.
            # Only receiver position (not transcoding progress) paces interpolation.
            a = self.cast.active
            if now >= self.progressive_next:
                self.progressive_next = now + 0.5
                r = self.progressive.ranges_text()
                if r != self.ranges_shown:
                    self.ipc.set("user-data/oh/ranges", r)
                    self.ranges_shown = r
                if a["mode"] == "airplay" and a["phase"] == "playing":
                    tp = a["local_pos"]
                    self.progressive.clock(
                        tp, self.ipc.get("user-data/oh/cast-paused") != "1",
                        subtitle_pos=self.subtitle_clock_position(tp, self.progressive.po))
                    if not self.progressive_sent and self._deps.processed_watched(tp, self.published_duration, False):
                        self.progressive_sent = True
                        if self.progressive.po.queue_id and self.link is not None:
                            self.link.post({"cmd": "watched", "id": self.progressive.po.queue_id})
            return
        f = self.progressive.feeder
        if f is not None and not self.progressive_source and not self.loaded and f.ready.is_set():
            self.set_cache_wait(self.args.seek_prebuffer)    # whole segments arrive at once: no reason to wait for more
            self.ipc.set("user-data/oh/served", self.served)
            self.ipc.set("user-data/oh/expect-load", f.url)   # oh_seek.lua: our load, not a drop
            self.progressive_active_url = f.url
            self.progressive_internal_urls()
            self.publish_compare_layers(f.compare_titles(), f.derive_1x())   # oh:* tracks of the segments
            self.ipc.command("loadfile", f.url, "replace")
            self.loaded, self.resume_wait, self.t_load = True, True, now
            self._deps.log(f"loadfile {f.url} ({now - self.t_req:.2f}s after request, stream from {f.first_s:.3f}s)")
        elif self.resume_wait and self.progressive_handoff is None and \
                self.ipc.get("path") == self.progressive_active_url and \
                self.ipc.get("user-data/oh/resumed") == self.served:
            self.resume_wait = False
            self.last_latency, self.last_kind = now - self.t_req, self.req_kind
            self._deps.log(f"playback resumed at {self.ipc.get('time-pos')}s (requested {self.progressive.target:.3f}s) "
                f"{now - self.t_req:.2f}s after request")
            po = self.progressive.po
            if po.pressed_at is not None:           # T1.3: once per press (a seek reloads the stream without it)
                self._deps.log(f"first playable {po.queue_id}: 按播放→可播 {self._deps.time.time() - po.pressed_at:.2f} s "
                    f"（播放器 request→可播 {now - self.t_req:.2f} s）")
                po.pressed_at = None
            if self.thumbs is not None and not self.prefetch_started:
                self.prefetch_started = True
                self.thumbs.start_prefetch(self.args.thumb_interval)
        if self.progressive_source:
            self.progressive_source_tick(now)
        else:
            self.progressive_check_handoff(now)
        if not self.progressive_source:
            self.progressive_continuation_tick(now)
        active_url = self.progressive_active_url or (f.url if f is not None else None)
        if self.loaded and self.ipc.get("path") == active_url \
                and self.ipc.get("track-list", []):
            self.load_progressive_subtitles(active_url)
        if now >= self.progressive_next:
            self.progressive_next = now + 0.5
            r = self.progressive.ranges_text()
            if r != self.ranges_shown:
                self.ipc.set("user-data/oh/ranges", r)
                self.ranges_shown = r
            clock_due = self.loaded and not self.resume_wait and self.progressive.clock_due()
            if self.loaded and not self.resume_wait and self.ipc.get("path") == active_url:
                self.remember_progressive_subtitle()
            if active_url is not None and self.loaded and (not self.progressive_sent or clock_due) and \
                    self.ipc.get("path") == active_url:
                tp = self.ipc.get("time-pos")
                if not self.progressive_sent and self._deps.processed_watched(
                        tp, self.published_duration, self.ipc.get("eof-reached", False) and (
                            self.progressive_source or (f is not None and active_url == f.url and f.ended) or (
                                self.progressive_clip is not None and self.progressive.feeder is None))):
                    self.progressive_sent = True
                    self._deps.log(f"progressive item watched (queue item {self.progressive.po.queue_id})")
                    if self.progressive.po.queue_id and self.link is not None:
                        self.link.post({"cmd": "watched", "id": self.progressive.po.queue_id})
                if clock_due and isinstance(tp, (int, float)):
                    # the worker's interpolation policy follows the viewer (queue -> worker stdin "clock")
                    self.progressive.clock(
                        tp, not self.ipc.get("core-idle", True) and not self.ipc.get("pause", False)
                        and not self.ipc.get("eof-reached", False),
                        subtitle_pos=self.subtitle_clock_position(tp, self.progressive.po))

    def add_to_queue(self):
        """Key 'a' / settings menu row: send the live source to the queue app with the current settings."""
        if self.processed is not None:
            self.osd("已是處理過的影片")
            return
        if self.progressive is not None:
            self.osd("已在處理佇列中")
            return
        if self.session is None or self.cur_open is None or not self.cur_open[0]:
            self.osd("沒有正在播放的影片")
            return
        cfg = (self.sc.cfg if self.sc is not None else self.engine.cfg)
        payload = self._deps.build_add_payload(
            self.cur_open[0], self.cur_open[1], self._deps.queue_quality(self.qc.qualities, self.qc.current),
            self.title, cfg, self.sc.fp16_on if self.sc is not None else bool(cfg.get("fp16")))
        if self.link is None:
            self.osd("處理佇列未開啟")
            return

        def done(reply):               # sender thread
            if reply is None:
                self.notices.put("處理佇列未開啟")
            elif reply.get("ok"):
                self.notices.put("已加入處理佇列")
            else:
                self.notices.put("加入處理佇列失敗：" + str(reply.get("error") or "未知錯誤")[:60])
        self.link.post(payload, done)
        self._deps.log(f"queue add: {payload['input'][:100]} quality={payload['quality']} settings={payload['settings']}")

    def switch_to(self, opened: Any):
        """Replace the current source by an opened (resolved + probed) one. Models stay loaded."""
        t0 = self._deps.time.perf_counter()
        self.stop_s = self.leave_current()
        if self.reload_pending:                      # file changes (settings_reload) first: overrides revert to / sit on them
            self.apply_reloaded_settings(t0)
        self.apply_overrides(getattr(opened, "overrides", None) or {})   # before the new restorer is made (begin_source)
        self.t_req = self._deps.time.perf_counter()
        self.begin_source(opened)
        self.publish_source()
        self.served, self.loaded, self.resume_wait = f"init{self.epoch}", False, False
        self.restore_deadline, self.n_req, self.t_stopped, self.t_load, self.switch_ctx = None, 0, self._deps.time.perf_counter(), None, None
        self.sw_note = f"source switch (stop old {self.stop_s:.2f}s, setup {self._deps.time.perf_counter() - t0:.2f}s) | "
        self.req_kind, self.last_latency, self.last_kind = "開啟", None, None
        self.proc_samples.clear()
        self.proc_fps_now = None                   # do not report the previous source's rate before a new sample
        self.hist.clear("ratio", "buffer")
        alloc = self._deps.device_backend.allocated_gib(getattr(self.engine, "device", None))
        self._deps.log(f"opened {opened.title!r}: {self.session.source.describe()} | {self.sw_note.rstrip(' |')} | "
            f"threads {self._deps.threading.active_count()} | torch allocated {alloc} GiB")

    def release_stream_item(self, item_id):
        """Tell the queue this stream item is no longer needed (stream_end, queue_ipc.py "Stream items"); the queue
        removes it and its folder. Fire-and-forget: queued on the sender thread, never blocks; the last ones of a
        closing player are flushed by link.close() (bounded wait). Any thread."""
        if not item_id or getattr(self, "link", None) is None:
            return
        try:
            if self.link.post({"cmd": "stream_end", "id": item_id}) is False:
                self._deps.log("stream subtitles: stream_end dropped (queue backlog full; the queue times the item out)")
            else:
                self._deps.log(f"stream subtitles: stream_end (stream item {item_id})")
        except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
            self._deps.log(f"stream subtitles: stream_end failed ({e!r})")

    def drop_pending_open(self):
        """Forget a resolved source that waits for the models and will never play (another source replaced it)."""
        opened, self.pending_open = self.pending_open, None
        self.release_stream_item(getattr(opened, "stream_item", None))

    def live_report(self):
        """The live-stream report that rides on player_busy (queue_ipc.py, "Live stream report"): what the queue's
        「串流中」 card shows. None unless a live-mode source is open (not a processed file / progressive playback) and
        has not reached its end. Called by the BusyReporter every LIVE_INTERVAL s; every mpv read is guarded."""
        if self.session is None or self.processed is not None or self.progressive is not None:
            return None

        def num(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
        try:
            ready = self.loaded and not self.resume_wait and self.cast.active is None
            if ready and self.ipc.get("eof-reached"):
                return None
            pos = num(self.ipc.get("time-pos")) if ready else None
            cache = num(self.ipc.get("demuxer-cache-time")) if ready else None
            cache_wait = bool(ready and self.ipc.get("paused-for-cache", False))
            playing = bool(ready and not cache_wait and not self.ipc.get("pause", False))
            done = bool(self.loaded and self.session.processing_done())
            fps, proc = num(self.session.fps), num(self.proc_fps_now)
            report = {
                "title": str(self.title or ""), "pos": pos, "duration": num(self.published_duration) or None,
                "playing": playing, "buffering": (not ready) or cache_wait,
                "ahead_s": round(max(0.0, cache - pos), 1) if pos is not None and cache is not None else None,
                "speed": round(proc / fps, 2) if proc is not None and fps and not done else None,
                "restore_done": done, "stream_item": self.stream_item,
            }
            paused = self.restore_pause.report(self.session) if getattr(self, "restore_pause", None) is not None else None
            if paused is not None:
                report["restore_pause"] = paused
            return report
        except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
            self._deps.log(f"live report skipped: {e!r}")
            return None

    def restore_probe(self):
        """(restored lead ahead_s, mpv is playing) for RestorePause's buffer_low check; read only while a pause lasts."""
        if not self.loaded or self.resume_wait or self.cast.active is not None:
            return None, False
        pos, cache = self.ipc.get("time-pos"), self.ipc.get("demuxer-cache-time")
        playing = not self.ipc.get("pause", False) and not self.ipc.get("paused-for-cache", False) \
            and not self.ipc.get("eof-reached", False)
        if isinstance(pos, bool) or isinstance(cache, bool) or not isinstance(pos, (int, float)) \
                or not isinstance(cache, (int, float)):
            return None, playing
        return max(0.0, cache - pos), playing

    def stream_link_tick(self, now):
        """Live mode with a queue stream item: the `watching` heartbeat, and the playback position (`clock`) once the
        stream plays. Only the subtitle scheduling uses them (queue_ipc.py, "Stream items"); player_busy is unchanged."""
        link = self.stream_link
        if link is None:
            return
        link.tick(now)                    # also while casting: the queue drops a stream item whose watching stops (60 s)

        if self.cast is not None and self.cast.active is not None:
            a = self.cast.active
            vod = a.get("vod")
            try:
                restarts = getattr(vod, "restarts", None) if vod is not None else None
                if isinstance(restarts, int) and restarts > getattr(self, "_cast_stream_restarts", 0):
                    self._cast_stream_restarts = restarts
                    plan = a.get("plan")
                    anchor = getattr(vod, "anchor", None) if vod is not None else None
                    if plan is not None and isinstance(anchor, int):
                        jump_pos = float(a.get("v0", 0.0)) + float(plan.start(anchor))
                        if anchor != getattr(self, "_cast_stream_want_k", None):
                            self._cast_stream_want_k = anchor
                            link.want(jump_pos, now)
                if link.clock_due(now):
                    mode = a.get("mode")
                    if mode == "airplay" and a.get("phase") == "playing":
                        tp = a.get("local_pos")
                        if isinstance(tp, (int, float)) and not isinstance(tp, bool):
                            cast_paused = self.ipc.get("user-data/oh/cast-paused") if self.ipc is not None else None
                            playing = cast_paused != "1"
                            link.clock(float(tp), playing, now)
                    elif mode == "hls" and vod is not None:
                        last_req = getattr(vod, "last_request", None)
                        if isinstance(last_req, dict) and "pos" in last_req:
                            pos = last_req["pos"]
                            if isinstance(pos, (int, float)) and not isinstance(pos, bool):
                                tp = float(a.get("v0", 0.0)) + float(pos)
                                link.clock(tp, True, now)
            except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
                self._deps.log(f"cast stream subtitle link skipped: {e!r}")
            return

        if self.session is None or not self.loaded or self.resume_wait or not link.clock_due(now):
            return
        try:
            if self.ipc.get("path") != self.session.url:
                return
            tp = self.ipc.get("time-pos")
            if isinstance(tp, bool) or not isinstance(tp, (int, float)):
                return
            link.clock(tp, not self.ipc.get("core-idle", True) and not self.ipc.get("pause", False)
                       and not self.ipc.get("eof-reached", False), now)
        except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
            self._deps.log(f"stream subtitle clock skipped: {e!r}")

    def submit_text(self, text: str, headers: dict, label: str, start: float = 0.0, quality=None, overrides=None,
                    stream_item=None):
        try:
            oi = self._deps.open_input.parse(text)
        except self._deps.open_input.OpenInputError as e:
            self._deps.log(f"open ({label}): {e}")
            self.show_error(str(e))
            return
        if headers and oi.kind == "url":
            oi.headers = {**oi.headers, **headers}
        self._deps.log(f"open ({label}): {oi.target[:120]} [{oi.describe()}]" + (f" notes: {oi.notes}" if oi.notes else "")
                       + (f" quality={quality}" if quality is not None else "")
                       + (f" overrides={overrides}" if overrides else ""))
        extra = {k: v for k, v in (("quality", quality), ("overrides", overrides), ("stream_item", stream_item)) if v}
        self.opener.submit(oi, label, start, **extra)   # (plain opens keep the three-argument call)
