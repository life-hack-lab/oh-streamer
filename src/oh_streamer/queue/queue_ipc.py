"""
Messages between the player (oh_player.py --player) and the queue app (queue_app.py). Stdlib only, no torch.
Both sides reuse single_instance: the queue app is the single instance named "queue-<name>" (<name> = the player
instance name, "default" unless --instance / OH_STREAMER_PLAYER_INSTANCE), so tests with their own instance name never talk
to the user's queue. Every message is one JSON object {"cmd": ..., ...}; the reply is a JSON object.

player -> queue (send_to_queue):
  {"cmd": "add", "input": str, "headers": {..}, "quality": int|str|None, "title": str|None, "settings": {..}}
      -> {"ok": true, "id": str}
  {"cmd": "player_busy", "busy": bool, "pid": int}   live restore running / stopped. While busy the player repeats it
      every BUSY_HEARTBEAT s; the queue treats a busy flag older than BUSY_EXPIRE s as idle (player crashed).
      Live stream report (T3.5 / T4.1; optional "live" key, absent from older players): while the live-mode player
      has a source open (restore running or done, with or without a subtitle stream item) it adds
        "live": {"title": str, "pos": float|None, "duration": float|None, "playing": bool, "buffering": bool,
                 "ahead_s": float|None, "speed": float|None, "restore_done": bool, "stream_item": str|None}
      sent with player_busy every LIVE_INTERVAL s, also while busy=false (restore finished, video still playing).
        pos       mpv time-pos in absolute source seconds (None until the source loads / while casting);
        duration  the source length in seconds (None = unknown);
        ahead_s   restored video that is not played yet = demuxer-cache-time - pos, >= 0 (None = unknown);
        speed     restore speed as a multiple of real time = processed fps / source fps (None = not measured yet or
                  restore_done); the lead + speed pair is what T4.1's GpuHost.restore_info() will read;
        stream_item  the subtitle stream item's id, None for a stream without subtitles.
        restore_pause  (T4.2, optional) {"state": "running"|"pausing"|"paused", "turn": str|None}: the restore pause the queue
                 asked for (restore_pause below). turn = the queue's token while pausing / paused, else None. While "paused" it may carry
                 "reserved_mib": the VRAM the paused restoration still holds (torch reserved after empty_cache); the queue
                 learns it per GPU for the T4.3 VRAM check and the pre-load free-VRAM check. Whenever the
                 state changes the player sends a player_busy with the live report at once, not at the next tick.
      When the report goes away (source ended / closed / replaced / eof) the player sends one last player_busy with
      "live": null; a message without the "live" key leaves the stored report untouched (heartbeats between reports).
      The queue shows it as the 「串流中」 card (get_state()["streaming"]) and treats a report older than BUSY_EXPIRE s
      as gone (player crashed). It never changes the busy / playing semantics above.
  {"cmd": "restore_needed", "id": <stream item id>, "reason": "seek"|"buffer_low"|"timeout"}   (T4.2) the player wants
      the GPU back while the restore is paused: a segment restart (seek / quality / settings switch; sent before the new
      segment starts restoring), the restored lead ahead_s fell below 30 s while mpv plays, or the pause lasted over 900 s.
      Fire-and-forget. The queue stops that stream item's local subtitle turn and answers restore_resume. The player
      waits at most 3 s for it, then resumes anyway (the picture wins).
      Queue side: the handler answers {"ok": true} at once (unknown id -> {"ok": false, "error": "unknown id"}); the
      cancel flags are set before the reply, the model processes are killed and restore_resume is sent from a short thread.
      The cut slice goes back to pending. A new pause request waits 10 s (60 s after reason "timeout").
  {"cmd": "player_playing", "playing": bool, "pid": int}   mpv is playing anything (also a processed file: VSR and
      decoding use the GPU). Heartbeat like player_busy. Does not hold the queue back; only keeps worker throughput
      observations made meanwhile out of the performance calibration.
  {"cmd": "lookup", "input": str, "quality": ..}      -> {"ok": true, "item": None | {"id", "state", "output", "progress", "engine"}}
      (engine: the restoration engine the item is made with, "lada" for items from before the setting; a done item
      plays whatever engine made it)
  {"cmd": "watched", "id": str}                       a processed file was played to (near) the end
  {"cmd": "show"}                                     bring the queue window to the front
  {"cmd": "want", "id": str, "k": int, "pos": float (optional), "foreground": bool (optional)} progressive playback
      requests segment k of this item. A foreground request is an explicit user play/seek target and becomes the
      running job (another item's worker is stopped and goes back to queued; player_busy does not hold it back while it
      is watched); a running worker of this item gets a goto. Background requests are feeder lookahead/repeats and can
      be served after current work without interrupting a subtitle slice. The user's pauses win: a paused item (or pause_all) is not
      resumed -> {"ok": false, "error": "paused"}; a failed one is not retried -> {"ok": false, "error": "failed"}.
      Otherwise {"ok": true}, or {"ok": false} (ignored) when the item's source is not downloaded yet / it is done. The
      player repeats it while it waits. Optional "pos" is the viewer's current non-negative absolute source position;
      on a seek/start want, the queue uses it to update subtitle scheduling before the next periodic clock arrives.
      Optional "foreground" is true for the user's explicit play/seek target and false for feeder lookahead/repeats;
      only a foreground want should interrupt an in-progress subtitle slice. Older players omit this field.
  {"cmd": "watching", "id": str}                      the player is showing this item progressively; repeated every
      BUSY_HEARTBEAT s, a watch older than BUSY_EXPIRE s is void. A watched item is never evicted, keeps priority over the
      other items and keeps its work/ dir (deleted once the watch expires).
  {"cmd": "clock", "id": str, "pos": float, "playing": bool}   the viewer's position in absolute source seconds (mpv
      time-pos) about every second while progressive playback runs; playing = advancing (not paused / buffering).
      Optional "subtitle_pos" is the same position relative to the opened output's first audio sample; the queue maps it
      back to the source audio clock for subtitle scheduling while forwarding unchanged "pos" to restore workers.
      Fire-and-forget. The queue forwards it to the item's running worker (stdin line {"cmd": "clock", "pos", "playing"}),
      which uses it to skip RIFE (interpolation "full") while it is only a little ahead of the viewer.
      -> {"ok": true}, or {"ok": false} when the item has no running worker (harmless).
  Stream items (live-mode streaming with subtitles; see "stream_item" of the open message below): the live player
  sends the same watching / clock / want with the stream item's id, for the subtitle scheduling only. The item is
  temporary and has no download, restore segments or worker, so the queue records the data and does nothing else
  (no goto, no download session, no seen / protected flags, never the player's "current item"):
    watching  every BUSY_HEARTBEAT s while the stream item's source is open (refreshes the watch lease).
    clock     {"id", "pos", "playing"} about once a second once the stream plays (same throttle as progressive):
              pos = mpv time-pos in absolute source seconds (the container time the subtitle slices are measured in;
              the scheduler applies the item's audio_offset, so there is no "subtitle_pos"). Reply {"ok": true}.
    want      {"id", "pos", "foreground": true} when the user seeks (and at open when it starts past 0): pos = the target
              in absolute source seconds. "k" is omitted: there are no repair segments to request, and a queue that
              gets one ignores it. The queue sets the clock to (pos, not playing) and remembers pos as the wanted
              position (the next clock tells whether playback goes on). Reply {"ok": true}, or {"ok": false,
              "error": "bad pos"} without a valid pos.
  An id the queue does not know (the stream item is gone) stays {"ok": false, "error": "unknown id"} for all three,
  which the player ignores. None of them touches player_busy / player_playing (the live restore's GPU use).
  Lifecycle (a stream item lives for one play and is removed with its streams/<id>/ folder):
  {"cmd": "stream_end", "id": str}   the player is done with the stream item: it replaced the source by another one,
      closed, ended the source for any other reason, or never used the item (the queue had a processed output, the open
      failed or was superseded). Fire-and-forget (queued like watching; the player's last messages are flushed with a
      bounded wait on close). NOT sent when only the segment restarts (quality switch, settings change, seek): the
      item stays. -> {"ok": true}; an unknown id is {"ok": false, "error": "unknown id"} and an id that is not a
      stream item {"ok": false, "error": "not a stream item"} (nothing is touched); the player ignores both.
  Safety net for a player that died without stream_end (queue_service.STREAM_WATCH_EXPIRE / STREAM_FIRST_WATCH_GRACE):
  the queue drops a stream item whose last watching is older than 60 s, or that never got one within 120 s of its
  creation. The queue also clears every stream item when it stops; one left by a crash is cleared at the next start.

queue -> player (send_to_player, the player's existing single_instance listener):
  {"input": <output.mkv path>, "headers": {}, "processed": true, "queue_id": str, "title": str,
   "subtitle_combos_dir": <items/<id>/subtitles/combos, only when its manifest exists>}
      play an already processed file as is (no Lada restore); the lookup reply item carries the same optional key
  {"input": <source path>, "progressive": true, "queue_id": str, "work_dir": <item work dir>, "title": str}
      source downloaded, not finished: play the segments of work_dir/segments.json as the worker produces them
  Both processed and progressive messages may include "subtitles": [{"path": <existing SRT>, "lang": str,
  "title": str, "default": bool}] and "subtitle_offset": <finite seconds>. SRT zero is the extracted audio's first
  sample; offset maps it onto this source/output's absolute clock. The player shifts a temporary SRT, never sub-delay.
  They may also include "subtitle_covered_s": <finite non-negative audio-clock seconds> and "subtitle_complete": bool
  so a partial subtitle head can report its coverage as soon as playback opens. Add subtitle_offset to covered_s to
  compare it with the absolute mpv position.
  Optional "subtitle_covered_ranges": [[start,end],...] describes discontinuous audio-clock coverage.
  Embedded AI tracks suppress external tracks for processed outputs. Optional "subtitle_error": true requests the
  OSD message "字幕產生失敗" after an automatic play whose subtitle generation failed.
  {"cmd": "subtitle_update", "item_id": <same queue_id>, "srt": <cumulative SRT path or list of per-track paths>,
   "covered_s": <finite non-negative audio-clock seconds>, "complete": bool,
   "covered_ranges": [[start,end],...],  # optional; audio-clock seconds, authoritative when present
   "subtitles": [{"path": str, "lang": str, "title": str, "default": bool}],
   "subtitle_offset": <finite seconds>}
      replace the current external subtitle snapshot for the matching progressive or processed queue item. The SRT
      list and optional metadata list correspond by index. Paths contain accumulated cues, including islands;
      covered_s is the continuous prefix from zero; covered_ranges includes completed/partially translated islands;
      subtitle_offset maps that audio clock onto the absolute source/output clock.
      this is sent once per completed block, not once per cue. Invalid or failed subtitle updates are harmless to video.
  {"cmd": "subtitle_status", "item_id": <queue_id or stream item id>, "state": str, "stage": "asr"|"mt"|None,
   "executor": "local"|"cloud"|None, "done": int, "total": int, "covered_s": <audio-clock seconds>|None,
   "offset": <seconds>, "complete": bool, "wait": {"kind", "reason", "remaining_s", "executor"}|None,
   "stuck": {"kind", "stage"}|None, "elsewhere": int, "error": str, "gpu_short": true (only when true)}
      how the subtitle job of the video the player is showing is doing (stream items and progressive items the
      player reported with watching; built by subtitle_player_status.py from item.subtitle). The player shows it in
      the 進階資訊 panel (字幕 section: state, slices done/total, how far the subtitles are ahead of the playback
      position = covered_s + offset - mpv time-pos) and shows a failure (state "failed" / "stuck") once as an OSD.
      state: complete / stuck / running / waiting_quota / retrying / waiting_gpu / waiting_executor / waiting_download /
      queued (the slice-table view states of subtitle_view.py) or preparing (no slice table yet) / failed (the job
      failed; error = the reason) / paused. wait.remaining_s is a countdown snapshot, the player keeps counting.
      Sent on a change of the above (not the countdown), again every 30 s, and 5 s after a send the player did not
      accept; never per scheduler loop. Reply {"ok": true}, or {"ok": false, "error": "invalid subtitle status"}.
      gpu_short (T4.3): a stream item on the local subtitle executor whose restore lead (player_busy live ahead_s)
      fell by > 5 s over >= 60 s while the subtitle lead is under urgent_s: the player shows an OSD once per source.
      A status for an item the player has not opened (yet) waits there (at most 16); one for another item is
      never shown. Harmless to video.
  {"cmd": "restore_pause", "stream_item": <id>, "turn": str} / {"cmd": "restore_resume", "stream_item": <id>, "turn": str}
      (T4.2) hold / release the live restore so a local subtitle turn can use the GPU: restore_pause holds the detector
      (Session.pause_restore: frames already detected are restored and sent, then the VRAM cache is freed; models and
      session stay, mpv plays what is buffered), restore_resume lets it run again (no restart). Reply {"ok": true} at
      once (the controller loop applies it; the state shows in the live report's "restore_pause"), or {"ok": false,
      "error": "not this stream"} when stream_item is not the live source's stream item / no live session. The player
      remembers the turn; a resume with another turn still resumes (logged). Source end / replace / close drops it.
  {"cmd": "open", "input": <url / path>, "headers": {..}, "quality": .., "overrides": {..}, "stream_item": <id>}
      source not downloaded yet: normal live restore (the download goes on). "stream_item" (optional) is the id of the
      temporary subtitle-only stream item the queue created for this stream (see "Stream items" above); the player
      remembers it for this source only (a new open without it, or any other source, clears it) and reports to it.
  Both of the first two may carry "start": seconds (>= 0, finite; see parse_start): begin playback there (a progressive
  start in an unprocessed part behaves like a user seek there). Absent = the old behaviour.
"""
import math

from oh_streamer.player import single_instance


BUSY_HEARTBEAT = 10.0
BUSY_EXPIRE = 30.0
LIVE_INTERVAL = 2.0        # seconds between live stream reports (the 「串流中」 card position updates)


def parse_start(value, duration=None) -> float | None:
    """A start position in seconds from a JS / message value (number or numeric string): None when absent or invalid
    (not a number, NaN / inf, negative, bool), else the value clamped to `duration` when that is known (> 0)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        s = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(s) or s < 0:
        return None
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration > 0:
        s = min(s, float(duration))
    return s


def queue_instance_name(player_name: str | None = None) -> str:
    return single_instance.instance_name(f"queue-{single_instance.instance_name(player_name)}")


def send_to_queue(payload: dict, player_name: str | None = None, timeout: float = 2.0, log=lambda m: None) -> dict | None:
    """Reply dict, or None when the queue app is not running / not reachable (short timeout: never block playback)."""
    return single_instance.forward(queue_instance_name(player_name), payload, timeout=timeout, log=log)


def send_to_player(payload: dict, player_name: str | None = None, timeout: float = 20.0, log=lambda m: None) -> dict | None:
    return single_instance.forward(single_instance.instance_name(player_name), payload, timeout=timeout, log=log)
