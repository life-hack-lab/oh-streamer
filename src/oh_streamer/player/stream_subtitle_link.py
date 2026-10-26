"""
What the live-mode player tells the queue about a stream item (queue_ipc.py, "Stream items"): the open message of a
stream with subtitles carries "stream_item": <id>, and while that source plays the player sends the same `watching` /
`clock` / `want` messages as progressive playback, only for the queue's subtitle scheduling. Stdlib only.

`send(payload)` must not block (QueueLink.post). Times are the caller's own monotonic clock (the controller loop's);
this class never reads a clock, so the throttles are testable.

`StreamSubtitles` (T3.2) is the other direction: the queue's growing subtitle tracks of that stream item, held by the
controller for the live source so load_progressive_subtitles can add them to every live segment mpv loads.
"""
import math

WATCHING_INTERVAL = 10.0     # = queue_ipc.BUSY_HEARTBEAT (not imported: same reason as progressive_stream)
CLOCK_INTERVAL = 1.0         # = progressive_stream.CLOCK_INTERVAL


def _position(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


class StreamSubtitleLink:
    def __init__(self, item_id: str, send, log=lambda m: None, watching_interval: float = WATCHING_INTERVAL,
                 clock_interval: float = CLOCK_INTERVAL):
        self.item_id, self._send, self._log = str(item_id), send, log
        self.watching_interval, self.clock_interval = watching_interval, clock_interval
        self._next_watching = 0.0
        self._next_clock = 0.0

    def tick(self, now: float) -> bool:
        """The `watching` heartbeat; True when it was sent."""
        if now < self._next_watching:
            return False
        self._next_watching = now + self.watching_interval
        self._post({"cmd": "watching", "id": self.item_id})
        return True

    def clock_due(self, now: float) -> bool:
        """True when clock() would send (the controller reads mpv's position only then)."""
        return now >= self._next_clock

    def clock(self, pos, playing: bool, now: float) -> bool:
        """The viewer's position (absolute source seconds, mpv time-pos); at most every clock_interval s."""
        position = _position(pos)
        if position is None or not self.clock_due(now):
            return False
        self._next_clock = now + self.clock_interval
        self._post({"cmd": "clock", "id": self.item_id, "pos": round(position, 3), "playing": bool(playing)})
        return True

    def want(self, pos, now: float) -> bool:
        """The user jumped to `pos` (absolute source seconds): the subtitle scheduling should work there first.
        No "k": a stream item has no repair segments. The next clock follows at once so the queue learns the
        playing state without waiting out the throttle."""
        position = _position(pos)
        if position is None:
            return False
        self._next_clock = 0.0
        self._log(f"stream subtitles: want {position:.2f}s (stream item {self.item_id})")
        self._post({"cmd": "want", "id": self.item_id, "pos": round(position, 3), "foreground": True})
        return True

    def _post(self, payload: dict) -> None:
        try:
            if self._send(payload) is False:
                self._log(f"stream subtitles: {payload['cmd']} dropped (queue backlog full)")
        except Exception as e:  # noqa: BLE001 - a hint to the queue never affects playback
            self._log(f"stream subtitles: {payload['cmd']} failed ({e!r})")


class StreamSubtitles:
    """The external subtitle snapshot of a live source's stream item (queue -> player `subtitle_update`).

    Carries the fields load_progressive_subtitles / apply_subtitle_update / subtitle_gap_tick read from a
    ProgressiveOpen or ProcessedOpen, so those functions serve live mode unchanged. `queue_id` is the stream item id.
    Until the first snapshot arrives `subtitles` is empty and `subtitle_covered_s` is None (no gap OSD, no tracks)."""
    is_stream = True
    source_url = None
    path = None

    def __init__(self, item_id: str):
        self.queue_id = str(item_id)
        self.subtitles: list = []
        self.subtitle_offset = 0.0
        self.subtitle_covered_s = None
        self.subtitle_covered_ranges = None
        self.subtitle_complete = False
