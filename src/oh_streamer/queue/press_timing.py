"""T1.3 timing log: 按播放 → 第一段可播 (queue side).

play() stamps the press; the worker reports when it was started, how long its models took to load and when each
segment is done. The first finished segment after a press gives one log line (queue_service). The wall-clock press
time also travels to the player (`pressed_at`), which logs the same span up to its first playable frame.

Logging only: nothing here changes what runs.
"""

from __future__ import annotations

import time


class PressTiming:
    def __init__(self, clock=time.monotonic, wall=time.time):
        self._clock, self._wall = clock, wall
        self._pending: dict[str, dict] = {}

    def press(self, item_id: str) -> None:
        """A 播放 press starts (or restarts) the timer of this item."""
        self._pending[item_id] = {"t": self._clock(), "wall": self._wall()}

    def pressed_at(self, item_id: str) -> float | None:
        """Wall-clock time of the last press, for the player (valid across processes); None when never pressed."""
        p = self._pending.get(item_id)
        return None if p is None else p["wall"]

    def spawned(self, item_id: str) -> None:
        p = self._pending.get(item_id)
        if p is not None:
            p.setdefault("spawn", self._clock())

    def models_loaded(self, item_id: str, seconds) -> None:
        p = self._pending.get(item_id)
        if p is not None and isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds >= 0:
            p["models"] = float(seconds)

    def segment_done(self, item_id: str) -> str | None:
        """The log line for the first finished segment after a press; None for later segments and unpressed items."""
        p = self._pending.get(item_id)
        if p is None or p.get("logged"):
            return None
        p["logged"] = True
        total = self._clock() - p["t"]
        parts = [f"worker 啟動 {p['spawn'] - p['t']:.1f} s" if "spawn" in p else "worker 已在執行"]
        if "models" in p:
            parts.append(f"模型載入 {p['models']:.1f} s")
        return f"first playable {item_id}: 按播放→第一段完成 {total:.1f} s（{'，'.join(parts)}）"
