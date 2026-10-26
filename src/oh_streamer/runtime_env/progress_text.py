"""Download-progress wording for launch.pyw's base-env window, the same as setup.js's download line
(「已下載 120 MB / 260 MB（46%） · 5.2 MB/s · 剩約 30 秒」). Stdlib only: launch.pyw uses it before the base env exists.
Units are binary (GB = 2^30), like setup.js and uv's own output."""

from __future__ import annotations

INSTALLING = "正在安裝套件…"
UNMEASURED = "正在下載並安裝套件…"                                      # no measurable download (warm cache, unreadable plan)


def fmt_bytes(n: float) -> str:
    if n >= 2 ** 30:
        return f"{n / 2 ** 30:.1f} GB"
    if n >= 2 ** 20:
        return f"{round(n / 2 ** 20)} MB"
    return f"{max(1, round(n / 2 ** 10))} KB"


def fmt_pair(done: float, total: float) -> str:
    """「120 / 260 MB」: one unit for both, taken from the total."""
    if total >= 2 ** 30:
        return f"{done / 2 ** 30:.1f} / {total / 2 ** 30:.1f} GB"
    if total >= 2 ** 20:
        return f"{round(done / 2 ** 20)} / {round(total / 2 ** 20)} MB"
    return f"{fmt_bytes(done)} / {fmt_bytes(total)}"


def fmt_speed(bps: float) -> str:
    mb = bps / 2 ** 20
    return f"{mb:.1f} MB/s" if mb >= 1 else f"{max(1, round(bps / 2 ** 10))} KB/s"


def fmt_eta(s: float) -> str:
    """Coarse on purpose, like setup.js: a jittery 「剩 2 分 37 秒」 reads worse than 「2 分 40 秒」."""
    if s < 10:
        return "剩不到 10 秒"
    if s < 60:
        return f"剩約 {round(s / 5) * 5} 秒"
    if s < 3600:
        t = round(s / 10) * 10
        m, r = divmod(t, 60)
        return f"剩約 {m} 分 {r} 秒" if r else f"剩約 {m} 分"
    t = round(s / 60)
    h, m = divmod(t, 60)
    return f"剩約 {h} 小時 {m} 分" if m else f"剩約 {h} 小時"


def describe(snapshot: dict | None) -> tuple[float | None, str]:
    """(fraction 0..1 or None for a moving bar, text) for a DownloadTracker snapshot (None = nothing measurable)."""
    if not snapshot:
        return None, UNMEASURED
    total = snapshot.get("total_bytes") or 0
    done = min(snapshot.get("downloaded_bytes") or 0, total)
    if total <= 0:
        return None, UNMEASURED
    if done >= total:
        return None, f"已下載 {fmt_bytes(total)}，{INSTALLING}"
    parts = [f"已下載 {fmt_pair(done, total)}（{int(done / total * 100)}%）"]
    if snapshot.get("speed_bps") is not None:
        parts.append(fmt_speed(snapshot["speed_bps"]))
    eta = snapshot.get("eta_s")
    parts.append(fmt_eta(eta) if eta is not None else "正在估計剩餘時間…")
    return done / total, " · ".join(parts)
