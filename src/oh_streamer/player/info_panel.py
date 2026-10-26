"""
Info panel (進階資訊) for oh_player: builds the display-ready JSON oh_seek.lua renders (user-data/oh/info),
plus a background GPU monitor (NVML via nvidia-ml-py, falling back to polling nvidia-smi).

build_info() is pure (see test_settings_info.py); all heavy gathering happens in GpuMonitor's own thread. History
keeps one sample per second for the trend charts drawn next to some rows (row keys "h": values oldest first, -1 = no
value; "hmax": top of the scale; "href": reference line), which oh_seek.lua shows when the panel has room for them.
"""

import collections
import os
import re
import subprocess
import threading
import time
import unicodedata
from urllib.parse import urlparse

from oh_streamer.player import subtitle_status as sub_status

GOOD, WARN, BAD, IDLE = "good", "warn", "bad", "idle"


# ---------------------------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------------------------

def display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def truncate(s: str, cols: int) -> str:
    if display_width(s) <= cols:
        return s
    out, w = [], 0
    for c in s:
        cw = display_width(c)
        if w + cw > cols - 1:
            break
        out.append(c)
        w += cw
    return "".join(out) + "…"


def fmt_clock(t, tenths=False) -> str:
    if t is None:
        return "—"
    t = max(0.0, float(t))
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    sec = f"{s:04.1f}" if tenths else f"{int(s):02d}"
    return f"{h}:{m:02d}:{sec}" if h else f"{m:02d}:{sec}"


def ratio_level(ratio) -> str:
    """Processing speed vs the source frame rate: green >= 1.2x, yellow 1.0-1.2x, red < 1.0x."""
    if ratio is None:
        return IDLE
    return GOOD if ratio >= 1.2 else WARN if ratio >= 1.0 else BAD


def buffer_level(secs, good: float = 5.0, warn: float = 1.5) -> str:
    if secs is None:
        return IDLE
    return GOOD if secs >= good else WARN if secs >= warn else BAD


def diagnose(pipe, ratio, ahead, util, done=False):
    """Which stage holds playback back, from Lada's queue fill levels: (text, level).

    pipe: {"feeder": (n, max), "clips": (n, max), "out": (n, max)} = decoder->detector, clips waiting for the
    restoration model, restored frames waiting for the muxer. Only judged while it matters (processing slower than
    realtime or the buffer nearly empty); a restart after a seek refills the buffer on its own and is "正常"."""
    if done:
        return "—", IDLE
    if not pipe:
        return "—", IDLE
    if (ratio is None or ratio >= 1.0) and (ahead is None or ahead >= 1.5):
        return "正常", GOOD

    def fill(k):
        n, mx = pipe.get(k) or (0, 0)
        return n / mx if mx else 0.0
    clips_n = (pipe.get("clips") or (0, 0))[0]
    if fill("out") >= 0.9:
        return "輸出／播放端太慢", WARN
    if clips_n >= 2:
        return "GPU 修復跟不上", BAD
    if fill("feeder") <= 0.25 and (util is None or util < 70):
        return "輸入端慢（讀取／網路／解碼）", BAD
    if fill("feeder") >= 0.6 or (util is not None and util >= 85):
        return "GPU 偵測跟不上", BAD
    return "不明（輸入與 GPU 都未飽和）", WARN


def source_label(src: str | None) -> str:
    if src and src.startswith(("http://", "https://")):
        return urlparse(src).hostname or src
    return "本地檔案"


def _num(v, fmt, unit=""):
    return "—" if v is None else f"{v:{fmt}}{unit}"


# ---------------------------------------------------------------------------------------------------------------
# trend charts
# ---------------------------------------------------------------------------------------------------------------

HISTORY_LEN = 120   # samples (one per second) shown by a chart


class History:
    """Per-second samples for the trend charts: processing speed (x realtime), buffer ahead (s), GPU use (%),
    VRAM (GiB), power (W). None = no value at that time."""

    KEYS = ("ratio", "buffer", "util", "vram", "power")

    def __init__(self, n: int = HISTORY_LEN):
        self.d = {k: collections.deque(maxlen=n) for k in self.KEYS}

    def add(self, **values):
        for k, q in self.d.items():
            q.append(values.get(k))

    def clear(self, *keys):
        """Forget the samples of these series (all without keys), e.g. the playback ones when the source changes."""
        for k in keys or self.KEYS:
            n = len(self.d[k])
            self.d[k].clear()
            self.d[k].extend([None] * n)   # keep the time axis aligned with the other series

    def snapshot(self) -> dict:
        return {k: list(q) for k, q in self.d.items()}


def _chart(row: dict, values, digits: int, hmax=None, floor: float = 1.0, href=None) -> dict:
    """Attach a trend chart to a panel row (nothing if there is no value at all)."""
    vals = [v for v in values or () if v is not None]
    if not vals:
        return row
    row["h"] = [-1 if v is None else round(max(0.0, float(v)), digits) for v in values]
    row["hmax"] = round(float(hmax) if hmax else max(floor, max(vals) * 1.15), 2)
    if href is not None:
        row["href"] = href
    return row


# ---------------------------------------------------------------------------------------------------------------
# the panel content
# ---------------------------------------------------------------------------------------------------------------

def vsr_value(status) -> tuple[str, str | None]:
    """(text, color level) of oh_seek.lua's user-data/oh/vsr-status ("開 1.48x（1080→1600）", "關", "不可用（…）",
    "失敗（…）", ...)."""
    if not isinstance(status, str) or not status:
        return "—", None
    level = BAD if status.startswith("失敗") else IDLE if status.startswith(("不可用", "關")) else \
        GOOD if "x（" in status else None
    return truncate(status, 34), level


def build_info(s: dict) -> dict:
    """Display-ready panel from a raw snapshot (missing keys are shown as "—").

    Snapshot keys: title, source, width, height, fps, codec, duration, quality, pos, cache_time, proc_fps,
    proc_done, frames_total, segment_start, last_latency, last_kind, state, detection_model, restoration_model,
    max_clip_length, fp16, encoder, prebuffer, torch_alloc_gib, vsr (oh_seek.lua's status text), smooth_playback
    ("on" / "off"), interp (adaptive interpolation, see interp_rows; mode "off" = only the 設定 row), gpu (dict from
    GpuMonitor or None), cast (dict from
    oh_player.CastManager.info_snapshot() while casting, else None: adds a 投放 section), history
    (History.snapshot(): trend charts on 處理速度, 預處理緩衝, 使用率, VRAM, 功耗), subtitle_status (a
    subtitle_status.SubtitleStatus of the queue's subtitle job for this video, or None: adds a 字幕 section; the
    countdown runs on "subtitle_now", a time.monotonic() value, default now).
    Returns {"state": str, "sections": [{"title": str, "rows": [{"k": label, "v": value, "c": color level?}]}]}.
    """
    g = s.get
    hist = g("history") or {}

    def row(k, v, c=None):
        r = {"k": k, "v": v}
        if c:
            r["c"] = c
        return r

    fps = g("fps")
    w, h = g("width"), g("height")
    video = [
        row("標題", truncate(str(g("title") or "—"), 34)),
        row("來源", truncate(str(g("source") or "—"), 34)),
        row("解析度", f"{w}x{h}" if w and h else "—"),
        row("影格率", _num(fps, ".3f", " fps").replace(".000 ", " ") + (out_fps_note(g("interp")) or
                                                                         rate_note(fps, g("out_fps")))),
        row("來源編碼", str(g("codec") or "—")),
        row("長度", fmt_clock(g("duration")) if g("duration") else "—"),
        row("目前畫質", str(g("quality") or "—")),
    ]

    if g("native_fps") and fps and abs(g("native_fps") - fps) > 0.001:
        video.insert(4, row("修復幀率", f"{g('native_fps'):.3f} → {fps:.3f} fps"))
    pos, cache_time = g("pos"), g("cache_time")
    ahead = max(0.0, cache_time - pos) if pos is not None and cache_time is not None else None
    proc_fps = g("proc_fps")
    ratio = proc_fps / fps if proc_fps is not None and fps else None
    if g("proc_done"):
        speed, speed_c = "已處理到結尾", IDLE
        buf_c = GOOD if ahead is not None else IDLE
    else:
        speed = "—" if proc_fps is None else f"{proc_fps:.1f} fps · {ratio:.2f}x 即時" if ratio else f"{proc_fps:.1f} fps"
        speed_c = ratio_level(ratio)
        buf_c = buffer_level(ahead, good=max(1.5, float(g("prebuffer") or 5.0)))
    lat = g("last_latency")
    proc = [
        row("播放位置", fmt_clock(pos, tenths=True)),
        _chart(row("預處理緩衝", _num(ahead, ".1f", " 秒"), buf_c), hist.get("buffer"), 1,
               floor=max(2.0, float(g("prebuffer") or 5.0))),
        _chart(row("處理速度", speed, speed_c), hist.get("ratio"), 2, floor=2.0, href=1.0),
        row("已處理幀數", f"{int(g('frames_total') or 0):,}"),
        row("分段起點", fmt_clock(g("segment_start"), tenths=True)),
        row("上次延遲", f"{lat:.2f} 秒（{g('last_kind') or 'seek'}）" if lat is not None else "—"),
        row("狀態", str(g("state") or "—")),
    ]
    pipe = g("pipe")
    gpu_util = (g("gpu") or {}).get("util")
    diag, diag_c = diagnose(pipe, ratio, ahead, gpu_util, bool(g("proc_done")))
    proc.insert(3, row("卡頓原因", diag, diag_c))
    if pipe:
        def q(k):
            n, mx = pipe.get(k) or (0, 0)
            return f"{n} / {mx}" if mx else f"{n}"
        proc.insert(4, row("解碼佇列", q("feeder")))
        proc.insert(5, row("待修復片段", q("clips")))
        proc.insert(6, row("輸出佇列", q("out")))
    if g("progressive"):
        proc = progressive_rows(g("progressive"), pos, ahead, buffer_level(ahead), hist, g("state"))
    drops = g("drops")
    if drops is not None:
        proc.append(row("播放端掉幀", f"{drops[0]} 掉 · {drops[1]} 延遲", WARN if drops[0] else None))
    perf = g("perf")
    if perf and not g("progressive"):
        proc.append(row("效能預估", f"{perf['ratio']:.1f}x（{perf['label']}）",
                        {"live": GOOD, "prebuffer": WARN}.get(perf.get("level"), BAD)))
    it = g("interp")
    if it and it.get("mode") == "auto":
        proc.extend(interp_rows(it))
    cmp = g("compare")
    if cmp and cmp.get("mode") == "on":
        proc.extend(compare_rows(cmp))

    fp16 = g("fp16")
    settings = [
        row("偵測模型", str(g("detection_model") or "—")),
        row("修復模型", str(g("restoration_model") or "—")),
        row("修復時間窗", f"{g('max_clip_length')} 幀" if g("max_clip_length") else "—"),
        row("FP16", "—" if fp16 is None else "開" if fp16 else "關"),
        row("編碼器", str(g("encoder") or "—")),
        row("RTX VSR", *vsr_value(g("vsr"))),
        row("平滑播放", {"on": "開", "off": "關"}.get(g("smooth_playback"), "—")),
        row("插幀（RIFE）", {"auto": "自動", "off": "關"}.get((g("interp") or {}).get("mode"), "—")),
    ]

    gpu = g("gpu")
    torch_alloc = g("torch_alloc_gib")
    if not gpu:
        gpu_rows = [row("狀態", "不可用", IDLE)]
        if torch_alloc is not None:
            gpu_rows.append(row("torch 配置", f"{torch_alloc:.2f} GB"))
    else:
        used, total = gpu.get("mem_used_gib"), gpu.get("mem_total_gib")
        vram = f"{used:.2f} / {total:.1f} GB" if used is not None and total is not None else "—"
        gpu_rows = [
            row("名稱", truncate(str(gpu.get("name") or "—").replace("NVIDIA GeForce ", ""), 34)),
            _chart(row("使用率", _num(gpu.get("util"), ".0f", " %")), hist.get("util"), 0, hmax=100),
            _chart(row("VRAM", vram), hist.get("vram"), 2, hmax=total),
            row("torch 配置", _num(torch_alloc, ".2f", " GB")),
            row("溫度", _num(gpu.get("temp"), ".0f", " °C")),
            row("SM 時脈", _num(gpu.get("sm_clock"), ".0f", " MHz")),
            _chart(row("功耗", _num(gpu.get("power_w"), ".1f", " W")), hist.get("power"), 1, floor=10.0),
        ]

    sections = [
        {"title": "影片", "rows": video},
        {"title": "處理狀態", "rows": proc},
        {"title": "設定", "rows": settings},
        {"title": "GPU", "rows": gpu_rows},
    ]
    if g("cast"):
        sections.insert(1, {"title": "投放", "rows": cast_rows(g("cast"))})
    if g("subtitle_status") is not None:
        sections.insert(3 if g("cast") else 2, {"title": "字幕", "rows": subtitle_rows(
            g("subtitle_status"), pos, g("subtitle_now"))})
    return {"state": str(g("state") or ""), "sections": sections}


def subtitle_rows(st, pos, now=None) -> list[dict]:
    """The 字幕 section: what the queue's subtitle job for this video is doing (player/subtitle_status.py) and how far
    ahead of the playback position the subtitles are. ``now`` is the monotonic clock the wait countdown runs on."""
    now = time.monotonic() if now is None else now

    def row(k, v, c=None):
        r = {"k": k, "v": v}
        if c:
            r["c"] = c
        return r
    rows = [row("字幕", truncate(sub_status.state_text(st, now), 34), sub_status.level(st, pos))]
    if st.state not in sub_status.PROBLEM_STATES and st.state not in ("preparing", "paused"):
        rows.append(row("字幕領先", sub_status.lead_text(st, pos), sub_status.lead_level(st, pos)))
    if getattr(st, "gpu_short", False):
        rows.append(row("算力", "不足（修復與本機字幕搶 GPU）", "bad"))
    return rows


def out_fps_note(it) -> str:
    """"（輸出 60）" when the output runs at a multiple of the source frame rate (interpolation)."""
    if it and it.get("mode") == "auto" and it.get("eligible") and it.get("out_fps"):
        return f"（輸出 {it['out_fps']:.5g}）"
    return ""


def rate_note(fps, out_fps) -> str:
    """"（插幀輸出 59.94）" when what plays runs at a multiple of the source frame rate (a queue item with RIFE)."""
    if fps and out_fps and out_fps > fps * 1.01:
        return f"（插幀輸出 {out_fps:.5g}）"
    return ""


def progressive_rows(p: dict, pos, ahead, buf_c, hist: dict, state) -> list[dict]:
    """處理狀態 rows while watching a queue item that is still processed (progressive_stream.status: processed / total
    source seconds, speed = x realtime, ahead = processed seconds right after the playback position, done)."""
    def row(k, v, c=None):
        return {"k": k, "v": v, "c": c} if c else {"k": k, "v": v}

    total, done_s, speed, lead = p.get("total") or 0.0, p.get("processed") or 0.0, p.get("speed"), p.get("ahead")
    if p.get("done"):
        prog, speed_txt, speed_c = "全部處理完成", "—", IDLE
    else:
        prog = f"{done_s / total:.0%}（{fmt_clock(done_s)} / {fmt_clock(total)}）" if total else fmt_clock(done_s)
        if speed is None:
            speed_txt, speed_c = "測量中…", None
        elif speed < 0.01:
            speed_txt, speed_c = "沒有進度（佇列暫停或等待中？）", WARN
        else:
            speed_txt, speed_c = f"{speed:.2f}x 即時", ratio_level(speed)
    to_end = bool(total) and isinstance(pos, (int, float)) and lead is not None and pos + lead >= total - 0.05
    if p.get("done"):
        eta, eta_c = "—", None
    elif lead is not None and lead <= 0.05:
        eta, eta_c = "播放位置還沒處理，背景處理中", WARN
    elif speed is not None and speed >= 0.01:
        left = max(0.0, total - done_s) / speed
        if speed < 1.0 and lead is not None and not to_end:
            # while playing, the processed lead shrinks by (1 - speed) s per second
            eta, eta_c = f"約 {fmt_clock(lead / (1.0 - speed))} 後追上（全部 {fmt_clock(left)}）", WARN
        else:
            eta, eta_c = f"全部完成約需 {fmt_clock(left)}", GOOD
    else:
        eta, eta_c = "—", None
    lead_c = None if lead is None or p.get("done") or to_end else GOOD if lead >= 60 else WARN if lead >= 10 else BAD
    return [
        row("播放位置", fmt_clock(pos, tenths=True)),
        _chart(row("預處理緩衝", _num(ahead, ".1f", " 秒"), buf_c), hist.get("buffer"), 1, floor=2.0),
        row("背景處理", prog, IDLE if p.get("done") else None),
        _chart(row("處理速度", speed_txt, speed_c), hist.get("ratio"), 2, floor=2.0, href=1.0),
        row("已處理領先", "—" if lead is None else "到結尾" if p.get("done") or to_end else fmt_clock(lead), lead_c),
        row("預計", eta, eta_c),
        row("狀態", str(state or "—")),
    ]


def interp_rows(it: dict) -> list[dict]:
    """Rows of the adaptive interpolation (snapshot "interp": mode, eligible, why, out_fps, ratio, rife_ms, state,
    lead, start_lead, budget_ms, model, error; see oh_player.Session.interp_snapshot)."""
    def row(k, v, col=None):
        r = {"k": k, "v": v}
        if col:
            r["c"] = col
        return r
    if not it.get("eligible"):
        return [row("插幀", truncate(f"未啟用（{it.get('why') or '—'}）", 34), IDLE)]
    ratio, ms, state = it.get("ratio"), it.get("rife_ms"), it.get("state")
    rows = [row("插幀", "自動 " + ("—" if ratio is None else f"{ratio * 100:.0f}%") + "（近 10 秒補幀比例）",
                GOOD if state == "on" else IDLE)]
    lead, start = it.get("lead"), it.get("start_lead")
    if state == "on":
        what, c = "補幀中", GOOD
    elif state == "loading":
        what, c = "載入模型中", IDLE
    elif state == "budget":
        what, c = "重複幀（GPU 太忙）", WARN
    elif state == "error":
        what, c = truncate("失敗：" + str(it.get("error") or "—"), 30), BAD
    elif state == "lead" and lead is not None:
        lead = max(0.0, lead)   # negative = the clips ahead need more than the buffer
        what, c = f"重複幀（緩衝 {lead:.1f}／{start:g} 秒）" if start else f"重複幀（緩衝 {lead:.1f} 秒）", IDLE
    else:
        what, c = "等待播放", IDLE
    rows.append(row("RIFE", truncate((f"{ms:.1f} ms · " if ms is not None else "") + what, 34), c))
    return rows


def compare_rows(c: dict) -> list[dict]:
    """The compare row (snapshot "compare": mode, layers, why; see oh_player.Session.compare_snapshot)."""
    layers = c.get("layers") or []
    if len(layers) > 1:
        names = {"processed": "處理後", "original": "原始", "restored": "修復(1x)"}
        return [{"k": "對照", "v": truncate("、".join(names.get(n, n) for n in layers), 34), "c": GOOD}]
    why = c.get("why")
    return [{"k": "對照", "v": truncate(f"未啟用（{why}）" if why else "等待片段開始", 34), "c": IDLE}]


def short_url(url: str | None, cols: int = 46) -> str:
    """http://host:port/…/name (the token is long; the full URL is in the log)."""
    if not url:
        return "—"
    p = urlparse(url)
    name = p.path.rsplit("/", 1)[-1]
    return truncate(f"{p.scheme}://{p.netloc}/…/{name}", cols)


def cast_rows(c: dict) -> list[dict]:
    """Rows of the 投放 section from CastManager.info_snapshot()."""
    def row(k, v, col=None):
        r = {"k": k, "v": v}
        if col:
            r["c"] = col
        return r
    phase = {"prepare": "準備中（製作第一段）", "connecting": "連線中", "playing": "投放中",
             "hls": "區網串流"}.get(c.get("phase"), str(c.get("phase") or "—"))
    rows = [row("狀態", phase, GOOD if c.get("phase") in ("playing", "hls") else WARN)]
    if c.get("device"):
        rows.append(row("裝置", truncate(str(c["device"]), 34)))
    rows.append(row("網址", short_url(c.get("url"))))
    if c.get("tv_pos") is not None:
        rows.append(row("電視位置", fmt_clock(c["tv_pos"], tenths=True)))
    prod = c.get("prod_pos")
    if prod is None:
        prod_v, prod_c = "閒置", IDLE
    elif c.get("paused"):
        prod_v, prod_c = f"{fmt_clock(prod)}（已領先，暫停）", IDLE
    else:
        prod_v, prod_c = fmt_clock(prod, tenths=True), GOOD
    rows.append(row("製作位置", prod_v, prod_c))
    ranges = c.get("cached_ranges") or []
    rtxt = "、".join(f"{fmt_clock(a)}–{fmt_clock(b)}" for a, b in ranges[:2]) + ("…" if len(ranges) > 2 else "")
    rows.append(row("已快取", f"{int(c.get('cached_segments') or 0)} 段 · "
                             f"{(c.get('cached_bytes') or 0) / 2**20:.0f} MB"))
    rows.append(row("快取範圍", truncate(rtxt, 34) if rtxt else "—"))
    lr = c.get("last_request")
    if lr:
        rows.append(row("最後請求", f"#{lr['k']} {fmt_clock(lr.get('pos'))} · {lr.get('how', '')} · "
                                   f"{max(0.0, lr.get('age') or 0.0):.0f} 秒前"))
    else:
        rows.append(row("最後請求", "—"))
    jl = c.get("jump_latency")
    rows.append(row("跳轉延遲", f"{jl:.2f} 秒" if jl is not None else "—",
                    None if jl is None else GOOD if jl < 5 else WARN if jl < 12 else BAD))
    rows.append(row("請求 / 重啟", f"{int(c.get('requests') or 0)} / {int(c.get('restarts') or 0)}"))
    if c.get("error"):
        rows.append(row("錯誤", truncate(str(c["error"]), 34), BAD))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# GPU monitor
# ---------------------------------------------------------------------------------------------------------------

SMI_FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw,power.limit"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_smi_line(line: str) -> dict | None:
    """One line of `nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits` -> stats dict."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7 or not parts[0]:
        return None
    mem_used, mem_total = _f(parts[2]), _f(parts[3])
    return {"name": parts[0], "util": _f(parts[1]),
            "mem_used_gib": mem_used / 1024 if mem_used is not None else None,     # MiB
            "mem_total_gib": mem_total / 1024 if mem_total is not None else None,
            "temp": _f(parts[4]), "sm_clock": _f(parts[5]), "power_w": _f(parts[6]),
            "power_limit_w": _f(parts[7]) if len(parts) > 7 else None}


class WinGpuEngines:
    """GPU utilisation the way Task Manager computes it, from the Windows "GPU Engine" performance counters (PDH):
    per engine (3D, Compute, Copy, VideoDecode, VideoEncode, ...) the sum over all processes; the headline % is the
    busiest engine. NVML's utilization.gpu is unreliable on hybrid laptops (it sticks at 100 % while the dGPU idles).
    The NVIDIA adapter is the LUID whose engines include videoencode / videodecode (AMD / Intel name theirs differently).
    Create and sample it from one thread only."""

    PATH = r"\GPU Engine(*)\Utilization Percentage"
    _INST = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+_eng_(\d+)_engtype_(.+)$", re.I)

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ct, self._wt = ctypes, wintypes
        self._pdh = ctypes.WinDLL("pdh")
        self._query = wintypes.HANDLE()
        self._counter = wintypes.HANDLE()
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQuery failed")
        if self._pdh.PdhAddEnglishCounterW(self._query, self.PATH, 0, ctypes.byref(self._counter)) != 0:
            self.close()
            raise OSError("GPU Engine counters not available")
        self._luid = None
        self._pdh.PdhCollectQueryData(self._query)      # rate counters need a first sample

    def close(self):
        try:
            self._pdh.PdhCloseQuery(self._query)
        except Exception:
            pass

    def _instances(self) -> list | None:
        ct = self._ct
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return None
        size, count = self._wt.DWORD(0), self._wt.DWORD(0)
        self._pdh.PdhGetFormattedCounterArrayW(self._counter, 0x200, ct.byref(size), ct.byref(count), None)   # DOUBLE
        if not size.value:
            return None
        buf = ct.create_string_buffer(size.value)
        if self._pdh.PdhGetFormattedCounterArrayW(self._counter, 0x200, ct.byref(size), ct.byref(count), buf) != 0:
            return None

        class Item(ct.Structure):
            _fields_ = [("name", ct.c_wchar_p), ("status", ct.c_uint32), ("value", ct.c_double)]
        items = (Item * count.value).from_buffer(buf)
        return [(it.name, it.value) for it in items if it.name and it.status in (0, 0x00000001)]

    def sample(self) -> dict | None:
        """{"util", "enc_util", "dec_util"} of the NVIDIA adapter, or None when it can't be told apart."""
        rows = self._instances()
        if not rows:
            return None
        parsed = []
        for name, v in rows:
            m = self._INST.search(name)
            if m:
                parsed.append((m.group(1).lower(), int(m.group(2)), m.group(3).lower(), v))
        if self._luid is None:
            self._luid = next((l for l, _e, t, _v in parsed if t in ("videoencode", "videodecode")), None)
            if self._luid is None:
                return None
        engines: dict = {}
        for l, e, t, v in parsed:
            if l == self._luid:
                engines[(e, t)] = engines.get((e, t), 0.0) + v        # sum over processes
        if not engines:
            return None

        def busiest(kind):
            vals = [v for (_e, t), v in engines.items() if t == kind]
            return min(100.0, max(vals)) if vals else 0.0
        return {"util": min(100.0, max(engines.values())), "enc_util": busiest("videoencode"),
                "dec_util": busiest("videodecode")}


class GpuMonitor:
    """Polls GPU stats every `interval` seconds in its own thread once started: always with NVML (a few cheap calls,
    the trend charts want history), the nvidia-smi fallback (a process per poll) only while active (panel visible)."""

    def __init__(self, index: int = 0, interval: float = 1.0):
        self.index, self.interval = index, interval
        self.lock = threading.Lock()
        self.active = threading.Event()
        self.closing = threading.Event()
        self.stats: dict | None = None
        self.backend = None          # "nvml" | "nvidia-smi" | None (unavailable)
        self.thread = None
        self._nvml = self._handle = None
        self._engines = None         # WinGpuEngines (Windows): Task-Manager-style utilisation, overrides NVML's

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="gpu stats", daemon=True)
            self.thread.start()

    def set_active(self, on: bool):
        if on:
            self.start()
            self.active.set()
        else:
            self.active.clear()

    def snapshot(self) -> dict | None:
        with self.lock:
            return dict(self.stats) if self.stats else None

    def close(self):
        self.closing.set()
        self.active.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    # -- worker -------------------------------------------------------------------------------------------------

    def _init_backend(self):
        if os.name == "nt":
            try:
                self._engines = WinGpuEngines()
            except Exception:
                self._engines = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._nvml = pynvml
            self.backend = "nvml"
            return
        except Exception:
            self._nvml = None
        if self._poll_smi() is not None:
            self.backend = "nvidia-smi"

    def _poll_nvml(self) -> dict:
        n, h = self._nvml, self._handle

        def get(fn, *a):
            try:
                return fn(h, *a)
            except Exception:
                return None
        name = get(n.nvmlDeviceGetName)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        util = get(n.nvmlDeviceGetUtilizationRates)
        mem = get(n.nvmlDeviceGetMemoryInfo)
        power = get(n.nvmlDeviceGetPowerUsage)
        limit = get(n.nvmlDeviceGetEnforcedPowerLimit)    # mW, the cap the driver actually applies
        enc = get(n.nvmlDeviceGetEncoderUtilization)      # (percent, sampling period us) | None
        dec = get(n.nvmlDeviceGetDecoderUtilization)
        return {"name": name, "util": float(util.gpu) if util is not None else None,
                "enc_util": _f(enc[0]) if enc else None, "dec_util": _f(dec[0]) if dec else None,
                "mem_used_gib": mem.used / 2**30 if mem is not None else None,
                "mem_total_gib": mem.total / 2**30 if mem is not None else None,
                "temp": _f(get(n.nvmlDeviceGetTemperature, n.NVML_TEMPERATURE_GPU)),
                "sm_clock": _f(get(n.nvmlDeviceGetClockInfo, n.NVML_CLOCK_SM)),
                "power_w": power / 1000 if power is not None else None,
                "power_limit_w": limit / 1000 if limit else None}

    def _poll_smi(self) -> dict | None:
        try:
            p = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits",
                                "-i", str(self.index)], capture_output=True, text=True, timeout=5,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (OSError, subprocess.SubprocessError):
            return None
        if p.returncode != 0:
            return None
        return parse_smi_line(p.stdout.strip().splitlines()[0]) if p.stdout.strip() else None

    def _run(self):
        try:
            self._init_backend()
        except Exception:
            self.backend = None
        while not self.closing.is_set():
            if self.backend != "nvml":
                self.active.wait()
            if self.closing.is_set():
                break
            stats = None
            try:
                if self.backend == "nvml":
                    stats = self._poll_nvml()
                elif self.backend == "nvidia-smi":
                    stats = self._poll_smi()
            except Exception:
                stats = None
            if stats is not None and self._engines is not None:
                try:
                    eng = self._engines.sample()
                except Exception:
                    eng = None
                if eng:
                    stats.update(eng)
            with self.lock:
                self.stats = stats
            self.closing.wait(self.interval if self.backend == "nvml" else max(2.0, self.interval))
        if self._engines is not None:
            self._engines.close()
