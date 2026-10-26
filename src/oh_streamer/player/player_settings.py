"""
Advanced player settings for oh_player: defaults, the persisted JSON file in the platform user-data directory, CLI
overrides, and the settings menu published to oh_seek.lua (user-data/oh/settings).

Precedence: built-in defaults < saved file < flags given explicitly on the command line. The file only stores what
the user picked in the menu, so a one-off CLI flag never leaks into it.

Pure Python (no torch / lada imports) so it can be unit-tested quickly: see test_settings_info.py.
"""

import json
import os
import sys
from pathlib import Path

from oh_streamer.platform import platform_support

DEFAULT_FILE = platform_support.data_dir() / "oh_settings.json"

DEFAULTS = {
    "max_clip_length": 240,
    "detection_model": "v4-fast",
    "restoration_model": "basicvsrpp-v1.2",
    "fp16": None,              # None = auto (Lada's gpu_has_fp16_acceleration)
    "seek_prebuffer": 1.0,
    "encoder": "auto",         # auto = h264_nvenc if usable, else libx264
    "vsr": "on",               # NVIDIA RTX Video Super Resolution in mpv (oh_seek.lua: d3d11vpp scaling-mode=nvidia)
    "smooth_playback": "on",   # mpv display-resample + interpolation (see smooth_props)
    "restoration_fps": "original",  # sample before detection/restoration
    "interpolation": "off",    # auto = adaptive RIFE 2x frame interpolation in the pipeline (rife_interp.py)
    "compare": "off",          # on = also encode the original (unrestored) frames as extra tracks
    "engine": "lada",          # restoration engine (engines/); an unknown / unavailable name falls back to lada when used
}
KEYS = tuple(DEFAULTS)

# player state that is remembered too, but is not a menu setting or a command-line flag
UI_DEFAULTS = {"volume": 100.0, "mute": False, "cast_vsr": "off", "cast_vsr_size": "2x"}
UI_KEYS = tuple(UI_DEFAULTS)

CLIP_OPTIONS = [30, 60, 90, 120, 180, 240, 300, 360]
PREBUFFER_OPTIONS = [0.5, 1.0, 2.0, 3.0]
ENCODERS = ["auto", "h264_nvenc", "h264_amf", "h264_videotoolbox", "libx264"]
ON_OFF = ["on", "off"]                     # vsr, smooth_playback (strings: they are passed on to mpv as such)
RESTORATION_FPS = ["original", "30"]
INTERPOLATION = ["off", "auto"]
COMPARE = ["off", "on"]

RELOAD_KEYS = frozenset({"detection_model", "restoration_model", "fp16"})   # models must be reloaded
LIVE_KEYS = frozenset({"seek_prebuffer", "vsr", "smooth_playback"})          # no restart needed (next seek / mpv)
RESTART_KEYS = frozenset(KEYS) - RELOAD_KEYS - LIVE_KEYS                     # restart the segment at the position

RECOMMENDED = {"v4-fast", "basicvsrpp-v1.2"}
_DETECTION_NAMES = {"v4-fast": "V4 快速", "v4-accurate": "V4 精準"}


def model_label(name: str) -> str:
    """Display name of a model: the V4 detectors by speed class, other "vN..." names with a capital V."""
    if name in _DETECTION_NAMES:
        return _DETECTION_NAMES[name]
    return "V" + name[1:] if len(name) > 1 and name[0] == "v" and name[1].isdigit() else name

# settings an "open" request may override for that source only (never written to the settings file)
OVERRIDE_KEYS = ("detection_model", "restoration_model", "interpolation", "restoration_fps")


# ---------------------------------------------------------------------------------------------------------------
# validation / file
# ---------------------------------------------------------------------------------------------------------------

def validate(key: str, v):
    """Typed, sane value for `key` or ValueError."""
    if key == "max_clip_length":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not 1 <= v <= 10000:
            raise ValueError(f"bad {key}: {v!r}")
        return int(v)
    if key in ("detection_model", "restoration_model", "engine"):
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"bad {key}: {v!r}")
        return v.strip()
    if key == "fp16":
        if v is not None and not isinstance(v, bool):
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "seek_prebuffer":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 60:
            raise ValueError(f"bad {key}: {v!r}")
        return float(v)
    if key == "encoder":
        if v not in ENCODERS:
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key in ("vsr", "smooth_playback", "cast_vsr"):
        if v not in ON_OFF:
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "cast_vsr_size":
        if v not in ("2x", "4k"):
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "restoration_fps":
        if v not in RESTORATION_FPS:
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "interpolation":
        if v not in INTERPOLATION:
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "compare":
        if v not in COMPARE:
            raise ValueError(f"bad {key}: {v!r}")
        return v
    if key == "volume":
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
            raise ValueError(f"bad {key}: {v!r}")
        return float(v)
    if key == "mute":
        if not isinstance(v, bool):
            raise ValueError(f"bad {key}: {v!r}")
        return v
    raise ValueError(f"unknown setting {key!r}")


def _join_warnings(*warnings) -> str | None:
    present = [warning for warning in warnings if warning]
    return " | ".join(present) if present else None


def load_file(path) -> tuple[dict, str | None]:
    """(valid saved settings, warning or None). Missing file = ({}, None); corrupt file / bad values are dropped."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as e:
        return {}, _join_warnings(f"ignoring unreadable settings file {path}: {e}")
    if not isinstance(raw, dict):
        return {}, _join_warnings(f"ignoring settings file {path}: not a JSON object")
    out, bad = {}, []
    for k, v in raw.items():
        if k not in DEFAULTS and k not in UI_DEFAULTS:
            continue
        try:
            out[k] = validate(k, v)
        except ValueError:
            bad.append(k)
    invalid_warning = f"ignoring invalid values in {path}: {', '.join(bad)}" if bad else None
    return out, invalid_warning


def save_file(path, data: dict):
    """Atomic write (a crash never leaves a half-written file). Raises OSError on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: validate(k, data[k]) for k in KEYS + UI_KEYS if k in data}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def cli_overrides(args) -> dict:
    """Settings given explicitly on the command line (argparse defaults for these flags are None)."""
    return {k: validate(k, getattr(args, k)) for k in KEYS if getattr(args, k, None) is not None}


def effective(saved: dict, cli: dict) -> dict:
    """The settings in use (volume / mute are not part of it: see ui_state)."""
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}, **cli}


def fix_platform(cfg: dict, platform=None) -> tuple[dict, list[str]]:
    """Runtime gates leave the saved file intact when settings travel between OSes."""
    platform = sys.platform if platform is None else platform
    cfg, notes = dict(cfg), []
    if platform != "win32" and cfg.get("encoder") == "h264_amf":
        notes.append("AMD AMF is unavailable on this platform; using libx264")
        cfg["encoder"] = "libx264"
    if platform == "darwin":
        for key in ("vsr", "interpolation"):
            if cfg.get(key, "off") != "off":
                notes.append(f"{key} is unavailable in the initial macOS port; using off")
                cfg[key] = "off"
        if cfg.get("encoder") == "h264_nvenc":
            notes.append("NVENC is unavailable on macOS; using the platform auto encoder")
            cfg["encoder"] = "auto"
    return cfg, notes


def ui_state(saved: dict) -> dict:
    return {**UI_DEFAULTS, **{k: v for k, v in saved.items() if k in UI_DEFAULTS}}


def fix_models(cfg: dict, cli: dict, det_names: list[str], res_names: list[str]) -> tuple[dict, list[str]]:
    """Replace saved / default model names that are not installed (CLI names are left alone: the caller errors).

    Returns (cfg, notes)."""
    cfg, notes = dict(cfg), []
    for key, names in (("detection_model", det_names), ("restoration_model", res_names)):
        if cfg[key] in names or key in cli or not names:
            continue
        repl = DEFAULTS[key] if DEFAULTS[key] in names else names[0]
        notes.append(f"{key} {cfg[key]!r} is not installed, using {repl!r}")
        cfg[key] = repl
    return cfg, notes


def clean_overrides(raw) -> tuple[dict, list[str]]:
    """(typed per-source overrides, notes about the entries left out) from the "overrides" of an open request.

    Only OVERRIDE_KEYS with a valid value are kept; whether the player can use a value (model installed, interpolation
    available) is checked against the menu when it is applied."""
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return {}, [f"overrides ignored: not an object ({type(raw).__name__})"]
    out, notes = {}, []
    for k, v in raw.items():
        if k not in OVERRIDE_KEYS:
            notes.append(f"override {k!r} ignored: not an overridable setting")
            continue
        try:
            out[k] = validate(k, v)
        except ValueError as e:
            notes.append(f"override ignored: {e}")
    return out, notes


def updated_file(saved: dict, changes: dict) -> dict:
    """What the file holds after the user changed `changes` in the menu."""
    return {**saved, **{k: validate(k, v) for k, v in changes.items()}}


def smooth_props(value: str) -> dict:
    """mpv properties of smooth_playback ("on" / "off"): playback paced to the display refresh (frames shown for an
    even number of vsyncs, speed nudged by <1 %, audio resampled to match) with the oversample tscale blending only the
    frames that straddle a vsync. "off" is mpv's default timing (no interpolation)."""
    if value == "on":
        return {"video-sync": "display-resample", "interpolation": "yes", "tscale": "oversample"}
    return {"video-sync": "audio", "interpolation": "no"}


# ---------------------------------------------------------------------------------------------------------------
# menu (published to oh_seek.lua) and requests coming back from it
# ---------------------------------------------------------------------------------------------------------------

def value_str(key: str, v) -> str:
    """Canonical string form of a value (what the menu and the requests use)."""
    if key == "fp16":
        return "on" if v else "off"
    if key == "seek_prebuffer":
        return f"{float(v):g}"
    return str(v)


def _num_options(values, current, fmt):
    vals = sorted(set(values) | {current})
    return [{"v": f"{x:g}", "label": fmt(x)} for x in vals]


def build_menu(cfg: dict, fp16_on: bool, det_models: list[str], res_models: list[str], nvenc_ok: bool,
               auto_encoder: str, *, videotoolbox_ok: bool = False, amf_ok: bool = False, platform=None,
               rife_ok: bool = True, overridden=(), engines=()) -> list[dict]:
    """The settings list: [{key, label, hint, kind (reload|restart|live), value, value_label, options:[{v,label}]}].

    overridden: keys whose value is a per-source override (shown in the hint; cfg holds the effective values).
    engines: [(name, label)] of the user-visible restoration engines; the 修復引擎 item exists only for more than one."""

    def model_opts(names, cur):
        names = list(names) + ([cur] if cur not in names else [])
        return [{"v": n, "label": n + ("（建議）" if n in RECOMMENDED else "")} for n in names]

    platform = sys.platform if platform is None else platform
    enc = [{"v": "auto", "label": f"自動（{auto_encoder}）"}]
    if platform != "darwin" and (nvenc_ok or cfg["encoder"] == "h264_nvenc"):
        enc.append({"v": "h264_nvenc", "label": "NVENC（h264_nvenc）"})
    if platform == "win32" and (amf_ok or cfg["encoder"] == "h264_amf"):
        enc.append({"v": "h264_amf", "label": "AMD AMF（H.264）"})
    if platform == "darwin" and (videotoolbox_ok or cfg["encoder"] == "h264_videotoolbox"):
        enc.append({"v": "h264_videotoolbox", "label": "Apple VideoToolbox（H.264）"})
    enc.append({"v": "libx264", "label": "x264（CPU）"})

    items = [
        {"key": "max_clip_length", "label": "修復時間窗", "kind": "restart",
         "hint": "較短：seek 較快、VRAM 較少，但時間穩定性較低",
         "options": _num_options(CLIP_OPTIONS, cfg["max_clip_length"], lambda x: f"{x:g} 幀")},
        {"key": "detection_model", "label": "偵測模型", "kind": "reload", "hint": "變更後會重新載入模型",
         "options": model_opts(det_models, cfg["detection_model"])},
        {"key": "restoration_model", "label": "修復模型", "kind": "reload", "hint": "變更後會重新載入模型",
         "options": model_opts(res_models, cfg["restoration_model"])},
        {"key": "fp16", "label": "FP16", "kind": "reload", "hint": "半精度：較快、較省 VRAM；變更後會重新載入模型",
         "options": [{"v": "on", "label": "開"}, {"v": "off", "label": "關"}]},
        {"key": "seek_prebuffer", "label": "Seek 預緩衝", "kind": "live",
         "hint": "seek 後開始播放前的緩衝秒數（下次 seek 生效）",
         "options": _num_options(PREBUFFER_OPTIONS, cfg["seek_prebuffer"], lambda x: f"{x:g} 秒")},
        {"key": "encoder", "label": "編碼器", "kind": "restart", "hint": "輸出串流的編碼器；x264 使用 CPU",
         "options": enc},
        {"key": "vsr", "label": "RTX 超解析度", "kind": "live",
         "hint": "NVIDIA RTX Video Super Resolution，放大到螢幕解析度（需在 NVIDIA App 開啟 RTX 影片增強）",
         "options": [{"v": "on", "label": "開"}, {"v": "off", "label": "關"}]},
        {"key": "smooth_playback", "label": "平滑播放", "kind": "live",
         "hint": "依螢幕更新率重新取樣播放節奏，減少頓挫（非補幀）",
         "options": [{"v": "on", "label": "開"}, {"v": "off", "label": "關"}]},
        {"key": "restoration_fps", "label": "修復幀率", "kind": "restart",
         "hint": "在偵測與修復前降幀以減少 GPU 負擔；60→30、59.94→29.97 fps，搭配自動插幀可補回 2 倍幀率",
         "options": [{"v": "original", "label": "原始"}, {"v": "30", "label": "最高 30fps"}]},
        {"key": "interpolation", "label": "插幀（RIFE）", "kind": "restart",
         "hint": "GPU 有餘裕時以 RIFE 補幀到 2 倍幀率；跟不上時改為重複幀，不會卡頓（修復幀率超過 40 fps 時不補幀）",
         "options": [{"v": "off", "label": "關"}, {"v": "auto", "label": "自動"}]},
        {"key": "compare", "label": "對照原始畫面", "kind": "restart",
         "hint": "額外編碼一條原始畫面（多佔用 NVENC / VRAM）；按住 o 對照，w 分割線",
         "options": [{"v": "off", "label": "關"}, {"v": "on", "label": "開"}]},
    ]
    if len(engines) > 1:
        items.insert(0, {"key": "engine", "label": "修復引擎", "kind": "restart",
                         "hint": "下次啟動播放器時才會換用此引擎；佇列也沿用此設定",
                         "options": [{"v": n, "label": label} for n, label in engines]})
    for it in items:
        if it["key"] == "vsr" and platform != "win32":
            it["options"] = [{"v": "off", "label": "不可用（需要 Windows／RTX）"}]
            it["hint"] = "此平台不支援 RTX 超解析度"
        if it["key"] == "interpolation" and (platform == "darwin" or not rife_ok):
            it["options"] = [{"v": "off", "label": "不可用（需要已驗證的 CUDA 後端）"}]
            it["hint"] = "macOS 首版暫不支援 RIFE 插幀"
        if it["key"] in overridden:
            it["hint"] = "此影片暫用的設定，不影響預設；在這裡變更才會存成預設"
        cur = fp16_on if it["key"] == "fp16" else cfg[it["key"]]
        it["value"] = value_str(it["key"], cur)
        it["value_label"] = next((o["label"] for o in it["options"] if o["v"] == it["value"]), it["value"])
    return items


def parse_value(key: str, s: str, menu: list[dict]):
    """Typed value of the option string `s` of setting `key`; only options offered in the menu are accepted."""
    item = next((it for it in menu if it["key"] == key), None)
    if item is None or not any(o["v"] == s for o in item["options"]):
        raise ValueError(f"{key}={s!r} is not offered")
    if key == "fp16":
        return s == "on"
    if key == "max_clip_length":
        return int(float(s))
    if key == "seek_prebuffer":
        return float(s)
    return s


def parse_request(req, menu: list[dict]) -> dict:
    """'<counter>:<json {key: option string}>' from oh_seek.lua -> {key: typed value}; invalid entries dropped."""
    if not isinstance(req, str) or ":" not in req:
        return {}
    try:
        wanted = json.loads(req.split(":", 1)[1])
    except ValueError:
        return {}
    if not isinstance(wanted, dict):
        return {}
    out = {}
    for k, s in wanted.items():
        try:
            out[k] = parse_value(k, str(s), menu)
        except ValueError:
            pass
    return out


def diff(cfg: dict, fp16_on: bool, wanted: dict) -> dict:
    """The entries of `wanted` that differ from what is in use (fp16 compares against the resolved value)."""
    out = {}
    for k, v in wanted.items():
        cur = fp16_on if k == "fp16" else cfg.get(k)
        if v != cur:
            out[k] = v
    return out
