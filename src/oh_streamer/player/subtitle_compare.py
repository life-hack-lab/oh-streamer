"""Subtitle-combo comparison for the player: pure logic, no mpv, no GPU.

A video can carry several generated subtitle combos (``<video>.subtitle-combos/manifest.json``, written by
the sliced scheduler's multi-track output since M6/M7). mpv renders only two
subtitle tracks natively (``sid`` + ``secondary-sid``), so to watch N combos at once this module builds one
COMPOSITE ASS file: one style per combo (own colour, own vertical band stacked bottom-up, own layer so libass never
shifts a band) and one Dialogue line per cue and combo, each prefixed with a dim 【label】.

The manifest folder is located with the same rule as ``combos.default_output_dir`` (re-implemented here so the player
does not import the subtitle package). Everything here is deterministic and only touches the filesystem in
``find_manifest`` / ``load`` / ``prepare``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "oh_streamer.subtitle_combos"
MAX_SCHEMA_VERSION = 1
FOLDER_SUFFIX = ".subtitle-combos"
COMPOSITE_TITLE = "比較模式"
COMPOSITE_FILENAME = "compare.ass"
MAX_COMBOS = 8                 # bands that still fit on screen
MAX_LABEL_CHARS = 40
PLAY_RES = (1280, 720)

# Distinct, readable on video (name, RGB). Label colour = the same hue pulled towards grey (see dim_color).
PALETTE = (
    ("白", (255, 255, 255)),
    ("黃", (255, 224, 74)),
    ("青", (92, 232, 255)),
    ("綠", (140, 255, 120)),
    ("粉紅", (255, 150, 214)),
    ("橘", (255, 168, 76)),
    ("紫", (196, 160, 255)),
    ("淺藍", (140, 190, 255)),
)

_CTRL_RE = re.compile(r"[\x00-\x1f\x7f  ]+")
_TAG_RE = re.compile(r"<[^>]*>|\{\\[^}]*\}")
_TIME_RE = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})")
_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")


class ManifestError(ValueError):
    """The manifest cannot be used (wrong format, newer schema, unreadable)."""


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ComboTrack:
    id: str
    label: str
    zh_path: Path
    ja_path: Path | None = None
    cue_count: int | None = None


@dataclass(frozen=True)
class ComboSet:
    folder: Path
    combos: tuple[ComboTrack, ...]
    video_changed: bool = False        # the manifest's size / mtime no longer match the video (audio hash is authoritative)
    skipped: tuple[str, ...] = ()      # human-readable reasons for combos that were left out


@dataclass(frozen=True)
class Band:
    """Where one combo's line sits in the composite (ASS pixels at PLAY_RES)."""
    index: int
    margin_v: int
    font_size: int
    layer: int


@dataclass
class Prepared:
    """What the player publishes to oh_sub_combos.lua."""
    composite: Path
    combos: list[dict] = field(default_factory=list)
    skipped: tuple[str, ...] = ()
    truncated: bool = False


# ---------------------------------------------------------------------------------------------- text helpers

def sanitize_label(label, fallback: str = "", max_chars: int = MAX_LABEL_CHARS) -> str:
    """A label that is safe inside ASS text, a Lua string and the OSD: no control characters, ASS override braces,
    backslashes or commas-in-fields trouble; whitespace collapsed; shortened with an ellipsis."""
    text = "" if label is None else str(label)
    text = _CTRL_RE.sub(" ", text)
    text = text.replace("{", "(").replace("}", ")").replace("\\", "／")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = _CTRL_RE.sub(" ", str(fallback or "")).strip()
        text = re.sub(r"\s+", " ", text.replace("{", "(").replace("}", ")").replace("\\", "／")).strip()
    if max_chars > 1 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def escape_ass_text(text) -> str:
    """Cue text as the text field of an ASS Dialogue line: newlines become \\N, braces (override blocks) are
    escaped, backslashes (which could form \\N / tags) become a full-width one."""
    s = "" if text is None else str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\", "＼").replace("{", "\\{").replace("}", "\\}")
    s = "\n".join(line.strip() for line in s.split("\n")).strip("\n")
    return s.replace("\n", "\\N")


def safe_style_name(combo_id: str, index: int) -> str:
    """Style / Name field value: ASCII, no commas or spaces, unique through the index."""
    base = _ID_SAFE_RE.sub("_", str(combo_id))[:32].strip("._-") or "combo"
    return f"c{index + 1}_{base}"


def format_ass_time(seconds: float) -> str:
    """H:MM:SS.cc (centiseconds, rounded; negative clamps to 0)."""
    cs = max(0, int(round(float(seconds) * 100)))
    h, rest = divmod(cs, 360000)
    m, rest = divmod(rest, 6000)
    s, c = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def ass_color(rgb: tuple[int, int, int], alpha: int = 0) -> str:
    """ASS &HAABBGGRR colour (alpha 0 = opaque)."""
    r, g, b = (max(0, min(255, int(v))) for v in rgb)
    return f"&H{max(0, min(255, alpha)):02X}{b:02X}{g:02X}{r:02X}"


def dim_color(rgb: tuple[int, int, int], amount: float = 0.45, grey: int = 150) -> tuple[int, int, int]:
    """The label colour: the combo colour blended towards mid grey (dimmer than the text, still tinted)."""
    amount = max(0.0, min(1.0, amount))
    return tuple(int(round(c * (1 - amount) + grey * amount)) for c in rgb)


def combo_color(index: int) -> tuple[str, tuple[int, int, int]]:
    return PALETTE[index % len(PALETTE)]


# ---------------------------------------------------------------------------------------------- SRT

def _srt_seconds(match: re.Match) -> float:
    h, m, s, frac = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> list[Cue]:
    """Tolerant SRT reader: BOM, CRLF/LF, missing or odd indices, '.' or ',' milliseconds, HTML/ASS tags removed.
    Blocks without a valid time line, or with empty text, or ending before they start are skipped. Sorted by start."""
    text = re.sub(r"\r+\n", "\n", (text or "").lstrip("\ufeff")).replace("\r", "\n")   # also "\r\r\n" (converted twice)
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n")]
        for i, ln in enumerate(lines):
            if "-->" not in ln:
                continue
            left, _, right = ln.partition("-->")
            a, b = _TIME_RE.search(left), _TIME_RE.search(right)
            if not a or not b:
                break
            start, end = _srt_seconds(a), _srt_seconds(b)
            body = "\n".join(x.strip() for x in lines[i + 1:])
            body = _TAG_RE.sub("", body).strip()
            if body and end > start:
                cues.append(Cue(start, end, body))
            break
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def load_srt(path: Path) -> list[Cue]:
    return parse_srt(Path(path).read_bytes().decode("utf-8-sig", errors="replace"))   # bytes: no newline translation


# ---------------------------------------------------------------------------------------------- layout

def layout_bands(count: int, play_res_y: int = PLAY_RES[1]) -> list[Band]:
    """Bands stacked bottom-up: combo 0 is the lowest. Font size shrinks with the count so everything stays inside
    ~88 % of the height; each combo has its own layer so libass does not push overlapping events around."""
    count = max(0, min(int(count), MAX_COMBOS))
    if count == 0:
        return []
    bottom = round(play_res_y * 0.035)
    usable = play_res_y * 0.88 - bottom
    size = 40 if count <= 2 else 34 if count <= 4 else 28
    line_factor = 1.5        # one line per band plus a small gap (was 2.35 for <=3 bands: too loose in the demo)
    pitch = size * line_factor
    while count * pitch > usable and size > 16:
        size -= 2
        pitch = size * line_factor
    return [Band(i, bottom + round(i * pitch), size, i) for i in range(count)]


# ---------------------------------------------------------------------------------------------- composite ASS

def build_composite_ass(tracks: list[tuple[str, str, list[Cue]]], play_res: tuple[int, int] = PLAY_RES,
                        title: str = COMPOSITE_TITLE) -> str:
    """tracks = [(combo id, label, cues)] in display order (first = lowest band). Style and Dialogue Name are
    ``safe_style_name(id, index)``; Lua relies on that to hide single bands."""
    tracks = list(tracks)[:MAX_COMBOS]
    bands = layout_bands(len(tracks), play_res[1])
    out = [
        "[Script Info]",
        f"Title: {sanitize_label(title, COMPOSITE_TITLE)}",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res[0]}",
        f"PlayResY: {play_res[1]}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
    ]
    names: list[str] = []
    for (cid, _label, _cues), band in zip(tracks, bands):
        name = safe_style_name(cid, band.index)
        names.append(name)
        rgb = combo_color(band.index)[1]
        out.append(
            f"Style: {name},Microsoft JhengHei UI,{band.font_size},{ass_color(rgb)},{ass_color(rgb)},"
            f"&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,2.4,1,2,40,40,{band.margin_v},1")
    out += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    events = []
    for (cid, label, cues), band, name in zip(tracks, bands, names):
        rgb = combo_color(band.index)[1]
        prefix = ("{\\1c&H" + ass_color(dim_color(rgb))[4:] + "&\\fs" + str(max(12, band.font_size * 2 // 3)) + "}【"
                  + escape_ass_text(sanitize_label(label, cid)) + "】{\\r}")
        for cue in cues:
            events.append((cue.start, band.index,
                           f"Dialogue: {band.layer},{format_ass_time(cue.start)},{format_ass_time(cue.end)},{name},{name},"
                           f"0,0,0,,{prefix}{escape_ass_text(cue.text)}"))
    events.sort(key=lambda e: (e[0], e[1]))
    out += [e[2] for e in events]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- manifest

def default_folder(video: Path | str) -> Path:
    """Same rule as combos.default_output_dir: <video dir>/<stem>.subtitle-combos."""
    p = Path(video)
    return p.parent / f"{p.stem}{FOLDER_SUFFIX}"


QUEUE_OUTPUT_NAME = "output.mkv"            # items/<id>/output.mkv: the queue keeps the combos in items/<id>/subtitles/combos
QUEUE_SUBFOLDER = ("subtitles", "combos")


def queue_item_folder(video: Path | str) -> Path | None:
    """The combos folder of a queue output (<item dir>/output.mkv -> <item dir>/subtitles/combos) when it holds a
    manifest, else None. Layout-only rule for players started with just the path (--processed)."""
    p = Path(video)
    if p.name.lower() != QUEUE_OUTPUT_NAME:
        return None
    folder = p.parent.joinpath(*QUEUE_SUBFOLDER)
    return folder if (folder / MANIFEST_NAME).is_file() else None


def resolve_folder(video: Path | str, folder: Path | str | None = None) -> Path:
    """Where the manifest of ``video`` lives: an explicit ``folder`` (the queue passes it for its items) when it holds
    a manifest, else the queue-item layout, else the sibling ``<stem>.subtitle-combos``. Other videos: unchanged."""
    if folder:
        explicit = Path(folder)
        if (explicit / MANIFEST_NAME).is_file():
            return explicit
    return queue_item_folder(video) or default_folder(video)


def find_manifest(video: Path | str, folder: Path | str | None = None) -> Path | None:
    path = resolve_folder(video, folder) / MANIFEST_NAME
    return path if path.is_file() else None


def parse_manifest(data, folder: Path, video: Path | str | None = None) -> ComboSet:
    """Validate a decoded manifest and keep the combos that are done (or running, M7) and whose zh SRT exists. Names in ``files`` are
    relative to the manifest folder; anything that would leave that folder is rejected."""
    if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
        raise ManifestError("not a subtitle combos manifest")
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1 or version > MAX_SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema_version {version!r}")
    raw = data.get("combos")
    if not isinstance(raw, list):
        raise ManifestError("combos is not a list")
    folder = Path(folder)
    combos, skipped, seen = [], [], set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        if cid in seen:
            skipped.append(f"{cid}: duplicate id")
            continue
        status = entry.get("status")
        if status not in ("done", "running"):          # running = the slice scheduler is still filling the SRT (M7)
            skipped.append(f"{cid}: {status}")
            continue
        files = entry.get("files") if isinstance(entry.get("files"), dict) else {}
        zh = _inside(folder, files.get("zh"))
        if zh is None or not zh.is_file():
            skipped.append(f"{cid}: zh-TW file missing")
            continue
        ja = _inside(folder, files.get("ja"))
        count = entry.get("cue_count")
        seen.add(cid)
        combos.append(ComboTrack(cid, sanitize_label(entry.get("label"), cid), zh,
                                 ja if ja is not None and ja.is_file() else None,
                                 count if isinstance(count, int) and not isinstance(count, bool) else None))
    return ComboSet(folder, tuple(combos), _video_changed(data.get("video"), video), tuple(skipped))


def _inside(folder: Path, name) -> Path | None:
    if not isinstance(name, str) or not name.strip():
        return None
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
        return None
    return folder / p


def _video_changed(info, video) -> bool:
    if video is None or not isinstance(info, dict):
        return False
    try:
        st = Path(video).stat()
    except OSError:
        return False
    size, mtime = info.get("size_bytes"), info.get("mtime_ns")
    return (isinstance(size, int) and size != st.st_size) or (isinstance(mtime, int) and mtime != st.st_mtime_ns)


def load(video: Path | str, folder: Path | str | None = None) -> ComboSet | None:
    """The usable combos next to a video, or None (no manifest, unreadable, newer schema, nothing usable). Never raises."""
    try:
        manifest = find_manifest(video, folder)
        if manifest is None:
            return None
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        found = parse_manifest(data, manifest.parent, video)
    except (OSError, ValueError):
        return None
    return found if found.combos else None


# ---------------------------------------------------------------------------------------------- growing manifests (M7)

POLL_INTERVAL_S = 5.0


def manifest_fingerprint(video: Path | str, folder: Path | str | None = None):
    """A comparable value that changes whenever the scheduler publishes progress: (updated_at, ((id, status,
    cue_count), ...)). None when there is no readable manifest. Never raises."""
    try:
        manifest = find_manifest(video, folder)
        if manifest is None:
            return None
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        raw = data.get("combos") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return None
        rows = tuple((e.get("id"), e.get("status"), e.get("cue_count")) for e in raw if isinstance(e, dict))
        return (data.get("updated_at"), rows)
    except (OSError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------------------------- player glue

def prepare(video: Path | str, work_dir: Path | str, folder: Path | str | None = None) -> Prepared | None:
    """Everything the player needs for a video: the composite ASS written to ``work_dir`` and the JSON-able combo list
    (``id, label, file, style, color, band``; ``file`` = the combo's zh-TW SRT). None when the video has no combos."""
    found = load(video, folder)
    if found is None:
        return None
    combos = list(found.combos)[:MAX_COMBOS]
    tracks, rows = [], []
    for combo in combos:
        try:
            cues = load_srt(combo.zh_path)
        except OSError:
            continue
        tracks.append((combo.id, combo.label, cues))
        index = len(rows)
        name, rgb = combo_color(index)
        rows.append({"id": combo.id, "label": combo.label, "file": str(combo.zh_path),
                     "style": safe_style_name(combo.id, index), "color": name, "rgb": ass_color(rgb)[4:]})
    if not rows:
        return None
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    composite = work / COMPOSITE_FILENAME
    tmp = composite.with_name(composite.name + ".tmp")        # replace atomically: mpv may re-read it while we write
    tmp.write_text(build_composite_ass(tracks), encoding="utf-8")
    tmp.replace(composite)
    return Prepared(composite, rows, found.skipped, truncated=len(found.combos) > MAX_COMBOS)


def publish_json(prepared: Prepared) -> str:
    """The value of user-data/oh/sub-combos (read by oh_sub_combos.lua)."""
    return json.dumps({"title": COMPOSITE_TITLE, "composite": str(prepared.composite), "combos": prepared.combos,
                       "truncated": prepared.truncated}, ensure_ascii=False, separators=(",", ":"))
