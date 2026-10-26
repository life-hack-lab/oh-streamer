"""
Persistent job list for the background video-processing queue (no torch, no lada: runs in the GUI process).

Layout under `root` (default %LOCALAPPDATA%\\oh_streamer\\queue\\):
  queue.json        item list + queue settings, written to a temp file first and swapped in with os.replace
  items/<id>/       source.* (downloaded original), work/ (worker scratch), output.mkv (finished video)

Item states: waiting -> downloading -> queued -> processing -> done, plus paused / failed. Items left in a transient
state (downloading / processing) by a closed or crashed process come back as waiting / queued on load.

Stream items (kind "stream", streaming-subtitles plan T2.1) are temporary subtitle-only items for a video the player
is streaming: they live in memory only (never in queue.json), keep their scratch files under streams/<id>/ (not items/),
are never downloaded / restored / finalized, and are invisible to items(), next_runnable, move, eviction and usage.
get / update / item_dir / remove_stream work on them by id. Whatever a previous run left in streams/ is wiped on start.

All public methods are thread-safe (one RLock). Mutations persist at once and then call the `on_change` callbacks
(outside the lock), so a UI can simply re-read items(). Eviction only ever deletes items that are done AND watched AND
not kept, least recently used first; if that cannot free enough room, nothing is deleted and ensure_space returns False.
"""

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .subtitle_recovery import LEGACY_SUBTITLE_FIELDS

QUEUE_FILE = "queue.json"
ITEMS_DIR = "items"
STREAMS_DIR = "streams"
STREAM_KIND = "stream"
STREAM_STATE = "streaming"
FORMAT_VERSION = 1

TRANSIENT_STATES = {"downloading": "waiting", "processing": "queued"}

SUBTITLE_STATES = {"queued", "waiting_gpu", "asr", "translate", "partial", "ready", "mux",
                   "mux_pending", "done", "failed", "skipped", "cancelled", "paused"}

DEFAULT_SETTINGS: dict[str, Any] = {
    "detection_model": None,  # inherit player settings until explicitly selected
    "restoration_model": None,
    "engine": None,           # restoration engine; None = inherit player settings (lada when neither names one)
    "max_clip_length": None,  # frames per restoration clip; None = inherit player settings
    "cap_gb": 50,
    "cache_dir": "",  # empty -> the store root
    "encoder": "hevc",
    "interpolation": "full",
    "save_compare": False,
    "compare_original": "copy",   # copy | encode: how the original-picture track is stored (only with save_compare)
    "compare_quality": "small",   # same | small | tiny: CQ offset 0 / +4 / +8 of an encoded original track
    "prevent_sleep": True,
    "shutdown_when_done": False,
    "show_thumbnails": True,      # UI only (film-strip thumbnails on the queue cards): not part of cache_key
    "show_thumbnail_cover": True,  # UI only: use the cover in the first film-strip slot
    "subtitle_asr": "whisper-ja-1.5B",  # retained for explicit diagnostic presets
    "subtitle_asr_mode": "dual",
    "subtitle_translation": "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
    "subtitle_primary": None,
    "subtitle_profile": None,
    "subtitle_comparison_enabled": None,
    "subtitle_comparison_rows": None,
    "subtitle_interjections": False,
    "subtitle_japanese": False,
    "subtitle_cloud_enabled": False,
    "subtitle_cloud_strategy": "smart_horizon",
    # Empty selection uses the production dual/single pipeline.
    "subtitle_combos_default": [],
    "subtitle_python": "",
    "subtitle_models": "",
    "subtitle_server": "",
    # Optional root for local subtitle models; OH_STREAMER_SUBTITLE_MODELS and subtitle_models win.
    # Empty keeps the existing default folder, so installed models never move.
    "custom_models_directory": "",
}

COMPARE_ORIGINAL_CHOICES = ("copy", "encode")
COMPARE_QUALITY_CHOICES = ("same", "small", "tiny")

# Settings that change the output pixels (compare/save_compare is folded into "compare"). An item may pin its own
# interpolation / encoder / save_compare in settings["pinned"] (intake panel overrides): _build_job and later queue
# settings changes leave those alone, and cache_key counts them.
PIXEL_SETTINGS = ("detection_model", "restoration_model", "fp16", "max_clip_length", "encoder", "interpolation")


OBSOLETE_SETTINGS = ("subtitle_head_minutes", "subtitle_head_seconds", "subtitle_translation_igpu",
                     "subtitle_igpu_server", "subtitle_igpu_device",
                     "subtitle_cloud_horizon_minutes")


def _migrate_subtitle(item: "Item", plan_exists: bool = False) -> None:
    """Load-time clean-up of an item written by a pre-M7 build (cheap: dict edits only; the slice table of an old
    item is rebuilt by the executor when it first starts). Never touches what a finished item needs: status,
    block_srt / block_subtitles / covered_s / audio_offset (the playable subtitle) and the files on disk."""
    sub = item.subtitle
    legacy_blocked = sub.get("status") == "blocked"          # item-level blocked no longer exists
    for key in LEGACY_SUBTITLE_FIELDS:
        sub.pop(key, None)
    sub.pop("view", None)                                    # derived: recomputed by the first scheduler tick
    if legacy_blocked:
        sub.update(status="partial" if sub.get("covered_s") else "queued", percent=0, error="")
    for key in OBSOLETE_SETTINGS:
        item.settings.pop(key, None)
    for key in ("subtitle_python", "subtitle_models", "subtitle_server"):     # resolved at use from the global settings
        item.settings.pop(key, None)
    if item.settings.get("subtitles") and not plan_exists and sub.get("status", "") in ("", "queued")             and not sub.get("covered_s") and not sub.get("block_srt"):
        # never started: follows the global subtitle settings (a started item keeps its snapshot as the locked one)
        from .subtitle_recovery import SNAPSHOT_KEYS
        for key in SNAPSHOT_KEYS:
            item.settings.pop(key, None)
    from oh_streamer.subtitles.configuration import drop_draft
    drop_draft(sub)                                          # D2: no draft -> refine flow; the field is just dropped
    drop_draft(item.settings)


@dataclass
class Item:
    id: str = ""
    title: str = ""
    target: str = ""
    kind: str = "url"  # url | file | stream (temporary subtitle-only item; see the module docstring)
    headers: dict = field(default_factory=dict)
    quality: Any = None
    settings: dict = field(default_factory=dict)
    cache_key: str = ""
    state: str = "waiting"
    order: int = 0
    created: float = 0.0
    last_used: float = 0.0
    watched: bool = False
    # the user has started playing this item at least once (drives the unwatched-finished sparkle); separate from
    # `watched` (played to the end, drives eviction). Items from older store files default to True on load.
    seen: bool = False
    keep: bool = False
    error: str = ""
    source_path: str = ""
    output_path: str = ""
    size_bytes: int = 0
    progress: dict = field(default_factory=dict)
    # Progressive HLS sources keep their local playlist stable while a sidecar tracks the cache lifecycle.
    # Empty values retain the pre-HLS queue format and behavior.
    source_status: str = ""  # "" | downloading | complete | failed | paused
    download_progress: dict = field(default_factory=dict)
    parent_id: str = ""  # previous processing version; target always remains the original locator
    source_origin: str = ""  # original | compare_copy | compare_encode
    subtitle: dict = field(default_factory=dict)
    play_after_subtitles: bool = False
    play_after_subtitles_start: Optional[float] = None
    # kind == "stream" only: how the player's source is reached and where subtitles take their audio from
    # ({"source_kind": "url" | "file", "audio_url", "audio_headers", ...}, filled in by the stream flow, T2.2/T2.3).
    stream: dict = field(default_factory=dict)

    @property
    def is_stream(self) -> bool:
        return self.kind == STREAM_KIND

    def to_dict(self) -> dict:
        return copy.deepcopy(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict, plan_exists: bool = False) -> "Item":
        """Tolerant: unknown keys are dropped, missing keys take their defaults."""
        known = {f.name for f in dataclasses.fields(cls)}
        if "seen" not in d:
            d = {**d, "seen": True}  # legacy store file: don't count old finished items as unplayed
        item = cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in known})
        item.id = str(item.id)
        item.play_after_subtitles = item.play_after_subtitles is True
        if not isinstance(item.headers, dict):
            item.headers = {}
        if not isinstance(item.settings, dict):
            item.settings = {}
        if not isinstance(item.progress, dict):
            item.progress = {}
        if not isinstance(item.download_progress, dict):
            item.download_progress = {}
        if not isinstance(item.subtitle, dict):
            item.subtitle = {}
        if not isinstance(item.stream, dict):
            item.stream = {}
        _migrate_subtitle(item, plan_exists)
        if not isinstance(item.settings.get("subtitle_combos", []), list):
            del item.settings["subtitle_combos"]
        if item.state == "done" and item.subtitle.get("status") in ("ready", "mux"):
            item.subtitle.update(status="mux_pending", percent=None, only=True, phase="", detail="")
        if item.subtitle.get("status") in (
            "asr",
            "translate",
            "combos",
            "mux",
            "waiting_gpu",
        ):
            item.subtitle.update(status="queued", percent=0, phase="", detail="")
        if item.source_status not in ("", "downloading", "complete", "failed", "paused"):
            item.source_status = ""
        return item


def cache_key(target: str, quality: Any, settings: dict) -> str:
    """sha1 of the canonical JSON of target + quality + the settings that affect the output pixels."""
    settings = settings or {}
    relevant = {k: settings.get(k) for k in PIXEL_SETTINGS}
    # Keep old auto-device cache keys stable; explicit choices may change precision.
    if settings.get("device", "auto") != "auto":
        relevant["device"] = settings["device"]
    compare = settings.get("save_compare", settings.get("compare", False))
    if isinstance(compare, str):
        compare = compare.lower() in ("on", "true", "1")
    pinned = settings.get("pinned") if isinstance(settings.get("pinned"), dict) else {}
    for k in ("interpolation", "encoder"):         # per-item overrides from the intake panel beat the queue settings
        if k in pinned:
            relevant[k] = pinned[k]
    if "save_compare" in pinned:
        compare = pinned["save_compare"]
    relevant["compare"] = bool(compare)
    engine = settings.get("engine")
    if isinstance(engine, str) and engine not in ("", "lada"):   # lada (and items from before the engine setting) keep their old keys
        relevant["engine"] = engine
    if compare:                    # the compare-track options only matter (and only count) with compare on
        orig, qual = settings.get("compare_original"), settings.get("compare_quality")
        relevant["compare_original"] = orig if orig in COMPARE_ORIGINAL_CHOICES else DEFAULT_SETTINGS["compare_original"]
        relevant["compare_quality"] = qual if qual in COMPARE_QUALITY_CHOICES else DEFAULT_SETTINGS["compare_quality"]
    blob = json.dumps({"target": target, "quality": quality, "settings": relevant},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


class QueueStore:
    """Ordered, persisted list of Items plus queue settings."""

    def __init__(self, root: Path, on_change: Optional[Callable[[], None]] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.on_change: list[Callable[[], None]] = [on_change] if on_change else []
        self._lock = threading.RLock()
        self._items: list[Item] = []
        self._streams: list[Item] = []  # temporary stream items: memory only, never persisted
        self._settings: dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        self._usage: Optional[int] = None
        self.protected_ids: set = set()  # items the player is watching: never evicted (set by the service)
        shutil.rmtree(self.root / STREAMS_DIR, ignore_errors=True)  # residue of a closed or crashed run
        self._load()

    # --- persistence ---

    @property
    def path(self) -> Path:
        return self.root / QUEUE_FILE

    def _load(self) -> None:
        with self._lock:
            data: dict = {}
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("queue.json is not an object")
                except (OSError, ValueError):
                    try:
                        os.replace(self.path, self.path.with_suffix(".json.bad"))
                    except OSError:
                        pass
                    data = {}
            stored = data.get("settings")
            if isinstance(stored, dict):
                self._settings.update(stored)
            for key in OBSOLETE_SETTINGS:                         # M7: the block-mode head window is gone (D5)
                self._settings.pop(key, None)
            from oh_streamer.subtitles.configuration import drop_draft
            drop_draft(self._settings)                            # D2: saved draft models / 'tiered' mode are dropped
            for key, choices in (("compare_original", COMPARE_ORIGINAL_CHOICES),
                                 ("compare_quality", COMPARE_QUALITY_CHOICES)):
                if self._settings.get(key) not in choices:       # missing (older queue.json) is already the default
                    self._settings[key] = DEFAULT_SETTINGS[key]
            items = []
            for raw in data.get("items") or []:
                if isinstance(raw, dict) and raw.get("id") and raw.get("kind") != STREAM_KIND:
                    plan = self.root / ITEMS_DIR / str(raw["id"]) / "subtitles" / "slice-plan.json"
                    items.append(Item.from_dict(raw, plan_exists=plan.is_file()))
            items.sort(key=lambda it: it.order)
            changed = False
            for it in items:
                new_state = TRANSIENT_STATES.get(it.state)
                if new_state:
                    it.state = new_state
                    changed = True
            self._items = items
            self._renumber()
            if self._settings.get('subtitle_profile') is None:
                from oh_streamer.subtitles.configuration import migrate
                from .subtitle_jobs import subtitle_configuration
                primary = subtitle_configuration(self._settings)['subtitle_primary']
                self._settings['subtitle_profile'] = migrate(self._settings, primary, fresh=not bool(stored))
                if not stored:
                    # A new install starts on the cloud-first contract: its preselected but unconfigured cloud profile
                    # blocks subtitle jobs (待設定) instead of silently running the legacy local pipeline, which a
                    # release may not even ship. Existing users keep 'legacy' until they save a profile.
                    self._settings.setdefault('subtitle_runtime_contract', 'cloud-first-1')
                changed = True
            if changed or not self.path.exists():
                self._save()

    def _renumber(self) -> None:
        for i, it in enumerate(self._items):
            it.order = i

    def _save(self) -> None:
        data = {"version": FORMAT_VERSION, "settings": self._settings, "items": [it.to_dict() for it in self._items]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def _notify(self) -> None:
        for cb in list(self.on_change):
            try:
                cb()
            except Exception:
                pass

    def _find(self, item_id: str) -> Item:
        for it in self._items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)

    def _find_any(self, item_id: str) -> Item:
        """Like _find, but also finds a stream item."""
        for it in self._streams:
            if it.id == item_id:
                return it
        return self._find(item_id)

    def _is_stream_id(self, item_id: str) -> bool:
        return any(it.id == item_id for it in self._streams)

    # --- queue settings ---

    def get_settings(self) -> dict:
        with self._lock:
            from .subtitle_jobs import subtitle_configuration
            return copy.deepcopy({**self._settings, **subtitle_configuration(self._settings)})

    def set_settings(self, **fields: Any) -> dict:
        with self._lock:
            from oh_streamer.subtitles.configuration import normalize, migrate
            from .subtitle_jobs import subtitle_configuration
            if 'subtitle_profile' in fields:
                fields['subtitle_profile'] = normalize(fields['subtitle_profile'])
                fields.setdefault('subtitle_runtime_contract', 'cloud-first-1')
            elif any(k in fields for k in ('subtitle_primary', 'subtitle_asr', 'subtitle_asr_mode', 'subtitle_translation', 'subtitle_japanese', 'subtitle_interjections', 'subtitle_cloud_strategy')):
                fields['subtitle_runtime_contract'] = 'legacy'
                # Legacy editor remains the runtime authority during S1. Reflect
                # its explicit edits in the saved mode without discarding others.
                merged = {**self._settings, **fields, 'subtitle_profile': None}
                migrated = migrate(merged, subtitle_configuration(merged)['subtitle_primary'])
                profile = copy.deepcopy(self._settings['subtitle_profile'])
                mode = migrated['active_mode']
                profile['profiles'][mode].update({k: migrated['profiles'][mode][k]
                    for k in ('asr', 'translation', 'processing', 'legacy_primary')})
                profile['active_mode'] = mode
                fields['subtitle_profile'] = profile
            self._settings.update(fields)
            self._save()
            result = copy.deepcopy(self._settings)
        self._notify()
        return result

    # --- items ---

    def item_dir(self, item_id: str) -> Path:
        with self._lock:
            stream = self._is_stream_id(item_id)
        return self.root / (STREAMS_DIR if stream else ITEMS_DIR) / item_id

    def items(self) -> list[Item]:
        """The regular queue items; stream items are not listed (see stream_items)."""
        with self._lock:
            return [copy.deepcopy(it) for it in self._items]

    def get(self, item_id: str) -> Optional[Item]:
        with self._lock:
            try:
                return copy.deepcopy(self._find_any(item_id))
            except KeyError:
                return None

    def _subtitle_add_settings(self, settings: Optional[dict]) -> dict:
        """Item settings for a new item. With subtitles the configuration is only VALIDATED here (same errors as a
        start would give); nothing of it is frozen: the item keeps ``subtitles`` and its per-video
        ``subtitle_overrides`` and follows the global subtitle settings until its first processing start."""
        settings = copy.deepcopy(settings or {})
        if settings.get("subtitles"):
            from .subtitle_jobs import derive_subtitle_snapshot, snapshot_subtitle_settings, SNAPSHOT_ITEM_KEYS
            glob = {key: value for key, value in self.get_settings().items() if key.startswith("subtitle_")}
            explicit = [key for key in settings if key.startswith("subtitle_") and key not in SNAPSHOT_ITEM_KEYS]
            if explicit:
                # the caller froze a configuration into the item on purpose (per-item configuration): keep it
                if settings.get("subtitle_combos") and "subtitle_configs" not in settings:
                    from .subtitle_jobs import load_combo_presets
                    presets, _ = load_combo_presets()
                    settings["subtitle_configs"] = [copy.deepcopy(presets[key]) for key in settings["subtitle_combos"] if key in presets]
                return snapshot_subtitle_settings({**glob, **settings})
            derive_subtitle_snapshot(self.get_settings(), settings)
        return settings

    # --- stream items (temporary, subtitle-only) ---

    def add_stream(self, target: str, source_kind: str, headers: Optional[dict], settings: Optional[dict],
                   title: Optional[str] = None, stream: Optional[dict] = None) -> Item:
        """Create a temporary stream item and its streams/<id>/ folder. No dedupe: every stream gets its own item.

        `source_kind` is "url" or "file"; `stream` carries extra source info (audio source, ...). With
        settings["subtitles"] the subtitle block starts "queued" and the subtitle settings are snapshotted exactly
        as add() does; the item has no restore job, so state stays "streaming" for its whole life.
        """
        settings = self._subtitle_add_settings(settings)
        now = time.time()
        item = Item(id=uuid.uuid4().hex[:8], title=title or target, target=target, kind=STREAM_KIND,
                    headers=copy.deepcopy(headers or {}), settings=settings, state=STREAM_STATE, created=now,
                    last_used=now, stream={**copy.deepcopy(stream or {}), "source_kind": source_kind})
        if settings.get("subtitles"):
            item.subtitle = dict(status="queued", percent=0, error="")
        with self._lock:
            self._streams.append(item)
            self.item_dir(item.id).mkdir(parents=True, exist_ok=True)
            result = copy.deepcopy(item)
        self._notify()
        return result

    def stream_items(self) -> list[Item]:
        with self._lock:
            return [copy.deepcopy(it) for it in self._streams]

    def remove_stream(self, item_id: str) -> bool:
        """Drop a stream item and delete its streams/<id>/ folder. False if the id is not a stream item."""
        with self._lock:
            for it in self._streams:
                if it.id == item_id:
                    self._streams.remove(it)
                    shutil.rmtree(self.root / STREAMS_DIR / item_id, ignore_errors=True)
                    break
            else:
                return False
        self._notify()
        return True

    def clear_streams(self) -> int:
        """Drop every stream item and wipe streams/ (queue shutdown). Returns how many items were dropped."""
        with self._lock:
            count = len(self._streams)
            self._streams.clear()
            shutil.rmtree(self.root / STREAMS_DIR, ignore_errors=True)
        if count:
            self._notify()
        return count

    def add(self, target: str, kind: str, headers: Optional[dict], quality: Any, settings: Optional[dict],
            title: Optional[str] = None, *, force: bool = False, state: str = "waiting") -> Item:
        """Append a waiting item; an existing item with the same cache_key is returned instead of a duplicate."""
        settings = self._subtitle_add_settings(settings)
        key = cache_key(target, quality, settings)
        with self._lock:
            for it in self._items:
                if not force and it.cache_key == key:
                    return copy.deepcopy(it)
            now = time.time()
            item = Item(id=uuid.uuid4().hex[:8], title=title or target, target=target, kind=kind,
                        headers=copy.deepcopy(headers or {}), quality=copy.deepcopy(quality), settings=settings,
                        cache_key=key, state=state, order=len(self._items), created=now, last_used=now)
            if settings.get("subtitles"):
                item.subtitle = dict(status="queued", percent=0, error="")
            self._items.append(item)
            self._save()
            result = copy.deepcopy(item)
        self._notify()
        return result

    def remove(self, item_id: str) -> bool:
        """Drop the item and delete its items/<id>/ directory. False if the id is unknown."""
        with self._lock:
            try:
                item = self._find(item_id)
            except KeyError:
                return False
            self._delete_locked(item)
            self._save()
        self._notify()
        return True

    def _delete_locked(self, item: Item) -> None:
        self._items.remove(item)
        self._renumber()
        shutil.rmtree(self.item_dir(item.id), ignore_errors=True)
        self._usage = None

    def move(self, item_id: str, new_index: int) -> None:
        with self._lock:
            item = self._find(item_id)
            self._items.remove(item)
            new_index = max(0, min(int(new_index), len(self._items)))
            self._items.insert(new_index, item)
            self._renumber()
            self._save()
        self._notify()

    def update(self, item_id: str, **fields: Any) -> Item:
        with self._lock:
            item = self._find_any(item_id)
            known = {f.name for f in dataclasses.fields(Item)} - {"id", "order", "kind"}
            for name, value in fields.items():
                if name not in known:
                    raise ValueError(f"cannot update field {name!r}")
                setattr(item, name, copy.deepcopy(value))
            if not item.is_stream:       # stream items are never persisted
                self._save()
            result = copy.deepcopy(item)
        self._notify()
        return result

    def clear_headers(self, item_id: str) -> None:
        self.update(item_id, headers={})

    def mark_seen(self, item_id: str) -> bool:
        """Set seen=True; persists / notifies only on a real change. Returns True when it changed."""
        with self._lock:
            try:
                item = self._find(item_id)
            except KeyError:
                return False
            if item.seen:
                return False
            item.seen = True
            self._save()
        self._notify()
        return True

    def mark_all_seen(self) -> int:
        """Set seen=True on every item (one save / notify when anything changed). Returns the item count."""
        with self._lock:
            changed = False
            for it in self._items:
                if not it.seen:
                    it.seen = True
                    changed = True
            if changed:
                self._save()
            count = len(self._items)
        if changed:
            self._notify()
        return count

    def mark_used(self, item_id: str) -> None:
        self.update(item_id, last_used=time.time())

    def next_runnable(self, state: str) -> Optional[Item]:
        """First item in queue order with exactly this state (paused / failed items are never picked)."""
        with self._lock:
            for it in self._items:
                if it.state == state:
                    return copy.deepcopy(it)
            return None

    # --- cache usage and eviction ---

    def usage_bytes(self, refresh: bool = False) -> int:
        with self._lock:
            if refresh or self._usage is None:
                total = 0
                for it in self._items:
                    total += _dir_size(self.item_dir(it.id))
                self._usage = total
            return self._usage

    def ensure_space(self, bytes_needed: int) -> bool:
        """Make room for `bytes_needed` under the cap by evicting done+watched+not-keep items, LRU first.

        Returns False (and deletes nothing) if even evicting every candidate would not be enough.
        """
        with self._lock:
            cap = int(float(self._settings.get("cap_gb", 50)) * 1024 ** 3)
            usage = self.usage_bytes(refresh=True)
            if usage + bytes_needed <= cap:
                return True
            protected = set(self.protected_ids)
            candidates = sorted((it for it in self._items if it.state == "done" and it.watched and not it.keep
                                 and it.id not in protected),
                                key=lambda it: it.last_used)
            plan = []
            for it in candidates:
                if usage + bytes_needed <= cap:
                    break
                usage -= _dir_size(self.item_dir(it.id))
                plan.append(it)
            if usage + bytes_needed > cap:
                return False
            for it in plan:
                self._delete_locked(it)
            self._save()
            self.usage_bytes(refresh=True)
        self._notify()
        return True
