"""Small queue subtitle contracts, independent of torch and the restoration runtime."""
import contextlib
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

from oh_streamer.subtitles import runtime
from .subtitle_recovery import SubtitleRecoveryMixin, without_legacy


ACTIVE = {"queued", "waiting_gpu", "asr", "translate", "mux", "partial"}
LABELS = dict(
    queued="等待 GPU",
    waiting_gpu="等待 GPU",
    asr="辨識中",
    translate="翻譯中",
    mux="封裝中",
    ready="等待封裝",
    partial="字幕前段完成",
    mux_pending="等關閉播放器後封裝",
    done="字幕完成",
    failed="字幕失敗",
    blocked="字幕暫停等待",
    skipped="字幕略過",
    cancelled="字幕已暫停",
    paused="字幕已暫停",
)


def progress_fields(stage, message):
    """Keep display phases separate from the scheduler's process stages."""
    # Stages this module still launches: the mux step; the plain asr / translate rows are kept as the generic
    # contract the progress tests exercise.
    allowed = {
        "asr": {"prepare_asr", "asr"},
        "translate": {"prepare_translate", "translate"},
        "mux": {"mux"},
    }
    phase = message.get("phase", stage)
    if not isinstance(phase, str) or phase not in allowed.get(stage, set()):
        return None
    percent = message.get("percent")
    if percent is not None:
        try:
            percent = float(percent)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            return None
    return dict(status=stage, phase=phase, percent=percent, detail=str(message.get("detail", "")))


@contextmanager
def replacement_guard(path):
    """On Windows deny readers until replacement completes, including paused mpv."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        # READ_ATTRIBUTES + DELETE, sharing DELETE only: concurrent readers
        # cannot enter; replace_guarded_file renames without dropping this lock.
        handle = create(str(Path(path).resolve()), 0x10080, 0x4, None, 3, 0, None)
        if handle == wintypes.HANDLE(-1).value:
            raise RuntimeError("影片仍被程式開啟或無法替換，請關閉播放器後只重跑字幕")
        try:
            yield
        finally:
            close(handle)
    else:
        import psutil
        target = Path(path).resolve()
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.pid == os.getpid():
                continue
            try:
                if any(Path(f.path).resolve() == target for f in proc.open_files()):
                    raise RuntimeError("影片仍被程式開啟，請關閉播放器後只重跑字幕")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        yield


def replace_guarded_file(source, target):
    """Rename under replacement_guard, without dropping the target reader lock.

    Windows FileRenameInfoEx supports replacing a delete-shared destination
    while its guard handle stays open. MoveFileEx/os.replace cannot do this.
    https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-file_rename_info
    """
    if os.name != "nt":
        os.replace(source, target)
        return
    expected = Path(source).stat()
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]
    name = str(Path(target).resolve()).encode("utf-16-le")
    # SetFileInformationByHandle's Win32 path conversion reads a WCHAR NUL even
    # though FileNameLength excludes it. Reserve it explicitly, never rely on
    # allocator padding (which can silently append garbage to the target name).
    buffer = ctypes.create_string_buffer(RenameInfo.FileName.offset + len(name) + 2)
    info = ctypes.cast(buffer, ctypes.POINTER(RenameInfo)).contents
    info.Flags = 0x1 | 0x2  # REPLACE_IF_EXISTS | POSIX_SEMANTICS
    info.RootDirectory = None
    info.FileNameLength = len(name)
    ctypes.memmove(
        ctypes.addressof(buffer) + RenameInfo.FileName.offset,
        name,
        len(name),
    )
    handle = create(str(Path(source).resolve()), 0x10000, 0x7, None, 3, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        rename = kernel.SetFileInformationByHandle
        rename.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        rename.restype = wintypes.BOOL
        if not rename(handle, 22, buffer, len(buffer)):  # FileRenameInfoEx
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close(handle)
    actual = Path(target).stat()
    if (actual.st_dev, actual.st_ino, actual.st_size) != (
        expected.st_dev,
        expected.st_ino,
        expected.st_size,
    ):
        raise RuntimeError("字幕檔案替換驗證失敗")


def subtitle_paths(folder, japanese=False):
    from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE
    full = folder / "sakura.zh-TW.full.srt"
    tracks = [(str(folder / "sakura.zh-TW.srt"), "chi", CONCISE_TITLE if full.is_file() else "繁體中文（AI）")]
    if full.is_file():
        tracks.append((str(full), "chi", FULL_TITLE))
    if japanese:
        tracks.append((str(folder / "ja.srt"), "jpn", "日文（AI）"))
    return tracks


def subtitle_configuration(settings):
    """Normalize saved editor rows, migrating flat settings and historical presets."""
    from oh_streamer.subtitles.combos import normalize_combo
    primary = settings.get("subtitle_primary")
    if primary is not None and not isinstance(primary, dict):
        raise ValueError("主字幕設定必須是物件")
    if not isinstance(primary, dict):
        primary = dict(id="primary", label="主字幕", asr=dict(mode=settings.get("subtitle_asr_mode", "dual"), model=settings.get("subtitle_asr") or "whisper-ja-1.5B"),
                       translation=dict(profile="sakura-galtransl-7b-v3.7", model_file=settings.get("subtitle_translation") or "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"),
                       sentence=dict(show_interjections=bool(settings.get("subtitle_interjections"))), japanese_track=bool(settings.get("subtitle_japanese")))
    primary = normalize_combo({**{key: value for key, value in primary.items() if key != "source"}, "id": "primary"})
    rows = settings.get("subtitle_comparison_rows")
    if rows is None:
        presets, _ = load_combo_presets()
        rows = [copy.deepcopy(presets[key]) for key in settings.get("subtitle_combos_default", []) if key in presets]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("比較字幕列必須是物件清單")
    rows = [normalize_combo({key: value for key, value in row.items() if key != "source"}) for row in rows]
    ids = [primary["id"]] + [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("字幕列 ID 不可重複")
    enabled = settings.get("subtitle_comparison_enabled")
    if enabled is None:
        enabled = bool(settings.get("subtitle_combos_default"))
    if not isinstance(enabled, bool):
        raise ValueError("比較模式必須是布林值")
    return dict(subtitle_primary=primary, subtitle_comparison_enabled=enabled, subtitle_comparison_rows=rows)


def _local_model_paths(settings):
    from oh_streamer.subtitles.environment import runtime_paths
    paths = runtime_paths(settings)
    return paths["models"], paths["server"]


def combo_availability_for(combo, settings=None):
    """combos.combo_availability against the models folder / llama-server the settings resolve to."""
    from oh_streamer.subtitles import combos
    models, server = _local_model_paths(settings or {})
    return combos.combo_availability(combo, models, server, settings)


def _apply_model_fallback(result, settings):
    """Use-time default fallback (the saved setting stays untouched): a local primary whose downloaded models are
    missing runs on the first available builtin combo (combos.FALLBACK_ORDER). With none available nothing changes,
    so the usual missing-model failure still shows. The note is kept as ``subtitle_combo_fallback``."""
    result.pop("subtitle_combo_fallback", None)
    if result.get("subtitle_runtime_contract") == "cloud-first-1" or not result.get("subtitle_configs"):
        return
    from oh_streamer.subtitles import combos
    primary = result["subtitle_configs"][0]
    if combos.is_cloud_combo(primary, settings):
        return
    try:
        models, server = _local_model_paths(settings)
        resolved = combos.resolve_available_combo(primary, models, server, settings, load_combo_presets()[0])
    except Exception:
        return
    if resolved["fallback"]:
        result["subtitle_configs"][0] = resolved["combo"]
        result["subtitle_combo_fallback"] = resolved["fallback"]


def snapshot_subtitle_settings(settings):
    from oh_streamer.subtitles.configuration import assert_secret_free
    assert_secret_free(settings)
    result = copy.deepcopy(settings)
    if not isinstance(result.get("subtitle_configs"), list):
        config = subtitle_configuration(settings)
        result.update(config)
        rows = [config["subtitle_primary"]] + (config["subtitle_comparison_rows"] if config["subtitle_comparison_enabled"] else [])
        result["subtitle_configs"] = copy.deepcopy(rows)
        result["subtitle_combos"] = [row["id"] for row in rows]
        _apply_model_fallback(result, settings)
    if 'subtitle_resolved' not in result:
        from oh_streamer.subtitles.configuration import migrate, resolve
        primary = result['subtitle_configs'][0] if result['subtitle_configs'] else {}
        profile = migrate(settings, primary)
        result['subtitle_profile'] = profile
        result['subtitle_resolved'] = resolve(profile, settings)
        # Explicit profile edits activate frozen execution for newly queued jobs.
        # Migrated settings and previously saved jobs retain legacy rows.
        result['subtitle_runtime_contract'] = settings.get('subtitle_runtime_contract', 'legacy')
        if result['subtitle_runtime_contract'] == 'cloud-first-1':
            from oh_streamer.subtitles.configuration import runtime_primary
            result['subtitle_configs'][0] = runtime_primary(result, result['subtitle_configs'][0])
            result['subtitle_primary'] = copy.deepcopy(result['subtitle_configs'][0])   # legacy mirror of the profile
            result['subtitle_cloud_strategy'] = result['subtitle_resolved']['processing']
    for key in PATH_KEYS:       # resolved at use from the global settings (environment.runtime_paths), never stored
        result.pop(key, None)
    return freeze_track_plans(result)


def _sync_legacy_primary(store):
    """Keep the legacy ``subtitle_primary`` row a mirror of the profile (the profile is the source of truth)."""
    from oh_streamer.subtitles.configuration import runtime_primary, resolve, ConfigurationError
    saved = store.get_settings()
    try:
        frozen = {**saved, 'subtitle_resolved': resolve(saved['subtitle_profile'], saved)}
        row = runtime_primary(frozen, subtitle_configuration(saved)['subtitle_primary'])
    except (ConfigurationError, TypeError, ValueError, KeyError):
        return
    if row != saved.get('subtitle_primary'):
        store.set_settings(subtitle_primary=row)


def _guard_settings(item, global_settings):
    """What the model-delete guard scans for an item: its locked snapshot, or - for an item that has not started
    (no snapshot, it follows the global settings) - the global subtitle settings it will adopt plus its overrides."""
    if not item.settings.get("subtitles") or item.settings.get("subtitle_configs") or item.settings.get("subtitle_resolved"):
        return item.settings
    base = {k: v for k, v in (global_settings or {}).items() if k.startswith("subtitle_")}
    return {**base, **item.settings}


# subtitle_* item keys that are not part of a snapshot: the user's choices for this video
SNAPSHOT_ITEM_KEYS = ("subtitle_overrides", "subtitle_hold", "subtitle_hold_sig")
PATH_KEYS = ("subtitle_python", "subtitle_models", "subtitle_server")
_VOLATILE_CONFIG_KEYS = ("label", "source")


def derive_subtitle_snapshot(global_settings, item_settings):
    """The subtitle snapshot an item would run with right now: the global ``subtitle_*`` settings plus the item's
    ``subtitle_overrides`` (only ``subtitle_comparison_enabled`` today). Never stores the python / models / server
    paths (they are resolved at use from the global settings); ``subtitle_track_plans`` is re-derived each time.
    Raises what ``snapshot_subtitle_settings`` raises when the configuration is not ready."""
    base = {key: copy.deepcopy(value) for key, value in (global_settings or {}).items()
            if key.startswith("subtitle_") and key != "subtitle_overrides"}
    overrides = (item_settings or {}).get("subtitle_overrides")
    if isinstance(overrides, dict) and isinstance(overrides.get("subtitle_comparison_enabled"), bool):
        base["subtitle_comparison_enabled"] = overrides["subtitle_comparison_enabled"]
    base.pop("subtitle_configs", None)           # rows follow the (overridden) global editor state
    base.pop("subtitle_combos", None)
    base["subtitles"] = True
    return snapshot_subtitle_settings(base)


def _identity_value(value):
    """JSON-stable copy with model paths reduced to their name (the model folder may move)."""
    if isinstance(value, dict):
        return {str(k): (Path(str(v)).name if k == "model_path" and isinstance(v, str) else _identity_value(v))
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0])) if k not in _VOLATILE_CONFIG_KEYS}
    if isinstance(value, (list, tuple)):
        return [_identity_value(v) for v in value]
    return value


def _binding_identity(row):
    if not isinstance(row, dict):
        return None
    return _identity_value({k: row[k] for k in ("id", "provider", "connection_id", "remote_model", "base_url",
                                                  "recipe", "model_path") if k in row})


def _digest(payload):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def _asr_identity(snapshot):
    snapshot = snapshot or {}
    configs = snapshot.get("subtitle_configs") if isinstance(snapshot.get("subtitle_configs"), list) else []
    primary = configs[0] if configs and isinstance(configs[0], dict) else {}
    resolved = snapshot.get("subtitle_resolved") if isinstance(snapshot.get("subtitle_resolved"), dict) else {}
    models = resolved.get("models") if isinstance(resolved.get("models"), dict) else {}
    return dict(row=_binding_identity(models.get("asr")), config=_identity_value(primary.get("asr") or {}),
                whisper_policy=resolved.get("whisper_policy"),
                whisper=[_binding_identity(r) if isinstance(r, dict) else r for r in resolved.get("whisper_models") or []])


def asr_signature(snapshot):
    """Digest of the primary ASR identity only (decides whether re-doing just the translation is possible)."""
    return _digest(_asr_identity(snapshot))


def content_signature(snapshot):
    """Stable digest of the content-affecting subtitle settings: primary ASR / MT identity, comparison rows,
    sentence / interjection / glossary / prompt settings. Model paths count by model id (name), never absolute;
    credentials, fallback / backup lists and policy, processing / cloud strategy and quota settings are ignored."""
    snapshot = snapshot or {}
    configs = snapshot.get("subtitle_configs") if isinstance(snapshot.get("subtitle_configs"), list) else []
    resolved = snapshot.get("subtitle_resolved") if isinstance(snapshot.get("subtitle_resolved"), dict) else {}
    models = resolved.get("models") if isinstance(resolved.get("models"), dict) else {}
    return _digest(dict(
        asr=_asr_identity(snapshot),
        mt=_binding_identity(models.get("translation")),
        configs=[_identity_value(c) for c in configs if isinstance(c, dict)],
        interjections=bool(snapshot.get("subtitle_interjections")),
        mode=resolved.get("mode")))


def freeze_track_plans(settings):
    """``settings`` with ``subtitle_track_plans`` (primary + comparison rows, `configuration.track_runtime_plans`)
    derived from its own rows and frozen into it: a queued job keeps running on the plans it was queued with, even
    when the global profile or the comparison rows change later (design 6.1 M7). The key is recomputed here,
    never copied from an older snapshot."""
    from oh_streamer.subtitles.configuration import track_runtime_plans
    plans = track_runtime_plans({key: value for key, value in settings.items() if key != 'subtitle_track_plans'})
    settings['subtitle_track_plans'] = copy.deepcopy(plans)
    return settings


# --------------------------------------------------------------------------- subtitle combos

# Measured once on the dev RTX laptop (10 min clip): peak VRAM over baseline, GiB.
_ASR_VRAM = {"anime-whisper": 1.9, "whisper-ja-1.5B": 4.7}
_TRANSLATION_VRAM = {"sakura-v1": 4.4, "sakura-galtransl-14b-v3.8": 5.4}


def load_combo_presets():
    """(presets by id, errors); the same loader the combos CLI and the worker use (builtin + user file)."""
    from oh_streamer.subtitles import combos
    return combos.load_presets()


def validate_combo_ids(ids, presets=None, settings=None):
    """Return ``(ordered unique ids, "")`` or ``(None, clear error)``; an empty list is valid (legacy pipeline)."""
    if not isinstance(ids, (list, tuple)) or any(not isinstance(i, str) for i in ids):
        return None, "字幕組合必須是組合 ID 的清單"
    unique = list(dict.fromkeys(ids))
    if presets is None:
        presets, _errors = load_combo_presets()
    unknown = [i for i in unique if i not in presets]
    if unknown:
        return None, "未知的字幕組合：" + "、".join(unknown)
    from oh_streamer.subtitles.combos import is_cloud_combo, is_cloud_enabled
    if not is_cloud_enabled(settings):
        for i in unique:
            preset = presets.get(i)
            if preset and is_cloud_combo(preset, settings):
                return None, f"雲端字幕功能未開啟：組合「{preset.get('label') or i}」包含雲端模型，請先在設定中啟用雲端字幕後端"
    return unique, ""


def gpu_hint(combo):
    asr = combo["asr"]["model"]
    profile = combo["translation"]["profile"]
    parts = []
    if asr in _ASR_VRAM:
        parts.append(f"辨識約 {_ASR_VRAM[asr]} GiB")
    if profile in _TRANSLATION_VRAM:
        parts.append(f"翻譯約 {_TRANSLATION_VRAM[profile]} GiB")
    if combo["correction"]["mode"] != "none":
        parts.append("含 LLM 校正（實驗）")
    if profile == "sakura-galtransl-14b-v3.8":
        parts.append("14B 部分卸載，約為 7B 的 4 倍時間")
    return "；".join(parts) + ("（實測 VRAM 增量，僅供參考）" if parts else "")


def _availability_fields(combo, settings):
    from oh_streamer.subtitles.combos import is_cloud_combo
    if is_cloud_combo(combo, settings):
        return dict(available=True, missing=[], unavailable_reason="", cloud=True)
    info = combo_availability_for(combo, settings)
    return dict(available=info["available"], missing=info["missing"], unavailable_reason=info["reason"], cloud=False)


def list_combos(settings=None):
    presets, _errors = load_combo_presets()
    return [
        dict(
            id=c["id"],
            label=c["label"],
            builtin=c.get("source") == "builtin",
            asr=c["asr"]["model"],
            translation=c["translation"]["profile"],
            approx_gpu_hint=gpu_hint(c),
            **_availability_fields(c, settings),
        )
        for c in presets.values()
    ]


def combos_dir(store, item_id):
    return store.item_dir(item_id) / "subtitles" / "combos"


def existing_combo_titles(folder):
    from oh_streamer.subtitles import combos
    manifest = combos.read_manifest(folder) or {}
    titles = []
    for entry in manifest.get("combos", []):
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if label:
            titles.append(str(label))
        for track in entry.get("tracks", []):
            titles.append(track["title"])
            if label:
                titles.append(str(label) + "・" + track["title"])
    return list(dict.fromkeys(titles))


class SubtitleQueueMixin(SubtitleRecoveryMixin):
    """Exclusive subtitle stages and an optional independent Vulkan translator."""

    def _subtitle_reserved_ids(self):
        return {
            it.id
            for it in self.store.items()
            if (
                it.subtitle.get("only")
                and it.subtitle.get("status") in ACTIVE | {"ready", "mux_pending"}
            )
        }

    def _subtitle_environment(self):
        from oh_streamer.subtitles.environment import detect
        try:
            return detect(self.store.get_settings())
        except Exception as exc:
            return dict(
                available=False,
                reason=f"無法偵測字幕環境：{exc}",
                asr_options=[],
                translation_options=[],
                python="",
                models="",
                server="",
            )

    def _subtitle_update(self, item_id, **fields):
        with self._lock:
            it = self.store.get(item_id)
            if it is None or item_id in self._tombstones:
                return
            if "status" in fields and "phase" not in fields:
                # Clear preparation details on pauses, retries, waits and completion.
                fields.update(phase="", detail="")
            subtitle = {**it.subtitle, **fields}
            if subtitle.get("status") in ("ready", "mux_pending", "done"):
                subtitle.pop("settings_pending", None)       # finished: nothing left to apply new settings to
            if subtitle.get("only") and subtitle.get("status") in ACTIVE | {"ready", "mux_pending"}:
                self.store.protected_ids = set(self.store.protected_ids) | {item_id}
            self.store.update(item_id, subtitle=subtitle)

    def _subtitle_check(self, run):
        from .queue_service import _Cancelled
        if run.cancel.is_set() or not self._run_is_current(run):
            raise _Cancelled()
        if self.perf.holding or self.is_player_busy():
            self._request_stop(run, "auto")
            raise _Cancelled()

    def _subtitle_stage(self, item, run, job, env):
        from .queue_service import NO_WINDOW, HERE
        self._subtitle_check(run)
        folder = self.store.item_dir(item.id) / "subtitles"
        folder.mkdir(parents=True, exist_ok=True)
        stage = job["stage"]
        job_path = folder / f"{stage}.job.json"
        runtime.atomic_write_json(job_path, job)
        phase = {"mux": "mux"}[stage]
        self._subtitle_update(
            item.id, status=stage, phase=phase, percent=None,
            detail="啟動字幕程序", error="",
        )
        environment = dict(
            os.environ,
            PYTHONPATH=str(HERE / "src"),
            PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        environment.pop("OH_STREAMER_SUBTITLE_IDLE_UI_PIDS", None)
        started = self._clock()
        error = ""
        proc = None
        with open(folder / f"{stage}.log", "ab") as stderr:
            try:
                with self._lock:
                    self._subtitle_check(run)
                    run.stage = stage
                    from oh_streamer.subtitles.environment import subtitle_command
                    run.proc = proc = subprocess.Popen(
                        # Frozen builds re-enter oh-streamer.exe (--subtitle-worker).
                        subtitle_command(env.get("runtime_python") or env["python"],
                                         "oh_streamer.subtitles.worker", "--job", job_path),
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        stderr=stderr,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=environment,
                        cwd=str(HERE),
                        creationflags=NO_WINDOW,
                        **({"start_new_session": True} if sys.platform != "win32" else {}),
                    )
                    self._subtitle_windows.append([started, None])
                    window = self._subtitle_windows[-1]
                for line in proc.stdout:
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if message.get("type") == "subtitle_progress":
                        with self._lock:
                            if not run.cancel.is_set() and self._run_is_current(run):
                                fields = progress_fields(stage, message)
                                if fields is not None:
                                    self._subtitle_update(item.id, **fields)
                    elif message.get("type") == "error":
                        error = str(message.get("message", ""))
                rc = proc.wait()
                self._subtitle_check(run)
                if rc:
                    raise RuntimeError(error or f"字幕程序退出：{rc}")
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        self._request_stop(run, "auto")
                    proc.wait()
                    proc.stdout.close()
                    proc.stdin.close()
                    with self._lock:
                        window[1] = self._clock()
                        # Keep one day of intervals for delayed performance observations.
                        self._subtitle_windows[:] = [
                            w
                            for w in self._subtitle_windows
                            if w[1] is None or w[1] > self._clock() - 86400
                        ]
                        if run.proc is proc:
                            run.proc = None

    def _subtitle_item_environment(self, item):
        env = dict(self._subtitle_environment())
        if item.settings.get("subtitle_configs"):
            from oh_streamer.subtitles.environment import resolve_runtime_python
            # Paths come from the global settings (never stored per item). Cloud rows need only an
            # interpreter: a missing venv falls back to the running app (dev interpreter or frozen exe).
            env["runtime_python"], env["runtime_available"] = resolve_runtime_python(env.get("python") or "")
            env["runtime_reason"] = "" if env["runtime_available"] else "找不到可執行的字幕 Python"
        return env

    @staticmethod
    def _subtitle_audio_offset(path):
        offset = 0.0
        from oh_streamer.sources import media_input
        try:
            # A range-cache marker is not media: probe its loopback runtime_url. No live session raises
            # SourceUnavailable (not swallowed: the host must not remember a wrong 0.0 offset).
            path, _opts = media_input.ffmpeg_input(path)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                "stream=start_time", "-of", "json", path,
            ], encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            offset = float((probe.get("streams") or [{}])[0].get("start_time") or 0.0)
            if not math.isfinite(offset):
                offset = 0.0
        except media_input.SourceUnavailable:
            raise
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            pass
        return offset

    def _add_external_subtitles(self, payload, item):
        # The scheduler publishes finished slices as `block_srt` / `block_subtitles` (host.publish): that is the
        # progressive subtitle. Comparison tracks are sidecars the player reads from the combos manifest (D3).
        has_partial = bool(item.subtitle.get("block_srt") or item.subtitle.get("covered_s"))
        is_progressive_sub = bool(item.subtitle.get("block_srt"))
        if item.subtitle.get("status") not in ("partial", "ready", "mux_pending", "done") and not has_partial:
            return
        # Progressive uses the original source even if the completed output has AI tracks.
        if item.subtitle.get("embedded", item.subtitle.get("status") == "done") and not payload.get("progressive"):
            return
        if is_progressive_sub:
            paths = item.subtitle["block_srt"]
            if isinstance(paths, dict):
                paths = [paths[k] for k in ("zh-TW", "full", "ja") if k in paths]
            elif isinstance(paths, str):
                paths = [paths]
            tracks = [(path, "chi" if i == 0 else "jpn", "繁體中文（AI）" if i == 0 else "日文（AI）") for i, path in enumerate(paths)]
            defaults = [i == 0 for i in range(len(tracks))]
            metadata = item.subtitle.get("block_subtitles")
            if metadata:
                tracks = [(t["path"], t["lang"], t["title"]) for t in metadata]
                defaults = [bool(t.get("default")) for t in metadata]
        else:
            tracks = subtitle_paths(self.store.item_dir(item.id) / "subtitles",
                                    self.store.get_settings().get("subtitle_japanese", False))
            defaults = [index == 0 for index in range(len(tracks))]
        external = [dict(path=path, lang=lang, title=title, default=default)
                    for (path, lang, title), default in zip(tracks, defaults) if Path(path).is_file()]
        if external:
            offset = (item.subtitle.get("audio_offset", 0.0) if payload.get("progressive")
                      else self._subtitle_audio_offset(item.output_path))
            payload.update(subtitles=external, subtitle_offset=offset)
            if is_progressive_sub:
                if not hasattr(self, "_subtitle_play_offsets"):
                    self._subtitle_play_offsets = {}
                self._subtitle_play_offsets[item.id] = offset
                payload.update(subtitle_covered_s=item.subtitle.get("covered_s", 0),
                               **({"subtitle_covered_ranges": item.subtitle["covered_ranges"]} if "covered_ranges" in item.subtitle else {}),
                               subtitle_complete=item.subtitle.get("status") in ("ready", "mux_pending", "done"))

    def _subtitle_mux(self, item, run, video):
        # The thumbnail thread backfills frames from a done item's output.mp4 with ffmpeg; an open reader there makes
        # replacement_guard fail ("file still open"). Hold it off for the whole mux stage. The wait for the in-flight
        # grab happens here, outside self._lock.
        hold = getattr(getattr(self, "thumbs", None), "hold", None)
        with (hold(item.id) if callable(hold) else contextlib.nullcontext()):
            return self._subtitle_mux_held(item, run, video)

    def _subtitle_mux_held(self, item, run, video):
        from .queue_service import _Cancelled
        folder = self.store.item_dir(item.id) / "subtitles"
        temporary = folder / "mux.partial.mkv"
        try:
            self._subtitle_check(run)
            current = self.store.get(item.id)
            if current is None or current.subtitle.get("status") == "skipped":
                return False
            # No replacement while our player has the file open (even when paused).
            if item.id in self._watched_ids():
                self._subtitle_update(item.id, status="mux_pending", only=True, percent=None, error="")
                return False
            env = getattr(run, "subtitle_env", None) or self._subtitle_item_environment(item)
            run_cfg = getattr(run, "subtitle_config", None) or {**self.store.get_settings(), **item.settings}
            # Only the primary subtitle goes into the mkv (D3); comparison tracks stay sidecars (combos manifest).
            tracks = subtitle_paths(
                folder,
                (item.settings.get("subtitle_configs") or [{}])[0].get("japanese_track",
                                                                       run_cfg.get("subtitle_japanese", False)),
            )
            job = dict(stage="mux", video=video, output=str(temporary), tracks=tracks)
            if (Path(tracks[0][0]).parent / "sakura.zh-TW.full.srt").is_file():
                job["dispositions"] = [index == 0 for index in range(len(tracks))]
            # Comparison tracks embedded by older versions are replaced, not duplicated.
            stale = existing_combo_titles(combos_dir(self.store, item.id))
            if stale:
                job["replace_titles"] = stale
            self._subtitle_stage(item, run, job, env)
            with self._lock:
                self._subtitle_check(run)
                if item.id in self._watched_ids():
                    self._subtitle_update(item.id, status="mux_pending", only=True, percent=None, error="")
                    return False
                with replacement_guard(video):
                    replace_guarded_file(temporary, video)
                self._subtitle_update(
                    item.id,
                    status="done",
                    percent=100,
                    error="",
                    only=False,
                    embedded=True,
                )
                self._safe_run_update(run, size_bytes=Path(video).stat().st_size)
            self._track_cache.pop(video, None)
            return True
        except _Cancelled:
            raise
        except Exception as exc:
            self._subtitle_update(
                item.id,
                status="failed",
                percent=0,
                error=str(exc)[-500:],
                only=False,
            )
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def _subtitle_interrupted(self, item, run):
        """A run that held the GPU lease (restoration worker or the mux step) was stopped. Slice work belongs to
        the executors (their threads settle their own status), so only the mux step needs bookkeeping: a mux that
        was interrupted by the queue (not by the user) is retried when the player is closed."""
        with self._lock:
            current = self.store.get(item.id)
            if current is None or item.id in self._tombstones:
                return
            status = current.subtitle.get("status")
            if current.state == "done" and status in ("ready", "mux"):
                self._subtitle_update(item.id, status="mux_pending", only=True, percent=None)
            elif status == "mux":
                self._subtitle_update(item.id, status="mux_pending", only=True, percent=None)

    def _run_subtitle_only(self, item, run):
        """Mux the finished subtitles of a completed video (the only subtitle step that still runs under the GPU
        lease; the slice work is done by the executors)."""
        from .queue_service import _Cancelled
        try:
            folder = self.store.item_dir(item.id) / "subtitles"
            if not (folder / "sakura.zh-TW.srt").is_file():
                self._subtitle_update(item.id, status="failed", error="找不到已完成的字幕檔，請重新處理字幕",
                                      only=False)
            else:
                run.subtitle_env = self._subtitle_item_environment(item)
                run.subtitle_config = {**self.store.get_settings(), **item.settings}
                self._subtitle_mux(item, run, item.output_path)
                if not run.cancel.is_set():
                    self._safe_run_update(
                        run,
                        size_bytes=Path(item.output_path).stat().st_size,
                    )
        except _Cancelled:
            pass
        except Exception as exc:
            self._subtitle_update(
                item.id,
                status="failed",
                error=str(exc)[-500:],
                only=False,
            )
        finally:
            if run.cancel.is_set():
                self._subtitle_interrupted(item, run)
            with self._lock:
                if self._wk is run:
                    self._wk = None
            self._wake.set()

    def _maybe_start_subtitle_only(self):
        """Start the mux of a completed video whose subtitles are finished (status ready / mux_pending)."""
        from .queue_service import _Run
        if self._paused_all or self.is_player_busy() or self.perf.holding:
            return
        for item in self.store.items():
            if (
                item.state != "done"
                or not item.subtitle.get("only")
                or item.subtitle.get("status") not in ("ready", "mux_pending")
            ):
                continue
            with self._lock:
                if self._wk is not None or getattr(self, "_subtitle_model_test_busy", False) or item.id in self._tombstones:
                    return
                if item.id in self._watched_ids():
                    continue
                if item.id in self._compare_busy or item.id in self._reprocessing:
                    continue
                run = _Run(item.id, self._generations.get(item.id, 0))
                run.subtitle_only = True
                run.stage = "waiting_gpu"
                self._wk = run
                run.thread = threading.Thread(
                    target=self._run_subtitle_only,
                    args=(item, run),
                    daemon=True,
                    name=f"subtitle-only-{item.id}",
                )
                run.thread.start()
            return

    def set_subtitles(self, item_id, enabled):
        if not isinstance(enabled, bool):
            return dict(ok=False, error="字幕設定必須為開關值")
        with self._lock:
            item = self.store.get(item_id)
            if (
                not item
                or item.state in ("done", "processing")
                or any(r.item_id == item_id for r in (self._wk,) if r)
            ):
                return dict(ok=False, error="處理中或已完成項目請使用只重跑字幕")
            env = self._subtitle_environment()
            if enabled and not env.get("runtime_available", env["available"]):
                return dict(ok=False, error=env["reason"])
            subtitle = dict(
                status="queued" if enabled else "skipped",
                percent=0,
                error="",
            )
            if enabled:
                try:            # validate only: an item that has not started follows the global settings (no freeze here)
                    derive_subtitle_snapshot(self.store.get_settings(), item.settings)
                except (ValueError, TypeError) as exc:
                    return dict(ok=False, error=getattr(exc, "message", None) or str(exc))
            settings = {**item.settings, "subtitles": enabled}
            for key in PATH_KEYS:
                settings.pop(key, None)
            item.settings = settings
            self.store.update(
                item_id,
                settings=settings,
                subtitle=subtitle,
            )
        if not enabled:
            self._subtitle_halt(item_id, "cancel", wait=0.0)
        self._subtitle_wake(item_id)
        return dict(ok=True)

    def list_subtitle_models(self):
        from oh_streamer.subtitles import model_registry
        return model_registry.list_models(self.store.get_settings())

    def list_subtitle_connections(self):
        from oh_streamer.subtitles.model_registry import list_connections
        return list_connections(self.store.get_settings())

    # -- local subtitle model downloads (one job at a time; starts only on an explicit user call)
    _MODEL_TERMINAL = ("done", "failed", "cancelled")

    def _model_dl(self):
        return self.__dict__.setdefault("_model_dl_state", dict(lock=threading.Lock(), job=None, items=None, cls=None, seen=True))

    def _model_dl_dir(self):
        from oh_streamer.subtitles.environment import runtime_paths
        return Path(runtime_paths(self.store.get_settings())["models"])

    def _model_dl_catalog(self):
        from oh_streamer.subtitles import model_download as md
        return self._model_dl()["items"] or md.catalog()

    def _invalidate_subtitle_environment(self):
        # detect() re-reads the models folder on every call; only the interpreter/Vulkan probes are memoised.
        from oh_streamer.subtitles import environment
        for name in ("_vulkan_devices", "_probe_python", "_probe_runtime_python"):
            fn = getattr(environment, name, None)
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()

    def subtitle_model_downloads(self):
        from oh_streamer.subtitles import model_download as md
        models, items = self._model_dl_dir(), self._model_dl_catalog()
        groups = {"asr": [], "translation": [], "runtime": []}
        for row in md.status(models, items):
            groups.setdefault(row["category"] or "translation", []).append(row)
        defaults = md.default_item_ids(items)
        probe = md.plan(models, [], items)
        return dict(ok=True, models_dir=str(models), groups=groups, default_ids=defaults,
                    free_bytes=probe["free_bytes"], progress=self.subtitle_model_download_progress()["progress"])

    _IN_USE_MESSAGE = "本機字幕正在使用 llama-server，請等字幕處理完成或暫停後再安裝。"

    def _local_subtitle_running(self):
        """A local (llama-server) subtitle run exists in the service's own state (friendly check)."""
        run = getattr(self, "_subtitle_local", None)
        return run is not None

    @staticmethod
    def _processes_in_folder(folder):
        """Exe paths of running processes located inside ``folder`` (read-only; the safety net that also sees
        the live-subtitle path and other app instances). Unreadable processes are ignored."""
        try:
            import psutil
        except ImportError:
            return []
        root = os.path.normcase(os.path.abspath(str(folder)))
        found = []
        for proc in psutil.process_iter(["exe"]):
            try:
                exe = proc.info.get("exe")
                if exe and os.path.normcase(os.path.abspath(exe)).startswith(root + os.sep):
                    found.append(exe)
            except Exception:
                continue
        return found

    def _model_dl_in_use(self, models, ids, items):
        """Message when the selection would overwrite files of a running llama-server, else ''. Only zip items
        that are not installed yet extract into an existing folder; plain model files are written to new paths
        (installed ones are skipped), so they never collide with open files."""
        from oh_streamer.subtitles import model_download as md
        folders = {item.id: {f.extract_to for f in item.files if f.extract_to}
                   for item in items if item.id in ids and not md._item_installed(models, item)}
        folders = {i: f for i, f in folders.items() if f}
        if not folders:
            return ""
        if self._local_subtitle_running():
            return self._IN_USE_MESSAGE
        for names in folders.values():
            for name in names:
                if self._processes_in_folder(Path(models) / name):
                    return self._IN_USE_MESSAGE
        return ""

    def plan_subtitle_model_download(self, item_ids):
        from oh_streamer.subtitles import model_download as md
        models, items = self._model_dl_dir(), self._model_dl_catalog()
        try:
            rules = md.check_selection(models, item_ids or [], items)
            plan = md.plan(models, rules["ids"], items)
        except md.DownloadError as exc:
            return dict(ok=False, message=str(exc), download_bytes=0, free_bytes=0, ids=[])
        plan["ids"] = rules["ids"]
        busy = self._model_dl_in_use(models, set(rules["ids"]), items)
        if busy:
            plan.update(ok=False, message=busy, in_use=True)
            return plan
        if not rules["ok"]:
            plan.update(ok=False, message=rules["message"])
        return plan

    def start_subtitle_model_download(self, item_ids):
        from oh_streamer.subtitles import model_download as md
        state = self._model_dl()
        with state["lock"]:
            job = state["job"]
            if job is not None and job.snapshot()["state"] not in self._MODEL_TERMINAL:
                return dict(ok=False, error="已有模型正在下載。")
            plan = self.plan_subtitle_model_download(item_ids)
            if not plan.get("ok"):
                return dict(ok=False, error=plan.get("message") or "無法開始下載。")
            if not plan["ids"] or plan["download_bytes"] <= 0:
                return dict(ok=False, error="所選模型都已安裝，不需要下載。")
            cls = state["cls"] or md.DownloadJob
            job = cls(self._model_dl_dir(), plan["ids"], state["items"])
            state["job"], state["seen"] = job, False
            job.start()
        return dict(ok=True, progress=self.subtitle_model_download_progress()["progress"])

    def cancel_subtitle_model_download(self):
        job = self._model_dl()["job"]
        if job is not None:
            job.cancel()
        return dict(ok=True)

    def subtitle_model_download_progress(self):
        state = self._model_dl()
        job = state["job"]
        if job is None:
            return dict(ok=True, progress=None)
        snap = job.snapshot()
        if snap["state"] in self._MODEL_TERMINAL and not state["seen"]:
            state["seen"] = True
            if snap["state"] == "done":
                self._invalidate_subtitle_environment()
        return dict(ok=True, progress=snap)

    def subtitle_usage(self):
        """Groq usage snapshot (contract version 1); never raises, errors come back as a dict."""
        try:
            from oh_streamer.subtitles.usage_monitor import usage_snapshot
            return usage_snapshot()
        except Exception as exc:
            return dict(version=1, providers=[], error=f"{type(exc).__name__}: {exc}")

    def test_save_subtitle_connection(self, spec):
        from oh_streamer.subtitles.model_registry import test_save_connection
        from oh_streamer.subtitles.combos import is_cloud_enabled
        if not is_cloud_enabled(self.store.get_settings()):
            self.store.set_settings(subtitle_cloud_enabled=True)
        return test_save_connection(spec, self.store.get_settings())

    def remove_subtitle_connection(self, connection_id):
        from oh_streamer.subtitles.model_registry import remove_connection, _document
        from oh_streamer.subtitles.configuration import references
        doc = _document(self.store.get_settings())
        associated_models = [m['id'] for m in doc.get('models', []) if m.get('connection_id') == connection_id]
        targets = [connection_id, *associated_models]
        if any(any(references(_guard_settings(item, self.store.get_settings()), t, include_history=False) for t in targets) for item in self.store.items()):
            return dict(ok=False, code='connection_in_use', error='服務連線仍被佇列引用')
        saved = self.store.get_settings()
        if any(references(saved, t, include_history=False) for t in targets):
            return dict(ok=False, code='connection_in_use', error='服務連線或其模型仍被字幕設定引用')
        return remove_connection(connection_id, self.store.get_settings())

    def set_subtitle_profile(self, profile):
        from oh_streamer.subtitles.configuration import project_connection_pool, resolve, ConfigurationError
        from oh_streamer.subtitles.combos import is_cloud_enabled
        try:
            profile = project_connection_pool(profile, self.store.get_settings())
            if profile.get('active_mode') == 'cloud' and not is_cloud_enabled(self.store.get_settings()):
                self.store.set_settings(subtitle_cloud_enabled=True)
            resolved = resolve(profile, self.store.get_settings())
            if any(e['code'] == 'local_mode_cloud_candidate' for e in resolved['errors']):
                return dict(ok=False, code='local_mode_cloud_candidate', error='混合雲端模型請使用自訂模式')
            self.store.set_settings(subtitle_profile=profile, subtitle_runtime_contract='cloud-first-1')
            _sync_legacy_primary(self.store)
            self._subtitle_thaw_all(refresh=True)       # new backups/models apply to jobs waiting on a limit
            return dict(ok=True, profile=profile, resolved=resolved, runtime_contract='cloud-first-1')
        except (ConfigurationError, TypeError, ValueError):
            return dict(ok=False, code='invalid_configuration', error='字幕配置格式無效')

    def remove_subtitle_pool_connection(self, connection_id):
        """Remove a managed connection and its models, protecting frozen jobs/other profiles."""
        from oh_streamer.subtitles import model_registry as registry
        from oh_streamer.subtitles.configuration import project_connection_pool, references, ConfigurationError
        with self._lock, registry._LOCK:
            saved = self.store.get_settings()
            doc = registry._document(saved)
            conn = next((c for c in doc['connections'] if c['id'] == connection_id), None)
            if not conn or conn.get('api_family') != 'groq':
                return dict(ok=False, error='找不到 Groq 連線')
            targets = [connection_id, *(m['id'] for m in doc['models'] if m.get('connection_id') == connection_id)]
            if any(any(references(_guard_settings(item, saved), t, include_history=False) for t in targets) for item in self.store.items()):
                return dict(ok=False, code='connection_in_use', error='此連線仍被佇列工作使用，請先移除相關工作')
            profile = copy.deepcopy(saved['subtitle_profile'])
            cloud = profile['profiles']['cloud']
            pool = cloud.get('connection_pool')
            if pool is None:
                return dict(ok=False, error='請先儲存連線池設定，再刪除連線')
            pool['connections'] = [c for c in pool['connections'] if c['id'] != connection_id]
            try:
                profile = project_connection_pool(profile, saved)
            except ConfigurationError:
                return dict(ok=False, error='連線池設定無效，請重新整理後重試')
            prospective = {**saved, 'subtitle_profile': profile}
            if any(references(prospective, t, include_history=False) for t in targets):
                return dict(ok=False, code='connection_in_use', error='此連線仍被其他字幕設定使用，請先更換引用的模型')
            self.store.set_settings(subtitle_profile=profile)
            try:
                result = registry.remove_connection(connection_id, saved, remove_models=True)
                if not result.get('ok'):
                    self.store.set_settings(subtitle_profile=saved['subtitle_profile'])
                    return result
            except Exception:
                self.store.set_settings(subtitle_profile=saved['subtitle_profile'])
                return dict(ok=False, error='無法刪除連線，已保留原有設定')
        self._subtitle_thaw_all(refresh=True)
        return dict(ok=True, profile=profile)

    def test_add_subtitle_model(self, spec):
        from oh_streamer.subtitles import model_registry
        local = isinstance(spec, dict) and spec.get("provider") == "local"
        if not local:
            return model_registry.test_and_add_model(spec, self.store.get_settings())
        with self._lock:
            if (getattr(self, "_subtitle_model_test_busy", False) or self._wk is not None
                    or self.is_player_busy() or self.perf.holding):
                return dict(ok=False, error="請先等待本機處理或模型測試完成，並關閉播放器與效能測試，再測試本機字幕模型")
            self._subtitle_model_test_busy = True
        try:
            return model_registry.test_and_add_model(spec, self.store.get_settings())
        finally:
            with self._lock:
                self._subtitle_model_test_busy = False
            self._wake.set()

    def remove_subtitle_model(self, model_id):
        from oh_streamer.subtitles import model_registry
        from oh_streamer.subtitles.configuration import references as contains_reference
        def references(value):
            return contains_reference(value, model_id, include_history=False)
        saved = self.store.get_settings()
        global_rows = [saved.get("subtitle_primary"), saved.get("subtitle_comparison_rows"), saved.get('subtitle_profile')]
        if references(global_rows) or any(references(_guard_settings(item, self.store.get_settings())) for item in self.store.items()):
            return dict(ok=False, error="模型仍被字幕設定或佇列項目使用，請先更換引用的模型")
        return model_registry.remove_model(model_id, saved)

    def update_subtitle_model_key(self, model_id, api_key):
        from oh_streamer.subtitles import model_registry
        result = model_registry.update_model_key(model_id, api_key, self.store.get_settings())
        if isinstance(result, dict) and result.get("ok", True):
            self._subtitle_thaw_all()                   # a corrected key may unblock jobs that waited for it
        return result

    def set_subtitle_configuration(self, primary, comparison_enabled, rows, item_id=None):
        try:
            config = subtitle_configuration(dict(subtitle_primary=primary, subtitle_comparison_enabled=comparison_enabled, subtitle_comparison_rows=rows))
        except (ValueError, TypeError) as exc:
            return dict(ok=False, error=str(exc))
        from oh_streamer.subtitles.combos import is_cloud_combo, is_cloud_enabled
        saved = self.store.get_settings()
        cloud_active = is_cloud_enabled(saved)
        active_rows = [config["subtitle_primary"]] + (config["subtitle_comparison_rows"] if config["subtitle_comparison_enabled"] else [])
        for row in active_rows:
            if is_cloud_combo(row) and not cloud_active:
                return dict(
                    ok=False,
                    error=f"雲端字幕功能未開啟，組合「{row.get('label') or row.get('id')}」包含雲端模型。請先在設定中啟用「允許使用雲端字幕後端」或設定環境變數 OH_STREAMER_SUBTITLE_CLOUD=1"
                )
        if item_id is None:
            # The retained editor edits the active plan, including output options.
            # Keep the other modes and policy intact; existing jobs keep snapshots.
            if saved.get('subtitle_runtime_contract') == 'cloud-first-1':
                from oh_streamer.subtitles.configuration import _profile, resolve
                profile = copy.deepcopy(saved['subtitle_profile'])
                active = profile['profiles'][profile['active_mode']]
                edited = _profile(config['subtitle_primary'])
                for key in ('asr', 'translation', 'legacy_primary'):
                    # Cloud's retained advanced editor only changes output options.
                    # Its legacy defaults must not fill unconfigured service slots.
                    if profile['active_mode'] == 'cloud' and key != 'legacy_primary' and active[key] is None:
                        continue
                    active[key] = edited[key]
                resolved = resolve(profile, saved)
                if any(e['code'] == 'local_mode_cloud_candidate' for e in resolved['errors']):
                    return dict(ok=False, code='local_mode_cloud_candidate', error='混合雲端模型請使用自訂模式')
                config['subtitle_profile'] = profile
            self.store.set_settings(**config)
            if saved.get('subtitle_runtime_contract') == 'cloud-first-1':
                _sync_legacy_primary(self.store)
            self._subtitle_thaw_all(refresh=True)
        else:
            if not self.store.get(item_id):
                return dict(ok=False, error="unknown id")
            if self._subtitle_mux_busy(item_id):
                return dict(ok=False, error="請先暫停字幕工作")
            if not self._subtitle_halt(item_id, "pause"):
                return dict(ok=False, error="字幕工作尚未停止，請稍後再試")
            with self._lock:
                item = self.store.get(item_id)
                if not item:
                    return dict(ok=False, error="unknown id")
                if self._subtitle_executor_running(item_id) or self._subtitle_mux_busy(item_id):
                    return dict(ok=False, error="請先暫停字幕工作")
                settings = {**item.settings, **config}
                settings.pop("subtitle_configs", None)
                for key in ('subtitle_profile', 'subtitle_resolved', 'subtitle_runtime_contract', 'subtitle_track_plans'):
                    settings.pop(key, None)
                settings = snapshot_subtitle_settings(settings)
                self.store.update(item_id, settings=settings, subtitle=self._subtitle_after_new_rows(item))
            self._subtitle_wake(item_id)
        return dict(ok=True, **config)

    def list_subtitle_combos(self):
        """[{id, label, builtin, asr, translation, approx_gpu_hint, available, missing, unavailable_reason, cloud}]
        for the UI picker (builtin first, then user). Availability is read from disk on every call, so a finished
        model download shows up without a restart."""
        return list_combos(self.store.get_settings())

    def subtitle_combo_fallback(self):
        """{fallback: {to_id,to_label,from_label,missing,message}|None} for the saved primary subtitle row: set when
        new items would run on another combo because the wanted one lacks downloaded models."""
        settings = self.store.get_settings()
        try:
            snap = snapshot_subtitle_settings({k: v for k, v in settings.items() if k.startswith("subtitle_") and k not in ("subtitle_configs", "subtitle_combo_fallback")})
        except Exception:
            return dict(ok=True, fallback=None)
        return dict(ok=True, fallback=snap.get("subtitle_combo_fallback"))

    def set_subtitle_combos(self, item_id, ids):
        """Pick the combos an item generates; [] returns it to the legacy single pipeline.

        Refused while the item's subtitle step is running or reserved (change it after pause/cancel); allowed on done
        items so 'rerun subtitles' can use the new selection.
        """
        from oh_streamer.subtitles import combos
        presets, _errors = load_combo_presets()
        checked, error = validate_combo_ids(ids, presets)
        if checked is None:
            return dict(ok=False, error=error)
        with self._lock:
            item = self.store.get(item_id)
            if not item or item_id in self._tombstones:
                return dict(ok=False, error="找不到佇列項目")
            if item.state == "processing" or self._subtitle_mux_busy(item_id):
                return dict(ok=False, error="字幕處理中，請先暫停或取消後再變更字幕組合")
            if checked:
                env = self._subtitle_environment()
                if not env.get("runtime_available", env["available"]):
                    return dict(ok=False, error=env.get("runtime_reason") or env["reason"])
        if not self._subtitle_halt(item_id, "pause"):
            return dict(ok=False, error="字幕工作尚未停止，請稍後再試")
        with self._lock:
            item = self.store.get(item_id)
            if not item or item_id in self._tombstones or self._subtitle_executor_running(item_id):
                return dict(ok=False, error="找不到佇列項目")
            settings = {**item.settings}
            if checked:
                settings["subtitle_combos"] = checked
                settings["subtitle_configs"] = [copy.deepcopy(presets[key]) for key in checked]
            else:
                settings.pop("subtitle_combos", None)
                settings.pop("subtitle_configs", None)
            freeze_track_plans(settings)
            self.store.update(item_id, settings=settings, subtitle=self._subtitle_after_new_rows(item))
        self._subtitle_wake(item_id)
        return dict(ok=True, ids=checked)

    def _subtitle_after_new_rows(self, item):
        """Subtitle dict of an item whose subtitle rows changed: legacy and published fields dropped, and (unless
        the user parked the item) queued again with a fresh slice table. The content-addressed cache stays, so
        only what the new rows need is recomputed."""
        from oh_streamer.subtitles import slice_plan as sp
        subtitle = without_legacy(item.subtitle, publish=True)
        if subtitle.get("status") not in ("done", "skipped", "paused", "cancelled"):
            subtitle.update(status="queued", percent=0, error="", phase="", detail="")
            (self.store.item_dir(item.id) / "subtitles" / sp.SLICE_PLAN_FILE).unlink(missing_ok=True)
        return subtitle

    def set_default_subtitle_combos(self, ids):
        """Global default for new subtitle items; [] = legacy pipeline. Existing items are untouched."""
        checked, error = validate_combo_ids(ids)
        if checked is None:
            return dict(ok=False, error=error)
        self.store.set_settings(subtitle_combos_default=checked)
        return dict(ok=True, ids=checked)

    def rerun_subtitles(self, item_id):
        with self._lock:
            item = self.store.get(item_id)
            if (
                not item
                or item.state != "done"
                or self._running_runs(item_id)
                or self._subtitle_executor_running(item_id)
                or item_id in self._compare_busy
                or item_id in self._reprocessing
            ):
                return dict(ok=False, error="只重跑字幕需要已完成的影片")
            if item.id in self._watched_ids():
                return dict(ok=False, error="請先關閉播放器中的影片")
            env = self._subtitle_item_environment(item)
            if not env.get("runtime_available", env["available"]):
                return dict(ok=False, error=env.get("runtime_reason") or env["reason"])
            # A new slice table; the content-keyed cache (subtitles/cache/) stays, so a rerun only redoes what is
            # missing or changed by the settings.
            from oh_streamer.subtitles import slice_plan as sp
            folder = self.store.item_dir(item_id) / "subtitles"
            for name in (sp.SLICE_PLAN_FILE, "sakura.translate.partial.json", "sakura.zh-TW.json"):
                (folder / name).unlink(missing_ok=True)
            self.store.update(
                item_id,
                settings={**item.settings, "subtitles": True},
                subtitle={
                    **without_legacy(item.subtitle, publish=True),
                    "status": "queued",
                    "percent": 0,
                    "error": "",
                    "only": True,
                },
            )
            self.store.protected_ids = set(self.store.protected_ids) | {item_id}
        self._subtitle_wake(item_id)
        return dict(ok=True)

    def cancel_subtitles(self, item_id):
        with self._lock:
            item = self.store.get(item_id)
            if not item or item.state != "done" or not item.subtitle.get("only"):
                return dict(ok=False, error="沒有進行中的只重跑字幕")
            self._subtitle_update(item_id, status="cancelled", percent=0, only=False)
            for run in self._running_runs(item_id):
                self._request_stop(run, "cancel")
        self._subtitle_halt(item_id, "cancel", wait=0.0)
        self._subtitle_wake(item_id)
        return dict(ok=True)

    def _skip_subtitles_for_play(self, item_id):
        """Playing starts: the play-after-subtitles wait ends and a running mux yields to the player. The
        subtitle work itself is never skipped (the executors keep filling the slice table while it plays)."""
        with self._lock:
            item = self.store.get(item_id)
            if item and getattr(item, "play_after_subtitles", False):
                self.store.update(item_id, play_after_subtitles=False, play_after_subtitles_start=None)
            if item and item.subtitle.get("status") == "mux":
                # The complete subtitles are already available. Opening playback
                # interrupts only replacement and leaves a deferred mux to retry.
                self._subtitle_update(item_id, status="mux_pending", only=True, percent=None)
                run = self._wk
                if run and run.item_id == item_id and run.stage == "mux":
                    self._request_stop(run, "preempt")
                return
