"""Subtitle combos: settings schema, presets and manifest helpers.

A *combo* selects ASR model/decoding, an optional correction stage, a translator
profile, sentence/short-reply options and a glossary. Since M7b the whole-file
pipeline that used to live here is gone: comparison tracks are produced by the
sliced scheduler (``queue/subtitle_scheduler.py``), which writes
``subtitles/combos/<id>.zh-TW.srt`` + ``manifest.json`` (contract in
the manifest contract, relied upon by the player).
This module keeps the combo/preset normalization, model-file tables and manifest
reader. Importing it does not import torch or any model runtime.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from . import runtime


MANIFEST_FORMAT = "oh_streamer.subtitle_combos"
SCHEMA_VERSION = 1
USER_PRESETS_FORMAT_VERSION = 1
OUTPUT_SUFFIX = ".subtitle-combos"
REPRODUCIBLE_SEED = 42
CORRECTION_MODES = ("none", "llm")

ASR_ALIASES = (
    "anime-whisper",
    "kotoba",
    "turbo",
    "whisper-ja-760M",
    "whisper-ja-1.5B",
    "whisper-large-v3",
    "qwen3-asr-1.7b",
)
# Repetition guard for the large OpenAI checkpoints (realistic PoC: ngram6).
# whisper-large-v3 already carries it in runtime.MODEL_SPECS; turbo's shared spec
# stays unchanged for the existing queue/PoC, so combos add it as a decode default.
ASR_DEFAULT_DECODE = {"turbo": {"no_repeat_ngram_size": 6}}

TRANSLATION_PROFILES = (
    "sakura-v1",
    "sakura-galtransl-7b-v3.7",
    "sakura-galtransl-14b-v3.8",
    "qwen3",
)
TRANSLATION_FILES = {
    "sakura-v1": "sakura/sakura-7b-qwen2.5-v1.0-iq4xs.gguf",
    "sakura-galtransl-7b-v3.7": "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
    "sakura-galtransl-14b-v3.8": "sakura/Sakura-Galtransl-14B-v3.8-IQ4_XS.gguf",
    "qwen3": "qwen3/Qwen3-8B-Q4_K_M.gguf",
}
# 14B v3.8 IQ4_XS needs partial offload on 8 GB (round-4 measured start point).
TRANSLATION_GPU_LAYERS = {"sakura-galtransl-14b-v3.8": 28}
CORRECTION_FILES = {"qwen3": "qwen3/Qwen3-8B-Q4_K_M.gguf", "sakura": TRANSLATION_FILES["sakura-v1"]}

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_TOP_LEVEL_KEYS = {
    "id", "label", "asr", "correction", "translation", "sentence", "glossary",
    "japanese_track", "description", "experimental",
    "cloud_strategy",
}

COMBO_TEMPLATE: dict[str, Any] = {
    "id": None,
    "label": None,
    "description": "",
    "experimental": False,
    "cloud_strategy": "smart_horizon",
    "asr": {
        "mode": "diagnostic",  # historical explicit comparison models
        "model": "anime-whisper",
        "decode": None,  # None = per-model default (ASR_DEFAULT_DECODE)
        "segmentation": "utterance",
        "vad_threshold": 0.3,
        "vad_rescue_chunk_seconds": 60.0,
    },
    "correction": {
        "mode": "none",
        "backend": "qwen3",
        "model_file": None,
        "prompt": "v1",
        "candidates": [],
    },
    "translation": {
        "mode": "standard",
        "debounce_cues": 24,
        "model_id": None,  # registered local or cloud model; credentials stay in the registry
        "backend": None,  # derived from profile
        "profile": "sakura-galtransl-7b-v3.7",
        "model_file": None,  # None = TRANSLATION_FILES[profile]
        "gpu_layers": None,  # None = TRANSLATION_GPU_LAYERS.get(profile, "auto")
        "context_size": 8192,
        "batch_size": 30,
        "prompt_mode": "official-numbered",
        # Legacy fields remain readable; every production translation is now
        # temperature zero with fixed seed and prompt caching disabled.
        "reproducible": False,
        "seed": None,  # only with reproducible; None = seed 42
        "temperature": None,  # None = profile default (Sakura v1.0 0.1, GalTransl 0.3)
    },
    "sentence": {
        "sentence_mode": True,
        "short_threshold": 1.0,
        "short_policy": "whitelist",
        "show_interjections": False,
        # Reserved for the short-reply option being added to sentences.py by a
        # parallel task; it is part of the cache identity but not yet passed on.
        "short_replies": None,
    },
    "glossary": {"generic_only": True, "path": None},
    "japanese_track": True,
}

BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "ai-default", "label": "雙模型 whisper-ja-1.5B＋anime-whisper＋GalTransl 7B",
        "description": "預設品質管線：雙模型融合，精簡與完整兩條字幕軌",
        "asr": {"mode": "dual", "model": "whisper-ja-1.5B"},
    },
    {
        "id": "ai-single", "label": "whisper-ja-1.5B＋GalTransl 7B",
        "description": "較快的單模型管線，精簡與完整兩條字幕軌",
        "asr": {"mode": "single", "model": "whisper-ja-1.5B"},
    },
    {
        "id": "anime-sakura1",
        "label": "anime-whisper＋Sakura v1.0",
        "description": "舊版 Sakura v1.0 管線",
        "translation": {"profile": "sakura-v1"},
    },
    {
        "id": "ja15b-sakura1",
        "label": "whisper-ja-1.5B＋Sakura v1.0",
        "asr": {"model": "whisper-ja-1.5B"},
        "translation": {"profile": "sakura-v1"},
    },
    {
        "id": "anime-gt14b",
        "label": "anime-whisper＋GalTransl 14B v3.8",
        "translation": {"profile": "sakura-galtransl-14b-v3.8"},
    },
    {
        "id": "ja15b-gt14b",
        "label": "whisper-ja-1.5B＋GalTransl 14B v3.8",
        "asr": {"model": "whisper-ja-1.5B"},
        "translation": {"profile": "sakura-galtransl-14b-v3.8"},
    },
    {
        "id": "anime-gt7b",
        "label": "anime-whisper＋GalTransl 7B v3.7",
        "translation": {"profile": "sakura-galtransl-7b-v3.7"},
    },
    {
        "id": "anime-llmfix-sakura1",
        "label": "anime-whisper＋LLM 校正＋Sakura v1.0（實驗）",
        "description": "佔位：需要 subtitles/correction.py；缺少時此組失敗並註明原因",
        "experimental": True,
        "correction": {"mode": "llm", "candidates": ["whisper-ja-1.5B"]},
        "translation": {"profile": "sakura-v1"},
    },
]


class ComboError(ValueError):
    """Invalid combo definition or selection."""



def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        # Deep copy: normalize_combo mutates the result, which must never alias COMBO_TEMPLATE
        # (a shared None default like asr.decode/translation.gpu_layers was overwritten for later combos).
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def canonical_asr_alias(name: Any) -> str:
    lowered = str(name).strip().lower()
    for alias in ASR_ALIASES:
        if alias.lower() == lowered:
            return alias
    if lowered.startswith("user-") and _ID_PATTERN.fullmatch(lowered):
        return lowered
    raise ComboError(f"unknown ASR model {name!r}; choose {', '.join(ASR_ALIASES)}")


def _require(condition: bool, combo_id: Any, field_name: str, message: str) -> None:
    if not condition:
        raise ComboError(f"combo {combo_id!r}: {field_name} {message}")


def _json_value(value: Any, combo_id: Any, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ComboError(f"combo {combo_id!r}: {field_name} must be JSON data") from exc


def normalize_combo(raw: Any, *, source: str = "user") -> dict[str, Any]:
    """Validate one combo definition and fill every default explicitly."""

    if not isinstance(raw, dict):
        raise ComboError("combo definition must be a JSON object")
    # Normalized settings are saved and normalized again on load.
    # "source" is derived; "cloud_horizon_minutes" is an obsolete key (M7b cleanup) from older combo files: ignored.
    raw = {key: value for key, value in raw.items() if key not in ("source", "cloud_horizon_minutes")}
    combo_id = raw.get("id")
    if isinstance(raw.get("translation"), dict) and (
            raw["translation"].get("mode") == "tiered" or "baseline_model_id" in raw["translation"]
            or "refine_model_id" in raw["translation"]):
        # D2: the draft -> refine flow is gone. A legacy 'tiered' combo becomes an ordinary one that
        # translates with its former refine model; the draft (baseline) model is dropped.
        legacy_tr = dict(raw["translation"])
        refine = legacy_tr.pop("refine_model_id", None)
        legacy_tr.pop("baseline_model_id", None)
        if legacy_tr.get("mode") == "tiered":
            if refine and not legacy_tr.get("model_id"):
                legacy_tr["model_id"] = refine
            if legacy_tr.get("batch_size") == 12 and not legacy_tr.get("model_id"):
                legacy_tr["batch_size"] = 30
        legacy_tr["mode"] = "standard"
        raw = {**raw, "translation": legacy_tr}
    unknown = set(raw) - _TOP_LEVEL_KEYS
    _require(not unknown, combo_id, "keys", f"are unknown: {', '.join(sorted(unknown))}")
    _require(isinstance(combo_id, str) and bool(_ID_PATTERN.match(combo_id)), combo_id, "id",
             "must be 1-64 lower-case letters, digits, '.', '_' or '-' and start with a letter/digit")
    for section in ("asr", "correction", "translation", "sentence", "glossary"):
        _require(section not in raw or isinstance(raw[section], dict), combo_id, section, "must be an object")
        unknown_fields = set(raw.get(section) or {}) - set(COMBO_TEMPLATE[section])
        _require(not unknown_fields, combo_id, section,
                 "has unknown fields: " + ", ".join(sorted(unknown_fields)))
    combo = _merge(COMBO_TEMPLATE, raw)
    label = combo["label"]
    _require(isinstance(label, str) and bool(label.strip()), combo_id, "label", "must be a non-empty string")
    combo["label"] = label.strip()
    _require(isinstance(combo["description"], str), combo_id, "description", "must be a string")
    _require(isinstance(combo["experimental"], bool), combo_id, "experimental", "must be true/false")
    _require(isinstance(combo["japanese_track"], bool), combo_id, "japanese_track", "must be true/false")

    asr = combo["asr"]
    _require(asr["mode"] in ("dual", "single", "diagnostic"), combo_id, "asr.mode",
             "must be dual, single or diagnostic")
    if asr["mode"] == "dual":
        asr["model"] = "whisper-ja-1.5B"
    if asr["mode"] != "diagnostic":
        asr["segmentation"] = "utterance"
    try:
        asr["model"] = canonical_asr_alias(asr["model"])
    except ComboError as exc:
        raise ComboError(f"combo {combo_id!r}: {exc}") from None
    if asr["decode"] is None:
        asr["decode"] = dict(ASR_DEFAULT_DECODE.get(asr["model"], {}))
    _require(isinstance(asr["decode"], dict), combo_id, "asr.decode", "must be an object")
    _json_value(asr["decode"], combo_id, "asr.decode")
    _require(asr["segmentation"] in ("utterance", "grouped"), combo_id, "asr.segmentation",
             "must be utterance or grouped")
    _require(asr["model"] != "qwen3-asr-1.7b" or asr["segmentation"] == "utterance", combo_id,
             "asr.segmentation", "must be utterance for qwen3-asr-1.7b")
    _require(isinstance(asr["vad_threshold"], (int, float)) and 0 < asr["vad_threshold"] < 1,
             combo_id, "asr.vad_threshold", "must be between 0 and 1")
    _require(isinstance(asr["vad_rescue_chunk_seconds"], (int, float)) and asr["vad_rescue_chunk_seconds"] >= 0,
             combo_id, "asr.vad_rescue_chunk_seconds", "must be non-negative")
    asr["vad_threshold"] = float(asr["vad_threshold"])
    asr["vad_rescue_chunk_seconds"] = float(asr["vad_rescue_chunk_seconds"])

    correction = combo["correction"]
    _require(correction["mode"] in CORRECTION_MODES, combo_id, "correction.mode",
             "must be one of " + ", ".join(CORRECTION_MODES))
    if correction["mode"] == "none":
        combo["correction"] = {"mode": "none"}
    else:
        _require(correction["backend"] in ("qwen3", "sakura"), combo_id, "correction.backend",
                 "must be qwen3 or sakura")
        if correction["model_file"] is None:
            correction["model_file"] = CORRECTION_FILES[correction["backend"]]
        _require(isinstance(correction["model_file"], str) and bool(correction["model_file"]), combo_id,
                 "correction.model_file", "must be a path relative to the models folder")
        _require(isinstance(correction["prompt"], str) and bool(correction["prompt"]), combo_id,
                 "correction.prompt", "must be a non-empty prompt id")
        _require(isinstance(correction["candidates"], list), combo_id, "correction.candidates", "must be a list")
        try:
            correction["candidates"] = [canonical_asr_alias(alias) for alias in correction["candidates"]]
        except ComboError as exc:
            raise ComboError(f"combo {combo_id!r}: correction.candidates: {exc}") from None

    translation = combo["translation"]
    _require(translation.get("mode", "standard") == "standard", combo_id, "translation.mode", "must be standard")
    translation["mode"] = "standard"
    _require(isinstance(translation.get("debounce_cues"), int) and not isinstance(translation.get("debounce_cues"), bool)
             and translation["debounce_cues"] > 0,
             combo_id, "translation.debounce_cues", "must be a positive integer")
    registered_id = translation.get("model_id")
    _require(registered_id is None or (isinstance(registered_id, str)
             and (registered_id.startswith("user-") or registered_id in TRANSLATION_PROFILES) and bool(_ID_PATTERN.fullmatch(registered_id))),
             combo_id, "translation.model_id", "must be a registered or built-in translation model ID or null")
    # Preserve the legacy serialized recipe for existing presets and caches.
    if registered_id is None:
        translation.pop("model_id", None)
    profile = str(translation["profile"]).strip().lower()
    profile = {"v1": "sakura-v1", "v1.0": "sakura-v1"}.get(profile, profile)
    _require(profile in TRANSLATION_PROFILES, combo_id, "translation.profile",
             "must be one of " + ", ".join(TRANSLATION_PROFILES))
    translation["profile"] = profile
    expected_backend = "qwen3" if profile == "qwen3" else "sakura"
    if translation["backend"] is None:
        translation["backend"] = expected_backend
    _require(translation["backend"] == expected_backend, combo_id, "translation.backend",
             f"must be {expected_backend} for profile {profile}")
    if translation["model_file"] is None:
        translation["model_file"] = TRANSLATION_FILES[profile]
    _require(isinstance(translation["model_file"], str) and bool(translation["model_file"]), combo_id,
             "translation.model_file", "must be a path relative to the models folder")
    if translation["gpu_layers"] is None:
        translation["gpu_layers"] = TRANSLATION_GPU_LAYERS.get(profile, "auto")
    layers = translation["gpu_layers"]
    _require(layers in ("auto", "all") or (isinstance(layers, int) and not isinstance(layers, bool) and layers >= 0),
             combo_id, "translation.gpu_layers", "must be auto, all or a non-negative integer")
    _require(isinstance(translation["context_size"], int) and translation["context_size"] > 0, combo_id,
             "translation.context_size", "must be a positive integer")
    bs = translation["batch_size"]
    allow_wide = bool(translation.get("model_id"))
    _require(isinstance(bs, int) and not isinstance(bs, bool) and (1 <= bs <= 100 if allow_wide else 20 <= bs <= 40),
             combo_id, "translation.batch_size", "must be 20-40 (or 1-100 for registered models)")
    _require(translation["prompt_mode"] in ("legacy", "official-lines", "official-numbered", "general-llm"), combo_id,
             "translation.prompt_mode", "is not supported")
    _require(profile in ("sakura-v1", "qwen3") or translation["prompt_mode"] not in ("legacy",), combo_id,
             "translation.prompt_mode", "legacy is not allowed with GalTransl profiles")
    _require(isinstance(translation["reproducible"], bool), combo_id, "translation.reproducible",
             "must be true/false")
    seed = translation["seed"]
    _require(seed is None or (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2 ** 31),
             combo_id, "translation.seed", "must be an integer between 0 and 2147483647")
    _require(seed is None or translation["reproducible"], combo_id, "translation.seed",
             "requires translation.reproducible=true")
    temperature = translation["temperature"]
    _require(temperature is None or (isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
                                     and 0 <= temperature <= 2),
             combo_id, "translation.temperature", "must be a number between 0 and 2")
    if temperature is not None:
        translation["temperature"] = float(temperature)
    # Keep old preset files readable; effective sampling is always included in
    # the stage identity below, even when these legacy fields are absent.
    for field_name, default in (("reproducible", False), ("seed", None), ("temperature", None)):
        if translation[field_name] == default and type(translation[field_name]) is type(default):
            del translation[field_name]

    sentence = combo["sentence"]
    _require(isinstance(sentence["sentence_mode"], bool), combo_id, "sentence.sentence_mode", "must be true/false")
    _require(isinstance(sentence["show_interjections"], bool), combo_id, "sentence.show_interjections",
             "must be true/false")
    _require(isinstance(sentence["short_threshold"], (int, float)) and sentence["short_threshold"] >= 0,
             combo_id, "sentence.short_threshold", "must be non-negative")
    sentence["short_threshold"] = float(sentence["short_threshold"])
    _require(sentence["short_policy"] in ("keep", "whitelist", "drop"), combo_id, "sentence.short_policy",
             "must be keep, whitelist or drop")
    _json_value(sentence["short_replies"], combo_id, "sentence.short_replies")

    glossary = combo["glossary"]
    _require(isinstance(glossary["generic_only"], bool), combo_id, "glossary.generic_only", "must be true/false")
    _require(glossary["path"] is None or (isinstance(glossary["path"], str) and bool(glossary["path"])),
             combo_id, "glossary.path", "must be a file path or null")
    _require(not (glossary["generic_only"] and glossary["path"]), combo_id, "glossary",
             "cannot combine generic_only with a path")

    strat = combo.get("cloud_strategy", "smart_horizon")
    _require(strat in ("smart_horizon", "full_preload"), combo_id, "cloud_strategy",
             "must be 'smart_horizon' or 'full_preload'")
    combo["cloud_strategy"] = strat

    combo["source"] = source
    return combo


def user_presets_path() -> Path:
    configured = os.environ.get("OH_STREAMER_SUBTITLE_COMBOS")
    if configured:
        return Path(configured).expanduser()
    try:
        from oh_streamer.platform.platform_support import data_dir
        return data_dir() / "subtitle_combos.json"
    except Exception:  # keep the subtitle runtime usable without the app package
        return runtime.default_cache_dir().parents[1] / "subtitle_combos.json"


def load_presets(user_file: str | os.PathLike[str] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return ``({id: combo}, errors)``; builtin presets first, then valid user presets."""

    presets: dict[str, dict[str, Any]] = {}
    for raw in BUILTIN_PRESETS:
        combo = normalize_combo(raw, source="builtin")
        presets[combo["id"]] = combo
    errors: list[str] = []
    path = Path(user_file) if user_file is not None else user_presets_path()
    if not path.is_file():
        return presets, errors
    try:
        document = json.loads(path.read_text("utf-8-sig"))
    except (OSError, ValueError) as exc:
        return presets, [f"{path}: cannot read user combos: {exc}"]
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict) and isinstance(document.get("combos"), list):
        version = document.get("schema_version", USER_PRESETS_FORMAT_VERSION)
        if version != USER_PRESETS_FORMAT_VERSION:
            return presets, [f"{path}: unsupported schema_version {version!r}"]
        entries = document["combos"]
    else:
        return presets, [f"{path}: expected {{\"schema_version\": 1, \"combos\": [...]}}"]
    for index, raw in enumerate(entries):
        try:
            combo = normalize_combo(raw, source="user")
        except ComboError as exc:
            errors.append(f"{path} combos[{index}]: {exc}")
            continue
        if combo["id"] in presets:
            owner = presets[combo["id"]]["source"]
            errors.append(f"{path} combos[{index}]: id {combo['id']!r} already defined ({owner})")
            continue
        presets[combo["id"]] = combo
    return presets, errors


# --------------------------------------------------------------------------- cloud / manifest


def is_cloud_combo(combo: dict[str, Any], settings=None) -> bool:
    from .configuration import runtime_plan
    frozen = runtime_plan(settings)
    if frozen and combo.get('id') == 'primary':
        from .model_registry import is_cloud_model
        return (any(is_cloud_model(m) for m in frozen['models'].values())
                or bool(combo.get('correction', {}).get('model_id')))
    asr = combo.get("asr", {})
    model = str(asr.get("model", ""))
    if model.startswith("user-") or model.startswith("whisper-large"):
        return True
    translation = combo.get("translation", {})
    if translation.get("model_id"):
        return True
    correction = combo.get("correction", {})
    if correction.get("model_id"):
        return True
    return False


def is_cloud_enabled(settings: dict[str, Any] | None = None) -> bool:
    env = os.environ.get("OH_STREAMER_SUBTITLE_CLOUD")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    if settings and isinstance(settings, dict):
        return bool(settings.get("subtitle_cloud_enabled", False))
    return False



def default_output_dir(video: str | os.PathLike[str]) -> Path:
    """Default combos folder for a local video: ``<folder>/<stem>.subtitle-combos``."""

    value = os.fspath(video)
    if value.lower().startswith(("http://", "https://")):
        raise ComboError("URL sources need an explicit output folder (--out)")
    path = Path(value).expanduser().resolve()
    return path.with_name(path.stem + OUTPUT_SUFFIX)


def read_manifest(folder: str | os.PathLike[str]) -> dict[str, Any] | None:
    try:
        manifest = json.loads((Path(folder) / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(manifest.get("combos"), list):
        return None
    return manifest


# --------------------------------------------------------------------------- local model availability

# First available wins when the wanted local combo lacks a downloaded model.
FALLBACK_ORDER = ("ai-default", "ai-single", "anime-gt7b", "ja15b-gt14b", "anime-gt14b",
                  "ja15b-sakura1", "anime-sakura1")
DOWNLOAD_HINT = "可到 設定 → 字幕 → 本機 下載本機字幕模型"
_TRANSLATION_LABELS = {
    "sakura-v1": "Sakura 7B v1.0", "sakura-galtransl-7b-v3.7": "GalTransl 7B v3.7",
    "sakura-galtransl-14b-v3.8": "GalTransl 14B v3.8", "qwen3": "Qwen3 8B",
}
_CLOUD_ASR_PREFIXES = ("user-", "whisper-large")


def _registered_local(model_id: Any, settings: Any) -> bool:
    """True when a registered ``user-*`` model id is a local runtime (needs files); unknown or cloud = False."""
    try:
        from .model_registry import get_model, is_cloud_model
        return not is_cloud_model(get_model(model_id, settings))
    except Exception:
        return False


def combo_requirements(combo: dict[str, Any], settings: Any = None) -> list[dict[str, str]]:
    """Local files a combo needs: ``[{kind, id, label, path}]`` (path relative to the models folder).

    Cloud parts need nothing local, using the rule of :func:`is_cloud_combo`: ASR ``user-*`` / ``whisper-large*``,
    and a translation/correction ``model_id`` (except the built-in local profile ids). ``llama-server`` is required
    (kind ``runtime``, empty path) whenever a local GGUF is.
    """

    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, rid: str, label: str, path: str) -> None:
        if rid not in seen:
            seen.add(rid)
            out.append(dict(kind=kind, id=rid, label=label, path=path))

    def asr(alias: Any) -> None:
        alias = str(alias)
        if alias.startswith(_CLOUD_ASR_PREFIXES):
            return
        add("asr", alias, alias, runtime.model_dir_name(alias))

    def gguf(rid: str, label: str, path: str) -> None:
        add("translation", rid, label, path)

    asr_cfg = combo.get("asr") or {}
    if asr_cfg.get("mode") == "dual":
        asr("whisper-ja-1.5B")
        asr("anime-whisper")
    else:
        asr(asr_cfg.get("model", ""))
    correction = combo.get("correction") or {}
    if correction.get("mode") == "llm":
        for alias in correction.get("candidates") or []:
            asr(alias)
        if not correction.get("model_id"):
            path = correction.get("model_file") or CORRECTION_FILES.get(correction.get("backend") or "qwen3", "")
            if path:
                gguf("correction:" + path, Path(path).name, path)
    translation = combo.get("translation") or {}
    model_id = translation.get("model_id")
    if model_id in TRANSLATION_FILES:
        gguf(model_id, _TRANSLATION_LABELS.get(model_id, model_id), TRANSLATION_FILES[model_id])
    elif model_id and not (str(model_id).startswith("user-") and _registered_local(model_id, settings)):
        pass  # cloud (or unreadable registry row, treated like is_cloud_combo): nothing local
    elif not model_id:
        profile = translation.get("profile")
        path = translation.get("model_file") or TRANSLATION_FILES.get(profile, "")
        if path:
            gguf(profile or path, _TRANSLATION_LABELS.get(profile) or Path(path).name, path)
    if any(row["kind"] == "translation" for row in out):
        add("runtime", "llama-server", "llama-server", "")
    return out


def _asr_dir_ready(folder: Path) -> bool:
    return (folder / "config.json").is_file() and any(
        p.is_file() for pattern in ("*.safetensors", "*.bin") for p in folder.glob(pattern))


def combo_availability(combo: dict[str, Any], models_dir: Any, server: Any = "", settings: Any = None) -> dict[str, Any]:
    """``{available, missing, missing_ids, reason}``; ``missing`` holds zh-TW labels of absent models.

    Checks the on-disk layout ``subtitles/environment.py detect()`` uses; never installs or probes anything.
    """

    models = Path(models_dir)
    missing: list[dict[str, str]] = []
    for row in combo_requirements(combo, settings):
        if row["kind"] == "asr":
            ok = _asr_dir_ready(models / row["path"])
        elif row["kind"] == "translation":
            ok = (models / row["path"]).is_file()
        else:
            ok = bool(server) and Path(server).is_file()
        if not ok:
            missing.append(row)
    labels = [row["label"] for row in missing]
    return dict(available=not missing, missing=labels, missing_ids=[row["id"] for row in missing],
                reason=(f"缺少 {'、'.join(labels)}，{DOWNLOAD_HINT}" if labels else ""))


def resolve_available_combo(combo: dict[str, Any], models_dir: Any, server: Any = "", settings: Any = None,
                            presets: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Use-time fallback for an unavailable local combo; the saved setting is never rewritten.

    Returns ``{combo, fallback}``. ``fallback`` is None when the combo is usable or when no preset in
    ``FALLBACK_ORDER`` is (then the combo is returned unchanged, so the usual missing-model message appears).
    Otherwise the model sections (asr/correction/translation) of the first available preset replace the
    combo's own and ``fallback`` = ``{to_id, to_label, from_label, missing, message}``.
    """

    info = combo_availability(combo, models_dir, server, settings)
    if info["available"]:
        return dict(combo=combo, fallback=None, availability=info)
    if presets is None:
        presets, _errors = load_presets()
    for preset_id in FALLBACK_ORDER:
        preset = presets.get(preset_id)
        if preset and combo_availability(preset, models_dir, server, settings)["available"]:
            effective = copy.deepcopy(combo)
            for section in ("asr", "correction", "translation"):
                effective[section] = copy.deepcopy(preset[section])
            if effective["asr"]["mode"] == "diagnostic":
                effective["asr"]["mode"] = "single"     # the primary row only knows dual / single
            label = preset.get("label") or preset_id
            return dict(combo=effective, availability=info, fallback=dict(
                to_id=preset_id, to_label=label, from_label=combo.get("label") or combo.get("id") or "",
                missing=info["missing"],
                message=f"預設組合缺少模型（{'、'.join(info['missing'])}），暫時改用 {label}"))
    return dict(combo=combo, fallback=None, availability=info)
