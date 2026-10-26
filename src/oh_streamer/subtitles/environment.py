"""Read-only discovery of the subtitle runtime; never installs files.

Installed builds (``OH_STREAMER_INSTALL_ROOT`` set) run local subtitles in the
app's own environment (``sys.executable``) with models under the user data folder.
A development checkout keeps preferring ``testdata/subtitle_poc`` when present.
Cloud-only subtitles need only stdlib + ffmpeg, so they run on whatever
interpreter is already running the app (or a frozen exe re-entered through a hidden route).
"""

from functools import lru_cache
import os
import re
from pathlib import Path
import subprocess
import sys


@lru_cache(maxsize=16)
def _vulkan_devices(server, stamp):
    """Probe only adapter enumeration, without loading a model or downloading."""
    try:
        result = subprocess.run([server, '--list-devices'], capture_output=True,
            timeout=15, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            return ()
        rows = []
        for key, name in re.findall(r'^\s*(Vulkan\d+):\s*(.+?)\s*\(\d+\s*MiB', result.stdout + '\n' + result.stderr, re.M):
            # Conservative integrated-adapter allowlist. An unknown/discrete GPU
            # falls back to CUDA rather than silently competing with repair.
            integrated = bool(re.search(r'Radeon.*\b[6789]\d{2}M\b|Intel.*(?:UHD|Iris|HD Graphics|Core.*Graphics)|Intel\(R\) Graphics', name, re.I)) and not re.search(r'\bRX\b', name, re.I)
            if integrated:
                rows.append((key, name.strip()))
        return tuple(rows)
    except (OSError, subprocess.SubprocessError):
        return ()


# Kept for the paused M8 dual-GPU work; no live caller.
def igpu_runtime(settings=None, root=None):
    settings = settings or {}
    root = Path(root) if root else Path(__file__).resolve().parents[4]
    override = settings.get('subtitle_igpu_server') or os.environ.get('OH_STREAMER_SUBTITLE_IGPU_SERVER')
    executable = 'llama-server.exe' if os.name == 'nt' else 'llama-server'
    candidates = ([Path(override)] if override else [
        root / 'testdata/subtitle_poc/models/llama-cpp-vulkan' / executable,
        root / 'testdata/igpu_feasibility_20261017/llama-b11703-vulkan' / executable])
    server = next((p.resolve() for p in candidates if p.is_file()), None)
    rows = _vulkan_devices(str(server), server.stat().st_mtime_ns) if server else ()
    requested = settings.get('subtitle_igpu_device') or os.environ.get('OH_STREAMER_SUBTITLE_IGPU_DEVICE')
    selected = next((row for row in rows if not requested or row[0] == requested), None)
    return dict(igpu_available=bool(selected), igpu_server=str(server) if server else '',
        igpu_device=selected[0] if selected else '', igpu_name=selected[1] if selected else '',
        igpu_reason='' if selected else '找不到可確認的 Vulkan 核顯或專用翻譯執行檔')


@lru_cache(maxsize=16)
def _probe_python(path, stamp):
    try:
        result = subprocess.run(
            [
                path,
                "-c",
                "import importlib.util, sys; "
                "sys.exit(0 if all(importlib.util.find_spec(name) "
                "for name in ('torch', 'transformers', 'silero_vad', "
                "'opencc', 'pynvml')) else 1)",
            ],
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=16)
def _probe_runtime_python(path, stamp):
    """Cloud rows only need a working interpreter; local workers validate dependencies."""
    try:
        result = subprocess.run([path, "-c", "import sys; sys.exit(0)"], capture_output=True,
                                timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# Hidden frozen_entry routes for subtitle child processes (``-m`` is unavailable in a frozen exe).
SUBTITLE_CHILD_ROUTES = {
    "oh_streamer.subtitles.worker": "--subtitle-worker",
}


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def subtitle_command(python, module, *args):
    """argv for a subtitle child process; a frozen build re-enters its own executable."""
    if is_frozen():
        return [sys.executable, SUBTITLE_CHILD_ROUTES[module], *map(str, args)]
    return [str(python or sys.executable), "-m", module, *map(str, args)]


def resolve_runtime_python(python):
    """(interpreter, ok) able to run the stdlib-only (cloud/mux) subtitle worker.

    Prefers the configured subtitle venv so local rows keep their packages; falls back
    to the running interpreter (dev checkout without the venv) or the frozen exe.
    """
    if is_frozen():
        return sys.executable, bool(sys.executable)
    candidate = Path(python) if python else None
    if candidate and candidate.is_file() and _probe_runtime_python(str(candidate.resolve()), candidate.stat().st_mtime_ns):
        return str(candidate.resolve()), True
    current = Path(sys.executable) if sys.executable else None
    if current and current.is_file():
        return str(current.resolve()), True
    return "", False


def is_installed():
    """True in an installed build (the launcher sets OH_STREAMER_INSTALL_ROOT)."""
    return bool(os.environ.get("OH_STREAMER_INSTALL_ROOT"))


def default_models_dir():
    """Where downloaded subtitle models live by default: <user data folder>/subtitle_models."""
    from oh_streamer.platform import platform_support

    return platform_support.data_dir() / "subtitle_models"


def _settings_text(settings, key):
    value = settings.get(key)
    return value.strip() if isinstance(value, str) else ""


def runtime_paths(settings=None, root=None):
    settings = settings or {}
    root = Path(root) if root else Path(__file__).resolve().parents[4]
    base = root / "testdata/subtitle_poc"
    dev_python = base / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    dev_models = base / "models"
    installed = is_installed()
    # Defaults only: installed = the app's own interpreter + user data folder;
    # dev checkout = the PoC venv/models when they exist, else the same shared defaults.
    default_python = Path(sys.executable) if installed or not dev_python.is_file() else dev_python
    default_models = default_models_dir() if installed or not dev_models.is_dir() else dev_models
    python = Path(
        settings.get("subtitle_python")
        or os.environ.get("OH_STREAMER_SUBTITLE_PYTHON")
        or default_python
    )
    # Precedence: OH_STREAMER_SUBTITLE_MODELS > subtitle_models > custom_models_directory > default.
    models = Path(
        os.environ.get("OH_STREAMER_SUBTITLE_MODELS")
        or _settings_text(settings, "subtitle_models")
        or _settings_text(settings, "custom_models_directory")
        or default_models
    )
    server = settings.get("subtitle_server") or os.environ.get("OH_STREAMER_SUBTITLE_SERVER")
    if not server:
        server = next(
            (
                str(path)
                for path in (models / "llama-cpp").rglob(
                    "llama-server.exe" if os.name == "nt" else "llama-server"
                )
            ),
            "",
        )
    return dict(python=str(python.resolve()), models=str(models.resolve()), server=str(Path(server).resolve()) if server else "")


def detect(settings=None, root=None):
    settings = settings or {}
    paths = runtime_paths(settings, root)
    python, models, server = Path(paths["python"]), Path(paths["models"]), paths["server"]
    asr_options = [
        dict(value=key, label=key)
        for key in ("whisper-ja-1.5B", "anime-whisper", "kotoba", "turbo")
        if (models / key / "config.json").is_file()
        and any((models / key).glob("*.safetensors"))
    ]
    translations = [
        dict(value=path.relative_to(models).as_posix(), label=path.name)
        for path in sorted((models / "sakura").glob("*.gguf"))
    ]
    asr_mode = settings.get("subtitle_asr_mode", "dual")
    asr = settings.get("subtitle_asr") or "whisper-ja-1.5B"
    translation = settings.get("subtitle_translation") or "sakura/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf"
    reasons = []
    if not python.is_file():
        reasons.append("找不到字幕執行環境的 Python")
    elif not _probe_python(str(python.resolve()), python.stat().st_mtime_ns):
        reasons.append("字幕執行環境缺少必要套件或無法執行")
    runtime_reasons = []
    runtime_python, runtime_ok = resolve_runtime_python(python)
    if not runtime_ok:
        runtime_reasons.append("找不到可執行的字幕 Python")
    available_asr = {option["value"] for option in asr_options}
    if asr_mode not in ("dual", "single"):
        reasons.append("ASR 模式必須為 dual 或 single")
    elif not ({"whisper-ja-1.5B", "anime-whisper"} & available_asr if asr_mode == "dual" else asr in available_asr):
        reasons.append("找不到可用的 ASR 本地模型" if asr_mode == "dual" else "找不到選定的 ASR 本地模型")
    if translation not in {option["value"] for option in translations}:
        reasons.append("找不到選定的 Sakura GGUF")
    if not server or not Path(server).is_file():
        reasons.append("找不到 llama-server")

    result = dict(
        available=not reasons,
        reason="；".join(reasons),
        # Subtitle combos pick their own ASR/GGUF models per combo, so only the
        # interpreter gates them (local requirements fail their own row only).
        runtime_available=not runtime_reasons,
        runtime_reason="；".join(runtime_reasons),
        # Interpreter for cloud/mux workers; equals ``python`` when the venv works.
        runtime_python=runtime_python,
        python=str(python.resolve()),
        models=str(models.resolve()),
        server=str(Path(server).resolve()) if server else "",
        asr=asr,
        asr_mode=asr_mode,
        translation=translation,
        asr_options=asr_options,
        translation_options=translations,
    )
    return result


def model_ref(row, settings=None):
    """Copy of a resolved model row in snapshot form: a model inside the models folder is stored as
    ``model_name`` (posix path relative to the models folder, ``.`` = the folder itself) and carries no
    ``model_path``; a model outside the folder (user-registered local model) keeps its absolute ``model_path``.
    The absolute location is rebuilt at use time by :func:`model_file`, so moving the models folder
    changes neither the stored row nor the cache identity."""
    row = dict(row)
    raw = row.get("model_path")
    if not raw or row.get("model_name"):
        return row
    try:
        root = Path(runtime_paths(settings)["models"])
        rel = Path(raw).expanduser().resolve().relative_to(root)
    except (ValueError, OSError):
        return row
    row.pop("model_path")
    row["model_name"] = rel.as_posix()
    return row


def model_file(row, settings=None, *, models=None):
    """Absolute path of a model row's files, or None. Accepts both snapshot forms: ``model_name`` (built under
    the current models folder) and legacy/outside rows with an absolute ``model_path`` (used as is when it
    exists, else looked up by basename inside the models folder). ``models`` is an explicit models root that
    overrides the one derived from ``settings``."""
    if not row:
        return None
    root = Path(models) if models else Path(runtime_paths(settings)["models"])
    name = row.get("model_name")
    if name:
        return str(root if name == "." else root / name)
    raw = row.get("model_path")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return str(path)
    alt = root / path.name
    return str(alt) if alt.exists() else str(path)
