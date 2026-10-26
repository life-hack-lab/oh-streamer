"""Small, lazy runtime helpers used by the local subtitle ASR stage.

Importing this module does not import torch, Transformers, Silero, or NVML.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Any

from oh_streamer.platform import platform_support


SAMPLE_RATE = 16_000
SCHEMA_VERSION = 1
VAD_PROFILE = {
    "threshold": 0.5,
    "min_speech_duration_ms": 150,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
}
GPU_UTIL_SAMPLE_COUNT = 3
GPU_UTIL_SAMPLE_INTERVAL_SECONDS = 0.2
GPU_BUSY_UTIL_THRESHOLD_PERCENT = 10.0

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "anime-whisper": {
        "model_id": "litagin/anime-whisper",
        # Match the official card. In particular, do not add an initial prompt.
        "generate_kwargs": {
            "language": "Japanese",
            "task": "transcribe",
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
        "torch_dtype": "float16",
    },
    "kotoba": {
        "model_id": "kotoba-tech/kotoba-whisper-v2.0",
        "generate_kwargs": {"language": "ja", "task": "transcribe"},
        "torch_dtype": "bfloat16",
    },
    "kotoba-v2.0": {
        "model_id": "kotoba-tech/kotoba-whisper-v2.0",
        "generate_kwargs": {"language": "ja", "task": "transcribe"},
        "torch_dtype": "bfloat16",
    },
    "turbo": {
        "model_id": "openai/whisper-large-v3-turbo",
        "generate_kwargs": {"language": "ja", "task": "transcribe"},
        "torch_dtype": "float16",
    },
    # ASR candidates from testdata/subtitle_poc/asr_candidates (run_candidate.py).
    # Keys are lower case for lookup; ``model_dir`` keeps the on-disk folder name.
    "whisper-ja-760m": {
        "model_id": "efwkjn/whisper-ja-760M",
        "model_dir": "whisper-ja-760M",
        "generate_kwargs": {"language": "ja", "task": "transcribe"},
        "torch_dtype": "float16",
    },
    "whisper-ja-1.5b": {
        "model_id": "efwkjn/whisper-ja-1.5B",
        "model_dir": "whisper-ja-1.5B",
        "generate_kwargs": {"language": "ja", "task": "transcribe", "no_repeat_ngram_size": 6},
        "torch_dtype": "float16",
    },
    "whisper-large-v3": {
        "model_id": "openai/whisper-large-v3",
        # Repetition guard: unguarded large-v3 produced 446-character loops in
        # the realistic PoC; ngram6 bounded the longest segment to 59 characters.
        "generate_kwargs": {"language": "ja", "task": "transcribe", "no_repeat_ngram_size": 6},
        "torch_dtype": "float16",
    },
    "qwen3-asr-1.7b": {
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "backend": "qwen-asr",
        "segmentations": ("utterance",),
        "generate_kwargs": {"language": "Japanese", "context": "", "max_new_tokens": 512},
        "torch_dtype": "bfloat16",
    },
}


def model_dir_name(alias: str) -> str:
    """On-disk folder name for a resolved alias (case preserved for new candidates)."""

    key = str(alias).strip().lower()
    return str(MODEL_SPECS.get(key, {}).get("model_dir", key))


class RuntimeCheckError(RuntimeError):
    """Raised when a required local runtime safety check cannot be completed."""


class GPUSafetyError(RuntimeCheckError):
    """Raised when using the CUDA device would overlap another GPU workload."""


def resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    """Resolve one supported alias or official model id to its checked-in recipe."""

    key = str(model).strip().lower()
    if key in MODEL_SPECS:
        spec = MODEL_SPECS[key]
        return key, {**spec, "generate_kwargs": dict(spec["generate_kwargs"])}
    for alias, spec in MODEL_SPECS.items():
        if key == spec["model_id"].lower():
            # The first alias for a repository is the stable public name.
            public_alias = "kotoba" if alias == "kotoba" else alias
            return public_alias, {**spec, "generate_kwargs": dict(spec["generate_kwargs"])}
    supported = (
        "anime-whisper, kotoba, turbo, whisper-ja-760M, whisper-ja-1.5B, "
        "whisper-large-v3, qwen3-asr-1.7b"
    )
    raise ValueError(f"unsupported ASR model {model!r}; choose {supported}")


def default_cache_dir() -> Path:
    """Return a user-owned cache directory outside the application/Lada virtualenvs."""

    return platform_support.cache_dir() / "subtitles"


def source_identity(source: str | os.PathLike[str]) -> tuple[str, dict[str, Any]]:
    """Return a stable cache key and safe-to-store summary for a local file or URL."""

    value = os.fspath(source)
    if value.lower().startswith(("http://", "https://")):
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(value)
        # Query strings often contain signed credentials; never persist them in result JSON.
        safe_url = urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
        descriptor = {"kind": "url", "source": safe_url, "identity": value}
    else:
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        descriptor = {
            "kind": "file",
            "source": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "identity": str(path),
        }
    encoded = json.dumps(descriptor, sort_keys=True, ensure_ascii=False).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    safe_descriptor = {k: v for k, v in descriptor.items() if k != "identity"}
    return key, safe_descriptor


def make_job_key(
    source_key: str,
    model_id: str,
    use_vad: bool,
    model_fingerprint: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_key": source_key,
        "model_id": model_id,
        "model_fingerprint": model_fingerprint,
        "use_vad": bool(use_vad),
        "vad_profile": VAD_PROFILE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def default_model_path(alias: str) -> Path:
    """Default model snapshot location; follows the same rule as environment.runtime_paths()."""

    from oh_streamer.subtitles import environment

    return Path(environment.runtime_paths()["models"]) / alias


def local_model_fingerprint(model_path: str | os.PathLike[str]) -> str:
    """Fingerprint local model file names, sizes, and mtimes without reading multi-GB weights."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"ASR model files were not found at {root}; prepare/download the model first and pass model_path"
        )
    files = []
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part.lower() in {".cache", ".git"} for part in relative.parts):
            continue
        try:
            if path.is_file():
                stat = path.stat()
                files.append((relative.as_posix(), stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    has_config = any(name == "config.json" for name, _size, _mtime in files)
    has_weights = any(
        name.endswith((".safetensors", ".bin")) or name.endswith((".safetensors.index.json", ".bin.index.json"))
        for name, _size, _mtime in files
    )
    if not files or not has_config or not has_weights:
        raise FileNotFoundError(
            f"ASR model files at {root} are incomplete; prepare/download the model snapshot first"
        )
    files.sort()
    payload = json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _replace_with_retry(source: str, destination: Path) -> None:
    """os.replace that rides out a transient Windows sharing violation (another reader in this process, e.g. the
    view tick reading the file without FILE_SHARE_DELETE, for a few ms): up to ~1 s, then the original PermissionError."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(0.03 * (attempt + 1))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: str | os.PathLike[str], value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def resolve_ffmpeg() -> str:
    """Find the user's or bundle's ffmpeg without consulting the Lada venv."""

    configured = os.environ.get("OH_STREAMER_FFMPEG")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found
    try:
        from oh_streamer.paths import bundle_root, checkout_root, upstream_root

        # upstream_root() is the install root (<root>/tools) and, in dev, the folder beside the checkout.
        roots = [root for root in (bundle_root(), checkout_root(), upstream_root()) if root is not None]
        names = ("ffmpeg.exe", "ffmpeg")
        candidates = [root / "tools" / name for root in roots for name in names]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    except Exception:  # paths.py is intentionally not required for the helper
        pass
    raise FileNotFoundError("ffmpeg was not found; install it on PATH or set OH_STREAMER_FFMPEG")


def valid_pcm_wav(path: str | os.PathLike[str], *, sample_rate: int = SAMPLE_RATE) -> bool:
    try:
        with wave.open(os.fspath(path), "rb") as audio:
            return (audio.getnchannels() == 1 and audio.getframerate() == sample_rate
                    and audio.getsampwidth() == 2 and audio.getnframes() > 0)
    except (OSError, wave.Error, EOFError):
        return False


def _probe_input_args(source) -> list[str]:
    from oh_streamer.sources import media_input
    arg, opts = media_input.ffmpeg_input(source)
    return [*opts, arg]


def scrub_remote(text: str | None, url: str | None) -> str:
    """Hide a remote marker's signed url in ffmpeg output before it reaches an error message or log."""
    text = text or ""
    return text.replace(url, "<remote>") if url else text


def raise_for_remote_failure(returncode: int, stderr: str | None, url: str | None) -> None:
    """Judge a finished ffmpeg run that read a remote marker (run with ``-loglevel warning``). An expired signed url
    raises ``RemoteExpired``; any other remote read problem raises ``RuntimeError`` even when the exit code is 0
    (an HLS segment that fails is skipped and ffmpeg still exits 0 with shorter audio). The url never appears in the
    message. Returns normally only when the run was clean."""
    from oh_streamer.sources import media_input
    kind = media_input.remote_failure(stderr)
    detail = scrub_remote((stderr or "").strip()[-1000:], url)
    if kind == "expired":
        raise media_input.RemoteExpired()
    if kind == "failed":
        raise RuntimeError(f"ffmpeg could not read the remote audio (exit {returncode}): {detail}")
    if returncode:
        raise RuntimeError(f"ffmpeg could not extract audio (exit {returncode}): {detail}")


def _probe_audio_duration(source: str | os.PathLike[str], ffmpeg: str, timeout: float | None) -> float | None:
    """Return a probed audio duration when ffprobe is available, without requiring it."""

    executable = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if executable is None:
        ffmpeg_path = Path(ffmpeg)
        sibling = ffmpeg_path.with_name("ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe")
        if sibling.is_file():
            executable = str(sibling)
    if executable is None:
        return None

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    probe_timeout = min(float(timeout), 10.0) if timeout is not None else 10.0
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration,start_time:format=duration",
                "-of",
                "json",
                *_probe_input_args(source),
            ],
            capture_output=True,
            text=True,
            timeout=probe_timeout,
            creationflags=flags,
            check=False,
        )
        if result.returncode:
            return None
        details = json.loads(result.stdout or "{}")
        streams = details.get("streams", [])
        stream = streams[0] if streams else {}
        try:
            stream_duration = float(stream.get("duration"))
        except (TypeError, ValueError):
            stream_duration = 0.0
        if stream_duration > 0:
            return stream_duration
        try:
            format_duration = float((details.get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            return None
        try:
            stream_start = float(stream.get("start_time"))
        except (TypeError, ValueError):
            stream_start = 0.0
        if stream_start > 0 and format_duration > stream_start:
            format_duration -= stream_start
        if format_duration > 0:
            return format_duration
    except (OSError, subprocess.SubprocessError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def _time_from_ffmpeg_progress(line: str) -> float | None:
    key, separator, value = line.strip().partition("=")
    if not separator:
        return None
    try:
        if key == "out_time_us":
            return max(0.0, int(value) / 1_000_000.0)
        if key == "out_time":
            hours, minutes, seconds = value.split(":", 2)
            return max(0.0, int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    except (TypeError, ValueError):
        return None
    return None


def _run_with_ffmpeg_progress(
    command: list[str],
    *,
    timeout: float | None,
    on_progress,
    total_duration: float | None,
    creationflags: int,
) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg while reading its actual media-time progress from stdout."""

    lines: queue.Queue[str | None] = queue.Queue()
    current = 0.0
    last_percent = -1
    last_unknown_bucket = 0
    process = None
    reader = None
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            if process.stdout is None:
                raise RuntimeError("ffmpeg progress output was not available")

            def read_lines() -> None:
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=read_lines, name="ffmpeg-progress-reader", daemon=True)
            reader.start()
            started = time.monotonic()
            finished_reading = False
            while not finished_reading:
                if timeout is not None and time.monotonic() - started >= timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    finished_reading = True
                    continue
                progress_time = _time_from_ffmpeg_progress(line)
                if progress_time is None:
                    continue
                current = max(current, progress_time)
                if total_duration is not None:
                    percent = min(100, max(0, int(current * 100 / total_duration)))
                    if percent != last_percent:
                        last_percent = percent
                        on_progress(current, total_duration)
                elif int(current // 30) > last_unknown_bucket:
                    last_unknown_bucket = int(current // 30)
                    on_progress(current, None)

            returncode = process.wait()
            reader.join(timeout=1.0)
            process.stdout.close()
            stderr_file.seek(0)
            stderr = stderr_file.read()
            return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if process is not None and process.stdout is not None:
                process.stdout.close()
            if reader is not None:
                reader.join(timeout=1.0)
            raise


def resample_to_wav(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    ffmpeg: str | None = None,
    timeout: float | None = None,
    on_progress=None,
) -> float:
    """Decode the first audio stream as cached 16 kHz mono PCM and return its duration."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_pcm_wav(target):
        duration = wav_duration(target)
        if on_progress:
            on_progress(duration, duration)
        return duration

    executable = ffmpeg or resolve_ffmpeg()
    input_args = _probe_input_args(source)
    total_duration = _probe_audio_duration(source, executable, timeout) if on_progress else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".partial.wav", dir=target.parent)
    os.close(fd)
    try:
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            *input_args[:-1], "-i", input_args[-1], "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
        ]
        if on_progress:
            command.extend(("-progress", "pipe:1", "-nostats"))
        command.append(temp_name)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            if on_progress:
                on_progress(0.0, total_duration)
                result = _run_with_ffmpeg_progress(
                    command,
                    timeout=timeout,
                    on_progress=on_progress,
                    total_duration=total_duration,
                    creationflags=flags,
                )
            else:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                        creationflags=flags, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg timed out while extracting 16 kHz audio from {source}") from exc
        if result.returncode:
            detail = (result.stderr or "").strip()[-1000:]
            raise RuntimeError(f"ffmpeg could not extract audio (exit {result.returncode}): {detail}")
        if not valid_pcm_wav(temp_name):
            raise RuntimeError("ffmpeg output is not a non-empty 16 kHz mono PCM WAV")
        os.replace(temp_name, target)
        duration = wav_duration(target)
        if on_progress:
            on_progress(duration, duration)
        return duration
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def resample_window_to_wav(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    start: float,
    end: float,
    *,
    ffmpeg: str | None = None,
    timeout: float | None = None,
    pad: bool = True,
    source_url: str | None = None,
) -> float:
    """Extract [start, end) of the first audio stream as 16 kHz mono PCM, padded with ``start`` seconds of
    silence in front so sample positions equal source timestamps (pad=False: the file starts at ``start``).
    A progressive source (HLS still downloading) is only touched where its segments are ready. HLS decodes the
    available prefix through ``end``: its demuxer does not reliably support sample-accurate input seeking."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_pcm_wav(target):
        return wav_duration(target)
    executable = ffmpeg or resolve_ffmpeg()
    start, end = max(0.0, float(start)), float(end)
    if end <= start:
        raise ValueError("audio window must be positive")
    from oh_streamer.sources import media_input
    is_remote = media_input.is_remote_marker(source)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".partial.wav", dir=target.parent)
    os.close(fd)
    try:
        # EVENT playlists are still live to FFmpeg until ENDLIST appears. Its default (-3) starts
        # near the latest segment, changing the timestamp origin and silently clipping requested
        # windows. Scheduler timestamps always refer to the complete prefix starting at segment 0.
        source_path = os.fspath(source).split("?", 1)[0].split("#", 1)[0]
        is_hls = Path(source_path).suffix.lower() in (".m3u8", ".m3u")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

        def build_command() -> tuple[list[str], str]:
            # A remote marker keeps -loglevel warning: a failed HLS segment is skipped with exit 0, and only
            # stderr tells. Built per attempt so a refreshed marker url is picked up by the retry.
            command = [executable, "-hide_banner", "-loglevel", "warning" if is_remote else "error",
                       "-nostdin", "-y"]
            filters = []
            if is_hls:
                command += ["-live_start_index", "0"]
                # HLS input -ss may land before/after the requested audio, even with accurate_seek.
                # Trim decoded timestamps instead; reset them before optional source-position padding.
                filters += [f"atrim=start={start:.3f}:end={end:.3f}", "asetpts=PTS-STARTPTS"]
            else:
                # Keep the established local-file extraction semantics and its existing content caches.
                # A remote marker (even one pointing at HLS) also lands here: input -ss fetches only the
                # needed segments/ranges, a prefix decode would read the stream from the start.
                if is_remote:
                    # -copyts is required: on an audio-only remote HLS, plain input -ss lands on a segment boundary
                    # and does not trim (up to one segment early, measured 6 s). With -copyts the demuxer's real
                    # timestamps are kept and ffmpeg trims to the exact start; wav output ignores the pts offset.
                    command += ["-copyts"]
                command += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
            input_arg, input_opts = media_input.ffmpeg_input(source, source_url)
            command += input_opts
            command += ["-i", input_arg,
                       "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)]
            if start > 0 and pad:
                filters.append(f"adelay={int(round(start * 1000))}:all=1")
            if filters:
                command += ["-af", ",".join(filters)]
            if is_hls:
                # Stop after the requested decoded samples, without waiting for later live segments.
                output_duration = end if pad else end - start
                command += ["-t", f"{output_duration:.3f}"]
            command += ["-c:a", "pcm_s16le", temp_name]
            return command, input_arg

        def attempt() -> None:
            cmd, input_arg = build_command()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                        creationflags=flags, check=False)
            except subprocess.TimeoutExpired as exc:
                if is_remote:  # the command line holds the signed url
                    raise RuntimeError("ffmpeg timed out while extracting an audio window from <remote>") from None
                raise RuntimeError(f"ffmpeg timed out while extracting an audio window from {source}") from exc
            if is_remote:
                raise_for_remote_failure(result.returncode, result.stderr, input_arg)
            elif result.returncode:
                raise RuntimeError(f"ffmpeg could not extract audio (exit {result.returncode}): "
                                   f"{(result.stderr or '').strip()[-1000:]}")
            if not valid_pcm_wav(temp_name):
                raise RuntimeError("ffmpeg output is not a non-empty 16 kHz mono PCM WAV")

        media_input.run_remote(source, attempt)
        os.replace(temp_name, target)
        return wav_duration(target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def wav_duration(path: str | os.PathLike[str]) -> float:
    with wave.open(os.fspath(path), "rb") as audio:
        if audio.getframerate() <= 0:
            raise RuntimeError("WAV has an invalid sample rate")
        return audio.getnframes() / float(audio.getframerate())


def load_wav_float32(path: str | os.PathLike[str]):
    """Read cached PCM WAV into a float32 numpy array; numpy is loaded only on demand."""

    import numpy as np

    with wave.open(os.fspath(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        width = audio.getsampwidth()
        frame_count = audio.getnframes()
        payload = audio.readframes(frame_count)
    if channels != 1 or sample_rate != SAMPLE_RATE or width != 2:
        raise RuntimeError("ASR input cache must be 16 kHz mono signed 16-bit PCM")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def _run_checked(command: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              creationflags=flags, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeCheckError(f"could not run required safety probe {command[0]!r}: {exc}") from exc


def _on_ac_power() -> dict[str, Any]:
    """Return a confirmed mains-power state; unknown states are errors."""

    if os.name == "nt":
        import ctypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = SYSTEM_POWER_STATUS()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_status = kernel32.GetSystemPowerStatus
        get_status.argtypes = [ctypes.POINTER(SYSTEM_POWER_STATUS)]
        get_status.restype = ctypes.c_bool
        if not get_status(ctypes.byref(status)):
            raise RuntimeCheckError("Windows could not report AC power status")
        if status.ACLineStatus != 1:
            state = "battery" if status.ACLineStatus == 0 else "unknown"
            raise RuntimeCheckError(f"GPU ASR requires AC power; Windows reports {state}")
        return {"on_ac_power": True, "source": "GetSystemPowerStatus"}

    if sys.platform == "darwin":
        result = _run_checked(["pmset", "-g", "batt"])
        if result.returncode != 0:
            raise RuntimeCheckError(f"pmset could not report power status: {result.stderr.strip()}")
        if "AC Power" in result.stdout:
            return {"on_ac_power": True, "source": "pmset"}
        if "Battery Power" in result.stdout:
            raise RuntimeCheckError("GPU ASR requires AC power; macOS reports battery power")
        raise RuntimeCheckError("macOS power status was ambiguous; refusing GPU ASR")

    supply_root = Path("/sys/class/power_supply")
    if supply_root.is_dir():
        known_mains = False
        for supply in supply_root.iterdir():
            try:
                supply_type = (supply / "type").read_text(encoding="ascii").strip().lower()
                if supply_type in {"mains", "usb", "usb_c"}:
                    known_mains = True
                    if (supply / "online").read_text(encoding="ascii").strip() == "1":
                        return {"on_ac_power": True, "source": str(supply)}
            except OSError:
                continue
        if known_mains:
            raise RuntimeCheckError("GPU ASR requires AC power; no connected mains supply was found")
    raise RuntimeCheckError("could not confirm AC power; refusing GPU ASR")


def _approved_idle_ui_pids() -> set[int]:
    """Session-only exception explicitly approved by the user; compute checks still apply."""
    value = os.environ.get('OH_STREAMER_SUBTITLE_IDLE_UI_PIDS', '')
    try:
        return {int(pid) for pid in value.split(',') if pid.strip()}
    except ValueError as exc:
        raise RuntimeCheckError('Invalid approved idle UI PID list') from exc


def _ignore_idle_ui(name: str, pid: int) -> bool:
    return name.lower() in {'pythonw', 'pythonw.exe'} and pid in _approved_idle_ui_pids()


def _process_names() -> tuple[list[str], str, list[str]]:
    errors: list[str] = []
    if os.name == "nt":
        try:
            result = _run_checked(["tasklist", "/FO", "CSV", "/NH"])
            if result.returncode == 0:
                rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
                if rows and all(len(row) >= 2 for row in rows):
                    return [row[0].strip().lower() for row in rows
                            if not _ignore_idle_ui(row[0].strip(), int(row[1].strip()))], "tasklist", errors
                errors.append("tasklist returned no parseable process rows")
            else:
                errors.append(f"tasklist exited {result.returncode}: {result.stderr.strip()}")
        except RuntimeCheckError as exc:
            errors.append(str(exc))

        # psutil avoids tasklist's account/session formatting issues when installed.
        try:
            import psutil

            names = [str(process.info.get("name") or "").lower() for process in psutil.process_iter(["name"])
                     if not _ignore_idle_ui(str(process.info.get('name') or ''), process.pid)]
            if names:
                return names, "psutil", errors
            errors.append("psutil returned no process rows")
        except Exception as exc:
            errors.append(f"psutil: {exc}")

        # Some managed Windows images lack tasklist or psutil; Get-Process is the last fallback.
        try:
            script = "Get-Process | ForEach-Object { '{0},{1}' -f $_.ProcessName,$_.Id }"
            fallback = _run_checked(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
            if fallback.returncode == 0:
                rows = list(csv.reader(line for line in fallback.stdout.splitlines() if line.strip()))
                names = [row[0].strip().lower() for row in rows
                         if not _ignore_idle_ui(row[0].strip(), int(row[1].strip()))]
                if names:
                    return names, "Get-Process", errors
            errors.append(f"Get-Process exited {fallback.returncode}: {fallback.stderr.strip()}")
        except RuntimeCheckError as exc:
            errors.append(str(exc))
        raise RuntimeCheckError("could not inspect Windows processes: " + " | ".join(errors))

    try:
        result = _run_checked(["ps", "-A", "-o", "comm="])
        if result.returncode == 0:
            names = [Path(line.strip()).name.lower() for line in result.stdout.splitlines() if line.strip()]
            if names:
                return names, "ps", errors
        errors.append(f"ps exited {result.returncode}: {result.stderr.strip()}")
    except RuntimeCheckError as exc:
        errors.append(str(exc))
    try:
        import psutil

        names = [str(process.info.get("name") or "").lower() for process in psutil.process_iter(["name"])]
        if names:
            return names, "psutil", errors
        errors.append("psutil returned no process rows")
    except Exception as exc:
        errors.append(f"psutil: {exc}")
    raise RuntimeCheckError("could not inspect process names before GPU ASR: " + " | ".join(errors))


def _compute_processes_nvml() -> tuple[list[dict[str, Any]], str]:
    """Enumerate compute PIDs via NVML, returning an error if any device cannot be inspected."""

    import pynvml

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        count = int(pynvml.nvmlDeviceGetCount())
        jobs: list[dict[str, Any]] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            getter = None
            for method_name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ):
                candidate = getattr(pynvml, method_name, None)
                if candidate is not None:
                    getter = candidate
                    break
            if getter is None:
                raise RuntimeCheckError("installed NVML bindings cannot enumerate compute processes")
            processes = getter(handle)
            for process in processes:
                jobs.append({
                    "pid": int(process.pid),
                    "gpu_index": index,
                    "used_gpu_memory_bytes": getattr(process, "usedGpuMemory", None),
                })
        return jobs, "NVML"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _compute_processes_smi() -> tuple[list[dict[str, Any]], str]:
    result = _run_checked([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        raise RuntimeCheckError(f"nvidia-smi could not enumerate compute processes: {result.stderr.strip()}")
    jobs = []
    for line in result.stdout.splitlines():
        if not line.strip() or "no running processes" in line.lower():
            continue
        parts = next(csv.reader([line]))
        if not parts or not parts[0].strip().isdigit():
            raise RuntimeCheckError("nvidia-smi returned an unrecognized compute-process row")
        memory = parts[2].strip().strip('[]') if len(parts) > 2 else "N/A"
        try:
            memory_bytes = None if memory in ("N/A", "Not Supported", "") else int(float(memory) * 1024 * 1024)
        except ValueError as exc:
            raise RuntimeCheckError("nvidia-smi returned unknown compute memory usage") from exc
        jobs.append({"pid": int(parts[0].strip()), "process_name": parts[1].strip() if len(parts) > 1 else None,
                     "used_gpu_memory_bytes": memory_bytes})
    return jobs, "nvidia-smi"


def _gpu_util_samples_nvml(
    sample_count: int = GPU_UTIL_SAMPLE_COUNT,
    interval_seconds: float = GPU_UTIL_SAMPLE_INTERVAL_SECONDS,
) -> tuple[list[float], str]:
    import pynvml

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handles = [pynvml.nvmlDeviceGetHandleByIndex(index)
                   for index in range(int(pynvml.nvmlDeviceGetCount()))]
        if not handles:
            raise RuntimeCheckError("NVML found no NVIDIA GPU devices")
        samples = []
        for sample_index in range(sample_count):
            samples.append(max(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) for handle in handles))
            if sample_index + 1 < sample_count:
                time.sleep(interval_seconds)
        return samples, "NVML"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _gpu_util_samples_smi(
    sample_count: int = GPU_UTIL_SAMPLE_COUNT,
    interval_seconds: float = GPU_UTIL_SAMPLE_INTERVAL_SECONDS,
) -> tuple[list[float], str]:
    samples = []
    for sample_index in range(sample_count):
        result = _run_checked([
            "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
        ])
        if result.returncode != 0:
            raise RuntimeCheckError(f"nvidia-smi could not sample GPU utilization: {result.stderr.strip()}")
        values = []
        for line in result.stdout.splitlines():
            value = line.strip().split()[0] if line.strip() else ""
            try:
                values.append(float(value))
            except ValueError as exc:
                raise RuntimeCheckError(f"nvidia-smi returned an unknown utilization value: {line!r}") from exc
        if not values:
            raise RuntimeCheckError("nvidia-smi returned no GPU utilization values")
        samples.append(max(values))
        if sample_index + 1 < sample_count:
            time.sleep(interval_seconds)
    return samples, "nvidia-smi"


def _gpu_utilization_samples() -> tuple[list[float], str, list[str]]:
    errors = []
    for probe in (_gpu_util_samples_nvml, _gpu_util_samples_smi):
        try:
            samples, source = probe()
            return samples, source, errors
        except Exception as exc:
            probe_name = getattr(probe, "__name__", type(probe).__name__)
            errors.append(f"{probe_name}: {exc}")
    raise RuntimeCheckError("could not verify GPU utilization: " + " | ".join(errors))


def assert_gpu_safe() -> dict[str, Any]:
    """Require AC power and inspect GPU activity, rather than blacklist app names.

    An enumerated CUDA context alone is not an active workload. Idle contexts with
    no reported CUDA allocation may coexist when every utilization sample is zero.
    Resident CUDA allocations, compute activity and uninspectable GPU state block.
    """

    power = _on_ac_power()
    try:
        _names, process_source, process_probe_errors = _process_names()
    except RuntimeCheckError as exc:
        # Process names are diagnostic; compute and utilization probes below
        # establish admission independently of app names or process visibility.
        process_source, process_probe_errors = "unavailable", [str(exc)]

    compute_jobs = None
    compute_source = None
    errors = []
    for probe in (_compute_processes_nvml, _compute_processes_smi):
        try:
            compute_jobs, compute_source = probe()
            break
        except Exception as exc:
            errors.append(f"{getattr(probe, '__name__', type(probe).__name__)}: {exc}")
    if compute_jobs is None:
        raise RuntimeCheckError("could not confirm that the NVIDIA GPU is idle; " + " | ".join(errors))
    utilization_samples, utilization_source, utilization_probe_errors = _gpu_utilization_samples()
    if not utilization_samples or any(not math.isfinite(value) or not 0 <= value <= 100
                                      for value in utilization_samples):
        raise RuntimeCheckError("could not verify valid GPU utilization samples")
    # NVML uses None (or its unsigned NOT_AVAILABLE sentinel) on WDDM. Neither
    # is evidence of an allocation; retain the context in the audit instead.
    allocated = [job for job in compute_jobs if isinstance(job.get("used_gpu_memory_bytes"), (int, float))
                 and 0 < job["used_gpu_memory_bytes"] < (1 << 60)]
    active_jobs = allocated or (compute_jobs if any(value > 0 for value in utilization_samples) else [])
    if active_jobs:
        summary = ", ".join(f"PID {job['pid']}" for job in active_jobs)
        raise GPUSafetyError(f"GPU ASR is blocked by another CUDA workload ({summary})")
    if all(value > GPU_BUSY_UTIL_THRESHOLD_PERCENT for value in utilization_samples):
        raise GPUSafetyError(
            "GPU is busy at " + ", ".join(f"{value:.0f}%" for value in utilization_samples)
            + " utilization, but no compute PID was reported; refusing GPU ASR as an unidentified workload"
        )

    return {
        "on_ac_power": True,
        "power_check": power,
        "process_check": {
            "source": process_source,
            "blocked_names_found": [],
            "probe_errors": process_probe_errors,
            "admission_rule": "GPU activity; application names do not block admission",
        },
        "compute_check": {"source": compute_source, "other_processes": 0, "probe_errors": errors,
                          "idle_contexts": compute_jobs},
        "utilization_check": {
            "source": utilization_source,
            "samples_percent": utilization_samples,
            "busy_threshold_percent": GPU_BUSY_UTIL_THRESHOLD_PERCENT,
            "probe_errors": utilization_probe_errors,
        },
    }


class NvmlMemorySampler:
    """Sample device-wide NVML VRAM while a model loads and transcribes."""

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.1):
        self.device_index = int(device_index)
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.available = False
        self.error: str | None = None
        self.device_name: str | None = None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.total_bytes: int | None = None
        self._nvml = None
        self._handle = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvmlMemorySampler":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.baseline_bytes = int(info.used)
            self.peak_bytes = self.baseline_bytes
            self.total_bytes = int(info.total)
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self.device_name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            self.available = True
            self._thread = threading.Thread(target=self._sample_loop, name="subtitle-nvml-sampler", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.error = f"NVML memory sampling unavailable: {exc}"
            self._shutdown()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                self.peak_bytes = max(self.peak_bytes or 0, used)
            except Exception as exc:
                self.error = f"NVML sampling stopped: {exc}"
                return

    def _shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 3))
        if self.available and self._nvml is not None and self._handle is not None:
            try:
                used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                self.peak_bytes = max(self.peak_bytes or 0, used)
            except Exception as sample_error:
                self.error = f"NVML final sample failed: {sample_error}"
        self._shutdown()

    def to_dict(self) -> dict[str, Any]:
        delta = None
        if self.peak_bytes is not None and self.baseline_bytes is not None:
            delta = max(0, self.peak_bytes - self.baseline_bytes)
        return {
            "available": self.available,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "baseline_bytes": self.baseline_bytes,
            "peak_bytes": self.peak_bytes,
            "peak_delta_bytes": delta,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "measurement": "device-wide NVML samples; peak may miss spikes shorter than the sample interval",
        }


def format_bytes_mib(value: int | None) -> float | None:
    return None if value is None else round(int(value) / (1024 * 1024), 2)
