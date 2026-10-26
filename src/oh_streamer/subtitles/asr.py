"""Local Japanese ASR with Transformers, optional Silero VAD, and resumable caches.

The heavyweight runtime is imported only inside :func:`run_asr`. This module is
safe to import from the main streamer process, which uses a separate venv from
the restoration engine.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import zlib

from . import runtime
from .quality import f7_low_confidence, japanese_similarity, load_quality_config


ASR_FEATURE_VERSION = "whisper-confidence-gap-empty-spans-v2"
SPAN_PLAN_VERSION = "shared-vad-gap-v1"


def _plan_settings(source_key, *, segmentation, use_vad, vad_threshold,
                   vad_rescue_chunk_seconds, config):
    return {"version": SPAN_PLAN_VERSION, "source_key": source_key,
            "segmentation": segmentation, "use_vad": bool(use_vad),
            "vad": dict(runtime.VAD_PROFILE, threshold=vad_threshold,
                        rescue_chunk_seconds=vad_rescue_chunk_seconds,
                        rescue_policy="disjoint-only-v1"),
            "gap_fill": dict(config["gap_fill"]), "utterance_max_seconds": 15}


def _plan_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read_shared_plan(value, expected_settings=None):
    plan = dict(value) if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))
    if plan.get("settings", {}).get("version") != SPAN_PLAN_VERSION:
        raise ValueError("unsupported shared ASR span plan version")
    if expected_settings is not None and plan.get("settings") != expected_settings:
        raise ValueError("shared ASR span plan source or VAD settings do not match")
    identity = {key: plan[key] for key in ("settings", "speech_intervals", "groups", "audio_duration_seconds")}
    if plan.get("plan_key") != _plan_digest(identity):
        raise ValueError("shared ASR span plan identity does not match spans")
    duration = float(plan["audio_duration_seconds"])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("invalid shared ASR audio duration")
    previous_end = 0.0
    for group in plan["groups"]:
        start, end = float(group["start"]), float(group["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + .0001):
            raise ValueError("invalid shared ASR span")
        if start < previous_end - .0001 or not isinstance(group.get("gap"), bool):
            raise ValueError("shared ASR spans must be ordered, disjoint, and gap-marked")
        previous_end = end
    if not Path(plan["audio_path"]).is_file():
        raise FileNotFoundError("shared ASR audio is missing")
    return plan


def prepare_shared_spans(input_path, out_dir, *, cache_dir=None, quality_config=None,
                         vad_threshold=None, vad_rescue_chunk_seconds=60.0,
                         segmentation="utterance", use_vad=True, gap_fill=True,
                         force=False, on_phase=None, window=None):
    """Prepare one CPU VAD/span plan and one PCM file for sequential model children.

    ``window=(start, end)`` limits decoding and VAD to that source range (progressive blocks of a source that
    is still downloading); the PCM keeps absolute timestamps by padding the front with silence.

    The returned dict is also saved to ``out_dir/asr-plan.json``. Pass that path
    or dict as ``run_asr(shared_plan=...)`` to reuse exact spans and skip VAD.
    Quality/fusion thresholds are excluded from this raw inference identity.
    """
    config = load_quality_config(quality_config)
    config["gap_fill"]["enabled"] = bool(gap_fill and config["gap_fill"]["enabled"])
    threshold = config["vad_threshold"] if vad_threshold is None else vad_threshold
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    if not math.isfinite(vad_rescue_chunk_seconds) or vad_rescue_chunk_seconds < 0:
        raise ValueError("VAD rescue chunk seconds must be non-negative")
    if segmentation not in ("utterance", "grouped") or (segmentation == "utterance" and not use_vad):
        raise ValueError("invalid shared ASR segmentation")
    source_key, source = runtime.source_identity(input_path)
    settings = _plan_settings(source_key, segmentation=segmentation, use_vad=use_vad,
                              vad_threshold=threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds,
                              config=config)
    if window is not None:
        settings["window"] = [float(window[0]), float(window[1])]
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "asr-plan.json"
    if not force and plan_path.is_file():
        try:
            return _read_shared_plan(plan_path, settings)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    import torch
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else runtime.default_cache_dir().resolve()
    wav_name = f"{source_key}.16k-mono.wav" if window is None else f"{source_key}.w{int(window[0] * 1000)}-{int(window[1] * 1000)}.16k-mono.wav"
    audio_path = cache_root / "audio" / wav_name
    started = time.perf_counter()
    def emit(percent, detail):
        if on_phase:
            on_phase("prepare_asr", percent, detail)
    emit(0, "準備共用 ASR 音訊與 VAD 分段")
    if window is None:
        runtime.resample_to_wav(input_path, audio_path)
    else:
        runtime.resample_window_to_wav(input_path, audio_path, window[0], window[1])
    audio = runtime.load_wav_float32(audio_path)
    duration = len(audio) / runtime.SAMPLE_RATE
    shift = 0.0
    if window is not None:
        shift = float(window[0])
        audio = audio[int(round(shift * runtime.SAMPLE_RATE)):]       # the padded silence is not analysed
    emit(40, "共用 PCM 音訊已就緒")
    intervals, vad_seconds, vad_source = _detect_speech_intervals(
        audio, torch=torch, threshold=threshold, rescue_chunk_seconds=vad_rescue_chunk_seconds,
        on_progress=lambda percent, detail: emit(None if percent is None else 40 + .4 * percent, detail),
    )
    groups = (utterance_groups(intervals, audio) if segmentation == "utterance" else
              _inference_groups(intervals, audio_duration=duration - shift, use_vad=use_vad))
    coverage = [{"start": start, "end": end} for start, end in groups]
    gaps = gap_inference_groups(coverage, audio, config=config) if use_vad else []
    if shift:
        intervals = [dict(row, start=row["start"] + shift, end=row["end"] + shift) for row in intervals]
        groups = [(start + shift, end + shift) for start, end in groups]
        gaps = [(start + shift, end + shift) for start, end in gaps]
    gap_set = set(gaps)
    rows = [{"start": start, "end": end, "gap": (start, end) in gap_set}
            for start, end in sorted([*groups, *gaps])]
    identity = {"settings": settings, "speech_intervals": intervals,
                "groups": rows, "audio_duration_seconds": duration}
    plan = dict(identity, plan_key=_plan_digest(identity), source=source,
                source_key=source_key, audio_duration=duration, vad_identity=dict(settings["vad"]),
                audio_path=str(audio_path), plan_path=str(plan_path),
                vad_seconds=vad_seconds, vad_source=vad_source,
                prepare_seconds=time.perf_counter() - started)
    runtime.atomic_write_json(plan_path, plan)
    emit(80, "共用 VAD 與補窗分段已就緒")
    return _read_shared_plan(plan, settings)


def segment_from_record(row):
    """Read new confidence records and historical caches through the same API."""
    from .core import Segment
    return Segment(
        float(row["start"]), float(row["end"]), str(row["text"]),
        no_speech_prob=row.get("no_speech_prob"), avg_logprob=row.get("avg_logprob"),
        compression_ratio=row.get("compression_ratio"),
        gap=bool(row.get("gap", row.get("src") == "gap")),
        low_confidence=bool(row.get("low_confidence", False)),
        alt_text=row.get("alt_text"), flags=tuple(row.get("flags", ())),
        asr_agreement=row.get("asr_agreement", japanese_similarity(str(row["text"]), row.get("alt_text") or "")
                              if row.get("alt_text") else None),
    )


def _compression_ratio(text):
    encoded = text.encode("utf-8")
    return len(encoded) / max(1, len(zlib.compress(encoded)))


class _WhisperConfidence:
    """Capture generation features, then score unmodified generated text tokens.

    Teacher forcing uses the funnel's Japanese/transcribe/no-timestamps prompt;
    scores include EOT, and no-speech probability is measured after SOT. It does
    not change decoding or request score tensors from the pipeline.
    """

    def __init__(self, torch, model, processor):
        self.torch, self.model, self.processor = torch, model, processor
        self.captured = []
        self.original = getattr(model, "generate", None)
        self.errors = []
        if self.original is not None and processor is not None:
            def generate(*args, **kwargs):
                output = self.original(*args, **kwargs)
                self.captured.append((kwargs.get("input_features", args[0] if args else None), output))
                return output
            model.generate = generate

    def reset(self):
        self.captured.clear()

    def close(self):
        if self.original is not None and self.processor is not None:
            self.model.generate = self.original
        self.captured.clear()

    def score(self, text=None):
        empty = {"avg_logprob": None, "no_speech_prob": None}
        if len(self.captured) != 1 or self.processor is None:
            return empty
        try:
            torch, model, tokenizer = self.torch, self.model, self.processor.tokenizer
            features, output = self.captured[0]
            parameter = next(model.parameters())
            features = features.to(parameter.device, parameter.dtype)
            eot = tokenizer.eos_token_id
            prompt = [tokenizer.convert_tokens_to_ids(token) for token in (
                "<|startoftranscript|>", "<|ja|>", "<|transcribe|>", "<|notimestamps|>")]
            if text is None:
                sequences = output if torch.is_tensor(output) else output["sequences"]
                ids = [token for token in sequences[0].tolist() if token < eot]
            else:
                ids = tokenizer.encode(text, add_special_tokens=False)
            full = torch.tensor([prompt + ids + [eot]], device=parameter.device)
            with torch.inference_mode():
                encoder = model.model.encoder(features)
                logits = model(encoder_outputs=encoder, decoder_input_ids=full[:, :-1]).logits[0].float().log_softmax(-1)
                probabilities = logits.gather(1, full[0, 1:, None])[:, 0]
                body = probabilities[len(prompt) - 1:]
                no_speech = None
                for token in ("<|nospeech|>", "<|nocaptions|>"):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None and token_id != tokenizer.unk_token_id:
                        no_speech = float(logits[0, token_id].exp())
                        break
                return {"avg_logprob": float(body.mean()) if len(body) else None,
                        "no_speech_prob": no_speech}
        except (AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError, StopIteration) as exc:
            self.errors.append(str(exc))
            return empty


def gap_inference_groups(intervals, audio, *, sample_rate=16000, config=None):
    """Uncovered >=2s, audible windows <=10s, matching the funnel span rules."""
    import numpy as np
    rules = (config or load_quality_config())["gap_fill"]
    if not rules["enabled"]:
        return []
    minimum, maximum = rules["min_seconds"], rules["max_seconds"]
    if not 0 < minimum <= maximum or rules["min_rms"] < 0:
        raise ValueError("invalid gap fill settings")
    duration = len(audio) / sample_rate
    cursor, groups = 0.0, []
    coverage = sorted((row["start"], row["end"]) for row in intervals)
    for start, end in coverage + [(duration, duration)]:
        if start - cursor >= minimum:
            position = cursor
            while start - position >= minimum:
                stop = min(start, position + maximum)
                samples = audio[int(position * sample_rate):int(stop * sample_rate)]
                if len(samples) and float(np.sqrt(np.mean(samples ** 2))) > rules["min_rms"]:
                    groups.append((position, stop))
                position = stop
        cursor = max(cursor, end)
    return groups


def _load_vad():
    """Load the Silero VAD dependency installed in the isolated subtitle runtime."""

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD is unavailable in the subtitle runtime; install it in the separate subtitle environment"
        ) from exc
    return load_silero_vad(onnx=False), get_speech_timestamps, "silero_vad package"


def _supports_silero_progress(get_speech_timestamps) -> bool:
    """Only pass Silero's progress callback when its installed signature advertises it."""

    try:
        return "progress_tracking_callback" in inspect.signature(
            get_speech_timestamps
        ).parameters
    except (TypeError, ValueError):
        return False


def rescue_missing_intervals(baseline, candidates):
    """Add only disjoint candidates, preserving every original speech boundary."""
    result = [dict(s) for s in baseline]
    for candidate in sorted(candidates, key=lambda s: (s["start"], s["end"])):
        if candidate["end"] <= candidate["start"]:
            continue
        if not any(
            candidate["start"] < segment["end"]
            and candidate["end"] > segment["start"]
            for segment in result
        ):
            result.append(dict(candidate))
    return sorted(result, key=lambda s: (s["start"], s["end"]))


def _detect_speech_intervals(
    audio,
    *,
    torch,
    threshold=0.3,
    rescue_chunk_seconds=60.0,
    on_progress=None,
) -> tuple[list[dict[str, float]], float, str]:
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    if not math.isfinite(rescue_chunk_seconds) or rescue_chunk_seconds < 0:
        raise ValueError("VAD rescue chunk seconds must be non-negative")
    started = time.perf_counter()
    if on_progress:
        on_progress(None, "正在載入 Silero VAD 模型")
    model, get_speech_timestamps, source = _load_vad()
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        model.to("cpu")
    waveform = torch.from_numpy(audio)
    supports_progress = _supports_silero_progress(get_speech_timestamps)
    duration = len(audio) / runtime.SAMPLE_RATE
    rescue_enabled = bool(rescue_chunk_seconds and duration > rescue_chunk_seconds)
    last_percent = -1

    def emit_progress(percent, detail, *, force=False):
        nonlocal last_percent
        if not on_progress:
            return
        if percent is None:
            on_progress(None, detail)
            return
        try:
            percent = max(0, min(100, int(float(percent))))
        except (TypeError, ValueError):
            return
        percent = max(last_percent, percent)
        if force or percent != last_percent:
            last_percent = percent
            on_progress(percent, detail)

    if on_progress:
        emit_progress(None, "Silero VAD 模型已就緒，準備掃描音訊", force=True)
        emit_progress(0, "以 Silero VAD 掃描音訊", force=True)

    def scan_vad(wave, *, base_percent, span_percent, detail):
        scan_last_percent = -1
        options = {
            "sampling_rate": runtime.SAMPLE_RATE,
            "threshold": threshold,
            "min_speech_duration_ms": runtime.VAD_PROFILE["min_speech_duration_ms"],
            "min_silence_duration_ms": runtime.VAD_PROFILE["min_silence_duration_ms"],
            "speech_pad_ms": runtime.VAD_PROFILE["speech_pad_ms"],
            "return_seconds": True,
        }
        if supports_progress and on_progress:
            def track_silero(percent):
                nonlocal scan_last_percent
                try:
                    value = float(percent)
                except (TypeError, ValueError):
                    return
                if not math.isfinite(value):
                    return
                value = max(0.0, min(100.0, value))
                mapped = base_percent + span_percent * value / 100.0
                whole_percent = int(mapped)
                if whole_percent != scan_last_percent:
                    scan_last_percent = whole_percent
                    emit_progress(mapped, detail)

            options["progress_tracking_callback"] = track_silero
        with torch.inference_mode():
            rows = get_speech_timestamps(wave, model, **options)
        end_percent = base_percent + span_percent
        if not supports_progress or scan_last_percent != int(end_percent):
            emit_progress(end_percent, f"{detail}完成", force=True)
        return rows

    primary_span = 50.0 if rescue_enabled else 100.0
    timestamps = scan_vad(
        waveform,
        base_percent=0.0,
        span_percent=primary_span,
        detail="以 Silero VAD 掃描整段音訊",
    )
    intervals = []
    for item in timestamps or []:
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            intervals.append({"start": start, "end": end})
    if rescue_enabled:
        candidates = []
        step = max(1, round(rescue_chunk_seconds * runtime.SAMPLE_RATE))
        offsets = list(range(0, len(audio), step))
        processed_samples = 0
        for chunk_index, offset in enumerate(offsets, start=1):
            model.reset_states()
            chunk = waveform[offset : offset + step]
            chunk_samples = len(chunk)
            base_percent = 50.0 + 50.0 * processed_samples / max(1, len(audio))
            span_percent = 50.0 * chunk_samples / max(1, len(audio))
            detail = f"以 VAD 掃描補漏分段 {chunk_index}/{len(offsets)}"
            emit_progress(base_percent, detail, force=True)
            rows = scan_vad(
                chunk,
                base_percent=base_percent,
                span_percent=span_percent,
                detail=detail,
            )
            processed_samples += chunk_samples
            for row in rows or []:
                start = max(0.0, float(row["start"])) + offset / runtime.SAMPLE_RATE
                end = (
                    min(len(chunk) / runtime.SAMPLE_RATE, float(row["end"]))
                    + offset / runtime.SAMPLE_RATE
                )
                if math.isfinite(start) and math.isfinite(end) and end > start:
                    candidates.append(dict(start=start, end=end))
        intervals = rescue_missing_intervals(intervals, candidates)
    emit_progress(100, "Silero VAD 掃描完成")
    intervals.sort(key=lambda item: (item["start"], item["end"]))
    return intervals, time.perf_counter() - started, source


def _inference_groups(
    intervals: list[dict[str, float]],
    *,
    audio_duration: float,
    use_vad: bool,
    max_group_seconds: float = 25.0,
    max_gap_seconds: float = 2.5,
) -> list[tuple[float, float]]:
    """Combine nearby VAD spans into short context windows for efficient ASR."""

    if not use_vad:
        return [(0.0, audio_duration)] if audio_duration > 0 else []
    if not intervals:
        return []
    groups: list[tuple[float, float]] = []
    start, end = intervals[0]["start"], intervals[0]["end"]
    for interval in intervals[1:]:
        next_start, next_end = interval["start"], interval["end"]
        if next_start - end <= max_gap_seconds and next_end - start <= max_group_seconds:
            end = max(end, next_end)
        else:
            groups.append((start, end))
            start, end = next_start, next_end
    groups.append((start, end))
    return groups


def _make_pipeline(torch, model_id: str, model_path: Path, model_dtype: str, device_index: int):
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is unavailable in the subtitle runtime; install it in the separate subtitle environment"
        ) from exc

    on_cuda = device_index >= 0
    dtype_name = model_dtype if on_cuda else "float32"
    dtype = getattr(torch, dtype_name, torch.float32)
    if model_dtype == "bfloat16" and on_cuda:
        try:
            if not torch.cuda.is_bf16_supported():
                dtype = torch.float16
        except (AttributeError, RuntimeError):
            dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
        "local_files_only": True,
    }
    if on_cuda:
        # Kotoba's official card recommends SDPA for CUDA. Turbo's card also
        # documents SDPA as the supported built-in speed/memory path.
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_path), **model_kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    device = torch.device(f"cuda:{device_index}" if on_cuda else "cpu")
    model.to(device)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    return asr_pipeline, model, processor, str(dtype)


class _QwenAsrPipeline:
    """Adapt Qwen3-ASR to the utterance-mode call shape used by :func:`run_asr`."""

    def __init__(self, model, sample_rate=16000):
        self.model = model
        self.sample_rate = sample_rate

    def __call__(self, payload, return_timestamps=False, generate_kwargs=None, **_ignored):
        if return_timestamps:
            raise ValueError("Qwen3-ASR supports utterance segmentation only")
        options = dict(generate_kwargs or {})
        results = self.model.transcribe(
            audio=(payload["array"], int(payload.get("sampling_rate", self.sample_rate))),
            context=str(options.get("context", "")),
            language=options.get("language", "Japanese"),
        )
        text = results[0].text if results else ""
        return {"text": text or ""}


def _make_qwen_asr_pipeline(torch, model_path: Path, model_dtype: str, device_index: int, generate_kwargs=None):
    """Load Qwen3-ASR lazily; the ``qwen-asr`` package is optional."""

    import sys
    import types

    # qwen_asr imports nagisa only for its forced aligner, which is not used here.
    sys.modules.setdefault("nagisa", types.ModuleType("nagisa"))
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR requires the optional 'qwen-asr' package in the separate subtitle "
            "environment (pip install qwen-asr); it is not installed"
        ) from exc
    on_cuda = device_index >= 0
    dtype = getattr(torch, model_dtype if on_cuda else "float32", torch.float32)
    options = dict(generate_kwargs or {})
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=f"cuda:{device_index}" if on_cuda else "cpu",
        max_inference_batch_size=1,
        max_new_tokens=int(options.get("max_new_tokens", 512)),
    )
    return _QwenAsrPipeline(model), model, None, str(dtype)


def utterance_groups(intervals, audio, *, sample_rate=16000, max_seconds=15.0):
    """Keep VAD boundaries; split long spans at the longest low-energy run.

    Search the latter half of each allowed window, using 20 ms RMS frames.
    If no silence is found, use the duration limit and record that hard cut.
    """
    import numpy as np
    if max_seconds <= 0 or sample_rate <= 0:
        raise ValueError("positive duration and sample rate required")
    groups = []
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        while end - start > max_seconds:
            lo, hi = start + max_seconds / 2, start + max_seconds
            frame = max(1, int(sample_rate * .02))
            samples = audio[int(lo * sample_rate) : int(hi * sample_rate)]
            usable = len(samples) // frame * frame
            rms = np.sqrt(np.mean(samples[:usable].reshape(-1, frame) ** 2, axis=1))
            runs, first = [], None
            for i, quiet in enumerate(list(rms < .01) + [False]):
                if quiet and first is None:
                    first = i
                elif not quiet and first is not None:
                    runs.append((i - first, first, i))
                    first = None
            cut = hi
            if runs:
                _, first, last = max(runs)
                cut = lo + (first + last) / 2 * frame / sample_rate
            groups.append((start, cut))
            start = cut
        if end > start:
            groups.append((start, end))
    return groups


def _json_chunks(result: Any, *, offset: float, piece_duration: float) -> list[dict[str, Any]]:
    """Convert Hugging Face timestamp chunks into global raw segment records."""

    if not isinstance(result, dict):
        return []
    chunks = result.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            return []
        return [{
            "start": round(offset, 3),
            "end": round(offset + piece_duration, 3),
            "text": text.strip(),
            "timestamp_source": "whole_input_fallback",
        }]

    valid_chunks = [
        c for c in chunks
        if isinstance(c, dict) and isinstance(c.get("text"), str) and c.get("text").strip()
    ]
    if not valid_chunks:
        text = result.get("text")
        if isinstance(text, str) and text.strip():
            return [{
                "start": round(offset, 3),
                "end": round(offset + piece_duration, 3),
                "text": text.strip(),
                "timestamp_source": "whole_input_fallback",
            }]
        return []

    output = []
    for i, chunk in enumerate(valid_chunks):
        text = chunk["text"].strip()
        timestamp = chunk.get("timestamp")
        start_local = end_local = None
        if isinstance(timestamp, (tuple, list)) and len(timestamp) >= 2:
            try:
                start_local = float(timestamp[0]) if timestamp[0] is not None else None
                end_local = float(timestamp[1]) if timestamp[1] is not None else None
            except (TypeError, ValueError):
                start_local = end_local = None
        if start_local is None or not math.isfinite(start_local):
            start_local = 0.0
        if end_local is None or not math.isfinite(end_local):
            # Look ahead: if next chunk has a valid start timestamp, snap end_local to it
            if i + 1 < len(valid_chunks):
                next_ts = valid_chunks[i + 1].get("timestamp")
                if isinstance(next_ts, (tuple, list)) and len(next_ts) >= 1 and next_ts[0] is not None:
                    try:
                        end_local = float(next_ts[0])
                    except (TypeError, ValueError):
                        end_local = piece_duration
                else:
                    end_local = piece_duration
            else:
                end_local = piece_duration

        start_local = min(piece_duration, max(0.0, start_local))
        end_local = min(piece_duration, max(start_local, end_local))
        output.append({
            "start": round(offset + start_local, 3),
            "end": round(offset + end_local, 3),
            "text": text,
            "timestamp_source": "transformers_chunk",
        })
    return output


def map_asr_chunks_to_intervals(
    chunks: Sequence[Union[dict[str, Any], Any]],
    intervals: Sequence[Union[tuple[float, float], list[float], dict[str, Any], Any]],
    *,
    snap_threshold: float = 0.8,
) -> list[list[Any]]:
    """Map ASR chunks to target intervals using maximum overlap clamping, preserving all chunks."""
    from .core import assign_segments_to_intervals
    return assign_segments_to_intervals(chunks, intervals, snap_threshold=snap_threshold)


def _cache_matches(path: Path, cache_key: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("cache_key") != cache_key:
        return None
    if not all(isinstance(value.get(key), list) for key in ("segments", "raw_segments", "speech_intervals")):
        return None
    if not isinstance(value.get("metrics"), dict):
        return None
    return value


def _atomic_remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class ASRSession:
    """Caller-owned, single-model pipeline reused across independent ASR jobs.

    Audio, VAD, progress and confidence capture remain local to each run_asr.
    The owner must close this session before loading another GPU model.
    """

    def __init__(self):
        self.identity = None
        self.pipeline = None
        self.torch = None
        self.device_index = -1

    def acquire(self, identity, factory, torch, device_index):
        if self.identity is not None and self.identity != identity:
            raise ValueError("ASR session cannot mix model or decoding identities")
        if self.pipeline is None:
            # Retain cleanup ownership even if model creation raises midway.
            self.torch, self.device_index = torch, device_index
            self.pipeline = factory()
            self.identity = identity
        return self.pipeline

    def close(self):
        self.pipeline = None
        self.identity = None
        if self.torch is not None:
            # Restoring a model's bound generate method can leave a self-reference cycle.
            # Collect its tensors before returning cached CUDA allocations to the allocator;
            # otherwise an error followed by a same-process reload may hold both models.
            gc.collect()
        if self.torch is not None and self.device_index >= 0:
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
        self.torch = None
        self.device_index = -1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def run_asr(
    input_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    model: str = "anime-whisper",
    use_vad: bool = True,
    cache_dir: str | os.PathLike[str] | None = None,
    force: bool = False,
    model_path: str | os.PathLike[str] | None = None,
    segmentation: str = "grouped",
    vad_threshold: float | None = None,
    vad_rescue_chunk_seconds: float = 60.0,
    resume_partial: bool = True,
    gpu_managed: bool = False,
    on_progress=None,
    on_phase=None,
    generate_overrides: dict[str, Any] | None = None,
    gap_fill: bool = True,
    quality_config: dict | None = None,
    shared_plan: dict | str | os.PathLike[str] | None = None,
    session: ASRSession | None = None,
) -> dict[str, Any]:
    """Transcribe local/URL audio to Japanese segments and write ``ja.json``/``ja.srt``.

    The returned dictionary has ``segments``, ``raw_segments``, ``speech_intervals``
    and ``metrics``. Whisper confidence is teacher-forced from captured generate
    features; unsupported backends expose explicit null confidence values.
    """

    started = time.perf_counter()
    config = load_quality_config(quality_config)
    config["gap_fill"]["enabled"] = bool(gap_fill and config["gap_fill"]["enabled"])
    if vad_threshold is None:
        vad_threshold = config["vad_threshold"]
    phase_percent: dict[str, int] = {}
    phase_none_detail: dict[str, str] = {}
    phase_detail: dict[str, str] = {}

    def report_phase(phase, percent, detail, *, force=False):
        if not callable(on_phase):
            return
        if percent is None:
            detail = str(detail)
            retained_percent = phase_percent.get(phase)
            if retained_percent is None:
                if not force and phase_none_detail.get(phase) == detail:
                    return
                phase_none_detail[phase] = detail
                on_phase(phase, None, detail)
            else:
                if not force and phase_detail.get(phase) == detail:
                    return
                phase_detail[phase] = detail
                on_phase(phase, retained_percent, detail)
            return
        try:
            value = max(0, min(100, int(round(float(percent)))))
        except (TypeError, ValueError):
            return
        value = max(phase_percent.get(phase, 0), value)
        if (
            phase_percent.get(phase) == value
            and (not force or phase_detail.get(phase) == str(detail))
        ):
            return
        phase_percent[phase] = value
        phase_detail[phase] = str(detail)
        on_phase(phase, value, str(detail))

    report_phase("prepare_asr", None, "正在檢查本機 ASR 模型檔案與音訊快取")
    if segmentation not in ("grouped", "utterance"):
        raise ValueError("segmentation must be grouped or utterance")
    if segmentation == "utterance" and not use_vad:
        raise ValueError("utterance segmentation requires VAD")
    if not math.isfinite(vad_threshold) or not 0 < vad_threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    if not math.isfinite(vad_rescue_chunk_seconds) or vad_rescue_chunk_seconds < 0:
        raise ValueError("VAD rescue chunk seconds must be non-negative")
    alias, spec = runtime.resolve_model(model)
    backend = spec.get("backend", "hf-pipeline")
    allowed_segmentations = spec.get("segmentations")
    if allowed_segmentations and segmentation not in allowed_segmentations:
        raise ValueError(
            f"{alias} supports segmentation {', '.join(allowed_segmentations)} only"
        )
    overrides = dict(generate_overrides or {})
    if overrides:
        spec["generate_kwargs"] = {**spec["generate_kwargs"], **overrides}
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else runtime.default_cache_dir()
    output_root = Path(out_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved_model_path = (
        Path(model_path).expanduser() if model_path is not None else runtime.default_model_path(alias)
    )
    resolved_model_path = resolved_model_path.resolve()
    model_fingerprint = runtime.local_model_fingerprint(resolved_model_path)

    source_key, source = runtime.source_identity(input_path)
    plan = None
    if shared_plan is not None:
        if segmentation != "utterance":
            raise ValueError("shared ASR spans require utterance segmentation")
        settings = _plan_settings(source_key, segmentation=segmentation, use_vad=use_vad,
                                  vad_threshold=vad_threshold, vad_rescue_chunk_seconds=vad_rescue_chunk_seconds,
                                  config=config)
        plan = _read_shared_plan(shared_plan, settings)
    model_id = spec["model_id"]
    cache_key = runtime.make_job_key(source_key, model_id, use_vad, model_fingerprint)
    cache_key = runtime.make_job_key(
        cache_key,
        ASR_FEATURE_VERSION + ":" + json.dumps(config["gap_fill"], sort_keys=True),
        use_vad,
    )
    if plan is not None:
        cache_key = runtime.make_job_key(cache_key, "shared-plan:" + plan["plan_key"], use_vad)
    if vad_rescue_chunk_seconds or vad_threshold != 0.5:
        cache_key = runtime.make_job_key(
            cache_key,
            f"vad-disjoint-rescue-v1:{vad_threshold}:{vad_rescue_chunk_seconds}",
            use_vad,
        )
    if segmentation != "grouped":
        cache_key = runtime.make_job_key(cache_key, "utterance-energy-v1-15s", use_vad)
    if overrides:
        cache_key = runtime.make_job_key(
            cache_key,
            "generate-overrides:" + json.dumps(overrides, sort_keys=True, ensure_ascii=False),
            use_vad,
        )
    json_path = output_root / "ja.json"
    srt_path = output_root / "ja.srt"
    cached = None if force else _cache_matches(json_path, cache_key)
    if cached is not None:
        cached = dict(cached)
        for field in ("segments", "raw_segments"):
            cached[field] = [dict(row, low_confidence=f7_low_confidence(segment_from_record(row), config))
                             for row in cached[field]]
        from .core import Segment, clean_segments
        from .postprocess import render_fixed_srt
        restored_segments = [segment_from_record(row) for row in cached["segments"]]
        restored_intervals = [tuple(interval) for interval in cached["speech_intervals"]]
        runtime.atomic_write_text(
            srt_path,
            render_fixed_srt(
                clean_segments(
                    restored_segments,
                    vad_intervals=restored_intervals,
                    quality_config=config,
                ),
                japanese=True,
            ),
        )
        cached = dict(cached)
        cached_metrics = dict(cached.get("metrics") or {})
        cache_wall_seconds = time.perf_counter() - started
        cached_metrics["cache_hit"] = True
        cached_metrics["cache_read_seconds"] = round(cache_wall_seconds, 3)
        cached["metrics"] = cached_metrics
        cached["json_path"] = str(json_path)
        cached["srt_path"] = str(srt_path)
        report_phase("prepare_asr", 100, "ASR 結果已由快取準備完成", force=True)
        report_phase("asr", 100, "已從快取還原完整 ASR 結果", force=True)
        return cached

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is unavailable in the subtitle runtime; use the separate subtitle environment"
        ) from exc

    use_cuda = bool(torch.cuda.is_available())
    gpu_safety: dict[str, Any]
    if use_cuda:
        # This check must run before model downloads or creation of a CUDA context.
        # Queue subprocesses already hold the same GPU lease as restoration.
        # Standalone PoC calls retain the strict idle-device safety audit.
        gpu_safety = (
            {"required": False, "status": "queue worker GPU admission"}
            if gpu_managed
            else runtime.assert_gpu_safe()
        )
        device_index = 0
    else:
        gpu_safety = {"required": False, "status": "cpu inference; CUDA safety probe not required"}
        device_index = -1

    if plan is not None:
        prepare_started = time.perf_counter()
        source_audio = Path(plan["audio_path"])
        audio = runtime.load_wav_float32(source_audio)
        audio_duration = len(audio) / runtime.SAMPLE_RATE
        if abs(audio_duration - plan["audio_duration_seconds"]) > 1 / runtime.SAMPLE_RATE:
            raise ValueError("shared ASR PCM duration no longer matches the span plan")
        speech_intervals = [dict(row) for row in plan["speech_intervals"]]
        inference_groups = [(row["start"], row["end"]) for row in plan["groups"]]
        gap_groups = [(row["start"], row["end"]) for row in plan["groups"] if row["gap"]]
        gap_spans = set(gap_groups)
        prepare_seconds = time.perf_counter() - prepare_started
        vad_seconds, vad_source, interval_setup_seconds = 0.0, "shared span plan (VAD reused)", 0.0
        report_phase("prepare_asr", 80, "已載入共用 PCM 音訊與相同 ASR 分段", force=True)
    else:
        audio_dir = cache_root / "audio"
        source_audio = audio_dir / f"{source_key}.16k-mono.wav"
        prepare_started = time.perf_counter()

        def report_audio_progress(current_seconds, total_seconds):
            if total_seconds is not None and total_seconds > 0:
                audio_percent = max(0.0, min(100.0, current_seconds * 100.0 / total_seconds))
                report_phase(
                    "prepare_asr",
                    40.0 * audio_percent / 100.0,
                    f"以 FFmpeg 擷取音訊（{current_seconds:.1f}/{total_seconds:.1f} 秒）",
                )
            else:
                report_phase(
                    "prepare_asr",
                    None,
                    f"FFmpeg 音訊目前已解碼至 {current_seconds:.1f} 秒；總長度未知",
                )

        report_phase("prepare_asr", None, "以 FFmpeg 準備 16 kHz 單聲道音訊")
        audio_duration = runtime.resample_to_wav(
            input_path,
            source_audio,
            on_progress=report_audio_progress if callable(on_phase) else None,
        )
        report_phase("prepare_asr", None, "將擷取的 PCM 音訊載入記憶體")
        audio = runtime.load_wav_float32(source_audio)
        report_phase("prepare_asr", 40, "音訊擷取與 PCM 載入完成", force=True)
        prepare_seconds = time.perf_counter() - prepare_started

        vad_started = time.perf_counter()
        # Keep the same Silero reference for the VAD-on/off comparison. The switch
        # controls only whether ASR receives the full waveform or VAD-trimmed groups.
        speech_intervals, vad_seconds, vad_source = _detect_speech_intervals(
            audio,
            torch=torch,
            threshold=vad_threshold,
            rescue_chunk_seconds=vad_rescue_chunk_seconds,
            on_progress=(
                (
                    lambda percent, detail: report_phase(
                        "prepare_asr",
                        None if percent is None else 40.0 + 0.4 * percent,
                        detail,
                        force=percent is None,
                    )
                )
                if callable(on_phase)
                else None
            ),
        )
        report_phase("prepare_asr", 80, "VAD 掃描完成，ASR 分組已就緒", force=True)
        inference_groups = _inference_groups(
            speech_intervals,
            audio_duration=audio_duration,
            use_vad=use_vad,
        )
        if segmentation == "utterance":
            inference_groups = utterance_groups(speech_intervals, audio)
        coverage = [{"start": start, "end": end} for start, end in inference_groups]
        gap_groups = gap_inference_groups(coverage, audio, config=config) if use_vad else []
        gap_spans = set(gap_groups)
        inference_groups = sorted([*inference_groups, *gap_groups])
        # VAD model initialization takes place above; retain wall time for any added
        # interval setup while exposing the VAD inference time separately.
        interval_setup_seconds = max(0.0, time.perf_counter() - vad_started - vad_seconds)

    progress_path = cache_root / "jobs" / f"{cache_key}.progress.json"
    progress: dict[str, Any] = {}
    if not force and resume_partial:
        try:
            candidate = json.loads(progress_path.read_text(encoding="utf-8"))
            if (isinstance(candidate, dict) and candidate.get("cache_key") == cache_key
                    and candidate.get("inference_groups") == [list(group) for group in inference_groups]
                    and isinstance(candidate.get("completed"), dict)):
                progress = candidate
        except (OSError, json.JSONDecodeError):
            progress = {}
    completed = progress.setdefault("completed", {})

    processing_started = time.perf_counter()
    model_load_seconds = 0.0
    inference_seconds = 0.0
    raw_by_group: dict[int, list[dict[str, Any]]] = {}
    for key, value in completed.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(inference_groups) and isinstance(value, list):
            raw_by_group[index] = value

    todo_group_indexes = [index for index in range(len(inference_groups)) if index not in raw_by_group]
    resumed_group_count = len(raw_by_group)
    sampler_context = runtime.NvmlMemorySampler(device_index) if use_cuda and todo_group_indexes else nullcontext(None)
    pipe = model_instance = processor = None
    nvml_sampler = None
    confidence = None
    confidence_errors = []
    actual_dtype = "not-loaded-no-speech" if not inference_groups else "not-loaded-all-groups-resumed"
    try:
        if todo_group_indexes:
            with sampler_context as sampler:
                nvml_sampler = sampler
                report_phase("prepare_asr", 80, "正在載入本機 ASR 模型與處理器", force=True)
                load_started = time.perf_counter()
                def create_pipeline():
                    if backend == "qwen-asr":
                        return _make_qwen_asr_pipeline(
                            torch, resolved_model_path, spec["torch_dtype"], device_index,
                            spec["generate_kwargs"],
                        )
                    return _make_pipeline(
                        torch, model_id, resolved_model_path, spec["torch_dtype"], device_index)

                identity = (backend, model_id, str(resolved_model_path), model_fingerprint,
                            spec["torch_dtype"], device_index,
                            json.dumps(spec["generate_kwargs"], sort_keys=True))
                pipe, model_instance, processor, actual_dtype = (
                    session.acquire(identity, create_pipeline, torch, device_index)
                    if session is not None else create_pipeline())
                model_load_seconds = time.perf_counter() - load_started
                confidence = _WhisperConfidence(torch, model_instance, processor)
                report_phase("prepare_asr", 100, "本機 ASR 模型與處理器已就緒", force=True)
                report_phase(
                    "asr",
                    0,
                    f"正在辨識 {len(todo_group_indexes)} 個待處理音訊分組",
                    force=True,
                )
                kwargs = dict(spec["generate_kwargs"])
                run_context = torch.inference_mode()
                for group_index in todo_group_indexes:
                    start_s, end_s = inference_groups[group_index]
                    first_sample = max(0, int(start_s * runtime.SAMPLE_RATE))
                    last_sample = min(len(audio), int(end_s * runtime.SAMPLE_RATE))
                    if last_sample <= first_sample:
                        raw_by_group[group_index] = []
                        continue
                    sample = audio[first_sample:last_sample]
                    payload = {"array": sample, "sampling_rate": runtime.SAMPLE_RATE}
                    call_started = time.perf_counter()
                    confidence.reset()
                    with run_context:
                        if segmentation == "utterance" or (start_s, end_s) in gap_spans:
                            result = pipe(
                                payload,
                                return_timestamps=False,
                                generate_kwargs=kwargs,
                            )
                        else:
                            result = pipe(
                                payload,
                                return_timestamps=True,
                                chunk_length_s=25 if alias.startswith("kotoba") else 30,
                                stride_length_s=5,
                                generate_kwargs=kwargs,
                            )
                    if segmentation == "utterance" or (start_s, end_s) in gap_spans:
                        raw_by_group[group_index] = (
                            [
                                {
                                    "start": round(start_s, 3),
                                    "end": round(end_s, 3),
                                    "text": result["text"],
                                    "timestamp_source": "vad_utterance_boundary",
                                }
                            ]
                        )
                    else:
                        raw_by_group[group_index] = _json_chunks(
                            result,
                            offset=start_s,
                            piece_duration=(last_sample - first_sample) / runtime.SAMPLE_RATE,
                        )
                    for row in raw_by_group[group_index]:
                        row.update(confidence.score(
                            None if segmentation == "utterance" or (start_s, end_s) in gap_spans
                            else row["text"]
                        ))
                        row["compression_ratio"] = _compression_ratio(row["text"])
                        row["gap"] = (start_s, end_s) in gap_spans
                        row["src"] = "gap" if row["gap"] else "vad"
                        row["low_confidence"] = f7_low_confidence(segment_from_record(row), config)
                    inference_seconds += time.perf_counter() - call_started
                    completed[str(group_index)] = raw_by_group[group_index]
                    if resume_partial:
                        runtime.atomic_write_json(
                            progress_path,
                            {
                                "schema_version": runtime.SCHEMA_VERSION,
                                "cache_key": cache_key,
                                "inference_groups": [
                                    list(group) for group in inference_groups
                                ],
                                "completed": completed,
                            },
                        )
                    if on_progress:
                        on_progress(len(raw_by_group), len(inference_groups))
                    report_phase(
                        "asr",
                        100.0 * len(raw_by_group) / max(1, len(inference_groups)),
                        f"已辨識 {len(raw_by_group)}/{len(inference_groups)} 個音訊分組",
                    )
                report_phase(
                    "asr",
                    100,
                    f"ASR 辨識完成，處理 {len(raw_by_group)}/{len(inference_groups)} 個音訊分組",
                )
                if use_cuda:
                    torch.cuda.synchronize(device_index)
        elif inference_groups:
            report_phase("prepare_asr", 100, "ASR 準備完成；所有分組皆已從部分結果還原", force=True)
            report_phase(
                "asr",
                100,
                f"已還原全部 {len(inference_groups)} 個 ASR 分組",
                force=True,
            )
        else:
            report_phase("prepare_asr", 100, "ASR 準備完成；沒有需要模型處理的語音分組", force=True)
            report_phase("asr", 100, "沒有語音分組需要辨識", force=True)
    finally:
        if confidence is not None:
            confidence_errors = confidence.errors
            confidence.close()
            confidence = None
        pipe = model_instance = processor = None
        if use_cuda and session is None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    processing_seconds = time.perf_counter() - processing_started
    raw_segments = [segment for index in range(len(inference_groups)) for segment in raw_by_group.get(index, [])]
    raw_segments.sort(key=lambda segment: (float(segment.get("start", 0.0)), float(segment.get("end", 0.0))))
    # A resumed raw checkpoint can predate a quality-only setting change.
    raw_segments = [dict(row, low_confidence=f7_low_confidence(segment_from_record(row), config))
                    for row in raw_segments]
    segments = [dict(item) for item in raw_segments if str(item.get("text", "")).strip()]

    # The core module owns cue timing/line wrapping. Raw ASR records remain intact
    # in JSON so later cleanup and subtitle-quality checks can inspect model output.
    from .core import Segment, clean_segments, inspect_hallucinations, render_srt

    core_segments = [
        segment_from_record(item)
        for item in segments
    ]
    vad_intervals = [
        (float(interval["start"]), float(interval["end"])) for interval in speech_intervals
    ]
    hallucinations = inspect_hallucinations(core_segments, vad_intervals=vad_intervals)
    cleaned_segments = clean_segments(core_segments, vad_intervals=vad_intervals, quality_config=config)
    from .postprocess import render_fixed_srt
    srt_text = render_fixed_srt(cleaned_segments, japanese=True)

    rtf = (
        audio_duration / processing_seconds
        if processing_seconds > 0 and inference_groups and not resumed_group_count
        else None
    )
    wall_seconds = time.perf_counter() - started
    wall_rtf = (
        audio_duration / wall_seconds
        if wall_seconds > 0 and not resumed_group_count
        else None
    )
    sampler_metrics = nvml_sampler.to_dict() if nvml_sampler is not None else {
        "available": False,
        "device_index": None,
        "device_name": None,
        "baseline_bytes": None,
        "peak_bytes": None,
        "peak_delta_bytes": None,
        "total_bytes": None,
        "error": "NVML sampling is only used for CUDA inference",
    }
    metrics = {
        "segmentation": segmentation,
        "vad_profile": dict(
            runtime.VAD_PROFILE,
            threshold=vad_threshold,
            rescue_chunk_seconds=vad_rescue_chunk_seconds,
            rescue_policy="disjoint-only-v1",
        ),
        "utterance_max_seconds": 15 if segmentation == "utterance" else None,
        "gap_fill": dict(config["gap_fill"]),
        "gap_group_count": len(gap_groups),
        "asr_feature_version": ASR_FEATURE_VERSION,
        "shared_plan_key": plan["plan_key"] if plan is not None else None,
        "shared_vad_reused": plan is not None,
        "model": alias,
        "model_id": model_id,
        "model_path": str(resolved_model_path),
        "asr_backend": backend,
        "generate_kwargs": dict(spec["generate_kwargs"]),
        "use_vad": bool(use_vad),
        "audio_duration_seconds": round(audio_duration, 3),
        "prepare_seconds": round(prepare_seconds, 3),
        "vad_seconds": round(vad_seconds, 3),
        "vad_interval_setup_seconds": round(interval_setup_seconds, 3),
        "vad_source": vad_source,
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "processing_seconds": round(processing_seconds, 3),
        "processing_seconds_definition": (
            "ASR model loading, inference, and progress checkpoints; excludes audio preparation and VAD"
        ),
        "wall_seconds": round(wall_seconds, 3),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "realtime_factor_wall": round(wall_rtf, 4) if wall_rtf is not None else None,
        "resumed_group_count": resumed_group_count,
        "segment_count": len(segments),
        "raw_segment_count": len(raw_segments),
        "cleaned_segment_count": len(cleaned_segments),
        "hallucinations": hallucinations,
        "device": "cuda:0" if use_cuda else "cpu",
        "torch_dtype": actual_dtype,
        "nvml": sampler_metrics,
        "gpu_safety": gpu_safety,
        "confidence_fields": {
            "no_speech_prob": "teacher-forced SOT no-speech probability; null if unavailable",
            "avg_logprob": "teacher-forced mean text plus EOT log probability; null if unavailable",
            "compression_ratio": "UTF-8 byte count / zlib compressed byte count",
        },
        "confidence_errors": confidence_errors[:5],
        "cache_hit": False,
    }
    serialized_speech_intervals = [
        [float(interval["start"]), float(interval["end"])] for interval in speech_intervals
    ]
    document = {
        "schema_version": runtime.SCHEMA_VERSION,
        "cache_key": cache_key,
        "source": source,
        "model": alias,
        "model_id": model_id,
        "model_path": str(resolved_model_path),
        "use_vad": bool(use_vad),
        "segments": segments,
        "raw_segments": raw_segments,
        "speech_intervals": serialized_speech_intervals,
        "shared_plan_path": plan.get("plan_path") if plan is not None else None,
        "shared_plan_key": plan["plan_key"] if plan is not None else None,
        "metrics": metrics,
        "json_path": str(json_path),
        "srt_path": str(srt_path),
    }
    runtime.atomic_write_json(json_path, document)
    runtime.atomic_write_text(srt_path, srt_text)
    _atomic_remove(progress_path)

    result = dict(document)
    result["metrics"] = dict(metrics)
    return result
