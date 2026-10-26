"""Chunk planning, ffmpeg silence snapping, overlap deduplication, and checkpoints for cloud ASR."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import runtime
from .core import Segment, render_srt
from .model_adapters import ModelAdapterError


def detect_silence_intervals(
    wav_path: str | os.PathLike[str],
    *,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -30.0,
    ffmpeg: Optional[str] = None,
    timeout: float = 60.0,
    close_open_at: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Detect silence intervals in a WAV file using ffmpeg silencedetect filter without PyTorch."""
    try:
        executable = ffmpeg or runtime.resolve_ffmpeg()
    except Exception:
        return []

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cmd = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        os.fspath(wav_path),
        "-af",
        f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
        output = proc.stderr or ""
    except (subprocess.SubprocessError, OSError):
        return []

    intervals: List[Tuple[float, float]] = []
    current_start: Optional[float] = None

    for line in output.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                current_start = max(0.0, float(m.group(1)))
        elif "silence_end:" in line:
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m and current_start is not None:
                end_ts = float(m.group(1))
                if end_ts > current_start:
                    intervals.append((current_start, end_ts))
                current_start = None

    if current_start is not None and close_open_at is not None and close_open_at > current_start:
        intervals.append((current_start, close_open_at))      # silent to the end of the audio
    return intervals


class ChunkSpec(NamedTuple):
    index: int
    start: float
    end: float


SNAP_RADIUS_SECONDS = 30.0      # a cut may move this far from its nominal time (both directions)


def cut_window(current_start: float, *, sp_end: float, target_chunk_sec: float = 300.0,
               min_chunk_sec: float = 120.0, max_chunk_sec: float = 600.0) -> Tuple[float, float, float]:
    """(ideal_cut, window_start, window_end) of the silence search for the chunk starting at current_start."""
    ideal_cut = current_start + target_chunk_sec
    window_start = max(current_start + min_chunk_sec, ideal_cut - SNAP_RADIUS_SECONDS)
    window_end = min(current_start + max_chunk_sec, ideal_cut + SNAP_RADIUS_SECONDS, sp_end - 10.0)
    return ideal_cut, window_start, window_end


def snap_cut(current_start: float, local_silences: Sequence[Tuple[float, float]], sp_end: float, *,
             target_chunk_sec: float = 300.0, min_chunk_sec: float = 120.0, max_chunk_sec: float = 600.0,
             overlap_sec: float = 0.0) -> Tuple[float, float]:
    """(chunk_end, next_start) of one chunk that is longer than target_chunk_sec: the silence nearest to the
    nominal cut inside the search window, else the nominal cut. Shared by the whole-file plan and by the lazy
    cuts decided while a source is still downloading, so both pick the same cut on the same audio."""
    ideal_cut, window_start, window_end = cut_window(
        current_start, sp_end=sp_end, target_chunk_sec=target_chunk_sec,
        min_chunk_sec=min_chunk_sec, max_chunk_sec=max_chunk_sec)
    if any(s_s <= ideal_cut <= s_e for s_s, s_e in local_silences):
        return ideal_cut, ideal_cut
    best_silence = None
    best_dist = float("inf")
    if window_start < window_end:
        for sil_start, sil_end in local_silences:
            if sil_end >= window_start and sil_start <= window_end:
                mid = (max(sil_start, window_start) + min(sil_end, window_end)) / 2.0
                dist = abs(mid - ideal_cut)
                if dist < best_dist:
                    best_dist = dist
                    best_silence = (sil_start, sil_end)
    if best_silence is not None:
        cut_point = (max(best_silence[0], window_start) + min(best_silence[1], window_end)) / 2.0
        return cut_point, cut_point
    return min(sp_end, ideal_cut + overlap_sec), ideal_cut


def detect_window_silences(source: str | os.PathLike[str], start: float, end: float, *,
                           ffmpeg: Optional[str] = None) -> List[Tuple[float, float]]:
    """Silence intervals (absolute seconds) of source[start, end) from a bounded ffmpeg decode, so only that
    range of a still-downloading source is read. A silence that runs to the window edge is clipped there."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="silence-window-") as tmp:
        wav = Path(tmp) / "window.wav"
        runtime.resample_window_to_wav(source, wav, start, end, ffmpeg=ffmpeg, pad=False)
        length = runtime.wav_duration(wav)
        found = detect_silence_intervals(wav, ffmpeg=ffmpeg, close_open_at=length)
    return [(start + a, start + b) for a, b in found]


def plan_chunks(
    duration: float,
    *,
    target_chunk_sec: float = 300.0,
    min_chunk_sec: float = 120.0,
    max_chunk_sec: float = 600.0,
    overlap_sec: float = 0.0,
    silence_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    skip_long_silence_sec: float = 10.0,
) -> List[Tuple[int, float, float]]:
    """Plan chunk boundaries, skipping long silence intervals and snapping to silence near target_chunk_sec.

    Returns a list of ChunkSpec (chunk_index, start_seconds, end_seconds).
    """
    if duration <= 0.0:
        return []

    silences = sorted(list(silence_intervals or []), key=lambda s: s[0])

    # 1. Identify long silence exclusions if skip_long_silence_sec is enabled
    exclusions: List[Tuple[float, float]] = []
    if skip_long_silence_sec and skip_long_silence_sec > 0.0:
        for s_start, s_end in silences:
            if (s_end - s_start) >= skip_long_silence_sec:
                # Keep 0.4s breathing buffer on both sides
                pad_start = 0.0 if s_start <= 0.05 else (s_start + 0.4)
                pad_end = duration if s_end >= (duration - 0.05) else (s_end - 0.4)
                if pad_end > pad_start:
                    exclusions.append((max(0.0, pad_start), min(duration, pad_end)))

    # Merge overlapping exclusion intervals
    merged_exclusions: List[Tuple[float, float]] = []
    for ex in exclusions:
        if not merged_exclusions:
            merged_exclusions.append(ex)
        else:
            prev_s, prev_e = merged_exclusions[-1]
            if ex[0] <= prev_e:
                merged_exclusions[-1] = (prev_s, max(prev_e, ex[1]))
            else:
                merged_exclusions.append(ex)

    # 2. Derive speech-active intervals from complements of exclusions
    speech_intervals: List[Tuple[float, float]] = []
    curr_t = 0.0
    for ex_s, ex_e in merged_exclusions:
        if ex_s > curr_t and (ex_s - curr_t) >= 0.05:
            speech_intervals.append((curr_t, ex_s))
        curr_t = max(curr_t, ex_e)
    if curr_t < duration and (duration - curr_t) >= 0.05:
        speech_intervals.append((curr_t, duration))

    # If entire audio was flagged as silence or no speech intervals remained, fallback to full duration
    if not speech_intervals:
        speech_intervals = [(0.0, duration)]

    # If only 1 speech interval covering whole duration and <= target_chunk_sec
    if len(speech_intervals) == 1 and (speech_intervals[0][1] - speech_intervals[0][0]) <= target_chunk_sec:
        return [ChunkSpec(0, speech_intervals[0][0], speech_intervals[0][1])]

    chunks: List[Tuple[int, float, float]] = []
    idx = 0

    for sp_start, sp_end in speech_intervals:
        sp_duration = sp_end - sp_start
        if sp_duration <= target_chunk_sec:
            chunks.append(ChunkSpec(idx, sp_start, sp_end))
            idx += 1
            continue

        # Sub-divide long speech interval using remaining shorter silences
        local_silences = [
            (s_s, s_e) for s_s, s_e in silences
            if s_e >= sp_start and s_s <= sp_end and ((s_e - s_s) < (skip_long_silence_sec or float("inf")))
        ]

        current_start = sp_start
        while current_start < sp_end:
            remaining = sp_end - current_start
            if remaining <= target_chunk_sec:
                chunks.append(ChunkSpec(idx, current_start, sp_end))
                idx += 1
                break

            chunk_end, next_start = snap_cut(
                current_start, local_silences, sp_end, target_chunk_sec=target_chunk_sec,
                min_chunk_sec=min_chunk_sec, max_chunk_sec=max_chunk_sec, overlap_sec=overlap_sec)

            chunks.append(ChunkSpec(idx, current_start, chunk_end))
            current_start = next_start
            idx += 1

    return chunks


# Degenerate segments (end <= start once the overlap / duration clamps are applied) are repaired, not fatal.
# A response is "badly broken" - and the chunk fails with kind='bad_response' - when more than
# min(MAX_REPAIRED_SEGMENTS, max(MIN_REPAIR_BUDGET, 10% of the segments)) segments needed a guessed repair
# (span or merge), or, for chunks of >= 4 segments, when more than half of them did. Repairs that follow
# the response's own word timestamps are exact and are not counted.
MIN_REPAIR_BUDGET = 3
MAX_REPAIRED_SEGMENTS = 10
REPAIR_BAD_FRACTION = 0.1
REPAIR_MIN_SPAN = 0.2        # a guessed span shorter than this is not worth a cue: merge the text instead
REPAIR_SEC_PER_CHAR = 0.12
REPAIR_MAX_SPAN = 2.0
_REPAIR_LOG_TEXT = 40
_WORD_MIN_SPAN = 0.05

_logger = logging.getLogger(__name__)


def _bad_response(message: str) -> ModelAdapterError:
    return ModelAdapterError(message, kind="bad_response")


def _clean_words(words: Any) -> List[Dict[str, Any]]:
    """Word timings of one segment as plain {word, start, end} rows; anything unusable is dropped."""
    rows: List[Dict[str, Any]] = []
    if not isinstance(words, list):
        return rows
    for w in words:
        try:
            row = dict(word=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(row["start"]) and math.isfinite(row["end"]):
            rows.append(row)
    return rows


def _log_repair(label: str, index: int, raw: Tuple[float, float], fixed: Optional[Tuple[float, float]],
                text: str, how: str) -> None:
    shown = text if len(text) <= _REPAIR_LOG_TEXT else text[:_REPAIR_LOG_TEXT] + "..."
    new = "-" if fixed is None else f"{fixed[0]:.3f}-{fixed[1]:.3f}"
    _logger.warning("Cloud ASR %s segment %d degenerate (raw %.3f-%.3f): repaired by %s -> %s %r",
                    label, index, raw[0], raw[1], how, new, shown)


def clamp_and_validate_segments(
    raw_segments: List[Dict[str, Any]],
    chunk_duration: float,
    *,
    max_tolerance: float = 0.1,
    label: str = "chunk",
) -> List[Dict[str, Any]]:
    """Validate segments, clamping minor floating point timestamp overlaps or end-bounds.

    Preserves avg_logprob, no_speech_prob, compression_ratio, and marks low_confidence. Keeps the word timings
    of a segment ('words') when the response carries them.

    A segment whose end is not after its start (a zero-length Whisper segment, or one the overlap / duration
    clamp emptied) is repaired and logged instead of failing the whole chunk: first from its own word
    timestamps, else a small span bounded by the next segment start and the chunk duration, else its text is
    merged into a neighbour. Text is never dropped. Overlaps and ends past the audio are clamped (logged when
    beyond the tolerance, and then counted as repairs); empty-text segments are skipped. Missing fields,
    non-finite times and too many repairs (see MAX_REPAIRED_SEGMENTS) raise ModelAdapterError(kind='bad_response').
    """
    if not isinstance(raw_segments, list):
        raise _bad_response("Cloud ASR requires verbose_json segments with ordered nonoverlapping timestamps")

    parsed: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for item in raw_segments:
        try:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            raise _bad_response("Cloud ASR segment missing valid start/end/text") from None
        if not (math.isfinite(start) and math.isfinite(end)):
            raise _bad_response("Cloud ASR segment with non-finite timestamp")
        if not text:                            # nothing to show: skip it rather than fail the whole chunk
            _logger.info("Cloud ASR %s segment with empty text skipped (%.3f-%.3f)", label, start, end)
            continue
        parsed.append((start, end, text, item))

    clamped: List[Dict[str, Any]] = []
    last_end = 0.0
    carry: Optional[Tuple[float, str, List[Dict[str, Any]]]] = None    # (start, text, words) merged into the next one
    guessed = 0

    for idx, (start, end, text, item) in enumerate(parsed):
        raw_pair = (start, end)
        words = _clean_words(item.get("words"))

        # Check start ordering / overlap
        if start < last_end:
            overlap = last_end - start
            if overlap > max_tolerance:         # clamped too, but counted against the repair budget below
                guessed += 1
                _logger.warning("Cloud ASR %s segment %d overlaps the previous one by %.3f s (raw %.3f-%.3f): "
                                "start clamped to %.3f", label, idx, overlap, start, end, last_end)
            start = last_end

        # Check end bound
        if end > chunk_duration:
            excess = end - chunk_duration
            if excess > max_tolerance + 0.1:    # clamped too, but counted against the repair budget below
                guessed += 1
                _logger.warning("Cloud ASR %s segment %d ends %.3f s after the audio (raw %.3f-%.3f): "
                                "end clamped to %.3f", label, idx, excess, start, end, chunk_duration)
            end = chunk_duration

        if carry is not None:                   # text of a degenerate predecessor, merged forward
            start, text, words = min(start, carry[0]), carry[1] + text, carry[2] + words
            carry = None

        repaired = None
        if end <= start:
            next_start = parsed[idx + 1][0] if idx + 1 < len(parsed) else chunk_duration
            room_end = min(max(next_start, start), chunk_duration)       # the repaired cue must end here
            word_end = max((w["end"] for w in words), default=start)
            guess = min(max(0.3, REPAIR_SEC_PER_CHAR * len(text)), REPAIR_MAX_SPAN)
            if min(word_end, room_end) - start >= _WORD_MIN_SPAN:
                end, repaired = min(word_end, room_end), "words"
            else:
                guessed += 1
                if min(start + guess, room_end) - start >= REPAIR_MIN_SPAN:
                    end, repaired = min(start + guess, room_end), "span"
            if repaired is not None:
                _log_repair(label, idx, raw_pair, (start, end), text,
                            "word timestamps" if repaired == "words" else "estimated span")
            elif idx + 1 < len(parsed):
                _log_repair(label, idx, raw_pair, None, text, "merging into the next segment")
                carry = (start, text, words)
                continue
            elif clamped:
                _log_repair(label, idx, raw_pair, None, text, "merging into the previous segment")
                prev = clamped[-1]
                prev["text"] += text
                if words:
                    prev["words"] = prev.get("words", []) + words
                prev["timing_repair"] = "merged"
                continue
            else:
                # The only segment of the chunk and no room left after its start: place it at the chunk end.
                span = min(guess, chunk_duration)
                if span < REPAIR_MIN_SPAN:
                    raise _bad_response("Cloud ASR segment end must be strictly greater than start")
                start, end, repaired = chunk_duration - span, chunk_duration, "span"
                _log_repair(label, idx, raw_pair, (start, end), text, "placement at the chunk end")

        # Extract confidence metrics if present
        avg_logprob = float(item["avg_logprob"]) if "avg_logprob" in item and item["avg_logprob"] is not None else None
        no_speech_prob = float(item["no_speech_prob"]) if "no_speech_prob" in item and item["no_speech_prob"] is not None else None
        compression_ratio = float(item["compression_ratio"]) if "compression_ratio" in item and item["compression_ratio"] is not None else None

        # Rule for low_confidence: high no_speech_prob (> 0.6) or very low avg_logprob (< -1.0)
        low_conf = False
        if avg_logprob is not None and avg_logprob < -1.0:
            low_conf = True
        if no_speech_prob is not None and no_speech_prob > 0.6:
            low_conf = True

        seg = dict(
            start=start,
            end=end,
            text=text,
            low_confidence=low_conf,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )
        if words:
            seg["words"] = words
        if repaired:
            seg["timing_repair"] = repaired
        clamped.append(seg)
        last_end = end

    allowed = min(MAX_REPAIRED_SEGMENTS, max(MIN_REPAIR_BUDGET, math.ceil(REPAIR_BAD_FRACTION * len(parsed))))
    if guessed > allowed or (len(parsed) >= 4 and guessed * 2 > len(parsed)):
        raise _bad_response(f"Cloud ASR response has too many degenerate segments ({guessed} of {len(parsed)})")
    return clamped


def deduplicate_segments(
    segments: List[Dict[str, Any]],
    *,
    match_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Deduplicate segments near chunk overlap boundaries."""
    if not segments:
        return []

    # Sort strictly by start time
    sorted_segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
    result: List[Dict[str, Any]] = []

    for seg in sorted_segs:
        if not result:
            result.append(dict(seg))
            continue

        prev = result[-1]
        seg_text = str(seg.get("text") or seg.get("text_ja") or "").strip()
        prev_text = str(prev.get("text") or prev.get("text_ja") or "").strip()
        # Check if current segment is duplicate of previous segment
        if seg["start"] < prev["end"]:
            # Overlapping segments: if texts match, drop or merge
            if seg_text and seg_text == prev_text:
                prev["end"] = max(prev["end"], seg["end"])
                continue
            # If start is slightly overlapping with previous end, clamp start
            if seg["end"] > prev["end"]:
                seg = dict(seg)
                seg["start"] = prev["end"]
                result.append(seg)
            # If completely enclosed and different, keep or drop depending on length
            continue

        result.append(dict(seg))

    return result


def reconstruct_timeline(
    chunks: Sequence[Tuple[int, float, float] | ChunkSpec],
    chunk_segments: Sequence[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Reconstruct whole-file timeline by shifting each chunk's segments by chunk.start and deduplicating."""
    combined: List[Dict[str, Any]] = []
    for chunk, segs in zip(chunks, chunk_segments):
        start_offset = chunk[1] if isinstance(chunk, (tuple, list)) else getattr(chunk, "start", 0.0)
        for s in segs:
            seg_copy = dict(s)
            seg_copy["start"] = round(float(s["start"]) + float(start_offset), 4)
            seg_copy["end"] = round(float(s["end"]) + float(start_offset), 4)
            if isinstance(s.get("words"), list):
                seg_copy["words"] = [dict(w, start=round(float(w["start"]) + float(start_offset), 4),
                                          end=round(float(w["end"]) + float(start_offset), 4)) for w in s["words"]]
            combined.append(seg_copy)
    return deduplicate_segments(combined)


def extract_chunk_audio(
    source_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    start_sec: float,
    duration_sec: float,
    *,
    fmt: str = "flac",
    ffmpeg: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Path:
    """Extract a slice of audio compressed as 16kHz mono FLAC (or WAV).

    A range_cache marker (``.rcache``) is read through ``source_url`` (its loopback runtime_url; resolved from
    the live session when omitted) instead of the marker file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    from oh_streamer.sources import media_input
    source_name = os.fspath(source_path).split("?", 1)[0].split("#", 1)[0]
    if media_input.needs_runtime_url(source_name):
        # Progressive single-file source: plain accurate input seek over HTTP Range, no WAV fallback on the marker.
        try:
            executable = ffmpeg or runtime.resolve_ffmpeg()
            input_arg, input_opts = media_input.ffmpeg_input(source_path, source_url)
            cmd = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *input_opts,
                   "-ss", f"{max(0.0, float(start_sec)):.3f}", "-t", f"{max(0.05, float(duration_sec)):.3f}",
                   "-i", input_arg, "-vn", "-ac", "1", "-ar", "16000"]
            cmd += (["-c:a", "flac", "-compression_level", "5"] if fmt.lower() == "flac"
                    else ["-c:a", "pcm_s16le"])
            cmd.append(os.fspath(out))
            subprocess.run(cmd, capture_output=True, check=True, creationflags=flags, timeout=180)
            if not out.is_file() or out.stat().st_size == 0:
                raise RuntimeError("FFmpeg did not produce the chunk audio")
            return out
        except Exception as exc:
            raise RuntimeError(f"Failed to extract chunk audio from {source_path}") from exc
    if media_input.is_remote_marker(source_name):
        # Streaming item: read the remote audio directly with input -ss plus -copyts (exact trim on audio-only HLS, see
        # runtime.resample_window_to_wav; only the needed segments/ranges are
        # fetched; no HLS prefix decode, no WAV fallback on the marker). Output goes to a partial file first so a
        # run that skipped a failed segment (exit 0, shorter audio) never leaves a chunk behind.
        partial = out.with_name(f".{out.stem}.partial{out.suffix}")

        def attempt() -> None:
            executable = ffmpeg or runtime.resolve_ffmpeg()
            input_arg, input_opts = media_input.ffmpeg_input(source_path)
            cmd = [executable, "-hide_banner", "-loglevel", "warning", "-nostdin", "-y", "-copyts", *input_opts,
                   "-ss", f"{max(0.0, float(start_sec)):.3f}", "-t", f"{max(0.05, float(duration_sec)):.3f}",
                   "-i", input_arg, "-vn", "-ac", "1", "-ar", "16000"]
            cmd += (["-c:a", "flac", "-compression_level", "5"] if fmt.lower() == "flac"
                    else ["-c:a", "pcm_s16le"])
            cmd.append(os.fspath(partial))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                        creationflags=flags, timeout=180, check=False)
            except subprocess.TimeoutExpired:
                raise RuntimeError("ffmpeg timed out while extracting chunk audio from <remote>") from None
            runtime.raise_for_remote_failure(result.returncode, result.stderr, input_arg)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("FFmpeg did not produce the chunk audio")

        try:
            media_input.run_remote(source_path, attempt)
            os.replace(partial, out)
            return out
        except Exception as exc:
            raise RuntimeError(f"Failed to extract chunk audio from {source_path}: {exc}") from None
        finally:
            try:
                os.unlink(partial)
            except OSError:
                pass
    if Path(source_name).suffix.lower() in (".m3u8", ".m3u"):
        # Share the local executor's exact HLS timestamp origin and decoded-prefix trim.
        # The generic input -ss below is not accurate for progressive HLS. In particular,
        # do not catch an HLS decode error and try the WAV-only fallback on its playlist.
        import tempfile
        try:
            executable = ffmpeg or runtime.resolve_ffmpeg()
            with tempfile.TemporaryDirectory(prefix=".hls-chunk-", dir=out.parent) as folder:
                decoded = Path(folder) / "window.wav"
                start = max(0.0, float(start_sec))
                duration = max(0.05, float(duration_sec))
                runtime.resample_window_to_wav(source_path, decoded, start, start + duration,
                                               ffmpeg=executable, pad=False, timeout=120)
                if fmt.lower() == "flac":
                    encoded = Path(folder) / "window.flac"
                    cmd = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-i", os.fspath(decoded), "-vn", "-ac", "1", "-ar", "16000",
                           "-c:a", "flac", "-compression_level", "5", os.fspath(encoded)]
                    subprocess.run(cmd, capture_output=True, check=True, creationflags=flags, timeout=120)
                    if not encoded.is_file() or encoded.stat().st_size == 0:
                        raise RuntimeError("FFmpeg did not produce the HLS FLAC window")
                else:
                    encoded = decoded
                os.replace(encoded, out)
            return out
        except Exception as exc:
            raise RuntimeError(f"Failed to extract HLS chunk audio from {source_path}") from exc

    try:
        executable = ffmpeg or runtime.resolve_ffmpeg()
    except Exception:
        executable = None

    if executable:
        cmd = [
            executable,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss", f"{max(0.0, float(start_sec)):.3f}",
            "-t", f"{max(0.05, float(duration_sec)):.3f}",
            "-i", os.fspath(source_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
        ]
        if fmt.lower() == "flac":
            cmd += ["-c:a", "flac", "-compression_level", "5", os.fspath(out)]
        else:
            cmd += ["-c:a", "pcm_s16le", os.fspath(out)]
        try:
            subprocess.run(cmd, capture_output=True, check=True, creationflags=flags)
            if out.is_file() and out.stat().st_size > 0:
                return out
        except Exception:
            pass

    # Fallback using wave if source is WAV and ffmpeg failed or not found
    try:
        import wave
        with wave.open(os.fspath(source_path), "rb") as audio:
            rate = audio.getframerate()
            params = audio.getparams()
            start_frame = int(max(0.0, start_sec) * rate)
            num_frames = int(max(0.05, duration_sec) * rate)
            audio.setpos(start_frame)
            frames = audio.readframes(num_frames)
            with wave.open(os.fspath(out), "wb") as target:
                target.setparams(params)
                target.writeframes(frames)
            return out
    except Exception as exc:
        raise RuntimeError(f"Failed to extract chunk audio from {source_path}") from exc


def extract_chunk_wav(
    source_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    start_sec: float,
    duration_sec: float,
    *,
    ffmpeg: Optional[str] = None,
) -> Path:
    """Backward-compatible wrapper to extract slice as WAV."""
    return extract_chunk_audio(source_path, out_path, start_sec, duration_sec, fmt="wav", ffmpeg=ffmpeg)


class ChunkCheckpoint:
    """Manages per-chunk serialization to resume interrupted cloud ASR without re-transcribing."""

    def __init__(self, folder: Path | str, audio_hash: str):
        self.folder = Path(folder)
        self.audio_hash = audio_hash
        self.folder.mkdir(parents=True, exist_ok=True)

    def _chunk_file(self, idx: int) -> Path:
        return self.folder / f"chunk_{idx:04d}.json"

    def get_chunk(self, idx: int) -> Optional[List[Dict[str, Any]]]:
        target = self._chunk_file(idx)
        if not target.is_file():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("audio_hash") == self.audio_hash and "segments" in data:
                return data["segments"]
        except Exception:
            return None
        return None

    def get_source(self, idx: int):
        try:
            data = json.loads(self._chunk_file(idx).read_text('utf-8'))
            return data.get('source') if data.get('audio_hash') == self.audio_hash else None
        except (OSError, ValueError):
            return None

    def save_chunk(self, idx: int, start: float, end: float, segments: List[Dict[str, Any]], source=None):
        target = self._chunk_file(idx)
        data = {
            "chunk_index": idx,
            "start": start,
            "end": end,
            "audio_hash": self.audio_hash,
            "segments": segments,
            "source": source,
        }
        runtime.atomic_write_json(target, data)
