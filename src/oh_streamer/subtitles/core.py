"""Pure subtitle parsing, cleanup, formatting, and translation helpers.

This module deliberately keeps ASR, model clients, and OpenCC optional. Callers
inject translation callbacks and, when desired, an OpenCC-compatible converter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Optional, Sequence, Union


MIN_DURATION = 0.7
MAX_DURATION = 7.0
MAX_LINE_CHARS = 18
DEFAULT_BATCH_SIZE = 30
DEFAULT_CONTEXT_SIZE = 5
DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_AVG_LOGPROB_THRESHOLD = -1.0


def has_japanese_residue(text: str) -> bool:
    """Kana in a Chinese result is a review hint, not a semantic accuracy score."""
    return bool(re.search(r"[\u3041-\u3096\u30a1-\u30fa]", text))

_TIME_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[,.](?P<millis>\d{1,3})$"
)
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.)、:：\-]\s*(.*?)\s*$")
_PUNCTUATION_RE = re.compile(
    r"[\s\u3000\.,!?;:'\"()\[\]{}<>，。！？；：、…「」『』【】（）《》〈〉・]+"
)
_INTERJECTION_RE = re.compile(
    r"^(?:あっ+|あぁ+|ああ+|んっ+|んん+|ん+|うっ+|うう+|はぁ+|はあ+|ふぅ+|ふう+|えっ+|ええ+|おお+|あんっ+)$"
)

REFUSAL_MARKERS = (
    "無法協助",
    "不能協助",
    "無法提供",
    "不能提供",
    "無法翻譯",
    "不能翻譯",
    "无法协助",
    "不能协助",
    "无法提供",
    "不能提供",
    "无法翻译",
    "不能翻译",
    "違反安全政策",
    "违反安全政策",
    "違反使用條款",
    "违反使用条款",
    "違反政策",
    "违反政策",
    "無法處理此請求",
    "无法处理此请求",
    "抱歉，我無法",
    "抱歉，我无法",
    "i cannot help",
    "i can't help",
    "i cannot provide",
    "i can't provide",
    "i cannot translate",
    "i can't translate",
    "i am unable to",
    "i'm unable to",
    "i cannot fulfill",
    "i can't fulfill",
    "as an ai",
    "content policy",
    "safety policy",
)


def is_refusal_text(text: str) -> bool:
    """Detect whether output represents an LLM safety refusal or policy block."""
    if not text:
        return False
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in REFUSAL_MARKERS)


@dataclass(frozen=True)
class Segment:
    """A subtitle cue with optional Whisper confidence values."""

    start: float
    end: float
    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    flags: tuple[str, ...] = ()
    compression_ratio: float | None = None
    gap: bool = False
    low_confidence: bool = False
    alt_text: str | None = None
    asr_agreement: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("start", self.start),
            ("end", self.end),
            ("no_speech_prob", self.no_speech_prob),
            ("avg_logprob", self.avg_logprob),
            ("compression_ratio", self.compression_ratio),
            ("asr_agreement", self.asr_agreement),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"subtitle {name} must be finite")
        if self.start < 0 or self.end < 0:
            raise ValueError("subtitle timestamps must be non-negative")
        if self.end < self.start:
            raise ValueError("subtitle end must be at or after start")


@dataclass(frozen=True)
class TranslationBatch:
    """One numbered batch plus unnumbered source context from the prior batch."""

    items: tuple[tuple[int, Segment], ...]
    context: tuple[Segment, ...] = ()

    @property
    def numbered_text(self) -> str:
        return "\n".join(f"{number}. {segment.text}" for number, segment in self.items)

    @property
    def context_text(self) -> str:
        if not self.context:
            return ""
        lines = "\n".join(f"- {segment.text}" for segment in self.context)
        return f"前文上下文（僅供參考，不要翻譯）：\n{lines}"

    @property
    def prompt(self) -> str:
        instruction = "請將以下日文逐句翻譯為簡體中文，保留每句編號；只輸出編號與簡體中文譯文。"
        parts = [instruction]
        if self.context_text:
            parts.append(self.context_text)
        parts.append(self.numbered_text)
        return "\n\n".join(parts)

    @property
    def expected_ids(self) -> tuple[int, ...]:
        return tuple(number for number, _ in self.items)


@dataclass(frozen=True)
class TranslationResult:
    source: Segment
    text: str
    preserved_original: bool = False
    is_filtered: bool = False
    filter_reason: str | None = None


@dataclass(frozen=True)
class TranslationRun:
    results: tuple[TranslationResult, ...]
    metrics: dict[str, int]

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


def parse_timestamp(value: str) -> float:
    """Parse an SRT timestamp such as ``00:00:01,250`` into seconds."""

    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid SRT timestamp: {value!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("millis").ljust(3, "0"))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp, rounding to the nearest millisecond."""

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timestamp must be finite and non-negative")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def parse_srt(text: str) -> list[Segment]:
    """Parse SRT text, preserving each cue's text lines as one string."""

    source = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", source.strip()) if source.strip() else []
    segments: list[Segment] = []
    for block_number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 2:
            raise ValueError(f"invalid SRT cue {block_number}: missing timestamp or text")
        timing_line_index = 1 if lines[0].strip().isdigit() else 0
        timing = lines[timing_line_index].strip()
        if "-->" not in timing:
            raise ValueError(f"invalid SRT cue {block_number}: missing time separator")
        start_text, end_text = (part.strip() for part in timing.split("-->", 1))
        if not start_text or not end_text:
            raise ValueError(f"invalid SRT cue {block_number}: incomplete timestamp")
        start_text, end_text = start_text.split()[0], end_text.split()[0]
        start, end = parse_timestamp(start_text), parse_timestamp(end_text)
        cue_text = "\n".join(lines[timing_line_index + 1 :]).strip()
        if not cue_text:
            raise ValueError(f"invalid SRT cue {block_number}: empty text")
        segments.append(Segment(start=start, end=end, text=cue_text))
    return segments


def _char_class(char: str) -> str:
    if "㐀" <= char <= "鿿" or "豈" <= char <= "﫿":
        return "han"
    if "぀" <= char <= "ヿ" or "ｦ" <= char <= "ﾟ":
        return "kana"
    if char.isascii() and char.isalnum():
        return "latin"
    return ""


def _safe_split_point(text: str) -> int | None:
    """Index at which to cut ``text`` near its middle without breaking a word: punctuation / spaces first, then a
    boundary between different scripts. None when the text has no such boundary (do not cut it)."""

    def allowed(i: int) -> bool:
        before, after = _char_class(text[i - 1]), _char_class(text[i])
        return not (before and before == after) and bool(text[:i].strip()) and bool(text[i:].strip())

    middle = len(text) / 2
    soft = [
        i for i in range(1, len(text))
        if allowed(i) and (text[i - 1] in "，。！？；：、,.!?;: 　" or text[i] in " 　")
    ]
    if soft:
        return min(soft, key=lambda i: abs(i - middle))
    hard = [i for i in range(1, len(text)) if allowed(i)]
    return min(hard, key=lambda i: abs(i - middle)) if hard else None


def _text_chunks(text: str, max_line_chars: int) -> list[str]:
    """Wrap text at a soft punctuation boundary and return chunks of two lines."""

    normalized = re.sub(r"[\s\u3000]+", " ", text).strip()
    if not normalized:
        return []
    lines: list[str] = []
    remaining = normalized
    while len(remaining) > max_line_chars:
        split_at = max_line_chars
        # Prefer a natural break in the latter half of the line.
        for index in range(max_line_chars, max(max_line_chars // 2, 0), -1):
            if index < len(remaining) and remaining[index - 1] in "，。！？；：、,.!?;: ":
                split_at = index
                break
        line = remaining[:split_at].strip()
        if not line:
            line = remaining[:max_line_chars]
            split_at = max_line_chars
        lines.append(line)
        remaining = remaining[split_at:].strip()
    if remaining:
        lines.append(remaining)
    return ["\n".join(lines[index : index + 2]) for index in range(0, len(lines), 2)]


def _allocate_durations(
    weights: Sequence[int],
    total_duration: float,
    min_duration: float,
    max_duration: float,
) -> list[float]:
    """Allocate a bounded interval proportionally while keeping its final edge exact."""

    if not weights:
        return []
    total_weight = sum(weights)
    count = len(weights)
    if total_duration <= 0:
        return [0.0] * count
    if total_duration < count * min_duration:
        # The available interval is too short to honor every minimum. Preserve
        # the text and its time range without creating overlap with the next cue.
        durations = [total_duration * weight / total_weight for weight in weights]
        durations[-1] = max(0.0, total_duration - math.fsum(durations[:-1]))
        return durations

    durations: list[float] = []
    remaining_duration = total_duration
    remaining_weight = total_weight
    for index, weight in enumerate(weights):
        remaining_count = count - index - 1
        if remaining_count == 0:
            duration = remaining_duration
        else:
            proportional = remaining_duration * weight / remaining_weight
            lowest = max(min_duration, remaining_duration - remaining_count * max_duration)
            highest = min(max_duration, remaining_duration - remaining_count * min_duration)
            duration = min(max(proportional, lowest), highest)
        durations.append(duration)
        remaining_duration -= duration
        remaining_weight -= weight
    durations[-1] = max(0.0, total_duration - math.fsum(durations[:-1]))
    return durations


def _merge_simultaneous_segments(segments: Iterable[Segment]) -> list[Segment]:
    """Combine cues with identical starts so the next-start bound is non-empty."""

    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: item.start):
        if not merged or merged[-1].start != segment.start:
            merged.append(segment)
            continue
        previous = merged[-1]
        merged[-1] = replace(previous,
            start=previous.start,
            end=max(previous.end, segment.end),
            text=" ".join(part for part in (previous.text.strip(), segment.text.strip()) if part),
            gap=previous.gap or segment.gap,
            low_confidence=previous.low_confidence or segment.low_confidence,
            flags=tuple(dict.fromkeys((*previous.flags, *segment.flags))),
        )
    return merged


def _segments_for_srt(
    segments: Iterable[Segment],
    *,
    max_line_chars: int,
    min_duration: float,
    max_duration: float,
) -> list[Segment]:
    if max_line_chars < 1:
        raise ValueError("max_line_chars must be positive")
    if (
        not math.isfinite(min_duration)
        or not math.isfinite(max_duration)
        or min_duration <= 0
        or max_duration <= 0
        or min_duration > max_duration
    ):
        raise ValueError("subtitle duration bounds must be positive and ordered")

    output: list[Segment] = []
    ordered_segments = _merge_simultaneous_segments(segments)
    for index, segment in enumerate(ordered_segments):
        chunks = _text_chunks(segment.text, max_line_chars)
        if not chunks:
            continue
        next_start = (
            ordered_segments[index + 1].start if index + 1 < len(ordered_segments) else None
        )
        target_end = max(segment.end, segment.start + min_duration)
        if next_start is not None:
            target_end = min(target_end, next_start)
        duration = max(0.0, target_end - segment.start)
        # Long utterances are divided into more cues when their text has room.
        desired_count = max(len(chunks), math.ceil(duration / max_duration))
        if desired_count > len(chunks):
            flat_text = "".join(chunks).replace("\n", "")
            chunks = _text_chunks(flat_text, max_line_chars)
            while len(chunks) < desired_count:
                # Cut the widest chunk that has a safe boundary: never inside a Han / kana / latin word.
                halves = None
                for widest in sorted(range(len(chunks)), key=lambda i: -len(chunks[i].replace("\n", ""))):
                    widest_text = chunks[widest].replace("\n", "")
                    cut = _safe_split_point(widest_text)
                    if cut is not None:
                        halves = _text_chunks(widest_text[:cut].strip(), max_line_chars) + _text_chunks(
                            widest_text[cut:].strip(), max_line_chars
                        )
                        break
                if halves is None:
                    break
                chunks[widest : widest + 1] = halves
            if len(chunks) < desired_count:
                # A very short utterance cannot be split into enough distinct
                # captions to cover a long interval without repeating text.
                duration = min(duration, len(chunks) * max_duration)

        weights = [max(1, len(chunk.replace("\n", ""))) for chunk in chunks]
        durations = _allocate_durations(weights, duration, min_duration, max_duration)
        cursor = segment.start
        for chunk, cue_duration in zip(chunks, durations):
            proposed_end = cursor + cue_duration
            output.append(
                replace(segment,
                    start=cursor,
                    end=proposed_end,
                    text=chunk,
                )
            )
            cursor = proposed_end
    return output


def render_srt(
    segments: Iterable[Segment],
    *,
    max_line_chars: int = MAX_LINE_CHARS,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> str:
    """Render bounded SRT cues without running one sentence into the next.

    The next cue's start is the hard boundary. If that leaves less than the
    requested minimum, the available interval wins; unsplittable text is capped
    at the maximum cue duration rather than repeating content.
    """

    cues = _segments_for_srt(
        segments,
        max_line_chars=max_line_chars,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    blocks = [
        f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues, 1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _load_blacklist() -> dict[str, list]:
    path = Path(__file__).with_name("blacklist.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entries": [], "credit_patterns": []}
    return data if isinstance(data, dict) else {"entries": [], "credit_patterns": []}


_DEFAULT_BLACKLIST = _load_blacklist()


def _compact(text: str) -> str:
    return _PUNCTUATION_RE.sub("", text).casefold()


def _has_speech_overlap(segment: Segment, vad_intervals: Sequence[tuple[float, float]]) -> bool:
    return any(end > segment.start and start < segment.end for start, end in vad_intervals)


def _blacklisted(
    text: str,
    *,
    speech_present: bool,
    blacklist: dict | Sequence[str],
) -> bool:
    compact_text = _compact(text)
    if isinstance(blacklist, dict):
        entries = blacklist.get("entries", ())
        patterns = blacklist.get("credit_patterns", ())
    else:
        entries = blacklist
        patterns = ()

    for entry in entries:
        if isinstance(entry, str):
            phrase, match_kind, non_speech_only = entry, "contains", False
        elif isinstance(entry, dict):
            phrase = str(entry.get("text", ""))
            match_kind = entry.get("match", "contains")
            non_speech_only = bool(entry.get("non_speech_only", False))
        else:
            continue
        if not phrase or (non_speech_only and speech_present):
            continue
        compact_phrase = _compact(phrase)
        if (match_kind == "exact" and compact_text == compact_phrase) or (
            match_kind != "exact" and compact_phrase in compact_text
        ):
            return True

    for pattern in patterns:
        try:
            if re.search(str(pattern), text, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _truncate_repeated_ngram(text: str, *, max_ngram: int = 12) -> str:
    """Compress repeated runs to two copies without discarding dialogue after them.

    A cue consisting entirely of a repeated run is noise; punctuation does not
    make such a cue dialogue. Keep the historical name for callers/diagnostics.
    """
    start = 0
    while start < len(text):
        max_size = min(max_ngram, (len(text) - start) // 3)
        matched = False
        for size in range(1, max_size + 1):
            ngram = text[start : start + size]
            if _compact(ngram) and text.startswith(ngram * 3, start):
                end = start + size * 3
                while text.startswith(ngram, end):
                    end += size
                tail = _compact(text[end:])
                if not _compact(text[:start]) and (
                    not tail or _compact(ngram).startswith(tail)
                ):
                    return ""
                text = text[:start] + ngram * 2 + text[end:]
                start += size * 2
                matched = True
                break
        if not matched:
            start += 1
    return text


def is_interjection(text: str) -> bool:
    """Return whether a cue is one or more common Japanese exclamations."""

    parts = [
        part.strip(" \t\r\n\u3000\"'「」『』()（）[]【】")
        for part in re.split(r"[,，、。.!！?？;；:：…・\s\u3000]+", text)
    ]
    parts = [part for part in parts if part]
    return bool(parts) and all(_INTERJECTION_RE.fullmatch(_compact(part)) for part in parts)


def clean_segments(
    segments: Iterable[Segment],
    *,
    blacklist: dict | Sequence[str] | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    show_interjections: bool = False,
    no_speech_threshold: float | None = None,
    avg_logprob_threshold: float | None = None,
    include_low_confidence: bool = False,
    quality_config: dict | None = None,
) -> list[Segment]:
    """Apply F7 and hard hallucination drops; retain flagged lows for the full track.

    VAD intervals remain available for legacy context-dependent blacklist entries,
    but never act as a cue deletion gate. Explicit legacy confidence overrides are
    supported for diagnostic callers; production uses the configured F7 rules.
    """

    from .quality import f7_low_confidence, hallucination_keys, key_text, load_quality_config, normalize_japanese

    config = load_quality_config(quality_config)
    hallucinations = hallucination_keys()
    repetition = config["f7"]
    loop = re.compile(r"(.{1," + str(repetition["repeat_max_unit"]) + r"}?)\1{" + str(repetition["repeat_count"] - 1) + r",}")

    rules = _DEFAULT_BLACKLIST if blacklist is None else blacklist
    cleaned: list[Segment] = []
    previous_key: str | None = None
    for segment in segments:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if not text:
            continue

        speech_present = True
        if vad_intervals is not None:
            speech_present = _has_speech_overlap(segment, vad_intervals)

        if (key_text(text) in hallucinations or loop.search(normalize_japanese(text))
                or _truncate_repeated_ngram(text) != text):
            continue
        low_confidence = f7_low_confidence(segment, config)
        if low_confidence and not include_low_confidence:
            continue

        if (
            no_speech_threshold is not None
            and segment.no_speech_prob is not None
            and segment.no_speech_prob > no_speech_threshold
        ):
            continue
        if (
            avg_logprob_threshold is not None
            and segment.avg_logprob is not None
            and segment.avg_logprob < avg_logprob_threshold
        ):
            continue
        if _blacklisted(text, speech_present=speech_present, blacklist=rules):
            continue
        if is_interjection(text) and not show_interjections:
            continue

        text = _truncate_repeated_ngram(text)
        key = _compact(text)
        if not text or (key and key == previous_key):
            continue
        flags = segment.flags
        if is_interjection(text) and "interjection" not in flags:
            flags = (*flags, "interjection")
        cleaned.append(
            replace(segment,
                text=text,
                flags=flags,
                low_confidence=low_confidence,
            )
        )
        previous_key = key
    return cleaned


def inspect_hallucinations(
    segments: Iterable[Segment],
    *,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    blacklist: dict | Sequence[str] | None = None,
) -> dict[str, int]:
    """Count suspected cues by cleanup rule; ``unique_suspect`` counts cue indices once."""

    rows = list(segments)
    rules = _DEFAULT_BLACKLIST if blacklist is None else blacklist
    counts: dict[str, int] = {
        "blacklist": 0,
        "consecutive_repeat": 0,
        "frequent_repeat": 0,
        "outside_vad": 0,
        "repeated_ngram": 0,
        "unique_suspect": 0,
    }
    suspects: set[int] = set()
    frequencies: dict[str, list[int]] = {}
    for index, segment in enumerate(rows):
        key = _compact(segment.text)
        if key:
            frequencies.setdefault(key, []).append(index)
        speech_present = True if vad_intervals is None else _has_speech_overlap(segment, vad_intervals)
        if _blacklisted(segment.text, speech_present=speech_present, blacklist=rules):
            counts["blacklist"] += 1
            suspects.add(index)
        if index and key and key == _compact(rows[index - 1].text):
            counts["consecutive_repeat"] += 1
            suspects.add(index)
        if vad_intervals is not None and not speech_present:
            counts["outside_vad"] += 1
            suspects.add(index)
        if _truncate_repeated_ngram(segment.text) != segment.text:
            counts["repeated_ngram"] += 1
            suspects.add(index)

    for indexes in frequencies.values():
        if len(indexes) >= 3:
            counts["frequent_repeat"] += len(indexes)
            suspects.update(indexes)
    counts["unique_suspect"] = len(suspects)
    return counts


def estimate_translation_max_tokens(
    japanese_items: Any,
    *,
    tokens_per_char: float = 1.25,
    cue_overhead: int = 4,
    safety_ratio: float = 0.20,
    min_safety: int = 32,
    min_tokens_per_cue: int = 16,
    cap: Optional[int] = None,
) -> int:
    """Calculate content-aware dynamic max_tokens with headroom for cloud gateway reservation.

    Formula derived from empirical subtitle length distributions:
      Content = sum(max(ceil(len(text) * tokens_per_char), min_tokens_per_cue))
      Overhead = N * cue_overhead (formatting: "1. ...\\n")
      Margin = max(min_safety, ceil(safety_ratio * Content))
      Total = Content + Overhead + Margin
    """
    if japanese_items is None:
        return min_safety

    texts: list[str] = []
    if isinstance(japanese_items, str):
        lines = [line.strip() for line in japanese_items.splitlines() if line.strip()]
        texts = lines if lines else ([japanese_items.strip()] if japanese_items.strip() else [])
    elif isinstance(japanese_items, Sequence):
        for item in japanese_items:
            if isinstance(item, dict):
                # Chat message format: extract user content
                if item.get("role") == "user":
                    content = str(item.get("content", ""))
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    texts.extend(lines if lines else ([content.strip()] if content.strip() else []))
            elif hasattr(item, "text"):
                texts.append(str(item.text))
            elif isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], "text"):
                # (index, Segment) tuple from batch.items
                texts.append(str(item[1].text))
            else:
                s = str(item).strip()
                if s:
                    texts.append(s)
    else:
        s = str(japanese_items).strip()
        if s:
            texts.append(s)

    if not texts:
        return min_safety

    content_est = sum(
        max(int(math.ceil(len(t) * tokens_per_char)), min_tokens_per_cue)
        for t in texts
    )
    structural = len(texts) * cue_overhead
    margin = max(min_safety, int(math.ceil(content_est * safety_ratio)))
    total = content_est + structural + margin
    if cap is not None:
        total = min(total, cap)
    return total


def make_translation_batches(
    segments: Sequence[Segment],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    min_batch_size: int = 20,
    max_batch_size: int = 40,
) -> list[TranslationBatch]:
    """Build cue batches with the preceding batch's last cues as context."""

    if not min_batch_size <= batch_size <= max_batch_size:
        raise ValueError(f"batch_size must be between {min_batch_size} and {max_batch_size}")
    if context_size < 0:
        raise ValueError("context_size must be non-negative")
    batches: list[TranslationBatch] = []
    for start in range(0, len(segments), batch_size):
        end = min(start + batch_size, len(segments))
        items = tuple((index - start + 1, segments[index]) for index in range(start, end))
        context = tuple(segments[max(0, start - context_size) : start])
        batches.append(TranslationBatch(items=items, context=context))
    return batches


def numbered_mismatch(output: str, expected_ids: int | Iterable[int]) -> dict[str, int]:
    """Counts describing why `output` fails strict parsing (for logs: numbers only, never subtitle text).
    expected / got (numbered lines) / unnumbered / empty items / first_bad = 1-based position of the first
    numbered line whose number is not the expected one (0 = none)."""

    expected = tuple(range(1, expected_ids + 1)) if isinstance(expected_ids, int) else tuple(expected_ids)
    ids: list[int] = []
    unnumbered = empty = 0
    for line in str(output).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        match = _NUMBERED_LINE_RE.fullmatch(line)
        if not match:
            unnumbered += 1
            continue
        ids.append(int(match.group(1)))
        empty += not match.group(2).strip()
    first_bad = next((i + 1 for i, n in enumerate(ids) if i >= len(expected) or n != expected[i]), 0)
    if not first_bad and len(ids) < len(expected):
        first_bad = len(ids) + 1
    return {"expected": len(expected), "got": len(ids), "unnumbered": unnumbered, "empty": empty,
            "first_bad": first_bad}


def parse_numbered_translations(
    output: str,
    expected_ids: int | Iterable[int],
    *,
    strict: bool = True,
) -> list[str]:
    """Parse one numbered translation per line and reject any unnumbered commentary (or extract leniently)."""

    expected = tuple(range(1, expected_ids + 1)) if isinstance(expected_ids, int) else tuple(expected_ids)
    if not expected:
        return []

    clean_output = str(output).strip()
    if not clean_output:
        raise ValueError("translation output is empty")

    if not strict:
        stripped = clean_output
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                data = json.loads(stripped)
                items_list = []
                if isinstance(data, list):
                    items_list = data
                elif isinstance(data, dict):
                    for k in ("translations", "results", "items", "segments", "data"):
                        if isinstance(data.get(k), list):
                            items_list = data[k]
                            break
                if items_list:
                    json_by_id: dict[int, str] = {}
                    for item in items_list:
                        if isinstance(item, dict):
                            cid = item.get("id") or item.get("index") or item.get("no")
                            val = item.get("zh") or item.get("text") or item.get("translation") or item.get("content")
                            if cid is not None and val is not None and str(val).strip():
                                try:
                                    json_by_id[int(cid)] = str(val).strip()
                                except (ValueError, TypeError):
                                    pass
                    if all(exp in json_by_id for exp in expected):
                        return [json_by_id[exp] for exp in expected]
            except Exception:
                pass

    parsed: list[tuple[int, str]] = []
    for line in output.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        match = _NUMBERED_LINE_RE.fullmatch(line)
        if not match:
            if strict:
                raise ValueError("translation output contains unnumbered text")
            continue
        parsed.append((int(match.group(1)), match.group(2).strip()))

    if not strict:
        by_id: dict[int, str] = {}
        for num, text in parsed:
            if num not in by_id and text:
                by_id[num] = text
        if all(exp in by_id for exp in expected):
            return [by_id[exp] for exp in expected]
        missing = set(expected) - set(by_id.keys())
        raise ValueError(f"translation output missing expected IDs: {sorted(missing)}")

    actual_ids = tuple(number for number, _ in parsed)
    if actual_ids != expected:
        raise ValueError(f"translation IDs {actual_ids!r} do not match expected {expected!r}")
    texts = [text for _, text in parsed]
    if any(not text for text in texts):
        raise ValueError("translation output contains an empty item")
    return texts


@lru_cache(maxsize=1)
def _default_opencc():
    try:
        from opencc import OpenCC  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCC is required for s2twp conversion") from exc
    return OpenCC("s2twp")


def to_taiwan_traditional(text: str, converter: Callable[[str], str] | object | None = None) -> str:
    """Convert through OpenCC s2twp, with an injectable callable/object for tests."""

    if converter is None:
        converter = _default_opencc()
    convert = converter if callable(converter) else getattr(converter, "convert", None)
    if not callable(convert):
        raise TypeError("converter must be callable or expose convert(text)")
    return str(convert(text))


def translate_segments(
    segments: Sequence[Segment],
    translate_batch: Callable[[TranslationBatch], str],
    translate_one: Callable[[str], str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    opencc: Callable[[str], str] | object | None = None,
    on_progress=None,
    on_batch=None,
    propagate_errors=(),
    tiered_retry: bool = False,
    budget_multiplier: float = 2.0,
    refusal_threshold: Optional[float] = None,
) -> TranslationRun:
    """Translate in numbered batches with optional tiered ladder retry and refusal detection."""

    results: list[TranslationResult] = []
    mismatch_batches = 0
    fallback_sentences = 0
    untranslated = 0
    suspected_refusals = 0
    residue_retry_sentences = 0
    residue_unresolved_sentences = 0

    def _check_fatal(exc: Exception):
        if isinstance(exc, propagate_errors):
            raise exc
        from .providers.http import ProviderError
        if isinstance(exc, ProviderError) and (
            not exc.retryable or exc.kind in {"auth", "quota_daily", "network", "rate_limit"}
        ):
            raise exc

    for batch in make_translation_batches(
        segments,
        batch_size=batch_size,
        context_size=context_size,
        min_batch_size=1 if batch_size < 20 else 20,
        max_batch_size=max(40, batch_size),
    ):
        # Per-batch request budget
        batch_budget = [max(2, int(math.ceil(budget_multiplier * len(batch.items))))]

        def _try_batch_call(b: TranslationBatch) -> tuple[bool, list[tuple[Segment, str, bool]]]:
            nonlocal suspected_refusals, untranslated
            if batch_budget[0] <= 0:
                return False, []
            batch_budget[0] -= 1
            try:
                raw = translate_batch(b)
                if not raw or not raw.strip():
                    return False, []
                try:
                    parsed = parse_numbered_translations(raw, b.expected_ids, strict=True)
                except ValueError:
                    parsed = parse_numbered_translations(raw, b.expected_ids, strict=False)

                rows_out = []
                for (_, seg), text in zip(b.items, parsed):
                    clean_t = text.strip()
                    if "[skip]" in clean_t.casefold():
                        rows_out.append((seg, "", False, True, "onomatopoeia_or_hallucination"))
                    elif tiered_retry and is_refusal_text(text):
                        suspected_refusals += 1
                        untranslated += 1
                        rows_out.append((seg, f"[原文] {seg.text}", True, False, None))
                    else:
                        rows_out.append((seg, text, False, False, None))
                return True, rows_out
            except Exception as exc:
                _check_fatal(exc)
                from .providers.http import ProviderError
                if isinstance(exc, ProviderError) and exc.kind == "content_blocked":
                    suspected_refusals += len(b.items)
                    untranslated += len(b.items)
                    return True, [(seg, f"[原文] {seg.text}", True, False, None) for _, seg in b.items]
                return False, []

        def _solve_sub_items(
            sub_items: Sequence[tuple[int, Segment]],
            context: Sequence[Segment],
            is_top: bool = False,
        ) -> list[tuple[Segment, str, bool, bool, Optional[str]]]:
            nonlocal mismatch_batches, fallback_sentences, untranslated, suspected_refusals

            sub_b = TranslationBatch(
                items=tuple((idx + 1, seg) for idx, (_, seg) in enumerate(sub_items)),
                context=context,
            )

            # Level 0: Try batch
            ok, batch_rows = _try_batch_call(sub_b)
            if ok:
                return batch_rows

            mismatch_batches += 1

            if tiered_retry:
                # Level 1: Retry same batch 1 time if top level
                if is_top and batch_budget[0] > 0:
                    ok, batch_rows = _try_batch_call(sub_b)
                    if ok:
                        return batch_rows

                # Level 2: Bisect if more than 2 items
                if len(sub_items) > 2 and batch_budget[0] > 0:
                    mid = len(sub_items) // 2
                    left_items = sub_items[:mid]
                    right_items = sub_items[mid:]
                    left_rows = _solve_sub_items(left_items, context, is_top=False)
                    right_context = tuple([s for _, s in left_items[-5:]])
                    right_rows = _solve_sub_items(right_items, right_context, is_top=False)
                    return left_rows + right_rows

            # Level 3: Single cue fallback
            single_rows = []
            for _, seg in sub_items:
                fallback_sentences += 1
                if batch_budget[0] <= 0:
                    untranslated += 1
                    single_rows.append((seg, f"[原文] {seg.text}", True, False, None))
                    continue
                batch_budget[0] -= 1
                try:
                    one_text = str(translate_one(seg.text)).strip()
                    if not one_text:
                        raise ValueError("empty single-cue translation")
                    if "[skip]" in one_text.casefold():
                        single_rows.append((seg, "", False, True, "onomatopoeia_or_hallucination"))
                    elif is_refusal_text(one_text):
                        suspected_refusals += 1
                        untranslated += 1
                        single_rows.append((seg, f"[原文] {seg.text}", True, False, None))
                    else:
                        single_rows.append((seg, one_text, False, False, None))
                except Exception as exc:
                    _check_fatal(exc)
                    from .providers.http import ProviderError
                    if isinstance(exc, ProviderError) and exc.kind == "content_blocked":
                        suspected_refusals += 1
                    untranslated += 1
                    single_rows.append((seg, f"[原文] {seg.text}", True, False, None))
            return single_rows

        rows = _solve_sub_items(batch.items, batch.context, is_top=True)

        for row_item in rows:
            if len(row_item) == 5:
                source, text, preserved, is_filtered, filter_reason = row_item
            else:
                source, text, preserved = row_item
                is_filtered, filter_reason = False, None

            if is_filtered:
                results.append(
                    TranslationResult(
                        source=source,
                        text="",
                        preserved_original=False,
                        is_filtered=True,
                        filter_reason=filter_reason,
                    )
                )
                continue

            if not preserved and has_japanese_residue(text):
                residue_retry_sentences += 1
                fallback_sentences += 1
                if batch_budget[0] > 0:
                    batch_budget[0] -= 1
                    try:
                        retry_text = str(translate_one(source.text)).strip()
                        if not retry_text or has_japanese_residue(retry_text):
                            raise ValueError("Japanese text remains after single-cue retry")
                        if "[skip]" in retry_text.casefold():
                            results.append(
                                TranslationResult(
                                    source=source,
                                    text="",
                                    preserved_original=False,
                                    is_filtered=True,
                                    filter_reason="onomatopoeia_or_hallucination",
                                )
                            )
                            continue
                        text = retry_text
                    except Exception as exc:
                        _check_fatal(exc)
                        from .providers.http import ProviderError
                        if isinstance(exc, ProviderError) and exc.kind == "content_blocked":
                            suspected_refusals += 1
                        text = f"[原文] {source.text}"
                        preserved = True
                        untranslated += 1
                        residue_unresolved_sentences += 1
                else:
                    text = f"[原文] {source.text}"
                    preserved = True
                    untranslated += 1
                    residue_unresolved_sentences += 1
            if not preserved:
                text = to_taiwan_traditional(text, converter=opencc)
                folded = text.casefold()
                if any(marker.casefold() in folded for marker in REFUSAL_MARKERS):
                    suspected_refusals += 1
            results.append(
                TranslationResult(
                    source=source,
                    text=text,
                    preserved_original=preserved,
                    is_filtered=False,
                    filter_reason=None,
                )
            )

        # Publish only after fallback, residue retries and traditional conversion.
        # Keep consumer exceptions outside model retry handling.
        if on_batch:
            on_batch(tuple(results))
        if on_progress:
            on_progress(len(results), len(segments))

    total_segments = len(segments)
    refusal_ratio = suspected_refusals / max(1, total_segments)
    if refusal_threshold is not None and total_segments > 0 and refusal_ratio > refusal_threshold:
        from .providers.http import ProviderError
        raise ProviderError(
            "content_blocked",
            f"Translation aborted: {suspected_refusals}/{total_segments} ({refusal_ratio * 100:.1f}%) "
            f"sentences were refused by provider safety filter (threshold is {refusal_threshold * 100:.0f}%).",
            retryable=False,
            scope="none",
        )

    return TranslationRun(
        results=tuple(results),
        metrics={
            "mismatch_batches": mismatch_batches,
            "fallback_sentences": fallback_sentences,
            "untranslated": untranslated,
            "suspected_refusals": suspected_refusals,
            "residue_retry_sentences": residue_retry_sentences,
            "residue_unresolved_sentences": residue_unresolved_sentences,
        },
    )


PROTECTED_TERMS: tuple[str, ...] = (
    "ボン、キュッ、ボン",
    "ボンキュッボン",
    "ボン・キュッ・ボン",
    "プルンプルン",
    "ぷるんぷるん",
    "チョメチョメ",
    "ちょめちょめ",
    "人たらし",
    "食われんように",
)

ONOMATOPOEIA_TOKENS: tuple[str, ...] = (
    "ちゅぷっ", "ちゅぱっ", "ちゅっ", "ちゅぷ", "ちゅぱ", "ぢゅぱっ", "ぢゅぱ", "チュッ", "チュ",
    "じゅるっ", "じゅる", "んっ",
    "パオ", "ぱお", "ふふっ", "ふふ", "おー", "ああ", "ぁむ", "うぇーい", "うぇい", "いえーい", "イエーイ",
    "ハハハハハ", "ははは", "がははは", "ハァハァ", "はぁはぁ", "くすくす",
    "啾啪", "啾噗", "啾", "啜飲", "擠擠", "擠", "嘖", "啪哦", "哼哼", "呵呵", "啊", "喔",
)

ASR_HALLUCINATIONS: tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
    "チャンネル登録",
)


def is_pure_onomatopoeia(text: str) -> bool:
    """Detect if text consists purely of onomatopoeia or sound tokens without dialogue.

    Strictly protects character body descriptions (e.g. ボン、キュッ、ボン, プルンプルン)
    and comedic/intimate slang (e.g. チョメチョメ, 人たらし).
    """
    if not text:
        return False
    # Protected comedic/character terms must never be filtered
    if any(p in text for p in PROTECTED_TERMS):
        return False
    t_clean = _PUNCTUATION_RE.sub("", text)
    if not t_clean:
        return False
    rem = t_clean
    for tok in sorted(ONOMATOPOEIA_TOKENS, key=len, reverse=True):
        rem = rem.replace(tok, "")
    return len(rem.strip()) == 0


def is_skip_text(text: str) -> bool:
    """Detect whether a line should be skipped (pure onomatopoeia or hallucination)."""
    if not text or "[skip]" in text.casefold():
        return True
    if any(h in text for h in ASR_HALLUCINATIONS):
        return True
    return is_pure_onomatopoeia(text)


def assign_segments_to_intervals(
    segments: Sequence[Union[Segment, dict[str, Any]]],
    intervals: Sequence[Union[tuple[float, float], list[float], dict[str, Any], Segment]],
    *,
    snap_threshold: float = 0.8,
) -> list[list[Any]]:
    """Assign segments to target intervals based on maximum overlap duration with distance snapping.

    Rules:
    1. Intersection length:
       ov = max(0.0, min(interval_end, seg_end) - max(interval_start, seg_start))
    2. Center-snapped maximum overlap:
       Assign segment to the interval that maximizes `ov` (`best_ov > 0.0`).
       Ties are broken by distance between interval center and segment center.
    3. Nearest snapping:
       If `best_ov == 0.0`, compute minimum distance to each interval:
       dist = max(0.0, interval_start - seg_end, seg_start - interval_end)
       Only if min(dist) < snap_threshold (default 0.8s), assign to that nearest interval.
       Otherwise, the segment remains unassigned.
    4. Segments assigned to each interval are preserved in strict chronological order.
    """
    if not intervals:
        return []

    parsed_intervals: list[tuple[float, float, float]] = []
    for interval in intervals:
        if isinstance(interval, (tuple, list)):
            bs, be = float(interval[0]), float(interval[1])
        elif isinstance(interval, dict):
            bs, be = float(interval["start"]), float(interval["end"])
        elif isinstance(interval, Segment):
            bs, be = float(interval.start), float(interval.end)
        else:
            bs, be = float(getattr(interval, "start")), float(getattr(interval, "end"))
        bs, be = min(bs, be), max(bs, be)
        mid = (bs + be) / 2.0
        parsed_intervals.append((bs, be, mid))

    assigned: list[list[Any]] = [[] for _ in range(len(parsed_intervals))]

    def _seg_time(s: Any) -> tuple[float, float]:
        if isinstance(s, dict):
            s_val, e_val = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        else:
            s_val, e_val = float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0))
        return min(s_val, e_val), max(s_val, e_val)

    sorted_segments = sorted(segments, key=_seg_time)

    for seg in sorted_segments:
        ws, we = _seg_time(seg)
        w_mid = (ws + we) / 2.0

        best_idx = -1
        best_ov = 0.0
        min_center_dist = float("inf")

        for idx, (bs, be, b_mid) in enumerate(parsed_intervals):
            ov = max(0.0, min(be, we) - max(bs, ws))
            if ov > best_ov:
                best_ov = ov
                best_idx = idx
                min_center_dist = abs(b_mid - w_mid)
            elif ov > 0.0 and abs(ov - best_ov) < 1e-6:
                center_dist = abs(b_mid - w_mid)
                if center_dist < min_center_dist:
                    best_idx = idx
                    min_center_dist = center_dist

        if best_ov <= 0.0:
            closest_dist = float("inf")
            for idx, (bs, be, _) in enumerate(parsed_intervals):
                dist = max(0.0, bs - we, ws - be)
                if dist < closest_dist and dist < snap_threshold:
                    closest_dist = dist
                    best_idx = idx

        if best_idx >= 0:
            assigned[best_idx].append(seg)

    return assigned


def join_assigned_segments(
    assigned_segs: Sequence[Union[Segment, dict[str, Any]]],
    sep: str = " ",
) -> str:
    """Join text of assigned segments cleanly into a single string."""
    texts = []
    for s in assigned_segs:
        if isinstance(s, dict):
            t = str(s.get("text") or s.get("text_ja") or s.get("ja") or "").strip()
        else:
            t = str(getattr(s, "text", "")).strip()
        if t:
            texts.append(t)
    return sep.join(texts)

