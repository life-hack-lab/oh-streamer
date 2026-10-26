"""Automatic Clause & Turn Decoupling for Subtitle Cues.

Joint acoustic-linguistic boundary detection with strict side-effect prevention.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from .core import Segment


@dataclass(frozen=True)
class DecoupleSegment:
    start: float
    end: float
    text: str
    parent_id: Optional[str] = None
    sub_index: int = 1


@dataclass(frozen=True)
class BoundaryCandidate:
    char_index: int          # Text split position (character index)
    tier: int                # 1: Strong termination, 2: Clause/turning, 3: Weak connection
    score: float             # Linguistic score (0.0 ~ 1.0)
    matched_pattern: str


class AutoClauseDecoupler:
    """Automatic clause and speaker turn decoupler (Acoustic + Linguistic Joint Pipeline)."""

    TIER1_PATTERNS = [
        r"[。！？!?…]+[」』）”\"]*",
        r"(?:だ|です|ます|でした|ました)[！!]+[」』）”\"]*",
        r"(?:だ|よ|ね|ぞ|ぜ|さ|わ|な)[！!]+[」』）”\"]*",
    ]
    TIER2_PATTERNS = [
        r"[、，,]",
        r"(?:てほしい|てください|てくれ|なさい)(?:[よねわ]|(?=[、\s]|$))",
        r"(?:けど|けれど|けれども|が|から|ので|のに)(?=[、\s]|$)",
        r"(?:たら|なら|ば)(?=[、\s]|$)",
    ]
    TIER3_PATTERNS = [
        r"(?:は|も)(?=[、\s]|$)",
    ]

    def __init__(
        self,
        sample_rate: int = 16000,
        min_cue_duration: float = 0.8,
        min_fast_pace_duration: float = 0.65,
        max_cue_duration: float = 6.0,
        max_cps: float = 15.0,
        min_chars: int = 3,
        onset_margin_sec: float = 0.03,
        decay_margin_sec: float = 0.03,
        search_radius_sec: float = 0.6,
    ):
        self.sample_rate = sample_rate
        self.min_cue_duration = min_cue_duration
        self.min_fast_pace_duration = min_fast_pace_duration
        self.max_cue_duration = max_cue_duration
        self.max_cps = max_cps
        self.min_chars = min_chars
        self.onset_margin_sec = onset_margin_sec
        self.decay_margin_sec = decay_margin_sec
        self.search_radius_sec = search_radius_sec

    def decouple(
        self,
        segment: Union[DecoupleSegment, Segment],
        audio: Optional[np.ndarray],
        speech_probs: Optional[np.ndarray] = None,
        *,
        max_depth: int = 3,
    ) -> List[Union[DecoupleSegment, Segment]]:
        """Decouple a single segment into sub-clauses if linguistic and acoustic dips coincide."""
        is_core_segment = isinstance(segment, Segment)
        if is_core_segment:
            d_seg = DecoupleSegment(start=segment.start, end=segment.end, text=segment.text)
        else:
            d_seg = segment

        result_dsegs = self._decouple_internal(d_seg, audio, speech_probs, depth=0, max_depth=max_depth)

        if is_core_segment:
            return [
                replace(
                    segment,
                    start=ds.start,
                    end=ds.end,
                    text=ds.text,
                )
                for ds in result_dsegs
            ]
        return result_dsegs

    def _decouple_internal(
        self,
        segment: DecoupleSegment,
        audio: Optional[np.ndarray],
        speech_probs: Optional[np.ndarray],
        depth: int,
        max_depth: int,
    ) -> List[DecoupleSegment]:
        if depth >= max_depth:
            return [segment]

        duration = segment.end - segment.start
        text = segment.text.strip()

        # Hard minimum checks: must have room for at least 2 minimum cues and enough characters
        min_req_duration = min(self.min_cue_duration, self.min_fast_pace_duration) * 2
        if duration < min_req_duration or len(text) < self.min_chars * 2:
            return [segment]

        if audio is None or len(audio) == 0:
            return [segment]

        candidates = self._find_linguistic_candidates(text)
        if not candidates:
            return [segment]

        rms, times = self._compute_rms_envelope(audio, segment.start, segment.end)
        if len(rms) == 0:
            return [segment]

        valid_split_points = []
        for cand in candidates:
            char_ratio = cand.char_index / len(text)
            est_time = segment.start + duration * char_ratio

            acoustic_point = self._find_best_acoustic_dip(
                rms=rms,
                times=times,
                speech_probs=speech_probs,
                target_time=est_time,
                search_radius=self.search_radius_sec,
                tier=cand.tier,
            )

            if acoustic_point is not None:
                split_time, dip_duration = acoustic_point
                if self._validate_split_constraints(
                    segment, split_time, cand.char_index, text, tier=cand.tier
                ):
                    dist = abs(split_time - est_time)
                    valid_split_points.append((cand.char_index, split_time, cand.tier, dist))

        if not valid_split_points:
            return [segment]

        # Prioritize Tier (Tier 1 > Tier 2 > Tier 3) then minimal distance to estimated time
        valid_split_points.sort(key=lambda x: (x[2], x[3]))
        splits = self._execute_split(segment, valid_split_points[0], text)

        # Recursively attempt further decoupling on each split part
        results = []
        for part in splits:
            sub_results = self._decouple_internal(
                part, audio, speech_probs, depth=depth + 1, max_depth=max_depth
            )
            results.extend(sub_results)

        # Ensure monotonic order and sub_index continuity
        out = []
        for idx, r in enumerate(results):
            out.append(
                replace(
                    r,
                    sub_index=idx + 1,
                    parent_id=segment.parent_id or f"{segment.start:.2f}_{segment.end:.2f}",
                )
            )
        return out

    def _is_valid_word_boundary(self, text: str, char_idx: int) -> bool:
        """Enforce morphological word boundaries and protect compound words / sokuon."""
        if char_idx <= 0 or char_idx >= len(text):
            return False

        c_prev = text[char_idx - 1]
        c_next = text[char_idx]

        # Sokuon (促音) strict protection
        if c_prev in "っッ" or c_next in "っッ":
            return False

        # Closing quotation or bracket protection (cannot cut between sentence end and closing quote)
        if c_next in "」』）)”\"'】》〉":
            return False

        # Kanji compound atomicity (cannot cut between two Kanji characters)
        if re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", c_prev) and re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", c_next):
            return False

        # Katakana loanword atomicity (cannot cut inside contiguous Katakana characters)
        if re.match(r"[\u30a0-\u30ffー]", c_prev) and re.match(r"[\u30a0-\u30ffー]", c_next):
            return False

        # Alphanumeric atomicity (cannot cut inside letters or numbers)
        if c_prev.isalnum() and c_next.isalnum() and not (
            re.match(r"[\u3040-\u309f]", c_prev) or re.match(r"[\u3040-\u309f]", c_next)
        ):
            return False

        return True

    def _find_linguistic_candidates(self, text: str) -> List[BoundaryCandidate]:
        candidates: List[BoundaryCandidate] = []
        for pat in self.TIER1_PATTERNS:
            for m in re.finditer(pat, text):
                candidates.append(BoundaryCandidate(m.end(), 1, 1.0, m.group()))
        for pat in self.TIER2_PATTERNS:
            for m in re.finditer(pat, text):
                candidates.append(BoundaryCandidate(m.end(), 2, 0.7, m.group()))
        for pat in self.TIER3_PATTERNS:
            for m in re.finditer(pat, text):
                candidates.append(BoundaryCandidate(m.end(), 3, 0.4, m.group()))

        candidates.sort(key=lambda c: (c.char_index, c.tier))
        unique: List[BoundaryCandidate] = []
        for c in candidates:
            if self._is_valid_word_boundary(text, c.char_index):
                if not unique or unique[-1].char_index != c.char_index:
                    unique.append(c)
                elif unique and unique[-1].char_index == c.char_index and c.tier < unique[-1].tier:
                    unique[-1] = c
        return unique

    def _compute_rms_envelope(
        self, audio: np.ndarray, start: float, end: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else -1)

        frame_len = int(self.sample_rate * 0.02)
        hop_len = int(self.sample_rate * 0.01)

        start_sample = max(0, int(round(start * self.sample_rate)))
        end_sample = min(len(audio), int(round(end * self.sample_rate)))
        clip = audio[start_sample:end_sample]

        if len(clip) < frame_len:
            return np.array([]), np.array([])

        num_frames = (len(clip) - frame_len) // hop_len + 1
        clip = np.ascontiguousarray(clip)
        frames = np.lib.stride_tricks.as_strided(
            clip,
            shape=(num_frames, frame_len),
            strides=(clip.strides[0] * hop_len, clip.strides[0]),
        )
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        window = min(5, len(rms))
        if window > 1:
            rms_smooth = np.convolve(rms, np.ones(window) / window, mode="same")
        else:
            rms_smooth = rms
        times = start + (np.arange(num_frames) * hop_len + frame_len / 2) / self.sample_rate
        return rms_smooth, times

    def _find_best_acoustic_dip(
        self,
        rms: np.ndarray,
        times: np.ndarray,
        speech_probs: Optional[np.ndarray],
        target_time: float,
        search_radius: float,
        tier: int,
    ) -> Optional[Tuple[float, float]]:
        if len(rms) == 0:
            return None

        mask = (times >= target_time - search_radius) & (times <= target_time + search_radius)
        if not np.any(mask):
            return None

        window_rms = rms[mask]
        window_times = times[mask]
        median_rms = np.median(rms) + 1e-9

        threshold_map = {
            1: {"min_pause_s": 0.16, "dip_ratio": 0.35, "vad_thresh": 0.40},
            2: {"min_pause_s": 0.24, "dip_ratio": 0.20, "vad_thresh": 0.25},
            3: {"min_pause_s": 0.32, "dip_ratio": 0.10, "vad_thresh": 0.15},
        }
        cfg = threshold_map.get(tier, threshold_map[2])

        is_dip = (window_rms / median_rms) <= cfg["dip_ratio"]
        if speech_probs is not None and len(speech_probs) == len(rms):
            window_probs = speech_probs[mask]
            is_dip = is_dip | (window_probs <= cfg["vad_thresh"])

        runs = []
        first = None
        for i, quiet in enumerate(list(is_dip) + [False]):
            if quiet and first is None:
                first = i
            elif not quiet and first is not None:
                runs.append((first, i))
                first = None

        valid_dips = []
        for start_idx, end_idx in runs:
            dip_start = window_times[start_idx]
            dip_end = window_times[min(end_idx, len(window_times) - 1)]
            pause_duration = dip_end - dip_start
            if pause_duration >= cfg["min_pause_s"]:
                center = (dip_start + dip_end) / 2.0
                dist = abs(center - target_time)
                valid_dips.append((dist, center, pause_duration))

        if not valid_dips:
            return None

        valid_dips.sort(key=lambda x: x[0])
        _, best_center, best_duration = valid_dips[0]
        return best_center, best_duration

    def _validate_split_constraints(
        self,
        parent: DecoupleSegment,
        split_time: float,
        char_idx: int,
        text: str,
        tier: int,
    ) -> bool:
        dur1 = split_time - parent.start
        dur2 = parent.end - split_time

        # Fast-paced shouts (Tier 1) allow hard lower bound 0.65s; otherwise min_cue_duration (0.8s)
        min_allowed = self.min_fast_pace_duration if tier == 1 else self.min_cue_duration
        if dur1 < min_allowed or dur2 < min_allowed:
            return False

        text1 = text[:char_idx].strip()
        text2 = text[char_idx:].strip()
        if len(text1) < self.min_chars or len(text2) < self.min_chars:
            return False

        # Reading rate (CPS) clamping
        if dur1 > 0 and len(text1) / dur1 > self.max_cps:
            return False
        if dur2 > 0 and len(text2) / dur2 > self.max_cps:
            return False

        if not self._is_valid_word_boundary(text, char_idx):
            return False

        return True

    def _execute_split(
        self,
        parent: DecoupleSegment,
        best_split: Tuple[int, float, int, float],
        text: str,
    ) -> List[DecoupleSegment]:
        char_idx, split_time, _, _ = best_split

        part1_end = round(split_time, 3)
        part2_start = round(split_time + self.onset_margin_sec, 3)

        # Ensure no inverted intervals
        if part1_end <= parent.start:
            part1_end = round(parent.start + 0.1, 3)
        if part2_start >= parent.end:
            part2_start = round(part1_end, 3)

        part1 = replace(
            parent,
            start=parent.start,
            end=part1_end,
            text=text[:char_idx].strip(),
            sub_index=1,
        )
        part2 = replace(
            parent,
            start=part2_start,
            end=parent.end,
            text=text[char_idx:].strip(),
            sub_index=2,
        )
        return [part1, part2]


def decouple_segments(
    segments: Sequence[Union[Segment, DecoupleSegment]],
    audio: Optional[np.ndarray],
    *,
    sample_rate: int = 16000,
    min_cue_duration: float = 0.8,
    min_fast_pace_duration: float = 0.65,
    max_cps: float = 15.0,
    min_chars: int = 3,
    speech_probs: Optional[np.ndarray] = None,
) -> List[Union[Segment, DecoupleSegment]]:
    """Convenience function to decouple a sequence of segments against an audio track."""
    if not segments:
        return []
    if audio is None or len(audio) == 0:
        return list(segments)

    decoupler = AutoClauseDecoupler(
        sample_rate=sample_rate,
        min_cue_duration=min_cue_duration,
        min_fast_pace_duration=min_fast_pace_duration,
        max_cps=max_cps,
        min_chars=min_chars,
    )

    out: List[Union[Segment, DecoupleSegment]] = []
    for s in segments:
        sub_segs = decoupler.decouple(s, audio, speech_probs=speech_probs)
        out.extend(sub_segs)
    return out
