"""Cue timing from word timestamps (cloud ASR).

Whisper / Groq segment times are contiguous: a segment starts where the previous one ended, so silence or music
before speech is absorbed into the cue. Word times are accurate, except that the first word after a pause is often
stretched back over the silence. One rule is shared by every consumer:

    start = max(first.start, first.end - max(0.25, 0.36 * len(first text)))
    end   = last.end + 0.25

(the caller clips ``end`` to the next cue's start). Segments without usable words keep their own times.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Optional, Sequence

FIRST_WORD_MIN_SECONDS = 0.25
FIRST_WORD_SECONDS_PER_CHAR = 0.36
TAIL_PAD_SECONDS = 0.25
MIN_CUE_SECONDS = 0.3
PAUSE_SPLIT_SECONDS = 0.8      # a pause this long inside one segment is usually a change of speaker
PAUSE_MIN_CHARS = 3            # ...but a side shorter than this (and not a short reply) stays attached
PAUSE_MIN_SECONDS = 0.5
_MIN_MATCH = 0.5


def clean_words(words: Any) -> list[tuple[str, float, float]]:
    """(text, start, end) of every well-formed, non-blank word."""
    out: list[tuple[str, float, float]] = []
    if not isinstance(words, (list, tuple)):
        return out
    for w in words:
        try:
            text = ''.join(str(w['word']).split())
            start, end = float(w['start']), float(w['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if text and end >= start:
            out.append((text, start, end))
    return out


def words_span(words: Sequence[tuple[str, float, float]]) -> Optional[tuple[float, float]]:
    """Cue (start, end) of a run of cleaned words, or None when empty."""
    if not words:
        return None
    text, w_start, w_end = words[0]
    start = max(w_start, w_end - max(FIRST_WORD_MIN_SECONDS, FIRST_WORD_SECONDS_PER_CHAR * len(text)))
    end = max(words[-1][2], start) + TAIL_PAD_SECONDS
    return round(start, 4), round(end, 4)


def align_words(text: str, words: Sequence[tuple[str, float, float]]) -> tuple[list[Optional[int]], float]:
    """Map every whitespace-free character of ``text`` to the index of the word that spells it (None when no
    word does) and return the matched fraction. Whisper may drop repeated text from the word list, so this aligns
    instead of comparing."""
    seg = ''.join(text.split())
    joined = ''.join(w[0] for w in words)
    word_of: list[int] = []
    for i, w in enumerate(words):
        word_of.extend([i] * len(w[0]))
    if seg == joined:
        return list(word_of), 1.0
    import difflib
    matched: list[Optional[int]] = [None] * len(seg)
    for x, y, size in difflib.SequenceMatcher(None, seg, joined, autojunk=False).get_matching_blocks():
        for k in range(size):
            matched[x + k] = word_of[y + k]
    ratio = sum(m is not None for m in matched) / len(seg) if seg else 0.0
    return matched, ratio


def retime_from_words(text: str, words: Any) -> Optional[tuple[float, float]]:
    """(start, end) of a whole segment from its words, or None when the words do not spell ``text``."""
    cleaned = clean_words(words)
    if not cleaned:
        return None
    matched, ratio = align_words(text, cleaned)
    if ratio < _MIN_MATCH:
        return None
    hits = [m for m in matched if m is not None]
    span = words_span(cleaned[min(hits):max(hits) + 1])
    if span is None or span[1] - span[0] <= 0:
        return None
    return span


def _split_at_separators(text: str, target: int) -> list[str]:
    """Split a too-long sentence at spaces / commas (the separators Whisper puts between phrases)."""
    text = text.strip()
    if len(text) <= target:
        return [text]
    points = [m.end() for m in re.finditer(r'[ 　]+|[、，,]', text)]
    points = [i for i in points if 0 < i < len(text) and text[:i].strip() and text[i:].strip()]
    if not points:
        return [text]
    cut = min(points, key=lambda i: abs(i - min(target, len(text) / 2)))
    return _split_at_separators(text[:cut], target) + _split_at_separators(text[cut:], target)


def _pause_chunks(text: str, words: Sequence[tuple[str, float, float]], matched: list[Optional[int]]) -> list[str]:
    """``text`` cut where two consecutive words are PAUSE_SPLIT_SECONDS apart, as long as both sides stay
    readable: at least PAUSE_MIN_CHARS characters and PAUSE_MIN_SECONDS of words, or a short reply (え, はい...)."""
    from .sentences import SHORT_RESPONSES, compact

    def readable(lo: int, hi: int) -> bool:          # whitespace-free char range [lo, hi)
        piece = ''.join(text.split())[lo:hi]
        if compact(piece) in SHORT_RESPONSES:
            return True
        got = [m for m in matched[lo:hi] if m is not None]
        return (len(compact(piece)) >= PAUSE_MIN_CHARS and bool(got)
                and words[max(got)][2] - words[min(got)][1] >= PAUSE_MIN_SECONDS)

    first_char: dict[int, int] = {}                   # word index -> its first char position
    for pos, m in enumerate(matched):
        if m is not None:
            first_char.setdefault(m, pos)
    candidates = [first_char[k] for k in range(1, len(words))
                  if k in first_char and words[k][1] - words[k - 1][2] >= PAUSE_SPLIT_SECONDS]
    cuts, last = [], 0
    for pos in candidates:
        if 0 < pos < len(matched) and readable(last, pos) and readable(pos, len(matched)):
            cuts.append(pos)
            last = pos
    if not cuts:
        return [text]
    chunks, start, seen = [], 0, 0                    # map whitespace-free positions back onto ``text``
    for i, c in enumerate(text):
        if c.isspace():
            continue
        if cuts and seen == cuts[0]:
            chunks.append(text[start:i])
            start = i
            cuts.pop(0)
        seen += 1
    chunks.append(text[start:])
    return [c for c in chunks if c.strip()]


def word_sentence_cues(segment, words: Any, max_chars: int = 36, maximum: float = 7.):
    """Sentence cues of one segment timed by their own words, or None when the words cannot be used (the caller
    then falls back to the character-proportional ``sentences.sentence_cues``)."""
    from .sentences import _split_long, split_sentences
    cleaned = clean_words(words)
    if not cleaned:
        return None
    matched, ratio = align_words(segment.text, cleaned)
    if ratio < _MIN_MATCH:
        return None
    hits = [m for m in matched if m is not None]
    whole = words_span(cleaned[min(hits):max(hits) + 1])
    if whole is None or whole[1] <= whole[0]:
        return None
    parts = [q for chunk in _pause_chunks(segment.text, cleaned, matched) for q in split_sentences(chunk)]
    if not parts:
        return []
    total = sum(len(''.join(p.split())) for p in parts)
    if total != len(matched):
        return None
    duration = whole[1] - whole[0]
    target = max(1, min(max_chars, int(total * maximum / duration)))
    pieces = [q.strip() for p in parts for s in _split_at_separators(p, target)
              for q in _split_long(s, target) if q.strip()]
    spans: list[list[float]] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        n = len(''.join(piece.split()))
        got = [m for m in matched[cursor:cursor + n] if m is not None]
        cursor += n
        span = words_span(cleaned[min(got):max(got) + 1]) if got else None
        if got:
            ranges.append((min(got), max(got)))
        if span is None:
            return None
        spans.append(list(span))
    for i in range(len(spans) - 1):
        nxt = spans[i + 1][0]
        if ranges[i + 1][0] <= ranges[i][1]:          # a word shared by two pieces: the later piece follows
            spans[i + 1][0] = nxt = spans[i][1]
            spans[i + 1][1] = max(spans[i + 1][1], nxt + MIN_CUE_SECONDS)
        elif spans[i][1] > nxt:
            spans[i][1] = nxt
    result = []
    for piece, (start, end) in zip(pieces, spans):
        if end <= start:
            end = start + MIN_CUE_SECONDS
        flags = tuple(segment.flags)
        if end - start > maximum + 1e-6:
            flags += ('indivisible_long_sentence',)
        result.append(replace(segment, start=start, end=end, text=piece, flags=flags))
    return result
