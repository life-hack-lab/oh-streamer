"""Utterance-local sentence cues. No model timestamps or translated-text slicing."""
from dataclasses import replace
import math
import re
import unicodedata
from .core import Segment, format_timestamp


SHORT_RESPONSES = frozenset(
    (
        'はい', 'うん', 'ううん', 'ええ', 'え', 'あ', 'ああ', 'あっ',
        'ん', 'んっ', 'はあ', 'はぁ', 'おう', 'おい', 'なあ', 'ほら',
        'そう', 'そうです', 'いいえ', 'ありがとう', 'すみません', 'ごめん',
        '部長', '社長', '店長',
    )
)


def compact(text):
    return ''.join(
        c
        for c in unicodedata.normalize('NFKC', text)
        if unicodedata.category(c)[0] not in 'PZS'
    )


def filter_short(segments, threshold=1.0, policy='whitelist'):
    """Classify by ORIGINAL VAD duration, before any sentence allocation."""
    if (
        not math.isfinite(threshold)
        or threshold < 0
        or policy not in ('keep', 'whitelist', 'drop', 'replies')
    ):
        raise ValueError('invalid short utterance policy')
    if policy == 'replies':  # opt-in rule, see short_replies.py
        from .short_replies import filter_short_replies

        return filter_short_replies(segments, threshold)
    kept, rejected = [], []
    for s in segments:
        short = round(s.end - s.start, 6) < threshold
        reject = short and (
            policy == 'drop'
            or (policy == 'whitelist' and compact(s.text) not in SHORT_RESPONSES)
        )
        (rejected if reject else kept).append(s)
    return kept, rejected


def split_sentences(text):
    # Leading ellipses belong to the following sentence; consecutive punctuation stays together.
    pieces = re.findall(
        r'.+?(?:[。？！?!…]+[」』）”\"]*|$)', text.replace('\n', ' ')
    )
    output, prefix = [], ''
    for part in pieces:
        if compact(part):
            output.append(prefix + part.strip())
            prefix = ''
        elif output:
            output[-1] += part.strip()
        else:
            prefix += part.strip()
    return output


def _split_long(text, target):
    if len(text) <= target:
        return [text]
    # A comma is preferred; Japanese particles are a fallback, never cut a Han word.
    points = [
        m.end()
        for m in re.finditer(
            r'[、，,]|[はがをにでとのもへ](?![ぁぃぅぇぉゃゅょっ])', text
        )
    ]
    points = [i for i in points if 0 < i < len(text)]
    if not points:
        return [text]
    cut = min(points, key=lambda i: abs(i - min(target, len(text) / 2)))
    return _split_long(text[:cut], target) + _split_long(text[cut:], target)


def allocate_durations(weights, duration, minimum=.7, maximum=7.):
    """Bounded proportional allocation; preserve the parent interval exactly.

    Impossible bounds are explicit errors, resolved by merging sentence fragments
    or recording an indivisible long sentence in sentence_cues.
    """
    n = len(weights)
    if (
        not n
        or duration <= 0
        or minimum <= 0
        or maximum < minimum
        or any(w <= 0 for w in weights)
    ):
        raise ValueError('invalid allocation')
    if duration < n * minimum - 1e-8 or duration > n * maximum + 1e-8:
        raise ValueError('infeasible duration bounds')
    lo, hi = 0., duration / min(weights)
    for _ in range(80):
        scale = (lo + hi) / 2
        if sum(min(maximum, max(minimum, w * scale)) for w in weights) < duration:
            lo = scale
        else:
            hi = scale
    values = [min(maximum, max(minimum, w * (lo + hi) / 2)) for w in weights]
    values[-1] += duration - sum(values)
    return values


def sentence_cues(segment, max_chars=36, minimum=.7, maximum=7.):
    duration = segment.end - segment.start
    if duration <= 0:
        raise ValueError('invalid utterance interval')
    parts = split_sentences(segment.text)
    if not parts:
        return []
    total = sum(map(len, parts))
    target = max(1, min(max_chars, int(total * maximum / duration)))
    parts = [piece for part in parts for piece in _split_long(part, target)]
    # Too many tiny sentences cannot all fit .7 s; combine the shortest with a neighbour.
    while len(parts) > 1 and len(parts) * minimum > duration + 1e-8:
        i = min(range(len(parts)), key=lambda j: len(parts[j]))
        j = i - 1 if i else 0
        parts[j:j + 2] = [''.join(parts[j:j + 2])]
    weights = [max(1, len(compact(p))) for p in parts]
    lower = min(minimum, duration / len(parts))
    upper = max(maximum, duration / len(parts))
    lengths = allocate_durations(weights, duration, lower, upper)
    result, cursor = [], segment.start
    for i, (part, length) in enumerate(zip(parts, lengths)):
        end = segment.end if i == len(parts) - 1 else cursor + length
        flags = segment.flags
        if end - cursor > maximum + 1e-6:
            flags += ('indivisible_long_sentence',)
        if end - cursor < minimum - 1e-6:
            flags += ('parent_shorter_than_minimum',)
        result.append(replace(segment, start=cursor, end=end, text=part, flags=flags))
        cursor = end
    return result


def wrap_two_lines(text, width=18):
    text = ' '.join(text.split())
    if len(text) <= width:
        return text
    # Never split adjacent Han characters; punctuation, whitespace, kana→Han are safe candidates.
    def han(c):
        return '\u3400' <= c <= '\u9fff'

    candidates = [
        i
        for i in range(1, len(text))
        if min(len(text[:i].strip()), len(text[i:].strip())) * 3
        >= len(text[:i].strip()) + len(text[i:].strip())
        and not (han(text[i - 1]) and han(text[i]))
        and text[i] not in '。、，？！?!…」』）'
        and (text[i - 1] in '。、，？！?!… ' or text[i] == ' ')
    ]
    if not candidates:
        return text  # Let the renderer wrap; do not invent a Chinese word boundary.
    cut = min(
        candidates,
        key=lambda i: (max(i, len(text) - i) > width, abs(i - len(text) / 2)),
    )
    return text[:cut].rstrip() + '\n' + text[cut:].lstrip()


def render_sentence_srt(segments):
    return '\n\n'.join(
        f'{i}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n'
        f'{wrap_two_lines(s.text)}'
        for i, s in enumerate(segments, 1)
    ) + ('\n' if segments else '')
