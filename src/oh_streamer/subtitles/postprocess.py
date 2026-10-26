"""Data-driven Taiwan wording and display policy; never infer ASR corrections."""
import json
from pathlib import Path
import re
from .core import Segment, format_timestamp, has_japanese_residue, to_taiwan_traditional


def taiwan_text(text, rules=None):
    if rules is None:
        rules = json.loads(
            Path(__file__).with_name('taiwan_terms.json').read_text('utf-8')
        )
    for rule in rules:
        source = rule['src']
        mode = rule.get('match', 'contains')
        if mode not in ('contains', 'word', 'exact') or not source:
            raise ValueError('invalid Taiwan replacement rule')
        protected = []
        for exception in rule.get('exceptions', []):
            protected.extend(
                (m.start(), m.end()) for m in re.finditer(re.escape(exception), text)
            )
        pattern = re.escape(source)
        if mode == 'word':
            pattern = r'(?<!\w)' + pattern + r'(?!\w)'
        elif mode == 'exact':
            pattern = '^' + pattern + '$'
        text = re.sub(
            pattern,
            lambda m: m.group()
            if any(a < m.end() and b > m.start() for a, b in protected)
            else rule['dst'],
            text,
        )
    text = text.translate(str.maketrans('?!,', '？！，'))
    return '\n'.join(
        re.sub(r'[。，]+$', '', line.rstrip()) for line in text.splitlines()
    ).strip()


def display_segments(rows, on_fail='hide', suspicious='keep'):
    if on_fail not in ('hide', 'japanese') or suspicious not in ('keep', 'hide', 'mark'):
        raise ValueError('invalid display policy')
    result = []
    for row in rows:
        failed = (
            row.get('preserved_original')
            or '[原文]' in row['text']
            or has_japanese_residue(row['text'])
        )
        if failed and on_fail == 'hide':
            continue
        if row.get('suspected_asr') and suspicious == 'hide':
            continue
        text = (
            row.get('text_ja', row['text'].replace('[原文]', '').strip())
            if failed
            else taiwan_text(row['text'])
        )
        if row.get('suspected_asr') and suspicious == 'mark':
            text = '（可能聽錯）' + text
        if text.strip():
            result.append(Segment(row['start'], row['end'], text))
    return result


def render_fixed_srt(segments, *, japanese=False):
    """Wrap Chinese visually only; keep every supplied cue edge exactly.

    Japanese has no automatic line or word cuts. Callers own cue timing.
    """
    blocks = []
    for s in segments:
        text = (
            s.text
            if japanese
            else '\n'.join(s.text[i:i + 18] for i in range(0, len(s.text), 18))
        )
        if text.strip():
            blocks.append(
                f'{len(blocks) + 1}\n{format_timestamp(s.start)} --> '
                f'{format_timestamp(s.end)}\n{text}'
            )
    return '\n\n'.join(blocks) + ('\n' if blocks else '')


def on_reference_timing(rows, reference):
    """Place revised text on an existing reviewed cue grid, without shifting it.

    Requires identical source segmentation; this does not align different ASRs.
    A source split into several reviewed cues gets proportional text portions.
    """
    assigned = {i: [] for i in range(len(rows))}
    for cue in reference:
        candidates = [
            i
            for i, r in enumerate(rows)
            if r['start'] <= cue.start + .001 and r['end'] > cue.start - .001
        ]
        if not candidates:
            raise ValueError('reference cue has no matching source span')
        index = max(candidates, key=lambda i: rows[i]['start'])
        assigned[index].append(cue)
    if any(not cues for cues in assigned.values()):
        raise ValueError('reference is missing a source translation')
    output = []
    for i, row in enumerate(rows):
        text = display_segments([row])
        if not text:
            continue
        text = text[0].text.replace('\n', '')
        cues = assigned[i]
        if len(text) < len(cues):
            raise ValueError('translation too short for fixed cue grid; review manually')
        weights = [max(1, len(c.text.replace('\n', ''))) for c in cues]
        total = sum(weights)
        cursor = weight = 0
        for j, cue in enumerate(cues):
            weight += weights[j]
            stop = (
                len(text)
                if j == len(cues) - 1
                else min(
                    len(text) - (len(cues) - j - 1),
                    max(cursor + 1, round(len(text) * weight / total)),
                )
            )
            output.append(Segment(cue.start, cue.end, text[cursor:stop]))
            cursor = stop
    return sorted(output, key=lambda s: s.start)


def _safe_to_taiwan(text, converter=None):
    """Convert to Taiwan Traditional; return the text unchanged if conversion fails."""
    try:
        if converter is not None:
            return to_taiwan_traditional(text, converter=converter)
        return to_taiwan_traditional(text)
    except Exception:
        return text
