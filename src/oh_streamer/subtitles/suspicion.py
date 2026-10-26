"""Experimental ASR disagreement, using identical utterance boundaries only."""
from difflib import SequenceMatcher
import re


def normalize(text):
    # The measured pair プルンプルン / ぷるんぷるん is phonetic agreement.
    text = ''.join(
        chr(ord(c) - 0x60) if '\u30a1' <= c <= '\u30f6' else c for c in text
    )
    return re.sub(r'[^\w]', '', text, flags=re.UNICODE).replace('_', '').casefold()


def mark_disagreement(primary, secondary, threshold=.45):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between zero and one')
    lookup = {(r['start'], r['end']): r for r in secondary}
    result = []
    for row in primary:
        key = (row['start'], row['end'])
        other = lookup.get(key)
        if other is None:
            result.append(
                dict(
                    row,
                    suspected_asr=False,
                    asr_similarity=None,
                    suspicion_status='unpaired: not evidence of correctness',
                )
            )
            continue
        a, b = normalize(row['text']), normalize(other['text'])
        score = SequenceMatcher(None, a, b, autojunk=False).ratio()
        result.append(
            dict(
                row,
                suspected_asr=score < threshold,
                asr_similarity=score,
                secondary_text=other['text'],
                suspicion_status='paired',
            )
        )
    return result
