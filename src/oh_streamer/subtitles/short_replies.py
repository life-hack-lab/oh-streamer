"""Opt-in rule for short (< threshold) VAD utterances: keep genuine short replies, drop noise.

Used through ``sentences.filter_short(..., policy='replies')``. The default ``'whitelist'``
policy is untouched. Deterministic; needs only text, duration and neighbouring VAD gaps
(no model confidences, so it works for any ASR backend).

Decision order for one short utterance:
1. hard reject  - empty/punctuation only, blacklist hit, repetition, foreign script,
                  Han-only multi-char text (Chinese-like), implausible chars/second.
2. tier A keep  - known short reply / back-channel / interjection (REPLY_LEXICON, regex).
3. tier B keep  - otherwise need ``min_signals`` of 4 weak signals (script, plausible
                  length+rate, clean edges, dialogue context).
"""
import re
import unicodedata

from .core import _DEFAULT_BLACKLIST, _blacklisted, _truncate_repeated_ngram

# Normalised form (no punctuation, no prolongation marks, no trailing small tsu, hiragana).
REPLY_LEXICON = frozenset(
    (
        # the existing production whitelist
        'はい', 'うん', 'ううん', 'ええ', 'え', 'あ', 'ああ', 'ん', 'はあ', 'おう', 'おい',
        'なあ', 'ほら', 'そう', 'そうです', 'いいえ', 'ありがとう', 'すみません', 'ごめん',
        '部長', '社長', '店長',
        # polite replies / acknowledgements
        'わかりました', '分かりました', 'わかった', '分かった', 'かしこまりました', 'はいはい',
        'いえ', 'いや', 'いえいえ', 'ありがとうございます', 'ごめんなさい', 'すいません',
        'どうも', 'いただきます', 'おやすみ', 'お願いします', 'おねがいします',
        'そうですね', 'そうですか', 'そうなんですか', 'そっか', 'そうか', 'そうだね', 'ですよね',
        # back-channel / fillers / small exclamations
        'なるほど', 'ふむ', 'ふん', 'へえ', 'ほう', 'ほお', 'うむ', 'あの', 'あのう', 'えっと',
        'ええと', 'その', 'まあ', 'ねえ', 'よし', 'よいしょ', 'あはは', 'ふふ', 'ははは',
        'えへへ', 'うふふ', 'へへ', 'あれ', 'あら', 'おや', '待って', 'だめ', 'やだ',
        'いやだ', 'やめて',
    )
)
_INTERJ = re.compile(r'^[あいうえおんはへふほぁぃぅぇぉ]{1,4}$')
_PUNCT = re.compile(
    r'[\s　.,!?;:\'"()\[\]{}<>，。！？；：、…‥・「」『』【】（）《》〈〉~～〜\-—―]+'
)
_PROLONG = re.compile(r'[ーｰ]+')
_JA = re.compile(r'^[ぁ-ゟ゠-ヿ㐀-鿿ー]+$')
_KANA = re.compile(r'[ぁ-ゟ゠-ヿー]')
# Stock phrases ASR models invent on near-silence (matched anywhere in the text).
HALLUCINATION_PHRASES = ('ご視聴', 'チャンネル登録', 'ご覧いただき', '字幕')
MAX_CPS = 9.0      # compact chars / second above which a < 1 s cue is not plausible speech
MAX_CPS_KNOWN = 14.0  # looser limit when the text is a known reply (fast 'ありがとうございます')
MIN_CPS = 1.0
MAX_LEN = 8
CONTEXT_GAP = 1.0  # neighbouring VAD utterance within this gap = inside a dialogue turn


def normalize(text: str) -> str:
    """Compact form for lexicon lookup: NFKC, punctuation and prolongation stripped,
    katakana->hiragana, trailing small tsu dropped, whole-string reduplication folded."""
    t = unicodedata.normalize('NFKC', text)
    t = _PROLONG.sub('', _PUNCT.sub('', t))
    t = ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in t)
    t = t.rstrip('っ')
    if len(t) >= 2 and len(t) % 2 == 0 and t[: len(t) // 2] == t[len(t) // 2:]:
        t = t[: len(t) // 2]
    return t


def is_known_reply(text: str) -> bool:
    n = normalize(text)
    return bool(n) and (n in REPLY_LEXICON or bool(_INTERJ.fullmatch(n)))


def hard_reject_reason(text: str, duration: float):
    """Return a reason if the text cannot be a genuine short reply, else None."""
    flat = _PUNCT.sub('', unicodedata.normalize('NFKC', text))
    n = normalize(text)
    if not n:
        return 'empty'
    if any(p in text for p in HALLUCINATION_PHRASES) or _blacklisted(
        text, speech_present=True, blacklist=_DEFAULT_BLACKLIST
    ):
        return 'blacklist'
    if len(flat) > 3 and _truncate_repeated_ngram(flat) != flat:
        return 'repetition'
    if not _JA.fullmatch(flat):
        return 'foreign_script'
    if not _KANA.search(flat) and len(flat) >= 2 and n not in REPLY_LEXICON:
        return 'han_only'
    limit = MAX_CPS_KNOWN if is_known_reply(text) else MAX_CPS
    if duration > 0 and len(flat) / duration > limit:
        return 'implausible_rate'
    return None


def weak_signals(text: str, duration: float, context_gap) -> dict:
    flat = _PUNCT.sub('', unicodedata.normalize('NFKC', text))
    kana = len(_KANA.findall(flat))
    return {
        'script': bool(flat) and kana / len(flat) >= 0.8,
        'plausible': 2 <= len(flat) <= MAX_LEN
        and duration > 0
        and MIN_CPS <= len(flat) / duration <= MAX_CPS,
        'clean_edges': not re.match(r'^[…‥.．・]', text.strip()),
        'context': context_gap is not None and context_gap <= CONTEXT_GAP,
    }


def classify_short(text: str, duration: float, context_gap=None, min_signals: int = 4):
    """-> (keep, reason)."""
    reason = hard_reject_reason(text, duration)
    if reason:
        return False, reason
    if is_known_reply(text):
        return True, 'known_reply'
    if sum(weak_signals(text, duration, context_gap).values()) >= min_signals:
        return True, 'signals'
    return False, 'weak_signals'


def filter_short_replies(segments, threshold=1.0, min_signals=4):
    """Same contract as ``sentences.filter_short``: (kept, rejected). Utterances at or above
    ``threshold`` are always kept; shorter ones go through ``classify_short``."""
    rows = list(segments)
    order = sorted(range(len(rows)), key=lambda i: (rows[i].start, rows[i].end))
    pos = {idx: k for k, idx in enumerate(order)}
    kept, rejected = [], []
    for i, s in enumerate(rows):
        duration = round(s.end - s.start, 6)
        if duration >= threshold:
            kept.append(s)
            continue
        k = pos[i]
        gaps = []
        if k > 0:
            gaps.append(s.start - rows[order[k - 1]].end)
        if k + 1 < len(order):
            gaps.append(rows[order[k + 1]].start - s.end)
        keep, _ = classify_short(s.text, duration, min(gaps) if gaps else None, min_signals)
        (kept if keep else rejected).append(s)
    return kept, rejected
