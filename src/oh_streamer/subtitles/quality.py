"""Versioned, CPU-only subtitle quality rules shared by ASR and subtitle tracks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher


def load_quality_config(overrides=None):
    config = json.loads(Path(__file__).with_name("quality.json").read_text(encoding="utf-8"))
    for key, value in (overrides or {}).items():
        config[key] = {**config[key], **value} if isinstance(value, dict) else value
    return config


def quality_fingerprint(config=None):
    """Invalidate derived quality artifacts without invalidating raw ASR."""
    payload = {"config": config or load_quality_config(), "hallucinations": _hallucinations()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize_japanese(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] not in "PSZC")
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def japanese_similarity(left, right):
    """Funnel goldlib.sim: kana-normalized difflib ratio, with empty = zero."""
    left, right = normalize_japanese(left), normalize_japanese(right)
    return SequenceMatcher(None, left, right, autojunk=False).ratio() if left and right else 0.0


def key_text(text):
    """Exact funnel whole-sentence normalization (gold2.norm then key_text)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[（(][^）)]*[）)]", "", text).lower()
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    return re.sub(r"[^ぁ-ゖァ-ヺ一-鿿0-9a-z]", "", text)


def _hallucinations():
    return json.loads(Path(__file__).with_name("hallucinations.json").read_text(encoding="utf-8"))["phrases"]


def hallucination_keys():
    return frozenset(key_text(text) for text in _hallucinations())


def is_onomatopoeia(text, config=None):
    config = config or load_quality_config()
    return bool(re.fullmatch(config["onomatopoeia_pattern"], normalize_japanese(text)))


def f7_low_confidence(segment, config=None):
    """Missing confidence passes that term; only gap cues use confidence gates."""
    stored_only = bool(segment.low_confidence and all(value is None for value in (
        segment.avg_logprob, segment.no_speech_prob, segment.compression_ratio)))
    if not segment.gap and not stored_only:
        return False
    rules = (config or load_quality_config())["f7"]
    low = stored_only or any((
        segment.avg_logprob is not None and segment.avg_logprob < rules["avg_logprob_min"],
        segment.no_speech_prob is not None and segment.no_speech_prob >= rules["no_speech_prob_max"],
        segment.compression_ratio is not None and segment.compression_ratio >= rules["compression_ratio_max"],
    ))
    if low and segment.alt_text and normalize_japanese(segment.alt_text):
        agreement = segment.asr_agreement
        if agreement is None:
            agreement = japanese_similarity(segment.text, segment.alt_text)
        if agreement >= rules["agreement_min"]:
            low = False
    return low
