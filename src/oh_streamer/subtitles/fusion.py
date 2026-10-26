"""Deterministic V3 span selection for the two sequential ASR models."""
from __future__ import annotations

import hashlib
import json
import math
import re

from .quality import japanese_similarity, load_quality_config, normalize_japanese

FUSION_VERSION = "a1-a2-v3-v1"


def fusion_identity(config=None):
    rules = (config or load_quality_config())["f7"]
    payload = {"version": FUSION_VERSION, "agreement_min": rules["agreement_min"],
               "compression_ratio_max": rules["compression_ratio_max"],
               "repeat_count": rules["repeat_count"], "repeat_max_unit": rules["repeat_max_unit"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def select_candidate(a1, a2, config=None):
    """Return selected model and reason; never concatenate candidate texts."""
    rules = (config or load_quality_config())["f7"]
    text1, text2 = normalize_japanese(a1["text"]), normalize_japanese(a2["text"])
    if text1 == text2:
        return "A1", "same"
    if not text1 or not text2:
        return ("A1" if text1 else "A2"), "one_empty"
    if japanese_similarity(text1, text2) >= rules["agreement_min"]:
        return "A1", "agree"
    loop = re.compile(r"(.{1," + str(rules["repeat_max_unit"]) + r"}?)\1{" + str(rules["repeat_count"] - 1) + r",}")
    bad1, bad2 = [bool(loop.search(text)) or (row.get("compression_ratio") or 0) >= rules["compression_ratio_max"]
                  for text, row in ((text1, a1), (text2, a2))]
    if bad1 != bad2:
        return ("A2" if bad1 else "A1"), "demote"
    # Missing scores use the funnel's -9 fallback. Zero is a valid logprob.
    score1 = a1.get("avg_logprob")
    score2 = a2.get("avg_logprob")
    score1 = -9 if score1 is None else score1
    score2 = -9 if score2 is None else score2
    return ("A2" if score2 > score1 else "A1"), "alp"


def _by_span(rows):
    output = {}
    for row in rows:
        key = (float(row["start"]), float(row["end"]))
        if not all(math.isfinite(value) for value in key) or key[0] < 0 or key[1] < key[0]:
            raise ValueError("invalid ASR span")
        if key in output:
            raise ValueError(f"duplicate ASR span {key}")
        output[key] = row
    return output


def fuse_segments(rows_a1, rows_a2=None, *, mode="dual", config=None):
    """Choose one complete record per identical span; single explicitly disables rescue."""
    if mode not in ("dual", "single"):
        raise ValueError("ASR mode must be dual or single")
    first = _by_span(rows_a1)
    if mode == "single":
        return [dict(first[key], alt_text="", asr_agreement=None,
                     selected_model="A1", fusion_reason="single") for key in sorted(first)]
    if rows_a2 is None:
        raise ValueError("dual fusion requires A2 segments")
    second = _by_span(rows_a2)
    if first.keys() != second.keys():
        raise ValueError("ASR models must use exactly the same span set")
    output = []
    for key in sorted(first):
        a1, a2 = first[key], second[key]
        model, reason = select_candidate(a1, a2, config)
        selected, alternate = (a1, a2) if model == "A1" else (a2, a1)
        output.append(dict(selected, alt_text=alternate["text"],
                           asr_agreement=japanese_similarity(selected["text"], alternate["text"]),
                           selected_model=model, fusion_reason=reason))
    return output
