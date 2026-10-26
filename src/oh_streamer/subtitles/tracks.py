"""Shared translation identity and the two AI subtitle display tracks."""
from dataclasses import replace
import hashlib
from pathlib import Path

CONCISE_TITLE = "繁體中文（AI・精簡）"
FULL_TITLE = "繁體中文（AI・完整）"
TRANSLATION_RECIPE = "deduplicated-ja-v1"
TAIWAN_TERMS_VERSION = "taiwan-terms-v1"
DEFAULT_SAMPLING = {"temperature": 0.0, "seed": 42, "cache_prompt": False}


def wording_identity():
    return {"version": TAIWAN_TERMS_VERSION, "sha256": hashlib.sha256(
        Path(__file__).with_name("taiwan_terms.json").read_bytes()).hexdigest()}


def deterministic_sampling(options=None):
    # A custom fixed seed is supported; temperature and prompt caching are fixed.
    seed = (options or {}).get("seed")
    return {**DEFAULT_SAMPLING, "seed": 42 if seed is None else seed}


def unique_segments(segments):
    """Exact Japanese strings, in first timestamp order; no fuzzy deduplication."""
    seen = set()
    result = []
    for segment in sorted(segments, key=lambda s: (s.start, s.end)):
        if segment.text not in seen:
            seen.add(segment.text)
            result.append(segment)
    return result


def render_tracks(rows, render, *, on_fail="hide", suspicious="keep"):
    from .postprocess import display_segments
    from .quality import is_onomatopoeia

    concise, full = [], []
    for row in rows:
        displayed = display_segments([row], on_fail, suspicious)
        if not displayed:
            continue
        low = bool(row.get("low_confidence"))
        onomatopoeia = bool(row.get("onomatopoeia", is_onomatopoeia(row.get("text_ja", ""))))
        if not low and not onomatopoeia:
            concise.extend(displayed)
        full.extend(replace(s, text="（低信心）" + s.text) if low else s for s in displayed)
    return render(concise), render(full)
