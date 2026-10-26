"""Result storage and SRT publication for the unified slice scheduler (design M2).

- media_id: HLS sidecar identity when the host knows it, otherwise a content fingerprint of the file;
- content-addressed result files under <subtitles>/cache/{asr,mt}/<key>.json (keys: slice_plan.*_cache_key);
- merging finished slices (and the already translated batches of a slice still running, design D2) into
  the partial/final SRT files the player reads. Same file names and result dict as the former cloud_horizon.publish_cloud_horizon.

No network, no GPU.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from oh_streamer.subtitles import cloud_chunks, runtime
from oh_streamer.subtitles.core import Segment, render_srt
from oh_streamer.subtitles.slice_plan import (
    DEFAULT_LEAD_KEEP_SECONDS, SlicePlan, asr_cache_path, mt_cache_path, slices_by_time,
)
from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE

FINGERPRINT_BLOCK = 1024 * 1024


def rcache_media_id(identity: str) -> str:
    return "rcache:" + str(identity)


def _rcache_fingerprint(path: str | os.PathLike[str]) -> Optional[str]:
    """Id of a range-cache source from its sidecar (the ``.rcache`` marker is a tiny constant JSON, so hashing it
    would give every item the same id). The sidecar identity survives pause/resume and completion; without it
    (no usable sidecar) fall back to request_key + total_bytes + etag, then None."""
    try:
        from oh_streamer.sources import range_cache
        data = range_cache._load_sidecar(range_cache.sidecar_path(path))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    if data.get("identity"):
        return rcache_media_id(data["identity"])
    if data.get("request_key") and data.get("total_bytes"):
        raw = f"{data['request_key']}|{data['total_bytes']}|{data.get('etag') or ''}"
        return "rcache:" + hashlib.sha256(raw.encode()).hexdigest()[:32]
    return None


def media_fingerprint(path: str | os.PathLike[str]) -> str:
    """Stable id of a media file: size + first / middle / last MiB. Independent of path and mtime (a playlist
    rewritten on resume or a moved file keeps its id). A range-cache ``.rcache`` marker is identified by its
    sidecar instead (the marker itself carries no source identity); with no sidecar the absolute path is mixed in
    so two markers can never collide."""
    p = Path(path)
    from oh_streamer.sources import media_input
    if media_input.is_remote_marker(p):
        # A streaming item's audio marker: identity is stable across signed-url refreshes (url/generation change).
        try:
            identity = media_input.read_remote_marker(p).get("identity")
        except media_input.RemoteMarkerError:
            identity = None
        if identity:
            return "stream:" + str(identity)
        return "stream-path:" + hashlib.sha256(os.path.normcase(os.path.abspath(p)).encode()).hexdigest()[:32]
    if media_input.needs_runtime_url(p):
        found = _rcache_fingerprint(p)
        if found:
            return found
        return "rcache-path:" + hashlib.sha256(os.path.normcase(os.path.abspath(p)).encode()).hexdigest()[:32]
    size = p.stat().st_size
    digest = hashlib.sha256(f"size:{size}".encode())
    with p.open("rb") as fh:
        for offset in sorted({0, max(0, size // 2 - FINGERPRINT_BLOCK // 2), max(0, size - FINGERPRINT_BLOCK)}):
            fh.seek(offset)
            digest.update(fh.read(FINGERPRINT_BLOCK))
    return "file:" + digest.hexdigest()[:32]


def hls_media_id(identity: str) -> str:
    return "hls:" + str(identity)


# --------------------------------------------------------------------------------------------------
# Cache files
# --------------------------------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_asr(folder: Path, key: str) -> Optional[dict]:
    data = _read_json(Path(folder) / asr_cache_path(key))
    return data if data and isinstance(data.get("segments"), list) else None


def write_asr(folder: Path, key: str, doc: Mapping[str, Any]) -> None:
    runtime.atomic_write_json(Path(folder) / asr_cache_path(key), dict(doc, key=key))


def read_mt(folder: Path, key: str) -> Optional[dict]:
    data = _read_json(Path(folder) / mt_cache_path(key))
    return data if data and isinstance(data.get("segments"), list) and data.get("complete", True) else None


def write_mt(folder: Path, key: str, doc: Mapping[str, Any]) -> None:
    runtime.atomic_write_json(Path(folder) / mt_cache_path(key), dict(doc, key=key, complete=True))


def partial_path(key: str) -> Path:
    return Path("cache") / "mt" / f"{key}.partial.json"


def read_partial(folder: Path, key: str) -> List[dict]:
    data = _read_json(Path(folder) / partial_path(key))
    return list(data.get("segments") or []) if data and data.get("key") == key else []


def write_partial(folder: Path, key: str, segments: Sequence[Mapping[str, Any]]) -> None:
    runtime.atomic_write_json(Path(folder) / partial_path(key), {"key": key, "segments": list(segments)})


def clear_partial(folder: Path, key: str) -> None:
    try:
        (Path(folder) / partial_path(key)).unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------------------------------

def cues_overlap(a: Mapping[str, Any], b: Mapping[str, Any], min_fraction: float = 0.3) -> bool:
    """Time overlap >= min_fraction of the shorter cue."""
    ov = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    shorter = max(1e-6, min(float(a["end"]) - float(a["start"]), float(b["end"]) - float(b["start"])))
    return ov > 0 and ov / shorter >= min_fraction


REPEAT_GAP_SECONDS = 1.0     # a pre-roll cue starting this soon after a prior cue may repeat it
REPEAT_MIN_RATIO = 0.8
REPEAT_MIN_CHARS = 6


def _compact_text(row: Mapping[str, Any]) -> str:
    return re.sub(r"[\W_]+", "", str(row.get("text_ja") or row.get("text") or ""))


def _repeats(r: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    """Pre-roll cue r repeats prior cue p: same text bar punctuation, starting right after p. Word timing
    can squeeze the repeat into a sliver past p's end (zero-length ASR segment at the cut), so no overlap."""
    gap = float(r["start"]) - float(p["end"])
    if not -0.05 <= gap <= REPEAT_GAP_SECONDS:
        return False
    a, b = _compact_text(r), _compact_text(p)
    # short interjections (だめ だめ) are often said twice for real
    return min(len(a), len(b)) >= REPEAT_MIN_CHARS and difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_MIN_RATIO


def resolve_preroll_overlaps(per_slice: Sequence[tuple], min_fraction: float = 0.3,
                             lead_keep_s: float = DEFAULT_LEAD_KEEP_SECONDS) -> List[Dict[str, Any]]:
    """[(slice start, rows with absolute times)] in slice order -> merged rows. A slice's pre-roll cue (it starts
    before the slice) is dropped when it overlaps (>= min_fraction of the shorter cue) a cue the previous slice
    kept, and ends no later than that prior cue (50 ms tolerance). A cue extending into new speech stays.
    Ownership by end time (slice_plan.owns_cue) leaves such
    pairs only in a small band before a cut, and the two ASR runs may split the sentence differently, so text
    deduplication cannot catch them (T3, batch of one slice). A slice with nothing published yet does not block
    its successor's pre-roll cues.

    A pre-roll cue ending at or before start - lead_keep_s (far pre-roll, kept by the cloud executor, see
    slice_plan.owns_cue keep_preroll) lies wholly in the previous slice's audio. It is dropped only when
    it overlaps a cue the previous slice published (whatever the end times: the two runs split sentences
    differently); with nothing overlapping, the previous slice lost it (tail guard) and it stays.

    A pre-roll cue that repeats a prior cue's text right after it (_repeats) is dropped too."""
    out: List[Dict[str, Any]] = []
    prev: List[Dict[str, Any]] = []

    def covered(r: Mapping[str, Any], start: float) -> bool:
        if float(r["start"]) >= start:
            return False
        far = float(r["end"]) <= start - lead_keep_s
        return any((cues_overlap(r, p, min_fraction) and (far or float(r["end"]) <= float(p["end"]) + 0.05))
                   or _repeats(r, p) for p in prev)

    for start, rows in per_slice:
        kept = [r for r in rows if not covered(r, float(start))]
        out.extend(kept)
        prev = kept
    return out


def _compose(plan: SlicePlan, done_segments: Mapping[int, Sequence[Mapping[str, Any]]],
             partial_segments: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """The combination rule shared by the primary publication and the comparison-track publication: finished
    slices plus the translated batches of a slice still running, pre-roll overlap removal, de-duplication,
    then the concise / full / Japanese SRT texts. Slices are taken in TIME order (T3.6: the list also holds
    islands and split pieces appended later). `covered_s` is the continuous coverage from 0 (the old meaning);
    `covered_ranges` lists every covered [start, end] (done slices whole, a slice still translating up to its
    last published cue), merged where they touch."""
    ordered = slices_by_time(plan)
    per_slice: List[tuple] = []
    done_indices: List[int] = []
    for s in ordered:
        rows = done_segments.get(s.index)
        if rows is not None:
            done_indices.append(s.index)
        else:
            rows = partial_segments.get(s.index) or ()
        absolute = []
        for seg in rows:
            copy = dict(seg)
            if "text_ja" not in copy and "text" in copy:
                copy["text_ja"] = copy["text"]
            copy["start"] = round(float(seg["start"]) + s.start, 3)
            copy["end"] = round(float(seg["end"]) + s.start, 3)
            absolute.append(copy)
        per_slice.append((s.start, absolute))
    all_segments = resolve_preroll_overlaps(per_slice)
    deduped = cloud_chunks.deduplicate_segments(all_segments)

    covered_s = 0.0
    cursor = 0.0                        # end of the continuous run from 0
    for s in ordered:
        if s.start > cursor + 1e-3:
            break                       # an uncut or unfinished gap: coverage from 0 ends here
        if s.index in done_segments:
            covered_s = max(covered_s, s.end)
            cursor = s.end
        else:
            rows = partial_segments.get(s.index)
            if rows:
                covered_s = max(covered_s, s.start + max(float(r["end"]) for r in rows))
            break

    ranges: List[List[float]] = []
    for s in ordered:
        if s.index in done_segments:
            end = s.end
        else:
            rows = partial_segments.get(s.index)
            end = min(s.end, s.start + max(float(r["end"]) for r in rows)) if rows else s.start
        if end <= s.start:
            continue
        if ranges and s.start <= ranges[-1][1] + 1e-3:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([s.start, end])
    covered_ranges = [[round(a, 3), round(b, 3)] for a, b in ranges]

    concise, full, ja = [], [], []
    for item in deduped:
        start_t, end_t = float(item["start"]), float(item["end"])
        ja_text = str(item.get("text_ja") or item.get("text") or "").strip()
        zh_text = str(item.get("zh") or item.get("mt_zh") or "").strip()
        filtered = bool(item.get("is_filtered")) or "[skip]" in zh_text.casefold()
        if zh_text and not filtered:
            # full_only / full_prefix: the local Sakura rows keep the blocks display rule (low-confidence and
            # onomatopoeia cues only in the full track, low confidence marked there)
            if not item.get("full_only"):
                concise.append(Segment(start=start_t, end=end_t, text=zh_text))
            shown = f"{item.get('full_prefix') or ''}{zh_text}"
            full.append(Segment(start=start_t, end=end_t,
                                text=f"{shown}\n{ja_text}" if ja_text and ja_text != zh_text else shown))
        if ja_text:
            ja.append(Segment(start=start_t, end=end_t, text=ja_text))
    return {"deduped": deduped, "covered_s": covered_s, "covered_ranges": covered_ranges,
            "done_indices": done_indices,
            "concise_srt": render_srt(concise), "full_srt": render_srt(full), "ja_srt": render_srt(ja),
            "cues": len(concise)}


def publish(folder: Path | str, plan: SlicePlan, done_segments: Mapping[int, Sequence[Mapping[str, Any]]],
            partial_segments: Optional[Mapping[int, Sequence[Mapping[str, Any]]]] = None, *,
            japanese: bool = False) -> Dict[str, Any]:
    """Write partial.*.srt (and, once every slice of a closed plan is done, the final files) from
    `done_segments` {slice index: translated segments relative to the slice start} plus `partial_segments`
    (batches already translated of slices still being translated: published per batch, design D2)."""
    root = Path(folder)
    composed = _compose(plan, done_segments, partial_segments or {})
    deduped, covered_s, done_indices = composed["deduped"], composed["covered_s"], composed["done_indices"]
    concise_srt, full_srt, ja_srt = composed["concise_srt"], composed["full_srt"], composed["ja_srt"]

    p_zh, p_full, p_ja = root / "partial.zh-TW.srt", root / "partial.zh-TW.full.srt", root / "partial.ja.srt"
    runtime.atomic_write_text(p_zh, concise_srt)
    runtime.atomic_write_text(p_full, full_srt)
    if japanese:
        runtime.atomic_write_text(p_ja, ja_srt)

    complete = bool(plan.slices) and not plan.open and len(done_indices) == len(plan.slices)
    if complete:
        runtime.atomic_write_text(root / "sakura.zh-TW.srt", concise_srt)
        runtime.atomic_write_text(root / "sakura.zh-TW.full.srt", full_srt)
        if japanese:
            runtime.atomic_write_text(root / "ja.srt", ja_srt)
        runtime.atomic_write_json(root / "sakura.zh-TW.json", {
            "recipe": "slice-plan-v2", "segments": deduped, "duration": plan.duration, "complete": True,
            "refinement_pending": False,
        })

    srt_map = {"zh-TW": str(p_zh), "full": str(p_full)}
    tracks = [{"path": str(p_zh), "lang": "chi", "title": CONCISE_TITLE, "default": True},
              {"path": str(p_full), "lang": "chi", "title": FULL_TITLE, "default": False}]
    if japanese:
        srt_map["ja"] = str(p_ja)
        tracks.append({"path": str(p_ja), "lang": "jpn", "title": "日文（AI）", "default": False})
    publication_key = hashlib.sha256(json.dumps(deduped, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "plan_key": f"{plan.media_id}:{publication_key}",
        "covered_s": covered_s,
        "covered_ranges": composed["covered_ranges"],
        "refinement_pending": False,
        "complete": complete,
        "done_indices": done_indices,
        "slices_done": len(done_indices),
        "slices_total": len(plan.slices),
        "cues": len(deduped),
        "srt": srt_map,
        "subtitles": tracks,
    }


# --------------------------------------------------------------------------------------------------
# Comparison tracks (M6b): <subtitles>/combos/<track>.zh-TW.srt + manifest.json
# --------------------------------------------------------------------------------------------------

COMBOS_DIR = "combos"
COMBOS_MANIFEST_FORMAT = "oh_streamer.subtitle_combos"
COMBOS_MANIFEST_SCHEMA = 1
MANAGED_BY = "slice_scheduler"       # manifest entries owned by the scheduler (others are never touched)
_STAGE_NAMES = {"asr": "asr", "mt": "translation"}
_STAGE_ZH = {"asr": "語音辨識", "mt": "翻譯", "config": "設定"}
_FILE_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")


def combo_file_stem(track: str) -> str:
    return _FILE_SAFE.sub("_", str(track)) or "combo"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def track_stage_summary(plan: SlicePlan, track: str) -> Dict[str, Any]:
    """Manifest `stages` of a track: asr / correction / translation with a status over all slices
    (done = every slice done, failed = a slice failed for good, running = some progress, else pending)."""
    from oh_streamer.subtitles import slice_plan as sp
    out: Dict[str, Any] = {}
    for stage, name in _STAGE_NAMES.items():
        states = [s.get(stage, track) for s in plan.slices]
        done = sum(1 for st in states if st.status == sp.DONE)
        failed = [st for st in states if st.status == sp.FAILED]
        if failed:
            status = "failed"
        elif plan.slices and done == len(states) and not plan.open:
            status = "done"
        elif done or any(st.status == sp.RUNNING for st in states):
            status = "running"
        else:
            status = "pending"
        out[name] = {"status": status, "key": None, "cache_hit": False, "seconds": 0.0,
                     "error": (failed[0].detail or failed[0].fail_kind or "failed") if failed else None,
                     "slices_done": done, "slices_total": len(states)}
    out["correction"] = {"status": "skipped", "key": None, "cache_hit": False, "seconds": 0.0, "error": None,
                         "mode": "none"}
    return {k: out[k] for k in ("asr", "correction", "translation")}


def track_failure(plan: SlicePlan, track: str) -> Optional[Dict[str, str]]:
    """{"stage": ..., "error": zh-TW text} of the first failed unit of a track, else None."""
    from oh_streamer.subtitles import slice_plan as sp
    for stage in (sp.STAGE_ASR, sp.STAGE_MT):
        for rank, s in enumerate(slices_by_time(plan)):          # "which part" counts in time order (T3.6)
            st = s.get(stage, track)
            if st.status == sp.FAILED:
                detail = str(st.detail or st.fail_kind or "未知原因")
                return {"stage": _STAGE_NAMES[stage],
                        "error": f"{_STAGE_ZH[stage]}失敗（第 {rank + 1} 段）：{detail}"[:400]}
    return None


def update_combos_manifest(folder: Path | str, entry: Optional[Mapping[str, Any]], *, order: Sequence[str],
                           video: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge `entry` (a scheduler-managed combo; None = only prune) into <folder>/combos/manifest.json and write
    it atomically. Entries of other writers stay; scheduler entries whose id is not in `order` (a removed row)
    are dropped, the rest follow `order`. The caller holds the PlanStore publish lock."""
    root = Path(folder) / COMBOS_DIR
    path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    existing = _read_json(path)
    if existing and existing.get("format") == COMBOS_MANIFEST_FORMAT \
            and existing.get("schema_version") == COMBOS_MANIFEST_SCHEMA and isinstance(existing.get("combos"), list):
        manifest = existing
    entries = [e for e in manifest.get("combos", []) if isinstance(e, dict)]
    foreign = [e for e in entries if e.get("managed_by") != MANAGED_BY]
    mine = {e.get("id"): e for e in entries if e.get("managed_by") == MANAGED_BY}
    if entry is not None:
        mine[entry["id"]] = dict(entry, managed_by=MANAGED_BY)
    ordered = [mine[t] for t in order if t in mine]
    video_info = dict(manifest.get("video") or {}) if isinstance(manifest.get("video"), dict) else {}
    video_info.update({k: v for k, v in (video or {}).items() if k not in video_info})
    out = {"format": COMBOS_MANIFEST_FORMAT, "schema_version": COMBOS_MANIFEST_SCHEMA, "updated_at": _now_iso(),
           "video": video_info, "combos": foreign + ordered}
    if isinstance(manifest.get("mux"), dict):
        out["mux"] = manifest["mux"]
    runtime.atomic_write_json(path, out)
    return out


def publish_combo(folder: Path | str, plan: SlicePlan, track: str, done_segments: Mapping[int, Sequence[Mapping[str, Any]]],
                  partial_segments: Optional[Mapping[int, Sequence[Mapping[str, Any]]]] = None, *,
                  label: str = "", combo: Optional[Mapping[str, Any]] = None, japanese: bool = False,
                  video: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Publish one comparison track as it grows: <folder>/combos/<track>.zh-TW.srt (+ .ja.srt) from the same
    combination as `publish`, and its manifest entry (running / done / failed). Returns the entry."""
    root = Path(folder) / COMBOS_DIR
    composed = _compose(plan, done_segments, partial_segments or {})
    stem = combo_file_stem(track)
    files: Dict[str, Optional[str]] = {"zh": f"{stem}.zh-TW.srt", "ja": None}
    runtime.atomic_write_text(root / files["zh"], composed["concise_srt"])
    if japanese:
        files["ja"] = f"{stem}.ja.srt"
        runtime.atomic_write_text(root / files["ja"], composed["ja_srt"])
    failure = track_failure(plan, track)
    complete = bool(plan.slices) and not plan.open and len(composed["done_indices"]) == len(plan.slices)
    status = "failed" if failure else ("done" if complete else "running")
    entry = {
        "id": track, "label": label or track, "source": "user", "status": status,
        "failed_stage": failure["stage"] if failure else None, "error": failure["error"] if failure else None,
        "params": dict(combo) if isinstance(combo, Mapping) else {},
        "stages": track_stage_summary(plan, track), "files": files, "cue_count": composed["cues"],
        "updated_at": _now_iso(),
    }
    update_combos_manifest(folder, entry, order=list(plan.tracks), video=video)
    return entry


def publish_combo_unready(folder: Path | str, plan: SlicePlan, track: str, reasons: Sequence[str], *,
                          label: str = "", video: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Manifest entry of a comparison row that cannot run (settings not ready): failed, with the zh-TW reasons."""
    blocked = {"status": "blocked", "key": None, "cache_hit": False, "seconds": 0.0, "error": None}
    entry = {
        "id": track, "label": label or track, "source": "user", "status": "failed", "failed_stage": "config",
        "error": "；".join(str(r) for r in reasons) or "比較字幕設定無效", "params": {},
        "stages": {"asr": dict(blocked),
                   "correction": {"status": "skipped", "key": None, "cache_hit": False, "seconds": 0.0,
                                  "error": None, "mode": "none"},
                   "translation": dict(blocked)},
        "files": {"zh": None, "ja": None}, "cue_count": 0, "updated_at": _now_iso(),
    }
    update_combos_manifest(folder, entry, order=list(plan.tracks), video=video)
    return entry


# --- helpers moved from the removed cloud_horizon.py (behaviour unchanged) ---------------------------------
CLOUD_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # cloud ASR APIs (Groq/OpenAI) reject files above 25 MB
CLOUD_UPLOAD_MIN_SPLIT_SECONDS = 30.0      # never split below this duration


def _clip_to_next_start(cues: list[Segment], timed: Sequence[bool]) -> None:
    """A word-timed cue (tail-padded) never runs into the next cue; at least 0.3 s stays where possible."""
    from dataclasses import replace
    from oh_streamer.subtitles.word_timing import MIN_CUE_SECONDS
    for i in range(len(cues) - 1):
        c, nxt = cues[i], cues[i + 1].start
        if timed[i] and c.end > nxt > c.start:
            cues[i] = replace(c, end=max(c.start + MIN_CUE_SECONDS, nxt))


def _prepare_slice_cues(
    raw_segments: Sequence[dict[str, Any]],
    config: Optional[dict[str, Any]] = None,
    settings: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Split raw ASR segments into sentence cues and filter short interjections."""
    from oh_streamer.subtitles.sentences import filter_short, sentence_cues

    sentence_cfg = (config or {}).get("sentence") or {}
    sentence_mode = bool(sentence_cfg.get("sentence_mode", True))
    short_threshold = float(sentence_cfg.get("short_threshold", 1.0))
    short_policy = str(sentence_cfg.get("short_policy", "whitelist"))
    show_interjections = bool(
        sentence_cfg.get("show_interjections", (settings or {}).get("subtitle_interjections", False))
    )

    from oh_streamer.subtitles import word_timing
    from oh_streamer.subtitles.core import _DEFAULT_BLACKLIST, _blacklisted

    segs: list[Segment] = []
    seg_words: list[Any] = []            # parallel to segs: the segment's word timestamps (cloud ASR) or None
    for item in raw_segments:
        if isinstance(item, Segment):
            segs.append(item)
            seg_words.append(None)
        elif isinstance(item, dict):
            text_ja = str(item.get("text_ja") or item.get("text") or "").strip()
            if not text_ja:
                continue
            if item.get("words") and _blacklisted(text_ja, speech_present=True, blacklist=_DEFAULT_BLACKLIST):
                continue                 # cloud replies skip core.clean_segments: credit lines like "by H." here
            start, end, words = float(item["start"]), float(item["end"]), None
            if item.get("words"):
                # segment times are contiguous (silence / music before speech is absorbed); the words are not
                span = word_timing.retime_from_words(text_ja, item["words"])
                if span is not None:
                    start, end, words = span[0], span[1], item["words"]
            seg_words.append(words)
            segs.append(
                Segment(
                    start=start,
                    end=end,
                    text=text_ja,
                    no_speech_prob=item.get("no_speech_prob"),
                    avg_logprob=item.get("avg_logprob"),
                    flags=tuple(item.get("flags") or ()),
                    gap=bool(item.get("gap")),
                    low_confidence=bool(item.get("low_confidence")),
                    compression_ratio=item.get("compression_ratio"),
                )
            )

    if sentence_mode:
        cues: list[Segment] = []
        timed: list[bool] = []
        for s, words in zip(segs, seg_words):
            made = word_timing.word_sentence_cues(s, words) if words else None
            if made is None:
                made = sentence_cues(s)
            cues.extend(made)
            timed.extend([bool(words)] * len(made))
        _clip_to_next_start(cues, timed)
        # filter_short's threshold was tuned on padded spans; word times are tight (a 0.7 s 「失礼」 is real speech),
        # so a word-timed cue is measured from the previous cue's end, like the contiguous segment Groq returned
        basis = [replace(c, start=cues[i - 1].end) if t and i and cues[i - 1].end < c.start else c
                 for i, (c, t) in enumerate(zip(cues, timed))]
        if short_policy != "keep" and not show_interjections:
            kept, _ = filter_short(basis, threshold=short_threshold, policy=short_policy)
            kept_ids = {id(b) for b in kept}
            cues = [c for c, b in zip(cues, basis) if id(b) in kept_ids]
    else:
        cues = segs
        _clip_to_next_start(cues, [w is not None for w in seg_words])

    return [
        {
            "start": round(c.start, 3),
            "end": round(c.end, 3),
            "text": c.text,
            "text_ja": c.text,
            "low_confidence": bool(c.low_confidence),
            "flags": list(c.flags),
            "gap": bool(c.gap),
            "avg_logprob": c.avg_logprob,
            "no_speech_prob": c.no_speech_prob,
            "compression_ratio": c.compression_ratio,
        }
        for c in cues
    ]


def _translate_batch(client, batch, parse):
    """Translate one batch of numbered lines in ONE request. A garbled numbering raises (bad_response /
    ValueError, see batch_needs_shrink); the executor retries a smaller batch and checkpoints what passed."""
    numbered = "\n".join(f"{i + 1}. {c['text_ja']}" for i, c in enumerate(batch))
    parsed = parse(client.translate(numbered), list(range(1, len(batch) + 1)), strict=False)
    if len(parsed) != len(batch):
        raise ValueError(f"MT batch returned {len(parsed)} lines, expected {len(batch)}")
    return parsed


def batch_needs_shrink(exc: BaseException) -> bool:
    """Would a smaller batch help? Garbled numbering (the model re-split lines) and a request too large for the
    provider's per-minute token limit (providers.http: bad_response + too_large) do; quota waits do not."""
    from .providers.http import ProviderError
    if isinstance(exc, ProviderError):
        return exc.kind == 'bad_response'
    return isinstance(exc, ValueError)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _audio_pieces(source_path, out_path: Path, start: float, duration: float, *, fmt: str = "flac") -> List[Tuple[Path, float]]:
    """Encode [start, start+duration) and return [(file, offset_seconds_from_start)].

    A file above CLOUD_UPLOAD_MAX_BYTES is replaced by two re-encoded halves (recursively, down to
    CLOUD_UPLOAD_MIN_SPLIT_SECONDS). Normal-size slices yield exactly [(out_path, 0.0)].
    """
    if not out_path.is_file():
        cloud_chunks.extract_chunk_audio(source_path=str(source_path), out_path=str(out_path),
                                         start_sec=start, duration_sec=duration, fmt=fmt)
    if _file_size(out_path) <= CLOUD_UPLOAD_MAX_BYTES or duration / 2.0 < CLOUD_UPLOAD_MIN_SPLIT_SECONDS:
        return [(out_path, 0.0)]
    half = duration / 2.0
    pieces: List[Tuple[Path, float]] = []
    for n, (off, dur) in enumerate(((0.0, half), (half, duration - half))):
        sub = out_path.with_name(f"{out_path.stem}_p{n}{out_path.suffix}")
        pieces += [(f, o + off) for f, o in _audio_pieces(source_path, sub, start + off, dur, fmt=fmt)]
    return pieces


def _shift_segment(seg: Any, offset: float) -> Any:
    if not offset or not isinstance(seg, dict):
        return seg
    seg = dict(seg)
    for key in ("start", "end"):
        if key in seg:
            seg[key] = round(float(seg[key]) + offset, 4)
    if isinstance(seg.get("words"), list):
        seg["words"] = [_shift_segment(w, offset) for w in seg["words"]]
    return seg
